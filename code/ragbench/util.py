"""Small shared helpers: atomic JSON writes, JSONL append/read, a sqlite key-value
cache for resumable work (embeddings, enrichment outputs), an inter-process file
lock, hashing and directory sizes."""
import hashlib
import json
import os
import sqlite3
import threading
import time
from contextlib import contextmanager
from pathlib import Path

try:
    import numpy as np
except Exception:  # noqa: BLE001
    np = None


def sha1_text(s):
    return hashlib.sha1((s or "").encode("utf-8")).hexdigest()


def sha1_json(obj):
    return sha1_text(json.dumps(obj, sort_keys=True, ensure_ascii=False))


def now_iso():
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def atomic_write_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(".%s.tmp-%d-%d" % (path.name, os.getpid(), threading.get_ident()))
    with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)
    for attempt in range(10):
        try:
            os.replace(tmp, path)
            return path
        except PermissionError:
            time.sleep(0.2 * (attempt + 1))
    os.replace(tmp, path)
    return path


def atomic_write_json(path, obj, indent=None):
    return atomic_write_text(path, json.dumps(obj, ensure_ascii=False, indent=indent))


def read_json(path, default=None):
    path = Path(path)
    if not path.is_file():
        return default
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


_APPEND_LOCK = threading.Lock()


def append_jsonl(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(obj, ensure_ascii=False, default=str) + "\n"
    with _APPEND_LOCK:
        with open(path, "a", encoding="utf-8", newline="\n") as fh:
            fh.write(line)
            fh.flush()


def read_jsonl(path):
    path = Path(path)
    if not path.is_file():
        return []
    out = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue  # a torn last line from an interrupted run is skipped
    return out


def dir_size_bytes(path):
    total = 0
    for root, _dirs, files in os.walk(path):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return total


def safe_name(s, maxlen=80):
    """Filesystem-safe ASCII name; non-ASCII content is replaced by a short hash."""
    keep = "".join(ch if (ch.isascii() and (ch.isalnum() or ch in "-_.")) else "_" for ch in s)
    if keep != s or len(keep) > maxlen:
        keep = keep[: maxlen - 9].strip("_") + "-" + sha1_text(s)[:8]
    return keep or sha1_text(s)[:12]


class FileLock:
    """Exclusive inter-process lock on a lock file (msvcrt on Windows, fcntl elsewhere).
    Re-entrant within a thread."""

    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = None
        self._depth = 0
        self._tlock = threading.RLock()

    def acquire(self, timeout=None, poll=0.05):
        self._tlock.acquire()
        if self._depth > 0:
            self._depth += 1
            return True
        start = time.time()
        fh = open(self.path, "a+b")
        while True:
            try:
                if os.name == "nt":
                    import msvcrt
                    fh.seek(0)
                    msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except OSError:
                if timeout is not None and time.time() - start > timeout:
                    fh.close()
                    self._tlock.release()
                    raise TimeoutError("lock busy: %s" % self.path)
                time.sleep(poll)
        self._fh = fh
        self._depth = 1
        return True

    def release(self):
        if self._depth > 1:
            self._depth -= 1
            self._tlock.release()
            return
        fh = self._fh
        self._fh = None
        self._depth = 0
        try:
            if fh is not None:
                if os.name == "nt":
                    import msvcrt
                    fh.seek(0)
                    try:
                        msvcrt.locking(fh.fileno(), msvcrt.LK_UNLCK, 1)
                    except OSError:
                        pass
                else:
                    import fcntl
                    fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
                fh.close()
        finally:
            self._tlock.release()

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, *exc):
        self.release()


class KVCache:
    """Durable key -> JSON / numpy-array cache in one sqlite file. Safe to resume:
    every put is committed. Keys are strings (use sha1 of the inputs + model tag).

        cache = KVCache(index_dir / "embeddings.sqlite")
        v = cache.get_array(key)
        if v is None: cache.put_array(key, vec)
    """

    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._db = sqlite3.connect(str(self.path), timeout=60, check_same_thread=False)
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("CREATE TABLE IF NOT EXISTS kv (k TEXT PRIMARY KEY, kind TEXT, v BLOB, dtype TEXT, shape TEXT)")
        self._db.commit()

    def __contains__(self, key):
        with self._lock:
            return self._db.execute("SELECT 1 FROM kv WHERE k=?", (key,)).fetchone() is not None

    def get(self, key, default=None):
        with self._lock:
            row = self._db.execute("SELECT kind, v FROM kv WHERE k=?", (key,)).fetchone()
        if row is None or row[0] != "json":
            return default
        return json.loads(row[1])

    def put(self, key, obj):
        with self._lock:
            self._db.execute("INSERT OR REPLACE INTO kv (k, kind, v, dtype, shape) VALUES (?,?,?,?,?)",
                             (key, "json", json.dumps(obj, ensure_ascii=False), None, None))
            self._db.commit()

    def get_array(self, key):
        with self._lock:
            row = self._db.execute("SELECT kind, v, dtype, shape FROM kv WHERE k=?", (key,)).fetchone()
        if row is None or row[0] != "array":
            return None
        shape = tuple(json.loads(row[3]))
        return np.frombuffer(row[1], dtype=row[2]).reshape(shape).copy()

    def put_array(self, key, arr):
        arr = np.asarray(arr)
        with self._lock:
            self._db.execute("INSERT OR REPLACE INTO kv (k, kind, v, dtype, shape) VALUES (?,?,?,?,?)",
                             (key, "array", arr.tobytes(), str(arr.dtype), json.dumps(list(arr.shape))))
            self._db.commit()

    def put_many(self, arrays=(), objs=()):
        """Several puts in ONE transaction (one commit): arrays = [(key, array)], objs = [(key, json-able)].
        Same rows as put_array / put; added 2026-09-24 for batch commits of the hybrid prebuild."""
        rows = [(k, "array", np.asarray(a).tobytes(), str(np.asarray(a).dtype), json.dumps(list(np.asarray(a).shape)))
                for k, a in arrays]
        rows += [(k, "json", json.dumps(o, ensure_ascii=False), None, None) for k, o in objs]
        if not rows:
            return 0
        with self._lock:
            self._db.executemany("INSERT OR REPLACE INTO kv (k, kind, v, dtype, shape) VALUES (?,?,?,?,?)", rows)
            self._db.commit()
        return len(rows)

    def missing(self, keys):
        """The keys (in input order, duplicates kept once) that have no row."""
        out, seen = [], set()
        with self._lock:
            for k in keys:
                if k in seen:
                    continue
                seen.add(k)
                if self._db.execute("SELECT 1 FROM kv WHERE k=?", (k,)).fetchone() is None:
                    out.append(k)
        return out

    def keys(self):
        with self._lock:
            return [r[0] for r in self._db.execute("SELECT k FROM kv")]

    def __len__(self):
        with self._lock:
            return self._db.execute("SELECT COUNT(*) FROM kv").fetchone()[0]

    def close(self):
        with self._lock:
            self._db.close()


@contextmanager
def timer():
    t = {"start": time.perf_counter(), "ms": None}
    try:
        yield t
    finally:
        t["ms"] = (time.perf_counter() - t["start"]) * 1000.0
