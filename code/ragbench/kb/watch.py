"""The inbox watcher: drop a report into `corpus/inbox`, and about ten seconds later it is answerable.

    python -m ragbench.kb watch [--inbox corpus/inbox] [--interval 2] [--no-embed]

The foreground loop polls the inbox every `interval_s` seconds and ingests every file whose size has stopped
changing between two polls (so a file still being copied is left alone). Ingestion is stages 1-7 of
`ingest.add_file`: no model call, the document is visible to queries the moment its `ready` stage commits.

A background thread then embeds what has just become ready (stage 8, bge-m3, the only GPU work), one document at a
time, so the answerable state never waits for the vectors. `--no-embed` leaves every document at `ready`, which is
what the lazy path of AGENTIC3 expects.

An ingested file is moved to `<inbox>/_ingested/`, a rejected or failing one to `<inbox>/_failed/`, so the inbox
itself always shows what is still waiting. Nothing depends on those moves: identity is the content hash, so a file
dropped in twice is a no-op either way, and a crash mid-ingest leaves the file in the inbox to be retried.
"""
import queue
import threading
import time
from pathlib import Path

from ..config import load_config
from . import ingest as I

SUFFIXES = (".pdf", ".html", ".htm")
DONE_DIR = "_ingested"
FAILED_DIR = "_failed"


class Watcher:
    def __init__(self, cfg=None, inbox=None, interval_s=2.0, embed=True, corpus=I.USER_CORPUS, log=print):
        self.cfg = cfg or load_config()
        self.inbox = Path(inbox) if inbox else I.inbox_path(self.cfg)
        self.interval_s = float(interval_s)
        self.embed = embed
        self.corpus = corpus
        self.log = log
        self.sizes = {}
        self.queue = queue.Queue()
        self.stop = threading.Event()
        self.embedded = []
        self.ingested = []

    # ---- stage 8, off the critical path ----------------------------------------------------------------------
    def _embed_loop(self):
        from ..llm import OllamaClient
        con = I.open_kb(self.cfg)
        client = OllamaClient(self.cfg)
        try:
            while not (self.stop.is_set() and self.queue.empty()):
                try:
                    doc_id = self.queue.get(timeout=self.interval_s)
                except queue.Empty:
                    continue
                try:
                    stats = I.embed_documents(con, [doc_id], self.cfg, client=client)
                    self.embedded.append(doc_id)
                    self.log("  embedded %s: %d passage(s), %d new, %.0f ms"
                             % (doc_id, stats["passages"], stats["embedded"], stats["wall_ms"]))
                except Exception as e:  # noqa: BLE001
                    self.log("  embed failed for %s: %s: %s" % (doc_id, type(e).__name__, e))
                finally:
                    self.queue.task_done()
        finally:
            con.close()

    # ---- the inbox -------------------------------------------------------------------------------------------
    def pending(self):
        """Files whose size did not change since the previous poll (a copy in progress is not ready yet)."""
        ready = []
        seen = set()
        for p in sorted(self.inbox.glob("*")):
            if p.is_dir() or p.suffix.lower() not in SUFFIXES:
                continue
            seen.add(p)
            size = p.stat().st_size
            if self.sizes.get(p) == size and size > 0:
                ready.append(p)
            self.sizes[p] = size
        self.sizes = {k: v for k, v in self.sizes.items() if k in seen}
        return ready

    def _move(self, path, folder):
        dest = self.inbox / folder
        dest.mkdir(parents=True, exist_ok=True)
        target = dest / path.name
        if target.exists():
            target.unlink()
        path.replace(target)
        return target

    def ingest_one(self, con, path):
        t0 = time.perf_counter()
        try:
            res = I.add_file(con, path, self.cfg, corpus=self.corpus)
        except Exception as e:  # noqa: BLE001
            self.log("%s FAILED: %s: %s" % (path.name, type(e).__name__, e))
            self._move(path, FAILED_DIR)
            return None
        docs = res.get("documents") or [res]
        for r in docs:
            self.log("%s -> %s %s (%.2f s to ready)" % (path.name, r.get("doc_id"), r.get("action"),
                                                        (time.perf_counter() - t0)))
            if r.get("action") == "rejected":
                continue
            self.ingested.append(r["doc_id"])
            if self.embed:
                self.queue.put(r["doc_id"])
        self._move(path, FAILED_DIR if docs[0].get("action") == "rejected" else DONE_DIR)
        return res

    def run(self, once=False):
        self.inbox.mkdir(parents=True, exist_ok=True)
        con = I.open_kb(self.cfg)
        thread = None
        if self.embed:
            thread = threading.Thread(target=self._embed_loop, name="kb-embed", daemon=True)
            thread.start()
        self.log("watching %s every %.1f s (embed %s); Ctrl-C to stop"
                 % (self.inbox, self.interval_s, "on" if self.embed else "off"))
        try:
            while True:
                if once:
                    self.pending()                          # the first poll only records sizes
                for p in self.pending():
                    self.ingest_one(con, p)
                if once:
                    break
                time.sleep(self.interval_s)
        except KeyboardInterrupt:
            self.log("\nstopping; every finished stage is committed")
        finally:
            self.stop.set()
            if thread is not None:
                self.queue.join()
                thread.join(timeout=30)
            con.close()
        return 0
