"""Plain-text section stores for grep, and the shared ripgrep wrapper.

Layout of a store (root = work/stores/<name>/ for the plain store; OURS writes its own
enriched variant with write_store under its index dir):
  <root>/manifest.json            {variant, created, files: {rel_path: {doc_id, section_id, level, title}}, ...}
  <root>/<doc_id>/<order:04d>.txt one file per section (its OWN units only)
  <root>/<doc_id>/_linemap.json   {rel_path: [[first_line, last_line, unit_id, tag], ...]} (1-based, inclusive)

Section file format (plain variant):
  line 1  "@@SECTION doc=<doc_id> sec=<section_id> level=<n>"
  line 2  "@@TITLE <section title>"
  line 3  "@@PATH <root title> > ... > <section title>"
  line 4  ""
  then each unit's text (tables span several lines), units separated by one blank line.
Header lines carry tag "meta" in the line map, unit lines tag "body". Enriched stores
may add their own tags (e.g. "header", "fact") through write_store.

    root = build_plain_store(load_set("A_en"))              # cached; rebuilt when the corpus changes
    hits = grep(root, r"revenue", ignore_case=True)            # -> [GrepHit]
    read_lines(root, hits[0].path, hits[0].line - 3, hits[0].line + 3)
"""
import json
import os
import subprocess
from dataclasses import dataclass, field, asdict
from pathlib import Path

from .config import load_config
from .util import atomic_write_json, atomic_write_text, read_json, now_iso, safe_name, sha1_text

STORE_FORMAT = 1


@dataclass
class StoreBlock:
    text: str                 # may contain newlines
    unit_id: str = ""         # "" for header/meta blocks
    tag: str = "body"         # meta | body | header | fact | ... (free for enriched stores)


@dataclass
class StoreFile:
    doc_id: str
    section_id: str
    rel_path: str             # "<doc_id>/<name>.txt", forward slashes
    level: int
    title: str
    blocks: list = field(default_factory=list)   # list[StoreBlock], written in order, one blank line between


def write_store(root, files, variant, extra_manifest=None):
    """Write StoreFiles + line maps + manifest atomically per file. Returns root Path."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    manifest_files = {}
    linemaps = {}
    for f in files:
        lines = []
        spans = []
        for bi, blk in enumerate(f.blocks):
            if bi > 0 and not (blk.tag == "meta" and f.blocks[bi - 1].tag == "meta"):
                lines.append("")
            blines = (blk.text or "").split("\n")
            start = len(lines) + 1
            lines.extend(blines)
            spans.append([start, len(lines), blk.unit_id, blk.tag])
        atomic_write_text(root / f.rel_path, "\n".join(lines) + "\n")
        manifest_files[f.rel_path] = {"doc_id": f.doc_id, "section_id": f.section_id, "level": f.level, "title": f.title}
        linemaps.setdefault(f.doc_id, {})[f.rel_path] = spans
    for doc_id, lm in linemaps.items():
        atomic_write_json(root / doc_id / "_linemap.json", lm)
    man = {"format": STORE_FORMAT, "variant": variant, "created": now_iso(), "n_files": len(manifest_files),
           "files": manifest_files}
    man.update(extra_manifest or {})
    atomic_write_json(root / "manifest.json", man)
    return root


def plain_files(doc, normalized=True):
    """StoreFiles for one Document (the plain variant used by B5)."""
    out = []
    for s in doc.sections:
        title = s.title if normalized else s.title_raw
        path = " > ".join(doc.section_path(s.id)) if normalized else s.title_raw
        blocks = [StoreBlock("@@SECTION doc=%s sec=%s level=%d" % (doc.id, s.id, s.level), "", "meta"),
                  StoreBlock("@@TITLE %s" % title.replace("\n", " "), "", "meta"),
                  StoreBlock("@@PATH %s" % path.replace("\n", " "), "", "meta")]
        for uid in s.unit_ids:
            u = doc.unit(uid)
            txt = u.text if normalized else u.text_raw
            if txt.strip():
                blocks.append(StoreBlock(txt, uid, "body"))
        out.append(StoreFile(doc.id, s.id, "%s/%04d.txt" % (safe_name(doc.id), s.order), s.level, s.title, blocks))
    return out


def build_plain_store(corpus_set, root=None, normalized=True, rebuild=False, cfg=None):
    """Plain section store for a CorpusSet; skipped when the manifest fingerprint matches."""
    cfg = cfg or load_config()
    name = corpus_set.name + ("" if normalized else "-raw")
    root = Path(root) if root else cfg.path("store_root") / name
    fp = corpus_set.fingerprint()
    man = read_json(root / "manifest.json")
    if not rebuild and man and man.get("corpus_fingerprint") == fp and man.get("format") == STORE_FORMAT:
        return root
    files = []
    for d in corpus_set.docs:
        files.extend(plain_files(d, normalized=normalized))
    return write_store(root, files, variant="plain" if normalized else "plain-raw",
                       extra_manifest={"corpus_set": corpus_set.name, "corpus_fingerprint": fp,
                                       "normalized": normalized})


# --------------------------------------------------------------------------
# grep
# --------------------------------------------------------------------------

@dataclass
class GrepHit:
    """One matching line."""
    path: str          # rel path inside the store
    doc_id: str
    section_id: str
    line: int          # 1-based
    text: str          # the matching line
    unit_id: str       # "" for header/meta lines
    tag: str           # line-map tag of that line
    submatches: list   # [[start, end], ...] character offsets in `text`

    def to_dict(self):
        return asdict(self)


import re as _re

_RE_EVENT_KEY = _re.compile(rb'"path":\{"text":"((?:[^"\\]|\\.)*)"\}.*?"line_number":(\d+)')


def rg_binary(cfg=None):
    cfg = cfg or load_config()
    env = os.environ.get("RAGBENCH_RG")
    if env and Path(env).is_file():
        return env
    p = cfg.path("rg")
    if p.is_file():
        return str(p)
    import shutil
    w = shutil.which("rg")
    if w:
        return w
    raise FileNotFoundError("ripgrep not found (config paths.rg, env RAGBENCH_RG or PATH)")


class _StoreIndex:
    _cache = {}

    @classmethod
    def get(cls, root):
        root = Path(root)
        mp = root / "manifest.json"
        key = str(root)
        mtime = mp.stat().st_mtime_ns if mp.is_file() else None
        hit = cls._cache.get(key)
        if hit and hit[0] == mtime:
            return hit[1]
        man = read_json(mp, {"files": {}})
        lm = {}
        for rel, info in man["files"].items():
            doc_dir = rel.split("/")[0]
            if doc_dir not in lm:
                lm[doc_dir] = read_json(root / doc_dir / "_linemap.json", {})
        obj = {"manifest": man, "linemaps": lm,
               "by_section": {info["section_id"]: rel for rel, info in man["files"].items()}}
        cls._cache[key] = (mtime, obj)
        return obj


def line_info(root, rel_path, line):
    """(unit_id, tag) of a 1-based line in a store file."""
    idx = _StoreIndex.get(root)
    spans = idx["linemaps"].get(rel_path.split("/")[0], {}).get(rel_path, [])
    for a, b, uid, tag in spans:
        if a <= line <= b:
            return uid, tag
    return "", "blank"


def grep(root, pattern, *, doc_ids=None, paths=None, globs=None, tags=None, ignore_case=True, fixed_strings=False,
         pcre2=False, max_hits=None, timeout_s=60, cfg=None):
    """Run ripgrep over a store. `pattern` may be a string or a list (OR-ed).
    doc_ids / paths restrict the files searched (an EMPTY list searches nothing and returns [];
    None searches the whole store); tags keeps only hits on lines with those line-map tags
    (e.g. ["body"] to ignore @@ header lines). Hits come back in file-path order then line order
    (sorted here, so ripgrep can search multi-threaded). Raises ValueError on an invalid regex."""
    cfg = cfg or load_config()
    root = Path(root)
    if (doc_ids is not None and not paths and len(doc_ids) == 0) or (paths is not None and len(paths) == 0):
        return []
    idx = _StoreIndex.get(root)
    files = idx["manifest"]["files"]
    cmd = [rg_binary(cfg), "--json", "--no-config", "--no-ignore", "--hidden", "-g", "*.txt"]
    if ignore_case:
        cmd.append("-i")
    if fixed_strings:
        cmd.append("-F")
    if pcre2:
        cmd.append("-P")
    for g in globs or []:
        cmd += ["-g", g]
    pats = [pattern] if isinstance(pattern, str) else list(pattern)
    for p in pats:
        cmd += ["-e", p]
    targets = []
    if paths:
        targets = [str(root / p) for p in paths]
    elif doc_ids:
        targets = [str(root / safe_name(d)) for d in doc_ids if (root / safe_name(d)).is_dir()]
        if not targets:
            return []
    else:
        targets = [str(root)]
    cmd += ["--"] + targets
    proc = subprocess.run(cmd, capture_output=True, timeout=timeout_s)
    if proc.returncode == 2:
        err = proc.stderr.decode("utf-8", "replace")
        raise ValueError("ripgrep error: %s" % err.strip()[:500])
    # ripgrep's own --sort path is single-threaded (about 9x slower on large stores); instead the
    # match events are ordered here by (path, line) from a cheap scan, and only as many as needed
    # are fully parsed
    # order = targets in the order given (as rg --sort path did), then path, then line
    rank = {}
    if paths:
        rank = {str(p).replace("\\", "/"): i for i, p in enumerate(paths)}
    elif doc_ids:
        rank = {safe_name(d): i for i, d in enumerate(doc_ids)}
    root_s = str(root)
    keyed = []
    for raw in proc.stdout.splitlines():
        if not raw.startswith(b'{"type":"match"'):
            continue
        m = _RE_EVENT_KEY.search(raw)
        try:
            if m:
                abspath, line_no = json.loads(b'"' + m.group(1) + b'"'), int(m.group(2))
            else:                               # unusual encoding (e.g. non-UTF-8 path as bytes): parse fully
                ev = json.loads(raw)
                abspath, line_no = ev["data"]["path"].get("text") or "", int(ev["data"]["line_number"])
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            continue
        rel = (abspath[len(root_s) + 1:] if abspath.startswith(root_s) else os.path.relpath(abspath, root)).replace("\\", "/")
        r = rank.get(rel if paths else rel.split("/")[0], 0) if rank else 0
        keyed.append((r, tuple(rel.split("/")), line_no, raw))
    keyed.sort(key=lambda k: (k[0], k[1], k[2]))
    hits = []
    for _r, _p, _l, raw in keyed:
        try:
            ev = json.loads(raw)
        except json.JSONDecodeError:
            continue
        d = ev["data"]
        abspath = d["path"].get("text") or ""
        rel = os.path.relpath(abspath, root).replace("\\", "/")
        info = files.get(rel)
        if info is None:
            continue
        text = (d["lines"].get("text") or "").rstrip("\n").rstrip("\r")
        # rg reports byte offsets; convert to character offsets
        tb = text.encode("utf-8")
        subs = [[len(tb[:m["start"]].decode("utf-8", "ignore")), len(tb[:m["end"]].decode("utf-8", "ignore"))]
                for m in d.get("submatches", [])]
        uid, tag = line_info(root, rel, d["line_number"])
        if tags and tag not in tags:
            continue
        hits.append(GrepHit(rel, info["doc_id"], info["section_id"], d["line_number"], text, uid, tag, subs))
        if max_hits and len(hits) >= max_hits:
            break
    return hits


def read_lines(root, rel_path, start, end):
    """Lines [start, end] (1-based, inclusive, clamped) of a store file -> (text, first, last, unit_ids)."""
    p = Path(root) / rel_path
    with open(p, encoding="utf-8") as fh:
        lines = fh.read().split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    a = max(1, int(start))
    b = min(len(lines), int(end))
    idx = _StoreIndex.get(root)
    spans = idx["linemaps"].get(rel_path.split("/")[0], {}).get(rel_path, [])
    uids = []
    for s0, s1, uid, tag in spans:
        if uid and s0 <= b and s1 >= a and uid not in uids:
            uids.append(uid)
    return "\n".join(lines[a - 1:b]), a, b, uids


def section_file(root, section_id):
    """rel path of a section's file in a store, or None."""
    return _StoreIndex.get(root)["by_section"].get(section_id)
