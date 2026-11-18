"""Ingestion: a report on disk -> answerable rows in `work/kb/kb.sqlite`, with no model call anywhere.

Eight stages (AGENTIC3_SPEC.md section 4.2). Each one is timed into `ingest_log` and committed on its own, so a
kill at any moment loses only the stage that was running and a restart redoes just that stage:

    register   sha256 of the source; company / year / language from the arguments, else from the file name pattern
               `<company>_<type>_<year>_<lang>.<ext>`, else from the title; the doc id `<corpus>.<key>.<lang>`
    extract    `tools/extract/extract.py` for a PDF or an HTML page, the `kareone` adapter for a KareOne report
               folder; both are idempotent, so an extraction that is already on disk is reused as it stands
    model      `corpus/extracted.py::build_extracted` -> Document (section tree + units), cached in
               `work/corpus/docs/<doc_id>.json` like every other Document
    load       ONE transaction: `documents`, `sections` (parent_id, path), `units`, and their `tri` rows
    facts      `ours/lite.py::extract_facts` per section -> `kb.typed` -> `facts` rows + `tri` rows
    passages   `agentic2/tree.py::build_passages` -> `passages` rows, `vec` still NULL
    ready      `documents.status = 'ready'`: the document is now answerable through the database path
    embed      bge-m3 over the passages that have no vector, on demand -> `status = 'vector_ready'`

Identity and idempotency. A document is identified by (doc_id, `content_sha1` = sha1 of `Document.text`) plus the
component versions of `schema.component_versions`. Same id, same content, current versions, status ready ->
nothing runs. Same id, NEW content -> one transaction deletes the old rows (ON DELETE CASCADE plus the trigram
trigger) and inserts the new ones, and the document stays invisible (`status = 'loading'`) until `ready`;
`verified_facts` rows keep the old `content_sha1` and simply stop matching. Same id, same content, a component
version bumped -> only the affected stages run (`schema.stale_stages`), never the extraction.

Two facts about the L0 extractor that this module has to get right:
  * `typed.repair_separators` is applied to a unit's text BEFORE `extract_facts`. The L0 number regex splits
    "2،788،550" into three numbers, so the repair has to happen upstream of it. Both repairs replace one character
    with one character, so `facts.char_start` / `char_end` still index into `units.text` as stored.
  * table units go through `normalize_lines`, not `normalize`: `normalize` collapses newlines and
    `lite.is_table_text` then no longer sees a table.

Fairness (OWNERSHIP rule 1). Only what every approach may see reaches a column: doc id, title, text, tree, unit
kind. `Document.meta` is question-authoring provenance and is never copied. `company` and `year_from/year_to` come
from `agentic/docmeta.py::doc_meta`, i.e. from the doc id and the title, exactly as the planner derives them today.
`units.table_id` is read off the unit's own local id ("<block>#table.<k>"), not off its meta.

Isolation. A file ingested with `add` gets corpus letter "U" and a row in `corpus/natural/manifest_user.csv`, so it
can never enter a benchmark set (`sync` loads A_fa, A_en, B_fa, B_en and C from the SAME cached Document objects
the other approaches use, so unit ids and evidence recall are identical).

Quality gate. Extraction verdict `fail` -> a `documents` row with `status = 'rejected'` and the reasons in the
`ingest_log`, nothing else loaded. Verdict `suspect` -> loaded with `quality = 'suspect'`.
"""
import csv
import hashlib
import json
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from ..config import load_config
from ..corpus import extracted, kareone
from ..corpus.model import Document, make_doc_id
from ..corpus.registry import CORPUS_LETTERS
from ..normalize import fold, normalize, normalize_lines
from ..approaches.agentic.docmeta import doc_key, doc_meta
from ..approaches.agentic.numbers import find_numbers
from ..approaches.agentic2.defaults import DEFAULTS as A2_DEFAULTS
from ..approaches.agentic2.tree import build_passages
from ..approaches.ours.lite import extract_facts
from ..util import sha1_text
from . import labels as L
from . import schema as S
from . import typed as T

USER_CORPUS = "U"                       # a document added by a user, never part of a benchmark set
BENCH_SETS = ("A_fa", "A_en", "B_fa", "B_en", "C")
DEFAULT_KB_PATH = "work/kb/kb.sqlite"
DEFAULT_INBOX = "corpus/inbox"
USER_MANIFEST = "manifest_user.csv"     # beside corpus/natural/manifest.csv, which stays the benchmark's
MANIFEST_COLUMNS = ("lane", "company", "title_fa", "title_en", "year_persian", "year_gregorian", "language",
                    "parallel_file", "format", "pages", "size_kb", "source_url", "landing_url", "local_path",
                    "sha256", "notes")

# <company>_<type>_<year>_<lang>.<ext>, the corpus/natural naming rule
_FILENAME = re.compile(r"^(?P<company>[^_]+)_(?P<kind>.+)_(?P<year>1[34]\d{2}|(?:19|20)\d{2})_(?P<lang>fa|en)$")
_NUM_TOKEN = re.compile(r"\d")


def _now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _ms(t0):
    return round((time.perf_counter() - t0) * 1000.0, 1)


# ---------------------------------------------------------------------------------------------------------------
# the file
# ---------------------------------------------------------------------------------------------------------------

def kb_path(cfg=None):
    """`kb.path` from config.yaml, or work/kb/kb.sqlite. config.yaml is frozen and has no `kb` block yet."""
    cfg = cfg or load_config()
    p = Path(cfg.get("kb.path", DEFAULT_KB_PATH))
    return p if p.is_absolute() else cfg.project_root / p


def inbox_path(cfg=None):
    cfg = cfg or load_config()
    p = Path(cfg.get("kb.inbox", DEFAULT_INBOX))
    return p if p.is_absolute() else cfg.project_root / p


def open_kb(cfg=None, path=None, read_only=False):
    """-> a connection to the knowledge base, schema applied (writable connections only)."""
    cfg = cfg or load_config()
    con = S.connect(path or kb_path(cfg), read_only=read_only)
    if not read_only:
        S.ensure_schema(con, cfg)
    return con


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------------------------------------------
# stage timing
# ---------------------------------------------------------------------------------------------------------------

def log_stage(con, doc_id, name, started, ms, rows=0, ok=1, detail=None):
    if isinstance(detail, (dict, list)):
        detail = json.dumps(detail, ensure_ascii=False)
    con.execute("INSERT INTO ingest_log (doc_id, stage, started_at, ms, rows, ok, detail) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (doc_id, name, started, ms, int(rows), int(ok), detail))
    con.commit()


class stage:
    """Times one stage into `ingest_log` and commits it.

        with stage(con, doc_id, "facts") as st:
            st.rows = n

    The row is written and committed whether the body succeeds or raises, so an interrupted ingest leaves its own
    evidence. `ok` is 0 on an exception and the exception text goes into `detail`.
    """

    def __init__(self, con, doc_id, name, detail=None):
        self.con, self.doc_id, self.name = con, doc_id, name
        self.rows = 0
        self.detail = detail
        self.ms = None

    def __enter__(self):
        self.started = _now()
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.ms = _ms(self.t0)
        detail = "%s: %s" % (type(exc).__name__, exc) if exc is not None else self.detail
        log_stage(self.con, self.doc_id, self.name, self.started, self.ms, self.rows, 0 if exc is not None else 1,
                  detail)
        return False


# ---------------------------------------------------------------------------------------------------------------
# stage 1: register
# ---------------------------------------------------------------------------------------------------------------

def parse_filename(path):
    """-> {company, kind, year, lang} from `<company>_<type>_<year>_<lang>.<ext>`; empty values when it does not fit."""
    m = _FILENAME.match(Path(path).stem)
    if not m:
        return {"company": None, "kind": None, "year": None, "lang": None}
    d = m.groupdict()
    return {"company": d["company"], "kind": d["kind"], "year": int(d["year"]), "lang": d["lang"]}


def read_manifest_rows(path):
    if not Path(path).is_file():
        return []
    with open(path, encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def _rel_key(p):
    return str(p).replace("\\", "/").strip().lower()


def _manifest_lookup(cfg, local_path):
    """The manifest row (benchmark first, then the user manifest) for a path relative to corpus/natural."""
    want = _rel_key(local_path)
    for name in (cfg.path("manifest"), cfg.path("manifest").parent / USER_MANIFEST):
        for row in read_manifest_rows(name):
            if _rel_key(row.get("local_path") or "") == want:
                return row, Path(name)
    return None, None


def _append_user_manifest(cfg, row):
    """Append a row to corpus/natural/manifest_user.csv (created with the manifest's columns if it is not there).

    The benchmark manifest is deliberately left alone: `registry._load_C` builds corpus C from every Persian row of
    it, so a user document written there would silently join a benchmark set.
    """
    path = cfg.path("manifest").parent / USER_MANIFEST
    exists = path.is_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(MANIFEST_COLUMNS), quoting=csv.QUOTE_ALL)
        if not exists:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in MANIFEST_COLUMNS})
    return path


def register(source, cfg, *, company=None, year=None, lang=None, corpus=USER_CORPUS, key=None, title=None,
             pair_of=None):
    """Stage 1. -> {doc_id, corpus, key, lang, company, year, kind, source_path, source_sha256, local_path,
    manifest, manifest_path, manifest_added, natural_path}.

    A folder is a KareOne report (kind "kareone"); a file is a PDF or an HTML page. The file is copied under
    `corpus/natural/<company>/` when it is not there already, because `tools/extract/extract.py` addresses its
    inputs by a path relative to the manifest's folder.
    """
    src = Path(source).resolve()
    if not src.exists():
        raise FileNotFoundError(str(src))
    if src.is_dir():
        return _register_kareone(src, cfg, company=company, year=year, corpus=corpus, key=key, title=title)
    parsed = parse_filename(src)
    company = company or parsed["company"]
    lang = lang or parsed["lang"]
    year = int(year) if year is not None else parsed["year"]
    natural = cfg.path("manifest").parent
    try:
        local_path = src.relative_to(natural.resolve()).as_posix()
        company = company or src.parent.name
    except ValueError:                        # bring the file under corpus/natural so extract.py can address it
        company = company or src.stem.split("_")[0] or "user"
        dest = natural / company / src.name
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.is_file() or dest.stat().st_size != src.stat().st_size:
            dest.write_bytes(src.read_bytes())
        src = dest
        local_path = src.relative_to(natural).as_posix()
    row, manifest_path = _manifest_lookup(cfg, local_path)
    if row is not None:
        lang = lang or (row.get("language") or "").strip().lower() or None
        title = title or (row.get("title_%s" % (lang or "fa")) or "").strip() or None
        if year is None:
            y = (row.get("year_persian") or "").strip()
            year = int(y) if y.isdigit() else None
    lang = lang or "fa"
    title = title or src.stem
    manifest_added = False
    if row is None:
        row = {"lane": "user_added", "company": company, "title_%s" % lang: title, "year_persian": str(year or ""),
               "language": lang, "format": src.suffix.lstrip(".").lower(), "local_path": local_path,
               "sha256": sha256_file(src), "notes": "added with `python -m ragbench.kb add` on %s" % _now()}
        manifest_path = _append_user_manifest(cfg, row)
        manifest_added = True
    stem_key = Path(local_path).stem
    if lang and stem_key.endswith("_" + lang):
        stem_key = stem_key[: -(len(lang) + 1)]
    key = key or stem_key
    return {"doc_id": make_doc_id(corpus, key, lang), "corpus": corpus, "key": key, "lang": lang,
            "company": company, "year": year, "title": title, "kind": "file",
            "source_path": str(src).replace("\\", "/"), "source_sha256": row.get("sha256") or sha256_file(src),
            "local_path": local_path, "manifest": row, "manifest_path": str(manifest_path),
            "manifest_added": manifest_added, "pair_id": pair_of, "natural_root": str(natural)}


def _register_kareone(folder, cfg, *, company=None, year=None, corpus=USER_CORPUS, key=None, title=None):
    key = key or folder.name
    return {"doc_id": None, "corpus": corpus, "key": key, "lang": None, "company": company or key, "year": year,
            "title": title, "kind": "kareone", "source_path": str(folder).replace("\\", "/"),
            "source_sha256": kareone.fingerprint(folder), "local_path": None, "manifest": None,
            "manifest_path": None, "manifest_added": False, "pair_id": None, "natural_root": None}


# ---------------------------------------------------------------------------------------------------------------
# stage 2: extract
# ---------------------------------------------------------------------------------------------------------------

def extraction_verdict(doc_dir):
    """-> (verdict, reasons). `verdict` is ok | suspect | fail | None (nothing extracted yet)."""
    st = extracted.read_status(doc_dir) or {}
    reasons = []
    q = Path(doc_dir) / "quality.json"
    if q.is_file():
        try:
            with open(q, encoding="utf-8") as fh:
                reasons = ((json.load(fh).get("document") or {}).get("reasons") or [])[:6]
        except ValueError:
            reasons = []
    return st.get("verdict"), reasons


_EXTRACTOR_PYTHON = []


def extractor_python(cfg):
    """The interpreter that runs tools/extract/extract.py: the first one that can import PyMuPDF.

    `sys.executable` is tried first, then `kb.extractor_python`, then the interpreter this virtual environment was
    created from (`pyvenv.cfg` `home`), where PyMuPDF is installed on this machine. The extractor is a standalone
    script and imports nothing from `ragbench`, so running it outside the venv is the normal case, not a fallback.
    """
    if _EXTRACTOR_PYTHON:
        return _EXTRACTOR_PYTHON[0]
    cands = [sys.executable, cfg.get("kb.extractor_python")]
    cfgfile = Path(sys.prefix) / "pyvenv.cfg"
    if cfgfile.is_file():
        for line in cfgfile.read_text(encoding="utf-8").splitlines():
            if line.split("=")[0].strip() == "home":
                cands.append(str(Path(line.split("=", 1)[1].strip()) / "python.exe"))
    for c in cands:
        if c and Path(c).is_file() and subprocess.run([c, "-c", "import fitz"], capture_output=True).returncode == 0:
            _EXTRACTOR_PYTHON.append(c)
            return c
    raise RuntimeError("no interpreter with PyMuPDF (fitz) found; tried %s" % [c for c in cands if c])


def extract(reg, cfg, force=False):
    """Stage 2. Runs tools/extract/extract.py unless the output folder is already complete for this sha256.

    -> {doc_dir, verdict, reasons, ran, output}. The extractor is idempotent and atomic on its own (it writes a
    temporary sibling folder and renames it into place), so this is a plain subprocess call; its one JSON status
    line is read, because an extractor error next to an older extraction would otherwise pass unnoticed.
    """
    root = cfg.path("extracted_root")
    doc_dir = extracted.out_dir_for(root, reg["local_path"])
    usable, _why = extracted.is_usable(doc_dir, exclude_verdicts=())
    if usable and not force:
        verdict, reasons = extraction_verdict(doc_dir)
        return {"doc_dir": doc_dir, "verdict": verdict, "reasons": reasons, "ran": False, "output": "cached"}
    cmd = [extractor_python(cfg), str(cfg.project_root / "tools" / "extract" / "extract.py"),
           "--manifest", reg["manifest_path"], "--out", str(root), "--files", reg["local_path"]]
    if force:
        cmd.append("--force")
    p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace",
                       cwd=str(cfg.project_root))
    out = (p.stdout or "").strip().splitlines()
    last = out[-1] if out else ""
    try:
        res = json.loads(last)
    except ValueError:
        res = {"status": "error", "error": (last or p.stderr or "")[:400]}
    if res.get("status") == "error":
        raise RuntimeError("extraction failed: %s" % res.get("error"))
    verdict, reasons = extraction_verdict(doc_dir)
    if verdict is None:
        raise RuntimeError("extraction produced nothing: %s" % last[:400])
    return {"doc_dir": doc_dir, "verdict": verdict, "reasons": reasons, "ran": True,
            "output": res, "seconds": res.get("seconds")}


# ---------------------------------------------------------------------------------------------------------------
# stage 3: model
# ---------------------------------------------------------------------------------------------------------------

def doc_cache_path(cfg, doc_id):
    return cfg.path("corpus_cache") / "docs" / ("%s.json" % doc_id)


def build_document(reg, doc_dir, cfg):
    """Stage 3. -> Document, also written to work/corpus/docs so a later rebuild of one stage needs no extraction."""
    doc = extracted.build_extracted(doc_dir, reg["doc_id"], reg["corpus"], reg["lang"], title=reg.get("title"),
                                    pair_id=reg.get("pair_id"),
                                    rows_per_unit=cfg.corpora["table_rows_per_unit"])
    doc.meta["cfg_rows"] = cfg.corpora["table_rows_per_unit"]
    doc.save(doc_cache_path(cfg, doc.id))
    return doc


def cached_document(cfg, doc_id):
    p = doc_cache_path(cfg, doc_id)
    return Document.load(p) if p.is_file() else None


# ---------------------------------------------------------------------------------------------------------------
# stage 4: load
# ---------------------------------------------------------------------------------------------------------------

def _table_id(unit):
    """The table a unit belongs to, read off its own local id ("p012_t1#table.3" -> "p012_t1"); never off its meta."""
    local = unit.local_id
    return local.split("#table.", 1)[0] if "#table." in local else None


def _section_rows(doc):
    for s in doc.sections:
        path = " > ".join(t for t in doc.section_path(s.id) if t)
        yield (s.id, doc.id, s.parent, s.level, s.order, s.title, fold(s.title or ""), path, fold(path),
               s.span[0], s.span[1], s.subtree_span[1], len(s.unit_ids), 0)


def _unit_rows(doc):
    for u in doc.units:
        yield (u.id, doc.id, u.section_id, u.order, u.kind, u.lang, u.text, fold(u.text),
               u.span[0], u.span[1], _table_id(u), len(find_numbers(u.text)) if _NUM_TOKEN.search(u.text) else 0)


def document_row(doc, *, source_path, source_sha256, quality, status, versions, content_sha1):
    meta = doc_meta(doc)                       # company and years from the id and the title only (fairness rule 1)
    years = meta["years"] or (None, None)
    title = normalize(doc.title or "")
    return (doc.id, doc.corpus, doc_key(doc.id), doc.lang, meta["company"], years[0], years[1],
            T.calendar_of(years[0]), title, fold(title), doc.pair_id, source_path, source_sha256, content_sha1,
            quality, status, len(doc.sections), len(doc.units), 0, len(doc.text),
            json.dumps(versions, sort_keys=True), _now())


def load_document(con, doc, cfg, *, source_path, source_sha256, quality, versions, content_sha1):
    """Stage 4, one transaction: the old rows of this doc_id go, the new documents / sections / units / tri rows
    arrive, and the document stays invisible (`status = 'loading'`) until the `ready` stage."""
    con.execute("DELETE FROM documents WHERE doc_id = ?", (doc.id,))
    con.execute("INSERT INTO documents VALUES (%s)" % ",".join("?" * 22),
                document_row(doc, source_path=source_path, source_sha256=source_sha256, quality=quality,
                             status="loading", versions=versions, content_sha1=content_sha1))
    secs = list(_section_rows(doc))
    units = list(_unit_rows(doc))
    con.executemany("INSERT INTO sections VALUES (%s)" % ",".join("?" * 14), secs)
    con.executemany("INSERT INTO units VALUES (%s)" % ",".join("?" * 12), units)
    con.executemany("INSERT INTO tri (text_fold, kind, ref_id, doc_id) VALUES (?, 'section_title', ?, ?)",
                    [(r[6], r[0], doc.id) for r in secs if r[6]])
    con.executemany("INSERT INTO tri (text_fold, kind, ref_id, doc_id) VALUES (?, 'unit', ?, ?)",
                    [(r[7], r[0], doc.id) for r in units if r[7]])
    con.commit()
    return {"sections": len(secs), "units": len(units)}


# ---------------------------------------------------------------------------------------------------------------
# stage 5: facts
# ---------------------------------------------------------------------------------------------------------------

def _section_units(con, doc_id):
    """-> [(section_id, title, [(unit_id, text)])] in document order, read from the database."""
    by_sec = {}
    for r in con.execute("SELECT unit_id, section_id, text FROM units WHERE doc_id = ? ORDER BY ord", (doc_id,)):
        by_sec.setdefault(r["section_id"], []).append((r["unit_id"], r["text"]))
    out = []
    for r in con.execute("SELECT section_id, title FROM sections WHERE doc_id = ? ORDER BY ord", (doc_id,)):
        out.append((r["section_id"], r["title"] or "", by_sec.get(r["section_id"], [])))
    return out


def build_facts(con, doc_id, cfg=None):
    """Stage 5, from the database alone (no Document needed). -> {facts, sections_with_facts, candidates}.

    `repair_separators` runs on every unit's text BEFORE `extract_facts`, because the L0 number regex would
    otherwise split "2،788،550" into "2", "788" and "550". Both repairs are one character for one character, so
    `facts.char_start` / `char_end` still index into `units.text` as it is stored.
    """
    row = con.execute("SELECT year_from, year_to FROM documents WHERE doc_id = ?", (doc_id,)).fetchone()
    doc_years = (row["year_from"], row["year_to"]) if row and row["year_from"] is not None else None
    if con.execute("SELECT 1 FROM facts WHERE doc_id = ? LIMIT 1", (doc_id,)).fetchone():
        con.execute("DELETE FROM tri WHERE doc_id = ? AND kind = 'fact_label'", (doc_id,))
        con.execute("DELETE FROM facts WHERE doc_id = ?", (doc_id,))
    # fact_id is allocated here rather than read back, so the trigram row of a fact is built in the same pass
    fact_id = (con.execute("SELECT coalesce(max(fact_id), 0) AS m FROM facts").fetchone()["m"]) + 1
    n_cand = 0
    per_section = {}
    rows, tri = [], []
    for section_id, title, units in _section_units(con, doc_id):
        if not units:
            continue
        repaired = {uid: T.repair_separators(normalize_lines(text or "")) for uid, text in units}
        facts, stats = extract_facts([(uid, repaired[uid]) for uid, _ in units], section_title=title)
        n_cand += stats["candidates"]
        for f in facts:
            typed = T.normalise_fact(f, unit_text=repaired.get(f["unit_id"], ""), doc_years=doc_years)
            rows.append((fact_id, doc_id, section_id, f["unit_id"], len(rows))
                        + tuple(typed[c] for c in S.FACT_COLUMNS[4:]))
            if typed["label_fold"]:
                tri.append((typed["label_fold"], str(fact_id), doc_id))
            fact_id += 1
        per_section[section_id] = per_section.get(section_id, 0) + len(facts)
    con.executemany("INSERT INTO facts (fact_id, %s) VALUES (%s)"
                    % (",".join(S.FACT_COLUMNS), ",".join("?" * (len(S.FACT_COLUMNS) + 1))), rows)
    con.executemany("INSERT INTO tri (text_fold, kind, ref_id, doc_id) VALUES (?, 'fact_label', ?, ?)", tri)
    con.execute("UPDATE sections SET n_facts = 0 WHERE doc_id = ?", (doc_id,))
    con.executemany("UPDATE sections SET n_facts = ? WHERE section_id = ?",
                    [(n, sid) for sid, n in per_section.items() if n])
    con.execute("UPDATE documents SET n_facts = ? WHERE doc_id = ?", (len(rows), doc_id))
    con.commit()
    return {"facts": len(rows), "sections_with_facts": sum(1 for n in per_section.values() if n),
            "candidates": n_cand}


# ---------------------------------------------------------------------------------------------------------------
# stage 6: passages
# ---------------------------------------------------------------------------------------------------------------

def passage_opts(cfg=None):
    return {k: A2_DEFAULTS[k] for k in ("passage_chars", "passage_max_chars", "passage_overlap_chars",
                                        "embed_with_path")}


def embed_text_of(title, path, text, with_path=True):
    """The text bge-m3 sees for a passage: "<doc title> > <section path>\\n<text>", as agentic2/tree.py builds it.

    Rebuilt at embed time from `documents.title`, `sections.path` and `passages.text`, and checked against
    `passages.embed_sha1`, so the embed stage needs no Document object.
    """
    crumb = " > ".join(p for p in (title or "", path or "") if p)
    return (crumb + "\n" + text) if (with_path and crumb) else text


def build_passage_rows(con, doc, cfg=None):
    """Stage 6. `vec` stays NULL: vectors are the only stage that costs GPU and they are filled on demand."""
    opts = passage_opts(cfg)
    title = con.execute("SELECT title FROM documents WHERE doc_id = ?", (doc.id,)).fetchone()["title"]
    paths = {r["section_id"]: r["path"] for r in
             con.execute("SELECT section_id, path FROM sections WHERE doc_id = ?", (doc.id,))}
    ps = build_passages(doc, opts)
    rows = []
    for p in ps:
        et = embed_text_of(title, paths.get(p.section_id, ""), p.text, opts["embed_with_path"])
        rows.append(("%s#p%04d" % (doc.id, p.pid), doc.id, p.section_id, p.pid,
                     json.dumps(p.unit_ids, ensure_ascii=False), p.text, sha1_text(et),
                     None, None, None, None))
    con.execute("DELETE FROM passages WHERE doc_id = ?", (doc.id,))
    con.executemany("INSERT INTO passages VALUES (%s)" % ",".join("?" * 11), rows)
    con.commit()
    return {"passages": len(rows), "chars": sum(len(r[5]) for r in rows)}


# ---------------------------------------------------------------------------------------------------------------
# stages 4-7 for one Document
# ---------------------------------------------------------------------------------------------------------------

def _versions(cfg):
    return S.component_versions(cfg)


def ingest_document(con, doc, cfg, *, source_path=None, source_sha256=None, quality="ok", force=False):
    """Stages 4-7 for a Document that is already built. -> {doc_id, action, stages, rows, ms}.

    `action` is "loaded" (new or replaced), "rebuilt" (only the stale stages ran) or "skipped" (same content, current
    versions, already ready). Every stage commits on its own; the document becomes visible at `ready`.
    """
    t0 = time.perf_counter()
    versions = _versions(cfg)
    content_sha1 = sha1_text(doc.text)
    row = con.execute("SELECT content_sha1, versions, status FROM documents WHERE doc_id = ?", (doc.id,)).fetchone()
    stale = S.stale_stages(row["versions"], current=versions) if row else S.STAGES
    same = bool(row) and row["content_sha1"] == content_sha1 and row["status"] in ("ready", "vector_ready")
    if same and not stale and not force:
        return {"doc_id": doc.id, "action": "skipped", "stages": {}, "rows": {}, "ms": _ms(t0)}
    todo = S.STAGES if (force or not same) else stale
    stages, rows = {}, {}
    if "embed" in todo and "passages" not in todo and "load" not in todo:
        con.execute("UPDATE passages SET vec = NULL, dim = NULL, model = NULL, embedded_at = NULL "
                    "WHERE doc_id = ?", (doc.id,))          # the embed model moved: the old vectors are not ours
        con.commit()
    if "load" in todo:
        with stage(con, doc.id, "load") as st:
            rows.update(load_document(con, doc, cfg, source_path=source_path, source_sha256=source_sha256,
                                      quality=quality, versions=versions, content_sha1=content_sha1))
            st.rows = rows["sections"] + rows["units"]
        stages["load"] = st.ms
    else:
        con.execute("UPDATE documents SET status = 'loading', versions = ? WHERE doc_id = ?",
                    (json.dumps(versions, sort_keys=True), doc.id))
        con.commit()
    if "facts" in todo:
        with stage(con, doc.id, "facts") as st:
            r = build_facts(con, doc.id, cfg)
            rows.update(r)
            st.rows = r["facts"]
            st.detail = {"candidates": r["candidates"]}
        stages["facts"] = st.ms
        # the label layer (labels.py) is derived from the facts just written, so it is rebuilt with them. It is
        # additive and versioned apart from component_versions: running it changes no fact and marks nothing stale.
        with stage(con, doc.id, "labels") as st:
            st.rows = L.build_doc(con, doc.id)["facts"]
        stages["labels"] = st.ms
        S.set_meta(con, "graph.dirty", "1")      # graph.py rebuilds once per command, not once per document
        con.commit()
    if "passages" in todo:
        with stage(con, doc.id, "passages") as st:
            r = build_passage_rows(con, doc, cfg)
            rows.update(r)
            st.rows = r["passages"]
        stages["passages"] = st.ms
    with stage(con, doc.id, "ready") as st:
        has_vec = con.execute("SELECT count(*) AS n FROM passages WHERE doc_id = ? AND vec IS NULL",
                              (doc.id,)).fetchone()["n"]
        con.execute("UPDATE documents SET status = ? WHERE doc_id = ?",
                    ("ready" if has_vec else "vector_ready", doc.id))
        con.commit()
        st.rows = 1
    stages["ready"] = st.ms
    return {"doc_id": doc.id, "action": "rebuilt" if same else "loaded", "stages": stages, "rows": rows,
            "chars": len(doc.text), "ms": _ms(t0)}


def rebuild_document(con, doc_id, cfg, stages_wanted=None, force=False):
    """Re-run the stale stages of an already loaded document from its cached Document. -> the ingest_document result."""
    doc = cached_document(cfg, doc_id)
    if doc is None:
        raise FileNotFoundError("no cached Document for %s (run `kb add` or `kb sync`)" % doc_id)
    row = con.execute("SELECT source_path, source_sha256, quality FROM documents WHERE doc_id = ?",
                      (doc_id,)).fetchone() or {}
    return ingest_document(con, doc, cfg, source_path=row.get("source_path"),
                           source_sha256=row.get("source_sha256"), quality=row.get("quality") or "ok", force=force)


# ---------------------------------------------------------------------------------------------------------------
# add: the whole pipeline for one file or KareOne folder
# ---------------------------------------------------------------------------------------------------------------

def _reject(con, reg, doc_id, reasons, cfg):
    con.execute("DELETE FROM documents WHERE doc_id = ?", (doc_id,))
    con.execute("INSERT INTO documents (doc_id, corpus, key, lang, company, title, source_path, source_sha256, "
                "quality, status, versions, ingested_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (doc_id, reg["corpus"], reg["key"], reg["lang"], reg["company"], reg.get("title") or "",
                 reg["source_path"], reg["source_sha256"], "fail", "rejected",
                 json.dumps(_versions(cfg), sort_keys=True), _now()))
    con.commit()
    return {"doc_id": doc_id, "action": "rejected", "reasons": reasons, "stages": {}, "rows": {}}


def add_file(con, source, cfg, *, company=None, year=None, lang=None, corpus=USER_CORPUS, key=None, title=None,
             pair_of=None, re_extract=False, force=False):
    """Stages 1-7 for one new file or KareOne report folder. -> the ingest_document result plus `register`/`extract`.

    A KareOne folder produces the fa and en editions together (the adapter builds both from one pass), so the
    result of that branch is a list under "documents".
    """
    t0 = time.perf_counter()
    started, t1 = _now(), time.perf_counter()
    reg = register(source, cfg, company=company, year=year, lang=lang, corpus=corpus, key=key, title=title,
                   pair_of=pair_of)
    register_ms = _ms(t1)
    log_stage(con, reg["doc_id"], "register", started, register_ms, 1 if reg["manifest_added"] else 0,
              detail={"sha256": reg["source_sha256"][:16], "manifest_added": reg["manifest_added"],
                      "kind": reg["kind"]})
    if reg["kind"] == "kareone":
        return _add_kareone(con, reg, cfg, register_ms, started, force=force, t0=t0)
    doc_id = reg["doc_id"]
    with stage(con, doc_id, "extract") as st:
        ex = extract(reg, cfg, force=re_extract)
        st.rows = 1 if ex["ran"] else 0
        st.detail = {"verdict": ex["verdict"], "ran": ex["ran"]}
    extract_ms = st.ms
    if ex["verdict"] == "fail":
        res = _reject(con, reg, doc_id, ex["reasons"], cfg)
        res.update({"register": reg, "extract": ex, "ms": _ms(t0),
                    "stages": {"register": register_ms, "extract": extract_ms}})
        return res
    with stage(con, doc_id, "model") as st:
        doc = build_document(reg, ex["doc_dir"], cfg)
        st.rows = len(doc.units)
        st.detail = {"sections": len(doc.sections), "chars": len(doc.text), "structure": doc.meta.get("structure")}
    model_ms = st.ms
    res = ingest_document(con, doc, cfg, source_path=reg["source_path"], source_sha256=reg["source_sha256"],
                          quality="suspect" if ex["verdict"] == "suspect" else "ok", force=force)
    res["stages"] = dict({"register": register_ms, "extract": extract_ms, "model": model_ms}, **res["stages"])
    res.update({"register": reg, "extract": ex, "ms": _ms(t0)})
    return res


def recorpus(doc, corpus):
    """The same Document under another corpus letter: every id re-prefixed, the tree untouched.

    `kareone.build_report` always builds corpus-A ids, so a KareOne folder handed to `kb add` would otherwise land
    on a benchmark doc id. Corpus "U" keeps a user report out of every benchmark set.
    """
    if doc.corpus == corpus:
        return doc
    old = doc.id
    new = make_doc_id(corpus, doc_key(old), doc.lang)

    def sw(i):
        return new + i[len(old):] if isinstance(i, str) and i.startswith(old) else i

    d = doc.to_dict()
    d["id"], d["corpus"] = new, corpus
    if d.get("pair_id"):
        d["pair_id"] = corpus + d["pair_id"][d["pair_id"].index("."):]
    for s in d["sections"]:
        s["id"], s["doc_id"], s["parent"] = sw(s["id"]), new, sw(s["parent"])
        s["children"] = [sw(c) for c in s["children"]]
        s["unit_ids"] = [sw(u) for u in s["unit_ids"]]
    for u in d["units"]:
        u["id"], u["doc_id"], u["section_id"] = sw(u["id"]), new, sw(u["section_id"])
    return Document.from_dict(d)


def _add_kareone(con, reg, cfg, register_ms, started, force=False, t0=None):
    t1 = time.perf_counter()
    pair = {l: recorpus(d, reg["corpus"]) for l, d in
            kareone.build_report(reg["key"], Path(reg["source_path"]),
                                 rows_per_unit=cfg.corpora["table_rows_per_unit"]).items()}
    model_ms = _ms(t1)
    out = []
    for lang in ("fa", "en"):
        doc = pair[lang]
        log_stage(con, doc.id, "model", started, model_ms, len(doc.units),
                  detail={"sections": len(doc.sections), "chars": len(doc.text)})
        doc.meta["cfg_rows"] = cfg.corpora["table_rows_per_unit"]
        doc.save(doc_cache_path(cfg, doc.id))
        r = ingest_document(con, doc, cfg, source_path=reg["source_path"], source_sha256=reg["source_sha256"],
                            force=force)
        r["stages"] = dict({"register": register_ms, "model": model_ms}, **r["stages"])
        out.append(r)
    return {"doc_id": None, "action": "loaded", "documents": out, "register": reg,
            "ms": _ms(t0) if t0 else None, "stages": out[0]["stages"] if out else {}}


# ---------------------------------------------------------------------------------------------------------------
# sync: the benchmark corpora, from the Document cache
# ---------------------------------------------------------------------------------------------------------------

def set_doc_ids(name, cfg):
    """Doc ids of a benchmark set, from work/corpus/sets/<name>.json when every cached Document is there.

    Falling back to `corpus.load_set` rebuilds the set manifest; it still reads the cached Documents and never
    re-extracts a PDF (the extraction outputs on disk are the adapter's input either way).
    """
    p = cfg.path("corpus_cache") / "sets" / ("%s.json" % name)
    if p.is_file():
        with open(p, encoding="utf-8") as fh:
            ids = [d["id"] for d in json.load(fh).get("docs") or []]
        if ids and all(doc_cache_path(cfg, i).is_file() for i in ids):
            return ids
    from ..corpus import load_set
    return load_set(name, cfg).doc_ids()


def sync_sets(con, names, cfg, *, force=False, log=None):
    """Load the benchmark corpora from the cached Documents, one document and one stage at a time.

    Resumable by construction: a document that is already loaded with the same content and current versions runs
    zero stages, so a restart after a kill continues exactly where the kill happened.
    """
    results = []
    for name in names:
        ids = set_doc_ids(name, cfg)
        for doc_id in ids:
            doc = cached_document(cfg, doc_id)
            if doc is None:
                results.append({"doc_id": doc_id, "action": "missing", "set": name})
                continue
            st = extracted.read_status(doc.source) or {}
            res = ingest_document(con, doc, cfg, source_path=doc.source, source_sha256=st.get("sha256"),
                                  quality="suspect" if st.get("verdict") == "suspect" else "ok", force=force)
            res["set"] = name
            results.append(res)
            if log:
                log(res)
    return results


# ---------------------------------------------------------------------------------------------------------------
# remove
# ---------------------------------------------------------------------------------------------------------------

CONTENT_TABLES = ("sections", "units", "facts", "passages")


def document_counts(con, doc_id):
    out = {t: con.execute("SELECT count(*) AS n FROM %s WHERE doc_id = ?" % t, (doc_id,)).fetchone()["n"]
           for t in CONTENT_TABLES}
    out["tri"] = con.execute("SELECT count(*) AS n FROM tri WHERE doc_id = ?", (doc_id,)).fetchone()["n"]
    out["documents"] = con.execute("SELECT count(*) AS n FROM documents WHERE doc_id = ?",
                                   (doc_id,)).fetchone()["n"]
    return out


def remove_document(con, doc_id):
    """Delete a document and everything the cascade and the trigram trigger own. -> {before, after}.

    `ingest_log` and `verified_facts` are kept on purpose: the log is the paper's measurement record and a verified
    fact keeps the content hash it was verified against, so it stops matching instead of being trusted stale.
    """
    before = document_counts(con, doc_id)
    con.execute("DELETE FROM documents WHERE doc_id = ?", (doc_id,))
    con.commit()
    return {"doc_id": doc_id, "before": before, "after": document_counts(con, doc_id)}


# ---------------------------------------------------------------------------------------------------------------
# stage 8: embed
# ---------------------------------------------------------------------------------------------------------------

class _EmbedKeys:
    """B3's embedding-cache location and key function, called unbound so no formula is copied (as lazyvec does)."""

    def __init__(self, cfg):
        self.cfg = cfg

    def path(self):
        from ..approaches.baselines import VectorRetrieval
        return VectorRetrieval._cache_path(self, None)

    def key(self, text):
        from ..approaches.baselines import VectorRetrieval
        return VectorRetrieval._text_key(self, text)


def pending_passages(con, doc_ids=None):
    """-> [row] of passages with no vector, joined to the title and section path that make up their embed text."""
    sql = ("SELECT p.passage_id, p.doc_id, p.text, p.embed_sha1, d.title, s.path "
           "FROM passages p JOIN documents d ON d.doc_id = p.doc_id "
           "LEFT JOIN sections s ON s.section_id = p.section_id WHERE p.vec IS NULL")
    args = ()
    if doc_ids is not None:
        ids = list(doc_ids)
        if not ids:
            return []
        sql += " AND p.doc_id IN (%s)" % ",".join("?" * len(ids))
        args = tuple(ids)
    return list(con.execute(sql + " ORDER BY p.doc_id, p.ord", args))


def embed_documents(con, doc_ids, cfg, *, client=None, log=None):
    """Stage 8. bge-m3 over every passage without a vector, committed per batch. -> stats.

    Vectors already in the shared embedding cache that B3 and AGENTIC2 fill are reused, so only genuinely new text
    costs GPU. Every model call goes through `OllamaClient.embed` inside a ledger scope of phase "index", which is
    what makes this show up in the ledger like any other index build.
    """
    import numpy as np
    from ..llm import OllamaClient
    rows = pending_passages(con, doc_ids)
    keys = _EmbedKeys(cfg)
    from ..util import KVCache
    cache = KVCache(keys.path())
    client = client or OllamaClient(cfg)
    opts = passage_opts(cfg)
    model = cfg.models["embed"]
    bs = int(cfg.embedding["batch_size"])
    stats = {"passages": len(rows), "from_cache": 0, "embedded": 0, "calls": 0, "tokens": 0, "embed_ms": 0.0,
             "mismatched_sha1": 0, "docs": 0}
    t0 = time.perf_counter()
    by_doc = {}
    for r in rows:
        by_doc.setdefault(r["doc_id"], []).append(r)
    try:
        with client.ledger.scope(phase="index", approach="KB", corpus_set="kb"):
            for doc_id, items in by_doc.items():
                with stage(con, doc_id, "embed") as st:
                    texts = []
                    for r in items:
                        et = embed_text_of(r["title"], r["path"], r["text"], opts["embed_with_path"])
                        if sha1_text(et) != r["embed_sha1"]:
                            stats["mismatched_sha1"] += 1
                        texts.append(et)
                    todo = [(i, t) for i, t in enumerate(texts) if cache.get_array("v:" + keys.key(t)) is None]
                    stats["from_cache"] += len(texts) - len(todo)
                    for j in range(0, len(todo), bs):
                        batch = todo[j:j + bs]
                        er = client.embed([t for _, t in batch], purpose="kb-index-embed", batch_size=bs)
                        for (_, t), v in zip(batch, er.vectors):
                            arr = np.asarray(v, dtype=np.float32)
                            cache.put_array("v:" + keys.key(t), arr)
                            cache.put("c:" + keys.key(t), {"tokens": er.prompt_tokens / len(batch),
                                                           "wall_ms": er.wall_ms / len(batch)})
                        stats["calls"] += er.calls
                        stats["tokens"] += er.prompt_tokens
                        stats["embed_ms"] += er.wall_ms
                        stats["embedded"] += len(batch)
                    now = _now()
                    for r, t in zip(items, texts):
                        arr = cache.get_array("v:" + keys.key(t))
                        if arr is None:
                            continue
                        con.execute("UPDATE passages SET vec = ?, dim = ?, model = ?, embedded_at = ? "
                                    "WHERE passage_id = ?",
                                    (arr.astype(np.float32).tobytes(), int(arr.shape[0]), model, now,
                                     r["passage_id"]))
                    con.execute("UPDATE documents SET status = 'vector_ready' WHERE doc_id = ? AND status = 'ready' "
                                "AND NOT EXISTS (SELECT 1 FROM passages WHERE doc_id = ? AND vec IS NULL)",
                                (doc_id, doc_id))
                    con.commit()
                    st.rows = len(items)
                    st.detail = {"embedded": len(todo), "cached": len(items) - len(todo),
                                 "embed_ms": round(stats["embed_ms"], 1)}
                stats["docs"] += 1
                if log:
                    log({"doc_id": doc_id, "passages": len(items), "embedded": len(todo), "ms": st.ms})
    finally:
        cache.close()
    stats["wall_ms"] = _ms(t0)
    return stats


def question_doc_ids(con, files, cfg):
    """Doc ids the given question files reference, the way `run.py --docs-from-questions` derives them.

    Keys are language-free, so both editions of an A or B document are included; only documents that are actually
    in the knowledge base come back.
    """
    from ..contracts import load_questions
    from ..run import referenced_doc_keys
    qs = [q for f in files for q in load_questions(f)]
    want = {(c, k) for c in CORPUS_LETTERS for k in referenced_doc_keys(qs, c)}
    return [r["doc_id"] for r in con.execute("SELECT doc_id, corpus, key FROM documents ORDER BY doc_id")
            if (r["corpus"], r["key"]) in want]


# ---------------------------------------------------------------------------------------------------------------
# status and the cost report
# ---------------------------------------------------------------------------------------------------------------

TABLES = ("documents", "sections", "units", "facts", "passages", "tri", "verified_facts", "ingest_log")


def db_size_bytes(path):
    p = Path(path)
    return sum(f.stat().st_size for f in (p, Path(str(p) + "-wal"), Path(str(p) + "-shm")) if f.is_file())


def _quantiles(values):
    if not values:
        return {"n": 0, "median": None, "mean": None, "max": None, "total": None}
    v = sorted(values)
    n = len(v)
    median = v[n // 2] if n % 2 else (v[n // 2 - 1] + v[n // 2]) / 2.0
    return {"n": n, "median": round(median, 1), "mean": round(sum(v) / n, 1), "max": round(v[-1], 1),
            "total": round(sum(v), 1)}


def stage_timings(con):
    """-> {stage: {n, median, mean, max, total, rows}} over the successful ingest_log rows, latest per doc+stage."""
    out = {}
    for name in ("register", "extract", "model") + S.STAGES + ("ready",):
        rows = list(con.execute(
            "SELECT ms, rows FROM ingest_log WHERE stage = ? AND ok = 1 AND id IN "
            "(SELECT max(id) FROM ingest_log WHERE stage = ? AND ok = 1 GROUP BY doc_id)", (name, name)))
        if not rows:
            continue
        q = _quantiles([r["ms"] for r in rows])
        q["rows"] = sum(r["rows"] or 0 for r in rows)
        out[name] = q
    return out


def status(con, cfg=None):
    cfg = cfg or load_config()
    counts = {t: con.execute("SELECT count(*) AS n FROM %s" % t).fetchone()["n"] for t in TABLES}
    by_status = {r["status"]: r["n"] for r in
                 con.execute("SELECT status, count(*) AS n FROM documents GROUP BY status ORDER BY status")}
    by_corpus = [dict(r) for r in con.execute(
        "SELECT corpus, count(*) AS docs, sum(n_sections) AS sections, sum(n_units) AS units, "
        "sum(n_facts) AS facts, sum(n_chars) AS chars FROM documents GROUP BY corpus ORDER BY corpus")]
    vec = con.execute("SELECT count(*) AS n, sum(vec IS NOT NULL) AS with_vec FROM passages").fetchone()
    return {"path": str(kb_path(cfg)), "size_bytes": db_size_bytes(kb_path(cfg)), "counts": counts,
            "by_status": by_status, "by_corpus": by_corpus, "stale": len(S.stale_documents(con, cfg)),
            "passages_with_vec": vec["with_vec"] or 0, "passages": vec["n"],
            "versions": S.component_versions(cfg), "stages": stage_timings(con)}


def extraction_costs(cfg=None):
    """Stage 2 over every extraction on disk: the seconds `tools/extract/extract.py` recorded for each one.

    Stage 2 is the only stage that is not re-run per ingest (the extractor is idempotent), so its cost is read from
    the `extract_status.json` files it wrote rather than from `ingest_log`.
    """
    cfg = cfg or load_config()
    secs, verdicts = [], {}
    for p in sorted(cfg.path("extracted_root").glob("*/*/extract_status.json")):
        try:
            with open(p, encoding="utf-8") as fh:
                st = json.load(fh)
        except ValueError:
            continue
        verdicts[st.get("verdict")] = verdicts.get(st.get("verdict"), 0) + 1
        if st.get("seconds") is not None:
            secs.append(float(st["seconds"]) * 1000.0)
    return {"seconds": _quantiles(secs), "verdicts": verdicts}


def per_document_rates(con):
    """-> [{doc_id, chars, facts, units, stage ms...}] for the throughput table."""
    per = {}
    for r in con.execute("SELECT doc_id, stage, ms FROM ingest_log WHERE ok = 1 AND doc_id IS NOT NULL AND id IN "
                         "(SELECT max(id) FROM ingest_log WHERE ok = 1 GROUP BY doc_id, stage)"):
        per.setdefault(r["doc_id"], {})[r["stage"]] = r["ms"]
    out = []
    for r in con.execute("SELECT doc_id, n_chars, n_facts, n_units, n_sections FROM documents ORDER BY doc_id"):
        d = per.get(r["doc_id"])
        if d and "load" in d:
            out.append(dict(r, **d))
    return out


def _md_table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    for r in rows:
        out.append("| " + " | ".join("" if c is None else str(c) for c in r) + " |")
    return "\n".join(out)


def cost_markdown(con, cfg=None):
    """work/runs/core/INGEST_COST.md: per stage timings and rows, per table counts, and the headline latency."""
    cfg = cfg or load_config()
    st = status(con, cfg)
    lines = ["# Ingestion cost (AGENTIC3 M2)", "",
             "Measured by `ragbench/kb/ingest.py`, one `ingest_log` row per stage per document, on this machine",
             "(%s). No model call is made in stages 1-7; stage 8 (embed) is the only GPU work." % _now(), ""]
    lines += ["Knowledge base: `%s`, %.1f MB." % (st["path"], st["size_bytes"] / 1e6), "",
              "## Rows per table", "", _md_table(["table", "rows"], sorted(st["counts"].items())), "",
              "## Documents", "",
              _md_table(["corpus", "docs", "sections", "units", "facts", "chars"],
                        [(r["corpus"], r["docs"], r["sections"], r["units"], r["facts"], r["chars"])
                         for r in st["by_corpus"]]),
              "", "Status: " + ", ".join("%s %d" % kv for kv in sorted(st["by_status"].items())) +
              "; passages with a vector %d of %d." % (st["passages_with_vec"], st["passages"]), ""]
    ex = extraction_costs(cfg)
    lines += ["## Per stage, per document (milliseconds)", "",
              _md_table(["stage", "documents", "median ms", "mean ms", "max ms", "total s", "rows produced"],
                        [(k + (" (paid by this database's ingests)" if k == "extract" else ""), v["n"],
                          v["median"], v["mean"], v["max"], round(v["total"] / 1000.0, 1), v["rows"])
                         for k, v in st["stages"].items()]
                        + [("extract (every extraction on disk, from extract_status.json)", ex["seconds"]["n"],
                            ex["seconds"]["median"], ex["seconds"]["mean"], ex["seconds"]["max"],
                            round((ex["seconds"]["total"] or 0) / 1000.0, 1), "")]), "",
              "Stage 2 is the only stage that is not re-run per ingest: `tools/extract/extract.py` is idempotent,",
              "so a document whose extraction is already on disk pays milliseconds for it. The second extract row",
              "is what the extractor itself recorded for each file it produced (verdicts: %s)."
              % ", ".join("%s %d" % kv for kv in sorted(ex["verdicts"].items(), key=lambda x: str(x[0]))),
              "Each row is the LATEST run of that stage per document, so the `embed` row is dominated by documents",
              "whose vectors were already cached; what a new vector costs is in the stage 8 section below.", ""]
    emb = []
    for r in con.execute("SELECT l.doc_id, l.ms, l.detail, "
                         "(SELECT sum(length(text)) FROM passages p WHERE p.doc_id = l.doc_id) AS chars "
                         "FROM ingest_log l WHERE l.stage = 'embed' AND l.ok = 1"):
        d = json.loads(r["detail"] or "{}")
        if d.get("embedded") and r["chars"]:
            emb.append((r["doc_id"], d["embedded"], r["chars"], r["ms"], r["ms"] / (r["chars"] / 1e5)))
    lines += ["## Stage 8, embedding (the only GPU work)", "",
              "Vectors already in the shared embedding cache that B3 and AGENTIC2 fill cost nothing: of the %d "
              % st["passages"] + "passages in the database, %d already had a cached vector." % st["passages_with_vec"],
              ""]
    if emb:
        lines += [_md_table(["document", "passages embedded", "chars", "ms", "ms per 100k chars"],
                            [(d, n, c, round(m, 1), round(r, 1)) for d, n, c, m, r in emb]), "",
                  "The first call of a process also loads bge-m3 (about 3.7 s here); the steady-state rate is the",
                  "one to quote.", ""]
    else:
        lines += ["No passage in this database needed a new vector.", ""]
    rates = per_document_rates(con)
    per100k = {s: _quantiles([d[s] / (d["n_chars"] / 1e5) for d in rates if d.get(s) and d["n_chars"] > 5000])
               for s in ("load", "facts", "passages")}
    lines += ["## Throughput per 100,000 characters of report text", "",
              _md_table(["stage", "documents", "median ms", "mean ms", "max ms"],
                        [(s, q["n"], q["median"], q["mean"], q["max"]) for s, q in per100k.items()]), ""]
    ready = [sum(d.get(s, 0.0) for s in ("load", "facts", "passages", "ready")) for d in rates]
    with_ex = [d["extract"] + sum(d.get(s, 0.0) for s in ("register", "model", "load", "facts", "passages", "ready"))
               for d in rates if d.get("extract")]
    q1, q2 = _quantiles(ready), _quantiles(with_ex)
    full = [(q1["median"] or 0) + (ex["seconds"]["median"] or 0)]
    lines += ["## Time from a new file on disk to the first answerable state", "",
              "The headline the design claims: a report is answerable through the database path with NO model call.",
              "", _md_table(["path", "documents", "median s", "mean s", "max s"],
                            [("load -> ready (extraction already on disk)", q1["n"],
                              None if q1["median"] is None else round(q1["median"] / 1000.0, 2),
                              None if q1["mean"] is None else round(q1["mean"] / 1000.0, 2),
                              None if q1["max"] is None else round(q1["max"] / 1000.0, 2)),
                             ("register -> ready, extraction included, as measured", q2["n"],
                              None if q2["median"] is None else round(q2["median"] / 1000.0, 2),
                              None if q2["mean"] is None else round(q2["mean"] / 1000.0, 2),
                              None if q2["max"] is None else round(q2["max"] / 1000.0, 2)),
                             ("register -> ready, median load-to-ready + median extraction", len(rates),
                              round(full[0] / 1000.0, 2), "", "")]), ""]
    return "\n".join(lines) + "\n"
