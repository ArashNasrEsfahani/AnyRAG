"""The knowledge base file: DDL, the connection factory, and the per-document version check.

One SQLite file (`work/kb/kb.sqlite`, `kb.path` in config.yaml) holds every ingested document:

    documents  one row per report      sections  the tree (parent_id, path)   units  the text blocks
    facts      the typed facts (typed.py)        passages  tree leaves + their bge-m3 vector (BLOB, lazily filled)
    tri        fts5 trigram index over folded text: an indexed PREFILTER for regex, never a ranker
    verified_facts  what the verifier accepted, for warm mode      ingest_log  one row per stage per document
    meta       schema and component versions

Conventions every later module codes against:
  * ids are the corpus ids (`corpus/model.py`): doc `<corpus>.<key>.<lang>`, section/unit `<doc_id>::<local>`.
  * `*_fold` columns hold `normalize.fold(normalize(text))` and are the only regex / trigram targets. Fold both
    sides or neither.
  * `facts.year` is the period's START year and `facts.year_to` its end (NULL when the period is not a range), so a
    year filter reads `year IS NULL OR (year <= :hi AND COALESCE(year_to, year) >= :lo)`.
  * `tri.kind` is one of `fact_label`, `unit`, `section_title`; `tri.ref_id` is the fact_id / unit_id / section_id.
  * a document is visible to queries only when `documents.status` is `ready` or `vector_ready`.

Deletion: `ON DELETE CASCADE` removes a document's sections, units, facts and passages (`PRAGMA foreign_keys = ON`,
set by `connect`). `tri` is an fts5 virtual table and takes no foreign key, so an AFTER DELETE trigger on
`documents` removes its trigram rows; that trigger is the one addition to the DDL of AGENTIC3_SPEC.md section 5.
`verified_facts` is deliberately NOT cascaded: its rows keep the old `doc_content_sha1` and simply stop matching.

Versions: `meta` holds the current component versions and `documents.versions` holds the ones each document was built
with, so `stale_stages` says per document WHICH stage must be rebuilt (facts only, passages only, embeddings only)
instead of forcing a re-extraction.

Checked on this machine (2026-09-20): Python 3.14.0 ships SQLite 3.50.4 and the fts5 trigram tokenizer matches
Persian substrings (`test_kb.py` re-checks both).
"""
import functools
import re
import sqlite3
import urllib.parse
from pathlib import Path

from ..approaches.agentic2.defaults import TREE_VERSION
from ..approaches.ours.lite import FACTX_VERSION
from .typed import TYPED_VERSION

SCHEMA_VERSION = "kb-schema-v1"

DDL = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);

CREATE TABLE IF NOT EXISTS documents (
  doc_id TEXT PRIMARY KEY, corpus TEXT, key TEXT, lang TEXT, company TEXT,
  year_from INTEGER, year_to INTEGER, calendar TEXT, title TEXT, title_fold TEXT, pair_id TEXT,
  source_path TEXT, source_sha256 TEXT, content_sha1 TEXT, quality TEXT, status TEXT,
  n_sections INTEGER, n_units INTEGER, n_facts INTEGER, n_chars INTEGER, versions TEXT, ingested_at TEXT);

CREATE TABLE IF NOT EXISTS sections (
  section_id TEXT PRIMARY KEY, doc_id TEXT NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
  parent_id TEXT, level INTEGER, ord INTEGER, title TEXT, title_fold TEXT, path TEXT, path_fold TEXT,
  span_start INTEGER, span_end INTEGER, subtree_end INTEGER, n_units INTEGER, n_facts INTEGER);

CREATE TABLE IF NOT EXISTS units (
  unit_id TEXT PRIMARY KEY, doc_id TEXT NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
  section_id TEXT NOT NULL, ord INTEGER, kind TEXT, lang TEXT, text TEXT, text_fold TEXT,
  span_start INTEGER, span_end INTEGER, table_id TEXT, n_numbers INTEGER);

CREATE TABLE IF NOT EXISTS facts (
  fact_id INTEGER PRIMARY KEY, doc_id TEXT NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
  section_id TEXT, unit_id TEXT, ord INTEGER,
  label TEXT, label_fold TEXT, row_label TEXT, col_header TEXT,
  value_text TEXT, value_num REAL, value_hi REAL, value_cmp REAL, alt_readings TEXT,
  unit_text TEXT, unit_norm TEXT, unit_kind TEXT, currency TEXT, is_percent INTEGER,
  period_text TEXT, year INTEGER, year_to INTEGER, period_kind TEXT, calendar TEXT, year_source TEXT,
  def_class TEXT, source_kind TEXT, char_start INTEGER, char_end INTEGER, quote TEXT, score REAL);

CREATE TABLE IF NOT EXISTS passages (
  passage_id TEXT PRIMARY KEY, doc_id TEXT NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
  section_id TEXT, ord INTEGER, unit_ids TEXT, text TEXT, embed_sha1 TEXT,
  vec BLOB, dim INTEGER, model TEXT, embedded_at TEXT);

CREATE VIRTUAL TABLE IF NOT EXISTS tri USING fts5 (text_fold, kind UNINDEXED, ref_id UNINDEXED, doc_id UNINDEXED,
                                                   tokenize = 'trigram');

CREATE TABLE IF NOT EXISTS verified_facts (
  vf_id INTEGER PRIMARY KEY, doc_id TEXT, doc_content_sha1 TEXT, unit_id TEXT, fact_id INTEGER,
  entity TEXT, metric TEXT, terms TEXT, value_text TEXT, value_num REAL, unit_norm TEXT, year INTEGER,
  def_class TEXT, definition TEXT, quote TEXT, guard TEXT, model TEXT, prompt_version TEXT,
  question_hash TEXT, run_id TEXT, verified_at TEXT);

CREATE TABLE IF NOT EXISTS ingest_log (id INTEGER PRIMARY KEY, doc_id TEXT, stage TEXT, started_at TEXT, ms REAL,
                                       rows INTEGER, ok INTEGER, detail TEXT);

CREATE INDEX IF NOT EXISTS ix_facts_doc_year ON facts (doc_id, year);
CREATE INDEX IF NOT EXISTS ix_facts_kind_year ON facts (unit_kind, year);
CREATE INDEX IF NOT EXISTS ix_facts_unit ON facts (unit_id);
CREATE INDEX IF NOT EXISTS ix_sections_doc ON sections (doc_id, ord);
CREATE INDEX IF NOT EXISTS ix_sections_parent ON sections (parent_id);
CREATE INDEX IF NOT EXISTS ix_units_section ON units (section_id, ord);
CREATE INDEX IF NOT EXISTS ix_docs_company ON documents (company, year_from);
CREATE INDEX IF NOT EXISTS ix_vf_lookup ON verified_facts (doc_id, year, unit_norm);

CREATE TRIGGER IF NOT EXISTS trg_documents_del_tri AFTER DELETE ON documents BEGIN
  DELETE FROM tri WHERE doc_id = old.doc_id;
END;
"""

FACT_COLUMNS = ("doc_id", "section_id", "unit_id", "ord",
                "label", "label_fold", "row_label", "col_header",
                "value_text", "value_num", "value_hi", "value_cmp", "alt_readings",
                "unit_text", "unit_norm", "unit_kind", "currency", "is_percent",
                "period_text", "year", "year_to", "period_kind", "calendar", "year_source",
                "def_class", "source_kind", "char_start", "char_end", "quote", "score")

TRI_KINDS = ("fact_label", "unit", "section_title")
STAGES = ("load", "facts", "passages", "embed")     # rebuildable stages, in pipeline order


# ---------------------------------------------------------------------------------------------------------------
# connection
# ---------------------------------------------------------------------------------------------------------------

@functools.lru_cache(maxsize=1024)
def _compiled(pattern):
    """Compile-cached; an unusable pattern caches as None instead of raising inside SQLite."""
    try:
        return re.compile(pattern, re.I)
    except re.error:
        return None


def _regexp(pattern, value):
    rx = _compiled(pattern or "")
    if rx is None or value is None:
        return 0
    return 1 if rx.search(str(value)) else 0


def _dict_row(cur, row):
    return {d[0]: row[i] for i, d in enumerate(cur.description)}


def connect(path, read_only=False):
    """-> sqlite3.Connection with WAL, foreign_keys ON, synchronous NORMAL, REGEXP, and dict rows.

    `value REGEXP pattern` is case-insensitive `re.search`; a pattern that does not compile matches nothing (0) and
    never raises, so one bad plan term cannot kill a query. `path` may be ":memory:" (tests). A read-only connection
    opens the existing file through a `mode=ro` URI and sets `query_only`; it is what `kb sql` and the analysis
    scripts use while an ingest is running.
    """
    p = str(path)
    if p == ":memory:":
        con = sqlite3.connect(p)
    elif read_only:
        uri = "file:" + urllib.parse.quote(Path(p).resolve().as_posix(), safe="/:") + "?mode=ro"
        con = sqlite3.connect(uri, uri=True)
    else:
        Path(p).parent.mkdir(parents=True, exist_ok=True)
        con = sqlite3.connect(p, timeout=60)      # parallel sweep lanes: a brief card refresh waits, never fails
    con.row_factory = _dict_row
    con.create_function("REGEXP", 2, _regexp, deterministic=True)
    con.execute("PRAGMA foreign_keys = ON")
    if read_only:
        con.execute("PRAGMA query_only = ON")
    elif p != ":memory:":
        con.execute("PRAGMA journal_mode = WAL")
        con.execute("PRAGMA synchronous = NORMAL")
    return con


# ---------------------------------------------------------------------------------------------------------------
# schema and versions
# ---------------------------------------------------------------------------------------------------------------

def component_versions(cfg=None):
    """The versions a document ingested NOW would carry: {schema, factx, typed, tree, embed}."""
    if cfg is None:
        from ..config import load_config
        cfg = load_config()
    return {"schema": SCHEMA_VERSION, "factx": FACTX_VERSION, "typed": TYPED_VERSION, "tree": TREE_VERSION,
            "embed": cfg.get("models.embed")}


def ensure_schema(con, cfg=None):
    """Apply the DDL if it is not there and record the current component versions in `meta`. -> the versions.

    Idempotent: every statement is IF NOT EXISTS, the version rows are rewritten with the same values, and
    `meta.created_at` is written once. Running it on a populated database changes nothing.
    """
    con.executescript(DDL)
    versions = component_versions(cfg)
    for name, value in versions.items():
        con.execute("INSERT INTO meta (key, value) VALUES (?, ?) "
                    "ON CONFLICT(key) DO UPDATE SET value = excluded.value", ("version." + name, str(value)))
    con.execute("INSERT OR IGNORE INTO meta (key, value) VALUES ('created_at', datetime('now'))")
    con.commit()
    return versions


def get_meta(con, key, default=None):
    row = con.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
    return default if row is None else row["value"]


def set_meta(con, key, value):
    con.execute("INSERT INTO meta (key, value) VALUES (?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value", (key, str(value)))


def stale_stages(stored, cfg=None, current=None):
    """Which stages of ONE document must be rebuilt, given the versions it was built with.

    `stored`: the `documents.versions` dict (or its JSON text, or None for a document from before versioning).
      schema changed -> everything, the row layout itself moved
      factx / typed changed -> ("facts",)          the regex facts and their typed columns
      tree changed -> ("passages", "embed")        the passage segmentation is the embedding cache key
      embed model changed -> ("embed",)
    Extraction (stages 1-3) is never in the list: a version change never re-reads the PDF.
    """
    cur = current or component_versions(cfg)
    if isinstance(stored, str):
        import json
        try:
            stored = json.loads(stored)
        except ValueError:
            stored = None
    if not isinstance(stored, dict):
        return STAGES
    if stored.get("schema") != cur["schema"]:
        return STAGES
    need = set()
    if stored.get("factx") != cur["factx"] or stored.get("typed") != cur["typed"]:
        need.add("facts")
    if stored.get("tree") != cur["tree"]:
        need.update(("passages", "embed"))
    if stored.get("embed") != cur["embed"]:
        need.add("embed")
    return tuple(s for s in STAGES if s in need)


def stale_documents(con, cfg=None):
    """-> [(doc_id, stages)] for every document whose versions no longer match, in doc_id order."""
    cur = component_versions(cfg)
    out = []
    for row in con.execute("SELECT doc_id, versions FROM documents ORDER BY doc_id"):
        stages = stale_stages(row["versions"], current=cur)
        if stages:
            out.append((row["doc_id"], stages))
    return out
