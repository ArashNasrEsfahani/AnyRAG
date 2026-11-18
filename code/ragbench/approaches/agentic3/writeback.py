"""Stage 8 - write back: what the verifier accepted goes into `verified_facts`, so the database gets better as it
is used ("index by use"). No model call, no network, one commit per row.

    h = question_hash(query.text, plan)
    w = write_rows(con, rows, plan=plan, run_id=run_id, model=model, prompt_version=VERIFY_PROMPT_VERSION,
                   question_hash=h)                     # -> {"written": n, "skipped": {reason: n}, "ms": ...}
    warm = read_rows(con, doc_ids=chosen, pattern=rx, years=(1398, 1404), warm=flags.get("warm", False))

Two rules decide everything here, and both are about not trusting a row we should not trust.

ONLY GUARD-PASSED ROWS ARE WRITTEN. The guard is the verbatim check: the quote is in the source text, the value is
in the quote, the metric term is in the quote. The model's own "relevant: true" is not enough and is not stored as
if it were. The caller must copy the verifier's guard result onto the row (`row["guard"] = item["guard"]`); a row
that does not carry one is skipped and counted under `no_guard`, never written on the assumption that it passed.

A STALE CONTENT HASH MAKES A ROW UNUSABLE. Every row is written with the document's `content_sha1` as of the moment
it was verified. `read_rows` joins `verified_facts` to `documents` ON THE HASH, so a document that has been
re-ingested with new content simply stops matching: the old rows are still there (they are the measurement record
of the warm experiment) but no query can reach them. `stale_rows` counts them for the report. There is no code path
that reads a verified fact without that join: `read_rows` below and `kb/query.py::verified_lookup`, the warm path
the agent actually calls, are the only two readers and both carry it.

ONE GUARD RULE, ONE PLACE. `GUARD_OK` below is that place. The writer refuses anything else and the reader accepts
nothing else - `kb/query.py::guard_ok_values()` imports this tuple rather than repeating it, because until
2026-09-20 it did repeat it as `coalesce(v.guard, 'ok') = 'ok'` and the two drifted: `ok_label_completed` rows were
written and then unreadable, and a row with no guard at all was read as if it had passed. A NULL guard is a miss on
both sides now.

Warm mode is OFF by default: `read_rows` returns [] unless the caller passes `warm=True`. COLD is the headline
protocol (spec section 8) - every question independent, comparable to every baseline - and warm is a separate
experiment. A wrong verified fact can spread to later questions, so it is gated by the guard, by the hash, and by
the flag.

Row format. `rows` are v1 rows (`reconcile.make_row`) with `guard` added; `unit_norm` comes from the row when the
knowledge base put it there and from `kb.typed.parse_unit` over the row's unit text otherwise. `fact_id` is stored
when the row came from the `facts` table and NULL when it came from text. A re-run of the same question writes
nothing new: the same (doc_id, hash, unit_id, value, year, question) is inserted once.
"""
import hashlib
import json
import time

from ...kb.typed import parse_unit
from ...normalize import fold, normalize

WRITEBACK_VERSION = "agentic3-writeback-v1"
GUARD_OK = ("ok", "ok_label_completed")

VF_COLUMNS = ("doc_id", "doc_content_sha1", "unit_id", "fact_id", "entity", "metric", "terms", "value_text",
              "value_num", "unit_norm", "year", "def_class", "definition", "quote", "guard", "model",
              "prompt_version", "question_hash", "run_id", "verified_at")

INSERT_SQL = "INSERT INTO verified_facts (%s) VALUES (%s)" % (", ".join(VF_COLUMNS), ", ".join("?" * len(VF_COLUMNS)))

EXISTS_SQL = """
SELECT vf_id FROM verified_facts
WHERE doc_id = ? AND doc_content_sha1 = ? AND unit_id IS ? AND value_text IS ? AND year IS ? AND question_hash IS ?
LIMIT 1
"""

# the only way a verified fact is read: joined to the document it came from, ON the content hash.
READ_SQL = """
SELECT v.vf_id, v.doc_id, v.unit_id, v.fact_id, v.entity, v.metric, v.terms, v.value_text, v.value_num,
       v.unit_norm, v.year, v.def_class, v.definition, v.quote, v.guard, v.model, v.prompt_version,
       v.question_hash, v.run_id, v.verified_at
FROM verified_facts v
JOIN documents d ON d.doc_id = v.doc_id AND d.content_sha1 = v.doc_content_sha1
WHERE d.status IN ('ready', 'vector_ready') AND v.guard IN (%s)
"""

STALE_SQL = """
SELECT COUNT(*) AS n FROM verified_facts v
LEFT JOIN documents d ON d.doc_id = v.doc_id AND d.content_sha1 = v.doc_content_sha1
WHERE d.doc_id IS NULL
"""


def question_hash(text, plan=None):
    """A short, stable id for one question as this approach saw it: the folded question plus the plan's entity,
    metric, kind and terms. Two runs of the same question share it; a changed plan does not."""
    plan = plan or {}
    terms = plan.get("terms") or {}
    parts = [fold(normalize(text or "")), str(plan.get("entity") or ""), str(plan.get("metric") or ""),
             str(plan.get("kind") or ""),
             "|".join(sorted(str(t) for lang in sorted(terms) for t in (terms.get(lang) or [])))]
    return hashlib.sha1("\x1f".join(parts).encode("utf-8")).hexdigest()[:12]


def _unit_norm(row):
    if row.get("unit_norm"):
        return row["unit_norm"]
    return parse_unit(row.get("unit") or "")["unit_norm"]


def content_sha1s(con, doc_ids):
    """{doc_id: content_sha1} for the documents a query touched."""
    ids = sorted({d for d in doc_ids if d})
    if not ids:
        return {}
    q = "SELECT doc_id, content_sha1 FROM documents WHERE doc_id IN (%s)" % ", ".join("?" * len(ids))
    return {r["doc_id"]: r["content_sha1"] for r in con.execute(q, ids).fetchall()}


def write_rows(con, rows, *, plan=None, run_id=None, model=None, prompt_version=None, question_hash=None, now=None):
    """Insert the guard-passed rows. One commit per row, so a stopped run keeps what it had already verified."""
    t0 = time.perf_counter()
    plan = plan or {}
    stamp = now or time.strftime("%Y-%m-%dT%H:%M:%S")
    terms = json.dumps(plan.get("terms") or {}, ensure_ascii=False)
    sha = content_sha1s(con, [r.get("doc_id") for r in rows])
    out = {"written": 0, "skipped": {}, "vf_ids": [], "ms": 0.0}

    def skip(reason):
        out["skipped"][reason] = out["skipped"].get(reason, 0) + 1

    for r in rows:
        guard = r.get("guard")
        if not guard:
            skip("no_guard")
            continue
        if guard not in GUARD_OK:
            skip("guard_rejected")
            continue
        doc_id = r.get("doc_id")
        if doc_id not in sha or not sha[doc_id]:
            skip("unknown_document")
            continue
        key = (doc_id, sha[doc_id], r.get("unit_id"), r.get("value"), r.get("period_year"), question_hash)
        if con.execute(EXISTS_SQL, key).fetchone():
            skip("already_written")
            continue
        con.execute(INSERT_SQL, (doc_id, sha[doc_id], r.get("unit_id"), r.get("fact_id"),
                                 plan.get("entity"), plan.get("metric"), terms, r.get("value"), r.get("value_num"),
                                 _unit_norm(r), r.get("period_year"), r.get("def_class"), r.get("definition"),
                                 r.get("quote"), guard, model, prompt_version, question_hash, run_id, stamp))
        con.commit()
        out["written"] += 1
        out["vf_ids"].append(con.execute("SELECT last_insert_rowid() AS id").fetchone()["id"])
    out["ms"] = round((time.perf_counter() - t0) * 1000, 3)
    return out


def read_rows(con, *, doc_ids=(), pattern=None, entity=None, years=None, warm=False, limit=24):
    """Verified facts still valid for the documents in scope. WARM MODE ONLY: returns [] unless `warm` is True.

    `pattern` is a regex matched against the stored metric and definition (`schema.connect` registers REGEXP);
    `years` is (lo, hi) and uses the contract's year filter, so a row with no year is never excluded by it.
    """
    if not warm:
        return []
    sql = READ_SQL % ", ".join("?" * len(GUARD_OK))
    args = list(GUARD_OK)
    ids = sorted({d for d in doc_ids if d})
    if ids:
        sql += " AND v.doc_id IN (%s)" % ", ".join("?" * len(ids))
        args += ids
    if entity:
        sql += " AND v.entity = ?"
        args.append(entity)
    if pattern:
        sql += " AND (COALESCE(v.metric, '') REGEXP ? OR COALESCE(v.definition, '') REGEXP ?)"
        args += [pattern, pattern]
    if years:
        sql += " AND (v.year IS NULL OR (v.year >= ? AND v.year <= ?))"
        args += [int(years[0]), int(years[1])]
    sql += " ORDER BY v.year, v.vf_id LIMIT ?"
    args.append(int(limit))
    return [dict(r) for r in con.execute(sql, args).fetchall()]


def stale_rows(con):
    """How many verified facts no longer match their document's content. They are kept, never read."""
    return con.execute(STALE_SQL).fetchone()["n"]


def counts(con):
    """For the warm-mode report: rows, live rows, stale rows, distinct questions."""
    live = con.execute("SELECT COUNT(*) AS n FROM verified_facts v JOIN documents d "
                       "ON d.doc_id = v.doc_id AND d.content_sha1 = v.doc_content_sha1").fetchone()["n"]
    total = con.execute("SELECT COUNT(*) AS n FROM verified_facts").fetchone()["n"]
    qs = con.execute("SELECT COUNT(DISTINCT question_hash) AS n FROM verified_facts").fetchone()["n"]
    return {"rows": total, "live": live, "stale": total - live, "questions": qs}
