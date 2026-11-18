"""Stage 6 - reconciliation in SQL: verified rows -> one series per question, plus gaps and conflicts.

The rules are `approaches/agentic/reconcile.py`'s, unchanged. They are why v2 states no false trend on a timeline,
so this module does not restate them in Python: it expresses them as SQL over a temp table of candidate rows joined
to `documents`, and it keeps v1's own predicates (`same_number`, `_differs`), v1's note texts (`table_notes`) and
v1's definition classes and qualifier rule by importing them rather than copying them.

    rec = reconcile(rows, kind="timeline", target_periods=[1398, ..., 1404], opts=o, q_lang="fa",
                    known_stems=stems, kb=con)          # kb: a `kb.schema.connect` connection, read-only is fine
    notes = table_notes(rec, "fa")                      # v1's texts, verbatim
    gaps = gap_queries(rec, kb=con)                     # program-only follow-up queries for the missing years

`rows` are v1 rows (`reconcile.make_row`): period, value, unit, definition, quote, doc_id, value_num, def_class,
role, period_year, doc_years, score, cid. The output dict has every key `reconcile.reconcile` returns, with the same
values, plus three SQL-only ones:

    conflict_detail   the spec's refined conflict rule (section 6 stage 6): one entry per (kept row, restatement)
                      pair of the same year and def_class that ALSO has the same unit kind, comes from a DIFFERENT
                      document, and differs beyond the tolerance in `value_cmp` (the rial equivalent), so a toman
                      restatement of a rial figure is not mistaken for a disagreement. `in_v1_conflicts` says
                      whether v1 flagged the same restatement.
    conflicts_resolved  the mirror image: the pairs v1 DID flag that the refined rule does not, each with its
                      `reason` (values_agree, different_unit_kind, same_document, no_number). Nothing is removed
                      from `conflicts`, which stays exactly v1's list so no v2 behaviour is lost; this list only
                      says which of its entries the typed columns can explain away. The kept (authoritative) value
                      stays in the series either way.
    periods_covered   the years of `target_periods` the series does cover (the complement of `periods_missing`).
    sql_ms            per-step timings of the SQL, for the paper.

Where the work happens. SQL decides every row-level question: the role of a row (a CASE over the document's years),
which rows a period keeps (authoritative first, else the rows of the latest restating report), the duplicate values
inside a kept set, which rows are narrower than the plainest one, which restatements conflict, which target years
have no row, and which documents could still hold them. Python assembles what SQL grouped - the class lists, the
main class, the trend direction - and the notes come from v1's own `table_notes`.

Working database. The candidate tables live in a private in-memory database, and the document years the role CASE
needs are mirrored into it from `documents` (or, for a document the knowledge base does not hold, from the row's own
`doc_years`, which is what v1 uses). That keeps the reconciler usable on a read-only knowledge-base connection,
where no TEMP table can be created, and runnable with no knowledge base at all.

Gap fill (`gap_queries`) is PROGRAM-ONLY and makes no model call. For each missing year it names the two documents
that can hold the number - the missing year's own report and the next year's report, where a restatement of it would
live - and builds a regex from the LABELS of the rows already verified, their wording being the best regex
available. It returns a ready `facts` statement and its parameters; the caller decides whether the time budget
allows running it and a second verify call on what comes back.
"""
import re
import sqlite3
import time

from ...corpus import doc_lang
from ...kb.typed import parse_unit
from ...normalize import fold, normalize
from ..agentic import planner as P
from ..agentic import reconcile as R
from ..agentic.docmeta import doc_key
from ..agentic.numbers import same_number
from ..agentic.reconcile import table_notes, table_text       # v1 texts, re-exported verbatim

RECONCILE_SQL_VERSION = "agentic3-reconcile-sql-v1"

CAND_COLUMNS = ("i", "doc_id", "period_year", "def_class", "value_text", "value_num", "value_cmp",
                "unit_kind", "unit_norm", "score", "n_qual", "row_years_from", "row_years_to")

DDL = """
CREATE TABLE doc (doc_id TEXT PRIMARY KEY, company TEXT, lang TEXT, year_from INTEGER, year_to INTEGER, title TEXT);
CREATE TABLE cand (i INTEGER PRIMARY KEY, doc_id TEXT, period_year INTEGER, def_class TEXT, value_text TEXT,
                   value_num REAL, value_cmp REAL, unit_kind TEXT, unit_norm TEXT, score REAL, n_qual INTEGER,
                   row_years_from INTEGER, row_years_to INTEGER);
CREATE TABLE target (year INTEGER);
"""

# the v1 role rule as one CASE. No document years (corpus A) -> the document speaks for its own statement.
ROLE_SQL = """
CREATE TABLE cand_role AS
SELECT c.*, COALESCE(d.year_from, c.row_years_from) AS doc_from, COALESCE(d.year_to, c.row_years_to) AS doc_to,
       CASE WHEN COALESCE(d.year_from, c.row_years_from) IS NULL THEN 'authoritative'
            WHEN c.period_year IS NULL THEN 'unknown_period'
            WHEN c.period_year BETWEEN COALESCE(d.year_from, c.row_years_from)
                                   AND COALESCE(d.year_to, c.row_years_to) THEN 'authoritative'
            WHEN c.period_year < COALESCE(d.year_from, c.row_years_from) THEN 'restatement'
            ELSE 'projection' END AS role
FROM cand c LEFT JOIN doc d ON d.doc_id = c.doc_id;
"""

# per period: keep the authoritative rows; with none, the rows of the LATEST restating report. `kept` is ordered by
# verification rank (v1 sorts by -score and Python's sort is stable, hence `i` as the tie-break).
PICK_SQL = """
CREATE TABLE period_pick AS
SELECT period_year, MAX(role = 'authoritative') AS has_auth,
       MAX(CASE WHEN role = 'restatement' THEN doc_to END) AS latest_to
FROM cand_role WHERE role IN ('authoritative', 'restatement') GROUP BY period_year;

CREATE TABLE kept AS
SELECT c.*, CASE WHEN p.has_auth = 1 THEN 'authoritative' ELSE 'latest_restatement' END AS kept_role,
       ROW_NUMBER() OVER (PARTITION BY c.period_year ORDER BY -COALESCE(c.score, 0), c.i) AS k_ord
FROM cand_role c JOIN period_pick p ON p.period_year IS c.period_year
WHERE (p.has_auth = 1 AND c.role = 'authoritative')
   OR (p.has_auth = 0 AND c.role = 'restatement' AND c.doc_to = p.latest_to);

CREATE TABLE rest AS
SELECT c.* FROM cand_role c JOIN period_pick p ON p.period_year IS c.period_year
WHERE c.role = 'restatement' AND (p.has_auth = 1 OR c.doc_to <> p.latest_to);
"""

# duplicates inside a kept set: an equal value of the same class (numbers through `same_number`), or, for a row with
# no number at all, an equal value string. Both relations are transitive, so "no EARLIER row matches" is v1's
# sequential rule ("no already-kept row matches"); the string test additionally ignores rows the number test dropped,
# which is the one case where the two would part.
UNIQ_SQL = """
CREATE TABLE kept_dup AS
SELECT a.i AS i, EXISTS (SELECT 1 FROM kept b
                         WHERE b.period_year IS a.period_year AND b.k_ord < a.k_ord AND b.def_class IS a.def_class
                           AND b.value_num IS NOT NULL AND a.value_num IS NOT NULL
                           AND same_num(b.value_num, a.value_num)) AS dup_value
FROM kept a;

CREATE TABLE uniq AS
SELECT a.* FROM kept a JOIN kept_dup da ON da.i = a.i
WHERE da.dup_value = 0
  AND NOT (a.value_num IS NULL
           AND EXISTS (SELECT 1 FROM kept b JOIN kept_dup db ON db.i = b.i
                       WHERE b.period_year IS a.period_year AND b.k_ord < a.k_ord AND db.dup_value = 0
                         AND b.value_text = a.value_text));
"""

# a period's subgroup rows count only when it has nothing else; a row with >= 2 unexplained definition words and more
# than the plainest row of its period is NARROWER (a programme's users, one service) and leaves the series.
POOL_SQL = """
CREATE TABLE pool AS
SELECT u.* FROM uniq u
JOIN (SELECT period_year, MAX(COALESCE(def_class, '') <> 'subgroup') AS has_main FROM uniq GROUP BY period_year) g
  ON g.period_year IS u.period_year
WHERE g.has_main = 0 OR COALESCE(u.def_class, '') <> 'subgroup'
"""
POOL_STAT_SQL = """
CREATE TABLE pool_stat AS
SELECT period_year, COUNT(*) AS n, MIN(n_qual) AS plainest FROM pool GROUP BY period_year
"""
NARROW_SQL = """
CREATE TABLE narrow AS
SELECT p.i AS i FROM pool p JOIN pool_stat s ON s.period_year IS p.period_year
WHERE :use_qual = 1 AND :is_single = 0 AND s.n > 1 AND p.n_qual >= 2 AND p.n_qual > s.plainest
"""
SERIES_SQL = """
CREATE TABLE series AS
SELECT p.*, ROW_NUMBER() OVER (ORDER BY (p.period_year IS NULL), p.period_year, p.k_ord) AS s_ord
FROM pool p WHERE p.i NOT IN (SELECT i FROM narrow)
"""

# a single-fact question: authoritative first, then the plainest definition, then the verification rank; one row per
# distinct value; at most `single_rows_max`.
SINGLE_ORD_SQL = """
CREATE TABLE single_ord AS
SELECT s.*, ROW_NUMBER() OVER (ORDER BY (s.kept_role <> 'authoritative'), s.n_qual, -COALESCE(s.score, 0), s.s_ord)
       AS q_ord
FROM series s
"""
SINGLE_KEEP_SQL = """
CREATE TABLE single_keep AS
SELECT a.* FROM single_ord a
WHERE NOT (a.value_num IS NOT NULL
           AND EXISTS (SELECT 1 FROM single_ord b WHERE b.q_ord < a.q_ord AND b.value_num IS NOT NULL
                       AND same_num(b.value_num, a.value_num)))
ORDER BY a.q_ord LIMIT :single_rows_max
"""

# v1's conflict: a restatement whose value differs from EVERY kept row of the same class. The kept value stays.
CONFLICT_SQL = """
SELECT r.i AS rest_i, r.period_year AS period_year, r.def_class AS def_class, r.doc_id AS restated_doc,
       r.value_text AS restated
FROM rest r
WHERE EXISTS (SELECT 1 FROM uniq u WHERE u.period_year IS r.period_year AND u.def_class IS r.def_class)
  AND NOT EXISTS (SELECT 1 FROM uniq u
                  WHERE u.period_year IS r.period_year AND u.def_class IS r.def_class
                    AND differs(u.value_num, r.value_num, :tol) = 0)
ORDER BY (r.period_year IS NULL), r.period_year, r.i
"""

# the spec's refined rule, one row per (kept row, restatement) pair of the same year and class, with the reason it
# is or is not a conflict: same year, same class, SAME UNIT KIND, DIFFERENT documents, and a rial-equivalent
# difference beyond the tolerance.
CONFLICT_PAIR_SQL = """
SELECT u.i AS kept_i, r.i AS rest_i, u.period_year AS period_year, u.def_class AS def_class,
       u.unit_kind AS kept_kind, r.unit_kind AS restated_kind, u.doc_id AS kept_doc, r.doc_id AS restated_doc,
       u.value_text AS kept, r.value_text AS restated, u.kept_role AS kept_role,
       COALESCE(u.value_cmp, u.value_num) AS kept_cmp, COALESCE(r.value_cmp, r.value_num) AS restated_cmp,
       CASE WHEN u.doc_id = r.doc_id THEN 'same_document'
            WHEN u.unit_kind IS NOT r.unit_kind THEN 'different_unit_kind'
            WHEN COALESCE(u.value_cmp, u.value_num) IS NULL
              OR COALESCE(r.value_cmp, r.value_num) IS NULL THEN 'no_number'
            WHEN differs(COALESCE(u.value_cmp, u.value_num),
                         COALESCE(r.value_cmp, r.value_num), :tol) = 0 THEN 'values_agree'
            ELSE 'conflict' END AS reason
FROM uniq u JOIN rest r ON r.period_year IS u.period_year
WHERE u.def_class IS r.def_class
ORDER BY (u.period_year IS NULL), u.period_year, r.i, u.k_ord
"""

SINGLE_SERIES_SQL = """
CREATE TABLE series AS
SELECT i, doc_id, period_year, def_class, value_text, value_num, value_cmp, unit_kind, unit_norm, score, n_qual,
       row_years_from, row_years_to, doc_from, doc_to, role, kept_role, k_ord, q_ord AS s_ord
FROM single_keep
"""


def _pstr(p):
    return "" if p is None else str(p)


def _unit_fields(row):
    """unit_kind / unit_norm / value_cmp of a row.

    A row that came from the `facts` table already carries them. A v1 row carries only the unit text, so it goes
    through the same typed normaliser `facts` is built with, and the unit field's scale is applied before the rial
    conversion ("50" + "میلیون تومان" -> 500,000,000 rial):
    v1 puts a scale word that sits in the unit field, not in the value token, outside `value_num`. `value_cmp` is
    used only by the conflict pairs, never by the series, so this never moves a number a reader sees.
    """
    if row.get("unit_kind") or row.get("value_cmp") is not None:
        return row.get("unit_kind"), row.get("unit_norm"), row.get("value_cmp")
    u = parse_unit(row.get("unit") or "")
    cmp_ = None
    if row.get("value_num") is not None and u["currency"] in ("toman", "rial"):
        cmp_ = float(row["value_num"]) * u["scale"] * (10.0 if u["currency"] == "toman" else 1.0)
    return u["unit_kind"], u["unit_norm"], cmp_


def working_db(rows, *, target_periods=(), kb=None):
    """A private in-memory database holding `cand`, `doc` and `target`, with `same_num` and `differs` registered."""
    con = sqlite3.connect(":memory:")
    con.row_factory = sqlite3.Row
    con.create_function("same_num", 2, lambda a, b: int(same_number(a, b)), deterministic=True)
    con.create_function("differs", 3, lambda a, b, t: int(R._differs(a, b, t)), deterministic=True)
    con.executescript(DDL)
    cand = []
    for i, r in enumerate(rows):
        kind, norm, cmp_ = _unit_fields(r)
        years = r.get("doc_years") or (None, None)
        cand.append((i, r.get("doc_id"), r.get("period_year"), r.get("def_class"), r.get("value"),
                     r.get("value_num"), cmp_, kind, norm, r.get("score"), len(r.get("qualifiers") or []),
                     years[0], years[1]))
    con.executemany("INSERT INTO cand (%s) VALUES (%s)" % (", ".join(CAND_COLUMNS), ", ".join("?" * len(CAND_COLUMNS))),
                    cand)
    con.executemany("INSERT INTO target (year) VALUES (?)", [(int(y),) for y in target_periods])
    doc_ids = sorted({r.get("doc_id") for r in rows if r.get("doc_id")})
    if kb is not None and doc_ids:
        q = ("SELECT doc_id, company, lang, year_from, year_to, title FROM documents WHERE doc_id IN (%s)"
             % ", ".join("?" * len(doc_ids)))
        con.executemany("INSERT INTO doc (doc_id, company, lang, year_from, year_to, title) VALUES (?, ?, ?, ?, ?, ?)",
                        [(d["doc_id"], d["company"], d["lang"], d["year_from"], d["year_to"], d["title"])
                         for d in kb.execute(q, doc_ids).fetchall()])
    return con


def reconcile(rows, *, kind, target_periods=(), opts=None, q_lang="fa", known_stems=None, kb=None):
    """The v1 reconciler's contract, decided in SQL. See the module docstring for the three added keys."""
    opts = opts or {}
    tol = float(opts.get("restatement_rel_tol", 0.005))
    if known_stems is not None:
        for r in rows:
            r["qualifiers"] = R.qualifiers(r.get("definition"), known_stems)
    con = working_db(rows, target_periods=target_periods, kb=kb)
    ms, t0 = {}, time.perf_counter()
    con.executescript(ROLE_SQL)
    con.executescript(PICK_SQL)
    con.executescript(UNIQ_SQL)
    ms["roles"] = round((time.perf_counter() - t0) * 1000, 3)

    t0 = time.perf_counter()
    con.execute(POOL_SQL)
    con.execute(POOL_STAT_SQL)
    con.execute(NARROW_SQL, {"use_qual": int(known_stems is not None), "is_single": int(kind == "single")})
    con.execute(SERIES_SQL)
    if kind == "single":
        con.execute(SINGLE_ORD_SQL)
        con.execute(SINGLE_KEEP_SQL, {"single_rows_max": int(opts.get("single_rows_max", 3))})
        con.execute("DROP TABLE series")
        con.execute(SINGLE_SERIES_SQL)
    ms["series"] = round((time.perf_counter() - t0) * 1000, 3)

    def keep(i, kept_role):
        return rows[i] if kept_role == "authoritative" else dict(rows[i], role="latest_restatement")

    role_of = {r["i"]: r["role"] for r in con.execute("SELECT i, role FROM cand_role").fetchall()}
    out = {"kind": kind, "series": [], "restatements": [], "conflicts": [], "projections": [],
           "unknown_period": [], "notes": [], "latest_restatement_periods": [], "narrower": [],
           "conflict_detail": [], "conflicts_resolved": [], "sql_ms": ms}
    out["projections"] = [r for i, r in enumerate(rows) if role_of.get(i) == "projection"]
    out["unknown_period"] = [r for i, r in enumerate(rows) if role_of.get(i) == "unknown_period"]
    out["latest_restatement_periods"] = [r["period_year"] for r in con.execute(
        "SELECT period_year FROM period_pick WHERE has_auth = 0 "
        "ORDER BY (period_year IS NULL), period_year").fetchall()]
    out["narrower"] = [keep(r["i"], r["kept_role"]) for r in con.execute(
        "SELECT p.i AS i, p.kept_role AS kept_role FROM narrow n JOIN pool p ON p.i = n.i "
        "ORDER BY (p.period_year IS NULL), p.period_year, p.k_ord").fetchall()]
    series = [keep(r["i"], r["kept_role"]) for r in
              con.execute("SELECT i, kept_role FROM series ORDER BY s_ord").fetchall()]
    out["series"] = series

    t0 = time.perf_counter()
    out["restatements"] = [rows[r["i"]] for r in con.execute(
        "SELECT i FROM rest ORDER BY (period_year IS NULL), period_year, i").fetchall()]
    v1_conflicts = con.execute(CONFLICT_SQL, {"tol": tol}).fetchall()
    flagged = {c["rest_i"] for c in v1_conflicts}
    for c in v1_conflicts:
        same = con.execute("SELECT value_text, doc_id FROM uniq WHERE period_year IS ? AND def_class IS ? "
                           "ORDER BY k_ord", (c["period_year"], c["def_class"])).fetchall()
        out["conflicts"].append({"period": str(c["period_year"]), "kept": [u["value_text"] for u in same],
                                 "kept_doc": same[0]["doc_id"], "restated": c["restated"],
                                 "restated_doc": c["restated_doc"], "def_class": c["def_class"]})
    for c in con.execute(CONFLICT_PAIR_SQL, {"tol": tol}).fetchall():
        pair = {"period": _pstr(c["period_year"]), "def_class": c["def_class"], "reason": c["reason"],
                "kept_kind": c["kept_kind"], "restated_kind": c["restated_kind"],
                "kept": c["kept"], "kept_doc": c["kept_doc"], "kept_role": c["kept_role"],
                "restated": c["restated"], "restated_doc": c["restated_doc"],
                "kept_cmp": c["kept_cmp"], "restated_cmp": c["restated_cmp"],
                "in_v1_conflicts": c["rest_i"] in flagged}
        if c["reason"] == "conflict":
            out["conflict_detail"].append(pair)
        elif pair["in_v1_conflicts"]:
            out["conflicts_resolved"].append(pair)
    ms["conflicts"] = round((time.perf_counter() - t0) * 1000, 3)

    t0 = time.perf_counter()
    out["periods_found"] = [_pstr(r["period_year"]) for r in con.execute(
        "SELECT DISTINCT period_year FROM series ORDER BY (period_year IS NULL), period_year").fetchall()]
    out["periods_missing"] = [str(r["year"]) for r in con.execute(
        "SELECT year FROM target WHERE year NOT IN "
        "(SELECT period_year FROM series WHERE period_year IS NOT NULL) ORDER BY rowid").fetchall()]
    out["periods_covered"] = [str(y) for y in target_periods if str(y) not in out["periods_missing"]]
    classes, class_points = {}, {}
    for r in con.execute("SELECT def_class, period_year, COUNT(*) AS n, MIN(value_text) AS value, MIN(s_ord) AS ord "
                         "FROM series WHERE COALESCE(def_class, '') <> 'subgroup' "
                         "GROUP BY def_class, period_year ORDER BY ord").fetchall():
        classes.setdefault(r["def_class"], []).append(_pstr(r["period_year"]))
        class_points.setdefault(r["def_class"], [])
        if r["n"] == 1:
            class_points[r["def_class"]].append((_pstr(r["period_year"]), r["value"]))
    out["definition_classes"] = classes
    out["class_points"] = class_points
    out["several_values_periods"] = [] if kind == "single" else [_pstr(r["period_year"]) for r in con.execute(
        "SELECT period_year FROM series WHERE period_year IS NOT NULL GROUP BY period_year HAVING COUNT(*) > 1 "
        "ORDER BY period_year").fetchall()]
    # the main class is a presentation pick over the SQL grouping: most periods, a named class before "unspecified",
    # then first appearance. `max` over a dict returns the first maximum, which is v1's tie-break.
    out["main_class"] = max(classes, key=lambda c: (len(classes[c]), c != "unspecified")) if classes else None
    dated = [q for q in out["periods_found"] if q]
    out["comparable"] = (len(classes) == 1) if (kind in ("timeline", "comparison") and len(dated) >= 2) else None
    out["trend"] = None
    pts = class_points.get(out["main_class"]) or []
    if out["comparable"] and out["main_class"] and len(pts) >= 2:
        first = [r for r in series if _pstr(r["period_year"]) == pts[0][0]][0]
        last = [r for r in series if _pstr(r["period_year"]) == pts[-1][0]][0]
        if first["value_num"] is not None and last["value_num"] is not None:
            direction = ("up" if last["value_num"] > first["value_num"]
                         else "down" if last["value_num"] < first["value_num"] else "flat")
            out["trend"] = {"class": out["main_class"], "from": [first["period"], first["value"]],
                            "to": [last["period"], last["value"]], "direction": direction}
    ms["summary"] = round((time.perf_counter() - t0) * 1000, 3)
    con.close()
    return out


# ---------------------------------------------------------------------------------------------------------------
# gap fill: the missing years, as targeted follow-up queries (no model)
# ---------------------------------------------------------------------------------------------------------------

_WORD_SPLIT = re.compile(r"[\s‌،,.:;()«»|/\-*]+")
_JOIN = r"[\s‌]{0,2}"
MAX_LABEL_WORDS = 3
_LABEL_STOP = {fold(w) for w in P.WEAK_WORDS | P.GENERIC_TERM_WORDS}

GAP_DOCS_SQL = """
SELECT doc_id, company, lang, year_from, year_to, title, status,
       CASE WHEN :y BETWEEN year_from AND year_to THEN 'own' ELSE 'next' END AS gap_role
FROM documents
WHERE status IN ('ready', 'vector_ready') AND year_from IS NOT NULL
  AND ((:y BETWEEN year_from AND year_to) OR (:y1 BETWEEN year_from AND year_to))
ORDER BY gap_role, year_from, doc_id
"""

GAP_FACTS_SQL = """
SELECT fact_id, doc_id, section_id, unit_id, label, value_text, value_num, unit_norm, unit_kind, year, year_source,
       def_class, quote, score
FROM facts
WHERE doc_id IN (%s)
  AND (year IS NULL OR (year <= :hi AND COALESCE(year_to, year) >= :lo))
  AND label_fold REGEXP :pattern
ORDER BY (year = :y) DESC, (year_source = 'own') DESC, score DESC, fact_id
LIMIT :limit
"""


def label_pattern(phrase):
    """One verified row's wording -> a regex over `facts.label_fold`.

    The target column is folded, so every literal is folded and the pattern is assembled afterwards. Folding a
    finished pattern instead is a silent corruption: `fold` strips the ezafe hamza out of
    `(?:شهرهای|شهر)ٔ?[ZWNJ ]?` and leaves
    `(?:شهرهای|شهر)?`, an optional group matching the empty string.
    Persian word forms come from `planner.fa_word_forms` / `fa_word_variants`, the same morphology the plan terms
    use. Counter and filler words are dropped with the planner's own `WEAK_WORDS` and `GENERIC_TERM_WORDS`
    ("تعداد کاربران فعال" ->
    "کاربران فعال"), not with the reconciler's qualifier
    stop list, which would also drop the metric word itself; a phrase made only of such words keeps them rather
    than returning nothing. At most MAX_LABEL_WORDS words are used.
    """
    words = [w for w in _WORD_SPLIT.split(normalize(phrase or "")) if w]
    words = [w for w in words if len(w) >= 3 and not re.search(r"[0-9]", w)]
    words = [w for w in words if fold(w) not in _LABEL_STOP] or words
    words = words[:MAX_LABEL_WORDS]
    if not words:
        return ""
    parts = []
    for n, w in enumerate(words):
        last = n == len(words) - 1
        if P._lang_of(w) == "fa":
            forms = P.fa_word_variants(w) if last else P.fa_word_forms(w)
        else:
            forms = [w.lower().rstrip("s") if last and len(w) > 4 and not w.lower().endswith("ss") else w.lower()]
        alts = list(dict.fromkeys(re.escape(fold(f)) for f in forms if fold(f)))
        parts.append(alts[0] if len(alts) == 1 else "(?:%s)" % "|".join(alts))
    return _JOIN.join(parts)


def gap_queries(rec, *, kb, target_periods=(), lang=None, max_labels=4, limit=40):
    """The missing years of a timeline, as follow-up queries the caller may or may not have time to run.

    One entry per missing year: the documents that can hold the number (`own` = that year's own report, `next` = the
    following year's report, where a restatement of it lives), a regex built from the labels of the rows already
    verified, and a ready `facts` query with its parameters. No model call and no vector call is made, and the facts
    query is NOT executed here (`run_gap_query` does that): whether it fits the time budget, and whether what comes
    back is worth a second verify call, is the caller's decision.
    """
    labels, seen = [], set()
    for r in list(rec.get("series") or []) + list(rec.get("restatements") or []):
        for text in (r.get("label"), r.get("definition")):
            pat = label_pattern(text)
            if pat and pat not in seen:
                seen.add(pat)
                labels.append({"text": normalize(text or ""), "pattern": pat})
        if len(labels) >= max_labels:
            break
    labels = labels[:max_labels]
    pattern = "|".join(l["pattern"] for l in labels)
    docs_of = [r["doc_id"] for r in (rec.get("series") or []) if r.get("doc_id")]
    companies = {re.split(r"[_\-]", doc_key(d))[0] for d in docs_of}
    langs = {doc_lang(d) for d in docs_of}
    want_lang = lang or (sorted(langs)[0] if len(langs) == 1 else None)
    missing = [int(y) for y in (rec.get("periods_missing") or [])]
    if target_periods:
        missing = [y for y in missing if y in {int(t) for t in target_periods}]
    out = []
    for y in missing:
        docs = [dict(d) for d in kb.execute(GAP_DOCS_SQL, {"y": y, "y1": y + 1}).fetchall()]
        if companies:
            docs = [d for d in docs if d["company"] in companies]
        if want_lang:
            docs = [d for d in docs if d["lang"] == want_lang]
        doc_ids = [d["doc_id"] for d in docs]
        params = {"d%d" % k: d for k, d in enumerate(doc_ids)}
        params.update({"y": y, "lo": y, "hi": y, "pattern": pattern, "limit": int(limit)})
        out.append({"year": y, "docs": docs, "doc_ids": doc_ids, "labels": labels, "pattern": pattern,
                    "sql": GAP_FACTS_SQL % (", ".join(":d%d" % k for k in range(len(doc_ids))) or "NULL"),
                    "params": params, "runnable": bool(doc_ids and pattern)})
    return out


def run_gap_query(kb, gap):
    """Execute one `gap_queries` entry against `facts`. Separate on purpose: the caller pays for it."""
    if not gap["runnable"]:
        return []
    return [dict(r) for r in kb.execute(gap["sql"], gap["params"]).fetchall()]
