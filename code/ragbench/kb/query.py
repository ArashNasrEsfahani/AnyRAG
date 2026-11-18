"""Retrieval over the knowledge base: the FIRST path of AGENTIC3, program only, no model call.

Everything here reads an open `schema.connect` connection (REGEXP and dict rows are registered there) and returns
either row dicts or `approaches/agentic/search.py::Candidate` objects, which is the shape AGENTIC2 already feeds to
the v1 verifier, the verbatim guard and the reconciler. A candidate carries:

    Candidate(doc_id, section_id, unit_ids, text, span, kind, hit_unit_ids, round_no)
      .text            the snippet the verifier reads, a verbatim slice of Document.text whenever `span` is set
      .span            [start, end) into Document.text, or None when the snippet was assembled (table rows)
      .kind            unit | table | clip | passage | lead   (the window kind, as in v1 and v2)
      .unit_ids        every unit whose text the snippet contains, in document order
      .hit_unit_ids    the unit(s) retrieval actually matched
      .score           the retrieval score (fact score, unit score, or vector similarity)
      .features        {"src": [...], "vec": float, "fact": {...}, "db": {...}}
      .cid             "c1", "c2", ... assigned by select_candidates; .verdict left None for the verifier

Retrieval paths (spec section 6 stage 2), all of them filtered to the target documents:

  documents   `document_metas` is the listing the planner chooses from; it is `agentic/docmeta.py` read from SQL
              instead of from Document objects, and it returns the same dicts, so `company_listing` and
              `select_docs` are imported from there rather than re-implemented.
  facts       trigram prefilter on the literal stems of the plan terms -> `label_fold REGEXP <terms>` -> typed
              features. Scored on distinct terms matched, all terms matched, unit-kind agreement, year agreement,
              table/kpi source, label specificity and the extractor score.
  sections    `title_fold` / `path_fold REGEXP <terms>`: the sections that NAME the topic. Used twice - as a tree
              prior over the facts and units underneath them (recursive CTE on `parent_id`), and as the answer to
              a locate question (`sections.path` IS the answer).
  units       for non-numeric answers: trigram prefilter -> `text_fold REGEXP` -> a term-near-term proximity
              window. This is the v1 grep search with ripgrep replaced by the index; the window rules are v1's.
  vectors     `passages.vec` (float32 BLOB) as one numpy matrix per document, cosine, with v2's blend
              0.65 * question + 0.35 * terms. The query vector is an ARGUMENT: this module never calls a model.
  verified    `verified_facts` for warm mode, behind `warm=True`, which defaults to OFF (spec section 8: COLD is
              the headline and every question must stay independent).

Tree expansion (stage 4) has the same semantics as `approaches/agentic2/tree.py` - a bare number gets its label, a
table row gets its caption and header row, every candidate carries its section path - but the mechanism is SQL:
section units by `units.ord`, ancestors and descendants by a recursive CTE on `sections.parent_id`, table parts by
`units.table_id`. The edge rules themselves are imported from `tree.py` (`_grow`, `_is_table`, `_letters`) so they
cannot drift; a candidate's text is rebuilt from `units.text` joined by "\\n\\n", which is exactly `Document.text`
over consecutive units of one section (a section block is its title, then its units separated by a blank line).

Four deliberate departures from the letter of AGENTIC3_SPEC.md section 6, each backed by a number in
`work/analysis/DB_CEILING.md`:

 1. The unit-kind filter is a SCORE, not a `WHERE` clause (`kind_filter="score"`, the default). The measured gain of
    the filter is real (pool 62 -> 31 rows, top-1 8 -> 15 of 47, median gold rank 8 -> 2) and a large score weight
    reproduces it for the top-k, but the hard clause also deleted 3 gold rows outright, among them
    `C.shaparak...::p14_b7` (`value='11/26' unit=''`, rank 144 -> lost). `kind_filter="hard"` and `"off"` exist for
    the ablation.
 2. The year is never a `WHERE` clause either (`year_filter="score"`). The spec's
    `year BETWEEN y0 AND y1 OR year IS NULL` lost the gold row in 5 of 47 pairs where the unit-kind filter had just
    put it at rank 1.
 3. Years are compared THROUGH the calendar. `B.digikala-1401.en` carries `year = 1401` on 368 of its 370 fact rows
    while an English plan asks for `[2022, 2023]`; `year_score` converts the period into the fact's calendar
    (jalali = gregorian - 621, one year of slack) before deciding, so that pair agrees instead of emptying the pool.
 4. The trigram prefilter is not applied to `sections`: the section search also reads `path_fold`, which has no
    trigram rows (`tri.kind = 'section_title'` indexes titles), so a prefilter there would lose ancestor matches.
    The sections table is ~100 rows per document, which is why that costs nothing.

Two smaller differences from AGENTIC2's in-memory fact table, both stated here because they change what is found.
The fact regex reads `label_fold` ALONE, where `factdb.query` reads `label OR unit`: `tri` indexes `label_fold`,
so adding a second column would make the prefilter unsound (a row matching only the unit word has no trigram row
for that stem), and a term that matches only a unit word is a weak term. What that costs is covered by the second
difference: when the fact path keeps fewer than `units_fallback_min` rows, `retrieve` also runs the unit path even
for a numeric question, because a value whose fact label names nothing (`C.snapp_annual-report_1398.fa::p1_b37`
labels 430 نفر with "پاسخ این سوال") is still reachable through the unit text.

Tunables live in `DEFAULTS` (not fairness-critical, copied into the trace by the approach); the fairness-critical
values stay in config.yaml.
"""
import json
import re
import time

import numpy as np

from ..normalize import fold, normalize
from ..approaches.agentic import search as S
from ..approaches.agentic.docmeta import company_listing, select_docs  # noqa: F401  (re-exported for the planner)
from ..approaches.agentic.planner import _FA_JOIN, term_patterns
from ..approaches.agentic2 import tree as T
from ..approaches.agentic2.defaults import DEFAULTS as A2_DEFAULTS

QUERY_VERSION = "kb-query-v1"

VISIBLE = ("ready", "vector_ready")

# the four expansion keys come from AGENTIC2 so tree expansion cannot drift from tree.py
EXPAND_KEYS = ("window_chars", "neighbour_chars", "window_neighbours", "table_max_chars")

DEFAULTS = {k: A2_DEFAULTS[k] for k in EXPAND_KEYS}
DEFAULTS.update({
    # ---- pools and caps -------------------------------------------------------------------------------------
    "facts_max_rows": 24,              # fact rows turned into candidates, best first
    "facts_rows_per_doc": 8,           # ... per document; one targeted document gets the whole budget
    "sections_max_rows": 12,
    "units_max_rows": 12,
    "units_rows_per_doc": 6,
    "units_fallback_min": 3,           # too few fact rows: run the unit path as well, even for a number
    "vec_top_k_single": 12,
    "vec_top_k_per_doc": 5,
    "vec_terms_weight": 0.35,          # v2's blend: (1 - w) * sim(question) + w * sim(terms)
    "max_candidates": 12,
    "max_candidates_multi": 24,

    # ---- typed filters --------------------------------------------------------------------------------------
    "kind_filter": "score",            # score | hard | off
    "year_filter": "score",            # score | hard | off
    "year_slack": 1,                   # years of slack after a calendar conversion

    # ---- fact score -----------------------------------------------------------------------------------------
    "w_term": 1.0,                     # per distinct term concept matched in the label
    "w_all_terms": 0.6,                # every term of one language matched
    "w_kind": 1.2,                     # unit_kind compatible with the plan's expected kind
    "w_kind_conflict": -1.5,           # a different NUMERIC kind (percent asked, money found)
    "w_year_own": 0.5,                 # the fact dates itself inside the period
    "w_year_doc": 0.15,                # ... or inherits the document's year
    "w_year_miss": -0.8,               # comparable calendars, year outside the period (halved for inherited years)
    "w_source_table": 0.4,
    "w_source_kpi": 0.3,
    "w_label_cover": 0.8,              # share of the label's words the matched terms cover
    "w_label_len": 0.4,                # label specificity: min(words, 6) / 6
    "w_score": 0.1,                    # the extractor's own score
    "w_section": 0.6,                  # tree prior: the row sits under a section whose title names the topic

    # ---- section score --------------------------------------------------------------------------------------
    "w_title": 1.0,                    # a term in the section's own title
    "w_path": 0.4,                     # ... in an ancestor title only
    "prior_decay": 0.5,                # the prior of a matched section, per level below it
    "prior_depth": 3,
    "lead_units": 3,                   # units of a section shown as the evidence of a locate answer

    # ---- unit search ----------------------------------------------------------------------------------------
    "near_chars": 120,                 # two DIFFERENT terms this close in one unit is the v1 proximity window
    "w_near": 1.0,
})

# the plan's `unit` (planner.UNIT_KINDS) -> the typed `facts.unit_kind` values that can answer it
KIND_COMPATIBLE = {
    "count": ("count", "people"),
    "percent": ("percent",),
    "currency": ("money",),
    "quantity": ("count", "mass", "energy", "length", "area", "time", "other"),
    "ratio": ("ratio", "percent"),
    "date": ("time",),
    "text": (),
}
NUMERIC_KINDS = ("people", "count", "money", "percent", "mass", "energy", "length", "area", "time", "ratio")


def _ms(t0):
    return round((time.perf_counter() - t0) * 1000.0, 1)


def _in(n):
    return ",".join("?" * n)


# ---------------------------------------------------------------------------------------------------------------
# terms: the plan's literals as folded regexes, and the trigram stems that prefilter them
# ---------------------------------------------------------------------------------------------------------------

_STRUCTURAL = set("|.^$*+?{}")
_WORD = re.compile(r"[^\W\d_]+", re.UNICODE)
_JOIN_MARK = "~j~"                  # stands in for planner._FA_JOIN while the pattern is folded


def fold_pattern(pattern):
    """`planner.term_patterns` builds its regexes over normalised text; this moves them onto the `*_fold` columns.

    Folding the pattern as a string is otherwise safe - every metacharacter `planner._esc` emits is ASCII
    punctuation, which `fold` leaves alone - with one exception that has to be protected: the Persian word join
    `ٔ?[ZWNJ ]?` starts with an optional ezafe hamza, and folding removes harakat, which would leave a bare `?`
    quantifying the preceding letter (`سهم?` = an optional `م`). The join is replaced first and restored after as
    `[ ]?`, which is what it means in folded text, where a ZWNJ is already a space.
    """
    return fold(pattern.replace(_FA_JOIN, _JOIN_MARK)).replace(_JOIN_MARK, "[ ]?")


def pattern_runs(pattern):
    """The literal runs a match is GUARANTEED to contain, in order.

    Character classes, quantifiers and anything inside a group break a run: only what sits at the top level of the
    pattern is required (`(?:سرمایگان|سرمایه)` requires neither alternative on its own). Only `term_patterns`
    output is parsed here, so the alphabet is literals, `\\`-escapes, `(?:...|...)` groups, `[...]` classes and `?`
    quantifiers.
    """
    runs, cur, depth, i, n = [], [], 0, 0, len(pattern)

    def flush():
        if cur:
            runs.append("".join(cur))
            cur.clear()

    while i < n:
        ch = pattern[i]
        if ch == "\\" and i + 1 < n:
            lit, i = pattern[i + 1], i + 2
        elif ch == "(":
            flush()
            depth += 1
            i += 3 if pattern[i:i + 3] == "(?:" else 1
            continue
        elif ch == ")":
            flush()
            depth -= 1
            i += 1
            if i < n and pattern[i] in "?*+":
                i += 1
            continue
        elif ch == "[":
            j = pattern.find("]", i + 1)
            i = n if j < 0 else j + 1
            if i < n and pattern[i] in "?*+":
                i += 1
            flush()
            continue
        elif ch in _STRUCTURAL:
            flush()
            i += 1
            continue
        else:
            lit, i = ch, i + 1
        if i < n and pattern[i] in "?*":
            flush()
            i += 1
            continue
        if i < n and pattern[i] == "+":
            i += 1
        if depth == 0:
            cur.append(lit)
    flush()
    return runs


class Terms:
    """The plan's terms as folded patterns, grouped into CONCEPTS (the v1 SearchContext rule: terms whose patterns
    coincide are one concept, so a singular and a plural do not count twice).

    `stems` are the literal runs of at least `min_stem` characters the trigram index can prefilter on. A term with
    no such run makes the whole prefilter unusable (`indexable` False) rather than silently losing rows.
    """

    def __init__(self, plan, d_lang=None, langs=("fa", "en"), min_stem=3):
        terms = plan.get("terms") or {}
        self.min_stem = min_stem
        self.d_lang = d_lang
        self.by_lang = {l: [t for t in (terms.get(l) or []) if t and t.strip()] for l in langs}
        self.items = []                     # [{term, lang, concept, pattern, rx, stems}]
        self.lang_concepts = {l: set() for l in langs}
        concept_of, n = {}, 0
        for lang in langs:
            for term in self.by_lang[lang]:
                pats = [p for p in (fold_pattern(x) for x in term_patterns(normalize(term), lang)) if p]
                if not pats:
                    continue
                ci = next((concept_of[p] for p in pats if p in concept_of), None)
                if ci is None:
                    ci = n
                    n += 1
                self.lang_concepts[lang].add(ci)
                for p in pats:
                    if p in concept_of:
                        continue
                    concept_of[p] = ci
                    self.items.append({"term": term, "lang": lang, "concept": ci, "pattern": p,
                                       "rx": re.compile(p, re.I),
                                       "stems": [r for r in pattern_runs(p) if len(r) >= min_stem]})
        self.n_concepts = n
        self.any_pattern = "|".join("(?:%s)" % it["pattern"] for it in self.items)
        self.indexable = bool(self.items) and all(it["stems"] for it in self.items)

    def __bool__(self):
        return bool(self.items)

    def concepts_in(self, text):
        """-> the set of concept ids whose pattern matches (`text` must already be folded)."""
        return {it["concept"] for it in self.items if it["rx"].search(text or "")}

    def spans(self, text):
        """-> [(concept, start, end)] of every match in folded text, in text order."""
        out = [(it["concept"], m.start(), m.end()) for it in self.items for m in it["rx"].finditer(text or "")]
        out.sort(key=lambda x: (x[1], x[2]))
        return out

    def covers_a_language(self, concepts):
        """True when the matched concepts cover every term of at least one language that has terms."""
        cs = set(concepts)
        return any(want and want <= cs for want in self.lang_concepts.values())

    def trace(self):
        return {"terms": [{"term": it["term"], "lang": it["lang"], "concept": it["concept"],
                           "pattern": it["pattern"], "stems": it["stems"]} for it in self.items],
                "concepts": self.n_concepts, "indexable": self.indexable}


# ---------------------------------------------------------------------------------------------------------------
# the trigram prefilter
# ---------------------------------------------------------------------------------------------------------------

def tri_query(terms):
    """The fts5 MATCH expression for `terms`: ("stem" AND "stem") OR ("stem") ... -> None when unusable.

    Sound by construction: a row whose text matches a term's regex contains every literal run of that pattern, so
    ANDing the runs of one pattern cannot exclude a row the regex would keep. The regex still decides; the index
    only says which rows CAN match.
    """
    if not terms.indexable:
        return None
    parts = ["(%s)" % " AND ".join('"%s"' % s.replace('"', '""') for s in it["stems"]) for it in terms.items]
    return " OR ".join(dict.fromkeys(parts))


def tri_filter(terms, kind, n_docs):
    """-> (SQL fragment, params) selecting the ref ids the index allows, or (None, []).

    A subquery rather than an id list: the prefilter of one Persian stem can return thousands of rows, well past
    SQLite's host-parameter limit.
    """
    q = tri_query(terms)
    if q is None:
        return None, []
    return ("SELECT ref_id FROM tri WHERE tri MATCH ? AND kind = ? AND doc_id IN (%s)" % _in(n_docs)), [q, kind]


def tri_refs(con, kind, doc_ids, terms):
    """-> (set of ref_id strings, info); `None` for the set means the index cannot prefilter this term set."""
    t0 = time.perf_counter()
    sql, params = tri_filter(terms, kind, len(doc_ids))
    if sql is None or not doc_ids:
        return None, {"used": False, "reason": "unindexable" if sql is None else "no_docs", "ms": _ms(t0)}
    refs = {r["ref_id"] for r in con.execute(sql, params + list(doc_ids))}
    return refs, {"used": True, "match": params[0], "refs": len(refs), "ms": _ms(t0)}


# ---------------------------------------------------------------------------------------------------------------
# stage 1: the document listing the planner chooses from
# ---------------------------------------------------------------------------------------------------------------

def document_metas(con, doc_ids=None, corpus=None, statuses=VISIBLE):
    """The visible documents as `agentic/docmeta.py` dicts, plus the columns only the database has.

    Same keys as `docmeta.doc_meta` - doc_id, key, company, years, title - so `company_listing` and `select_docs`
    read these rows unchanged; `lang`, `calendar`, `corpus`, `quality`, `status`, `n_facts`, `n_units`, `pair_id`
    and `content_sha1` are the extras the query side needs.
    """
    sql = ("SELECT doc_id, corpus, key, lang, company, year_from, year_to, calendar, title, pair_id, quality, "
           "status, n_facts, n_units, content_sha1 FROM documents WHERE status IN (%s)" % _in(len(statuses)))
    args = list(statuses)
    if doc_ids is not None:
        sql += " AND doc_id IN (%s)" % _in(len(doc_ids))
        args += list(doc_ids)
    if corpus:
        sql += " AND corpus = ?"
        args.append(corpus)
    out = []
    for r in con.execute(sql + " ORDER BY doc_id", args):
        out.append(dict(r, years=(r["year_from"], r["year_to"]) if r["year_from"] is not None else None))
    return out


def select_documents(con, *, doc_scope=None, companies=None, period=None, kind="single", question="",
                     max_docs=12, selection=True, corpus=None):
    """Stage 1's document choice: the listing from SQL, the rule from `docmeta.select_docs` (unchanged)."""
    metas = document_metas(con, doc_ids=doc_scope, corpus=corpus)
    chosen, info = select_docs(metas, companies_chosen=companies, period=period, kind=kind, question=question,
                               max_docs=max_docs, selection=selection)
    info["n_visible"] = len(metas)
    return chosen, info


# ---------------------------------------------------------------------------------------------------------------
# typed scoring: unit kind and year
# ---------------------------------------------------------------------------------------------------------------

def compatible_kinds(plan_unit):
    return KIND_COMPATIBLE.get(plan_unit or "", ())


def kind_score(row, plan_unit, opts):
    """-> (score, label). `unit_norm` is the EMPTY STRING for a bare count, never NULL, so the kind column decides."""
    want = compatible_kinds(plan_unit)
    if not want:
        return 0.0, "no_kind"
    kind = row["unit_kind"] or "other"
    if kind in want or (plan_unit == "percent" and row["is_percent"]):
        return float(opts["w_kind"]), "match"
    if kind in NUMERIC_KINDS:
        return float(opts["w_kind_conflict"]), "conflict"
    return 0.0, "other"


def calendar_of(year):
    if year is None:
        return None
    if 1200 <= year < 1600:
        return "jalali"
    if 1700 <= year < 2200:
        return "gregorian"
    return None


def convert_years(period, to_calendar):
    """-> ((lo, hi) in `to_calendar`, converted?). Jalali = gregorian - 621."""
    lo, hi = period
    src = calendar_of(lo if lo is not None else hi)
    if src is None or to_calendar is None or src == to_calendar:
        return period, False
    shift = -621 if to_calendar == "jalali" else 621
    return (None if lo is None else lo + shift, None if hi is None else hi + shift), True


def year_score(row, period, opts):
    """-> (score, label). Never a filter: a year mismatch is a penalty (DB_CEILING instruction 3).

    The period is converted into the FACT's calendar before comparing, so an English plan asking for [2022, 2023]
    agrees with the Jalali year 1401 that `B.digikala-1401.en` inherits on 368 of its 370 rows.
    """
    if not period or all(p is None for p in period):
        return 0.0, "no_period"
    if row["year"] is None:
        return 0.0, "no_year"
    cal = row["calendar"] or calendar_of(row["year"])
    (lo, hi), converted = convert_years(tuple(period), cal)
    slack = int(opts["year_slack"]) if converted else 0
    lo = -10 ** 9 if lo is None else lo - slack
    hi = 10 ** 9 if hi is None else hi + slack
    y0 = row["year"]
    y1 = row["year_to"] if row["year_to"] is not None else y0
    own = (row["year_source"] or "") == "own"
    if y0 <= hi and y1 >= lo:
        return float(opts["w_year_own"] if own else opts["w_year_doc"]), "in_period"
    return float(opts["w_year_miss"]) * (1.0 if own else 0.5), "out_of_period"


# ---------------------------------------------------------------------------------------------------------------
# 2a. facts
# ---------------------------------------------------------------------------------------------------------------

FACT_SELECT = ("SELECT f.fact_id, f.doc_id, f.section_id, f.unit_id, f.ord, f.label, f.label_fold, f.row_label, "
               "f.col_header, f.value_text, f.value_num, f.value_hi, f.value_cmp, f.alt_readings, f.unit_text, "
               "f.unit_norm, f.unit_kind, f.currency, f.is_percent, f.period_text, f.year, f.year_to, "
               "f.period_kind, f.calendar, f.year_source, f.def_class, f.source_kind, f.char_start, f.char_end, "
               "f.quote, f.score, s.path AS section_path, s.title AS section_title "
               "FROM facts f LEFT JOIN sections s ON s.section_id = f.section_id")


def facts_search(con, doc_ids, terms, *, plan_unit=None, period=None, priors=None, opts=None, trigram=True,
                 max_rows=None, per_doc=None):
    """The typed fact rows that can answer the question, best first. -> (rows, info).

    A row is its `facts` row plus `section_path`, `section_title`, `concepts` (the term concepts its label matches),
    `features` (the score breakdown) and `score`.
    """
    o = dict(DEFAULTS, **(opts or {}))
    t0 = time.perf_counter()
    info = {"docs": list(doc_ids), "kind_filter": o["kind_filter"], "year_filter": o["year_filter"],
            "plan_unit": plan_unit, "period": list(period) if period else None}
    if not doc_ids or not terms:
        info.update(matched=0, kept=0, ms=_ms(t0))
        return [], info
    sql = FACT_SELECT + " WHERE f.doc_id IN (%s) AND f.label_fold REGEXP ?" % _in(len(doc_ids))
    args = list(doc_ids) + [terms.any_pattern]
    sub, params = tri_filter(terms, "fact_label", len(doc_ids)) if trigram else (None, [])
    info["trigram"] = {"used": sub is not None,
                       "reason": None if sub is not None else ("disabled" if not trigram else "unindexable")}
    if sub is not None:
        sql += " AND f.fact_id IN (SELECT CAST(ref_id AS INTEGER) FROM (%s))" % sub
        args += params + list(doc_ids)
    if o["kind_filter"] == "hard" and compatible_kinds(plan_unit):
        kinds = compatible_kinds(plan_unit)
        sql += " AND (f.unit_kind IN (%s) OR f.is_percent = 1)" % _in(len(kinds))
        args += list(kinds)
    if o["year_filter"] == "hard" and period and any(p is not None for p in period):
        sql += " AND (f.year IS NULL OR (f.year <= ? AND coalesce(f.year_to, f.year) >= ?))"
        args += [period[1] if period[1] is not None else 10 ** 9,
                 period[0] if period[0] is not None else -10 ** 9]
    t1 = time.perf_counter()
    rows = [dict(r) for r in con.execute(sql, args)]
    info["sql_ms"] = _ms(t1)
    info["matched"] = len(rows)
    priors = priors or {}
    for r in rows:
        r["concepts"] = sorted(terms.concepts_in(r["label_fold"] or ""))
        r["score"], r["features"] = score_fact(r, terms, plan_unit, period, priors, o)
    rows.sort(key=lambda r: (-r["score"], -(r["value_num"] is not None), r["doc_id"], r["ord"]))
    cap = int(max_rows if max_rows is not None else o["facts_max_rows"])
    # one targeted document gets the whole budget (v2's rule); several share it, so every period is represented
    cap_doc = int(per_doc if per_doc is not None else (cap if len(doc_ids) == 1 else o["facts_rows_per_doc"]))
    kept, per = [], {}
    for r in rows:
        if per.get(r["doc_id"], 0) >= cap_doc:
            continue
        per[r["doc_id"]] = per.get(r["doc_id"], 0) + 1
        kept.append(r)
        if len(kept) >= cap:
            break
    info.update(kept=len(kept), per_doc=cap_doc, cap=cap, ms=_ms(t0))
    return kept, info


def label_cover(terms, label_fold):
    """-> the share of the label's words that lie inside a term match. A bare label (`شرکت`, `firms`) and a
    specific one (`مجموع نیروی انسانی گروه شرکتی`) both match one term; this is what tells them apart."""
    words = list(_WORD.finditer(label_fold or ""))
    if not words:
        return 0.0, 0
    hits = terms.spans(label_fold or "")
    covered = sum(1 for w in words if any(a < w.end() and b > w.start() for _c, a, b in hits))
    return covered / len(words), len(words)


def score_fact(row, terms, plan_unit, period, priors, opts):
    """-> (score, features). Ranks for SPECIFICITY, not only for term count (DB_CEILING instruction 4)."""
    o = opts
    f = {}
    n = len(row["concepts"])
    s = float(o["w_term"]) * n
    f["terms"] = n
    if n and terms.covers_a_language(row["concepts"]):
        s += float(o["w_all_terms"])
        f["all_terms"] = 1
    ks, f["kind"] = kind_score(row, plan_unit, o)
    ys, f["year"] = year_score(row, period, o)
    s += ks + ys
    src = row["source_kind"] or "text"
    f["source_kind"] = src
    if src == "table":
        s += float(o["w_source_table"])
    elif src == "kpi":
        s += float(o["w_source_kpi"])
    cover, n_words = label_cover(terms, row["label_fold"])
    s += float(o["w_label_cover"]) * cover + float(o["w_label_len"]) * (min(n_words, 6) / 6.0)
    f["label_cover"] = round(cover, 3)
    f["label_words"] = n_words
    s += float(o["w_score"]) * float(row["score"] or 0.0)
    prior = priors.get(row["section_id"], 0.0)
    if prior:
        s += float(o["w_section"]) * prior
        f["section_prior"] = round(prior, 3)
    return round(s, 4), f


# ---------------------------------------------------------------------------------------------------------------
# 2b. sections: the tree prior, and the answer to a locate question
# ---------------------------------------------------------------------------------------------------------------

def sections_search(con, doc_ids, terms, *, opts=None, max_rows=None):
    """Sections whose own TITLE, or an ancestor's, names the topic - best first. -> (rows, info).

    `sections.path` is the answer to a locate question; the same rows feed `section_priors`.
    """
    o = dict(DEFAULTS, **(opts or {}))
    t0 = time.perf_counter()
    info = {"docs": list(doc_ids), "trigram": {"used": False, "reason": "path_fold_not_indexed"}}
    if not doc_ids or not terms:
        info.update(matched=0, kept=0, ms=_ms(t0))
        return [], info
    sql = ("SELECT section_id, doc_id, parent_id, level, ord, title, title_fold, path, path_fold, span_start, "
           "span_end, subtree_end, n_units, n_facts FROM sections WHERE doc_id IN (%s) "
           "AND (title_fold REGEXP ? OR path_fold REGEXP ?)" % _in(len(doc_ids)))
    rows = [dict(r) for r in con.execute(sql, list(doc_ids) + [terms.any_pattern, terms.any_pattern])]
    info["matched"] = len(rows)
    for r in rows:
        title_hits = terms.concepts_in(r["title_fold"] or "")
        path_hits = terms.concepts_in(r["path_fold"] or "")
        r["concepts"] = sorted(title_hits | path_hits)
        r["score"] = round(float(o["w_title"]) * len(title_hits)
                           + float(o["w_path"]) * len(path_hits - title_hits), 4)
        r["features"] = {"title_terms": len(title_hits), "path_terms": len(path_hits)}
    rows = [r for r in rows if r["score"] > 0]
    rows.sort(key=lambda r: (-r["score"], r["doc_id"], r["ord"]))
    kept = rows[:int(max_rows if max_rows is not None else o["sections_max_rows"])]
    info.update(kept=len(kept), ms=_ms(t0))
    return kept, info


def subtree(con, section_ids, max_depth=None):
    """-> [(root_id, section_id, depth)] for each matched section and everything under it, by a recursive CTE on
    `sections.parent_id`."""
    ids = list(section_ids)
    if not ids:
        return []
    sql = ("WITH RECURSIVE sub(root, section_id, depth) AS ("
           " SELECT section_id, section_id, 0 FROM sections WHERE section_id IN (%s)"
           " UNION ALL"
           " SELECT sub.root, s.section_id, sub.depth + 1 FROM sections s JOIN sub ON s.parent_id = sub.section_id"
           " %s) "
           "SELECT root, section_id, min(depth) AS depth FROM sub GROUP BY root, section_id"
           % (_in(len(ids)), "WHERE sub.depth < ?" if max_depth is not None else ""))
    args = ids + ([int(max_depth)] if max_depth is not None else [])
    return [(r["root"], r["section_id"], r["depth"]) for r in con.execute(sql, args)]


def descendants(con, section_ids, max_depth=None):
    """-> {section_id: depth below the nearest of `section_ids`}."""
    out = {}
    for _root, sid, depth in subtree(con, section_ids, max_depth):
        if depth < out.get(sid, 10 ** 9):
            out[sid] = depth
    return out


def ancestors(con, section_id):
    """-> [section row] from the root down to `section_id` (recursive CTE on `parent_id`)."""
    sql = ("WITH RECURSIVE up(section_id, parent_id, title, level, ord, steps) AS ("
           " SELECT section_id, parent_id, title, level, ord, 0 FROM sections WHERE section_id = ?"
           " UNION ALL"
           " SELECT s.section_id, s.parent_id, s.title, s.level, s.ord, up.steps + 1"
           " FROM sections s JOIN up ON s.section_id = up.parent_id) "
           "SELECT section_id, parent_id, title, level, ord FROM up ORDER BY steps DESC")
    return [dict(r) for r in con.execute(sql, (section_id,))]


def section_priors(con, section_rows, opts=None):
    """-> {section_id: weight in (0, 1]} for the matched sections and everything under them, decayed per level."""
    o = dict(DEFAULTS, **(opts or {}))
    if not section_rows:
        return {}
    best = {}
    for r in section_rows:
        best[r["section_id"]] = max(best.get(r["section_id"], 0.0), float(r["score"]))
    top = max(best.values()) or 1.0
    decay = float(o["prior_decay"])
    out = {}
    for root, sid, depth in subtree(con, list(best), max_depth=int(o["prior_depth"])):
        w = (decay ** depth) * best[root] / top
        if w > out.get(sid, 0.0):
            out[sid] = round(w, 4)
    return out


# ---------------------------------------------------------------------------------------------------------------
# 2c. units: the v1 grep search moved into the database
# ---------------------------------------------------------------------------------------------------------------

def units_search(con, doc_ids, terms, *, opts=None, trigram=True, priors=None, max_rows=None, per_doc=None):
    """Units whose text carries the terms, scored by a term-near-term proximity window. -> (rows, info).

    The window rule is v1's: a unit counts when two DIFFERENT term concepts occur within `near_chars` of each
    other, and the closest such pair scores highest. A single term still matches, at a lower score.
    """
    o = dict(DEFAULTS, **(opts or {}))
    t0 = time.perf_counter()
    info = {"docs": list(doc_ids)}
    if not doc_ids or not terms:
        info.update(matched=0, kept=0, ms=_ms(t0))
        return [], info
    sql = ("SELECT unit_id, doc_id, section_id, ord, kind, lang, text, text_fold, span_start, span_end, table_id, "
           "n_numbers FROM units WHERE doc_id IN (%s) AND text_fold REGEXP ?" % _in(len(doc_ids)))
    args = list(doc_ids) + [terms.any_pattern]
    sub, params = tri_filter(terms, "unit", len(doc_ids)) if trigram else (None, [])
    info["trigram"] = {"used": sub is not None,
                       "reason": None if sub is not None else ("disabled" if not trigram else "unindexable")}
    if sub is not None:
        sql += " AND unit_id IN (%s)" % sub
        args += params + list(doc_ids)
    t1 = time.perf_counter()
    rows = [dict(r) for r in con.execute(sql, args)]
    info["sql_ms"] = _ms(t1)
    info["matched"] = len(rows)
    priors = priors or {}
    for r in rows:
        spans = terms.spans(r["text_fold"] or "")
        r["concepts"] = sorted({c for c, _a, _b in spans})
        gap = nearest_pair(spans)
        r["near_chars"] = gap
        s = float(o["w_term"]) * len(r["concepts"])
        if gap is not None and gap <= int(o["near_chars"]):
            s += float(o["w_near"])
        if terms.covers_a_language(r["concepts"]):
            s += float(o["w_all_terms"])
        prior = priors.get(r["section_id"], 0.0)
        s += float(o["w_section"]) * prior
        r["score"] = round(s, 4)
        r["features"] = {"terms": len(r["concepts"]), "near_chars": gap, "section_prior": round(prior, 3)}
    rows.sort(key=lambda r: (-r["score"], r["doc_id"], r["ord"]))
    cap_doc = int(per_doc if per_doc is not None else o["units_rows_per_doc"])
    kept, per = [], {}
    for r in rows:
        if per.get(r["doc_id"], 0) >= cap_doc:
            continue
        per[r["doc_id"]] = per.get(r["doc_id"], 0) + 1
        kept.append(r)
        if len(kept) >= int(max_rows if max_rows is not None else o["units_max_rows"]):
            break
    info.update(kept=len(kept), ms=_ms(t0))
    return kept, info


def nearest_pair(spans):
    """-> the smallest gap in characters between two matches of DIFFERENT concepts, or None."""
    best = None
    for i, (ci, _ai, bi) in enumerate(spans):
        for cj, aj, _bj in spans[i + 1:]:
            if cj == ci:
                continue
            gap = max(0, aj - bi)
            if best is None or gap < best:
                best = gap
    return best


# ---------------------------------------------------------------------------------------------------------------
# stage 4: tree expansion, the semantics of agentic2/tree.py read through SQL
# ---------------------------------------------------------------------------------------------------------------

class _Unit:
    """What `tree.py`'s edge rules need of a unit: `.id`, `.text`, `.kind`, `.span`."""
    __slots__ = ("id", "section_id", "ord", "kind", "text", "span", "table_id")

    def __init__(self, row):
        self.id = row["unit_id"]
        self.section_id = row["section_id"]
        self.ord = row["ord"]
        self.kind = row["kind"]
        self.text = row["text"] or ""
        self.span = [row["span_start"], row["span_end"]]
        self.table_id = row["table_id"]


class DocReader:
    """One question's read cache: a document's units (by section, in `units.ord` order), its sections, its title.

    A candidate's text is rebuilt as "\\n\\n".join of consecutive units of one section, which is exactly
    `Document.text[first.span_start:last.span_end]`, so an expanded candidate is still a verbatim slice and keeps
    its span - what the verbatim guard checks.
    """

    def __init__(self, con):
        self.con = con
        self._units = {}         # doc_id -> {section_id: [_Unit]}
        self._by_id = {}         # doc_id -> {unit_id: (section_id, index)}
        self._sections = {}      # doc_id -> {section_id: row}
        self._titles = {}

    def load(self, doc_id):
        if doc_id in self._units:
            return
        by_sec, by_id = {}, {}
        for r in self.con.execute(
                "SELECT unit_id, section_id, ord, kind, text, span_start, span_end, table_id FROM units "
                "WHERE doc_id = ? ORDER BY ord", (doc_id,)):
            u = _Unit(r)
            lst = by_sec.setdefault(u.section_id, [])
            by_id[u.id] = (u.section_id, len(lst))
            lst.append(u)
        self._units[doc_id] = by_sec
        self._by_id[doc_id] = by_id
        self._sections[doc_id] = {r["section_id"]: dict(r) for r in self.con.execute(
            "SELECT section_id, parent_id, level, ord, title, path, span_start, span_end, subtree_end "
            "FROM sections WHERE doc_id = ?", (doc_id,))}
        row = self.con.execute("SELECT title FROM documents WHERE doc_id = ?", (doc_id,)).fetchone()
        self._titles[doc_id] = (row or {}).get("title") or doc_id

    def units(self, doc_id, section_id):
        self.load(doc_id)
        return self._units[doc_id].get(section_id, [])

    def index_of(self, doc_id, unit_id):
        """-> (section_id, index within the section) or None."""
        self.load(doc_id)
        return self._by_id[doc_id].get(unit_id)

    def section(self, doc_id, section_id):
        self.load(doc_id)
        return self._sections[doc_id].get(section_id, {})

    def title(self, doc_id):
        self.load(doc_id)
        return self._titles[doc_id]

    def path(self, doc_id, section_id):
        return self.section(doc_id, section_id).get("path") or ""

    def breadcrumb(self, doc_id, section_id):
        """"<doc title> > <section path>": the header the verifier and the answer call see."""
        return " > ".join(p for p in (self.title(doc_id), self.path(doc_id, section_id)) if p)

    def table_parts(self, doc_id, table_id):
        """-> [_Unit] of one table in document order (corpus A `#table.<k>`, a PDF `p012_t1`)."""
        self.load(doc_id)
        if not table_id:
            return []
        out = [u for sec in self._units[doc_id].values() for u in sec if u.table_id == table_id]
        return sorted(out, key=lambda u: u.ord)

    def text_of(self, units, lo, hi):
        """-> (text, span) of units[lo..hi] of one section, verbatim in Document.text."""
        return "\n\n".join(u.text for u in units[lo:hi + 1]), [units[lo].span[0], units[hi].span[1]]


def table_candidate(reader, doc_id, section_id, idx, ctx, opts, value_tokens=(), round_no=1, tree=True):
    """`tree.py::table_candidate` over the database: caption + header row + the rows that carry a plan term or the
    matched value, then the following rows while `table_max_chars` allows.

    One addition a Document cannot make: when the unit is a LATER part of a multi-part table (`units.table_id`) its
    caption sits before the FIRST part, not before it, so the caption is looked up there and the candidate is then
    not a contiguous slice (span None). Every part repeats the header row, so line 0 is the header as in `tree.py`.
    """
    units = reader.units(doc_id, section_id)
    unit = units[idx]
    max_chars = int(opts["table_max_chars"])
    lines = unit.text.split("\n")
    if not tree:
        text = unit.text[:max_chars]
        return S.Candidate(doc_id, section_id, [unit.id], text, [unit.span[0], unit.span[0] + len(text)], "table",
                           [unit.id], round_no)
    want = set()
    for i, ln in enumerate(lines[1:], 1):
        if ctx is not None and ctx.has_term(ln):
            want.add(i)
        if value_tokens and any(v and v in ln for v in value_tokens):
            want.add(i)
    rows = [0] + sorted(want)
    used = sum(len(lines[i]) + 1 for i in rows)
    for i in range(1, len(lines)):          # fill with the remaining rows in order while they fit
        if i in want:
            continue
        if used + len(lines[i]) + 1 > max_chars:
            continue
        rows.append(i)
        used += len(lines[i]) + 1
    rows = sorted(set(rows))
    while len(rows) > 1 and sum(len(lines[i]) + 1 for i in rows) > max_chars:
        drop = next((i for i in reversed(rows) if i not in want and i != 0), None) or rows[-1]
        rows.remove(drop)
    body = "\n".join(lines[i] for i in rows)
    uids, text, start = [unit.id], body, unit.span[0]
    cap = _caption_index(reader, doc_id, section_id, idx, opts)
    adjacent = cap == idx - 1
    if cap is not None and len(units[cap].text) + len(body) + 2 <= max_chars + 250:
        prev = units[cap]
        text = prev.text + "\n\n" + body
        uids = [prev.id, unit.id]
        start = prev.span[0]
    else:
        adjacent = True
    span = [start, start + len(text)] if (adjacent and unit.text.startswith(body)) else None
    return S.Candidate(doc_id, section_id, uids, text, span, "table", [unit.id], round_no)


def _caption_index(reader, doc_id, section_id, idx, opts):
    """The index of the short caption / heading unit that labels this table, or None (`tree.py`'s rule, plus the
    first-part lookup for a continuation of a multi-part table)."""
    units = reader.units(doc_id, section_id)
    unit = units[idx]
    i = idx
    if unit.table_id:
        parts = [u.ord for u in reader.table_parts(doc_id, unit.table_id)]
        if parts and unit.ord != parts[0]:
            first = next((j for j, u in enumerate(units) if u.ord == parts[0]), None)
            i = idx if first is None else first
    if i <= 0:
        return None
    prev = units[i - 1]
    if prev.kind in ("caption", "heading") or (len(prev.text) <= int(opts["neighbour_chars"])
                                               and not T._is_table(prev)):
        return i - 1
    return None


def expand_unit(reader, doc_id, unit_id, value, ctx, opts, unit_kind, tree=True, round_no=1):
    """`tree.py::expand_unit` over the database -> Candidate (None when the unit is not in the database)."""
    at = reader.index_of(doc_id, unit_id)
    if at is None:
        return None
    section_id, idx = at
    units = reader.units(doc_id, section_id)
    unit = units[idx]
    max_chars = int(opts["window_chars"])
    if T._is_table(unit):
        return table_candidate(reader, doc_id, section_id, idx, ctx, opts, value_tokens=[value], round_no=round_no,
                               tree=tree)
    if len(unit.text) > max_chars:
        p = unit.text.find(value) if value else -1
        clips = S.clip_windows(None, unit, [max(p, 0)], max_chars, 1)
        if clips:
            text, span = clips[0]
            return S.Candidate(doc_id, section_id, [unit_id], text, span, "clip", [unit_id], round_no)
    if not tree:
        return S.Candidate(doc_id, section_id, [unit_id], unit.text, list(unit.span), "unit", [unit_id], round_no)
    lo, hi = T._grow(None, units, idx, idx, unit_kind, opts)
    if lo == hi == idx and T._letters(unit.text) < 12 and idx > 0:
        prev = units[idx - 1]           # a value whose label sits in the previous short unit (KPI line)
        if len(prev.text) <= int(opts["neighbour_chars"]) and not T._is_table(prev):
            lo = idx - 1
    text, span = reader.text_of(units, lo, hi)
    return S.Candidate(doc_id, section_id, [u.id for u in units[lo:hi + 1]], text, span, "unit", [unit_id],
                       round_no)


def expand_passage(reader, passage, ctx, opts, unit_kind, tree=True, round_no=1):
    """`tree.py::expand_passage` over the database -> [Candidate].

    `passages` stores no span, so a passage that is a WINDOW of one long unit is located by finding its text inside
    that unit; a passage covering whole units runs from the first unit's start to the last unit's end, which is how
    `tree.build_passages` built it.
    """
    doc_id, section_id = passage["doc_id"], passage["section_id"]
    units = reader.units(doc_id, section_id)
    pos = {u.id: i for i, u in enumerate(units)}
    pu = json.loads(passage["unit_ids"]) if isinstance(passage["unit_ids"], str) else list(passage["unit_ids"] or [])
    idxs = [pos[u] for u in pu if u in pos]
    out = []
    if not idxs:
        return out
    table_idx = [i for i in idxs if T._is_table(units[i])]
    if table_idx:
        for i in table_idx:
            out.append(table_candidate(reader, doc_id, section_id, i, ctx, opts, round_no=round_no, tree=tree))
        idxs = [i for i in idxs if i not in table_idx]
        if not idxs:
            return out
    lo, hi = min(idxs), max(idxs)
    a, b = passage_span(units, lo, hi, passage["text"])
    partial = len(idxs) == 1 and (a > units[lo].span[0] or b < units[hi].span[1])
    if partial or not tree:
        out.append(S.Candidate(doc_id, section_id, [units[i].id for i in range(lo, hi + 1)], passage["text"],
                               [a, b], "clip" if partial else "passage", [units[lo].id], round_no))
        return out
    lo, hi = T._grow(None, units, lo, hi, unit_kind, dict(opts, window_chars=max(int(opts["window_chars"]), b - a)))
    text, span = reader.text_of(units, lo, hi)
    out.append(S.Candidate(doc_id, section_id, [u.id for u in units[lo:hi + 1]], text, span, "passage",
                           [units[i].id for i in idxs], round_no))
    return out


def passage_span(units, lo, hi, text):
    """-> (start, end) of a passage in Document.text."""
    if lo == hi and text != units[lo].text:
        off = units[lo].text.find(text)
        if off >= 0:
            return units[lo].span[0] + off, units[lo].span[0] + off + len(text)
    return units[lo].span[0], units[hi].span[1]


def expand_fact(reader, row, ctx, opts, unit_kind, tree=True, round_no=1):
    """A fact row -> the Candidate the verifier reads: its unit, expanded around the value."""
    return expand_unit(reader, row["doc_id"], row["unit_id"], row["value_text"], ctx, opts, unit_kind, tree=tree,
                       round_no=round_no)


def lead_candidate(reader, doc_id, section_id, opts, round_no=1):
    """The first units of a section as one candidate - the evidence behind a locate answer (v1's "lead" window)."""
    units = reader.units(doc_id, section_id)
    if not units:
        return None
    max_chars = int(opts["window_chars"])
    hi, last = 0, min(len(units), int(opts["lead_units"])) - 1
    while hi < last and units[hi + 1].span[1] - units[0].span[0] <= max_chars:
        hi += 1
    text, span = reader.text_of(units, 0, hi)
    return S.Candidate(doc_id, section_id, [u.id for u in units[:hi + 1]], text, span, "lead", [], round_no)


# ---------------------------------------------------------------------------------------------------------------
# 2e. vectors: float32 BLOBs, one matrix per document
# ---------------------------------------------------------------------------------------------------------------

def load_matrix(con, doc_id):
    """-> ([passage row without `vec`], L2-normalised float32 matrix) in `passages.ord` order.

    Vectors are stored exactly as bge-m3 returned them, so they are normalised here the way `agentic2/lazyvec.py`
    normalises its matrix: same similarities, same ordering.
    """
    rows = [dict(r) for r in con.execute(
        "SELECT passage_id, doc_id, section_id, ord, unit_ids, text, embed_sha1, vec, dim FROM passages "
        "WHERE doc_id = ? AND vec IS NOT NULL ORDER BY ord", (doc_id,))]
    if not rows:
        return [], np.zeros((0, 1), dtype=np.float32)
    m = np.vstack([np.frombuffer(r.pop("vec"), dtype=np.float32) for r in rows]).astype(np.float32)
    n = np.linalg.norm(m, axis=1, keepdims=True)
    return rows, m / np.where(n > 0, n, 1.0)


def vector_search(con, doc_ids, qvec, terms_vec=None, *, weight=None, top_k=None, opts=None):
    """Cosine search over the stored passage vectors. -> (hits, info); a hit is (passage row, similarity).

    `qvec` and `terms_vec` are ARGUMENTS - embedding calls belong to the approach, never to this module. The blend
    is v2's: (1 - w) * sim(question) + w * sim(terms), w = `vec_terms_weight` (0.35).
    """
    o = dict(DEFAULTS, **(opts or {}))
    t0 = time.perf_counter()
    w = float(o["vec_terms_weight"] if weight is None else weight) if terms_vec is not None else 0.0
    k = int(top_k if top_k is not None else (o["vec_top_k_single"] if len(doc_ids) == 1 else o["vec_top_k_per_doc"]))
    q = unit_vector(qvec)
    tv = unit_vector(terms_vec) if terms_vec is not None else None
    hits, per_doc, unit_sim, missing = [], {}, {}, []
    for doc_id in doc_ids:
        rows, m = load_matrix(con, doc_id)
        if not len(m):
            missing.append(doc_id)
            continue
        sims = m @ q
        if w:
            sims = (1.0 - w) * sims + w * (m @ tv)
        for r, s in zip(rows, sims.tolist()):
            for u in json.loads(r["unit_ids"]):
                if s > unit_sim.get(u, -1.0):
                    unit_sim[u] = s
        order = np.argsort(-sims, kind="stable")[:k]
        per_doc[doc_id] = [[rows[i]["passage_id"], round(float(sims[i]), 4)] for i in order]
        hits += [(rows[i], float(sims[i])) for i in order]
    return hits, {"k": k, "terms_weight": w, "top": per_doc, "no_vectors": missing, "unit_sim": unit_sim,
                  "ms": _ms(t0)}


def unit_vector(v):
    a = np.asarray(v, dtype=np.float32).reshape(-1)
    n = float(np.linalg.norm(a))
    return a / (n if n > 0 else 1.0)


# ---------------------------------------------------------------------------------------------------------------
# 2d. verified facts (warm mode only)
# ---------------------------------------------------------------------------------------------------------------

def guard_ok_values():
    """The guard results that count as passed, taken from the ONE place that defines them:
    `approaches/agentic3/writeback.py::GUARD_OK`, which is also what the write path enforces.

    Imported inside the function on purpose: `ragbench.approaches.agentic3.__init__` imports `approach.py`, which
    imports this module, so a top-level import here would be a cycle. The tuple is tiny and the call is once per
    warm lookup.
    """
    from ..approaches.agentic3.writeback import GUARD_OK
    return tuple(GUARD_OK)


def verified_lookup(con, doc_ids, terms, *, period=None, enabled=False, max_rows=12):
    """Rows the verifier accepted earlier, for the SAME document content. -> (rows, info).

    OFF unless `enabled=True` (spec section 8: COLD mode is the headline and every question stays independent).
    Only guard-passed rows whose `doc_content_sha1` still matches the document are returned, so a rebuilt document
    invalidates its warm rows instead of answering from stale text.

    "Guard-passed" means `v.guard IN writeback.GUARD_OK` - the SAME tuple the write path checks, so a guard result
    the writer accepts is a guard result the reader can find. Until 2026-09-20 this read `coalesce(v.guard, 'ok')
    = 'ok'`, which had two consequences: `ok_label_completed` rows (12% of the live table) were written and then
    unreadable, and a NULL guard was silently treated as a pass. A NULL guard is now a miss, which is the safe
    direction: a row whose guard was never recorded is not a verified fact.
    """
    info = {"enabled": bool(enabled), "docs": list(doc_ids)}
    guard_ok = guard_ok_values()
    info["guard_ok"] = list(guard_ok)
    if not enabled or not doc_ids or not terms:
        info["rows"] = 0
        return [], info
    sql = ("SELECT v.* FROM verified_facts v JOIN documents d ON d.doc_id = v.doc_id "
           "AND d.content_sha1 = v.doc_content_sha1 "
           "WHERE v.doc_id IN (%s) AND v.guard IN (%s) "
           "AND (v.metric REGEXP ? OR v.terms REGEXP ?)" % (_in(len(doc_ids)), _in(len(guard_ok))))
    args = list(doc_ids) + list(guard_ok) + [terms.any_pattern, terms.any_pattern]
    if period and any(p is not None for p in period):
        sql += " AND (v.year IS NULL OR (v.year >= ? AND v.year <= ?))"
        args += [period[0] if period[0] is not None else -10 ** 9,
                 period[1] if period[1] is not None else 10 ** 9]
    rows = [dict(r) for r in con.execute(sql + " ORDER BY v.verified_at DESC LIMIT ?", args + [int(max_rows)])]
    info["rows"] = len(rows)
    return rows, info


# ---------------------------------------------------------------------------------------------------------------
# assembly: candidates in the AGENTIC2 shape
# ---------------------------------------------------------------------------------------------------------------

def rank_candidates(cands):
    """Dedupe and order as `agentic2/approach.py::rank` does: a candidate whose units lie inside a kept one of the
    same section adds nothing, and on a tie the wider window wins."""
    def same_text(a, b):
        if a.doc_id != b.doc_id or a.section_id != b.section_id or not set(b.unit_ids) <= set(a.unit_ids):
            return False
        if b.kind == "clip" or a.kind == "clip":
            if b.text in a.text:
                return True
            if a.span and b.span:
                ov = min(a.span[1], b.span[1]) - max(a.span[0], b.span[0])
                return ov >= 0.5 * max(1, min(b.span[1] - b.span[0], a.span[1] - a.span[0]))
            return False
        return True

    kept = []
    for c in sorted(cands, key=lambda c: (-round(c.score, 3), -len(c.unit_ids), c.doc_id, c.unit_ids)):
        dup = next((k for k in kept if same_text(k, c)), None)
        if dup is not None:
            dup.features["src"] = sorted(set(dup.features.get("src") or []) | set(c.features.get("src") or []))
            continue
        for k in [k for k in kept if same_text(c, k) and k.score <= c.score + 0.02]:
            c.features["src"] = sorted(set(c.features.get("src") or []) | set(k.features.get("src") or []))
            kept.remove(k)
        kept.append(c)
    kept.sort(key=lambda c: (-c.score, c.doc_id, c.unit_ids))
    return kept


def select_candidates(ranked, doc_order, cap):
    """Several documents: round robin in the given document order, so every period is verified before any period
    gets its fifth candidate (`agentic2/approach.py::select`). Assigns `cid` and clears `verdict`."""
    if len(doc_order) <= 1:
        out = list(ranked[:cap])
    else:
        by = {}
        for c in ranked:
            by.setdefault(c.doc_id, []).append(c)
        order = [d for d in doc_order if d in by]
        out, taken = [], {d: 0 for d in order}
        while len(out) < cap and any(taken[d] < len(by[d]) for d in order):
            for d in order:
                if len(out) >= cap:
                    break
                if taken[d] < len(by[d]):
                    out.append(by[d][taken[d]])
                    taken[d] += 1
    for i, c in enumerate(out, 1):
        c.cid = "c%d" % i
        c.verdict = None
    return out


def retrieve(con, plan, doc_ids, *, d_lang="fa", qvec=None, terms_vec=None, opts=None, trigram=True, warm=False,
             tree=True, units=None, cap=None, reader=None):
    """The whole database retrieval for one question (stages 2 and 4). -> (candidates, info).

    Order of work: sections (the tree prior) -> facts -> units (a text answer, a thin fact pool, or `units=True`)
    -> vectors (only when a query vector is handed in) -> tree expansion -> dedupe, rank, cap. `info` carries the
    per-stage timings the paper reports and the row counts of every path.
    """
    o = dict(DEFAULTS, **(opts or {}))
    t0 = time.perf_counter()
    reader = reader or DocReader(con)
    unit_kind = plan.get("unit") or "count"
    period = plan.get("period")
    terms = Terms(plan, d_lang=d_lang)
    ctx = T.search_context(plan, d_lang)
    info = {"version": QUERY_VERSION, "docs": list(doc_ids), "trigram": trigram, "unit_kind": unit_kind,
            "terms": terms.trace()}

    secs, sinfo = sections_search(con, doc_ids, terms, opts=o)
    priors = section_priors(con, secs, opts=o)
    info["sections"] = dict(sinfo, priors=len(priors))

    facts, finfo = facts_search(con, doc_ids, terms, plan_unit=unit_kind, period=period, priors=priors, opts=o,
                                trigram=trigram)
    info["facts"] = finfo

    # the unit path answers text questions, and catches a number whose fact label cannot be named at all
    # (`C.snapp_annual-report_1398.fa::p1_b37` labels 430 نفر with "پاسخ این سوال")
    thin = len(facts) < int(o["units_fallback_min"])
    want_units = (unit_kind == "text" or thin) if units is None else bool(units)
    urows, uinfo = ([], {"skipped": "numeric_question"})
    if want_units:
        urows, uinfo = units_search(con, doc_ids, terms, opts=o, trigram=trigram, priors=priors)
    info["units"] = dict(uinfo, fallback=bool(thin and unit_kind != "text"))

    vhits, vinfo = ([], {"skipped": "no_query_vector"})
    if qvec is not None:
        vhits, vinfo = vector_search(con, doc_ids, qvec, terms_vec, opts=o)
    info["vectors"] = {k: v for k, v in vinfo.items() if k != "unit_sim"}
    unit_sim = vinfo.get("unit_sim") or {}

    vf, vfinfo = verified_lookup(con, doc_ids, terms, period=period, enabled=warm)
    info["verified"] = dict(vfinfo, matches=vf)

    t_exp = time.perf_counter()
    cands = []
    floor = min(unit_sim.values()) if unit_sim else 0.0
    for r in facts:
        c = expand_fact(reader, r, ctx, o, unit_kind, tree=tree)
        if c is None:
            continue
        c.score = r["score"]
        c.features = {"src": ["facts"], "db": r["features"],
                      "vec": round(max([unit_sim.get(u, floor) for u in c.unit_ids] or [floor]), 4),
                      "fact": {"fact_id": r["fact_id"], "label": r["label"], "value": r["value_text"],
                               "unit": r["unit_text"], "unit_kind": r["unit_kind"], "year": r["year"],
                               "def_class": r["def_class"], "source_kind": r["source_kind"]}}
        cands.append(c)
    for r in urows:
        c = expand_unit(reader, r["doc_id"], r["unit_id"], None, ctx, o, unit_kind, tree=tree)
        if c is None:
            continue
        c.score = r["score"]
        c.features = {"src": ["units"], "db": r["features"]}
        cands.append(c)
    for row, sim in vhits:
        for c in expand_passage(reader, row, ctx, o, unit_kind, tree=tree):
            c.score = sim
            c.features = {"src": ["vector"], "vec": round(sim, 4), "pid": row["passage_id"]}
            cands.append(c)
    info["expand_ms"] = _ms(t_exp)

    t_rank = time.perf_counter()
    ranked = rank_candidates(cands)
    limit = int(cap if cap is not None else (o["max_candidates"] if len(doc_ids) <= 1 else o["max_candidates_multi"]))
    sel = select_candidates(ranked, list(doc_ids), limit)
    info.update(rank_ms=_ms(t_rank), pre_cap=len(ranked), selected=len(sel), cap=limit,
                sections_rows=len(secs), facts_rows=len(facts), units_rows=len(urows), vector_hits=len(vhits),
                verified_rows=len(vf), ms=_ms(t0))
    return sel, info
