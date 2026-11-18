"""AGENTIC3: a persistent typed knowledge base as the FIRST retrieval path, with a budgeted agent on top.

Eight stages, at most `caps.agentic3_max_calls` (4) model calls and `caps.agentic3_max_seconds` (60) of query
clock per question:

    1 plan        (1 call)  the v1 planner, its document listing now read from `documents` instead of from
                            Document objects, plus the one prompt change the spec allows: up to 3 FACETS, each a
                            group of terms naming one of the things a multi-section question has to find.
    2 retrieve    (no LLM)  `kb/query.py`: sections (the tree prior), typed facts (regex on `label_fold`, scored on
                            unit kind, year, label specificity, source kind), units, and one pass per facet.
    3 route       (no LLM)  `route.py` decides FAST | STANDARD | ESCALATED | TIMELINE and records why.
    4 expand      (no LLM)  tree expansion in SQL (`kb/query.py`), capped at 12 candidates (24 over several
                            documents), every facet keeping at least 2 slots.
    5 verify      (0-2)     the v1 batched verifier and the verbatim guard, with v2's two repairs and one new
                            escape hatch (below). FAST skips it; TIMELINE always runs it and keeps its gate.
    6 reconcile   (no LLM)  `reconcile_sql.py` (flag `no_sql_reconcile` falls back to v1's Python reconciler), then
                            program-only gap queries for the missing years, and one more verify call if the clock
                            allows.
    7 answer      (1 call)  the SHARED answer prompt, the shared timeline prompt, the shared 6000-char budget,
                            through the per-question call cache so a resumed question makes the same decisions.
    8 write back  (no LLM)  `writeback.py`: guard-passed rows into `verified_facts`. Read back only with `warm`.

THE VERIFIER RANKS RATHER THAN GATES - as the evidence supports it, not as the spec states it. Spec section 6
stage 5 asks for a plain-candidates pack when the verifier accepts nothing, "with no 'nothing was verified' note".
`work/analysis/V2_FAILURES.md` section 2.1 shows that this is already v2's behaviour (`evidence_items` returns
`unverified_candidates`, `pack_evidence` never renders `meta`, and 12 of the 18 misses - 9 of the 11 NOT_FOUNDs -
already went down it), so implementing the spec's letter recovers nothing. What the forensics do support is two
changes, both here:

  * THE GUARD ESCAPE HATCH. `quote_without_term` fired 149 times over 79 lookups and destroyed the correct value
    in 7 of the 18 misses, every time as a synonym gap between the plan's terms and the report's wording
    (`employees` for `staff`, `تعطیل‌شده` for `متوقف`, `rounds` for `capital raises`). In those seven the
    verifier's own `definition` is a correct label for a correct value, so the model's output contradicts the
    guard. A rejection for that one reason is re-checked with the term test removed, and accepted when a plan term
    is in the verifier's definition or in the fact row's own label. Every other guard reason, and the whole
    verbatim check, is untouched: the quote must still be in the snippet and the value in the quote.
  * ORDER BY THE VERIFIER'S JUDGEMENT. When nothing is accepted, the pack is ordered with the candidates the model
    called relevant first and the ones it rejected last, instead of by retrieval score alone. Miss #13
    (customs-02 en-fa) is the case: the verifier read the right value out of the right KPI line, the guard vetoed
    it, and the candidate then sat at rank 9 of 7 packed items.

`verify_gate` restores v2 exactly: the hard veto and the retrieval-score order.

Flags (ablations): no_typed (the typed filters and weights off, i.e. v2's text fact table), no_trigram,
no_sql_reconcile, no_fastpath, verify_gate, vectors_always, no_vectors, no_facets, no_tree_prior, warm,
candidates=<n>. `AGENTIC3_OPTIONS` (a JSON object of DEFAULTS overrides) is DEV tuning only and is recorded in
trace.params; `core_plan.sh v3test` refuses to run while it is set.

Resumable like v1 and v2: every model call is cached in work/runs/<out>/agentic_calls_<variant>.sqlite, the
knowledge base commits per document and per stage, vectors go to the shared embedding cache, and write-back
commits per row.
"""
import json
import math
import os
import time

from ...contracts import AnswerResult, Approach, RetrievedItem
from ...kb import ingest as I
from ...kb import query as Q
from ...kb import schema as KB
from ...llm import BudgetExceeded
from ...normalize import normalize
from ...util import atomic_write_json, read_json, safe_name
from ..agentic import planner as P
from ..agentic import reconcile as R
from ..agentic import verify as V
from ..agentic.approach import CallCache, _series_row
from ..agentic.defaults import VERIFY_PROMPT_VERSION
from ..agentic2.approach import _Session2
from ..agentic2.lazyvec import LazyVectors
from . import reconcile_sql as RS
from . import retrieve as RT
from . import route as RO
from . import writeback as WB
from .defaults import APPROACH_VERSION, DEFAULTS, FLAGS, PLAN_PROMPT_VERSION, ROUTE_VERSION, VALUE_FLAGS

OPTIONS_ENV = "AGENTIC3_OPTIONS"     # DEV tuning only: a JSON object of DEFAULTS overrides (recorded in trace.params)

FACETS_RULE = ('- "facets": for a question that needs several different parts of the report, 2 or 3 groups of '
               'search terms, each group naming ONE of the different things to find (2 to 4 short terms per group, '
               'Persian or English). [] when one topic answers the question. Facets are only for searching: they '
               'never change "kind", and a question about one value stays "single".')

# how good an unverified candidate is as evidence, by what the verifier said about it. A value the model read out
# and only the guard refused is the best unverified evidence there is; a snippet the model called irrelevant is the
# worst (V2_FAILURES 2.2 and miss #13).
VERDICT_ORDER = {"accepted": 0, "guard": 1, "duplicate": 2, "model_omitted": 3, "unparsed": 3,
                 "not_verified_budget": 4, "not_verified_time": 4, None: 4, "model_rejected": 5}


def parse_flags(flags):
    if flags is None:
        return {}
    if isinstance(flags, str):
        d = {}
        for p in flags.replace("+", ",").split(","):
            p = p.strip()
            if not p:
                continue
            k, _, v = p.partition("=")
            d[k.strip()] = v.strip() if v else True
        flags = d
    out = {}
    for k, v in dict(flags).items():
        if k not in FLAGS:
            raise ValueError("unknown AGENTIC3 flag %r; known: %s" % (k, ", ".join(FLAGS)))
        if k in VALUE_FLAGS:
            try:
                n = int(v)
            except (TypeError, ValueError):
                raise ValueError("AGENTIC3 flag %s takes an integer (got %r)" % (k, v))
            if n < 1:
                raise ValueError("AGENTIC3 flag %s must be >= 1" % k)
            out[k] = n
            continue
        v = v if isinstance(v, bool) else str(v).lower() in ("1", "true", "yes", "on")
        if v:
            out[k] = True
    return out


def _ms(t0):
    return round((time.perf_counter() - t0) * 1000.0, 1)


# ---------------------------------------------------------------------------------------------------------------
# stage 1: the plan prompt, v1's with the facets field appended
# ---------------------------------------------------------------------------------------------------------------

def plan_messages(question, metas, choose_companies, opts):
    """v1's messages, the facets rule appended to the system message."""
    msgs = P.plan_messages(question, metas, choose_companies, opts)
    if opts["facets"]:
        msgs[0]["content"] = msgs[0]["content"] + "\n" + FACETS_RULE
    return msgs


def plan_schema(company_keys, opts):
    """v1's schema with `facets` LAST, so every field the planner already produced is decoded from the same
    prefix and the added field cannot reorder the others."""
    schema = P.plan_schema(company_keys)
    if opts["facets"]:
        schema["properties"]["facets"] = {
            "type": "array", "maxItems": int(opts["max_facets"]),
            "items": {"type": "array", "items": {"type": "string"}, "maxItems": int(opts["max_facet_terms"])}}
        schema["required"].append("facets")
    return schema


def validate_facets(obj, plan, opts):
    """The model's facet groups -> [{name, terms {fa, en}, raw, dropped}], cleaned like the plan's own terms.

    DB_CEILING instruction 8 asks for facets that cover DISJOINT vocabulary, so a term already claimed by an
    earlier facet is removed from a later one; a facet left with nothing is dropped.
    """
    from ...normalize import fold
    groups = obj.get("facets") if isinstance(obj, dict) else None
    out, claimed = [], set()
    for group in (groups or [])[: int(opts["max_facets"])]:
        values = [group] if isinstance(group, str) else [x for x in (group or []) if isinstance(x, str)]
        terms, dropped = P.clean_terms(values, int(opts["max_facet_terms"]), int(opts["max_term_chars"]),
                                       stop_extra=plan["entity_terms"])
        terms, _report = P.refine_terms(terms, plan["entity_terms"], int(opts["max_facet_terms"]))
        overlap = []
        for lang in ("fa", "en"):
            keep = []
            for t in terms[lang]:
                if fold(t) in claimed:
                    overlap.append(t)
                    continue
                keep.append(t)
            terms[lang] = keep
        if not (terms["fa"] or terms["en"]):
            continue
        for lang in ("fa", "en"):
            claimed.update(fold(t) for t in terms[lang])
        out.append({"name": "f%d" % (len(out) + 1), "terms": terms, "raw": values,
                    "dropped": dropped, "overlap_dropped": overlap})
    return out


# ---------------------------------------------------------------------------------------------------------------
# the approach
# ---------------------------------------------------------------------------------------------------------------

class TypedKBAgent(Approach):
    name = "AGENTIC3"
    needs_gold = False

    def __init__(self, cfg=None, client=None, flags=None, options=None):
        super().__init__(cfg=cfg, client=client, flags=parse_flags(flags))
        env = os.environ.get(OPTIONS_ENV)
        self.env_options = json.loads(env) if env else None
        self.opts = dict(DEFAULTS, **(self.env_options or {}), **(options or {}))
        f = self.flags
        o = self.opts
        if f.get("candidates"):
            o["max_candidates"] = o["max_candidates_multi"] = int(f["candidates"])
        if f.get("no_typed"):            # v2's text fact table: no typed filter and no typed weight
            o.update(kind_filter="off", year_filter="off", w_kind=0.0, w_kind_conflict=0.0, w_year_own=0.0,
                     w_year_doc=0.0, w_year_miss=0.0, w_source_table=0.0, w_source_kpi=0.0)
        if f.get("no_tree_prior"):
            o["w_section"] = 0.0
        if f.get("no_facets"):
            o["facets"] = False
        if f.get("verify_gate"):         # v2 exactly: the hard term veto, and no verdict-aware ordering
            o.update(guard_escape=False, rank_by_verdict=False)
        if f.get("no_sql_reconcile"):
            o["sql_reconcile"] = False
        self.warm = bool(f.get("warm"))
        self.use_trigram = not f.get("no_trigram")
        self.use_vectors = not f.get("no_vectors")
        self.use_verify = True
        self.use_tree = True
        self.use_factdb = True
        self.line_windows = False
        self.doc_selection = True
        caps = self.cfg.caps or {}
        self.max_calls = int(caps.get("agentic3_max_calls") or o["max_calls_default"])
        self.max_seconds = float(caps.get("agentic3_max_seconds") or o["max_seconds_default"])
        self._cache = None
        self._cache_for = None
        self._con = None
        self.lazy = LazyVectors(self)

    # -- the knowledge base ------------------------------------------------------------------------------------
    def kb(self):
        if self._con is None:
            self._con = RT.open_kb(self.cfg)
        return self._con

    def config_signature(self):
        """What makes an index directory stale: the knowledge base's component versions (schema, typed normaliser,
        fact extractor, tree, embed model), its path, and the retrieval contract."""
        return {"kb": KB.component_versions(self.cfg), "kb_path": str(I.kb_path(self.cfg)),
                "query": Q.QUERY_VERSION, "approach": APPROACH_VERSION}

    def _build_index(self, corpus_set, index_dir):
        """Ingestion: every document of the corpus set into the shared knowledge base, no model call.

        This is the only place the approach sees a corpus without a question (fairness rule 2). `index_dir` holds
        the pointer, the ingestion statistics and the row counts; the data itself lives at one shared path, so
        several corpus sets share one database and a document ingested for corpus A_fa is not ingested again for
        A_en. The wall time of this method is the ingestion cost the paper reports (`build_stats.json:attempts[].
        build_ms`); `kb_index.json` next to it breaks the same time down by stage and document.
        """
        t0 = time.perf_counter()
        con = self.kb()
        stats = RT.ingest_corpus(con, corpus_set, self.cfg)
        payload = {"store_root": None, "lazy": True, "kb_path": str(I.kb_path(self.cfg)),
                   "n_docs": len(corpus_set.docs), "doc_ids": corpus_set.doc_ids(),
                   "versions": KB.component_versions(self.cfg), "ingest": stats,
                   "kb_status": {k: v for k, v in I.status(con, self.cfg).items() if k != "stages"},
                   "build_ms": _ms(t0)}
        atomic_write_json(index_dir / "kb_index.json", payload, indent=1)
        return payload

    def _load_index(self, corpus_set, index_dir):
        """A finished index is a pointer: check that the knowledge base still holds every document of the set and
        re-ingest only what is missing (the file can be rebuilt or removed between runs)."""
        payload = read_json(index_dir / "kb_index.json") or {}
        missing = RT.missing_documents(self.kb(), corpus_set.doc_ids())
        if missing or not payload:
            return self._build_index(corpus_set, index_dir)
        return payload

    def call_cache(self):
        path = self.opts.get("call_cache_path")
        if not path and self.opts.get("call_cache", "run") == "run" and getattr(self.client.ledger, "path", None):
            from pathlib import Path
            path = Path(self.client.ledger.path).parent / ("agentic_calls_%s.sqlite" % safe_name(self.variant))
        if self.opts.get("call_cache") == "off":
            path = None
        if self._cache is None or str(path) != str(self._cache_for):
            self._cache = CallCache(path)
            self._cache_for = path
        return self._cache

    def _answer(self, query, index):
        return _Session3(self, query, index).run()


# ---------------------------------------------------------------------------------------------------------------
# one question
# ---------------------------------------------------------------------------------------------------------------

class _Session3(_Session2):
    """One question. Inherits from v2: the query clock (lazy embedding excluded, replayed calls counted with their
    recorded wall time), the two verify repairs, and from v1 the stage timing, the call cache, the batched
    verifier, the guard, the verified table and the shared answer call."""

    def __init__(self, ap, query, index):
        super().__init__(ap, query, index)
        self.con = ap.kb()
        self.reader = Q.DocReader(self.con)
        self.stats.update({"route_fast": 0, "route_standard": 0, "route_escalated": 0, "route_timeline": 0,
                           "facets": 0, "facet_candidates": 0, "db_facts_rows": 0, "db_units_rows": 0,
                           "db_sections_rows": 0, "db_ms": 0.0, "guard_escaped": 0, "vector_hits": 0,
                           "docs_embedded": 0, "passages_embedded": 0, "gap_queries": 0, "gap_rows": 0,
                           "gap_candidates": 0, "warm_rows": 0, "written_back": 0, "cand_units": 0, "cand_gap": 0,
                           "narrow_suppressed": 0})

    # -- stage 1 ---------------------------------------------------------------------------------------------
    def plan(self, metas):
        """v1's plan call and repair ladder, with the facets field (PLAN_PROMPT_VERSION agentic3-plan-v1)."""
        o = self.o
        choose = len(metas) > 1 and self.a.doc_selection
        company_keys = sorted({m["company"] for m in metas}) if choose else None
        msgs = plan_messages(self.q.text, metas, choose, o)
        schema = plan_schema(company_keys, o)
        r = self.chat("plan", msgs, fmt=schema, num_predict=int(o["plan_num_predict"]),
                      version=PLAN_PROMPT_VERSION)
        obj, how = P.parse_plan_text(r.text)
        problem = P.plan_problem(obj)
        rep = {"parse": how, "problem": problem, "model_output": (r.text or "")[:1500], "done_reason": r.done_reason}
        if how in ("extract", "suffix_repair"):
            self.stats["plan_parse_repaired"] = 1
        if problem and o["plan_repair_call"] and self.budget.can(1, reserve=2):
            msgs2 = msgs + [{"role": "assistant", "content": (r.text or "")[:1500]},
                            {"role": "user", "content": P.REPAIR_USER.format(problem=problem)}]
            r2 = self.chat("plan_repair", msgs2, fmt=schema, num_predict=int(o["plan_num_predict"]),
                           version=PLAN_PROMPT_VERSION)
            obj2, how2 = P.parse_plan_text(r2.text)
            problem2 = P.plan_problem(obj2)
            rep["repair"] = {"parse": how2, "problem": problem2, "model_output": (r2.text or "")[:1500]}
            self.stats["plan_repair"] = 1
            if not problem2:
                obj, problem = obj2, None
        if problem:
            plan = P.program_plan(self.q.text, self.q.q_lang, metas, choose, o)
            self.stats["plan_program"] = 1
        else:
            plan = P.validate_plan(obj, question=self.q.text, q_lang=self.q.q_lang, metas=metas,
                                   choose_companies=choose, opts=o)
        plan["facets"] = validate_facets(obj, plan, o) if (o["facets"] and not problem) else []
        plan["model"] = rep
        self.stats["facets"] = len(plan["facets"])
        return plan

    # -- stage 5: the guard escape hatch -----------------------------------------------------------------------
    def verify_batch(self, batch, plan, purpose, rows, meta_by, batch_no):
        out = super().verify_batch(batch, plan, purpose, rows, meta_by, batch_no)
        if self.o["guard_escape"]:
            self.accept_by_definition(out, batch, plan, rows, meta_by)
        self.annotate_rows(out, rows, batch)
        return out

    def term_source(self, rec, cand, term_rx):
        """Where a plan term stands when the quote does not carry one: in the verifier's own definition, or in the
        fact row's label. -> "definition" | "label" | None."""
        definition = normalize(str(rec.get("definition") or ""))
        if definition and any(rx.search(definition) for rx in term_rx):
            return "definition"
        if self.o["guard_escape_label"]:
            label = normalize(str(((cand.features or {}).get("fact") or {}).get("label") or ""))
            if label and any(rx.search(label) for rx in term_rx):
                return "label"
        return None

    def accept_by_definition(self, out, batch, plan, rows, meta_by):
        """Re-check a `quote_without_term` rejection with the term test removed (V2_FAILURES 2.2).

        Accepted only when the verbatim guard passes on its own - the quote is in the snippet and the value is in
        the quote - AND a plan term stands in the verifier's definition or in the fact row's label. The guard's
        purpose, a number labelled with a DIFFERENT metric, is what is left: such a value has the term nowhere.
        """
        unit_kind = plan.get("unit") or "count"
        if unit_kind == "text" or not self.o["quote_term_guard"]:
            return
        term_rx = self.metric_term_rx(plan)
        by_cid = {c.cid: c for c in batch}
        done = 0
        for rec in out["items"]:
            if rec.get("verdict") != "guard:quote_without_term":
                continue
            c = by_cid.get(rec["cid"])
            if c is None:
                continue
            where = self.term_source(rec, c, term_rx)
            if where is None:
                continue
            it = {k: rec.get(k) for k in ("value", "unit", "period", "definition", "quote")}
            ok, _reason, ginfo = V.guard(it, c.text, unit_kind=unit_kind,
                                         min_quote_letters=int(self.o["min_quote_letters"]), term_rx=None)
            if not ok:
                continue
            if any(r["cid"] == c.cid and r.get("value_num") is not None and ginfo.get("value_num") is not None
                   and abs(r["value_num"] - ginfo["value_num"]) < 1e-9 for r in rows):
                continue
            uid = V.value_unit_id(c, self.corpus, ginfo.get("value_token"), str(it.get("quote") or ""))
            row = R.make_row(it, c, ginfo, meta_by.get(c.doc_id, {}), uid, verified=True)
            row["guard_escape"] = where
            rows.append(row)
            rec.update(verdict="accepted", guard="ok", term_source=where, guard_escape=where,
                       value_num=ginfo.get("value_num"), row_period=row["period"], def_class=row["def_class"],
                       role=row["role"])
            c.verdict = "accepted"
            out["accepted"] += 1
            out["guard_rejected"] -= 1
            self.stats["accepted"] += 1
            self.stats["guard_rejected"] -= 1
            out["guard_reasons"]["quote_without_term"] = out["guard_reasons"].get("quote_without_term", 1) - 1
            self.guard_reasons["quote_without_term"] = self.guard_reasons.get("quote_without_term", 1) - 1
            done += 1
        out["guard_escaped"] = done
        self.stats["guard_escaped"] += done

    def annotate_rows(self, out, rows, batch):
        """Write-back's caller contract: the guard result and the fact id belong on the row, not only on the
        verify record (`writeback.write_rows` skips a row without a guard as `no_guard`)."""
        fact_by_cid = {c.cid: ((c.features or {}).get("fact") or {}) for c in batch}
        for rec in out["items"]:
            if rec.get("verdict") != "accepted":
                continue
            row = next((r for r in rows if r.get("guard") is None and r.get("cid") == rec.get("cid")
                        and r.get("value_num") == rec.get("value_num")), None)
            if row is None:
                continue
            row["guard"] = rec.get("guard") or "ok"
            row["fact_id"] = fact_by_cid.get(rec["cid"], {}).get("fact_id")
            row["label"] = fact_by_cid.get(rec["cid"], {}).get("label")

    # -- stage 7: the evidence pack -----------------------------------------------------------------------------
    def evidence_items(self, rec, cands, verified, *, verified_ran=True, warm_rows=()):
        """[table] + its source windows when rows were verified (v2's pack, 88% accurate against 57% for the plain
        one), else the candidates in the ordinary format - ordered by what the verifier said about them.

        THE SERIES TABLE IS LEFT AS THE RECONCILER BUILT IT, and round 2 measured why. v3 packs 1.68 verified rows
        per period against v2's 1.12 (42 rows over 25 dev periods against 27 over 24), which reads badly, so both
        obvious cures were tried on the recorded dev timelines. The text scorer reads the FIRST value the answer
        gives for a period, so what matters is which row of a period comes first. Re-ordering each period by the
        main definition class, then authoritative, then the plainest definition - v1's own single-fact pick,
        applied per period - gets the first value right for 10 of the 25 dev periods against 11 today. Keeping only
        each period's first row and dropping the others gains 1 period and loses 2, because on 4 of the 9 dev
        periods that offer several rows the 4B answerer does NOT copy the first one, and twice it chose better than
        the reconciler's score order (27.1 million over 97, 2,788,550 over 12,993,237). The extra rows are ugly;
        removing them costs accuracy. The gap to v2 on timelines is which rows the verifier accepts, not how they
        are shown.
        """
        o = self.o
        items = [RetrievedItem(text=r["quote"] or r["value_text"] or "", doc_id=r["doc_id"], unit_ids=[r["unit_id"]]
                               if r.get("unit_id") else [], kind="fact", meta={"verified": True, "warm": True})
                 for r in warm_rows]
        if not verified_ran:
            items += [c.to_item(meta={"verified": None}) for c in cands]
            ev_from = "warm_verified" if warm_rows else "database_candidates"
        elif not rec["series"]:
            pool = self.by_verdict(cands) if o["rank_by_verdict"] else list(cands)
            if not o["fill_unverified"]:
                pool = pool[: int(o["unverified_fallback_items"])]
            items += [c.to_item(meta={"verified": False}) for c in pool]
            ev_from = "unverified_candidates"
            self.stats["unverified_fallback"] = 1
        else:
            table, ev_from = self.evidence(rec, verified)
            items += table
            if o["fill_unverified"]:
                have = {it.meta.get("cid") for it in items}
                extra = [c for c in cands if c.cid not in have]
                items += [c.to_item(meta={"verified": getattr(c, "verdict", None) == "accepted"}) for c in extra]
                self.stats["fill_items"] = len(extra)
        for i, it in enumerate(items):
            it.rank = i + 1
        return items, ev_from

    def count_sources(self, cands):
        """The per-path candidate counters, under v2's names so the two approaches' `ts_cand_*` columns compare."""
        src = [set((c.features or {}).get("src") or []) for c in cands]
        self.stats.update(candidates=len(cands),
                          cand_factdb=sum(1 for s in src if "facts" in s and "vector" not in s),
                          cand_vector=sum(1 for s in src if "vector" in s and "facts" not in s),
                          cand_both=sum(1 for s in src if "vector" in s and "facts" in s),
                          cand_units=sum(1 for s in src if s == {"units"}),
                          cand_gap=sum(1 for s in src if "gap" in s))

    @staticmethod
    def by_verdict(cands):
        order = {c.cid: i for i, c in enumerate(cands)}
        def rank(c):
            v = getattr(c, "verdict", None)
            return VERDICT_ORDER.get(v.split(":")[0] if isinstance(v, str) else v, 4)
        return sorted(cands, key=lambda c: (rank(c), order[c.cid]))

    # -- retrieval ---------------------------------------------------------------------------------------------
    def merge(self, pool, doc_ids, facets, cap):
        """The candidate merge (retrieve.py::merge). A method only so that a subclass can reserve slots for a source
        of its own - AGENTIC4 reserves them for the graph. For AGENTIC3 it is exactly RT.merge (2026-09-21: the two
        call sites below used to call RT.merge directly; replaying v3 records from their call cache is byte-identical
        before and after)."""
        return RT.merge(pool, doc_ids, facets, self.o, cap)

    def retrieve(self, plan, doc_ids, facets, cap):
        o = self.o
        trigram = self.a.use_trigram and (len(doc_ids) > 1 or o["trigram_single_doc"])
        cands, info = RT.retrieve_all(self.con, plan, doc_ids, d_lang=self.q.d_lang, opts=o, trigram=trigram,
                                      warm=self.a.warm, tree=self.a.use_tree, reader=self.reader, facets=facets,
                                      cap=cap)
        info["trigram_used"] = trigram
        return cands, info

    def vectors(self, plan, doc_ids, cap, pool, facets):
        """The escalated path: embed what has never been embedded (off the clock), then merge the passage hits."""
        o = self.o
        with self.stage("lazy-index") as st:
            emb = RT.embed_pending(self.con, doc_ids, self.cfg, self.a.client, self.a.lazy)
            self.stats["docs_embedded"] = emb["docs"]
            self.stats["passages_embedded"] = emb["embedded"]
            st.update(docs=emb["docs"], passages=emb["passages"], embedded=emb["embedded"],
                      from_cache=emb["from_cache"], embed_calls=emb["calls"], embed_ms=emb["embed_ms"])
        qvec, tvec, qinfo = RT.query_vectors(self.a.lazy, self.a.client, self.q.text, plan, self.q.d_lang, o)
        self.stats["query_embed_calls"] += qinfo["embed_calls"]
        vcands, vinfo = RT.vector_candidates(self.con, doc_ids, qvec, tvec, plan=plan, d_lang=self.q.d_lang,
                                             opts=o, reader=self.reader, tree=self.a.use_tree)
        self.stats["vec_hits"] = vinfo["hits"]
        self.stats["vector_hits"] = vinfo["hits"]
        sel, minfo = self.merge(pool + vcands, doc_ids, facets, cap)
        return sel, {"embed": emb, "query": qinfo, "search": vinfo, "merge": minfo}

    # -- stage 6 -----------------------------------------------------------------------------------------------
    def narrowing_applies(self, rows, plan):
        """Should the reconciler's NARROW rule run on these rows?

        The rule drops a row whose definition carries two or more words the question does not explain and more of
        them than the plainest row of its period: a programme's users against all users, one service against the
        company. It assumes the rows of a period are competing readings of ONE metric. A plan that returned FACETS
        asserts the opposite - its rows are deliberately about different things - and when such a plan's rows also
        carry no period at all, every one of them sits in the same empty period and the rule deletes the question's
        SECOND TOPIC rather than a sub-definition.

        Measured on the 30 dev records: the rule removes a row in 7 of them. In the three that meet both conditions
        - A-rpt-f-03 en-fa, fa-en and fa-fa, the multi-section question facets exist for - the verifier had
        accepted both halves of the answer and the rule deleted one of them (113.4, then 113.4 and 72.71, then
        78.7), leaving a one-row table that the answerer refused to answer half a question from. The other four
        carry dated rows and are left exactly as they were, which is why both conditions are required.

        A question where the verifier accepted nothing also takes this branch (no row carries a period because
        there are no rows); with nothing to narrow the reconciler returns the same empty series either way, and
        the dev run confirms it - C-TL-divar-01 records `narrow_suppressed` and its answer is unchanged.
        """
        if self.o["narrow_faceted_topics"] or not plan.get("facets") or plan["kind"] == "single":
            return True
        return any(r.get("period_year") for r in rows)

    def reconcile(self, rows, plan, target_periods):
        o = self.o
        stems = self.known_stems(plan)
        for r in rows:                      # kept here so the trace still shows them when narrowing is suppressed
            r["qualifiers"] = R.qualifiers(r.get("definition"), stems)
        narrow = self.narrowing_applies(rows, plan)
        self.stats["narrow_suppressed"] = int(not narrow)
        stems = stems if narrow else None   # the reconciler's own switch: no stems, no narrowing (v1 and SQL alike)
        if o["sql_reconcile"]:
            return RS.reconcile(rows, kind=plan["kind"], target_periods=target_periods, opts=o,
                                q_lang=self.q.q_lang, known_stems=stems, kb=self.con)
        return R.reconcile(rows, kind=plan["kind"], target_periods=target_periods, opts=o, q_lang=self.q.q_lang,
                           known_stems=stems)

    def gap_fill(self, rec, plan, target_periods, scope_ids, rows, meta_by, cands, fallback=None):
        """Program-only gap queries for the years with no value, then ONE verify call if the clock allows.

        `reconcile_sql.gap_queries` searches every visible document by design; a benchmark question may only see
        its own corpus set, so every gap is restricted to `scope_ids` first and what that drops is recorded.
        `fallback` is a pattern used for a gap the reconciler could not build one for (it draws its wording from
        the rows already verified, and a question where nothing was verified has none).
        """
        o = self.o
        info = {"ran": False, "missing": list(rec["periods_missing"]), "queries": [], "dropped_out_of_scope": 0,
                "pattern_from": "plan" if fallback else "verified_rows"}
        if not rec["periods_missing"] or not o["gap_fill"]:
            return [], info
        gaps = RS.gap_queries(rec, kb=self.con, target_periods=target_periods, lang=self.q.d_lang,
                              limit=int(o["gap_rows_per_year"]))
        seen = {u for c in cands for u in c.unit_ids}
        grows, info["ran"] = [], True
        for gap in gaps[: int(o["gap_max_years"])]:
            g, dropped = RT.scope_gap(gap, scope_ids)
            if not g["pattern"] and fallback:
                g.update(pattern=fallback, runnable=bool(g["doc_ids"]))
                g["params"]["pattern"] = fallback
            info["dropped_out_of_scope"] += dropped
            found = RS.run_gap_query(self.con, g) if g["runnable"] else []
            grows += [r for r in found if r["unit_id"] not in seen]
            info["queries"].append({"year": g["year"], "docs": g["doc_ids"], "pattern": g["pattern"],
                                    "rows": len(found), "new": len([r for r in found if r["unit_id"] not in seen]),
                                    "dropped_out_of_scope": dropped, "runnable": g["runnable"]})
        self.stats["gap_queries"] = len(info["queries"])
        self.stats["gap_rows"] = len(grows)
        gcands = RT.gap_candidates(self.reader, grows, plan, self.q.d_lang, o, tree=self.a.use_tree)
        gcands = Q.select_candidates(Q.rank_candidates(gcands), list({r["doc_id"] for r in grows}),
                                     int(o["gap_max_candidates"]))
        for i, c in enumerate(gcands, 1):
            c.cid = "g%d" % i
        self.stats["gap_candidates"] = len(gcands)
        info["candidates"] = [c.trace(int(o["trace_text_chars"])) for c in gcands]
        return gcands, info

    def gap_batch(self, gcands, skipped, gap):
        """-> (the batch the call held back for the gap round verifies, the NEW candidates it adds). The gap round's
        candidates when it found any, else the candidates the first round had to skip, so the reserve never costs a
        verification. A hook (AGENTIC4 adds the graph's candidates for the missing years here); with gap candidates
        the skipped list is untouched, without them the batch IS the skipped head - v3's behaviour either way."""
        return (list(gcands) or list(skipped[: int(self.o["verify_batch_max"])])), gcands

    # -- the pipeline --------------------------------------------------------------------------------------------
    def run(self):
        t_all = time.perf_counter()
        o = self.o
        trace = {"variant": self.a.variant, "max_calls": self.budget.max, "max_seconds": self.a.max_seconds,
                 "prompts": {"plan": PLAN_PROMPT_VERSION, "verify": VERIFY_PROMPT_VERSION, "answer": "shared"},
                 "versions": {"approach": APPROACH_VERSION, "route": ROUTE_VERSION, "query": Q.QUERY_VERSION,
                              "reconcile_sql": RS.RECONCILE_SQL_VERSION, "writeback": WB.WRITEBACK_VERSION,
                              "kb": KB.component_versions(self.cfg)},
                 "params": {k: o[k] for k in sorted(o) if not k.startswith("call_cache") and not k.endswith("_path")},
                 "options_env": self.a.env_options, "call_cache": self.cache.path, "stages": {},
                 "flags": dict(self.a.flags)}
        scope_ids = [d.id for d in self.a.docs_in_scope(self.q, self.index)]
        self.stats["docs_scope"] = len(scope_ids)
        with self.a.client.ledger.scope() as whole:
            # 1. plan, over the document listing read from SQL
            with self.stage("plan") as st:
                metas = Q.document_metas(self.con, doc_ids=scope_ids)
                plan = self.plan(metas)
                chosen, sel = RT.target_documents(self.con, scope_ids, plan, o, question=self.q.text)
                meta_by = {m["doc_id"]: m for m in metas}
                doc_ids = [m["doc_id"] for m in chosen]
                self.stats["docs_targeted"] = len(doc_ids)
                st.update(docs_scope=len(scope_ids), docs_visible=len(metas), docs_targeted=len(doc_ids),
                          kind=plan["kind"], facets=len(plan["facets"]), repair=self.stats["plan_repair"],
                          program_plan=self.stats["plan_program"])
            trace["stages"]["plan"] = {k: plan[k] for k in ("entity", "metric", "kind", "unit", "terms",
                                                            "entity_terms", "period", "companies", "facets",
                                                            "dropped_terms", "fixes", "model")}
            trace["stages"]["plan"]["doc_selection"] = sel
            trace["stages"]["plan"]["missing_documents"] = RT.missing_documents(self.con, scope_ids)

            # 2 + 4. database retrieval and tree expansion (no model)
            facets = RT.facet_plans(plan, o) if o["facets"] else []
            multi = len(doc_ids) > 1
            cap = int(o["max_candidates_multi"] if multi else o["max_candidates"])
            with self.stage("retrieve") as st:
                pool, rinfo = self.retrieve(plan, doc_ids, facets, cap)
                cands, minfo = self.merge(pool, doc_ids, facets, cap)
                # the route asks the merge which facets it could not fill (`route.starved_facets`), so the merge
                # info has to be on `rinfo` before stage 3 reads it, not only in the trace afterwards
                rinfo["merge"] = minfo
                self.stats.update(db_facts_rows=rinfo.get("facts_rows", 0), db_units_rows=rinfo.get("units_rows", 0),
                                  db_sections_rows=rinfo.get("sections_rows", 0), db_ms=rinfo.get("ms", 0.0),
                                  facet_candidates=rinfo.get("facet_candidates", 0),
                                  candidates_pre_cap=minfo["pre_cap"])
                self.count_sources(cands)
                st.update(candidates=len(cands), pool=len(pool), pre_cap=minfo["pre_cap"], db_ms=rinfo.get("ms"),
                          facts_rows=rinfo.get("facts_rows"), units_rows=rinfo.get("units_rows"),
                          sections_rows=rinfo.get("sections_rows"), facet_slots=minfo["facet_slots"],
                          trigram=rinfo["trigram_used"])
            warm_rows = ((rinfo.get("verified") or {}).get("matches") or []) if self.a.warm else []
            self.stats["warm_rows"] = len(warm_rows)

            # 3. the sufficiency check (no model)
            with self.stage("route") as st:
                decision = RO.decide(plan, cands, rinfo, o, flags=self.a.flags, warm_rows=warm_rows)
                self.stats["route_" + decision["route"].lower()] = 1
                st.update(route=decision["route"], reason=decision["reason"], vectors=int(decision["vectors"]),
                          verify=int(decision["verify"]))
            trace["stages"]["route"] = decision

            # 2e. vectors, only when the route asks for them (first touch is preprocessing, off the query clock)
            vinfo = None
            if decision["vectors"] and self.a.use_vectors and doc_ids:
                with self.stage("vectors") as st:
                    cands, vinfo = self.vectors(plan, doc_ids, cap, pool, facets)
                    self.count_sources(cands)
                    st.update(hits=vinfo["search"]["hits"], candidates=len(cands),
                              no_vectors=vinfo["search"].get("no_vectors"),
                              embed_calls=self.stats["query_embed_calls"])
            rinfo["candidates"] = [c.trace(int(o["trace_text_chars"])) for c in cands]
            rinfo["facet_sections"] = {n: sorted({c.section_id for c in cands if n in RT.facets_of(c)})
                                       for n, _p in facets}
            trace["stages"]["retrieve"] = rinfo
            trace["stages"]["vectors"] = vinfo

            # 5. verification (skipped on FAST; gated for the series on TIMELINE)
            rows, logs, skipped, time_skipped = [], [], [], []
            with self.stage("verify") as st:
                if decision["verify"] and cands:
                    # a multi-period plan holds one call back for the gap round (v1's reserve rule)
                    reserve = 2 if (plan["kind"] != "single" and o["gap_fill"] and self.budget.left() >= 3) else 1
                    logs, skipped, time_skipped = self.verify_rounds(cands, plan, rows, meta_by, "verify",
                                                                     reserve=reserve)
                else:
                    skipped = list(cands) if decision["verify"] else []
                for c in skipped:
                    c.verdict = "not_verified_budget"
                for c in time_skipped:
                    c.verdict = "not_verified_time"
                self.stats["candidates_unverified_budget"] = len(skipped)
                self.stats["time_skipped_candidates"] = len(time_skipped)
                st.update(ran=int(decision["verify"]), batches=len(logs), candidates=len(cands),
                          accepted=sum(b["accepted"] for b in logs),
                          model_rejected=sum(b["model_rejected"] for b in logs),
                          guard_rejected=sum(b["guard_rejected"] for b in logs),
                          guard_escaped=sum(b.get("guard_escaped", 0) for b in logs),
                          skipped_for_budget=len(skipped), skipped_for_time=len(time_skipped))
            trace["stages"]["verify"] = {"ran": decision["verify"], "batches": logs,
                                         "skipped_for_budget": [c.cid for c in skipped],
                                         "skipped_for_time": [c.cid for c in time_skipped]}
            verified = [c for c in cands if getattr(c, "verdict", None) not in (None, "not_verified_budget",
                                                                                "not_verified_time")]

            # 6. reconcile in SQL, then the gap round
            with self.stage("reconcile") as st:
                target_periods = self.target_periods(plan, chosen)
                rec = self.reconcile(rows, plan, target_periods)
                gap = {"ran": False}
                if plan["kind"] != "single" and self.budget.can(1, reserve=1):
                    gcands, gap = self.gap_fill(rec, plan, target_periods, scope_ids, rows, meta_by, cands)
                    if not gcands and not skipped:
                        alt, gap2 = self.gap_fill(rec, plan, target_periods, scope_ids, rows, meta_by, cands,
                                                  fallback=RT.plan_gap_pattern(plan, self.q.d_lang, o))
                        if alt:
                            gcands, gap = alt, gap2
                    # nothing to fill: the call held back for the gap round goes to the candidates the first
                    # round had to skip, so the reserve never costs a verification
                    extra, gcands = self.gap_batch(gcands, skipped, gap)
                    purpose = "verify_gap" if gcands else "verify"
                    if extra and o["gap_verify_call"] and self.time_for_verify(len(extra), plan):
                        logs.append(self.verify_batch(extra, plan, purpose, rows, meta_by, len(logs) + 1))
                        gap["accepted"] = logs[-1]["accepted"]
                        gap["second_round"] = purpose
                        if gcands:
                            cands = cands + gcands
                        skipped = [c for c in skipped if c not in extra]
                        verified += [c for c in extra if getattr(c, "verdict", None) is not None]
                        rec = self.reconcile(rows, plan, target_periods)
                    self.stats["gap_fill"] = int(bool(gcands))
                    self.stats["candidates_unverified_budget"] = len(skipped)
                items, ev_from = self.evidence_items(rec, cands, verified,
                                                     verified_ran=decision["verify"], warm_rows=warm_rows)
                evidence = self.a.final_pack(items, self.index, self.q)
                self.stats.update(rows=len(rec["series"]), periods_found=len(rec["periods_found"]),
                                  periods_missing=len(rec["periods_missing"]), conflicts=len(rec["conflicts"]),
                                  comparable=(-1 if rec["comparable"] is None else int(bool(rec["comparable"]))))
                st.update(rows_accepted=len(rows), series_rows=len(rec["series"]), gap=gap.get("ran", False),
                          periods_found=len(rec["periods_found"]), periods_missing=len(rec["periods_missing"]),
                          conflicts=len(rec["conflicts"]), evidence_from=ev_from, evidence_offered=len(items),
                          evidence_items=evidence.n_included, evidence_chars=evidence.chars,
                          sql_ms=rec.get("sql_ms"))
            trace["stages"]["gap_fill"] = gap

            # 7. the answer call (always kept)
            with self.stage("answer") as st:
                if not self.budget.can(1):
                    raise BudgetExceeded("AGENTIC3: no call left for the answer")
                answer = self.answer_call(evidence)
                st.update(answer_chars=len(answer))

            # 8. write back (no model call, one commit per row)
            with self.stage("writeback") as st:
                wb = self.write_back(rows, plan)
                st.update(**{k: wb[k] for k in ("written", "ms")})
            whole_summary = whole.summary()
        total_ms = _ms(t_all)
        self.stats["query_clock_ms"] = round(self.clock_s() * 1000.0, 1)
        self.stats["over_time"] = int(self.clock_s() > self.a.max_seconds)
        self.stats["lazy_index_ms"] = next((s["wall_ms"] for s in self.timing if s["stage"] == "lazy-index"), 0.0)
        structured = {
            "kind": plan["kind"], "entity": plan["entity"], "metric": plan["metric"], "unit_kind": plan["unit"],
            "route": decision["route"], "route_reason": decision["reason"],
            "series": [_series_row(r) for r in rec["series"]],
            "comparable": rec["comparable"], "main_class": rec["main_class"],
            "definition_classes": rec["definition_classes"], "trend": rec["trend"],
            "notes": RS.table_notes(rec, self.q.q_lang) if rec["series"] else [],
            "periods_found": rec["periods_found"], "periods_missing": rec["periods_missing"],
            "target_periods": [str(p) for p in target_periods], "conflicts": rec["conflicts"],
            "restatements": [_series_row(r) for r in rec["restatements"]],
            "narrower": [_series_row(r) for r in rec["narrower"]],
            "projections": [_series_row(r) for r in rec["projections"]],
            "rows_all": [_series_row(r) for r in rows], "evidence_from": ev_from,
        }
        trace["stages"]["answer"] = {"evidence_from": ev_from, "n_offered": evidence.n_offered,
                                     "n_included": evidence.n_included, "chars": evidence.chars,
                                     "table": items[0].text if (items and ev_from == "verified_table") else None}
        trace["stages"]["writeback"] = wb
        trace["structured"] = structured
        trace["guard_reasons"] = self.guard_reasons
        trace["calls"] = self.calls
        trace["calls_by_purpose"] = {p: self.budget.log.count(p) for p in dict.fromkeys(self.budget.log)}
        trace["timing"] = self._timing(whole_summary, total_ms)
        trace["timing"]["query_clock_ms"] = self.stats["query_clock_ms"]
        trace["timing"]["lazy_index_ms"] = self.stats["lazy_index_ms"]
        # the two latencies the paper reports: the 60 s budget excludes the first touch of a document's vectors,
        # the person waiting does not. Both are measured the same way (program time plus each call's recorded wall
        # time), so a question replayed from the call cache reports the latency of the run that produced it.
        trace["timing"]["wall_with_embedding_ms"] = round(self.stats["query_clock_ms"]
                                                          + self.stats["lazy_index_ms"], 1)
        trace["ledger_calls"] = whole_summary["llm_calls"]
        self.stats["program_ms"] = trace["timing"]["totals"]["program_ms"]
        trace["stats"] = self.stats
        order = {c.cid: i for i, c in enumerate(cands)}
        ranked = sorted(cands, key=lambda c: (getattr(c, "verdict", None) != "accepted", order.get(c.cid, 0)))
        retrieved = [c.to_item(rank=i, meta={"verdict": getattr(c, "verdict", None),
                                             "src": c.features.get("src"), "facet": c.features.get("facet")})
                     for i, c in enumerate(ranked, 1)]
        return AnswerResult(answer_text=answer, evidence=evidence, retrieved=retrieved, trace=trace,
                            structured=structured)

    # -- helpers of run() ----------------------------------------------------------------------------------------
    def target_periods(self, plan, chosen):
        if plan["kind"] == "single":
            return []
        p0, p1 = plan["period"]
        years = []
        for m in chosen:
            if m["years"]:
                for y in range(m["years"][0], m["years"][1] + 1):
                    if (p0 is None or y >= p0) and (p1 is None or y <= p1) and y not in years:
                        years.append(y)
        return sorted(years)

    def reserve_s(self, plan):
        """The approach never reads the query's answer format (fairness): a multi-period PLAN reserves the longer
        answer time."""
        return float(self.o["answer_reserve_timeline_s"] if plan["kind"] != "single" else self.o["answer_reserve_s"])

    def time_for_verify(self, n, plan, in_stage_s=0.0):
        est = float(self.o["verify_s_fixed"]) + float(self.o["verify_s_per_candidate"]) * n
        return self.budget.can(1, reserve=1) and (self.left_s() - in_stage_s >= est + self.reserve_s(plan))

    def verify_rounds(self, cands, plan, rows, meta_by, purpose, reserve=1):
        """v2's batching: as many verify calls as the cap and the clock allow, the answer call always reserved.

        `reserve` is v1's rule for a multi-period plan: one more call is held back for the gap round, so a timeline
        spends plan + verify + gap + answer rather than two candidate batches and no gap fill. When there turns
        out to be no gap to fill, `run` spends the held-back call on the candidates this round had to skip.
        """
        o = self.o
        n_calls = min(int(o["max_verify_calls"]), max(0, self.budget.left() - 1))
        n_batches = min(n_calls, math.ceil(len(cands) / float(o["verify_batch_max"])))
        if n_batches < 1:
            return [], list(cands), []
        size = math.ceil(len(cands) / float(n_batches))
        batches = [cands[i:i + size] for i in range(0, len(cands), size)][:n_batches]
        logs, skipped, time_skipped = [], list(cands[size * n_batches:]), []
        self.stats["verify_batches_planned"] = len(batches)
        in_stage_s = 0.0
        for k, b in enumerate(batches):
            if not self.budget.can(1, reserve=reserve):
                skipped += b
                continue
            if not self.time_for_verify(len(b), plan, in_stage_s):
                time_skipped += b
                self.stats["time_skipped_batches"] += 1
                continue
            lg = self.verify_batch(b, plan, purpose, rows, meta_by, k + 1)
            lg["clock_before_s"] = round(self.clock_s() + in_stage_s, 2)
            in_stage_s += (float(lg.get("llm_wall_ms") or 0.0) + float(lg.get("prompt_build_ms") or 0.0)
                           + float(lg.get("parse_guard_ms") or 0.0)) / 1000.0
            logs.append(lg)
        return logs, skipped, time_skipped

    def write_back(self, rows, plan):
        if not self.o["write_back"]:
            return {"written": 0, "skipped": {"disabled": len(rows)}, "ms": 0.0}
        wb = WB.write_rows(self.con, rows, plan=plan, run_id=self.a.client.ledger.run_id,
                           model=self.cfg.models["chat"], prompt_version=VERIFY_PROMPT_VERSION,
                           question_hash=WB.question_hash(self.q.text, plan))
        self.stats["written_back"] = wb["written"]
        return wb
