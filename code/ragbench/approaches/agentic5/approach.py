"""AGENTIC5: budgeted verification and reconciliation of an explicit EVIDENCE-REQUIREMENT MATRIX.

A new, separately named method (AGENTIC3 keeps its exact behaviour; nothing in agentic/, agentic2/ or agentic3/ is
changed). It reuses AGENTIC3's knowledge base, its call cache, its query clock and its shared answer call, and it
replaces the parts the reviewer found unsound for multi-document questions (work/analysis/UPGRADE_PLAN.md items 1, 5
and 7; the design at the end of work/analysis/UPGRADES_ASTRA_session.log):

    1 plan        (1 call)   AGENTIC3's plan + entities / metrics / periods / operation -> the requirement matrix
                             (`requirements.py`). Required periods come from the question and the plan, never from the
                             selected documents. A malformed plan falls back field by field; an optional repair call
                             consumes the optional second verify slot, and that loss is recorded.
    2 route       (no LLM)   the question's pooled retrieval over the WHOLE allowed pool, cards + the capped best 1-2
                             chunk hits per report, a shortlist of <= 8 that retains explicit entity/period matches, and
                             the pooled candidates outside it kept as an escape channel (`routing.py`).
    3 retrieve    (no LLM)   each requirement's own retrieval (its wording) over the documents that can serve it; a
                             requirement no shortlisted report can serve expands to its own entity's reports first.
                             Allocation across REQUIREMENTS, then channels; batch 1 <= 12 (`allocate.py`).
    4 verify 1    (1 call)   typed claims: requirement id, entity, metric, period, definition, value, unit, quote; the
                             verbatim guard; no model-definition escape (`claims.py`).
    5 reconcile   (no LLM)   identity = entity x metric x period x definition x unit family (`reconcile5.py`).
    6 verify 2    (0-1 call) only for MISSING or CONFLICTING requirements: their remaining candidates plus a gap
                             retrieval with the requirement's own wording, entity and period (`channels.py`).
    7 compute     (no LLM)   deterministic arithmetic from verified compatible operands (`arith.py`).
    8 evidence    (no LLM)   the table (company | period | metric/definition | value/unit | status | sources) and the
                             source snippets built jointly under the 6000-char budget, citations numbered once after the
                             final packing (`evidence5.py`).
    9 answer      (1 call)   the SHARED answer prompt and budget.

Cold and isolated by default: warm=False, write_back=False, a read-only knowledge-base connection for the question,
no query-time document embedding, and the remaining deadline enforced at every model request (the read timeout of
the call is cut to what is left; a verify call is not started without time for it and the answer).
"""
import json
import math  # noqa: F401
import os
import time

from ...contracts import AnswerResult, Approach
from ...kb import query as Q
from ...kb import schema as KB
from ...llm import BudgetExceeded
from ...util import safe_name  # noqa: F401
from ..agentic import planner as P
from ..agentic import verify as V
from ..agentic.approach import _series_row
from ..agentic2.approach import _Session2
from ..agentic2.lazyvec import LazyVectors
from ..agentic3 import retrieve as RT
from ..agentic3 import writeback as WB
from ..agentic3.approach import TypedKBAgent, _Session3
from . import allocate as AL
from . import arith as AR
from . import channels as CH
from . import claims as CL
from . import evidence5 as EV
from . import reconcile5 as RC
from . import requirements as RQ
from . import routing as RO5
from .defaults import (APPROACH_VERSION, DEFAULTS, EVIDENCE_VERSION, FLAGS, PLAN_PROMPT_VERSION, RECONCILE_VERSION,
                       ROUTE_VERSION, VALUE_FLAGS, VERIFY_PROMPT_VERSION)

OPTIONS_ENV = "AGENTIC5_OPTIONS"     # DEV tuning only: a JSON object of DEFAULTS overrides (recorded in trace.params)


class DeadlineSkip(Exception):
    """A model request that would start without the time it needs (the answer's reserve included)."""


def parse_flags(flags):
    if flags is None:
        return {}
    if isinstance(flags, str):
        d = {}
        for p in flags.replace("+", ",").split(","):
            p = p.strip()
            if p:
                k, _, v = p.partition("=")
                d[k.strip()] = v.strip() if v else True
        flags = d
    out = {}
    for k, v in dict(flags).items():
        if k not in FLAGS:
            raise ValueError("unknown AGENTIC5 flag %r; known: %s" % (k, ", ".join(FLAGS)))
        if k in VALUE_FLAGS:
            try:
                n = int(v)
            except (TypeError, ValueError):
                raise ValueError("AGENTIC5 flag %s takes an integer (got %r)" % (k, v))
            if n < 1:
                raise ValueError("AGENTIC5 flag %s must be >= 1" % k)
            out[k] = n
            continue
        v = v if isinstance(v, bool) else str(v).lower() in ("1", "true", "yes", "on")
        if v:
            out[k] = True
    return out


def _ms(t0):
    return round((time.perf_counter() - t0) * 1000.0, 1)


class RequirementAgent(TypedKBAgent):
    name = "AGENTIC5"
    needs_gold = False

    def __init__(self, cfg=None, client=None, flags=None, options=None):
        Approach.__init__(self, cfg=cfg, client=client, flags=parse_flags(flags))
        env = os.environ.get(OPTIONS_ENV)
        self.env_options = json.loads(env) if env else None
        self.opts = dict(DEFAULTS, **(self.env_options or {}), **(options or {}))
        f, o = self.flags, self.opts
        if f.get("candidates"):
            o["batch1_max"] = int(f["candidates"])
        if f.get("shortlist"):
            o["shortlist_max"] = int(f["shortlist"])
        if f.get("write_back"):
            o["write_back"] = True
        if f.get("warm"):
            o["warm"] = True
        if f.get("no_gap_fill"):
            o["gap_fill"] = False
        if f.get("no_arith"):
            o["arith"] = False
        if f.get("no_escape"):
            o["escape_max"] = 0
        if f.get("no_dense"):
            o["dense"] = "off"
        if f.get("no_hybrid"):
            o["hybrid"] = False
        o["facets"] = False                 # metrics replace facets; no disjoint-vocabulary rule
        self.warm = bool(o["warm"])
        self.use_trigram = True
        self.use_vectors = o.get("dense") != "off"
        self.use_verify = True
        self.use_tree = True
        self.use_factdb = True
        self.line_windows = False
        self.doc_selection = True
        caps = self.cfg.caps or {}
        self.max_calls = int(caps.get("agentic5_max_calls") or caps.get("agentic3_max_calls") or o["max_calls_default"])
        self.max_seconds = float(caps.get("agentic5_max_seconds") or caps.get("agentic3_max_seconds")
                                 or o["max_seconds_default"])
        self._cache = None
        self._cache_for = None
        self._con = None
        self._qcon = None
        self._cards = {}
        self.lazy = LazyVectors(self)

    # the ragbench/variants.py options act at query time only: they name the run variant, not the index
    index_neutral_flags = ("doc_card", "section_expand")

    def config_signature(self):
        sig = super().config_signature()
        sig["approach"] = APPROACH_VERSION
        return sig

    def variant_routed_ids(self, query, index):
        """variants-v2 card hook. AGENTIC5 routes AFTER, and from, its plan call (the shortlist needs the requirement
        matrix), so nothing is routed when the card is built: the card is question-ranked, and the session records
        afterwards how the card's shortlist compares with the routed one (trace.variants.card_vs_route)."""
        return None

    def _build_index(self, corpus_set, index_dir):
        """AGENTIC3's ingestion into the knowledge base, plus the shared hybrid index over the same set (chunks and
        BM25; vectors from the shared cache only - the prebuild CLI embeds, this never does)."""
        payload = super()._build_index(corpus_set, index_dir)
        payload["hybrid"] = CH.prebuild_hybrid(self, corpus_set)
        return payload

    def query_kb(self):
        """The question's connection: READ-ONLY (mode=ro + query_only) unless write-back or warm reads were asked
        for, so an experiment cannot change the knowledge base it is measured on."""
        if self.opts["write_back"]:
            return self.kb()
        if self._qcon is None:
            self._qcon = RT.open_kb(self.cfg, read_only=True)
        return self._qcon

    def card(self, doc):
        if doc.id not in self._cards:
            from ...variants import document_card
            self._cards[doc.id] = document_card(doc)
        return self._cards[doc.id]

    def _answer(self, query, index):
        return _Session5(self, query, index).run()


# ---------------------------------------------------------------------------------------------------------------
# one question
# ---------------------------------------------------------------------------------------------------------------

class _Session5(_Session3):
    def __init__(self, ap, query, index):
        _Session2.__init__(self, ap, query, index)       # NOT _Session3's: it opens the writable knowledge base
        self.con = ap.query_kb()
        self.reader = Q.DocReader(self.con)
        self.stats.update({"requirements": 0, "req_verified": 0, "req_conflict": 0, "req_missing": 0,
                           "claims": 0, "claims_unbound": 0, "shortlist": 0, "escape": 0, "expanded_docs": 0,
                           "verify2": 0, "verify2_lost_to_repair": 0, "deadline_skips": 0, "computed": 0,
                           "written_back": 0, "guard_escaped": 0, "source_label_escape": 0, "label_completed": 0})
        self.deadlines = []

    # -- every model request is bounded by the remaining deadline ------------------------------------------------
    def chat(self, purpose, messages, *, fmt=None, num_predict=None, version=""):
        o = self.o
        left = self.left_s()
        reserve = 0.0 if purpose == "answer" else self.reserve_s(self._plan or {"kind": "single"})
        if purpose != "answer" and purpose not in ("plan",) and left < float(o["call_min_s"]) + reserve:
            self.stats["deadline_skips"] += 1
            self.deadlines.append({"purpose": purpose, "left_s": round(left, 2), "skipped": True})
            raise DeadlineSkip("%s: %.1f s left, %.1f s needed" % (purpose, left, float(o["call_min_s"]) + reserve))
        limit = max(float(o["answer_floor_s"]), left) if purpose == "answer" else max(float(o["call_min_s"]),
                                                                                      left - reserve)
        client = self.a.client
        old = getattr(client, "timeout", None)
        bounded = isinstance(old, tuple) and len(old) == 2
        if bounded:
            client.timeout = (old[0], max(1.0, min(float(old[1]), limit)))
        self.deadlines.append({"purpose": purpose, "left_s": round(left, 2), "read_timeout_s": round(limit, 2),
                               "applied": bounded})
        try:
            return super().chat(purpose, messages, fmt=fmt, num_predict=num_predict, version=version)
        finally:
            if bounded:
                client.timeout = old

    _plan = None

    # -- stage 1 -------------------------------------------------------------------------------------------------
    def plan5(self, metas):
        o = self.o
        choose = len(metas) > 1
        company_keys = sorted({m["company"] for m in metas if m.get("company")}) if choose else None
        msgs = RQ.plan_messages(self.q.text, metas, choose, o)
        schema = RQ.plan_schema(company_keys, o)
        r = self.chat("plan", msgs, fmt=schema, num_predict=int(o["plan_num_predict"]), version=PLAN_PROMPT_VERSION)
        obj, how = P.parse_plan_text(r.text)
        problem = P.plan_problem(obj)
        rep = {"parse": how, "problem": problem, "model_output": (r.text or "")[:1500], "done_reason": r.done_reason}
        if how in ("extract", "suffix_repair"):
            self.stats["plan_parse_repaired"] = 1
        r2 = None
        if problem and o["plan_repair_call"] and self.budget.can(1, reserve=2):
            msgs2 = msgs + [{"role": "assistant", "content": (r.text or "")[:1500]},
                            {"role": "user", "content": P.REPAIR_USER.format(problem=problem)}]
            try:
                r2 = self.chat("plan_repair", msgs2, fmt=schema, num_predict=int(o["plan_num_predict"]),
                               version=PLAN_PROMPT_VERSION)
            except DeadlineSkip as e:
                rep["repair"] = {"skipped": str(e)}
        if r2 is not None:
            obj2, how2 = P.parse_plan_text(r2.text)
            problem2 = P.plan_problem(obj2)
            rep["repair"] = {"parse": how2, "problem": problem2, "model_output": (r2.text or "")[:1500]}
            self.stats["plan_repair"] = 1
            if not problem2:
                obj, problem = obj2, None
            elif isinstance(obj2, dict) and not isinstance(obj, dict):
                obj = obj2
        if problem:
            plan = P.program_plan(self.q.text, self.q.q_lang, metas, choose, o)
            self.stats["plan_program"] = 1
        else:
            plan = P.validate_plan(obj, question=self.q.text, q_lang=self.q.q_lang, metas=metas,
                                   choose_companies=choose, opts=o)
        plan["facets"] = []
        plan["model"] = rep
        matrix = RQ.build_requirements(obj if isinstance(obj, dict) else {}, plan, self.q.text, metas, o)
        if problem:
            matrix["fixes"].append("program_plan")
        self.stats["requirements"] = len(matrix["requirements"])
        self._plan = plan
        return plan, matrix

    # -- stage 4: typed verification -------------------------------------------------------------------------------
    def verify5(self, batch, matrix, plan, claims, meta_by, purpose, batch_no):
        o = self.o
        t0 = time.perf_counter()
        fixed = CL.fix_thousands(batch)
        reqs = matrix["requirements"]
        msgs = CL.verify_messages(self.q.text, RQ.summary(matrix), batch, self.corpus)
        n = len(batch)
        num_predict = int(o["verify5_num_predict_base"]) + int(o["verify5_num_predict_per_item"]) * n
        prompt_ms = _ms(t0)
        out = {"batch": batch_no, "purpose": purpose, "candidates": [c.cid for c in batch], "n": n,
               "num_predict": num_predict, "items": [], "thousands_fixed": fixed}
        try:
            r = self.chat(purpose, msgs, fmt=CL.verify_schema(n), num_predict=num_predict,
                          version=VERIFY_PROMPT_VERSION)
        except DeadlineSkip as e:
            for c in batch:
                c.verdict = "not_verified_time"
            out.update(skipped=str(e), accepted=0)
            return out
        call = self.calls[-1]
        self.stats["verify_calls"] += 1
        t1 = time.perf_counter()
        items, how = CL.parse_items(r.text, n)
        out.update(call=call, parse=how)
        rx_by_metric = {m["id"]: CL.term_rx(m["terms"]) for m in matrix["metrics"]}
        all_rx = [x for v in rx_by_metric.values() for x in v]
        unit_kind = plan.get("unit") or "count"
        min_letters = int(o["min_quote_letters"])
        counts = {"accepted": 0, "unbound": 0, "model_rejected": 0, "model_omitted": 0, "guard_rejected": 0,
                  "unparsed": 0, "duplicate": 0}
        seen = set()
        for i, c in enumerate(batch, 1):
            its = items.get(i) or []
            self.stats["candidates_verified"] += 1
            if not its:
                v = "model_omitted" if how else "unparsed"
                c.verdict = v
                counts[v] += 1
                out["items"].append({"cid": c.cid, "doc_id": c.doc_id, "verdict": v})
                continue
            verdicts = []
            label = CL.source_label(c)
            for it in its:
                rec = {"cid": c.cid, "doc_id": c.doc_id}
                rec.update({k: str(it.get(k) or "")[:300] for k in CL.VALUE_FIELDS})
                rel = it.get("relevant")
                rel = rel if isinstance(rel, bool) else str(rel).strip().lower() in ("true", "yes", "1")
                if not rel:
                    rec["verdict"] = "model_rejected"
                    verdicts.append("model_rejected")
                    out["items"].append(rec)
                    continue
                ok, reason, ginfo = V.guard(it, c.text, unit_kind=unit_kind, min_quote_letters=min_letters,
                                            term_rx=all_rx or None)
                guard_how = "ok"
                if not ok and reason == "quote_without_term" and o["source_label_escape"] and label \
                        and CL.has_term(label, all_rx):
                    ok, reason, ginfo = V.guard(it, c.text, unit_kind=unit_kind, min_quote_letters=min_letters,
                                                term_rx=None)
                    if ok:
                        guard_how = "ok_source_label"
                        self.stats["source_label_escape"] += 1
                if not ok and reason == "quote_no_label":
                    q2, g2 = CL.complete_label(it, c, unit_kind, all_rx, min_letters)
                    if q2:
                        it = dict(it, quote=q2)
                        ok, reason, ginfo, guard_how = True, "ok", g2, "ok_label_completed"
                        self.stats["label_completed"] += 1
                if not ok:
                    rec["verdict"] = "guard:" + reason
                    rec["guard"] = reason
                    counts["guard_rejected"] += 1
                    self.guard_reasons[reason] = self.guard_reasons.get(reason, 0) + 1
                    verdicts.append(rec["verdict"])
                    out["items"].append(rec)
                    continue
                key = (c.cid, ginfo.get("value_num"), V.canon(str(it.get("quote") or "")), str(it.get("req") or ""))
                if key in seen:
                    rec["verdict"] = "duplicate"
                    counts["duplicate"] += 1
                    out["items"].append(rec)
                    continue
                seen.add(key)
                dm = meta_by.get(c.doc_id) or {}
                cper, per_how = CL.claim_period(it.get("period"), it.get("quote"), dm.get("years"))
                req, bhow, detail = CL.bind(it, c, reqs, rx_by_metric, dm, cper, label)
                uid = V.value_unit_id(c, self.corpus, ginfo.get("value_token"), str(it.get("quote") or ""))
                claim = CL.make_claim(it, c, ginfo, dm, uid, req, bhow, detail, cper, per_how, self.corpus)
                fact = (c.features or {}).get("fact") or {}
                claim.update(claim_id="k%d" % (len(claims) + 1), guard=guard_how, batch=batch_no,
                             fact_id=fact.get("fact_id"), label=fact.get("label"))
                claims.append(claim)
                rec.update(guard=guard_how, value_num=ginfo.get("value_num"), claim_id=claim["claim_id"],
                           bound_to=claim["req_id"], bind=bhow, period_key=claim["period_key"],
                           unit_status=claim["unit_status"])
                if req is None:
                    rec["verdict"] = "unbound:" + bhow
                    counts["unbound"] += 1
                    self.stats["claims_unbound"] += 1
                else:
                    rec["verdict"] = "accepted"
                    counts["accepted"] += 1
                verdicts.append("accepted" if req is not None else "unbound")
                out["items"].append(rec)
            if verdicts and all(v == "model_rejected" for v in verdicts):
                counts["model_rejected"] += 1
            order = {"accepted": 0, "unbound": 1, "guard": 2, "duplicate": 3, "model_rejected": 4}
            c.verdict = min(verdicts, key=lambda v: order.get(v.split(":")[0], 9)) if verdicts else "duplicate"
        for k in ("accepted", "model_rejected", "model_omitted", "guard_rejected", "unparsed"):
            self.stats[k] += counts[k]
        self.stats["claims"] = len(claims)
        out.update(counts, prompt_build_ms=prompt_ms, parse_guard_ms=_ms(t1), wall_ms=_ms(t0),
                   llm_wall_ms=call["wall_ms"], cache_hit=call["cache_hit"], guard_reasons=dict(self.guard_reasons))
        return out

    # -- helpers -----------------------------------------------------------------------------------------------
    def cards(self, metas):
        out = {}
        for m in metas:
            d = m["doc_id"]
            if self.corpus.has_doc(d):
                out[d] = self.a.card(self.corpus.doc(d))
            else:
                out[d] = m.get("title") or d
        return out

    def requirement_docs(self, req, metas, short, scored, used_expansion):
        docs = RO5.serving_docs(req, metas, short)
        if docs:
            return docs, False
        exp = RO5.expansion_docs(req, metas, short, scored, self.o)
        for d in exp:
            used_expansion.setdefault(d, []).append(req["id"])
        return (exp or list(short)), bool(exp)

    def unverified_items(self, cands):
        pool = self.by_verdict(cands)
        return [c.to_item(meta={"verified": False}) for c in pool]

    @staticmethod
    def block_requirements(reqs):
        """The requirement matrix in `evidence.build_blocks`' shape (terms / entity / period, from the QUESTION)."""
        out = []
        for r in reqs:
            terms = [t for lang in ("fa", "en") for t in (r.get("terms") or {}).get(lang) or []]
            spec = {"id": r["id"], "terms": terms or ([r["metric"]] if r.get("metric") else [])}
            ents = [t for t in (r.get("entity_terms") or []) if t]
            if ents:
                spec["entity"] = ents
            if r.get("period"):
                spec["period"] = [str(r["period"]["year"])]
            out.append(spec)
        return out

    def blocks(self, cands, reqs, budget):
        """Unverified support as the shared foundation's evidence blocks (`ragbench.evidence.build_blocks`), in the
        verdict-aware order, one block reserved per requirement. -> (items, report) or (None, reason)."""
        try:
            from ...evidence import build_blocks
        except ImportError:
            return None, {"skipped": "absent"}
        if budget <= 0 or not cands:
            return [], {"skipped": "no_room" if budget <= 0 else "no_candidates"}
        try:
            items, report = build_blocks(self.by_verdict(cands), self.corpus, budget=int(budget),
                                         requirements=self.block_requirements(reqs), question=self.q.text,
                                         cfg=self.cfg)
        except Exception as e:  # noqa: BLE001  (the plain candidate pack is the fallback, recorded)
            return None, {"skipped": "build_blocks_failed: %s" % str(e)[:200]}
        for it in items:
            it.meta = dict(it.meta or {}, verified=False)
        return items, {k: report.get(k) for k in ("n_blocks", "rendered_chars", "requirements", "n_dropped",
                                                   "warnings")}

    def time_ok(self, n, plan):
        return self.time_for_verify(n, plan)

    # -- the pipeline --------------------------------------------------------------------------------------------
    def run(self):
        t_all = time.perf_counter()
        o = self.o
        trace = {"variant": self.a.variant, "max_calls": self.budget.max, "max_seconds": self.a.max_seconds,
                 "prompts": {"plan": PLAN_PROMPT_VERSION, "verify": VERIFY_PROMPT_VERSION, "answer": "shared"},
                 "versions": {"approach": APPROACH_VERSION, "route": ROUTE_VERSION, "reconcile": RECONCILE_VERSION,
                              "evidence": EVIDENCE_VERSION, "query": Q.QUERY_VERSION,
                              "kb": KB.component_versions(self.cfg)},
                 "params": {k: (o[k] if not callable(o[k]) else "callable") for k in sorted(o)
                            if not k.startswith("call_cache") and not k.endswith("_path")},
                 "options_env": self.a.env_options, "call_cache": self.cache.path, "stages": {},
                 "flags": dict(self.a.flags),
                 "isolation": {"warm": bool(o["warm"]), "write_back": bool(o["write_back"]),
                               "kb_read_only": not o["write_back"], "query_time_document_embedding": False}}
        scope_ids = [d.id for d in self.a.docs_in_scope(self.q, self.index)]
        self.stats["docs_scope"] = len(scope_ids)
        claims, logs = [], []
        with self.a.client.ledger.scope() as whole:
            # 1. plan -> the requirement matrix
            with self.stage("plan") as st:
                metas = Q.document_metas(self.con, doc_ids=scope_ids)
                meta_by = {m["doc_id"]: m for m in metas}
                plan, matrix = self.plan5(metas)
                reqs = matrix["requirements"]
                st.update(requirements=len(reqs), operation=matrix["operation"], repair=self.stats["plan_repair"],
                          program_plan=self.stats["plan_program"])
            trace["stages"]["plan"] = {k: plan.get(k) for k in ("entity", "metric", "kind", "unit", "terms",
                                                                 "entity_terms", "period", "companies",
                                                                 "dropped_terms", "fixes", "model")}
            trace["stages"]["requirements"] = {"operation": matrix["operation"], "fixes": matrix["fixes"],
                                               "entities": matrix["entities"], "metrics": matrix["metrics"],
                                               "periods": matrix["periods"], "open_series": matrix["open_series"],
                                               "truncated": matrix["truncated"], "requirements": reqs,
                                               "lines": RQ.summary(matrix)}
            trace["stages"]["plan"]["missing_documents"] = RT.missing_documents(self.con, scope_ids)

            # 2. routing over the pool, and the escape channel
            with self.stage("route") as st:
                ch = CH.Channels(self)
                kb_ids = [m["doc_id"] for m in metas]
                # the hybrid index covers the whole corpus set; the KB path can only read what was ingested
                pool, pinfo = ch.pooled(plan, scope_ids if ch.hybrid is not None else kb_ids)
                short, rinfo = RO5.shortlist(metas, self.cards(metas) if len(metas) > 1 else {}, pool, matrix, o)
                escape = RO5.escape_candidates(pool, short, o) if int(o["escape_max"]) > 0 else []
                self.stats.update(shortlist=len(short), escape=len(escape), docs_targeted=len(short))
                st.update(pool=len(pool), shortlist=len(short), escape=len(escape), hybrid=ch.diag["hybrid"])
            trace["stages"]["route"] = dict(rinfo, pool=pinfo, escape=[c.trace(160) for c in escape])
            vt = self.a._variant_trace
            if vt is not None and vt.get("card_selection"):
                shown = [r["doc_id"] for r in vt["card_selection"].get("shown") or []]
                vt["card_vs_route"] = {"routed_ids_at_card_time": None,
                                       "reason": "AGENTIC5 routes after its plan call",
                                       "route_shortlist": list(short),
                                       "shortlist_shown_in_card": [d for d in short if d in shown]}

            # 3. each requirement's own retrieval, then allocation (requirements first, then channels)
            with self.stage("retrieve") as st:
                cands, rq_info, expanded = [], {}, {}
                scored = rinfo.get("scores") or []
                for r in reqs:
                    docs, exp = self.requirement_docs(r, metas, short, scored, expanded)
                    rplan = RQ.requirement_plan(plan, r)
                    cs, info = ch.for_requirement(r, rplan, docs, int(o["req_pool_cap"]))
                    cands += cs
                    rq_info[r["id"]] = dict(info, docs=docs, expanded=exp)
                # the question's own pooled hits inside the shortlist serve every requirement their report can hold
                for c in pool:
                    if c.doc_id in short:
                        tags = [r["id"] for r in reqs if c.doc_id in RO5.serving_docs(r, metas, short)]
                        if tags:
                            c.features[CH.REQ_KEY] = sorted(set(c.features.get(CH.REQ_KEY) or []) | set(tags))
                            cands.append(c)
                batch1, rest, ainfo = AL.allocate(cands, reqs, int(o["batch1_max"]), escape=escape)
                self.stats["expanded_docs"] = len(expanded)
                self.stats["candidates"] = len(batch1) + len(rest)
                st.update(candidates=len(cands), batch1=len(batch1), rest=len(rest), starved=len(ainfo["starved"]))
            trace["stages"]["retrieve"] = {"requirements": rq_info, "allocation": ainfo, "expanded": expanded,
                                           "channels": ch.diag,
                                           "batch1": [c.trace(int(o["trace_text_chars"])) for c in batch1]}

            # 4. verify batch 1 (the plan call and the answer call are always accounted for)
            with self.stage("verify") as st:
                ran = False
                if batch1 and self.budget.can(1, reserve=1) and self.time_ok(len(batch1), plan):
                    logs.append(self.verify5(batch1, matrix, plan, claims, meta_by, "verify", 1))
                    ran = not logs[-1].get("skipped")
                else:
                    for c in batch1:
                        c.verdict = "not_verified_budget" if not self.budget.can(1, reserve=1) else "not_verified_time"
                st.update(ran=int(ran), candidates=len(batch1), accepted=sum(b.get("accepted", 0) for b in logs))
            recon = RC.reconcile(claims, matrix, known_stems=self.known_stems(plan), opts=o)

            # 6. verify batch 2: only missing / conflicting requirements (a repair call has consumed the slot)
            gap = {"ran": False, "targets": recon["missing"] + recon["conflicting"]}
            batch2 = []
            with self.stage("gap_fill") as st:
                targets = [r for r in reqs if r["id"] in set(gap["targets"])]
                if targets and not o["gap_fill"]:
                    gap["skipped"] = "flag_no_gap_fill"
                elif targets and not self.budget.can(1, reserve=1):
                    gap["skipped"] = "call_budget"
                    if self.stats["plan_repair"]:
                        gap["skipped"] = "slot_used_by_plan_repair"
                        self.stats["verify2_lost_to_repair"] = 1
                elif targets:
                    pool2, ginfo = [], {}
                    for r in targets:
                        docs, _exp = self.requirement_docs(r, metas, short, scored, expanded)
                        more = RO5.expansion_docs(r, metas, docs, scored, o)
                        for d in more:
                            expanded.setdefault(d, []).append(r["id"])
                        rplan = RQ.requirement_plan(plan, r)
                        gc, gi = ch.for_requirement(r, rplan, list(dict.fromkeys(docs + more)),
                                                    int(o["req_pool_cap"]), round_no=2)
                        for c in gc:
                            c.features["src"] = sorted(set(c.features.get("src") or []) | {"gap"})
                        pool2 += gc
                        ginfo[r["id"]] = dict(gi, docs=docs, added_docs=more)
                    pool2 += [c for c in rest if set(c.features.get(CH.REQ_KEY) or []) & {r["id"] for r in targets}]
                    seen_keys = {(c.doc_id, tuple(sorted(c.unit_ids))) for c in batch1}
                    pool2 = [c for c in pool2 if (c.doc_id, tuple(sorted(c.unit_ids))) not in seen_keys]
                    esc2 = [c for c in escape if c not in batch1]
                    batch2, _rest2, ainfo2 = AL.allocate(pool2, targets, int(o["batch2_max"]), escape=esc2,
                                                         prefix="d")
                    gap.update(retrieval=ginfo, allocation=ainfo2, candidates=len(batch2))
                    if batch2 and self.time_ok(len(batch2), plan):
                        logs.append(self.verify5(batch2, matrix, plan, claims, meta_by, "verify_gap", 2))
                        gap["ran"] = not logs[-1].get("skipped")
                        self.stats["verify2"] = int(gap["ran"])
                        recon = RC.reconcile(claims, matrix, known_stems=self.known_stems(plan), opts=o)
                    elif batch2:
                        gap["skipped"] = "time"
                        for c in batch2:
                            c.verdict = "not_verified_time"
                    self.stats["expanded_docs"] = len(expanded)
                st.update(ran=int(gap["ran"]), targets=len(gap["targets"]), candidates=len(batch2))
            gap["after"] = {"missing": recon["missing"], "conflicting": recon["conflicting"]}
            trace["stages"]["gap_fill"] = gap
            trace["stages"]["verify"] = {"batches": logs}

            # 7 + 8. arithmetic and the evidence
            with self.stage("reconcile") as st:
                computed = AR.compute(matrix, recon, o) if o["arith"] else []
                self.stats["computed"] = sum(1 for c in computed if c["status"] == "ok")
                verified_cands = [c for c in batch1 + batch2]
                by_cid = {c.cid: c for c in verified_cands}
                any_value = any(e["status"] != "missing" for e in recon["requirements"].values())
                blocks_info = None
                if any_value:
                    missing_ids = set(recon["missing"])
                    extra = []
                    if o["fill_unverified"] and missing_ids:
                        pool_x = [c for c in batch1 + batch2 + rest
                                  if set(c.features.get(CH.REQ_KEY) or []) & missing_ids
                                  and getattr(c, "verdict", None) != "accepted"]
                        miss_reqs = [r for r in reqs if r["id"] in missing_ids]
                        if pool_x:
                            def extra(room, _pool=pool_x, _reqs=miss_reqs):
                                nonlocal blocks_info
                                its, blocks_info = self.blocks(_pool, _reqs, room)
                                return its if its is not None else [
                                    c.to_item(meta={"verified": False}) for c in self.by_verdict(_pool)[:6]]
                    matrix["doc_hint"] = next((c.doc_id for c in verified_cands), "")
                    expand = bool(self.a.flags.get("section_expand"))
                    est = {} if expand else None
                    evidence, einfo, table_text = EV.build(matrix, recon, computed, by_cid, self.index, self.cfg,
                                                           self.q.q_lang, o, extra=extra, state=est)
                    ev_from = "requirement_table"
                    if expand:
                        # section_expand: once, at the final boundary; citations re-numbered after it, or skipped
                        pack2, text2, sx = EV.expand_after_fit(est, self.corpus, self.q.text, self.cfg)
                        if pack2 is not None:
                            evidence, table_text = pack2, text2
                            einfo["citations"] = sx["citations_after"]
                            einfo["chars"] = pack2.chars
                        einfo["section_expand"] = sx
                        if self.a._variant_trace is not None:
                            self.a._variant_trace["section_expand"] = sx
                else:
                    pool_u = batch1 + batch2 + rest
                    items, blocks_info = self.blocks(pool_u, reqs, int(self.cfg.evidence["max_chars"]))
                    if items is None:
                        items = self.unverified_items(pool_u)
                    # no table, no citations to keep: the shared final boundary (== pack() unless section_expand)
                    evidence = self.a.final_pack(items, self.index, self.q)
                    einfo, table_text = {"version": EVIDENCE_VERSION, "unverified": len(items)}, None
                    ev_from = "unverified_blocks" if blocks_info and "skipped" not in blocks_info \
                        else "unverified_candidates"
                    self.stats["unverified_fallback"] = 1
                einfo["blocks"] = blocks_info
                self.stats.update(req_verified=sum(1 for e in recon["requirements"].values()
                                                   if e["status"] == "verified"),
                                  req_conflict=len(recon["conflicting"]), req_missing=len(recon["missing"]),
                                  rows=len(recon["rows"]))
                st.update(evidence_from=ev_from, evidence_items=evidence.n_included, evidence_chars=evidence.chars,
                          computed=len(computed))
            trace["stages"]["reconcile"] = {"summary": RC.summary(recon), "missing": recon["missing"],
                                            "conflicting": recon["conflicting"],
                                            "unbound": [_series_row(c) for c in recon["unbound"]],
                                            "computed": computed}
            trace["stages"]["evidence"] = dict(einfo, evidence_from=ev_from)

            # 9. the answer call (always kept)
            with self.stage("answer") as st:
                if not self.budget.can(1):
                    raise BudgetExceeded("AGENTIC5: no call left for the answer")
                answer = self.answer_call(evidence)
                st.update(answer_chars=len(answer))

            # write-back only when asked for (off by default: cold and isolated)
            with self.stage("writeback") as st:
                wb = self.write_back5(claims, plan)
                st.update(written=wb.get("written", 0))
            whole_summary = whole.summary()
        total_ms = _ms(t_all)
        self.stats["query_clock_ms"] = round(self.clock_s() * 1000.0, 1)
        self.stats["over_time"] = int(self.clock_s() > self.a.max_seconds)
        series_rows = [c for c in recon["rows"] if c.get("req_status") in ("verified", "conflict")]
        structured = {
            "kind": plan["kind"], "operation": matrix["operation"], "entity": plan["entity"], "metric": plan["metric"],
            "unit_kind": plan["unit"], "route": "REQUIREMENTS",
            "requirements": [{"id": r["id"], "entity": r["entity"], "metric": r["metric"], "period": r["period_key"],
                              "status": recon["requirements"][r["id"]]["status"]} for r in reqs],
            "series": [dict(_series_row(c), req_id=c.get("req_id"), entity=c.get("entity_name"),
                            period_key=c.get("period_key"), status=c.get("req_status")) for c in series_rows],
            "computed": computed, "comparable": None, "trend": None,
            "periods_missing": [recon["requirements"][rid]["requirement"]["period_key"] for rid in recon["missing"]
                                if recon["requirements"][rid]["requirement"]["period_key"]],
            "missing": recon["missing"], "conflicting": recon["conflicting"],
            "rows_all": [_series_row(c) for c in claims], "evidence_from": ev_from,
        }
        trace["stages"]["answer"] = {"evidence_from": ev_from, "n_offered": evidence.n_offered,
                                     "n_included": evidence.n_included, "chars": evidence.chars,
                                     "table": table_text}
        trace["stages"]["writeback"] = wb
        trace["structured"] = structured
        trace["guard_reasons"] = self.guard_reasons
        trace["calls"] = self.calls
        trace["deadlines"] = self.deadlines
        trace["calls_by_purpose"] = {p: self.budget.log.count(p) for p in dict.fromkeys(self.budget.log)}
        trace["timing"] = self._timing(whole_summary, total_ms)
        trace["timing"]["query_clock_ms"] = self.stats["query_clock_ms"]
        trace["ledger_calls"] = whole_summary["llm_calls"]
        self.stats["program_ms"] = trace["timing"]["totals"]["program_ms"]
        trace["stats"] = self.stats
        allc = batch1 + batch2
        ranked = sorted(allc, key=lambda c: (getattr(c, "verdict", None) != "accepted", allc.index(c)))
        retrieved = [c.to_item(rank=i, meta={"verdict": getattr(c, "verdict", None), "src": c.features.get("src"),
                                             "req": c.features.get(CH.REQ_KEY)})
                     for i, c in enumerate(ranked, 1)]
        return AnswerResult(answer_text=answer, evidence=evidence, retrieved=retrieved, trace=trace,
                            structured=structured)

    def write_back5(self, claims, plan):
        if not self.o["write_back"]:
            return {"written": 0, "skipped": {"disabled": len(claims)}, "ms": 0.0}
        rows = [c for c in claims if c.get("req_id")]
        wb = WB.write_rows(self.a.kb(), rows, plan=plan, run_id=self.a.client.ledger.run_id,
                           model=self.cfg.models["chat"], prompt_version=VERIFY_PROMPT_VERSION,
                           question_hash=WB.question_hash(self.q.text, plan))
        self.stats["written_back"] = wb["written"]
        return wb
