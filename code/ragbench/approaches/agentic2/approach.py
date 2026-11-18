"""AGENTIC2 ("AGENTIC v2"): tree + fact DB + verify, at most 4 model calls and 60 s per question.

    1 plan          (1 call)  the v1 planner: entity, metric, kind, fa+en terms, period, target documents from metadata
    2 retrieve      (no LLM)  a. lazy bge-m3 vectors of the target documents' tree leaves (first touch only; stage
                                 "lazy-index", reported as preprocessing, excluded from the 60 s query clock)
                              b. vector search with the question (+ a terms query vector), top-k passages per document
                              c. tree expansion: table caption + header + rows, the label of a detached number, the
                                 section path in every header
                              d. fact DB: L0 regex facts of the target documents in an in-memory SQLite table, rows whose
                                 label matches a plan term (both languages) and whose year is in the period
                              e. union, dedupe, rank (vector score + number + fact-DB + term bonuses), cap 12 (24 over
                                 several documents)
    3 verify        (1 call, at most 2)  the v1 batched verifier with the verbatim code guard
    4 reconcile + answer (1 call)  the v1 reconciliation table (+ source windows, then the remaining candidates in rank
                                 order while the 6000-char budget allows) -> the SHARED answer prompt (timeline prompt for
                                 timeline questions)

Budget: caps.agentic2_max_calls (4) and caps.agentic2_max_seconds (60). The clock is the question's program time plus
each model call's wall time as recorded (a call replayed from the per-call cache counts with its original wall time, so a
resumed question takes the same decisions). A verify batch runs only when its estimated time plus the answer reserve
fits in what is left; otherwise the answer is given from what is verified (stats time_skipped_batches).

Flags (ablations): no_vectors (v1 grep windows instead of vectors), no_factdb, no_tree (passages / units without
expansion), no_verify (ranked candidates straight to the answer call), candidates=<n> (cap; two verify calls of
ceil(n/2) when n > verify_batch_max).
Resumable like v1: every model call is cached in work/runs/<out>/agentic_calls_<variant>.sqlite; vectors in the shared
embedding cache; fact rows per document content hash.
"""
import json
import math
import os
import re
import time
from pathlib import Path

import numpy as np

from ...contracts import AnswerResult, Approach, RetrievedItem
from ...llm import BudgetExceeded
from ...store import STORE_FORMAT, build_plain_store, grep
from ...util import read_json, safe_name
from ..agentic import reconcile as R
from ..agentic import search as S
from ..agentic.approach import CallBudget, CallCache, _Session, _series_row
from ..agentic.defaults import PLAN_PROMPT_VERSION, VERIFY_PROMPT_VERSION
from ..agentic.docmeta import doc_meta, select_docs
from ..agentic.numbers import candidate_numbers, find_numbers, unit_kind_match
from . import tree as T
from .defaults import DEFAULTS, FACTDB_VERSION, FLAGS, RANK_VERSION, TREE_VERSION, VALUE_FLAGS
from .factdb import FactDB
from .lazyvec import LazyVectors

OPTIONS_ENV = "AGENTIC2_OPTIONS"     # DEV tuning only: a JSON object of DEFAULTS overrides (recorded in trace.params)
_AR_THOUSANDS = re.compile("(?<=[0-9])،(?=[0-9]{3}(?![0-9]))")   # 2،788،550 (Arabic comma as thousands separator)


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
            raise ValueError("unknown AGENTIC2 flag %r; known: %s" % (k, ", ".join(FLAGS)))
        if k in VALUE_FLAGS:
            try:
                n = int(v)
            except (TypeError, ValueError):
                raise ValueError("AGENTIC2 flag %s takes an integer (got %r)" % (k, v))
            if n < 1:
                raise ValueError("AGENTIC2 flag %s must be >= 1" % k)
            out[k] = n
            continue
        v = v if isinstance(v, bool) else str(v).lower() in ("1", "true", "yes", "on")
        if v:
            out[k] = True
    return out


def _ms(t0):
    return round((time.perf_counter() - t0) * 1000.0, 1)


class TreeFactVerify(Approach):
    name = "AGENTIC2"
    needs_gold = False

    def __init__(self, cfg=None, client=None, flags=None, options=None):
        super().__init__(cfg=cfg, client=client, flags=parse_flags(flags))
        env = os.environ.get(OPTIONS_ENV)
        self.env_options = json.loads(env) if env else None
        self.opts = dict(DEFAULTS, **(self.env_options or {}), **(options or {}))
        f = self.flags
        if f.get("candidates"):
            self.opts["max_candidates"] = self.opts["max_candidates_multi"] = int(f["candidates"])
        self.use_vectors = not f.get("no_vectors")
        self.use_factdb = not f.get("no_factdb")
        self.use_tree = not f.get("no_tree")
        self.use_verify = not f.get("no_verify")
        self.line_windows = False
        self.doc_selection = True
        caps = self.cfg.caps or {}
        self.max_calls = int(caps.get("agentic2_max_calls") or self.opts["max_calls_default"])
        self.max_seconds = float(caps.get("agentic2_max_seconds") or self.opts["max_seconds_default"])
        self._cache = None
        self._cache_for = None
        self.lazy = LazyVectors(self)
        self.factdb = FactDB(self.cfg, cache_path=self.opts.get("factdb_cache_path"))

    def config_signature(self):
        sig = {"lazy": True, "tree": TREE_VERSION, "factdb": FACTDB_VERSION}
        if not self.use_vectors:
            sig.update(store="plain", store_format=STORE_FORMAT, normalized=True)
        return sig

    def _build_index(self, corpus_set, index_dir):
        """No index build: vectors and fact rows are made lazily per targeted document. Only the no_vectors ablation
        writes the v1 plain store (seconds, CPU) for ripgrep."""
        out = {"store_root": None, "lazy": True, "n_docs": len(corpus_set.docs)}
        if not self.use_vectors:
            t0 = time.perf_counter()
            root = build_plain_store(corpus_set, root=Path(index_dir) / "store", cfg=self.cfg)
            man = read_json(Path(root) / "manifest.json", {}) or {}
            try:
                grep(root, "@@AGENTIC-WARM-UP-NO-SUCH-LINE@@", fixed_strings=True, timeout_s=600, cfg=self.cfg)
            except Exception:  # noqa: BLE001
                pass
            out.update(store_root=str(root), n_files=man.get("n_files", 0), store_ms=_ms(t0))
        return out

    def _load_index(self, corpus_set, index_dir):
        return self._build_index(corpus_set, index_dir)

    def call_cache(self):
        path = self.opts.get("call_cache_path")
        if not path and self.opts.get("call_cache", "run") == "run" and getattr(self.client.ledger, "path", None):
            path = Path(self.client.ledger.path).parent / ("agentic_calls_%s.sqlite" % safe_name(self.variant))
        if self.opts.get("call_cache") == "off":
            path = None
        if self._cache is None or str(path) != str(self._cache_for):
            self._cache = CallCache(path)
            self._cache_for = path
        return self._cache

    def _answer(self, query, index):
        return _Session2(self, query, index).run()


# --------------------------------------------------------------------------
# one question
# --------------------------------------------------------------------------

class _Session2(_Session):
    def __init__(self, ap, query, index):
        super().__init__(ap, query, index)
        self.stats.update({"lazy_docs_embedded": 0, "lazy_passages_embedded": 0, "lazy_embed_calls": 0,
                           "lazy_index_ms": 0.0, "query_embed_calls": 0, "vec_hits": 0, "factdb_rows": 0,
                           "factdb_matched": 0, "cand_vector": 0, "cand_factdb": 0, "cand_both": 0, "cand_grep": 0,
                           "candidates_pre_cap": 0, "verify_batches_planned": 0, "time_skipped_batches": 0,
                           "time_skipped_candidates": 0, "fill_items": 0, "query_clock_ms": 0.0, "over_time": 0,
                           "thousands_fixed": 0, "label_completed": 0})

    # -- the query clock (lazy-index excluded; replayed calls count with their recorded wall time) --------------------
    def clock_s(self):
        total = 0.0
        for st in self.timing:
            if st["stage"] == "lazy-index":
                continue
            calls_wall = sum(c["wall_ms"] for c in self.calls if c.get("stage") == st["stage"])
            total += st["wall_ms"] - st.get("call_elapsed_ms", 0.0) + calls_wall
        return total / 1000.0

    def left_s(self):
        return self.a.max_seconds - self.clock_s()

    # -- retrieval ------------------------------------------------------------------------------------------------------
    def lazy_index(self, docs):
        infos = []
        for d in docs:
            _ps, _m, info = self.a.lazy.ensure(d, self.o, self.a.client)
            infos.append(info)
            if info.get("embedded_now"):
                self.stats["lazy_docs_embedded"] += 1
                self.stats["lazy_passages_embedded"] += info["embedded_now"]
                self.stats["lazy_embed_calls"] += info["embed_calls"]
        return infos

    def terms_text(self, plan):
        terms = plan.get("terms") or {}
        d, other = self.q.d_lang, ("en" if self.q.d_lang == "fa" else "fa")
        parts = [plan.get("entity") or "", plan.get("metric") or ""] + list(terms.get(d) or []) + list(terms.get(other) or [])
        return " ; ".join(p for p in parts if p).strip()

    def vector_hits(self, plan, docs):
        o = self.o
        texts = [self.q.text]
        tt = self.terms_text(plan) if o["vec_terms_query"] else ""
        if tt:
            texts.append(tt)
        qv, qinfo = self.a.lazy.query_vectors(texts, self.a.client)
        self.stats["query_embed_calls"] += qinfo["embed_calls"]
        w = float(o["vec_terms_weight"]) if len(qv) > 1 else 0.0
        k = int(o["vec_top_k_single"] if len(docs) == 1 else o["vec_top_k_per_doc"])
        hits, unit_sim, per_doc = [], {}, {}
        for d in docs:
            ps, m, _info = self.a.lazy.ensure(d, self.o, self.a.client)
            if not len(ps):
                continue
            sims = m @ qv[0]
            if w:
                sims = (1.0 - w) * sims + w * (m @ qv[1])
            for p, s in zip(ps, sims.tolist()):
                for u in p.unit_ids:
                    if s > unit_sim.get(u, -1.0):
                        unit_sim[u] = s
            order = np.argsort(-sims, kind="stable")[:k]
            per_doc[d.id] = [[ps[i].pid, round(float(sims[i]), 4)] for i in order]
            hits += [(d, ps[i], float(sims[i])) for i in order]
        return hits, unit_sim, {"query": qinfo, "k": k, "terms_text": tt, "terms_weight": w, "top": per_doc}

    def candidates(self, plan, chosen, meta_by):
        o = self.o
        corpus = self.corpus
        docs = [corpus.doc(m["doc_id"]) for m in chosen if corpus.has_doc(m["doc_id"])]
        unit_kind = plan.get("unit") or "count"
        ctx = T.search_context(plan, self.q.d_lang)
        info = {}
        cands, unit_sim = [], {}
        t0 = time.perf_counter()
        if self.a.use_vectors:
            hits, unit_sim, vinfo = self.vector_hits(plan, docs)
            info["vector"] = vinfo
            info["vector_ms"] = _ms(t0)
            self.stats["vec_hits"] = len(hits)
            t1 = time.perf_counter()
            for d, p, s in hits:
                for c in T.expand_passage(d, p, ctx, o, unit_kind, tree=self.a.use_tree):
                    c.score = s
                    c.features = {"vec": round(s, 4), "pid": p.pid, "src": ["vector"]}
                    cands.append(c)
            info["expand_vector_ms"] = _ms(t1)
        else:
            gc, ginfo = S.search(self.root, corpus, plan, chosen, d_lang=self.q.d_lang, question=self.q.text, opts=o,
                                 round_no=1, cfg=self.cfg)
            for c in gc:          # v1 lexical score (about 0-15) mapped to the vector scale
                c.features = dict(c.features or {}, src=["grep"], grep_score=c.score)
                c.score = max(0.0, min(1.0, c.score / 15.0))
                cands.append(c)
            info["grep"] = {k: ginfo.get(k) for k in ("patterns", "hit_lines", "windows", "kept", "selected", "rg_ms")}
            info["grep_ms"] = _ms(t0)
            self.stats["cand_grep"] = len(gc)
        fact_rows = []
        if self.a.use_factdb:
            t2 = time.perf_counter()
            rows, builds = [], []
            for d in docs:
                rs, binfo = self.a.factdb.rows_for(d, (meta_by.get(d.id) or {}).get("years"))
                rows += rs
                builds.append(binfo)
            con = self.a.factdb.table(rows)
            period = plan.get("period") if plan.get("period") and any(x is not None for x in plan["period"]) else None
            per_doc = int(o["factdb_max_rows"]) if len(docs) == 1 else int(o["factdb_rows_per_doc"])
            fact_rows, qinfo = self.a.factdb.query(con, plan, self.q.d_lang, [d.id for d in docs], period=period,
                                                   max_rows=int(o["factdb_max_rows"]), per_doc=per_doc)
            con.close()
            self.stats["factdb_rows"] = len(rows)
            self.stats["factdb_matched"] = qinfo.get("matched", 0)
            floor = min(unit_sim.values()) if unit_sim else 0.0
            for r in fact_rows:
                d = corpus.doc(r["doc_id"])
                c = T.expand_unit(d, r["unit_id"], r["value"], ctx, o, unit_kind, tree=self.a.use_tree)
                if c is None:
                    continue
                s = max([unit_sim.get(u, floor) for u in c.unit_ids] or [floor])
                c.score = s
                c.features = {"vec": round(s, 4), "src": ["factdb"], "fact": {"label": r["label"], "value": r["value"],
                                                                               "year": r["year"]}}
                cands.append(c)
            info["factdb"] = dict(qinfo, builds=builds, rows=len(rows), ms=_ms(t2),
                                  kept_rows=[{k: r[k] for k in ("doc_id", "year", "label", "value", "unit", "unit_id",
                                                               "matched_terms")} for r in fact_rows[:30]])
        t3 = time.perf_counter()
        ranked = self.rank(cands, fact_rows, ctx, unit_kind)
        self.stats["candidates_pre_cap"] = len(ranked)
        multi = len(docs) > 1
        cap = int(o["max_candidates_multi"] if multi else o["max_candidates"])
        sel = self.select(ranked, [d.id for d in docs], cap) if multi else ranked[:cap]
        for i, c in enumerate(sel, 1):
            c.cid = "c%d" % i
            c.verdict = None
            src = set(c.features.get("src") or [])
            self.stats["cand_vector"] += int("vector" in src and "factdb" not in src)
            self.stats["cand_factdb"] += int("factdb" in src and "vector" not in src)
            self.stats["cand_both"] += int("vector" in src and "factdb" in src)
        info.update(rank_ms=_ms(t3), cap=cap, pre_cap=len(ranked), selected=len(sel), multi_doc=multi)
        return sel, info

    def rank(self, cands, fact_rows, ctx, unit_kind):
        """Dedupe (a candidate whose units lie inside a kept one of the same section adds nothing) and score."""
        o = self.o
        fact_units = {}
        for r in fact_rows:
            fact_units.setdefault(r["unit_id"], []).append(r["value"])
        for c in cands:
            bonus = 0.0
            nums = candidate_numbers(find_numbers(c.text), unit_kind) if unit_kind != "text" else []
            if nums and (unit_kind in ("text", None) or any(unit_kind_match(c.text, t, unit_kind) for t in nums)
                         or unit_kind in ("quantity", "ratio")):
                bonus += float(o["number_bonus"])
                c.features["number"] = 1
            if any(u in fact_units and any(v in c.text for v in fact_units[u]) for u in c.unit_ids):
                bonus += float(o["factdb_bonus"])
                c.features["factdb_match"] = 1
            if ctx is not None and ctx.has_term(c.text):
                bonus += float(o["term_bonus"])
                c.features["term"] = 1
            c.features["bonus"] = round(bonus, 3)
            c.score = round(float(c.score) + bonus, 4)
            c.number_unit_ids = [u for u in c.unit_ids]
        def same_text(a, b):
            """b adds nothing to a: its units lie inside a's (same section) and, for clips of one long unit, its span
            overlaps a's by at least half"""
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
        # best first; on a tie the wider window first, so the narrower windows inside it are dropped
        for c in sorted(cands, key=lambda c: (-round(c.score, 3), -len(c.unit_ids), c.doc_id, c.unit_ids)):
            dup = next((k for k in kept if same_text(k, c)), None)
            if dup is not None:
                dup.features["src"] = sorted(set(dup.features.get("src") or []) | set(c.features.get("src") or []))
                continue
            wider = [k for k in kept if same_text(c, k) and k.score <= c.score + 0.02]
            for k in wider:                  # a kept narrower window inside this one (within 0.02 of its score)
                c.features["src"] = sorted(set(c.features.get("src") or []) | set(k.features.get("src") or []))
                kept.remove(k)
            kept.append(c)
        kept.sort(key=lambda c: (-c.score, c.doc_id, c.unit_ids))
        return kept

    @staticmethod
    def select(ranked, doc_order, cap):
        """Several documents: round robin in year order (each period gets its best candidates first)."""
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
        return out

    # -- verification: the v1 verifier + guard, with two tree-aware repairs ---------------------------------------------
    def verify_batch(self, batch, plan, purpose, rows, meta_by, batch_no):
        """v1 verify_batch (same prompt, schema, guard, re-attribution) plus:
          * thousands_fix: a Persian thousands separator written as the Arabic comma ("2،788،550") is shown to the
            verifier and the guard as "2,788,550" (the v1 number reader splits it into three numbers); the candidate
            is then no longer a verbatim slice (span dropped);
          * label completion: a value whose quote is only the number (guard quote_no_label; an infographic label
            extracted on its own line) is re-checked by the SAME guard with the nearest label line of the snippet
            added to the quote (previous line first, then next): the verbatim, value and metric-term checks all still
            apply."""
        fixed = []
        if self.o.get("thousands_fix", True):
            for c in batch:
                t = _AR_THOUSANDS.sub(",", c.text)
                if t != c.text:
                    c.text, c.span = t, None
                    fixed.append(c.cid)
        out = super().verify_batch(batch, plan, purpose, rows, meta_by, batch_no)
        out["thousands_fixed"] = fixed
        self.stats["thousands_fixed"] += len(fixed)
        if self.o.get("label_completion", True):
            self.complete_labels(out, batch, plan, rows, meta_by)
        return out

    def complete_labels(self, out, batch, plan, rows, meta_by):
        from ..agentic import verify as V
        unit_kind = plan.get("unit") or "count"
        term_rx = self.metric_term_rx(plan) if (self.o["quote_term_guard"] and unit_kind != "text") else None
        by_cid = {c.cid: c for c in batch}
        done = 0
        for rec in out["items"]:
            if rec.get("verdict") != "guard:quote_no_label":
                continue
            c = by_cid.get(rec["cid"])
            if c is None:
                continue
            lines = c.text.split("\n")
            cq = V.canon(rec.get("quote") or "")
            idx = next((i for i, ln in enumerate(lines) if cq and cq in V.canon(ln)), None)
            if idx is None:
                continue
            labels = []
            for j in (idx - 1, idx - 2, idx + 1, idx + 2):
                if 0 <= j < len(lines) and lines[j].strip() and T._label_only(lines[j], unit_kind):
                    labels.append(j)
            for j in labels[:2]:
                quote = (lines[j] + "\n" + lines[idx]) if j < idx else (lines[idx] + "\n" + lines[j])
                it = {k: rec.get(k) for k in ("value", "unit", "period", "definition")}
                it["quote"] = quote
                ok, reason, ginfo = V.guard(it, c.text, unit_kind=unit_kind,
                                            min_quote_letters=int(self.o["min_quote_letters"]), term_rx=term_rx)
                if not ok:
                    continue
                if any(r["cid"] == c.cid and r.get("value_num") is not None and ginfo.get("value_num") is not None
                       and abs(r["value_num"] - ginfo["value_num"]) < 1e-9 for r in rows):
                    break
                uid = V.value_unit_id(c, self.corpus, ginfo.get("value_token"), quote)
                row = R.make_row(it, c, ginfo, meta_by.get(c.doc_id, {}), uid, verified=True)
                row["label_completed"] = True
                rows.append(row)
                rec.update(verdict="accepted", guard="ok_label_completed", completed_quote=quote,
                           value_num=ginfo.get("value_num"), row_period=row["period"], def_class=row["def_class"],
                           role=row["role"])
                c.verdict = "accepted"
                out["accepted"] += 1
                out["guard_rejected"] -= 1
                self.stats["accepted"] += 1
                self.stats["guard_rejected"] -= 1
                out["guard_reasons"]["quote_no_label"] = out["guard_reasons"].get("quote_no_label", 1) - 1
                self.guard_reasons["quote_no_label"] = self.guard_reasons.get("quote_no_label", 1) - 1
                done += 1
                break
        out["label_completed"] = done
        self.stats["label_completed"] += done

    # -- evidence -----------------------------------------------------------------------------------------------------
    def evidence_items(self, rec, cands, verified):
        o = self.o
        if not self.a.use_verify:
            return [c.to_item(rank=i, meta={"verified": None}) for i, c in enumerate(cands, 1)], "ranked_candidates"
        if not rec["series"]:
            items, ev_from = [], "unverified_candidates"
            pool = cands if o["fill_unverified"] else cands[: int(o["unverified_fallback_items"])]
            self.stats["unverified_fallback"] = 1
            return [c.to_item(rank=i, meta={"verified": False}) for i, c in enumerate(pool, 1)], ev_from
        items, ev_from = self.evidence(rec, verified)
        if o["fill_unverified"]:
            have = {it.meta.get("cid") for it in items}
            extra = [c for c in cands if c.cid not in have]
            items += [c.to_item(meta={"verified": getattr(c, "verdict", None) == "accepted"}) for c in extra]
            self.stats["fill_items"] = len(extra)
        for i, it in enumerate(items):
            it.rank = i + 1
        return items, ev_from

    # -- the pipeline ------------------------------------------------------------------------------------------------------
    def run(self):
        t_all = time.perf_counter()
        o = self.o
        trace = {"variant": self.a.variant, "max_calls": self.budget.max, "max_seconds": self.a.max_seconds,
                 "prompts": {"plan": PLAN_PROMPT_VERSION, "verify": VERIFY_PROMPT_VERSION, "answer": "shared"},
                 "versions": {"tree": TREE_VERSION, "factdb": FACTDB_VERSION, "rank": RANK_VERSION},
                 "params": {k: o[k] for k in sorted(o) if not k.startswith("call_cache") and not k.endswith("_path")},
                 "options_env": self.a.env_options, "call_cache": self.cache.path, "stages": {},
                 "flags": dict(self.a.flags)}
        scope_docs = self.a.docs_in_scope(self.q, self.index)
        metas = [doc_meta(d) for d in scope_docs]
        meta_by = {m["doc_id"]: m for m in metas}
        self.stats["docs_scope"] = len(metas)
        with self.a.client.ledger.scope() as whole:
            # 1. plan
            with self.stage("plan") as st:
                plan = self.plan(metas)
                period = plan["period"] if any(p is not None for p in plan["period"]) else None
                chosen, sel = select_docs(metas, companies_chosen=plan["companies"], period=period, kind=plan["kind"],
                                          question=self.q.text, max_docs=int(o["max_target_docs"]), selection=True)
                self.stats["docs_targeted"] = len(chosen)
                st.update(docs_scope=len(metas), docs_targeted=len(chosen), kind=plan["kind"],
                          repair=self.stats["plan_repair"], program_plan=self.stats["plan_program"])
            trace["stages"]["plan"] = {k: plan[k] for k in ("entity", "metric", "kind", "unit", "terms", "entity_terms",
                                                             "period", "companies", "dropped_terms", "fixes", "model")}
            trace["stages"]["plan"]["doc_selection"] = sel
            docs = [self.corpus.doc(m["doc_id"]) for m in chosen if self.corpus.has_doc(m["doc_id"])]

            # 2a. lazy index (first touch of a document; preprocessing, not on the query clock)
            with self.stage("lazy-index") as st:
                infos = self.lazy_index(docs) if self.a.use_vectors else []
                st.update(docs=len(infos), embedded_docs=self.stats["lazy_docs_embedded"],
                          passages=sum(i["passages"] for i in infos),
                          embedded_passages=self.stats["lazy_passages_embedded"],
                          embed_calls=self.stats["lazy_embed_calls"],
                          embed_tokens=sum(i.get("embed_tokens_now") or 0 for i in infos),
                          embed_ms=round(sum(i.get("embed_ms_now") or 0 for i in infos), 1),
                          cached_cost_ms=round(sum(i.get("cached_cost_ms") or 0 for i in infos), 1))
            trace["stages"]["lazy_index"] = infos

            # 2b-e. retrieve (no LLM)
            with self.stage("retrieve") as st:
                cands, rinfo = self.candidates(plan, chosen, meta_by)
                st.update(candidates=len(cands), pre_cap=rinfo.get("pre_cap"), vec_hits=self.stats["vec_hits"],
                          factdb_rows=self.stats["factdb_rows"], factdb_matched=self.stats["factdb_matched"],
                          cand_vector=self.stats["cand_vector"], cand_factdb=self.stats["cand_factdb"],
                          cand_both=self.stats["cand_both"], embed_calls=self.stats["query_embed_calls"])
            self.stats["candidates"] = len(cands)
            rinfo["candidates"] = [c.trace(int(o["trace_text_chars"])) for c in cands]
            trace["stages"]["retrieve"] = rinfo

            # 3. verify (1 call, at most 2) within the call cap and the query clock
            rows, logs, skipped, time_skipped = [], [], [], []
            with self.stage("verify") as st:
                if self.a.use_verify and cands:
                    n_calls = min(int(o["max_verify_calls"]), max(0, self.budget.left() - 1))
                    n_batches = min(n_calls, math.ceil(len(cands) / float(o["verify_batch_max"])))
                    if n_batches >= 1:
                        size = math.ceil(len(cands) / float(n_batches))
                        batches = [cands[i:i + size] for i in range(0, len(cands), size)][:n_batches]
                        rest = cands[size * n_batches:]
                        skipped += rest
                        self.stats["verify_batches_planned"] = len(batches)
                        # the approach never reads the query's answer format (fairness rule): a multi-period plan reserves the
                        # longer answer time
                        reserve_s = float(o["answer_reserve_timeline_s"] if plan["kind"] != "single"
                                          else o["answer_reserve_s"])
                        in_stage_s = 0.0      # finished batches of this stage (recorded call wall + program time)
                        for k, b in enumerate(batches):
                            est = float(o["verify_s_fixed"]) + float(o["verify_s_per_candidate"]) * len(b)
                            if not self.budget.can(1, reserve=1):
                                skipped += b
                                continue
                            if self.left_s() - in_stage_s < est + reserve_s:
                                time_skipped += b
                                self.stats["time_skipped_batches"] += 1
                                continue
                            lg = self.verify_batch(b, plan, "verify", rows, meta_by, k + 1)
                            lg["estimate_s"] = round(est, 1)
                            lg["clock_before_s"] = round(self.clock_s() + in_stage_s, 2)
                            in_stage_s += (float(lg.get("llm_wall_ms") or 0.0) + float(lg.get("prompt_build_ms") or 0.0)
                                           + float(lg.get("parse_guard_ms") or 0.0)) / 1000.0
                            logs.append(lg)
                    else:
                        skipped += cands
                for c in skipped + time_skipped:
                    c.verdict = "not_verified_budget" if c in skipped else "not_verified_time"
                self.stats["candidates_unverified_budget"] = len(skipped)
                self.stats["time_skipped_candidates"] = len(time_skipped)
                st.update(batches=len(logs), candidates=len(cands), verified=sum(b["n"] for b in logs),
                          skipped_for_budget=len(skipped), skipped_for_time=len(time_skipped),
                          accepted=sum(b["accepted"] for b in logs),
                          model_rejected=sum(b["model_rejected"] for b in logs),
                          model_omitted=sum(b["model_omitted"] for b in logs),
                          reattributed=sum(b["reattributed"] for b in logs),
                          guard_rejected=sum(b["guard_rejected"] for b in logs),
                          unparsed=sum(b["unparsed"] for b in logs))
            trace["stages"]["verify"] = {"batches": logs, "skipped_for_budget": [c.cid for c in skipped],
                                         "skipped_for_time": [c.cid for c in time_skipped]}
            verified = [c for c in cands if getattr(c, "verdict", None) not in (None, "not_verified_budget",
                                                                                 "not_verified_time")]

            # 4a. reconcile (code) + evidence
            with self.stage("reconcile") as st:
                target_periods = []
                if plan["kind"] != "single":
                    p0, p1 = plan["period"]
                    for m in chosen:
                        if m["years"]:
                            for y in range(m["years"][0], m["years"][1] + 1):
                                if (p0 is None or y >= p0) and (p1 is None or y <= p1) and y not in target_periods:
                                    target_periods.append(y)
                    target_periods.sort()
                rec = R.reconcile(rows, kind=plan["kind"], target_periods=target_periods, opts=o, q_lang=self.q.q_lang,
                                  known_stems=self.known_stems(plan))
                items, ev_from = self.evidence_items(rec, cands, verified)
                evidence = self.a.pack(items, self.index)
                self.stats.update(rows=len(rec["series"]), periods_found=len(rec["periods_found"]),
                                  periods_missing=len(rec["periods_missing"]), conflicts=len(rec["conflicts"]),
                                  comparable=(-1 if rec["comparable"] is None else int(bool(rec["comparable"]))))
                st.update(rows_accepted=len(rows), series_rows=len(rec["series"]), restatements=len(rec["restatements"]),
                          conflicts=len(rec["conflicts"]), periods_found=len(rec["periods_found"]),
                          periods_missing=len(rec["periods_missing"]), evidence_from=ev_from,
                          evidence_offered=len(items), evidence_items=evidence.n_included, evidence_chars=evidence.chars)

            # 4b. the answer call (always kept)
            with self.stage("answer") as st:
                if not self.budget.can(1):
                    raise BudgetExceeded("AGENTIC2: no call left for the answer")
                answer = self.answer_call(evidence)
                st.update(answer_chars=len(answer))
            whole_summary = whole.summary()
        total_ms = _ms(t_all)
        self.stats["query_clock_ms"] = round(self.clock_s() * 1000.0, 1)
        self.stats["over_time"] = int(self.clock_s() > self.a.max_seconds)
        self.stats["lazy_index_ms"] = next((s["wall_ms"] for s in self.timing if s["stage"] == "lazy-index"), 0.0)
        structured = {
            "kind": plan["kind"], "entity": plan["entity"], "metric": plan["metric"], "unit_kind": plan["unit"],
            "series": [_series_row(r) for r in rec["series"]],
            "comparable": rec["comparable"], "main_class": rec["main_class"], "definition_classes": rec["definition_classes"],
            "trend": rec["trend"], "notes": R.table_notes(rec, self.q.q_lang) if rec["series"] else [],
            "periods_found": rec["periods_found"], "periods_missing": rec["periods_missing"],
            "target_periods": [str(p) for p in target_periods], "conflicts": rec["conflicts"],
            "restatements": [_series_row(r) for r in rec["restatements"]],
            "narrower": [_series_row(r) for r in rec["narrower"]],
            "projections": [_series_row(r) for r in rec["projections"]],
            "rows_all": [_series_row(r) for r in rows], "evidence_from": ev_from,
        }
        trace["stages"]["answer"] = {"evidence_from": ev_from,
                                     "table": items[0].text if (items and ev_from == "verified_table") else None,
                                     "n_offered": evidence.n_offered, "n_included": evidence.n_included,
                                     "chars": evidence.chars}
        trace["structured"] = structured
        trace["guard_reasons"] = self.guard_reasons
        trace["calls"] = self.calls
        trace["calls_by_purpose"] = {p: self.budget.log.count(p) for p in dict.fromkeys(self.budget.log)}
        trace["timing"] = self._timing(whole_summary, total_ms)
        trace["timing"]["query_clock_ms"] = self.stats["query_clock_ms"]
        trace["timing"]["lazy_index_ms"] = self.stats["lazy_index_ms"]
        trace["ledger_calls"] = whole_summary["llm_calls"]
        self.stats["program_ms"] = trace["timing"]["totals"]["program_ms"]
        trace["stats"] = self.stats
        order = {c.cid: i for i, c in enumerate(cands)}
        ranked = sorted(cands, key=lambda c: (getattr(c, "verdict", None) != "accepted", order.get(c.cid, 0)))
        retrieved = [c.to_item(rank=i, meta={"verdict": getattr(c, "verdict", None), "src": c.features.get("src")})
                     for i, c in enumerate(ranked, 1)]
        return AnswerResult(answer_text=answer, evidence=evidence, retrieved=retrieved, trace=trace,
                            structured=structured)
