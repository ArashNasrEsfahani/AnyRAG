"""LEX6 approach: document-aware local retrieval with bounded evidence replacement (core.py; spec
work/analysis/LEX6_DESIGN_FINAL.md = Astra R2 section 5).

    --approach lex6                          the main arm: S2+S5 document ranking, top-3 nominations, fresh local
                                             retrieval, package-aware slots, 3,000-char anchor, bounded exact packing,
                                             the shared GLM answer (no online LLM call besides the answer)
    --ablation rank=s2 | rank=s5             one document channel only (V chunk prior | centroid)
    --ablation absent_only                   ABSENT-ONLY: reserve only for nominated documents absent from B0
    --ablation global_only                   alternatives from the global candidates only (no fresh local search)
    --ablation no_anchor                     packing ablation: no baseline anchor (the 3,000 addition cap stays)
    --ablation cards                         CARDS: half-weight card BM25 + card dense in F, card nominations
    --ablation sections                      SECTIONS: <= 2 section cards per nominated doc -> optional candidates
    --ablation router                        ROUTER: the optional bounded routing call (router.py)
    --ablation select                        LEX6+S: the LEX5 selector over the nominated documents' local packages
                                             (a labelled ablation: its selector call cannot fit the token gate)
Flags combine (e.g. cards,sections -> LEX6+S in the eval is 'LEX6+cards,sections'). Card-dependent flags need
work/datasets_v2/lex6/cards/ (cardsrc.py); without cards for the scope they fall back to the main arm and say so.
trace['lex6'] records the document ranking, shortlist, nominations, router input/output/status, every slot's
alternatives (ranks, chars, provenance), B0, the anchor, the reservations, the displaced B0 packages, the final pack
with a reason per package, the provenance of every displayed interval, and each stage's wall time.
"""
import time

from ...retrieval import hybrid as H
from ..hybrid import HybridRetrieval
from ..lex4 import core as K
from ..lex5 import bundles as B
from ..lex5.approach import PROMPT as SELECT_PROMPT, _packager, freeze
from . import core as K6

LEX6_VERSION = "lex6-v1"
LEX6_DEFAULTS = {"depth": 40, "k": 80, "budget": 6000, "select_per_doc": 8, "select_top": 24,
                 "select_deadline_s": 3.0, "select_max_tokens": 1024}
_STORE = {}


def card_store():
    if "s" not in _STORE:
        from .cardsrc import CardStore
        _STORE["s"] = CardStore()
    return _STORE["s"]


class Lex6(HybridRetrieval):
    name = "LEX6"
    index_neutral_flags = ("rank", "absent_only", "global_only", "no_anchor", "cards", "sections", "router",
                           "select")

    def config_signature(self):
        from . import router as RT
        return {"hybrid": H.signature(self.cfg), "lex6": LEX6_DEFAULTS, "core": K6.C, "version": LEX6_VERSION,
                "core_version": K6.LEX6_CORE_VERSION, "router": RT.R, "lex4": K.CORE_VERSION,
                "bundles": B.BUNDLES_VERSION}

    def _prep(self, hx):
        from ..lex6 import docrank
        self._cents = docrank.centroids(hx, list(hx.doc_ids))          # outside the timed query
        fl = self.flags or {}
        if fl.get("cards") or fl.get("sections") or fl.get("router"):
            card_store()
        if fl.get("router"):
            try:
                self._router_client().warm_deadline()
            except Exception:  # noqa: BLE001
                pass
        if fl.get("select"):
            try:
                self._selector().warm_deadline()
            except Exception:  # noqa: BLE001
                pass

    def _router_client(self):
        if getattr(self, "_rc", None) is None:
            from . import router as RT
            self._rc = RT.client(self.cfg, self.client.ledger)
        return self._rc

    def _selector(self):
        if getattr(self, "_sel", None) is None:
            from ...llm_openrouter import OpenRouterReader, model_info
            mid = "openai/gpt-oss-20b"
            info = model_info(mid, self.cfg.project_root / "work" / "openrouter" / "catalog.json")
            self._sel = OpenRouterReader(self.cfg, self.client.ledger, mid, supported=info.get("params"), attempts=1,
                                         deadline_s=LEX6_DEFAULTS["select_deadline_s"],
                                         max_tokens_total=LEX6_DEFAULTS["select_max_tokens"], reasoning_effort="low")
        return self._sel

    def _build_index(self, corpus_set, index_dir):
        hx = super()._build_index(corpus_set, index_dir)
        self._prep(hx)
        return hx

    def _load_index(self, corpus_set, index_dir):
        hx = super()._load_index(corpus_set, index_dir)
        self._prep(hx)
        return hx

    def _answer(self, query, index):
        return run_lex6(self, query, index)


def select_stage(ap, P, qtext, out, cost, budget):
    """LEX6+S: the LEX5 selector (one call, strict grammar, exact bundles) over the nominated documents' local
    packages (<= select_per_doc per document, interleaved by local rank, <= select_top), then fill from the global
    order. -> (shown or None, trace)."""
    from ..lex5 import select as S
    from ...llm import BudgetExceeded
    per = {}
    for s in out["slots"]:
        if s["kind"] != "doc":
            continue
        lo = K6.local_order(out["ctx"], s["doc"], qtext)
        per.setdefault(s["doc"], [t[0] for t in lo[:LEX6_DEFAULTS["select_per_doc"]]])
    cands = []
    for r in range(LEX6_DEFAULTS["select_per_doc"]):
        for d, lst in per.items():
            if r < len(lst) and lst[r] not in cands:
                cands.append(lst[r])
    cands = cands[:LEX6_DEFAULTS["select_top"]]
    pk = [freeze(P.package(c)) for c in cands]
    st = {"shown": len(pk), "candidates": cands}
    if not pk:
        st.update(status="skipped", reason="no_candidates")
        return None, st
    msgs = [{"role": "user", "content": SELECT_PROMPT.format(question=qtext.strip(),
                                                             passages="\n\n".join(S.views(P, pk, qtext, "full")))}]
    t = time.perf_counter()
    text, finish = None, None
    try:
        res = ap._selector().chat(msgs, purpose="lex6_select", num_predict=LEX6_DEFAULTS["select_max_tokens"])
        text, finish = res.text, res.done_reason
        st.update(prompt_tokens=res.prompt_tokens, completion_tokens=res.completion_tokens, cost_usd=res.cost_usd,
                  finish=finish, raw=(text or "")[:1000], usage_known=True)
    except BudgetExceeded:
        raise
    except Exception as e:  # noqa: BLE001
        st.update(status="fail", reason="deadline" if "deadline" in str(e).lower() else "call_error",
                  usage_known=False)
    st["t_select_ms"] = round((time.perf_counter() - t) * 1000.0, 1)
    if text is None:
        return None, st
    pr = B.parse(text, len(pk), finish_reason=finish)
    st.update(status=pr["status"], reason=pr["reason"])
    if not pr["ok"]:
        return None, st
    parts_pk = [[tuple(pk[i - 1] for i in b) for b in alts] for alts in pr["parts"]]
    rrf_rank = {p["chunk"]: i for i, p in enumerate(out["rrf_order"], 1)}
    shown, info = B.pack_exact(parts_pk, rrf_rank, cost, budget)
    shown, finfo = B.fill(shown, parts_pk, out["rrf_order"], cost, budget, bundles=True, rrf=True)
    st.update(pack=info, fill=finfo)
    return shown, st


def run_lex6(ap, query, index):
    from ...contracts import AnswerResult
    from ...llm import BudgetExceeded
    fl = dict(ap.flags or {})
    budget = int(ap.cfg.evidence["max_chars"])
    st = {"version": LEX6_VERSION, "budget": budget, "flags": {k: v for k, v in fl.items() if v}}
    T0 = time.perf_counter()
    hx = index.payload
    docs = list(hx.doc_ids) if query.doc_scope in (None, "corpus") else list(query.doc_scope)
    # 1. question vector once, then the hybrid with it (BM25 top 40 U dense top 40, RRF)
    t = time.perf_counter()
    qv, _qi = hx.query_vector(query.text, ap.client)
    st["t_embed_ms"] = round((time.perf_counter() - t) * 1000.0, 1)
    t = time.perf_counter()
    hits = hx.search(query.text, doc_ids=None if query.doc_scope in (None, "corpus") else docs,
                     k=LEX6_DEFAULTS["k"], depth=LEX6_DEFAULTS["depth"], query_vector=qv)
    fused = [(h.chunk, h.rrf) for h in hits]
    st["t_hybrid_ms"] = round((time.perf_counter() - t) * 1000.0, 1)
    P = _packager(ap, index)
    cost = B.Coster(P.cost)
    view, notes = None, []
    if fl.get("cards") or fl.get("sections") or fl.get("router"):
        store = card_store()
        if store.ready(docs):
            view = store.view(docs)
        else:
            notes.append("cards missing for the scope: card flags inactive")
            fl = {k: v for k, v in fl.items() if k not in ("cards", "sections", "router")}
    route = None
    if fl.get("router") and view is not None:
        from . import router as RT
        route = RT.make_router(ap._router_client(), view.compact)
    t = time.perf_counter()
    out, integrity = None, None
    try:
        out = K6.plan(hx, P, freeze, cost, docs, query.text, qv, fused, getattr(ap, "_cents", {}), opts=fl,
                      cards=view, router=route)
        shown, tr = out["shown"], out["trace"]
    except BudgetExceeded:
        raise
    except Exception as e:  # noqa: BLE001  integrity failure -> B0 (HYBRID-intact), recorded
        integrity = "%s: %s" % (type(e).__name__, str(e)[:300])
        rrf_order = [freeze(P.package(c)) for c, _ in fused]
        shown = B.hybrid_intact(rrf_order, cost, budget)
        tr = {"b0": [p["chunk"] for p in shown], "final": [p["chunk"] for p in shown]}
    st["t_plan_ms"] = round((time.perf_counter() - t) * 1000.0, 1)
    if fl.get("select") and out is not None:
        out["ctx"] = out.get("ctx") or K6.QCtx(hx, qv)
        s_shown, s_tr = select_stage(ap, P, query.text, out, cost, budget)
        st["select"] = s_tr
        if s_shown is not None:
            shown = s_shown
            tr["final"] = [p["chunk"] for p in shown]
            tr["final_why"] = {p["chunk"]: "select" for p in shown}
    st["lex6"] = tr
    st["integrity_failure"] = integrity
    st["notes"] = notes
    items = [P.item(p) for p in shown]
    ev = ap.pack(items, index)
    st.update(evidence_items=ev.n_included, evidence_chars=ev.chars, cut=ev.last_item_cut, exact_cost=cost(shown),
              cost_matches=cost(shown) == ev.chars, shown_chunks=[p["chunk"] for p in shown],
              b0_chunks=tr.get("b0"), no_change=[p["chunk"] for p in shown] == tr.get("b0"),
              top_rrf=[c for c, _ in fused[:40]])
    st["t_retrieval_ms"] = round((time.perf_counter() - T0) * 1000.0, 1)
    t = time.perf_counter()
    answer = ap.generate_answer(query, ev)
    st["t_answer_ms"] = round((time.perf_counter() - t) * 1000.0, 1)
    trace = {"hybrid": hits.diagnostics, "lex6": st, "retrieval_ms": st["t_retrieval_ms"],
             "evidence_items": ev.n_included, "last_item_cut": ev.last_item_cut}
    return AnswerResult(answer_text=answer, evidence=ev, retrieved=items, trace=trace)
