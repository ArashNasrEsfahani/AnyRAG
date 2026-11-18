"""LEX11-M (2026-09-26, work/datasets_v2/lex11/PROGRESS.md; spec: work/analysis/LEX11_ASTRA_R1.log steps 1-6).

    1. retrieve once: authorized pool scope; one bge-m3 query encoding (dense + sparse); BM25 + bge-m3 dense + sparse,
       top 40 valid chunks per channel, union U (lex8 chunk channels, identity-checked artifacts)
    2. F2 MaxMean exactly as lex8/fusion.py; the first 40
    3. the ordinary chunk packer (pack_evidence) on the F2 top 40 at the cell budget -> membership + displayed text
       FROZEN (clipped last item included)
    4. reorder only the frozen items by HYBRID's H(c) (lex11/order.py); HYBRID's own BM25 / dense ranks come from
       HybridIndex.search (k=80 so every channel-top-40 member keeps its rank); re-render; assert identical
       membership, displayed content and chars
    5. the unchanged shared answer call (one call)
    6. trace: channel lists, F2 scores of the pack, pre/post order, H, ranks, intervals, chars, stage times

Cells (run with --approach-class; not in the shared registry):
    Lex11M6000   LEX11-M at 6,000 rendered chars
    Lex11M15500  LEX11-M at 15,500 rendered chars
    F2At6000     the F2 order alone at 6,000 (LEX10's pipeline at 6,000; steps 1-3 + 5, no reorder, no HYBRID search)
"""
import time

from ...contracts import pack_evidence
from ...normalize import normalize
from ..lex10.approach import DEPTH, Lex10
from ..lex8 import fusion as FU
from ..lex8 import retrieve as RV
from ..lex8.approach import Lex8
from . import order as OR

LEX11_VERSION = "lex11m-v1"


class _Lex11Base(Lex10):
    budget_chars = 6000
    reorder = True

    def config_signature(self):
        sig = dict(super().config_signature())
        sig.pop("lex10", None)
        sig["lex11"] = {"version": LEX11_VERSION, "order": OR.ORDER_VERSION if self.reorder else "f2",
                        "budget_chars": self.budget_chars, "depth": DEPTH, "fusion": "maxmean",
                        "packer": "pack_evidence (chunk), membership frozen before reorder",
                        "h_ranks": "HybridIndex.search bm25 + dense (k=80, depth=40)" if self.reorder else None}
        return sig

    def _answer(self, query, index):
        T0 = time.perf_counter()
        six = self._six
        B = int(self.budget_chars)
        st = {"version": LEX11_VERSION, "cell": self.name, "device": getattr(Lex8._enc, "device", None), "budget": B}
        doc_ids = None if query.doc_scope in (None, "corpus") else list(query.doc_scope)
        t = time.perf_counter()
        q = Lex8._enc.encode([normalize(query.text)], max_length=128, colbert=False)[0]
        st["t_encode_ms"] = round((time.perf_counter() - t) * 1000.0, 1)
        t = time.perf_counter()
        mask, _ = six.hx.scope_mask(doc_ids)
        arrays, valid = RV.chunk_channels(six, query.text, q, mask)
        lists = {ch: FU.top_ids(arr, valid[ch], DEPTH) for ch, arr in arrays.items()}
        forder, fsc, _N = FU.fuse(arrays, lists, "maxmean")
        order = [int(c) for c in forder[:DEPTH]]
        st["t_channels_ms"] = round((time.perf_counter() - t) * 1000.0, 1)
        t = time.perf_counter()
        items = [six.hx.hit_for_chunk(c).to_item(rank=i) for i, c in enumerate(order)]
        frozen = pack_evidence(items, index.corpus, budget=B, cfg=self.cfg)
        st["t_pack_ms"] = round((time.perf_counter() - t) * 1000.0, 1)
        ev, info = frozen, None
        if self.reorder:
            t = time.perf_counter()
            hits = six.hx.search(query.text, doc_ids=doc_ids, k=2 * DEPTH, depth=DEPTH, client=self.client)
            bm = [int(h.chunk) for h in sorted((h for h in hits if h.ranks.get("bm25")), key=lambda h: h.ranks["bm25"])]
            dn = [int(h.chunk) for h in sorted((h for h in hits if h.ranks.get("dense")), key=lambda h: h.ranks["dense"])]
            st["t_hybrid_ranks_ms"] = round((time.perf_counter() - t) * 1000.0, 1)
            t = time.perf_counter()
            ev, info = OR.lex11_pack(frozen, OR.channel_ranks(bm, dn), order, index.corpus)
            st["t_reorder_ms"] = round((time.perf_counter() - t) * 1000.0, 1)
            st["hybrid_lists"] = {"bm25": bm, "dense": dn}
            st["hybrid_dense_complete"] = (hits.diagnostics.get("dense") or {}).get("complete")
        st["t_retrieval_ms"] = round((time.perf_counter() - T0) * 1000.0, 1)
        t = time.perf_counter()
        answer = self.generate_answer(query, ev)
        st["t_answer_ms"] = round((time.perf_counter() - t) * 1000.0, 1)
        st.update({"channels": {k: [int(x) for x in v] for k, v in lists.items()}, "order40": order,
                   "f2_scores_pack": {str(int(it.meta["chunk"])): round(float(fsc[int(it.meta["chunk"])]), 5)
                                      for it in frozen.items},
                   "intervals": [[int(it.meta["chunk"]), it.doc_id, list(it.span) if it.span else None,
                                  len((it.text or "").strip()), bool((it.meta or {}).get("cut_from_chars"))]
                                 for it in ev.items],
                   "reorder": info, "evidence_items": ev.n_included, "evidence_chars": ev.chars,
                   "frozen_chars": frozen.chars})
        trace = {"lex11": st, "retrieval_ms": st["t_retrieval_ms"], "chunks_total": six.hx.n,
                 "evidence_items": ev.n_included, "last_item_cut": ev.last_item_cut}
        return self._result(answer, ev, items, trace)


class Lex11M6000(_Lex11Base):
    name = "LEX11M_6000"
    budget_chars = 6000


class Lex11M15500(_Lex11Base):
    name = "LEX11M_15500"
    budget_chars = 15500


class F2At6000(_Lex11Base):
    name = "F2_6000"
    budget_chars = 6000
    reorder = False
