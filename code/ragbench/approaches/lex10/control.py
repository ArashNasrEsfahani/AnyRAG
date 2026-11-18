"""HYBRID@15,500 (2026-09-26, work/datasets_v2/lex10/b_control/PROGRESS.md): the size-matched control for LEX10b.

    plain HYBRID fused order (BM25 + dense over the shared HybridIndex chunks, equal-weight RRF k=60, depth 40, exactly
    HybridRetrieval's hx.search) -> the first 40 chunks -> pack_evidence at LEX10_BUDGET (15,500 rendered chars, the
    same chunk packer and size as LEX10b) -> the shared answer call. Only the ORDER differs from LEX10b (F2 MaxMean).
    This is the offline arm "HYBRID@B**" of lex10/b/eval10b.py (pack_at(hybrid_order40, 15,500)) run online.

Run with --approach-class ragbench.approaches.lex10.control:Hybrid15500 (not added to the shared registry).
"""
import time

from ...contracts import pack_evidence
from ...retrieval import hybrid as H
from ..hybrid import HybridRetrieval
from .approach import DEPTH, LEX10_BUDGET

CONTROL_VERSION = "hybrid15500-v1"


class Hybrid15500(HybridRetrieval):
    name = "HYBRID15500"
    index_neutral_flags = ()

    def config_signature(self):
        sig = super().config_signature()
        sig["control"] = {"version": CONTROL_VERSION, "budget": LEX10_BUDGET, "depth": DEPTH, "order": "hybrid rrf",
                          "packer": "pack_evidence (chunk)"}
        return sig

    def _answer(self, query, index):
        T0 = time.perf_counter()
        hx = index.payload
        doc_ids = None if query.doc_scope in (None, "corpus") else list(query.doc_scope)
        hits = hx.search(query.text, doc_ids=doc_ids, k=DEPTH, depth=DEPTH, client=self.client)
        items = [h.to_item(rank=r) for r, h in enumerate(hits)]
        t = time.perf_counter()
        ev = pack_evidence(items, index.corpus, budget=LEX10_BUDGET, cfg=self.cfg)
        pack_ms = (time.perf_counter() - t) * 1000.0
        retrieval_ms = (time.perf_counter() - T0) * 1000.0
        t = time.perf_counter()
        answer = self.generate_answer(query, ev)
        st = {"version": CONTROL_VERSION, "budget": LEX10_BUDGET, "t_pack_ms": round(pack_ms, 1),
              "t_retrieval_ms": round(retrieval_ms, 1), "t_answer_ms": round((time.perf_counter() - t) * 1000.0, 1),
              "order40": [int(h.chunk) for h in hits], "evidence_items": ev.n_included, "evidence_chars": ev.chars}
        trace = {"hybrid": hits.diagnostics, "control": st, "retrieval_ms": st["t_retrieval_ms"], "chunks_total": hx.n,
                 "evidence_items": ev.n_included, "last_item_cut": ev.last_item_cut, "rrf_k": H.RRF_K}
        return self._result(answer, ev, items, trace)
