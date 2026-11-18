"""HYBRID: the B2+B3 reciprocal-rank-fusion control (2026-09-24; a reviewer's demanded baseline).

    HYBRID         BM25 (B2) and bge-m3 cosine (B3) over the SAME chunks of the shared HybridIndex, fused by
                   equal-weight RRF (k=60, one contribution per channel per chunk). The fused chunks are offered to
                   the shared packer exactly as B2/B3 offer theirs (at least chunking.top_k, more while every offered
                   chunk fits whole; max_candidates 40), then the shared answer call.
    HYBRID+blocks  the same fused candidates (top 40) turned into source-grounded evidence blocks by
                   ragbench.evidence.build_blocks (merge before selection, sentence-bounded 700-1,000 char prose,
                   row-aware tables, exact budget) - isolates evidence construction from retrieval.
    HYBRID+rerank  (2026-09-25; a BASELINE / comparison arm only - LEX, LEX3 and AGENTIC* never use it) the top
                   RERANK_DEFAULTS["candidates"] (50) fused hits re-scored by the cross-encoder BAAI/bge-reranker-v2-m3
                   on (question, chunk text) (ragbench/retrieval/rerank.py) and re-ordered; the top `candidates` (40)
                   of the reranked list then go to the same packer / blocks / budget / shared answer call. Combines with
                   blocks. Index-neutral (same shared HybridIndex). trace["rerank"]: candidates_reranked, ms, device,
                   fp16, fallback (CPU because the GPU was busy / out of memory), load_ms (model load, outside the
                   timed query: it happens when the index is loaded).

Index: the shared HybridIndex under <index_root>/_hybrid/<set>-<fp8>/ (python -m ragbench.retrieval.prebuild fills
its vectors ahead of time). This approach's own index dir only records which shared index it used. At build time
missing chunk vectors are embedded (index phase, like B3) unless HYBRID_DEFAULTS["embed_at_build"] is False; at query
time only the question is embedded - a chunk still without a vector is reported in trace.hybrid.dense, never embedded.
"""
import time

from ..retrieval import hybrid as H
from ..util import atomic_write_json
from .baselines import DEFAULTS as BL_DEFAULTS, _ChunkRetrieval

HYBRID_DEFAULTS = {
    "embed_at_build": True,
    "candidates": 40,            # fused hits requested (= baselines max_candidates)
    "depth": 40,                 # per-channel depth fed to RRF
}


class HybridRetrieval(_ChunkRetrieval):
    """HYBRID / HYBRID+blocks (flag blocks). Flags doc_card / section_expand as for B2/B3."""
    name = "HYBRID"
    # rerank re-orders the candidates of the same shared index at query time
    index_neutral_flags = ("doc_card", "section_expand", "rerank", "intact", "kw", "doc")
    reranker = None                  # tests inject an object with .score(question, texts) -> (scores, info)

    def config_signature(self):
        return {"hybrid": H.signature(self.cfg), "defaults": HYBRID_DEFAULTS}

    def _get_reranker(self):
        if self.reranker is None:
            from ..retrieval import rerank as R
            self.reranker = R.get_reranker()
        load = getattr(self.reranker, "load", None)
        if callable(load):
            load()
        return self.reranker

    def _build_index(self, corpus_set, index_dir):
        hx = H.HybridIndex.open(corpus_set, self.cfg, client=self.client, embed=bool(HYBRID_DEFAULTS["embed_at_build"]))
        if HYBRID_DEFAULTS["embed_at_build"] and not hx.dense_complete:
            hx = H.HybridIndex.build(corpus_set, self.cfg, client=self.client, embed=True)
        atomic_write_json(index_dir / "hybrid_ref.json", hx.status(), indent=1)
        if self.flags.get("rerank"):
            self._get_reranker()         # the model load stays outside the timed query
        self._kw_warm(hx)
        return hx

    def _kw_warm(self, hx):
        """KW / DOC (2026-09-26, approaches/kwdoc.py): the keyword index is built at load, outside the timed query."""
        if self.flags.get("kw") or self.flags.get("doc"):
            from . import kwdoc
            kwdoc.keyword_index(hx)

    def _load_index(self, corpus_set, index_dir):
        hx = H.HybridIndex.load(corpus_set, self.cfg, client=self.client)
        if self.flags.get("rerank"):
            self._get_reranker()
        self._kw_warm(hx)
        return hx

    def _answer(self, query, index):
        if self.flags.get("intact"):     # HYBRID-intact (2026-09-26): LEX5's fallback packing, no selector
            from .lex5.approach import answer_intact
            return answer_intact(self, query, index)
        if self.flags.get("kw") or self.flags.get("doc"):   # KW / DOC ideas (approaches/kwdoc.py)
            from .kwdoc import answer_kw
            return answer_kw(self, query, index)
        t0 = time.perf_counter()
        hx = index.payload
        doc_ids = None if query.doc_scope in (None, "corpus") else list(query.doc_scope)
        k = max(int(HYBRID_DEFAULTS["candidates"]), int(self.cfg.chunking["top_k"]))
        rerank = bool(self.flags.get("rerank"))
        if rerank:
            from ..retrieval import rerank as R
            n_rr = max(int(R.RERANK_DEFAULTS["candidates"]), k)
            fused = hx.search(query.text, doc_ids=doc_ids, k=n_rr, depth=int(HYBRID_DEFAULTS["depth"]), client=self.client)
            rr = self._get_reranker()
            kept, rinfo, rscores = R.rerank_hits(query.text, fused, rr, keep=k)
            rinfo["load_ms"] = getattr(rr, "load_ms", None)
            hits = H.HitList(kept, fused.diagnostics)
            fpos = {id(h): i for i, h in enumerate(fused)}
            items = []
            for r, (h, s) in enumerate(zip(hits, rscores)):
                it = h.to_item(rank=r)
                it.score = round(float(s), 6)            # the reranker's score; rrf stays in meta
                it.meta["rerank"] = round(float(s), 6)
                it.meta["rrf_rank"] = fpos[id(h)]
                items.append(it)
        else:
            hits = hx.search(query.text, doc_ids=doc_ids, k=k, depth=int(HYBRID_DEFAULTS["depth"]), client=self.client)
            items = [h.to_item(rank=r) for r, h in enumerate(hits)]
        trace = {"hybrid": hits.diagnostics}
        if rerank:
            trace["rerank"] = rinfo
        if self.flags.get("blocks"):
            from ..evidence import build_blocks
            offered, rep = build_blocks(hits, index.corpus, budget=int(self.cfg.evidence["max_chars"]),
                                        question=query.text, cfg=self.cfg)
            trace["blocks"] = rep
        else:
            offered = self._offer(items, index)
        ev = self.final_pack(offered, index, query)
        retrieval_ms = (time.perf_counter() - t0) * 1000.0
        answer = self.generate_answer(query, ev)
        trace.update({"chunks_total": hx.n, "chunks_in_scope": hits.diagnostics.get("chunks_in_scope"),
                      "scope_docs_found": hits.diagnostics.get("scope_docs"), "offered": len(offered),
                      "evidence_items": ev.n_included, "last_item_cut": ev.last_item_cut,
                      "retrieval_ms": round(retrieval_ms, 1), "dense_complete": hx.dense_complete,
                      "params": {"top_k": int(self.cfg.chunking["top_k"]), "fill_budget": bool(BL_DEFAULTS["fill_budget"]),
                                 "max_candidates": int(BL_DEFAULTS["max_candidates"]), "rrf_k": H.RRF_K,
                                 "candidates": k, "depth": int(HYBRID_DEFAULTS["depth"]),
                                 "blocks": bool(self.flags.get("blocks")), "rerank": rerank,
                                 "rerank_candidates": (n_rr if rerank else None)},
                      # [chunk, rrf, bm25 rank, dense rank] (+ [rerank score, rrf rank] with rerank), in offered order
                      "top": [[h.chunk, round(h.rrf, 6), h.ranks.get("bm25"), h.ranks.get("dense")]
                              + ([it.meta["rerank"], it.meta["rrf_rank"]] if rerank else []) for h, it in zip(hits, items)]})
        return self._result(answer, ev, items, trace)

    @staticmethod
    def _result(answer, ev, items, trace):
        from ..contracts import AnswerResult
        return AnswerResult(answer_text=answer, evidence=ev, retrieved=items, trace=trace)
