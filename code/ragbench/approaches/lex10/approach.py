"""LEX10b (2026-09-26, work/datasets_v2/lex10/b/PROGRESS.md): F2 MaxMean chunk order + the unchanged chunk packer at a
wider pack, for the same single answer call. No cross-encoder, no extra LLM call.

    question encoded once by bge-m3 (dense + sparse); BM25 + bge-m3 dense + bge-m3 sparse top 40 over the shared
    chunks (in scope); MaxMean fusion over the REAL channel scores (lex8/fusion.py, as the offline LEX8 F2 arm);
    the first 40 chunks of that order -> pack_evidence at LEX10_BUDGET rendered chars -> the shared answer call.

LEX10_BUDGET = 15,500 chars is B**, fixed by the declared token rule (the largest 500-step pack whose predicted mean
answer-call tokens <= 2.40x HYBRID, tiktoken o200k x per-language GLM factor; lex10/tokbudget.py). The per-language
variant tied on retention and was not chosen (declared tie rule). Index-time artifacts are LEX8's (identity-checked).
"""
import time

from ...contracts import pack_evidence
from ...normalize import normalize
from ..lex8 import fusion as FU
from ..lex8 import retrieve as RV
from ..lex8.approach import Lex8

LEX10_VERSION = "lex10b-v1"
LEX10_BUDGET = 15500
DEPTH = 40


class Lex10(Lex8):
    name = "LEX10"
    index_neutral_flags = ()

    def config_signature(self):
        sig = super().config_signature()
        sig["lex10"] = {"version": LEX10_VERSION, "budget": LEX10_BUDGET, "depth": DEPTH, "fusion": "maxmean",
                        "packer": "pack_evidence (chunk)"}
        return sig

    def _answer(self, query, index):
        T0 = time.perf_counter()
        six = self._six
        st = {"version": LEX10_VERSION, "device": getattr(Lex8._enc, "device", None), "budget": LEX10_BUDGET}
        doc_ids = None if query.doc_scope in (None, "corpus") else list(query.doc_scope)
        t = time.perf_counter()
        q = Lex8._enc.encode([normalize(query.text)], max_length=128, colbert=False)[0]
        st["t_encode_ms"] = round((time.perf_counter() - t) * 1000.0, 1)
        t = time.perf_counter()
        mask, _ = six.hx.scope_mask(doc_ids)
        arrays, valid = RV.chunk_channels(six, query.text, q, mask)
        lists = {ch: FU.top_ids(arr, valid[ch], DEPTH) for ch, arr in arrays.items()}
        order, _sc, _N = FU.fuse(arrays, lists, "maxmean")
        order = [int(c) for c in order[:DEPTH]]
        st["t_channels_ms"] = round((time.perf_counter() - t) * 1000.0, 1)
        t = time.perf_counter()
        items = [six.hx.hit_for_chunk(c).to_item(rank=i) for i, c in enumerate(order)]
        ev = pack_evidence(items, index.corpus, budget=LEX10_BUDGET, cfg=self.cfg)
        st["t_pack_ms"] = round((time.perf_counter() - t) * 1000.0, 1)
        st["t_retrieval_ms"] = round((time.perf_counter() - T0) * 1000.0, 1)
        t = time.perf_counter()
        answer = self.generate_answer(query, ev)
        st["t_answer_ms"] = round((time.perf_counter() - t) * 1000.0, 1)
        st.update({"channels": {k: [int(x) for x in v] for k, v in lists.items()}, "order40": order,
                   "evidence_items": ev.n_included, "evidence_chars": ev.chars})
        trace = {"lex10": st, "retrieval_ms": st["t_retrieval_ms"], "chunks_total": six.hx.n,
                 "evidence_items": ev.n_included, "last_item_cut": ev.last_item_cut}
        return self._result(answer, ev, items, trace)
