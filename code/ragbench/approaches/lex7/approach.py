"""LEX7.1 (2026-09-26): three-channel bi-encoder retrieval + MMR + a small-budget order fixer. No cross-encoder.

    LEX7    BM25 (the shared HybridIndex, B2 tokenizer) + bge-m3 dense + bge-m3 sparse lexical weights, fused by
            equal-weight RRF (k=60, depth 40 per channel) -> MMR (lambda 0.7, redundancy = cosine of the bge-m3 dense
            chunk vectors) over the top 40 -> order fixer v2 (lex7/orderfix.py: one gpt-oss-20b call over the top-12
            compact views, 200-token cap incl. reasoning, 1.5 s deadline, base order on any failure) -> HYBRID's own
            offer / final pack / answer path over the first 40.
Chunk-side dense vectors and lexical weights are INDEX-time artifacts (m3multi one-pass encode of H.index_text,
lex_research/m3, compacted to <LEX7_M3_DIR or work/datasets_v2/lex7/m3idx>/<set>.npz and checked against the shared
index by chunk count and a doc-id checksum). At query time only the question is encoded (bge-m3, one forward pass,
GPU when free); the model load and the fixer's connection warm-up happen at index load, outside the timed query.
Flags: no_fix (the MMR-a1 base order only). trace["lex7"]: channel ranks, base order, fixer info, stage ms.
"""
import hashlib
import os
import time
from pathlib import Path

import numpy as np

from ...normalize import normalize
from ...retrieval import hybrid as H
from ..hybrid import HybridRetrieval
from . import orderfix as OF

LEX7_VERSION = "lex7.1-v1"
LEX7_DEFAULTS = {"depth": 40, "rrf_k": H.RRF_K, "mmr_lambda": 0.7, "mmr_n": 40, "candidates": 40,
                 "query_max_length": 128, "fixer": "v2"}


def m3_dir(cfg):
    return Path(os.environ.get("LEX7_M3_DIR") or (Path(cfg.project_root) / "work" / "datasets_v2" / "lex7" / "m3idx"))


class M3Index:
    def __init__(self, path, hx):
        from scipy.sparse import csr_matrix
        z = np.load(path)
        n = int(z["n"])
        ck = hashlib.sha1("\n".join(c["doc_id"] for c in hx.chunks).encode("utf-8")).hexdigest()
        if n != hx.n or str(z["docsha"]) != ck:
            raise RuntimeError("lex7 m3 index %s does not match the shared index (%d vs %d chunks)" % (path, n, hx.n))
        self.dense = z["dense"].astype(np.float32)
        self.S = csr_matrix((z["data"], z["indices"], z["indptr"]), shape=tuple(int(x) for x in z["shape"]))
        self.n = n

    def sparse_scores(self, q):
        qv = np.zeros(self.S.shape[1], np.float32)
        for t, w in q.items():
            qv[int(t)] = w
        return np.asarray(self.S @ qv).ravel()


def _top(scores, valid, k):
    o = np.lexsort((np.arange(len(scores)), -scores))
    return [int(i) for i in o[valid[o]][:k]]


def rrf(lists, k=H.RRF_K):
    fused = H.rrf_fuse(dict(enumerate(lists)), k=k)
    return [c for c, _s, _r in fused], {c: s for c, s, _r in fused}


def mmr(order, score, vec, lam, n):
    """lex_research/evaluate.py mmr, verbatim semantics: min-max normalised RRF score minus max cosine to picked."""
    head = order[:n]
    v = np.array([score[c] for c in head], np.float64)
    lo, hi = v.min(), v.max()
    r = {c: (float((score[c] - lo) / (hi - lo)) if hi > lo else 1.0) for c in head}
    pos = {c: i for i, c in enumerate(order)}
    left, out = list(head), []
    while left:
        def val(c):
            red = max((float(vec[c] @ vec[s]) for s in out), default=0.0)
            return (lam * r[c] - (1 - lam) * red, -pos[c])
        best = max(left, key=val)
        out.append(best)
        left.remove(best)
    return out + order[n:]


class Lex7(HybridRetrieval):
    name = "LEX7"
    index_neutral_flags = ("no_fix",)
    _enc = None

    def config_signature(self):
        return {"hybrid": H.signature(self.cfg), "lex7": LEX7_DEFAULTS, "version": LEX7_VERSION,
                "orderfix": OF.R2, "orderfix_version": OF.ORDERFIX_V2, "prompt": OF.PROMPT}

    def _prep(self, hx):
        from ...retrieval import m3multi as M3
        name = getattr(self, "_set_name", None)
        self._m3 = M3Index(m3_dir(self.cfg) / ("%s.npz" % name), hx)
        if Lex7._enc is None:
            t = time.perf_counter()
            Lex7._enc = M3.M3Encoder().load()
            Lex7._enc.encode(["warm up"], max_length=16, colbert=False)
            self._load_ms = (time.perf_counter() - t) * 1000.0
        if not (self.flags or {}).get("no_fix"):
            # the fixer's tokenizer / snippet code load once here, outside the timed query (the smoke run spent 3.8 s
            # of its first query in the first build_prompt)
            OF.build_prompt("warm up question", [{"path": "a > b", "text": "warm up text"}] * 6)
            try:
                self._fixer().warm_deadline()
            except Exception:  # noqa: BLE001
                pass

    def _fixer(self):
        if getattr(self, "_fx", None) is None:
            self._fx = OF.client(self.cfg, self.client.ledger, caps=OF.R2)
        return self._fx

    def _build_index(self, corpus_set, index_dir):
        self._set_name = corpus_set.name
        hx = super()._build_index(corpus_set, index_dir)
        self._prep(hx)
        return hx

    def _load_index(self, corpus_set, index_dir):
        self._set_name = corpus_set.name
        hx = super()._load_index(corpus_set, index_dir)
        self._prep(hx)
        return hx

    def _answer(self, query, index):
        from ...llm import BudgetExceeded
        d = LEX7_DEFAULTS
        T0 = time.perf_counter()
        hx, X = index.payload, self._m3
        st = {"version": LEX7_VERSION, "device": getattr(Lex7._enc, "device", None),
              "load_ms": round(getattr(self, "_load_ms", 0.0), 1)}
        doc_ids = None if query.doc_scope in (None, "corpus") else list(query.doc_scope)
        mask, _ = hx.scope_mask(doc_ids)
        t = time.perf_counter()
        q = Lex7._enc.encode([normalize(query.text)], max_length=d["query_max_length"], colbert=False)[0]
        st["t_encode_ms"] = round((time.perf_counter() - t) * 1000.0, 1)
        t = time.perf_counter()
        s_bm, _ = hx.bm25_scores(query.text)
        bm = _top(np.asarray(s_bm, np.float32), mask & (np.asarray(s_bm) > 0.0), d["depth"])
        qd = np.asarray(q["dense"], np.float32)
        dn = _top(X.dense @ qd, mask.copy(), d["depth"])
        s_sp = X.sparse_scores(q["sparse"])
        sp = _top(s_sp, mask & (s_sp > 0), d["depth"])
        base, score = rrf([bm, dn, sp], k=d["rrf_k"])
        st["t_channels_ms"] = round((time.perf_counter() - t) * 1000.0, 1)
        t = time.perf_counter()
        base = mmr(base, score, X.dense, d["mmr_lambda"], d["mmr_n"])
        st["t_mmr_ms"] = round((time.perf_counter() - t) * 1000.0, 1)
        order, info = list(base), None
        if not (self.flags or {}).get("no_fix"):
            def view(c):
                ch = hx.chunks[c]
                return {"path": ch.get("breadcrumb") or "", "text": ch["text"]}
            try:
                order, info = OF.fix(self._fixer(), query.text, base, view, caps=OF.R2)
            except BudgetExceeded:
                raise
            except Exception as e:  # noqa: BLE001  never fails the query: base order
                order, info = list(base), {"status": "fail", "reason": "exception", "error": str(e)[:200]}
        st["t_fix_ms"] = round((info or {}).get("t_total_ms") or 0.0, 1)
        k = max(int(d["candidates"]), int(self.cfg.chunking["top_k"]))
        items = [hx.hit_for_chunk(c).to_item(rank=r) for r, c in enumerate(order[:k])]
        offered = self._offer(items, index)
        ev = self.final_pack(offered, index, query)
        st["t_retrieval_ms"] = round((time.perf_counter() - T0) * 1000.0, 1)
        t = time.perf_counter()
        answer = self.generate_answer(query, ev)
        st["t_answer_ms"] = round((time.perf_counter() - t) * 1000.0, 1)
        st.update({"channels": {"bm25": bm, "dense": dn, "sparse": sp}, "base40": base[:40], "order40": order[:40],
                   "fixer": info, "offered": len(offered), "evidence_items": ev.n_included,
                   "evidence_chars": ev.chars})
        trace = {"lex7": st, "retrieval_ms": st["t_retrieval_ms"], "chunks_total": hx.n,
                 "evidence_items": ev.n_included, "last_item_cut": ev.last_item_cut}
        return self._result(answer, ev, items, trace)
