"""S15 (post-hoc, Nov 2026): cross-lingual query swap and Persian orthographic perturbation.

Retrieval reuses S10's verified code path (s10lib: frozen HybridIndex.search, HYBRID packer) and S9's HR path
(fused top 50, depth 40 -> rerank_hits keep 40 -> HYBRID _offer + final_pack at 6,000). Recall@40 uses S7's span rule
(t5_channels.gold_cov / bundle_cls, copied); pack reach uses S7's text rule (s7lib.gold_reach, imported).
Frozen code @77924d4 imported read-only.
"""
import dataclasses
import json
import os
import re
import sys
from functools import lru_cache
from pathlib import Path

HERE = Path(__file__).resolve().parent
PH = HERE.parent
for p in (str(PH / "S10_retrieval"), str(PH / "S7_analysis")):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np  # noqa: E402

import s10lib as S10  # noqa: E402   (sets CUDA_VISIBLE_DEVICES="" ; GPU scripts pop it before importing torch)
from ragbench.normalize import normalize  # noqa: E402
from ragbench.retrieval import hybrid as H  # noqa: E402

CFG = S10.CFG
FZ, TR, DEVQ = S10.FZ, S10.TR, S10.DEVQ
OTHER = {"fa": "en", "en": "fa"}
LANG_NAME = {"fa": "Persian", "en": "English"}


def jl(p):
    p = Path(p)
    return [json.loads(l) for l in open(p, encoding="utf-8-sig") if l.strip()] if p.is_file() else []


def append(p, rec):
    with open(p, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def qids(split="test"):
    return S10.qids(split)


def lang_of(qid):
    return S10.pool_of(qid).split("-")[1]


def is_ab(qid):
    return "-AB-" in qid


def tier_of(qid):
    m = re.search(r"-(T[123])-", qid)
    return m.group(1) if m else "AB"


def orig_query(qid, split="test"):
    return S10.query_for(qid, split)


def swapped_query(qid, text, split="test"):
    """The question's own pool / scope / format, the query text in the other language (q_lang = that language)."""
    q = orig_query(qid, split)
    ql = OTHER[q.q_lang]
    return dataclasses.replace(q, text=text, q_lang=ql, setting="%s-%s" % (ql, q.d_lang))


def text_query(qid, text, split="test"):
    """Same question and language, different query text (orthographic variants)."""
    return dataclasses.replace(orig_query(qid, split), text=text)


# ------------------------------------------------------------------ retrieval
def doc_ids(query):
    return None if query.doc_scope in (None, "corpus") else list(query.doc_scope)


def search(query, channels, qvec, k=40, depth=40, hx=None):
    hx = hx or S10.hx(query.corpus_set)
    return hx.search(query.text, doc_ids=doc_ids(query), k=k, depth=depth, channels=channels,
                     query_vector=qvec if "dense" in channels else None)


def hr_list(query, qvec, rr):
    """HR as in approaches/hybrid.py (rerank flag): fused k=50 depth 40, rerank keep 40; items carry the rerank score."""
    from ragbench.retrieval import rerank as R
    fused = search(query, ("bm25", "dense"), qvec, k=50, depth=40)
    kept, info, sc = R.rerank_hits(query.text, fused, rr, keep=40)
    fpos = {id(h): i for i, h in enumerate(fused)}
    items = []
    for r, (h, s) in enumerate(zip(kept, sc)):
        it = h.to_item(rank=r)
        it.score = round(float(s), 6)
        it.meta["rerank"] = round(float(s), 6)
        it.meta["rrf_rank"] = fpos[id(h)]
        items.append(it)
    return items, [h.chunk for h in fused], info


def pack6000(items, query):
    ev = S10.pack(items, query.corpus_set, 6000)
    msgs, npred = S10.messages(query, ev)
    return ev, msgs, npred


# ------------------------------------------------------------------ gold measures (S7)
@lru_cache(maxsize=None)
def qv2():
    return {d["id"]: d["meta"]["v2"] for d in jl(FZ / "questions" / "test_v2judge_all.jsonl")}


def gold_cov(hx, q, chunk_ids):
    """S7 t5_channels.gold_cov (copied): fraction of each gold span covered by the union of the chunks' spans."""
    out = {}
    for e in qv2()[q]["evidence"]:
        a, b = e["start"], e["end"]
        cov = np.zeros(max(1, b - a), bool)
        for c in chunk_ids:
            ch = hx.chunks[c]
            if ch["doc_id"] != e["doc_id"] or not ch.get("span"):
                continue
            s0, s1 = ch["span"]
            lo, hi = max(a, s0), min(b, s1)
            if hi > lo:
                cov[lo - a:hi - a] = True
        out[e["id"]] = float(cov.mean())
    return out


def bundle_cls(q, cov, thr=0.9):
    fr = [sum(cov[x] >= thr for x in bun) / len(bun) for bun in qv2()[q]["evidence_bundles"]]
    best = max(fr)
    return "complete" if best == 1.0 else ("partial" if best > 0 else "none")


def item_recall(q, cov, thr=0.9):
    ids = sorted({x for bun in qv2()[q]["evidence_bundles"] for x in bun})
    return sum(cov[x] >= thr for x in ids), len(ids)


def recall40(qid, set_name, chunks):
    hx = S10.hx(set_name)
    cov = gold_cov(hx, qid, list(chunks)[:40])
    h, t = item_recall(qid, cov)
    return {"cls": bundle_cls(qid, cov), "items_hit": h, "items": t}


_S7 = {}


def reach(qid, msgs):
    if "L" not in _S7:
        import s7lib
        _S7["L"] = s7lib
    r = _S7["L"].gold_reach(qid, msgs)
    return {"cls": r["cls"], "best_frac": r["best_frac"]}


# ------------------------------------------------------------------ raw (un-normalised) BM25, S7 t5_channels copy
_RE_WORD_RAW = re.compile(r"\d+(?:[.,/]\d+)*%?|[^\W\d_]+(?:[‌'][^\W\d_]+)*", re.UNICODE)


def tok_raw(text):
    return [t for t in _RE_WORD_RAW.findall((text or "").lower())]


@lru_cache(maxsize=None)
def raw_model(set_name):
    import bm25s
    from ragbench.approaches import baselines as BL
    hx = S10.hx(set_name)
    toks = [tok_raw(H.index_text(c)) for c in hx.chunks]
    p = BL.DEFAULTS["bm25"]
    m = bm25s.BM25(k1=p["k1"], b=p["b"], method=p["method"])
    m.index(toks, show_progress=False)
    return m


class RawBM25:
    """Context manager: the set's HybridIndex scores BM25 with the plain tokenizer on BOTH sides (S7 'raw_both') or
    the plain query tokenizer on the frozen index ('raw_query'). Everything else in HybridIndex.search is unchanged."""

    def __init__(self, set_name, mode="raw_both"):
        self.hx, self.mode, self.sn = S10.hx(set_name), mode, set_name

    def _scores(self, text):
        m = raw_model(self.sn) if self.mode == "raw_both" else self.hx.bm25
        known = [t for t in dict.fromkeys(tok_raw(text)) if t in m.vocab_dict]
        if not known:
            return np.zeros(self.hx.n, np.float32), {"raw": self.mode, "tokens": 0}
        return np.asarray(m.get_scores(known), dtype=np.float32), {"raw": self.mode, "tokens": len(known)}

    def __enter__(self):
        self.hx.bm25_scores = self._scores
        return self.hx

    def __exit__(self, *a):
        del self.hx.bm25_scores


# ------------------------------------------------------------------ vectors cache
VEC = HERE / "vectors.jsonl"     # {"key": sha1(text_sent), "text_sent": ..., "vec": [...]} ; text_sent is exactly what Ollama saw


def text_key(t):
    import hashlib
    return hashlib.sha1(t.encode("utf-8")).hexdigest()


@lru_cache(maxsize=1)
def _vecs():
    return {r["key"]: np.asarray(r["vec"], np.float32) for r in jl(VEC)}


def vec(text_sent):
    return _vecs()[text_key(text_sent)]


def saved_orig_vec(qid):
    return S10.qvecs()[qid]
