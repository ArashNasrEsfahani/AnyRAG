"""S13 shared helpers: questions/queries (frozen make_query), HybridIndex per set, e5 vectors, frozen pack path.

Pack/message building reuses S2a's packs.py (imported read-only from ../S2a_packs; it only wraps frozen ragbench code).
"""
import json
import os
import sys
from functools import lru_cache
from pathlib import Path

HERE = Path(__file__).resolve().parent
PH = HERE.parent
for _p in (str(PH / "S2a_packs"),):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np  # noqa: E402

import packs as PK  # noqa: E402  (sets CUDA_VISIBLE_DEVICES="" for CPU-only processes)
from ragbench.contracts import load_questions  # noqa: E402
from ragbench.normalize import normalize  # noqa: E402
from ragbench.retrieval import hybrid as H  # noqa: E402

CFG = PK.CFG
DS = PK.WORK / "datasets_v2"
QFILES = {"dev": DS / "devrun" / "questions", "test": DS / "test_freeze" / "questions"}
POOLS = {"RPT-fa": "fa-fa", "RPT-en": "en-en", "BOOK-fa": "fa-fa", "BOOK-en": "en-en", "LAW-fa": "fa-fa",
         "LAW-en": "en-en"}
SETS = ["BOOK_en", "BOOK_fa", "C", "D_en", "LAW_en", "LAW_fa"]
VEC = HERE / "vectors"
E5_MODEL = "intfloat/multilingual-e5-large"


def qlist():
    """-> [(split, pool_id, Query)] for the 96 test + 60 dev questions (frozen HybridRetrieval.make_query)."""
    out = []
    for split in ("test", "dev"):
        for pid, setting in POOLS.items():
            for q in load_questions(QFILES[split] / ("%s_v2run_%s.jsonl" % (split, pid))):
                out.append((split, pid, PK._approach(pid).make_query(q, setting)))
    return out


@lru_cache(maxsize=None)
def hx(set_name):
    return H.HybridIndex.load(PK.corpus(set_name), CFG, client=None, refresh_vectors=False)


def l2(v):
    v = np.asarray(v, np.float32)
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.where(n > 0, n, 1.0)


@lru_cache(maxsize=None)
def e5_chunks(set_name):
    return l2(np.load(VEC / ("e5_chunks_%s.npy" % set_name)).astype(np.float32))


@lru_cache(maxsize=None)
def e5_queries():
    z = np.load(VEC / "e5_queries.npz")
    return {str(q): l2(v) for q, v in zip(z["qids"], z["vecs"].astype(np.float32))}


@lru_cache(maxsize=None)
def m3_queries():
    """S9's Ollama bge-m3 query vectors (the deployed HYBRID query path), 156 questions."""
    z = np.load(PH / "S9_gpu" / "align_vectors.npz")
    return {str(q): v for q, v in zip(z["qids"], z["q_oll"])}


@lru_cache(maxsize=None)
def hx_e5(set_name):
    """The same HybridIndex (chunks, BM25) with the dense matrix swapped for e5 chunk vectors."""
    h = hx(set_name)
    return H.HybridIndex(h.corpus_set, CFG, h.index_dir, h.chunks, h.bm25, e5_chunks(set_name), np.ones(h.n, bool),
                         h.meta)


def wjson(path, obj):
    Path(path).write_text(json.dumps(obj, indent=1, ensure_ascii=False), encoding="utf-8")
