"""S10 (post-hoc, Nov 2026): single-channel retrieval arms with HYBRID's frozen packer and answer prompt. No GPU.

    DENSE_<B>  HybridIndex.search(channels=("dense",), k=40, depth=40) with the saved Ollama bge-m3 query vector
               (S9 align_vectors.npz q_oll) over the frozen stored chunk vectors (_hybrid/<set>/embeddings.npy)
    BM25_<B>   HybridIndex.search(channels=("bm25",), k=40, depth=40) (frozen bm25s index + tokenizer, CPU)
    B = 6000:  HybridRetrieval._offer + final_pack at 6,000 (HYBRID's own path)
    B = 15500: pack_evidence over the first 40 at 15,500 (HYBRID15500's lex10/control.py path)
A single channel through the frozen rrf_fuse gives score 1/(60+rank), i.e. the channel's own order.
Pack rendering: contracts.pack_evidence + build_answer_messages (frozen code, imported read-only).
"""
import hashlib
import json
import os
import sys
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace

os.environ["CUDA_VISIBLE_DEVICES"] = ""
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_k] = "2"
FROZEN = r"D:\KareOne\RAG-Paper-upgrades"
if FROZEN not in sys.path:
    sys.path.insert(0, FROZEN)

import numpy as np  # noqa: E402

from ragbench.config import load_config  # noqa: E402
from ragbench.contracts import answer_num_predict, build_answer_messages, load_questions, pack_evidence  # noqa: E402
from ragbench.corpus.registry import load_set  # noqa: E402
from ragbench.pools import load_pool, resolve_pool_path  # noqa: E402
from ragbench.approaches.hybrid import HybridRetrieval  # noqa: E402
from ragbench.retrieval import hybrid as H  # noqa: E402
from ragbench.util import read_jsonl  # noqa: E402

ROOT = Path(r"D:\KareOne\RAG Paper")
WORK = ROOT / "work"
FZ = WORK / "datasets_v2" / "test_freeze"
TR = WORK / "datasets_v2" / "test_run"
DEVQ = WORK / "datasets_v2" / "devrun" / "questions"
HERE = Path(__file__).resolve().parent
CFG = load_config()
DEPTH = 40
BUDGETS = (6000, 15500)
ARMS = ["DENSE_6000", "DENSE_15500", "BM25_6000", "BM25_15500"]
CHANNEL = {"DENSE": ("dense",), "BM25": ("bm25",), "HYB": ("bm25", "dense")}
PACKDIR = {"HYBRID_6000": "HYBRID", "HYBRID_15500": "HYBRID15500"}


def sha256_msgs(messages):
    return hashlib.sha256(json.dumps(messages, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


def pool_of(q):
    return "-".join(q.split("-")[2:4])


def qfile(split, pool=None):
    d = FZ / "questions" if split == "test" else DEVQ
    return d / ("%s_v2run_%s.jsonl" % (split, pool or "all"))


def qids(split="test"):
    return [json.loads(l)["id"] for l in open(qfile(split), encoding="utf-8") if l.strip()]


@lru_cache(maxsize=None)
def questions(split, pool):
    return {q.id: q for q in load_questions(qfile(split, pool))}


@lru_cache(maxsize=None)
def _approach(pool):
    a = HybridRetrieval(cfg=CFG, client=SimpleNamespace(ledger=None))
    a.pool = load_pool(resolve_pool_path(pool, CFG))
    a.variants_version = "variants-v2"
    return a


SETTING = {"RPT-fa": "fa-fa", "RPT-en": "en-en", "BOOK-fa": "fa-fa", "BOOK-en": "en-en", "LAW-fa": "fa-fa",
           "LAW-en": "en-en"}


def query_for(qid, split="test"):
    pool = pool_of(qid)
    return _approach(pool).make_query(questions(split, pool)[qid], SETTING[pool])


@lru_cache(maxsize=None)
def corpus(set_name):
    return load_set(set_name, CFG)


@lru_cache(maxsize=None)
def hx(set_name):
    return H.HybridIndex.load(corpus(set_name), CFG, client=None, refresh_vectors=False)


@lru_cache(maxsize=None)
def qvecs():
    z = np.load(WORK / "posthoc_2026-11" / "S9_gpu" / "align_vectors.npz")
    return {str(q): np.asarray(v, np.float32) for q, v in zip(z["qids"], z["q_oll"])}


def search(query, channel, k=DEPTH):
    doc_ids = None if query.doc_scope in (None, "corpus") else list(query.doc_scope)
    qv = qvecs()[query.question_id] if "dense" in CHANNEL[channel] else None
    return hx(query.corpus_set).search(query.text, doc_ids=doc_ids, k=k, depth=DEPTH, channels=CHANNEL[channel],
                                       query_vector=qv)


class _Offer(HybridRetrieval):
    def __init__(self, budget):
        super().__init__(cfg=CFG, client=SimpleNamespace(ledger=None))
        self.budget_chars = budget

    def pack(self, items, index, budget=None):
        return pack_evidence(items, index.corpus, budget=budget or self.budget_chars, cfg=self.cfg)


def pack(items, set_name, budget):
    """HYBRID's packer: 6000 -> _offer + final_pack (HybridRetrieval); 15500 -> pack_evidence over the first 40."""
    cs = corpus(set_name)
    if budget == 6000:
        a = _Offer(budget)
        ns = SimpleNamespace(corpus=cs)
        return a.final_pack(a._offer(items, ns), ns, None)
    return pack_evidence(items[:DEPTH], cs, budget=budget, cfg=CFG)


def messages(query, ev):
    return build_answer_messages(query, ev, CFG), answer_num_predict(query, CFG)


@lru_cache(maxsize=None)
def frozen_records(arm):
    out = {}
    for d in sorted((TR / "runs").glob("*/" + arm)):
        for f in d.glob("runs_*.jsonl"):
            for r in read_jsonl(f):
                old = out.get(r["question_id"])
                if old is None or old.get("error") or not r.get("error"):
                    out[r["question_id"]] = r
    return out


def frozen_pack(arm, qid, setting):
    p = TR / "packs" / PACKDIR[arm] / ("%s__%s.json" % (qid, setting))
    return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else None
