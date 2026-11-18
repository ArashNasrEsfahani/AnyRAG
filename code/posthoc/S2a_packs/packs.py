"""S2a: rebuild answer packs offline with the FROZEN ragbench code (77924d4, read-only, imported from its worktree).

Nothing here re-implements rendering: items come from the saved traces (HYBRID / HR retrieved_units, LEX5B chunk ids
through lex4.core.Packager over the shared HybridIndex chunks.jsonl) and are rendered by contracts.pack_evidence and
contracts.build_answer_messages. No GPU, no encoder, no reranker.
"""
import json
import os
import sys
from functools import lru_cache
from pathlib import Path
from types import MappingProxyType, SimpleNamespace

os.environ["CUDA_VISIBLE_DEVICES"] = ""
FROZEN = r"D:\KareOne\RAG-Paper-upgrades"
if FROZEN not in sys.path:
    sys.path.insert(0, FROZEN)

from ragbench.config import load_config                       # noqa: E402
from ragbench.contracts import (RetrievedItem, answer_num_predict, build_answer_messages,  # noqa: E402
                                load_questions, pack_evidence)
from ragbench.corpus.registry import load_set                 # noqa: E402
from ragbench.pools import load_pool, resolve_pool_path       # noqa: E402
from ragbench.approaches.hybrid import HybridRetrieval        # noqa: E402
from ragbench.approaches.lex4 import core as K                # noqa: E402
from ragbench.approaches.lex5 import bundles as B             # noqa: E402
from ragbench.util import read_jsonl                          # noqa: E402

ROOT = Path(r"D:\KareOne\RAG Paper")
WORK = ROOT / "work"
FZ = WORK / "datasets_v2" / "test_freeze"
TR = WORK / "datasets_v2" / "test_run"
HERE = Path(__file__).resolve().parent
CFG = load_config()

ARM_RUNFILE = {"HYBRID_6000": "runs_HYBRID.jsonl", "HR_6000": None, "LEX5B_6000": "runs_LEX5B.jsonl",
               "HYBRID_15500": "runs_HYBRID15500.jsonl", "HR_15500": None, "LEX5B_15500": "runs_LEX5B_15500.jsonl"}
ARM_PACKDIR = {"HYBRID_6000": "HYBRID", "HR_6000": "HYBRID_rerank-0c0612fe", "LEX5B_6000": "LEX5B",
               "HYBRID_15500": "HYBRID15500", "HR_15500": "HR15500_rerank-dcb91474", "LEX5B_15500": "LEX5B_15500"}


def sha256_msgs(messages):
    import hashlib
    return hashlib.sha256(json.dumps(messages, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


def pool_of(q):
    return "-".join(q.split("-")[2:4])


def qids_all():
    return [json.loads(l)["id"] for l in open(FZ / "questions" / "test_v2run_all.jsonl", encoding="utf-8") if l.strip()]


@lru_cache(maxsize=None)
def questions(pool):
    return {q.id: q for q in load_questions(FZ / "questions" / ("test_v2run_%s.jsonl" % pool))}


@lru_cache(maxsize=None)
def question_dicts():
    return {d["id"]: d for d in (json.loads(l) for l in open(FZ / "questions" / "test_v2run_all.jsonl",
                                                             encoding="utf-8") if l.strip())}


@lru_cache(maxsize=None)
def corpus(set_name):
    return load_set(set_name, CFG)


@lru_cache(maxsize=None)
def hx_chunks(set_name):
    ref = None
    for f in (WORK / "indexes" / "LEX5B").glob(set_name + "-*/hybrid_ref.json"):
        ref = json.loads(f.read_text(encoding="utf-8"))
        break
    d = Path(ref["index_dir"])
    return SimpleNamespace(chunks=read_jsonl(d / "chunks.jsonl"), index_dir=d)


@lru_cache(maxsize=None)
def packager(set_name):
    return K.Packager(hx_chunks(set_name), corpus(set_name))


@lru_cache(maxsize=None)
def _approach(pool):
    a = HybridRetrieval(cfg=CFG, client=SimpleNamespace(ledger=None))
    a.pool = load_pool(resolve_pool_path(pool, CFG))
    a.variants_version = "variants-v2"
    return a


def query_for(qid, setting):
    pool = pool_of(qid)
    return _approach(pool).make_query(questions(pool)[qid], setting)


@lru_cache(maxsize=None)
def frozen_records(arm):
    """{qid: latest non-error record} of a frozen GLM arm."""
    out = {}
    for d in sorted((TR / "runs").glob("*/" + arm)):
        for f in d.glob("runs_*.jsonl"):
            for r in read_jsonl(f):
                old = out.get(r["question_id"])
                if old is None or old.get("error") or not r.get("error"):
                    out[r["question_id"]] = r
    return out


def frozen_pack(arm, qid, setting):
    p = TR / "packs" / ARM_PACKDIR[arm] / ("%s__%s.json" % (qid, setting))
    return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else None


# ------------------------------------------------------------------ pack builders
def _ns(cs):
    return SimpleNamespace(corpus=cs)


def items_from_record(rec):
    return [RetrievedItem(**d) for d in rec["retrieved_units"]]


def pack_hybrid_direct(items, cs, budget):
    """HYBRID15500's path (lex10/control.py): pack_evidence over the ranked top 40 at the budget."""
    return pack_evidence(items, cs, budget=budget, cfg=CFG)


class _Offer(HybridRetrieval):
    """HybridRetrieval._offer + final_pack with only the pack budget set (the budget_matrix._Wide rule)."""

    def __init__(self, budget):
        super().__init__(cfg=CFG, client=SimpleNamespace(ledger=None))
        self.budget_chars = budget

    def pack(self, items, index, budget=None):
        return pack_evidence(items, index.corpus, budget=budget or self.budget_chars, cfg=self.cfg)


def pack_offer(items, cs, budget):
    a = _Offer(budget)
    offered = a._offer(items, _ns(cs))
    return a.final_pack(offered, _ns(cs), None)


def lex_packages(set_name, chunk_ids):
    P = packager(set_name)
    return [MappingProxyType(dict(P.package(int(c)))) for c in chunk_ids]


def pack_lex_chunks(set_name, chunk_ids, budget):
    """LEX5B's step 7: P.item per shown package -> pack_evidence at the budget."""
    P = packager(set_name)
    items = [P.item(p) for p in lex_packages(set_name, chunk_ids)]
    return pack_evidence(items, corpus(set_name), budget=budget, cfg=CFG)


def fusion_fallback(set_name, f2_top40, budget):
    """lex5b approach step 2: B.hybrid_intact over the F2 packages (fusion order, whole, dedupe, exact cost)."""
    P = packager(set_name)
    pks = lex_packages(set_name, f2_top40)
    cost = B.Coster(P.cost)
    return B.hybrid_intact(pks, cost, budget), cost


def messages_for(qid, setting, ev):
    q = query_for(qid, setting)
    return build_answer_messages(q, ev, CFG), answer_num_predict(q, CFG)


# ------------------------------------------------------------------ GOLD
def gold_items(qid, set_name):
    """The question's FIRST evidence bundle, as unit-kind items over the frozen corpus offsets (bundle order)."""
    v2 = question_dicts()[qid]["meta"]["v2"]
    bundles = v2.get("evidence_bundles") or []
    if not bundles:
        return None, "no evidence bundle (unanswerable)"
    ev = {e["id"]: e for e in v2.get("evidence") or []}
    cs = corpus(set_name)
    import hashlib
    items, notes = [], []
    for eid in bundles[0]:
        e = ev[eid]
        doc = cs.doc(e["doc_id"])
        a, b = int(e["start"]), int(e["end"])
        txt = doc.text[a:b]
        ok = hashlib.sha256(txt.encode("utf-8")).hexdigest() == e.get("text_sha256")
        if not ok:
            notes.append("%s sha mismatch" % eid)
        sec = None
        try:
            if e.get("unit_id") and doc.has_unit(e["unit_id"]):
                sec = cs.unit(e["unit_id"]).section_id
        except Exception:  # noqa: BLE001
            sec = None
        items.append(RetrievedItem(text=txt, doc_id=doc.id, section_id=sec,
                                   unit_ids=[u.id for u in doc.units_overlapping(a, b)], kind="unit", span=[a, b],
                                   meta={"evidence_id": eid, "sha_ok": ok}))
    return items, ";".join(notes)


GOLD_BUDGET = 10 ** 6       # budget-free: every gold item whole
