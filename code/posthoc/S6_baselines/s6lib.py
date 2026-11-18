"""S6 baselines: shared helpers (frozen ragbench @77924d4 imported read-only; no GPU, no Ollama).

Records come from work/datasets_v2/test_run/runs/<pool>/<arm>/runs_*.jsonl (latest non-error record per question,
the same rule as report_test.load). Packing uses the frozen HybridRetrieval._offer + final_pack + build_answer_messages
over RetrievedItems rebuilt from the saved retrieved_units (text, doc_id, section_id, span, unit_ids, meta)."""
import json
import os
import sys
import types
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
CODE = r"D:\KareOne\RAG-Paper-upgrades"
if CODE not in sys.path:
    sys.path.insert(0, CODE)

ROOT = Path(r"D:\KareOne\RAG Paper")
DS = ROOT / "work" / "datasets_v2"
FZ = DS / "test_freeze"
TR = DS / "test_run"
OUT = ROOT / "work" / "posthoc_2026-11" / "S6_baselines"

from ragbench.config import load_config  # noqa: E402
from ragbench.contracts import RetrievedItem, build_answer_messages, load_questions, answer_num_predict  # noqa: E402
from ragbench.corpus.registry import load_set  # noqa: E402
from ragbench.run import answer_messages_sha256  # noqa: E402

CFG = load_config()
CFG.models["chat"] = "openrouter/z-ai/glm-5.3-flash"   # as ragbench.run --reader does (no effect on messages)
CFG.models["chat_digest"] = "openrouter"
MAN = json.loads((FZ / "MANIFEST.json").read_text(encoding="utf-8"))
POOLS = ["BOOK-en", "BOOK-fa", "LAW-en", "LAW-fa", "RPT-en", "RPT-fa"]


def pool_of(q):
    return "-".join(q.split("-")[2:4])


def jl(p):
    p = Path(p)
    return [json.loads(l) for l in open(p, encoding="utf-8-sig") if l.strip()] if p.is_file() else []


QUESTIONS = {}
for _p in POOLS:
    for _q in load_questions(FZ / "questions" / ("test_v2run_%s.jsonl" % _p)):
        QUESTIONS[_q.id] = (_q, MAN["pools"][_p]["setting"])
    for _q in load_questions(DS / "devrun" / "questions" / ("dev_v2run_%s.jsonl" % _p)):   # smoke tests only
        QUESTIONS[_q.id] = (_q, MAN["pools"][_p]["setting"])
QIDS = [json.loads(l)["id"] for l in open(FZ / "questions" / "test_v2run_all.jsonl", encoding="utf-8") if l.strip()]
DEV60 = DS / "dev60" / "runs"


def records(arm, base=TR / "runs"):
    """{qid: latest non-error record (or last error record)} of a frozen GLM arm."""
    out = {}
    for p in POOLS:
        d = base / p / arm
        for f in sorted(d.glob("runs_*.jsonl")):
            for r in jl(f):
                q = r["question_id"]
                old = out.get(q)
                if old is None or old.get("error") or not r.get("error"):
                    out[q] = r
    return out


_SETS = {}


def corpus(set_name):
    if set_name not in _SETS:
        _SETS[set_name] = load_set(set_name, CFG)
    return _SETS[set_name]


_APP = {}


def approach(kind="HYBRID"):
    """Frozen approach objects (no client calls are made; OllamaClient is constructed lazily but never used)."""
    if kind not in _APP:
        from ragbench.approaches.hybrid import HybridRetrieval
        from ragbench.approaches.setr.approach import SetR
        cls = {"HYBRID": HybridRetrieval, "SETR": SetR}[kind]
        a = cls(cfg=CFG, client=types.SimpleNamespace(ledger=None), flags={})
        a.variants_version = "variants-v2"
        _APP[kind] = a
    return _APP[kind]


def query_of(qid, app=None):
    q, setting = QUESTIONS[qid]
    return (app or approach()).make_query(q, setting)


def handle(query):
    return types.SimpleNamespace(corpus=corpus(query.corpus_set), payload=None)


def items_of(rec):
    return [RetrievedItem(**d) for d in rec["retrieved_units"]]


def pack_messages(ordered_items, qid, app=None, no_fill=False):
    """-> (messages, num_predict, ev). HYBRID's frozen offer + packer over the given order (no_fill: the items are
    offered as they are, the packer alone; see RESULTS.md)."""
    app = app or approach()
    query = query_of(qid, app)
    idx = handle(query)
    offered = list(ordered_items) if no_fill else app._offer(ordered_items, idx)
    ev = app.final_pack(offered, idx, query)
    msgs = build_answer_messages(query, ev, CFG)
    return msgs, answer_num_predict(query, CFG), ev, offered


def closed_book_messages(qid):
    query = query_of(qid)
    return build_answer_messages(query, None, CFG), answer_num_predict(query, CFG)


def sha(msgs):
    return answer_messages_sha256(msgs)


def pack_file(variant, qid):
    q, setting = QUESTIONS[qid]
    return TR / "packs" / variant / ("%s__%s.json" % (qid, setting))
