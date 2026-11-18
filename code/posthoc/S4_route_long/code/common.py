"""Shared helpers for stream S4 (routing and long context). Read-only use of frozen code and data."""
import json
import os
import sys
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
ROOT = Path(r"D:\KareOne\RAG Paper")
CODE = Path(r"D:\KareOne\RAG-Paper-upgrades")
WORK = ROOT / "work"
S4 = WORK / "posthoc_2026-11" / "S4_route_long"
TR = WORK / "datasets_v2" / "test_run"
FZ = WORK / "datasets_v2" / "test_freeze"
QDIR = WORK / "datasets_v2" / "questions"
POOLS = ["RPT-fa", "RPT-en", "BOOK-fa", "BOOK-en", "LAW-fa", "LAW-en"]
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))


def jl(p):
    p = Path(p)
    return [json.loads(l) for l in open(p, encoding="utf-8-sig") if l.strip()] if p.is_file() else []


def append(p, obj):
    with open(p, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def test_questions():
    return {d["id"]: d for d in jl(FZ / "questions" / "test_v2run_all.jsonl")}


def pool_manifest(pool):
    return json.loads((WORK / "multidoc" / "pools" / ("%s.json" % pool)).read_text(encoding="utf-8"))


_SETS = {}


def pool_docs(pool):
    """-> list of ragbench Document objects of the frozen pool, verified against the pool manifest."""
    from ragbench.pools import load_pool
    from ragbench.corpus.registry import load_set
    p = load_pool(WORK / "multidoc" / "pools" / ("%s.json" % pool))
    out = []
    for e in p.manifest["docs"]:
        if e["set"] not in _SETS:
            _SETS[e["set"]] = load_set(e["set"])
        cs = _SETS[e["set"]]
        d = cs.doc(e["doc_id"])
        out.append(d)
    probs = p.verify(_SETS)
    if probs:
        raise RuntimeError("pool %s verify: %s" % (pool, probs[:3]))
    return out


def openrouter_key():
    for line in open(ROOT / ".env", encoding="utf-8"):
        if line.strip().startswith("OPENROUTER_API_KEY"):
            return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise KeyError("OPENROUTER_API_KEY")


def limit_remaining():
    import urllib.request
    req = urllib.request.Request("https://openrouter.ai/api/v1/key",
                                 headers={"Authorization": "Bearer " + openrouter_key()})
    with urllib.request.urlopen(req, timeout=30) as r:
        d = json.loads(r.read().decode("utf-8"))["data"]
    return d.get("limit_remaining")


def spent():
    """USD spent by this stream, from S4/ledger.jsonl."""
    return sum((r.get("cost_usd") or 0) for r in jl(S4 / "ledger.jsonl"))
