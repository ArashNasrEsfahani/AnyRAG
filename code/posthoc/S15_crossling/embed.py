"""Step 2 (GPU via Ollama): encode query texts with the frozen Ollama bge-m3 path (OllamaClient.embed, one text per
call, as HybridIndex.query_vector; the repo path sends normalize(text)). Cached in vectors.jsonl by sha1(text sent).

    texts: original test questions (normalize; to check against S9's saved q_oll), translated questions (normalize),
    Persian orthographic variants of the 45 answerable fa questions: normalize(variant) (repo path) and the raw
    variant as written (no normalisation).
"""
import json
import time

import numpy as np

import s15lib as L
from variants import VARIANTS
from ragbench.llm import Ledger, OllamaClient


def texts():
    out = []
    tr = {r["qid"]: r for r in L.jl(L.HERE / "translations.jsonl") if r["direction"] == "forward" and not r.get("error")}
    for qid in L.qids("test"):
        o = L.orig_query(qid).text
        out.append(("orig_norm", qid, L.normalize(o)))
        out.append(("trans_norm", qid, L.normalize(tr[qid]["translation"])))
        if L.lang_of(qid) == "fa" and not L.is_ab(qid):
            for v, f in VARIANTS.items():
                out.append(("var_norm:" + v, qid, L.normalize(f(o))))
                out.append(("var_raw:" + v, qid, f(o)))
    return out


def main():
    print(L.S10.CFG.ollama.get("cross_process_lock"))
    L.CFG.ollama["cross_process_lock"] = False        # no cross-process lock (COMMON.md GPU rules)
    oc = OllamaClient(L.CFG, ledger=Ledger(path=str(L.HERE / "ollama_ledger.jsonl")))
    have = {r["key"] for r in L.jl(L.VEC)}
    todo, seen = [], set()
    for kind, qid, t in texts():
        k = L.text_key(t)
        if k in have or k in seen:
            continue
        seen.add(k)
        todo.append((kind, qid, t, k))
    print("to embed:", len(todo), flush=True)
    t0 = time.perf_counter()
    with oc.ledger.scope(phase="query", approach="S15-embed"):
        for i, (kind, qid, t, k) in enumerate(todo):
            v = oc.embed([t], purpose="query-embed").vectors[0]
            L.append(L.VEC, {"key": k, "kind": kind, "qid": qid, "text_sent": t, "vec": [float(x) for x in v]})
    wall = time.perf_counter() - t0
    print("embedded %d in %.1f s" % (len(todo), wall), flush=True)
    # check the fresh original vectors against S9's saved q_oll
    L._vecs.cache_clear()
    cs = []
    for qid in L.qids("test"):
        a = L.vec(L.normalize(L.orig_query(qid).text))
        b = L.saved_orig_vec(qid)
        cs.append(float(a @ b / np.linalg.norm(a) / np.linalg.norm(b)))
    res = {"n_embedded_this_run": len(todo), "wall_s": round(wall, 1),
           "orig_vs_S9_saved_cosine": {"n": len(cs), "min": min(cs), "mean": float(np.mean(cs))}}
    print(json.dumps(res))
    L.append(L.HERE / "embed_runs.jsonl", res)


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    main()
