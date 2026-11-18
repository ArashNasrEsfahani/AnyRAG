"""Build the size-S hybrid index (chunks + BM25 on CPU; bge-m3 vectors through the frozen OllamaClient, exactly the
frozen HybridIndex.build path) into S12_chunks/idx<S>/.

    python index.py --size 500 --probe      200 sampled chunks: throughput + GPU memory
    python index.py --size 500              every set (resumable; vectors cached in idx<S>/_embed_cache + embeddings.npy)
"""
import argparse
import json
import random
import time

import s12
import packs as PK
from ragbench.llm import Ledger, OllamaClient
from ragbench.retrieval import hybrid as H
from ragbench.approaches import baselines as BL

ap = argparse.ArgumentParser()
ap.add_argument("--size", type=int, required=True)
ap.add_argument("--probe", action="store_true")
ap.add_argument("--sets", default=",".join(s12.SETS))
a = ap.parse_args()
s12.below_normal()
cfg = s12.cfg_for(a.size)
assert a.size != 1000
LOG = s12.HERE / ("index_%d.jsonl" % a.size)
client = OllamaClient(cfg, ledger=Ledger(path=str(s12.HERE / ("ollama_ledger_%d.jsonl" % a.size))))


def log(rec):
    rec["t"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")
    print(json.dumps(rec), flush=True)


SCOPE = client.ledger.scope(phase="index", approach="S12-chunks-%d" % a.size)
SCOPE.__enter__()
if a.probe:
    allt = []
    for s in s12.SETS:
        cs = PK.corpus(s)
        allt += [H.index_text(c) for c in BL.build_chunks(cs, cfg, BL.DEFAULTS["chunker"])]
    texts = random.Random(0).sample(allt, 200)
    before = s12.nvsmi()
    t0 = time.perf_counter()
    st = H.embed_missing(cfg, texts, client)
    log({"phase": "probe", "size": a.size, "n": 200, "chars": sum(map(len, texts)), "wall_s": round(time.perf_counter() - t0, 1),
         "stats": st, "nvsmi_before": before, "nvsmi_after": s12.nvsmi()})
else:
    for s in a.sets.split(","):
        cs = PK.corpus(s)
        t0 = time.perf_counter()
        for attempt in range(40):          # Ollama under shared load: transient HTTP 400 (port exhaustion); resume
            try:
                hx = H.HybridIndex.build(cs, cfg, client=client, embed=True, commit_every=8,
                                         progress=lambda d: d.get("phase") != "batch" or d["done"] % 1600 == 0 and print(d["msg"], flush=True))
                break
            except Exception as e:  # noqa: BLE001
                log({"phase": "retry", "size": a.size, "set": s, "attempt": attempt, "error": "%s: %s" % (type(e).__name__, str(e)[:200])})
                time.sleep(30)
        else:
            raise SystemExit("gave up on %s" % s)
        vm = json.loads((hx.index_dir / "vector_meta.json").read_text(encoding="utf-8"))
        log({"phase": "build", "size": a.size, "set": s, "status": hx.status(), "wall_s": round(time.perf_counter() - t0, 1),
             "embed_stats": vm.get("embed_stats"), "nvsmi": s12.nvsmi()})
