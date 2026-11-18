"""Build-path checks at size 1000 (does the index path used for 500/2000 rebuild the frozen 1000 index?):
(a) chunks via BL.build_chunks with a copy_with() config: chunks fingerprint == frozen chunks_meta, per set;
(b) BM25 tokens of the frozen index text: vocab size equal to the frozen bm25_meta (CPU, no files written);
(c) 64 seeded frozen chunks re-embedded through the frozen OllamaClient at batch 16: max |diff| to the stored vectors.
-> check_build.json"""
import json
import random

import numpy as np

import s12
import packs as PK
from ragbench.approaches import baselines as BL
from ragbench.retrieval import hybrid as H
from ragbench.llm import Ledger, OllamaClient
from ragbench.util import read_json

out = {"sets": {}}
cfgx = PK.CFG.copy_with({"chunking": {"chunk_chars": 1000, "overlap_chars": 200},
                         "paths": {"index_root": str(s12.HERE / "idx_check_unused")}})
samp = []
for s in s12.SETS:
    cs = PK.corpus(s)
    d = H.index_dir_for(cs, PK.CFG)
    cm = read_json(d / "chunks_meta.json")
    bm = read_json(d / "bm25_meta.json")
    ch = BL.build_chunks(cs, cfgx, BL.DEFAULTS["chunker"])
    import bm25s
    toks = [BL.tokenize(H.index_text(c), BL.DEFAULTS["tokenizer"]) for c in ch]
    vocab = len({t for tt in toks for t in tt})
    out["sets"][s] = {"n": len(ch), "n_frozen": cm["n_chunks"],
                      "fingerprint_equal": BL.chunks_fingerprint(ch) == cm["chunks_fingerprint"],
                      "vocab": vocab, "vocab_frozen": bm["vocab"],
                      "sig_equal_ignoring_paths": H.signature(cfgx) == H.signature(PK.CFG)}
    hx = H.HybridIndex.load(cs, PK.CFG, refresh_vectors=False)
    rnd = random.Random(s)
    for i in rnd.sample(range(hx.n), 64 // len(s12.SETS) + 1):
        samp.append((H.index_text(hx.chunks[i]), hx.E[i]))
client = OllamaClient(PK.CFG, ledger=Ledger(path=str(s12.HERE / "ollama_ledger_check.jsonl")))
with client.ledger.scope(phase="smoke", approach="S12-check"):
    er = client.embed([t for t, _ in samp], purpose="s12-check", batch_size=16)
V = np.asarray(er.vectors, np.float32)
V = V / np.linalg.norm(V, axis=1, keepdims=True)
E = np.stack([e for _, e in samp])
out["reembed"] = {"n": len(samp), "max_abs_diff": float(np.abs(V - E).max()),
                  "min_cos": float(np.min(np.sum(V * E, axis=1)))}
(s12.HERE / "check_build.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
print(json.dumps(out, indent=1))
