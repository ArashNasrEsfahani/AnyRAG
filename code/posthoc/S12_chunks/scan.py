"""Scan: per test question corpus_set / doc_scope; chunk counts at 1000 (frozen index), 500 and 2000 (CPU chunker)."""
import json, time
from collections import defaultdict
import packs as PK
from ragbench.approaches import baselines as BL
from ragbench.retrieval import hybrid as H
out = {"questions": {}}
sets = defaultdict(set)
for qid in PK.qids_all():
    r = PK.frozen_records("HYBRID_6000")[qid]
    q = PK.query_for(qid, r["setting"])
    out["questions"][qid] = {"set": q.corpus_set, "scope": None if q.doc_scope in (None, "corpus") else list(q.doc_scope), "setting": r["setting"]}
    sets[q.corpus_set].update(q.doc_scope or [])
for s in sorted(sets):
    cs = PK.corpus(s)
    print(s, "docs", len(cs.doc_ids()), "scope-union docs", len(sets[s]), "frozen dir", H.index_dir_for(cs, PK.CFG))
    for size, ov in ((1000, 200), (500, 100), (2000, 400)):
        cfg = PK.CFG.copy_with({"chunking": {"chunk_chars": size, "overlap_chars": ov}})
        t = time.time(); ch = BL.build_chunks(cs, cfg, BL.DEFAULTS["chunker"])
        ins = [c for c in ch if c["doc_id"] in sets[s]]
        print("  ", size, "chunks", len(ch), "in-scope", len(ins), "in-scope chars", sum(len(H.index_text(c)) for c in ins), "%.1fs" % (time.time()-t), flush=True)
json.dump(out, open("scan.json", "w"), indent=1)
