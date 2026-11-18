"""S12 retrieval + packs, frozen code paths only (approaches/hybrid.HybridRetrieval._answer, without the answer call).

    python retrieve.py qvec               96 test question vectors via the frozen OllamaClient (HybridIndex.query_vector
                                          path: embed([normalize(query.text)])) -> qvec.npz (shared by all sizes)
    python retrieve.py search --size S    HYBRID top 40 (k=40, depth=40) and HR candidates (k=50, depth=40) -> search_<S>.jsonl
    python retrieve.py rerank             ONE GPU phase: bge-reranker-v2-m3 (frozen rerank.py, fp16 cuda) on the top 50 of
                                          every question and size -> rerank_scores.jsonl (cached; resumable)
    python retrieve.py pack --size S      HYBRID_6000 / HR_6000 packs (frozen _offer + final_pack at 6000,
                                          build_answer_messages) -> packs/<ARM>/<qid>.json, packs_index.jsonl rows;
                                          at S=1000 compares every pack with the saved frozen pack (verify_1000.json)
"""
import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np

import s12
import packs as PK
from ragbench.normalize import normalize
from ragbench.retrieval import hybrid as H
from ragbench.approaches.hybrid import HYBRID_DEFAULTS

HERE = s12.HERE
BUDGET = 6000


def jl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()] if os.path.isfile(p) else []


def questions():
    out = []
    recs = PK.frozen_records("HYBRID_6000")
    for qid in PK.qids_all():
        st = recs[qid]["setting"]
        out.append((qid, st, PK.query_for(qid, st)))
    return out


def arm(pipe, size):
    return "%s_6000" % pipe if size == 1000 else "%s_C%d_6000" % (pipe, size)


def cmd_qvec():
    from ragbench.llm import Ledger, OllamaClient
    client = OllamaClient(PK.CFG, ledger=Ledger(path=str(HERE / "ollama_ledger_query.jsonl")))
    Q = questions()
    V = []
    t0 = time.perf_counter()
    with client.ledger.scope(phase="query", approach="S12-qvec"):
        for qid, st, q in Q:
            er = client.embed([normalize(q.text or "")], purpose="query-embed")
            V.append(np.asarray(er.vectors[0], dtype=np.float32))
    np.savez(HERE / "qvec.npz", qids=np.asarray([x[0] for x in Q]), V=np.stack(V))
    rec = {"phase": "qvec", "n": len(V), "wall_s": round(time.perf_counter() - t0, 1), "nvsmi": s12.nvsmi(),
           "t": time.strftime("%Y-%m-%dT%H:%M:%S")}
    with open(HERE / "gpu_phases.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")
    print(rec)


def qvecs():
    z = np.load(HERE / "qvec.npz")
    return {str(q): z["V"][i] for i, q in enumerate(z["qids"])}


_HX = {}


def hx_of(set_name, size):
    k = (set_name, size)
    if k not in _HX:
        cfg = s12.cfg_for(size)
        _HX[k] = H.HybridIndex.load(PK.corpus(set_name), cfg, client=None, refresh_vectors=False)
        assert _HX[k].dense_complete, (set_name, size)
    return _HX[k]


def cmd_search(size):
    out = HERE / ("search_%d.jsonl" % size)
    done = {r["qid"] for r in jl(out)}
    QV = qvecs()
    for qid, st, q in questions():
        if qid in done:
            continue
        hx = hx_of(q.corpus_set, size)
        doc_ids = None if q.doc_scope in (None, "corpus") else list(q.doc_scope)
        k = max(int(HYBRID_DEFAULTS["candidates"]), int(PK.CFG.chunking["top_k"]))
        h40 = hx.search(q.text, doc_ids=doc_ids, k=k, depth=int(HYBRID_DEFAULTS["depth"]), query_vector=QV[qid])
        h50 = hx.search(q.text, doc_ids=doc_ids, k=50, depth=int(HYBRID_DEFAULTS["depth"]), query_vector=QV[qid])
        row = {"qid": qid, "size": size, "set": q.corpus_set, "chunks_in_scope": h40.diagnostics["chunks_in_scope"],
               "chunks_total": hx.n, "hyb40": [h.chunk for h in h40], "fused50": [h.chunk for h in h50],
               "hyb40_ranks": [[h.ranks.get("bm25"), h.ranks.get("dense")] for h in h40]}
        with open(out, "a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
    print("search", size, "done", len(jl(out)))


def cmd_rerank(sizes):
    os.environ.pop("CUDA_VISIBLE_DEVICES", None)          # packs.py blanks it; this is the GPU phase
    out = HERE / "rerank_scores.jsonl"
    done = {(r["size"], r["qid"]) for r in jl(out)}
    Q = {qid: q for qid, st, q in questions()}
    todo = []
    for size in sizes:
        for r in jl(HERE / ("search_%d.jsonl" % size)):
            if (size, r["qid"]) not in done:
                todo.append((size, r))
    print("rerank todo", len(todo), "nvidia-smi before:", s12.nvsmi(), flush=True)
    if not todo:
        return
    import torch
    from ragbench.retrieval import rerank as R
    rr = R.get_reranker()
    rr.load()
    print("reranker device", rr.device, "fp16", rr.fp16, "fallback", rr.fallback, "load_ms", rr.load_ms, flush=True)
    if rr.device != "cuda":
        sys.exit("reranker not on cuda (%s): the frozen HR ran fp16 cuda; stop" % rr.fallback)
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    n = 0
    for size, r in todo:
        hx = hx_of(r["set"], size)
        texts = [hx.chunks[i]["text"] for i in r["fused50"]]       # frozen rerank_hits text_fn: h.text = chunk text
        s, info = rr.score(Q[r["qid"]].text, texts)
        if info.get("fallback") or info.get("oom_this_call"):
            sys.exit("reranker fell back to CPU: %s" % info)
        with open(out, "a", encoding="utf-8") as f:
            f.write(json.dumps({"size": size, "qid": r["qid"], "chunks": r["fused50"], "scores": s,
                                "ms": info["ms"], "tokens_nonpad": info["tokens_nonpad"],
                                "tokens_padded": info["tokens_padded"]}) + "\n")
        n += 1
        if n % 48 == 0:
            print("rerank", n, "/", len(todo), "%.1fs" % (time.perf_counter() - t0), s12.nvsmi(), flush=True)
    wall = time.perf_counter() - t0
    rec = {"phase": "rerank", "n_questions": n, "wall_s": round(wall, 1),
           "peak_alloc_mb": round(torch.cuda.max_memory_allocated() / 2 ** 20, 1),
           "nvsmi_end": s12.nvsmi(), "device": rr.device, "fp16": rr.fp16, "batch_size": rr.batch_size,
           "load_ms": rr.load_ms, "sizes": sizes, "t": time.strftime("%Y-%m-%dT%H:%M:%S")}
    with open(HERE / "gpu_phases.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")
    print(rec)
    rr._model = None
    torch.cuda.empty_cache()


class Cached:
    """A reranker that returns the cached scores (same .score(question, texts) contract as rerank.py)."""

    def __init__(self, scores):
        self.s = scores

    def score(self, question, texts):
        assert len(texts) == len(self.s)
        return list(self.s), {"n": len(texts), "ms": 0.0, "cached": True}


def sha(msgs):
    return hashlib.sha256(json.dumps(msgs, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


def cmd_pack(size):
    from ragbench.retrieval import rerank as R
    S = {r["qid"]: r for r in jl(HERE / ("search_%d.jsonl" % size))}
    RS = {r["qid"]: r for r in jl(HERE / "rerank_scores.jsonl") if r["size"] == size}
    QV = qvecs()
    idx_rows, ver = [], {"HYBRID": [0, 0], "HR": [0, 0], "bad": []}
    trace_rows = []
    for qid, st, q in questions():
        hx = hx_of(q.corpus_set, size)
        cs = PK.corpus(q.corpus_set)
        doc_ids = None if q.doc_scope in (None, "corpus") else list(q.doc_scope)
        k = max(int(HYBRID_DEFAULTS["candidates"]), int(PK.CFG.chunking["top_k"]))
        res = {}
        # HYBRID (approaches/hybrid.py, rerank off)
        hits = hx.search(q.text, doc_ids=doc_ids, k=k, depth=int(HYBRID_DEFAULTS["depth"]), query_vector=QV[qid])
        assert [h.chunk for h in hits] == S[qid]["hyb40"]
        items = [h.to_item(rank=r) for r, h in enumerate(hits)]
        res["HYBRID"] = (PK.pack_offer(items, cs, BUDGET), hits)
        # HR (rerank on): top 50 fused, rerank_hits keep 40, items carry the reranker score (as in _answer)
        if qid in RS:
            fused = hx.search(q.text, doc_ids=doc_ids, k=max(int(R.RERANK_DEFAULTS["candidates"]), k),
                              depth=int(HYBRID_DEFAULTS["depth"]), query_vector=QV[qid])
            assert [h.chunk for h in fused] == RS[qid]["chunks"]
            kept, rinfo, rscores = R.rerank_hits(q.text, fused, Cached(RS[qid]["scores"]), keep=k)
            fpos = {id(h): i for i, h in enumerate(fused)}
            its = []
            for r, (h, s) in enumerate(zip(kept, rscores)):
                it = h.to_item(rank=r)
                it.score = round(float(s), 6)
                it.meta["rerank"] = round(float(s), 6)
                it.meta["rrf_rank"] = fpos[id(h)]
                its.append(it)
            res["HR"] = (PK.pack_offer(its, cs, BUDGET), kept)
        for pipe, (ev, hl) in res.items():
            msgs, npred = PK.messages_for(qid, st, ev)
            h = sha(msgs)
            a = arm(pipe, size)
            packed = [{"chunk": it.meta.get("chunk"), "doc_id": it.doc_id, "span": it.span} for it in ev.items]
            trace_rows.append({"arm": a, "qid": qid, "size": size, "packed": packed, "chars": ev.chars,
                               "n_items": ev.n_included, "top40": [x.chunk for x in hl][:40]})
            if size == 1000:
                fp = PK.frozen_pack(a, qid, st)
                ok = fp is not None and fp["messages_sha256"] == h and fp["messages"] == msgs and fp["num_predict"] == npred
                ver[pipe][0] += ok
                ver[pipe][1] += 1
                if not ok:
                    ver["bad"].append([pipe, qid])
                continue
            same = []
            for fa in ("HYBRID_6000", "HR_6000"):
                fp = PK.frozen_pack(fa, qid, st)
                if fp is not None and fp["messages_sha256"] == h:
                    same.append(fa)
            d = HERE / "packs" / a
            d.mkdir(parents=True, exist_ok=True)
            (d / (qid + ".json")).write_text(json.dumps({"qid": qid, "setting": st, "messages": msgs,
                                                          "messages_sha256": h, "num_predict": npred},
                                                         ensure_ascii=False), encoding="utf-8")
            idx_rows.append({"arm": a, "qid": qid, "setting": st, "sha": h, "chars": ev.chars, "num_predict": npred,
                             "n_items": ev.n_included, "last_item_cut": ev.last_item_cut, "same_as_frozen": same})
    keep = [r for r in jl(HERE / "pack_traces.jsonl") if r["size"] != size]
    with open(HERE / "pack_traces.jsonl", "w", encoding="utf-8") as f:
        for r in keep + trace_rows:
            f.write(json.dumps(r) + "\n")
    if size == 1000:
        (HERE / "verify_1000.json").write_text(json.dumps(ver, indent=1), encoding="utf-8")
        print("verify 1000:", ver)
        return
    names = {arm(p, size) for p in ("HYBRID", "HR")}
    keep = [r for r in jl(HERE / "packs_index.jsonl") if r["arm"] not in names]
    with open(HERE / "packs_index.jsonl", "w", encoding="utf-8") as f:
        for r in keep + idx_rows:
            f.write(json.dumps(r) + "\n")
    print("packs", size, len(idx_rows), "same-as-frozen", sum(bool(r["same_as_frozen"]) for r in idx_rows))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd")
    ap.add_argument("--size", type=int)
    ap.add_argument("--sizes", default="1000,500,2000")
    a = ap.parse_args()
    s12.below_normal()
    if a.cmd == "qvec":
        cmd_qvec()
    elif a.cmd == "search":
        cmd_search(a.size)
    elif a.cmd == "rerank":
        cmd_rerank([int(x) for x in a.sizes.split(",")])
    elif a.cmd == "pack":
        cmd_pack(a.size)
