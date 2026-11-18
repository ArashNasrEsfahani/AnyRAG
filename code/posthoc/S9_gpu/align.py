"""S9 job 1: are HYBRID's bge-m3 (Ollama 'bge-m3:latest', GGUF) and LEX5B's bge-m3 (HF BAAI/bge-m3 via the frozen
m3multi.M3Encoder, fp16 cuda) the same model in practice?

    python align.py probe        one question + one chunk through both paths, peak GPU memory
    python align.py encode       156 questions (96 test + 60 dev) and a seeded sample of 1,000 chunks, both paths
    python align.py lists        top-40 dense lists on the 96 test questions + counterfactual packs (needs encode)

Paths exactly as at query/index time in the frozen code:
  Ollama  query: OllamaClient.embed([normalize(q)]) (HybridIndex.query_vector), num_ctx 8192, truncate; L2-normalised
          chunk: OllamaClient.embed(index_text(chunk)) batches of 16 (embedding.batch_size), as the index was built
  HF      query: M3Encoder.encode([normalize(q)], max_length=128) (LEX8/LEX5B query path); also max_length 512
          chunk: M3Encoder.encode(index_text(chunk), max_length=512, batch 16) (the lex_research chunk artifact)
Stored vectors are compared too: HybridIndex E (Ollama, from the shared cache) and lex8/idx/chunks/<set>.npz (HF f16).
Outputs: align_vectors.npz, align_results.json, align_lists.jsonl, align_stdout.log (via the caller).
"""
import gpuenv  # noqa: F401

import json
import random
import sys
import time
from pathlib import Path

import numpy as np

from ragbench.config import load_config
from ragbench.normalize import normalize
from ragbench.retrieval import hybrid as H

HERE = Path(__file__).resolve().parent
WORK = HERE.parent.parent
LEX8_CHUNKS = WORK / "datasets_v2" / "lex8" / "idx" / "chunks"
SEED, N_CHUNKS = 0, 1000
cfg = load_config()

sys.path.insert(0, str(HERE))
import hybrid_agent as HA  # noqa: E402  (question loading / make_query, S3's code)


def qlist():
    out = []
    for split in ("test", "dev"):
        for q, pid in HA.load_split(split, cfg):
            query = HA.make_query(q, pid, cfg)
            out.append((split, pid, query))
    return out


_HXS, _LX = {}, {}


def hx_of(name):
    if name not in _HXS:
        _HXS[name] = H.HybridIndex.load(HA.corpus_set(name, cfg), cfg, client=None, refresh_vectors=False)
    return _HXS[name]


def lex_dense(name):
    if name not in _LX:
        _LX[name] = np.load(LEX8_CHUNKS / ("%s.npz" % name))["dense"].astype(np.float32)
    return _LX[name]


def l2(v):
    v = np.asarray(v, np.float32)
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.where(n > 0, n, 1.0)


def cos_rows(a, b):
    return np.sum(l2(a) * l2(b), axis=1)


def stats(x):
    x = np.asarray(x, np.float64)
    return {"n": int(x.size), "mean": round(float(x.mean()), 6), "min": round(float(x.min()), 6),
            "p01": round(float(np.percentile(x, 1)), 6), "p05": round(float(np.percentile(x, 5)), 6),
            "median": round(float(np.median(x)), 6), "max": round(float(x.max()), 6)}


def encoders():
    from ragbench.llm import Ledger, OllamaClient
    from ragbench.retrieval import m3multi as M3
    oc = OllamaClient(cfg, ledger=Ledger(path=str(HERE / "ollama_ledger.jsonl")))
    enc = M3.M3Encoder().load()
    return oc, enc


def oll(oc, texts, purpose, bs):
    with oc.ledger.scope(phase="smoke", approach="S9-align"):
        return l2(oc.embed(texts, purpose=purpose, batch_size=bs).vectors)


def sample_chunks():
    sets = sorted({q.corpus_set for _s, _p, q in qlist()})
    allc = [(s, i) for s in sets for i in range(hx_of(s).n)]
    rnd = random.Random(SEED)
    return sets, sorted(rnd.sample(allc, N_CHUNKS)), len(allc)


def probe():
    import torch
    oc, enc = encoders()
    q = qlist()[0][2]
    hx = hx_of(q.corpus_set)
    t = H.index_text(hx.chunks[0])
    torch.cuda.reset_peak_memory_stats()
    a = oll(oc, [normalize(q.text)], "s9-align-q", 1)[0]
    b = enc.encode([normalize(q.text)], max_length=128, colbert=False)[0]["dense"]
    c = oll(oc, [t], "s9-align-c", 16)[0]
    d = enc.encode([t] * 16, max_length=512, colbert=False)[0]["dense"]
    print("probe q cos %.5f  chunk cos %.5f  HF peak %s MB  nvidia-smi %s  HF device %s fp16 %s rev %s" % (
        float(a @ l2(b)), float(c @ l2(d)), gpuenv.gpu_peak_mb(), gpuenv.nvsmi(), enc.device, enc.fp16, enc.revision))


def encode():
    oc, enc = encoders()
    T0 = time.perf_counter()
    Q = qlist()
    qtexts = [normalize(q.text) for _s, _p, q in Q]
    t = time.perf_counter()
    q_oll = np.stack([oll(oc, [x], "s9-align-q", 1)[0] for x in qtexts])          # one at a time, as at query time
    t_qo = time.perf_counter() - t
    t = time.perf_counter()
    q_hf128 = np.stack([enc.encode([x], max_length=128, colbert=False)[0]["dense"] for x in qtexts])
    t_qh = time.perf_counter() - t
    q_hf512 = np.stack([enc.encode([x], max_length=512, colbert=False)[0]["dense"] for x in qtexts])
    q_tok = [len(enc.tok(x)["input_ids"]) for x in qtexts]
    sets, samp, n_all = sample_chunks()
    ctexts = [H.index_text(hx_of(s).chunks[i]) for s, i in samp]
    t = time.perf_counter()
    c_oll = oll(oc, ctexts, "s9-align-c", int(cfg.embedding["batch_size"]))
    t_co = time.perf_counter() - t
    t = time.perf_counter()
    c_hf = np.stack([r["dense"] for r in enc.encode(ctexts, max_length=512, batch_size=16, colbert=False)])
    t_ch = time.perf_counter() - t
    c_tok = [len(enc.tok(x)["input_ids"]) for x in ctexts]
    c_store_oll = np.stack([hx_of(s).E[i] for s, i in samp])
    c_store_hf = np.stack([lex_dense(s)[i] for s, i in samp])
    np.savez(HERE / "align_vectors.npz", q_oll=q_oll, q_hf128=q_hf128, q_hf512=q_hf512, c_oll=c_oll, c_hf=c_hf,
             c_store_oll=c_store_oll, c_store_hf=c_store_hf, q_tok=np.asarray(q_tok), c_tok=np.asarray(c_tok),
             samp_set=np.asarray([s for s, _ in samp]), samp_i=np.asarray([i for _, i in samp]),
             qids=np.asarray([q.question_id for _s, _p, q in Q]), qsplit=np.asarray([s for s, _p, _q in Q]))
    res = {"n_questions": len(Q), "n_test": sum(1 for s, _, _ in Q if s == "test"),
           "n_dev": sum(1 for s, _, _ in Q if s == "dev"), "chunk_population": n_all, "sets": sets,
           "seed": SEED, "n_chunks": len(samp), "hf_device": enc.device, "hf_fp16": enc.fp16,
           "hf_revision": enc.revision, "ollama_model": cfg.models["embed"], "ollama_digest": cfg.models.get("embed_digest"),
           "ollama_num_ctx": cfg.embedding["num_ctx"], "ollama_truncate": cfg.embedding["truncate"],
           "wall_s": {"q_ollama": round(t_qo, 1), "q_hf128": round(t_qh, 1), "c_ollama": round(t_co, 1),
                      "c_hf": round(t_ch, 1), "total_encode": round(time.perf_counter() - T0, 1)},
           "hf_peak_alloc_mb": gpuenv.gpu_peak_mb(), "nvidia_smi_end": gpuenv.nvsmi()}
    qt = np.asarray(q_tok)
    res["question_tokens"] = {"max": int(qt.max()), "over_128": int((qt > 128).sum())}
    ct = np.asarray(c_tok)
    res["chunk_tokens"] = {"mean": round(float(ct.mean()), 1), "max": int(ct.max()), "over_512": int((ct > 512).sum())}
    cq128 = cos_rows(q_oll, q_hf128)
    res["cos"] = {
        "questions_ollama_vs_hf128 (as deployed)": stats(cq128),
        "questions_ollama_vs_hf512": stats(cos_rows(q_oll, q_hf512)),
        "questions_test_only_ollama_vs_hf128": stats(cq128[[s == "test" for s, _, _ in Q]]),
        "chunks_fresh_ollama_vs_fresh_hf512": stats(cos_rows(c_oll, c_hf)),
        "chunks_stored_hybrid_vs_stored_lex8": stats(cos_rows(c_store_oll, c_store_hf)),
        "chunks_fresh_ollama_vs_stored_hybrid": stats(cos_rows(c_oll, c_store_oll)),
        "chunks_fresh_hf_vs_stored_lex8": stats(cos_rows(c_hf, c_store_hf)),
    }
    for lg in ("fa", "en"):
        m = np.asarray([q.d_lang == lg for _s, _p, q in Q])
        res["cos"]["questions_%s_ollama_vs_hf128" % lg] = stats(cq128[m])
        mc = np.asarray([s.endswith("_en") or s == "D_en" for s, _ in samp]) if lg == "en" else \
            np.asarray([not (s.endswith("_en") or s == "D_en") for s, _ in samp])
        res["cos"]["chunks_%s_fresh_ollama_vs_hf" % lg] = stats(cos_rows(c_oll, c_hf)[mc])
    # every stored chunk of every set (no encoding): HYBRID E vs lex8 npz
    allc = []
    for s in sets:
        allc.append(cos_rows(hx_of(s).E, lex_dense(s)))
    res["cos"]["chunks_ALL_stored_hybrid_vs_stored_lex8"] = stats(np.concatenate(allc))
    worst = np.argsort(cq128)[:5]
    res["worst_questions"] = [{"id": Q[i][2].question_id, "cos": round(float(cq128[i]), 5), "tokens": q_tok[i]}
                              for i in worst]
    (HERE / "align_results.json").write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(res, indent=1, ensure_ascii=False))


# ------------------------------------------------------------------------------------------------ lists + packs
def topk(scores, valid, k=40):
    o = np.lexsort((np.arange(len(scores)), -np.asarray(scores)))
    return [int(i) for i in o[np.asarray(valid)[o]][:k]]


def spearman(a, b):
    from scipy.stats import spearmanr
    return float(spearmanr(a, b).statistic)


def kendall(a, b):
    from scipy.stats import kendalltau
    return float(kendalltau(a, b).statistic)


def lists():
    """Per test question: the HYBRID dense channel (Ollama q . stored Ollama chunks) vs the LEX dense channel (HF q@128
    . stored HF chunks), both over the question's pool scope, top 40. Then counterfactual retrieval for the frozen
    arms with the other model's dense channel, and whether the pack (exact answer messages) would change."""
    from ragbench.contracts import build_answer_messages, pack_evidence
    from ragbench.approaches.baselines import DEFAULTS as BLD
    from ragbench.approaches.lex8 import fusion as FU
    from ragbench.approaches.lex8.index import _M3
    from ragbench.retrieval import rerank as R
    import hashlib
    z = np.load(HERE / "align_vectors.npz")
    qids = list(z["qids"])
    Q = {q.question_id: (pid, q) for s, pid, q in qlist() if s == "test"}
    TR = WORK / "datasets_v2" / "test_run"
    rr = R.get_reranker().load()
    print("reranker", rr.device, rr.fp16, gpuenv.nvsmi(), flush=True)

    def rec(pid, arm, qid):
        d = TR / "runs" / pid / arm
        f = next(d.glob("runs_*.jsonl"))
        best = None
        for line in open(f, encoding="utf-8"):
            r = json.loads(line)
            if r["question_id"] == qid and not r.get("error"):
                best = r
        return best

    PACKDIR = {"HYBRID_6000": "HYBRID", "HYBRID_15500": "HYBRID15500", "HR_6000": "HYBRID_rerank-0c0612fe",
               "HR_15500": "HR15500_rerank-dcb91474"}

    def saved_sha(arm, query):
        p = TR / "packs" / PACKDIR[arm] / ("%s__%s.json" % (query.question_id, query.setting))
        return json.loads(p.read_text(encoding="utf-8"))["messages_sha256"]

    def msha(query, ev):
        msgs = build_answer_messages(query, ev, cfg)
        CONTENT[0] = sorted("%s@%s" % (it.doc_id, (it.span or [None])[0]) for it in ev.items)
        return hashlib.sha256(json.dumps(msgs, ensure_ascii=False, sort_keys=True).encode()).hexdigest()

    CONTENT = [None]

    def offer_pack(items, corpus, budget):
        """HybridRetrieval._offer + final_pack (pack_evidence) at `budget` (the 6,000 / HR_15500 cells)."""
        k = int(cfg.chunking["top_k"])
        offered = list(items[:k])
        cap = int(BLD["max_candidates"])
        nxt = len(offered)
        pk = lambda its: pack_evidence(its, corpus, budget=budget, cfg=cfg)  # noqa: E731
        if BLD["fill_budget"]:
            while nxt < min(len(items), cap):
                p1 = pk(offered)
                if p1.last_item_cut or p1.n_included < len(offered):
                    break
                offered.append(items[nxt])
                nxt += 1
                p2 = pk(offered)
                if p2.n_included == p1.n_included:
                    offered.pop()
                    break
        return pk(offered)

    def fused_hits(hx, query, qvec, k, depth):
        doc_ids = list(query.doc_scope)
        return hx.search(query.text, doc_ids=doc_ids, k=k, depth=depth, query_vector=qvec)

    def arm_packs(hx, query, qvec, corpus):
        """-> {arm: messages sha} for HYBRID_6000/15500 and HR_6000/15500 with this query vector (and E)."""
        out, extra = {}, {}
        hits = fused_hits(hx, query, qvec, 40, 40)
        items = [h.to_item(rank=r) for r, h in enumerate(hits)]
        out["HYBRID_6000"] = msha(query, offer_pack(items, corpus, 6000))
        extra["c:HYBRID_6000"] = CONTENT[0]
        out["HYBRID_15500"] = msha(query, pack_evidence(items, corpus, budget=15500, cfg=cfg))
        extra["c:HYBRID_15500"] = CONTENT[0]
        extra["hyb40"] = [h.chunk for h in hits]
        fused = fused_hits(hx, query, qvec, 50, 40)
        kept, info, sc = R.rerank_hits(query.text, fused, rr, keep=40)
        its = []
        for r, (h, s) in enumerate(zip(kept, sc)):
            it = h.to_item(rank=r)
            it.score = round(float(s), 6)
            its.append(it)
        out["HR_6000"] = msha(query, offer_pack(its, corpus, 6000))
        extra["c:HR_6000"] = CONTENT[0]
        out["HR_15500"] = msha(query, offer_pack(its, corpus, 15500))
        extra["c:HR_15500"] = CONTENT[0]
        extra["hr50"] = [h.chunk for h in fused]
        extra["hr40"] = [h.chunk for h in kept]
        extra["rerank_ms"] = info.get("ms")
        return out, extra

    outf = HERE / "align_lists.jsonl"
    done = set()
    if outf.is_file():
        done = {json.loads(l)["qid"] for l in open(outf, encoding="utf-8") if l.strip()}
    for j, qid in enumerate(qids):
        if z["qsplit"][j] != "test" or qid in done:
            continue
        pid, query = Q[qid]
        hx = hx_of(query.corpus_set)
        corpus = HA.corpus_set(query.corpus_set, cfg)
        mask, _ = hx.scope_mask(list(query.doc_scope))
        E_l = lex_dense(query.corpus_set)
        qo, qh = z["q_oll"][j], z["q_hf128"][j]
        s_o = hx.E @ l2(qo)                      # HYBRID dense channel (query re-normalised as in hx.search)
        s_h = E_l @ np.asarray(qh, np.float32)   # LEX8/LEX5B dense channel (lex8.retrieve.chunk_channels)
        A = topk(s_o, mask & hx.has_vec)
        B = topk(s_h, mask)
        U = sorted(set(A) | set(B))
        I = [c for c in A if c in set(B)]
        row = {"qid": qid, "pool": pid, "lang": query.d_lang, "scope_chunks": int(mask.sum()),
               "overlap40": len(set(A) & set(B)), "top1_same": A[0] == B[0],
               "overlap10": len(set(A[:10]) & set(B[:10])),
               "spearman_union_scores": spearman(s_o[U], s_h[U]),
               "kendall_ranks_intersection": (kendall([A.index(c) for c in I], [B.index(c) for c in I])
                                              if len(I) > 2 else None),
               "spearman_scope_scores": spearman(s_o[mask], s_h[mask])}
        # counterfactual packs: (a) reproduction with the fresh Ollama query vector, (b) HF vectors in HYBRID / HR
        rep, ex_r = arm_packs(hx, query, qo, corpus)
        hx_hf = H.HybridIndex(hx.corpus_set, cfg, hx.index_dir, hx.chunks, hx.bm25, l2(E_l), np.ones(hx.n, bool),
                              hx.meta)
        cf, ex_c = arm_packs(hx_hf, query, np.asarray(qh, np.float32), corpus)
        row["packs"] = {}
        for arm in rep:
            sv = saved_sha(arm, query)
            row["packs"][arm] = {"repro_equal_saved": rep[arm] == sv, "hf_dense_equal_saved": cf[arm] == sv,
                                 "hf_dense_equal_repro": cf[arm] == rep[arm],
                                 "hf_dense_same_chunk_set": ex_c["c:" + arm] == ex_r["c:" + arm]}
        row["hyb40_same_order"] = ex_r["hyb40"] == ex_c["hyb40"]
        row["hyb40_overlap"] = len(set(ex_r["hyb40"]) & set(ex_c["hyb40"]))
        row["hr50_same_set"] = set(ex_r["hr50"]) == set(ex_c["hr50"])
        row["hr40_same_order"] = ex_r["hr40"] == ex_c["hr40"]
        # saved HYBRID trace order vs reproduction
        r6 = rec(pid, "HYBRID_6000", qid)
        row["hyb40_repro_equal_trace"] = [t[0] for t in r6["trace"]["top"]] == ex_r["hyb40"][:len(r6["trace"]["top"])]
        # LEX5B: F2 MaxMean top 40 with its own HF dense vs with Ollama dense (query + chunks)
        X = _M3(LEX8_CHUNKS / ("%s.npz" % query.corpus_set))
        s_bm, _ = hx.bm25_scores(query.text)
        s_bm = np.asarray(s_bm, np.float32)
        qm = enc_sparse(query.text)
        s_sp = X.sparse_scores(qm["sparse"])
        f2 = {}
        for name, dn in (("hf", s_h), ("ollama", s_o.astype(np.float32))):
            arrays = {"bm25": s_bm, "dense": dn, "sparse": s_sp}
            valid = {"bm25": mask & (s_bm > 0), "dense": mask.copy(), "sparse": mask & (s_sp > 0)}
            ls = {ch: FU.top_ids(arr, valid[ch], 40) for ch, arr in arrays.items()}
            order, _sc, _N = FU.fuse(arrays, ls, "maxmean")
            f2[name] = [int(c) for c in order[:40]]
        lx = rec(pid, "LEX5B_6000", qid)
        saved_f2 = lx["trace"]["lex5b"].get("f2_top40")
        saved_f2 = [int(c[0] if isinstance(c, list) else c) for c in saved_f2] if saved_f2 else None
        row["lex5b"] = {"repro_f2_equal_saved": saved_f2 == f2["hf"], "ollama_f2_same_order": f2["ollama"] == f2["hf"],
                        "ollama_f2_overlap40": len(set(f2["ollama"]) & set(f2["hf"])),
                        "ollama_top24_same": f2["ollama"][:24] == f2["hf"][:24],
                        "ollama_top24_same_set": set(f2["ollama"][:24]) == set(f2["hf"][:24])}
        with open(outf, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(row) + "\n")
        print(qid, "ov40", row["overlap40"], "rho %.3f" % row["spearman_union_scores"],
              {a: (v["repro_equal_saved"], v["hf_dense_equal_saved"]) for a, v in row["packs"].items()},
              "lex", row["lex5b"]["repro_f2_equal_saved"], row["lex5b"]["ollama_f2_same_order"], flush=True)
    summarize_lists()


_ENC = {}


def enc_sparse(text):
    """The LEX query's sparse weights (HF, max_length 128), needed for the F2 fusion."""
    if "e" not in _ENC:
        from ragbench.retrieval import m3multi as M3
        _ENC["e"] = M3.M3Encoder().load()
    return _ENC["e"].encode([normalize(text)], max_length=128, colbert=False)[0]


def summarize_lists():
    rows = [json.loads(l) for l in open(HERE / "align_lists.jsonl", encoding="utf-8") if l.strip()]
    res = json.loads((HERE / "align_results.json").read_text(encoding="utf-8"))
    ov = np.asarray([r["overlap40"] for r in rows])
    out = {"n": len(rows), "overlap40": {"mean": round(float(ov.mean()), 2), "min": int(ov.min()),
                                         "median": float(np.median(ov)), "mean_frac": round(float(ov.mean() / 40), 3),
                                         "n_identical_sets": int((ov == 40).sum())},
           "overlap10_mean": round(float(np.mean([r["overlap10"] for r in rows])), 2),
           "top1_same": sum(r["top1_same"] for r in rows),
           "spearman_union_scores": stats([r["spearman_union_scores"] for r in rows]),
           "kendall_ranks_intersection": stats([r["kendall_ranks_intersection"] for r in rows
                                                if r["kendall_ranks_intersection"] is not None]),
           "spearman_scope_scores": stats([r["spearman_scope_scores"] for r in rows])}
    for lg in ("fa", "en"):
        o = [r["overlap40"] for r in rows if r["lang"] == lg]
        out["overlap40_mean_" + lg] = round(float(np.mean(o)), 2)
    packs = {}
    for arm in rows[0]["packs"]:
        packs[arm] = {k: sum(r["packs"][arm][k] for r in rows) for k in rows[0]["packs"][arm]}
        packs[arm]["would_change_vs_saved"] = sum(not r["packs"][arm]["hf_dense_equal_saved"] for r in rows)
        packs[arm]["would_change_chunk_set"] = sum(not r["packs"][arm]["hf_dense_same_chunk_set"] for r in rows)
        packs[arm]["would_change_order_only"] = packs[arm]["would_change_vs_saved"] - packs[arm]["would_change_chunk_set"]
    out["packs_hybrid_hr_with_hf_dense"] = packs
    out["hyb40_same_order"] = sum(r["hyb40_same_order"] for r in rows)
    out["hyb40_overlap_mean"] = round(float(np.mean([r["hyb40_overlap"] for r in rows])), 2)
    out["hyb40_repro_equal_trace"] = sum(r["hyb40_repro_equal_trace"] for r in rows)
    out["hr50_same_set"] = sum(r["hr50_same_set"] for r in rows)
    out["hr40_same_order"] = sum(r["hr40_same_order"] for r in rows)
    out["lex5b_f2"] = {k: sum(r["lex5b"][k] for r in rows) for k in rows[0]["lex5b"] if k != "ollama_f2_overlap40"}
    out["lex5b_f2"]["ollama_f2_overlap40_mean"] = round(float(np.mean([r["lex5b"]["ollama_f2_overlap40"] for r in rows])), 2)
    res["lists"] = out
    (HERE / "align_results.json").write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    t0 = time.perf_counter()
    {"probe": probe, "encode": encode, "lists": lists, "summary": summarize_lists}[sys.argv[1]]()
    print("wall %.1f s" % (time.perf_counter() - t0))
