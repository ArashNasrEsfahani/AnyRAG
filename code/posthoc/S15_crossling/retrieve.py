"""Step 3: retrieval lists, recall@40, 6,000 packs and pack reach (no LLM calls; HR reranker on GPU).

Experiment X (cross-lingual), 96 test questions, conditions orig | trans, arms BM25 | DENSE | HYB | HR:
    orig  = the frozen query (S9's saved Ollama vector); verified against the frozen HYBRID/HR packs and S10's packs
    trans = the GLM translation as query text (q_lang = other language), own pool/scope, vector = Ollama(normalize(t))
Experiment P (orthography), 45 answerable Persian test questions, conditions = variants.VARIANTS, arms
    BM25 (frozen tokenizer) | BM25RAW (plain tokenizer both sides, S7 raw_both) | BM25RAWQ (plain query tokenizer on the
    frozen index, S7 raw_query) | DENSE (Ollama(normalize(v)), the repo path) | DENSERAW (Ollama(v) as written) |
    HYB (BM25 + DENSE, the repo path) | HYBRAW (BM25RAW + DENSERAW, same RRF)
-> rows.jsonl (one line per (exp, cond, arm, qid)), packs/<ARM>_<cond>/<qid>.json for the packs that are answered.
"""
import argparse
import json
import os
import time

os.environ["HF_HOME"] = r"D:\KareOne\RAG Paper\work\models\hf"     # as S9 gpuenv.py; never the frozen tree
os.environ["HF_HUB_OFFLINE"] = "1"
import s15lib as L  # noqa: E402
from variants import VARIANTS

os.environ.pop("CUDA_VISIBLE_DEVICES", None)
OUT = L.HERE / "rows.jsonl"
SAVE = {("X", "trans", "HR"), ("X", "orig", "HR")}           # packs written to disk (HR_6000 translated is answered)
HRDIR = "HYBRID_rerank-0c0612fe"


def s10_shas():
    out = {}
    for r in L.jl(L.HERE.parent / "S10_retrieval" / "packs_index.jsonl"):
        out[(r["arm"], r["qid"])] = r["sha"]
    return out


def frozen_sha(packdir, qid, setting):
    p = L.TR / "packs" / packdir / ("%s__%s.json" % (qid, setting))
    return json.loads(p.read_text(encoding="utf-8"))["messages_sha256"] if p.is_file() else None


def row(exp, cond, arm, qid, query, items, extra=None):
    ev, msgs, npred = L.pack6000(items, query)
    sha = L.S10.sha256_msgs(msgs)
    lst = [it.meta["chunk"] for it in items]
    r = {"exp": exp, "cond": cond, "arm": arm, "qid": qid, "lang": L.lang_of(qid), "tier": L.tier_of(qid),
         "ab": L.is_ab(qid), "setting": query.setting, "query": query.text, "list40": lst[:40], "n_list": len(lst),
         "sha": sha, "chars": ev.chars, "packed": [it.meta.get("chunk") for it in ev.items]}
    if not L.is_ab(qid):
        r["recall40"] = L.recall40(qid, query.corpus_set, lst)
        r["reach"] = L.reach(qid, msgs)
    if (exp, cond, arm) in SAVE:
        d = L.HERE / "packs" / ("%s_6000_%s" % (arm, cond))
        d.mkdir(parents=True, exist_ok=True)
        (d / (qid + ".json")).write_text(json.dumps({"arm": "%s_6000_%s" % (arm, cond), "question_id": qid,
                                                     "setting": query.setting, "messages": msgs, "num_predict": npred,
                                                     "messages_sha256": sha, "evidence_chars": ev.chars},
                                                    ensure_ascii=False, indent=1), encoding="utf-8")
    r.update(extra or {})
    return r


def items_of(hits):
    return [h.to_item(rank=i) for i, h in enumerate(hits)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="X,P")
    a = ap.parse_args()
    done = {(r["exp"], r["cond"], r["arm"], r["qid"]) for r in L.jl(OUT)}
    tr = {r["qid"]: r for r in L.jl(L.HERE / "translations.jsonl") if r["direction"] == "forward" and not r.get("error")}
    rr = None
    t_rr = 0.0
    if "X" in a.exp:
        from ragbench.retrieval import rerank as R
        import torch
        rr = R.get_reranker().load()
        print("reranker", rr.device, rr.fp16, L.S10.__name__, flush=True)
        S10SHA = s10_shas()
        for qid in L.qids("test"):
            for cond in ("orig", "trans"):
                if all(("X", cond, arm, qid) in done for arm in ("BM25", "DENSE", "HYB", "HR")):
                    continue
                if cond == "orig":
                    q, qv = L.orig_query(qid), L.saved_orig_vec(qid)
                else:
                    t = tr[qid]["translation"]
                    q, qv = L.swapped_query(qid, t), L.vec(L.normalize(t))
                rows = []
                for arm, ch in (("BM25", ("bm25",)), ("DENSE", ("dense",)), ("HYB", ("bm25", "dense"))):
                    rows.append(row("X", cond, arm, qid, q, items_of(L.search(q, ch, qv))))
                t0 = time.perf_counter()
                its, fused50, info = L.hr_list(q, qv, rr)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                dt = time.perf_counter() - t0
                t_rr += dt
                rows.append(row("X", cond, "HR", qid, q, its, {"hr_fused50": fused50, "rerank_ms": info.get("ms"),
                                                              "rerank_wall_ms": round(dt * 1000, 1)}))
                if cond == "orig":                                    # verification against frozen packs / S10 packs
                    st = q.setting
                    for r in rows:
                        if r["arm"] == "HYB":
                            rec = L.S10.frozen_records("HYBRID_6000")[qid]
                            r["verify"] = {"list_eq_frozen": r["list40"] == [u["meta"]["chunk"] for u in rec["retrieved_units"]],
                                           "sha_eq_frozen_pack": r["sha"] == frozen_sha("HYBRID", qid, st)}
                        elif r["arm"] == "HR":
                            rec = L.S10.frozen_records("HR_6000")[qid]
                            r["verify"] = {"list_eq_frozen": r["list40"] == [u["meta"]["chunk"] for u in rec["retrieved_units"]],
                                           "sha_eq_frozen_pack": r["sha"] == frozen_sha(HRDIR, qid, st)}
                        else:
                            r["verify"] = {"sha_eq_S10_pack": r["sha"] == S10SHA.get((r["arm"] + "_6000", qid))}
                for r in rows:
                    L.append(OUT, r)
                print(qid, cond, [(r["arm"], (r.get("recall40") or {}).get("cls"), (r.get("reach") or {}).get("cls"),
                                   r.get("verify")) for r in rows], flush=True)
        print("rerank wall total %.1f s" % t_rr, flush=True)
        del rr
        torch.cuda.empty_cache()
    if "P" in a.exp:
        for qid in L.qids("test"):
            if L.lang_of(qid) != "fa" or L.is_ab(qid):
                continue
            o = L.orig_query(qid).text
            for v, f in VARIANTS.items():
                if all(("P", v, arm, qid) in done for arm in ("BM25", "BM25RAW", "BM25RAWQ", "DENSE", "DENSERAW",
                                                               "HYB", "HYBRAW")):
                    continue
                t = f(o)
                q = L.text_query(qid, t)
                vn, vr = L.vec(L.normalize(t)), L.vec(t)
                ex = {"changed": t != o}
                rows = [row("P", v, "BM25", qid, q, items_of(L.search(q, ("bm25",), None)), ex),
                        row("P", v, "DENSE", qid, q, items_of(L.search(q, ("dense",), vn)), ex),
                        row("P", v, "HYB", qid, q, items_of(L.search(q, ("bm25", "dense"), vn)), ex),
                        row("P", v, "DENSERAW", qid, q, items_of(L.search(q, ("dense",), vr)), ex)]
                with L.RawBM25(q.corpus_set, "raw_both") as hx:
                    rows.append(row("P", v, "BM25RAW", qid, q, items_of(L.search(q, ("bm25",), None, hx=hx)), ex))
                    rows.append(row("P", v, "HYBRAW", qid, q, items_of(L.search(q, ("bm25", "dense"), vr, hx=hx)), ex))
                with L.RawBM25(q.corpus_set, "raw_query") as hx:
                    rows.append(row("P", v, "BM25RAWQ", qid, q, items_of(L.search(q, ("bm25",), None, hx=hx)), ex))
                for r in rows:
                    L.append(OUT, r)
            print(qid, "P done", flush=True)


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    main()
