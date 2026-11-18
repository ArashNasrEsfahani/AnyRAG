"""S10 step 1 (no model call, no GPU): verify inputs, then build the DENSE / BM25 packs.

    python build.py verify            index files vs the frozen manifest; query vectors vs saved dense cosines;
                                      channel lists vs saved per-channel ranks; HYBRID fused list + packer vs saved packs
                                      -> verify.json, verify_rows.jsonl
    python build.py build [--split dev --ids A,B,C]
                                      -> packs/<ARM>/<qid>.json, packs_index.jsonl (dev: dev_packs/, dev_packs_index.jsonl)
"""
import argparse
import hashlib
import json
from collections import Counter

import numpy as np

import s10lib as L

SRC_ARMS = ("HYBRID_6000", "HR_6000", "HYBRID_15500", "HR_15500")


def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def verify_index_files(sets):
    man = json.loads((L.FZ / "index_manifests" / "_hybrid.json").read_text(encoding="utf-8"))["rows"]
    out = {}
    for s in sets:
        d = L.hx(s).index_dir
        name = d.name
        ref = json.loads(next((L.WORK / "indexes" / "HYBRID").glob(s + "-*/hybrid_ref.json")).read_text(encoding="utf-8"))
        rows = {k: v for k, v in man.items() if k.startswith(name + "/")}
        ok = bad = 0
        for k, v in rows.items():
            p = d.parent / k
            if p.is_file() and sha_file(p) == v["sha256"]:
                ok += 1
            else:
                bad += 1
        out[s] = {"index_dir": str(d), "same_as_HYBRID_hybrid_ref": str(d) == ref["index_dir"],
                  "manifest_files": len(rows), "sha_ok": ok, "sha_bad": bad, "n_chunks": L.hx(s).n,
                  "dense_complete": L.hx(s).dense_complete}
    return out


def verify():
    rows, agg = [], Counter()
    maxdiff = {"dense": 0.0, "bm25": 0.0}
    sets = set()
    for qid in L.qids("test"):
        q = L.query_for(qid)
        sets.add(q.corpus_set)
        X = L.hx(q.corpus_set)
        qv = L.qvecs()[qid]
        qn = qv / np.linalg.norm(qv)
        s_bm, _ = X.bm25_scores(q.text)
        dl = [h.chunk for h in L.search(q, "DENSE")]
        bl = [h.chunk for h in L.search(q, "BM25")]
        dpos = {c: i + 1 for i, c in enumerate(dl)}
        bpos = {c: i + 1 for i, c in enumerate(bl)}
        saved_d, saved_b = {}, {}
        r = {"qid": qid, "lang": q.d_lang, "set": q.corpus_set}
        for arm in SRC_ARMS:
            rec = L.frozen_records(arm)[qid]
            for u in rec["retrieved_units"]:
                m = u["meta"]
                c, rk, sc = m["chunk"], m.get("ranks") or {}, m.get("scores") or {}
                if rk.get("dense"):
                    saved_d[c] = rk["dense"]
                    dd = abs(float(X.E[c] @ qn) - sc["dense"])
                    maxdiff["dense"] = max(maxdiff["dense"], dd)
                    agg["dense_score_within_1e-5" if dd <= 1e-5 else "dense_score_off"] += 1
                if rk.get("bm25"):
                    saved_b[c] = rk["bm25"]
                    bd = abs(float(s_bm[c]) - sc["bm25"])
                    maxdiff["bm25"] = max(maxdiff["bm25"], bd)
                    agg["bm25_score_within_1e-4" if bd <= 1e-4 else "bm25_score_off"] += 1
        r["dense_ranks_checked"] = len(saved_d)
        r["dense_ranks_match"] = sum(dpos.get(c) == k for c, k in saved_d.items())
        r["bm25_ranks_checked"] = len(saved_b)
        r["bm25_ranks_match"] = sum(bpos.get(c) == k for c, k in saved_b.items())
        known = set(saved_d.values())
        t = 0
        while t + 1 in known:
            t += 1
        r["dense_saved_contiguous_depth"] = t
        r["dense_top_contiguous_equal"] = dl[:t] == [c for c, _ in sorted(saved_d.items(), key=lambda kv: kv[1])][:t]
        r["bm25_full40_known"] = all(k in set(saved_b.values()) for k in range(1, len(bl) + 1))
        r["bm25_list_len"] = len(bl)
        # HYBRID fused list and packer
        hits = L.search(q, "HYB")
        fused = [h.chunk for h in hits]
        rec6 = L.frozen_records("HYBRID_6000")[qid]
        r["fused40_equal_saved"] = fused == [u["meta"]["chunk"] for u in rec6["retrieved_units"]]
        r["fused40_equal_trace_top"] = fused == [t_[0] for t_ in rec6["trace"]["top"]]
        items = [h.to_item(rank=i) for i, h in enumerate(hits)]
        for b in L.BUDGETS:
            arm = "HYBRID_%d" % b
            ev = L.pack(items, q.corpus_set, b)
            msgs, npred = L.messages(q, ev)
            fp = L.frozen_pack(arm, qid, q.setting)
            rec = L.frozen_records(arm)[qid]
            byte_eq = json.dumps(msgs, ensure_ascii=False) == json.dumps(fp["messages"], ensure_ascii=False)
            r["pack_%d" % b] = {"bytes_equal_pack_file": byte_eq, "sha_equal_pack_file": L.sha256_msgs(msgs) == fp["messages_sha256"],
                                "sha_equal_record": L.sha256_msgs(msgs) == rec["meta"]["answer_messages_sha256"],
                                "num_predict_equal": npred == fp["num_predict"], "chars_equal": ev.chars == rec["evidence_chars"]}
        rows.append(r)
        print(qid, r["dense_ranks_match"], "/", r["dense_ranks_checked"], r["bm25_ranks_match"], "/",
              r["bm25_ranks_checked"], r["fused40_equal_saved"], r["pack_6000"]["bytes_equal_pack_file"],
              r["pack_15500"]["bytes_equal_pack_file"], flush=True)
    s = {"n_questions": len(rows),
         "dense_ranks": [sum(r["dense_ranks_match"] for r in rows), sum(r["dense_ranks_checked"] for r in rows)],
         "bm25_ranks": [sum(r["bm25_ranks_match"] for r in rows), sum(r["bm25_ranks_checked"] for r in rows)],
         "questions_all_dense_ranks_match": sum(r["dense_ranks_match"] == r["dense_ranks_checked"] for r in rows),
         "questions_all_bm25_ranks_match": sum(r["bm25_ranks_match"] == r["bm25_ranks_checked"] for r in rows),
         "dense_saved_contiguous_depth": {"mean": round(float(np.mean([r["dense_saved_contiguous_depth"] for r in rows])), 1),
                                          "min": min(r["dense_saved_contiguous_depth"] for r in rows),
                                          "n_full40": sum(r["dense_saved_contiguous_depth"] >= 40 for r in rows)},
         "dense_top_contiguous_equal": sum(r["dense_top_contiguous_equal"] for r in rows),
         "bm25_full_list_known": sum(r["bm25_full40_known"] for r in rows),
         "score_checks": dict(agg), "max_abs_diff": maxdiff,
         "fused40_equal_saved": sum(r["fused40_equal_saved"] for r in rows),
         "fused40_equal_trace_top": sum(r["fused40_equal_trace_top"] for r in rows)}
    for b in L.BUDGETS:
        k = "pack_%d" % b
        s[k] = {x: sum(r[k][x] for r in rows) for x in rows[0][k]}
    s["index_files"] = verify_index_files(sorted(sets))
    with open(L.HERE / "verify_rows.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    (L.HERE / "verify.json").write_text(json.dumps(s, indent=1), encoding="utf-8")
    print(json.dumps(s, indent=1))


def frozen_sha_index():
    idx = {}
    for arm in SRC_ARMS:
        for qid, r in L.frozen_records(arm).items():
            h = (r.get("meta") or {}).get("answer_messages_sha256")
            if h and not r.get("error"):
                idx.setdefault(h, []).append(arm)
    return idx


def build(split, ids):
    out = L.HERE / ("packs" if split == "test" else "dev_packs")
    fsha = frozen_sha_index() if split == "test" else {}
    rows = []
    for qid in ids:
        q = L.query_for(qid, split)
        for ch in ("DENSE", "BM25"):
            hits = L.search(q, ch)
            items = [h.to_item(rank=i) for i, h in enumerate(hits)]
            for b in L.BUDGETS:
                arm = "%s_%d" % (ch, b)
                ev = L.pack(items, q.corpus_set, b)
                msgs, npred = L.messages(q, ev)
                h = L.sha256_msgs(msgs)
                d = out / arm
                d.mkdir(parents=True, exist_ok=True)
                (d / (qid + ".json")).write_text(json.dumps(
                    {"arm": arm, "question_id": qid, "setting": q.setting, "messages": msgs, "num_predict": npred,
                     "messages_sha256": h, "evidence_chars": ev.chars}, ensure_ascii=False, indent=1), encoding="utf-8")
                rows.append(dict(arm=arm, qid=qid, setting=q.setting, sha=h, num_predict=npred, chars=ev.chars,
                                 items=ev.n_included, offered=ev.n_offered, cut=ev.last_item_cut,
                                 list40=[x.chunk for x in hits], n_list=len(hits),
                                 packed_chunks=[it.meta.get("chunk") for it in ev.items] if hasattr(ev, "items") else None,
                                 same_as_frozen=fsha.get(h, [])))
        print(qid, flush=True)
    name = "packs_index.jsonl" if split == "test" else "dev_packs_index.jsonl"
    with open(L.HERE / name, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    c = Counter()
    for r in rows:
        c[(r["arm"], "n")] += 1
        c[(r["arm"], "chars")] += r["chars"]
        c[(r["arm"], "same_as_frozen")] += bool(r["same_as_frozen"])
        c[(r["arm"], "cut")] += bool(r["cut"])
        c[(r["arm"], "short_list")] += r["n_list"] < 40
    for a in L.ARMS:
        if c[(a, "n")]:
            print("%-12s n=%d mean_chars=%.0f same_as_frozen=%d cut=%d list<40=%d" % (
                a, c[(a, "n")], c[(a, "chars")] / c[(a, "n")], c[(a, "same_as_frozen")], c[(a, "cut")],
                c[(a, "short_list")]))


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd")
    ap.add_argument("--split", default="test")
    ap.add_argument("--ids", default=None)
    a = ap.parse_args()
    if a.cmd == "verify":
        verify()
    else:
        ids = a.ids.split(",") if a.ids else L.qids(a.split)
        build(a.split, ids)
