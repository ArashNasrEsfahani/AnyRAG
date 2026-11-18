"""S13 retrieval + packs (CPU, no model calls; needs vectors/ from embed.py).

Per question (96 test; --dev: the first 5 dev questions for the smoke test):
  M3_HYBRID  reproduction of the frozen HYBRID with S9's saved Ollama bge-m3 query vector (check: pack sha == frozen)
  M3_DENSE   bge-m3 cosine only (stored HybridIndex E, same query vector)          - retrieval stats only
  E5_DENSE   e5 cosine only (HybridIndex.search(channels=("dense",)) over the e5 matrix)
  E5_HYBRID  BM25 + e5, equal-weight RRF k=60, depth 40 (HybridIndex.search with the dense matrix swapped)
All with k=40 (HYBRID_DEFAULTS), the pool's doc scope, and packed exactly as HYBRID_6000: HybridRetrieval._offer +
final_pack at 6,000 (S2a packs.pack_offer), answer messages by the frozen build_answer_messages.
Also: recall@40 of complete gold bundles (span-based: a gold item is covered when chunks of the same document in the
top 40 cover >= 90% of its character span; a bundle when all its items are).
-> packs/<ARM>/<qid>.json, packs_index.jsonl, lists.jsonl, retrieve_checks.json
"""
import json
import sys
from collections import defaultdict

import numpy as np

import s13lib as S

PK = S.PK
ARMS_NEW = ("E5_DENSE", "E5_HYBRID")
COVER = 0.9


def frozen_sha(arm, qid):
    r = PK.frozen_records(arm).get(qid)
    return r and (r.get("meta") or {}).get("answer_messages_sha256")


def search(h, query, qvec, channels=("bm25", "dense")):
    doc_ids = None if query.doc_scope in (None, "corpus") else list(query.doc_scope)
    return h.search(query.text, doc_ids=doc_ids, k=40, depth=40, channels=channels, query_vector=qvec)


def pack(query, hits):
    items = [x.to_item(rank=r) for r, x in enumerate(hits)]
    ev = PK.pack_offer(items, PK.corpus(query.corpus_set), 6000)
    msgs, npred = PK.messages_for(query.question_id, query.setting, ev) if query.question_id in PK.question_dicts() \
        else (PK.build_answer_messages(query, ev, PK.CFG), PK.answer_num_predict(query, PK.CFG))
    return ev, msgs, npred


def bundle_recall(qid, h, chunk_ids):
    """-> (complete bundle covered?, item coverage dict) over the top-40 chunk ids (span-based)."""
    v2 = PK.question_dicts()[qid]["meta"]["v2"]
    if not v2.get("evidence_bundles"):
        return None, {}
    spans = defaultdict(list)
    for c in chunk_ids:
        ch = h.chunks[int(c)]
        spans[ch["doc_id"]].append(tuple(ch["span"]))
    cov = {}
    for e in v2["evidence"]:
        a, b = int(e["start"]), int(e["end"])
        mask = np.zeros(max(1, b - a), bool)
        for x, y in spans.get(e["doc_id"], []):
            lo, hi = max(a, x), min(b, y)
            if hi > lo:
                mask[lo - a:hi - a] = True
        cov[e["id"]] = float(mask.mean())
    ok = any(all(cov[x] >= COVER for x in bun) for bun in v2["evidence_bundles"])
    return ok, cov


def main():
    dev = "--dev" in sys.argv
    Q = [(s, p, q) for s, p, q in S.qlist() if s == ("dev" if dev else "test")]
    if dev:
        seen, pick = set(), []
        for s, p, q in Q:                       # one per pool (5 of 6 pools), deterministic
            if p not in seen and len(pick) < 5:
                seen.add(p)
                pick.append((s, p, q))
        Q = pick
    e5q, m3q = S.e5_queries(), S.m3_queries()
    out_idx = S.HERE / ("packs_index_dev.jsonl" if dev else "packs_index.jsonl")
    out_lists = S.HERE / ("lists_dev.jsonl" if dev else "lists.jsonl")
    checks = defaultdict(int)
    rows, lrows = [], []
    for split, pid, query in Q:
        qid = query.question_id
        h, he = S.hx(query.corpus_set), S.hx_e5(query.corpus_set)
        L = {"M3_HYBRID": search(h, query, m3q[qid]),
             "M3_DENSE": search(h, query, m3q[qid], ("dense",)),
             "E5_DENSE": search(he, query, e5q[qid], ("dense",)),
             "E5_HYBRID": search(he, query, e5q[qid])}
        lrow = {"qid": qid, "pool": pid, "lang": query.d_lang, "split": split,
                "scope_chunks": L["E5_DENSE"].diagnostics["chunks_in_scope"]}
        for a, hits in L.items():
            ids = [x.chunk for x in hits]
            lrow[a] = ids
            if not dev:
                ok, cov = bundle_recall(qid, h, ids)
                lrow["bundle40_" + a] = ok
        # BM25 channel is shared: identical bm25 ranks in M3_HYBRID and E5_HYBRID
        rb = lambda hits: sorted((x.chunk, x.ranks.get("bm25")) for x in hits if x.ranks.get("bm25"))  # noqa: E731
        lrow["overlap40_e5_m3_dense"] = len(set(lrow["E5_DENSE"]) & set(lrow["M3_DENSE"]))
        lrow["overlap40_e5_m3_hybrid"] = len(set(lrow["E5_HYBRID"]) & set(lrow["M3_HYBRID"]))
        if not dev:
            fr = PK.frozen_records("HYBRID_6000")[qid]
            checks["n"] += 1
            checks["m3_hybrid_top_equal_trace"] += [t[0] for t in fr["trace"]["top"]] == lrow["M3_HYBRID"][:len(fr["trace"]["top"])]
            _ev, msgs, _np = pack(query, L["M3_HYBRID"])
            checks["m3_hybrid_pack_sha_equal_frozen"] += PK.sha256_msgs(msgs) == frozen_sha("HYBRID_6000", qid)
            checks["setting_equal_frozen"] += fr["setting"] == query.setting
            tb = {t[0]: t[2] for t in fr["trace"]["top"] if t[2]}
            checks["bm25_ranks_consistent"] += all(tb.get(c) in (None, r) for c, r in rb(L["E5_HYBRID"]))
        for a in ARMS_NEW:
            ev, msgs, npred = pack(query, L[a])
            sha = PK.sha256_msgs(msgs)
            d = S.HERE / ("packs_dev" if dev else "packs") / a
            d.mkdir(parents=True, exist_ok=True)
            (d / (qid + ".json")).write_text(json.dumps(
                {"arm": a, "question_id": qid, "setting": query.setting, "messages": msgs, "num_predict": npred,
                 "messages_sha256": sha, "evidence_chars": ev.chars}, ensure_ascii=False, indent=1), encoding="utf-8")
            same = [] if dev else [fa for fa in ("HYBRID_6000", "HR_6000") if frozen_sha(fa, qid) == sha]
            rows.append(dict(arm=a, qid=qid, split=split, setting=query.setting, sha=sha, num_predict=npred,
                             chars=ev.chars, items=ev.n_included, cut=ev.last_item_cut, same_as_frozen=same,
                             top40=lrow[a]))
        lrows.append(lrow)
        print(qid, {a: lrow.get("bundle40_" + a) for a in L}, lrow["overlap40_e5_m3_dense"], flush=True)
    with open(out_idx, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(out_lists, "w", encoding="utf-8") as f:
        for r in lrows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    agg = defaultdict(lambda: [0, 0, 0])
    for r in rows:
        agg[r["arm"]][0] += 1
        agg[r["arm"]][1] += r["chars"]
        agg[r["arm"]][2] += bool(r["same_as_frozen"])
    checks["arms"] = {a: {"n": v[0], "mean_chars": round(v[1] / v[0]), "pack_identical_to_frozen": v[2]}
                      for a, v in agg.items()}
    if not dev:
        S.wjson(S.HERE / "retrieve_checks.json", checks)
    print(json.dumps(checks, indent=1))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
