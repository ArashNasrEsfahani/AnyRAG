"""Build every new pack (no model call). -> packs/<ARM>/<qid>.json + packs_index.jsonl (one row per arm x question).

Arms
  FUSION_<B>   B in 3000, 6000, 10000, 15500, 24000: lex5b fallback = B.hybrid_intact(F2 packages of f2_top40, exact
               cost, budget B) rendered as LEX5B renders its shown packages (never clipped).
  HYBRID_<B>   B in 3000, 10000, 24000: pack_evidence over the saved HYBRID top 40 at B (lex10/control.py path).
  HR_<B>       B in 3000, 10000, 24000: HybridRetrieval._offer + final_pack at B over the saved reranked top 40
               (budget_matrix._Wide path).
  GOLD         first evidence bundle of each answerable question, unit items from the frozen offsets, budget-free.
  KNAP_6000    (arm 3) cost-aware greedy knapsack over the 40 F2 packages: score / exact char cost, whole items only.
  MMR_6000     (arm 3) MMR over the F2 packages with the saved bge-m3 chunk embeddings (cosine), whole items only.
Frozen arms (HYBRID/HR/LEX5B at 6000 and 15500) are not rebuilt here; verify.py shows they reproduce byte-for-byte.
"""
import json
import sys
from pathlib import Path

import numpy as np

import packs as PK

SWEEP = [3000, 10000, 24000]
OUT = PK.HERE / "packs"
ARM3 = "--arm3" in sys.argv


def frozen_sha_index():
    idx = {}
    for arm in PK.ARM_PACKDIR:
        for qid, r in PK.frozen_records(arm).items():
            h = (r.get("meta") or {}).get("answer_messages_sha256")
            if h and not r.get("error"):
                idx.setdefault(h, []).append(arm)
    return idx


FSHA = frozen_sha_index()


def write(arm, qid, setting, ev, extra):
    msgs, npred = PK.messages_for(qid, setting, ev)
    h = PK.sha256_msgs(msgs)
    d = OUT / arm
    d.mkdir(parents=True, exist_ok=True)
    (d / (qid + ".json")).write_text(json.dumps({"arm": arm, "question_id": qid, "setting": setting, "messages": msgs,
                                                 "num_predict": npred, "messages_sha256": h,
                                                 "evidence_chars": ev.chars}, ensure_ascii=False, indent=1),
                                     encoding="utf-8")
    row = dict(arm=arm, qid=qid, setting=setting, sha=h, num_predict=npred, chars=ev.chars, items=ev.n_included,
               offered=ev.n_offered, cut=ev.last_item_cut, same_as_frozen=FSHA.get(h, []), **extra)
    return row


# --------------------------------------------------------------- arm 3 helpers
@__import__("functools").lru_cache(maxsize=None)
def embeddings(set_name):
    d = PK.hx_chunks(set_name).index_dir
    E = np.load(d / "embeddings.npy", mmap_mode="r")
    return E


def knapsack(set_name, f2, scores, budget):
    """Greedy by score per rendered char; exact cost of the final ordered list (Coster); an item is kept only if the
    whole list still fits; represented (contained) packages are skipped by the frozen _try_add / canonical."""
    P = PK.packager(set_name)
    pks = PK.lex_packages(set_name, f2)
    cost = PK.B.Coster(P.cost)
    dens = []
    for i, p in enumerate(pks):
        c1 = P.cost(p, 2)                                   # standalone cost incl. separator (header length ~n-free)
        dens.append((-(float(scores[str(p["chunk"])]) / max(1, c1)), i))
    shown = []
    for _d, i in sorted(dens):
        nxt = PK.B._try_add(shown, [pks[i]], cost, budget)
        if nxt is not None:
            shown = nxt
    return shown, cost


def mmr(set_name, f2, scores, budget, lam=0.7):
    """MMR over the F2 packages: rel = F2 score (min-max over the 40), sim = cosine of the saved chunk embeddings;
    pick argmax lam*rel - (1-lam)*max sim to picked, add whole if the exact cost still fits (else skip)."""
    E = embeddings(set_name)
    P = PK.packager(set_name)
    pks = PK.lex_packages(set_name, f2)
    cost = PK.B.Coster(P.cost)
    V = np.asarray([E[int(c)] for c in f2], dtype=np.float32)
    V /= np.maximum(1e-9, np.linalg.norm(V, axis=1, keepdims=True))
    S = V @ V.T
    rel = np.array([float(scores[str(c)]) for c in f2])
    rel = (rel - rel.min()) / max(1e-9, rel.max() - rel.min())
    left, picked, shown = list(range(len(f2))), [], []
    while left:
        best = max(left, key=lambda i: lam * rel[i] - (1 - lam) * (max(S[i, j] for j in picked) if picked else 0.0))
        left.remove(best)
        nxt = PK.B._try_add(shown, [pks[best]], cost, budget)
        if nxt is not None:
            shown = nxt
            picked.append(best)
    return shown, cost


def main():
    L6, L15 = PK.frozen_records("LEX5B_6000"), PK.frozen_records("LEX5B_15500")
    H6, H15 = PK.frozen_records("HYBRID_6000"), PK.frozen_records("HYBRID_15500")
    R6, R15 = PK.frozen_records("HR_6000"), PK.frozen_records("HR_15500")
    rows, checks = [], {"f2_same_6000_15500": 0, "hybrid_top40_same": 0, "hr_top40_same": 0,
                        "hybrid_direct_eq_offer_6000": 0, "n": 0}
    for qid in PK.qids_all():
        l6, h6, r6 = L6[qid], H6[qid], R6[qid]
        s, setn = l6["setting"], l6["corpus_set"]
        cs = PK.corpus(setn)
        checks["n"] += 1
        checks["f2_same_6000_15500"] += l6["trace"]["lex5b"]["f2_top40"] == L15[qid]["trace"]["lex5b"]["f2_top40"]
        ids = lambda r: [d["meta"]["chunk"] for d in r["retrieved_units"]]   # noqa: E731
        checks["hybrid_top40_same"] += ids(h6) == ids(H15[qid])
        checks["hr_top40_same"] += ids(r6) == ids(R15[qid])
        hi, ri = PK.items_from_record(h6), PK.items_from_record(r6)
        a = PK.sha256_msgs(PK.messages_for(qid, s, PK.pack_hybrid_direct(hi, cs, 6000))[0])
        checks["hybrid_direct_eq_offer_6000"] += a == h6["meta"]["answer_messages_sha256"]
        st = l6["trace"]["lex5b"]
        f2 = st["f2_top40"]
        P = PK.packager(setn)
        cand_lex = sum(P.cost(p, n) for n, p in enumerate(PK.lex_packages(setn, f2), 1))
        cand_h = PK.pack_hybrid_direct(hi, cs, 10 ** 7).chars
        cand_r = PK.pack_hybrid_direct(ri, cs, 10 ** 7).chars
        if ARM3:
            for arm, fn in (("KNAP_6000", knapsack), ("MMR_6000", mmr)):
                shown, cost = fn(setn, f2, st["f2_scores"], 6000)
                ev = PK.pack_lex_chunks(setn, [p["chunk"] for p in shown], 6000)
                rows.append(write(arm, qid, s, ev, {"exact_cost": cost(shown) == ev.chars,
                                                    "chunks": [p["chunk"] for p in shown]}))
            continue
        for B in [3000, 6000, 10000, 15500, 24000]:
            fb, cost = PK.fusion_fallback(setn, f2, B)
            chunks = [p["chunk"] for p in fb]
            if B == 6000:
                assert chunks == st["fallback_chunks"]
            if B == 15500:
                assert chunks == L15[qid]["trace"]["lex5b"]["fallback_chunks"]
            ev = PK.pack_lex_chunks(setn, chunks, B)
            rows.append(write("FUSION_%d" % B, qid, s, ev, {"exact_cost": cost(fb) == ev.chars, "chunks": chunks,
                                                            "candidate_chars": cand_lex}))
        for B in SWEEP:
            rows.append(write("HYBRID_%d" % B, qid, s, PK.pack_hybrid_direct(hi, cs, B), {"candidate_chars": cand_h}))
            rows.append(write("HR_%d" % B, qid, s, PK.pack_offer(ri, cs, B), {"candidate_chars": cand_r}))
        items, note = PK.gold_items(qid, setn)
        if items is not None:
            ev = PK.pack_evidence(items, cs, budget=PK.GOLD_BUDGET, cfg=PK.CFG)
            rows.append(write("GOLD", qid, s, ev, {"note": note, "n_gold": len(items)}))
    name = "packs_index_arm3.jsonl" if ARM3 else "packs_index.jsonl"
    with open(PK.HERE / name, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(json.dumps(checks))
    (PK.HERE / ("build_checks%s.json" % ("_arm3" if ARM3 else ""))).write_text(json.dumps(checks, indent=1))
    from collections import defaultdict
    agg = defaultdict(lambda: [0, 0, 0, 0, 0])
    for r in rows:
        a = agg[r["arm"]]
        a[0] += 1
        a[1] += r["chars"]
        a[2] += bool(r["same_as_frozen"])
        a[3] += r["cut"]
        a[4] += r.get("candidate_chars", 0) < int(r["arm"].split("_")[-1]) if r["arm"][-1].isdigit() else 0
    for k, v in sorted(agg.items()):
        print("%-14s n=%d mean_chars=%.0f same_as_frozen=%d cut=%d candidates<budget=%d" % (
            k, v[0], v[1] / v[0], v[2], v[3], v[4]))


if __name__ == "__main__":
    main()
