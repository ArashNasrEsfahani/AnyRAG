"""Where the 0.08 of evidence recall between AGENTIC2 and AGENTIC3 actually goes, from the recorded runs alone.

No model call, no replay: this reads `work/runs/core_dev_wide`'s three JSONL files and the gold units of
pilot.jsonl and decomposes the difference per record.

  ret_recall   share of the question's gold units on the candidates the approach SELECTED (before the answer pack)
  ev_recall    ... on the units that reached the answer prompt

The point of separating them: a gap that is already present at `ret_recall` is a retrieval or selection problem,
and a gap that opens between `ret_recall` and `ev_recall` is a packing problem (the 6000-char evidence budget,
or the verifier's ordering). They have different fixes and only one of them is about the candidate cap.

    .venv/Scripts/python.exe -m ragbench.tools.v3_recall_gap
"""
import json
import math
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
WIDE = ROOT / "work" / "runs" / "core_dev_wide"
QUESTIONS = ROOT / "ragbench" / "data" / "questions" / "pilot.jsonl"

ARMS = {
    "AGENTIC3": "runs_AGENTIC3.jsonl",
    "AGENTIC3+vectors_always": "runs_AGENTIC3_vectors_always-7afdfb27.jsonl",
    "AGENTIC2": "runs_AGENTIC2.jsonl",
}


def sign_test(wins, losses):
    n = wins + losses
    if n == 0:
        return 1.0
    k = min(wins, losses)
    return min(1.0, 2.0 * sum(math.comb(n, i) for i in range(0, k + 1)) / float(2 ** n))


def load():
    qs = {}
    for line in QUESTIONS.open(encoding="utf-8"):
        q = json.loads(line)
        qs[q["id"]] = q
    per = {}
    for arm, fn in ARMS.items():
        for line in (WIDE / fn).open(encoding="utf-8"):
            rec = json.loads(line)
            q = qs[rec["question_id"]]
            gold = list(dict.fromkeys((q.get("gold_units") or {}).get(rec["d_lang"]) or []))
            ret = {u for it in (rec.get("retrieved_units") or []) for u in (it.get("unit_ids") or [])}
            ev = set(rec.get("evidence_units") or [])
            per[(arm, rec["question_id"], rec["setting"])] = {
                "n_gold": len(gold),
                "ret": (len(set(gold) & (ret | ev)) / len(gold)) if gold else None,
                "ev": (len(set(gold) & ev) / len(gold)) if gold else None,
                "qtype": q.get("type"), "corpus": rec["corpus"],
                "n_cands": rec["trace"]["stats"]["candidates"],
                "abstained": bool(rec["abstained"]),
            }
    return qs, per


def main():
    qs, per = load()
    keys = sorted({k[1:] for k in per if k[0] == "AGENTIC2"})
    keys = [k for k in keys if all(("%s" % a, ) + k in per for a in ARMS)]
    gold_keys = [k for k in keys if per[("AGENTIC2",) + k]["n_gold"]]
    print("paired records: %d (%d with gold units)\n" % (len(keys), len(gold_keys)))

    print("| arm | ret_recall | ev_recall | pack loss | mean candidates |")
    print("|---|---|---|---|---|")
    for a in ARMS:
        ret = sum(per[(a,) + k]["ret"] for k in gold_keys) / len(gold_keys)
        ev = sum(per[(a,) + k]["ev"] for k in gold_keys) / len(gold_keys)
        nc = sum(per[(a,) + k]["n_cands"] for k in keys) / len(keys)
        print("| %-24s | %.3f | %.3f | %.3f | %.1f |" % (a, ret, ev, ret - ev, nc))

    for a, b in (("AGENTIC3+vectors_always", "AGENTIC2"), ("AGENTIC3", "AGENTIC2")):
        print("\n### %s against %s, evidence recall, paired over %d records" % (a, b, len(gold_keys)))
        w = [k for k in gold_keys if per[(a,) + k]["ev"] > per[(b,) + k]["ev"]]
        l = [k for k in gold_keys if per[(a,) + k]["ev"] < per[(b,) + k]["ev"]]
        print("  wins %d, losses %d, ties %d, sign test p = %.4f"
              % (len(w), len(l), len(gold_keys) - len(w) - len(l), sign_test(len(w), len(l))))
        loss = sum(per[(b,) + k]["ev"] - per[(a,) + k]["ev"] for k in l)
        gain = sum(per[(a,) + k]["ev"] - per[(b,) + k]["ev"] for k in w)
        print("  recall lost on the %d losing records: %.2f; gained on the %d winning ones: %.2f; net %.2f"
              % (len(l), loss, len(w), gain, gain - loss))
        by = defaultdict(lambda: [0.0, 0])
        for k in l:
            by[per[(b,) + k]["qtype"]][0] += per[(b,) + k]["ev"] - per[(a,) + k]["ev"]
            by[per[(b,) + k]["qtype"]][1] += 1
        print("  where it is lost, by question type:")
        for t, (s, c) in sorted(by.items(), key=lambda x: -x[1][0]):
            print("    %-16s %d records, %.2f of recall" % (t, c, s))
        nbig = [k for k in l if per[(b,) + k]["n_gold"] >= 4]
        print("  of the %d losing records, %d have 4+ gold units (%.2f of the %.2f lost)"
              % (len(l), len(nbig), sum(per[(b,) + k]["ev"] - per[(a,) + k]["ev"] for k in nbig), loss))

    print("\n### gold units per record and the cap of 12\n")
    buckets = [(1, 1), (2, 3), (4, 8), (9, 99)]
    print("| gold units | records | v2 ev_recall | v3+va ev_recall | v3+va ret_recall |")
    print("|---|---|---|---|---|")
    for lo, hi in buckets:
        ks = [k for k in gold_keys if lo <= per[("AGENTIC2",) + k]["n_gold"] <= hi]
        if not ks:
            continue
        print("| %d-%d | %d | %.3f | %.3f | %.3f |"
              % (lo, hi, len(ks),
                 sum(per[("AGENTIC2",) + k]["ev"] for k in ks) / len(ks),
                 sum(per[("AGENTIC3+vectors_always",) + k]["ev"] for k in ks) / len(ks),
                 sum(per[("AGENTIC3+vectors_always",) + k]["ret"] for k in ks) / len(ks)))


if __name__ == "__main__":
    main()
