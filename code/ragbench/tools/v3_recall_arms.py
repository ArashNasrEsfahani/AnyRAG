"""The recall experiment: does raising v3's vector reservation close the evidence-recall gap to AGENTIC2?

Written 2026-09-20 by the recall agent. Four arms over the SAME 64 dev records (pilot.jsonl, corpora A, B and C,
36 questions), each read from its own directory so the variant name `AGENTIC3` does not collide:

  v2            work/runs/core_dev_wide      AGENTIC2                   the arm every axis leans towards
  wide-default  work/runs/core_dev_wide      AGENTIC3+vectors_always    the headline v3 row (route-v1 records)
  control       work/runs/v3recall_ctl       AGENTIC3                   today's default, re-run under route-v2
  vec8          work/runs/v3recall_vec8      AGENTIC3, vec_min_slots=8  the arm under test

`control` exists because `vectors_always` was an ABLATION FLAG when the wide set ran and is the DEFAULT now
(defaults.py `vector_policy`, ROUTE_VERSION agentic3-route-v2). Comparing `vec8` against the wide row would mix
the reservation change with the route version; comparing it against `control`, which shares the wide run's call
cache, isolates it. Both control and vec8 are dev-only AGENTIC3_OPTIONS runs, which `--final` refuses outright.

Every comparison is paired over the same (question, setting) keys with the exact two-sided sign test, and the
project's noise band is quoted with it: at n = 64 a difference of fewer than 6 records (0.094) is not separated
from chance by the no-op model of V3_ABLATIONS.md.

    .venv/Scripts/python.exe -m ragbench.tools.v3_recall_arms
"""
import argparse
import math
import statistics
from pathlib import Path

from .v3_wide_report import latency_s, quant, records, scored, sign_test, wilson

ROOT = Path(__file__).resolve().parents[2]
NOISE_RECORDS = 6           # V3_ABLATIONS.md no-op model at n = 64: |delta| < 6 records is not separated
NOISE = NOISE_RECORDS / 64.0

ARMS = [
    ("v2",           ROOT / "work" / "runs" / "core_dev_wide", "AGENTIC2"),
    ("wide-default", ROOT / "work" / "runs" / "core_dev_wide", "AGENTIC3+vectors_always"),
    ("control",      ROOT / "work" / "runs" / "v3recall_ctl",  "AGENTIC3"),
    ("vec8",         ROOT / "work" / "runs" / "v3recall_vec8", "AGENTIC3"),
]


def load():
    """-> ({label: {key: strict row}}, {label: {key: key-fact row}}, {label: {key: record}}), key = (qid, setting)."""
    strict, keyfact, recs = {}, {}, {}
    for label, d, approach in ARMS:
        strict[label] = {k[1:]: v for k, v in scored(d).items() if k[0] == approach}
        keyfact[label] = {k[1:]: v for k, v in scored(d, "results_keyfacts").items() if k[0] == approach}
        recs[label] = {k[1:]: v for k, v in records(d).items() if k[0] == approach}
    return strict, keyfact, recs


def common(strict):
    keys = None
    for label in strict:
        ks = set(strict[label])
        keys = ks if keys is None else (keys & ks)
    return sorted(keys or [])


def mean(xs):
    xs = [x for x in xs if x is not None]
    return statistics.mean(xs) if xs else float("nan")


def table(strict, keyfact, recs, keys):
    print("\n## The four arms over the same %d records\n" % len(keys))
    print("| arm | strict | 95% CI | key-fact | ev-recall | ret-recall | abstain | calls | p50 s | p95 s | max s |"
          " model s |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for label, _d, _a in ARMS:
        sc = [strict[label][k]["match_correct"] for k in keys]
        kf = [keyfact[label][k]["match_correct"] for k in keys if k in keyfact[label]]
        ev = [strict[label][k]["ev_recall"] for k in keys if strict[label][k]["n_gold"]]
        rt = [strict[label][k]["ret_recall"] for k in keys if strict[label][k]["n_gold"]]
        ab = sum(1 for k in keys if strict[label][k]["abstained"])
        lat = [latency_s(recs[label][k]) for k in keys if k in recs[label]]
        calls = [recs[label][k]["llm_calls"] for k in keys if k in recs[label]]
        model = sum(float(recs[label][k].get("llm_wall_ms") or 0.0) for k in keys if k in recs[label]) / 1000.0
        lo, hi = wilson(sum(sc), len(sc))
        print("| %-12s | %.3f | %.2f-%.2f | %.3f | %.3f | %.3f | %d/%d | %.2f | %.1f | %.1f | %.1f | %.0f |"
              % (label, mean(sc), lo, hi, mean(kf), mean(ev), mean(rt), ab, len(keys),
                 mean(calls), quant(lat, 0.5), quant(lat, 0.95), max(lat), model))


def paired_binary(rows, keys, a, b, label):
    w = [k for k in keys if rows[a][k]["match_correct"] > rows[b][k]["match_correct"]]
    l = [k for k in keys if rows[a][k]["match_correct"] < rows[b][k]["match_correct"]]
    d = len(w) - len(l)
    p = sign_test(len(w), len(l))
    band = "INSIDE the noise band" if abs(d) < NOISE_RECORDS else "OUTSIDE the noise band"
    print("\n**%s** — %s wins %d, %s wins %d, ties %d; net %+d records (%+.3f), %s (|delta| < %d = %.3f); "
          "exact two-sided sign test p = %.4f (%s)."
          % (label, a, len(w), b, len(l), len(keys) - len(w) - len(l), d, d / float(len(keys)), band,
             NOISE_RECORDS, NOISE, p, "separated from chance" if p < 0.05 else "NOT separated from chance"))
    if w:
        print("  - only %s: %s" % (a, ", ".join("%s %s" % k for k in sorted(w))))
    if l:
        print("  - only %s: %s" % (b, ", ".join("%s %s" % k for k in sorted(l))))


def paired_recall(strict, keys, a, b, label):
    ks = [k for k in keys if strict[a][k]["n_gold"] and strict[b][k]["n_gold"]]
    w = [k for k in ks if strict[a][k]["ev_recall"] > strict[b][k]["ev_recall"]]
    l = [k for k in ks if strict[a][k]["ev_recall"] < strict[b][k]["ev_recall"]]
    da = mean([strict[a][k]["ev_recall"] for k in ks]) - mean([strict[b][k]["ev_recall"] for k in ks])
    p = sign_test(len(w), len(l))
    print("\n**%s (evidence recall)** — %s better on %d records, %s better on %d, equal on %d of %d; "
          "mean delta %+.3f; exact two-sided sign test p = %.4f (%s)."
          % (label, a, len(w), b, len(l), len(ks) - len(w) - len(l), len(ks), da, p,
             "separated from chance" if p < 0.05 else "NOT separated from chance"))


def by_qtype(strict, keyfact, keys):
    print("\n## Key-fact accuracy and evidence recall by question type\n")
    types = sorted({strict["v2"][k]["qtype"] for k in keys})
    print("| arm | " + " | ".join("%s (n)" % t for t in types) + " |")
    print("|---" * (len(types) + 1) + "|")
    for label, _d, _a in ARMS:
        cells = []
        for t in types:
            ks = [k for k in keys if strict[label][k]["qtype"] == t and k in keyfact[label]]
            cells.append("%.2f (%d)" % (mean([keyfact[label][k]["match_correct"] for k in ks]), len(ks))
                         if ks else "-")
        print("| %-12s | %s |" % (label, " | ".join(cells)))
    print()
    print("| arm | " + " | ".join("%s ev-recall" % t for t in types) + " |")
    print("|---" * (len(types) + 1) + "|")
    for label, _d, _a in ARMS:
        cells = []
        for t in types:
            ks = [k for k in keys if strict[label][k]["qtype"] == t and strict[label][k]["n_gold"]]
            cells.append("%.3f" % mean([strict[label][k]["ev_recall"] for k in ks]) if ks else "-")
        print("| %-12s | %s |" % (label, " | ".join(cells)))


def cache_and_cost(recs, keys):
    print("\n## Call cache and model cost\n")
    print("| arm | calls | cache hits | share | prompt tok | completion tok |")
    print("|---|---|---|---|---|---|")
    for label, _d, _a in ARMS:
        rs = [recs[label][k] for k in keys if k in recs[label]]
        calls = sum(r["llm_calls"] for r in rs)
        hits = sum((r["trace"].get("stats") or {}).get("cache_hits", 0) for r in rs)
        print("| %-12s | %d | %d | %.0f%% | %d | %d |"
              % (label, calls, hits, 100.0 * hits / max(calls, 1),
                 sum(r.get("prompt_tokens") or 0 for r in rs), sum(r.get("completion_tokens") or 0 for r in rs)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--section", default="all")
    a = ap.parse_args()
    strict, keyfact, recs = load()
    missing = [l for l in strict if not strict[l]]
    if missing:
        print("no scored records for arm(s) %s: run the arm, then ragbench.eval.aggregate" % missing)
        return 1
    keys = common(strict)
    table(strict, keyfact, recs, keys)
    print("\n## Paired comparisons (same records, exact sign test, noise band |delta| < %d records = %.3f)"
          % (NOISE_RECORDS, NOISE))
    for a_, b_, lab in (("vec8", "control", "vec8 against today's default, strict"),
                        ("vec8", "v2", "vec8 against AGENTIC2, strict"),
                        ("control", "v2", "today's default against AGENTIC2, strict"),
                        ("control", "wide-default", "route-v2 control against the wide row, strict")):
        paired_binary(strict, keys, a_, b_, lab)
    kk = [k for k in keys if all(k in keyfact[l] for l in keyfact)]
    for a_, b_, lab in (("vec8", "control", "vec8 against today's default, key-fact"),
                        ("vec8", "v2", "vec8 against AGENTIC2, key-fact"),
                        ("control", "v2", "today's default against AGENTIC2, key-fact")):
        paired_binary(keyfact, kk, a_, b_, lab)
    for a_, b_, lab in (("vec8", "control", "vec8 against today's default"),
                        ("vec8", "v2", "vec8 against AGENTIC2"),
                        ("control", "v2", "today's default against AGENTIC2")):
        paired_recall(strict, keys, a_, b_, lab)
    by_qtype(strict, keyfact, keys)
    cache_and_cost(recs, keys)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
