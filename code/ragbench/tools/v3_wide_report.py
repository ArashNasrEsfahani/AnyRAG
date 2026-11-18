"""The WIDE dev evaluation: does AGENTIC3 beat AGENTIC2, and must the vectors always be on?

Written 2026-09-20 by the decision agent, after the adversarial review of AGENTIC v3 returned FIX-FIRST. It reads
`work/runs/core_dev_wide` (stage `v3wide` of `tools/core_plan.sh`: pilot.jsonl, 36 dev questions over corpora A, B
and C, three arms) and the existing timeline directory, and prints the tables the decision is made from. It makes
no model call and changes nothing.

WHY IT EXISTS. `work/runs/core/V3_ABLATIONS.md` showed that on the 20 corpus-A dev records of stage `v3dev` a
PROVABLE no-op - `no_fastpath`, with the FAST path firing 0 times in both arms - moved dev A strict accuracy by
0.10 and rewrote 15 of 28 answer texts. At that sample size the decoding noise of the 4B chat model is wider than
almost every effect in the ablation table, so no tuning decision could be read off it. This module therefore
reports, for every comparison:

  * the paired counts (wins, losses, ties) over the SAME (question, setting) records, not two independent means;
  * the exact two-sided sign test on the discordant pairs, which is the right test for paired 0/1 outcomes and
    needs no normal approximation at n = 64;
  * a 95% Wilson interval on each arm's accuracy, so the reader sees the width of what is being compared.

A difference that the sign test does not separate from chance is reported as "not separated from chance", never
as a win. The point of the wide set is to be able to say that honestly.

LATENCY. Never `RunRecord.latency_ms`: a record replayed from the per-question call cache has a wall clock of
milliseconds. `trace.timing.query_clock_ms` (program time + the RECORDED wall time of each call) is cache-proof
and is the number the cap applies to; `wall_with_embedding_ms` adds the first-touch embedding, which is what a
person actually waits for. AGENTIC2's trace carries `query_clock_ms` and `lazy_index_ms` but not their sum, so
the sum is taken here.

    .venv/Scripts/python.exe -m ragbench.tools.v3_wide_report            # every table
    .venv/Scripts/python.exe -m ragbench.tools.v3_wide_report --section routes
"""
import argparse
import json
import math
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
WIDE = ROOT / "work" / "runs" / "core_dev_wide"
TIMELINE = ROOT / "work" / "runs" / "core_dev_pilot_timeline"

ARMS = ["AGENTIC3", "AGENTIC3+vectors_always", "AGENTIC2"]


# ---------------------------------------------------------------------------------------------------------------
# statistics
# ---------------------------------------------------------------------------------------------------------------

def sign_test(wins, losses):
    """Exact two-sided sign test on the discordant pairs. -> p.

    Under the null the two arms are equally likely to win a discordant pair, so the number of wins is
    Binomial(wins + losses, 1/2). The two-sided p is the total probability of an outcome at least as extreme.
    """
    n = wins + losses
    if n == 0:
        return 1.0
    k = min(wins, losses)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / float(2 ** n)
    return min(1.0, 2.0 * tail)


def wilson(k, n, z=1.96):
    """95% Wilson score interval for a proportion. -> (lo, hi)."""
    if n == 0:
        return (0.0, 0.0)
    p = k / float(n)
    d = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, centre - half), min(1.0, centre + half))


def pct(x):
    return "-" if x is None else "%.3f" % x


# ---------------------------------------------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------------------------------------------

def read_jsonl(path):
    if not Path(path).is_file():
        return []
    out = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out


def scored(run_dir, scoring="results"):
    """{(approach, question_id, setting): scored record} from one aggregation directory."""
    rows = read_jsonl(Path(run_dir) / scoring / "scored.jsonl")
    return {(r["approach"], r["question_id"], r["setting"]): r for r in rows if not r.get("errored")}


def records(run_dir):
    """{(approach, question_id, setting): RunRecord} for every runs_*.jsonl of a directory."""
    out = {}
    for f in sorted(Path(run_dir).glob("runs_*.jsonl")):
        for r in read_jsonl(f):
            if r.get("error"):
                continue
            out[(r["approach"], r["question_id"], r["setting"])] = r
    return out


def latency_s(rec):
    """Wall clock a person waits for: query clock + first-touch embedding, from the recorded call wall times."""
    t = (rec.get("trace") or {}).get("timing") or {}
    if "wall_with_embedding_ms" in t:
        return float(t["wall_with_embedding_ms"]) / 1000.0
    return (float(t.get("query_clock_ms") or 0.0) + float(t.get("lazy_index_ms") or 0.0)) / 1000.0


def clock_s(rec):
    """The 60 s cap's own clock: no embedding."""
    t = (rec.get("trace") or {}).get("timing") or {}
    return float(t.get("query_clock_ms") or 0.0) / 1000.0


def route_of(rec):
    st = ((rec.get("trace") or {}).get("stages") or {}).get("route") or {}
    return st.get("route") or "-"


def quant(xs, q):
    if not xs:
        return None
    xs = sorted(xs)
    if len(xs) == 1:
        return xs[0]
    i = q * (len(xs) - 1)
    lo, hi = int(math.floor(i)), int(math.ceil(i))
    return xs[lo] + (xs[hi] - xs[lo]) * (i - lo)


# ---------------------------------------------------------------------------------------------------------------
# tables
# ---------------------------------------------------------------------------------------------------------------

def accuracy_table(strict, keyfact, recs, arms):
    keys = None
    for a in arms:
        ks = {k[1:] for k in strict if k[0] == a}
        keys = ks if keys is None else (keys & ks)
    keys = sorted(keys or [])
    print("\n## Accuracy over the %d (question, setting) records every arm answered\n" % len(keys))
    print("| Arm | n | strict | 95%% CI | key-fact | calls | p50 s | p95 s | max s | model s |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for a in arms:
        sc = [strict[(a,) + k]["match_correct"] for k in keys]
        kf = [keyfact[(a,) + k]["match_correct"] for k in keys if (a,) + k in keyfact]
        lat = [latency_s(recs[(a,) + k]) for k in keys if (a,) + k in recs]
        calls = [recs[(a,) + k]["llm_calls"] for k in keys if (a,) + k in recs]
        model = sum(float(recs[(a,) + k].get("llm_wall_ms") or 0.0) for k in keys if (a,) + k in recs) / 1000.0
        lo, hi = wilson(sum(sc), len(sc))
        print("| %s | %d | %.3f | %.2f-%.2f | %.3f | %.2f | %.1f | %.1f | %.1f | %.0f |"
              % (a, len(sc), statistics.mean(sc), lo, hi, statistics.mean(kf) if kf else float("nan"),
                 statistics.mean(calls), quant(lat, 0.5), quant(lat, 0.95), max(lat), model))
    return keys


def paired(strict, keys, a, b, label):
    wins = [k for k in keys if strict[(a,) + k]["match_correct"] > strict[(b,) + k]["match_correct"]]
    losses = [k for k in keys if strict[(a,) + k]["match_correct"] < strict[(b,) + k]["match_correct"]]
    p = sign_test(len(wins), len(losses))
    verdict = "separated from chance" if p < 0.05 else "NOT separated from chance"
    print("\n**%s** — %s wins %d, %s wins %d, ties %d of %d; exact two-sided sign test p = %.3f (%s)."
          % (label, a, len(wins), b, len(losses), len(keys) - len(wins) - len(losses), len(keys), p, verdict))
    if wins:
        print("  - only %s: %s" % (a, ", ".join("%s %s" % k for k in sorted(wins))))
    if losses:
        print("  - only %s: %s" % (b, ", ".join("%s %s" % k for k in sorted(losses))))
    return {"wins": len(wins), "losses": len(losses), "p": p}


def by_corpus(strict, keys, arms):
    print("\n## Accuracy by corpus (strict)\n")
    corpora = sorted({strict[(arms[0],) + k]["corpus"] for k in keys})
    print("| Arm | " + " | ".join("%s (n)" % c for c in corpora) + " |")
    print("|---" * (len(corpora) + 1) + "|")
    for a in arms:
        cells = []
        for c in corpora:
            ks = [k for k in keys if strict[(a,) + k]["corpus"] == c]
            sc = [strict[(a,) + k]["match_correct"] for k in ks]
            cells.append("%.2f (%d)" % (statistics.mean(sc), len(sc)))
        print("| %s | %s |" % (a, " | ".join(cells)))


def by_qtype(strict, keyfact, keys, arms):
    print("\n## Key-fact accuracy by question type\n")
    types = sorted({strict[(arms[0],) + k]["qtype"] for k in keys})
    print("| Arm | " + " | ".join("%s (n)" % t for t in types) + " |")
    print("|---" * (len(types) + 1) + "|")
    for a in arms:
        cells = []
        for t in types:
            ks = [k for k in keys if strict[(a,) + k]["qtype"] == t and (a,) + k in keyfact]
            sc = [keyfact[(a,) + k]["match_correct"] for k in ks]
            cells.append("%.2f (%d)" % (statistics.mean(sc), len(sc)) if sc else "-")
        print("| %s | %s |" % (a, " | ".join(cells)))


def routes(strict, recs, keys, arms):
    print("\n## Routes taken, with accuracy and latency\n")
    print("| Arm | Route | n | share | strict | p50 s | calls |")
    print("|---|---|---|---|---|---|---|")
    for a in arms:
        if a == "AGENTIC2":
            continue
        rs = {}
        for k in keys:
            rec = recs.get((a,) + k)
            if rec is None:
                continue
            rs.setdefault(route_of(rec), []).append(k)
        for r in ["FAST", "STANDARD", "ESCALATED", "TIMELINE"]:
            ks = rs.get(r, [])
            if not ks:
                print("| %s | %s | 0 | 0.0%% | - | - | - |" % (a, r))
                continue
            sc = [strict[(a,) + k]["match_correct"] for k in ks]
            lat = [latency_s(recs[(a,) + k]) for k in ks]
            calls = [recs[(a,) + k]["llm_calls"] for k in ks]
            print("| %s | %s | %d | %.1f%% | %.2f | %.1f | %.2f |"
                  % (a, r, len(ks), 100.0 * len(ks) / len(keys), statistics.mean(sc),
                     quant(lat, 0.5), statistics.mean(calls)))


def fast_gate(recs, keys, arm="AGENTIC3"):
    print("\n## The FAST gate: where it was reached and why it refused\n")
    reached, reasons, esc = 0, {}, {}
    for k in keys:
        rec = recs.get((arm,) + k)
        if rec is None:
            continue
        d = ((rec.get("trace") or {}).get("stages") or {}).get("route") or {}
        det = d.get("detail") or {}
        if "escalate" in det:
            esc[det["escalate"].get("reason", "?")] = esc.get(det["escalate"].get("reason", "?"), 0) + 1
        f = det.get("fast")
        if f is None:
            continue
        reached += 1
        reasons[f.get("reason", "?")] = reasons.get(f.get("reason", "?"), 0) + 1
    print("Gate reached on %d of %d records. Refusals:" % (reached, len(keys)))
    for r, n in sorted(reasons.items(), key=lambda x: -x[1]):
        print("  - `%s` %d" % (r, n))
    print("\nPool test (`escalate_check`) outcomes over the same records:")
    for r, n in sorted(esc.items(), key=lambda x: -x[1]):
        print("  - `%s` %d" % (r, n))


def timeline_table():
    """The 8-question dev timeline set (pilot_timeline.jsonl) — already run for every arm, no new GPU time."""
    import csv
    path = TIMELINE / "results" / "timeline.csv"
    if not path.is_file():
        print("\n(no timeline.csv yet)")
        return
    recs = records(TIMELINE)
    print("\n## Timelines: the 8 dev timeline questions (pilot_timeline.jsonl, corpus C fa-fa)\n")
    print("| Arm | n | coverage | precision | fully correct | false trends | comparability flag | calls | p50 s |"
          " p95 s |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    rows = []
    with open(path, encoding="utf-8-sig", newline="") as fh:
        rows = [r for r in csv.DictReader(fh) if r["setting"] == "ALL" and r["approach"] in ARMS]
    for row in sorted(rows, key=lambda r: ARMS.index(r["approach"])):
        # latency from the records, never from the CSV: `latency_ms` is the wall clock of a run that may have
        # replayed its calls from the cache, and the v3 timeline rows were replayed.
        lat = [latency_s(v) for k, v in recs.items() if k[0] == row["approach"]]
        print("| %s | %s | %s | %s | %s | %s | %s | %s | %.1f | %.1f |"
              % (row["approach"], row["n"], pct(float(row["tl_coverage"])), pct(float(row["tl_precision"])),
                 pct(float(row["tl_series_correct"])), pct(float(row["tl_false_trend"])),
                 pct(float(row["tl_comparability_flag"])), row["llm_calls_mean"],
                 quant(lat, 0.5) or 0.0, quant(lat, 0.95) or 0.0))


def warm_table():
    """The warm experiment before and after the route fix of 2026-09-20, against the matching cold arm.

    The cold reference is `AGENTIC3+vectors_always` from `core_dev` / `core_dev_pilot_timeline`, because that
    variant is what the DEFAULT now is: comparing a warm arm that runs today's code against a cold arm that ran
    the database-first routing would mix the two changes.
    """
    before = records(ROOT / "work" / "runs" / "v3warm")
    after = records(ROOT / "work" / "runs" / "v3warm2")
    cold = {}
    for d in (ROOT / "work" / "runs" / "core_dev", TIMELINE):
        for k, v in records(d).items():
            if k[0] == "AGENTIC3+vectors_always":
                cold[k[1:]] = v
    scored_before = scored(ROOT / "work" / "runs" / "v3warm")
    scored_after = scored(ROOT / "work" / "runs" / "v3warm2")
    scold = {}
    for d in (ROOT / "work" / "runs" / "core_dev", TIMELINE):
        for k, v in scored(d).items():
            if k[0] == "AGENTIC3+vectors_always":
                scold[k[1:]] = v
    order = []
    path = ROOT / "work" / "runs" / "v3warm" / "order.tsv"
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split("\t")
        if len(parts) >= 5:
            order.append((parts[3], parts[4]))
    print("\n## Warm mode, before and after the route fix (28 dev records, seeded order, one per process)\n")
    print("| # | question | setting | route before | route after | warm rows after | refusal after | calls b/a |"
          " correct b/a/cold |")
    print("|---|---|---|---|---|---|---|---|---|")
    agg = {"before": [], "after": [], "cold": []}
    for i, key in enumerate(order, 1):
        b = next((v for k, v in before.items() if k[1:] == key), None)
        a = next((v for k, v in after.items() if k[1:] == key), None)
        c = cold.get(key)
        sb = next((v for k, v in scored_before.items() if k[1:] == key), None)
        sa = next((v for k, v in scored_after.items() if k[1:] == key), None)
        sc = scold.get(key)
        if a is None:
            continue
        det = ((a.get("trace") or {}).get("stages") or {}).get("route", {}).get("detail") or {}
        why = (det.get("warm") or {}).get("reason", "")
        print("| %d | %s | %s | %s | %s | %s | %s | %s/%s | %s/%s/%s |"
              % (i, key[0], key[1], route_of(b) if b else "-", route_of(a),
                 ((a.get("trace") or {}).get("stats") or {}).get("warm_rows", 0), why,
                 b["llm_calls"] if b else "-", a["llm_calls"],
                 _mark(sb), _mark(sa), _mark(sc)))
        for name, rec, sco in (("before", b, sb), ("after", a, sa), ("cold", c, sc)):
            if rec is not None and sco is not None:
                agg[name].append((rec, sco))
    print("\n| arm | n | calls mean | p50 latency s | FAST records | correct |")
    print("|---|---|---|---|---|---|")
    for name in ("before", "after", "cold"):
        rows = agg[name]
        if not rows:
            continue
        lat = [latency_s(r) for r, _s in rows]
        fast = sum(1 for r, _s in rows if route_of(r) == "FAST")
        ok = sum(_s["match_correct"] for _r, _s in rows)
        print("| %s | %d | %.2f | %.1f | %d | %.0f/%d (%.2f) |"
              % (name, len(rows), statistics.mean([r["llm_calls"] for r, _s in rows]), quant(lat, 0.5),
                 fast, ok, len(rows), ok / len(rows)))


def _mark(s):
    return "-" if s is None else ("1" if s["match_correct"] else "0")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=str(WIDE))
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--section", default="all")
    args = ap.parse_args()
    arms = [a for a in args.arms.split(",") if a]
    strict, keyfact, recs = scored(args.runs), scored(args.runs, "results_keyfacts"), records(args.runs)
    if not strict:
        print("no scored records in %s: run `core_plan.sh v3wide` then `core_plan.sh aggregate`" % args.runs)
        return 1
    keys = accuracy_table(strict, keyfact, recs, arms)
    if args.section in ("all", "paired"):
        print("\n## Paired comparisons (same records, exact sign test)")
        if len(arms) > 2:
            paired(strict, keys, arms[0], arms[2], "v3 against v2, strict")
            paired(strict, keys, arms[1], arms[2], "v3+vectors_always against v2, strict")
            paired(strict, keys, arms[1], arms[0], "vectors_always against v3 as built, strict")
            kk = [k for k in keys if (arms[0],) + k in keyfact and (arms[2],) + k in keyfact]
            paired(keyfact, kk, arms[0], arms[2], "v3 against v2, key-fact")
            paired(keyfact, kk, arms[1], arms[0], "vectors_always against v3 as built, key-fact")
    if args.section in ("all", "corpus"):
        by_corpus(strict, keys, arms)
        by_qtype(strict, keyfact, keys, arms)
    if args.section in ("all", "routes"):
        routes(strict, recs, keys, arms)
        fast_gate(recs, keys)
    if args.section in ("all", "timeline"):
        timeline_table()
    if args.section in ("all", "warm"):
        warm_table()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
