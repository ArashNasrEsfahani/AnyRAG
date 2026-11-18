"""AGENTIC2 cost report from its run records (no model calls): calls, latency, the query clock and the lazy first-touch
embedding ("lazy-index") that the paper reports as preprocessing.

    python -m ragbench.tools.agentic2_report --runs runs/core_test --runs runs/core_test_timeline [--variant AGENTIC2]
        [--out work/runs/core/AGENTIC2_COST.md]

Per corpus block (corpus x setting): n, calls mean/max, verify calls, latency p50/p95/max (whole answer(), incl. lazy
indexing), query clock p50/p95/max (lazy indexing excluded; the 60 s cap applies to it), records over 60 s.
Lazy preprocessing: documents embedded on first touch in these runs (passages, embed calls, embed wall ms), and the
cache-inclusive embedding cost of every document the variant touched (cached_cost_ms: the original batch cost of every
vector, whichever run embedded it; so documents embedded during the dev runs are counted too).
"""
import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

from ..config import load_config
from ..eval.aggregate import resolve_path
from ..util import read_jsonl, safe_name


def pct(vals, q):
    v = sorted(float(x) for x in vals if x is not None)
    if not v:
        return None
    pos = (len(v) - 1) * q / 100.0
    lo = int(math.floor(pos))
    hi = min(lo + 1, len(v) - 1)
    return v[lo] + (v[hi] - v[lo]) * (pos - lo)


def s(ms):
    return "-" if ms is None else "%.1f" % (ms / 1000.0)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", action="append", required=True)
    ap.add_argument("--variant", default="AGENTIC2")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass
    cfg = load_config()
    cap_s = float(cfg.caps.get("agentic2_max_seconds") or 60)
    recs = []
    for r in args.runs:
        d = resolve_path(r, cfg)
        for rec in read_jsonl(d / ("runs_%s.jsonl" % safe_name(args.variant))):
            if rec.get("error"):
                continue
            rec["_dir"] = d.name
            recs.append(rec)
    blocks = defaultdict(list)
    for r in recs:
        tl = (r.get("meta") or {}).get("answer_format") == "timeline"
        key = ("TL" if tl else r["corpus"], r["setting"])
        blocks[key].append(r)
        blocks[("ALL", "timeline" if tl else "single-fact")].append(r)
        blocks[("ALL", "all")].append(r)
    L = ["# %s cost report" % args.variant, "", "Runs: %s. Records without errors: %d." % (", ".join(args.runs), len(recs)), "",
         "| Block | n | Calls mean (max) | Verify calls mean | Latency p50 / p95 / max s | Query clock p50 / p95 / max s | "
         "Over %d s (clock) | Over %d s (latency) | Lazy-index s total |" % (cap_s, cap_s),
         "|---|---|---|---|---|---|---|---|---|"]
    order = sorted(blocks, key=lambda k: (k[0] == "ALL", k[0], k[1]))
    for k in order:
        rs = blocks[k]
        lat = [r["latency_ms"] for r in rs]
        clock = [((r.get("trace") or {}).get("timing") or {}).get("query_clock_ms") for r in rs]
        lazy = [((r.get("trace") or {}).get("timing") or {}).get("lazy_index_ms") or 0.0 for r in rs]
        vc = [((r.get("trace") or {}).get("stats") or {}).get("verify_calls", 0) for r in rs]
        L.append("| %s %s | %d | %.2f (%d) | %.2f | %s / %s / %s | %s / %s / %s | %d | %d | %.1f |" % (
            k[0], k[1], len(rs), sum(r["llm_calls"] for r in rs) / len(rs), max(r["llm_calls"] for r in rs),
            sum(vc) / len(rs), s(pct(lat, 50)), s(pct(lat, 95)), s(max(lat)), s(pct(clock, 50)), s(pct(clock, 95)),
            s(max(c for c in clock if c is not None)), sum(1 for c in clock if c and c > cap_s * 1000),
            sum(1 for x in lat if x > cap_s * 1000), sum(lazy) / 1000.0))
    # lazy preprocessing per document
    first, touched = {}, {}
    for r in sorted(recs, key=lambda r: r.get("ts") or ""):
        for info in ((r.get("trace") or {}).get("stages") or {}).get("lazy_index") or []:
            d = info["doc_id"]
            if info.get("cached_cost_ms") is not None:
                touched[d] = info
            if info.get("embedded_now") and d not in first:
                first[d] = info
    L += ["", "## Lazy first-touch embedding (preprocessing)", "",
          "| | Documents | Passages | Embed calls | Embed wall s | Chars |", "|---|---|---|---|---|---|"]
    fe = list(first.values())
    L.append("| embedded in these runs | %d | %d | %d | %.1f | %d |" % (
        len(fe), sum(i["embedded_now"] for i in fe), sum(i["embed_calls"] for i in fe),
        sum(i["embed_ms_now"] for i in fe) / 1000.0, sum(i.get("chars") or 0 for i in fe)))
    tv = list(touched.values())
    L.append("| every document touched, cache-inclusive cost | %d | %d | - | %.1f | %d |" % (
        len(tv), sum(i["passages"] for i in tv), sum(i["cached_cost_ms"] for i in tv) / 1000.0,
        sum(i.get("chars") or 0 for i in tv)))
    if tv:
        per_doc = [i["cached_cost_ms"] / 1000.0 for i in tv]
        L.append("")
        L.append("Per document (cache-inclusive): mean %.1f s, max %.1f s; per 100k characters: %.1f s." % (
            sum(per_doc) / len(per_doc), max(per_doc),
            100000.0 * sum(per_doc) / max(1, sum(i.get("chars") or 0 for i in tv))))
    L += ["", "Latency = the whole answer() incl. first-touch embedding; query clock = latency without the lazy-index stage "
          "(model calls at their recorded wall time); the 60 s cap applies to the clock."]
    text = "\n".join(L) + "\n"
    print(text)
    if args.out:
        Path(resolve_path(args.out, cfg)).write_text(text, encoding="utf-8")
        print("-> %s" % args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
