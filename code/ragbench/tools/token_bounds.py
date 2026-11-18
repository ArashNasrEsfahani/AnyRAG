"""Token accounting with an explicit UPPER BOUND for calls whose usage is unknown (FREEZE_ASTRA_R1 fix d).

A query-phase chat call either reports its usage (ok and usage_known: prompt + completion tokens, reasoning included in
completion) or it does not (a deadline-cancelled or failed call: the provider may still have processed and billed it).
An unknown call is never counted as 0 in the upper bound and never as more than 0 in the lower bound:

    known call     lower = upper = prompt_tokens + completion_tokens
    unknown call   lower = 0
                   upper = ceil(prompt_chars x r) + completion_cap
                     r               the calibrated tokens-per-character ratio of that (model, purpose, language):
                                     the MAXIMUM observed prompt_tokens / prompt_chars over the calibration ledgers'
                                     known calls, times a safety margin (default 1.10); fallbacks (model, purpose, any
                                     language), then (model, any purpose, any language) - the maximum of each group
                     completion_cap  the call's own max_tokens (recorded on every call since 2026-10-01), else the
                                     purpose's frozen cap in the calibration file
Bounds are SUMMED PER CALL (a question with two unknown calls gets two allowances; the dev report60 expression added one
allowance per question - the generalisation bug this module replaces).
A ratio of two arms is reported as [lower_num / upper_den, upper_num / lower_den]; a gate 'ratio <= g' is PASSED when
the upper ratio <= g, FAILED when the lower ratio > g, else UNRESOLVED (never passed by treating missing usage as 0).

    python -m ragbench.tools.token_bounds calibrate --ledgers <dir> [--ledgers <dir> ...] --out calib.json
    python -m ragbench.tools.token_bounds report --runs <dir> --calib calib.json
Only kind 'chat', phase 'query' calls count (warm-up 'smoke' calls and local embeddings do not).
"""
import argparse
import glob
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

TOKEN_BOUNDS_VERSION = "token-bounds-v1"
DEFAULT_MARGIN = 1.10


def _lang(rec):
    s = rec.get("setting") or ""
    return s[:2] if len(s) >= 2 else "*"


def is_query_chat(rec):
    return rec.get("kind") == "chat" and rec.get("phase") == "query"


def is_known(rec):
    return bool(rec.get("ok")) and bool(rec.get("usage_known"))


def read_ledgers(paths):
    out = []
    for p in paths:
        p = Path(p)
        files = [p] if p.is_file() else sorted(Path(x) for x in glob.glob(str(p / "**" / "ledger_*.jsonl"),
                                                                         recursive=True))
        for f in files:
            for line in open(f, encoding="utf-8"):
                if line.strip():
                    out.append(json.loads(line))
    return out


def calibrate(records, margin=DEFAULT_MARGIN, caps=None):
    """-> calibration dict from known query chat calls: max prompt_tokens / prompt_chars per key x margin."""
    mx, n = defaultdict(float), defaultdict(int)
    cap_seen = defaultdict(int)
    for r in records:
        if not is_query_chat(r) or not is_known(r):
            continue
        pc, pt = r.get("prompt_chars") or 0, r.get("prompt_tokens") or 0
        if pc <= 0 or pt <= 0:
            continue
        ratio = pt / pc
        m, pu, lg = r.get("model"), r.get("purpose"), _lang(r)
        for k in ("%s|%s|%s" % (m, pu, lg), "%s|%s|*" % (m, pu), "%s|*|*" % m):
            mx[k] = max(mx[k], ratio)
            n[k] += 1
        if r.get("max_tokens"):
            cap_seen[pu] = max(cap_seen[pu], int(r["max_tokens"]))
    return {"version": TOKEN_BOUNDS_VERSION, "margin": margin,
            "observed_max_ratio": {k: round(v, 6) for k, v in sorted(mx.items())},
            "ratios": {k: round(v * margin, 6) for k, v in sorted(mx.items())},
            "n_calls": dict(sorted(n.items())),
            "completion_caps": dict(sorted((caps or cap_seen).items()))}


def ratio_for(rec, calib):
    m, pu, lg = rec.get("model"), rec.get("purpose"), _lang(rec)
    for k in ("%s|%s|%s" % (m, pu, lg), "%s|%s|*" % (m, pu), "%s|*|*" % m):
        if k in calib["ratios"]:
            return calib["ratios"][k], k
    raise KeyError("no calibrated ratio for %s|%s|%s" % (m, pu, lg))


def call_bounds(rec, calib):
    """-> dict(lower, upper, known, rule) for one query chat call."""
    if is_known(rec):
        t = int(rec.get("prompt_tokens") or 0) + int(rec.get("completion_tokens") or 0)
        return {"lower": t, "upper": t, "known": True, "rule": "reported"}
    r, key = ratio_for(rec, calib)
    cap = rec.get("max_tokens") or (calib.get("completion_caps") or {}).get(rec.get("purpose"))
    if not cap:
        raise KeyError("no completion cap for an unknown-usage %s call" % rec.get("purpose"))
    ub = int(math.ceil((rec.get("prompt_chars") or 0) * r)) + int(cap)
    return {"lower": 0, "upper": ub, "known": False, "rule": "ceil(%d chars x %.4f [%s]) + %d" % (
        rec.get("prompt_chars") or 0, r, key, int(cap))}


def question_bounds(records, calib):
    """Sum per call over the query chat calls of ONE question/cell. -> dict(lower, upper, calls, unknown)."""
    lo = up = calls = unk = 0
    for r in records:
        if not is_query_chat(r):
            continue
        b = call_bounds(r, calib)
        lo += b["lower"]
        up += b["upper"]
        calls += 1
        unk += 0 if b["known"] else 1
    return {"lower": lo, "upper": up, "calls": calls, "unknown": unk}


def ratio_bounds(num, den):
    """num, den: dict(lower, upper) totals of two arms -> (lower ratio, upper ratio)."""
    lo = num["lower"] / den["upper"] if den["upper"] else float("inf")
    up = num["upper"] / den["lower"] if den["lower"] else float("inf")
    return lo, up


def gate(num, den, g):
    lo, up = ratio_bounds(num, den)
    return {"ratio_lower": lo, "ratio_upper": up, "gate": g,
            "status": "passed" if up <= g else "failed" if lo > g else "unresolved"}


def main(argv=None):
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("calibrate")
    c.add_argument("--ledgers", action="append", required=True)
    c.add_argument("--out", required=True)
    c.add_argument("--margin", type=float, default=DEFAULT_MARGIN)
    rp = sub.add_parser("report")
    rp.add_argument("--runs", required=True)
    rp.add_argument("--calib", required=True)
    a = ap.parse_args(argv)
    if a.cmd == "calibrate":
        cal = calibrate(read_ledgers(a.ledgers), margin=a.margin)
        cal["sources"] = a.ledgers
        Path(a.out).write_text(json.dumps(cal, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        print(json.dumps({k: cal[k] for k in ("observed_max_ratio", "completion_caps")}, ensure_ascii=False))
        return 0
    cal = json.loads(Path(a.calib).read_text(encoding="utf-8"))
    by = defaultdict(list)
    for r in read_ledgers([a.runs]):
        by[r.get("question_id")].append(r)
    tot = {"lower": 0, "upper": 0, "calls": 0, "unknown": 0}
    for q, rs in sorted(by.items(), key=lambda kv: str(kv[0])):
        b = question_bounds(rs, cal)
        for k in tot:
            tot[k] += b[k]
    print(json.dumps(tot))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
