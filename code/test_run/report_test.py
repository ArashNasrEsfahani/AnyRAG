"""STEP 4: the true-test report (PROTOCOL.md sections 5 and 9). Reads only saved records, ledgers and verdicts; no
model call. Writes TEST_RESULTS.md, test_results.json, registry rows (set "test96") and the blinded review page.

    python report_test.py [--registry]       # --registry appends the 48 reader x arm rows to RESULTS_REGISTRY.jsonl
"""
import json
import math
import os
import random
import statistics as stx
import sys
from collections import defaultdict
from pathlib import Path

# Paths (release edit, results-neutral): the original script lived next to its data (work/datasets_v2/test_run). In
# the AnyRAG repository it lives in code/test_run and reads results/test, results/protocol and code/ragbench. The
# environment variables ANYRAG_TEST_DIR, ANYRAG_FREEZE_DIR and ANYRAG_CODE_DIR override the detection.
_SELF = Path(__file__).resolve()
HERE = _SELF.parent
if os.environ.get("ANYRAG_TEST_DIR"):
    HERE = Path(os.environ["ANYRAG_TEST_DIR"]).resolve()
elif not (HERE / "runs").is_dir() and (_SELF.parents[2] / "results" / "test").is_dir():
    HERE = _SELF.parents[2] / "results" / "test"
DS = HERE.parent
FZ = Path(os.environ.get("ANYRAG_FREEZE_DIR") or (DS / "protocol" if (DS / "protocol").is_dir() else DS / "test_freeze"))
WT = os.environ.get("ANYRAG_CODE_DIR") or (str(_SELF.parents[1]) if (_SELF.parents[1] / "ragbench").is_dir()
                                           else str(_SELF.parents[4] / "RAG-Paper-upgrades"))
sys.path.insert(0, WT)
from ragbench.tools import judge_multi as JM  # noqa: E402
from ragbench.tools import token_bounds as TB  # noqa: E402

M = json.loads((FZ / "MANIFEST.json").read_text(encoding="utf-8"))
ARMS = [a["arm"] for a in M["arms"]]
VAR = {a["arm"]: a["variant"] for a in M["arms"]}
READERS = M["readers"]
PRIMARY = M["primary_reader"]
JUDGE, PIN = M["judge"]["model"], M["judge"]["provider_pin"]
CAL = json.loads((FZ / "token_calibration.json").read_text(encoding="utf-8"))
# The frozen calibration covers only the dev models (GLM + gpt-oss selectors). Failed replay answer calls of the other
# readers (HTTP 429 etc., usage not reported) need a ratio for their UPPER token bound. Fill ONLY missing keys from
# the test's own known-usage replay calls, same method and margin (TB.calibrate); frozen keys are never changed.
# Affects descriptive non-primary token bounds only (the gate uses GLM cells).
CAL_EXTENDED_KEYS = []


def _extend_cal():
    recs = []
    for f in (HERE / "replay").rglob("ledger_*.jsonl"):
        recs += [json.loads(l) for l in open(f, encoding="utf-8") if l.strip()]
    ext = TB.calibrate(recs, margin=CAL.get("margin", TB.DEFAULT_MARGIN))
    for k, v in ext["ratios"].items():
        if k not in CAL["ratios"]:
            CAL["ratios"][k] = v
            CAL_EXTENDED_KEYS.append(k)


_extend_cal()
SHORT = {"z-ai/glm-5.3-flash": "GLM-5.3-flash", "openai/gpt-oss-20b": "gpt-oss-20b",
         "qwen/qwen3-235b-a22b-2507": "Qwen3-235B-2507", "meta-llama/llama-3.3-70b-instruct": "Llama-3.3-70B",
         "google/gemma-3-27b-it": "Gemma-3-27B", "deepseek/deepseek-v4-flash": "DeepSeek-V4-flash"}


def slug(m):
    return m.replace("/", "__").replace(":", "_")


def pool_of(q):
    return "-".join(q.split("-")[2:4])


def jl(p):
    p = Path(p)
    return [json.loads(l) for l in open(p, encoding="utf-8-sig") if l.strip()] if p.is_file() else []


# ---------------------------------------------------------------- questions
QS = {}
for d in jl(FZ / "questions" / "test_v2judge_all.jsonl"):
    v2 = d["meta"]["v2"]
    QS[d["id"]] = {"lang": v2["language"], "tier": v2.get("tier") or v2.get("intended_tier"), "domain": v2["domain"],
                   "family": v2.get("source_family"), "answerable": d["answer_type"] != "unanswerable",
                   "question": v2["question"], "gloss": v2.get("question_gloss_en"),
                   "reference": v2.get("expected_answer"), "reference_gloss": v2.get("expected_answer_gloss_en")}
QIDS = list(QS)
ANS = [q for q in QIDS if QS[q]["answerable"]]
AB = [q for q in QIDS if not QS[q]["answerable"]]


# ---------------------------------------------------------------- records + verdicts
def cell_dir(reader, q, arm):
    if reader == PRIMARY:
        return HERE / "runs" / pool_of(q) / arm
    return HERE / "replay" / slug(reader) / pool_of(q) / arm


def load():
    """-> REC[reader][arm][q] = latest non-error record (or the last error record), VER[reader] = final_verdicts."""
    REC, VER = {}, {}
    for rd in READERS:
        REC[rd] = {a: {} for a in ARMS}
        for a in ARMS:
            for pool in {pool_of(q) for q in QIDS}:
                d = (HERE / "runs" / pool / a) if rd == PRIMARY else (HERE / "replay" / slug(rd) / pool / a)
                for f in sorted(d.glob("runs_*.jsonl")) if d.is_dir() else []:
                    for r in jl(f):
                        q = r["question_id"]
                        old = REC[rd][a].get(q)
                        if old is None or old.get("error") or not r.get("error"):
                            REC[rd][a][q] = r
        VER[rd] = final_verdicts_by_answer(HERE / "judge" / slug(rd))
    return REC, VER



def final_verdicts_by_answer(out_dir):
    """JM.final_verdicts keyed by (rid, answer_sha1) instead of rid. Same policy (this judge + pin only; FIRST valid
    verdict of a key is final; blinded adjudication resolves needs_adjudication). Needed because a cell whose first
    record was an infrastructure error had that empty answer judged by rule (strict 0) before its rerun's real answer
    was judged under the same rid; the rid-keyed view would return the stale empty-answer verdict. The judge's own
    skip key already includes answer_sha1, so no answer was judged twice."""
    out_dir, js = Path(out_dir), JM._slug(JUDGE)
    rows = {}
    for v in jl(out_dir / ("verdicts_%s.jsonl" % js)):
        k = (v["rid"], v["answer_sha1"])
        if v.get("judge") != JUDGE or v.get("judge_provider_pin") != PIN or k in rows:
            continue
        rows[k] = dict(status="judged", strict=v["strict"], answer_sha1=v["answer_sha1"], verdict=v)
    adj = {}
    for a in jl(out_dir / ("adjudications_%s.jsonl" % js)):
        if a.get("blinded") is True and a.get("strict") in (0, 1):
            adj.setdefault((a["rid"], a["answer_sha1"]), a)
    for n in jl(out_dir / ("needs_adjudication_%s.jsonl" % js)):
        k = (n["rid"], n["answer_sha1"])
        if n.get("judge") != JUDGE or n.get("judge_provider_pin") != PIN or k in rows:
            continue
        a = adj.get(k)
        rows[k] = (dict(status="adjudicated", strict=a["strict"], answer_sha1=n["answer_sha1"], adjudication=a)
                   if a is not None else dict(status="needs_adjudication", strict=None, answer_sha1=n["answer_sha1"]))
    return rows


def score(REC, VER, rd, a, q):
    """-> (lower, upper, status). Missing record or error = system failure 0; needs_adjudication = [0, 1]."""
    r = REC[rd][a].get(q)
    if r is None:
        return 0, 0, "missing"
    if r.get("error"):
        return 0, 0, "system_failure"
    rid = r["meta"]["identity"]["rid"]
    v = VER[rd].get((rid, JM.answer_sha1(r)))
    if v is None:
        return 0, 1, "unjudged"
    if v["status"] == "needs_adjudication":
        return 0, 1, "needs_adjudication"
    return v["strict"], v["strict"], v["status"]


def sign_p(b, w):
    n = b + w
    if n == 0:
        return 1.0
    k = min(b, w)
    p = sum(math.comb(n, i) for i in range(0, k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def holm(ps):
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    adj, run = [0.0] * len(ps), 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (len(ps) - rank) * ps[i]))
        adj[i] = run
    return adj


def boot_ci(diffs, n=10000, seed=0, clusters=None):
    rnd = random.Random(seed)
    if clusters is None:
        idx = list(range(len(diffs)))
        sims = []
        for _ in range(n):
            s = [diffs[rnd.choice(idx)] for _ in idx]
            sims.append(sum(s))
    else:
        groups = defaultdict(list)
        for d, c in zip(diffs, clusters):
            groups[c].append(d)
        keys = list(groups)
        sims = []
        for _ in range(n):
            s = 0
            for _k in keys:
                s += sum(groups[rnd.choice(keys)])
            sims.append(s)
    sims.sort()
    return sims[int(0.025 * n)], sims[int(0.975 * n) - 1]


def pct(xs, p):
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return None
    k = (len(xs) - 1) * p
    lo, hi = math.floor(k), math.ceil(k)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


# ---------------------------------------------------------------- tokens and latency
def ledgers(d):
    out = []
    for f in Path(d).glob("ledger_*.jsonl"):
        out += jl(f)
    return out


def primary_calls(q, a):
    return [c for c in ledgers(HERE / "runs" / pool_of(q) / a) if c.get("question_id") == q
            and TB.is_query_chat(c)]


def token_cell(REC, rd, a, q):
    """(lower, upper, unknown) query-LLM tokens of one cell. Non-primary = shared selector calls + own answer."""
    pc = primary_calls(q, a)
    shared = [c for c in pc if c.get("purpose") != "answer"]
    if rd == PRIMARY:
        calls = pc
    else:
        own = [c for c in ledgers(cell_dir(rd, q, a)) if c.get("question_id") == q and TB.is_query_chat(c)]
        calls = shared + own
    b = TB.question_bounds(calls, CAL) if calls else {"lower": 0, "upper": 0, "unknown": 0}
    return b["lower"], b["upper"], b["unknown"]


def latency_cell(REC, rd, a, q):
    pr = REC[PRIMARY][a].get(q)
    if pr is None or pr.get("error"):
        return None
    if rd == PRIMARY:
        return pr["latency_ms"]
    r = REC[rd][a].get(q)
    if r is None or r.get("error"):
        return None
    ans = [c for c in primary_calls(q, a) if c.get("purpose") == "answer" and c.get("ok")]
    pre = pr["latency_ms"] - (ans[-1].get("wall_total_ms") or 0) if ans else pr["latency_ms"]
    return pre + r["latency_ms"]


def selector_stats(REC, a):
    out = {"n": 0, "fail": 0, "reasons": defaultdict(int), "all_none": 0, "unchanged_pack": 0, "unchanged_set": 0}
    for q in QIDS:
        r = REC[PRIMARY][a].get(q)
        if r is None or r.get("error"):
            continue
        t = r.get("trace") or {}
        out["n"] += 1
        if "lex5b" in t:
            lx = t["lex5b"]
            s = lx.get("selector") or {}
            if s.get("status") == "all_none":
                out["all_none"] += 1
            elif s.get("status") != "ok":
                out["fail"] += 1
                out["reasons"][str(s.get("reason"))] += 1
            out["unchanged_pack"] += lx.get("shown_chunks") == lx.get("fallback_chunks")
            out["unchanged_set"] += set(lx.get("shown_chunks") or []) == set(lx.get("fallback_chunks") or [])
        elif "setr" in t:
            s = t["setr"].get("selector") or {}
            if s.get("fallback"):
                out["fail"] += 1
                out["reasons"][str(s.get("reason"))] += 1
            sel = s.get("selected") or []
            same = bool(s.get("fallback")) or sel == list(range(1, len(sel) + 1))   # HYBRID order kept
            out["unchanged_pack"] += same
            out["unchanged_set"] += same
    out["reasons"] = dict(out["reasons"])
    return out


# ---------------------------------------------------------------- analysis
def main(registry=False):
    REC, VER = load()
    res = {"manifest_code_commit": M["code_commit"], "readers": READERS, "arms": ARMS, "cells": {}, "pairs": {},
           "needs_adjudication": [], "system_failures": [], "tokens": {}, "latency": {}, "selector": {}}
    S = {}            # S[rd][a][q] = (lo, hi, status)
    for rd in READERS:
        S[rd] = {}
        for a in ARMS:
            S[rd][a] = {q: score(REC, VER, rd, a, q) for q in QIDS}
            for q, (lo, hi, st) in S[rd][a].items():
                if st in ("needs_adjudication", "unjudged"):
                    res["needs_adjudication"].append({"reader": rd, "arm": a, "q": q, "status": st})
                if st in ("missing", "system_failure"):
                    res["system_failures"].append({"reader": rd, "arm": a, "q": q, "status": st})

    def tally(rd, a, qs):
        lo = sum(S[rd][a][q][0] for q in qs)
        hi = sum(S[rd][a][q][1] for q in qs)
        return lo, hi

    for rd in READERS:
        for a in ARMS:
            c = {"all": tally(rd, a, QIDS), "answerable": tally(rd, a, ANS), "ab": tally(rd, a, AB)}
            for lg in ("fa", "en"):
                c[lg] = tally(rd, a, [q for q in ANS if QS[q]["lang"] == lg])
            for tr in ("T1", "T2", "T3"):
                c[tr] = tally(rd, a, [q for q in ANS if QS[q]["tier"] == tr])
            for dm in sorted({QS[q]["domain"] for q in QIDS}):
                c["dom_" + dm] = tally(rd, a, [q for q in ANS if QS[q]["domain"] == dm])
            # tokens / latency
            toks = [token_cell(REC, rd, a, q) for q in QIDS]
            lat = [latency_cell(REC, rd, a, q) for q in QIDS]
            c["tokens_lower"] = sum(t[0] for t in toks)
            c["tokens_upper"] = sum(t[1] for t in toks)
            c["tokens_unknown_calls"] = sum(t[2] for t in toks)
            c["lat_p50_s"] = round((pct(lat, 0.5) or 0) / 1000, 2)
            c["lat_p95_s"] = round((pct(lat, 0.95) or 0) / 1000, 2)
            res["cells"]["%s|%s" % (rd, a)] = c

    def pair(rd, a, b, qs=ANS):
        """a vs b on qs (machine scores, needs_adjudication as bounds)."""
        better = [q for q in qs if S[rd][a][q][0] == 1 and S[rd][b][q][1] == 0]
        worse = [q for q in qs if S[rd][a][q][1] == 0 and S[rd][b][q][0] == 1]
        unres = [q for q in qs if q not in better and q not in worse and
                 (S[rd][a][q][0] != S[rd][a][q][1] or S[rd][b][q][0] != S[rd][b][q][1])]
        la, ha = tally(rd, a, qs)
        lb, hb = tally(rd, b, qs)
        diffs = [S[rd][a][q][0] - S[rd][b][q][0] for q in qs]
        return {"a": a, "b": b, "reader": rd, "net": la - lb, "net_lower": la - hb, "net_upper": ha - lb,
                "better": better, "worse": worse, "unresolved": unres, "sign_p": sign_p(len(better), len(worse)),
                "ci95": boot_ci(diffs), "ci95_cluster_family": boot_ci(diffs, clusters=[QS[q]["family"] for q in qs])}

    comps = [("LEX5B_6000", "HYBRID_6000"), ("LEX5B_6000", "HR_6000"), ("SETR_6000", "LEX5B_6000"),
             ("SETR_6000", "HYBRID_6000"), ("SETR_6000", "HR_6000"), ("LEX5B_15500", "HYBRID_15500"),
             ("LEX5B_15500", "HR_15500"), ("SETR_15500", "LEX5B_15500"), ("HYBRID_15500", "HYBRID_6000"),
             ("HR_15500", "HR_6000"), ("LEX5B_15500", "LEX5B_6000"), ("SETR_15500", "SETR_6000")]
    for rd in READERS:
        for a, b in comps:
            res["pairs"]["%s|%s|%s" % (rd, a, b)] = pair(rd, a, b)
    # primary gate (GLM)
    g1, g2 = res["pairs"]["%s|LEX5B_6000|HYBRID_6000" % PRIMARY], res["pairs"]["%s|LEX5B_6000|HR_6000" % PRIMARY]
    adj = holm([g1["sign_p"], g2["sign_p"]])
    acc = ("passed" if min(g1["net_lower"], g2["net_lower"]) >= 5 else
           "failed" if min(g1["net_upper"], g2["net_upper"]) < 5 else "unresolved")
    cL, cH, cR = (res["cells"]["%s|%s" % (PRIMARY, x)] for x in ("LEX5B_6000", "HYBRID_6000", "HR_6000"))
    tg = TB.gate({"lower": cL["tokens_lower"], "upper": cL["tokens_upper"]},
                 {"lower": cH["tokens_lower"], "upper": cH["tokens_upper"]}, 2.5)
    tr_hr = TB.ratio_bounds({"lower": cL["tokens_lower"], "upper": cL["tokens_upper"]},
                            {"lower": cR["tokens_lower"], "upper": cR["tokens_upper"]})
    res["gate"] = {"accuracy": acc, "net_vs_hybrid": [g1["net_lower"], g1["net_upper"]],
                   "net_vs_hr": [g2["net_lower"], g2["net_upper"]], "tokens": tg, "tokens_vs_hr": tr_hr,
                   "sign_p": [g1["sign_p"], g2["sign_p"]], "holm_p": adj,
                   "superiority": [p < 0.05 for p in adj]}
    for a in ("LEX5B_6000", "LEX5B_15500", "SETR_6000", "SETR_15500"):
        res["selector"][a] = selector_stats(REC, a)
    # interaction (GLM): LEX5B gain at 6000 minus at 15500, per control
    res["interaction"] = {c: res["pairs"]["%s|LEX5B_6000|%s_6000" % (PRIMARY, c)]["net"] -
                          res["pairs"]["%s|LEX5B_15500|%s_15500" % (PRIMARY, c)]["net"] for c in ("HYBRID", "HR")}
    (HERE / "test_results.json").write_text(json.dumps(res, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    write_md(res, S)
    review_page(REC, S)
    if registry:
        write_registry(res)
    print(json.dumps({"gate": res["gate"], "needs_adjudication": len(res["needs_adjudication"]),
                      "system_failures": len(res["system_failures"])}, default=str))


def fmt(lohi):
    lo, hi = lohi
    return "%d" % lo if lo == hi else "%d-%d" % (lo, hi)


def write_md(res, S):
    C, P = res["cells"], res["pairs"]
    L = ["# TRUE TEST RESULTS (one-shot, 96 test questions; frozen protocol PROTOCOL.md section 9, code %s)" %
         res["manifest_code_commit"][:7], "",
         "Machine scores of the pinned judge (gpt-oss-120b @ deepinfra/bf16, strict). `a-b` = bounds from "
         "needs_adjudication cells (never counted as 0). 90 answerable + 6 unanswerable (AB = correct abstention).", ""]
    g = res["gate"]
    cHg = C["%s|HYBRID_6000" % PRIMARY]
    L += ["## 1. Primary gate (reader GLM-5.3-flash)", "",
          "| check | result |", "|---|---|",
          "| accuracy: LEX5B_6000 >= +5/90 vs HYBRID_6000 AND vs HR_6000 | **%s** (net vs HYBRID %s, vs HR %s) |" % (
              g["accuracy"].upper(), "%+d..%+d" % tuple(g["net_vs_hybrid"]) if g["net_vs_hybrid"][0] != g["net_vs_hybrid"][1] else "%+d" % g["net_vs_hybrid"][0],
              "%+d..%+d" % tuple(g["net_vs_hr"]) if g["net_vs_hr"][0] != g["net_vs_hr"][1] else "%+d" % g["net_vs_hr"][0]),
          "| tokens: LEX5B_6000 <= 2.5x contemporaneous HYBRID_6000 (96 q) | **%s** (ratio %.2f-%.2f; vs HR %.2f-%.2f) |" % (
              g["tokens"]["status"].upper(), g["tokens"]["ratio_lower"], g["tokens"]["ratio_upper"], g["tokens_vs_hr"][0], g["tokens_vs_hr"][1]),
          "| superiority (exact sign test, Holm over 2) | vs HYBRID p=%.3f (Holm %.3f), vs HR p=%.3f (Holm %.3f) -> %s |" % (
              g["sign_p"][0], g["holm_p"][0], g["sign_p"][1], g["holm_p"][1],
              "supported" if all(g["superiority"]) else "supported vs one control" if any(g["superiority"]) else "not supported"),
          ""]
    for a, b in (("LEX5B_6000", "HYBRID_6000"), ("LEX5B_6000", "HR_6000"), ("SETR_6000", "LEX5B_6000")):
        p = P["%s|%s|%s" % (PRIMARY, a, b)]
        L.append("- **%s vs %s** (GLM, 90 answerable): net %+d, better %d / worse %d (sign p %.3f), 95%% bootstrap CI of "
                 "the net [%+d, %+d], family-cluster CI [%+d, %+d]. Better: %s. Worse: %s.%s" % (
                     a, b, p["net"], len(p["better"]), len(p["worse"]), p["sign_p"], p["ci95"][0], p["ci95"][1],
                     p["ci95_cluster_family"][0], p["ci95_cluster_family"][1],
                     ", ".join(q[8:] for q in p["better"]) or "-", ", ".join(q[8:] for q in p["worse"]) or "-",
                     (" Unresolved: %s." % ", ".join(q[8:] for q in p["unresolved"])) if p["unresolved"] else ""))
    L += ["", "## 2. Every reader x arm (correct)", "",
          "| reader | arm | /96 | /90 ans | AB /6 | fa /45 | en /45 | T1 | T2 | T3 | tokens/q | p50 s | p95 s |",
          "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    nt = {t: sum(1 for q in ANS if QS[q]["tier"] == t) for t in ("T1", "T2", "T3")}
    for rd in READERS:
        for a in ARMS:
            c = C["%s|%s" % (rd, a)]
            tk = "%d" % (c["tokens_lower"] / 96) if c["tokens_lower"] == c["tokens_upper"] else "%d-%d" % (
                c["tokens_lower"] / 96, c["tokens_upper"] / 96)
            L.append("| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %.2f | %.2f |" % (
                SHORT[rd], a, fmt(c["all"]), fmt(c["answerable"]), fmt(c["ab"]), fmt(c["fa"]), fmt(c["en"]),
                fmt(c["T1"]), fmt(c["T2"]), fmt(c["T3"]), tk, c["lat_p50_s"], c["lat_p95_s"]))
    L += ["", "Tier sizes (answerable): %s. Tokens = query-LLM tokens per question (selector + reader; non-GLM "
          "readers: the shared GLM selector calls + their own answer call); latency of non-GLM readers = the shared "
          "pre-answer time of the GLM cell + their own answer call." % nt, ""]
    # generality
    L += ["## 3. Generality across readers (descriptive; 90 answerable, nets with better/worse)", "",
          "| reader | LEX5B-HYBRID @6000 | LEX5B-HR @6000 | SETR-LEX5B @6000 | SETR-HYBRID @6000 | LEX5B-HYBRID @15500 | LEX5B-HR @15500 | HYBRID 15500-6000 | HR 15500-6000 |",
          "|---|---|---|---|---|---|---|---|---|"]
    for rd in READERS:
        row = []
        for a, b in (("LEX5B_6000", "HYBRID_6000"), ("LEX5B_6000", "HR_6000"), ("SETR_6000", "LEX5B_6000"),
                     ("SETR_6000", "HYBRID_6000"), ("LEX5B_15500", "HYBRID_15500"), ("LEX5B_15500", "HR_15500"),
                     ("HYBRID_15500", "HYBRID_6000"), ("HR_15500", "HR_6000")):
            p = P["%s|%s|%s" % (rd, a, b)]
            row.append("%+d (%d/%d, p %.2f)" % (p["net"], len(p["better"]), len(p["worse"]), p["sign_p"]))
        L.append("| %s | %s |" % (SHORT[rd], " | ".join(row)))
    # tokens / latency ratios GLM
    L += ["", "## 4. Tokens and latency (GLM cells, 96 questions)", ""]
    cH = C["%s|HYBRID_6000" % PRIMARY]
    for a in ARMS:
        c = C["%s|%s" % (PRIMARY, a)]
        lo, hi = TB.ratio_bounds({"lower": c["tokens_lower"], "upper": c["tokens_upper"]},
                                 {"lower": cH["tokens_lower"], "upper": cH["tokens_upper"]})
        L.append("- %s: tokens %d-%d (%d unknown-usage calls) = %.2f-%.2fx HYBRID_6000; latency p50 %.2f s (%.2fx), "
                 "p95 %.2f s (%.2fx)" % (a, c["tokens_lower"], c["tokens_upper"], c["tokens_unknown_calls"], lo, hi,
                                         c["lat_p50_s"], c["lat_p50_s"] / max(cH["lat_p50_s"], 1e-9), c["lat_p95_s"],
                                         c["lat_p95_s"] / max(cH["lat_p95_s"], 1e-9)))
    L += ["", "Selector behaviour (GLM cells): " + "; ".join(
        "%s n %d, fallbacks %d %s, all-NONE %d, final pack = fallback pack %d (same set %d)" % (
            a, s["n"], s["fail"], s["reasons"], s.get("all_none", 0), s["unchanged_pack"], s.get("unchanged_set", 0))
        for a, s in res["selector"].items()), ""]
    # budget effect + interaction
    L += ["## 5. Budget effect and LEX-by-budget interaction (GLM)", ""]
    for a, b in (("HYBRID_15500", "HYBRID_6000"), ("HR_15500", "HR_6000"), ("LEX5B_15500", "LEX5B_6000"),
                 ("SETR_15500", "SETR_6000"), ("LEX5B_15500", "HYBRID_15500"), ("LEX5B_15500", "HR_15500"),
                 ("SETR_15500", "LEX5B_15500")):
        p = P["%s|%s|%s" % (PRIMARY, a, b)]
        L.append("- %s vs %s: net %+d (%d/%d, p %.3f)" % (a, b, p["net"], len(p["better"]), len(p["worse"]),
                                                         p["sign_p"]))
    L.append("- Interaction (LEX5B gain at 6,000 minus gain at 15,500): vs HYBRID %+d, vs HR %+d (includes "
             "independent selector calls per budget)." % (res["interaction"]["HYBRID"], res["interaction"]["HR"]))
    # adjudication
    L += ["", "## 6. needs_adjudication and system failures", ""]
    if res["needs_adjudication"]:
        for x in res["needs_adjudication"]:
            L.append("- %s %s %s (%s)" % (SHORT[x["reader"]], x["arm"], x["q"], x["status"]))
    else:
        L.append("- needs_adjudication: none")
    if res["system_failures"]:
        for x in res["system_failures"]:
            L.append("- system failure: %s %s %s (%s)" % (SHORT[x["reader"]], x["arm"], x["q"], x["status"]))
    else:
        L.append("- system failures: none")
    # run notes (operational; no method, arm, reader setting or gate changed) - added 2026-10-04 before the one run
    L += ["", "## 7. Run notes (operational)", ""]
    bs = json.loads((HERE / "budget_start.json").read_text(encoding="utf-8"))
    try:
        endf = float(json.loads((DS.parent / "openrouter" / "reader_budget.json").read_text(encoding="utf-8-sig"))["cost_usd"])
    except Exception:  # noqa: BLE001
        endf = None
    L.append("- Spend (shared reader file: answers + selectors + replays; the judge is ledgered separately): start "
             "$%.4f, end %s -> %s." % (bs["start_usd"], "$%.4f" % endf if endf is not None else "n/a",
                                      "$%.4f" % (endf - bs["start_usd"]) if endf is not None else "n/a"))
    L.append("- The run was interrupted and resumed: started 2026-10-01 23:09, stopped by the user at about primary "
             "question 4/96 (GPU needed elsewhere), resumed 2026-10-04 17:53 on the same frozen commit, completed "
             "2026-10-04. Ollama (bge-m3 query embeddings for HYBRID / HR / SETR) died and was restarted twice "
             "(~19:40-19:58 and ~20:38; a watchdog restarted it on failure afterwards). Cells that hit those gaps got "
             "their infrastructure rerun. **Latency p50/p95 are therefore measured under interrupted / resumed "
             "conditions with GPU contention and two Ollama restarts; they are a descriptive trade-off, not a gate.**")
    rr = jl(HERE / "reruns.jsonl")
    ic = [r for r in rr if r.get("kind") == "infra-cleanup"]
    icr = HERE / "infra_cleanup_result.json"
    if ic or icr.is_file():
        fin = json.loads(icr.read_text(encoding="utf-8")) if icr.is_file() else {}
        L.append("- Infra clean-up (PROTOCOL.md section 10, amendment 2): cells whose every record was an HTTP 429 or "
                 "Ollama-connection error were re-tried at concurrency 1 with back-off (max 5 tries): %d tries logged; "
                 "final outcome per reader %s. Cells still failed are the system failures of section 6." % (
                     len(ic), {SHORT.get(k, k): {x: sum(1 for y in v if y["final"] == x)
                                                 for x in sorted({y["final"] for y in v})} for k, v in fin.items()}))
    L.append("- Reruns logged in reruns.jsonl: %d (of which infra-cleanup tries %d)." % (len(rr), len(ic)))
    L.append("- The 14 DeepSeek replay cells never attempted before the 2026-10-04 stop were answered as ordinary replay "
             "calls at concurrency 1 (`finish_todo.py`).")
    if CAL_EXTENDED_KEYS:
        L.append("- Token bounds: the frozen calibration has no ratio for non-primary reader models; failed (usage "
                 "not reported) answer calls of those readers get an upper bound from a ratio calibrated on the "
                 "test's own known-usage replay calls (same method and margin, missing keys only: %s). Frozen keys "
                 "and every GLM / gate number are unaffected." % ", ".join(CAL_EXTENDED_KEYS))
    L += ["", "One-shot execution: run-to-run model variance is not estimated. Subgroups, abstention, budget and "
          "reader-generality results are descriptive. Human review of the GLM 6,000 discordant answers: "
          "`human_check/review_test.html` (key: `human_check/key_test.json`).", ""]
    (HERE / "TEST_RESULTS.md").write_text("\n".join(L), encoding="utf-8")


def review_page(REC, S):
    """Blinded review of the GLM 6,000 discordant answers: LEX5B vs HYBRID / HR, and SETR vs LEX5B."""
    out = DS / ("human_review" if (DS / "human_review").is_dir() else "human_check")
    items = {}
    comps = (("LEX5B_6000", "HYBRID_6000"), ("LEX5B_6000", "HR_6000"), ("SETR_6000", "LEX5B_6000"))
    for q in QIDS:
        arms = set()
        for a, b in comps:
            sa, sb = S[PRIMARY][a][q], S[PRIMARY][b][q]
            if sa[0] != sb[0] or sa[1] != sb[1]:
                arms |= {a, b}
        for a in sorted(arms):
            r = REC[PRIMARY][a].get(q)
            ans = ((r or {}).get("answer_text") or "").strip()
            k = (q, " ".join(ans.split()))
            it = items.setdefault(k, {"qid": q, "answer": ans, "arms": [], "strict": {}})
            it["arms"].append(a)
            it["strict"][a] = S[PRIMARY][a][q][0] if S[PRIMARY][a][q][0] == S[PRIMARY][a][q][1] else None
    items = list(items.values())
    random.Random(11).shuffle(items)
    key, page = [], []
    for i, it in enumerate(items, 1):
        iid = "T%02d" % i
        qd = QS[it["qid"]]
        key.append({"item": iid, "qid": it["qid"], "arms": it["arms"], "strict_verdicts": it["strict"]})
        page.append({"item": iid, "lang": qd["lang"], "question": qd["question"], "question_gloss": qd["gloss"],
                     "reference": qd["reference"] if qd["answerable"] else
                     "(This question is UNANSWERABLE from the corpus; a correct answer abstains / says not found.)",
                     "reference_gloss": qd["reference_gloss"] if qd["answerable"] else None, "answer": it["answer"]})
    (out / "key_test.json").write_text(json.dumps({"seed": 11, "source": "test_run GLM @6000 discordant pairs "
                                                   "(LEX5B vs HYBRID/HR, SETR vs LEX5B)", "items": key},
                                                  ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    tpl = (out / "review_dev.template.html").read_text(encoding="utf-8")
    tpl = (tpl.replace('var KEY = "dev_review_v1";', 'var KEY = "test_review_v1";')
              .replace("<title>Dev Answer Check</title>", "<title>Test Answer Check</title>")
              .replace('set:"dev60_discordant@6000"', 'set:"test96_discordant@6000"'))
    data = json.dumps(page, ensure_ascii=False).replace("</", "<\\/")
    (out / "review_test.html").write_text(tpl.replace("/*__DATA__*/[]", data), encoding="utf-8")
    print("review items", len(items), "questions", len({k["qid"] for k in key}))


def write_registry(res):
    rows = []
    for rd in READERS:
        for a in ARMS:
            c = res["cells"]["%s|%s" % (rd, a)]
            rows.append({"arm": a, "variant": VAR[a], "set": "test96", "n": 96, "reader": "openrouter:" + rd,
                         "reader_pin": (M["reader_pins"].get(rd) or {}).get("provider"),
                         "primary_reader": rd == PRIMARY,
                         "pack": "own GLM cell" if rd == PRIMARY else "replay of the GLM cell's pack",
                         "judge": "%s @ %s strict v2" % (JUDGE, PIN), "experiment": M["experiment"],
                         "correct": list(c["all"]), "correct_answerable_90": list(c["answerable"]),
                         "ab_abstain_6": list(c["ab"]), "fa": list(c["fa"]), "en": list(c["en"]),
                         "tokens_q": [round(c["tokens_lower"] / 96), round(c["tokens_upper"] / 96)],
                         "p50_s": c["lat_p50_s"], "p95_s": c["lat_p95_s"],
                         "budget_chars": 15500 if a.endswith("15500") else 6000, "date": "2026-10-04",
                         "code": M["code_commit"][:7], "results": "work/datasets_v2/test_run (TEST_RESULTS.md)"})
    with open(DS / "RESULTS_REGISTRY.jsonl", "a", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("registry rows", len(rows))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main(registry="--registry" in sys.argv[1:])
