"""S16 analysis (no model calls) -> results.json, tables.md. Based on S8_factorial/analyze.py.
Judge 1 (pinned gpt-oss-120b): the variant's judge-1 dir via judge_multi.final_verdicts (needs_adjudication = 0..1);
  cells reused from a frozen arm take report_test.score of that arm.
Judge 2 (mistral-large-2512 @ mistral/zdr): judge2/<VARIANT> (first valid verdict of (rid, answer_sha1) is final;
  needs_adjudication / unjudged = 0..1); frozen-reuse cells and the frozen reference arms take S1's verdicts
  (j2lib.judge2).
Statistics: revision_stats.py helpers extracted with ast and run unchanged (as S8 does), DP sign range for >12
unresolved cells; agreement and Cohen's kappa via j2lib.kappa over cells resolved by both judges."""
import ast
import itertools
import json
import os
import random
import sys
from collections import defaultdict
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import s16  # noqa: E402
import judge2_posthoc as J2P  # noqa: E402
import j2lib  # noqa: E402
import report_test as RT  # noqa: E402
from ragbench.tools import judge_multi as JM  # noqa: E402

REPO = J2P.REPO
FZ = REPO / "work" / "datasets_v2" / "test_freeze"
P = RT.PRIMARY
REC, VER = RT.load()
ANS, AB = RT.ANS, RT.AB

# ---------------------------------------------------------------- revision_stats helpers (unchanged source)
SRC = REPO / "paper" / "analysis" / "revision_stats.py"
tree = ast.parse(SRC.read_text(encoding="utf-8"))
want = {"find", "sign_p", "diff_vectors", "sign_range", "boot_mean", "compare"}
mod = ast.Module(body=[n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in want], type_ignores=[])
NS = {"RT": RT, "itertools": itertools, "random": random, "defaultdict": defaultdict,
      "N_BOOT": 10000, "SEED": 0, "ANS": ANS, "parent": {}}
exec(compile(mod, str(SRC), "exec"), NS)
QROWS = {}
for line in open(FZ / "questions" / "test_v2judge_all.jsonl", encoding="utf-8-sig"):
    if line.strip():
        d = json.loads(line)
        QROWS[d["id"]] = d["meta"]["v2"]
cons = {q: set(QROWS[q]["source_family"].split("+")) for q in ANS}
for q in ANS:
    c = sorted(cons[q])
    for o in c[1:]:
        NS["parent"][NS["find"](o)] = NS["find"](c[0])
NS["comp"] = {q: NS["find"](sorted(cons[q])[0]) for q in ANS}
NS["block"] = {q: QROWS[q]["source_block"] for q in ANS}
_orig_sign_range = NS["sign_range"]


def sign_range_dp(fa, fb, qs):
    states = {(0, 0)}
    k = 0
    for q in qs:
        al, ah = fa(q)
        bl, bh = fb(q)
        k += (al != ah) + (bl != bh)
        opts = {(int(x > y), int(x < y)) for x in {al, ah} for y in {bl, bh}}
        states = {(b + db, w + dw) for b, w in states for db, dw in opts}
    ps = [NS["sign_p"](b, w) for b, w in states]
    nets = [b - w for b, w in states]
    return min(ps), max(ps), min(nets), max(nets), k


def sign_range_auto(fa, fb, qs):
    n = sum((fa(q)[0] != fa(q)[1]) + (fb(q)[0] != fb(q)[1]) for q in qs)
    return _orig_sign_range(fa, fb, qs) if n <= 12 else sign_range_dp(fa, fb, qs)


NS["sign_range"] = sign_range_auto
compare = NS["compare"]


# ---------------------------------------------------------------- cells
def j1_frozen(arm, q):
    lo, hi, _ = RT.score(REC, VER, P, arm, q)
    return lo, hi


def j2_frozen(arm, q):
    return j2lib.judge2(arm, q)[:2]


_C = {}


def _load(v):
    if v not in _C:
        d, _, _ = s16.VARIANTS[v]
        fv = JM.final_verdicts(d, "openai/gpt-oss-120b", s16.J1PIN)
        _, cell = s16.records(v)
        slug = JM._slug(J2P.JUDGE2)
        rows, pend = {}, {}
        od = s16.OUT / v
        for x in sorted(s16.jl(od / ("verdicts_%s.jsonl" % slug)), key=lambda x: x["at"]):
            if x.get("judge") == J2P.JUDGE2 and x.get("judge_provider_pin") == J2P.PIN2:
                rows.setdefault((x["rid"], x["answer_sha1"]), x)
        for n in s16.jl(od / ("needs_adjudication_%s.jsonl" % slug)):
            pend.setdefault((n["rid"], n["answer_sha1"]), n)
        _C[v] = (cell, fv, rows, pend)
    return _C[v]


def j1(v, q):
    cell, fv, _, _ = _load(v)
    c = cell[q]
    if c[0] == "frozen":
        return j1_frozen(c[1], q)
    f = fv[c[1]]
    return f["strict_lower"], f["strict_upper"]


def j2(v, q):
    cell, _, rows, _ = _load(v)
    c = cell[q]
    if c[0] == "frozen":
        return j2_frozen(c[1], q)
    k = (c[1], c[2])
    if k in rows:
        return rows[k]["strict"], rows[k]["strict"]
    return 0, 1


def j2_status(v):
    cell, _, rows, pend = _load(v)
    st = defaultdict(int)
    for q in ANS + AB:
        c = cell[q]
        if c[0] == "frozen":
            st["frozen_S1"] += 1
        elif (c[1], c[2]) in rows:
            st["judged"] += 1
        elif (c[1], c[2]) in pend:
            st["needs_adjudication"] += 1
        else:
            st["unjudged"] += 1
    return dict(st)


F1 = {v: (lambda v: lambda q: j1(v, q))(v) for v in s16.ORDER}
F2 = {v: (lambda v: lambda q: j2(v, q))(v) for v in s16.ORDER}
for a in ("HYBRID_6000", "HR_6000"):
    F1[a] = (lambda a: lambda q: j1_frozen(a, q))(a)
    F2[a] = (lambda a: lambda q: j2_frozen(a, q))(a)


def tot(f, qs):
    return [sum(f(q)[0] for q in qs), sum(f(q)[1] for q in qs)]


def groups(f):
    g = {"all": tot(f, ANS), "abstain_ok_6": tot(f, AB)}
    for t in ("T1", "T2", "T3"):
        g[t] = tot(f, [q for q in ANS if "-%s-" % t in q])
    for lang in ("fa", "en"):
        g[lang] = tot(f, [q for q in ANS if "-%s-" % lang in q])
    return g


def cmp(fa, fb):
    c = compare(fa, fb, ANS)
    for key, qs in [("net_" + t, [q for q in ANS if "-%s-" % t in q]) for t in ("T1", "T2", "T3")] + \
                   [("net_" + l, [q for q in ANS if "-%s-" % l in q]) for l in ("fa", "en")]:
        lo, up, _ = NS["diff_vectors"](fa, fb, qs)
        c[key] = [sum(lo), sum(up)]
    return c


def agreement(v, qs):
    pairs = []
    for q in qs:
        a, b = F1[v](q), F2[v](q)
        if a[0] == a[1] and b[0] == b[1]:
            pairs.append((a[0], b[0]))
    n = len(pairs)
    return {"resolved_both": n, "agree": sum(x == y for x, y in pairs),
            "pct": None if not n else 100.0 * sum(x == y for x, y in pairs) / n,
            "kappa": j2lib.kappa(pairs), "j1_1_j2_0": sum(x > y for x, y in pairs),
            "j1_0_j2_1": sum(x < y for x, y in pairs)}


# comparisons: every variant vs HYBRID_6000 and HR_6000; plus the paper's own reference where different
PAIRS = [(v, r) for v in s16.ORDER for r in ("HYBRID_6000", "HR_6000")] + [("E5_DENSE", "DENSE_6000"),
                                                                           ("HR_6000", "HYBRID_6000")]


def fmt(lh):
    return "%d" % lh[0] if lh[0] == lh[1] else "%d..%d" % (lh[0], lh[1])


def fmts(lh):
    f = lambda x: ("%+d" % x) if x else "0"  # noqa: E731
    return f(lh[0]) if lh[0] == lh[1] else "%s..%s" % (f(lh[0]), f(lh[1]))


def fp(pr):
    a, b = pr
    return ("%.3g" % a) if abs(a - b) < 1e-12 else "%.3g-%.3g" % (a, b)


def ci(x):
    return "[%.1f, %.1f]" % tuple(x)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    res = {"status_j2": {}, "acc_j1": {}, "acc_j2": {}, "cmp_j1": {}, "cmp_j2": {}, "agree_96": {}, "agree_90": {}}
    for v in s16.ORDER:
        res["status_j2"][v] = j2_status(v)
        res["agree_96"][v] = agreement(v, ANS + AB)
        res["agree_90"][v] = agreement(v, ANS)
    for a in s16.ORDER + ["HYBRID_6000", "HR_6000"]:
        res["acc_j1"][a] = groups(F1[a])
        res["acc_j2"][a] = groups(F2[a])
    for a, b in PAIRS:
        k = "%s vs %s" % (a, b)
        res["cmp_j1"][k] = cmp(F1[a], F1[b])
        res["cmp_j2"][k] = cmp(F2[a], F2[b])
    pooled = []
    for v in s16.ORDER:
        for q in ANS + AB:
            a, b = F1[v](q), F2[v](q)
            if a[0] == a[1] and b[0] == b[1]:
                pooled.append((a[0], b[0]))
    res["agree_pooled_96"] = {"n": len(pooled), "pct": 100.0 * sum(x == y for x, y in pooled) / len(pooled),
                              "kappa": j2lib.kappa(pooled)}
    (HERE / "results.json").write_text(json.dumps(res, indent=1), encoding="utf-8")

    T = ["# S16 tables (generated by analyze.py)", "",
         "## Accuracy, judge 1 vs judge 2 (correct /90; abstention /6)", "",
         "| arm | J1 /90 | J2 /90 | J1 fa/en | J2 fa/en | J2 T1/T2/T3 | J2 abst /6 | J2 status (96 cells) |",
         "|---|---|---|---|---|---|---|---|"]
    for a in s16.ORDER + ["HYBRID_6000", "HR_6000"]:
        g1, g2 = res["acc_j1"][a], res["acc_j2"][a]
        st = res["status_j2"].get(a, "frozen (S1)")
        st = st if isinstance(st, str) else ", ".join("%s %d" % kv for kv in sorted(st.items()))
        T.append("| %s | %s | %s | %s/%s | %s/%s | %s/%s/%s | %s | %s |" % (
            a, fmt(g1["all"]), fmt(g2["all"]), fmt(g1["fa"]), fmt(g1["en"]), fmt(g2["fa"]), fmt(g2["en"]),
            fmt(g2["T1"]), fmt(g2["T2"]), fmt(g2["T3"]), fmt(g2["abstain_ok_6"]), st))
    T += ["", "## Paired comparisons (net /90; b/w = better/worse at resolved pairs; sign p range; 95% CI pp over "
          "questions / 20 components)", "",
          "| comparison | judge | net | b/w | sign p | CI q | CI comp | net fa / en | net T1/T2/T3 |",
          "|---|---|---|---|---|---|---|---|---|"]
    for a, b in PAIRS:
        k = "%s vs %s" % (a, b)
        for j in ("j1", "j2"):
            c = res["cmp_" + j][k]
            T.append("| %s | %s | %s | %d/%d | %s | %s | %s | %s / %s | %s/%s/%s |" % (
                k, j.upper(), fmts(c["net_bounds"]), c["better_worse_resolved"][0], c["better_worse_resolved"][1],
                fp(c["sign_p_range"]), ci(c["ci_pp_question"]), ci(c["ci_pp_component"]), fmts(c["net_fa"]),
                fmts(c["net_en"]), fmts(c["net_T1"]), fmts(c["net_T2"]), fmts(c["net_T3"])))
    T += ["", "## Agreement with judge 1 (cells resolved by both judges)", "",
          "| arm | n (96) | agree % | kappa | J1=1,J2=0 | J1=0,J2=1 | n (90 ans.) | agree % | kappa |",
          "|---|---|---|---|---|---|---|---|---|"]
    kf = lambda x: "-" if x is None else "%.3f" % x  # noqa: E731
    pf = lambda x: "-" if x is None else "%.1f" % x  # noqa: E731
    for v in s16.ORDER:
        a, b = res["agree_96"][v], res["agree_90"][v]
        T.append("| %s | %d | %s | %s | %d | %d | %d | %s | %s |" % (
            v, a["resolved_both"], pf(a["pct"]), kf(a["kappa"]), a["j1_1_j2_0"], a["j1_0_j2_1"],
            b["resolved_both"], pf(b["pct"]), kf(b["kappa"])))
    p = res["agree_pooled_96"]
    T.append("| pooled (12 variants) | %d | %.1f | %.3f | | | | | |" % (p["n"], p["pct"], p["kappa"]))
    (HERE / "tables.md").write_text("\n".join(T) + "\n", encoding="utf-8")
    print("\n".join(T))


if __name__ == "__main__":
    main()
