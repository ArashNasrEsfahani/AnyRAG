"""S1 task 1 analysis: judge 2 (mistralai/mistral-large-2512 @ mistral/zdr) vs judge 1 (pinned gpt-oss-120b) vs the
human symmetric merge, GLM-5.3-flash, 8 arm-budget cells x 96 test questions. No model calls.

    python analyze_judge2.py     -> judge2_results.json, judge2_results.md
"""
import json
import random
from collections import defaultdict

import j2lib as L

RT = L.RT
ARMS = RT.ARMS
QIDS, ANS, AB = RT.QIDS, RT.ANS, RT.AB
N_BOOT, SEED = 10000, 0
HUM, SEL = L.human_sym()
COMP = L.components(ANS)


def J1(a):
    return lambda q: L.judge1(a, q)


def J2(a):
    return lambda q: L.judge2(a, q)[:2]


def SYM1(a):
    """the paper's symmetric human merge: human on the 30 reviewed questions, judge 1 elsewhere."""
    return lambda q: L.human_sym_cell(a, q)


def SYM2(a):
    """same merge with judge 2 outside the 30 reviewed questions (6,000 arms only)."""
    return lambda q: (HUM[(a, q)], HUM[(a, q)]) if (a, q) in HUM else L.judge2(a, q)[:2]


def tot(f, qs):
    return [sum(f(q)[0] for q in qs), sum(f(q)[1] for q in qs)]


def b(x):
    return "%d" % x[0] if x[0] == x[1] else "%d-%d" % tuple(x)


# ------------------------------------------------------------------ paired comparison (exact ranges by DP)
def sign_range(fa, fb, qs):
    states = {(0, 0)}
    k = 0
    for q in qs:
        (al, ah), (bl, bh) = fa(q), fb(q)
        k += (al != ah) + (bl != bh)
        outs = {(x > y, x < y) for x in {al, ah} for y in {bl, bh}}
        states = {(s0 + o0, s1 + o1) for s0, s1 in states for o0, o1 in outs}
    ps = [L.sign_p(x, y) for x, y in states]
    nets = [x - y for x, y in states]
    return min(ps), max(ps), min(nets), max(nets), k


def boot(diffs, clusters=None):
    rnd = random.Random(SEED)
    n = len(diffs)
    sims = []
    if clusters is None:
        for _ in range(N_BOOT):
            sims.append(100.0 * sum(diffs[rnd.randrange(n)] for _ in range(n)) / n)
    else:
        g = defaultdict(list)
        for d, c in zip(diffs, clusters):
            g[c].append(d)
        keys = sorted(g)
        for _ in range(N_BOOT):
            t = c = 0
            for _k in keys:
                grp = g[keys[rnd.randrange(len(keys))]]
                t += sum(grp)
                c += len(grp)
            sims.append(100.0 * t / c)
    sims.sort()
    return sims[int(0.025 * N_BOOT)], sims[int(0.975 * N_BOOT) - 1]


def compare(fa, fb, qs=ANS):
    lo = [fa(q)[0] - fb(q)[1] for q in qs]
    up = [fa(q)[1] - fb(q)[0] for q in qs]
    p_lo, p_hi, n_lo, n_hi, k = sign_range(fa, fb, qs)
    out = {"net_range": [n_lo, n_hi], "unresolved_cells": k, "sign_p_range": [p_lo, p_hi]}
    for name, cl in (("question", None), ("component", [COMP[q] for q in qs])):
        a1, a2 = boot(lo, cl), boot(up, cl)
        out["ci_pp_" + name] = [min(a1[0], a2[0]), max(a1[1], a2[1])]
    bw = [0, 0]
    for q in qs:
        (al, ah), (bl, bh) = fa(q), fb(q)
        if al == ah and bl == bh:
            bw[0] += al > bl
            bw[1] += al < bl
    out["better_worse_resolved"] = bw
    return out


PAIRS = [("LEX5B_6000", "HYBRID_6000"), ("LEX5B_6000", "HR_6000"), ("SETR_6000", "LEX5B_6000"),
         ("HR_6000", "HYBRID_6000"), ("SETR_6000", "HYBRID_6000"), ("SETR_6000", "HR_6000"),
         ("LEX5B_15500", "HYBRID_15500"), ("LEX5B_15500", "HR_15500"), ("SETR_15500", "LEX5B_15500"),
         ("HYBRID_15500", "HYBRID_6000"), ("HR_15500", "HR_6000"), ("LEX5B_15500", "LEX5B_6000"),
         ("SETR_15500", "SETR_6000"), ("HYBRID_15500", "LEX5B_6000")]


def main():
    res = {"judge2": {"model": L.J2_MODEL, "provider_pin": L.J2_PIN, "generation": {"temperature": 0, "top_p": 1.0,
                                                                                   "seed": 42}},
           "status_counts": {}, "cells": {}}
    # ---------------------------------------------------------- 1. per-cell accuracy
    for a in ARMS:
        st = defaultdict(int)
        for q in QIDS:
            st[L.judge2(a, q)[2]] += 1
        res["status_counts"][a] = dict(st)
        res["cells"][a] = {"j1_ans90": tot(J1(a), ANS), "j2_ans90": tot(J2(a), ANS), "j1_all96": tot(J1(a), QIDS),
                           "j2_all96": tot(J2(a), QIDS), "j1_unans6": tot(J1(a), AB), "j2_unans6": tot(J2(a), AB)}
        if a in L.ARMS4:
            res["cells"][a]["sym_human_j1_ans90"] = tot(SYM1(a), ANS)
            res["cells"][a]["sym_human_j2_ans90"] = tot(SYM2(a), ANS)
    # by tier / language (answerable)
    groups = defaultdict(list)
    for q in ANS:
        groups["tier " + str(RT.QS[q]["tier"])].append(q)
        groups["lang " + RT.QS[q]["lang"]].append(q)
    res["by_group"] = {g: {"n": len(qs), **{a: {"j1": tot(J1(a), qs), "j2": tot(J2(a), qs)} for a in ARMS}}
                       for g, qs in sorted(groups.items())}
    # ---------------------------------------------------------- 2. judge-judge agreement
    ag = {}
    for grp, arms in [("all8", ARMS), ("6000x4", L.ARMS4)] + [(a, [a]) for a in ARMS]:
        pairs, unres = [], 0
        for a in arms:
            for q in QIDS:
                x, y = L.judge1(a, q), L.judge2(a, q)[:2]
                if x[0] != x[1] or y[0] != y[1]:
                    unres += 1
                    continue
                pairs.append((x[0], y[0]))
        ag[grp] = {"cells": len(pairs) + unres, "comparable": len(pairs), "unresolved": unres,
                   "agree": sum(x == y for x, y in pairs), "j1_1_j2_0": sum(x > y for x, y in pairs),
                   "j1_0_j2_1": sum(x < y for x, y in pairs),
                   "agree_pct": 100.0 * sum(x == y for x, y in pairs) / len(pairs) if pairs else None,
                   "kappa": L.kappa(pairs)}
    res["judge_agreement"] = ag
    # ---------------------------------------------------------- 3. judges vs human on the 120 reviewed cells
    hv = {}
    for grp, arms in [("all", L.ARMS4)] + [(a, [a]) for a in L.ARMS4]:
        o = {}
        for jn, jf in (("j1", L.judge1), ("j2", lambda a, q: L.judge2(a, q)[:2])):
            pairs, unres = [], 0
            for a in arms:
                for q in SEL:
                    m = jf(a, q)
                    if m[0] != m[1]:
                        unres += 1
                        continue
                    pairs.append((m[0], HUM[(a, q)]))
            o[jn] = {"comparable": len(pairs), "unresolved": unres, "agree": sum(x == y for x, y in pairs),
                     "stricter_m0_h1": sum(x < y for x, y in pairs), "laxer_m1_h0": sum(x > y for x, y in pairs),
                     "agree_pct": 100.0 * sum(x == y for x, y in pairs) / len(pairs) if pairs else None,
                     "kappa": L.kappa(pairs), "judge_correct": sum(x for x, _ in pairs),
                     "human_correct": sum(y for _, y in pairs)}
        hv[grp] = o
    res["vs_human_120"] = hv
    # ---------------------------------------------------------- 4. ranking at 6000
    rk = {}
    for name, F in (("judge1", J1), ("judge2", J2), ("sym_human_j1", SYM1), ("sym_human_j2", SYM2)):
        rk[name] = {a: tot(F(a), ANS) for a in L.ARMS4}
    res["ranking_6000_ans90"] = rk
    # ---------------------------------------------------------- 5. paired comparisons
    cmp_ = {}
    for a, b2 in PAIRS:
        cmp_["judge2 %s vs %s" % (a, b2)] = compare(J2(a), J2(b2))
        cmp_["judge1 %s vs %s" % (a, b2)] = compare(J1(a), J1(b2))
    for a, b2 in PAIRS[:6]:
        cmp_["sym_human_j2 %s vs %s" % (a, b2)] = compare(SYM2(a), SYM2(b2))
        cmp_["sym_human_j1 %s vs %s" % (a, b2)] = compare(SYM1(a), SYM1(b2))
    # by tier / language for the two primary comparisons under judge 2
    for g, qs in sorted(groups.items()):
        for a, b2 in PAIRS[:2]:
            sr = sign_range(J2(a), J2(b2), qs)
            cmp_["judge2 [%s] %s vs %s" % (g, a, b2)] = {"n": len(qs), "net_range": [sr[2], sr[3]],
                                                         "sign_p_range": [sr[0], sr[1]], "unresolved_cells": sr[4]}
    res["pairs"] = cmp_
    (L.HERE / "judge2_results.json").write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8")
    (L.HERE / "judge2_results.md").write_text(md(res), encoding="utf-8")
    print(md(res))


def f(x, d=1):
    return "-" if x is None else "%.*f" % (d, x)


def md(res):
    o = ["# Judge 2 (%s @ %s, T=0, seed 42) vs judge 1 (gpt-oss-120b @ deepinfra/bf16) vs human" %
         (res["judge2"]["model"], res["judge2"]["provider_pin"]), "", "## 1. Correct answers per cell (GLM-5.3-flash)",
         "", "Bounds a-b span unresolved cells (needs_adjudication / unjudged).", "",
         "| cell | J1 /90 | J2 /90 | J1 /96 | J2 /96 | J1 unans /6 | J2 unans /6 | sym-human(J1) /90 | sym-human(J2) /90 | J2 statuses |",
         "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    for a, c in res["cells"].items():
        o.append("| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
            a, b(c["j1_ans90"]), b(c["j2_ans90"]), b(c["j1_all96"]), b(c["j2_all96"]), b(c["j1_unans6"]),
            b(c["j2_unans6"]), b(c["sym_human_j1_ans90"]) if "sym_human_j1_ans90" in c else "",
            b(c["sym_human_j2_ans90"]) if "sym_human_j2_ans90" in c else "",
            ", ".join("%s %d" % kv for kv in sorted(res["status_counts"][a].items()))))
    o += ["", "## 2. Judge 1 vs judge 2 agreement (cells where both are resolved)", "",
          "| cells | comparable | agree | J1=1,J2=0 | J1=0,J2=1 | agree % | kappa |", "|---|---:|---:|---:|---:|---:|---:|"]
    for g, x in res["judge_agreement"].items():
        o.append("| %s | %d | %d | %d | %d | %s | %s |" % (g, x["comparable"], x["agree"], x["j1_1_j2_0"], x["j1_0_j2_1"],
                                                       f(x["agree_pct"]), f(x["kappa"], 3)))
    o += ["", "## 3. Each judge vs the human on the 120 reviewed cells (30 questions x 4 pipelines @6,000)", "",
          "stricter = judge 0, human 1; laxer = judge 1, human 0.", "",
          "| pipeline | J1 agree | J1 stricter | J1 laxer | J1 kappa | J2 agree | J2 stricter | J2 laxer | J2 unres. | J2 kappa | human correct |",
          "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for g, x in res["vs_human_120"].items():
        j1, j2 = x["j1"], x["j2"]
        o.append("| %s | %d/%d | %d | %d | %s | %d/%d | %d | %d | %d | %s | %d |" % (
            g, j1["agree"], j1["comparable"], j1["stricter_m0_h1"], j1["laxer_m1_h0"], f(j1["kappa"], 3), j2["agree"],
            j2["comparable"], j2["stricter_m0_h1"], j2["laxer_m1_h0"], j2["unresolved"], f(j2["kappa"], 3),
            j2["human_correct"]))
    o += ["", "## 4. Ranking of the four pipelines @6,000 (correct /90)", "",
          "| scoring | HYBRID | HR | LEX5B | SETR |", "|---|---:|---:|---:|---:|"]
    for n, r in res["ranking_6000_ans90"].items():
        o.append("| %s | %s |" % (n, " | ".join(b(r[a]) for a in L.ARMS4)))
    o += ["", "## 5. Paired comparisons (90 answerable; net = a - b; exact sign-p range over unresolved cells; 95% "
          "bootstrap CI in pp over questions and over source-family components, union over the two extreme "
          "assignments)", "", "| comparison | net | better/worse (resolved) | unresolved | sign p | CI pp (question) | CI pp (component) |",
          "|---|---|---|---:|---|---|---|"]
    for k, x in res["pairs"].items():
        if "ci_pp_question" not in x:
            continue
        o.append("| %s | %s | %d/%d | %d | %s | [%s, %s] | [%s, %s] |" % (
            k, ("%+d" % x["net_range"][0]) if x["net_range"][0] == x["net_range"][1] else "%+d..%+d" % tuple(x["net_range"]),
            x["better_worse_resolved"][0], x["better_worse_resolved"][1], x["unresolved_cells"],
            ("%.4f" % x["sign_p_range"][0]) if abs(x["sign_p_range"][0] - x["sign_p_range"][1]) < 1e-12
            else "%.4f-%.4f" % tuple(x["sign_p_range"]),
            f(x["ci_pp_question"][0]), f(x["ci_pp_question"][1]), f(x["ci_pp_component"][0]), f(x["ci_pp_component"][1])))
    o += ["", "### By tier and language (judge 2, primary comparisons)", "", "| group | n | comparison | net | sign p |",
          "|---|---:|---|---|---|"]
    for k, x in res["pairs"].items():
        if "ci_pp_question" in x:
            continue
        g = k[k.index("[") + 1:k.index("]")]
        o.append("| %s | %d | %s | %+d..%+d | %.3f-%.3f |" % (g, x["n"], k[k.index("]") + 2:], x["net_range"][0],
                                                            x["net_range"][1], x["sign_p_range"][0], x["sign_p_range"][1]))
    o += ["", "### Accuracy by tier and language (answerable; J1 / J2)", "",
          "| group | n | " + " | ".join(ARMS) + " |", "|---|---:|" + "---|" * len(ARMS)]
    for g, x in res["by_group"].items():
        o.append("| %s | %d | %s |" % (g, x["n"], " | ".join("%s / %s" % (b(x[a]["j1"]), b(x[a]["j2"])) for a in ARMS)))
    return "\n".join(o) + "\n"


if __name__ == "__main__":
    main()
