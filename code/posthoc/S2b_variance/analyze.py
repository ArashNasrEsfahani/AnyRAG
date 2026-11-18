"""S2b analysis: reads only files written by this stream + the frozen saved records/verdicts. No model call.
Writes results.json (all numbers in RESULTS.md come from it)."""
import itertools
import json
import math
import random
import statistics as stx
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, r"D:\KareOne\RAG Paper\work\posthoc_2026-11\S2b_variance")
import lib as L  # noqa: E402

sys.path.insert(0, str(L.TR))
import report_test as RT  # noqa: E402
from ragbench.tools import judge_multi as JM  # noqa: E402

REC, VER = RT.load()
P = RT.PRIMARY
ANS, AB = RT.ANS, RT.AB
N_BOOT, SEED = 10000, 0

# ---------------------------------------------------------------- components (same rule as revision_stats.py)
QROWS = {}
for line in open(L.FZ / "questions" / "test_v2judge_all.jsonl", encoding="utf-8-sig"):
    if line.strip():
        d = json.loads(line)
        QROWS[d["id"]] = d["meta"]["v2"]
parent = {}


def find(x):
    parent.setdefault(x, x)
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x


cons = {q: sorted(set(QROWS[q]["source_family"].split("+"))) for q in ANS}
for q in ANS:
    for o in cons[q][1:]:
        parent[find(o)] = find(cons[q][0])
COMP = {q: find(cons[q][0]) for q in ANS}
TIER = {q: RT.QS[q]["tier"] for q in RT.QIDS}
LANG = {q: RT.QS[q]["lang"] for q in RT.QIDS}


def sign_p(b, w):
    return RT.sign_p(b, w)


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
            tot = cnt = 0
            for _k in keys:
                grp = g[keys[rnd.randrange(len(keys))]]
                tot += sum(grp)
                cnt += len(grp)
            sims.append(100.0 * tot / cnt)
    sims.sort()
    return [sims[int(0.025 * N_BOOT)], sims[int(0.975 * N_BOOT) - 1]]


def compare(fa, fb, qs=ANS):
    """fa, fb: q -> (lo, hi) (may be fractional means). Net bounds, sign-test range over unresolved cells (exact
    enumeration when <= 12 unresolved cells, else the two extremes), bootstrap CIs (union over the two extremes)."""
    lo = [fa(q)[0] - fb(q)[1] for q in qs]
    up = [fa(q)[1] - fb(q)[0] for q in qs]
    cells = [(q, t) for q in qs for f, t in ((fa, "a"), (fb, "b")) if f(q)[0] != f(q)[1]]
    ps = []
    assigns = itertools.product((0, 1), repeat=len(cells)) if len(cells) <= 12 else [(0,) * len(cells),
                                                                                      (1,) * len(cells)]
    for asg in assigns:
        val = dict(zip(cells, asg))
        b = w = 0
        for q in qs:
            av = val.get((q, "a"), fa(q)[0]) if (q, "a") in val else fa(q)[0]
            bv = val.get((q, "b"), fb(q)[0]) if (q, "b") in val else fb(q)[0]
            b += av > bv
            w += av < bv
        ps.append(sign_p(b, w))
    if len(cells) > 12:   # extremes for a: up/down, b: down/up
        for sa, sb in ((1, 0), (0, 1)):
            b = w = 0
            for q in qs:
                av = fa(q)[sa] if fa(q)[0] != fa(q)[1] else fa(q)[0]
                bv = fb(q)[sb] if fb(q)[0] != fb(q)[1] else fb(q)[0]
                b += av > bv
                w += av < bv
            ps.append(sign_p(b, w))
    cq = [boot(lo), boot(up)]
    cc = [boot(lo, [COMP[q] for q in qs]), boot(up, [COMP[q] for q in qs])]
    b = sum(1 for q in qs if fa(q)[0] == fa(q)[1] and fb(q)[0] == fb(q)[1] and fa(q)[0] > fb(q)[0])
    w = sum(1 for q in qs if fa(q)[0] == fa(q)[1] and fb(q)[0] == fb(q)[1] and fa(q)[0] < fb(q)[0])
    return {"net": [round(sum(lo), 3), round(sum(up), 3)], "pp": [round(100 * sum(lo) / len(qs), 2),
                                                                 round(100 * sum(up) / len(qs), 2)],
            "better_worse_resolved": [b, w], "unresolved_cells": len(cells),
            "sign_p": [round(min(ps), 4), round(max(ps), 4)],
            "ci_pp_question": [round(min(cq[0][0], cq[1][0]), 2), round(max(cq[0][1], cq[1][1]), 2)],
            "ci_pp_component": [round(min(cc[0][0], cc[1][0]), 2), round(max(cc[0][1], cc[1][1]), 2)]}


def r0(arm):
    return lambda q: RT.score(REC, VER, P, arm, q)[:2]


def by_group(f, qs=ANS):
    out = {}
    for key, g in (("tier", TIER), ("lang", LANG)):
        for v in sorted({g[q] for q in qs}):
            sub = [q for q in qs if g[q] == v]
            out["%s=%s" % (key, v)] = [sum(f(q)[0] for q in sub), sum(f(q)[1] for q in sub), len(sub)]
    return out


def acc(f, qs=ANS):
    return [sum(f(q)[0] for q in qs), sum(f(q)[1] for q in qs)]


RES = {}

# ================================================================ experiment 1
E1 = {}
for arm in L.ARMS:
    v = L.verdicts(L.HERE / "exp1_reader" / arm)
    E1[arm] = {0: {q: RT.score(REC, VER, P, arm, q) for q in RT.QIDS}}
    for rep in (1, 2):
        E1[arm][rep] = {q: v.get((q, rep, L.primary(arm, q)["meta"]["identity"]["method"]), (0, 1, "missing"))
                        for q in RT.QIDS}


def f1(arm, rep):
    return lambda q: E1[arm][rep][q][:2]


def fmean(arm, reps=(0, 1, 2)):
    return lambda q: (sum(E1[arm][r][q][0] for r in reps) / len(reps), sum(E1[arm][r][q][1] for r in reps) / len(reps))


def answers(arm, rep):
    if rep == 0:
        return {q: (REC[P][arm][q].get("answer_text") or "") for q in RT.QIDS}
    out = {}
    for r in L.jl(L.HERE / "exp1_reader" / arm / "runs.jsonl"):
        if r["meta"]["identity"]["replicate"] == rep and not r.get("error"):
            out[r["question_id"]] = r.get("answer_text") or ""
    return out


exp1 = {"cells": {}, "comparisons": {}}
for arm in L.ARMS:
    a = {rep: acc(f1(arm, rep)) for rep in (0, 1, 2)}
    resolved = [q for q in ANS if all(E1[arm][r][q][0] == E1[arm][r][q][1] for r in (0, 1, 2))]
    flips = [q for q in resolved if len({E1[arm][r][q][0] for r in (0, 1, 2)}) > 1]
    pair_flip = {"%d-%d" % (x, y): sum(1 for q in ANS if E1[arm][x][q][0] == E1[arm][x][q][1]
                                       and E1[arm][y][q][0] == E1[arm][y][q][1]
                                       and E1[arm][x][q][0] != E1[arm][y][q][0]) for x, y in ((0, 1), (0, 2), (1, 2))}
    ans = {rep: answers(arm, rep) for rep in (0, 1, 2)}
    same_text = {"%d-%d" % (x, y): sum(1 for q in RT.QIDS if q in ans[x] and q in ans[y] and ans[x][q] == ans[y][q])
                 for x, y in ((0, 1), (0, 2), (1, 2))}
    # reader-noise variance of one run's accuracy: sum over q of the within-question sample variance (resolved q)
    s2 = [stx.variance([E1[arm][r][q][0] for r in (0, 1, 2)]) for q in resolved]
    sd_reader_pp = 100.0 * math.sqrt(sum(s2) * len(ANS) / max(1, len(resolved))) / len(ANS)
    accs_lo = [a[r][0] for r in (0, 1, 2)]
    accs_hi = [a[r][1] for r in (0, 1, 2)]
    exp1["cells"][arm] = {
        "acc_by_rep": {str(r): a[r] for r in (0, 1, 2)},
        "unresolved_by_rep": {str(r): sum(1 for q in ANS if E1[arm][r][q][0] != E1[arm][r][q][1]) for r in (0, 1, 2)},
        "missing_or_error": {str(r): sum(1 for q in RT.QIDS if E1[arm][r][q][2] in ("missing", "system_failure",
                                                                                  "unjudged")) for r in (1, 2)},
        "between_rep_sd_count": [round(stx.stdev(accs_lo), 3), round(stx.stdev(accs_hi), 3)],
        "between_rep_sd_pp": [round(100 * stx.stdev(accs_lo) / 90, 2), round(100 * stx.stdev(accs_hi) / 90, 2)],
        "resolved_all3": len(resolved), "flip_questions": len(flips),
        "flip_rate": round(len(flips) / max(1, len(resolved)), 4), "pairwise_flips": pair_flip,
        "identical_answer_text_96": same_text, "reader_sd_pp_from_within_q_var": round(sd_reader_pp, 2),
        "by_group_rep": {str(r): by_group(f1(arm, r)) for r in (0, 1, 2)},
        "abstention_unanswerable": {str(r): [sum(E1[arm][r][q][0] for q in AB), sum(E1[arm][r][q][1] for q in AB),
                                             len(AB)] for r in (0, 1, 2)}}

PAIRS = [("LEX5B_6000", "HYBRID_6000"), ("LEX5B_6000", "HR_6000"), ("HYBRID_15500", "HYBRID_6000"),
         ("HR_6000", "HYBRID_6000"), ("SETR_6000", "LEX5B_6000"), ("LEX5B_15500", "HYBRID_15500"),
         ("LEX5B_15500", "HR_15500"), ("LEX5B_15500", "LEX5B_6000"), ("HR_15500", "HR_6000")]
for a, b in PAIRS:
    c = {"same_rep": {str(r): compare(f1(a, r), f1(b, r)) for r in (0, 1, 2)}}
    cross = {}
    for ra in (0, 1, 2):
        for rb in (0, 1, 2):
            lo = sum(E1[a][ra][q][0] - E1[b][rb][q][1] for q in ANS)
            hi = sum(E1[a][ra][q][1] - E1[b][rb][q][0] for q in ANS)
            cross["%d-%d" % (ra, rb)] = [lo, hi]
    c["cross_rep_net"] = cross
    s0 = c["same_rep"]["0"]["net"]
    sign0 = 1 if s0[0] > 0 else -1 if s0[1] < 0 else 0
    c["cross_same_sign_as_r0"] = sum(1 for v in cross.values() if (v[0] > 0 and sign0 > 0) or (v[1] < 0 and sign0 < 0))
    c["cross_n"] = len(cross)
    c["mean_of_3"] = compare(fmean(a), fmean(b))
    # reader-noise SD of a one-run paired difference (within-question variances of both arms, resolved q)
    s2 = []
    for q in ANS:
        if all(E1[x][r][q][0] == E1[x][r][q][1] for x in (a, b) for r in (0, 1, 2)):
            s2.append(stx.variance([E1[a][r][q][0] for r in (0, 1, 2)]) + stx.variance([E1[b][r][q][0] for r in (0, 1, 2)]))
    c["reader_sd_net_one_run"] = round(math.sqrt(sum(s2) * len(ANS) / max(1, len(s2))), 2)
    c["mean_of_3_by_group"] = {k: [round(x, 3) for x in v[:2]] + [v[2]] for k, v in
                               by_group(lambda q: (fmean(a)(q)[0] - fmean(b)(q)[1], fmean(a)(q)[1] - fmean(b)(q)[0])).items()}
    exp1["comparisons"]["%s vs %s" % (a, b)] = c
RES["exp1"] = exp1


# ================================================================ experiments 2-5
def exp_scores(exp):
    """{(q, rep, budget): (lo, hi, status, row)} for an experiment's assign rows."""
    vnew = L.verdicts(L.HERE / exp) if (L.HERE / exp / "judge").is_dir() else {}
    e1v = {arm: L.verdicts(L.HERE / "exp1_reader" / arm) for arm in ("LEX5B_6000", "SETR_6000")}
    out = {}
    for row in L.jl(L.HERE / exp / "assign.jsonl"):
        q, rep, arm = row["q"], row["rep"], row["arm"]
        src = row["answer_source"]
        if src == "r0":
            s = RT.score(REC, VER, P, arm, q)
        elif src.startswith("exp1_rep"):
            k = int(src[-1])
            s = e1v[arm].get((q, k, L.primary(arm, q)["meta"]["identity"]["method"]), (0, 1, "missing"))
        else:
            m = L.primary(arm, q)["meta"]["identity"]["method"] + "|%s_b%d" % (exp, row["budget"])
            s = vnew.get((q, rep, m), (0, 1, "unjudged"))
        out[(q, rep, row["budget"])] = (s[0], s[1], s[2], row)
    return out


def sel_summary(rows):
    n = len(rows)
    reasons = defaultdict(int)
    for r in rows:
        if r["fallback"]:
            reasons[str(r["reason"])] += 1
    sel = [r["selector"] for r in rows if not r["selector"].get("failed") and r["selector"].get("prompt_tokens")]
    lat = sorted(r["selector"]["t_select_ms"] for r in rows if r["selector"].get("t_select_ms") is not None)
    return {"n": n, "pack_changed": sum(1 for r in rows if not r["pack_unchanged"]),
            "shown_changed": sum(1 for r in rows if r["shown_chunks"] != r["saved_shown"]),
            "fallback": sum(1 for r in rows if r["fallback"]), "fallback_reasons": dict(reasons),
            "deadline": sum(1 for r in rows if r["selector"].get("reason") == "deadline"),
            "selector_calls_with_usage": len(sel),
            "selector_prompt_tokens_mean": round(stx.mean(x["prompt_tokens"] for x in sel), 1) if sel else None,
            "selector_completion_tokens_mean": round(stx.mean(x["completion_tokens"] for x in sel), 1) if sel else None,
            "selector_reasoning_tokens_mean": round(stx.mean((x.get("reasoning_tokens") or 0) for x in sel), 1) if sel else None,
            "selector_cost_usd": round(sum(x.get("cost_usd") or 0 for x in sel), 5),
            "selector_latency_ms_median": lat[len(lat) // 2] if lat else None,
            "selector_latency_ms_p90": lat[int(0.9 * (len(lat) - 1))] if lat else None,
            "answer_new": sum(1 for r in rows if r["answer_source"] == "new"),
            "providers": dict(__import__("collections").Counter(str(r["selector"].get("provider")) for r in rows))}


def saved_sel_summary(arm):
    rows = []
    for q in RT.QIDS:
        s = L.primary(arm, q)["trace"]["lex5b"]["selector"]
        rows.append(s)
    reasons = defaultdict(int)
    for s in rows:
        if s.get("fallback"):
            reasons[str(s.get("reason"))] += 1
    us = [s for s in rows if s.get("prompt_tokens")]
    return {"fallback": sum(1 for s in rows if s.get("fallback")), "fallback_reasons": dict(reasons),
            "selector_prompt_tokens_mean": round(stx.mean(s["prompt_tokens"] for s in us), 1),
            "selector_completion_tokens_mean": round(stx.mean(s["completion_tokens"] for s in us), 1)}


RES["saved_selector_LEX5B_6000"] = saved_sel_summary("LEX5B_6000")
RES["saved_selector_LEX5B_15500"] = saved_sel_summary("LEX5B_15500")

for exp in ("exp2", "exp3", "exp4", "exp5", "exp6"):
    if not (L.HERE / exp / "assign.jsonl").is_file():
        continue
    S = exp_scores(exp)
    R = {}
    if exp == "exp2":
        for rep in (1, 2):
            rows = [S[(q, rep, 6000)][3] for q in RT.QIDS if (q, rep, 6000) in S]
            if len(rows) < 96:
                R["rep%d_incomplete" % rep] = len(rows)
                continue
            f = lambda q, rep=rep: S[(q, rep, 6000)][:2]  # noqa: E731
            R["rep%d" % rep] = dict(sel_summary(rows), acc=acc(f), by_group=by_group(f),
                                    vs_LEX5B_r0=compare(f, r0("LEX5B_6000")),
                                    vs_HYBRID_6000_same_reader_rep=compare(f, f1("HYBRID_6000", rep)),
                                    vs_HR_6000_same_reader_rep=compare(f, f1("HR_6000", rep)),
                                    vs_LEX5B_reader_rep_only=compare(f, f1("LEX5B_6000", rep)))
            R["rep%d" % rep]["changed_pack_scores"] = {
                "n": sum(1 for r in rows if not r["pack_unchanged"] and r["q"] in ANS),
                "new_pack": acc(f, [r["q"] for r in rows if not r["pack_unchanged"] and r["q"] in ANS]),
                "reader_rep_same_q": acc(f1("LEX5B_6000", rep), [r["q"] for r in rows if not r["pack_unchanged"]
                                                                and r["q"] in ANS])}
        if "rep1" in R and "rep2" in R:
            accs = [acc(r0("LEX5B_6000")), R["rep1"]["acc"], R["rep2"]["acc"]]
            R["full_pipeline_between_rep_sd_count"] = [round(stx.stdev(a[0] for a in accs), 3),
                                                       round(stx.stdev(a[1] for a in accs), 3)]
            # pack identity between the two new selector replicates
            R["rep1_rep2_same_pack"] = sum(1 for q in RT.QIDS if S[(q, 1, 6000)][3]["pack_sha"] == S[(q, 2, 6000)][3]["pack_sha"])
    elif exp == "exp6":
        rows = [v[3] for v in S.values() if v[3]["rep"] == 1]
        if len(rows) == 96:
            f = lambda q: S[(q, 1, 6000)][:2]  # noqa: E731
            R["rep1"] = dict(sel_summary(rows), acc=acc(f), acc_r0=acc(r0("SETR_6000")),
                             same_selection=sum(1 for r in rows if r["selected"] == r["saved_selected"]),
                             vs_SETR_r0=compare(f, r0("SETR_6000")),
                             vs_SETR_reader_rep1_only=compare(f, f1("SETR_6000", 1)),
                             vs_LEX5B_6000_selector_rep1=compare(f, lambda q, S2=exp_scores("exp2"): S2[(q, 1, 6000)][:2])
                             if (L.HERE / "exp2" / "assign.jsonl").is_file() else None)
        else:
            R["rep1_incomplete"] = len(rows)
    elif exp in ("exp3", "exp4"):
        rows = [v[3] for v in S.values()]
        qs = [r["q"] for r in rows]
        f = lambda q: S[(q, 0, 6000)][:2]  # noqa: E731
        qa = [q for q in qs if q in ANS]
        R = dict(sel_summary(rows), questions=len(qs), answerable=len(qa), acc=acc(f, qa),
                 acc_r0_same_q=acc(r0("LEX5B_6000"), qa), selector_model=rows[0]["selector_model"] if rows else None)
        if exp == "exp3" and len(rows) == 96:
            R["by_group"] = by_group(f)
            R["by_group_r0"] = by_group(r0("LEX5B_6000"))
            for b in ("LEX5B_6000", "HYBRID_6000", "HR_6000"):
                R["vs_%s_r0" % b] = compare(f, r0(b))
            R["abstention_unanswerable"] = [sum(f(q)[0] for q in AB), sum(f(q)[1] for q in AB), len(AB)]
        if exp == "exp4":
            R["now_succeeded"] = sum(1 for r in rows if not r["fallback"])
            R["status_counts"] = dict(__import__("collections").Counter(str(r["status"]) for r in rows))
            R["latencies_ms"] = sorted(r["selector"]["t_select_ms"] for r in rows)
            R["vs_r0"] = compare(f, r0("LEX5B_6000"), qa)
    elif exp == "exp5":
        k15 = [q for q in RT.QIDS if (q, 0, 15500) in S]
        k6 = [q for q in RT.QIDS if (q, 0, 6000) in S]
        if len(k15) == 96 and len(k6) == 96:
            a15 = lambda q: S[(q, 0, 15500)][:2]  # 6000 reply packed at 15500  # noqa: E731
            a6 = lambda q: S[(q, 0, 6000)][:2]    # 15500 reply packed at 6000   # noqa: E731
            R["reply6000_at15500"] = dict(sel_summary([S[(q, 0, 15500)][3] for q in RT.QIDS]), acc=acc(a15))
            R["reply15500_at6000"] = dict(sel_summary([S[(q, 0, 6000)][3] for q in RT.QIDS]), acc=acc(a6))
            R["same_parsed_parts_6000_vs_15500_reply"] = sum(
                1 for q in RT.QIDS
                if L.primary("LEX5B_6000", q)["trace"]["lex5b"]["selector"].get("parts")
                == L.primary("LEX5B_15500", q)["trace"]["lex5b"]["selector"].get("parts")
                and L.primary("LEX5B_6000", q)["trace"]["lex5b"]["selector"].get("status")
                == L.primary("LEX5B_15500", q)["trace"]["lex5b"]["selector"].get("status"))
            R["budget_effect_frozen"] = compare(r0("LEX5B_15500"), r0("LEX5B_6000"))
            R["budget_effect_shared_6000reply"] = compare(a15, r0("LEX5B_6000"))
            R["budget_effect_shared_15500reply"] = compare(r0("LEX5B_15500"), a6)
            R["selection_effect_at6000"] = compare(a6, r0("LEX5B_6000"))
            R["selection_effect_at15500"] = compare(r0("LEX5B_15500"), a15)
            R["vs_HYBRID_15500_reply6000_at15500"] = compare(a15, r0("HYBRID_15500"))
    RES[exp] = R

RES["spend"] = L.StreamBudget().state()
(L.HERE / "results.json").write_text(json.dumps(RES, indent=1), encoding="utf-8")
print("written results.json")
