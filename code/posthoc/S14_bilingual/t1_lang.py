"""S14 task 1: language interaction and replication table (FRAMING section 2, extra analysis 2). Saved data only.
-> lang.json, lang_tables.md

For each key contrast A - B (paired, 45 answerable questions per language):
  primary      GLM-5.3-flash r0, judge 1: per-language net, b/w, exact sign-p range, 95% bootstrap CI over questions and
               over source-family components (revision_stats.compare); language x arm permutation test (question-level
               random relabelling and exact component-level relabelling) and a stratified bootstrap CI of the interaction
               T = mean_en(d) - mean_fa(d)
  readers      the five other readers on byte-identical packs (both arms frozen only)
  replicates   GLM reader replicates r1, r2 (S2b exp1) for every arm that has them; arms without replicates stay at r0
               ("one side" replicated)
  judge 2      mistral-large-2512 (S1 for the frozen arms; S8 partial for post-hoc arms)
  no-contam    every condition again without the 26 closed-book-correct questions (S7 contam.json): fa 39, en 25
Within-language MDE at n = 45 (S5's exact sign-test method, alpha .05, power .80) from each contrast's observed
discordance."""
import json
import time

import numpy as np

import s14lib as L

T0 = time.time()
CONTRASTS = [
    ("HR vs HYBRID @6000", "HR_6000", "HYBRID_6000"),
    ("Lex-RAG vs FUSION @6000", "LEX5B_6000", "FUSION_6000"),
    ("Lex-RAG vs FUSION @3000", "LEX5B_3000", "FUSION_3000"),
    ("Lex-RAG vs HYBRID @6000", "LEX5B_6000", "HYBRID_6000"),
    ("HYBRID budget 15500 vs 6000", "HYBRID_15500", "HYBRID_6000"),
    ("GREP_AGENT vs HYBRID_6000", "GREP_AGENT", "HYBRID_6000"),
    ("GREP_AGENT vs HR_6000", "GREP_AGENT", "HR_6000"),
    ("ITER_BM25 vs HYBRID_6000", "ITER_BM25", "HYBRID_6000"),
    ("ITER_BM25 vs HR_6000", "ITER_BM25", "HR_6000"),
    ("HYBRID_AGENT vs HYBRID_6000", "HYBRID_AGENT", "HYBRID_6000"),
    ("HYBRID_AGENT vs HR_6000", "HYBRID_AGENT", "HR_6000"),
    ("SETR_NOFILL vs HYBRID_3000", "SETR_NOFILL", "HYBRID_3000"),
    ("GOLD vs HR_6000 (best frozen single-pass)", "GOLD", "HR_6000"),
    ("GOLD vs HYBRID_24000 (best single-pass of the sweep)", "GOLD", "HYBRID_24000"),
]
SHORT = {r: r.split("/")[-1].replace("google/", "") for r in L.READERS}
J2_MAX_UNRES_HALF = 6      # a judge-2 half with more unresolved cells than this is reported but marked undetermined
NOC = [q for q in L.ANS if q not in L.CONTAM]


def quick(fa, fb, qs):
    lo = sum(fa(q)[0] - fb(q)[1] for q in qs)
    hi = sum(fa(q)[1] - fb(q)[0] for q in qs)
    p_lo, p_hi, n_lo, n_hi, k = L.S8A.sign_range_auto(fa, fb, qs)
    b = w = 0
    for q in qs:
        (al, ah), (bl, bh) = fa(q), fb(q)
        if al == ah and bl == bh:
            b += al > bl; w += al < bl
    return {"n": len(qs), "net_bounds": [n_lo, n_hi], "bw": [b, w], "sign_p_range": [p_lo, p_hi], "unresolved_cells": k}


def full(fa, fb, qs):
    c = L.compare(fa, fb, qs)
    return {"n": len(qs), "net_bounds": c["net_enum"], "bw": c["better_worse_resolved"], "sign_p_range": c["sign_p_range"],
            "unresolved_cells": c["unresolved_cells"], "ci_pp_question": c["ci_pp_question"],
            "ci_pp_component": c["ci_pp_component"]}


def conditions(A, B):
    """[(name, fa_fn, fb_fn, kind)]"""
    out = [("GLM r0, judge 1 (primary)", L.j1(A), L.j1(B), "primary")]
    if A in L.FROZEN_ARMS and B in L.FROZEN_ARMS:
        for rd in L.READERS[1:]:
            out.append(("reader %s" % SHORT[rd], L.reader(rd, A), L.reader(rd, B), "reader"))
    ra, rb = L.rep(A, 1), L.rep(B, 1)
    if ra is not None or rb is not None:
        side = "both" if (ra is not None and rb is not None) else ("A only" if ra is not None else "B only")
        for k in (1, 2):
            fa = L.rep(A, k) or L.j1(A)
            fb = L.rep(B, k) or L.j1(B)
            out.append(("GLM replicate r%d (%s replicated)" % (k, side), fa, fb, "replicate"))
    if L.j2(A) is not None and L.j2(B) is not None:
        out.append(("judge 2 (GLM r0)", L.j2(A), L.j2(B), "judge2"))
    return out


def run_condition(name, fa, fb, kind, boot):
    row = {"condition": name, "kind": kind}
    for scope, qs_all in (("all90", L.ANS), ("nocontam64", NOC)):
        r = {}
        for lg in ("fa", "en"):
            qs = [q for q in qs_all if L.LANG[q] == lg]
            r[lg] = full(fa, fb, qs) if boot else quick(fa, fb, qs)
        r["all"] = full(fa, fb, qs_all) if boot else quick(fa, fb, qs_all)
        r["interaction_perm"] = L.perm_interaction(fa, fb, qs_all)
        if boot:
            r["interaction_ci"] = L.boot_interaction(fa, fb, qs_all)
        if kind == "judge2":
            r["undetermined"] = any(r[lg]["unresolved_cells"] > J2_MAX_UNRES_HALF for lg in ("fa", "en"))
        row[scope] = r
    return row


def mde_rows(fa, fb):
    out = {}
    for lg, qs in (("fa", L.FA), ("en", L.EN)):
        res = [q for q in qs if fa(q)[0] == fa(q)[1] and fb(q)[0] == fb(q)[1]]
        disc = sum(fa(q)[0] != fb(q)[0] for q in res)
        psi = disc / len(res) if res else 0
        out[lg] = {"psi_resolved": psi, "n_resolved": len(res),
                   "mde_exact_points_of_45": L.mde_exact(45, psi) * 45 if psi > 0 else None,
                   "mde_normal_points_of_45": L.mde_normal(45, psi) * 45 if psi > 0 else None}
    return out


RES = {"contrasts": {}, "meta": {"n_fa": len(L.FA), "n_en": len(L.EN), "nocontam_fa": sum(L.LANG[q] == "fa" for q in NOC),
                                 "nocontam_en": sum(L.LANG[q] == "en" for q in NOC),
                                 "j2_max_unresolved_per_half": J2_MAX_UNRES_HALF,
                                 "run_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                                 "j2_resolved_answerable_cells_at_run": {a: sum(L.j2(a)(q)[0] == L.j2(a)(q)[1] for q in L.ANS)
                                     for a in sorted({x for _, A, B in CONTRASTS for x in (A, B)}) if L.j2(a) is not None}}}
for name, A, B in CONTRASTS:
    rows = []
    for cn, fa, fb, kind in conditions(A, B):
        rows.append(run_condition(cn, fa, fb, kind, boot=(kind == "primary")))
        print("%6.0fs %s | %s" % (time.time() - T0, name, cn), flush=True)
    RES["contrasts"][name] = {"A": A, "B": B, "rows": rows, "mde": mde_rows(L.j1(A), L.j1(B))}

# ------------------------------------------------------------------ generic MDE table at n = 45 (and 90, for reference)
RES["mde_generic"] = {"%.2f" % psi: {"n45_exact": L.mde_exact(45, psi) * 45, "n45_normal": L.mde_normal(45, psi) * 45,
                                     "n90_exact": L.mde_exact(90, psi) * 90, "n39_exact": L.mde_exact(39, psi) * 39,
                                     "n25_exact": L.mde_exact(25, psi) * 25}
                      for psi in (0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40)}

# ------------------------------------------------------------------ classification
SIG = 0.05


def sig_dir(c, sign):
    """significant in the pooled direction under every assignment of unresolved cells."""
    lo, hi = c["net_bounds"]
    ok_dir = lo > 0 if sign > 0 else hi < 0
    return ok_dir and c["sign_p_range"][1] < SIG


def same_dir(c, sign):
    lo, hi = c["net_bounds"]
    return lo > 0 if sign > 0 else hi < 0


summary = []
for name, v in RES["contrasts"].items():
    prim = v["rows"][0]["all90"]
    nb = prim["all"]["net_bounds"]
    sign = 1 if sum(nb) >= 0 else -1
    s = {"contrast": name, "pooled_net": nb, "pooled_p": prim["all"]["sign_p_range"]}
    for scope in ("all90", "nocontam64"):
        pr = v["rows"][0][scope]
        cls = {lg: sig_dir(pr[lg], sign) for lg in ("fa", "en")}
        s[scope] = {"class": "both" if cls["fa"] and cls["en"] else "English only" if cls["en"] else
                    "Persian only" if cls["fa"] else "neither",
                    "fa_net": pr["fa"]["net_bounds"], "fa_p": pr["fa"]["sign_p_range"],
                    "en_net": pr["en"]["net_bounds"], "en_p": pr["en"]["sign_p_range"],
                    "interaction_T_pp": pr["interaction_perm"]["T_pp"],
                    "interaction_p_question": pr["interaction_perm"]["p_question"],
                    "interaction_p_component": pr["interaction_perm"]["p_component"]}
    # direction consistency over replication conditions (all90): share of usable conditions with the pooled sign held
    for lg in ("fa", "en"):
        usable = [r for r in v["rows"][1:] if not r["all90"].get("undetermined")]
        held = sum(same_dir(r["all90"][lg], sign) for r in usable)
        s["%s_direction_held_in_replications" % lg] = [held, len(usable)]
    summary.append(s)
# Holm within the comparison families (primary, worst-case p)
for scope in ("all90", "nocontam64"):
    ps = [x[scope][lg + "_p"][1] for x in summary for lg in ("fa", "en")]
    hp = L.holm(ps)
    for i, x in enumerate(summary):
        x[scope]["fa_p_holm"] = hp[2 * i]; x[scope]["en_p_holm"] = hp[2 * i + 1]
    for kind in ("question", "component"):
        ip = [x[scope]["interaction_p_%s" % kind][1] for x in summary]
        hi = L.holm(ip)
        for i, x in enumerate(summary):
            x[scope]["interaction_p_%s_holm" % kind] = hi[i]
    for x in summary:
        x[scope]["class_holm"] = ("both" if x[scope]["fa_p_holm"] < SIG and x[scope]["en_p_holm"] < SIG and
                                  x[scope]["class"] == "both" else
                                  "English only" if x[scope]["en_p_holm"] < SIG and x[scope]["class"] in ("both", "English only") else
                                  "Persian only" if x[scope]["fa_p_holm"] < SIG and x[scope]["class"] in ("both", "Persian only") else
                                  "neither")
for x in summary:
    for scope in ("all90", "nocontam64"):
        s = x[scope]
        sign = 1 if sum(x["pooled_net"]) >= 0 else -1
        dfa = (s["fa_net"][0] > 0) if sign > 0 else (s["fa_net"][1] < 0)
        den = (s["en_net"][0] > 0) if sign > 0 else (s["en_net"][1] < 0)
        if scope == "all90" and s["interaction_p_component_holm"] < SIG and s["interaction_p_question_holm"] < SIG:
            pat = "language-dependent (interaction, Holm-%d)" % len(CONTRASTS)
        elif s["interaction_p_component"][1] < SIG and s["interaction_p_question"][1] < SIG:
            pat = "language-dependent (interaction, raw p only)"
        elif dfa and den:
            pat = "same direction in both halves"
        elif den:
            pat = "English half only (interaction not shown)"
        elif dfa:
            pat = "Persian half only (interaction not shown)"
        else:
            pat = "undetermined"
        s["pattern"] = pat
RES["summary"] = summary
RES["family"] = {"contrasts": len(CONTRASTS), "within_language_tests": 2 * len(CONTRASTS),
                 "interaction_tests": len(CONTRASTS), "alpha": SIG,
                 "rule": "a half 'shows' a contrast if its exact sign test is p < .05 under every assignment of "
                         "unresolved cells with the net in the pooled direction (GLM r0, judge 1)"}
L.wjson(L.HERE / "lang.json", RES)

# ------------------------------------------------------------------ markdown
f = L.fnet


def cell(c, show_ci=False):
    s = "%s (%d/%d) p %s" % (f(*c["net_bounds"]), c["bw"][0], c["bw"][1], L.fp(*c["sign_p_range"]))
    if show_ci:
        s += "; CI q [%.1f, %.1f], c [%.1f, %.1f]" % (*c["ci_pp_question"], *c["ci_pp_component"])
    return s


md = ["# S14 task 1: language interaction and replication (generated by t1_lang.py -> lang.json)", "",
      "Paired A - B on the answerable questions. fa/en: 45 each (no-contam: fa 39, en 25). Net in answers, (wins/losses at",
      "resolved pairs), exact two-sided sign-p range over unresolved cells. CI = 95% bootstrap in pp of that half's n",
      "(q = questions, c = source-family components; 10,000, seed 0). Interaction T = mean d (en) - mean d (fa) in pp;",
      "p_q = 20,000 random language relabellings of questions; p_c = exact relabelling of the 20 components (184,756).",
      "Judge-2 rows with more than %d unresolved cells in a half are undetermined. Judge-2 snapshot %s, resolved answerable" % (
          J2_MAX_UNRES_HALF, RES["meta"]["run_at"]),
      "cells per arm: " + ", ".join("%s %d" % kv for kv in RES["meta"]["j2_resolved_answerable_cells_at_run"].items()), ""]
for name, v in RES["contrasts"].items():
    md += ["## %s (%s - %s)" % (name, v["A"], v["B"]), ""]
    md += ["| condition | scope | fa | en | all 90 / 64 | T pp | p_q | p_c |", "|---|---|---|---|---|---|---|---|"]
    for r in v["rows"]:
        if r["all90"].get("undetermined"):
            x = r["all90"]
            md.append("| %s | all | undetermined: %d / %d unresolved cells (fa / en) | | | | | |" % (
                r["condition"], x["fa"]["unresolved_cells"], x["en"]["unresolved_cells"]))
            continue
        for scope in ("all90", "nocontam64"):
            x = r[scope]
            ip = x["interaction_perm"]
            star = "*" if x.get("undetermined") else ""
            md.append("| %s%s | %s | %s | %s | %s | %s | %s | %s |" % (
                r["condition"], star, "all" if scope == "all90" else "no-contam", cell(x["fa"], "ci_pp_question" in x["fa"]),
                cell(x["en"], "ci_pp_question" in x["en"]), cell(x["all"]),
                ("%+.1f" % ip["T_pp"][0]) if abs(ip["T_pp"][0] - ip["T_pp"][1]) < 0.05 else "%+.1f..%+.1f" % tuple(ip["T_pp"]),
                L.fp(*ip["p_question"]), L.fp(*ip["p_component"])))
        if "interaction_ci" in r["all90"]:
            ic = r["all90"]["interaction_ci"]; icn = r["nocontam64"]["interaction_ci"]
            md.append("| (primary interaction CI) | all / no-contam | | | | q [%.1f, %.1f] c [%.1f, %.1f] / q [%.1f, %.1f] c [%.1f, %.1f] | | |" % (
                *ic["ci_pp_question"], *ic["ci_pp_component"], *icn["ci_pp_question"], *icn["ci_pp_component"]))
    m = v["mde"]
    md += ["", "Within-language MDE (exact sign test, n = 45, alpha .05, power .80) at this contrast's observed discordance: "
           "fa psi %.2f -> %s of 45; en psi %.2f -> %s of 45." % (
               m["fa"]["psi_resolved"], "%.1f" % m["fa"]["mde_exact_points_of_45"] if m["fa"]["mde_exact_points_of_45"] else "-",
               m["en"]["psi_resolved"], "%.1f" % m["en"]["mde_exact_points_of_45"] if m["en"]["mde_exact_points_of_45"] else "-"), ""]
md += ["## Minimum detectable effect by discordance (exact sign test, alpha .05, power .80; points)", "",
       "| psi (discordant share) | n=45 exact | n=45 normal | n=39 exact | n=25 exact | n=90 exact |", "|---|---|---|---|---|---|"]
for psi, r in RES["mde_generic"].items():
    md.append("| %s | %.1f | %.1f | %.1f | %.1f | %.1f |" % (psi, r["n45_exact"], r["n45_normal"], r["n39_exact"], r["n25_exact"],
                                                         r["n90_exact"]))
md += ["", "## Summary: where does each contrast replicate? (GLM r0, judge 1; family = %d contrasts x 2 halves = %d "
       "within-language tests, %d interaction tests)" % (len(CONTRASTS), 2 * len(CONTRASTS), len(CONTRASTS)), "",
       "| contrast | pooled /90 | fa /45 (p) | en /45 (p) | class (raw) | class (Holm-%d) | direction held in replications fa / en | "
       "T pp (p_q; p_c) | no-contam fa /39 | no-contam en /25 | no-contam class (raw) | no-contam T (p_q; p_c) |" % (2 * len(CONTRASTS)),
       "|---|---|---|---|---|---|---|---|---|---|---|---|"]
for s in summary:
    a, n = s["all90"], s["nocontam64"]
    md.append("| %s | %s | %s (%s) | %s (%s) | **%s** | %s | %d/%d / %d/%d | %s (%s; %s) | %s (%s) | %s (%s) | %s | %s (%s; %s) |" % (
        s["contrast"], f(*s["pooled_net"]), f(*a["fa_net"]), L.fp(*a["fa_p"]), f(*a["en_net"]), L.fp(*a["en_p"]), a["class"],
        a["class_holm"], *s["fa_direction_held_in_replications"], *s["en_direction_held_in_replications"],
        "%+.1f" % a["interaction_T_pp"][0] if abs(a["interaction_T_pp"][0] - a["interaction_T_pp"][1]) < .05 else
        "%+.1f..%+.1f" % tuple(a["interaction_T_pp"]),
        L.fp(*a["interaction_p_question"]), L.fp(*a["interaction_p_component"]),
        f(*n["fa_net"]), L.fp(*n["fa_p"]), f(*n["en_net"]), L.fp(*n["en_p"]), n["class"],
        "%+.1f" % n["interaction_T_pp"][0] if abs(n["interaction_T_pp"][0] - n["interaction_T_pp"][1]) < .05 else
        "%+.1f..%+.1f" % tuple(n["interaction_T_pp"]),
        L.fp(*n["interaction_p_question"]), L.fp(*n["interaction_p_component"])))
md += ["", "### Pattern (direction rule: a half carries the effect if its net is in the pooled direction under every "
       "assignment; 'language-dependent' if both permutation p (question and component level, worst case) < .05)", "",
       "| contrast | all 90 | without the 26 closed-book-correct |", "|---|---|---|"]
for s in summary:
    md.append("| %s | %s | %s |" % (s["contrast"], s["all90"]["pattern"], s["nocontam64"]["pattern"]))
md += ["", "Holm-adjusted interaction p (worst case), all 90: " + "; ".join(
    "%s q %.3f c %.3f" % (s["contrast"].split(" (")[0], s["all90"]["interaction_p_question_holm"],
                          s["all90"]["interaction_p_component_holm"]) for s in summary)]
(L.HERE / "lang_tables.md").write_text("\n".join(md) + "\n", encoding="utf-8")
print("\n".join(md[-25:]))
print("done %.0fs" % (time.time() - T0))
