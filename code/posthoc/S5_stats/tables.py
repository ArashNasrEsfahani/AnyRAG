"""Render markdown tables (tables.md) from glmm_final_results.json and stats_results.json (no new statistics)."""
import json, sys
from pathlib import Path
OUT = Path(__file__).resolve().parent
GF = json.loads((OUT / "glmm_final_results.json").read_text())
F = GF["fits"]
S = json.loads((OUT / "stats_results.json").read_text())
L = ["## GLMM", "", "Estimators: " + "; ".join("%s = %s" % kv for kv in GF["method"].items()), "",
     "### Fit summary", "", "| fit | n | fixed | logLik | converged | max abs grad | min Hess eig | sigma_q | sigma_comp |", "|---|---|---|---|---|---|---|---|---|"]
for k, v in F.items():
    L.append("| %s | %d | %d | %.3f | %s | %.1e | %.2g | %.3f | %s |" % (k, v["n"], v["n_fixed"], v["loglik"], v["converged"], v["grad_max"],
             v["min_hess_eig"], v["sigma_question"], "%.3f" % v["sigma_component"] if v["sigma_component"] is not None else "-"))
f = lambda x: "%+.3f (%.3f); OR %.2f [%.2f, %.2f]; p=%.3f" % (x["logOR"], x["se"], x["OR"], *x["OR_ci95"], x["p_wald"])
for scope in ("answerable90", "all96"):
    for form in (0, 1):
        L += ["", "### Contrasts: AGQ, question intercept, %s, form %d (%s)" % (
            scope, form, "arm + reader" if form == 0 else "arm + reader + pipeline:reader; arm effects averaged over readers (sum coding)"),
              "", "log-odds ratio (SE); OR [95% Wald CI]; Wald p", "", "| contrast | unresolved = wrong | unresolved = right |", "|---|---|---|"]
        a, b = F["AGQ|%s|unres=0|form=%d|re=q" % (scope, form)], F["AGQ|%s|unres=1|form=%d|re=q" % (scope, form)]
        for c in a["contrasts"]:
            L.append("| %s | %s | %s |" % (c, f(a["contrasts"][c]), f(b["contrasts"][c])))
        L += ["", "Population-averaged accuracy (%%, mean over readers): unres=0: %s. unres=1: %s." % (
            ", ".join("%s %.1f" % kv for kv in a["pop_avg_acc_pct"].items()), ", ".join("%s %.1f" % kv for kv in b["pop_avg_acc_pct"].items()))]
L += ["", "### Estimator sensitivity (answerable90, form 0): LEX5B-HR and LEX5B-HYBRID @6000 log-odds ratios (SE)", "",
      "| fit | LEX5B-HR | LEX5B-HYBRID | sigma_q | sigma_c |", "|---|---|---|---|---|"]
for fill in (0, 1):
    for est, re in (("AGQ", "q"), ("LAPLACE", "q"), ("LAPLACE", "cq")):
        v = F["%s|answerable90|unres=%d|form=0|re=%s" % (est, fill, re)]
        c = v["contrasts"]
        L.append("| %s re=%s unres=%d | %+.3f (%.3f) | %+.3f (%.3f) | %.3f | %s |" % (est, re, fill, c["LEX5B_6000 vs HR_6000"]["logOR"], c["LEX5B_6000 vs HR_6000"]["se"],
                 c["LEX5B_6000 vs HYBRID_6000"]["logOR"], c["LEX5B_6000 vs HYBRID_6000"]["se"], v["sigma_question"],
                 "%.3f" % v["sigma_component"] if v["sigma_component"] is not None else "-"))
L += ["", "### Likelihood-ratio tests", "", "| test | stat | df | p |", "|---|---|---|---|"]
for k, v in GF["lrt"].items():
    L.append("| %s | %.3f | %d | %.3g |" % (k, v["stat"], v["df"], v["p"]))
L += ["", "### Per-reader LEX5B-HR @6000 log-odds ratio implied by form 1 (AGQ, answerable90)", "", "| reader | unres=0 | unres=1 |", "|---|---|---|"]
a, b = F["AGQ|answerable90|unres=0|form=1|re=q"]["per_reader_LEX5B_vs_HR_6000"], F["AGQ|answerable90|unres=1|form=1|re=q"]["per_reader_LEX5B_vs_HR_6000"]
for r in a:
    L.append("| %s | %+.3f (%.3f) | %+.3f (%.3f) |" % (r, a[r]["logOR"], a[r]["se"], b[r]["logOR"], b[r]["se"]))

GP = json.loads((OUT / "glmm_pack_results.json").read_text())
FP = GP["fits"]
L += ["", "## GLMM with pack random effect (PRIMARY model): " + GP["method"], "",
      "| fit | n | fixed | logLik | converged | max abs grad | min Hess eig | sigma_question | sigma_pack |", "|---|---|---|---|---|---|---|---|---|"]
for k, v in FP.items():
    L.append("| %s | %d | %d | %.3f | %s | %.1e | %.2g | %.3f | %.3f |" % (k, v["n"], v["n_fixed"], v["loglik"], v["converged"], v["grad_max"],
             v["min_hess_eig"], v["sigma_question"], v["sigma_pack"]))
for scope in ("answerable90", "all96"):
    for form in (0, 1):
        L += ["", "### Contrasts: Laplace, question + pack intercepts, %s, form %d (%s)" % (
            scope, form, "arm + reader" if form == 0 else "arm + reader + pipeline:reader; arm effects averaged over readers (sum coding)"),
              "", "| contrast | unresolved = wrong | unresolved = right |", "|---|---|---|"]
        a, b = FP["LAPLACE|%s|unres=0|form=%d|re=q+pack" % (scope, form)], FP["LAPLACE|%s|unres=1|form=%d|re=q+pack" % (scope, form)]
        for c in a["contrasts"]:
            L.append("| %s | %s | %s |" % (c, f(a["contrasts"][c]), f(b["contrasts"][c])))
        L += ["", "Population-averaged accuracy (%%, mean over readers): unres=0: %s. unres=1: %s." % (
            ", ".join("%s %.1f" % kv for kv in a["pop_avg_acc_pct"].items()), ", ".join("%s %.1f" % kv for kv in b["pop_avg_acc_pct"].items()))]
L += ["", "### LRTs (question + pack model)", "", "| test | stat | df | p |", "|---|---|---|---|"]
for k, v in GP["lrt"].items():
    L.append("| %s | %.3f | %d | %.3g |" % (k, v["stat"], v["df"], v["p"]))
L += ["", "### Per-reader LEX5B-HR @6000 log-odds ratio implied by form 1 (question + pack, answerable90)", "", "| reader | unres=0 | unres=1 |", "|---|---|---|"]
a, b = FP["LAPLACE|answerable90|unres=0|form=1|re=q+pack"]["per_reader_LEX5B_vs_HR_6000"], FP["LAPLACE|answerable90|unres=1|form=1|re=q+pack"]["per_reader_LEX5B_vs_HR_6000"]
for r in a:
    L.append("| %s | %+.3f (%.3f) | %+.3f (%.3f) |" % (r, a[r]["logOR"], a[r]["se"], b[r]["logOR"], b[r]["se"]))

L += ["", "## TOST / non-inferiority, LEX5B vs HR @6000 (GLM reader), margins post hoc", ""]
for k, v in S["tost"].items():
    L += ["### %s (net %+d..%+d of 90)" % (k, *v["net_bounds"]), "",
          "| bootstrap | 90% CI pp (union over extremes) | 90% CI points of 90 | smallest equiv. margin (pts) | smallest NI margin (pts) |", "|---|---|---|---|---|"]
    for sch in ("component", "question"):
        s = v[sch]
        L.append("| %s | [%.2f, %.2f] | [%.2f, %.2f] | %.2f | %.2f |" % (sch, *s["ci90_pp_union"], *s["ci90_points_of_90"],
                 s["smallest_equivalence_margin_points"], s["smallest_noninferiority_margin_points"]))
    L += ["", "| margin (pts / pp) | TOST p comp. | equivalent (comp.) | TOST p quest. | NI p comp. | non-inferior (comp.) | NI p quest. | cluster-t TOST p | cluster-t NI p |",
          "|---|---|---|---|---|---|---|---|---|"]
    for m in ("3", "5", "8"):
        c, q, t = v["component"]["margins"][m], v["question"]["margins"][m], v["cluster_t_check"]["tost_p_by_margin"][m]
        L.append("| %s / %.2f | %.4f | %s | %.4f | %.4f | %s | %.4f | %.4f | %.4f |" % (m, c["margin_pp"], c["tost_p"], c["equivalent_at_5pct"],
                 q["tost_p"], c["noninf_p"], c["noninferior_at_5pct_one_sided"], q["noninf_p"], t["tost_p"], t["noninf_p"]))
    L.append("")
L += ["## Minimum detectable effect (paired, exact two-sided sign test, alpha .05, power .80, n=90)", "",
      "| comparison | discordance basis | psi | MDE exact (pts of 90 / pp) | MDE normal (pts / pp) | MDE exact x sqrt(deff) (pts) | power for 5/90 at n=90 | n for 5/90 exact | n normal | n normal x deff |",
      "|---|---|---|---|---|---|---|---|---|---|"]
for k, v in S["mde"].items():
    for nm, r in v["by_psi"].items():
        L.append("| %s (b/w %d/%d of %d resolved; deff %.2f) | %s | %.3f | %.2f / %.2f | %.2f / %.2f | %.2f | %.3f | %s | %.0f | %.0f |" % (
            k, *v["better_worse_resolved"], v["n_resolved_pairs"], v["design_effect_component_vs_question"], nm, r["psi"],
            r["mde_exact_points_of_90"], r["mde_exact_pp"], r["mde_normal_points_of_90"], r["mde_normal_pp"], r["mde_exact_points_with_deff"],
            r["power_at_n90_for_5_of_90"], r["n_for_5_of_90_exact_first"], r["n_for_5_of_90_normal"], r["n_for_5_of_90_normal_with_deff"]))
L += ["", "## Multiplicity across readers (tables/readers.tex); raw and adjusted p as min-max over unresolved assignments", "",
      "| reader | comparison | net | raw p | Holm (6 readers) | BH (6 readers) | Holm (24 LEX-vs-control) | BH (24) | Holm (48) | BH (48) |",
      "|---|---|---|---|---|---|---|---|---|---|"]
for t in S["readers_multiplicity"]["tests"]:
    g = lambda f: "%.3f-%.3f" % (t[f + "_p_min"], t[f + "_p_max"]) if (f + "_p_min") in t else "-"
    L.append("| %s | %s | %+d..%+d | %.4f-%.4f | %s | %s | %s | %s | %s | %s |" % (t["reader"], t["comparison"], *t["net"], t["p_min"], t["p_max"],
             g("per_column_6_readers_holm"), g("per_column_6_readers_bh"), g("lex_vs_control_24_holm"), g("lex_vs_control_24_bh"),
             g("whole_table_48_holm"), g("whole_table_48_bh")))
(OUT / "tables.md").write_text("\n".join(L) + "\n", encoding="utf-8")
print("\n".join(L))
