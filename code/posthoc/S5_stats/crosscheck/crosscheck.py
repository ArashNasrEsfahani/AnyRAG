"""Independent cross-check of the S5 primary GLMM (custom ML-Laplace, question + pack random intercepts) with
statsmodels. Reads cells.json directly (does NOT import glmm*.py / s5_*.py); builds the design with patsy formulas.
  1. BinomialBayesMixedGLM (variance components: question, pack = question x arm): fit_vb (mean-field VB) and
     fit_map (posterior mode), with statsmodels' default priors and with diffuse priors (fe_p=10, vcp_p=5).
  2. GEE logistic, exchangeable working correlation, groups = question, robust (sandwich) SE (+ bias-reduced).
  3. GLM logistic, cluster-robust SE by question (and naive iid SE, for reference).
  4. pipeline x reader interaction: GEE robust Wald (15 df); GLM LR tests with question FE and with pack FE.
No API, no GPU.

    crosscheck.py freq                       -> crosscheck_freq_results.json (GEE, GLM, interaction tests)
    crosscheck.py bayes <vb|map> <default|diffuse> <0|1> <ref arm>  -> parts/*.json (one Bayes fit)
    run_all.sh runs all of them in parallel; combine.py builds crosscheck_results.json and the table.

"""
import json, os, time, warnings
os.environ["CUDA_VISIBLE_DEVICES"] = ""
from pathlib import Path
import numpy as np, pandas as pd
import statsmodels, statsmodels.api as sm, statsmodels.formula.api as smf
from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM
from scipy.stats import norm, chi2

HERE = Path(__file__).resolve().parent
S5 = HERE.parent
cells = json.loads((S5 / "cells.json").read_text(encoding="utf-8"))
CONTRASTS = {"LEX5B_6000 vs HYBRID_6000": ("LEX5B_6000", "HYBRID_6000"), "LEX5B_6000 vs HR_6000": ("LEX5B_6000", "HR_6000"),
             "HR_6000 vs HYBRID_6000": ("HR_6000", "HYBRID_6000"), "SETR_6000 vs LEX5B_6000": ("SETR_6000", "LEX5B_6000"),
             "budget HYBRID (15500-6000)": ("HYBRID_15500", "HYBRID_6000"), "budget HR (15500-6000)": ("HR_15500", "HR_6000"),
             "budget LEX5B (15500-6000)": ("LEX5B_15500", "LEX5B_6000"), "budget SETR (15500-6000)": ("SETR_15500", "SETR_6000")}
FIX = "y ~ C(arm, Treatment('{ref}')) + C(reader, Sum)"
READERS_ORDER = [r.split("/")[1] for r in cells["readers"]]
INT_COLS = ["I_%s_%d" % (p, k) for p in ("HR", "LEX5B", "SETR") for k in range(len(READERS_ORDER) - 1)]
INT = " + ".join(INT_COLS)   # pipeline (treatment vs HYBRID) x reader (sum-to-zero) = 3 x 5 = 15 columns, built by hand
FIX_INT = "y ~ C(arm, Treatment('HYBRID_6000')) + C(reader, Sum) + " + INT


def frame(scope, fill):
    qs = set(cells["ans"] if scope == "answerable90" else cells["qids"])
    rows = []
    for key, (lo, hi, st) in cells["cells"].items():
        rd, arm, q = key.split("|")
        if q not in qs:
            continue
        rows.append(dict(reader=rd.split("/")[1], arm=arm, q=q, pack=q + "|" + arm, pipe=arm.split("_")[0],
                         y=float(lo if lo == hi else fill)))
    df = pd.DataFrame(rows).sort_values(["q", "arm", "reader"]).reset_index(drop=True)
    last = READERS_ORDER[-1]
    for p in ("HR", "LEX5B", "SETR"):
        for k in range(len(READERS_ORDER) - 1):
            rs = (df.reader == READERS_ORDER[k]).astype(float) - (df.reader == last).astype(float)
            df["I_%s_%d" % (p, k)] = (df["pipe"] == p).astype(float) * rs
    df["qid"] = pd.factorize(df.q)[0]
    return df


def cname(ref, a):
    return "C(arm, Treatment('%s'))[T.%s]" % (ref, a)


def lin(names, a, b, ref="HYBRID_6000"):
    v = np.zeros(len(names))
    if a != ref: v[names.index(cname(ref, a))] += 1
    if b != ref: v[names.index(cname(ref, b))] -= 1
    return v


def summ(est, se):
    return dict(logOR=float(est), se=float(se), OR=float(np.exp(est)),
                OR_ci95=[float(np.exp(est - 1.96 * se)), float(np.exp(est + 1.96 * se))], p=float(2 * norm.sf(abs(est / se))))


def bayes(df, method, fe_p, vcp_p, refs=("HYBRID_6000", "HR_6000", "LEX5B_6000", "SETR_6000")):
    """Mean-field VB reports only marginal posterior SDs, so each contrast is read off as a single coefficient by
    refitting with the contrast's reference arm as the treatment baseline (same for MAP, for comparability)."""
    out, info = {}, {}
    vc = {"question": "0 + C(q)", "pack": "0 + C(pack)"}
    for ref in refs:
        m = BinomialBayesMixedGLM.from_formula(FIX.format(ref=ref), vc, df, vcp_p=vcp_p, fe_p=fe_p)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = m.fit_vb() if method == "vb" else m.fit_map()
        names = list(m.exog_names)
        sd = np.asarray(r.fe_sd)
        for k, (a, b) in CONTRASTS.items():
            if b == ref:
                i = names.index(cname(ref, a))
                out[k] = summ(r.fe_mean[i], sd[i])
        info[ref] = {"sigma_" + n: float(np.exp(r.vcp_mean[j])) for j, n in enumerate(m.vcp_names)}
    return out, info


def gee(df, formula):
    m = smf.gee(formula, "qid", df, family=sm.families.Binomial(), cov_struct=sm.cov_struct.Exchangeable())
    return m, m.fit(maxiter=200)


def contrasts_from(params, names, cov):
    b = np.asarray(params)
    return {k: summ(lin(names, a, c) @ b, np.sqrt(lin(names, a, c) @ cov @ lin(names, a, c))) for k, (a, c) in CONTRASTS.items()}


def wald(params, names, cov, pat):
    idx = [i for i, n in enumerate(names) if pat in n]
    b = np.asarray(params)[idx]; V = cov[np.ix_(idx, idx)]
    st = float(b @ np.linalg.solve(V, b))
    return dict(stat=st, df=len(idx), p=float(chi2.sf(st, len(idx))))


def glm_lr(df, group):
    """LR test of pipeline:reader in a plain logistic GLM with fixed effects for `group` (question or pack). Groups
    with a constant outcome are dropped: their FE diverge and they carry no information about within-group terms."""
    d = df[df.groupby(group).y.transform("nunique") > 1].copy()
    base = "y ~ 0 + C(%s) + C(reader, Sum)" % group
    if group == "q":
        base += " + C(arm, Treatment('HYBRID_6000'))"
    full = base + " + " + INT
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r0 = smf.glm(base, d, family=sm.families.Binomial()).fit(maxiter=300)
        r1 = smf.glm(full, d, family=sm.families.Binomial()).fit(maxiter=300)
    st = 2 * (r1.llf - r0.llf); dfree = int(round(r1.df_model - r0.df_model))
    return dict(stat=float(st), df=dfree, p=float(chi2.sf(st, dfree)), converged=[bool(r0.converged), bool(r1.converged)], n_used=int(len(d)), n_groups_used=int(d[group].nunique()))


PRIORS = {"default": (2.0, 0.5), "diffuse": (10.0, 5.0)}   # (fe_p, vcp_p): prior SD of fixed effects / of log-SD of each variance component


def run_bayes(meth, lab, fill, ref):
    """One BinomialBayesMixedGLM fit on answerable90 (the primary scope). Writes parts/bayes_<meth>_<lab>_unres<fill>_<ref>.json"""
    t0 = time.time()
    df = frame("answerable90", fill)
    c, info = bayes(df, meth, *PRIORS[lab], refs=(ref,))
    (HERE / "parts").mkdir(exist_ok=True)
    (HERE / "parts" / ("bayes_%s_%s_unres%d_%s.json" % (meth, lab, fill, ref))).write_text(
        json.dumps(dict(contrasts=c, sigmas=info[ref], secs=time.time() - t0), indent=1), encoding="utf-8")
    print("bayes %s %s unres=%d ref=%s %s sig=%s (%.0fs)" % (meth, lab, fill, ref, {k: round(v["logOR"], 3) for k, v in c.items()},
                                                           info[ref], time.time() - t0), flush=True)


def run_freq():
    t0 = time.time()
    custom = json.loads((S5 / "glmm_pack_results.json").read_text())["fits"]
    res = {"statsmodels": statsmodels.__version__, "pandas": pd.__version__, "fits": {}}
    for scope in ("answerable90", "all96"):
        for fill in (0, 1):
            df = frame(scope, fill)
            key = "%s|unres=%d" % (scope, fill)
            R = res["fits"][key] = {"n": int(len(df)), "n_correct": int(df.y.sum()), "n_q": int(df.q.nunique()),
                                    "n_pack": int(df.pack.nunique())}
            ck = custom["LAPLACE|%s|unres=%d|form=0|re=q+pack" % (scope, fill)]
            R["custom_ML_Laplace"] = {k: dict(logOR=v["logOR"], se=v["se"], OR=v["OR"], OR_ci95=v["OR_ci95"], p=v["p_wald"])
                                      for k, v in ck["contrasts"].items() if k in CONTRASTS}
            R["custom_sigmas"] = dict(sigma_question=ck["sigma_question"], sigma_pack=ck["sigma_pack"])
            print("%s n=%d correct=%d (custom n=%d correct=%d)" % (key, R["n"], R["n_correct"], ck["n"], ck["n_correct"]), flush=True)
            f0 = FIX.format(ref="HYBRID_6000")
            m, r = gee(df, f0)
            names = list(m.exog_names)
            R["GEE_exch_q"] = contrasts_from(r.params, names, np.asarray(r.cov_params()))
            R["GEE_exch_q_working_corr"] = float(r.cov_struct.dep_params)
            rb = m.fit(maxiter=200, cov_type="bias_reduced")
            R["GEE_exch_q_biasreduced"] = contrasts_from(rb.params, names, np.asarray(rb.cov_params()))
            g = smf.glm(f0, df, family=sm.families.Binomial()).fit(cov_type="cluster", cov_kwds={"groups": df.qid.values})
            R["GLM_cluster_q"] = contrasts_from(g.params, list(g.model.exog_names), np.asarray(g.cov_params()))
            gp = smf.glm(f0, df, family=sm.families.Binomial()).fit()
            R["GLM_naive_iid"] = contrasts_from(gp.params, list(gp.model.exog_names), np.asarray(gp.cov_params()))
            mi, ri = gee(df, FIX_INT)
            ni = list(mi.exog_names)
            rib = mi.fit(maxiter=200, cov_type="bias_reduced")
            R["interaction"] = {"GEE_robust_wald_pipeline:reader": wald(ri.params, ni, np.asarray(ri.cov_params()), "I_"),
                                "GEE_biasreduced_wald_pipeline:reader": wald(rib.params, ni, np.asarray(rib.cov_params()), "I_"),
                                "GLM_LR_questionFE": glm_lr(df, "q"), "GLM_LR_packFE": glm_lr(df, "pack")}
            for lab in ("custom_ML_Laplace", "GEE_exch_q", "GEE_exch_q_biasreduced", "GLM_cluster_q", "GLM_naive_iid"):
                c = R[lab]
                print("  %-24s " % lab + "  ".join("%s %.3f(%.3f) p=%.4f" % (k.split(" (")[0].replace("_6000", ""), v["logOR"], v["se"], v["p"])
                                                  for k, v in c.items()))
            print("  GEE working rho=%.3f" % R["GEE_exch_q_working_corr"])
            for k, v in R["interaction"].items():
                print("  interaction %-40s %s" % (k, v))
            print("  (%.0fs)" % (time.time() - t0), flush=True)
    (HERE / "crosscheck_freq_results.json").write_text(json.dumps(res, indent=1), encoding="utf-8")


if __name__ == "__main__":
    import sys
    if sys.argv[1] == "freq":
        run_freq()
    else:
        run_bayes(sys.argv[2], sys.argv[3], int(sys.argv[4]), sys.argv[5])
