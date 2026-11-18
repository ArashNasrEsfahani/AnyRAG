"""Analysis 1 (final driver). Two estimators:
  * AGQ  : ML with adaptive Gauss-Hermite quadrature (25 nodes) -- question random intercept only (glmm_agq.py).
  * LAPLACE: ML with the Laplace approximation (as lme4::glmer's default nAGQ=1), joint mode over all random effects
             by block Newton; random intercepts for question, or for component + question-in-component.
           Gradient by central differences; Wald SEs from a finite-difference Hessian.
The AGQ question-only fit is the reference for how much the Laplace approximation shifts the estimates; the nested
(component) model is available only under Laplace (adaptive quadrature over the nested pair did not stabilise: see
RESULTS.md). Fixed-effect design, codings and contrasts are glmm.py's.

    .venv/Scripts/python.exe work/posthoc_2026-11/S5_stats/glmm_final.py
"""
import json, os, sys, time
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import chi2
sys.argv = [sys.argv[0], "25", "10"]
import glmm_agq as A
G0 = A.G0
import s5_lib as L


def laplace_nll(theta, Xs, ys, qcomp, nc, use_c, state):
    p = Xs.shape[2]
    beta, sq = theta[:p], np.exp(theta[p])
    sc = np.exp(theta[p + 1]) if use_c else None
    nq = Xs.shape[0]
    eta0 = Xs @ beta
    b = state.get("b", np.zeros(nq)).copy()
    u = state.get("u", np.zeros(nc)).copy() if use_c else np.zeros(nc)

    def logf(b, u):
        e = eta0 + (b + u[qcomp])[:, None]
        v = (ys * e - np.logaddexp(0, e)).sum() - (b ** 2).sum() / (2 * sq ** 2)
        if use_c:
            v -= (u ** 2).sum() / (2 * sc ** 2)
        return v
    cur = logf(b, u)
    for _ in range(100):
        e = eta0 + (b + u[qcomp])[:, None]
        pr = expit(e)
        r = (ys - pr).sum(1)
        d = (pr * (1 - pr)).sum(1)
        gb = r - b / sq ** 2
        Aq = d + 1 / sq ** 2
        if use_c:
            gu = np.zeros(nc); np.add.at(gu, qcomp, r); gu -= u / sc ** 2
            S = np.zeros(nc); np.add.at(S, qcomp, d - d ** 2 / Aq); S += 1 / sc ** 2
            rhs = gu.copy(); np.add.at(rhs, qcomp, -d * gb / Aq)
            du = rhs / S
            db = (gb - d * du[qcomp]) / Aq
        else:
            du = np.zeros(nc); db = gb / Aq
        t = 1.0
        while True:
            nb, nu = b + t * db, u + t * du
            new = logf(nb, nu)
            if new >= cur - 1e-12 or t < 1e-6:
                break
            t /= 2
        b, u = nb, nu
        conv = abs(new - cur) < 1e-12 and max(np.abs(db).max(), np.abs(du).max() if use_c else 0) < 1e-8
        cur = new
        if conv:
            break
    e = eta0 + (b + u[qcomp])[:, None]
    pr = expit(e); d = (pr * (1 - pr)).sum(1); Aq = d + 1 / sq ** 2
    logdet = np.log(Aq).sum()
    val = cur - nq * np.log(sq)        # prior normalisers (the 2*pi terms cancel against the Laplace sqrt(2*pi)^n)
    if use_c:
        S = np.zeros(nc); np.add.at(S, qcomp, d - d ** 2 / Aq); S += 1 / sc ** 2
        logdet += np.log(S).sum()
        val -= nc * np.log(sc)
    state["b"], state["u"] = b, u
    return -(val - 0.5 * logdet)


def fit_laplace(X, y, qi, ci, nq, nc, re):
    Xs, ys, qcomp = A.prep(X, y, qi, ci, nq)
    p = X.shape[1]; use_c = re == "cq"
    state = {}
    f0 = lambda t: laplace_nll(t, Xs, ys, qcomp, nc, use_c, state)

    def fg(t, h=1e-5):
        v = f0(t); g = np.zeros(len(t))
        for i in range(len(t)):
            e = np.zeros(len(t)); e[i] = h
            g[i] = (f0(t + e) - f0(t - e)) / (2 * h)
        return v, g
    t0 = np.zeros(p + (2 if use_c else 1)); t0[p] = np.log(2.5)
    if use_c:
        t0[p + 1] = np.log(0.5)
    r = minimize(fg, t0, jac=True, method="BFGS", options={"gtol": 1e-5, "maxiter": 2000})
    t = r.x; hh = 1e-3
    H = np.zeros((len(t), len(t)))
    for i in range(len(t)):
        e = np.zeros(len(t)); e[i] = hh
        H[:, i] = (fg(t + e)[1] - fg(t - e)[1]) / (2 * hh)
    H = (H + H.T) / 2
    ev = np.linalg.eigvalsh(H)
    gmax = float(np.abs(fg(t)[1]).max())
    if use_c:     # glmm.py layout: (beta, log sc, log sq)
        perm = list(range(p)) + [p + 1, p]; t2, H2 = t[perm], H[np.ix_(perm, perm)]
    else:
        t2, H2 = t, H
    return dict(theta=t2, cov=np.linalg.inv(H2), nll=float(f0(t)), converged=bool((r.success or gmax < 1e-2) and ev.min() > 0),
                gmax=gmax, min_hess_eig=float(ev.min()), msg=str(r.message), nit=int(r.nit))


def record(r, names, form, re, y, nq, nc):
    p = len(names)
    rec = dict(n=int(len(y)), n_correct=int(y.sum()), n_q=nq, n_comp=nc, n_fixed=p, loglik=-r["nll"], nll=r["nll"],
               converged=r["converged"], grad_max=r["gmax"], min_hess_eig=r["min_hess_eig"], optimizer=r["msg"], nit=r["nit"],
               sigma_question=float(np.exp(r["theta"][-1])), sigma_component=float(np.exp(r["theta"][p])) if re == "cq" else None,
               log_sigma_question_se=float(np.sqrt(r["cov"][-1, -1])))
    if re == "cq":
        rec["log_sigma_component_se"] = float(np.sqrt(r["cov"][p, p]))
    rec["contrasts"] = {k: G0.contrast(r, names, G0.sub(G0.arm_vec(a), G0.arm_vec(b))) for k, (a, b) in G0.CONTRASTS.items()}
    zz, ww = np.polynomial.hermite.hermgauss(60)
    sd = np.sqrt(rec["sigma_question"] ** 2 + (rec["sigma_component"] or 0) ** 2)
    rec["pop_avg_acc_pct"] = {a: 100 * float(np.mean([(ww / np.sqrt(np.pi) * expit(
        np.array(G0.xrow(a, ri, form)[0]) @ r["theta"][:p] + np.sqrt(2) * sd * zz)).sum() for ri in range(G0.NR)])) for a in G0.ARMS}
    if form == 1:
        per = {}
        for ri in range(G0.NR):
            v = np.zeros(len(r["theta"]))
            v[:p] = np.array(G0.xrow("LEX5B_6000", ri, 1)[0]) - np.array(G0.xrow("HR_6000", ri, 1)[0])
            per[G0.RN[ri]] = dict(logOR=float(v @ r["theta"]), se=float(np.sqrt(v @ r["cov"] @ v)))
        rec["per_reader_LEX5B_vs_HR_6000"] = per
    return rec


def main():
    t0 = time.time()
    res = {"method": {"AGQ": "ML, adaptive Gauss-Hermite quadrature, 25 nodes, question random intercept",
                      "LAPLACE": "ML, Laplace approximation (lme4 nAGQ=1 equivalent), block-Newton joint mode"}, "fits": {}}
    F = res["fits"]
    for scope, qset in (("answerable90", G0.D.ans), ("all96", G0.D.qids)):
        for fill in (0, 1):
            for form in (0, 1, 2):
                X, y, qi, ci, names, nq, nc = G0.build(qset, fill, form)
                for est, re in (("AGQ", "q"), ("LAPLACE", "q"), ("LAPLACE", "cq")):
                    key = "%s|%s|unres=%d|form=%d|re=%s" % (est, scope, fill, form, re)
                    r = A.fit(X, y, qi, ci, nq, nc, "q") if est == "AGQ" else fit_laplace(X, y, qi, ci, nq, nc, re)
                    F[key] = rec = record(r, names, form, re, y, nq, nc)
                    c = rec["contrasts"]
                    print("%-46s nll=%.4f conv=%s g=%.1e sq=%.3f sc=%s LEX-HYB %.3f(%.3f) LEX-HR %.3f(%.3f) (%.0fs)" % (
                        key, rec["nll"], rec["converged"], rec["grad_max"], rec["sigma_question"],
                        "%.3f" % rec["sigma_component"] if rec["sigma_component"] is not None else "-",
                        c["LEX5B_6000 vs HYBRID_6000"]["logOR"], c["LEX5B_6000 vs HYBRID_6000"]["se"],
                        c["LEX5B_6000 vs HR_6000"]["logOR"], c["LEX5B_6000 vs HR_6000"]["se"], time.time() - t0), flush=True)
    lrt = {}
    for est, re in (("AGQ", "q"), ("LAPLACE", "q"), ("LAPLACE", "cq")):
        for scope in ("answerable90", "all96"):
            for fill in (0, 1):
                b = "%s|%s|unres=%d|form=%%d|re=%s" % (est, scope, fill, re)
                for hi, lo, what in ((1, 0, "pipeline:reader"), (2, 1, "budget:reader + pipeline:budget:reader"), (2, 0, "all reader interactions")):
                    st = 2 * (F[b % lo]["nll"] - F[b % hi]["nll"]); df = F[b % hi]["n_fixed"] - F[b % lo]["n_fixed"]
                    lrt["%s|%s|unres=%d|re=%s: form%d vs form%d (%s)" % (est, scope, fill, re, hi, lo, what)] = dict(
                        stat=st, df=df, p=float(chi2.sf(max(st, 0), df)))
    for scope in ("answerable90", "all96"):
        for fill in (0, 1):
            for form in (0, 1, 2):
                bq, bc = ("LAPLACE|%s|unres=%d|form=%d|re=q" % (scope, fill, form), "LAPLACE|%s|unres=%d|form=%d|re=cq" % (scope, fill, form))
                st = 2 * (F[bq]["nll"] - F[bc]["nll"])
                lrt["LAPLACE|%s|unres=%d|form=%d: re=cq vs re=q (component variance; boundary, chi2(1)/2)" % (scope, fill, form)] = dict(
                    stat=st, df=1, p=float(0.5 * chi2.sf(max(st, 0), 1)))
    res["lrt"] = lrt
    for k, v in lrt.items():
        print("LRT %-100s stat=%.3f df=%d p=%.4g" % (k, v["stat"], v["df"], v["p"]))
    (L.OUT / "glmm_final_results.json").write_text(json.dumps(res, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
