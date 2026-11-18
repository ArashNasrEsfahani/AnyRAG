"""Analysis 1 (final): GLMM by maximum likelihood with ADAPTIVE Gauss-Hermite quadrature for the question random
intercept (nodes re-centred at each question's conditional mode and scaled by its curvature, per component node), and
adaptive Gauss-Hermite (J nodes) over the component random intercept (mode/curvature by Newton with finite differences). Replaces glmm.py, whose non-adaptive quadrature
did not converge in K for the variance components (see RESULTS.md). Design matrices, codings and contrasts are those of
glmm.py. Gradient: importance-weighted score with node locations held fixed (exact at the AGQ approximation's limit);
BFGS; Wald SEs from a finite-difference Hessian of that gradient.

    .venv/Scripts/python.exe work/posthoc_2026-11/S5_stats/glmm_agq.py [Kq] [Jc]
"""
import json, os, sys, time
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp, expit
from scipy.stats import chi2
sys.argv += [] if len(sys.argv) > 1 else []
Kq = int(sys.argv[1]) if len(sys.argv) > 1 else 15
Jc = int(sys.argv[2]) if len(sys.argv) > 2 else 30
_argv = sys.argv; sys.argv = [sys.argv[0]]
import glmm as G0          # reuse build / xrow / contrast / CONTRASTS (its own K is irrelevant here)
sys.argv = _argv
import s5_lib as L

xq, wq = np.polynomial.hermite.hermgauss(Kq)
LWQ = np.log(wq)
xc, wc = np.polynomial.hermite.hermgauss(Jc)
ZC = np.sqrt(2.0) * xc
LWC = np.log(wc / np.sqrt(np.pi))


def prep(X, y, qi, ci, nq):
    R = X.shape[0] // nq
    order = np.argsort(qi, kind="stable")
    assert (np.bincount(qi) == R).all()
    Xs, ys = X[order].reshape(nq, R, -1), y[order].reshape(nq, R)
    qcomp = np.zeros(nq, int); qcomp[qi] = ci
    return Xs, ys, qcomp


def agq_parts(beta, sq, off, Xs, ys):
    """inner adaptive GH over b_q. off: Q x M offsets (component value per question and outer node).
    -> log-integral (Q x M), posterior node weights pik (Q x M x K), linear predictors ek (Q x R x M x K)."""
    e = (Xs @ beta)[:, :, None] + off[:, None, :]            # Q x R x M
    b = np.zeros(off.shape)
    yy = ys[:, :, None]
    for _ in range(60):                                       # Newton for the conditional mode of b_q (concave)
        p = expit(e + b[:, None, :])
        g1 = (yy - p).sum(1) - b / sq ** 2
        g2 = -(p * (1 - p)).sum(1) - 1 / sq ** 2
        step = g1 / g2
        b = b - step
        if np.abs(step).max() < 1e-11:
            break
    p = expit(e + b[:, None, :])
    s = 1 / np.sqrt((p * (1 - p)).sum(1) + 1 / sq ** 2)
    bk = b[:, :, None] + np.sqrt(2) * s[:, :, None] * xq[None, None, :]
    ek = e[:, :, :, None] + bk[:, None, :, :]
    ll = (yy[..., None] * ek - np.logaddexp(0, ek)).sum(1)
    lg = ll - bk ** 2 / (2 * sq ** 2) + xq[None, None, :] ** 2 + LWQ[None, None, :]
    lse = logsumexp(lg, axis=2)
    lqm = lse + np.log(np.sqrt(2) * s) - np.log(np.sqrt(2 * np.pi) * sq)
    return lqm, np.exp(lg - lse[:, :, None]), ek


def nll(theta, Xs, ys, qcomp, nc, use_c):
    """negative log marginal likelihood and its beta-gradient (fixed-node score; log-SD entries filled in by fit())."""
    p = Xs.shape[2]
    beta, sq = theta[:p], np.exp(theta[p])
    nq = Xs.shape[0]
    if not use_c:
        lqm, pik, ek = agq_parts(beta, sq, np.zeros((nq, 1)), Xs, ys)
        tot, W = lqm.sum(), pik
    else:
        sc = np.exp(theta[p + 1])
        # outer adaptive GH: Newton on u_c for h(u) = sum_q lq(u) - u^2/(2 sc^2), derivatives by central differences
        u, hh = np.zeros(nc), 1e-4
        for _ in range(40):
            off = (u[qcomp][:, None] + np.array([-hh, 0.0, hh])[None, :])
            lqm, _, _ = agq_parts(beta, sq, off, Xs, ys)
            T = np.zeros((nc, 3)); np.add.at(T, qcomp, lqm)
            T -= (u[:, None] + np.array([-hh, 0.0, hh])[None, :]) ** 2 / (2 * sc ** 2)
            g1 = (T[:, 2] - T[:, 0]) / (2 * hh)
            g2 = (T[:, 2] - 2 * T[:, 1] + T[:, 0]) / hh ** 2
            g2 = np.minimum(g2, -1 / sc ** 2 * 0.5)
            step = g1 / g2
            u = u - step
            if np.abs(step).max() < 1e-7:
                break
        su = 1 / np.sqrt(-g2)
        uj = u[:, None] + np.sqrt(2) * su[:, None] * xc[None, :]           # C x J
        lqm, pik, ek = agq_parts(beta, sq, uj[qcomp], Xs, ys)
        T = np.zeros((nc, Jc)); np.add.at(T, qcomp, lqm)
        B = T - uj ** 2 / (2 * sc ** 2) + xc[None, :] ** 2 + np.log(wc)[None, :]
        lse = logsumexp(B, axis=1)
        tot = (lse + np.log(np.sqrt(2) * su) - np.log(np.sqrt(2 * np.pi) * sc)).sum()
        W = np.exp(B - lse[:, None])[qcomp][:, :, None] * pik
    r = ys[:, :, None, None] - expit(ek)
    gb = np.einsum("qrp,qr->p", Xs, (r * W[:, None, :, :]).sum((2, 3)))
    g = np.zeros(len(theta)); g[:p] = gb
    return -tot, -g


def fit(X, y, qi, ci, nq, nc, re):
    Xs, ys, qcomp = prep(X, y, qi, ci, nq)
    p = X.shape[1]; use_c = re == "cq"
    f0 = lambda t: nll(t, Xs, ys, qcomp, nc, use_c)

    def f(t):
        """analytic beta-gradient; central differences for the log-SD parameters (whose fixed-node score is approximate)."""
        v, g = f0(t)
        for i in range(p, len(t)):
            e = np.zeros(len(t)); e[i] = 1e-5
            g[i] = (f0(t + e)[0] - f0(t - e)[0]) / 2e-5
        return v, g
    t0 = np.zeros(p + (2 if use_c else 1)); t0[p] = np.log(2.5)
    if use_c:
        t0[p + 1] = np.log(0.5)
    r = minimize(f, t0, jac=True, method="BFGS", options={"gtol": 1e-6, "maxiter": 3000})
    t, h = r.x, 1e-4
    H = np.zeros((len(t), len(t)))
    for i in range(len(t)):
        e = np.zeros(len(t)); e[i] = h
        H[:, i] = (f(t + e)[1] - f(t - e)[1]) / (2 * h)
    H = (H + H.T) / 2
    ev = np.linalg.eigvalsh(H)
    gmax = float(np.abs(f(t)[1]).max())
    # re-order theta to glmm.py layout (beta, [log sc], log sq) for contrast()
    if use_c:
        perm = list(range(p)) + [p + 1, p]
        t2, H2 = t[perm], H[np.ix_(perm, perm)]
    else:
        t2, H2 = t, H
    return dict(theta=t2, cov=np.linalg.inv(H2), nll=float(f(t)[0]), converged=bool((r.success or gmax < 2e-2) and ev.min() > 0), gmax=gmax,
                min_hess_eig=float(ev.min()), msg=str(r.message), nit=int(r.nit))


def main():
    t0 = time.time()
    results = {"method": "ML; adaptive Gauss-Hermite quadrature, %d nodes, for the question intercept (per component node); "
                         "%d-node adaptive Gauss-Hermite for the component intercept; BFGS; Wald SEs from finite-difference Hessian" % (Kq, Jc),
               "fits": {}}
    for scope, qset in (("answerable90", G0.D.ans), ("all96", G0.D.qids)):
        for fill in (0, 1):
            for form in (0, 1, 2):
                X, y, qi, ci, names, nq, nc = G0.build(qset, fill, form)
                for re in ("q", "cq"):
                    key = "%s|unres=%d|form=%d|re=%s" % (scope, fill, form, re)
                    r = fit(X, y, qi, ci, nq, nc, re)
                    p = len(names)
                    rec = dict(n=int(len(y)), n_correct=int(y.sum()), n_q=nq, n_comp=nc, n_fixed=p, loglik=-r["nll"], nll=r["nll"],
                               converged=r["converged"], grad_max=r["gmax"], min_hess_eig=r["min_hess_eig"], optimizer=r["msg"],
                               nit=r["nit"], sigma_question=float(np.exp(r["theta"][-1])),
                               sigma_component=float(np.exp(r["theta"][p])) if re == "cq" else None)
                    if re == "cq":
                        rec["log_sigma_component_se"] = float(np.sqrt(r["cov"][p, p]))
                    rec["log_sigma_question_se"] = float(np.sqrt(r["cov"][-1, -1]))
                    rec["contrasts"] = {k: G0.contrast(r, names, G0.sub(G0.arm_vec(a), G0.arm_vec(b))) for k, (a, b) in G0.CONTRASTS.items()}
                    # population-averaged accuracy: integrate total RE N(0, sq^2 + sc^2) with 60-node GH
                    zz, ww = np.polynomial.hermite.hermgauss(60)
                    sd = np.sqrt(rec["sigma_question"] ** 2 + (rec["sigma_component"] or 0) ** 2)
                    rec["pop_avg_acc_pct"] = {a: 100 * float(np.mean([(ww / np.sqrt(np.pi) * expit(
                        np.array(G0.xrow(a, ri, form)[0]) @ r["theta"][:p] + np.sqrt(2) * sd * zz)).sum() for ri in range(G0.NR)]))
                        for a in G0.ARMS}
                    if form == 1:
                        per = {}
                        for ri in range(G0.NR):
                            v = np.zeros(len(r["theta"]))
                            v[:p] = np.array(G0.xrow("LEX5B_6000", ri, 1)[0]) - np.array(G0.xrow("HR_6000", ri, 1)[0])
                            per[G0.RN[ri]] = dict(logOR=float(v @ r["theta"]), se=float(np.sqrt(v @ r["cov"] @ v)))
                        rec["per_reader_LEX5B_vs_HR_6000"] = per
                    results["fits"][key] = rec
                    c = rec["contrasts"]
                    print("%-34s nll=%.4f conv=%s g=%.1e sq=%.3f sc=%s LEX-HYB %.3f(%.3f) LEX-HR %.3f(%.3f) (%.0fs)" % (
                        key, r["nll"], r["converged"], r["gmax"], rec["sigma_question"],
                        "%.3f" % rec["sigma_component"] if rec["sigma_component"] is not None else "-",
                        c["LEX5B_6000 vs HYBRID_6000"]["logOR"], c["LEX5B_6000 vs HYBRID_6000"]["se"],
                        c["LEX5B_6000 vs HR_6000"]["logOR"], c["LEX5B_6000 vs HR_6000"]["se"], time.time() - t0), flush=True)
    lrt, F = {}, results["fits"]
    for scope in ("answerable90", "all96"):
        for fill in (0, 1):
            for re in ("q", "cq"):
                b = "%s|unres=%d|form=%%d|re=%s" % (scope, fill, re)
                for hi, lo, what in ((1, 0, "pipeline:reader"), (2, 1, "budget:reader + pipeline:budget:reader"), (2, 0, "all reader interactions")):
                    st = 2 * (F[b % lo]["nll"] - F[b % hi]["nll"]); df = F[b % hi]["n_fixed"] - F[b % lo]["n_fixed"]
                    lrt["%s|unres=%d|re=%s: form%d vs form%d (%s)" % (scope, fill, re, hi, lo, what)] = dict(stat=st, df=df, p=float(chi2.sf(max(st, 0), df)))
            for form in (0, 1, 2):
                bq, bc = "%s|unres=%d|form=%d|re=q" % (scope, fill, form), "%s|unres=%d|form=%d|re=cq" % (scope, fill, form)
                st = 2 * (F[bq]["nll"] - F[bc]["nll"])
                lrt["%s|unres=%d|form=%d: re=cq vs re=q (component variance; boundary, chi2(1)/2)" % (scope, fill, form)] = dict(
                    stat=st, df=1, p=float(0.5 * chi2.sf(max(st, 0), 1)))
    results["lrt"] = lrt
    for k, v in lrt.items():
        print("LRT %-100s stat=%.3f df=%d p=%.4g" % (k, v["stat"], v["df"], v["p"]))
    (L.OUT / ("glmm_agq_results_K%d_J%d.json" % (Kq, Jc))).write_text(json.dumps(results, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
