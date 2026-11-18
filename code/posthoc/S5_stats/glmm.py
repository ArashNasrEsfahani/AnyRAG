"""Analysis 1: mixed-effects logistic models, maximum likelihood with (non-adaptive) Gauss-Hermite quadrature
(own implementation, numpy/scipy; ML on the quadrature-approximated marginal likelihood, not Laplace, not Bayesian).
Random intercepts for question, and question nested in shared-family component. Fixed effects: arm (pipeline x budget,
8 cells, treatment vs HYBRID_6000) + reader (sum-to-zero coding, so arm effects are averages over the six readers on
the logit scale) [form 1: + pipeline:reader] [form 2: + budget:reader + pipeline:budget:reader = saturated fixed part].
Unresolved judge cells: fitted twice (all 0 / all 1).

    .venv/Scripts/python.exe work/posthoc_2026-11/S5_stats/glmm.py [K]
"""
import json, os, sys, time
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp, expit
from scipy.stats import chi2, norm
import s5_lib as L

D = L.load()
K = int(sys.argv[1]) if len(sys.argv) > 1 else 25
ARMS = ["HYBRID_6000", "HR_6000", "LEX5B_6000", "SETR_6000", "HYBRID_15500", "HR_15500", "LEX5B_15500", "SETR_15500"]
PIPES = ["HYBRID", "HR", "LEX5B", "SETR"]
READERS = D.readers
RN = [r.split("/")[1] for r in READERS]
NR = len(READERS)


def xrow(a, r, form):
    pipe, bud = a.split("_")
    x, nm = [1.0], ["(Intercept)"]
    for a2 in ARMS[1:]:
        x.append(float(a == a2)); nm.append("arm[%s]" % a2)
    rs = [(1.0 if r == k else 0.0) - (1.0 if r == NR - 1 else 0.0) for k in range(NR - 1)]
    x += rs; nm += ["reader[%s]" % RN[k] for k in range(NR - 1)]
    pd = [float(pipe == p) for p in PIPES[1:]]
    b15 = float(bud == "15500")
    if form >= 1:
        for pi, p in enumerate(PIPES[1:]):
            for k in range(NR - 1):
                x.append(pd[pi] * rs[k]); nm.append("%s:reader[%s]" % (p, RN[k]))
    if form >= 2:
        for k in range(NR - 1):
            x.append(b15 * rs[k]); nm.append("15500:reader[%s]" % RN[k])
        for pi, p in enumerate(PIPES[1:]):
            for k in range(NR - 1):
                x.append(pd[pi] * b15 * rs[k]); nm.append("%s:15500:reader[%s]" % (p, RN[k]))
    return x, nm


def build(qset, fill, form):
    rows, y, qi, ci = [], [], [], []
    qs = sorted(qset, key=lambda q: (str(D.q[q]["comp"]), q))
    comps = sorted({str(D.q[q]["comp"]) for q in qs})
    cidx = {c: i for i, c in enumerate(comps)}
    names = None
    for j, q in enumerate(qs):
        for r, rd in enumerate(READERS):
            for a in ARMS:
                lo, hi = D.m(rd, a, q)
                x, names = xrow(a, r, form)
                rows.append(x); y.append(lo if lo == hi else fill); qi.append(j); ci.append(cidx[str(D.q[q]["comp"])])
    return np.array(rows), np.array(y, float), np.array(qi), np.array(ci), names, len(qs), len(comps)


GH_x, GH_w = np.polynomial.hermite.hermgauss(K)
Z = np.sqrt(2.0) * GH_x
LW = np.log(GH_w / np.sqrt(np.pi))


def nll_q(theta, X, y, qi, nq):
    """question random intercept only. theta = (beta, log sigma_q)"""
    p = X.shape[1]
    beta, s = theta[:p], np.exp(theta[p])
    eta = (X @ beta)[:, None] + s * Z[None, :]
    ll = y[:, None] * eta - np.logaddexp(0, eta)
    S = np.zeros((nq, K)); np.add.at(S, qi, ll)
    A = LW[None, :] + S
    lq = logsumexp(A, axis=1)
    W = np.exp(A - lq[:, None])[qi]
    Wr = W * (y[:, None] - expit(eta))
    gb = X.T @ Wr.sum(1)
    gs = (Wr * Z[None, :]).sum() * s
    return -lq.sum(), -np.concatenate([gb, [gs]])


def nll_cq(theta, X, y, qi, ci, nq, nc, qcomp):
    """component + question-in-component random intercepts. theta = (beta, log sigma_c, log sigma_q)"""
    p = X.shape[1]
    beta, sc, sq = theta[:p], np.exp(theta[p]), np.exp(theta[p + 1])
    eta = (X @ beta)[:, None, None] + sc * Z[None, :, None] + sq * Z[None, None, :]
    ll = y[:, None, None] * eta - np.logaddexp(0, eta)
    S = np.zeros((nq, K, K)); np.add.at(S, qi, ll)
    A = LW[None, None, :] + S
    lqj = logsumexp(A, axis=2)
    pik = np.exp(A - lqj[:, :, None])
    T = np.zeros((nc, K)); np.add.at(T, qcomp, lqj)
    B = LW[None, :] + T
    lc = logsumexp(B, axis=1)
    pij = np.exp(B - lc[:, None])
    W = pij[ci][:, :, None] * pik[qi]
    Wr = W * (y[:, None, None] - expit(eta))
    gb = X.T @ Wr.sum((1, 2))
    gc = (Wr * Z[None, :, None]).sum() * sc
    gq = (Wr * Z[None, None, :]).sum() * sq
    return -lc.sum(), -np.concatenate([gb, [gc, gq]])


def fit(X, y, qi, ci, nq, nc, re):
    p = X.shape[1]
    qcomp = np.zeros(nq, int); qcomp[qi] = ci
    if re == "q":
        f = lambda t: nll_q(t, X, y, qi, nq); t0 = np.zeros(p + 1); t0[p] = np.log(1.5)
    else:
        f = lambda t: nll_cq(t, X, y, qi, ci, nq, nc, qcomp); t0 = np.zeros(p + 2); t0[p] = np.log(0.5); t0[p + 1] = np.log(1.5)
    r = minimize(f, t0, jac=True, method="BFGS", options={"gtol": 1e-5, "maxiter": 3000})
    t, h = r.x, 1e-4
    H = np.zeros((len(t), len(t)))
    for i in range(len(t)):
        e = np.zeros(len(t)); e[i] = h
        H[:, i] = (f(t + e)[1] - f(t - e)[1]) / (2 * h)
    H = (H + H.T) / 2
    ev = np.linalg.eigvalsh(H)
    cov = np.linalg.inv(H)
    gmax = float(np.abs(f(t)[1]).max())
    return dict(theta=t, cov=cov, nll=float(r.fun), converged=bool(gmax < 1e-3 and ev.min() > 0), gmax=gmax,
                min_hess_eig=float(ev.min()), msg=str(r.message), nit=int(r.nit))


def contrast(res, names, c):
    v = np.zeros(len(res["theta"]))
    for nm, w in c.items():
        v[names.index(nm)] = w
    est = float(v @ res["theta"]); se = float(np.sqrt(v @ res["cov"] @ v))
    return dict(logOR=est, se=se, OR=float(np.exp(est)), OR_ci95=[float(np.exp(est - 1.96 * se)), float(np.exp(est + 1.96 * se))],
                p_wald=float(2 * norm.sf(abs(est / se))))


def arm_vec(a):
    return {} if a == "HYBRID_6000" else {"arm[%s]" % a: 1.0}


def sub(c1, c2):
    out = dict(c1)
    for k, w in c2.items():
        out[k] = out.get(k, 0.0) - w
    return out


CONTRASTS = {"LEX5B_6000 vs HYBRID_6000": ("LEX5B_6000", "HYBRID_6000"), "LEX5B_6000 vs HR_6000": ("LEX5B_6000", "HR_6000"),
             "HR_6000 vs HYBRID_6000": ("HR_6000", "HYBRID_6000"), "SETR_6000 vs LEX5B_6000": ("SETR_6000", "LEX5B_6000"),
             "SETR_6000 vs HYBRID_6000": ("SETR_6000", "HYBRID_6000"),
             "budget HYBRID (15500-6000)": ("HYBRID_15500", "HYBRID_6000"), "budget HR (15500-6000)": ("HR_15500", "HR_6000"),
             "budget LEX5B (15500-6000)": ("LEX5B_15500", "LEX5B_6000"), "budget SETR (15500-6000)": ("SETR_15500", "SETR_6000")}


def pop_avg(res, names, form, sig_idx):
    """population-averaged accuracy (%) per arm, averaged over readers (point estimates; RE integrated by GH)."""
    p = len(names)
    beta = res["theta"][:p]
    sd = np.sqrt(sum(np.exp(res["theta"][i]) ** 2 for i in sig_idx))
    out = {}
    for a in ARMS:
        accs = [float((np.exp(LW) * expit(np.array(xrow(a, r, form)[0]) @ beta + sd * Z)).sum()) for r in range(NR)]
        out[a] = 100 * float(np.mean(accs))
    return out


def main():
    t0 = time.time()
    results = {"method": "ML, non-adaptive Gauss-Hermite quadrature, K=%d nodes per random effect (nested: K x K); "
                         "BFGS with analytic gradient; Wald SEs from finite-difference Hessian" % K, "fits": {}}
    for scope, qset in (("answerable90", D.ans), ("all96", D.qids)):
        for fill in (0, 1):
            for form in (0, 1, 2):
                X, y, qi, ci, names, nq, nc = build(qset, fill, form)
                for re in ("q", "cq"):
                    key = "%s|unres=%d|form=%d|re=%s" % (scope, fill, form, re)
                    r = fit(X, y, qi, ci, nq, nc, re)
                    p = len(names)
                    rec = dict(n=int(len(y)), n_correct=int(y.sum()), n_q=nq, n_comp=nc, n_fixed=p, loglik=-r["nll"], nll=r["nll"],
                               converged=r["converged"], grad_max=r["gmax"], min_hess_eig=r["min_hess_eig"],
                               optimizer=r["msg"], nit=r["nit"],
                               sigma_question=float(np.exp(r["theta"][p + (1 if re == "cq" else 0)])),
                               sigma_component=float(np.exp(r["theta"][p])) if re == "cq" else None)
                    rec["contrasts"] = {k: contrast(r, names, sub(arm_vec(a), arm_vec(b))) for k, (a, b) in CONTRASTS.items()}
                    rec["pop_avg_acc_pct"] = pop_avg(r, names, form, [p] if re == "q" else [p, p + 1])
                    if form == 1:   # per-reader LEX5B vs HR at 6000 implied by the pipeline:reader model
                        per = {}
                        for ri in range(NR):
                            xa = np.array(xrow("LEX5B_6000", ri, 1)[0]); xb = np.array(xrow("HR_6000", ri, 1)[0])
                            v = np.zeros(len(r["theta"])); v[:p] = xa - xb
                            per[RN[ri]] = dict(logOR=float(v @ r["theta"]), se=float(np.sqrt(v @ r["cov"] @ v)))
                        rec["per_reader_LEX5B_vs_HR_6000"] = per
                    results["fits"][key] = rec
                    c = rec["contrasts"]
                    print("%-34s n=%d nll=%.3f conv=%s g=%.1e sq=%.2f sc=%s LEX-HYB %.3f(%.3f) LEX-HR %.3f(%.3f) (%.0fs)" % (
                        key, len(y), r["nll"], r["converged"], r["gmax"], rec["sigma_question"],
                        "%.3f" % rec["sigma_component"] if rec["sigma_component"] is not None else "-",
                        c["LEX5B_6000 vs HYBRID_6000"]["logOR"], c["LEX5B_6000 vs HYBRID_6000"]["se"],
                        c["LEX5B_6000 vs HR_6000"]["logOR"], c["LEX5B_6000 vs HR_6000"]["se"], time.time() - t0), flush=True)
    lrt = {}
    F = results["fits"]
    for scope in ("answerable90", "all96"):
        for fill in (0, 1):
            for re in ("q", "cq"):
                b = "%s|unres=%d|form=%%d|re=%s" % (scope, fill, re)
                for hi, lo, what in ((1, 0, "pipeline:reader"), (2, 1, "budget:reader + pipeline:budget:reader"),
                                     (2, 0, "all reader interactions")):
                    st = 2 * (F[b % lo]["nll"] - F[b % hi]["nll"]); df = F[b % hi]["n_fixed"] - F[b % lo]["n_fixed"]
                    lrt["%s: form%d vs form%d (%s)" % (b.replace("|form=%d", ""), hi, lo, what)] = dict(stat=st, df=df, p=float(chi2.sf(st, df)))
            for form in (0, 1, 2):
                bq, bc = "%s|unres=%d|form=%d|re=q" % (scope, fill, form), "%s|unres=%d|form=%d|re=cq" % (scope, fill, form)
                st = 2 * (F[bq]["nll"] - F[bc]["nll"])
                lrt["%s|unres=%d|form=%d: re=cq vs re=q (component variance; boundary, chi2(1)/2)" % (scope, fill, form)] = dict(
                    stat=st, df=1, p=float(0.5 * chi2.sf(max(st, 0), 1)))
    results["lrt"] = lrt
    for k, v in lrt.items():
        print("LRT %-100s stat=%.2f df=%d p=%.4g" % (k, v["stat"], v["df"], v["p"]))
    (L.OUT / ("glmm_results_K%d.json" % K)).write_text(json.dumps(results, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
