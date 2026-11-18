"""What arXiv:2608.08390 (Mesforush & Parhi, lasso universality under X = AZB) actually lets us do with LEX's window
design - the two experiments it licenses, run on the training dump of tools/lex_learn.py. No model call, no GPU.

THE THEOREM, in the terms we need. y = X beta* + xi, X = A Z B: Z has i.i.d. mean-zero unit-variance sub-Gaussian
entries, A (n x n) mixes ROWS and B (d x d) mixes COLUMNS, both bounded in operator norm; d/n -> gamma in (0, inf);
beta* is k-sparse with k = o(n) ("the sparse regime" is about the SIGNAL, not about d > n); xi ~ N(0, sigma^2 I);
lambda FIXED. Then (Prop. 1) the support-restricted "oracle" lasso concentrates on a deterministic problem that sees
A only through tau_n = tr(A'A)/n and B only through B_S'B_S; (Thm. 1) under a directional strict dual-feasibility
condition (Assumption 5) the full lasso EQUALS the oracle lasso w.h.p. - no false positives; (Cor. 2) the squared
error is the same whether Z is Gaussian or not. Their simulations (Figs. 4-6) plant geometric / power / anchor-diffuse
signals with k = n^(1/3), sigma = 0.2, lambda = gamma, and swap Gaussian, Rademacher, uniform and centred-Bernoulli
latent matrices under synthetic A and B.

    twin      THE PAPER'S EXPERIMENT ON OUR REAL DESIGN. The candidate-window matrix X (4122 windows x ~360 features)
              is written EXACTLY as A Z-hat B: B = the column covariance's square root, A = a per-record
              compound-symmetry block (the windows of one question share terms and reports), Z-hat = A^-1 X B^-1 -
              our empirical latent matrix, visibly non-Gaussian (binary indicators). Its Gaussian twin is A G B. The
              paper's planted signals are fitted on both, full and oracle, over lambda. If the squared errors match
              and full = oracle, our retrieval design sits in the theorem's universality class and Gaussian-design
              lasso theory may be used for it; if full != oracle, Assumption 5 fails on our columns.
    support   THE SPARSE REGIME ON THE REAL LABELS. The earlier fits chose lambda by cross-validation and came out
              DENSE (230-284 of ~290 coefficients nonzero): outside Assumption 1, so the theorem said nothing about
              them. Here lambda is fixed to give k = 5..40 nonzero phrases (k << n), the hand-set score enters as an
              UNPENALISED column (so a large lambda gives back exactly the hand-set ranking and the lasso only adds
              phrase corrections), and we report out of fold (same folds as before), the empirical dual-feasibility
              margin (Assumption 5's eta) and which phrases are selected stably across folds and resamples.

    .venv/Scripts/python.exe -m ragbench.tools.lex_universality twin    [--trials 20]
    .venv/Scripts/python.exe -m ragbench.tools.lex_universality support
    .venv/Scripts/python.exe -m ragbench.tools.lex_universality selftest

Out: work/runs/lex_learn_univ/{TWIN.md, SUPPORT.md, twin.json, support.json}.
CAVEATS (stated in the reports too): our response is 0/1, not a linear signal plus Gaussian noise, so the real-label
fits are outside the theorem's noise model - which is why the universality check uses the paper's planted signals;
and a finite n cannot verify ASYMPTOTIC operator-norm bounds, only report them.
"""
import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from ..approaches.lex import learn as LN
from .lex_learn import _chars_recall, _recall_at_k, load

ROOT = Path(__file__).resolve().parents[2]
DUMP = ROOT / "work" / "runs" / "lex_learn" / "windows.jsonl"
FOLDS = ROOT / "work" / "runs" / "lex_learn_1se" / "folds.json"
OUT = ROOT / "work" / "runs" / "lex_learn_univ"


# ---------------------------------------------------------------------------------------------------------------
# a fast lasso in Gram form (n > d here), with per-coordinate penalty factors
# ---------------------------------------------------------------------------------------------------------------

def lasso_gram(G, c, lam, pen=None, sweeps=1000, tol=1e-9, b0=None):
    """min_b 1/2 b'Gb - c'b + lam sum pen_j |b_j|  (G = X'X/n, c = X'y/n, X and y centred): the lasso (1.2) of the
    paper in Gram form, by coordinate descent (exact zeros, exact KKT). -> (b, grad) where grad = c - G b =
    X'(y - Xb)/n, the KKT correlations at the solution. `b0` warm-starts it."""
    d = len(c)
    pen = np.ones(d) if pen is None else np.asarray(pen, dtype=float)
    b = np.zeros(d) if b0 is None else np.array(b0, dtype=float)
    grad = np.array(c, dtype=float) - (G @ b if b0 is not None else 0.0)
    diag = np.diag(G).copy()
    for j in np.where((diag <= 0) & (b != 0))[0]:     # no variance (e.g. a phrase absent from a resample): no weight
        grad += G[:, j] * b[j]
        b[j] = 0.0
    active = list(range(d))
    for _ in range(sweeps):
        big = 0.0
        for j in active:
            if diag[j] <= 0:
                continue
            z = grad[j] + diag[j] * b[j]
            t = lam * pen[j]
            nb = math.copysign(max(abs(z) - t, 0.0), z) / diag[j]
            dl = nb - b[j]
            if dl != 0.0:
                grad -= G[:, j] * dl
                b[j] = nb
                big = max(big, abs(dl))
        if len(active) == d:
            if big < tol:
                break
            active = [j for j in range(d) if b[j] != 0.0] or list(range(d))
        elif big < tol:
            active = list(range(d))
    return b, grad


def lasso_fista(G, c, lam, pen=None, iters=5000, tol=1e-10):
    """The same problem by FISTA with adaptive restart (the solver the paper's simulations use), vectorised: fast to
    get near the solution; `solve` then polishes with coordinate descent for exact zeros and KKT."""
    d = len(c)
    pen = np.ones(d) if pen is None else np.asarray(pen, dtype=float)
    L = float(np.linalg.eigvalsh(G).max())
    thr = lam * pen / L
    b = np.zeros(d)
    v, t = b.copy(), 1.0
    for _ in range(iters):
        z = v - (G @ v - c) / L
        nb = np.sign(z) * np.maximum(np.abs(z) - thr, 0.0)
        if np.dot(nb - b, v - nb) > 0:            # restart when momentum points uphill
            t = 1.0
        nt = (1.0 + math.sqrt(1.0 + 4.0 * t * t)) / 2.0
        v = nb + ((t - 1.0) / nt) * (nb - b)
        if np.max(np.abs(nb - b)) < tol:
            b = nb
            break
        b, t = nb, nt
    return b


def solve(G, c, lam, pen=None, b0=None):
    """FISTA to get close (or a warm start), then coordinate descent to finish exactly."""
    if b0 is None:
        b0 = lasso_fista(G, c, lam, pen)
    return lasso_gram(G, c, lam, pen, b0=b0)


def dual_margin(b, grad, lam, pen=None):
    """Assumption 5, measured: eta = 1 - max over INACTIVE penalised j of |x_j'r/n| / lam. The KKT conditions force
    eta >= 0; eta well above 0 is STRICT dual feasibility (no inactive column is about to enter, the support is
    stable); eta near 0 means one is on the edge."""
    pen = np.ones(len(b)) if pen is None else np.asarray(pen)
    inact = [j for j in range(len(b)) if b[j] == 0.0 and pen[j] > 0]
    if not inact:
        return None
    return float(1.0 - max(abs(grad[j]) / (lam * pen[j]) for j in inact))


# ---------------------------------------------------------------------------------------------------------------
# the design
# ---------------------------------------------------------------------------------------------------------------

def dedupe(X, names, thr=0.999, tol=1e-8):
    """Drop every column that is a linear combination of the columns before it (Gram-Schmidt on standardised
    columns). Pairs: nested phrases («اگهی رد» inside «درصد اگهی رد شده»), synonyms the model always picks together,
    complementary kind indicators; triples: n_terms = n_phrase + n_relaxed. Such columns in one support break the
    paper's Assumption 4 (lambda_min of X_S'X_S/n >= kappa > 0) and make the lasso non-unique.
    -> (X, names, {kept name or "(combination)": [dropped names]})."""
    n = len(X)
    sd = X.std(axis=0)
    Z = (X - X.mean(axis=0)) / np.where(sd > 1e-12, sd, 1.0)
    C = np.abs(Z.T @ Z / n)
    Q = np.zeros((n, 0))
    keep, merged = [], {}
    for j in range(X.shape[1]):
        v = Z[:, j]
        r = v - Q @ (Q.T @ v) if Q.shape[1] else v
        if float(r @ r) / n > tol:
            keep.append(j)
            Q = np.column_stack([Q, r / math.sqrt(float(r @ r))])
        else:
            twin = next((i for i in keep if C[i, j] > thr), None)
            merged.setdefault(names[twin] if twin is not None else "(combination)", []).append(names[j])
    return X[:, keep], [names[j] for j in keep], merged


def design(min_df=2, with_hand=False):
    """The dump -> (X standardised, y, names, record group per row, question per row, hand score per row, records)."""
    data, recs = load(DUMP)
    X, y, names = data.matrix(min_df=min_df)
    grp, qid, hand = [], [], []
    for ri, r in enumerate(recs):
        for w in r["windows"]:
            grp.append(ri)
            qid.append(r["question_id"])
            hand.append(float(w["hand"] or 0.0))
    hand = np.array(hand)
    if with_hand:
        X = np.column_stack([X, hand])
        names = list(names) + ["hand"]
    sd = X.std(axis=0)
    keep = sd > 1e-12
    X = (X[:, keep] - X[:, keep].mean(axis=0)) / sd[keep]
    names = [n for n, k in zip(names, keep) if k]
    X, names, merged = dedupe(X, names)
    design.merged = merged
    return X, y, names, np.array(grp), np.array(qid), hand, recs


def sym_sqrt(S, floor):
    ev, V = np.linalg.eigh(S)
    ev = np.maximum(ev, floor)
    return (V * np.sqrt(ev)) @ V.T, (V / np.sqrt(ev)) @ V.T, ev


def cs_apply(M, groups, rho, power):
    """Apply R_g^power blockwise, R_g = (1 - rho) I + rho 11' (compound symmetry, size m): eigenvalue 1 - rho on the
    complement of 1, 1 + (m - 1) rho on 1."""
    out = np.empty_like(M)
    for idx in groups:
        m = len(idx)
        a = (1.0 - rho) ** power
        e = (1.0 + (m - 1) * rho) ** power
        blk = M[idx]
        mean = blk.mean(axis=0, keepdims=True)
        out[idx] = a * (blk - mean) + e * mean
    return out


def decompose(X, grp, floor_rel=1e-3):
    """X = A Z-hat B exactly: B from the column covariance (eigenvalues floored), A compound symmetry per record with
    rho estimated from the column-whitened rows."""
    n, d = X.shape
    S = X.T @ X / n
    floor = floor_rel * float(np.trace(S)) / d
    B, Binv, ev = sym_sqrt(S, floor)
    W = X @ Binv
    groups = [np.where(grp == g)[0] for g in np.unique(grp)]
    num = den = 0.0
    for idx in groups:
        m = len(idx)
        if m < 2:
            continue
        s = W[idx].sum(axis=0)
        num += float(s @ s) - float((W[idx] ** 2).sum())
        den += d * m * (m - 1)
    rho = min(max(num / den if den else 0.0, 0.0), 0.95)
    Zhat = cs_apply(W, groups, rho, -0.5)
    m_max = max(len(i) for i in groups)
    info = {"n": n, "d": d, "gamma": d / n, "rho_within_record": rho, "records": len(groups),
            "record_size_max": m_max, "record_size_median": float(np.median([len(i) for i in groups])),
            "A_op_norm": math.sqrt(1.0 + (m_max - 1) * rho), "tau_n": 1.0,
            "B_op_norm": float(math.sqrt(ev.max())), "B_min_sv": float(math.sqrt(ev.min())),
            "Zhat_var": float(Zhat.var()), "Zhat_kurtosis": float(((Zhat - Zhat.mean()) ** 4).mean() / Zhat.var() ** 2),
            "Zhat_share_near_mode": float(np.mean(np.abs(Zhat - np.median(Zhat)) < 0.05))}
    return B, groups, rho, Zhat, info


# ---------------------------------------------------------------------------------------------------------------
# twin: the paper's simulation protocol on our design
# ---------------------------------------------------------------------------------------------------------------

def profile(kind, k, rng):
    s = rng.choice([-1.0, 1.0], size=k)
    i = np.arange(1, k + 1, dtype=float)
    if kind == "geometric":                                   # (2.2), r = 0.25
        r = 0.25
        v = np.sqrt((1 - r ** 2) / (1 - r ** (2 * k))) * r ** (i - 1)
    elif kind == "power":                                     # (2.4), p = 1.5
        v = i ** -1.5 / math.sqrt(float((i ** -3.0).sum()))
    else:                                                     # (2.3), r = 0.25, theta = 0.75
        r, th = 0.25, 0.75
        v = np.empty(k)
        v[0] = math.sqrt(th)
        v[1:] = np.sqrt((1 - th) * (1 - r ** 2) / (1 - r ** (2 * (k - 1)))) * r ** (i[1:] - 2)
    return s * v


def fit_pair(X, y, S, lam):
    """Full and oracle lasso, plus the primal-dual witness check: the oracle solution IS a full-lasso solution iff
    every inactive column's correlation with its residual stays strictly below lambda (Theorem 1's mechanism)."""
    n = X.shape[0]
    G, c = X.T @ X / n, X.T @ y / n
    b, grad = solve(G, c, lam)
    bo, _g = solve(G[np.ix_(S, S)], c[S], lam)
    full_o = np.zeros_like(b)
    full_o[S] = bo
    g_or = c - G @ full_o
    fit_pair.last_oracle = bo
    out = np.ones(len(b), dtype=bool)
    out[S] = False
    cert = bool(np.max(np.abs(g_or[out])) < lam * (1 - 1e-9)) if out.any() else True
    return b, full_o, grad, cert


def cmd_twin(a):
    rng = np.random.RandomState(a.seed)
    X, _y, names, grp, _q, _h, _recs = design(min_df=2)
    B, groups, rho, Zhat, info = decompose(X, grp)
    n, d = X.shape
    Bsq = B @ B
    k = int(n ** (1.0 / 3.0))                                  # the paper's k = n^(1/3)
    gamma = d / n
    lams = [gamma * f for f in a.lam_factors]
    designs = ["real (A Z-hat B)", "Gaussian twin (A G B)", "twin, rows independent (G B)", "i.i.d. Gaussian"]
    res = {p: {dn: {lam: defaultdict(list) for lam in lams} for dn in designs}
           for p in ("geometric", "power", "anchor-diffuse")}
    kappa = []
    C = np.corrcoef(X.T)
    np.fill_diagonal(C, 0.0)
    for t in range(a.trials):
        if a.support == "clustered":
            # Assumption 5 under stress: a seed column and its most correlated neighbours, every other one active,
            # so each active column has an INACTIVE near-twin (phrases that co-occur in the same reports)
            seed = int(rng.randint(d))
            near = [seed] + [int(j) for j in np.argsort(-np.abs(C[seed]))[:2 * k]]
            S = sorted(set(near[0::2][:k]))
            while len(S) < k:
                S = sorted(set(S) | {int(rng.randint(d))})
        else:
            S = sorted(rng.choice(d, size=k, replace=False).tolist())
        G0 = rng.randn(n, d)
        mats = {designs[0]: X,
                designs[1]: cs_apply(G0, groups, rho, 0.5) @ B,
                designs[2]: G0 @ B,
                designs[3]: rng.randn(n, d)}
        kappa.append(float(np.linalg.eigvalsh(X[:, S].T @ X[:, S] / n).min()))     # Assumption 4 on the real design
        for p in res:
            beta = np.zeros(d)
            beta[S] = profile(p, k, rng)
            xi = a.sigma * rng.randn(n)
            for dn, M in mats.items():
                yv = M @ beta + xi
                yv = yv - yv.mean()
                Mc = M - M.mean(axis=0)
                for lam in lams:
                    b, bo, grad, cert = fit_pair(Mc, yv, S, lam)
                    rr = res[p][dn][lam]
                    rr["se_full"].append(float(((b - beta) ** 2).sum()))
                    rr["se_oracle"].append(float(((bo - beta) ** 2).sum()))
                    fp = [j for j in range(d) if b[j] != 0.0 and j not in set(S)]
                    rr["false_pos"].append(len(fp))
                    rr["full_eq_oracle"].append(float(cert))
                    rr["eta"].append(dual_margin(b, grad, lam))
                    # Prop. 1: the oracle lasso concentrates on argmin 1/2 (b*_S - b)'Q(b*_S - b) + lam|b|_1 with
                    # Q = tau_n B_S'B_S; tau_n = 1 here (A has a unit diagonal) and B'B = the floored column covariance
                    # (i.i.d. Gaussian: B = I)
                    Q = np.eye(k) if dn == designs[3] else Bsq[np.ix_(S, S)]
                    bdet, _g = solve(Q, Q @ beta[S], lam)
                    rr["dist_det"].append(float(np.sqrt(((fit_pair.last_oracle - bdet) ** 2).sum())))
        print("trial %d/%d" % (t + 1, a.trials), flush=True)

    def m(v):
        v = [x for x in v if x is not None]
        return float(np.mean(v)) if v else float("nan")

    OUT.mkdir(parents=True, exist_ok=True)
    lines = ["# The paper's universality experiment on LEX's real window design", "",
             "arXiv:2608.08390, Figs. 4-6 protocol: k = floor(n^(1/3)) = %d planted coefficients (geometric r=0.25, "
             "power p=1.5, anchor-diffuse r=0.25 theta=0.75), ||beta*|| = 1, sigma = 0.2, lambda a multiple of "
             "gamma = d/n; %d trials, support redrawn each trial (%s supports)." % (k, a.trials, a.support), "",
             "## The design, written as A Z-hat B", "",
             "| n (windows) | d (features) | gamma = d/n | records (A blocks) | median / max block | rho within record "
             "| op-norm A | op-norm B | smallest sv of B |", "|---|---|---|---|---|---|---|---|---|",
             "| %d | %d | %.3f | %d | %.0f / %d | %.3f | %.2f | %.2f | %.3f |" % (
                 n, d, gamma, info["records"], info["record_size_median"], info["record_size_max"], rho,
                 info["A_op_norm"], info["B_op_norm"], info["B_min_sv"]), "",
             "Z-hat (our empirical latent matrix): variance %.3f, kurtosis %.1f (Gaussian: 3), %.0f%% of entries "
             "within 0.05 of the median - far from Gaussian, which is what makes the comparison informative." % (
                 info["Zhat_var"], info["Zhat_kurtosis"], 100 * info["Zhat_share_near_mode"]), "",
             "Exactly collinear columns merged before anything else (Assumption 4): %d (e.g. %s)." % (
                 sum(len(v) for v in design.merged.values()),
                 "; ".join("%s = %s" % (k2, v[0]) for k2, v in list(design.merged.items())[:3])), "",
             "Assumption 4 on the real design (smallest eigenvalue of X_S'X_S/n over the drawn supports): median "
             "%.3f, min %.3f." % (float(np.median(kappa)), float(np.min(kappa))), ""]
    for p in res:
        lines += ["## %s profile" % p, "",
                  "| lambda / gamma | design | SE full | SE oracle | false positives | full = oracle | dual margin eta "
                  "| oracle to deterministic (Prop. 1) |", "|---|---|---|---|---|---|---|---|"]
        for lam, f in zip(lams, a.lam_factors):
            for dn in designs:
                rr = res[p][dn][lam]
                lines.append("| %.2f | %s | %.4f | %.4f | %.1f | %.0f%% | %.3f | %.4f |" % (
                    f, dn, m(rr["se_full"]), m(rr["se_oracle"]), m(rr["false_pos"]),
                    100 * m(rr["full_eq_oracle"]), m(rr["eta"]), m(rr["dist_det"])))
        lines.append("")
    text = "\n".join(lines)
    (OUT / ("TWIN_%s.md" % a.support)).write_text(text + "\n", encoding="utf-8")
    (OUT / ("twin_%s.json" % a.support)).write_text(json.dumps({"info": info, "k": k, "lams": lams, "kappa": kappa, "results": {
        p: {dn: {str(l): {kk: m(v) for kk, v in rr.items()} for l, rr in dd.items()} for dn, dd in pp.items()}
        for p, pp in res.items()}}, indent=1), encoding="utf-8")
    print(text)
    return 0


# ---------------------------------------------------------------------------------------------------------------
# support: the sparse regime on the real labels
# ---------------------------------------------------------------------------------------------------------------

def lam_path(G, c, pen, kmax, n=80, ratio=1e-3):
    """A decreasing penalty path with warm starts, stopped once more than kmax penalised coefficients are nonzero.
    -> [(lam, b, grad, nonzero)]. The unpenalised columns are fitted first (the path's start)."""
    free = np.where(pen == 0)[0]
    b = np.zeros(len(c))
    if len(free):
        b[free] = np.linalg.lstsq(G[np.ix_(free, free)], c[free], rcond=None)[0]
    grad0 = c - G @ b
    hi = float(np.max(np.abs(grad0[pen > 0]))) * 1.001
    out = []
    for i in range(n):
        lam = hi * ratio ** (i / (n - 1))
        b, grad = lasso_gram(G, c, lam, pen, b0=b)
        nz = int(np.sum((b != 0.0) & (pen > 0)))
        out.append((lam, b.copy(), grad.copy(), nz))
        if nz > kmax:
            break
    return out


def lam_for_k(G, c, pen, k, path=None):
    """The path point whose nonzero count is closest to k (ties: the larger penalty)."""
    path = path or lam_path(G, c, pen, k)
    return min(path, key=lambda t: (abs(t[3] - k), -t[0]))


def standardise(Xtr, Xte):
    mu, sd = Xtr.mean(axis=0), Xtr.std(axis=0)
    sd[sd < 1e-12] = 1.0
    return (Xtr - mu) / sd, (Xte - mu) / sd


def cmd_support(a):
    data, recs = load(DUMP)
    Xraw, y, names = data.matrix(min_df=2)
    Xraw, names, merged = dedupe(Xraw, list(names))
    hand = np.array([float(w["hand"] or 0.0) for r in recs for w in r["windows"]])
    rowrec = np.array([ri for ri, r in enumerate(recs) for _w in r["windows"]])
    rowq = np.array([r["question_id"] for r in recs for _w in r["windows"]])
    assign = json.loads(FOLDS.read_text(encoding="utf-8"))["assign"]
    ks = a.ks
    variants = {"offset": True, "plain": False}                 # hand-set score as an unpenalised column, or not
    oof = {(v, k): np.zeros(len(y)) for v in variants for k in ks}
    margins = defaultdict(list)
    lamk = defaultdict(list)
    for f in sorted(set(assign.values())):
        tr = np.array([assign.get(q) is not None and assign[q] != f for q in rowq])
        te = np.array([assign.get(q) == f for q in rowq])
        for v, off in variants.items():
            Xf = np.column_stack([Xraw, hand]) if off else Xraw
            pen = np.ones(Xf.shape[1])
            if off:
                pen[-1] = 0.0
            Xtr, Xte = standardise(Xf[tr], Xf[te])
            ytr = y[tr] - y[tr].mean()
            G, c = Xtr.T @ Xtr / len(ytr), Xtr.T @ ytr / len(ytr)
            path = lam_path(G, c, pen, max(ks))
            for k in ks:
                lam, b, grad, nz = lam_for_k(G, c, pen, k, path)
                oof[(v, k)][te] = Xte @ b
                margins[(v, k)].append(dual_margin(b, grad, lam, pen))
                lamk[(v, k)].append((lam, nz))
        print("fold %d done" % f, flush=True)

    # out-of-fold retrieval, the same metrics as tools/lex_learn.py report
    def evaluate(score):
        rows = defaultdict(list)
        for i in range(len(y)):
            if assign.get(rowq[i]) is None:
                continue
            rows[rowrec[i]].append({"gold": bool(y[i] > 0), "hand": hand[i], "s": score[i],
                                    "chars": recs[rowrec[i]]["windows"][0]["chars"]})
        # chars per window, in row order
        return rows
    chars = np.array([w["chars"] for r in recs for w in r["windows"]])
    by_rec = defaultdict(list)
    for i in range(len(y)):
        if assign.get(rowq[i]) is not None:
            by_rec[rowrec[i]].append(i)

    def read(score):
        k8, bud, better, worse, n = [], [], 0, 0, 0
        for _r, idx in by_rec.items():
            ws = [{"gold": bool(y[i] > 0), "chars": int(chars[i]), "i": i} for i in idx]
            h = _recall_at_k(ws, lambda w: hand[w["i"]], 8)
            s = _recall_at_k(ws, lambda w: score[w["i"]], 8)
            if h is None:
                continue
            n += 1
            k8.append(s)
            bud.append(_chars_recall(ws, lambda w: score[w["i"]]))
            better += s > h
            worse += s < h
        return float(np.mean(k8)), float(np.mean(bud)), better, worse, n

    hk8, hbud, _b, _w, nrec = read(hand)
    lines = ["# The sparse regime on the real labels - out of fold (same 5 folds as lex_learn_1se)", "",
             "%d exactly collinear columns merged first (Assumption 4); %d features remain." % (
                 sum(len(v) for v in merged.values()), Xraw.shape[1]), "",
             "Hand-set score: gold recall @ top 8 **%.3f**, @ 6000 chars **%.3f** (%d records)." % (hk8, hbud, nrec), "",
             "| model | k nonzero phrases/features | recall @8 | recall @6000 chars | better / worse vs hand | "
             "dual margin eta (min over folds) |", "|---|---|---|---|---|---|"]
    table = []
    for v in variants:
        for k in ks:
            k8, bud, bt, wr, _n = read(oof[(v, k)])
            eta = [e for e in margins[(v, k)] if e is not None]
            nzs = [nz for _l, nz in lamk[(v, k)]]
            lines.append("| %s | %s | %.3f | %.3f | %d / %d | %.3f |" % (
                "hand-set score + sparse corrections" if v == "offset" else "sparse lasso alone", "%d (got %s)" % (
                    k, "/".join(str(x) for x in nzs)), k8, bud, bt, wr, min(eta) if eta else float("nan")))
            table.append({"variant": v, "k": k, "recall8": k8, "recall_budget": bud, "better": bt, "worse": wr,
                          "eta": eta, "lam_nz": lamk[(v, k)]})

    # stability selection: fixed lambda (k = a.stab_k on all data), refit on question bootstrap resamples + folds
    Xf = np.column_stack([Xraw, hand])
    fnames = list(names) + ["hand"]
    pen = np.ones(Xf.shape[1])
    pen[-1] = 0.0
    Xs, _ = standardise(Xf, Xf[:1])
    yc = y - y.mean()
    G, c = Xs.T @ Xs / len(y), Xs.T @ yc / len(y)
    lam0, b0, grad0, nz0 = lam_for_k(G, c, pen, a.stab_k)
    rng = np.random.RandomState(a.seed)
    qs = sorted(set(rowq))
    counts, signs = defaultdict(int), defaultdict(float)
    reps = 0
    for rep in range(a.boot):
        pick = rng.choice(len(qs), size=len(qs), replace=True)
        w = np.zeros(len(y))
        qpos = {q: i for i, q in enumerate(qs)}
        mult = np.bincount(pick, minlength=len(qs))
        for i in range(len(y)):
            w[i] = mult[qpos[rowq[i]]]
        if w.sum() == 0:
            continue
        sw = w / w.sum()
        mu = sw @ Xf
        Xc = Xf - mu
        sd = np.sqrt(sw @ (Xc ** 2))
        sd[sd < 1e-12] = 1.0
        Xc = Xc / sd
        yy = y - sw @ y
        Gb = (Xc * sw[:, None]).T @ Xc
        cb = (Xc * sw[:, None]).T @ yy
        b, _g = solve(Gb, cb, lam0, pen, b0=b0)
        reps += 1
        for j in range(len(b)):
            if b[j] != 0.0 and pen[j] > 0:
                counts[fnames[j]] += 1
                signs[fnames[j]] += math.copysign(1.0, b[j])
    stable = sorted(((cnt / reps, n, signs[n] / cnt) for n, cnt in counts.items()), reverse=True)
    lines += ["", "## Stability selection (lambda fixed at %.5f = %d nonzero on all dev data, hand-set score "
              "unpenalised; %d question-bootstrap resamples)" % (lam0, nz0, reps), "",
              "Dual margin at that lambda on all data: eta = %.3f." % (dual_margin(b0, grad0, lam0, pen) or float("nan")),
              "", "| selected in | feature | mean sign |", "|---|---|---|"]
    for fr, n, sg in stable[:30]:
        lines.append("| %.0f%% | %s | %+.2f |" % (100 * fr, n, sg))
    lines += ["", "Columns selected in >= 80%% of resamples: %d of %d candidates." % (
        sum(1 for fr, _n, _s in stable if fr >= 0.8), Xf.shape[1] - 1)]
    OUT.mkdir(parents=True, exist_ok=True)
    text = "\n".join(lines)
    (OUT / "SUPPORT.md").write_text(text + "\n", encoding="utf-8")
    (OUT / "support.json").write_text(json.dumps({"hand": [hk8, hbud, nrec], "table": table, "stability": [
        {"freq": fr, "feature": n, "sign": sg} for fr, n, sg in stable], "lam0": lam0, "nz0": nz0}, indent=1,
        ensure_ascii=False), encoding="utf-8")
    print(text)
    return 0


def cmd_selftest(_a):
    """The Gram solver against learn.lasso (the one the method uses), and eta >= 0 at a solution."""
    rng = np.random.RandomState(1)
    X = rng.binomial(1, 0.3, size=(400, 30)).astype(float)
    beta = np.zeros(30)
    beta[[2, 9, 20]] = [1.5, -1.0, 0.7]
    y = X @ beta + 0.1 * rng.randn(400)
    lam = 0.02
    b0, b1 = LN.lasso(X, y, lam, n_sweeps=2000, tol=1e-12)
    mu, sd = X.mean(axis=0), X.std(axis=0)
    Z = (X - mu) / sd
    G, c = Z.T @ Z / 400, Z.T @ (y - y.mean()) / 400
    bg, grad = lasso_gram(G, c, lam)
    ok1 = np.allclose(bg / sd, b1, atol=1e-5)
    eta = dual_margin(bg, grad, lam)
    ok2 = eta is not None and eta >= -1e-9
    print(("ok  " if ok1 else "FAIL") + " Gram lasso = learn.lasso (max diff %.2e)" % float(np.abs(bg / sd - b1).max()))
    print(("ok  " if ok2 else "FAIL") + " dual margin at a solution is >= 0 (%.3f)" % eta)
    R = cs_apply(rng.randn(12, 4), [np.arange(0, 5), np.arange(5, 12)], 0.3, 0.5)
    back = cs_apply(cs_apply(R, [np.arange(0, 5), np.arange(5, 12)], 0.3, -0.5), [np.arange(0, 5), np.arange(5, 12)],
                    0.3, 0.5)
    ok3 = np.allclose(back, R)
    print(("ok  " if ok3 else "FAIL") + " compound-symmetry square root and its inverse round-trip")
    return 0 if (ok1 and ok2 and ok3) else 1


def main(argv=None):
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("twin")
    p.add_argument("--trials", type=int, default=20)
    p.add_argument("--sigma", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--lam-factors", type=float, nargs="+", default=[0.25, 0.5, 1.0, 2.0], dest="lam_factors")
    p.add_argument("--support", default="random", choices=["random", "clustered"])
    p = sub.add_parser("support")
    p.add_argument("--ks", type=int, nargs="+", default=[5, 10, 20, 40])
    p.add_argument("--stab-k", type=int, default=10, dest="stab_k")
    p.add_argument("--boot", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    sub.add_parser("selftest")
    a = ap.parse_args(argv)
    return {"twin": cmd_twin, "support": cmd_support, "selftest": cmd_selftest}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
