"""Analyses 2-4 (no API calls, CPU only): equivalence / non-inferiority (TOST), minimum detectable effect, and
multiplicity correction across readers. Reads cells.json written by s5_data.py. Writes stats_results.json.

    .venv/Scripts/python.exe work/posthoc_2026-11/S5_stats/analyses.py
"""
import itertools, json, math, os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import numpy as np
from scipy.stats import binom, norm, t as tdist
import s5_lib as L

D = L.load()
P = D.primary
ANS = D.ans
N = len(ANS)
COMP = [D.q[q]["comp"] for q in ANS]
MARGINS = [3, 5, 8]          # points of 90, chosen post hoc
res = {"note": "post hoc; margins 3/5/8 points of 90 chosen after unsealing", "N": N}


# ------------------------------------------------------------------ 2. TOST / non-inferiority, LEX5B vs HR @6000
def cluster_t(diffs, clusters):
    """ratio-estimator mean (pp) with cluster-robust SE (CR1-type, G/(G-1)), df=G-1."""
    d = np.array(diffs, float); g = {}
    for x, c in zip(d, clusters):
        g.setdefault(c, []).append(x)
    G = len(g); mean = d.mean()
    u = np.array([sum(v) - len(v) * mean for v in g.values()])
    se = math.sqrt(G / (G - 1) * (u ** 2).sum()) / len(d)
    return 100 * mean, 100 * se, G - 1


def tost(fa, fb, label):
    lo, up = L.diff_vectors(fa, fb, ANS)
    out = {"net_bounds": [sum(lo), sum(up)], "pp_bounds": [100 * sum(lo) / N, 100 * sum(up) / N]}
    for scheme, cl in (("component", COMP), ("question", None)):
        s_lo, s_up = L.boot_sims(lo, cl), L.boot_sims(up, cl)
        B = len(s_lo)
        ci90 = [s_lo[int(0.05 * B)], s_up[int(0.95 * B) - 1]]
        ci95 = [s_lo[int(0.025 * B)], s_up[int(0.975 * B) - 1]]
        rows = {}
        for m in MARGINS:
            mpp = 100.0 * m / N
            p_lower = sum(x <= -mpp for x in s_lo) / B     # H0: diff <= -m  (worst-case extreme: lower vector)
            p_upper = sum(x >= mpp for x in s_up) / B      # H0: diff >= +m  (worst-case extreme: upper vector)
            rows["%d" % m] = {"margin_pp": mpp, "equivalent_at_5pct": bool(ci90[0] > -mpp and ci90[1] < mpp),
                              "tost_p": max(p_lower, p_upper), "p_lower": p_lower, "p_upper": p_upper,
                              "noninferior_at_5pct_one_sided": bool(ci90[0] > -mpp), "noninf_p": p_lower}
        out[scheme] = {"ci90_pp_union": ci90, "ci95_pp_union": ci95,
                       "ci90_points_of_90": [x * N / 100 for x in ci90],
                       "smallest_equivalence_margin_points": max(-ci90[0], ci90[1]) * N / 100,
                       "smallest_noninferiority_margin_points": -ci90[0] * N / 100, "margins": rows}
    # parametric check: cluster-robust t over components, at both extremes
    ct = {}
    for nm, v in (("lower_extreme", lo), ("upper_extreme", up)):
        mu, se, df = cluster_t(v, COMP)
        ct[nm] = {"mean_pp": mu, "se_pp": se, "df": df,
                  "ci90_pp": [mu - tdist.ppf(.95, df) * se, mu + tdist.ppf(.95, df) * se]}
    ct["tost_p_by_margin"] = {}
    for m in MARGINS:
        mpp = 100.0 * m / N
        a, b = ct["lower_extreme"], ct["upper_extreme"]
        pl = tdist.sf((a["mean_pp"] + mpp) / a["se_pp"], a["df"])
        pu = tdist.cdf((b["mean_pp"] - mpp) / b["se_pp"], b["df"])
        ct["tost_p_by_margin"]["%d" % m] = {"tost_p": float(max(pl, pu)), "noninf_p": float(pl)}
    out["cluster_t_check"] = ct
    return out


res["tost"] = {
    "machine LEX5B_6000 vs HR_6000": tost(lambda q: D.m(P, "LEX5B_6000", q), lambda q: D.m(P, "HR_6000", q), "machine"),
    "human_sym LEX5B_6000 vs HR_6000": tost(lambda q: D.h("LEX5B_6000", q), lambda q: D.h("HR_6000", q), "human"),
}


# ------------------------------------------------------------------ 3. MDE (paired binary, exact sign test)
def exact_power(n, psi, delta, alpha=0.05):
    """power of the exact two-sided sign test when P(a>b)=(psi+delta)/2, P(a<b)=(psi-delta)/2."""
    if delta >= psi:
        return 1.0
    th = (psi + delta) / (2 * psi)
    pw = 0.0
    pD = binom.pmf(np.arange(n + 1), n, psi)
    for Dd in range(n + 1):
        if pD[Dd] < 1e-14:
            continue
        b = np.arange(Dd + 1)
        k = np.minimum(b, Dd - b)
        cdf = binom.cdf(np.arange(Dd + 1), Dd, 0.5)
        pv = np.minimum(1.0, 2 * cdf[k])
        rej = pv <= alpha
        pw += pD[Dd] * binom.pmf(b, Dd, th)[rej].sum()
    return float(pw)


def mde_exact(n, psi, power=0.8):
    lo, hi = 0.0, psi
    for _ in range(60):
        mid = (lo + hi) / 2
        if exact_power(n, psi, mid) >= power:
            hi = mid
        else:
            lo = mid
    return hi


def mde_normal(n, psi, alpha=0.05, power=0.8):
    za, zb = norm.ppf(1 - alpha / 2), norm.ppf(power)
    lo, hi = 1e-9, psi
    for _ in range(100):
        d = (lo + hi) / 2
        need = (za * math.sqrt(psi) + zb * math.sqrt(max(psi - d * d, 0))) ** 2 / d ** 2
        if need <= n:
            hi = d
        else:
            lo = d
    return hi


def n_normal(psi, delta, alpha=0.05, power=0.8):
    za, zb = norm.ppf(1 - alpha / 2), norm.ppf(power)
    return (za * math.sqrt(psi) + zb * math.sqrt(psi - delta ** 2)) ** 2 / delta ** 2


def n_exact(psi, delta, power=0.8, start=50, stop=3000):
    first = None
    stable = None
    ok = [exact_power(n, psi, delta) >= power for n in range(start, stop + 1)]
    for i, v in enumerate(ok):
        if v and first is None:
            first = start + i
    for i in range(len(ok) - 1, -1, -1):
        if not ok[i]:
            stable = start + i + 1
            break
    return first, stable


mde = {}
for a, b in (("LEX5B_6000", "HYBRID_6000"), ("LEX5B_6000", "HR_6000")):
    fa, fb = (lambda q, a=a: D.m(P, a, q)), (lambda q, b=b: D.m(P, b, q))
    bb, ww = L.bw_resolved(fa, fb, ANS)
    nres = sum(1 for q in ANS if fa(q)[0] == fa(q)[1] and fb(q)[0] == fb(q)[1])
    # discordant-count range over every assignment of unresolved cells
    cells = [(q, t) for q in ANS for f, t in ((fa, "a"), (fb, "b")) if f(q)[0] != f(q)[1]]
    dcs = []
    for asg in itertools.product((0, 1), repeat=len(cells)):
        val = dict(zip(cells, asg))
        dcs.append(sum(val.get((q, "a"), fa(q)[0]) != val.get((q, "b"), fb(q)[0]) for q in ANS))
    psis = {"resolved": (bb + ww) / nres, "min_over_assignments": min(dcs) / N, "max_over_assignments": max(dcs) / N}
    lo, up = L.diff_vectors(fa, fb, ANS)
    # design effect from bootstrap SDs (component vs question), averaged over the two extremes
    sd = lambda s: float(np.std(s))
    deff = np.mean([(sd(L.boot_sims(v, COMP)) / sd(L.boot_sims(v, None))) ** 2 for v in (lo, up)])
    r = {"better_worse_resolved": [bb, ww], "n_resolved_pairs": nres, "discordant_range": [min(dcs), max(dcs)],
         "design_effect_component_vs_question": float(deff), "by_psi": {}}
    for nm, psi in psis.items():
        de, dn = mde_exact(N, psi), mde_normal(N, psi)
        d5 = 5 / N
        f_, s_ = n_exact(psi, d5)
        r["by_psi"][nm] = {"psi": psi, "mde_exact_points_of_90": de * N, "mde_exact_pp": 100 * de,
                           "mde_normal_points_of_90": dn * N, "mde_normal_pp": 100 * dn,
                           "mde_exact_points_with_deff": de * N * math.sqrt(deff),
                           "n_for_5_of_90_normal": n_normal(psi, d5), "n_for_5_of_90_exact_first": f_,
                           "n_for_5_of_90_exact_stable": s_, "n_for_5_of_90_normal_with_deff": n_normal(psi, d5) * deff,
                           "power_at_n90_for_5_of_90": exact_power(N, psi, d5)}
    mde["%s vs %s" % (a, b)] = r
res["mde"] = mde


# ------------------------------------------------------------------ 4. multiplicity across readers (tables/readers.tex)
def perm_p_exact(d):
    """exact two-sided sign-flip permutation p for sum of integer paired contrasts d (zeros drop out)."""
    a = [abs(int(x)) for x in d if x != 0]
    obs = abs(sum(int(x) for x in d))
    if not a:
        return 1.0
    tot = sum(a)
    dist = np.zeros(2 * tot + 1); dist[tot] = 1.0
    for v in a:
        nd = np.zeros_like(dist)
        nd[v:] += 0.5 * dist[:-v] if v else 0
        nd[:-v] += 0.5 * dist[v:]
        dist = nd
    s = np.arange(-tot, tot + 1)
    return float(min(1.0, dist[np.abs(s) >= obs - 1e-9].sum()))


def inter_range(rd, ctrl):
    arms = ["LEX5B_6000", ctrl + "_6000", "LEX5B_15500", ctrl + "_15500"]
    cells = [(q, a) for q in ANS for a in arms if D.m(rd, a, q)[0] != D.m(rd, a, q)[1]]
    ps = []
    for asg in itertools.product((0, 1), repeat=len(cells)):
        val = dict(zip(cells, asg))
        g = lambda q, a: val.get((q, a), D.m(rd, a, q)[0])
        d = [g(q, arms[0]) - g(q, arms[1]) - g(q, arms[2]) + g(q, arms[3]) for q in ANS]
        ps.append(perm_p_exact(d))
    return min(ps), max(ps), len(cells)


def holm(ps):
    order = sorted(range(len(ps)), key=lambda i: ps[i]); adj = [0.0] * len(ps); run = 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (len(ps) - rank) * ps[i])); adj[i] = run
    return adj


def bh(ps):
    m = len(ps); order = sorted(range(m), key=lambda i: ps[i], reverse=True); adj = [0.0] * m; run = 1.0
    for rank, i in enumerate(order):
        run = min(run, ps[i] * m / (m - rank)); adj[i] = run
    return adj


COLS = [("LEX5B_6000", "HYBRID_6000"), ("LEX5B_6000", "HR_6000"), ("LEX5B_15500", "HYBRID_15500"),
        ("LEX5B_15500", "HR_15500"), ("HYBRID_15500", "HYBRID_6000"), ("HR_15500", "HR_6000"),
        ("interaction", "HYBRID"), ("interaction", "HR")]
tests = []
for rd in D.readers:
    for a, b in COLS:
        if a == "interaction":
            plo, phi, k = inter_range(rd, b)
            nl, nh = L.inter_bounds(D, rd, b)
            name, test = "interaction vs %s" % b, "exact sign-flip permutation on per-question contrast"
        else:
            r = L.sign_range(lambda q: D.m(rd, a, q), lambda q: D.m(rd, b, q), ANS)
            plo, phi, k, nl, nh = r["p_lo"], r["p_hi"], r["k"], r["net_lo"], r["net_hi"]
            name, test = "%s vs %s" % (a, b), "exact sign test"
        tests.append(dict(reader=rd.split("/")[1], comparison=name, test=test, net=[nl, nh], unresolved_cells=k,
                          p_min=plo, p_max=phi))
fams = {"per_column_6_readers": lambda t: t["comparison"],
        "lex_vs_control_24": lambda t: "lex" if t["comparison"].startswith("LEX5B") else None,
        "whole_table_48": lambda t: "all"}
for fname, key in fams.items():
    groups = {}
    for i, t in enumerate(tests):
        g = key(t)
        if g is not None:
            groups.setdefault(g, []).append(i)
    for g, idx in groups.items():
        for bound in ("p_min", "p_max"):
            ps = [tests[i][bound] for i in idx]
            for meth, fn in (("holm", holm), ("bh", bh)):
                for i, v in zip(idx, fn(ps)):
                    tests[i]["%s_%s_%s" % (fname, meth, bound)] = v
res["readers_multiplicity"] = {"note": "adjusted p are monotone in each raw p, so adjusting the min-p and max-p vectors "
                                       "bounds the adjusted p over all assignments of unresolved cells",
                               "tests": tests}

(L.OUT / "stats_results.json").write_text(json.dumps(res, indent=1), encoding="utf-8")

# console summary
for k, v in res["tost"].items():
    print(k, "net", v["net_bounds"])
    for sch in ("component", "question"):
        s = v[sch]
        print("  %s 90%%CI pp [%.2f, %.2f] = points [%.2f, %.2f]; smallest eq margin %.2f pts; NI margin %.2f" % (
            sch, *s["ci90_pp_union"], *s["ci90_points_of_90"], s["smallest_equivalence_margin_points"],
            s["smallest_noninferiority_margin_points"]))
        for m, r in s["margins"].items():
            print("    m=%s (%.2fpp) equiv=%s tost_p=%.4f  NI=%s ni_p=%.4f" % (m, r["margin_pp"], r["equivalent_at_5pct"], r["tost_p"],
                                                                         r["noninferior_at_5pct_one_sided"], r["noninf_p"]))
    print("  cluster-t:", json.dumps(v["cluster_t_check"]))
for k, v in mde.items():
    print(k, v["better_worse_resolved"], v["n_resolved_pairs"], v["discordant_range"], "deff %.2f" % v["design_effect_component_vs_question"])
    for nm, r in v["by_psi"].items():
        print("   ", nm, {kk: (round(x, 3) if isinstance(x, float) else x) for kk, x in r.items()})
for t in tests:
    print("%-24s %-28s net %s k=%d p %.4f-%.4f | col Holm %.3f-%.3f BH %.3f-%.3f | 48 Holm %.3f-%.3f BH %.3f-%.3f" % (
        t["reader"], t["comparison"], t["net"], t["unresolved_cells"], t["p_min"], t["p_max"],
        t["per_column_6_readers_holm_p_min"], t["per_column_6_readers_holm_p_max"], t["per_column_6_readers_bh_p_min"],
        t["per_column_6_readers_bh_p_max"], t["whole_table_48_holm_p_min"], t["whole_table_48_holm_p_max"],
        t["whole_table_48_bh_p_min"], t["whole_table_48_bh_p_max"]))
