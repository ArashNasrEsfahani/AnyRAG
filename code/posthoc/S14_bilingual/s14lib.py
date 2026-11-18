"""S14 "bilingual, saved data": shared loaders and statistics. No API calls, no GPU (CUDA_VISIBLE_DEVICES="").

Verdict sources (all read-only, all reused from the owning streams' own loaders):
  judge 1, GLM r0, every arm except HYBRID_AGENT   S8_factorial/analyze.py J1[arm]  (frozen arms via report_test.score,
                                                     S2a arms via S2a's rule, S3/S6 arms via their judge dirs, S8 lexbud)
  judge 1, HYBRID_AGENT                             S9_gpu runs + judge dir (same rule as S9_gpu/analyze.py score())
  judge 2 (mistral-large-2512)                      S8_factorial/analyze.py J2[arm] (frozen 8 arms = S1 j2lib; post-hoc
                                                     arms = S8 judge2/<ARM>, partial)
  other five readers (judge 1)                      report_test.score(REC, VER, reader, arm, q) (= S5 cells.json)
  GLM reader replicates r1, r2 (judge 1)            S2b_variance/exp1_reader/<arm> (same rule as S2b lib.verdicts)
Statistics: revision_stats.compare (via S8's ast-extracted copy, with S8's exact DP sign range when >12 unresolved
cells), 10,000 resamples, seed 0, over questions and the 20 source-family components.
"""
import importlib.util
import itertools
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = ""
ROOT = Path(r"D:\KareOne\RAG Paper")
PH = ROOT / "work" / "posthoc_2026-11"
HERE = PH / "S14_bilingual"
TR = ROOT / "work" / "datasets_v2" / "test_run"
FZ = ROOT / "work" / "datasets_v2" / "test_freeze"
FROZEN_CODE = r"D:\KareOne\RAG-Paper-upgrades"
for p in (FROZEN_CODE, str(TR), str(PH / "S8_factorial"), str(PH / "S1_evaluator")):
    if p not in sys.path:
        sys.path.insert(0, p)


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


S8A = _load("s8_analyze", PH / "S8_factorial" / "analyze.py")   # main() is guarded; import only builds loaders
RT = S8A.RT
JM = S8A.JM
REC, VER = S8A.REC, S8A.VER
P = RT.PRIMARY
READERS = list(RT.READERS)
ANS = list(RT.ANS)
QS = RT.QS
LANG = {q: QS[q]["lang"] for q in RT.QIDS}
TIER = {q: QS[q]["tier"] for q in RT.QIDS}
FA = [q for q in ANS if LANG[q] == "fa"]
EN = [q for q in ANS if LANG[q] == "en"]
COMP = S8A.NS["comp"]
compare = S8A.compare                       # revision_stats.compare (unchanged source) + exact DP sign range
CONTAM = json.loads((PH / "S7_analysis" / "contam.json").read_text(encoding="utf-8"))["closedbook_correct"]["questions"]
assert len(CONTAM) == 26
FROZEN_ARMS = list(RT.ARMS)


def jl(p):
    p = Path(p)
    return [json.loads(l) for l in open(p, encoding="utf-8-sig") if l.strip()] if p.is_file() else []


# ------------------------------------------------------------------ HYBRID_AGENT (S9), same rule as S9 analyze.score
_HA = {}


def _ha():
    if not _HA:
        rec = {}
        for r in jl(PH / "S9_gpu" / "runs" / "HYBRID_AGENT" / "test" / "runs_HYBRID_AGENT.jsonl"):
            old = rec.get(r["question_id"])
            if old is None or old.get("error") or not r.get("error"):
                rec[r["question_id"]] = r
        _HA["rec"] = rec
        _HA["ver"] = RT.final_verdicts_by_answer(PH / "S9_gpu" / "judge" / "HYBRID_AGENT" / "test")
    return _HA["rec"], _HA["ver"]


def hybrid_agent(q):
    rec, ver = _ha()
    r = rec.get(q)
    if r is None or r.get("error"):
        return 0, 0
    v = ver.get((r["meta"]["identity"]["rid"], JM.answer_sha1(r)))
    if v is None or v["status"] == "needs_adjudication":
        return 0, 1
    return v["strict"], v["strict"]


def j1(arm):
    """judge 1, GLM, original run (r0)."""
    if arm == "HYBRID_AGENT":
        return hybrid_agent
    return S8A.J1[arm]


def j2(arm):
    """judge 2 or None when the arm was never sent to judge 2."""
    return S8A.J2.get(arm)


def reader(rd, arm):
    """judge 1 on another reader's answer to the byte-identical frozen pack (frozen arms only)."""
    if arm not in FROZEN_ARMS:
        return None
    return lambda q: RT.score(REC, VER, rd, arm, q)[:2]


# ------------------------------------------------------------------ S2b reader replicates (same rule as S2b lib)
_SLUG = JM._slug(json.loads((FZ / "MANIFEST.json").read_text(encoding="utf-8"))["judge"]["model"])
_E1 = {}


def _rep_verdicts(arm):
    if arm in _E1:
        return _E1[arm]
    d = PH / "S2b_variance" / "exp1_reader" / arm
    recs = {}
    for r in jl(d / "runs.jsonl"):
        rid = r["meta"]["identity"]["rid"]
        if rid not in recs or recs[rid].get("error") or not r.get("error"):
            recs[rid] = r
    V = {}
    for v in jl(d / "judge" / ("verdicts_%s.jsonl" % _SLUG)):
        V.setdefault((v["rid"], v["answer_sha1"]), ("judged", v["strict"]))
    for n in jl(d / "judge" / ("needs_adjudication_%s.jsonl" % _SLUG)):
        V.setdefault((n["rid"], n["answer_sha1"]), ("needs_adjudication", None))
    out = {}
    for rid, r in recs.items():
        key = (r["question_id"], r["meta"]["identity"]["replicate"], r["meta"]["identity"]["method"])
        if r.get("error"):
            out[key] = (0, 0)
            continue
        v = V.get((rid, JM.answer_sha1(r)))
        out[key] = (0, 1) if v is None or v[0] == "needs_adjudication" else (v[1], v[1])
    _E1[arm] = out
    return out


def _method(arm, q):
    best = None
    for f in sorted((TR / "runs" / RT.pool_of(q) / arm).glob("runs_*.jsonl")):
        for r in jl(f):
            if r["question_id"] == q and (best is None or best.get("error") or not r.get("error")):
                best = r
    return best["meta"]["identity"]["method"]


_METH = {}


def rep(arm, k):
    """GLM reader replicate k (0 = saved test answer) of a frozen arm. For FUSION_<B> cells whose S2a answer is the frozen
    LEX5B_<B> answer (identical pack) replicate k takes LEX5B_<B>'s replicate k; other non-frozen cells stay at r0."""
    if k == 0:
        return j1(arm)
    if arm in FROZEN_ARMS:
        v = _rep_verdicts(arm)

        def f(q):
            if (arm, q) not in _METH:
                _METH[(arm, q)] = _method(arm, q)
            return v.get((q, k, _METH[(arm, q)]), (0, 1))
        return f
    if arm.startswith("FUSION_"):
        if not any((S8A.S2A_ROWS.get((arm, q)) or {}).get("source", "").split(":", 1)[-1] in FROZEN_ARMS
                   and S8A.S2A_ROWS[(arm, q)]["source"].startswith("frozen:") for q in ANS):
            return None          # no cell of this FUSION budget is a frozen answer: nothing is replicated
        base = j1(arm)

        def g(q):
            r = S8A.S2A_ROWS.get((arm, q))
            if r is not None and r["source"].startswith("frozen:"):
                src = r["source"].split(":", 1)[1]
                if src in FROZEN_ARMS:
                    return rep(src, k)(q)
            return base(q)
        return g
    return None


# ------------------------------------------------------------------ statistics
def fvec(f, qs):
    return np.array([f(q)[0] for q in qs]), np.array([f(q)[1] for q in qs])


def unresolved_cells(fa, fb, qs):
    return [(q, t) for q in qs for f, t in ((fa, "a"), (fb, "b")) if f(q)[0] != f(q)[1]]


def assignments(fa, fb, qs, max_enum=8):
    """yield per-question difference vectors (len(qs)) for each assignment of unresolved cells; if more than max_enum
    unresolved cells, only the 4 language-corner assignments (each half at its lo or hi bound) are yielded."""
    cells = unresolved_cells(fa, fb, qs)
    if len(cells) <= max_enum:
        for asg in itertools.product((0, 1), repeat=len(cells)):
            val = dict(zip(cells, asg))
            yield np.array([val.get((q, "a"), fa(q)[0]) - val.get((q, "b"), fb(q)[0]) for q in qs]), "enum"
    else:
        lo = {q: fa(q)[0] - fb(q)[1] for q in qs}
        hi = {q: fa(q)[1] - fb(q)[0] for q in qs}
        for x, y in (("lo", "lo"), ("lo", "hi"), ("hi", "lo"), ("hi", "hi")):
            yield np.array([(lo if (LANG[q] == "fa" and x == "lo") or (LANG[q] == "en" and y == "lo") else hi)[q]
                            for q in qs]), "corners"


_PERM_Q = {}


def _perm_matrix(qs, n_perm=20000, seed=0):
    key = (tuple(qs), n_perm, seed)
    if key not in _PERM_Q:
        rng = np.random.default_rng(seed)
        is_en = np.array([LANG[q] == "en" for q in qs])
        M = np.empty((n_perm, len(qs)), dtype=bool)
        for i in range(n_perm):
            M[i] = rng.permutation(is_en)
        _PERM_Q[key] = M
    return _PERM_Q[key]


_PERM_C = {}


def _comp_partitions(qs):
    """all ways to label the components present in qs as en, with the observed number of en components
    (components nest within language); exact."""
    key = tuple(qs)
    if key not in _PERM_C:
        comps = sorted({COMP[q] for q in qs})
        n_en = len({COMP[q] for q in qs if LANG[q] == "en"})
        idx = {c: i for i, c in enumerate(comps)}
        C = np.zeros((len(qs), len(comps)))
        for j, q in enumerate(qs):
            C[j, idx[COMP[q]]] = 1
        combos = np.array(list(itertools.combinations(range(len(comps)), n_en)))
        lab = np.zeros((len(combos), len(comps)), dtype=bool)
        lab[np.arange(len(combos))[:, None], combos] = True
        _PERM_C[key] = (C, lab, comps)
    return _PERM_C[key]


def interaction_stat(d, is_en):
    """mean difference in the English half minus mean difference in the Persian half (per question)."""
    return d[is_en].mean() - d[~is_en].mean()


def perm_interaction(fa, fb, qs, max_enum=8):
    """language x arm permutation test on the paired contrast d_q = a_q - b_q.
    question level: 20,000 random relabellings of language over qs (fixed n_fa/n_en), seed 0;
    component level: exact, every relabelling of the source-family components with the observed number of English
    components (language is a property of the source family). Two-sided, |T*| >= |T_obs|.
    Returns ranges over assignments of unresolved cells."""
    is_en = np.array([LANG[q] == "en" for q in qs])
    M = _perm_matrix(qs)
    C, lab, comps = _comp_partitions(qs)
    nq = C.sum(0)
    out = {"T_obs": [], "p_question": [], "p_component": [], "mode": None}
    for d, mode in assignments(fa, fb, qs, max_enum):
        out["mode"] = mode
        t = interaction_stat(d, is_en)
        n_en = is_en.sum()
        s_en = (M * d).sum(1) / n_en
        s_fa = ((~M) * d).sum(1) / (len(qs) - n_en)
        T = s_en - s_fa
        pq = (np.sum(np.abs(T) >= abs(t) - 1e-12) + 1) / (len(T) + 1)
        cs = d @ C                         # per-component sum
        se = lab @ cs; ne = lab @ nq
        sf = cs.sum() - se; nf = nq.sum() - ne
        Tc = se / ne - sf / nf
        pc = np.mean(np.abs(Tc) >= abs(t) - 1e-12)
        out["T_obs"].append(float(t)); out["p_question"].append(float(pq)); out["p_component"].append(float(pc))
    return {"T_pp": [100 * min(out["T_obs"]), 100 * max(out["T_obs"])],
            "p_question": [min(out["p_question"]), max(out["p_question"])],
            "p_component": [min(out["p_component"]), max(out["p_component"])],
            "n_component_partitions": int(len(lab)), "mode": out["mode"],
            "unresolved_cells": len(unresolved_cells(fa, fb, qs))}


def boot_interaction(fa, fb, qs, n_boot=10000, seed=0):
    """95% CI (pp) of T = mean_en(d) - mean_fa(d), stratified bootstrap (questions within language; components within
    language), union over the two corner assignments that minimise and maximise T."""
    rng = np.random.default_rng(seed)
    is_en = np.array([LANG[q] == "en" for q in qs])
    lo = np.array([fa(q)[0] - fb(q)[1] for q in qs]); hi = np.array([fa(q)[1] - fb(q)[0] for q in qs])
    dmin = np.where(is_en, lo, hi); dmax = np.where(is_en, hi, lo)
    res = {}
    # question level
    ie, iff = np.where(is_en)[0], np.where(~is_en)[0]
    be = rng.integers(0, len(ie), (n_boot, len(ie))); bf = rng.integers(0, len(iff), (n_boot, len(iff)))
    qs_ci = []
    for d in (dmin, dmax):
        T = d[ie][be].mean(1) - d[iff][bf].mean(1)
        qs_ci.append(np.percentile(T, [2.5, 97.5]))
    res["ci_pp_question"] = [100 * min(c[0] for c in qs_ci), 100 * max(c[1] for c in qs_ci)]
    # component level (question-weighted, as revision_stats.boot_mean)
    comps_en = sorted({COMP[q] for q in qs if LANG[q] == "en"}); comps_fa = sorted({COMP[q] for q in qs if LANG[q] == "fa"})
    cc = []
    for d in (dmin, dmax):
        se = np.array([sum(d[j] for j, q in enumerate(qs) if COMP[q] == c) for c in comps_en])
        ne = np.array([sum(1 for q in qs if COMP[q] == c) for c in comps_en])
        sf = np.array([sum(d[j] for j, q in enumerate(qs) if COMP[q] == c) for c in comps_fa])
        nf = np.array([sum(1 for q in qs if COMP[q] == c) for c in comps_fa])
        ke = rng.integers(0, len(comps_en), (n_boot, len(comps_en))); kf = rng.integers(0, len(comps_fa), (n_boot, len(comps_fa)))
        T = se[ke].sum(1) / ne[ke].sum(1) - sf[kf].sum(1) / nf[kf].sum(1)
        cc.append(np.percentile(T, [2.5, 97.5]))
    res["ci_pp_component"] = [100 * min(c[0] for c in cc), 100 * max(c[1] for c in cc)]
    return res


# ------------------------------------------------------------------ MDE (copied from S5_stats/analyses.py)
from scipy.stats import binom, norm  # noqa: E402


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


def holm(ps):
    idx = sorted(range(len(ps)), key=lambda i: ps[i])
    out = [0.0] * len(ps)
    run = 0.0
    for r, i in enumerate(idx):
        run = max(run, min(1.0, (len(ps) - r) * ps[i]))
        out[i] = run
    return out


def wjson(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=1), encoding="utf-8")


def fnet(lo, hi):
    s = lambda x: "%+d" % x
    return s(lo) if lo == hi else "%s..%s" % (s(lo), s(hi))


def fp(a, b):
    return "%.3f" % a if abs(a - b) < 5e-4 else "%.3f-%.3f" % (a, b)
