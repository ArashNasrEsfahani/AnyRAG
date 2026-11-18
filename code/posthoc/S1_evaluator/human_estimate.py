"""S1 task 2(c): merge completed human reviews and estimate per-pipeline HUMAN accuracy on the 90 answerable test
questions (GLM-5.3-flash @6,000), with Cohen's kappa between reviewers. No model calls.

    python human_estimate.py --a A_export.json [--b B_export.json] [--judge 1|2] [--out DIR]

--a  completed packet A review (24 sampled machine-concordant questions x 4 pipelines). Either the stand-alone page's
     Export JSON ({"set", "seconds", "results": [{"item", "verdict", "note"}]}) or the review game's
     game/A_review.answercheck.jsonl (latest line per (qid, answer_id) wins).
--b  completed packet B review (second reviewer, 120 cells), same two formats.
Verdict rule (as in the earlier merge): correct = 1; wrong / partial / unsupported = 0; a note containing
"half correct" (any case) = 0 regardless of the button; unsure / no verdict = not usable (dropped, counted).

Design. The 90 answerable questions split into two strata.
  R  the 30 human-reviewed questions: a census; the first reviewer's symmetric-merge verdicts are used as they are.
  C  the 60 machine-concordant questions: the judge's verdict M is known for all 60; the human verdict H only on the
     simple random sample S of n = 24 (packet A, seed in its key).
Estimators of the number of human-correct answers in C, per pipeline (and of paired differences between pipelines):
  naive       sum_C M                                  (what the paper reports now: machine verdicts on C)
  PPI         sum_C M + (N/n) sum_S (H - M)            prediction-powered / difference estimator (unbiased under SRS);
              Var = N^2 (1 - n/N) s^2(H - M) / n       (finite-population correction; the 90 questions are the target)
  RG          Rogan-Gladen misclassification correction (cf. Lee et al. 2025, arXiv 2511.21140):
              theta = (p_obs + spec - 1) / (sens + spec - 1), clipped to [0, 1], with sens = P(M=1 | H=1) and
              spec = P(M=0 | H=0) estimated on S; p_obs = mean_C M is a census (no sampling variance), so the
              delta-method variance carries only the calibration terms:
              Var = [(1 - theta)^2 spec(1 - spec)/m0 + theta^2 sens(1 - sens)/m1] / (sens + spec - 1)^2.
              Reported per pipeline and with sens/spec pooled over the four pipelines. Undefined when the sample has
              no H=0 or no H=1 cell, or sens + spec <= 1.
Totals over the 90 = census count on R + estimate on C; 95% intervals are normal (z = 1.96), clipped to [0, 90].
Unresolved machine cells (needs_adjudication) in C are evaluated at both extremes; results are reported as ranges.
"""
import argparse
import itertools
import json
import math
from pathlib import Path

import j2lib as L

Z = 1.96
PAIRS = [("LEX5B_6000", "HYBRID_6000"), ("LEX5B_6000", "HR_6000"), ("SETR_6000", "LEX5B_6000"),
         ("SETR_6000", "HYBRID_6000"), ("SETR_6000", "HR_6000"), ("HR_6000", "HYBRID_6000")]
HERE = L.HERE


def value(verdict, note):
    if "half correct" in (note or "").lower():
        return 0
    return {"correct": 1, "wrong": 0, "partial": 0, "unsupported": 0}.get(verdict)


def read_review(path, key):
    """-> ({(arm, qid): 0/1}, n_unusable_cells). Accepts the page export JSON or a review-game answercheck JSONL."""
    path = Path(path)
    items = {k["item"]: k for k in key["items_key"]}
    by_aid = {(k["qid"], k["answer_id"]): k for k in key["items_key"]}
    txt = path.read_text(encoding="utf-8-sig").strip()
    got = {}
    try:
        d = json.loads(txt)
    except ValueError:
        d = None                                         # several JSON lines: a review-game answercheck file
    if isinstance(d, dict) and "results" in d:
        if d.get("set") != key["set"]:
            raise SystemExit("export set %r does not match key set %r" % (d.get("set"), key["set"]))
        for r in d["results"]:
            got[r["item"]] = value(r.get("verdict"), r.get("note"))
    else:
        latest = {}
        for line in txt.splitlines():
            if line.strip():
                r = json.loads(line)
                latest[(r["qid"], r["answer_id"])] = r
        for k, r in latest.items():
            if k in by_aid:
                got[by_aid[k]["item"]] = value(r.get("verdict"), r.get("note"))
    out, unusable = {}, 0
    for iid, k in items.items():
        v = got.get(iid)
        for a in k["arms"]:
            if v is None:
                unusable += 1
            else:
                out[(a, k["qid"])] = v
    return out, unusable


def machine_fn(judge):
    if judge == 1:
        return lambda a, q: L.judge1(a, q)
    return lambda a, q: L.judge2(a, q)[:2]


def extremes(cells, mach):
    """Unresolved machine cells -> list of assignments {(a, q): 0/1} (all combos when <= 10, else the 2 extremes)."""
    unres = [c for c in cells if mach(*c)[0] != mach(*c)[1]]
    if len(unres) <= 10:
        return [dict(zip(unres, bits)) for bits in itertools.product((0, 1), repeat=len(unres))], unres
    return [{c: 0 for c in unres}, {c: 1 for c in unres}], unres


def ppi_total(Mc, Ms, Hs, N):
    """Mc: machine values on all of C; Ms/Hs: machine/human on the sample. -> (estimate, se)."""
    n = len(Hs)
    d = [h - m for h, m in zip(Hs, Ms)]
    mean_d = sum(d) / n
    s2 = sum((x - mean_d) ** 2 for x in d) / (n - 1) if n > 1 else 0.0
    est = sum(Mc) + N * mean_d
    se = math.sqrt(N * N * (1 - n / N) * s2 / n) if n else float("nan")
    return est, se


def rg_total(Mc, Ms, Hs, N):
    m1 = [m for m, h in zip(Ms, Hs) if h == 1]
    m0 = [m for m, h in zip(Ms, Hs) if h == 0]
    if not m1 or not m0:
        return None
    sens, spec = sum(m1) / len(m1), 1 - sum(m0) / len(m0)
    if sens + spec <= 1:
        return None
    p_obs = sum(Mc) / len(Mc)
    th = (p_obs + spec - 1) / (sens + spec - 1)
    thc = min(1.0, max(0.0, th))
    var = ((1 - thc) ** 2 * spec * (1 - spec) / len(m0) + thc ** 2 * sens * (1 - sens) / len(m1)) / (sens + spec - 1) ** 2
    return {"estimate": N * thc, "se": N * math.sqrt(var), "sens": sens, "spec": spec, "m1": len(m1), "m0": len(m0),
            "theta_unclipped": th}


def ci(est, se, lo=0, hi=90):
    return [max(lo, est - Z * se), min(hi, est + Z * se)]


def run(a_path, b_path, judge, out_dir, keys_dir=HERE / "packets"):
    keyA = json.loads((keys_dir / "A_concordant24" / "key_A_concordant24.json").read_text(encoding="utf-8"))
    keyB = json.loads((keys_dir / "B_second_reviewer" / "key_B_second_reviewer.json").read_text(encoding="utf-8"))
    mach = machine_fn(judge)
    H1, R = L.human_sym()
    C = keyA["population_qids"]
    S = keyA["sample_qids"]
    N = len(C)
    res = {"judge": judge, "N_C": N, "n_S": len(S), "R_questions": len(R)}
    HA, unA = read_review(a_path, keyA)
    res["A_unusable_cells"] = unA
    per = {}
    allcells = [(a, q) for a in L.ARMS4 for q in C]
    assigns, unres = extremes(allcells, mach)
    res["C_unresolved_machine_cells"] = [list(c) for c in unres]

    def mv(a, q, asg):
        lo, hi = mach(a, q)
        return asg.get((a, q), lo)

    # pooled calibration (all four pipelines) for the pooled RG variant
    for a in L.ARMS4:
        rows = []
        for asg in assigns:
            Mc = [mv(a, q, asg) for q in C]
            samp = [q for q in S if (a, q) in HA]
            Ms = [mv(a, q, asg) for q in samp]
            Hs = [HA[(a, q)] for q in samp]
            census = sum(H1[(a, q)] for q in R)
            e, se = ppi_total(Mc, Ms, Hs, N)
            pooled_M = [mv(b, q, asg) for b in L.ARMS4 for q in S if (b, q) in HA]
            pooled_H = [HA[(b, q)] for b in L.ARMS4 for q in S if (b, q) in HA]
            rg = rg_total(Mc, Ms, Hs, N)
            rgp = rg_total(Mc, pooled_M, pooled_H, N)
            stricter = sum(1 for m, h in zip(Ms, Hs) if m == 0 and h == 1)
            laxer = sum(1 for m, h in zip(Ms, Hs) if m == 1 and h == 0)
            rows.append({
                "census_R": census, "naive_total": census + sum(Mc),
                "ppi_total": census + e, "ppi_ci95": ci(census + e, se), "ppi_se": se,
                "rg_total": None if rg is None else census + rg["estimate"],
                "rg_ci95": None if rg is None else ci(census + rg["estimate"], rg["se"]),
                "rg_sens_spec": None if rg is None else [rg["sens"], rg["spec"], rg["m1"], rg["m0"]],
                "rg_pooled_total": None if rgp is None else census + rgp["estimate"],
                "rg_pooled_ci95": None if rgp is None else ci(census + rgp["estimate"], rgp["se"]),
                "sample_cells": len(Hs), "sample_judge_stricter": stricter, "sample_judge_laxer": laxer,
                "sample_agree": len(Hs) - stricter - laxer})
        per[a] = rows if len(rows) > 1 else rows[0]
    res["per_pipeline"] = per
    # paired differences (PPI on the difference)
    pairs = {}
    for a, b in PAIRS:
        rows = []
        for asg in assigns:
            Dc = [mv(a, q, asg) - mv(b, q, asg) for q in C]
            samp = [q for q in S if (a, q) in HA and (b, q) in HA]
            Ds = [mv(a, q, asg) - mv(b, q, asg) for q in samp]
            Hd = [HA[(a, q)] - HA[(b, q)] for q in samp]
            census = sum(H1[(a, q)] - H1[(b, q)] for q in R)
            e, se = ppi_total(Dc, Ds, Hd, N)
            rows.append({"census_R": census, "naive_net": census + sum(Dc), "ppi_net": census + e,
                         "ppi_ci95": [census + e - Z * se, census + e + Z * se]})
        pairs["%s vs %s" % (a, b)] = rows if len(rows) > 1 else rows[0]
    res["pairs"] = pairs
    if b_path:
        HB, unB = read_review(b_path, keyB)
        res["B_unusable_cells"] = unB
        agr = {}
        for grp, arms in [("all", L.ARMS4)] + [(a, [a]) for a in L.ARMS4]:
            cells = [(a, q) for a in arms for q in R if (a, q) in HB]
            p12 = [(H1[c], HB[c]) for c in cells]
            out = {"cells": len(cells), "agree": sum(x == y for x, y in p12), "r1_only_correct": sum(x > y for x, y in p12),
                   "r2_only_correct": sum(x < y for x, y in p12), "kappa_r1_r2": L.kappa(p12),
                   "r2_correct": sum(y for _, y in p12), "r1_correct": sum(x for x, _ in p12)}
            for j, f in (("j1", machine_fn(1)), ("j2", machine_fn(2))):
                pj = [(f(*c)[0], HB[c]) for c in cells if f(*c)[0] == f(*c)[1]]
                out["kappa_%s_r2" % j] = L.kappa(pj)
                out["%s_stricter_than_r2" % j] = sum(1 for m, h in pj if m == 0 and h == 1)
                out["%s_laxer_than_r2" % j] = sum(1 for m, h in pj if m == 1 and h == 0)
            agr[grp] = out
        res["reviewers"] = agr
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / ("human_estimate_judge%d.json" % judge)).write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8")
    (out_dir / ("human_estimate_judge%d.md" % judge)).write_text(to_md(res), encoding="utf-8")
    return res


def _f(x, d=1):
    return "-" if x is None else ("%.*f" % (d, x))


def to_md(res):
    L_ = ["# Human-accuracy estimate (judge %d as the predictor), GLM @6,000, 90 answerable" % res["judge"], "",
          "Census on the %d reviewed questions + estimate on the %d machine-concordant questions from a sample of %d. "
          "Unusable (unsure/blank) cells in packet A: %d. Unresolved machine cells on C: %d."
          % (res["R_questions"], res["N_C"], res["n_S"], res["A_unusable_cells"], len(res["C_unresolved_machine_cells"])), "",
          "| pipeline | census R /30 | naive (machine on C) /90 | PPI /90 [95% CI] | RG per-pipeline /90 [95% CI] (sens, spec) | RG pooled /90 [95% CI] | sample: judge stricter / laxer / agree |",
          "|---|---:|---:|---|---|---|---|"]
    for a, r in res["per_pipeline"].items():
        rows = r if isinstance(r, list) else [r]
        def rng(k, d=1):
            vals = [x[k] for x in rows if x[k] is not None]
            if not vals:
                return "-"
            lo, hi = min(vals), max(vals)
            return _f(lo, d) if abs(hi - lo) < 1e-9 else "%s..%s" % (_f(lo, d), _f(hi, d))
        def cis(k):
            vals = [x[k] for x in rows if x[k] is not None]
            if not vals:
                return ""
            return "[%s, %s]" % (_f(min(v[0] for v in vals)), _f(max(v[1] for v in vals)))
        ss = rows[0]["rg_sens_spec"]
        L_.append("| %s | %d | %s | %s %s | %s %s %s | %s %s | %d / %d / %d |" % (
            a, rows[0]["census_R"], rng("naive_total", 0), rng("ppi_total"), cis("ppi_ci95"), rng("rg_total"),
            cis("rg_ci95"), "" if ss is None else "(%.2f, %.2f)" % (ss[0], ss[1]), rng("rg_pooled_total"),
            cis("rg_pooled_ci95"), rows[0]["sample_judge_stricter"], rows[0]["sample_judge_laxer"], rows[0]["sample_agree"]))
    L_ += ["", "| paired net /90 | census R | naive | PPI [95% CI] |", "|---|---:|---:|---|"]
    for k, r in res["pairs"].items():
        rows = r if isinstance(r, list) else [r]
        L_.append("| %s | %+d | %s | %s [%s, %s] |" % (
            k, rows[0]["census_R"], "..".join(sorted({"%+d" % x["naive_net"] for x in rows})),
            "..".join(sorted({"%+.1f" % x["ppi_net"] for x in rows})), _f(min(x["ppi_ci95"][0] for x in rows)),
            _f(max(x["ppi_ci95"][1] for x in rows))))
    if "reviewers" in res:
        L_ += ["", "Second reviewer (packet B) vs first reviewer (symmetric merge) and the judges; unusable cells: %d."
               % res["B_unusable_cells"], "",
               "| group | cells | agree | kappa R1-R2 | R1 correct | R2 correct | kappa J1-R2 | J1 stricter/laxer vs R2 | kappa J2-R2 | J2 stricter/laxer vs R2 |",
               "|---|---:|---:|---:|---:|---:|---:|---|---:|---|"]
        for g, o in res["reviewers"].items():
            L_.append("| %s | %d | %d | %s | %d | %d | %s | %d / %d | %s | %d / %d |" % (
                g, o["cells"], o["agree"], _f(o["kappa_r1_r2"], 3), o["r1_correct"], o["r2_correct"],
                _f(o["kappa_j1_r2"], 3), o["j1_stricter_than_r2"], o["j1_laxer_than_r2"], _f(o["kappa_j2_r2"], 3),
                o["j2_stricter_than_r2"], o["j2_laxer_than_r2"]))
    return "\n".join(L_) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", default=None)
    ap.add_argument("--judge", type=int, choices=(1, 2), default=1)
    ap.add_argument("--out", default=str(HERE / "human_results"))
    a = ap.parse_args()
    run(a.a, a.b, a.judge, a.out)
    print((Path(a.out) / ("human_estimate_judge%d.md" % a.judge)).read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
