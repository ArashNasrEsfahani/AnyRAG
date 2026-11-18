"""Analysis of the multi-document experiment (2026-09-24, P0 list). final_analysis.py stays frozen with the finished study.

    .venv/Scripts/python.exe -m ragbench.tools.multidoc_analysis --design work/multidoc/design.json --runs <dir> \
        --verdicts <verdicts file> [--failures <failures file>] [--out work/multidoc/RESULTS.md]

Refuses (exit 2) unless check_grid finds the expected grid complete (every record and verdict; see check_grid.py).
Everything prespecified comes from the frozen design JSON: methods, readers, replicates, settings, the PRIMARY
contrasts [[id, method 1, method 2, metric]] (Holm-adjusted together), secondary contrasts (unadjusted), bootstrap
draws and seed.

Estimand of a contrast: per question, the mean over its cells (pools, readers, replicates, settings) of
metric(method 1) - metric(method 2) - the two methods paired cell by cell; questions weigh equally (a ten-point list
does not outweigh a two-part comparison: judge_multi already averages units within a question). Uncertainty: 95%
interval from a source-block cluster bootstrap (source blocks resampled with replacement); p-value from a block-level
sign-flip permutation (exact over all flips for <= 16 blocks, else `permutations` random flips).
Metrics per cell: strict, recall, precision, f1 (judge_multi verdicts) and routing / document coverage computed HERE
from the records and the question's gold evidence sets - never from an approach's own statistics:
  route_all_retrieved   some accepted evidence set lies wholly within the documents of the retrieved items
  route_recall_retrieved  best share of an accepted set's documents among them
  route_all_evidence / route_recall_evidence   the same over the FINAL evidence given to the answer call
(unanswerable questions have no evidence set and no routing metric). Per-family tables report every metric per method.
"""
import argparse
import itertools
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from ..corpus.model import doc_lang, swap_lang
from . import check_grid as CG
from .verify_multidoc import gold_doc_sets

VERDICT_METRICS = ("strict", "recall", "precision", "f1")
ROUTE_METRICS = ("route_all_retrieved", "route_recall_retrieved", "route_all_evidence", "route_recall_evidence")
METRICS = VERDICT_METRICS + ROUTE_METRICS


def _in_lang(doc_id, lang):
    try:
        return swap_lang(doc_id, lang) if doc_lang(doc_id) != lang else doc_id
    except Exception:  # noqa: BLE001
        return doc_id


def routing(qd, rec):
    """Coverage of the question's accepted evidence sets by the retrieved and the final evidence documents."""
    sets = gold_doc_sets(qd)
    if not sets:
        return {k: None for k in ROUTE_METRICS}
    lang = (rec.get("d_lang") or rec.get("setting", "xx-xx")[-2:])
    sets = [{_in_lang(d, lang) for d in s} for s in sets]
    got = {"retrieved": {it.get("doc_id") for it in rec.get("retrieved_units") or []},
           "evidence": {it.get("doc_id") for it in rec.get("evidence_items") or []}}
    out = {}
    for k, docs in got.items():
        out["route_all_%s" % k] = float(any(s <= docs for s in sets))
        out["route_recall_%s" % k] = max(len(s & docs) / len(s) for s in sets)
    return out


def cell_values(rep, questions):
    """{(method, question_id, cell rest): {metric: value}} over the complete grid."""
    vals = {}
    for cell, (rec, v) in rep["cells"].items():
        qid, pid, m, rd, repl, s = cell
        row = {k: (float(v[k]) if v.get(k) is not None else None) for k in VERDICT_METRICS}
        row.update(routing(questions[qid], rec))
        vals[(m, qid, (pid, rd, repl, s))] = row
    return vals


def per_question(vals, m1, m2, metric, questions):
    """{question_id: mean paired difference} (cells where both methods have the metric)."""
    acc = defaultdict(list)
    for (m, qid, rest), row in vals.items():
        if m != m1 or row.get(metric) is None:
            continue
        other = vals.get((m2, qid, rest))
        if other is None or other.get(metric) is None:
            continue
        acc[qid].append(row[metric] - other[metric])
    return {q: float(np.mean(ds)) for q, ds in acc.items()}


def by_block(diffs, questions):
    blocks = defaultdict(list)
    for qid, d in diffs.items():
        blocks[questions[qid].get("source_block") or qid].append(d)
    return blocks


def estimate(blocks):
    xs = [d for ds in blocks.values() for d in ds]
    return float(np.mean(xs)) if xs else float("nan")


def bootstrap(blocks, n=10000, seed=0):
    keys = sorted(blocks)
    if not keys:
        return float("nan"), float("nan")
    rng = np.random.RandomState(seed)
    draws = []
    for _ in range(n):
        pick = rng.randint(0, len(keys), size=len(keys))
        xs = [d for i in pick for d in blocks[keys[i]]]
        draws.append(np.mean(xs))
    return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def permutation(blocks, n_random=20000, seed=0):
    keys = sorted(blocks)
    sums = np.array([sum(blocks[k]) for k in keys])
    n = sum(len(blocks[k]) for k in keys)
    if not keys or n == 0:
        return 1.0, "no data"
    obs = abs(sums.sum() / n)
    if len(keys) <= 16:
        hits = sum(1 for s in itertools.product((1.0, -1.0), repeat=len(keys))
                   if abs(np.dot(s, sums) / n) >= obs - 1e-12)
        return hits / 2 ** len(keys), "exact over %d flips" % 2 ** len(keys)
    rng = np.random.RandomState(seed)
    hits = sum(abs(np.dot(rng.choice((1.0, -1.0), size=len(keys)), sums) / n) >= obs - 1e-12 for _ in range(n_random))
    return (hits + 1) / (n_random + 1), "%d random flips" % n_random


def holm(ps):
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    adj, running = [0.0] * len(ps), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(ps) - rank) * ps[i]))
        adj[i] = running
    return adj


def contrast_rows(design, vals, questions, which):
    bs = design.get("bootstrap") or {}
    rows = []
    for c in design.get(which) or []:
        cid, m1, m2, metric = c[0], c[1], c[2], (c[3] if len(c) > 3 else "strict")
        diffs = per_question(vals, m1, m2, metric, questions)
        blocks = by_block(diffs, questions)
        est = estimate(blocks)
        lo, hi = bootstrap(blocks, n=int(bs.get("n", 10000)), seed=int(bs.get("seed", 0)))
        p, how = permutation(blocks, n_random=int(bs.get("permutations", 20000)), seed=int(bs.get("seed", 0)))
        rows.append({"id": cid, "m1": m1, "m2": m2, "metric": metric, "estimate": est, "ci": [lo, hi], "p": p,
                     "p_how": how, "n_questions": len(diffs), "n_blocks": len(blocks)})
    if which == "primary_contrasts" and rows:
        for r, a in zip(rows, holm([r["p"] for r in rows])):
            r["p_holm"] = a
    return rows


def family_table(vals, questions, methods):
    """{family: {method: {metric: mean of per-question means}}}."""
    acc = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for (m, qid, rest), row in vals.items():
        for k, x in row.items():
            if x is not None:
                acc[qid][m][k].append(x)
    fam = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for qid, per_m in acc.items():
        f = questions[qid].get("task_family") or "?"
        for m, ks in per_m.items():
            for k, xs in ks.items():
                fam[f][m][k].append(float(np.mean(xs)))
                fam["ALL"][m][k].append(float(np.mean(xs)))
    return {f: {m: {k: round(float(np.mean(xs)), 4) for k, xs in ks.items()} for m, ks in per_m.items()}
            for f, per_m in fam.items()}


def analyse(design, questions, records, verdicts, failures=()):
    rep = CG.grid_gaps(design, questions, records, verdicts, failures)
    if not rep["complete"]:
        return None, rep
    vals = cell_values(rep, questions)
    res = {"experiment_id": design["experiment_id"], "design_sha1": design.get("design_sha1"),
           "cells": len(vals), "primary": contrast_rows(design, vals, questions, "primary_contrasts"),
           "secondary": contrast_rows(design, vals, questions, "secondary_contrasts"),
           "families": family_table(vals, questions, design["methods"])}
    return res, rep


def render(res):
    L = ["# Multi-document experiment %s - analysis" % res["experiment_id"], "",
         "Design sha1 %s; %d complete cells." % (str(res["design_sha1"])[:12], res["cells"]), "",
         "## Primary contrasts (Holm)", "",
         "| id | contrast | metric | estimate (pp) | 95% CI (source-block bootstrap) | p | Holm p | questions / blocks |",
         "|---|---|---|---|---|---|---|---|"]
    for r in res["primary"]:
        L.append("| %s | %s - %s | %s | %+.1f | [%+.1f, %+.1f] | %.4f (%s) | %.4f | %d / %d |" % (
            r["id"], r["m1"], r["m2"], r["metric"], 100 * r["estimate"], 100 * r["ci"][0], 100 * r["ci"][1], r["p"],
            r["p_how"], r["p_holm"], r["n_questions"], r["n_blocks"]))
    L += ["", "## Secondary contrasts (no multiplicity claim)", "",
          "| id | contrast | metric | estimate (pp) | 95% CI | p |", "|---|---|---|---|---|---|"]
    for r in res["secondary"]:
        L.append("| %s | %s - %s | %s | %+.1f | [%+.1f, %+.1f] | %.4f |" % (
            r["id"], r["m1"], r["m2"], r["metric"], 100 * r["estimate"], 100 * r["ci"][0], 100 * r["ci"][1], r["p"]))
    L += ["", "## Per family (mean of per-question means)", ""]
    for f, per_m in sorted(res["families"].items()):
        L += ["### %s" % f, "", "| method | " + " | ".join(METRICS) + " |", "|---" * (len(METRICS) + 1) + "|"]
        for m, ks in sorted(per_m.items()):
            L.append("| %s | %s |" % (m, " | ".join("-" if ks.get(k) is None else "%.3f" % ks[k] for k in METRICS)))
        L.append("")
    return "\n".join(L)


def main(argv=None):
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--design", required=True)
    ap.add_argument("--runs", action="append", required=True)
    ap.add_argument("--verdicts", required=True)
    ap.add_argument("--failures", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    design, qs, recs, verdicts, fails = CG.load_inputs(a.design, a.runs, a.verdicts, a.failures)
    res, rep = analyse(design, qs, recs, verdicts, fails)
    if res is None:
        print("\n".join(CG.summary_lines(rep)))
        print("REFUSED: the expected record/verdict grid is incomplete; close every gap first (check_grid.py)")
        return 2
    text = render(res)
    if a.out:
        Path(a.out).write_text(text + "\n", encoding="utf-8")
        Path(a.out).with_suffix(".json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
