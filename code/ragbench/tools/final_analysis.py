"""The pre-registered analysis of the fresh-holdout final test (work/final/PREREG.md). Frozen with it.

    .venv/Scripts/python.exe -m ragbench.tools.final_analysis --verdicts work/final/verdicts_primary.jsonl
        [--questions ragbench/data/questions/test_fresh.jsonl] [--out work/final/RESULTS_primary.md] [--seed 0]

Input verdicts: one JSON line per graded answer {"method", "question_id", "setting", "correct": 0|1} (the primary
judge's; ragbench/tools/judge_final.py writes it). Questions give the corpus, the language and the company
(meta.family).

Estimand (per contrast M1 - M2): over the PRIMARY cells (corpus C fa-fa, corpus D en-en), the paired difference per
question; a company's questions averaged first; companies averaged within a language; the two languages averaged
(equal weights). Inference: 95% CI from a company-cluster bootstrap (companies resampled within each language,
10,000 draws); p-value from a company-level sign-flip permutation test (exact over all flips when the primary cells
hold <= 20 companies in total, else 100,000 random flips). Primary family {H1 LEX2 - LEX2+llm_regex, H2 LEX2 - B2}
Holm-adjusted at 0.05. Everything else is reported as secondary: H3, B3, per-cell exact McNemar, absolute scores.
"""
import argparse
import itertools
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
PRIMARY = [("C", "fa-fa"), ("D", "en-en")]
H_PRIMARY = [("H1", "LEX2", "LEX2+llm_regex"), ("H2", "LEX2", "B2")]
H_SECONDARY = [("H3", "LEX2", "LEX2+free_keywords"), ("S1", "LEX2", "B3"), ("S2", "B2", "B3")]
METHODS = ["LEX2", "LEX2+llm_regex", "LEX2+free_keywords", "B2", "B3"]


def load_questions(path):
    qs = {}
    for line in open(path, encoding="utf-8"):
        q = json.loads(line)
        meta = q.get("meta") or {}
        qs[q["id"]] = {"corpus": q["corpus"], "family": meta.get("family") or q["id"].split("-")[2],
                       "cell": meta.get("cell"), "type": q.get("type")}
    return qs


def load_verdicts(path):
    v = {}
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        v[(r["method"], r["question_id"], r["setting"])] = 1.0 if r.get("correct") else 0.0
    return v


def paired(v, qs, m1, m2, cells):
    """-> {lang: {company: [d, ...]}} of paired differences over the given cells."""
    out = defaultdict(lambda: defaultdict(list))
    for (m, qid, setting), c1 in v.items():
        if m != m1 or qid not in qs:
            continue
        q = qs[qid]
        if (q["corpus"], setting) not in cells:
            continue
        c2 = v.get((m2, qid, setting))
        if c2 is None:
            continue
        out[setting[:2]][q["family"]].append(c1 - c2)
    return out


def estimate(groups):
    langs = [np.mean([np.mean(ds) for ds in comp.values()]) for comp in groups.values() if comp]
    return float(np.mean(langs)) if langs else float("nan")


def bootstrap(groups, n=10000, seed=0):
    rng = np.random.RandomState(seed)
    langs = {l: [np.mean(ds) for ds in comp.values()] for l, comp in groups.items() if comp}
    draws = []
    for _ in range(n):
        per = [np.mean(rng.choice(means, size=len(means), replace=True)) for means in langs.values()]
        draws.append(np.mean(per))
    return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def permutation(groups, n_random=100000, seed=0):
    """Company-level sign flips. Exact over all flips when there are <= 20 companies in total."""
    cells = [(l, np.mean(ds)) for l, comp in groups.items() for ds in comp.values()]
    langs = sorted({l for l, _m in cells})
    obs = estimate(groups)

    def stat(signs):
        per = []
        for l in langs:
            vals = [s * m for (ll, m), s in zip(cells, signs) if ll == l]
            per.append(np.mean(vals))
        return float(np.mean(per))

    k = len(cells)
    if k <= 20:
        flips = itertools.product((1.0, -1.0), repeat=k)
        total = 2 ** k
        hits = sum(1 for s in flips if abs(stat(s)) >= abs(obs) - 1e-12)
        return hits / total, "exact over %d flips" % total
    rng = np.random.RandomState(seed)
    hits = 0
    for _ in range(n_random):
        s = rng.choice((1.0, -1.0), size=k)
        hits += abs(stat(s)) >= abs(obs) - 1e-12
    return (hits + 1) / (n_random + 1), "%d random flips" % n_random


def mcnemar_exact(v, qs, m1, m2, cell):
    b = c = 0
    for (m, qid, setting), x in v.items():
        if m != m1 or qid not in qs or (qs[qid]["corpus"], setting) != cell:
            continue
        y = v.get((m2, qid, setting))
        if y is None:
            continue
        b += x > y
        c += y > x
    n = b + c
    if n == 0:
        return b, c, 1.0
    p = sum(math.comb(n, i) for i in range(0, min(b, c) + 1)) / 2 ** n * 2
    return b, c, min(1.0, p)


def fmt_p(p):
    return "<0.0001" if p < 1e-4 else "%.4f" % p


def holm(ps):
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    adj, running = [0.0] * len(ps), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(ps) - rank) * ps[i]))
        adj[i] = running
    return adj


def main(argv=None):
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--verdicts", required=True)
    ap.add_argument("--questions", default=str(ROOT / "ragbench" / "data" / "questions" / "test_fresh.jsonl"))
    ap.add_argument("--out", default=str(ROOT / "work" / "final" / "RESULTS_primary.md"))
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    qs = load_questions(a.questions)
    v = load_verdicts(a.verdicts)
    L = ["# Pre-registered analysis - fresh holdout (primary judge)", ""]

    # absolute scores per method and cell
    L += ["## Correctness by method and cell (questions answered correctly / graded)", "",
          "| method | " + " | ".join("%s %s" % c for c in PRIMARY) + " | primary, weighted |", "|---|---|---|---|"]
    for m in METHODS:
        cells = []
        for cell in PRIMARY:
            xs = [x for (mm, qid, s), x in v.items() if mm == m and qid in qs and (qs[qid]["corpus"], s) == cell]
            cells.append("%d/%d (%.1f%%)" % (sum(xs), len(xs), 100 * np.mean(xs)) if xs else "-")
        g = defaultdict(lambda: defaultdict(list))
        for (mm, qid, s), x in v.items():
            if mm == m and qid in qs and (qs[qid]["corpus"], s) in PRIMARY:
                g[s[:2]][qs[qid]["family"]].append(x)
        L.append("| %s | %s | %s |" % (m, " | ".join(cells), "%.1f%%" % (100 * estimate(g)) if g else "-"))

    # primary family
    L += ["", "## Primary contrasts (Holm over H1, H2)", "",
          "| | contrast | estimate (pp) | 95% CI (company bootstrap) | p (permutation) | Holm p | companies fa/en | questions |",
          "|---|---|---|---|---|---|---|---|"]
    rows, ps = [], []
    for h, m1, m2 in H_PRIMARY:
        g = paired(v, qs, m1, m2, PRIMARY)
        est = estimate(g)
        lo, hi = bootstrap(g, seed=a.seed)
        p, how = permutation(g, seed=a.seed)
        rows.append((h, m1, m2, est, lo, hi, p, how, g))
        ps.append(p)
    for (h, m1, m2, est, lo, hi, p, how, g), padj in zip(rows, holm(ps)):
        nq = sum(len(ds) for comp in g.values() for ds in comp.values())
        L.append("| %s | %s − %s | %+.1f | [%+.1f, %+.1f] | %s (%s) | %s | %d / %d | %d |" % (
            h, m1, m2, 100 * est, 100 * lo, 100 * hi, fmt_p(p), how, fmt_p(padj), len(g.get("fa", {})), len(g.get("en", {})), nq))

    # secondary
    L += ["", "## Secondary contrasts (no multiplicity claim)", "",
          "| | contrast | estimate (pp) | 95% CI | p (permutation) |", "|---|---|---|---|---|"]
    for h, m1, m2 in H_SECONDARY:
        g = paired(v, qs, m1, m2, PRIMARY)
        if not g:
            continue
        est = estimate(g)
        lo, hi = bootstrap(g, seed=a.seed)
        p, how = permutation(g, seed=a.seed)
        L.append("| %s | %s − %s | %+.1f | [%+.1f, %+.1f] | %s (%s) |" % (h, m1, m2, 100 * est, 100 * lo, 100 * hi,
                                                                        fmt_p(p), how))
    L += ["", "## Exact McNemar per cell (secondary)", "", "| contrast | cell | M1 only | M2 only | p |",
          "|---|---|---|---|---|"]
    for h, m1, m2 in H_PRIMARY + H_SECONDARY:
        for cell in PRIMARY:
            b, c, p = mcnemar_exact(v, qs, m1, m2, cell)
            L.append("| %s %s − %s | %s %s | %d | %d | %.4f |" % (h, m1, m2, cell[0], cell[1], b, c, p))
    text = "\n".join(L)
    Path(a.out).write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
