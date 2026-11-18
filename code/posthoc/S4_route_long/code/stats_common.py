"""Frozen-test loading and paired statistics for S4. The statistics functions are copied from
paper/analysis/revision_stats.py (find / diff_vectors / sign_range / boot_mean / compare; same seeds and resample
counts) because that module runs and writes its own outputs on import. Scores come from report_test.load/score."""
import itertools
import json
import os
import random
import sys
from collections import defaultdict
from pathlib import Path

import common as C

os.environ["ANYRAG_TEST_DIR"] = str(C.TR)
os.environ["ANYRAG_FREEZE_DIR"] = str(C.FZ)
os.environ["ANYRAG_CODE_DIR"] = str(C.CODE)
sys.path.insert(0, str(C.TR))
import report_test as RT  # noqa: E402

N_BOOT, SEED = 10000, 0
P = RT.PRIMARY
QS, ANS, AB, QIDS = RT.QS, RT.ANS, RT.AB, RT.QIDS
_LOADED = {}


def load():
    if not _LOADED:
        _LOADED["REC"], _LOADED["VER"] = RT.load()
    return _LOADED["REC"], _LOADED["VER"]


def mach(a, q):
    REC, VER = load()
    lo, hi, _ = RT.score(REC, VER, P, a, q)
    return lo, hi


# ---------------------------------------------------------------- components (as revision_stats)
QROWS = {}
for line in open(C.FZ / "questions" / "test_v2judge_all.jsonl", encoding="utf-8-sig"):
    if line.strip():
        d = json.loads(line)
        QROWS[d["id"]] = d["meta"]["v2"]
constituents = {q: set(QROWS[q]["source_family"].split("+")) for q in ANS}
parent = {}


def find(x):
    parent.setdefault(x, x)
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x


for q in ANS:
    c = sorted(constituents[q])
    for o in c[1:]:
        parent[find(o)] = find(c[0])
comp = {q: find(sorted(constituents[q])[0]) for q in ANS}


def tier(q):
    return QS[q]["tier"] or QROWS[q].get("intended_tier")


def intended_tier(q):
    return QROWS[q].get("intended_tier") or QS[q]["tier"]


def lang(q):
    return QS[q]["lang"]


def diff_vectors(fa, fb, qs):
    lo, up, unres = [], [], []
    for q in qs:
        al, ah = fa(q)
        bl, bh = fb(q)
        lo.append(al - bh)
        up.append(ah - bl)
        if al != ah or bl != bh:
            unres.append(q)
    return lo, up, unres


def sign_range(fa, fb, qs):
    cells = []
    for q in qs:
        for f, tag in ((fa, "a"), (fb, "b")):
            l, h = f(q)
            if l != h:
                cells.append((q, tag))
    ps, nets = [], []
    for assign in itertools.product((0, 1), repeat=len(cells)):
        val = dict(zip(cells, assign))
        b = w = net = 0
        for q in qs:
            av = val.get((q, "a"), fa(q)[0])
            bv = val.get((q, "b"), fb(q)[0])
            net += av - bv
            b += av > bv
            w += av < bv
        ps.append(RT.sign_p(b, w))
        nets.append(net)
    return min(ps), max(ps), min(nets), max(nets), len(cells)


def boot_mean(diffs, clusters=None):
    rnd = random.Random(SEED)
    n = len(diffs)
    sims = []
    if clusters is None:
        for _ in range(N_BOOT):
            s = [diffs[rnd.randrange(n)] for _ in range(n)]
            sims.append(100.0 * sum(s) / n)
    else:
        g = defaultdict(list)
        for d, c in zip(diffs, clusters):
            g[c].append(d)
        keys = sorted(g)
        for _ in range(N_BOOT):
            tot = cnt = 0
            for _k in keys:
                grp = g[keys[rnd.randrange(len(keys))]]
                tot += sum(grp)
                cnt += len(grp)
            sims.append(100.0 * tot / cnt)
    sims.sort()
    return sims[int(0.025 * N_BOOT)], sims[int(0.975 * N_BOOT) - 1]


def compare(fa, fb, qs=None):
    qs = qs or ANS
    lo, up, unres = diff_vectors(fa, fb, qs)
    p_lo, p_hi, n_lo, n_hi, k = sign_range(fa, fb, qs)
    out = {"net_bounds": [sum(lo), sum(up)], "pp_bounds": [round(100 * sum(lo) / len(qs), 2),
                                                          round(100 * sum(up) / len(qs), 2)],
           "unresolved_cells": k, "sign_p_range": [round(p_lo, 4), round(p_hi, 4)]}
    for name, cl in (("question", None), ("component", [comp[q] for q in qs])):
        a1, a2 = boot_mean(lo, cl), boot_mean(up, cl)
        out["ci_pp_" + name] = [round(min(a1[0], a2[0]), 2), round(max(a1[1], a2[1]), 2)]
    b = w = 0
    for q in qs:
        al, ah = fa(q)
        bl, bh = fb(q)
        if al == ah and bl == bh:
            b += al > bl
            w += al < bl
    out["wins_losses_resolved"] = [b, w]
    return out


def acc_bounds(f, qs):
    return [sum(f(q)[0] for q in qs), sum(f(q)[1] for q in qs)]


def by_group(f, qs=None):
    qs = qs or ANS
    out = {"all": acc_bounds(f, qs) + [len(qs)]}
    for t in ("T1", "T2", "T3"):
        s = [q for q in qs if tier(q) == t]
        out[t] = acc_bounds(f, s) + [len(s)]
    for l in ("fa", "en"):
        s = [q for q in qs if lang(q) == l]
        out[l] = acc_bounds(f, s) + [len(s)]
    return out


def fmt_b(lohi):
    lo, hi = lohi[0], lohi[1]
    return ("%d" % lo) if lo == hi else ("%d..%d" % (lo, hi))
