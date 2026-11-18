"""Shared helpers for S5 (copies of the revision_stats.py logic, same RNG/seed/percentile convention)."""
import itertools, json, math, random
from collections import defaultdict
from pathlib import Path

OUT = Path(__file__).resolve().parent
N_BOOT, SEED = 10000, 0


class Data:
    def __init__(self, d):
        self.__dict__.update(d)

    def m(self, rd, a, q):
        lo, hi, _ = self.cells["%s|%s|%s" % (rd, a, q)]
        return lo, hi

    def h(self, a, q):
        lo, hi, _ = self.human_sym[a][q]
        return lo, hi


def load():
    return Data(json.loads((OUT / "cells.json").read_text(encoding="utf-8")))


def sign_p(b, w):
    n = b + w
    if n == 0:
        return 1.0
    k = min(b, w)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def diff_vectors(fa, fb, qs):
    lo, up = [], []
    for q in qs:
        al, ah = fa(q); bl, bh = fb(q)
        lo.append(al - bh); up.append(ah - bl)
    return lo, up


def bw_resolved(fa, fb, qs):
    b = w = 0
    for q in qs:
        al, ah = fa(q); bl, bh = fb(q)
        if al == ah and bl == bh:
            b += al > bl; w += al < bl
    return b, w


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
            av = val.get((q, "a"), fa(q)[0]); bv = val.get((q, "b"), fb(q)[0])
            net += av - bv; b += av > bv; w += av < bv
        ps.append(sign_p(b, w)); nets.append(net)
    return dict(p_lo=min(ps), p_hi=max(ps), net_lo=min(nets), net_hi=max(nets), k=len(cells))


def boot_sims(diffs, clusters=None, n_boot=N_BOOT, seed=SEED):
    """sorted bootstrap means in pp (question-weighted), same scheme as revision_stats.boot_mean."""
    rnd = random.Random(seed)
    n = len(diffs)
    sims = []
    if clusters is None:
        for _ in range(n_boot):
            s = [diffs[rnd.randrange(n)] for _ in range(n)]
            sims.append(100.0 * sum(s) / n)
    else:
        g = defaultdict(list)
        for d, c in zip(diffs, clusters):
            g[c].append(d)
        keys = sorted(g)
        for _ in range(n_boot):
            tot = cnt = 0
            for _k in keys:
                grp = g[keys[rnd.randrange(len(keys))]]
                tot += sum(grp); cnt += len(grp)
            sims.append(100.0 * tot / cnt)
    sims.sort()
    return sims


def boot_ci(diffs, clusters=None, level=0.95):
    s = boot_sims(diffs, clusters)
    a = (1 - level) / 2
    return s[int(a * N_BOOT)], s[int((1 - a) * N_BOOT) - 1]


def inter_bounds(D, rd, ctrl):
    lo = hi = 0
    for q in D.ans:
        l6, c6 = D.m(rd, "LEX5B_6000", q), D.m(rd, ctrl + "_6000", q)
        l15, c15 = D.m(rd, "LEX5B_15500", q), D.m(rd, ctrl + "_15500", q)
        lo += l6[0] - c6[1] - l15[1] + c15[0]
        hi += l6[1] - c6[0] - l15[0] + c15[1]
    return lo, hi
