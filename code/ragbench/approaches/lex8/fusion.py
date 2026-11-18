"""LEX8 score fusion (Astra R1, 2026-09-26): explicit, bounded, score-aware; no RRF, no MMR.

For a candidate union U and channels j (BM25, bge-m3 dense, bge-m3 sparse), s_j(x) is the REAL channel score of every
x in U (computed from the full score array; never zero merely because x missed channel j's top-k list).

    z_j(x) = (s_j(x) - min_U s_j) / (max_U s_j - min_U s_j)      (0 when max == min); non-members of U clipped to [0, 1]
    CombSUM-norm(x) = sum_j z_j(x)
    MaxMean R(x)    = 0.5 max_j z_j(x) + 0.5 mean_j z_j(x)
    unit utility u(a) = 0.8 R_unit(a) + 0.2 max_{p in parents(a)} R_chunk(p)
    with synthetic questions u_exp(a) = 0.8 u(a) + 0.2 e(a)
"""
import numpy as np

FUSION_VERSION = "lex8-fusion-v1"
W_UNIT, W_PARENT = 0.8, 0.2
W_BASE, W_SYNTH = 0.8, 0.2


def top_ids(scores, valid, k):
    """indices of the top k valid scores, best first; ties by index."""
    o = np.lexsort((np.arange(len(scores)), -np.asarray(scores)))
    return [int(i) for i in o[np.asarray(valid)[o]][:k]]


class Normalizer:
    """Per-channel min-max over a union U; applies to any id (clipped to [0, 1])."""

    def __init__(self, arrays, union):
        self.arrays = arrays                     # {channel: full score array}
        self.union = list(union)
        self.lo, self.hi = {}, {}
        idx = np.asarray(self.union, np.int64)
        for ch, arr in arrays.items():
            v = np.asarray(arr)[idx] if len(idx) else np.zeros(1)
            self.lo[ch], self.hi[ch] = float(v.min()), float(v.max())

    def z(self, ch, x):
        lo, hi = self.lo[ch], self.hi[ch]
        if hi <= lo:
            return 0.0
        return float(min(1.0, max(0.0, (float(self.arrays[ch][x]) - lo) / (hi - lo))))

    def zs(self, x):
        return [self.z(ch, x) for ch in self.arrays]

    def zvec(self, ids):
        """[len(ids), channels] array of z (vectorised)."""
        ids = np.asarray(list(ids), np.int64)
        out = np.zeros((len(ids), len(self.arrays)), np.float64)
        for j, (ch, arr) in enumerate(self.arrays.items()):
            lo, hi = self.lo[ch], self.hi[ch]
            if hi > lo and len(ids):
                out[:, j] = np.clip((np.asarray(arr, np.float64)[ids] - lo) / (hi - lo), 0.0, 1.0)
        return out


def combsum(zs):
    return float(sum(zs))


def maxmean(zs):
    return 0.5 * max(zs) + 0.5 * (sum(zs) / len(zs)) if zs else 0.0


def maxmean_vec(Z):
    return 0.5 * Z.max(axis=1) + 0.5 * Z.mean(axis=1) if Z.shape[0] else np.zeros(0)


def fuse(arrays, lists, rule):
    """arrays {channel: full score array}; lists {channel: top-k ids}. -> (order, score dict, normalizer).
    Order: score desc, ties best single-channel rank, then id."""
    union = sorted(set().union(*[set(v) for v in lists.values()]))
    N = Normalizer(arrays, union)
    f = combsum if rule == "combsum" else maxmean
    sc = {x: f(N.zs(x)) for x in union}
    rank = {}
    for ch, v in lists.items():
        for r, x in enumerate(v, 1):
            rank[x] = min(rank.get(x, 10 ** 9), r)
    order = sorted(union, key=lambda x: (-sc[x], rank.get(x, 10 ** 9), x))
    return order, sc, N


def unit_utility(r_unit, parents, r_chunk):
    """u(a) = 0.8 R_unit(a) + 0.2 max over parents R_chunk(p) (0 without parents)."""
    pr = max((r_chunk(p) for p in parents), default=0.0)
    return W_UNIT * r_unit + W_PARENT * pr


def expanded_utility(u, e):
    return W_BASE * u + W_SYNTH * e
