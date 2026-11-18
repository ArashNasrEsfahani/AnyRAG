"""LEX with the window score LEARNED instead of hand-set (flag `learned`; off by default, every other arm untouched).

WHAT IS LEARNED. LEX scores a window by `sum(term_weight)` over the terms the window's text shows, plus three hand-set
bonuses (a number when the answer is a number, a year the question asks about, a name the question names). Every
coefficient in that sum is a constant somebody chose: w_picked, w_keyword, w_weak, w_relaxed, w_number, w_year, w_name.
Here the same window is described by a FEATURE VECTOR and scored by a fitted linear model. The candidate windows are
built exactly as before - only their score, and so the order `select` packs them in, changes.

WHY THE LASSO, AND WHY THIS DESIGN IS THE AWKWARD KIND. The feature vector is the generic features above PLUS one
indicator per PHRASE in the vocabulary the model may pick («سود خالص», «درآمد عملیاتی», ...). Few phrases should
matter, so the coefficient vector should be sparse: that is what the L1 penalty is for. The design is not the
textbook one in three ways at once:
  - the entries are binary term-match indicators, not Gaussian;
  - the ROWS are dependent - every candidate window of one record is a row, and they share terms and reports;
  - the COLUMNS are dependent - phrases co-occur by construction, and some are EXACTLY collinear (a phrase nested in
    another, synonyms the model always picks together), which the fit must merge first.
arXiv:2608.08390 (Mesforush and Parhi) proves, for X = A Z B (A mixes rows, B mixes columns, Z i.i.d. sub-Gaussian),
a k = o(n)-sparse signal, Gaussian noise and a FIXED penalty, that the lasso's solution and squared error are the
same as with a Gaussian Z, and that under a strict dual-feasibility condition the full lasso has no false positives.
What that does and does not cover here:
  - its "sparse regime" is the SIGNAL's sparsity (k = o(n)), not more features than rows: this design is n ~ 4100
    windows by d ~ 320 features (gamma ~ 0.08);
  - the cross-validated fits of tools/lex_learn.py come out DENSE (hundreds of nonzero coefficients), outside its
    Assumption 1; tools/lex_universality.py `support` fits at a fixed penalty with k << n instead;
  - our response is 0/1 (gold or not), not a linear signal plus Gaussian noise, so fits on the real labels sit
    outside its noise model; tools/lex_universality.py `twin` runs the paper's own planted-signal protocol on this
    real design, which is the part the theorem speaks to.
Results: work/analysis/LEX_LEARNED.md; citation: work/analysis/SOTA_2026.md section 5.

HONESTY. The fit uses gold unit ids, so a learned arm is only reportable OUT OF FOLD: tools/lex_learn.py trains on
k-1 folds of the DEV questions and scores the held-out fold with weights that never saw it. Dev is 64 wide records;
treat any difference under a few points as noise.
"""
import json
import math
import re
from collections import defaultdict

import numpy as np

from ...normalize import fold

FEATURE_VERSION = "lex-learn-v1"
_DIGIT = re.compile(r"\d")


# ---------------------------------------------------------------------------------------------------------------
# features
# ---------------------------------------------------------------------------------------------------------------

def features(w, got, terms, kw, years, names):
    """One candidate window -> {feature name: value}, sparse.

    The generic block mirrors what the hand-set score already uses (so the learner CAN recover it), then the phrase
    block gives every matched term its own coefficient. `got` is {term index: relaxed} as `credit` returns it."""
    f = defaultdict(float)
    text, kind, table = w["text"], w["kind"], w.get("table")
    folded = fold(text)
    f["n_terms"] = float(len(got))
    for i, rel in got.items():
        t = terms[i]
        f["idf_relaxed" if rel else "idf_phrase"] += t.idf
        f["n_relaxed" if rel else "n_phrase"] += 1.0
        f["w_source:%s" % t.source] += 1.0
        if t.weak:
            f["n_weak"] += 1.0
        if len(fold(t.text).split()) > 1:
            f["n_multiword"] += 1.0
        f["ph:%s" % fold(t.text)] = 1.0            # the p >> n block
    f["kind:%s" % kind] = 1.0
    if table:
        f["is_table"] = 1.0
    f["log_chars"] = math.log(1.0 + len(text or ""))
    if _DIGIT.search(text or ""):
        f["has_digit"] = 1.0
        if kw.get("answer") == "number":
            f["number_wanted"] = 1.0
    if years and any(y in (text or "") for y in years):
        f["year_hit"] = 1.0
    if names and any(nm in folded for nm in names):
        f["name_hit"] = 1.0
    return dict(f)


def scorer(model, keep=False):
    """-> a `search.windows` score_fn. `keep` also hangs the feature vector on the window (the training dump)."""
    def score_fn(w, got, terms, kw, years, names):
        f = features(w, got, terms, kw, years, names)
        if keep:
            w["feats"] = f
        return model.score(f) if model is not None else w["hand_score"]
    return score_fn


# ---------------------------------------------------------------------------------------------------------------
# the lasso (squared loss, coordinate descent, active set)
# ---------------------------------------------------------------------------------------------------------------

def lasso(X, y, lam, n_sweeps=60, tol=1e-6):
    """min_b 1/(2n) ||y - b0 - Xb||^2 + lam ||b||_1, on standardised columns. -> (b0, b) in the ORIGINAL units.

    Cyclic coordinate descent with an active set: one full sweep, then only the coefficients that left zero, until
    a full sweep changes nothing. Columns are centred and scaled because the phrase indicators and the idf sums live
    on very different scales and one penalty has to be fair to both."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    n, p = X.shape
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd[sd < 1e-12] = 1.0
    Z = (X - mu) / sd
    ybar = float(y.mean())
    r = y - ybar
    b = np.zeros(p)
    csq = (Z * Z).sum(axis=0)
    active = list(range(p))
    for sweep in range(n_sweeps):
        big = 0.0
        for j in active:
            if csq[j] <= 0:
                continue
            rho = float(Z[:, j] @ r) + b[j] * csq[j]
            nb = np.sign(rho) * max(abs(rho) - lam * n, 0.0) / csq[j]
            d = nb - b[j]
            if d != 0.0:
                r -= Z[:, j] * d
                b[j] = nb
                big = max(big, abs(d))
        if len(active) == p:
            nz = [j for j in range(p) if b[j] != 0.0]
            if big < tol:
                break
            active = nz or list(range(p))
        elif big < tol:
            active = list(range(p))          # one more full sweep to confirm
    w = b / sd
    return float(ybar - float(mu @ w)), w


def lam_max(X, y):
    """The smallest penalty that zeroes every coefficient - the top of the path."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    mu, sd = X.mean(axis=0), X.std(axis=0)
    sd[sd < 1e-12] = 1.0
    Z = (X - mu) / sd
    return float(np.abs(Z.T @ (y - y.mean())).max() / max(1, len(y)))


def path(n=12, hi=None, ratio=1e-3):
    hi = hi or 1.0
    return [hi * (ratio ** (i / max(1, n - 1))) for i in range(n)]


# ---------------------------------------------------------------------------------------------------------------
# dataset / model
# ---------------------------------------------------------------------------------------------------------------

class Dataset:
    """Rows of (question key, feature dict, label). `matrix` fixes a column order and drops rare phrase columns."""

    def __init__(self):
        self.rows = []

    def add(self, key, feats, label):
        self.rows.append({"key": key, "f": dict(feats), "y": float(label)})

    def keys(self):
        out = []
        for r in self.rows:
            if r["key"] not in out:
                out.append(r["key"])
        return out

    def subset(self, keys):
        d = Dataset()
        keep = set(keys)
        d.rows = [r for r in self.rows if r["key"] in keep]
        return d

    def matrix(self, min_df=2, names=None, phrases=True):
        if names is None:
            df = defaultdict(int)
            for r in self.rows:
                for k in r["f"]:
                    df[k] += 1
            names = sorted(k for k, c in df.items()
                           if not k.startswith("ph:") or (phrases and c >= min_df))
        idx = {k: i for i, k in enumerate(names)}
        X = np.zeros((len(self.rows), len(names)))
        for i, r in enumerate(self.rows):
            for k, v in r["f"].items():
                j = idx.get(k)
                if j is not None:
                    X[i, j] = v
        y = np.array([r["y"] for r in self.rows])
        return X, y, names


class Model:
    """A fitted window scorer. `score(feats)` is the dot product; unknown features (a phrase never seen in training)
    contribute nothing, which is the honest behaviour for an out-of-fold question."""

    def __init__(self, bias, weights, meta=None):
        self.bias = float(bias)
        self.w = {k: float(v) for k, v in weights.items() if v}
        self.meta = meta or {}

    def score(self, feats):
        s = self.bias
        for k, v in feats.items():
            c = self.w.get(k)
            if c:
                s += c * v
        return s

    def to_json(self):
        return {"feature_version": FEATURE_VERSION, "bias": self.bias, "weights": self.w, "meta": self.meta}

    def save(self, path_):
        with open(path_, "w", encoding="utf-8") as fh:
            json.dump(self.to_json(), fh, ensure_ascii=False, indent=1, sort_keys=True)

    @classmethod
    def load(cls, path_):
        with open(path_, encoding="utf-8") as fh:
            d = json.load(fh)
        if d.get("feature_version") != FEATURE_VERSION:
            raise ValueError("learned model was fitted with %s, this is %s" % (d.get("feature_version"),
                                                                               FEATURE_VERSION))
        return cls(d["bias"], d["weights"], d.get("meta"))


def fit(data, lam, min_df=2, names=None, phrases=True):
    X, y, names = data.matrix(min_df=min_df, names=names, phrases=phrases)
    b0, b = lasso(X, y, lam)
    m = Model(b0, dict(zip(names, b)), {"lam": lam, "phrases": bool(phrases), "n_rows": int(X.shape[0]),
                                        "p": int(X.shape[1]),
                                        "nonzero": int((b != 0).sum()), "n_questions": len(data.keys()),
                                        "positives": int(y.sum())})
    return m


def rank_score(data, model, k=8):
    """How well a model ranks: the share of each question's gold windows that reach its top k. This is what the score
    is FOR (the packer takes the best windows), so the penalty is chosen on it rather than on squared error."""
    by = defaultdict(list)
    for r in data.rows:
        by[r["key"]].append(r)
    hits, tot = 0.0, 0
    for _key, rows in by.items():
        gold = sum(1 for r in rows if r["y"] > 0)
        if not gold:
            continue
        top = sorted(rows, key=lambda r: -model.score(r["f"]))[:k]
        hits += sum(1 for r in top if r["y"] > 0) / gold
        tot += 1
    return hits / tot if tot else 0.0


def choose_lam(data, folds=4, n_lam=10, min_df=2, k=8, seed=0, rule="best", phrases=True):
    """Inner cross-validation over the penalty path, on QUESTIONS (never on rows: the windows of one question are the
    dependent rows). -> (lam, [(lam, mean rank score, standard error)]).

    `rule`: "best" takes the top of the path, "1se" the LARGEST penalty (simplest model) whose mean is within one
    standard error of the top - the usual guard when the path is as flat and noisy as 43 questions make it."""
    keys = data.keys()
    rng = np.random.RandomState(seed)
    order = list(keys)
    rng.shuffle(order)
    parts = [order[i::folds] for i in range(folds)]
    X, y, _names = data.matrix(min_df=min_df, phrases=phrases)
    lams = path(n_lam, hi=lam_max(X, y))
    table = []
    for lam in lams:
        s = []
        for i in range(folds):
            te = parts[i]
            tr = [q for q in order if q not in set(te)]
            if not tr or not te:
                continue
            m = fit(data.subset(tr), lam, min_df=min_df, phrases=phrases)
            s.append(rank_score(data.subset(te), m, k=k))
        mean = float(np.mean(s)) if s else 0.0
        se = float(np.std(s) / math.sqrt(len(s))) if len(s) > 1 else 0.0
        table.append((lam, mean, se))
    top = max(table, key=lambda t: (t[1], t[0]))
    if rule == "1se":
        floor = top[1] - top[2]
        best = max([t for t in table if t[1] >= floor] or [top], key=lambda t: t[0])[0]
    else:
        best = top[0]
    return best, table
