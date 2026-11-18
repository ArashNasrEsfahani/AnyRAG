"""LEX9 learned scorer: a frozen logistic regression (json) or LightGBM lambdarank booster (txt) over
features.FEATURES. score(X) -> one float per candidate; higher = earlier in the order."""
import json
from pathlib import Path

import numpy as np

from .features import FEATURES

MODEL_VERSION = "lex9-model-v1"


class Scorer:
    def __init__(self, path):
        path = Path(path)
        meta = json.loads((path / "model.json").read_text(encoding="utf-8"))
        if meta["features"] != FEATURES:
            raise ValueError("feature list differs from the frozen model's")
        self.kind = meta["kind"]
        self.meta = meta
        if self.kind == "lr":
            self.mu = np.asarray(meta["mean"])
            self.sd = np.asarray(meta["scale"])
            self.w = np.asarray(meta["coef"])
            self.b = float(meta["intercept"])
        else:
            import lightgbm as lgb
            self.booster = lgb.Booster(model_file=str(path / "lgbm.txt"))

    def score(self, X):
        if not len(X):
            return np.zeros(0)
        if self.kind == "lr":
            return ((X - self.mu) / self.sd) @ self.w + self.b
        return self.booster.predict(X, num_threads=1)


def order(U, scores, tiebreak):
    """candidates by learned score desc, ties by tiebreak desc (F2 MaxMean), then chunk id."""
    idx = np.lexsort((np.asarray(U), -np.asarray(tiebreak), -np.asarray(scores)))
    return [int(U[i]) for i in idx]
