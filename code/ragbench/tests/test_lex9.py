"""LEX9 offline checks: leakage control in the synthetic chunk field, number folding, order ties, frozen LR scorer."""
import json

import numpy as np

from ragbench.approaches.lex9 import features as F
from ragbench.approaches.lex9 import model as M


class _Retr:
    vocab_dict = {"steel": 0, "output": 1}

    def get_scores(self, toks):
        return np.asarray([5.0, 3.0, 1.0, 4.0], np.float32)


def _field():
    f = F.SynthChunkField.__new__(F.SynthChunkField)
    f.retr, f.qchunk, f.n = _Retr(), np.asarray([0, 0, 1, 2]), 3
    return f


def test_self_question_excluded():
    f = _field()
    full = f.chunk_scores("steel output")
    assert list(full) == [5.0, 1.0, 4.0]
    ex = f.chunk_scores("steel output", exclude_q=0)          # sibling question 1 of chunk 0 remains
    assert list(ex) == [3.0, 1.0, 4.0]
    ex3 = f.chunk_scores("steel output", exclude_q=3)
    assert ex3[2] == 0.0


def test_numbers_fold():
    assert F.numbers("تولید ۱۲٬۵۰۰ تن در 1402") == {"12500", "1402"}
    assert F.numbers("3.5% and 007") == {"3.5", "7"}


def test_order_ties():
    U = np.asarray([7, 3, 9])
    assert M.order(U, [1.0, 1.0, 2.0], [0.1, 0.5, 0.0]) == [9, 3, 7]
    assert M.order(U, [1.0, 1.0, 1.0], [0.2, 0.2, 0.2]) == [3, 7, 9]


def test_lr_scorer(tmp_path):
    k = len(F.FEATURES)
    meta = {"kind": "lr", "features": F.FEATURES, "mean": [0.0] * k, "scale": [1.0] * k,
            "coef": [1.0] + [0.0] * (k - 1), "intercept": 0.5}
    (tmp_path / "model.json").write_text(json.dumps(meta), encoding="utf-8")
    S = M.Scorer(tmp_path)
    X = np.zeros((2, k))
    X[1, 0] = 2.0
    assert list(S.score(X)) == [0.5, 2.5]
    meta["features"] = F.FEATURES[:-1]
    (tmp_path / "model.json").write_text(json.dumps(meta), encoding="utf-8")
    try:
        M.Scorer(tmp_path)
        assert False
    except ValueError:
        pass


if __name__ == "__main__":
    import sys
    import tempfile
    from pathlib import Path
    sys.stdout.reconfigure(encoding="utf-8")
    fails = 0
    for name, fn in [("self_question_excluded", test_self_question_excluded), ("numbers_fold", test_numbers_fold),
                     ("order_ties", test_order_ties)]:
        try:
            fn()
            print("ok", name)
        except AssertionError as e:
            fails += 1
            print("FAIL", name, e)
    with tempfile.TemporaryDirectory() as d:
        try:
            test_lr_scorer(Path(d))
            print("ok lr_scorer")
        except AssertionError as e:
            fails += 1
            print("FAIL lr_scorer", e)
    print("%d failed" % fails)
    raise SystemExit(1 if fails else 0)
