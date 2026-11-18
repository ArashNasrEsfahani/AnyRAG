"""Offline tests of LEX11-M (approaches/lex11/order.py). No model, no network.

    .venv/Scripts/python.exe -m ragbench.tests.test_lex11
"""
import math
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ragbench.approaches.lex11 import order as OR
from ragbench.contracts import RetrievedItem, pack_evidence


class _NoCorpus:
    def doc(self, d):
        raise KeyError(d)

    def unit(self, u):
        raise KeyError(u)


CFG = SimpleNamespace(evidence={"max_chars": 6000, "min_item_chars": 40})


def _items(n, length=300):
    return [RetrievedItem(text=("chunk %d " % c) + "x" * length, doc_id="D%d" % (c % 3), kind="chunk", rank=i,
                          meta={"chunk": c}) for i, c in enumerate(range(100, 100 + n))]


def test_h_score_and_ranks():
    assert OR.h_score(None, None) == 0.0
    assert math.isclose(OR.h_score(1, None), 1 / 61)
    assert math.isclose(OR.h_score(1, 1), 2 / 61)
    assert math.isclose(OR.h_score(41, 3), 1 / 63)          # beyond depth 40 contributes 0
    r = OR.channel_ranks([5, 6, 7], [7, 8])
    assert r == {5: (1, None), 6: (2, None), 7: (3, 1), 8: (None, 2)}


def test_reorder_ties_and_sparse_only_last():
    its = _items(4)
    ranks = OR.channel_ranks([101, 103], [103, 101])             # 101 and 103 tie on H; 100/102 sparse-only
    f2 = {100: 1, 101: 2, 102: 3, 103: 4}
    new, keys = OR.reorder(its, ranks, f2)
    assert [it.meta["chunk"] for it in new] == [101, 103, 100, 102]   # tie -> best rank 1 both -> F2 rank
    assert keys[-1][0] == 0 and keys[-2][0] == 0


def test_frozen_membership_text_and_chars():
    its = _items(30)
    frozen = pack_evidence(its, _NoCorpus(), budget=6000, cfg=CFG)
    assert frozen.last_item_cut                                   # the clipped item is part of the frozen set
    rev = list(reversed([int(it.meta["chunk"]) for it in frozen.items]))
    ranks = OR.channel_ranks(rev, [])
    ev, info = OR.lex11_pack(frozen, ranks, [it.meta["chunk"] for it in its], _NoCorpus())
    assert info["post"] == rev and ev.chars == frozen.chars and ev.n_included == frozen.n_included
    assert sorted(OR.displayed(ev)) == sorted(OR.displayed(frozen))
    # identity order renders exactly like pack_evidence
    same = OR.render(frozen.items, _NoCorpus(), frozen)
    assert same.text == frozen.text


def test_invariance_check_raises():
    its = _items(5)
    frozen = pack_evidence(its, _NoCorpus(), budget=6000, cfg=CFG)
    bad = OR.render(frozen.items[:-1], _NoCorpus(), frozen)
    try:
        OR.check_invariance(frozen, bad)
        raise RuntimeError("invariance check did not raise")
    except AssertionError:
        pass


def test_cells():
    from ragbench.approaches.lex11 import approach as A
    assert A.Lex11M6000.budget_chars == 6000 and A.Lex11M6000.reorder
    assert A.Lex11M15500.budget_chars == 15500 and A.Lex11M15500.reorder
    assert A.F2At6000.budget_chars == 6000 and not A.F2At6000.reorder


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for f in fns:
        f()
        print("ok", f.__name__)
    print("%d tests passed" % len(fns))
