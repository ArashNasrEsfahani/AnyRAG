def test_budget_matrix_cells_only_raise_the_pack_budget():
    from ragbench.approaches import budget_matrix as BM
    from ragbench.approaches.hybrid import HybridRetrieval
    from ragbench.approaches.lex10.approach import LEX10_BUDGET
    from ragbench.approaches.lex5b.approach import LEX5B_DEFAULTS, Lex5b
    from ragbench.approaches.lex7.approach import Lex7
    assert BM.BUDGET == LEX10_BUDGET == 15500
    assert LEX5B_DEFAULTS["budget"] == 6000 and getattr(Lex5b, "budget_chars", None) is None   # 6,000 arm unchanged
    for cls, base in ((BM.HR15500, HybridRetrieval), (BM.Lex7_15500, Lex7), (BM.Lex5b_15500, Lex5b)):
        assert issubclass(cls, base) and cls.budget_chars == 15500 and cls.__mro__[1] is BM._Wide
    hr = BM.HR15500()
    assert hr.flags.get("rerank") is True and hr.index_variant == "HR15500"
    assert hr.config_signature()["budget_matrix"]["budget_chars"] == 15500


def test_wide_pack_uses_15500_by_default():
    from ragbench.approaches import budget_matrix as BM
    seen = {}

    def fake(items, corpus, budget=None, cfg=None):
        seen["budget"] = budget
        return None

    orig = BM.pack_evidence
    BM.pack_evidence = fake
    try:
        hr = BM.HR15500()
        hr.pack([], type("I", (), {"corpus": None})())
        assert seen["budget"] == 15500
        hr.pack([], type("I", (), {"corpus": None})(), budget=700)
        assert seen["budget"] == 700
    finally:
        BM.pack_evidence = orig
