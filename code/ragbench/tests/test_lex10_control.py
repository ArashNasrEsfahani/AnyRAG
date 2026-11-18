def test_hybrid15500_control_matches_lex10b_budget():
    from ragbench.approaches.hybrid import HybridRetrieval
    from ragbench.approaches.lex10 import approach as A
    from ragbench.approaches.lex10 import control as C
    assert issubclass(C.Hybrid15500, HybridRetrieval) and C.Hybrid15500.name == "HYBRID15500"
    assert C.LEX10_BUDGET == A.LEX10_BUDGET == 15500 and C.DEPTH == A.DEPTH == 40
