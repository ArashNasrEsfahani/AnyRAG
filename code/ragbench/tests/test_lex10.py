from ragbench.approaches.lex10 import tokbudget as T


def test_fit_predict_pick():
    calls = [{"lang": "fa", "o200k": 100, "glm_prompt": 120, "glm_completion": 10},
             {"lang": "fa", "o200k": 300, "glm_prompt": 348, "glm_completion": 30},
             {"lang": "en", "o200k": 200, "glm_prompt": 200, "glm_completion": 50}]
    m = T.fit(calls)
    assert abs(m["fa"]["k"] - 1.17) < 1e-9 and m["fa"]["completion"] == 20
    assert T.predict(m, "en", 100) == 150
    assert T.pick_budget({6000: 2100, 8500: 3000, 9000: 3100}, 1.40 * 2169) == 8500
    assert T.pick_budget({6000: 5000}, 3036.6) is None
    # LEX10b: the amended cap (2.40x HYBRID) picks 15,500 from the saved prediction table
    assert T.pick_budget({15000: 4991.8, 15500: 5135.2, 16000: 5306.2}, 2.40 * 2169) == 15500


def test_lex10_registered():
    from ragbench.approaches import APPROACHES
    from ragbench.approaches.lex10 import approach as A
    assert APPROACHES["LEX10"] == "ragbench.approaches.lex10.approach:Lex10"
    assert A.LEX10_BUDGET == 15500 and A.Lex10.name == "LEX10"
