"""Offline tests for SETR (approaches/setr): prompt fidelity, parser, reorder, packing order. No model call."""
from ragbench.approaches import APPROACHES
from ragbench.approaches.setr import approach as S


def test_registered():
    assert APPROACHES["SETR"] == "ragbench.approaches.setr.approach:SetR"


def test_prompt_verbatim_and_rendering():
    msgs, cut = S.build_prompt("Q?", ["a\n\nb", "c" * 2000])
    assert msgs[0] == {"role": "system", "content": S.SYSTEM}
    u = msgs[1]["content"]
    assert u.startswith("I will provide you with 2 passages, each indicated by a numerical identifier []. "
                        "Select the passages based on their relevance to the search query: Q?.\n\n[1] a b\n[2] ")
    assert "Search Query: Q?\n\nPlease follow the steps below:\nStep 1. Please list up the information requirements" in u
    assert u.endswith("e.g., ### Final Selection: [4] [2].")
    assert cut == 1 and ("c" * S.PASSAGE_CAP + " …") in u and ("c" * (S.PASSAGE_CAP + 1)) not in u


def test_parse():
    P = S.parse_selection
    assert P("Step 1 ... [3] has X\n### Final Selection: [4] [2] [4] [99]", 20) == ([4, 2], "ok")
    assert P("**Final Selection:** [1][3]", 20) == ([1, 3], "ok")
    # the LAST marker wins (a CoT may quote the format first)
    assert P("format '### Final Selection: [] []'\n...\n### Final Selection: [5]", 20) == ([5], "ok")
    assert P("I pick [1] [2]", 20) == ([], "no_marker")
    assert P("### Final Selection: []", 20) == ([], "empty_selection")
    assert P("### Final Selection: [21]", 20) == ([], "empty_selection")
    assert P("", 20) == ([], "empty_reply") and P(None, 20) == ([], "empty_reply")


def test_reorder():
    items = list("abcdefghij")
    assert S.reorder(items, [3, 1], 5) == list("cabdefghij")
    assert S.reorder(items, [], 5) == items
    assert S.reorder(items, [7], 5) == items          # outside the shown top is ignored


def test_defaults_declared():
    d = S.SETR_DEFAULTS
    assert d["top"] == 20 and d["deadline_s"] == 8.0 and d["attempts"] == 1 and d["budget"] == 6000
    assert d["selector"] == "z-ai/glm-5.3-flash"


if __name__ == "__main__":
    import sys
    n = 0
    for k, f in list(globals().items()):
        if k.startswith("test_") and callable(f):
            f()
            n += 1
    print("test_setr: %d/%d ok" % (n, n))
    sys.exit(0)
