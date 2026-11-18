"""Offline tests of LEX6 (approaches/lex6: core.py packing and document ranking, router.py validation, approach.py).

    .venv/Scripts/python.exe -m ragbench.tests.test_lex6

No network, no model: fake packages for the packer, the synthetic corpus + FakeOllama for the approach, fake
router / selector clients.
"""
import sys
import tempfile
import time
from pathlib import Path
from types import MappingProxyType

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[2]))

from ragbench.approaches.lex5 import bundles as B  # noqa: E402
from ragbench.approaches.lex6 import core as K6  # noqa: E402
from ragbench.approaches.lex6 import router as RT  # noqa: E402

RESULTS = {}


def check(name, cond, detail=""):
    RESULTS[name] = {"ok": bool(cond), "detail": detail}
    print(("PASS " if cond else "FAIL ") + name + (" - " + str(detail)[:300] if detail and not cond else ""), flush=True)
    return cond


def pkg(c, doc="d", a=None, size=1000):
    a = c * 10000 if a is None else a
    return MappingProxyType({"chunk": c, "doc_id": doc, "text": "x" * size, "provenance": ((doc, a, a + size, "body"),),
                             "_cost1": size})


def flat_cost(p, n):
    return len(p["text"])


def ids(xs):
    return [p["chunk"] for p in xs]


# ------------------------------------------------------------------------------------------------------------------
def fresh():
    return B.Coster(flat_cost)          # Coster caches per (chunk, n): one per scenario


def test_anchor():
    cost = fresh()
    b0 = [pkg(1, size=1200), pkg(2, size=1200), pkg(3, size=1200), pkg(4, size=1200), pkg(5, size=1000)]
    check("anchor.shortest_prefix_reaching_3000", ids(K6.anchor_of(b0, cost)) == [1, 2, 3])
    check("anchor.exact_3000_stops", ids(K6.anchor_of([pkg(1, size=1500), pkg(2, size=1500), pkg(3)], fresh())) == [1, 2])
    check("anchor.all_when_short", ids(K6.anchor_of([pkg(1, size=800), pkg(2, size=900)], fresh())) == [1, 2])
    check("anchor.none_for_no_anchor", K6.anchor_of(b0, cost, target=0) == [])


def test_pack():
    cost = B.Coster(flat_cost)
    b0 = [pkg(1, size=1200), pkg(2, size=1200), pkg(3, size=1200), pkg(4, size=1200), pkg(5, size=1100)]
    rrf = b0 + [pkg(6, size=500)]
    Hs = {p["chunk"]: 1.0 / (60 + i) for i, p in enumerate(rrf, 1)}
    # no-change branch 1: every slot already represented in B0 (an alternative contained in a B0 package)
    inside = pkg(90, a=20000 + 100, size=300)              # inside package 2's interval
    shown, info = K6.pack(b0, [{"key": "s", "alts": [inside, pkg(91)]}], rrf, Hs, cost)
    check("pack.package_aware_represented_no_change", shown == b0 and not info["changed"]
          and info["reason"] == "no_unresolved_slot", info)
    # a slot is NOT represented by a mere same-document package that does not contain it
    near = pkg(92, a=20000 + 1100, size=400)               # overlaps package 2 partially
    shown, info = K6.pack(b0, [{"key": "s", "alts": [near]}], rrf, Hs, cost)
    check("pack.partial_overlap_is_unresolved", info["unresolved"] == [0] and 92 in ids(shown), info)
    # replacement keeps the anchor, displaces from the tail, fills from the global order
    shown, info = K6.pack(b0, [{"key": "s", "alts": [pkg(20, doc="e", size=900)]}], rrf, Hs, cost)
    check("pack.anchor_kept_first", ids(shown)[:3] == [1, 2, 3] and ids(shown)[3] == 20, ids(shown))
    check("pack.budget_and_displacement", cost(shown) <= 6000 and info["displaced"] and info["changed"], info)
    # addition cap: an alternative whose addition exceeds 3,000 cannot be placed -> B0 exactly
    cost = fresh()
    b0s = [pkg(1, size=1000)]                     # short B0: the cap binds before the budget
    big = MappingProxyType(dict(pkg(30, doc="e", size=3100), _cost1=10))   # _cost1 lies; the pack uses exact cost
    shown, info = K6.pack(b0s, [{"key": "s", "alts": [big]}], b0s, {}, cost)
    check("pack.addition_cap_3000", shown == b0s and info["reason"] == "no_slot_fits", info)
    ok_add = pkg(31, doc="e", size=2900)
    shown, info = K6.pack(b0s, [{"key": "s", "alts": [ok_add]}], b0s, {}, cost)
    check("pack.addition_under_cap_placed", 31 in ids(shown) and info["additions_cost"] <= 3000, info)
    # alternative preference: alt1 when both fit; alt2 when only alt2 fits
    cost = fresh()
    a1, a2 = pkg(40, doc="e", size=800), pkg(41, doc="e", size=700)
    shown, info = K6.pack(b0, [{"key": "s", "alts": [a1, a2]}], rrf, Hs, cost)
    check("pack.prefers_alt1", 40 in ids(shown) and 41 not in ids(shown) and info["alt_rank_sum"] == 0, info)
    a1b = pkg(42, doc="e", size=3200)
    shown, info = K6.pack(b0, [{"key": "s", "alts": [a1b, a2]}], rrf, Hs, cost)
    check("pack.alt2_when_alt1_too_big", 41 in ids(shown) and info["alt_rank_sum"] == 1, info)
    # most slots first: two 1,400-char slots beat one
    s1, s2 = pkg(50, doc="e", size=1200), pkg(51, doc="f", size=1200)
    shown, info = K6.pack(b0, [{"key": "a", "alts": [s1]}, {"key": "b", "alts": [s2]}], rrf, Hs, cost)
    check("pack.most_slots", info["covered"] == 2 and {50, 51} <= set(ids(shown)), info)
    # 3^8 enumeration bound and speed
    slots = [{"key": str(i), "alts": [pkg(100 + 2 * i, doc="g%d" % i, size=300), pkg(101 + 2 * i, doc="g%d" % i, size=200)]}
             for i in range(9)]
    t = time.perf_counter()
    shown, info = K6.pack(b0, slots, rrf, Hs, cost)
    dt = time.perf_counter() - t
    check("pack.max_8_slots", len(info["unresolved"]) == 8, info["unresolved"])
    check("pack.enumeration_bounded_3pow8", info["evaluated"] <= 3 ** 8, info["evaluated"])
    check("pack.enumeration_fast", dt < 20.0, dt)
    check("pack.enumeration_covers_all_fitting", info["covered"] == 8 and cost(shown) <= 6000, info)


def test_docrank():
    V = {"b": 0.03, "a": 0.03, "c": 0.0, "d": 0.02}
    cent = {"a": 0.5, "b": 0.7}
    R = K6.rank_docs(V, cent)
    check("docrank.zero_evidence_not_ranked", "c" not in R["V"] and "c" not in R["F"] and "d" in R["F"], R["F"])
    check("docrank.doc_id_ties", R["V"][:2] == ["a", "b"], R["V"])
    check("docrank.missing_centroid_no_credit", R["F_scores"]["d"] == 1.0 / 63, R["F_scores"])
    R2 = K6.rank_docs(V, cent, channels="s5")
    check("docrank.channel_s5_only", R2["F"] == ["b", "a"], R2["F"])
    Rc = K6.rank_docs(V, cent, {"c": 2.0, "a": 0.0}, {"c": 0.4, "d": 0.3})
    check("docrank.cards_half_weight", abs(Rc["F_scores"]["c"] - 1.0 / 61) < 1e-12 and "a" not in Rc["cardB"],
          Rc["F_scores"])
    sl = K6.shortlist({"V": ["a", "b", "c"], "C": ["a", "d", "e"], "F": ["a", "f", "b", "g", "h", "i", "j", "k", "l"]})
    check("docrank.shortlist_order_and_cap", sl == ["a", "b", "d", "f", "g", "h", "i", "j"], sl)


def test_router_parse():
    ok = RT.parse("P1: 1-4 | D2 D5\nP2: 6-9 | D1\nEND", 10, 6)
    check("router.parse_valid", ok["ok"] and ok["parts"] == [{"words": [1, 4], "ids": [2, 5]},
                                                             {"words": [6, 9], "ids": [1]}], ok)
    check("router.parse_all_none", RT.parse("NONE\nEND", 10, 6)["status"] == "all_none")
    check("router.parse_persian_digits", RT.parse("P1: ۱-۳ | D۲\nEND", 10, 6)["ok"])
    bad = {"empty": "", "no_end": "P1: 1-2 | D1", "text_after_end": "P1: 1-2 | D1\nEND\nok",
           "bad_line": "Part one: D1\nEND", "part_index": "P2: 1-2 | D1\nEND", "bad_range": "P1: 5-2 | D1\nEND",
           "id_out_of_range": "P1: 1-2 | D9\nEND", "duplicate_id_in_part": "P1: 1-2 | D1 D1\nEND",
           "overflow_ids": "P1: 1-2 | D1 D2 D3\nEND", "bad_ids": "P1: 1-2 | doc1\nEND",
           "overflow_parts": "".join("P%d: 1-2 | D1\n" % i for i in range(1, 6)) + "END",
           "overflow_docs": "P1: 1-2 | D1 D2\nP2: 3-4 | D3 D4\nP3: 5-6 | D5\nEND",
           "range_beyond_question": "P1: 1-11 | D1\nEND"}
    for why, text in bad.items():
        r = RT.parse(text, 10, 6)
        want = "bad_range" if why == "range_beyond_question" else why
        check("router.reject.%s" % why, not r["ok"] and r["reason"] == want, r)
    tr = RT.parse("P1: 1-2 | D1\nEND", 10, 6, finish_reason="length")
    check("router.reject.truncated", not tr["ok"] and tr["reason"] == "truncated", tr)


def test_router_prompt():
    if not RT.TOKENIZER.exists():
        check("router.prompt_tokenizer_present", False, str(RT.TOKENIZER))
        return
    cards = {"d%d" % i: ("[D] document %d about topic %d " % (i, i)) + "word " * 55 for i in range(8)}
    p, shown, t = RT.build_prompt("What was the revenue of the company in 2021 and 2022?", list(cards), cards.get)
    check("router.prompt_fits_cap", p is not None and t <= RT.R["content_max"] and len(shown) >= 2, (t, len(shown)))
    check("router.prompt_prunes_from_end", shown == list(cards)[:len(shown)], shown)
    pp, sp, _ = RT.build_prompt("What was the revenue?", list(cards), cards.get, permute=True)
    check("router.permute_same_set_other_order", set(sp) == set(shown) or len(sp) != len(shown), (sp, shown))
    huge = {"a": "w " * 900, "b": "w " * 900}
    check("router.prompt_no_fit", RT.build_prompt("q?", list(huge), huge.get)[0] is None)


def test_flags():
    from ragbench.run import parse_flags
    check("flags.lex6_default_no_variant", parse_flags(["rank=s2s5"], "LEX6") == {})
    check("flags.lex6_values", parse_flags(["rank=s5,absent_only,cards"], "LEX6") ==
          {"rank": "s5", "absent_only": True, "cards": True})
    for bad in ("rank=s7", "rerank"):
        try:
            parse_flags([bad], "LEX6")
            check("flags.lex6_reject.%s" % bad, False)
        except SystemExit:
            check("flags.lex6_reject.%s" % bad, True)


class FakeChat:
    model_id, provider = "fake/router", ["fake"]

    def __init__(self, reply, finish="stop", fail=None):
        self.reply, self.finish, self.fail, self.calls = reply, finish, fail, []

    def chat(self, msgs, **kw):
        from ragbench.llm import ChatResult
        self.calls.append(msgs)
        if self.fail:
            raise self.fail
        r = ChatResult(text=self.reply, prompt_tokens=500, completion_tokens=40, wall_ms=10.0,
                       done_reason=self.finish, cost_usd=0.0)
        r.reasoning_tokens, r.provider = 10, "fake"
        return r

    def warm_deadline(self):
        return 200


def test_approach(tmp):
    from ragbench.approaches import get_approach
    from ragbench.llm import Ledger
    from ragbench.tests.test_baselines import FakeOllama
    from ragbench.tests.test_hybrid import make_set, tcfg
    from ragbench.tests.test_rerank import question
    cs = make_set()
    cfg = tcfg(tmp / "idx")
    fo = FakeOllama(cfg, Ledger(path=None))
    hy = get_approach("HYBRID", cfg=cfg, client=fo, flags={"intact": True})
    hy.build_index(cs)
    base = hy.answer(question(), "en-en")
    b0 = base.trace["lex5"]["shown_chunks"]
    ap = get_approach("LEX6", cfg=cfg, client=FakeOllama(cfg, Ledger(path=None)), flags=None)
    ap.build_index(cs)
    rec = ap.answer(question(), "en-en")
    lt = rec.trace.get("lex6") or {}
    check("approach.ok", rec.error is None and lt.get("integrity_failure") is None, (rec.error, lt.get("integrity_failure")))
    check("approach.b0_equals_hybrid_intact", lt.get("b0_chunks") == b0, (lt.get("b0_chunks"), b0))
    check("approach.exact_cost_no_cut", lt.get("cost_matches") and not lt.get("cut") and rec.evidence_chars <= 6000, lt)
    check("approach.one_llm_call", rec.llm_calls == 1, rec.llm_calls)
    tr = lt.get("lex6") or {}
    check("approach.trace_fields", all(k in tr for k in ("docrank", "shortlist", "nominated", "slots", "pack",
                                                         "final", "final_why", "provenance")), list(tr))
    check("approach.nominates_top3", len(tr.get("nominated") or []) <= 3 and tr["nominated"] == [d for d, _ in tr["docrank"]["F"][:3]])
    check("approach.provenance_in_items", all("provenance" in (u.get("meta") or {}) for u in rec.retrieved_units if isinstance(u, dict)) and len(rec.retrieved_units) > 0, "")
    check("approach.variant", ap.variant == "LEX6")
    # the router path with a fake client (cards are faked through a tiny view)
    from ragbench.approaches.lex6 import approach as A6

    class View:
        def compact(self, d):
            return "card of %s" % d

        def doc_scores(self, q, v):
            return {}, {}

        def sections(self, d, v, n):
            return []

    class Store:
        def ready(self, docs):
            return True

        def view(self, docs):
            return View()
    A6._STORE["s"] = Store()
    for name, fake in (("ok", FakeChat("P1: 1-3 | D1\nEND")), ("deadline", FakeChat("", fail=Exception("deadline 0.7s exceeded"))),
                       ("invalid", FakeChat("P1: 1-3 | D9\nEND"))):
        apr = get_approach("LEX6", cfg=cfg, client=FakeOllama(cfg, Ledger(path=None)), flags={"router": True})
        apr._rc = fake
        apr.build_index(cs)
        orig = RT.trigger
        RT.trigger = lambda info: {"triggered": True, "forced": True}
        try:
            import dataclasses
            r = apr.answer(dataclasses.replace(question(), doc_scope=["T.alpha.en", "T.beta.en"]), "en-en")
        finally:
            RT.trigger = orig
        rt = ((r.trace.get("lex6") or {}).get("lex6") or {}).get("router") or {}
        want = {"ok": ("ok", True), "deadline": ("fail", False), "invalid": ("fail", False)}[name]
        check("approach.router.%s" % name, r.error is None and rt.get("status") == want[0] and rt.get("use") == want[1]
              and len(fake.calls) == 1, (r.error, rt))
    del A6._STORE["s"]


def main():
    test_anchor()
    test_pack()
    test_docrank()
    test_router_parse()
    test_router_prompt()
    test_flags()
    with tempfile.TemporaryDirectory() as t:
        test_approach(Path(t))
    bad = [k for k, v in RESULTS.items() if not v["ok"]]
    print("%d/%d passed" % (len(RESULTS) - len(bad), len(RESULTS)))
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
