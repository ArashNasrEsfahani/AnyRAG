"""Offline tests of LEX5 (approaches/lex5: bundles.py, approach.py), HYBRID-intact, the lex4 fixes and the OpenRouter
selector options.

    .venv/Scripts/python.exe -m ragbench.tests.test_lex5

No network, no model: fake packages for the packers, the synthetic corpus + FakeOllama of test_hybrid /
test_baselines for the approach, and a fake selector client.
"""
import sys
import tempfile
import time
from pathlib import Path
from types import MappingProxyType

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[2]))

from ragbench.approaches import get_approach  # noqa: E402
from ragbench.approaches.lex5 import bundles as B  # noqa: E402
from ragbench.llm import ChatResult, Ledger  # noqa: E402
from ragbench.tests.test_baselines import FakeOllama  # noqa: E402
from ragbench.tests.test_hybrid import make_set, tcfg  # noqa: E402
from ragbench.tests.test_rerank import question  # noqa: E402

RESULTS = {}


def check(name, cond, detail=""):
    RESULTS[name] = {"ok": bool(cond), "detail": detail}
    print(("PASS " if cond else "FAIL ") + name + (" - " + str(detail)[:300] if detail and not cond else ""), flush=True)
    return cond


# --------------------------------------------------------------------------------------------------------------
# parser
# --------------------------------------------------------------------------------------------------------------

def test_parser():
    ok = B.parse("parts=2\nP1: 3+9 | 12\nP2: NONE\nEND", 24)
    check("parse.valid", ok["ok"] and ok["status"] == "ok" and ok["parts"] == [[(3, 9), (12,)], []], ok)
    an = B.parse("parts=1\nP1: NONE\nEND", 24)
    check("parse.all_none_is_valid_and_distinct", an["ok"] and an["status"] == "all_none", an)
    fa = B.parse("parts=۲\nP1: ۳+۹\nP2: هیچ\nEND", 24)
    check("parse.persian_digits_and_none", fa["ok"] and fa["parts"] == [[(3, 9)], []], fa)
    fence = B.parse("```\nparts=1\nP1: 4\nEND\n```", 24)
    check("parse.surrounding_fence", fence["ok"] and fence["parts"] == [[(4,)]], fence)
    bad = {
        "empty": "",
        "no_parts_line": "P1: 3\nEND",
        "bad_part_count": "parts=0\nEND",
        "overflow_parts": "parts=9\n" + "".join("P%d: 1\n" % i for i in range(1, 10)) + "END",
        "no_end": "parts=1\nP1: 3",
        "text_after_end": "parts=1\nP1: 3\nEND\nthanks",
        "part_count_mismatch": "parts=2\nP1: 3\nEND",
        "part_index": "parts=2\nP1: 3\nP3: 4\nEND",
        "bad_line": "parts=1\nThe first part: 3\nEND",
        "overflow_alternatives": "parts=1\nP1: 1 | 2 | 3\nEND",
        "empty_bundle": "parts=1\nP1: 3 |\nEND",
        "bad_ids": "parts=1\nP1: 3, 4\nEND",
        "id_out_of_range": "parts=1\nP1: 25\nEND",
        "duplicate_id_in_bundle": "parts=1\nP1: 3+3\nEND",
        "duplicate_alternative": "parts=1\nP1: 3+4 | 3+4\nEND",
    }
    for why, text in bad.items():
        r = B.parse(text, 24)
        check("parse.reject.%s" % why, not r["ok"] and r["status"] == "fail" and r["reason"] == why, r)
    tr = B.parse("parts=1\nP1: 3\nEND", 24, finish_reason="length")
    check("parse.reject.truncated", not tr["ok"] and tr["reason"] == "truncated", tr)
    zero = B.parse("parts=1\nP1: 0\nEND", 24)
    check("parse.reject.id_zero", not zero["ok"] and zero["reason"] == "id_out_of_range", zero)
    # v1.1 (FREEZE_ASTRA_R1): alternatives are compared as unordered passage sets
    for text in ("parts=1\nP1: 1+2 | 2+1\nEND", "parts=1\nP1: 3+9+5 | 5+3+9\nEND"):
        r = B.parse(text, 24)
        check("parse.reject.duplicate_alternative_as_set %s" % text.split("\n")[1], not r["ok"]
              and r["reason"] == "duplicate_alternative", r)
    keep = B.parse("parts=1\nP1: 9+3 | 3+4\nEND", 24)
    check("parse.distinct_sets_keep_presentation_order", keep["ok"] and keep["parts"] == [[(9, 3), (3, 4)]], keep)
    sub = B.parse("parts=1\nP1: 1+2 | 1\nEND", 24)
    check("parse.subset_is_a_different_set", sub["ok"] and sub["parts"] == [[(1, 2), (1,)]], sub)


# --------------------------------------------------------------------------------------------------------------
# packers on fake packages (cost = chars of the package text; no headers)
# --------------------------------------------------------------------------------------------------------------

def pkg(c, doc="d", a=None, b=None, size=1000, prefix=None):
    a = c * 10000 if a is None else a
    b = a + size if b is None else b
    prov = ([(doc, prefix[0], prefix[1], "prefix")] if prefix else []) + [(doc, a, b, "body")]
    return MappingProxyType({"chunk": c, "doc_id": doc, "text": "x" * (b - a), "provenance": tuple(prov)})


def flat_cost(p, n):
    return len(p["text"])


def test_packers():
    cost = B.Coster(flat_cost)
    a, b_, c, d, e, f, g, h = (pkg(i) for i in range(1, 9))
    # Astra R4 counterexample: P1 preferred = 3 packages (3,000), alternative = 1 package (1,000); P2, P3 = 2 each
    parts = [[(a, b_, c), (d,)], [(e, f)], [(g, h)]]
    rrf = {p["chunk"]: p["chunk"] for p in (a, b_, c, d, e, f, g, h)}
    gs, gi = B.pack_greedy(parts, cost, 6000)
    xs, xi = B.pack_exact(parts, rrf, cost, 6000)
    check("greedy.counterexample_covers_2", gi["covered"] == 2 and [p["chunk"] for p in gs] == [1, 2, 3, 5, 6], gi)
    check("exact.counterexample_covers_3", xi["covered"] == 3 and xi["chars"] == 5000
          and [p["chunk"] for p in xs] == [4, 5, 6, 7, 8] and xi["choice"] == [1, 0, 0], xi)
    check("exact.bounded_enumeration", xi["evaluated"] <= 3 ** 8, xi)
    # prefer the preferred alternative when both fit; then min chars
    xs2, xi2 = B.pack_exact([[(a, b_), (d,)]], rrf, cost, 6000)
    check("exact.prefers_first_alternative", [p["chunk"] for p in xs2] == [1, 2] and xi2["alt_rank_sum"] == 0, xi2)
    # a shared package is shown once and render order is part then bundle order
    xs3, _ = B.pack_exact([[(e, a)], [(a, b_)]], rrf, cost, 6000)
    check("exact.shared_once_part_order", [p["chunk"] for p in xs3] == [5, 1, 2], [p["chunk"] for p in xs3])
    # nothing fits -> empty
    big = pkg(20, size=7000)
    xs4, xi4 = B.pack_exact([[(big,)]], rrf, cost, 6000)
    check("exact.infeasible_part_uncovered", xs4 == [] and xi4["covered"] == 0, xi4)
    # 8 parts x 2 alternatives stays fast
    many = [[(pkg(100 + 2 * i, size=300),), (pkg(101 + 2 * i, size=200),)] for i in range(8)]
    t = time.perf_counter()
    _, xi5 = B.pack_exact(many, {}, cost, 2000)
    check("exact.worst_case_fast", time.perf_counter() - t < 5.0 and xi5["covered"] == 8, (time.perf_counter() - t, xi5))
    # fill: complete bundles only, then RRF
    small = pkg(40, size=400)
    fs, fi = B.fill([d], parts, [a, small, e], cost, 3500)
    check("fill.complete_bundles_then_rrf", [p["chunk"] for p in fs] == [4, 5, 6, 40] and fi["fill_bundles"] == 1
          and fi["fill_rrf"] == 1, ([p["chunk"] for p in fs], fi))
    # flat: individual packages, no completeness
    fl, fli = B.pack_flat(parts, cost, 2500)
    check("flat.first_id_per_part_first", [p["chunk"] for p in fl] == [1, 5], [p["chunk"] for p in fl])
    # representation with provenance: containment of the body alone is not enough when a prefix is displayed
    outer = pkg(30, a=0, b=2000)
    inner = pkg(31, a=500, b=900)
    inner_pref = pkg(32, a=500, b=900, prefix=(5000, 5050))
    check("dedupe.contained", B.canonical([outer, inner]) == [outer])
    check("dedupe.contained_but_prefix_not_shown_kept", B.canonical([outer, inner_pref]) == [outer, inner_pref])
    dup = pkg(33, a=0, b=2000)
    check("dedupe.exact_keeps_first", [p["chunk"] for p in B.canonical([outer, dup])] == [30])
    part_ov = pkg(34, a=1500, b=2500)
    check("dedupe.partial_overlap_kept", len(B.canonical([outer, part_ov])) == 2)
    other_doc = pkg(35, doc="e", a=500, b=900)
    check("dedupe.other_document_kept", len(B.canonical([outer, other_doc])) == 2)
    hi = B.hybrid_intact([a, outer, inner, b_, big, c], cost, 5000)
    check("intact.rrf_order_whole_dedupe", [p["chunk"] for p in hi] == [1, 30, 2, 3], [p["chunk"] for p in hi])
    try:
        a["chunk"] = 99
        check("packages.immutable", False)
    except TypeError:
        check("packages.immutable", True)


# --------------------------------------------------------------------------------------------------------------
# lex4 fixes
# --------------------------------------------------------------------------------------------------------------

def test_lex4_fixes():
    src = (HERE.parents[1] / "approaches" / "lex4" / "core.py").read_text(encoding="utf-8")
    code = "\n".join(l.split("#")[0] for l in src.splitlines())
    check("lex4.no_package_mutation", 'for p["chunk"] in' not in code)
    from ragbench.approaches.lex4 import spans as SP
    rec = {"answerable": True, "spans": [{"sid": "e1", "doc_id": "A", "canon": SP.canon("the value is 42 units")}],
           "requirements": [], "bundles": [["e1"]]}
    same = SP.retention(rec, [("A", "... The value is 42 units ...")])
    other = SP.retention(rec, [("B", "... The value is 42 units ...")])
    legacy = SP.retention(rec, ["... The value is 42 units ..."])
    check("retention.source_identity", same["full"] and not other["full"] and same["source_identity"], (same, other))
    check("retention.legacy_text_only_flagged", legacy["full"] and not legacy["source_identity"], legacy)


# --------------------------------------------------------------------------------------------------------------
# flags and the approach end to end (fake selector)
# --------------------------------------------------------------------------------------------------------------

def test_flags():
    from ragbench.run import parse_flags
    check("flags.default_names_no_variant", parse_flags(["selector=oss20b,view=full,top=24"], "LEX5") == {})
    check("flags.values", parse_flags(["selector=glm,top=40,view=head,greedy,no_fill"], "LEX5")
          == {"selector": "glm", "top": 40, "view": "head", "greedy": True, "no_fill": True})
    for bad in ("selector=gpt5", "view=snippet", "top=x", "rerank"):
        try:
            parse_flags([bad], "LEX5")
            check("flags.reject.%s" % bad, False)
        except SystemExit:
            check("flags.reject.%s" % bad, True)
    check("flags.hybrid_intact", parse_flags(["intact"], "HYBRID") == {"intact": True})


class FakeSelector:
    model_id, provider = "fake/selector", ["fake"]

    def __init__(self, reply, finish="stop", fail=None):
        self.reply, self.finish, self.fail, self.calls = reply, finish, fail, []

    def chat(self, msgs, **kw):
        self.calls.append(msgs)
        if self.fail:
            raise self.fail
        r = ChatResult(text=self.reply, prompt_tokens=1000, completion_tokens=50, wall_ms=10.0,
                       done_reason=self.finish, cost_usd=0.0)
        r.reasoning_tokens, r.provider = 20, "fake"
        return r

    def warm_deadline(self):
        return 200


def run_lex5(cfg, cs, flags, sel):
    fo = FakeOllama(cfg, Ledger(path=None))
    ap = get_approach("LEX5", cfg=cfg, client=fo, flags=flags)
    ap._sel = sel
    ap.build_index(cs)
    return ap, ap.answer(question(), "en-en")


def test_approach(tmp):
    from ragbench.approaches.lex5.bundles import parse
    from ragbench.llm_openrouter import DeadlineExceeded
    cs = make_set()
    cfg = tcfg(tmp / "idx")
    fo = FakeOllama(cfg, Ledger(path=None))
    hy = get_approach("HYBRID", cfg=cfg, client=fo, flags={"intact": True})
    hy.build_index(cs)
    base = hy.answer(question(), "en-en")
    lx = base.trace.get("lex5") or {}
    check("intact.variant_and_no_selector", hy.variant == "HYBRID+intact" and base.error is None
          and "selector" not in lx and base.llm_calls == 1, (base.error, lx.keys()))
    check("intact.exact_cost_no_cut", lx.get("cost_matches") and not lx.get("cut")
          and base.evidence_chars <= cfg.evidence["max_chars"], lx)
    sel = FakeSelector("parts=2\nP1: 2+3\nP2: 1 | 4\nEND")
    ap, rec = run_lex5(cfg, cs, None, sel)
    s = (rec.trace.get("lex5") or {}).get("selector") or {}
    check("lex5.ok", rec.error is None and s.get("status") == "ok" and not s.get("fallback"), (rec.error, s))
    check("lex5.one_selector_call_shared_answer", len(sel.calls) == 1 and rec.llm_calls == 1, rec.llm_calls)
    check("lex5.shows_top24_full", s.get("shown") == min(24, rec.trace["hybrid"]["fused"]) and s.get("view") == "full"
          and "Passages:" in sel.calls[0][0]["content"], s)
    lt = rec.trace["lex5"]
    top = lt["top_rrf"]
    check("lex5.selected_first_in_part_order", lt["shown_chunks"][:3] == [top[1], top[2], top[0]], (lt["shown_chunks"], top[:5]))
    check("lex5.exact_cost_budget", lt["cost_matches"] and not lt["cut"] and rec.evidence_chars <= 6000, lt)
    check("lex5.stage_times", all(k in lt for k in ("t_hybrid_ms", "t_package_ms", "t_pack_ms", "t_answer_ms"))
          and "t_select_ms" in s)
    # failures -> the fallback, identical to HYBRID-intact
    for why, fs in (("bad_grammar", FakeSelector("P1: 3, 4")), ("truncated", FakeSelector("parts=1\nP1: 2\nEND", "length")),
                    ("deadline", FakeSelector("", fail=DeadlineExceeded("deadline 3.0s exceeded")))):
        _, r2 = run_lex5(cfg, cs, None, fs)
        s2 = r2.trace["lex5"]["selector"]
        check("lex5.fallback.%s" % why, s2["fallback"] and r2.trace["lex5"]["shown_chunks"] == lx["shown_chunks"]
              and r2.error is None, (s2.get("reason"), r2.error))
    _, r3 = run_lex5(cfg, cs, None, FakeSelector("parts=1\nP1: NONE\nEND"))
    s3 = r3.trace["lex5"]["selector"]
    check("lex5.all_none_not_a_fallback_but_fills", s3["status"] == "all_none" and not s3["fallback"]
          and r3.trace["lex5"]["shown_chunks"] == lx["shown_chunks"], s3)
    _, r4 = run_lex5(cfg, cs, {"no_fill": True}, FakeSelector("parts=1\nP1: 2\nEND"))
    check("lex5.no_fill", r4.trace["lex5"]["shown_chunks"] == [r4.trace["lex5"]["top_rrf"][1]], r4.trace["lex5"]["shown_chunks"])
    ap5, r5 = run_lex5(cfg, cs, {"selector": "glm", "greedy": True}, FakeSelector("parts=1\nP1: 2\nEND"))
    check("lex5.variant_name", ap5.variant == "LEX5+greedy,selector=glm" and ap5.index_variant == "LEX5"
          and r5.trace["lex5"]["selector"]["packer"] == "greedy")
    check("lex5.prompt_domain_neutral", not any(w in sel.calls[0][0]["content"].split("Passages:")[0].lower()
                                                 for w in ("company", "revenue", "year", "report", "fiscal", "metric")))


def test_client_options():
    """The payload of the selector client: pinned provider without fallback, one cap incl. reasoning, low effort."""
    from ragbench.llm_openrouter import OpenRouterReader
    from ragbench.config import load_config
    cfg = load_config()
    c = OpenRouterReader(cfg, Ledger(path=None), "openai/gpt-oss-20b", supported=["reasoning", "temperature", "seed"],
                         provider="groq", attempts=1, deadline_s=3.0, max_tokens_total=1024)
    p = c._payload([{"role": "user", "content": "x"}], 200, None)
    check("client.pinned_no_fallback", p.get("provider") == {"order": ["groq"], "allow_fallbacks": False,
                                                             "require_parameters": True}, p.get("provider"))
    check("client.cap_includes_reasoning", p["max_tokens"] == 1024 and p["reasoning"]["effort"] == "low", p)
    check("client.one_attempt", c.or_retries == 0 and c.deadline_s == 3.0)
    d = OpenRouterReader(cfg, Ledger(path=None), "z-ai/glm-5.3-flash", supported=["reasoning"])
    pd = d._payload([{"role": "user", "content": "x"}], 200, None)
    check("client.default_unchanged", "provider" not in pd and pd["max_tokens"] == 3200 and d.or_retries == 6, pd)


def test_deadline_cancels():
    """A real total deadline: a local server that answers after 5 s; the call must return at ~0.5 s, and the server
    must see the connection dropped (the request was cancelled, not abandoned)."""
    import http.server
    import threading
    from ragbench.config import load_config
    from ragbench.llm_openrouter import DeadlineExceeded, OpenRouterReader
    seen = {"broken": None}

    class Slow(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            self.rfile.read(int(self.headers.get("Content-Length") or 0))
            time.sleep(2.0)
            try:
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b"{}")
                self.wfile.flush()
                seen["broken"] = False
            except OSError:
                seen["broken"] = True

        def log_message(self, *a):
            pass

    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Slow)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    c = OpenRouterReader(load_config(), Ledger(path=None), "x/y", attempts=1, deadline_s=0.5)
    t = time.perf_counter()
    try:
        c._post_deadline("http://127.0.0.1:%d/" % srv.server_address[1], {}, {"a": 1}, 0.5)
        check("deadline.raises", False)
    except DeadlineExceeded:
        dt = time.perf_counter() - t
        check("deadline.raises", True)
        check("deadline.caller_released_at_deadline", dt < 0.9, dt)
    time.sleep(2.5)
    check("deadline.server_side_observed", seen["broken"] is not None, seen)   # informational: True = write failed
    srv.shutdown()


def main():
    with tempfile.TemporaryDirectory(prefix="ragbench_lex5_", ignore_cleanup_errors=True) as d:
        tmp = Path(d)
        test_parser()
        test_packers()
        test_lex4_fixes()
        test_flags()
        test_approach(tmp)
        test_client_options()
        test_deadline_cancels()
    bad = [k for k, v in RESULTS.items() if not v["ok"]]
    print("\n%d checks, %d failed%s" % (len(RESULTS), len(bad), (": " + ", ".join(bad)) if bad else ""))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
