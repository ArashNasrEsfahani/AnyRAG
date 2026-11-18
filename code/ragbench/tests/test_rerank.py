"""Offline tests of HYBRID+rerank (ragbench/retrieval/rerank.py + approaches/hybrid.py flag rerank).

    .venv/Scripts/python.exe -m ragbench.tests.test_rerank

No model, no torch, no network: a fake cross-encoder (keyword counting) replaces bge-reranker-v2-m3; the synthetic
corpus, fake embedder and fake chat come from test_hybrid / test_baselines. Nothing is written outside a temp dir.
"""
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[2]))

from ragbench.approaches import HYBRID_FLAGS, get_approach  # noqa: E402
from ragbench.contracts import Question  # noqa: E402
from ragbench.llm import Ledger  # noqa: E402
from ragbench.retrieval import hybrid as H  # noqa: E402
from ragbench.retrieval import rerank as R  # noqa: E402
from ragbench.tests.test_baselines import FakeOllama  # noqa: E402
from ragbench.tests.test_hybrid import make_set, tcfg  # noqa: E402

RESULTS = {}


def check(name, cond, detail=""):
    RESULTS[name] = {"ok": bool(cond), "detail": detail}
    print(("PASS " if cond else "FAIL ") + name + (" - " + str(detail)[:300] if detail and not cond else ""), flush=True)
    return cond


class FakeReranker:
    """score = occurrences of `word` in the text (ties -> 0). Records every call."""

    def __init__(self, word="maintenance", device="fake"):
        self.word, self.device, self.calls, self.loaded, self.load_ms = word.lower(), device, [], 0, 0.0

    def load(self):
        self.loaded += 1
        return self

    def score(self, question, texts):
        self.calls.append((question, list(texts)))
        return [float(t.lower().count(self.word)) for t in texts], {"model": "fake", "device": self.device,
                                                                     "fp16": False, "n": len(texts), "ms": 0.1}


class _H:
    def __init__(self, i, text):
        self.i, self.text = i, text


# --------------------------------------------------------------------------------------------------------------

def test_rerank_hits():
    hits = [_H(0, "a"), _H(1, "b maintenance"), _H(2, "c"), _H(3, "maintenance maintenance"), _H(4, "d maintenance")]
    fr = FakeReranker()
    out, info, scores = R.rerank_hits("q?", hits, fr)
    check("order.by_score_then_input_rank", [h.i for h in out] == [3, 1, 4, 0, 2], [h.i for h in out])
    check("order.scores_aligned", scores == [2.0, 1.0, 1.0, 0.0, 0.0], scores)
    check("order.all_scored_once", len(fr.calls) == 1 and fr.calls[0][1] == [h.text for h in hits])
    check("info.fields", info["candidates_reranked"] == 5 and info["kept"] == 5 and info["device"] == "fake"
          and info["order"][0] == [3, 2.0] and "ms" in info, info)
    out2, info2, _ = R.rerank_hits("q?", hits, FakeReranker(), keep=2)
    check("keep.truncates_after_scoring", [h.i for h in out2] == [3, 1] and info2["candidates_reranked"] == 5
          and info2["kept"] == 2)
    a = [h.i for h in R.rerank_hits("q?", hits, FakeReranker())[0]]
    b = [h.i for h in R.rerank_hits("q?", hits, FakeReranker())[0]]
    check("determinism", a == b)
    e, ei, es = R.rerank_hits("q?", [], FakeReranker())
    check("empty", e == [] and es == [] and ei["candidates_reranked"] == 0)

    class Bad(FakeReranker):
        def score(self, q, texts):
            return [1.0], {}
    try:
        R.rerank_hits("q?", hits, Bad())
        check("misaligned_scores_rejected", False)
    except ValueError:
        check("misaligned_scores_rejected", True)


def test_flags():
    from ragbench.run import parse_flags
    check("flags.registered", "rerank" in HYBRID_FLAGS and parse_flags(["rerank"], "HYBRID") == {"rerank": True}
          and parse_flags(["rerank,blocks"], "HYBRID") == {"rerank": True, "blocks": True})
    for ap in ("LEX", "LEX3", "AGENTIC", "AGENTIC5", "B2", "B3"):
        try:
            parse_flags(["rerank"], ap)
            check("flags.not_for_%s" % ap, False)
        except SystemExit:
            check("flags.not_for_%s" % ap, True)
    # no method other than approaches/hybrid.py may reach the reranker
    pkg = HERE.parents[1]
    users = sorted(p.relative_to(pkg).as_posix() for p in pkg.rglob("*.py")
                   if "tests" not in p.parts and p.name != "rerank.py"
                   and ("retrieval import rerank" in p.read_text(encoding="utf-8", errors="replace")
                        or "retrieval.rerank" in p.read_text(encoding="utf-8", errors="replace")))
    check("isolation.only_hybrid_imports_rerank", users == ["approaches/hybrid.py"], users)


def question():
    return Question(id="t-steel", corpus="T", question_fa="", question_en="How many tonnes of steel did Alpha Corp produce?",
                    answer_fa="", answer_en="4.2 million tonnes", answer_type="numeric",
                    gold_units={"en": ["T.alpha.en::ops#p0"]}, doc_scope=["T.alpha.en"], type="lookup", answer_value=4.2e6)


def run_variant(cfg, cs, flags, reranker=None):
    fo = FakeOllama(cfg, Ledger(path=None))
    ap = get_approach("HYBRID", cfg=cfg, client=fo, flags=flags)
    if reranker is not None:
        ap.reranker = reranker
    ap.build_index(cs)
    return ap, ap.answer(question(), "en-en"), fo


def test_approach(tmp):
    cs = make_set()
    cfg = tcfg(tmp / "idx")
    base_ap, base, _ = run_variant(cfg, cs, None)
    fr = FakeReranker("maintenance")
    ap, rec, fo = run_variant(cfg, cs, {"rerank": True}, fr)
    tr = rec.trace or {}
    ri = tr.get("rerank", {})
    check("approach.no_error", rec.error is None and base.error is None, (rec.error, base.error))
    check("approach.variant_name", ap.variant == "HYBRID+rerank" and rec.approach == "HYBRID+rerank")
    check("approach.index_neutral", ap.index_variant == "HYBRID" and ap.index_dir(cs) == base_ap.index_dir(cs)
          and rec.index_fingerprint == base.index_fingerprint)
    check("approach.reranker_loaded_at_index", fr.loaded >= 1 and len(fr.calls) == 1)
    n_in_scope = tr["hybrid"]["chunks_in_scope"]
    want = min(R.RERANK_DEFAULTS["candidates"], tr["hybrid"]["fused"])
    check("approach.candidates_reranked", ri.get("candidates_reranked") == want == len(fr.calls[0][1])
          and tr["params"]["rerank_candidates"] == R.RERANK_DEFAULTS["candidates"], (ri.get("candidates_reranked"), want, n_in_scope))
    check("approach.trace_fields", ri.get("device") == "fake" and "ms" in ri and "load_ms" in ri
          and tr["params"]["rerank"] is True, ri)
    # the reranked order: every retrieved item carries its rerank score, non-increasing; ties keep the RRF order
    sc = [it["score"] for it in rec.retrieved_units]
    rrf_ranks = [it["meta"]["rrf_rank"] for it in rec.retrieved_units]
    ok = all(sc[i] >= sc[i + 1] for i in range(len(sc) - 1)) and all(
        rrf_ranks[i] < rrf_ranks[i + 1] for i in range(len(sc) - 1) if sc[i] == sc[i + 1])
    check("approach.retrieved_in_reranked_order", ok, list(zip(sc, rrf_ranks))[:10])
    top_text = rec.retrieved_units[0]["text"].lower()
    best = max(t.lower().count("maintenance") for t in fr.calls[0][1])
    check("approach.reranker_pick_first", best > 0 and top_text.count("maintenance") == best, rec.retrieved_units[0]["meta"])
    check("approach.kept_at_most_hybrid_candidates", len(rec.retrieved_units) <= 40)
    check("approach.same_budget_and_single_answer_call", 0 < rec.evidence_chars <= cfg.evidence["max_chars"]
          and rec.llm_calls == 1 == base.llm_calls and rec.embed_calls == 1, (rec.evidence_chars, rec.llm_calls))
    check("approach.evidence_in_scope", all(it["doc_id"] == "T.alpha.en" for it in rec.evidence_items))
    # same candidate set as HYBRID asked for 50 (the reranker only reorders + truncates)
    hx = H.HybridIndex.load(cs, cfg)
    with fo.ledger.scope(phase="smoke"):
        fused50 = hx.search(question().question_en, doc_ids=["T.alpha.en"], k=R.RERANK_DEFAULTS["candidates"], depth=40,
                            client=fo)
    check("approach.reranks_the_hybrid_candidates", fr.calls[0][1] == [h.text for h in fused50])
    # determinism: a second run gives the same order and evidence
    ap2, rec2, _ = run_variant(cfg, cs, {"rerank": True}, FakeReranker("maintenance"))
    check("approach.deterministic", [u["meta"]["chunk"] for u in rec2.retrieved_units]
          == [u["meta"]["chunk"] for u in rec.retrieved_units] and rec2.evidence_units == rec.evidence_units)
    # an identity reranker (all ties) reproduces plain HYBRID's order
    ap3, rec3, _ = run_variant(cfg, cs, {"rerank": True}, FakeReranker("zzzz-nothing"))
    check("approach.all_ties_equals_hybrid", [u["meta"]["chunk"] for u in rec3.retrieved_units]
          == [u["meta"]["chunk"] for u in base.retrieved_units] and rec3.evidence_units == base.evidence_units)
    # plain HYBRID never touches a reranker
    base_ap.reranker = FakeReranker()
    base_ap.answer(question(), "en-en")
    check("approach.plain_hybrid_no_rerank", base_ap.reranker.calls == [] and "rerank" not in (base.trace or {}))
    # blocks combine with rerank
    ap4, rec4, _ = run_variant(cfg, cs, {"rerank": True, "blocks": True}, FakeReranker("maintenance"))
    check("approach.rerank_blocks", rec4.error is None and ap4.variant == "HYBRID+blocks,rerank"
          and rec4.trace.get("blocks", {}).get("n_blocks") == len(rec4.evidence_items)
          and rec4.evidence_chars <= cfg.evidence["max_chars"], rec4.error)


def main():
    with tempfile.TemporaryDirectory(prefix="ragbench_rerank_", ignore_cleanup_errors=True) as d:
        tmp = Path(d)
        test_rerank_hits()
        test_flags()
        test_approach(tmp)
    bad = [k for k, v in RESULTS.items() if not v["ok"]]
    print("\n%d checks, %d failed%s" % (len(RESULTS), len(bad), (": " + ", ".join(bad)) if bad else ""))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
