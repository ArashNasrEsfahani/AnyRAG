"""Offline tests for the multi-document P0 fixes (2026-09-24): frozen pools and --pool, answer formats decoupled from
retrieval and options, effective-request identity in call caches, identity-keyed records and resume, judge_multi,
check_grid, multidoc_analysis, verify_multidoc, and variants-v2 (query-ranked cards, safe expansion, card time
measured). No model calls: every client is a fake.

    .venv/Scripts/python.exe -m ragbench.tests.test_multidoc_pipeline
"""
import json
import re
import shutil
import sys
import tempfile
import time
import types
from pathlib import Path

from .. import run as RUN
from .. import variants as V
from ..config import load_config
from ..contracts import (ANSWER_PROMPT_VERSIONS, ANSWER_SYSTEM, MULTIDOC_FORMATS, MULTIDOC_NUM_PREDICT,
                         MULTIDOC_PROMPT_VERSIONS, MULTIDOC_SYSTEMS, AnswerResult, Approach, IndexHandle, Query,
                         Question, RetrievedItem, answer_format, answer_num_predict, answer_prompt_version,
                         build_answer_messages, pack_evidence)
from ..corpus.model import DocBuilder
from ..corpus.registry import CorpusSet
from ..llm import ChatResult, Ledger
from .. import pools as P
from ..tools import check_grid as CG
from ..tools import judge_final as JF
from ..tools import judge_multi as JM
from ..tools import multidoc_analysis as MA
from ..tools import verify_multidoc as VM

RESULTS = []
PKG = Path(__file__).resolve().parents[1]


def check(name, cond, detail=""):
    RESULTS.append((name, bool(cond)))
    print(("PASS " if cond else "FAIL ") + name + (" - " + str(detail)[:400] if detail != "" and not cond else ""),
          flush=True)
    return cond


def raises(fn, exc=Exception):
    try:
        fn()
    except exc:
        return True
    except Exception:  # noqa: BLE001
        return False
    return False


# --------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------

def words_text(tag, n_chars):
    out, i = [], 0
    while len(" ".join(out)) < n_chars:
        out.append("%s%d" % (tag, i))
        i += 1
    return " ".join(out)[:n_chars].rstrip()


def make_doc(doc_id, title, sections):
    """sections: [(local, title, [unit texts])]"""
    b = DocBuilder(doc_id, doc_id.split(".")[0], doc_id.rsplit(".", 1)[1], title, source="test",
                   meta={"fingerprint": "fp-" + doc_id})
    for local, st, units in sections:
        b.add_section(local, st)
        for i, t in enumerate(units):
            b.add_unit(local, "%s-u%d" % (local, i), "paragraph", t)
    return b.build()


class FakeClient:
    def __init__(self, reply="ok"):
        self.ledger = Ledger(path=None)
        self.calls = []
        self.reply = reply

    def options(self, num_predict=None, **kw):
        return {"num_predict": num_predict}

    def chat(self, messages, purpose=None, num_predict=None, **kw):
        self.calls.append({"messages": messages, "purpose": purpose, "num_predict": num_predict})
        text = self.reply(messages) if callable(self.reply) else self.reply
        return ChatResult(text=text, prompt_tokens=1, completion_tokens=1, wall_ms=1.0, call_id="c%d" % len(self.calls))


class _Stub(Approach):
    name = "STUB"

    def _build_index(self, corpus_set, index_dir):
        return {}

    def _answer(self, query, index):
        return AnswerResult(answer_text=self.generate_answer(query, None))


class _PoolStub(Approach):
    """For run.py tests: no model call; retrieves one item per document of the (pool-masked) scope."""
    name = "POOLSTUB"

    def _build_index(self, corpus_set, index_dir):
        return {}

    def _answer(self, query, index):
        items = [RetrievedItem(text=(index.corpus.doc(d).title or d) + " text", doc_id=d, kind="unit")
                 for d in (query.doc_scope if isinstance(query.doc_scope, list) else [])]
        return AnswerResult(answer_text="stub answer", evidence=self.pack(items, index), retrieved=items,
                            trace={"scope": query.doc_scope, "pool_id": query.pool_id})


def _q(qid="t1", qtype="lookup", answer_type="short", **kw):
    return Question(id=qid, corpus="C", question_fa="پرسش؟", question_en="Question?", answer_fa="", answer_en="",
                    answer_type=answer_type, gold_units={"fa": ["C.x.fa::p1"]}, doc_scope=["C.x.fa"], type=qtype, **kw)


# --------------------------------------------------------------------------
# 2. answer formats
# --------------------------------------------------------------------------

def test_formats():
    cfg = load_config()
    check("formats.legacy_versions_unchanged", ANSWER_PROMPT_VERSIONS == {"default": "answer-v2",
                                                                          "timeline": "answer-timeline-v1"})
    check("formats.old_questions_unchanged", answer_format(_q()) == "default"
          and answer_format(_q(qtype="timeline", answer_type="series")) == "timeline")
    check("formats.family_maps", answer_format(_q(task_family="comparison")) == "comparison"
          and answer_format(_q(task_family="aggregation")) == "comparison"
          and answer_format(_q(task_family="cross_year")) == "series"
          and answer_format(_q(task_family="single_fact")) == "single"
          and answer_format(_q(task_family="list")) == "list")
    check("formats.explicit_wins", answer_format(_q(task_family="comparison", answer_format="list")) == "list")
    check("formats.unknown_family_raises", raises(lambda: answer_format(_q(task_family="poetry")), ValueError))
    check("formats.validate_flags_bad_fields", "task_family" in _q(task_family="poetry").validate()
          and "answer_format" in _q(answer_format="haiku").validate())
    ev = types.SimpleNamespace(text="[1] X\nbody")
    for fmt in MULTIDOC_FORMATS:
        qq = Query("q", "Which?", "en", "en", "en-en", "D", "D_en", "corpus", answer_format=fmt)
        m = build_answer_messages(qq, ev, cfg)
        check("formats.%s_prompt" % fmt, m[0]["content"] == MULTIDOC_SYSTEMS[fmt].format(
            source_rule="Use only the numbered evidence passages in the user message.",
            abstain_rule="If the answer is not there, reply with exactly NOT_FOUND and nothing else.",
            answer_lang="English", abstain="NOT_FOUND") and answer_prompt_version(qq) == MULTIDOC_PROMPT_VERSIONS[fmt]
              and answer_num_predict(qq, cfg) == MULTIDOC_NUM_PREDICT[fmt], m[0]["content"][:200])
    check("formats.config_override", answer_num_predict("list", cfg.copy_with({"answer": {"list_num_predict": 1000}})) == 1000)
    check("formats.default_prompt_is_answer_v2", build_answer_messages(
        Query("q", "x", "en", "en", "en-en", "A", "A_en", ["A.x.en"]), ev, cfg)[0]["content"].startswith(
        ANSWER_SYSTEM.split("{")[0]))
    # Approach.answer: the question's format drives prompt and allowance
    client = FakeClient("1. a\n2. b")
    ap = _Stub(cfg=cfg, client=client)
    ap._indexes["C"] = IndexHandle("STUB", "C", Path("."), "fp", None, {}, {})
    rec = ap.answer(_q(task_family="list"), "fa-fa")
    call = client.calls[-1]
    check("formats.approach_list_question", call["num_predict"] == 900 and "numbered list" in call["messages"][0]["content"]
          and rec.meta == {"answer_format": "list", "answer_prompt": "answer-list-v1", "answer_num_predict": 900}, rec.meta)
    old = _q()
    check("formats.old_to_dict_shape", set(old.to_dict()) == {"id", "corpus", "question_fa", "question_en", "answer_fa",
                                                               "answer_en", "answer_type", "gold_units", "doc_scope",
                                                               "type", "answer_value", "answer_unit", "meta"})
    check("formats.new_fields_round_trip", Question.from_dict(_q(task_family="list", pool_id="P").to_dict()).task_family
          == "list")
    check("sig.v1_unchanged_by_family", RUN.question_sig(_q()) == RUN.question_sig(_q(task_family="list")))
    check("sig.v2_sees_family_format_pool", len({RUN.question_sig_v2(_q()), RUN.question_sig_v2(_q(task_family="list")),
                                                 RUN.question_sig_v2(_q(answer_format="series")),
                                                 RUN.question_sig_v2(_q(pool_id="P1"))}) == 4)


# --------------------------------------------------------------------------
# 7. variants-v2 and the decoupling of list rule / allowance from expansion
# --------------------------------------------------------------------------

def expansion_corpus():
    a_units = [words_text("alpha%d_" % i, 800) for i in range(5)]
    A = make_doc("C.aaa.fa", "Report A", [("a0", "Intro", [words_text("intro", 300)]),
                                          ("a1", "Five key findings", a_units)])
    B = make_doc("C.bbb.fa", "Report B", [("b1", "Market", [words_text("beta", 1700)]),
                                          ("b2", "Prices", [words_text("delta", 1400)])])
    Cc = make_doc("C.ccc.fa", "Report C", [("c1", "Costs", [words_text("gamma", 1500)])])
    return CorpusSet("C", [A, B, Cc]), A, B, Cc


def expansion_items(A, B, Cc):
    sec = A.section("C.aaa.fa::a1")
    u0, u1 = [A.unit(u) for u in sec.unit_ids[:2]]
    it = lambda d, u: RetrievedItem(text=u.text, doc_id=d.id, section_id=u.section_id, unit_ids=[u.id], kind="unit",
                                    span=list(u.span))    # noqa: E731
    return [it(A, u0), it(A, u1), it(B, B.units[0]), it(Cc, Cc.units[0])]


def test_variants(tmp):
    cfg = load_config()
    cs, A, B, Cc = expansion_corpus()
    items = expansion_items(A, B, Cc)
    q = "What are the five key findings?"
    new, dec = V.expand_sections(items, cs, q, 6000)
    before = pack_evidence(items, cs, 6000, cfg)
    after = pack_evidence(new, cs, 6000, cfg)
    check("expand.v1_applies_and_loses_a_document", dec["applied"] and {i.doc_id for i in before.items} == {
        "C.aaa.fa", "C.bbb.fa", "C.ccc.fa"} and "C.ccc.fa" not in {i.doc_id for i in after.items},
          (dec, [i.doc_id for i in after.items]))
    pk = lambda its: pack_evidence(its, cs, 6000, cfg)    # noqa: E731
    new2, dec2 = V.expand_sections_safe(items, cs, q, 6000, pk)
    check("expand.v2_abandons_when_a_document_is_lost", new2 == items and not dec2["applied"] and dec2.get("abandoned")
          and dec2["lost_docs"] == ["C.ccc.fa"], dec2)
    small = [items[0], items[1], items[2]]
    new3, dec3 = V.expand_sections_safe(small, cs, q, 6000, pk)
    check("expand.v2_keeps_a_safe_expansion", dec3["applied"] and any(i.kind == "section" for i in new3), dec3)
    u2 = B.units[1]
    req = RetrievedItem(text=u2.text, doc_id=B.id, section_id=u2.section_id, unit_ids=[u2.id], kind="unit",
                        span=list(u2.span), meta={"requirement": "req-b2"})
    new4, dec4 = V.expand_sections_safe(small + [req], cs, q, 6000, pk)
    _, dec4v1 = V.expand_sections(small + [req], cs, q, 6000)
    check("expand.v2_reserved_requirement_guard", dec4v1["applied"] and not dec4["applied"]
          and dec4["lost_requirements"] == ["req-b2"] and dec4["lost_docs"] == [] and new4 == small + [req], dec4)

    # final_pack: list rule + allowance only under v1 and only for the old formats
    def fp(version, fmt):
        client = FakeClient()
        ap = _Stub(cfg=cfg, client=client, flags={"section_expand": True})
        ap.variants_version = version
        idx = IndexHandle("STUB", "C", Path("."), "fp", cs, {}, {})
        qq = Query("q", q, "en", "fa", "en-fa", "C", "C", ["C.aaa.fa", "C.bbb.fa", "C.ccc.fa"], answer_format=fmt)
        ap._begin_variants(qq, idx)
        ev = ap.final_pack(small, idx, qq)
        ap.generate_answer(qq, ev)
        return ap.client.list_answer, client.calls[-1]["num_predict"], client.calls[-1]["messages"][0]["content"]
    la, n, _ = fp("variants-v1", "default")
    check("decouple.v1_default_keeps_old_coupling", la is True and n == V.LIST_NUM_PREDICT, (la, n))
    for fmt in ("single", "comparison", "list"):
        la, n, sysmsg = fp("variants-v1", fmt)
        check("decouple.v1_%s_format_is_never_rewritten" % fmt, la is False and n == MULTIDOC_NUM_PREDICT[fmt]
              and V.LIST_ANSWER_RULE not in sysmsg, (la, n))
    la, n, _ = fp("variants-v2", "default")
    check("decouple.v2_never_couples", la is False and n == 256, (la, n))

    # cards: v2 ranks by the question and logs its choice; v1 unchanged (first 12 in corpus order)
    docs = [make_doc("C.d%02d.fa" % i, "Report %02d" % i, [("s", "Part", [("Term%d " % i) * 30 + words_text("x", 100)])])
            for i in range(15)]
    docs[13] = make_doc("C.d13.fa", "Steel exports report", [("s", "Steel exports", [("steel exports tonnage " * 40)])])
    text, sel = V.card_shortlist(docs, "How did steel exports change?")
    check("cards.v2_query_ranked", sel["shown"][0]["doc_id"] == "C.d13.fa" and sel["n_shown"] == 12
          and sel["n_candidates"] == 15 and "... and 3 more" in text, sel["shown"][:3])
    text2, sel2 = V.card_shortlist(docs, "How did steel exports change?", routed_ids=["C.d02.fa"])
    check("cards.v2_routed_first", [s["doc_id"] for s in sel2["shown"][:2]] == ["C.d02.fa", "C.d13.fa"], sel2["shown"][:3])
    cs2 = CorpusSet("C", docs)
    idx2 = IndexHandle("STUB", "C", Path("."), "fp", cs2, {}, {})
    ap = _Stub(cfg=cfg, client=FakeClient(), flags={"doc_card": True})
    qq = Query("q", "How did steel exports change?", "en", "fa", "en-fa", "C", "C", [d.id for d in docs])
    ap._begin_variants(qq, idx2)
    check("cards.v1_unchanged", ap.client.card == V.card_context([V.document_card(d) for d in docs])
          and "card_selection" not in ap._variant_trace)
    ap.variants_version = "variants-v2"
    ap._begin_variants(qq, idx2)
    check("cards.v2_logged_in_trace", ap._variant_trace["card_selection"]["shown"][0]["doc_id"] == "C.d13.fa"
          and ap._variant_trace["version"] == "variants-v2")

    # card construction inside the measured query time under v2 (before the timer under v1)
    orig_card, orig_doc_card = V._card_of, V.document_card
    try:
        V._card_of = lambda d: (time.sleep(0.25), "card")[1]
        V.document_card = lambda d: (time.sleep(0.25), "card")[1]
        lat = {}
        for version in ("variants-v1", "variants-v2"):
            ap = _Stub(cfg=cfg, client=FakeClient(), flags={"doc_card": True})
            ap.variants_version = version
            ap._indexes["C"] = IndexHandle("STUB", "C", Path("."), "fp", CorpusSet("C", [docs[0]]), {}, {})
            qd = Question(id="c1", corpus="C", question_fa="x", question_en="x", answer_fa="", answer_en="",
                          answer_type="short", gold_units={"fa": ["C.d00.fa::s-u0"]}, doc_scope=["C.d00.fa"], type="lookup")
            lat[version] = ap.answer(qd, "fa-fa").latency_ms
    finally:
        V._card_of, V.document_card = orig_card, orig_doc_card
    check("cards.v2_timed_v1_untimed", lat["variants-v1"] < 150 <= lat["variants-v2"], lat)


# --------------------------------------------------------------------------
# 3. effective request identity
# --------------------------------------------------------------------------

def test_effective_request(tmp):
    cfg = load_config()
    inner = FakeClient()
    vc = V.VariantClient(inner)
    msgs = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]
    check("eff.none_without_injection", vc.effective_request_sig(msgs, "plan", 50) is None
          and vc.effective_request_sig(msgs, "answer", 50) is None)
    vc.card = "CARD A"
    s1 = vc.effective_request_sig(msgs, "plan", 50)
    vc.card = "CARD B"
    s2 = vc.effective_request_sig(msgs, "plan", 50)
    check("eff.card_changes_sig", s1 and s2 and s1 != s2 and vc.effective_request_sig(msgs, "answer", 50) is None)
    vc.card, vc.list_answer = None, True
    m2, n2, ch = vc.effective_request(msgs, "answer", 256)
    check("eff.list_rule_and_allowance", ch and n2 == V.LIST_NUM_PREDICT and V.LIST_ANSWER_RULE in m2[0]["content"]
          and vc.effective_request_sig(msgs, "answer", 256))
    vc.chat(msgs, purpose="answer", num_predict=256)
    check("eff.chat_sends_the_effective_request", inner.calls[-1]["num_predict"] == V.LIST_NUM_PREDICT
          and inner.calls[-1]["messages"] == m2)
    vc.list_answer = False
    vc.chat(msgs, purpose="plan")
    check("eff.chat_unchanged_without_injection", inner.calls[-1]["messages"] == msgs and inner.calls[-1]["num_predict"] is None)
    # the AGENTIC call cache (LEX uses the same scheme): the effective request enters the key
    from ..approaches.agentic.approach import AgenticPipeline, _Session
    fake = FakeClient()
    ap = AgenticPipeline(cfg=cfg, client=fake, options={"call_cache_path": str(tmp / "calls.sqlite")})
    idx = types.SimpleNamespace(corpus=None, payload={"store_root": str(tmp)})
    qq = Query("q", "x", "en", "en", "en-en", "D", "D_en", "corpus")
    sess = lambda: _Session(ap, qq, idx)     # noqa: E731
    sess().chat("plan", msgs, num_predict=50)
    sess().chat("plan", msgs, num_predict=50)
    check("eff.plain_cache_hit", len(fake.calls) == 1)
    ap.client = V.VariantClient(fake)
    sess().chat("plan", msgs, num_predict=50)
    check("eff.no_injection_keeps_the_old_key", len(fake.calls) == 1)
    ap.client.card = "CARD A"
    sess().chat("plan", msgs, num_predict=50)
    sess().chat("plan", msgs, num_predict=50)
    ap.client.card = "CARD B"
    sess().chat("plan", msgs, num_predict=50)
    check("eff.revised_card_never_replays", len(fake.calls) == 3 and fake.calls[1]["messages"][-1]["content"].startswith(
        "CARD A") and fake.calls[2]["messages"][-1]["content"].startswith("CARD B"), len(fake.calls))
    lex_src = (PKG / "approaches" / "lex" / "approach.py").read_text(encoding="utf-8")
    check("eff.lex_cache_uses_it", 'payload["effective_request"] = eff' in lex_src)


# --------------------------------------------------------------------------
# 1. pools
# --------------------------------------------------------------------------

def test_pools(tmp):
    d1 = make_doc("C.alpha.fa", "Alpha", [("s", "S", ["alpha text one"])])
    d2 = make_doc("C.beta.fa", "Beta", [("s", "S", ["beta text two"])])
    d3 = make_doc("C.gamma.fa", "Gamma", [("s", "S", ["gamma text three"])])
    cs = CorpusSet("C", [d1, d2, d3])
    check("pool.refuses_gold_derived", raises(lambda: P.build_manifest("p", "C", ["C.alpha.fa"], {"C": cs}, "gold"),
                                              P.PoolError))
    check("pool.refuses_unknown_doc", raises(lambda: P.build_manifest("p", "C", ["C.zzz.fa"], {"C": cs}, "manual"),
                                             P.PoolError))
    m = P.build_manifest("P1", "C", ["C.beta.fa", "C.alpha.fa"], {"C": cs}, "manual", "test", created="t0")
    path = tmp / "pools" / "P1.json"
    P.write_manifest(path, m)
    P.write_manifest(path, m)                                   # same content: no-op
    m2 = P.build_manifest("P1", "C", ["C.alpha.fa"], {"C": cs}, "manual", "test", created="t0")
    check("pool.frozen_refuses_other_content", raises(lambda: P.write_manifest(path, m2), P.PoolError))
    pool = P.load_pool(path)
    check("pool.sorted_ids_and_mask", pool.doc_ids() == ["C.alpha.fa", "C.beta.fa"]
          and pool.mask("corpus", "C") == ["C.alpha.fa", "C.beta.fa"]
          and pool.mask(["C.gamma.fa", "C.beta.fa"], "C") == ["C.beta.fa"] and pool.mask("corpus", "D_en") == [])
    check("pool.verify_ok", pool.verify({"C": cs}) == [])
    d2b = make_doc("C.beta.fa", "Beta", [("s", "S", ["beta text CHANGED"])])
    d2b.meta["fingerprint"] = "re-extracted"
    probs = pool.verify({"C": CorpusSet("C", [d1, d2b, d3])})
    check("pool.verify_detects_changes", any("text_sha1" in p for p in probs) and any("fingerprint" in p for p in probs),
          probs)
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw["docs"].append({"doc_id": "C.gamma.fa", "set": "C"})
    (tmp / "pools" / "bad.json").write_text(json.dumps(raw), encoding="utf-8")
    check("pool.tamper_detected", raises(lambda: P.load_pool(tmp / "pools" / "bad.json"), P.PoolError))
    # make_query masks the corpus scope; the index is untouched
    ap = _Stub(cfg=load_config(), client=FakeClient())
    ap.pool = pool
    qd = Question(id="m1", corpus="C", question_fa="x", question_en="x", answer_fa="", answer_en="", answer_type="short",
                  gold_units={"fa": ["C.alpha.fa::s-u0"]}, doc_scope="corpus", type="lookup")
    qq = ap.make_query(qd, "fa-fa")
    check("pool.make_query_mask", qq.doc_scope == ["C.alpha.fa", "C.beta.fa"] and qq.pool_id == "P1", qq)
    check("pool.real_tiny_manifest", P.load_pool(P.pool_root(load_config()) / "C-tiny-test.json").pool_id == "C-tiny-test")


# --------------------------------------------------------------------------
# 1 + 3 + 4. run.py: refusals, pool mask, identity records, resume, stale refusal
# --------------------------------------------------------------------------

TINY = ["C.dataak_annual-report_1403.fa", "C.dataak_annual-report_1404.fa", "C.irancell_sustainability-report_1399.fa"]


def md_question(qid, fam="comparison", pools=("C-tiny-test",), **kw):
    d = {"schema": "multidoc-q-v1", "id": qid, "corpus": "C", "question_fa": "پرسش %s؟" % qid,
         "question_en": "Question %s?" % qid, "answer_fa": "x", "answer_en": "x", "answer_type": "multi",
         "gold_units": {"fa": [TINY[0] + "::u1", TINY[2] + "::u2"]}, "doc_scope": "corpus", "type": "multidoc",
         "task_family": fam, "pool_id": list(pools), "meta": {"experiment": "multidoc", "experiment_id": "md-test"}}
    d.update(kw)
    return d


def test_run(tmp):
    cfg = load_config().copy_with({"paths": {"index_root": str(tmp / "idx")}})
    qf = tmp / "dev_md.jsonl"
    qf.write_text("".join(json.dumps(q, ensure_ascii=False) + "\n" for q in [
        md_question("MD-1"), md_question("MD-2", fam="list"), md_question("MD-3", pools=("other-pool",))]), encoding="utf-8")
    out = tmp / "out"
    base = ["--approach-class", "ragbench.tests.test_multidoc_pipeline:_PoolStub", "--questions", str(qf), "--corpus", "C",
            "--settings", "fa-fa", "--out", str(out), "--no-health", "--no-warmup"]
    logs = []

    def run(extra):
        logs.clear()
        return RUN.main(base + extra, cfg=cfg, log=logs.append)
    check("run.refuses_docs_from_questions", run(["--docs-from-questions", "--pool", "C-tiny-test"]) == 2
          and any("derives the search pool" in l for l in logs), logs[-1:])
    check("run.refuses_docs", run(["--docs", "dataak_annual-report_1403", "--pool", "C-tiny-test"]) == 2)
    check("run.requires_pool", run([]) == 2 and any("needs --pool" in l for l in logs), logs[-1:])
    rc = run(["--pool", "C-tiny-test", "--dry-run"])
    js = [json.loads(l[len("DRYRUN-JSON "):]) for l in logs if l.startswith("DRYRUN-JSON ")]
    check("run.dry_identity_plan", rc == 0 and js and js[0]["identity_mode"] and js[0]["planned"] == 2
          and js[0]["pool"]["pool_id"] == "C-tiny-test" and js[0]["variants_version"] == "variants-v2"
          and len(js[0]["rids"]) == 2 and js[0]["by_format"] == {"comparison": {"planned": 1, "todo": 1},
                                                                  "list": {"planned": 1, "todo": 1}}, (rc, js, logs[-3:]))
    rc = run(["--pool", "C-tiny-test"])
    recs = [json.loads(l) for l in open(out / "runs_POOLSTUB.jsonl", encoding="utf-8")]
    ok = rc == 0 and len(recs) == 2 and all(r["trace"]["scope"] == TINY and r["trace"]["pool_id"] == "C-tiny-test"
                                            for r in recs)
    ident = recs[0]["meta"]["identity"] if recs else {}
    check("run.pool_masks_scope", ok, (rc, logs[-3:]))
    check("run.identity_on_records", set(ident) >= {"experiment_id", "question_id", "pool_id", "method", "options", "reader",
                                                    "replicate", "setting", "rid", "code_sig", "question_sig", "pool_sha1"}
          and ident["experiment_id"] == "md-test" and ident["options"]["variants_version"] == "variants-v2"
          and recs[0]["meta"]["pool"]["pool_id"] == "C-tiny-test", ident)
    idx_dirs = [p.name for p in (tmp / "idx" / "POOLSTUB").iterdir()]
    full_fp = __import__("ragbench.corpus.registry", fromlist=["load_set"]).load_set("C", cfg).fingerprint()
    check("run.index_over_full_set", len(idx_dirs) == 1 and json.loads(
        (tmp / "idx" / "POOLSTUB" / idx_dirs[0] / "build_stats.json").read_text(encoding="utf-8"))["fingerprint"]
          == _PoolStub(cfg=cfg, client=FakeClient()).index_fingerprint(__import__(
              "ragbench.corpus.registry", fromlist=["load_set"]).load_set("C", cfg)) and bool(full_fp), idx_dirs)
    rc = run(["--pool", "C-tiny-test"])
    n_after = len(open(out / "runs_POOLSTUB.jsonl", encoding="utf-8").read().splitlines())
    check("run.resume_by_rid", rc == 0 and n_after == 2 and any("0 to run" in l for l in logs), logs[:2])
    rc = run(["--pool", "C-tiny-test", "--replicate", "1"])
    n_rep = len(open(out / "runs_POOLSTUB.jsonl", encoding="utf-8").read().splitlines())
    check("run.replicate_is_a_new_identity", rc == 0 and n_rep == 4)
    rc = run(["--pool", "C-tiny-test", "--variants-version", "variants-v1"])
    check("run.variants_version_change_refused", rc == 2 and any("variants-v2" in l and "semantics" in l for l in logs),
          logs[-3:])
    lines = open(out / "runs_POOLSTUB.jsonl", encoding="utf-8").read().splitlines()
    r0 = json.loads(lines[0])
    r0["meta"]["identity"]["code_sig"] = "stale-code"
    (out / "runs_POOLSTUB.jsonl").write_text("\n".join([json.dumps(r0, ensure_ascii=False)] + lines[1:]) + "\n",
                                             encoding="utf-8")
    rc = run(["--pool", "C-tiny-test"])
    check("run.stale_record_refused", rc == 2 and any("REFUSED" in l and "code_sig" in l for l in logs), logs[-2:])
    # a pool whose frozen document changed is refused
    raw = json.loads((P.pool_root(load_config()) / "C-tiny-test.json").read_text(encoding="utf-8"))
    raw["pool_id"] = "C-tiny-tampered"
    raw["docs"][0]["text_sha1"] = "0" * 40
    raw["manifest_sha1"] = P.manifest_sha1(raw)
    tp = tmp / "C-tiny-tampered.json"
    tp.write_text(json.dumps(raw), encoding="utf-8")
    qf2 = tmp / "dev_md2.jsonl"
    qf2.write_text(json.dumps(md_question("MD-9", pools=("C-tiny-tampered",)), ensure_ascii=False) + "\n", encoding="utf-8")
    logs.clear()
    rc = RUN.main(["--approach-class", "ragbench.tests.test_multidoc_pipeline:_PoolStub", "--questions", str(qf2),
                   "--corpus", "C", "--settings", "fa-fa", "--out", str(tmp / "out2"), "--no-health", "--no-warmup",
                   "--pool", str(tp), "--dry-run"], cfg=cfg, log=logs.append)
    check("run.changed_pool_document_refused", rc == 2 and any("POOL:" in l and "text_sha1" in l for l in logs), logs[-3:])
    check("run.dependency_sha1_covers_shared_modules", {p.name for p in RUN.dependency_files()} >= {
        "contracts.py", "variants.py", "pools.py", "baselines.py", "model.py"} and "run.py" not in
          {p.name for p in RUN.dependency_files()})


# --------------------------------------------------------------------------
# 6. verify_multidoc
# --------------------------------------------------------------------------

def verify_fixture():
    qa = "Alpha revenue in 1402 was 12345 billion rials according to the audited statements of the company"
    qb = "Beta revenue in 1402 was 23456 billion rials according to the audited statements of the company"
    da = make_doc("C.alpha.fa", "Alpha", [("s", "S", [qa])])
    db = make_doc("C.beta.fa", "Beta", [("s", "S", [qb])])
    dc = make_doc("C.gamma.fa", "Gamma", [("s", "S", ["unrelated gamma text"])])
    cs = CorpusSet("C", [da, db, dc])

    def span(d):
        u = d.units[0]
        return {"doc_id": d.id, "unit_id": u.id, "start": u.span[0], "end": u.span[1], "quote": u.text}
    q = {"schema": "multidoc-q-v1", "id": "V1", "corpus": "C", "type": "multidoc", "answer_type": "multi",
         "question_fa": "درآمد آلفا و بتا در ۱۴۰۲ چقدر فرق داشت؟", "question_en": "How far apart were the two companies' sales?",
         "answer_fa": "x", "answer_en": "x", "gold_units": {"fa": [da.units[0].id, db.units[0].id]}, "doc_scope": "corpus",
         "task_family": "comparison", "pool_id": ["PV"], "source_companies": ["alpha", "beta"], "source_block": "b1",
         "information_need_id": "n1",
         "content_units": [{"claim_id": "c1", "company": "alpha", "period": "1402", "metric": "revenue", "value": "12345",
                            "unit": "billion rials", "role": "operand", "source_spans": [span(da)]},
                           {"claim_id": "c2", "company": "beta", "period": "1402", "metric": "revenue", "value": "23456",
                            "unit": "billion rials", "role": "operand", "source_spans": [span(db)]},
                           {"claim_id": "c3", "company": "beta-alpha", "period": "1402", "metric": "revenue difference",
                            "value": "11111", "unit": "billion rials", "role": "derived",
                            "source_spans": [span(da), span(db)]}],
         "accepted_evidence_sets": [["C.alpha.fa", "C.beta.fa"]],
         "multidoc_necessity": {"declared": True, "indispensable_docs": ["C.alpha.fa", "C.beta.fa"],
                                "single_doc_complete": False, "audit": "each report read alone"},
         "meta": {"experiment": "multidoc", "experiment_id": "md-test"}}
    pools = {"PV": P.Pool(P.build_manifest("PV", "C", ["C.alpha.fa", "C.beta.fa", "C.gamma.fa"], {"C": cs}, "manual",
                                           created="t"))}
    return q, cs, pools


def test_verify():
    q, cs, pools = verify_fixture()
    rows = VM.check_set([q], cs, pools)
    check("verify.good_question_passes", rows[0]["ok"] and not rows[0]["flags"], rows[0])
    bad = json.loads(json.dumps(q))
    bad["multidoc_necessity"] = {"declared": False}
    e, _ = VM.check_question(bad, cs, pools)
    check("verify.necessity_required", any("declared" in x for x in e) and any("indispensable" in x for x in e), e)
    one = json.loads(json.dumps(q))
    one["content_units"] = one["content_units"][:1]
    one["accepted_evidence_sets"] = [["C.alpha.fa"]]
    e, _ = VM.check_question(one, cs, pools)
    check("verify.single_document_refused", any("1 document" in x for x in e) and any("one document" in x for x in e), e)
    gp = {"PV": P.Pool(P.build_manifest("PV", "C", ["C.alpha.fa", "C.beta.fa"], {"C": cs}, "manual", created="t"))}
    e, _ = VM.check_question(q, cs, gp)
    check("verify.gold_equal_pool_refused", any("gold-derived" in x for x in e), e)
    e, _ = VM.check_question(dict(q, doc_scope=["C.alpha.fa", "C.beta.fa"]), cs, pools)
    check("verify.listed_scope_refused", any("doc_scope" in x for x in e), e)
    mis = json.loads(json.dumps(q))
    mis["content_units"][0]["source_spans"][0]["quote"] = "something else"
    e, _ = VM.check_question(mis, cs, pools)
    check("verify.quote_checked_against_corpus", any("quote" in x for x in e), e)
    leak = dict(q, question_en="Alpha revenue in 1402 was 12345 billion rials according to the audited statements?")
    _, f = VM.check_question(leak, cs, pools)
    check("verify.copied_span_and_leakage_flags", any(x.startswith("copied_span") for x in f)
          and any(x.startswith("answer_leakage") for x in f), f)
    q2 = dict(json.loads(json.dumps(q)), id="V2", source_block="b2", question_en="What separated the two sales totals?",
              question_fa="فاصله فروش دو شرکت چقدر بود؟")
    rows = VM.check_set([q, q2], cs, pools)
    check("verify.need_across_blocks_refused", all(any("spans source blocks" in x for x in r["errors"]) for r in rows))
    q3 = dict(json.loads(json.dumps(q)), id="V3")
    rows = VM.check_set([q, q3], cs, pools)
    check("verify.duplicate_text_refused", all(any("duplicate question text" in x for x in r["errors"]) for r in rows))
    un = {"schema": "multidoc-q-v1", "id": "U1", "corpus": "C", "type": "unanswerable", "answer_type": "unanswerable",
          "question_fa": "سود گاما در ۱۳۹۰؟", "question_en": "Gamma profit in 1390?", "answer_fa": "NOT_FOUND",
          "answer_en": "NOT_FOUND", "gold_units": {}, "doc_scope": "corpus", "task_family": "unanswerable",
          "pool_id": ["PV"], "source_block": "b9", "information_need_id": "n9", "meta": {"absent_search": "grep x"}}
    e, _ = VM.check_question(un, cs, pools)
    check("verify.unanswerable_ok", not e, e)
    e, _ = VM.check_question(dict(un, meta={}), cs, pools)
    check("verify.unanswerable_needs_absent_search", any("absent_search" in x for x in e), e)


# --------------------------------------------------------------------------
# 4. judge_multi
# --------------------------------------------------------------------------

def scripted_judge(messages):
    """Grades GOOD answers correct, BAD answers wrong; replies in the rubric's own format."""
    sysm, user = messages[0]["content"], messages[1]["content"]
    good = "Candidate answer: GOOD" in user or "Candidate answer:\nGOOD" in user
    if sysm == JF.JUDGE_V2_SYSTEM:
        return json.dumps({"correct": good, "reason": "scripted"})
    ids = sorted(set(re.findall(r'"claim_id": "(\w+)"', user)))
    if "WRONGYEAR" in user:
        return json.dumps({"claims": [{"text": "a", "unit": ids[0], "status": "wrong_binding"},
                                      {"text": "b", "unit": ids[1], "status": "correct"}],
                           "units": dict({i: "missing" for i in ids}, **{ids[0]: "wrong", ids[1]: "correct"})})
    return json.dumps({"claims": [{"text": i, "unit": i, "status": "correct" if good else "wrong_value"} for i in ids],
                       "units": {i: "correct" if good else "wrong" for i in ids}, "reason": "scripted"})


def mk_record(qd, method, answer, retrieved=(), evidence=(), error=None, reader="chatA", replicate=0, setting="fa-fa",
              experiment="md-test", pool="P1", vv="variants-v2"):
    cfg = load_config().copy_with({"models": {"chat": reader, "chat_digest": None}})
    ident = RUN.record_identity(experiment, qd["id"], pool, method, {}, vv, cfg, replicate, setting)
    ident.update(code_sig="c", question_sig="q", pool_sha1="p")
    return {"question_id": qd["id"], "approach": method, "setting": setting, "d_lang": setting[-2:],
            "answer_text": answer, "error": error, "trace": {}, "retrieved_units": [{"doc_id": d} for d in retrieved],
            "evidence_items": [{"doc_id": d} for d in evidence], "meta": {"identity": ident}}


def grid_questions():
    out = {}
    for qid, fam, block, n in (("G1", "comparison", "b1", 2), ("G2", "list", "b1", 3), ("G3", "series", "b2", 2),
                               ("G4", "single_fact", "b3", 1)):
        units = [{"claim_id": "c%d" % i, "company": "co%d" % i, "period": "1402", "metric": "m", "value": str(100 + i),
                  "unit": "u", "role": "operand" if fam == "comparison" else "item",
                  "source_spans": [{"doc_id": "C.d%d.fa" % i, "quote": "value %d" % (100 + i)}]} for i in range(n)]
        out[qid] = {"schema": "multidoc-q-v1", "id": qid, "corpus": "C", "type": "multidoc", "answer_type": "multi"
                    if n > 1 else "short", "question_fa": "پرسش %s" % qid, "question_en": "Q %s" % qid,
                    "answer_fa": "gold", "answer_en": "gold", "gold_units": {"fa": ["C.d0.fa::u"]}, "doc_scope": "corpus",
                    "task_family": fam, "pool_id": ["P1"], "source_block": block, "information_need_id": "n" + qid,
                    "content_units": units, "accepted_evidence_sets": [["C.d%d.fa" % i for i in range(n)]]}
    return out


def test_judge(tmp):
    qs = grid_questions()
    j = FakeClient(scripted_judge)
    recs = {r["meta"]["identity"]["rid"]: r for r in (
        mk_record(qs["G1"], "M1", "GOOD"), mk_record(qs["G4"], "M1", "GOOD"), mk_record(qs["G1"], "M2", "WRONGYEAR"),
        mk_record(qs["G2"], "M2", "NOT_FOUND"))}
    s = JM.judge_records(recs, qs, j, "J", tmp / "judge", log=lambda x: None)
    vs = {(v["question_id"], v["method"]): v for v in JM.read_jsonl(tmp / "judge" / "verdicts_J.jsonl")}
    check("judge.all_judged", s["judged"] == 4 and s["failures"] == 0 and s["rule"] == 1, s)
    g1 = vs[("G1", "M1")]
    check("judge.multi_rubric_scores", g1["rubric"] == "judge-multi-v1" and g1["strict"] == 1 and g1["recall"] == 1.0
          and g1["f1"] == 1.0 and g1["rid"] and g1["answer_sha1"] and g1["rubric_sha1"] and g1["packet_sha1"], g1)
    single_call = [c for c in j.calls if c["messages"][0]["content"] == JF.JUDGE_V2_SYSTEM]
    check("judge.single_uses_judge_final_v3_unchanged", vs[("G4", "M1")]["rubric"] == "judge-final-v3" and single_call
          and single_call[0]["messages"] == JF.messages(Question.from_dict(qs["G4"]), recs[[k for k, r in recs.items()
                                                                                             if r["question_id"] == "G4"][0]]))
    wy = vs[("G1", "M2")]
    check("judge.wrong_binding_earns_nothing", wy["strict"] == 0 and wy["recall"] == 0.5 and wy["precision"] == 0.5
          and wy["wrong"] == 1, wy)
    ab = vs[("G2", "M2")]
    check("judge.false_abstention_by_rule", ab["source"] == "rule" and ab["strict"] == 0 and ab["false_abstention"], ab)
    multi_user = [c for c in j.calls if c["messages"][0]["content"] == JM.MULTI_SYSTEM][0]["messages"][1]["content"]
    check("judge.packet_is_method_independent", "M1" not in multi_user and "retrieved" not in multi_user
          and '"claim_id": "c0"' in multi_user and "value 100" in multi_user)
    n0 = len(j.calls)
    s2 = JM.judge_records(recs, qs, j, "J", tmp / "judge", log=lambda x: None)
    check("judge.resume_skips_done", s2["skipped_done"] == 4 and len(j.calls) == n0, s2)
    # failures are recorded apart and retried only on request
    bad = FakeClient(lambda m: json.dumps({"claims": [], "units": {"c0": "correct"}}))       # omits c1
    r3 = mk_record(qs["G3"], "M1", "GOOD")
    recs3 = {r3["meta"]["identity"]["rid"]: r3}
    s3 = JM.judge_records(recs3, qs, bad, "J", tmp / "judge", log=lambda x: None, on_malformed="failure",
                          max_attempts=1)                       # the legacy rule, kept as an option
    fails = JM.read_jsonl(tmp / "judge" / "judge_failures_J.jsonl")
    check("judge.incomplete_reply_is_a_failure_not_a_verdict", s3["failures"] == 1 and s3["judged"] == 0
          and "omits" in fails[-1]["error"] and not [v for v in JM.read_jsonl(tmp / "judge" / "verdicts_J.jsonl")
                                                     if v["question_id"] == "G3"], fails[-1:])
    s4 = JM.judge_records(recs3, qs, bad, "J", tmp / "judge", log=lambda x: None, on_malformed="failure",
                          max_attempts=1)
    check("judge.failure_not_silently_retried", s4["skipped_failed"] == 1 and len(bad.calls) == 1, s4)
    s5 = JM.judge_records(recs3, qs, FakeClient(scripted_judge), "J", tmp / "judge", retry_failures=True,
                          log=lambda x: None)
    check("judge.explicit_retry", s5["judged"] == 1, s5)
    r1b = mk_record(qs["G1"], "M1", "GOOD, revised")
    s6 = JM.judge_records({r1b["meta"]["identity"]["rid"]: r1b}, qs, FakeClient(scripted_judge), "J", tmp / "judge",
                          log=lambda x: None)
    check("judge.changed_answer_is_rejudged", s6["judged"] == 1, s6)
    # parser / scorer units
    p, why = JM.parse_multi('{"claims": [{"text": "x", "unit": "c9", "status": "correct"}], "units": {"c0": "correct"}}',
                            ["c0"])
    check("judge.parse_rejects_unknown_unit", p is None and "unknown unit" in why, why)
    pk = {"units": [{"claim_id": "a", "role": "operand"}, {"claim_id": "b", "role": "operand"},
                    {"claim_id": "d", "role": "derived"}]}
    sc = JM.score_multi({"units": {"a": "correct", "b": "correct", "d": "missing"},
                         "claims": [{"status": "correct"}, {"status": "correct"}, {"status": "duplicate"},
                                    {"status": "unsupported"}]}, pk)
    check("judge.score_multi", sc["strict"] == 0 and abs(sc["recall"] - 2 / 3) < 1e-3 and abs(sc["precision"] - 2 / 3) < 1e-3
          and sc["duplicates"] == 1 and sc["unsupported"] == 1 and sc["omissions"] == 1
          and sc["recall_by_role"] == {"derived": 0.0, "operand": 1.0}, sc)


def test_judge_policy(tmp):
    """FREEZE_ASTRA_R1 fixes a + b: endpoint pin recorded and keyed; bounded identical retries; malformed ->
    needs_adjudication (terminal, never 0); call errors only -> judge_failures; blinded adjudication resolves."""
    qs = grid_questions()
    d = tmp / "judge_policy"
    r1 = mk_record(qs["G1"], "M1", "GOOD")
    recs = {r1["meta"]["identity"]["rid"]: r1}
    # 1. malformed twice, then valid: identical messages each time, verdict on the 3rd attempt, pin recorded
    seq = iter(["not json", '{"claims": [], "units": {"c0": "correct"}}'])
    flaky = FakeClient(lambda m: next(seq, None) or scripted_judge(m))
    s = JM.judge_records(recs, qs, flaky, "J", d, log=lambda x: None, provider_pin="prov/x")
    v = JM.read_jsonl(d / "verdicts_J.jsonl")
    check("judge_policy.bounded_identical_retries", s["judged"] == 1 and len(flaky.calls) == 3
          and all(c["messages"] == flaky.calls[0]["messages"] for c in flaky.calls) and s["retried_records"] == 1, s)
    check("judge_policy.pin_recorded", v and v[0]["judge_provider_pin"] == "prov/x" and v[0]["judge_attempts"] == 3, v)
    # 2. the same record under ANOTHER pin is not 'done' (pin is part of the key); unpinned old verdicts never reused
    s2 = JM.judge_records(recs, qs, FakeClient(scripted_judge), "J", d, log=lambda x: None, provider_pin=None)
    check("judge_policy.pin_in_key", s2["judged"] == 1 and s2["skipped_done"] == 0, s2)
    s2b = JM.judge_records(recs, qs, FakeClient(scripted_judge), "J", d, log=lambda x: None, provider_pin="prov/x")
    check("judge_policy.same_pin_resumes", s2b["skipped_done"] == 1, s2b)
    # 3. always malformed -> needs_adjudication after exactly 3 calls; terminal; never a verdict, never 0
    r2 = mk_record(qs["G3"], "M1", "GOOD")
    recs2 = {r2["meta"]["identity"]["rid"]: r2}
    bad = FakeClient(lambda m: "garbage")
    s3 = JM.judge_records(recs2, qs, bad, "J", d, log=lambda x: None, provider_pin="prov/x")
    na = JM.read_jsonl(d / "needs_adjudication_J.jsonl")
    check("judge_policy.malformed_to_adjudication", s3["needs_adjudication"] == 1 and len(bad.calls) == 3
          and na and na[-1]["n_attempts"] == 3 and na[-1]["status"] == "needs_adjudication"
          and not [x for x in JM.read_jsonl(d / "verdicts_J.jsonl") if x["question_id"] == "G3"], s3)
    s4 = JM.judge_records(recs2, qs, FakeClient(scripted_judge), "J", d, log=lambda x: None, provider_pin="prov/x",
                          retry_failures=True)
    check("judge_policy.adjudication_is_terminal", s4["skipped_needs_adjudication"] == 1 and s4["judged"] == 0, s4)
    fv = JM.final_verdicts(d, "J", "prov/x")
    rid2 = r2["meta"]["identity"]["rid"]
    check("judge_policy.unjudged_is_bounds_not_zero", fv[rid2]["status"] == "needs_adjudication"
          and fv[rid2]["strict"] is None and (fv[rid2]["strict_lower"], fv[rid2]["strict_upper"]) == (0, 1), fv[rid2])
    with open(d / "adjudications_J.jsonl", "a", encoding="utf-8") as fh:
        fh.write(json.dumps({"rid": rid2, "answer_sha1": na[-1]["answer_sha1"], "strict": 1, "adjudicator": "t",
                             "blinded": True}) + "\n")
    fv = JM.final_verdicts(d, "J", "prov/x")
    check("judge_policy.blinded_adjudication_resolves", fv[rid2]["status"] == "adjudicated" and fv[rid2]["strict"] == 1
          and fv[rid2]["machine_strict"] is None, fv[rid2])
    check("judge_policy.first_verdict_is_final", fv[r1["meta"]["identity"]["rid"]]["status"] == "judged")
    # 4. call errors only -> infrastructure failure (judge_failures), retryable on request
    class Down(FakeClient):
        def chat(self, messages, purpose=None, num_predict=None, **kw):
            self.calls.append(1)
            raise RuntimeError("HTTP 503")
    r3 = mk_record(qs["G2"], "M1", "GOOD")
    recs3 = {r3["meta"]["identity"]["rid"]: r3}
    dn = Down()
    s5 = JM.judge_records(recs3, qs, dn, "J", d, log=lambda x: None, provider_pin="prov/x")
    check("judge_policy.call_errors_are_failures", s5["failures"] == 1 and s5["needs_adjudication"] == 0
          and len(dn.calls) == 3, s5)
    s6 = JM.judge_records(recs3, qs, FakeClient(scripted_judge), "J", d, log=lambda x: None, provider_pin="prov/x",
                          retry_failures=True)
    check("judge_policy.failure_retry_on_request", s6["judged"] == 1, s6)
    check("judge_policy.served_match", JM._served_matches("deepinfra/bf16", "DeepInfra") is True
          and JM._served_matches("deepinfra/bf16", "Groq") is False)


# --------------------------------------------------------------------------
# 5. check_grid + multidoc_analysis
# --------------------------------------------------------------------------

def test_grid_and_analysis(tmp):
    qs = grid_questions()
    design = {"experiment_id": "md-test", "questions": "unused.jsonl", "methods": ["M1", "M2"], "readers": ["chatA"],
              "replicates": [0], "settings": {"C": ["fa-fa"]}, "variants_version": "variants-v2", "judge": "J",
              "primary_contrasts": [["H1", "M1", "M2", "strict"], ["H2", "M1", "M2", "route_all_evidence"]],
              "secondary_contrasts": [["S1", "M1", "M2", "f1"]], "bootstrap": {"n": 300, "seed": 0}}
    dpath = CG.freeze_design(design, tmp / "design.json")
    design = CG.load_design(dpath)
    check("grid.design_frozen", raises(lambda: CG.freeze_design(dict(design, methods=["M1"]), dpath), ValueError))
    all_docs = lambda qd: [s["doc_id"] for u in qd["content_units"] for s in u["source_spans"]]    # noqa: E731
    recs = []
    for qid, qd in qs.items():
        good = qid != "G4"
        recs.append(mk_record(qd, "M1", "GOOD" if good else "BAD", retrieved=all_docs(qd), evidence=all_docs(qd)))
        recs.append(mk_record(qd, "M2", "BAD", retrieved=all_docs(qd), evidence=all_docs(qd)[:1]))
    rep = CG.grid_gaps(design, qs, recs, [])
    check("grid.no_verdicts_incomplete", not rep["complete"] and len(rep["missing_verdict"]) == 8, CG.summary_lines(rep))
    JM.judge_records({r["meta"]["identity"]["rid"]: r for r in recs}, qs, FakeClient(scripted_judge), "J", tmp / "j",
                     log=lambda x: None)
    verdicts = JM.read_jsonl(tmp / "j" / "verdicts_J.jsonl")
    rep = CG.grid_gaps(design, qs, recs, verdicts)
    check("grid.complete", rep["complete"] and rep["expected"] == 8, CG.summary_lines(rep))
    rep2 = CG.grid_gaps(design, qs, recs[:-1], verdicts)
    check("grid.missing_record_listed", not rep2["complete"] and len(rep2["missing_record"]) == 1)
    errd = [dict(r) for r in recs]
    errd[0]["error"] = "boom"
    check("grid.error_record_listed", len(CG.grid_gaps(design, qs, errd, verdicts)["error_record"]) == 1)
    fl = [v for v in verdicts if v["question_id"] == "G1" and v["method"] == "M1"]
    rest = [v for v in verdicts if v not in fl]
    rep3 = CG.grid_gaps(design, qs, recs, rest, failures=[dict(fl[0], error="x")])
    check("grid.judge_failure_listed", len(rep3["judge_failure"]) == 1 and not rep3["complete"])
    stale = [dict(r) for r in recs]
    stale[0] = dict(stale[0], answer_text="GOOD but changed")
    check("grid.verdict_bound_to_answer_hash", len(CG.grid_gaps(design, qs, stale, verdicts)["missing_verdict"]) == 1)
    res, _ = MA.analyse(design, qs, recs[:-1], verdicts)
    check("analysis.refuses_incomplete_grid", res is None)
    res, rep = MA.analyse(design, qs, recs, verdicts)
    h1 = {r["id"]: r for r in res["primary"]}
    check("analysis.paired_estimate", abs(h1["H1"]["estimate"] - 0.75) < 1e-9 and h1["H1"]["n_blocks"] == 3
          and abs(h1["H1"]["p"] - 0.5) < 1e-9 and "exact" in h1["H1"]["p_how"], h1["H1"])
    check("analysis.holm", h1["H1"]["p_holm"] >= h1["H1"]["p"] and "p_holm" in h1["H2"])
    check("analysis.routing_outside_approaches", abs(h1["H2"]["estimate"] - 0.75) < 1e-9, h1["H2"])   # G4: 1 doc, covered by both
    fam = res["families"]
    check("analysis.per_family", fam["comparison"]["M1"]["strict"] == 1.0 and fam["single_fact"]["M1"]["strict"] == 0.0
          and fam["ALL"]["M2"]["route_all_retrieved"] == 1.0 and fam["ALL"]["M2"]["route_all_evidence"] == 0.25, fam["ALL"])
    check("analysis.renders", "Primary contrasts" in MA.render(res))


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass
    tmp = Path(tempfile.mkdtemp(prefix="ragbench_multidoc_"))
    try:
        for name, fn in (("formats", test_formats), ("variants", lambda: test_variants(tmp)),
                         ("effective", lambda: test_effective_request(tmp)), ("pools", lambda: test_pools(tmp)),
                         ("verify", test_verify), ("judge", lambda: test_judge(tmp)),
                         ("judge_policy", lambda: test_judge_policy(tmp)),
                         ("grid", lambda: test_grid_and_analysis(tmp)), ("run", lambda: test_run(tmp))):
            try:
                fn()
            except Exception as e:  # noqa: BLE001
                import traceback
                traceback.print_exc()
                check("%s.raised" % name, False, "%s: %s" % (type(e).__name__, e))
    finally:
        failed = [n for n, ok in RESULTS if not ok]
        print("\n%d checks, %d failed%s" % (len(RESULTS), len(failed), (": " + ", ".join(failed)) if failed else ""))
        shutil.rmtree(tmp, ignore_errors=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
