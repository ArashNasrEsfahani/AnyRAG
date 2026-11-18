"""Offline smoke test for eval/ and run.py: no model calls, no OpenRouter credit.

    .venv/Scripts/python.exe -m ragbench.eval.test_eval

Everything is written to a temporary directory (indexes, runs, ledger, judge cache and a
scratch judge budget file), never to work/.
"""
import json
import random
import shutil
import sys
import tempfile
from pathlib import Path

from ..config import load_config
from ..contracts import Approach, AnswerResult, Question, RetrievedItem, RunRecord, save_questions, load_questions
from ..corpus.model import swap_lang
from ..corpus.registry import load_set
from ..util import append_jsonl, atomic_write_json, read_jsonl
from . import aggregate as AGG
from . import judge as J
from . import keyfacts as KF
from . import metrics as M

QUESTION_DIR = Path(__file__).resolve().parents[1] / "data" / "questions"
KEY_FACT_FILES = ["test_core.jsonl", "dev_core.jsonl", "pilot.jsonl"]
KEY_FACT_TYPES = ("locate", "multi-section", "cross-year")

RESULTS = []


def check(name, cond, detail=""):
    RESULTS.append((name, bool(cond)))
    print(("PASS " if cond else "FAIL ") + name + (" - " + str(detail) if detail != "" else ""), flush=True)
    return cond


# --------------------------------------------------------------------------
# a retrieval approach that never calls a model (for run.py plumbing)
# --------------------------------------------------------------------------

class FakeRetrieval(Approach):
    """Takes the first 3 units with a digit in the scoped documents, packs them with the
    shared budget and 'answers' with the first number of the evidence. No model calls."""
    name = "FAKE"

    def _build_index(self, corpus_set, index_dir):
        atomic_write_json(Path(index_dir) / "fake_index.json", {"docs": corpus_set.doc_ids()})
        return {"docs": corpus_set.doc_ids()}

    def _answer(self, query, index):
        items = []
        for doc in self.docs_in_scope(query, index):
            for u in doc.units:
                if any(ch.isdigit() for ch in u.text) and len(items) < 3:
                    items.append(RetrievedItem(text=u.text, doc_id=doc.id, section_id=u.section_id, unit_ids=[u.id],
                                               kind="unit", rank=len(items), span=list(u.span)))
        ev = self.pack(items, index)
        nums = M.extract_numbers(ev.text.split("\n", 1)[1] if "\n" in ev.text else ev.text)
        return AnswerResult(answer_text=nums[0].text if nums else "NOT_FOUND", evidence=ev, retrieved=items,
                            trace={"fake": True})


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------

def test_metrics():
    cases = [
        ("264,926 t of the sample product were made in 2024 [2].", 264926, "t", True),
        ("about 265,000 tonnes were made", 264926, "t", True),
        ("درآمد ۱۲٫۵ میلیارد ریال بود", 12.5, "میلیارد ریال", True),
        ("درآمد 12,500 میلیون ریال", 12.5, "billion rials", True),
        ("1.25 میلیارد تومان", 12.5, "میلیارد ریال", True),
        ("۱۲٫۵ همت", 12.5, "هزار میلیارد تومان", True),
        ("revenue was $12.5bn in 2020", 12.5, "billion USD", True),
        ("۳/۵ درصد", 3.5, "%", True),
        ("fell 3.5%", -3.5, "%", True),
        ("12.3 million users", 12345678, "users", True),
        ("12 million users", 12345678, "users", False),
        ("13 million", 12.5, "million", False),
        ("در ۱۴۰۱/۰۵/۱۲ رشد ۵ درصد", 12, "%", False),
        ("NOT_FOUND", 5, None, False),
        ("12.5", 13, None, False),
    ]
    bad = [(t, v, u, M.numeric_match(t, v, u)) for t, v, u, exp in cases if M.numeric_match(t, v, u)["match"] != exp]
    check("metrics.numeric_match", not bad, "%d cases, bad=%s" % (len(cases), bad))
    r = M.numeric_match("revenue was 12.5B", 12500000000, None, gold_text="12.5 billion")
    check("metrics.numeric_unit_from_gold_text", r["match"], r)
    cfg_strict = M.MatchConfig(rel_tol=0.0, allow_rounding=False)
    check("metrics.numeric_tolerance_configurable", not M.numeric_match("about 265,000 t", 264926, "t", cfg=cfg_strict)["match"])
    s1 = M.string_match("دیجی‌كالا [1].", ["دیجی کالا", "Digikala"])
    s2 = M.string_match("The largest is Digikala.", ["دیجی‌کالا", "Digikala"])
    s3 = M.string_match("Snapp", ["دیجی‌کالا", "Digikala"])
    check("metrics.string_match", s1["exact"] and s2["contains"] and not s3["match"], (s1, s2, s3))
    s4 = M.string_match("دیجیکالا", ["دیجی‌کالا", "Digikala"])                  # ZWNJ vs no space
    s5 = M.string_match("بزرگترین فروشگاه دیجی کالا است", ["دیجیکالا"])          # space vs no space, inside a sentence
    s6 = M.string_match("Snappfood", ["Snapp"])                                  # no match inside a longer word
    check("metrics.string_match_persian_spacing", s4["exact"] and s5["contains"] and not s6["match"], (s4, s5, s6))
    check("metrics.scale_words_need_word_start", M.parse_unit("column").scale == 1.0 and M.parse_unit("autumn").scale == 1.0
          and M.parse_unit("mn rials").scale == 1e6 and not M.numeric_match("5 million", 5, "column")["match"]
          and not M.numeric_match("5m3", 5000000, "m3")["match"],
          (M.parse_unit("column"), M.numeric_match("5m3", 5000000, "m3")))
    abst = [M.detect_abstention(x) for x in ["NOT_FOUND", "not_found.", "NOT_FOUND [2]", "The evidence does not mention this.",
                                             "اطلاعاتی در این باره یافت نشد."]]
    nonabst = [M.detect_abstention(x) for x in ["Revenue was 5 billion rials", "Digikala", ""]]
    check("metrics.abstention", all(abst) and not any(nonabst), (abst, nonabst))
    rec = {"evidence_units": ["d::a", "d::b"], "retrieved_units": [{"unit_ids": ["d::a", "d::b", "d::c"]}]}
    er = M.evidence_recall(rec, ["d::b", "d::c", "d::z"])
    check("metrics.evidence_recall", er["ev_hit"] == 1.0 and abs(er["ev_recall"] - 1 / 3) < 1e-9
          and abs(er["ret_recall"] - 2 / 3) < 1e-9, er)
    check("metrics.norm_text", M.norm_text("۱٬۲۳۴٫۵ ميليون ريال، [3] Test.") == "1234.5 میلیون ریال test",
          M.norm_text("۱٬۲۳۴٫۵ ميليون ريال، [3] Test."))


# --------------------------------------------------------------------------
# judge (no network: _call is replaced)
# --------------------------------------------------------------------------

def _q(i, at="numeric", corpus="A"):
    if at == "unanswerable":
        return Question(id="q%02d" % i, corpus=corpus, question_fa="پرسش %d؟" % i, question_en="Question %d?" % i,
                        answer_fa="", answer_en="", answer_type="unanswerable", gold_units={"fa": [], "en": []},
                        doc_scope="corpus", type="unanswerable")
    if at == "short":
        return Question(id="q%02d" % i, corpus=corpus, question_fa="بزرگترین فروشگاه؟ %d" % i, question_en="Largest store? %d" % i,
                        answer_fa="دیجی‌کالا", answer_en="Digikala", answer_type="short",
                        gold_units={"fa": ["A.x.fa::s#u%d" % i], "en": ["A.x.en::s#u%d" % i]},
                        doc_scope=["A.x.fa"], type="lookup")
    return Question(id="q%02d" % i, corpus=corpus, question_fa="درآمد سال %d چقدر بود؟" % i,
                    question_en="What was revenue in year %d?" % i, answer_fa="%d میلیارد ریال" % (100 + i),
                    answer_en="%d billion rials" % (100 + i), answer_type="numeric",
                    gold_units={"fa": ["A.x.fa::s#u%d" % i, "A.x.fa::s#v%d" % i], "en": ["A.x.en::s#u%d" % i, "A.x.en::s#v%d" % i]},
                    doc_scope=["A.x.fa"], type="lookup", answer_value=100 + i, answer_unit="billion rials")


def test_judge(cfg, tmp):
    q = _q(1)
    rec = {"question_id": "q01", "approach": "B2", "setting": "fa-en", "q_lang": "fa", "d_lang": "en",
           "answer_text": "درآمد ۱۰۱ میلیارد ریال بود [1]", "error": None}
    msgs = J.build_judge_messages(q, rec)
    u = msgs[1]["content"]
    check("judge.prompt_english_with_both_golds", "Gold answer (Persian): 101 میلیارد ریال" in u and
          "Gold answer (English): 101 billion rials" in u and "Model answer: درآمد" in u and "Same question in English" in u)
    check("judge.parse_strict", J.parse_verdict('{"correct": true, "partially_correct": true, "reason": "ok"}') ==
          {"correct": True, "partially_correct": False, "reason": "ok", "strict_json": True})
    check("judge.parse_fenced", (J.parse_verdict('```json\n{"correct": false, "partially_correct": true, "reason": "unit"}\n```') or {}).get("partially_correct") is True)
    trunc = J.parse_verdict('{"correct": false, "partially_correct": true, "reason": "the model gave the right num')
    check("judge.parse_truncated", trunc and trunc["correct"] is False and trunc["partially_correct"] is True and not trunc["strict_json"], trunc)
    check("judge.parse_garbage", J.parse_verdict("I think it is right") is None)

    check("judge.none", J.make_judge("none", cfg).verdict(q, rec) is None)

    jcfg = cfg.copy_with({"judge": {"budget_file": str(tmp / "judge_budget.json"), "max_calls_total": 3}})
    calls = []

    def fake_call(msgs, ctx):
        calls.append(ctx)
        return '{"correct": true, "partially_correct": false, "reason": "same number"}', 0.00002

    j = J.make_judge("openrouter", jcfg, max_calls=2, max_cost_usd=0.01, cache_path=tmp / "judge_cache.sqlite")
    j._call = fake_call
    check("judge.label", j.label == "openrouter:" + cfg.judge["model"], j.label)
    v1 = j.verdict(q, rec)
    v2 = j.verdict(q, rec)
    check("judge.cache", v1["source"] == "judge" and v2["source"] == "cache" and len(calls) == 1 and v2["correct"] is True,
          (v1.get("source"), v2.get("source"), len(calls)))
    err = j.verdict(q, dict(rec, error="ModelError: boom", answer_text=""))
    ab_ans = j.verdict(q, dict(rec, answer_text="NOT_FOUND"))
    ab_un = j.verdict(_q(2, "unanswerable"), dict(rec, question_id="q02", answer_text="NOT_FOUND [1]"))
    check("judge.rules_without_calls", err["source"] == "rule" and err["correct"] is False and ab_ans["correct"] is False
          and ab_un["correct"] is True and len(calls) == 1)
    j.verdict(_q(3), dict(rec, question_id="q03", answer_text="103 billion"))
    v4 = j.verdict(_q(4), dict(rec, question_id="q04", answer_text="104 billion"))
    check("judge.per_invocation_call_cap", v4["source"] == "budget" and v4["correct"] is None and len(calls) == 2, (v4, len(calls)))
    j2 = J.make_judge("openrouter", jcfg, max_calls=100, max_cost_usd=0.0001, cache_path=tmp / "judge_cache.sqlite")
    j2._call = fake_call
    v5 = j2.verdict(_q(5), dict(rec, question_id="q05", answer_text="105 billion"))
    check("judge.estimated_cost_cap", v5["source"] == "budget" and "cost" in v5["reason"], v5.get("reason"))
    check("judge.parse_string_booleans", (J.parse_verdict('{"correct": "true", "partially_correct": "false", "reason": "x"}') or {}).get("correct") is True)

    # failed requests use the persisted workflow budget: they count toward the cap and 3 in a row stop judging
    fails = []

    def failing_call(msgs, ctx):
        fails.append(ctx)
        raise J.ModelError("OpenRouter HTTP 401: no auth")

    j3 = J.make_judge("openrouter", jcfg, max_calls=100, max_cost_usd=1.0, cache_path=tmp / "judge_cache.sqlite")
    j3._call = failing_call
    vs = [j3.verdict(_q(30 + i), dict(rec, question_id="q%02d" % (30 + i), answer_text="%d billion" % (130 + i)))
          for i in range(6)]
    check("judge.error_breaker", len(fails) == 3 and j3.calls == 3 and [v["source"] for v in vs] ==
          ["error", "error", "error", "budget", "budget", "budget"] and "consecutive" in (j3.exhausted or ""),
          (len(fails), [v["source"] for v in vs], j3.exhausted))
    j4 = J.make_judge("openrouter", jcfg, max_calls=2, max_cost_usd=1.0, cache_path=tmp / "judge_cache.sqlite")
    j4._call = failing_call
    fails.clear()
    j4.max_consecutive_errors = 99
    [j4.verdict(_q(40 + i), dict(rec, question_id="q%02d" % (40 + i), answer_text="%d billion" % (140 + i))) for i in range(4)]
    check("judge.errors_count_toward_call_cap", len(fails) == 2, len(fails))
    oj = J.make_judge("ollama", cfg, cache_path=tmp / "judge_cache.sqlite")
    check("judge.ollama_labelled_weaker", oj.weaker and "WEAKER" in oj.label, oj.label)
    check("judge.no_key_in_label_or_summary", "sk-or" not in json.dumps(j.summary()) + j.label)


# --------------------------------------------------------------------------
# fake RunRecords -> aggregate
# --------------------------------------------------------------------------

def _record(q, approach, base, setting, correct, ev_hit, calls, rng, error=None, flags=None):
    ql, dl = setting.split("-")
    gold = q.gold(dl)
    if q.answer_type == "unanswerable":
        ans = "NOT_FOUND" if correct else "۹۹ میلیارد ریال"
    elif q.answer_type == "short":
        ans = ("Digikala" if ql == "en" else "دیجی‌کالا") if correct else "Snapp"
    else:
        ans = ("%d billion rials [2]" % q.answer_value if ql == "en" else "%s میلیارد ریال" % str(int(q.answer_value)).translate(str.maketrans("0123456789", "۰۱۲۳۴۵۶۷۸۹"))) if correct else "NOT_FOUND"
    ev_units = (gold[:1] if ev_hit and gold else []) + ["A.x.%s::other#%d" % (dl, rng.randint(0, 99))]
    return RunRecord(
        question_id=q.id, approach=approach, approach_base=base, flags=flags or {}, setting=setting, q_lang=ql, d_lang=dl,
        corpus=q.corpus, corpus_set="%s_%s" % (q.corpus, dl), answer_text="" if error else ans,
        abstained=(ans == "NOT_FOUND"), retrieved_units=[{"unit_ids": ev_units, "doc_id": "A.x.%s" % dl}],
        evidence_units=[] if base == "B0" else ev_units, evidence_items=[], evidence_chars=0 if base == "B0" else 3000,
        llm_calls=calls, embed_calls=1 if base == "B3" else 0, prompt_tokens=calls * 900, completion_tokens=calls * 20,
        embed_tokens=0, llm_wall_ms=calls * 800.0, load_ms=0.0, latency_ms=calls * 1000.0 + rng.random() * 100,
        failed_calls=0, error=error, trace=None, models={"chat": "fake", "embed": "fake"}, config_hash="cfgfake",
        index_fingerprint="fp", ts="2026-09-15T00:00:00", run_id="fake").to_dict()


def _kf_q(qid, facts, answer_type="short"):
    return Question(id=qid, corpus="A", question_fa="س", question_en="q", answer_fa="پ", answer_en="a",
                    answer_type=answer_type, gold_units={"fa": [], "en": []}, doc_scope=["A.x.fa"], type="locate",
                    meta={"key_facts": facts})


def test_keyfacts(cfg, tmp):
    """The scorer itself, then every key fact authored in the real question files."""
    num = _kf_q("kf-num", [{"kind": "number", "value": 4400000, "aliases": [], "why": ""}])
    forms = ["4,400,000 rials", "4.4 million", "4.4 میلیون تومان", "۴٬۴۰۰٬۰۰۰ ریال", "4.4M"]
    got = [KF.score_key_facts(num, f)["kf_correct"] for f in forms]
    check("keyfacts.number_readings", all(x == 1.0 for x in got), list(zip(forms, got)))
    check("keyfacts.number_wrong_value", KF.score_key_facts(num, "4.5 million")["kf_correct"] == 0.0)
    k = _kf_q("kf-k", [{"kind": "number", "value": 308000, "aliases": [], "why": ""}])
    check("keyfacts.spaced_scale_letter", KF.score_key_facts(k, "there were 308 K Sellers")["kf_correct"] == 1.0
          and KF.score_key_facts(num, "4.4 M rials")["kf_correct"] == 1.0
          and KF.score_key_facts(k, "308 kg of goods")["kf_correct"] == 0.0)
    slash = _kf_q("kf-slash", [{"kind": "number", "value": 4.7, "aliases": [], "why": ""}])
    check("keyfacts.persian_decimal_slash", KF.score_key_facts(slash, "حداقل امتیاز 4/7 است")["kf_correct"] == 1.0)
    word = _kf_q("kf-word", [{"kind": "number", "value": 4.9e9, "aliases": ["49 billion"], "why": ""}])
    check("keyfacts.number_alias_surface", KF.score_key_facts(word, "customers gave 49 billion rials")["kf_correct"] == 1.0
          and KF.score_key_facts(word, "4.9 میلیارد تومان")["kf_correct"] == 1.0)
    st = _kf_q("kf-str", [{"kind": "string", "value": "زنجیره‌تأمین", "aliases": ["supply chain"], "why": ""}])
    check("keyfacts.string_zwnj_and_alias",
          KF.score_key_facts(st, "درباره زنجیره تامین است")["kf_correct"] == 1.0
          and KF.score_key_facts(st, "It is in the supply chain chapter.")["kf_correct"] == 1.0
          and KF.score_key_facts(st, "It is in the logistics chapter.")["kf_correct"] == 0.0)
    two = _kf_q("kf-two", [{"kind": "number", "value": 447, "aliases": [], "why": ""},
                           {"kind": "string", "value": "هوافضا", "aliases": ["aerospace"], "why": ""}])
    part = KF.score_key_facts(two, "Aerospace is the smallest family.")
    check("keyfacts.all_or_nothing", part["kf_correct"] == 0.0 and part["kf_found"] == 1 and part["kf_recall"] == 0.5
          and KF.score_key_facts(two, "Aerospace, with 447 firms.")["kf_correct"] == 1.0, part)
    none = KF.score_key_facts(_kf_q("kf-none", []), "anything")
    check("keyfacts.no_key_facts_is_none", none["kf_total"] == 0 and none["kf_correct"] is None, none)

    # the authored key facts: a key fact the gold answer itself does not contain is a bug
    bad, n_q, n_facts = [], 0, 0
    for name in KEY_FACT_FILES:
        for q in load_questions(QUESTION_DIR / name):
            facts = KF.key_facts_of(q)
            if q.type in KEY_FACT_TYPES and not facts:
                bad.append((q.id, "no key facts authored"))
            if not facts:
                continue
            n_q += 1
            n_facts += len(facts)
            if not 2 <= len(facts) <= 4:
                bad.append((q.id, "%d key facts (want 2-4)" % len(facts)))
            for lang in ("fa", "en"):
                r = KF.score_key_facts(q, q.answer(lang), lang)
                if r["kf_correct"] != 1.0:
                    bad.append((q.id, lang, r["kf_missing"]))
    check("keyfacts.gold_answers_contain_every_key_fact", not bad,
          "%d questions, %d key facts; bad=%s" % (n_q, n_facts, bad[:6]))


def test_scoring_modes(cfg, tmp):
    """--scoring keyfacts changes only the questions that carry key facts, and always keeps the
    strict number in acc_match_strict."""
    rng = random.Random(3)
    lookup = _q(1, "short")
    prose = Question(id="q90", corpus="A", question_fa="کجا؟", question_en="Where?",
                     answer_fa="در فصل لجستیک، با 447 شرکت.", answer_en="In the logistics chapter, with 447 firms.",
                     answer_type="short", gold_units={"fa": ["A.x.fa::s#u1"], "en": ["A.x.en::s#u1"]},
                     doc_scope=["A.x.fa"], type="locate",
                     meta={"key_facts": [{"kind": "number", "value": 447, "aliases": [], "why": ""},
                                         {"kind": "string", "value": "لجستیک", "aliases": ["logistics"], "why": ""}]})
    qpath = tmp / "questions_scoring.jsonl"
    save_questions(qpath, [lookup, prose])
    run_dir = tmp / "runs" / "scoring"
    f = run_dir / "runs_FAKE.jsonl"
    for q in (lookup, prose):
        for s in ("fa-fa", "en-en"):
            r = _record(q, "FAKE", "FAKE", s, True, True, 1, rng)
            if q.id == "q90":
                # a correct answer in other words: every key fact, none of the gold wording
                r["answer_text"] = ("این عدد در بخش لجستیک آمده و 447 بنگاه دارد" if s == "fa-fa"
                                    else "That figure sits in the logistics part, which holds 447 businesses")
            append_jsonl(f, r)
    res = {}
    for mode in AGG.SCORING_MODES:
        res[mode] = AGG.aggregate_runs([str(run_dir)], [str(qpath)], out_dir=str(tmp / ("results_" + mode)),
                                       judge_kind="none", n_boot=100, cfg=cfg, scoring=mode, log=lambda s: None)
    by = {m: {(r["question_id"], r["setting"]): r for r in res[m]["scored"]} for m in res}
    strict, keyf = by["strict"][("q90", "fa-fa")], by["keyfacts"][("q90", "fa-fa")]
    check("scoring.keyfacts_promotes_kf_correct",
          strict["acc_match"] == 0.0 and keyf["acc_match"] == 1.0 and keyf["acc_match_strict"] == 0.0
          and strict["kf_correct"] == 1.0, (strict["acc_match"], keyf["acc_match"], strict["kf_correct"]))
    check("scoring.lookup_unchanged_by_mode",
          by["strict"][("q01", "fa-fa")]["acc_match"] == by["keyfacts"][("q01", "fa-fa")]["acc_match"]
          and by["keyfacts"][("q01", "fa-fa")].get("kf_correct") is None)
    rows = {m: {(r["approach"], r["setting"]): r for r in res[m]["rows"]} for m in res}
    check("scoring.rows_report_both",
          rows["keyfacts"][("FAKE", "ALL")]["acc_match"] == 1.0
          and rows["keyfacts"][("FAKE", "ALL")]["acc_match_strict"] == 0.5
          and rows["strict"][("FAKE", "ALL")]["acc_match"] == 0.5
          and rows["strict"][("FAKE", "ALL")]["kf_recall"] == 1.0
          and rows["strict"][("FAKE", "ALL")]["n_keyfact"] == 2,
          {k: rows["keyfacts"][("FAKE", "ALL")].get(k) for k in ("acc_match", "acc_match_strict", "kf_recall", "n_keyfact")})
    md = (res["keyfacts"]["out"] / "results.md").read_text(encoding="utf-8")
    check("scoring.md_shows_key_fact_columns", "Key-fact recall %" in md and "Scoring: **keyfacts**" in md)


def test_aggregate(cfg, tmp):
    rng = random.Random(7)
    qs = [_q(i, "numeric") for i in range(1, 17)] + [_q(i, "short") for i in range(17, 21)] + \
         [_q(i, "unanswerable") for i in range(21, 25)]
    qpath = tmp / "questions.jsonl"
    save_questions(qpath, qs)
    run_dir = tmp / "runs" / "fake"
    settings = ["fa-fa", "en-en"]
    truth = {}
    spec = {  # approach -> (base, p_correct, p_ev, calls)
        "B0": ("B0", 0.1, 0.0, 1), "B1": ("B1", 0.95, 1.0, 1), "B2": ("B2", 0.45, 0.6, 1),
        "OURS": ("OURS", 0.8, 0.85, 2), "OURS+no_facts": ("OURS", 0.6, 0.7, 2),
    }
    for approach, (base, pc, pe, calls) in spec.items():
        flags = {"no_facts": True} if "no_facts" in approach else {}
        f = run_dir / ("runs_%s.jsonl" % approach.replace("+", "_"))
        for q in qs:
            for s in settings:
                c = rng.random() < pc
                e = rng.random() < pe
                if approach == "B2" and q.id == "q01" and s == "fa-fa":
                    # an errored attempt followed by a successful retry: the retry must win
                    append_jsonl(f, _record(q, approach, base, s, False, False, 0, rng, error="ModelError: timeout", flags=flags))
                if approach == "B2" and q.id == "q02" and s == "en-en":
                    append_jsonl(f, _record(q, approach, base, s, False, False, 0, rng, error="ModelError: timeout", flags=flags))
                    truth[(approach, q.id, s)] = (0.0, 0.0)
                    continue
                append_jsonl(f, _record(q, approach, base, s, c, e, calls, rng, flags=flags))
                truth[(approach, q.id, s)] = (float(c), float(e) if (q.answer_type != "unanswerable" and base != "B0") else None)
        ablation = approach != base
        atomic_write_json(run_dir / ("index_%s_A_fa.json" % approach.replace("+", "_")),
                          {"approach": approach, "corpus_set": "A_fa", "complete": True, "size_bytes": 1024 * 1024,
                           # the ablation's store was assembled from the shared enrichment cache: 0 ledgered calls
                           "totals": {"llm_calls": 957 if (base == "OURS" and not ablation) else 0,
                                      "embed_calls": 60 if base == "B3" else 0,
                                      "build_ms": 3.6e6 if (base == "OURS" and not ablation) else 1000.0},
                           "cached_cost": ({"llm_calls": 957, "prompt_tokens": 5e5, "completion_tokens": 4e5,
                                            "llm_wall_ms": 3.5e6, "embed_tokens": 0, "embed_wall_ms": 0,
                                            "sources": ["index.json:enrichment.cost_all_cached_calls"]}
                                           if base == "OURS" else None)})
    res = AGG.aggregate_runs([str(run_dir)], [str(qpath)], out_dir=str(tmp / "results"), judge_kind="none",
                             n_boot=2000, seed=1, cfg=cfg, log=lambda s: print("   " + s))
    out = res["out"]
    check("aggregate.outputs", all((out / n).is_file() for n in ("results.csv", "results.md", "comparisons.csv", "scored.jsonl", "judge_summary.json")))
    rows = {(r["approach"], r["setting"]): r for r in res["rows"]}
    check("aggregate.dedupe_retry_wins", len(res["scored"]) == len(spec) * len(qs) * len(settings), len(res["scored"]))
    ok_all = True
    for approach in spec:
        for s in settings + ["ALL"]:
            keys = [k for k in truth if k[0] == approach and (s == "ALL" or k[2] == s)]
            exp_acc = sum(truth[k][0] for k in keys) / len(keys)
            evs = [truth[k][1] for k in keys if truth[k][1] is not None]
            if approach == "B2":   # q02 en-en errored: ev_hit comes from its record (no gold in evidence)
                evs = [truth[k][1] for k in keys if truth[k][1] is not None]
            r = rows[(approach, s)]
            if abs(r["acc_match"] - exp_acc) > 1e-9:
                ok_all = False
                print("   acc mismatch", approach, s, r["acc_match"], exp_acc)
            if approach != "B0" and evs and abs(r["ev_hit"] - sum(evs) / len(evs)) > 1e-9:
                ok_all = False
                print("   ev mismatch", approach, s, r["ev_hit"], sum(evs) / len(evs))
    check("aggregate.acc_match_and_ev_hit_equal_ground_truth", ok_all)
    check("aggregate.b0_no_evidence_recall", rows[("B0", "ALL")]["ev_hit"] is None)
    check("aggregate.errors_counted", rows[("B2", "en-en")]["n_errors"] == 1 and rows[("B2", "fa-fa")]["n_errors"] == 0,
          (rows[("B2", "en-en")]["n_errors"], rows[("B2", "fa-fa")]["n_errors"]))
    check("aggregate.index_columns", rows[("OURS", "ALL")]["index_llm_calls"] == 957 and rows[("OURS", "fa-fa")]["index_build_ms"] == 3.6e6
          and rows[("OURS", "en-en")]["index_llm_calls"] is None, (rows[("OURS", "ALL")]["index_llm_calls"], rows[("OURS", "en-en")]["index_llm_calls"]))
    nf = rows[("OURS+no_facts", "ALL")]
    check("aggregate.index_cost_includes_cached_items", nf["index_llm_calls"] == 0 and nf["index_full_llm_calls"] == 957
          and rows[("OURS", "ALL")]["index_full_llm_calls"] == 957 and rows[("B2", "ALL")]["index_full_llm_calls"] == 0
          and "cost_all_cached_calls" in (nf["index_cost_source"] or ""),
          {k: nf[k] for k in ("index_llm_calls", "index_full_llm_calls", "index_cost_source")})

    class OrderJudge:
        provider = "stub"
        label = "stub"
        weaker = False

        def __init__(self):
            self.seen = []

        def verdict(self, q, r):
            self.seen.append((r["question_id"], r["setting"], r["approach"]))
            return None

        def summary(self):
            return {}

    recs, _ = AGG.load_run_records([str(run_dir)], cfg)
    oj = OrderJudge()
    AGG.score_records(recs, qs, oj, log=lambda s: None)
    first = oj.seen[:len(spec)]
    check("aggregate.judge_order_interleaves_approaches", len({k[:2] for k in first}) == 1
          and {k[2] for k in first} == set(spec), first)
    cov = AGG.coverage_warnings([r for r in recs if not (r["approach"] == "OURS" and r["question_id"] == "q05")])
    check("aggregate.coverage_warning", len(cov) == 1 and "OURS 2" in cov[0]
          and not AGG.coverage_warnings(recs), cov)
    check("aggregate.latency_percentiles", rows[("OURS", "ALL")]["latency_p50_ms"] >= 2000 and rows[("OURS", "ALL")]["latency_p95_ms"] <= 2100)
    comps = res["comparisons"]
    pairs = {(c["a"], c["b"]) for c in comps}
    check("aggregate.comparison_pairs", {("OURS", "B0"), ("OURS", "B1"), ("OURS", "B2"), ("OURS+no_facts", "B2"),
                                         ("OURS+no_facts", "OURS")} <= pairs and ("B2", "OURS") not in pairs, sorted(pairs))
    c = [x for x in comps if x["a"] == "OURS" and x["b"] == "B2" and x["setting"] == "ALL" and x["metric"] == "acc_match"][0]
    exp_diff = rows[("OURS", "ALL")]["acc_match"] - rows[("B2", "ALL")]["acc_match"]
    check("aggregate.bootstrap_ci", abs(c["diff"] - exp_diff) < 1e-9 and c["ci_low"] <= c["diff"] <= c["ci_high"]
          and c["n_clusters"] == len(qs) and c["n_pairs"] == len(qs) * 2, c)
    b = AGG.paired_bootstrap([("q%d" % i, 1.0, 0.0) for i in range(30)], n_boot=500)
    z = AGG.paired_bootstrap([("q%d" % i, float(i % 2), float(i % 2)) for i in range(30)], n_boot=500)
    check("aggregate.bootstrap_degenerate", b["ci_low"] == 1.0 and b["ci_high"] == 1.0 and z["diff"] == 0 and z["ci_low"] == 0 and z["ci_high"] == 0)
    md = (out / "results.md").read_text(encoding="utf-8")
    check("aggregate.markdown", "## Corpus A" in md and "| OURS+no_facts |" in md and "Paired bootstrap" in md and "Acc match" in md)
    head = (out / "results.csv").read_text(encoding="utf-8-sig").splitlines()[0]
    check("aggregate.csv_columns", all(k in head.split(",") for k in ("approach", "corpus", "setting", "acc_judge", "acc_match", "ev_hit",
                                                                       "ev_recall", "llm_calls_mean", "prompt_tokens_mean", "latency_p50_ms",
                                                                       "latency_p95_ms", "index_llm_calls", "index_build_ms", "index_size_bytes")))
    fw = AGG.fairness_warnings([{"approach": "OURS+no_facts", "approach_base": "OURS", "llm_calls": 4, "evidence_chars": 100},
                                {"approach": "B2", "approach_base": "B2", "llm_calls": 1, "evidence_chars": cfg.evidence["max_chars"] + 1},
                                {"approach": "B4", "approach_base": "B4", "llm_calls": 10, "evidence_chars": 10}], cfg)
    check("aggregate.fairness_warnings", len(fw) == 2 and any("OURS+no_facts" in w and "cap (3)" in w for w in fw)
          and any(w.startswith("B2") and "budget" in w for w in fw) and not any("unexpected" in w for w in res["meta"]["warnings"]), fw)
    print("\n" + md[:2500] + "\n...")


# --------------------------------------------------------------------------
# run.py end to end with the fake approach (no model), then resume
# --------------------------------------------------------------------------

def test_run_cli(cfg, tmp):
    from .. import run as RUN
    A_en = load_set("A_en", cfg)
    doc = A_en.doc("A.rpt-a.en")
    unit = next(u for u in doc.units if u.kind == "paragraph" and M.extract_numbers(u.text))
    val = M.extract_numbers(unit.text)[0]
    qs = [Question(id="fake-1", corpus="A", question_fa="پرسش آزمایشی؟", question_en="Test question?",
                   answer_fa=val.text, answer_en=val.text, answer_type="numeric",
                   gold_units={"en": [unit.id], "fa": [swap_lang(unit.id, "fa")]}, doc_scope=["A.rpt-a.en"], type="lookup",
                   answer_value=val.value, answer_unit=None),
          Question(id="fake-2", corpus="A", question_fa="بی‌پاسخ؟", question_en="Unanswerable?", answer_fa="", answer_en="",
                   answer_type="unanswerable", gold_units={"fa": [], "en": []}, doc_scope=["A.rpt-a.en"], type="unanswerable"),
          Question(id="fake-3", corpus="C", question_fa="سی؟", question_en="C?", answer_fa="x", answer_en="x", answer_type="short",
                   gold_units={"fa": ["C.x.fa::p1_b1"]}, doc_scope="corpus", type="lookup")]
    qpath = tmp / "run_questions.jsonl"
    save_questions(qpath, qs)
    out = tmp / "runs" / "cli"
    argv = ["--approach-class", "ragbench.eval.test_eval:FakeRetrieval", "--questions", str(qpath), "--corpus", "A",
            "--settings", "en-en,fa-fa", "--out", str(out), "--no-warmup", "--no-health"]
    logs = []
    rc = RUN.main(argv, cfg=cfg, log=lambda s: (logs.append(s), print("   " + s)))
    recs = read_jsonl(out / "runs_FAKE.jsonl")
    check("run.records_written", rc == 0 and len(recs) == 4 and {r["setting"] for r in recs} == {"en-en", "fa-fa"}, (rc, len(recs)))
    check("run.index_stats_copied", (out / "index_FAKE_A_en.json").is_file() and (out / "index_FAKE_A_fa.json").is_file())
    meta = json.loads((out / "run_meta_FAKE.json").read_text(encoding="utf-8"))
    check("run.meta", meta["invocations"][0]["n_planned"] == 4 and "ledger_totals" in meta["invocations"][0]
          and meta["invocations"][0]["status"] == "ok", meta["invocations"][0].get("status"))
    r1 = [r for r in recs if r["question_id"] == "fake-1" and r["setting"] == "en-en"][0]
    check("run.fake_record_shape", r1["evidence_chars"] <= cfg.evidence["max_chars"] and r1["llm_calls"] == 0 and not r1["error"],
          (r1["evidence_chars"], r1["answer_text"]))
    logs.clear()
    RUN.main(argv, cfg=cfg, log=lambda s: (logs.append(s), print("   " + s)))
    check("run.resume_skips_done", len(read_jsonl(out / "runs_FAKE.jsonl")) == 4 and any("0 to run" in s for s in logs))
    extra = [Question(id="fake-4", corpus="A", question_fa="بی‌پاسخ ۲؟", question_en="Unanswerable 2?", answer_fa="",
                      answer_en="", answer_type="unanswerable", gold_units={"fa": [], "en": []}, doc_scope=["A.rpt-a.en"],
                      type="unanswerable")]
    save_questions(qpath, qs + extra)
    cfg_changed = cfg.copy_with({"evidence": {"max_chars": 5000}})
    rc_mixed = RUN.main(argv, cfg=cfg_changed, log=lambda s: print("   " + s))
    check("run.resume_guard_refuses_other_config", rc_mixed == 2 and len(read_jsonl(out / "runs_FAKE.jsonl")) == 4, rc_mixed)
    rc_more = RUN.main(argv, cfg=cfg, log=lambda s: print("   " + s))
    check("run.resume_runs_only_new_keys", rc_more == 0 and len(read_jsonl(out / "runs_FAKE.jsonl")) == 6)
    edited = [Question.from_dict(dict(qs[0].to_dict(), question_en="Test question, reworded?"))] + qs[1:] + extra
    save_questions(qpath, edited)
    rc_edit = RUN.main(argv, cfg=cfg, log=lambda s: print("   " + s))
    check("run.resume_guard_refuses_changed_question", rc_edit == 2 and len(read_jsonl(out / "runs_FAKE.jsonl")) == 6, rc_edit)
    meta = json.loads((out / "run_meta_FAKE.json").read_text(encoding="utf-8"))
    check("run.meta_code_hash_and_question_sigs", meta["invocations"][-1].get("approach_code_sha1")
          and "fake-1" in meta.get("question_sigs", {}))
    idx_copy = json.loads((out / "index_FAKE_A_en.json").read_text(encoding="utf-8"))
    check("run.index_copy_has_cached_cost_key", "cached_cost" in idx_copy and idx_copy["cached_cost"] is None)

    # an Ollama runtime update between invocations refuses to mix records (health() patched: no model touched)
    from ..llm import OllamaClient
    real_health = OllamaClient.health
    argv_h = [a for a in argv if a != "--no-health"]

    def unans(i):
        return Question(id="fake-%d" % i, corpus="A", question_fa="بی‌پاسخ %d؟" % i, question_en="Unanswerable %d?" % i,
                        answer_fa="", answer_en="", answer_type="unanswerable", gold_units={"fa": [], "en": []},
                        doc_scope=["A.rpt-a.en"], type="unanswerable")
    try:
        fake_h = {"reachable": True, "chat_present": True, "embed_present": True, "chat_digest_match": True,
                  "embed_digest_match": True}
        OllamaClient.health = lambda self: dict(fake_h, version="0.32.15")
        save_questions(qpath, qs + extra + [unans(5)])
        rc_v1 = RUN.main(argv_h, cfg=cfg, log=lambda s: print("   " + s))
        OllamaClient.health = lambda self: dict(fake_h, version="0.34.0")
        save_questions(qpath, qs + extra + [unans(5), unans(6)])
        logs_v = []
        rc_v2 = RUN.main(argv_h, cfg=cfg, log=lambda s: (logs_v.append(s), print("   " + s)))
        check("run.resume_guard_refuses_ollama_update", rc_v1 == 0 and rc_v2 == 2 and any("Ollama version" in s for s in logs_v)
              and len(read_jsonl(out / "runs_FAKE.jsonl")) == 8, (rc_v1, rc_v2, len(read_jsonl(out / "runs_FAKE.jsonl"))))
    finally:
        OllamaClient.health = real_health
    save_questions(qpath, qs)
    try:
        RUN.parse_flags(["no_facts"], "B2")
        bad_flag = False
    except SystemExit:
        bad_flag = True
    check("run.ablation_flags", RUN.parse_flags(["no_facts,granularity=always_window"], "OURS") ==
          {"no_facts": True, "granularity": "always_window"} and bad_flag)
    res = AGG.aggregate_runs([str(out)], [str(qpath)], judge_kind="none", n_boot=200, cfg=cfg, log=lambda s: print("   " + s))
    rows = {(r["approach"], r["setting"]): r for r in res["rows"]}
    check("run.aggregate_on_cli_output", rows[("FAKE", "ALL")]["n"] == 4 and rows[("FAKE", "ALL")]["index_sets"] == "A_en,A_fa",
          {k: rows[("FAKE", "ALL")][k] for k in ("n", "acc_match", "ev_hit", "index_sets")})


def test_splits_final_docs(cfg, tmp):
    """dev/test split stamping, the test-file guard, --final refusals, --docs-from-questions, directory-level
    Ollama version guard and per-split aggregation (fake approach, no model)."""
    from .. import run as RUN
    from ..llm import OllamaClient
    A_en = load_set("A_en", cfg)
    doc = A_en.doc("A.rpt-a.en")
    unit = next(u for u in doc.units if u.kind == "paragraph" and M.extract_numbers(u.text))
    val = M.extract_numbers(unit.text)[0]

    def q(qid, scope):
        return Question(id=qid, corpus="A", question_fa="پرسش؟", question_en="Question?", answer_fa=val.text,
                        answer_en=val.text, answer_type="numeric",
                        gold_units={"en": [unit.id], "fa": [swap_lang(unit.id, "fa")]}, doc_scope=scope,
                        type="lookup", answer_value=val.value, answer_unit=None)

    check("split.names", RUN.question_split("ragbench/data/questions/pilot.jsonl") == "dev"
          and RUN.question_split("test_sample.jsonl") == "test" and RUN.question_split("test.jsonl") == "test"
          and RUN.question_split("run_questions.jsonl") == "other" and AGG.question_file_split("pilot.jsonl") == "dev")
    dev_path = tmp / "pilot_fake.jsonl"
    save_questions(dev_path, [q("dev-1", ["A.rpt-a.en"]), q("dev-2", "corpus")])
    test_path = tmp / "test_fake.jsonl"
    save_questions(test_path, [Question.from_dict(dict(q("tst-1", ["A.rpt-a.en"]).to_dict()))])
    out = tmp / "runs" / "splits"
    base = ["--approach-class", "ragbench.eval.test_eval:FakeRetrieval", "--corpus", "A", "--settings", "en-en",
            "--out", str(out), "--no-warmup"]
    real_health = OllamaClient.health
    fake_h = {"reachable": True, "chat_present": True, "embed_present": True, "chat_digest_match": True,
              "embed_digest_match": True}
    try:
        OllamaClient.health = lambda self: dict(fake_h, version="0.32.15")
        logs = []
        rc = RUN.main(base + ["--questions", str(dev_path), "--docs-from-questions"], cfg=cfg,
                      log=lambda s: (logs.append(s), print("   " + s)))
        recs = read_jsonl(out / "runs_FAKE.jsonl")
        meta = json.loads((out / "run_meta_FAKE.json").read_text(encoding="utf-8"))
        inv = meta["invocations"][-1]
        idx = json.loads((out / "index_FAKE_A_en.json").read_text(encoding="utf-8"))
        check("split.dev_stamped_on_records", rc == 0 and len(recs) == 2 and all(r["split"] == "dev" for r in recs)
              and all(r["meta"]["ollama_version"] == "0.32.15" for r in recs), (rc, [r.get("split") for r in recs]))
        check("docs_from_questions.restricts_every_set", inv["docs_restriction"]["keys"] == ["rpt-a"]
              and idx["docs"] == ["A.rpt-a.en"] and recs[0]["meta"]["docs_restriction"]["mode"] == "docs_from_questions"
              and inv["split"] == "dev" and inv["ollama_version"] == "0.32.15",
              (inv.get("docs_restriction"), idx.get("docs")))
        check("docs_from_questions.corpus_scope_question_still_runs", any(r["question_id"] == "dev-2" for r in recs)
              and not any("skipping" in s for s in logs))
        check("docs_from_questions.referenced_keys_use_gold_units",
              RUN.referenced_doc_keys([q("x", "corpus")], "A") == {"rpt-a"}
              and RUN.referenced_doc_keys([q("x", ["A.rpt-g.fa"])], "A") == {"rpt-a", "rpt-g"}
              and RUN.referenced_doc_keys([q("x", ["A.rpt-g.fa"])], "B") == set())
        # test split: refused without --final, and --final refuses undeclared variants and overrides
        rc_t = RUN.main(base + ["--questions", str(test_path)], cfg=cfg, log=lambda s: print("   " + s))
        check("split.test_file_needs_final", rc_t == 2 and len(read_jsonl(out / "runs_FAKE.jsonl")) == 2, rc_t)
        logs_f = []
        rc_f = RUN.main(base + ["--questions", str(test_path), "--final", "--docs-from-questions", "--ids", "tst-1"],
                        cfg=cfg, log=lambda s: (logs_f.append(s), print("   " + s)))
        check("final.refuses_undeclared_variant_and_subsets", rc_f == 2 and any("FAKE is not declared" in s for s in logs_f)
              and any("--ids" in s for s in logs_f) and any("ragbench/config.yaml" in s for s in logs_f), rc_f)
        base_cfg = load_config()
        # (config final.question_files since the CORE PLAN integration: the file name must be declared)
        ok_args = RUN.build_parser().parse_args(["--approach", "ours", "--questions", "test_core.jsonl", "--corpus", "A",
                                                 "--out", "o", "--final", "--docs-from-questions"])
        from ..contracts import Setting
        sets4 = [Setting.parse(s) for s in ok_args.settings.split(",")]
        check("final.declared_run_is_allowed", RUN.final_problems(ok_args, base_cfg, "OURS", sets4) == [],
              RUN.final_problems(ok_args, base_cfg, "OURS", sets4))
        bad_args = RUN.build_parser().parse_args(["--approach", "ours", "--questions", "test_core.jsonl", "--corpus", "A",
                                                  "--out", "o", "--final", "--allow-mixed", "--no-health", "--limit", "3"])
        probs = RUN.final_problems(bad_args, base_cfg, "OURS+no_facts,granularity=always_window", sets4)
        check("final.refuses_overrides", len(probs) == 5 and any("docs_from_questions" in p for p in probs)
              and any("not declared" in p for p in probs), probs)
        # an Ollama update refuses to add records to the directory, even for another variant's first records
        OllamaClient.health = lambda self: dict(fake_h, version="0.34.0")
        logs_v = []
        rc_v = RUN.main(base + ["--questions", str(dev_path), "--docs-from-questions", "--approach-class",
                                "ragbench.eval.test_eval:FakeRetrieval2"], cfg=cfg,
                        log=lambda s: (logs_v.append(s), print("   " + s)))
        check("ollama.directory_guard_covers_other_variants", rc_v == 2 and any("Ollama version" in s for s in logs_v)
              and not (out / "runs_FAKE2.jsonl").exists(), rc_v)
    finally:
        OllamaClient.health = real_health
    # aggregation keeps splits apart
    other_rec = dict(read_jsonl(out / "runs_FAKE.jsonl")[0], question_id="tst-1", split="test")
    append_jsonl(out / "runs_FAKE.jsonl", other_rec)
    legacy = {k: v for k, v in read_jsonl(out / "runs_FAKE.jsonl")[1].items() if k not in ("split", "meta")}
    legacy.update(question_id="dev-legacy")
    save_questions(dev_path, [q("dev-1", ["A.rpt-a.en"]), q("dev-2", "corpus"), q("dev-legacy", ["A.rpt-a.en"])])
    append_jsonl(out / "runs_FAKE.jsonl", legacy)
    res = AGG.aggregate_runs([str(out)], [str(dev_path), str(test_path)], out_dir=str(tmp / "results_splits"),
                             judge_kind="none", n_boot=100, cfg=cfg, log=lambda s: print("   " + s))
    rows = {(r["split"], r["approach"], r["setting"]): r for r in res["rows"]}
    md = (res["out"] / "results.md").read_text(encoding="utf-8")
    check("aggregate.rows_per_split", rows[("dev", "FAKE", "ALL")]["n"] == 3 and rows[("test", "FAKE", "ALL")]["n"] == 1
          and "## Corpus A, dev split" in md and "## Corpus A, TEST split" in md
          and any("no split stamped" in w for w in res["meta"]["warnings"]), sorted(rows))


class FakeRetrieval2(FakeRetrieval):
    name = "FAKE2"


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass
    base = load_config()
    tmp = Path(tempfile.mkdtemp(prefix="ragbench_eval_test_"))
    cfg = base.copy_with({"paths": {"work_root": str(tmp / "work"), "index_root": str(tmp / "work" / "indexes"),
                                    "runs_root": str(tmp / "work" / "runs"), "ledger_root": str(tmp / "work" / "ledger"),
                                    "store_root": str(tmp / "work" / "stores")},
                          "judge": {"budget_file": str(tmp / "judge_budget_unused.json")}})
    print("scratch:", tmp)
    try:
        test_metrics()
        test_keyfacts(cfg, tmp)
        test_judge(cfg, tmp)
        test_aggregate(cfg, tmp)
        test_scoring_modes(cfg, tmp)
        test_run_cli(cfg, tmp)
        test_splits_final_docs(cfg, tmp)
    finally:
        failed = [n for n, ok in RESULTS if not ok]
        print("\n%d checks, %d failed%s" % (len(RESULTS), len(failed), (": " + ", ".join(failed)) if failed else ""))
        if not failed:
            shutil.rmtree(tmp, ignore_errors=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
