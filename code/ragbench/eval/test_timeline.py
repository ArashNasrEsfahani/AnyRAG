"""Offline tests for TIMELINE (series) questions: the Persian/English period-value parser, series scoring,
structured output and citations, aggregate integration (timeline.csv, cost by question type) and the series
judge rubric (fake calls). No model calls, no OpenRouter credit.

    .venv/Scripts/python.exe -m ragbench.eval.test_timeline
"""
import json
import shutil
import sys
import tempfile
from pathlib import Path

from ..config import load_config
from ..contracts import Question, RunRecord, save_questions
from ..util import append_jsonl
from . import aggregate as AGG
from . import judge as J
from . import metrics as M
from . import timeline as TL

RESULTS = []


def check(name, cond, detail=""):
    RESULTS.append((name, bool(cond)))
    print(("PASS " if cond else "FAIL ") + name + (" - " + str(detail) if detail != "" else ""), flush=True)
    return cond


def series_dict(text, known, pct=None):
    return {p: [round(m.value, 6) for m in ms] for p, ms in TL.extract_series(text, known, pct).items()}


def first_values(text, known, pct=None):
    return {p: v[0] for p, v in series_dict(text, known, pct).items()}


# --------------------------------------------------------------------------
# numbers
# --------------------------------------------------------------------------

def test_numbers():
    def vals(text):
        return [(round(m.value, 6), m.percent) for m in TL.parse_numbers(text)]
    check("numbers.persian_digits_and_decimal", vals("۲۳٫۱۷ درصد") == [(23.17, True)], vals("۲۳٫۱۷ درصد"))
    check("numbers.thousands_separators", vals("۲٬۷۸۸٬۵۵۰ و 2,502,679 و 1،530،000") == [(2788550, False), (2502679, False),
                                                                                       (1530000, False)],
          vals("۲٬۷۸۸٬۵۵۰ و 2,502,679 و 1،530،000"))
    check("numbers.arabic_comma_is_not_a_list_separator_before_space", vals("1398، 1399") == [(1398, False), (1399, False)],
          vals("1398، 1399"))
    check("numbers.slash_decimal", vals("%23/17 و 13/3 میلیون") == [(23.17, True), (13.3e6, False)], vals("%23/17 و 13/3 میلیون"))
    check("numbers.date_and_year_span", vals("1401/05/12") == [] and vals("1398/99") == [(1398, False)],
          (vals("1401/05/12"), vals("1398/99")))
    check("numbers.scale_words", vals("12.5 میلیارد، 44 million، 1.2bn، 44M، 3 هزار میلیارد") ==
          [(12.5e9, False), (44e6, False), (1.2e9, False), (44e6, False), (3e12, False)],
          vals("12.5 میلیارد، 44 million، 1.2bn، 44M، 3 هزار میلیارد"))
    check("numbers.composite_persian", vals("18 میلیون و 519 هزار شماره") == [(18519000, False)]
          and vals("18 هزار و 800 برنامه") == [(18800, False)] and vals("(هزار و 546 نفر)") == [(1546, False)],
          (vals("18 میلیون و 519 هزار شماره"), vals("18 هزار و 800 برنامه"), vals("(هزار و 546 نفر)")))
    check("numbers.composite_stops_at_larger_scale", vals("13.3 میلیون و 18.5 میلیون") == [(13.3e6, False), (18.5e6, False)],
          vals("13.3 میلیون و 18.5 میلیون"))
    check("numbers.word_numbers", vals("چهل‌میلیون کاربر و forty million") == [(40e6, False), (40e6, False)],
          vals("چهل‌میلیون کاربر و forty million"))
    check("numbers.glued_and_plus", vals("15میلیون تراکنش، 40+ میلیون، +45میلیون") == [(15e6, False), (40e6, False), (45e6, False)],
          vals("15میلیون تراکنش، 40+ میلیون، +45میلیون"))
    check("numbers.percent_forms", [m.percent for m in TL.parse_numbers("%24 و 24 درصد و 24 percent و 24%")] == [True] * 4,
          vals("%24 و 24 درصد و 24 percent و 24%"))
    check("numbers.citation_markers_removed", vals("795 نفر [2]") == [(795, False)], vals("795 نفر [2]"))


# --------------------------------------------------------------------------
# periods and pairing
# --------------------------------------------------------------------------

def test_pairing():
    k3 = ["1398", "1399", "1400"]
    c = {
        "fa_year_first": ("در 1398، 13.3 میلیون و در 1399، 18.5 میلیون و در 1400، 21.8 میلیون کاربر", k3,
                          {"1398": 13.3e6, "1399": 18.5e6, "1400": 21.8e6}),
        "en_value_first": ("13.3 million in 1398, 18.5 million in 1399 and 21.8 million in 1400.", k3,
                           {"1398": 13.3e6, "1399": 18.5e6, "1400": 21.8e6}),
        "list_colons_semicolons": ("۱۳۹۸: ۳۵ میلیون؛ ۱۳۹۹: ۴۴ میلیون؛ ۱۴۰۰: ۵۳٫۱ میلیون", k3,
                                   {"1398": 35e6, "1399": 44e6, "1400": 53.1e6}),
        "markdown_table": ("| سال | تعداد |\n|---|---|\n| 1398 | 35 |\n| 1399 | 44 |", k3, {"1398": 35, "1399": 44}),
        "bullets": ("- 1398 → 795 نفر\n- 1399 → 1,029 نفر", k3, {"1398": 795, "1399": 1029}),
        "respectively": ("1397 و 1398 به ترتیب 350 و 795 نفر", ["1397", "1398"], {"1397": 350, "1398": 795}),
        "mixed_orientation": ("250 هزار خودرو (1401) و در سال 1402، 70 هزار خودرو", ["1401", "1402"], {"1401": 250e3, "1402": 70e3}),
        "users_then_phones": ("1398: 35M users, 44M phones in 1399", ["1398", "1399"], {"1398": 35e6, "1399": 44e6}),
        "gregorian": ("Revenue: 2021: $5.2 million; 2022: $7.9 million", ["2021", "2022"], {"2021": 5.2e6, "2022": 7.9e6}),
        "two_digit_jalali": ("در سال 98 نرخ %23/17 بود و در سال 99 به 20 درصد رسید", ["1398", "1399"], {"1398": 23.17, "1399": 20}),
    }
    bad = {}
    for name, (text, known, exp) in c.items():
        got = first_values(text, known)
        if {p: round(v, 3) for p, v in got.items()} != {p: round(v, 3) for p, v in exp.items()}:
            bad[name] = got
    check("pairing.shapes", not bad, bad)
    # a value that looks like a Jalali year is not a period
    got = first_values("Divar staff rose from 350 in 1397 to 795 in 1398, 1,029 in 1399 and 1340 in 1400.",
                       ["1397", "1398", "1399", "1400"])
    check("periods.year_like_values", got == {"1397": 350, "1398": 795, "1399": 1029, "1400": 1340}, got)
    got = first_values("در سال 1400 دیوار 1340 نفر نیرو داشت", ["1400"])
    check("periods.value_with_unit_not_a_year", got == {"1400": 1340}, got)
    # the report's own year is a source, not a period, when a period is named
    got = series_dict("گزارش 1400 نرخ رد سال 1399 را 18 درصد و نرخ 1400 را 24 درصد اعلام کرد", ["1399", "1400"], True)
    check("periods.report_year_is_source", got == {"1399": [18], "1400": [24]}, got)
    got = first_values("The 1399 annual report gives 20%.", ["1399"], True)
    check("periods.report_year_promoted_when_alone", got == {"1399": 20}, got)
    got = first_values("کارنامه در 3 سال اخیر 250 هزار خودرو کارشناسی کرده است (1401)", ["1401"])
    check("periods.durations_are_not_values", got == {"1401": 250e3}, got)
    got = first_values("1399: 20% of reviews, a 12% rise; 1400: 24%", ["1399", "1400"], False)
    check("pairing.want_percent_false_drops_percentages", got == {}, got)
    check("pairing.no_periods_no_pairs", TL.extract_series("about 35 million users", ["1398"]) == {})


# --------------------------------------------------------------------------
# scoring
# --------------------------------------------------------------------------

def _item(period, value, unit, units=None, percent=None, doc="C.x_annual-report_%s.fa"):
    u = M.parse_unit(unit)
    return {"period": period, "value": value, "unit": unit, "value_base": value * (1 if u.percent else u.scale),
            "percent": bool(u.percent), "definition_en": "d", "definition_fa": "d", "doc": doc % period,
            "gold_units": {"fa": units or ["C.x_annual-report_%s.fa::p1_b1" % period], "en": []}}


def tl_question(qid="tl-1", comparable=True, restated=None, supplementary=None, cues=None, series=None):
    series = series or [_item("1398", 13.3, "million users"), _item("1399", 18.5, "million users"),
                        _item("1400", 21.8, "million users")]
    meta = {"schema_version": TL.SCHEMA_VERSION, "answer_series": series, "comparable": comparable,
            "series_notes_fa": "یادداشت", "series_notes_en": "note", "comparability_cues": cues or {},
            "reconciliation": {"rule": TL.RECONCILIATION_RULE, "restatements": restated or [], "supplementary": supplementary or []}}
    return Question(id=qid, corpus="C", question_fa="چند کاربر؟", question_en="How many users?", answer_fa="-", answer_en="-",
                    answer_type="series", gold_units={"fa": [u for it in series for u in it["gold_units"]["fa"]], "en": []},
                    doc_scope=sorted({it["doc"] for it in series}), type="timeline", meta=meta)


def rec(text="", structured=None, trace=None, error=None, **kw):
    r = {"question_id": "tl-1", "approach": "OURS", "approach_base": "OURS", "setting": "fa-fa", "q_lang": "fa", "d_lang": "fa",
         "corpus": "C", "corpus_set": "C", "answer_text": text, "error": error, "trace": trace, "evidence_units": [],
         "retrieved_units": [], "llm_calls": 2, "embed_calls": 0, "prompt_tokens": 2500, "completion_tokens": 80,
         "embed_tokens": 0, "latency_ms": 3200.0, "load_ms": 0.0, "evidence_chars": 4000}
    if structured is not None:
        r["structured"] = structured
    r.update(kw)
    return r


def test_scoring():
    q = tl_question(restated=[dict(_item("1398", 11.0, "million users"), doc="C.x_annual-report_1399.fa")],
                    supplementary=[_item("1397", 9.0, "million users")])
    s = TL.score_series(q, rec("1398: 13.3 میلیون؛ 1399: 18.5 میلیون؛ 1400: 21.8 میلیون کاربر آگهی‌گذار"))
    check("score.full_text_answer", s["tl_coverage"] == 1.0 and s["tl_precision"] == 1.0 and s["tl_series_correct"] == 1.0
          and s["tl_source"] == "text" and s["tl_comparability_flag"] is None and s["tl_false_trend"] is None, s)
    s = TL.score_series(q, rec("11 million in 1398, 18.5 million in 1399 and 22 million in 1400"))
    check("score.restated_and_rounding", abs(s["tl_coverage"] - 2 / 3.) < 1e-6 and abs(s["tl_restated_used"] - 1 / 3.) < 1e-6
          and s["tl_series_correct"] == 0.0 and s["tl_period_hits"]["1400"], s)
    s = TL.score_series(q, rec("13.4 million in 1398, 18.5 million in 1399 and 21.8 million in 1400"))
    check("score.within_tolerance_is_gold", s["tl_coverage"] == 1.0 and s["tl_restated_used"] == 0.0, s)
    s = TL.score_series(q, rec("1397: 9 million; 1398: 13.3 million; 1399: 18.5 million; 1400: 21.8 million"))
    check("score.supplementary_counts_correct", s["tl_precision"] == 1.0 and s["tl_n_pred_periods"] == 4 and s["tl_series_correct"] == 1.0, s)
    s = TL.score_series(q, rec("1395: 2 million; 1398: 13.3 million; 1399: 18.5 million; 1400: 21.8 million"))
    check("score.out_of_range_not_counted", s["tl_precision"] == 1.0 and s["tl_n_out_of_range"] == 1, s)
    q2 = tl_question(series=[_item("1398", 13.3, "million users"), _item("1400", 21.8, "million users")])
    s = TL.score_series(q2, rec("1398: 13.3 million; 1399: 17 million; 1400: 21.8 million"))
    check("score.wrong_period_inside_range_lowers_precision", s["tl_coverage"] == 1.0 and abs(s["tl_precision"] - 2 / 3.) < 1e-6
          and s["tl_series_correct"] == 0.0, s)
    s = TL.score_series(q, rec("NOT_FOUND"), abstained=True)
    check("score.abstention", s["tl_coverage"] == 0.0 and s["tl_precision"] is None and s["tl_series_correct"] == 0.0, s)
    # comparability
    qn = tl_question(comparable=False, cues={"fa": ["گوشی"], "en": ["phones"]})
    s1 = TL.score_series(qn, rec("Users grew from 13.3 million in 1398 to 18.5 million in 1399 and 21.8 million in 1400."))
    s2 = TL.score_series(qn, rec("1398: 13.3 million; 1399: 18.5 million; 1400: 21.8 million. The figures are not directly "
                                 "comparable because the definition changes."))
    s3 = TL.score_series(qn, rec("1398: 13.3 میلیون؛ 1399: 18.5 میلیون؛ 1400: 21.8 میلیون؛ از 1399 گوشی‌ها شمرده شده‌اند و رشد کرده"))
    s4 = TL.score_series(qn, rec("1398: 13.3 million; 1399: 18.5 million; 1400: 21.8 million."))
    check("score.false_trend_without_flag", s1["tl_false_trend"] == 1.0 and s1["tl_comparability_flag"] == 0.0
          and s1["tl_coverage"] == 1.0 and s1["tl_series_correct"] == 0.0, s1)
    check("score.flagged_change", s2["tl_comparability_flag"] == 1.0 and s2["tl_false_trend"] == 0.0 and s2["tl_series_correct"] == 1.0, s2)
    check("score.question_cues_count_as_flag", s3["tl_comparability_flag"] == 1.0 and s3["tl_false_trend"] == 0.0, s3)
    check("score.unflagged_no_trend", s4["tl_comparability_flag"] == 0.0 and s4["tl_false_trend"] == 0.0 and s4["tl_series_correct"] == 0.0, s4)
    check("flags.persian_phrases", TL.flags_definition_change("این ارقام مستقیما قابل‌مقایسه نیستند")
          and TL.flags_definition_change("رقم 1401 تجمعی است") and not TL.flags_definition_change("طبق گزارش 1400، 24 درصد"))
    # one word between «قابل مقایسه» and «نیست», as English allows "not directly comparable" (dev, 2026-09-21: v4's
    # «... متفاوت است و قابل مقایسه مستقیم نیست» scored as no flag)
    check("flags.persian_word_between", TL.flags_definition_change("قابل مقایسه مستقیم نیست")
          and TL.flags_definition_change("ارقام قابل‌مقایسه کامل نیستند")
          and not TL.flags_definition_change("این رقم قابل مقایسه با سال قبل است و رشد کرده"))
    check("trend.detection", TL.states_trend("کاربران دیوار رشد کرد") and TL.states_trend("the count doubled")
          and not TL.states_trend("1398: 35 million; 1399: 44 million"))
    # percentages are strict for answers
    qp = tl_question(series=[_item("1398", 23.17, "percent"), _item("1399", 20, "percent")])
    s = TL.score_series(qp, rec("1398: 23% ; 1399: 20 نفر"))
    check("score.percent_strict_and_rounding", s["tl_period_hits"] == {"1398": True, "1399": False}, s)


def test_structured():
    q = tl_question()
    items = [{"period": "1398", "value": 13.3, "unit": "million users", "unit_id": "C.x_annual-report_1398.fa::p1_b1",
              "doc_id": "C.x_annual-report_1398.fa", "definition": "registered", "quote": "13/3 میلیون"},
             {"period": 1399, "value": "18.5 میلیون", "unit": "", "unit_id": "C.x_annual-report_1399.fa::p9_b9"},
             {"period": "۱۴۰۰", "value": 21800000, "unit": "users", "unit_id": "C.x_annual-report_1400.fa::p2_b2"}]
    texts = {"C.x_annual-report_1399.fa::p9_b9": "از کل 18/5 میلیون کاربری که آگهی گذاشتند",
             "C.x_annual-report_1400.fa::p2_b2": "+21.8 میلیون کاربر یکتای آگهی گذار"}
    s = TL.score_series(q, rec("ignored text 1398: 1 million", structured={"series": items}), unit_resolver=texts.get)
    check("structured.scored_directly", s["tl_source"] == "structured" and s["tl_coverage"] == 1.0 and s["tl_precision"] == 1.0, s)
    check("structured.citations", s["tl_citation_correct"] == 1.0 and s["tl_n_citations"] == 3, s)
    texts_bad = dict(texts, **{"C.x_annual-report_1400.fa::p2_b2": "آمار مربوط به 1400 نیست"})
    s = TL.score_series(q, rec("", structured={"series": items}), unit_resolver=texts_bad.get)
    check("structured.wrong_citation", abs(s["tl_citation_correct"] - 2 / 3.) < 1e-6, s)
    s = TL.score_series(q, rec("", trace={"structured": {"series": items[:2]}}))
    check("structured.from_trace_without_resolver", s["tl_source"] == "structured" and abs(s["tl_coverage"] - 2 / 3.) < 1e-6
          and s["tl_citation_correct"] == 1.0 and s["tl_n_citations"] == 1, s)
    qn = tl_question(comparable=False)
    s = TL.score_series(qn, rec("Users grew every year.", structured={"series": items, "comparable": False}))
    check("structured.comparable_false_is_a_flag", s["tl_comparability_flag"] == 1.0 and s["tl_false_trend"] == 0.0
          and s["tl_series_correct"] == 1.0, s)
    s = TL.score_series(qn, rec("", structured={"series": items, "trend": "steady growth"}))
    check("structured.trend_field_without_flag", s["tl_false_trend"] == 1.0, s)
    s = TL.score_series(q, rec("1398: 13.3 million; 1399: 18.5 million; 1400: 21.8 million", structured={"series": []}))
    check("structured.empty_series_falls_back_to_text", s["tl_source"] == "text" and s["tl_coverage"] == 1.0, s)
    # score_record integration: match_correct, cost fields kept
    row = M.score_record(rec("1398: 13.3 million; 1399: 18.5 million; 1400: 21.8 million"), q)
    check("score_record.series", row["match_correct"] == 1.0 and row["tl_coverage"] == 1.0 and row["string_match"] is None
          and row["qtype"] == "timeline" and row["latency_ms"] == 3200.0 and row["llm_calls"] == 2
          and row["prompt_tokens"] == 2500 and row["latency_noload_ms"] == 3200.0, {k: row.get(k) for k in (
              "match_correct", "tl_coverage", "qtype", "latency_ms", "llm_calls")})
    row = M.score_record(rec("", error="ModelError: boom"), q)
    check("score_record.series_error", row["match_correct"] == 0.0 and row["tl_coverage"] == 0.0, row.get("tl_coverage"))


def test_dev_file(cfg):
    """The DEV timeline questions' own reference answers score as fully correct in both languages; a naive trend
    answer to the Divar staff question is caught."""
    from ..contracts import load_questions
    path = cfg.path("questions_root") / "pilot_timeline.jsonl"
    if not path.is_file():
        check("dev_file.present", False, path)
        return
    qs = {q.id: q for q in load_questions(path)}
    bad = {}
    for q in qs.values():
        for lang in ("fa", "en"):
            row = M.score_record(rec(q.answer(lang), question_id=q.id, q_lang=lang, setting="%s-fa" % lang), q)
            if row["match_correct"] != 1.0:
                bad["%s/%s" % (q.id, lang)] = {k: row.get(k) for k in ("tl_coverage", "tl_precision", "tl_comparability_flag",
                                                                       "tl_false_trend", "tl_period_hits")}
    check("dev_file.reference_answers_score_correct", not bad and len(qs) == 8, bad)
    q = qs["C-TL-divar-01"]
    row = M.score_record(rec("نیروی کار دیوار از 350 نفر در سال 1397 به 1340 نفر در سال 1400 رسید؛ رشدی نزدیک به چهار برابر.",
                             question_id=q.id), q)
    check("dev_file.naive_trend_caught", row["tl_false_trend"] == 1.0 and row["tl_coverage"] == 0.5 and row["match_correct"] == 0.0
          and row["tl_precision"] == 1.0, {k: row.get(k) for k in ("tl_false_trend", "tl_coverage", "tl_precision")})
    q = qs["C-TL-divar-02"]
    row = M.score_record(rec("1398: 23.17%; 1399: 18%; 1400: 24%", question_id=q.id), q)
    check("dev_file.restated_value_caught", abs(row["tl_restated_used"] - 1 / 3.) < 1e-6 and row["tl_period_hits"]["1399"] is False,
          {k: row.get(k) for k in ("tl_restated_used", "tl_period_hits")})
    q = qs["C-TL-tipax-01"]
    row = M.score_record(rec("۱۳۹۹: ۱٬۵۳۰٬۰۰۰؛ ۱۴۰۰: ۲٬۰۴۷٬۸۵۸؛ ۱۴۰۱: ۲٬۵۰۲٬۶۷۹؛ ۱۴۰۲: ۲٬۶۸۱٬۲۴۰؛ ۱۴۰۳: ۲٬۷۸۸٬۵۵۰ مشتری فرستنده",
                             question_id=q.id), q)
    check("dev_file.supplementary_periods", row["tl_coverage"] == 1.0 and row["tl_precision"] == 1.0 and row["tl_n_pred_periods"] == 5,
          {k: row.get(k) for k in ("tl_coverage", "tl_precision", "tl_n_pred_periods")})


def test_test_file(cfg):
    """The TEST timeline questions' own reference answers score as fully correct in both languages (skipped while the
    file does not exist)."""
    from ..contracts import load_questions
    path = cfg.path("questions_root") / "test_timeline.jsonl"
    if not path.is_file():
        print("  skip test_file (no %s)" % path)
        return
    qs = [q for q in load_questions(path) if TL.is_timeline(q)]
    bad = {}
    for q in qs:
        for lang in ("fa", "en"):
            row = M.score_record(rec(q.answer(lang), question_id=q.id, q_lang=lang, setting="%s-fa" % lang), q)
            if row["match_correct"] != 1.0:
                bad["%s/%s" % (q.id, lang)] = {k: row.get(k) for k in ("tl_coverage", "tl_precision", "tl_comparability_flag",
                                                                       "tl_false_trend", "tl_period_hits")}
    check("test_file.reference_answers_score_correct", qs and not bad, "%d questions; %s" % (len(qs), bad))


def test_parser_fixes():
    """2026-09-16 (integration): dot-grouped thousands and series measured in a duration."""
    def vals(text):
        return [(round(m.value, 6), m.percent) for m in TL.parse_numbers(text)]
    check("numbers.dot_grouped_thousands", vals("4.318.701 سفر و 12.345.678.901 ریال") == [(4318701, False), (12345678901, False)],
          vals("4.318.701 سفر و 12.345.678.901 ریال"))
    check("numbers.dot_grouped_thousands_persian", vals("۴٫۳۱۸٫۷۰۱ و ۴٬۳۱۸٬۷۰۱") == [(4318701, False), (4318701, False)],
          vals("۴٫۳۱۸٫۷۰۱ و ۴٬۳۱۸٬۷۰۱"))
    check("numbers.dot_grouped_sentence_end", vals("In 1402 it was 4.318.701.") == [(1402, False), (4318701, False)],
          vals("In 1402 it was 4.318.701."))
    check("numbers.single_dot_stays_decimal", vals("4.318 و 12.5 درصد و 1398.01") == [(4.318, False), (12.5, True), (1398.01, False)],
          vals("4.318 و 12.5 درصد و 1398.01"))
    got = first_values("1401: 4.318.701 سفر؛ 1402: ۵٫۱۲۰٫۰۰۰ سفر", ["1401", "1402"])
    check("pairing.dot_grouped_values", got == {"1401": 4318701, "1402": 5120000}, got)
    # durations: dropped as values unless the gold unit is that duration
    check("duration.unit_kind", TL.unit_duration_kind("hours") == "hour" and TL.unit_duration_kind("ساعت") == "hour"
          and TL.unit_duration_kind("million hours") == "hour" and TL.unit_duration_kind("روز") == "day"
          and TL.unit_duration_kind("rides per day") is None and TL.unit_duration_kind("روزانه") is None
          and TL.unit_duration_kind("people") is None,
          [TL.unit_duration_kind(u) for u in ("hours", "ساعت", "million hours", "روز", "rides per day", "روزانه", "people")])
    text = "1401: 48 ساعت؛ 1402: 36 ساعت (طی 3 سال گذشته)"
    got = first_values(text, ["1401", "1402"])
    check("duration.dropped_without_duration_gold", got == {}, got)
    got = {p: [round(m.value, 6) for m in ms] for p, ms in TL.extract_series(text, ["1401", "1402"], None, {"hour"}).items()}
    check("duration.kept_when_gold_is_that_duration", got == {"1401": [48], "1402": [36]}, got)
    got = {p: [round(m.value, 6) for m in ms] for p, ms in
           TL.extract_series("1401: 2 روز؛ 1402: 30 ساعت", ["1401", "1402"], None, {"hour"}).items()}
    check("duration.other_duration_still_dropped", got == {"1402": [30]}, got)
    got = first_values("در سال 1400 میانگین 1440 دقیقه", ["1400"])
    check("duration.year_like_value_before_duration_is_not_a_period", "1440" not in got, got)
    qd = tl_question(series=[_item("1401", 48, "hours"), _item("1402", 36, "hours")])
    for i, it in enumerate(qd.meta["answer_series"]):
        it["unit_fa"] = "ساعت"
    s = TL.score_series(qd, rec("1401: 48 ساعت\n1402: 36 ساعت\nزمان تحویل کوتاه‌تر شد."))
    check("duration.series_in_hours_scores", s["tl_coverage"] == 1.0 and s["tl_precision"] == 1.0 and s["tl_series_correct"] == 1.0, s)
    s = TL.score_series(tl_question(), rec("1401: 48 ساعت"))
    check("duration.non_duration_series_unchanged", s["tl_n_pred_periods"] == 0, s)
    # the shared timeline answer prompt's format (contracts answer-timeline-v1): one line per period, NOT_FOUND lines
    q = tl_question()
    s = TL.score_series(q, rec("1398: 13.3 million users\n1399: NOT_FOUND\n1400: 21.8 million users\nUsers grew."))
    check("score.answer_timeline_v1_lines_with_not_found", s["tl_period_hits"] == {"1398": True, "1399": False, "1400": True}
          and s["tl_precision"] == 1.0, s)
    s = TL.score_series(q, rec("1398: 13.3 میلیون کاربر\n1399: 18.5 میلیون کاربر\n1400: 21.8 میلیون کاربر\nتعداد کاربران افزایش یافت."))
    check("score.answer_timeline_v1_lines_persian", s["tl_series_correct"] == 1.0, s)


def test_validation_helpers():
    it = _item("1399", 20, "percent")
    check("unit_contains.detached_percent", TL.unit_text_contains_value("20", it)
          and not TL.unit_text_contains_value("20", it, percent_strict=True))
    it = _item("1398", 13.3, "million users")
    check("unit_contains.detached_scale", TL.unit_text_contains_value("13/3", it) and not TL.unit_text_contains_value("13", it))
    it = _item("1399", 18.5, "million users")
    check("unit_contains.words_plus_digits", TL.unit_text_contains_value("18 میلیون و 519 هزار شماره", it))
    check("period_key.order", sorted(["1400", "1398", "2021", "1399"], key=TL.period_key) == ["1398", "1399", "1400", "2021"])


# --------------------------------------------------------------------------
# aggregate
# --------------------------------------------------------------------------

def _run_record(q, approach, base, answer, calls, latency, structured=None):
    r = RunRecord(question_id=q.id, approach=approach, approach_base=base, flags={}, setting="fa-fa", q_lang="fa", d_lang="fa",
                  corpus="C", corpus_set="C", answer_text=answer, abstained=answer == "NOT_FOUND",
                  retrieved_units=[{"unit_ids": q.gold("fa")[:1], "doc_id": "C.x.fa"}], evidence_units=q.gold("fa")[:1],
                  evidence_items=[], evidence_chars=3000, llm_calls=calls, embed_calls=0, prompt_tokens=calls * 1000,
                  completion_tokens=calls * 40, embed_tokens=0, llm_wall_ms=calls * 900.0, load_ms=0.0, latency_ms=latency,
                  failed_calls=0, error=None, trace={"structured": structured} if structured else None,
                  models={"chat": "fake"}, config_hash="cfgfake", index_fingerprint="fp", ts="2026-09-16T00:00:00",
                  run_id="fake", split="dev").to_dict()
    return r


def test_aggregate(cfg, tmp):
    tq = [tl_question("tl-%d" % i, comparable=(i % 2 == 0)) for i in range(4)]
    nq = [Question(id="num-%d" % i, corpus="C", question_fa="چند؟", question_en="How many?", answer_fa="795 نفر",
                   answer_en="795 people", answer_type="numeric", gold_units={"fa": ["C.x.fa::p1_b1"], "en": []},
                   doc_scope=["C.x.fa"], type="lookup", answer_value=795, answer_unit="people") for i in range(3)]
    qpath = tmp / "pilot_tl.jsonl"
    save_questions(qpath, tq + nq)
    run_dir = tmp / "runs" / "tl"
    good = "1398: 13.3 میلیون؛ 1399: 18.5 میلیون؛ 1400: 21.8 میلیون. تعریف تغییر کرده و قابل مقایسه نیست."
    for q in tq:
        append_jsonl(run_dir / "runs_OURS.jsonl", _run_record(q, "OURS", "OURS", good, 3, 9000.0))
        append_jsonl(run_dir / "runs_B2.jsonl", _run_record(q, "B2", "B2", "Users grew to 21.8 million in 1400.", 1, 2000.0))
        append_jsonl(run_dir / "runs_AGENT.jsonl", _run_record(q, "AGENT", "AGENT", "", 7, 30000.0, structured={
            "series": [{"period": "1398", "value": 13.3, "unit": "million users", "unit_id": q.gold("fa")[0]},
                       {"period": "1399", "value": 18.5, "unit": "million users"},
                       {"period": "1400", "value": 21.8, "unit": "million users"}], "comparable": False}))
    for q in nq:
        for ap, base, calls in (("OURS", "OURS", 2), ("B2", "B2", 1), ("AGENT", "AGENT", 5)):
            append_jsonl(run_dir / ("runs_%s.jsonl" % ap), _run_record(q, ap, base, "795 نفر", calls, 1500.0))
    res = AGG.aggregate_runs([str(run_dir)], [str(qpath)], out_dir=str(tmp / "results"), judge_kind="none", n_boot=200,
                             cfg=cfg, log=lambda s: None)
    out = res["out"]
    check("aggregate.timeline_outputs", (out / "timeline.csv").is_file() and (out / "cost_by_qtype.csv").is_file())
    tl = {(r["approach"], r["setting"]): r for r in res["timeline_rows"]}
    ours, b2, agent = tl[("OURS", "ALL")], tl[("B2", "ALL")], tl[("AGENT", "ALL")]
    check("aggregate.timeline_metrics", ours["tl_coverage"] == 1.0 and ours["tl_series_correct"] == 1.0
          and ours["tl_comparability_flag"] == 1.0 and b2["tl_false_trend"] == 1.0 and abs(b2["tl_coverage"] - 1 / 3.) < 1e-6
          and agent["n_structured"] == 4 and agent["tl_citation_correct"] == 1.0 and ours["n_non_comparable"] == 2,
          {k: (ours.get(k), b2.get(k), agent.get(k)) for k in ("tl_coverage", "tl_series_correct", "tl_comparability_flag",
                                                               "tl_false_trend", "tl_citation_correct")})
    check("aggregate.timeline_cost_fields", ours["llm_calls_mean"] == 3 and ours["prompt_tokens_mean"] == 3000
          and ours["latency_p50_ms"] == 9000.0 and agent["latency_p95_ms"] == 30000.0 and ours["n"] == 4,
          {k: ours.get(k) for k in ("llm_calls_mean", "prompt_tokens_mean", "latency_p50_ms")})
    rows = {(r["split"], r["approach"], r["setting"]): r for r in res["rows"]}
    r_all = rows[("dev", "OURS", "ALL")]
    check("aggregate.results_rows_carry_timeline_means", r_all["n_timeline"] == 4 and r_all["tl_coverage"] == 1.0
          and r_all["n"] == 7, {k: r_all.get(k) for k in ("n_timeline", "tl_coverage", "n")})
    qt = {(r["approach"], r["qtype"]): r for r in res["qtype_rows"]}
    check("aggregate.cost_by_qtype", qt[("AGENT", "timeline")]["llm_calls_mean"] == 7 and qt[("AGENT", "lookup")]["llm_calls_mean"] == 5
          and qt[("OURS", "timeline")]["latency_p50_ms"] == 9000.0 and qt[("B2", "lookup")]["acc_match"] == 1.0,
          {k: v["llm_calls_mean"] for k, v in qt.items()})
    md = (out / "results.md").read_text(encoding="utf-8")
    check("aggregate.markdown_sections", "### Timeline questions" in md and "### Cost by question type" in md and "| AGENT |" in md)
    comps = [c for c in res["comparisons"] if c["metric"] == "tl_coverage" and c["a"] == "OURS" and c["b"] == "B2" and c["setting"] == "ALL"]
    check("aggregate.tl_coverage_compared", comps and comps[0]["n_pairs"] == 4, comps[:1])
    head = (out / "timeline.csv").read_text(encoding="utf-8-sig").splitlines()[0].split(",")
    check("aggregate.timeline_csv_columns", all(k in head for k in ("tl_coverage", "tl_precision", "tl_false_trend", "llm_calls_mean",
                                                                   "prompt_tokens_mean", "latency_p50_ms", "latency_noload_p50_ms")))
    # --timeline-scoring text (2026-09-16): every approach from its answer text; AGENT's empty text scores 0 and its
    # structured series moves to the secondary tls_* columns
    res_t = AGG.aggregate_runs([str(run_dir)], [str(qpath)], out_dir=str(tmp / "results_text"), judge_kind="none", n_boot=200,
                               cfg=cfg, log=lambda s: None, timeline_scoring="text")
    tlt = {(r["approach"], r["setting"]): r for r in res_t["timeline_rows"]}
    at, ot = tlt[("AGENT", "ALL")], tlt[("OURS", "ALL")]
    check("aggregate.text_scoring_ignores_structured", at["tl_coverage"] == 0.0 and at["tl_citation_correct"] is None
          and at["tls_coverage"] == 1.0 and at["tls_series_correct"] == 1.0 and at["n_structured_scored"] == 4
          and at["timeline_scoring"] == "text" and ot["tl_coverage"] == 1.0 and ot["tls_coverage"] is None,
          {k: (at.get(k), ot.get(k)) for k in ("tl_coverage", "tls_coverage", "tls_series_correct", "n_structured_scored")})
    md_t = (tmp / "results_text" / "results.md").read_text(encoding="utf-8")
    check("aggregate.text_scoring_markdown", "answer text for every approach" in md_t and "Struct. coverage %" in md_t
          and res_t["meta"]["timeline_scoring"] == "text")
    q0 = tq[1]
    r_agent = _run_record(q0, "AGENT", "AGENT", "Users grew every year.", 7, 30000.0, structured={
        "series": [{"period": "1398", "value": 13.3, "unit": "million users"}], "comparable": False})
    s_auto = TL.score_series(q0, r_agent)
    s_text = TL.score_series(q0, r_agent, scoring="text")
    check("structured.text_mode_flag_and_trend_from_text", s_auto["tl_comparability_flag"] == 1.0 and s_auto["tl_false_trend"] == 0.0
          and s_text["tl_comparability_flag"] == 0.0 and s_text["tl_false_trend"] == 1.0 and s_text["tl_source"] == "text",
          (s_auto, s_text))


# --------------------------------------------------------------------------
# judge (series rubric, no network)
# --------------------------------------------------------------------------

def test_judge(cfg, tmp):
    q, r = J.series_probe_example()
    msgs = J.build_series_judge_messages(q, r)
    u = msgs[1]["content"]
    check("judge.series_prompt", "- 1399: 44 million phones (report: C.divar_annual-report_1399.fa; counts: unique phones" in u
          and "Restated values that are NOT correct for their period: 1399: 44.1 million phones" in u
          and "WARNING: the figures of this series are NOT comparable across periods. 1398 counts active users" in u
          and u.index("WARNING") < u.index("Gold series")
          and '{"flags_definition_change": true or false, "false_trend": true or false, "period_correct": {"1398": true or false, '
              '"1399": true or false, "1400": true or false}' in u
          and "Model answer: Divar grew steadily" in u, u[:400])
    qc = tl_question(comparable=True)
    uc = J.build_series_judge_messages(qc, rec("x"))[1]["content"]
    check("judge.series_prompt_comparable", "WARNING" not in uc and "The figures are comparable across periods." in uc
          and '"false_trend": false (the series is comparable).' in uc and "flags the change" not in uc, uc[-900:])
    # the live v1 probe: every period right, no flag, yet "correct": true and "false_trend": false on a non-comparable series
    v = J.parse_series_verdict('{"period_correct": {"1398": true, "1399": true, "1400": true}, "flags_definition_change": false, '
                               '"false_trend": false, "correct": true, "partially_correct": false, "reason": "right values"}',
                               ["1398", "1399", "1400"], comparable=False)
    check("judge.series_correct_consistent_with_fields", v["correct"] is False and v["partially_correct"] is True
          and v["series"]["correct_raw"] is True, v)
    v = J.parse_series_verdict('{"period_correct": {"1398": true, "1399": true, "1400": false}, "flags_definition_change": false, '
                               '"false_trend": true, "correct": false, "partially_correct": true, "reason": "trend"}',
                               ["1398", "1399", "1400"])
    check("judge.series_parse_strict", v["strict_json"] and abs(v["series"]["coverage"] - 2 / 3.) < 1e-9 and v["partially_correct"]
          and v["series"]["false_trend"] is True, v)
    v = J.parse_series_verdict('```json\n{"period_correct": {"1398": "true", "1399": true}, "flags_definition_change": true}\n```',
                               ["1398", "1399", "1400"])
    check("judge.series_parse_derives_correct", v and v["correct"] is False and v["partially_correct"] and not v["strict_json"]
          and v["series"]["period_correct"]["1400"] is False, v)
    v = J.parse_series_verdict('{"period_correct": {"1398": true, "1399": true, "1400": true}, "flags_definition_cha',
                               ["1398", "1399", "1400"])
    check("judge.series_parse_truncated", v and v["series"]["coverage"] == 1.0 and v["correct"] is True, v)
    check("judge.series_parse_garbage", J.parse_series_verdict("looks right to me", ["1398"]) is None)
    jcfg = cfg.copy_with({"judge": {"budget_file": str(tmp / "judge_budget.json"), "max_calls_total": 5}})
    seen = []

    def fake_call(msgs, ctx):
        seen.append((j._series_call, j._call_max_tokens()))
        return ('{"period_correct": {"1398": true, "1399": true, "1400": true}, "flags_definition_change": false, '
                '"false_trend": true, "correct": false, "partially_correct": true, "reason": "unflagged trend"}'), 0.00003

    j = J.make_judge("openrouter", jcfg, max_calls=3, max_cost_usd=0.01, cache_path=tmp / "judge_cache.sqlite")
    j._call = fake_call
    v1 = j.verdict(q, r)
    v2 = j.verdict(q, r)
    check("judge.series_verdict_and_cache", v1["source"] == "judge" and v2["source"] == "cache" and len(seen) == 1
          and v2["series"]["false_trend"] is True and v1["correct"] is False and seen[0] == (True, J.SERIES_MAX_TOKENS),
          (v1.get("source"), v2.get("source"), seen))
    na = j.verdict(q, dict(r, answer_text="NOT_FOUND"))
    check("judge.series_rule_abstain", na["source"] == "rule" and na["correct"] is False and len(seen) == 1)
    k1 = j.cache_key(q, r)
    q2 = Question.from_dict(json.loads(json.dumps(q.to_dict())))
    q2.meta["answer_series"][2]["value"] = 53.2
    check("judge.series_cache_key_includes_gold", k1 != j.cache_key(q2, r))
    oj = J.make_judge("ollama", cfg, cache_path=tmp / "judge_cache2.sqlite")
    check("judge.ollama_series_schema", "period_correct" in oj.SERIES_SCHEMA["properties"])
    # aggregate copies the judge's series fields
    tq = tl_question(comparable=False)

    class StubJudge:
        provider, label, weaker = "stub", "stub", False

        def verdict(self, q, r):
            return {"correct": False, "partially_correct": True, "source": "judge",
                    "series": {"coverage": 0.5, "flags_definition_change": False, "false_trend": True}}

        def summary(self):
            return {}
    rows = AGG.score_records([rec("1398: 13.3 million", question_id="tl-1")], [tq], StubJudge(), log=lambda s: None)
    check("aggregate.judge_series_fields", rows[0]["tl_judge_coverage"] == 0.5 and rows[0]["tl_judge_false_trend"] == 1.0
          and rows[0]["tl_judge_flag"] == 0.0 and rows[0]["acc_judge"] == 0.0, {k: rows[0].get(k) for k in rows[0] if "judge" in k})


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass
    base = load_config()
    tmp = Path(tempfile.mkdtemp(prefix="ragbench_timeline_test_"))
    cfg = base.copy_with({"paths": {"work_root": str(tmp / "work"), "index_root": str(tmp / "work" / "indexes"),
                                    "runs_root": str(tmp / "work" / "runs"), "ledger_root": str(tmp / "work" / "ledger"),
                                    "store_root": str(tmp / "work" / "stores")},
                          "judge": {"budget_file": str(tmp / "judge_budget_unused.json")}})
    print("scratch:", tmp)
    try:
        test_numbers()
        test_pairing()
        test_scoring()
        test_structured()
        test_validation_helpers()
        test_parser_fixes()
        test_dev_file(base)
        test_test_file(base)
        test_aggregate(cfg, tmp)
        test_judge(cfg, tmp)
    finally:
        failed = [n for n, ok in RESULTS if not ok]
        print("\n%d checks, %d failed%s" % (len(RESULTS), len(failed), (": " + ", ".join(failed)) if failed else ""))
        if not failed:
            shutil.rmtree(tmp, ignore_errors=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
