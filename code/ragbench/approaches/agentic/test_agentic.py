"""AGENTIC offline tests (no model, no GPU; a scripted client replaces the HTTP call).

    .venv/Scripts/python.exe -m ragbench.approaches.agentic.test_agentic

Covers: planner JSON parsing/repair and validation, Persian term morphology, proximity windows on Persian
text with detached numbers (and the line_windows ablation that misses them), the verbatim guard, number
readings (scale words, right-to-left glitch groups), reconciliation (authoritative vs restated, conflicts,
non-comparable definitions, subgroups, single-fact tables), the call budget, the per-call cache (replay and
resume after a failure), and a full Approach.answer() RunRecord with structured.series.
Writes only under work/test_agentic/.
"""
import json
import re
import shutil
import sys
from pathlib import Path

from ...config import load_config
from ...contracts import Question
from ...corpus.model import DocBuilder
from ...corpus.registry import CorpusSet
from ...llm import BudgetExceeded, Ledger, ModelError, OllamaClient
from ...normalize import normalize
from ...store import build_plain_store
from . import planner as P
from . import reconcile as R
from . import search as S
from . import verify as V
from .approach import AgenticPipeline, CallBudget, SERIES_FIELDS, parse_flags
from .defaults import DEFAULTS
from .docmeta import doc_meta, select_docs
from .numbers import find_numbers

ZWNJ = "‌"
FAILS = []
N = [0]


def check(cond, label, detail=""):
    N[0] += 1
    print(("  ok   " if cond else "  FAIL ") + label + ((" - " + str(detail)) if (detail and not cond) else ""))
    if not cond:
        FAILS.append(label)
    return cond


# --------------------------------------------------------------------------
# a small synthetic Persian corpus shaped like the Snapp extraction problems
# --------------------------------------------------------------------------

FILLER = ("این بخش دربارهٔ برنامه‌های رفاهی و خدمات پشتیبانی است و جزئیات بیشتری از همکاری‌ها و طرح‌های سال را شرح می‌دهد. " * 8)


def _doc(doc_id, title, sections):
    b = DocBuilder(doc_id, "C", "fa", title, source="test")
    for sec_local, sec_title, units in sections:
        b.add_section(sec_local, sec_title)
        for k, (kind, text) in enumerate(units):
            b.add_unit(sec_local, "%s_b%d" % (sec_local, k), kind, text)
    d = b.build()
    d.meta["fingerprint"] = "test-" + doc_id
    return d


def make_corpus():
    d1398 = _doc("C.testco_annual-report_1398.fa", "گزارش عملکرد تست‌کو در سال ۱۳۹۸", [
        ("p1", "مقدمه", [("paragraph", "تعداد راننده‌های تست‌کو تا پایان سال ۹۸ به بیش از ۲میلیون نفر می‌رسد که حدود ۳درصد آن‌ها زن هستند. " + FILLER)]),
    ])
    d1400 = _doc("C.testco_annual-report_1400.fa", "گزارش عملکرد سالانه گروه تست‌کو ۱۴۰۰", [
        ("p1", "تست‌کو خودرو؛ آمار", [("paragraph", "کل راننده ها 3,761,925"), ("paragraph", "رانندگان زن 100,263"),
                                   ("paragraph", "راننده های فعال"), ("paragraph", "رانندگان مرد"),
                                   ("paragraph", "1,896,210"), ("paragraph", "3,661,662")]),
        ("p2", "باشگاه", [("paragraph", "در سالی که گذشت بیش از 110هزار کاربر راننده از خدمات باشگاه رانندگان استفاده کردند. " + FILLER)]),
    ])
    d1401 = _doc("C.testco_annual-report_1401.fa", "گزارش عملکرد سالانه گروه تست‌کو ۱۴۰۱", [
        ("p1", "نسبت کاربران راننده زن و مرد", [("paragraph", "رانندگان زن"), ("paragraph", "567„ 150 نفر"),
                                             ("paragraph", "رانندگان مرد"), ("paragraph", "979„379„4 نفر")]),
    ])
    d1402 = _doc("C.testco_annual-report_1402.fa", "گزارش عملکرد سالانه گروه تست‌کو ۱۴۰۲", [
        ("p1", "تست‌کو خودرو", [("paragraph", "محمد مدیرعامل تست‌کو"), ("paragraph", "تعداد کاربران راننده"),
                               ("paragraph", "6,152,509 نفر"), ("paragraph", "تعداد کاربران راننده" + ZWNJ + "ی فعال در 1402"),
                               ("paragraph", "3,305,147 نفر")]),
        ("p2", "سرویس‌ها", [("caption", "جدول سرویس‌ها"),
                            ("table", "سرویس | تعداد راننده | سال\nخودرو | 6,152,509 | 1402\nباکس | 1,103,992 | 1402\nغذا | 134,006 | 1402")]),
        ("p3", "بازگویی", [("paragraph", "در سال 1400 تعداد کل راننده ها 3,800,000 بود.")]),
    ])
    dother = _doc("C.otherco_annual-report_1402.fa", "گزارش سالانه دیگرکو ۱۴۰۲", [
        ("p1", "آمار", [("paragraph", "تعداد کاربران راننده 99,999 نفر")]),
    ])
    return CorpusSet("C", [d1398, d1400, d1401, d1402, dother])


PLAN_OBJ = {"entity": "تست‌کو", "metric": "تعداد رانندگان", "kind": "timeline",
            "terms_fa": ["راننده", "رانندگان فعال", "کاربران راننده"], "terms_en": ["drivers", "active drivers"],
            "entity_terms": ["تست‌کو", "Testco"], "unit": "count", "period_from": "", "period_to": "",
            "companies": ["testco"]}


# --------------------------------------------------------------------------
# a scripted client (the real OllamaClient/ledger path, the HTTP call replaced)
# --------------------------------------------------------------------------

class ScriptedClient(OllamaClient):
    def __init__(self, cfg, ledger, responder, fail_at=None):
        super().__init__(cfg, ledger)
        self.responder = responder
        self.payloads = []
        self.fail_at = fail_at

    def _post(self, endpoint, payload):
        self.payloads.append(payload)
        if self.fail_at is not None and len(self.payloads) == self.fail_at:
            raise ModelError("simulated server restart")
        content = self.responder(payload)
        chars = sum(len(m["content"]) for m in payload["messages"])
        return ({"message": {"content": content}, "prompt_eval_count": chars // 4,
                 "eval_count": max(1, len(content) // 4), "done_reason": "stop"}, 1.0, 0.0, 1)


def _snippets(user):
    parts = re.split(r"\n\n(?=\[\d+\] \()", "\n\n" + user.split("Snippets:\n", 1)[1])
    out = []
    for p in parts:
        m = re.match(r"\[(\d+)\] \([^\n]*\)\n(.*)", p.strip(), re.S)
        if m:
            out.append((int(m.group(1)), m.group(2)))
    return out


RULES = [   # (must be in the snippet, item) - what a careful verifier would answer
    ("6,152,509 نفر", {"relevant": True, "value": "6,152,509", "unit": "نفر", "period": "1402",
                       "definition": "کاربران راننده", "quote": "تعداد کاربران راننده 6,152,509 نفر"}),
    ("3,305,147", {"relevant": True, "value": "3,305,147", "unit": "نفر", "period": "1402",
                   "definition": "کاربران راننده فعال", "quote": "تعداد کاربران راننده" + ZWNJ + "ی فعال در 1402 3,305,147 نفر"}),
    ("کل راننده ها 3,761,925", {"relevant": True, "value": "3,761,925", "unit": "", "period": "",
                                 "definition": "کل رانندگان", "quote": "کل راننده ها 3,761,925"}),
    ("1,896,210", {"relevant": True, "value": "1,896,210", "unit": "", "period": "1400",
                   "definition": "رانندگان فعال", "quote": "راننده های فعال رانندگان مرد 1,896,210"}),
    ("2میلیون", {"relevant": True, "value": "2,000,000", "unit": "نفر", "period": "پایان سال 98",
                 "definition": "رانندگان", "quote": "تعداد راننده‌های تست‌کو تا پایان سال 98 به بیش از 2میلیون نفر"}),
]


def verify_responder(payload, rules=RULES, wrong=False, omit=False):
    """Keyed reply {"s1": {"relevant", "values": [...]}, ...} as the v2 schema forces."""
    user = payload["messages"][-1]["content"]
    out = {}
    for i, text in _snippets(user):
        found = []
        for must, it in rules:
            if must in text and len(found) < 3:
                v = {k: it[k] for k in ("value", "unit", "period", "definition", "quote")}
                if wrong and it["value"] == "3,761,925":
                    v["value"] = "100,263"             # the nearby wrong number, with the right quote
                found.append(v)
        if not found and omit:
            continue                                   # a reply that skips snippets (the v1 live failure)
        out["s%d" % i] = {"relevant": bool(found), "values": found}
    return json.dumps(out, ensure_ascii=False)


def make_responder(plan=PLAN_OBJ, plan_text=None, answer="پاسخ", wrong=False, omit=False, shift=False):
    def respond(payload):
        system = payload["messages"][0]["content"]
        if system.startswith("You plan a search"):
            return plan_text if plan_text is not None else json.dumps(plan, ensure_ascii=False)
        if system.startswith("You check candidate snippets"):
            text = verify_responder(payload, wrong=wrong, omit=omit)
            if shift:     # every snippet's entry filed under the previous key (the live v2 failure)
                obj = json.loads(text)
                n = len(obj)
                text = json.dumps({"s%d" % i: obj.get("s%d" % (i + 1), {"relevant": False, "values": []})
                                   for i in range(1, n + 1)}, ensure_ascii=False)
            return text
        return answer
    return respond


# --------------------------------------------------------------------------
# tests
# --------------------------------------------------------------------------

def test_numbers():
    t = find_numbers("رانندگان مرد 979„379„4 نفر")[0]
    check(t.glitch and t.main == 4379979.0, "numbers.rtl_glitch_reversed_groups", t.readings)
    t = find_numbers("567„ 150 نفر")[0]
    check(t.main == 150567.0 and 567150.0 in t.readings, "numbers.rtl_glitch_ambiguous_keeps_both", t.readings)
    t = find_numbers(normalize("به بیش از ۲میلیون نفر"))[0]
    check(t.readings == [2.0, 2000000.0], "numbers.scale_word_glued", t.readings)
    toks = find_numbers("%97.4 و 1403/05/12 و سال 1402 و 4.4M")
    check(toks[0].percent and toks[1].is_date and toks[2].is_year and toks[3].main == 4.4e6, "numbers.percent_date_year_suffix",
          [x.to_dict() for x in toks])


def test_planner():
    obj, how = P.parse_plan_text('Here is the plan:\n```json\n{"entity": "x", "terms_fa": ["راننده"]}\n```')
    check(obj and how == "extract", "plan.parse_fenced_chatter", how)
    obj, how = P.parse_plan_text('{"entity": "اسنپ", "metric": "رانندگان", "terms_fa": ["راننده", "رانندگان فعال"')
    check(obj and how == "suffix_repair" and obj["terms_fa"] == ["راننده", "رانندگان فعال"], "plan.parse_cut_reply", (how, obj))
    obj, how = P.parse_plan_text("I cannot help with that")
    check(obj is None and P.plan_problem(obj), "plan.parse_garbage")
    check(P.plan_problem({"terms_fa": [], "terms_en": [""]}) is not None, "plan.problem_no_terms")
    metas = [doc_meta(d) for d in make_corpus().docs]
    plan = P.validate_plan({"kind": "trend", "unit": "people", "entity": "تست‌کو",
                            "terms_fa": ["رانندگان", "راننده|سفیر", "1402", "تست‌کو", "(کاربران راننده)"],
                            "terms_en": ["drivers", "drivers"], "period_from": "۱۳۹۸", "period_to": "1404",
                            "companies": ["testco", "nosuch"]},
                           question="تعداد رانندگان تست‌کو طی سال‌ها چگونه تغییر کرده است؟", q_lang="fa", metas=metas,
                           choose_companies=True)
    check(plan["kind"] == "timeline" and "kind_inferred" in plan["fixes"], "plan.validate_kind_inferred", plan["kind"])
    check(plan["unit"] == "count", "plan.validate_unit_inferred", plan["unit"])
    check(plan["terms"]["fa"] == ["رانندگان", "راننده سفیر", "کاربران راننده", "راننده", "سفیر", "کاربران"]
          and plan["terms"]["en"] == ["drivers"],
          "plan.validate_terms_literal_no_year_no_entity_dedupe_plus_head_words", plan["terms"])
    check(plan["period"] == (None, None) and "period_ignored_not_in_question" in plan["fixes"] and plan["companies"] == ["testco"],
          "plan.period_not_in_question_ignored_companies_kept", (plan["period"], plan["companies"], plan["fixes"]))
    plan_y = P.validate_plan({"kind": "timeline", "unit": "count", "terms_fa": ["راننده"], "period_from": "1399",
                              "period_to": "1402"}, question="تعداد رانندگان تست‌کو از سال ۱۳۹۹ تا ۱۴۰۲", q_lang="fa",
                             metas=metas, choose_companies=True)
    check(plan_y["period"] == (1399, 1402), "plan.period_kept_when_question_names_years", plan_y["period"])
    # the live Snapp plan (2026-09-16): entity names and question words inside terms
    live = P.validate_plan({"entity": "اسنپ", "kind": "timeline", "unit": "count",
                            "terms_fa": ["تعداد رانندگان اسنپ", "تغییر رانندگان اسنپ", "گروه اسنپ"],
                            "terms_en": ["Snapp drivers count", "Snapp driver change", "Snapp group"],
                            "entity_terms": ["اسنپ", "Snapp"]},
                           question="اسنپ چند راننده دارد و تعداد رانندگان آن در طول سال‌ها چگونه تغییر کرده است؟",
                           q_lang="fa", metas=metas, choose_companies=True)
    check(live["terms"]["fa"] == ["تعداد رانندگان", "رانندگان"] and not any("snapp" in t.lower() or "group" in t.lower()
                                                                            for t in live["terms"]["en"])
          and "گروه اسنپ" in live["term_refinement"]["dropped"], "plan.refine_terms_entity_and_generic_words_removed",
          (live["terms"], live["term_refinement"]))
    pp = P.program_plan("تعداد رانندگان تست‌کو طی سال‌ها چگونه تغییر کرده است؟", "fa", metas, True)
    check(pp["kind"] == "timeline" and pp["companies"] == ["testco"] and pp["terms"]["fa"], "plan.program_plan_fallback",
          (pp["kind"], pp["companies"], pp["terms"]))
    chosen, info = select_docs(metas, companies_chosen=["testco"], period=(1400, 1402), kind="timeline", question="")
    check([m["doc_id"] for m in chosen] == ["C.testco_annual-report_1400.fa", "C.testco_annual-report_1401.fa",
                                            "C.testco_annual-report_1402.fa"], "docmeta.select_company_and_period", info)
    msgs = P.plan_messages("سوال", metas, True)
    check("testco: 1398, 1400, 1401, 1402 | 4 documents" in msgs[1]["content"] and "otherco" in msgs[1]["content"],
          "plan.company_listing_in_prompt", msgs[1]["content"][:300])
    sch = P.plan_schema(["otherco", "testco"])
    check(sch["properties"]["companies"]["items"]["enum"] == ["otherco", "testco"], "plan.schema_company_enum")


def test_term_patterns():
    def any_match(pats, text):
        return any(re.search(p, text, re.I) for p in pats)
    pr = P.term_patterns("راننده", "fa")
    check(any_match(pr, "رانندگان مرد") and any_match(pr, "راننده" + ZWNJ + "ها") and any_match(pr, "کاربران راننده" + ZWNJ + "ی فعال"),
          "terms.fa_plural_stem_heh_gan", pr)
    pk = P.term_patterns("کاربران راننده", "fa")
    check(any_match(pk, "کاربر راننده") and any_match(pk, "کاربران" + ZWNJ + "راننده") and any_match(pk, "تعداد کاربران رانندگان"),
          "terms.fa_phrase_forms_and_joiners", pk)
    pe = P.term_patterns("active drivers", "en")
    check(any_match(pe, "Active driver accounts") and any_match(pe, "active-drivers"), "terms.en_plural_and_joiner", pe)
    ws = P.score_words({"fa": ["تعداد رانندگان"]})
    check([w for w, _rx, weak in ws if weak] == ["تعداد"] and [w for w, _rx, weak in ws if not weak] == ["رانندگان"],
          "terms.weak_measure_words", [(w, weak) for w, _r, weak in ws])


def test_windows(cfg, root, cs):
    plan = P.validate_plan(PLAN_OBJ, question="تعداد رانندگان تست‌کو طی سال‌ها چگونه تغییر کرده است؟", q_lang="fa",
                           metas=[doc_meta(d) for d in cs.docs], choose_companies=True)
    metas = [doc_meta(d) for d in cs.docs if d.id.startswith("C.testco")]
    opts = dict(DEFAULTS)
    cands, info = S.search(root, cs, plan, metas, d_lang="fa", question="تعداد رانندگان", opts=opts, cfg=cfg)
    texts = [c.text for c in cands]
    w1402 = [c for c in cands if c.doc_id.endswith("1402.fa") and "تعداد کاربران راننده" in c.text and "6,152,509" in c.text]
    check(bool(w1402), "windows.detached_number_after_label (1402)", texts)
    w1400 = [c for c in cands if c.doc_id.endswith("1400.fa") and "راننده های فعال" in c.text and "1,896,210" in c.text]
    check(bool(w1400), "windows.scattered_infographic_labels (1400)", texts)
    w1401 = [c for c in cands if c.doc_id.endswith("1401.fa") and "979„379„4" in c.text]
    check(bool(w1401), "windows.rtl_glitch_numbers (1401)", texts)
    clip = [c for c in cands if c.doc_id.endswith("1398.fa")]
    check(bool(clip) and clip[0].kind == "clip" and "2میلیون" in clip[0].text and len(clip[0].text) <= opts["clip_chars"],
          "windows.long_unit_clip_keeps_term_and_number", [(c.kind, len(c.text)) for c in clip])
    tab = [c for c in cands if c.kind == "table"]
    check(bool(tab) and tab[0].text.startswith("جدول سرویس‌ها\n\nسرویس | تعداد راننده | سال"), "windows.table_caption_header_rows",
          tab[0].text if tab else None)
    doc = cs.doc(w1402[0].doc_id) if w1402 else None
    check(bool(w1402) and doc.text[w1402[0].span[0]:w1402[0].span[1]] == w1402[0].text, "windows.verbatim_span")
    check(all(c.doc_id.startswith("C.testco") for c in cands), "windows.only_target_docs")
    per = {}
    for c in cands:
        per[c.doc_id] = per.get(c.doc_id, 0) + 1
    check(max(per.values()) <= opts["per_doc_cap"] and len(per) == 4, "windows.per_doc_cap_round_robin", per)
    check(cands[0].doc_id.endswith("1398.fa") and cands[1].doc_id.endswith("1400.fa"), "windows.round_robin_year_order",
          [c.doc_id for c in cands[:4]])
    # the line_windows ablation: the hit unit alone never holds the detached number
    lc, _ = S.search(root, cs, plan, metas, d_lang="fa", question="تعداد رانندگان", opts=opts, line_only=True, cfg=cfg)
    check(not any("تعداد کاربران راننده" in c.text and "6,152,509" in c.text and c.kind == "window" for c in lc),
          "windows.line_only_misses_detached_numbers", [c.text for c in lc if c.doc_id.endswith("1402.fa")])
    # widened gap round: bigger windows, excludes already-seen units
    seen = set()
    for c in cands:
        seen.update(c.number_unit_ids)
        seen.update(c.hit_unit_ids)
    g, ginfo = S.search(root, cs, plan, metas, d_lang="fa", question="تعداد رانندگان تست‌کو", opts=opts, round_no=2,
                        per_doc_cap=3, global_cap=8, exclude_units=seen, only_docs={"C.testco_annual-report_1402.fa"}, cfg=cfg)
    check(all(c.doc_id.endswith("1402.fa") for c in g) and ginfo.get("excluded_seen", 0) >= 1, "windows.gap_round_only_missing_docs_excludes_seen",
          ginfo)


def test_guard():
    snip = "کل راننده ها 3,761,925\n\nرانندگان زن 100,263\n\nراننده های فعال\n\nرانندگان مرد\n\n1,896,210\n\n3,661,662"
    ok = {"value": "3,761,925", "quote": "کل راننده ها 3,761,925"}
    r = V.guard(ok, snip, unit_kind="count")
    check(r[0] and r[2]["value_num"] == 3761925.0, "guard.accepts_verbatim", r)
    r = V.guard({"value": "3761925", "quote": "کل راننده‌ها 3761925"}, snip, unit_kind="count")
    check(r[0], "guard.accepts_separator_and_zwnj_differences", r)
    r = V.guard({"value": "3,761,925", "quote": "تعداد کل رانندگان 3,761,925"}, snip, unit_kind="count")
    check(not r[0] and r[1] == "quote_not_in_snippet", "guard.rejects_paraphrased_quote", r)
    r = V.guard({"value": "3,761,952", "quote": "کل راننده ها 3,761,925"}, snip, unit_kind="count")
    check(not r[0] and r[1] == "value_not_in_snippet", "guard.rejects_changed_digits", r)
    r = V.guard({"value": "100,263", "quote": "کل راننده ها 3,761,925"}, snip, unit_kind="count")
    check(not r[0] and r[1] == "value_not_in_quote", "guard.rejects_nearby_wrong_number", r)
    r = V.guard({"value": "3.8 میلیون", "quote": "کل راننده ها 3,761,925"}, snip, unit_kind="count")
    check(not r[0] and r[1] == "value_not_in_snippet", "guard.rejects_rounded_value", r)
    r = V.guard({"value": "3,761,925", "quote": "3,761,925"}, snip, unit_kind="count")
    check(not r[0] and r[1] == "quote_no_label", "guard.rejects_quote_without_label", r)
    r = V.guard({"value": "1402", "quote": "فعال در 1402"}, "تعداد کاربران راننده فعال در 1402", unit_kind="count")
    check(not r[0] and r[1] == "value_is_year", "guard.rejects_year_as_value", r)
    r = V.guard({"value": "3,761,925 و 100,263", "quote": "کل راننده ها 3,761,925 رانندگان زن 100,263"}, snip, unit_kind="count")
    check(not r[0] and r[1] == "value_has_several_numbers", "guard.rejects_summed_or_listed_values", r)
    g = "رانندگان مرد\n\n979„379„4 نفر"
    r = V.guard({"value": "4,379,979", "quote": "رانندگان مرد 979„379„4 نفر"}, g, unit_kind="count")
    check(r[0] and r[2]["value_num"] == 4379979.0 and r[2]["reading"] == "reversed_groups", "guard.rtl_glitch_reading", r)
    r = V.guard({"value": "2,000,000", "quote": "به بیش از 2میلیون نفر"}, "رانندگان تا پایان سال 98 به بیش از 2میلیون نفر می‌رسد",
                unit_kind="count")
    check(r[0] and r[2]["value_num"] == 2e6, "guard.scale_word_reading", r)
    r = V.guard({"value": "8 میکروگرم", "quote": "حد مجاز 8 میکروگرم بر کیلوگرم"}, "حد مجاز 8 میکروگرم بر کیلوگرم است", unit_kind="text")
    check(r[0], "guard.text_question_with_number", r)
    r = V.guard({"value": "صید دریایی", "quote": "سه دسته: صید دریایی، سخت‌پوستان"}, "دقیقاً سه دسته: صید دریایی، سخت‌پوستان پرورشی", unit_kind="text")
    check(r[0], "guard.text_value_in_quote", r)
    r = V.guard({"value": "صید رودخانه‌ای", "quote": "سه دسته: صید دریایی"}, "دقیقاً سه دسته: صید دریایی، سخت‌پوستان پرورشی", unit_kind="text")
    check(not r[0] and r[1] == "text_value_not_in_snippet", "guard.text_value_invented", r)
    term_rx = [re.compile(p) for p in P.term_patterns("راننده", "fa")]
    staff = "خانواده ی 6,561 نفری گروه اسنپ با همراهی میلیون ها راننده"
    r = V.guard({"value": "6,561", "quote": "خانواده ی 6,561 نفری گروه اسنپ"}, staff, unit_kind="count", term_rx=term_rx)
    check(not r[0] and r[1] == "quote_without_term", "guard.rejects_number_labelled_with_another_metric", r)
    r = V.guard({"value": "3,761,925", "quote": "کل راننده ها 3,761,925"}, snip, unit_kind="count", term_rx=term_rx)
    check(r[0], "guard.term_guard_accepts_metric_label", r)
    items, how = V.parse_items('{"items": [{"id": 1, "relevant": true, "value": "5", "quote": "x 5"}, {"id": 2, "relev', 3)
    check(how == "salvaged" and list(items) == [1], "verify.parse_cut_reply_salvages_complete_items", (how, items))
    items, how = V.parse_items('{"s1": {"relevant": false, "values": []}, "s2": {"relevant": true, "values": [{"value": "5", '
                               '"unit": "", "period": "", "definition": "a", "quote": "x 5"}, {"value": "7", "unit": "", '
                               '"period": "", "definition": "b", "quote": "y 7"}]}, "s3": {"relevant": tr', 3)
    check(how == "salvaged" and sorted(items) == [1, 2] and len(items[2]) == 2 and items[2][1]["value"] == "7"
          and items[1][0]["relevant"] is False, "verify.parse_keyed_reply_several_values_cut_tail", (how, items))
    sch = V.verify_schema(3)
    check(sch["required"] == ["s1", "s2", "s3"] and sch["properties"]["s2"]["properties"]["values"]["maxItems"] == 3,
          "verify.schema_requires_every_snippet_key")
    axis = "400,000,000\n\nمسافر\n\n350,000,000\n\nراننده\n\n300,000,000\n\n250,000,000"
    r = V.guard({"value": "350,000,000", "quote": "350,000,000\n\nراننده"}, axis, unit_kind="count", term_rx=term_rx)
    check(not r[0] and r[1] == "value_is_axis_tick", "guard.rejects_chart_axis_numbers", r)
    r = V.guard({"value": "6,152,509", "quote": "تعداد کاربران راننده 6,152,509"},
                "تعداد کاربران راننده 6,152,509\n\nفعال 3,305,147\n\nزن 233,214", unit_kind="count", term_rx=term_rx)
    check(r[0], "guard.axis_check_ignores_irregular_numbers", r)


class _C:
    def __init__(self, cid, doc_id, score=1.0):
        self.cid, self.doc_id, self.section_id, self.unit_ids, self.score, self.round = cid, doc_id, "s", ["u"], score, 1


def _row(value, quote, definition, period, doc_year, cid, score=1.0, value_num=None):
    item = {"value": value, "quote": quote, "definition": definition, "period": period, "unit": "نفر"}
    meta = {"years": (doc_year, doc_year)}
    num = value_num if value_num is not None else float(value.replace(",", ""))
    return R.make_row(item, _C(cid, "C.x_annual-report_%d.fa" % doc_year, score), {"value_num": num, "value_token": value},
                      meta, "u")


def test_reconcile():
    rows = [
        _row("2,431,581", "تعداد رانندگانی که در اسنپ ثبت‌نام کرده‌اند 2,431,581", "رانندگان ثبت‌نام‌شده", "", 1399, "a"),
        _row("3,761,925", "کل راننده ها 3,761,925", "کل رانندگان", "", 1400, "b"),
        _row("1,896,210", "راننده های فعال 1,896,210", "رانندگان فعال", "1400", 1400, "c"),
        _row("100,263", "رانندگان زن 100,263", "رانندگان زن", "1400", 1400, "d"),
        _row("3,305,147", "تعداد کاربران راننده‌ی فعال در 1402 3,305,147 نفر", "فعال", "1402", 1402, "e"),
        _row("3,800,000", "در سال 1400 تعداد کل راننده ها 3,800,000 بود", "کل رانندگان", "سال 1400", 1402, "f"),
        _row("3,692,657", "تعداد کاربران راننده ی فعال* 3,692,657", "فعال", "", 1403, "g"),
    ]
    check([r["role"] for r in rows] == ["authoritative"] * 5 + ["restatement", "authoritative"], "reconcile.roles",
          [r["role"] for r in rows])
    check(rows[0]["def_class"] == "registered" and rows[1]["def_class"] == "total" and rows[2]["def_class"] == "active"
          and rows[3]["def_class"] == "subgroup", "reconcile.definition_classes_from_quote",
          [r["def_class"] for r in rows])
    rec = R.reconcile(rows, kind="timeline", target_periods=[1399, 1400, 1401, 1402, 1403], opts=DEFAULTS, q_lang="fa")
    periods = [(r["period"], r["value"]) for r in rec["series"]]
    check(periods == [("1399", "2,431,581"), ("1400", "3,761,925"), ("1400", "1,896,210"), ("1402", "3,305,147"),
                      ("1403", "3,692,657")] or sorted(periods) == sorted([("1399", "2,431,581"), ("1400", "3,761,925"),
                                                                          ("1400", "1,896,210"), ("1402", "3,305,147"),
                                                                          ("1403", "3,692,657")]),
          "reconcile.series_authoritative_no_subgroup_no_restatement", periods)
    check(rec["conflicts"] and rec["conflicts"][0]["restated"] == "3,800,000" and rec["conflicts"][0]["kept"] == ["3,761,925"],
          "reconcile.restatement_conflict_noted", rec["conflicts"])
    check(rec["comparable"] is False and rec["trend"] is None, "reconcile.not_comparable_no_trend",
          (rec["comparable"], rec["definition_classes"]))
    check(rec["periods_missing"] == ["1401"], "reconcile.missing_period", rec["periods_missing"])
    text = R.table_text(rec, "fa", {"a": 2, "b": 3, "c": 4, "e": 5, "g": 6})
    check("یک سری قابل مقایسه نیستند" in text and "| [3]" in text and "1401" in text and "فعال: 1400، 1402، 1403" in text,
          "reconcile.table_text_notes_and_citations", text)
    same = [r for r in rows if r["def_class"] == "active"]
    rec2 = R.reconcile(same, kind="timeline", target_periods=[1400, 1402, 1403], opts=DEFAULTS, q_lang="en")
    check(rec2["comparable"] is True and rec2["trend"] and rec2["trend"]["direction"] == "up"
          and rec2["trend"]["from"] == ["1400", "1,896,210"], "reconcile.comparable_series_trend", rec2["trend"])
    only_rest = [_row("3,800,000", "در سال 1400 تعداد کل راننده ها 3,800,000 بود", "کل", "1400", 1402, "f")]
    rec3 = R.reconcile(only_rest, kind="timeline", target_periods=[1400], opts=DEFAULTS, q_lang="en")
    check(rec3["series"] and rec3["series"][0]["role"] == "latest_restatement" and rec3["latest_restatement_periods"] == [1400],
          "reconcile.restatement_used_when_period_report_silent", rec3["series"])
    single = R.reconcile(rows[:3], kind="single", opts=dict(DEFAULTS, single_rows_max=2), q_lang="fa")
    check(len(single["series"]) == 2 and single["comparable"] is None, "reconcile.single_fact_table_rows", single["series"])
    known = {"راننده", "رانندگان", "کاربر", "کاربران", "فعال", "اسنپ"}
    nar = [_row("2,000,000", "تعداد راننده‌ها به بیش از 2,000,000 نفر", "تعداد رانندگان اسنپ", "1398", 1398, "p"),
           _row("11,000", "بیش از 11,000 راننده دارای معلولیت", "راننده دارای معلولیت در اسنپ", "1398", 1398, "q"),
           _row("3,000,000", "خدمات به 3,000,000 کاربر راننده", "کاربر راننده", "1400", 1400, "r"),
           _row("15,000", "15,000 کاربر راننده ی معاف از کمیسیون", "کاربر راننده معاف از کمیسیون", "1400", 1400, "s")]
    recn = R.reconcile(nar, kind="timeline", target_periods=[1398, 1400], opts=DEFAULTS, q_lang="fa", known_stems=known)
    check(sorted(r["value"] for r in recn["series"]) == ["2,000,000", "3,000,000"]
          and sorted(r["value"] for r in recn["narrower"]) == ["11,000", "15,000"],
          "reconcile.narrower_rows_leave_the_series", ([r["value"] for r in recn["series"]], [r["qualifiers"] for r in nar]))
    p = R.parse_period("پایان سال 98", "", (1398, 1398))
    check(p == (1398, "period_short_year"), "reconcile.period_short_year", p)
    p = R.parse_period("", "در سال 1396 شروع شد", (1403, 1403))
    check(p == (1403, "doc_year"), "reconcile.period_quote_year_outside_doc_ignored", p)


def test_budget_flags():
    b = CallBudget(3)
    b.spend("plan")
    check(b.can(1, reserve=1) and not b.can(2, reserve=1), "budget.can_with_reserve")
    b.spend("verify")
    b.spend("answer")
    try:
        b.spend("extra")
        check(False, "budget.raises_past_cap")
    except BudgetExceeded:
        check(True, "budget.raises_past_cap")
    check(parse_flags("no_guard,synth_answer") == {"no_guard": True, "synth_answer": True}, "flags.parse")
    try:
        parse_flags({"granularity": "x"})
        check(False, "flags.unknown_rejected")
    except ValueError:
        check(True, "flags.unknown_rejected")


def _question(scope="corpus"):
    return Question(id="t-timeline", corpus="C", question_fa="تعداد رانندگان تست‌کو طی سال‌ها چگونه تغییر کرده است؟",
                    question_en="How has the number of Testco drivers changed over the years?", answer_fa="", answer_en="",
                    answer_type="short", gold_units={"fa": ["C.testco_annual-report_1402.fa::p1_b2"]}, doc_scope=scope,
                    type="cross-year")


def _approach(cfg, cs, responder, options=None, fail_at=None, flags=None):
    ledger = Ledger(path=None)
    client = ScriptedClient(cfg, ledger, responder, fail_at=fail_at)
    ap = AgenticPipeline(cfg=cfg, client=client, flags=flags, options=options)
    ap.build_index(cs, ledger)
    return ap, client


def test_pipeline(cfg, cs, tmp):
    ap, client = _approach(cfg, cs, make_responder(), options={"call_cache": "off"})
    rec = ap.answer(_question(), "fa-fa")
    tr = rec.trace
    check(not rec.error, "pipeline.no_error", rec.error or "")
    st = tr.get("stats") or {}
    check(rec.llm_calls == st.get("calls") == tr.get("ledger_calls") and rec.llm_calls <= 8, "pipeline.calls_from_ledger_within_cap",
          (rec.llm_calls, st.get("calls"), tr.get("ledger_calls")))
    check(tr["calls"][0]["purpose"] == "plan" and tr["calls"][-1]["purpose"] == "answer", "pipeline.plan_first_answer_last",
          [c["purpose"] for c in tr["calls"]])
    tm = tr["timing"]
    check([s["stage"] for s in tm["stages"]] == ["plan", "search", "verify", "gap_fill", "reconcile", "answer"],
          "timing.stage_records", [s["stage"] for s in tm["stages"]])
    check(tm["reconciles"] and tm["totals"]["llm_calls"] == rec.llm_calls and tm["totals"]["prompt_tokens"] == rec.prompt_tokens
          and tm["totals"]["completion_tokens"] == rec.completion_tokens and abs(tm["totals"]["llm_wall_ms"] - rec.llm_wall_ms) <= 1.0,
          "timing.stage_totals_reconcile_with_ledger", (tm["totals"], rec.llm_calls, rec.prompt_tokens))
    srch = tm["stages"][1]
    check(srch["llm_calls"] == 0 and srch["rg_ms"] > 0 and srch["hit_lines"] > 0 and srch["selected"] == tr["stats"]["candidates"],
          "timing.search_stage_rg_ms_and_counts", srch)
    vb = tr["stages"]["verify"]["batches"]
    check(vb and all("wall_ms" in b and b["llm_calls"] == 1 and "guard_ms" in b for b in vb)
          and sum(b["accepted"] for b in vb) == tm["stages"][2]["accepted"], "timing.verify_batch_records", vb[0] if vb else None)
    staged = [r for r in client.ledger.records if r.get("question_id") == "t-timeline" and r.get("kind") == "chat"]
    check(staged and all(r.get("stage") in ("plan", "verify", "gap_fill", "answer") for r in staged),
          "timing.ledger_records_carry_stage", [r.get("stage") for r in staged])
    sel = tr["stages"]["plan"]["doc_selection"]
    check("C.otherco_annual-report_1402.fa" not in sel["chosen"] and len(sel["chosen"]) == 4, "pipeline.doc_selection_by_company",
          sel)
    series = tr["structured"]["series"]
    check(series and all(all(k in r for k in SERIES_FIELDS) for r in series), "pipeline.structured_series_fields",
          series[:1])
    vals = {(r["period"], r["value"]) for r in series}
    check(("1402", "6,152,509") in vals and ("1400", "1,896,210") in vals and ("1398", "2میلیون") in vals,
          "pipeline.series_values", sorted(vals))
    check(tr["structured"]["comparable"] is False, "pipeline.not_comparable_detected", tr["structured"]["definition_classes"])
    try:   # the timeline scorer (eval/timeline.py, owned by the evaluation agent) reads this structured output
        from ...eval import timeline as TL
        st_of = TL.structured_of(json.loads(json.dumps(rec.to_dict(), ensure_ascii=False, default=str)))
        ser, _items = TL.structured_series(st_of, known_periods=("1398", "1400", "1401", "1402"))
        base = {p: sorted(m.value for m in ms) for p, ms in ser.items()}
        check(base.get("1398") == [2000000.0] and 6152509.0 in base.get("1402", []) and 1896210.0 in base.get("1400", [])
              and TL.flags_definition_change("", structured=st_of) and not TL.states_trend("", structured=st_of),
              "pipeline.timeline_scorer_reads_structured_series", base)
    except ImportError as e:
        print("  skip timeline scorer check (%s)" % e)
    check(rec.evidence_items and rec.evidence_items[0]["kind"] == "fact" and rec.evidence_chars <= cfg.evidence["max_chars"],
          "pipeline.evidence_table_first_within_budget", rec.evidence_items[:2])
    check(tr["stages"]["answer"]["table"] and "[2]" in tr["stages"]["answer"]["table"], "pipeline.table_cites_windows",
          tr["stages"]["answer"]["table"])
    ans_payload = client.payloads[-1]
    check(ans_payload["messages"][0]["content"].startswith("You answer questions about company") and
          ans_payload["options"]["num_predict"] == cfg.generation["num_predict_answer"], "pipeline.shared_answer_prompt")
    check(rec.structured is not None and rec.structured.get("series") == tr["structured"]["series"],
          "pipeline.runrecord_structured_field", type(rec.structured))
    # a question of type timeline gets the shared TIMELINE answer prompt and its budget (contracts, 2026-09-16)
    from ...contracts import ANSWER_TIMELINE_SYSTEM, answer_num_predict
    qt = _question()
    qt.type, qt.answer_type = "timeline", "series"
    apt, clientt = _approach(cfg, cs, make_responder(), options={"call_cache": "off"})
    rect = apt.answer(qt, "fa-fa")
    pt = clientt.payloads[-1]
    check(not rect.error and pt["messages"][0]["content"].startswith(ANSWER_TIMELINE_SYSTEM.split("{source_rule}")[0]) and
          "one line per period" in pt["messages"][0]["content"] and pt["options"]["num_predict"] == answer_num_predict("timeline", cfg)
          and answer_num_predict("timeline", cfg) != cfg.generation["num_predict_answer"]
          and rect.meta.get("answer_prompt") == "answer-timeline-v1",
          "pipeline.timeline_question_uses_shared_timeline_prompt", (pt["options"].get("num_predict"), rect.meta, rect.error))
    check(getattr(rec, "retrieved_units") and rec.retrieved_units[0]["meta"]["verdict"] == "accepted", "pipeline.retrieved_accepted_first")
    check(tr["stages"]["gap_fill"]["missing_docs"] == ["C.testco_annual-report_1401.fa"], "pipeline.gap_targets_missing_period",
          tr["stages"]["gap_fill"]["missing_docs"])

    # the guard catches the nearby wrong number that the (scripted) verifier returns with the right quote
    ap2, _ = _approach(cfg, cs, make_responder(wrong=True), options={"call_cache": "off"})
    rec2 = ap2.answer(_question(), "fa-fa")
    check(rec2.trace["guard_reasons"].get("value_not_in_quote", 0) >= 1 and
          ("1400", "100,263") not in {(r["period"], r["value"]) for r in rec2.trace["structured"]["series"]},
          "pipeline.guard_rejects_nearby_wrong_number", rec2.trace["guard_reasons"])
    ap3, _ = _approach(cfg, cs, make_responder(wrong=True), options={"call_cache": "off"}, flags={"no_guard": True})
    rec3 = ap3.answer(_question(), "fa-fa")
    check(("1400", "100,263") in {(r["period"], r["value"]) for r in rec3.trace["structured"]["series"]} and rec3.approach == "AGENTIC+no_guard",
          "pipeline.no_guard_ablation_lets_it_through", rec3.trace["structured"]["series"])

    # a verifier that lists only the snippets it accepts: the others are model_omitted, not unparsed
    apo, _ = _approach(cfg, cs, make_responder(omit=True), options={"call_cache": "off"})
    reco = apo.answer(_question(), "fa-fa")
    so = reco.trace["stats"]
    check(so["model_omitted"] > 0 and so["unparsed"] == 0 and so["accepted"] >= 4, "pipeline.omitted_snippets_not_unparsed",
          {k: so[k] for k in ("accepted", "model_omitted", "unparsed", "model_rejected")})

    # values filed under the neighbouring snippet key: re-attributed only where the whole guard passes
    aps, _ = _approach(cfg, cs, make_responder(shift=True), options={"call_cache": "off"})
    recs_ = aps.answer(_question(), "fa-fa")
    ss = recs_.trace["stats"]
    vals_s = {(r["period"], r["value"]) for r in recs_.trace["structured"]["series"]}
    check(ss["reattributed"] >= 1 and ("1402", "6,152,509") in vals_s, "pipeline.shifted_keys_reattributed_by_guard",
          ({k: ss[k] for k in ("accepted", "reattributed", "guard_rejected")}, sorted(vals_s)))
    apn, _ = _approach(cfg, cs, make_responder(shift=True), options={"call_cache": "off", "reattribute": False})
    recn_ = apn.answer(_question(), "fa-fa")
    check(recn_.trace["stats"]["reattributed"] == 0 and recn_.trace["guard_reasons"].get("quote_not_in_snippet", 0) >= 1,
          "pipeline.shifted_keys_rejected_without_reattribution", recn_.trace["guard_reasons"])

    # a plan reply that is not JSON: one repair call, then a program plan
    ap4, c4 = _approach(cfg, cs, make_responder(plan_text="sorry, no JSON"), options={"call_cache": "off"})
    rec4 = ap4.answer(_question(), "fa-fa")
    purposes = [c["purpose"] for c in rec4.trace["calls"]]
    check(purposes[:2] == ["plan", "plan_repair"] and rec4.trace["stats"]["plan_program"] == 1 and rec4.llm_calls <= 8,
          "pipeline.plan_repair_then_program_plan", purposes)

    # budget: a tight cap still leaves the answer call, skipped candidates are recorded
    # (config caps.agentic_max_calls wins over DEFAULTS max_calls_default since the registration patch: set both)
    ap5, _ = _approach(cfg.copy_with({"caps": {"agentic_max_calls": 4}}), cs, make_responder(),
                       options={"call_cache": "off", "max_calls_default": 4, "verify_batch_size": 2})
    rec5 = ap5.answer(_question(), "fa-fa")
    p5 = [c["purpose"] for c in rec5.trace["calls"]]
    check(rec5.llm_calls == 4 and p5[-1] == "answer" and rec5.trace["stats"]["candidates_unverified_budget"] > 0,
          "budget.tight_cap_keeps_answer_call", (rec5.llm_calls, p5, rec5.trace["stats"]["candidates_unverified_budget"]))
    ap6, _ = _approach(cfg.copy_with({"caps": {"agentic_max_calls": 3}}), cs, make_responder(),
                       options={"call_cache": "off", "max_calls_default": 3, "verify_batch_size": 2})
    rec6 = ap6.answer(_question(), "fa-fa")
    check(rec6.llm_calls == 3 and [c["purpose"] for c in rec6.trace["calls"]] == ["plan", "verify", "answer"],
          "budget.cap3_plan_verify_answer", [c["purpose"] for c in rec6.trace["calls"]])

    # per-call cache: a rerun replays every call; a failure mid-question repeats only the call in flight
    cache_path = tmp / "calls.sqlite"
    if cache_path.exists():
        cache_path.unlink()
    ap7, c7 = _approach(cfg, cs, make_responder(), options={"call_cache_path": str(cache_path)}, fail_at=3)
    rec7 = ap7.answer(_question(), "fa-fa")
    check(bool(rec7.error) and "simulated" in rec7.error, "cache.first_attempt_fails_at_call_3", rec7.error)
    ap8, c8 = _approach(cfg, cs, make_responder(), options={"call_cache_path": str(cache_path)})
    rec8 = ap8.answer(_question(), "fa-fa")
    check(not rec8.error and rec8.trace["stats"]["cache_hits"] == 2 and len(c8.payloads) == rec8.llm_calls - 2,
          "cache.resume_repeats_only_the_call_in_flight", (rec8.trace["stats"]["cache_hits"], len(c8.payloads), rec8.llm_calls))
    ap9, c9 = _approach(cfg, cs, make_responder(), options={"call_cache_path": str(cache_path)})
    rec9 = ap9.answer(_question(), "fa-fa")
    check(len(c9.payloads) == 0 and rec9.trace["stats"]["cache_hits"] == rec9.llm_calls == rec8.llm_calls
          and rec9.answer_text == rec8.answer_text and rec9.prompt_tokens == rec8.prompt_tokens,
          "cache.full_replay_same_record_costs", (len(c9.payloads), rec9.llm_calls, rec9.prompt_tokens, rec8.prompt_tokens))
    check(rec9.trace["timing"]["reconciles"] and rec9.trace["timing"]["totals"]["cache_hits"] == rec9.llm_calls
          and all(r.get("cache_hit") and r.get("stage") for r in c9.ledger.records if r.get("kind") == "chat"),
          "cache.replayed_calls_ledgered_with_stage_and_flag", rec9.trace["timing"]["totals"])

    # single-document scope, single fact, synth answer ablation
    q = Question(id="t-single", corpus="C", question_fa="در گزارش 1402 تست‌کو تعداد کاربران راننده فعال چند نفر است؟",
                 question_en="", answer_fa="3,305,147", answer_en="", answer_type="numeric",
                 gold_units={"fa": ["C.testco_annual-report_1402.fa::p1_b4"]}, doc_scope=["C.testco_annual-report_1402.fa"],
                 type="lookup", answer_value=3305147)
    plan_single = dict(PLAN_OBJ, kind="single", terms_fa=["کاربران راننده فعال", "راننده"], companies=[])
    ap10, c10 = _approach(cfg, cs, make_responder(plan=plan_single), options={"call_cache": "off"}, flags={"synth_answer": True})
    rec10 = ap10.answer(q, "fa-fa")
    s10 = rec10.trace["structured"]
    check(not rec10.error and s10["kind"] == "single" and s10["series"] and rec10.trace["stages"]["plan"]["doc_selection"]["rule"] == "single_doc_scope",
          "pipeline.single_doc_single_fact", (rec10.error, s10["series"][:2]))
    check(c10.payloads[-1]["messages"][0]["content"].startswith("You answer questions about company, market and sector reports.\nUse only")
          and "Passage [1] is a table" in c10.payloads[-1]["messages"][0]["content"], "pipeline.synth_answer_prompt")
    check("company_listing" not in json.dumps(c10.payloads[0]["messages"], ensure_ascii=False) and
          "companies" not in c10.payloads[0]["format"]["properties"], "pipeline.single_doc_plan_has_no_company_choice")


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass
    base = load_config()
    tmp = base.project_root / "work" / "test_agentic"
    if tmp.exists():
        shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True, exist_ok=True)
    cfg = base.copy_with({"paths": {"index_root": str(tmp / "indexes")}})
    cs = make_corpus()
    root = build_plain_store(cs, root=tmp / "store", cfg=cfg)
    print("numbers");   test_numbers()
    print("planner");   test_planner()
    print("terms");     test_term_patterns()
    print("windows");   test_windows(cfg, root, cs)
    print("guard");     test_guard()
    print("reconcile"); test_reconcile()
    print("budget");    test_budget_flags()
    print("pipeline");  test_pipeline(cfg, cs, tmp)
    print("\n%d checks, %d failure(s)" % (N[0], len(FAILS)))
    for f in FAILS:
        print("  FAILED:", f)
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
