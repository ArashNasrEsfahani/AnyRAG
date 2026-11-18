"""Pilot question set test (offline; no model calls, no OpenRouter).

    .venv/Scripts/python.exe -m ragbench.tests.test_questions

Checks the matching key, that pilot.jsonl validates with zero errors against the current
corpora, that the validator catches deliberately broken copies (missing gold id, broken fa/en
alignment, a gold unit that does not contain its quote, English gold on corpus C, a numeric
answer without a value), and that a rebuild from pilot.src.yaml reproduces pilot.jsonl.
The DEV timeline questions (pilot_timeline.jsonl) get the same treatment: clean validation, composition,
reproducible build, and broken copies (value not in its unit, period order, duplicate period, doc outside the
scope, a unit from another report, a restatement from the period's own report, missing notes, gold union).
Run from D:/KareOne/RAG Paper.
"""
import copy
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[2]))

import yaml  # noqa: E402

from ragbench.config import load_config  # noqa: E402
from ragbench.contracts import ALL_SETTINGS, Question, load_questions  # noqa: E402
from ragbench.corpus.model import swap_lang  # noqa: E402
from ragbench.tools import questions as T  # noqa: E402

RESULTS = {}


def check(name, cond, detail=""):
    RESULTS[name] = bool(cond)
    print(("PASS " if cond else "FAIL ") + name + (" - " + str(detail) if detail else ""), flush=True)
    return cond


def errors_for(qdicts, corp, levels=("error",)):
    qs = [Question.from_dict(d) for d in qdicts]
    rep = T.Report()
    T.validate_questions(qs, qdicts, corp, rep)
    return [(qid, msg) for lv, qid, msg in rep.items if lv in levels]


def test_timeline(corp, root):
    """pilot_timeline.jsonl (DEV timeline questions): validates clean, rebuilds identically, and the timeline
    validator catches broken copies."""
    from ragbench.tools import timeline_questions as TQ
    path = root / "pilot_timeline.jsonl"
    raw = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    qs = load_questions(path)
    probs = errors_for(raw, corp, ("error", "warn"))
    check("timeline.validates_clean", not probs, probs[:3])
    comp = TQ.timeline_composition(qs)
    check("timeline.composition", all(r[4] for r in comp) and len(qs) == 8, comp)
    check("timeline.dev_split_name", path.name.startswith("pilot"))
    check("timeline.C_supports_fa_docs_only", all(q.supports("fa-fa") and q.supports("en-fa") and not q.supports("en-en") for q in qs))
    with open(root / "pilot_timeline.src.yaml", encoding="utf-8") as fh:
        specs = yaml.safe_load(fh)["questions"]
    rebuilt = [T.build_question(s, corp, source="pilot_timeline.src.yaml")[0] for s in specs]
    same = json.loads(json.dumps(rebuilt, ensure_ascii=False)) == raw
    check("timeline.build_reproducible", same, "" if same else "pilot_timeline.jsonl differs from its source (run build)")
    by_id = {d["id"]: i for i, d in enumerate(raw)}

    def broken(qid, mutate, levels=("error",)):
        dd = copy.deepcopy(raw)
        mutate(dd[by_id[qid]])
        return [m for i, m in errors_for(dd, corp, levels) if i == qid]

    def value_not_in_unit(d):
        s = d["meta"]["answer_series"][0]
        s["value"], s["value_base"] = 351, 351.0
    check("timeline.catch_value_not_in_gold_unit", any("does not contain the value" in m for m in broken("C-TL-divar-01", value_not_in_unit)))

    def unordered(d):
        s = d["meta"]["answer_series"]
        s[0], s[1] = s[1], s[0]
    check("timeline.catch_period_order", any("chronological" in m for m in broken("C-TL-divar-01", unordered)))

    def duplicate(d):
        d["meta"]["answer_series"][1]["period"] = d["meta"]["answer_series"][0]["period"]
    check("timeline.catch_duplicate_period", any("not unique" in m for m in broken("C-TL-divar-02", duplicate)))

    def scope(d):
        d["doc_scope"] = d["doc_scope"][1:]
    check("timeline.catch_doc_not_in_scope", any("not in doc_scope" in m for m in broken("C-TL-divar-01", scope)))

    def foreign_unit(d):
        s = d["meta"]["answer_series"]
        s[0]["gold_units"]["fa"] = s[0]["gold_units"]["fa"] + s[1]["gold_units"]["fa"]
    check("timeline.catch_unit_from_other_report", any("not in the period's report" in m for m in broken("C-TL-tipax-01", foreign_unit)))

    def own_restatement(d):
        d["meta"]["reconciliation"]["restatements"][0]["doc"] = d["meta"]["answer_series"][1]["doc"]
    check("timeline.catch_restatement_from_own_report", any("own report" in m for m in broken("C-TL-divar-02", own_restatement)))

    def no_notes(d):
        d["meta"]["series_notes_en"] = ""
    check("timeline.catch_noncomparable_without_notes", any("series_notes_en" in m for m in broken("C-TL-karnameh-01", no_notes)))

    def union(d):
        d["gold_units"]["fa"] = d["gold_units"]["fa"][:-1]
    check("timeline.catch_gold_union", any("union" in m for m in broken("C-TL-cafebazaar-02", union)))

    def not_bool(d):
        d["meta"]["comparable"] = "no"
    check("timeline.catch_comparable_not_bool", any("true or false" in m for m in broken("C-TL-tapsell-01", not_bool)))

    def answer_missing_period(d):
        d["answer_en"] = "1397: 163 thousand; 1399: 161 thousand."
    check("timeline.warn_reference_answer_unparsed", any("series parser" in m for m in broken("C-TL-cafebazaar-02", answer_missing_period,
                                                                                              ("warn",))))


def test_matching():
    check("match.persian_decimal", T.matches_all(T.match_key("93/9% از سفارش‌ها"), ["93.9%"]))
    check("match.thousands", T.match_key("1,418,000 t") == "1418000 t", T.match_key("1,418,000 t"))
    check("match.digit_boundary", not T.matches_all(T.match_key("15.19 and 2,161"), ["5.19"])
          and not T.matches_all(T.match_key("1610 firms"), ["161"]))
    check("match.zwnj_fold", T.matches_all(T.match_key("دیجی\u200cکالا سرویس"), ["دیجی‌کالا سرویس"]))
    check("match.alternatives", T.matches_all(T.match_key("308 K Sellers"), [["308,000", "sellers"], ["308 k", "sellers"]]))
    check("match.numbers_in", T.numbers_in("150,000,000 تومان و 9/85 میلیون") == [150000000.0, 9.85])
    # mixed terms keep digit boundaries ("700 kt" is not "1,700 kt"; "24 درصد" is not "124" or "0.24")
    mixed_bad = [("1,700 kt of billet", ["700 kt"]), ("124 درصد", ["24 درصد"]), ("0.24 درصد", ["24 درصد"]),
                 ("1164 عضو", ["164 عضو"]), ("198.5 million rials", ["98.5 million rials"]), ("2022 | 71", ["2022 | 7"])]
    check("match.mixed_digit_boundary", not any(T.matches_all(T.match_key(t), k) for t, k in mixed_bad),
          [t for t, k in mixed_bad if T.matches_all(T.match_key(t), k)])
    check("match.mixed_ok", T.matches_all(T.match_key("حدود 700 هزار تن شمش"), ["700 هزار تن"])
          and T.matches_all(T.match_key("2022 | 7 | 2023"), ["2022 | 7"]))
    # a "%" term needs a percent marker on either side; RTL extraction often writes "%85"
    check("match.percent_needs_marker", not T.matches_all(T.match_key("4 روز تاخیر"), ["4%"])
          and not T.matches_all(T.match_key("16 billets"), ["16%"]))
    check("match.percent_forms", all(T.matches_all(T.match_key(t), ["4%"]) for t in ("4% of orders", "4 % از", "4 درصد", "%4 از", "4 percent")))
    # letter-initial terms start on a word boundary ("رد" is not inside "عملکرد"), suffixes stay allowed
    check("match.word_start", not T.matches_all(T.match_key("عملکرد سال"), ["رد"])
          and T.matches_all(T.match_key("24 درصد به ردشدن"), ["رد"])
          and T.matches_all(T.match_key("در خانه‌ها"), ["خانه"])
          and not T.matches_all(T.match_key("کارخانه"), ["خانه"]))


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    cfg = load_config()
    root = cfg.path("questions_root")
    path = root / T.OUT_NAME
    test_matching()

    raw = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    qs = load_questions(path)
    corp = T.Corpora(cfg)
    errs = errors_for(raw, corp)
    check("pilot.validates", not errs, errs[:3])
    check("pilot.size", len(qs) == 36, len(qs))
    comp = T.composition(qs)
    bad = [r for r in comp if not (r[2] >= r[3] if r[1] == "reports" else r[2] == r[3])]
    check("pilot.composition", not bad, bad)
    check("pilot.ids_unique", len({q.id for q in qs}) == len(qs))
    check("pilot.A_all_settings", all(len([s for s in ALL_SETTINGS if q.supports(s)]) == 4 for q in qs if q.corpus == "A"))
    check("pilot.C_fa_docs_only", all(q.supports("fa-fa") and q.supports("en-fa") and not q.supports("en-en")
                                      for q in qs if q.corpus == "C"))

    # the validator must catch broken copies
    by_id = {d["id"]: i for i, d in enumerate(raw)}

    def broken(qid, mutate):
        dd = copy.deepcopy(raw)
        mutate(dd[by_id[qid]])
        return errors_for(dd, corp)

    def missing_gold(d):
        d["gold_units"]["fa"][0] = d["gold_units"]["fa"][0] + "_nope"
        d["meta"]["evidence"][0]["units"]["fa"][0] = d["gold_units"]["fa"][0]
    check("catch.missing_gold", any("does not exist" in m for _, m in broken("A-rpt-a-01", missing_gold)))

    def misalign(d):
        d["gold_units"]["en"] = d["gold_units"]["en"][:-1]
    check("catch.alignment", any("not aligned" in m for _, m in broken("A-rpt-a-02", misalign)))

    def wrong_unit(d):
        other = [u for u in corp.doc("A.rpt-a.fa").units if "<gold value>" not in u.text][5].id
        d["gold_units"]["fa"] = [other]
        d["gold_units"]["en"] = [swap_lang(other, "en")]
        d["meta"]["evidence"][0]["units"] = {"fa": [other], "en": [swap_lang(other, "en")]}
    check("catch.quote_support", any("does not contain its quote" in m for _, m in broken("A-rpt-a-01", wrong_unit)))

    def c_english(d):
        d["gold_units"]["en"] = ["C.x.en::p1_b1"]
    check("catch.C_english_gold", any("English gold" in m for _, m in broken("C-snapp-01", c_english)))

    def no_value(d):
        d["answer_value"] = None
    check("catch.numeric_value", any("answer_value" in m for _, m in broken("B-digikala-02", no_value)))

    # gold units longer than the evidence budget are left out when the fact has other gold units
    max_chars = int(cfg.evidence["max_chars"])
    too_long = [(q.id, u) for q in qs for l in ("fa", "en") for u in q.gold(l)
                if corp.unit(u) is not None and len(corp.unit(u).text) > max_chars]
    check("gold.fits_budget", not too_long, too_long[:3])
    rq = next(d for d in raw if d["id"] == "A-rpt-g-02")
    check("gold.oversize_recorded", any((e.get("oversize") or {}).get("en") for e in rq["meta"]["evidence"]))
    d_rq = corp.doc("A.rpt-g.en")
    big = "method/method_fixes/x_nowhere#table.0"
    small = "front/front/k_front#kpi.1"
    kept, dropped = T._drop_oversize({"en": [big, small]}, {"en": d_rq}, max_chars)
    kept_sole, dropped_sole = T._drop_oversize({"en": [big]}, {"en": d_rq}, max_chars)
    check("gold.oversize_rule", kept == {"en": [small]} and dropped == [big] and kept_sole == {"en": [big]} and not dropped_sole)
    check("meta.lexical_overlap", all((d["meta"].get("lexical_overlap") or {}).get("fa") for d in raw
                                      if d["answer_type"] != "unanswerable"))

    # rebuild reproduces the committed file (quotes still link to the same units)
    with open(root / T.SRC_NAME, encoding="utf-8") as fh:
        specs = yaml.safe_load(fh)["questions"]
    rebuilt = [T.build_question(s, corp)[0] for s in specs]
    same = [q["id"] for q, r in zip(rebuilt, raw) if q["gold_units"] != r["gold_units"]]
    check("build.reproducible", not same, "gold changed for %s (extraction re-run? run build)" % same if same else "")
    stale = [q["id"] for q, r in zip(rebuilt, raw)
             if json.loads(json.dumps(q["meta"]["evidence"], ensure_ascii=False)) != r["meta"]["evidence"]]
    check("build.meta_current", not stale, "meta.evidence differs from pilot.src.yaml for %s (run build)" % stale if stale else "")

    test_timeline(corp, root)

    n_fail = sum(1 for v in RESULTS.values() if not v)
    print("%d checks, %d failed" % (len(RESULTS), n_fail))
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
