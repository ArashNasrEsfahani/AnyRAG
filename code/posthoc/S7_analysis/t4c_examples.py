"""Task 4c: four short examples for a paper figure, exact text from the saved packs, answers and verdicts.
English glosses of Persian strings are this stream's own translations (marked gloss_en). -> examples.json, examples.md"""
import json
import sys

import s7lib as L

sys.path.insert(0, str(L.PH / "S1_evaluator"))
import j2lib as J2  # noqa: E402

J2._CACHE["rv"] = (L.REC, L.VER)
rows2, _ = J2._j2_rows()
SYM = json.loads((L.TR / "test_results_human_sym.json").read_text(encoding="utf-8"))["per_question_glm_6000_sym"]
REACH = {(r["arm"], r["q"]): r for r in json.loads((L.HERE / "reach.json").read_text(encoding="utf-8"))["per_cell"]}


def verdicts(arm, q):
    out = {"J1_machine": list(L.score(arm, q))}
    if arm in L.FROZEN_ARMS:
        r = L.REC[L.P][arm][q]
        k = (r["meta"]["identity"]["rid"], L.JM.answer_sha1(r))
        v1 = L.VER[L.P].get(k)
        out["J1_reason"] = v1 and v1.get("verdict", {}).get("reason")
        out["J1_units"] = v1 and v1.get("verdict", {}).get("units")
        v2 = rows2.get(k)
        out["J2_machine"] = v2 and v2["strict"]
        out["J2_reason"] = v2 and v2.get("reason")
        out["J2_units"] = v2 and v2.get("units")
        if arm in SYM:
            h = SYM[arm][q]
            out["human_sym"] = h[0] if h[2] != "judged" else "not reviewed"
    return out


def item_headers(arm, q):
    return [h for h, _ in L.parse_items(L.evidence_block(L.messages(arm, q)))]


def gold(q, ids):
    return [{"id": e["id"], "quote": e["quote"], "region": e.get("region")} for e in L.QV2[q]["evidence"] if e["id"] in ids]


def sel_raw(q):
    return L.REC[L.P]["LEX5B_6000"][q]["trace"]["lex5b"]["selector"].get("raw")


EX = []
# (i) selector rescue, Persian -------------------------------------------------------------------------
q = "V2-test-RPT-fa-T2-005"
EX.append({
    "id": "ex1_selector_rescue_fa", "kind": "selector rescues evidence that fusion order missed", "q": q,
    "question": L.QV2[q]["question"], "question_gloss_en": L.QV2[q]["question_gloss_en"],
    "reference_gloss_en": L.QV2[q]["expected_answer_gloss_en"],
    "gold_needed": gold(q, ["E3", "E6"]),
    "gold_gloss_en": {"E3": "41,579,509 toman", "E6": "A user who completes more than 40 successful trips a month."},
    "FUSION_6000": {"pack_items": item_headers("FUSION_6000", q), "gold_item_coverage": REACH[("FUSION_6000", q)]["items"],
                    "answer_first_sentence": "میانگین درآمد ماهانهٔ کاربران رانندهٔ تمام‌وقت اسنپ در سال ۱۴۰۳ برابر 41,579,509 تومان بود [3].",
                    "answer_gloss_en": "The average monthly income of Snapp's full-time driver users in 1403 was 41,579,509 toman [3]. "
                                       "(The rest of the answer says Snapp splits drivers into full-time and part-time but gives no definition.)",
                    "verdict": verdicts("FUSION_6000", q)},
    "LEX5B_6000": {"selector_raw": sel_raw(q), "pack_items": item_headers("LEX5B_6000", q),
                   "gold_item_coverage": REACH[("LEX5B_6000", q)]["items"],
                   "answer_first_sentence": "میانگین درآمد ماهانهٔ رانندگان تمام‌وقت اسنپ در ۱۴۰۳ برابر ۴۱,۵۷۹,۵۰۹ تومان بود و اسنپ راننده‌ای را تمام‌وقت می‌داند که در ماه بیش از ۴۰ سفر موفق انجام دهد.",
                   "answer_gloss_en": "The average monthly income of Snapp's full-time drivers in 1403 was 41,579,509 toman, and Snapp "
                                      "counts as full-time a driver who completes more than 40 successful trips a month.",
                   "verdict": verdicts("LEX5B_6000", q)},
    "note": "The selector split the question into two parts (P1: passage 3, the income table; P2: passage 9, the 1403 "
            "glossary). The fusion pack never reaches the 1403 glossary item (header '... 1403 > واژه‌نامه' = 'glossary'); "
            "Lex-RAG puts it second.",
})
for k in ("FUSION_6000", "LEX5B_6000"):
    EX[-1][k]["answer_full"] = L.answer_text(k, q)
# (i) selector rescue, English -------------------------------------------------------------------------
q = "V2-test-LAW-en-T2-004"
EX.append({
    "id": "ex2_selector_rescue_en", "kind": "selector rescues evidence that fusion order missed", "q": q,
    "question": L.QV2[q]["question"], "reference": L.QV2[q]["expected_answer"], "gold_needed": gold(q, ["E1", "E2", "E3"]),
    "FUSION_6000": {"pack_items": item_headers("FUSION_6000", q), "gold_item_coverage": REACH[("FUSION_6000", q)]["items"],
                    "answer_full": L.answer_text("FUSION_6000", q), "verdict": verdicts("FUSION_6000", q)},
    "LEX5B_6000": {"selector_raw": sel_raw(q), "pack_items": item_headers("LEX5B_6000", q),
                   "gold_item_coverage": REACH[("LEX5B_6000", q)]["items"],
                   "answer_full": L.answer_text("LEX5B_6000", q), "verdict": verdicts("LEX5B_6000", q)},
    "note": "Fusion order reaches Art. 3(3) but not the full Art. 30(3) fine clause with point (b); the reader then answers "
            "the fine part with NOT_FOUND. The selector's second part (passages 15+21) is exactly Art. 30(3) + (b).",
})
# (ii) judge disagreement, Persian ---------------------------------------------------------------------
q = "V2-test-LAW-fa-T1-002"
EX.append({
    "id": "ex3_judge_disagreement_fa", "kind": "judge disagreement: Persian synonym not recognised by judge 1", "q": q,
    "question": L.QV2[q]["question"], "question_gloss_en": L.QV2[q]["question_gloss_en"],
    "reference_gloss_en": L.QV2[q]["expected_answer_gloss_en"],
    "requirement_R2": [x["proposition"] for x in L.QV2[q]["requirements"] if x["id"] == "R2"][0],
    "cell": "LEX5B_6000", "answer_full": L.answer_text("LEX5B_6000", q),
    "answer_gloss_en": "Under Art. 64 of the Constitution the religious minorities have 5 representatives in total: "
                       "1. Zoroastrians: one representative; 2. Kalimian [the Persian term for Jews]: one representative; "
                       "3. Assyrian and Chaldean Christians: one representative jointly; 4. Armenian Christians of the "
                       "south: one representative; 5. Armenian Christians of the north: one representative.",
    "verdict": verdicts("LEX5B_6000", q),
    "same_answer_other_cells": "all 8 GLM cells give the same five seats with the word kalimian; judge 1 failed 7 of "
                               "them (it passed HYBRID_6000), judge 2 passed all 8 (j12_disagreements.json)",
    "note": "Judge 1 marks the Jews requirement 'missing' and the claim 'کلیمیان: یک نماینده' (Kalimian: one "
            "representative) as 'extra': it does not recognise کلیمیان as 'Jews'. Judge 2 and the human accept the answer.",
})
# (ii) judge disagreement, English ---------------------------------------------------------------------
q = "V2-test-RPT-en-T2-002"
ev = L.evidence_block(L.messages("LEX5B_6000", q))
i = ev.find("Profitability lagged")
EX.append({
    "id": "ex4_judge_disagreement_en", "kind": "judge disagreement: true added detail failed as 'unsupported' by judge 2", "q": q,
    "question": L.QV2[q]["question"], "reference": L.QV2[q]["expected_answer"], "cell": "LEX5B_6000",
    "answer_full": L.answer_text("LEX5B_6000", q), "verdict": verdicts("LEX5B_6000", q),
    "gold_E1_quote": [e["quote"] for e in L.QV2[q]["evidence"] if e["id"] == "E1"][0],
    "pack_sentence_after_E1": ev[i:i + 150],
    "note": "The sentence judge 2 calls unsupported is verbatim in the reader's pack, immediately after the gold quote E1; "
            "the judge sees only the annotated excerpts. Judge 2 fails 6 of the 8 GLM cells of this question (5 of them passed by judge 1); judge 1 fails 1.",
})
L.wjson(L.HERE / "examples.json", EX)

md = ["# S7 task 4c: examples for the paper figure (exact text; generated by t4c_examples.py -> examples.json)", ""]
for e in EX:
    md += ["## " + e["id"] + ": " + e["kind"], "", "- question (%s): %s" % (e["q"], e["question"])]
    if e.get("question_gloss_en"):
        md.append("- gloss: " + e["question_gloss_en"])
    if e.get("reference_gloss_en"):
        md.append("- reference (gloss): " + e["reference_gloss_en"])
    if e.get("reference"):
        md.append("- reference: " + e["reference"])
    for g in e.get("gold_needed", []):
        md.append("- gold %s: “%s”%s" % (g["id"], g["quote"], (" (gloss: %s)" % e["gold_gloss_en"][g["id"]]) if e.get("gold_gloss_en", {}).get(g["id"]) else ""))
    for arm in ("FUSION_6000", "LEX5B_6000"):
        if arm in e:
            x = e[arm]
            md.append("- **%s** pack items: %s" % (arm, " | ".join(x["pack_items"])))
            md.append("  - gold coverage: %s" % x["gold_item_coverage"])
            if x.get("selector_raw"):
                md.append("  - selector reply: `%s`" % x["selector_raw"].replace("\n", " / "))
            md.append("  - answer: %s" % (x.get("answer_first_sentence") or x.get("answer_full")).replace("\n", " / "))
            if x.get("answer_gloss_en"):
                md.append("  - answer gloss: " + x["answer_gloss_en"])
            v = x["verdict"]
            md.append("  - machine judge 1: %s; judge 2: %s; human: %s" % (v["J1_machine"], v.get("J2_machine"), v.get("human_sym", "n/a")))
    if "answer_full" in e and "FUSION_6000" not in e:
        md.append("- answer (%s): %s" % (e["cell"], e["answer_full"].replace("\n", " / ")))
        if e.get("answer_gloss_en"):
            md.append("- answer gloss: " + e["answer_gloss_en"])
        v = e["verdict"]
        md.append("- judge 1: strict %s, units %s, reason “%s”" % (v["J1_machine"], v.get("J1_units"), v.get("J1_reason")))
        md.append("- judge 2: strict %s, units %s, reason “%s”" % (v.get("J2_machine"), v.get("J2_units"), v.get("J2_reason")))
        md.append("- human (symmetric review): %s" % v.get("human_sym"))
    if e.get("pack_sentence_after_E1"):
        md.append("- gold E1: “%s”; pack continues: “%s”" % (e["gold_E1_quote"], e["pack_sentence_after_E1"]))
    md += ["- note: " + e["note"], ""]
(L.HERE / "examples.md").write_text("\n".join(md) + "\n", encoding="utf-8")
print("\n".join(md))
