"""Task 4 (review item 16): error analysis.
(a) judge-1 vs judge-2 disagreements on the primary reader, classified by cause (manual reading of both judges' saved
    units/claims/reasons in j12_disagreements_dump.md; labels below, one per cell, index = order in j12_disagreements.json)
(b) Lex-RAG (LEX5B_6000) vs HYBRID_6000 and vs FUSION_6000 wins/losses by the question's integration operation
    (subtype field) and language, with the gold-bundle reach of both packs (reach.json)
(c) candidate examples for the paper figure (selected ones are written by t4c_examples.py)
-> errors.json, errors_tables.md"""
import json
from collections import Counter, defaultdict

import s7lib as L

D = json.loads((L.HERE / "j12_disagreements.json").read_text(encoding="utf-8"))
# Manual labels (read from the dump; see RESULTS.md for the definitions)
#   extra_unsupported  all required units correct under both judges; the stricter judge failed an added claim as
#                      unsupported by the packet (the added detail is mostly true and taken from the evidence)
#   omission_disputed  a required unit judged missing that is only implied, partly stated or stated differently
#   judge_misread      the stricter judge calls a unit missing/wrong although the answer states it plainly
#   synonym_fa         Persian term not recognised (kalimian = Jews) -> required unit 'missing'
#   numeric_tolerance  '9.2%' against the reference's 'about 9%'
#   substantive_error  a real defect of the answer that only the stricter judge caught
LAB = {0: "extra_unsupported", 1: "extra_unsupported", 2: "extra_unsupported", 3: "omission_disputed",
       4: "substantive_error", 5: "omission_disputed", 6: "numeric_tolerance", 7: "extra_unsupported",
       8: "extra_unsupported", 9: "extra_unsupported", 10: "omission_disputed", 11: "synonym_fa",
       12: "extra_unsupported", 13: "omission_disputed", 14: "substantive_error", 15: "extra_unsupported",
       16: "extra_unsupported", 17: "judge_misread", 18: "extra_unsupported", 19: "judge_misread",
       20: "judge_misread", 21: "extra_unsupported", 22: "synonym_fa", 23: "extra_unsupported",
       24: "extra_unsupported", 25: "extra_unsupported", 26: "omission_disputed", 27: "synonym_fa",
       28: "extra_unsupported", 29: "extra_unsupported", 30: "omission_disputed", 31: "judge_misread",
       32: "judge_misread", 33: "substantive_error", 34: "synonym_fa", 35: "substantive_error",
       36: "omission_disputed", 37: "omission_disputed", 38: "numeric_tolerance", 39: "extra_unsupported",
       40: "extra_unsupported", 41: "extra_unsupported", 42: "substantive_error", 43: "substantive_error",
       44: "synonym_fa", 45: "extra_unsupported", 46: "omission_disputed", 47: "substantive_error",
       48: "judge_misread", 49: "numeric_tolerance", 50: "extra_unsupported", 51: "omission_disputed",
       52: "synonym_fa", 53: "extra_unsupported", 54: "extra_unsupported", 55: "extra_unsupported",
       56: "numeric_tolerance", 57: "extra_unsupported", 58: "extra_unsupported", 59: "extra_unsupported",
       60: "omission_disputed", 61: "extra_unsupported", 62: "synonym_fa", 63: "extra_unsupported",
       64: "extra_unsupported", 65: "omission_disputed"}
assert len(LAB) == len(D) == 66
for i, d in enumerate(D):
    d["label"] = LAB[i]
    d["idx"] = i
CATS = ["extra_unsupported", "omission_disputed", "judge_misread", "synonym_fa", "numeric_tolerance", "substantive_error"]
a = {"n": len(D), "by_label": {}, "by_label_question": {}, "by_label_lang": {}, "by_arm": {}}
for c in CATS:
    cs = [d for d in D if d["label"] == c]
    a["by_label"][c] = {"cells": len(cs), "J1_stricter": sum(d["stricter"] == "J1" for d in cs),
                        "J2_stricter": sum(d["stricter"] == "J2" for d in cs), "questions": len({d["q"] for d in cs}),
                        "fa": sum(d["lang"] == "fa" for d in cs), "en": sum(d["lang"] == "en" for d in cs),
                        "six_thousand_cells": sum(d["arm"].endswith("_6000") for d in cs)}
a["by_arm"] = {arm: dict(Counter(d["label"] for d in D if d["arm"] == arm)) for arm in L.FROZEN_ARMS}
a["questions_total"] = len({d["q"] for d in D})
a["top_questions"] = Counter(d["q"] for d in D).most_common(8)
a["structural_flags_of_stricter_verdict"] = dict(Counter(f for d in D for f in d["structural"]))

# ------------------------------------------------------------------ (b) wins / losses by operation
reach = {(r["arm"], r["q"]): r for r in json.loads((L.HERE / "reach.json").read_text(encoding="utf-8"))["per_cell"]}
b = {}
for ctrl in ("HYBRID_6000", "FUSION_6000", "HR_6000"):
    tab = defaultdict(lambda: {"win": 0, "loss": 0, "tie1": 0, "tie0": 0, "unres": 0, "n": 0})
    detail = []
    for q in L.ANS:
        x, y = L.score("LEX5B_6000", q), L.score(ctrl, q)
        st = L.QV2[q]["subtype"]
        for key in (st, "%s|%s" % (st, L.lang(q)), "ALL|" + L.lang(q), "ALL"):
            t = tab[key]
            t["n"] += 1
            if x[0] != x[1] or y[0] != y[1]:
                t["unres"] += 1
            elif x[0] > y[0]:
                t["win"] += 1
            elif x[0] < y[0]:
                t["loss"] += 1
            elif x[0] == 1:
                t["tie1"] += 1
            else:
                t["tie0"] += 1
        if x[0] == x[1] and y[0] == y[1] and x[0] != y[0]:
            detail.append({"q": q, "subtype": st, "lang": L.lang(q), "result": "win" if x[0] > y[0] else "loss",
                           "lex_reach": reach[("LEX5B_6000", q)]["reach"], "ctrl_reach": reach[(ctrl, q)]["reach"],
                           "lex_chars": reach[("LEX5B_6000", q)]["evidence_chars"],
                           "ctrl_chars": reach[(ctrl, q)]["evidence_chars"]})
    # reach transitions on decisive pairs
    trans = Counter("%s:%s->%s" % (d["result"], d["ctrl_reach"], d["lex_reach"]) for d in detail)
    b[ctrl] = {"table": dict(tab), "decisive": detail, "reach_transitions": dict(trans)}

# selector-fallback status of LEX5B on decisive pairs vs FUSION (fallback -> identical pack)
sel = {}
for q in L.ANS:
    tr = L.REC[L.P]["LEX5B_6000"][q]["trace"]["lex5b"]
    sel[q] = {"fallback": tr["selector"].get("fallback"), "status": tr["selector"].get("status")}
b["selector_on_decisive_vs_FUSION"] = dict(Counter("%s|fallback=%s" % (d["result"], sel[d["q"]]["fallback"])
                                                   for d in b["FUSION_6000"]["decisive"]))
res = {"a_judge_disagreements": a, "a_cells": [{k: d[k] for k in ("idx", "arm", "q", "lang", "tier", "stricter",
                                                                  "label", "structural")} for d in D],
       "b_wins_losses": b}
L.wjson(L.HERE / "errors.json", res)

# ------------------------------------------------------------------ markdown
md = ["# S7 task 4: error analysis (generated by t4_errors.py -> errors.json)", "",
      "## (a) judge 1 (gpt-oss-120b) vs judge 2 (Mistral Large 3) disagreements, GLM reader, 8 frozen cells", "",
      "%d cells on %d questions (stricter judge: J1 %d, J2 %d)." % (len(D), a["questions_total"],
                                                                  sum(d["stricter"] == "J1" for d in D), sum(d["stricter"] == "J2" for d in D)), "",
      "| cause | cells | J1 stricter | J2 stricter | questions | fa | en | in 6000 cells |", "|---|---|---|---|---|---|---|---|"]
for c in CATS:
    t = a["by_label"][c]
    md.append("| %s | %d | %d | %d | %d | %d | %d | %d |" % (c, t["cells"], t["J1_stricter"], t["J2_stricter"], t["questions"],
                                                          t["fa"], t["en"], t["six_thousand_cells"]))
md += ["", "Most frequent questions: " + ", ".join("%s (%d)" % kv for kv in a["top_questions"]), ""]
md += ["## (b) LEX5B_6000 vs controls by integration operation (machine strict; win/loss/tie-correct/tie-wrong/unresolved)", ""]
subs = sorted({L.QV2[q]["subtype"] for q in L.ANS})
for ctrl in ("HYBRID_6000", "FUSION_6000", "HR_6000"):
    tab = b[ctrl]["table"]
    md += ["### vs " + ctrl, "", "| operation (subtype) | n | W/L all | W/L fa | W/L en |", "|---|---|---|---|---|"]
    for s in subs + ["ALL"]:
        t = tab[s]
        f = tab.get("%s|fa" % s) if s != "ALL" else tab["ALL|fa"]
        e = tab.get("%s|en" % s) if s != "ALL" else tab["ALL|en"]
        g = lambda t: "%d/%d" % (t["win"], t["loss"]) + (" (+%d unres)" % t["unres"] if t["unres"] else "") if t else "-"
        md.append("| %s | %d | %s | %s | %s |" % (s, t["n"], g(t), g(f), g(e)))
    md += ["", "Reach of the control pack -> Lex-RAG pack on the decisive questions: " +
           ", ".join("%s %d" % kv for kv in sorted(b[ctrl]["reach_transitions"].items())), ""]
md.append("Selector status on the decisive questions vs FUSION: %s" % b["selector_on_decisive_vs_FUSION"])
(L.HERE / "errors_tables.md").write_text("\n".join(md) + "\n", encoding="utf-8")
print("\n".join(md))
for d in b["FUSION_6000"]["decisive"]:
    print(d)
