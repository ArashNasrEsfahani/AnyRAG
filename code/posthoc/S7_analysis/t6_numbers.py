"""Task 6 (review item 19): numeric loose ends, from saved files only.
(1) App. F "15 of the 27 human-accepted SetR answers": the machine x human 2x2 per pipeline on the 120 reviewed cells
(2) dev gold-only T2 12/18 vs test GOLD T2 30/30: what each number is
(3) Table 13 (tab:devtest): which judge produced each dev machine number
-> numbers.json, numbers_tables.md"""
import json
from collections import Counter

import s7lib as L

out = {}
# ------------------------------------------------------------------ (1) 2x2 per pipeline
sym = json.loads((L.TR / "test_results_human_sym.json").read_text(encoding="utf-8"))
SYM = sym["per_question_glm_6000_sym"]
sel = sym["selected_questions"]
selq = sel if isinstance(sel, list) else list(sel)
two = {}
for a in ["HYBRID_6000", "HR_6000", "LEX5B_6000", "SETR_6000"]:
    c = Counter()
    for q in L.ANS:
        h = SYM[a][q]
        if h[2] == "judged":
            continue
        hv = h[0]
        m = L.score(a, q)
        mv = "unres" if m[0] != m[1] else str(m[0])
        c["m%s_h%d" % (mv, hv)] += 1
    two[a] = dict(sorted(c.items()))
    two[a]["reviewed"] = sum(c.values())
    two[a]["human_accepted"] = c["m1_h1"] + c["m0_h1"] + c["munres_h1"]
    two[a]["human_accepted_machine_resolved"] = c["m1_h1"] + c["m0_h1"]
    two[a]["judge_rejected_of_human_accepted"] = c["m0_h1"]
out["appF_2x2"] = two
out["appF_2x2_source"] = ("work/datasets_v2/test_run/test_results_human_sym.json per_question_glm_6000_sym (status != judged = "
                          "human-reviewed cell, half-correct rule applied) x machine verdicts via report_test.score")

# ------------------------------------------------------------------ (2) dev gold-only T2 vs test GOLD T2
dg = L.WORK / "datasets_v2" / "diagnosis"
ver = L.jl(dg / "judge" / "gold__z-ai_glm-5.3-flash" / "verdicts_openai__gpt-oss-120b.jsonl")
fail = L.jl(dg / "judge" / "gold__z-ai_glm-5.3-flash" / "judge_failures_openai__gpt-oss-120b.jsonl")
recs = [r for f in sorted((dg / "runs" / "gold__z-ai_glm-5.3-flash").glob("*/runs_*.jsonl")) for r in L.jl(f)]
t2 = sorted({r["question_id"] for r in recs if "-T2-" in r["question_id"]})
vq = {}
for v in ver:
    vq.setdefault(v["question_id"], v)
dev = {"questions_T2": len(t2), "judged_T2": sum(q in vq for q in t2),
       "correct_T2": sum(vq[q]["strict"] for q in t2 if q in vq),
       "unjudged_T2": [q for q in t2 if q not in vq],
       "wrong_T2": {q: vq[q].get("reason") for q in t2 if q in vq and vq[q]["strict"] == 0},
       "judge_pin": sorted({str(v.get("judge_provider_pin")) for v in ver}),
       "judge_served": sorted({str(v.get("judge_provider_served")) for v in ver}),
       "judged_at": sorted({v["at"][:10] for v in ver}),
       "pack": "B1 GoldEvidence: the question's gold units (whole units, every annotated evidence unit, author order)",
       "source": "work/datasets_v2/diagnosis/T2_T3_DIAGNOSIS.md sec. 2; judge/gold__z-ai_glm-5.3-flash/verdicts_*.jsonl",
       "T3": {"judged": sum(1 for v in ver if "-T3-" in v["question_id"]),
              "correct": sum(v["strict"] for v in ver if "-T3-" in v["question_id"])},
       "judge_failure_rows": len(fail)}
# do the dev gold packs equal the current dev evidence units?
cur = {}
for f in (L.WORK / "datasets_v2" / "questions").glob("*_dev.jsonl"):
    for d in L.jl(f):
        cur[d["id"]] = d
dev["gold_units_equal_current_dev_evidence"] = sum(
    set(r.get("evidence_units") or []) == {e["unit_id"] for e in cur[r["question_id"]]["evidence"]} for r in recs)
dev["records"] = len(recs)
test = {"T2": list(L.total("GOLD", [q for q in L.ANS if L.tier(q) == "T2"])),
        "T3": list(L.total("GOLD", [q for q in L.ANS if L.tier(q) == "T3"])),
        "T1": list(L.total("GOLD", [q for q in L.ANS if L.tier(q) == "T1"])),
        "pack": "first annotated evidence bundle, doc.text[start:end] slices (S2a GOLD)",
        "judge": "pinned gpt-oss-120b deepinfra/bf16 (S2a judge.py)",
        "source": "work/posthoc_2026-11/S2a_packs/results.json; recomputed here from S2a verdicts"}
out["gold_only_dev_vs_test"] = {"dev": dev, "test": test}

# ------------------------------------------------------------------ (3) Table 13 judges
d60 = L.jl(L.WORK / "datasets_v2" / "dev60" / "judge" / "verdicts_openai__gpt-oss-120b.jsonl")
st = L.jl(L.WORK / "datasets_v2" / "setr" / "judge" / "verdicts_openai__gpt-oss-120b.jsonl")
out["table13_judges"] = {
    "dev60 (HYBRID, HR, LEX5B dev machine)": {"rows": len(d60), "by_method_pin": dict(Counter(
        "%s|pin=%s|served=%s" % (v.get("method"), v.get("judge_provider_pin"), v.get("judge_provider_served")) for v in d60)),
        "dates": sorted({v["at"][:10] for v in d60})},
    "setr (SETR dev machine)": {"rows": len(st), "by_source_pin": dict(Counter(
        "%s|pin=%s|served=%s" % (v.get("source"), v.get("judge_provider_pin"), v.get("judge_provider_served")) for v in st)),
        "dates": sorted({v["at"][:10] for v in st})},
    "note": "dev60 rows carry no provider pin (field absent -> None) and no served provider; SETR rows are pinned to "
            "deepinfra/bf16 (49 judge calls served by DeepInfra, 9 rule verdicts). The 24 idea_test questions inside dev60 "
            "reuse earlier verdicts (DEV60.md 'saved 24').",
}
L.wjson(L.HERE / "numbers.json", out)

md = ["# S7 task 6: numeric loose ends (generated by t6_numbers.py -> numbers.json)", "",
      "## (1) machine judge x human, 120 reviewed cells (30 questions x 4 pipelines, GLM @6000)", "",
      "| pipeline | reviewed | m1 h1 | m0 h1 (judge stricter) | m1 h0 | m0 h0 | unresolved h1 | unresolved h0 | human accepted | of which judge rejected |",
      "|---|---|---|---|---|---|---|---|---|---|"]
for a, c in two.items():
    md.append("| %s | %d | %d | %d | %d | %d | %d | %d | %d | %d |" % (a, c["reviewed"], c.get("m1_h1", 0), c.get("m0_h1", 0),
                                                                 c.get("m1_h0", 0), c.get("m0_h0", 0), c.get("munres_h1", 0),
                                                                 c.get("munres_h0", 0), c["human_accepted"], c["m0_h1"] if "m0_h1" in c else 0))
md += ["", "## (2) gold-only accuracy, dev vs test", "",
       "- dev (diagnosis, 2026-09-26): T2 %d correct of %d (%d judged, unjudged %s counted wrong); T3 %d/%d judged; judge pin %s, served %s; pack = %s" % (
           dev["correct_T2"], dev["questions_T2"], dev["judged_T2"], dev["unjudged_T2"], dev["T3"]["correct"], dev["T3"]["judged"],
           dev["judge_pin"], dev["judge_served"], dev["pack"]),
       "- dev T2 wrong: " + "; ".join("%s: %s" % kv for kv in dev["wrong_T2"].items()),
       "- dev gold units equal the current dev evidence unit sets in %d of %d records" % (dev["gold_units_equal_current_dev_evidence"], dev["records"]),
       "- test (S2a GOLD): T1 %s, T2 %s, T3 %s of 30; %s; %s" % (L.fmt(*test["T1"]), L.fmt(*test["T2"]), L.fmt(*test["T3"]), test["pack"], test["judge"]),
       "", "## (3) Table 13 judges", "", "```", json.dumps(out["table13_judges"], indent=1), "```"]
(L.HERE / "numbers_tables.md").write_text("\n".join(md) + "\n", encoding="utf-8")
print("\n".join(md))
