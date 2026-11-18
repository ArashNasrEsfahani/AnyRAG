"""Task 1: where the answer is lost. For every answerable test question and arm, did a complete gold bundle reach the
reader (verbatim, same document), a partial one, or none? Crossed with the saved machine verdict.
-> reach.json, reach_tables.md, fig_reach_arms.dat, fig_reach_tier.dat, fig_reach_lang.dat"""
import json
from collections import Counter, defaultdict

import s7lib as L

MAIN = ["HYBRID_6000", "HR_6000", "LEX5B_6000", "SETR_6000", "FUSION_6000", "SETR_NOFILL", "LISTWISE",
        "HYBRID_15500", "HR_15500", "ITER_BM25", "GREP_AGENT"]
EXTRA = ["LEX5B_15500", "SETR_15500", "HYBRID_3000", "HYBRID_10000", "HYBRID_24000", "HR_3000", "HR_10000",
         "HR_24000", "FUSION_3000", "FUSION_10000", "FUSION_15500", "FUSION_24000", "KNAP_6000", "MMR_6000", "GOLD"]
ARMS = MAIN + EXTRA
SYM = json.loads((L.TR / "test_results_human_sym.json").read_text(encoding="utf-8"))["per_question_glm_6000_sym"]
ABST = "NOT_FOUND"


def outcome(arm, q, reach):
    lo, hi = L.score(arm, q)
    if lo != hi:
        return "unresolved"
    if lo == 1:
        return "correct_" + reach
    return "miss_" + reach


rows = {}
for a in ARMS:
    for q in L.ANS:
        m = L.messages(a, q)
        r = L.gold_reach(q, m)
        lo, hi = L.score(a, q)
        ans = (L.answer_text(a, q) or "").strip()
        rows[(a, q)] = dict(arm=a, q=q, tier=L.tier(q), lang=L.lang(q), pool=L.pool(q), reach=r["cls"],
                            best_frac=r["best_frac"], items=r["items"], lo=lo, hi=hi,
                            outcome=outcome(a, q, r["cls"]), abstained=ans == ABST,
                            evidence_chars=len(L.evidence_block(m)) if m else 0)

# --------------------------------------------------------------- cross-check against unit ids for the record-based arms
xc = {}
for a in ["HYBRID_6000", "HR_6000", "SETR_6000", "HYBRID_15500", "HR_15500", "LEX5B_6000"]:
    agree = n = 0
    dis = []
    for q in L.ANS:
        rec = L.REC[L.P][a][q]
        eu = set(rec.get("evidence_units") or [])
        for e in L.QV2[q]["evidence"]:
            t = rows[(a, q)]["items"][e["id"]] >= L.PRESENT
            u = e["unit_id"] in eu
            n += 1
            agree += t == u
            if t != u:
                dis.append([q, e["id"], "text" if t else "unit"])
    xc[a] = {"items": n, "agree": agree, "text_only": sum(d[2] == "text" for d in dis),
             "unit_only": sum(d[2] == "unit" for d in dis)}

CATS = ["correct_complete", "correct_partial", "correct_none", "miss_complete", "miss_partial", "miss_none",
        "unresolved"]


def tally(arm, qs):
    c = Counter(rows[(arm, q)]["outcome"] for q in qs)
    reach = Counter(rows[(arm, q)]["reach"] for q in qs)
    abst = sum(rows[(arm, q)]["abstained"] for q in qs if rows[(arm, q)]["outcome"] == "miss_complete")
    out = {k: c.get(k, 0) for k in CATS}
    out.update({"n": len(qs), "reach_complete": reach.get("complete", 0), "reach_partial": reach.get("partial", 0),
                "reach_none": reach.get("none", 0), "miss_complete_abstained": abst})
    return out


res = {"definition": {
    "present": "an evidence item reached the reader if >= %.0f%% of its normalised quote occurs verbatim (segments of "
               ">= %d chars) inside pack items whose header names the same document" % (100 * L.PRESENT, L.MIN_SEG),
    "complete": "every item of at least one annotated bundle present", "partial": "some but not all (best bundle)",
    "none": "no item of any bundle present",
    "miss": "machine strict 0 (resolved)", "unresolved": "needs_adjudication cell (counted separately)"},
    "cross_check_unit_ids": xc, "arms": {}, "tier": {}, "lang": {}, "pool": {}}
for a in ARMS:
    res["arms"][a] = tally(a, L.ANS)
    for t in ("T1", "T2", "T3"):
        res["tier"]["%s|%s" % (a, t)] = tally(a, [q for q in L.ANS if L.tier(q) == t])
    for lg in ("fa", "en"):
        res["lang"]["%s|%s" % (a, lg)] = tally(a, [q for q in L.ANS if L.lang(q) == lg])
    for pl in sorted({L.pool(q) for q in L.ANS}):
        res["pool"]["%s|%s" % (a, pl)] = tally(a, [q for q in L.ANS if L.pool(q) == pl])

# human view of "reached but missed" for the four frozen 6000 arms (symmetric human merge, 30 reviewed questions)
hum = {}
for a in ["HYBRID_6000", "HR_6000", "LEX5B_6000", "SETR_6000"]:
    c = Counter()
    for q in L.ANS:
        r = rows[(a, q)]
        h = SYM[a][q]
        reviewed = h[2] != "judged"
        if r["outcome"] in ("miss_complete", "miss_partial", "miss_none", "unresolved"):
            key = r["outcome"] + ("|reviewed_human_%d" % h[0] if reviewed else "|not_reviewed")
            c[key] += 1
    hum[a] = dict(sorted(c.items()))
res["human_on_misses"] = hum

# questions on which no 6000 retrieval arm delivered a complete bundle
six = ["HYBRID_6000", "HR_6000", "LEX5B_6000", "SETR_6000", "FUSION_6000", "SETR_NOFILL", "LISTWISE"]
never = [q for q in L.ANS if all(rows[(a, q)]["reach"] != "complete" for a in six)]
res["no_6000_arm_complete"] = {"n": len(never), "by_tier": dict(Counter(L.tier(q) for q in never)),
                               "by_lang": dict(Counter(L.lang(q) for q in never)), "questions": never}
anyc = [q for q in L.ANS if any(rows[(a, q)]["reach"] == "complete" for a in MAIN)]
res["any_main_arm_complete"] = len(anyc)
res["per_cell"] = [{k: v for k, v in r.items() if k != "items"} | {"items": r["items"]} for r in rows.values()]
L.wjson(L.HERE / "reach.json", res)

# --------------------------------------------------------------- .dat files (pgfplots, whitespace-separated)
LABEL = {"HYBRID_6000": "HYBRID", "HR_6000": "HR", "LEX5B_6000": "LexRAG", "SETR_6000": "SetR",
         "FUSION_6000": "FUSION", "SETR_NOFILL": "SetR-nofill", "LISTWISE": "Listwise", "HYBRID_15500": "HYBRID-15.5k",
         "HR_15500": "HR-15.5k", "ITER_BM25": "Iter-BM25", "GREP_AGENT": "Grep-agent"}
hdr = "arm label n correct_complete correct_partial correct_none miss_complete miss_partial miss_none unresolved " \
      "reach_complete reach_partial reach_none miss_complete_abstained"


def datrow(a, t):
    return " ".join(str(x) for x in [a, LABEL.get(a, a), t["n"]] + [t[k] for k in CATS] +
                    [t["reach_complete"], t["reach_partial"], t["reach_none"], t["miss_complete_abstained"]])


with open(L.HERE / "fig_reach_arms.dat", "w", encoding="utf-8") as fh:
    fh.write("# S7 task 1: GLM-5.3-flash, 90 answerable test questions, machine strict verdicts (reach.json)\n")
    fh.write(hdr + "\n")
    for a in MAIN:
        fh.write(datrow(a, res["arms"][a]) + "\n")
for key, fn, groups in (("tier", "fig_reach_tier.dat", ("T1", "T2", "T3")), ("lang", "fig_reach_lang.dat", ("fa", "en"))):
    with open(L.HERE / fn, "w", encoding="utf-8") as fh:
        fh.write("# S7 task 1 by %s (30 per tier / 45 per language); reach.json\n" % key)
        fh.write("group " + hdr + "\n")
        for g in groups:
            for a in MAIN:
                fh.write(g + " " + datrow(a, res[key]["%s|%s" % (a, g)]) + "\n")

# --------------------------------------------------------------- markdown
md = ["# S7 task 1: where the answer is lost (generated by t1_reach.py from reach.json)", "",
      "Definition: " + res["definition"]["present"] + ". complete = all items of some bundle; partial = some; none = no item.",
      "Outcome = machine strict verdict (resolved); unresolved judge cells are a separate column.", "",
      "| arm | correct /90 | reach complete | partial | none | miss: evidence complete (reader/judge) | of which NOT_FOUND | miss: partial | miss: none (never reached) | correct w/o complete bundle | unresolved |",
      "|---|---|---|---|---|---|---|---|---|---|---|"]
for a in ARMS:
    t = res["arms"][a]
    cor = t["correct_complete"] + t["correct_partial"] + t["correct_none"]
    md.append("| %s | %d | %d | %d | %d | %d | %d | %d | %d | %d | %d |" % (
        a, cor, t["reach_complete"], t["reach_partial"], t["reach_none"], t["miss_complete"],
        t["miss_complete_abstained"], t["miss_partial"], t["miss_none"], t["correct_partial"] + t["correct_none"],
        t["unresolved"]))
for key, groups in (("tier", ("T1", "T2", "T3")), ("lang", ("fa", "en"))):
    md += ["", "## by %s" % key, "",
           "| arm | " + " | ".join("%s: reach C/P/N | %s: miss C/P/N" % (g, g) for g in groups) + " |",
           "|---|" + "---|---|" * len(groups)]
    for a in MAIN:
        cells = []
        for g in groups:
            t = res[key]["%s|%s" % (a, g)]
            cells.append("%d/%d/%d" % (t["reach_complete"], t["reach_partial"], t["reach_none"]))
            cells.append("%d/%d/%d" % (t["miss_complete"], t["miss_partial"], t["miss_none"]))
        md.append("| %s | %s |" % (a, " | ".join(cells)))
md += ["", "## cross-check: text match vs gold unit id in the record's evidence_units (per evidence item)", "",
       "| arm | items | agree | text only | unit only |", "|---|---|---|---|---|"]
for a, x in xc.items():
    md.append("| %s | %d | %d | %d | %d |" % (a, x["items"], x["agree"], x["text_only"], x["unit_only"]))
md += ["", "## human verdicts on the frozen 6000 misses (symmetric merge; 30 reviewed questions)", ""]
for a, c in hum.items():
    md.append("- %s: %s" % (a, ", ".join("%s %d" % kv for kv in c.items())))
md += ["", "No 6000 retrieval arm delivered a complete bundle on %d questions (%s; %s)." % (
    len(never), res["no_6000_arm_complete"]["by_tier"], res["no_6000_arm_complete"]["by_lang"]),
    "At least one of the 11 main arms delivered a complete bundle on %d of 90." % len(anyc)]
(L.HERE / "reach_tables.md").write_text("\n".join(md) + "\n", encoding="utf-8")
print("\n".join(md))
