"""Symmetric human-adjudicated true-test results (GLM-5.3-flash reader, 6,000 review budget). No model calls.

Inputs: human_check/key_test.json + human_test_export.json (first, asymmetric review: 77 items / 78 cells),
human_check/key_test_sym.json + human_test_sym_export.json (symmetric completion: 39 items / 40 cells shown + cells
inherited from a byte-identical reviewed answer of the same question). Machine verdicts via report_test.load()/score().
Rule (user, 2026-10-05): note containing "half correct" (any case) = WRONG regardless of button; "unsure" keeps the
machine verdict. Outside the 30 reviewed questions: machine strict verdicts; needs_adjudication = bounds [0, 1].
Writes TEST_RESULTS_HUMAN_SYM.md + test_results_human_sym.json next to this file."""
import itertools, json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import report_test as RT  # noqa: E402

# Paths (release edit, results-neutral): data directory as resolved by report_test (results/test in the repository);
# review keys/exports in results/human_review (workspace: human_check); the superseded asymmetric merge, read only to
# reproduce it cell for cell, in results/test/superseded_asymmetric_review (workspace: next to the data).
HERE = RT.HERE
HC = HERE.parent / ("human_review" if (HERE.parent / "human_review").is_dir() else "human_check")
_PREV = HERE / "superseded_asymmetric_review" / "test_results_human.json"
PREV_PATH = _PREV if _PREV.is_file() else HERE / "test_results_human.json"

ARMS4 = ["HYBRID_6000", "HR_6000", "LEX5B_6000", "SETR_6000"]
P = RT.PRIMARY
QIDS, ANS, QS = RT.QIDS, RT.ANS, RT.QS
REC, VER = RT.load()
MACH = {a: {q: RT.score(REC, VER, P, a, q) for q in QIDS} for a in ARMS4}


def short(q):
    return q.replace("V2-test-", "")


def human_val(r):
    """export row -> (value or None, basis)."""
    note = (r.get("note") or "").lower()
    if "half correct" in note:
        return 0, "human_half_correct_rule"
    v = r.get("verdict")
    if v == "correct":
        return 1, "human_correct"
    if v == "wrong":
        return 0, "human_wrong"
    return None, "human_unsure"


def load_review(key_name, exp_name):
    key = json.loads((HC / key_name).read_text(encoding="utf-8-sig"))
    exp = {r["item"]: r for r in json.loads((HC / exp_name).read_text(encoding="utf-8-sig"))["results"]}
    return key, exp


# ---------------------------------------------------------------- per-cell human verdicts
key1, exp1 = load_review("key_test.json", "human_test_export.json")
key2, exp2 = load_review("key_test_sym.json", "human_test_sym_export.json")
missing_items = [it["item"] for it in key1["items"] if it["item"] not in exp1] + \
                [it["item"] for it in key2["items"] if it["item"] not in exp2]

H = {}   # (q, arm) -> dict(value, basis, source, item)
for it in key1["items"]:
    if it["item"] not in exp1:
        continue
    v, b = human_val(exp1[it["item"]])
    for a in it["arms"]:
        H[(it["qid"], a)] = dict(value=v, basis=b, source="direct_first" if len(it["arms"]) == 1
                                 else "direct_first_dedup", item=it["item"])
OLD = dict(H)
for it in key2["items"]:
    if it["item"] not in exp2:
        continue
    v, b = human_val(exp2[it["item"]])
    for a in it["arms"]:
        assert (it["qid"], a) not in H, (it, a)
        H[(it["qid"], a)] = dict(value=v, basis=b, source="direct_sym" if len(it["arms"]) == 1
                                 else "direct_sym_dedup", item=it["item"])
inh_check = []
for x in key2["inherited"]:
    assert any(OLD.get((x["qid"], a), {}).get("item") == x["from_item"] for a in ARMS4), x
    # re-derive with the half-correct rule from the source item (the key stores the raw button)
    v, b = human_val(exp1[x["from_item"]])
    inh_check.append(dict(qid=x["qid"], arm=x["arm"], from_item=x["from_item"], key_verdict=x["verdict"], rule_value=v))
    assert (x["qid"], x["arm"]) not in H
    H[(x["qid"], x["arm"])] = dict(value=v, basis=b + "_inherited", source="inherited_identical_answer",
                                   item=x["from_item"])

SEL = sorted({q for q, _ in H}, key=QIDS.index)
gaps = [(q, a) for q in SEL for a in ARMS4 if (q, a) not in H]
unsure_cells = [(q, a) for (q, a), h in H.items() if h["value"] is None]


def merged(hmap):
    out = {}
    for a in ARMS4:
        out[a] = {}
        for q in QIDS:
            lo, hi, st = MACH[a][q]
            h = hmap.get((q, a))
            if h is not None and h["value"] is not None:
                out[a][q] = (h["value"], h["value"], h["basis"])
            else:
                out[a][q] = (lo, hi, st)
    return out


S_M = {a: dict(MACH[a]) for a in ARMS4}
S_O = merged(OLD)
S_N = merged(H)

# reproduce the earlier (asymmetric) report exactly
prev = json.loads(PREV_PATH.read_text(encoding="utf-8"))
for a in ARMS4:
    pq = prev["per_question_glm_6000_human"][a]
    for q in QIDS:
        assert list(S_O[a][q][:2]) == pq[q][:2], ("old merge mismatch", a, q, S_O[a][q], pq[q])


def tally(S, a, qs):
    return sum(S[a][q][0] for q in qs), sum(S[a][q][1] for q in qs)


def fmt(t):
    return "%d" % t[0] if t[0] == t[1] else "%d-%d" % t


def outcomes(S, a, b, q):
    """possible diffs (a - b) on q over all assignments of unresolved cells."""
    va = [S[a][q][0]] if S[a][q][0] == S[a][q][1] else [0, 1]
    vb = [S[b][q][0]] if S[b][q][0] == S[b][q][1] else [0, 1]
    return sorted({x - y for x in va for y in vb})


def pair(S, a, b, qs=ANS):
    better = [q for q in qs if S[a][q][0] == 1 and S[b][q][1] == 0]
    worse = [q for q in qs if S[a][q][1] == 0 and S[b][q][0] == 1]
    unres = [q for q in qs if q not in better and q not in worse and
             (S[a][q][0] != S[a][q][1] or S[b][q][0] != S[b][q][1])]
    la, ha = tally(S, a, qs)
    lb, hb = tally(S, b, qs)
    ps = []
    for combo in itertools.product(*[outcomes(S, a, b, q) for q in unres]):
        nb = len(better) + sum(1 for d in combo if d > 0)
        nw = len(worse) + sum(1 for d in combo if d < 0)
        ps.append(RT.sign_p(nb, nw))
    return {"a": a, "b": b, "net_lower": la - hb, "net_upper": ha - lb, "better": better, "worse": worse,
            "unresolved": unres, "n_better": len(better), "n_worse": len(worse),
            "sign_p": RT.sign_p(len(better), len(worse)), "sign_p_range": [min(ps), max(ps)]}


COMPS = [("LEX5B_6000", "HYBRID_6000"), ("LEX5B_6000", "HR_6000"), ("SETR_6000", "LEX5B_6000")]
VERS = {"machine": S_M, "asym_human_old": S_O, "sym_human_new": S_N}
pairs = {v: {"%s vs %s" % c: pair(S, *c) for c in COMPS} for v, S in VERS.items()}
for v in VERS:
    g = [pairs[v]["%s vs %s" % c] for c in COMPS[:2]]
    adj = RT.holm([x["sign_p"] for x in g])
    adj_hi = RT.holm([x["sign_p_range"][1] for x in g])
    for x, h, hh in zip(g, adj, adj_hi):
        x["holm_p"], x["holm_p_worst"] = h, hh
    acc = ("passed" if min(x["net_lower"] for x in g) >= 5 else
           "failed" if min(x["net_upper"] for x in g) < 5 else "unresolved")
    pairs[v]["gate_accuracy"] = acc

table = {}
for a in ARMS4:
    table[a] = {v: {"all_96": tally(S, a, QIDS), "answerable_90": tally(S, a, ANS)} for v, S in VERS.items()}
    table[a]["cells_human_first"] = sum(1 for (q, x) in OLD if x == a)
    table[a]["cells_human_sym_total"] = sum(1 for (q, x) in H if x == a)

# ---------------------------------------------------------------- judge vs human agreement
def agree_rows(include_inherited):
    rows = []
    for (q, a), h in H.items():
        if h["source"] == "inherited_identical_answer" and not include_inherited:
            continue
        lo, hi, st = MACH[a][q]
        m = lo if lo == hi else None
        hv = h["value"]
        if m is None:
            k = "machine_needs_adj"
        elif hv is None:
            k = "human_unsure"
        elif m == hv:
            k = "agree"
        elif m == 0:
            k = "judge_strict(m0,h1)"
        else:
            k = "judge_lenient(m1,h0)"
        rows.append(dict(q=q, arm=a, item=h["item"], source=h["source"], machine=m, human=hv, kind=k,
                         basis=h["basis"]))
    return rows


KINDS = ["agree", "judge_strict(m0,h1)", "judge_lenient(m1,h0)", "machine_needs_adj", "human_unsure"]


def agree_summary(rows):
    by = {a: {k: 0 for k in KINDS} for a in ARMS4}
    tot = {k: 0 for k in KINDS}
    for r in rows:
        by[r["arm"]][r["kind"]] += 1
        tot[r["kind"]] += 1
    comp = tot["agree"] + tot["judge_strict(m0,h1)"] + tot["judge_lenient(m1,h0)"]
    return {"cells": len(rows), "overall": tot, "by_arm": by, "comparable": comp,
            "agreement_rate": round(tot["agree"] / comp, 4) if comp else None}


rows_direct = agree_rows(False)
rows_all = agree_rows(True)
AG = {"direct_reviewed": agree_summary(rows_direct), "including_inherited": agree_summary(rows_all),
      "sym_review_only": agree_summary([r for r in rows_direct if r["source"].startswith("direct_sym")]),
      "first_review_only": agree_summary([r for r in rows_direct if r["source"].startswith("direct_first")])}

# needs_adjudication cells of the four arms after the merge
na_left = [(a, q) for a in ARMS4 for q in QIDS if S_N[a][q][0] != S_N[a][q][1]]
na_resolved = [(a, q, S_N[a][q][0], S_N[a][q][2]) for a in ARMS4 for q in QIDS
               if MACH[a][q][2] == "needs_adjudication" and S_N[a][q][0] == S_N[a][q][1]]

res = {"rule": prev["rule"] + "; symmetric completion (key_test_sym.json) adds direct, deduplicated-identical and "
       "inherited-identical-answer verdicts so all 4 arms are human-judged on the 30 selected questions",
       "selected_questions": SEL, "coverage": {"cells_expected": 4 * len(SEL), "cells_covered": len(H),
       "gaps": gaps, "unsure_cells": unsure_cells, "items_missing_from_exports": missing_items,
       "by_source": {s: sum(1 for h in H.values() if h["source"] == s) for s in sorted({h["source"] for h in H.values()})}},
       "inherited_check": inh_check, "glm_6000_table": table, "pairs": pairs, "agreement": AG,
       "agreement_cells": rows_all, "needs_adjudication_unresolved_4arms": na_left,
       "needs_adjudication_resolved_by_human_4arms": na_resolved,
       "per_question_glm_6000_sym": {a: {q: list(S_N[a][q]) for q in QIDS} for a in ARMS4}}
(HERE / "test_results_human_sym.json").write_text(json.dumps(res, ensure_ascii=False, indent=1) + "\n",
                                                  encoding="utf-8")

CHANGES = """- **Ranking at 6,000 changes.** With every arm human-judged on the same 30 questions, the /90 order is SETR 78 > HR 76 = LEX5B 76 > HYBRID 71 (old asymmetric: LEX5B 76 > HR 73 > SETR 72 > HYBRID 68). LEX5B is no longer the best GLM @6,000 arm.
- **Withdrawn: "LEX5B beats HYBRID significantly under human adjudication" (+8, 9/1, Holm 0.043).** Symmetric: +5 (7/2), p 0.180, Holm 0.359, not significant. The old margin was an artifact of asymmetric review: LEX5B was reviewed on all 30 questions but the controls only on discordant ones, and the judge errs almost only in the strict direction (37 m0->h1, 0 m1->h0), so reviewing more cells of one arm inflates that arm.
- **Withdrawn: "LEX5B leads HR by +3".** Symmetric: 0 (5/5), p 1.000.
- **Reversed: "SETR is worse than LEX5B" (-4, 2/6).** Symmetric: +2 (3/1), p 0.625, i.e. no detectable difference. The judge under-scored SETR most (15 of 30 SETR cells were m0,h1; agreement 50%).
- **Unchanged:** the pre-registered primary (machine) gate FAILED; the secondary human re-check also FAILS (net vs HYBRID exactly +5 meets that half, net vs HR 0 does not); no superiority claim survives Holm; the token gate still passes; the judge is never lenient on reviewed cells, and overall judge-human agreement stays about two-thirds (79/116 = 68.1%).
- **Caveat:** the 66 unreviewed questions keep machine verdicts for all arms, so absolute totals remain lower bounds of human-judged accuracy where the judge was strict; the comparison is now symmetric only within the 30 selected questions, which were chosen as discordant under the machine judge.
"""

# ---------------------------------------------------------------- markdown
L = []
w = L.append
cov = res["coverage"]
w("# TRUE TEST RESULTS, SYMMETRIC HUMAN ADJUDICATION (GLM-5.3-flash reader, 6,000-character review)\n")
w("Computed 2026-10-06 by `test_run/human_merge_sym.py` (no model calls). Inputs: first review "
  "(`human_check/key_test.json` + `human_test_export.json`, 77 items = 78 cells) and the symmetric completion "
  "(`key_test_sym.json` + `human_test_sym_export.json`, 39 items = 40 cells shown, plus %d cells inherited from a "
  "byte-identical reviewed answer of the same question). Machine verdicts from `report_test.load()/score()`; the "
  "earlier asymmetric merge (`test_results_human.json`) is reproduced cell-for-cell before the new verdicts are "
  "added.\n" % len(key2["inherited"]))
w("**Rule:** a note containing \"half correct\" (any case) = WRONG regardless of button; \"unsure\" keeps the "
  "machine verdict. On the 30 selected questions every arm now carries a human verdict; on the other 66 the machine "
  "strict verdict stands, and a needs_adjudication cell there would be a bound `a-b`.\n")
w("**Coverage:** %d of %d cells on %d questions are human-judged (%s). Gaps: %s. Unsure cells: %s. Export items "
  "missing: %s.\n" % (cov["cells_covered"], cov["cells_expected"], len(SEL),
                      ", ".join("%s %d" % kv for kv in cov["by_source"].items()),
                      "none" if not gaps else "; ".join("%s %s" % (short(q), a) for q, a in gaps),
                      "none" if not unsure_cells else "; ".join("%s %s" % (short(q), a) for q, a in unsure_cells),
                      "none" if not missing_items else ", ".join(missing_items)))
w("Inherited cells (rule re-applied to the source item): " + "; ".join(
    "%s %s <- %s = %s" % (short(x["qid"]), x["arm"], x["from_item"], x["rule_value"]) for x in inh_check) + ".\n")

w("## 1. GLM-5.3-flash @6,000: correct answers\n")
w("| arm | machine /96 | asym. human (old) /96 | **sym. human (new) /96** | machine /90 | asym. human (old) /90 | "
  "**sym. human (new) /90** | human cells old -> new |")
w("|---|---:|---:|---:|---:|---:|---:|---:|")
for a in ARMS4:
    t = table[a]
    w("| %s | %s | %s | **%s** | %s | %s | **%s** | %d -> %d |" % (
        a, fmt(t["machine"]["all_96"]), fmt(t["asym_human_old"]["all_96"]), fmt(t["sym_human_new"]["all_96"]),
        fmt(t["machine"]["answerable_90"]), fmt(t["asym_human_old"]["answerable_90"]),
        fmt(t["sym_human_new"]["answerable_90"]), t["cells_human_first"], t["cells_human_sym_total"]))
w("\nThe 15,500 arms and the five other readers are unchanged from TEST_RESULTS_HUMAN.md (machine only).\n")

w("## 2. Paired comparisons (90 answerable; exact two-sided sign test)\n")
w("| comparison | machine net (b/w, p) | asym. human old net (b/w, p) | **sym. human new net (b/w, p, Holm)** |")
w("|---|---|---|---|")


def fmtn(lo, hi):
    return "%+d" % lo if lo == hi else "%+d..%+d" % (lo, hi)


def cellp(x, holm=True):
    s = "%s (%d/%d, p %.4f" % (fmtn(x["net_lower"], x["net_upper"]), x["n_better"], x["n_worse"], x["sign_p"])
    if x["sign_p_range"][0] != x["sign_p_range"][1]:
        s += ", range %.4f-%.4f" % tuple(x["sign_p_range"])
    if holm and "holm_p" in x:
        s += ", Holm %.4f" % x["holm_p"]
    return s + ")"


for c in COMPS:
    k = "%s vs %s" % c
    w("| %s | %s | %s | **%s** |" % (k, cellp(pairs["machine"][k]), cellp(pairs["asym_human_old"][k]),
                                     cellp(pairs["sym_human_new"][k])))
w("\nHolm is over the two gate comparisons (LEX5B vs HYBRID, LEX5B vs HR); SETR vs LEX5B is descriptive. "
  "\"range\" = sign p over every assignment of unresolved (needs_adjudication) cells; absent when there are none.\n")
for c in COMPS:
    k = "%s vs %s" % c
    x = pairs["sym_human_new"][k]
    w("- **%s** (symmetric): net %s, better %d / worse %d, p %.4f%s; sign-p over unresolved assignments %.4f-%.4f." % (
        k, fmtn(x["net_lower"], x["net_upper"]), x["n_better"], x["n_worse"], x["sign_p"],
        (", Holm %.4f (worst case %.4f)" % (x["holm_p"], x["holm_p_worst"])) if "holm_p" in x else "",
        x["sign_p_range"][0], x["sign_p_range"][1]))
    w("  - Better: %s." % (", ".join(map(short, x["better"])) or "none"))
    w("  - Worse: %s." % (", ".join(map(short, x["worse"])) or "none"))
    w("  - Unresolved: %s." % (", ".join(map(short, x["unresolved"])) or "none"))
w("")

w("## 3. Pre-registered gate re-checked (SECONDARY analysis)\n")
w("The pre-registered primary analysis (machine judge) FAILED and stays the primary result. Criterion: LEX5B_6000 net "
  ">= +5/90 vs HYBRID_6000 AND vs HR_6000; superiority = Holm-adjusted sign p < 0.05.\n")
w("| check | machine (primary) | asym. human (old) | **sym. human (new)** |")
w("|---|---|---|---|")
for c in COMPS[:2]:
    k = "%s vs %s" % c
    w("| net vs %s /90 | %s | %s | **%s** |" % (c[1], *(fmtn(pairs[v][k]["net_lower"], pairs[v][k]["net_upper"]) for v in VERS)))
w("| accuracy gate | %s | %s | **%s** |" % tuple(pairs[v]["gate_accuracy"].upper() for v in VERS))
for c in COMPS[:2]:
    k = "%s vs %s" % c
    w("| superiority vs %s (Holm < 0.05) | %s | %s | **%s** |" % (c[1], *(
        ("yes" if pairs[v][k]["holm_p"] < 0.05 else "no") + " (%.4f)" % pairs[v][k]["holm_p"] for v in VERS)))
w("| tokens (<= 2.5x HYBRID) | PASSED (2.04-2.32x) | unchanged | unchanged |\n")

w("## 4. Judge-human agreement on all reviewed cells\n")
w("Unit = one (question, arm) cell. \"Judge strict\" = machine 0, human 1; \"judge lenient\" = machine 1, human 0. "
  "needs_adjudication (machine) and unsure (human) cells are not comparable. The first review covered discordant "
  "questions only; the symmetric completion covers the remaining arms of the same questions.\n")
for name, lab in (("direct_reviewed", "Directly reviewed cells (both reviews)"),
                  ("including_inherited", "Including the inherited cells"),
                  ("first_review_only", "First review only"), ("sym_review_only", "Symmetric review only")):
    g = AG[name]
    o = g["overall"]
    w("**%s** (%d cells): agree %d, judge strict %d, judge lenient %d, machine needs_adj. %d, human unsure %d; "
      "agreement on comparable cells %d/%d = %.1f%%.\n" % (lab, g["cells"], o["agree"], o["judge_strict(m0,h1)"],
      o["judge_lenient(m1,h0)"], o["machine_needs_adj"], o["human_unsure"], o["agree"], g["comparable"],
      100 * g["agreement_rate"]))
w("| arm | agree | judge strict (m0,h1) | judge lenient (m1,h0) | machine needs_adj. | human unsure | agreement |")
w("|---|---:|---:|---:|---:|---:|---:|")
for a in ARMS4:
    b = AG["including_inherited"]["by_arm"][a]
    comp = b["agree"] + b["judge_strict(m0,h1)"] + b["judge_lenient(m1,h0)"]
    w("| %s | %d | %d | %d | %d | %d | %d/%d = %.1f%% |" % (a, *(b[k] for k in KINDS), b["agree"], comp,
                                                           100 * b["agree"] / comp if comp else 0))
w("\n(Table includes the %d inherited cells.) Non-agreeing cells: %s.\n" % (
    len(key2["inherited"]),
    "; ".join("%s %s %s m=%s h=%s%s" % (r["item"], short(r["q"]), r["arm"], r["machine"], r["human"],
              " (half correct)" if "half" in r["basis"] else "") for r in rows_all if r["kind"] != "agree")))

w("## 5. needs_adjudication cells (four GLM @6,000 arms)\n")
w("Resolved by human review: " + ("; ".join("%s %s -> %d (%s)" % (a, short(q), v, b) for a, q, v, b in na_resolved)
                                  or "none") + ".")
w("Still unresolved (bounds): " + ("; ".join("%s %s" % (a, short(q)) for a, q in na_left) or "none") + ".\n")

w("## 6. What changes relative to TEST_RESULTS_HUMAN.md\n")
w(CHANGES)
(HERE / "TEST_RESULTS_HUMAN_SYM.md").write_text("\n".join(L), encoding="utf-8")

print(json.dumps({"coverage": {k: cov[k] for k in ("cells_expected", "cells_covered", "gaps", "unsure_cells",
                                                   "items_missing_from_exports", "by_source")},
                  "table": {a: {v: table[a][v] for v in VERS} for a in ARMS4},
                  "pairs": {v: {k: (x if isinstance(x, str) else {kk: x[kk] for kk in ("net_lower", "net_upper",
                            "n_better", "n_worse", "sign_p", "sign_p_range") + (("holm_p",) if "holm_p" in x else ())})
                                for k, x in pairs[v].items()} for v in VERS},
                  "agreement": {k: {"overall": g["overall"], "rate": g["agreement_rate"]} for k, g in AG.items()},
                  "na_left": na_left, "inh": inh_check}, indent=1, default=str))
