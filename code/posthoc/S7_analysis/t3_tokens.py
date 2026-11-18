"""Task 3 (review item 12): budgets in reader tokens per language, a token-matched view, the Persian/English
tokens-per-character ratio reconciled, and SETR_NOFILL vs HYBRID_3000 / HR_3000.
-> tokens.json, tokens_tables.md, fig_tokens_acc.dat"""
import json
import statistics as st
from collections import defaultdict

import numpy as np

import s7lib as L

SWEEP = {"HYBRID": [3000, 6000, 10000, 15500, 24000], "HR": [3000, 6000, 10000, 15500, 24000],
         "FUSION": [3000, 6000, 10000, 15500, 24000], "LEX5B": [6000, 15500], "SETR": [6000, 15500]}
EXTRA = ["SETR_NOFILL", "LISTWISE", "GOLD", "ITER_BM25", "GREP_AGENT", "CLOSEDBOOK"]
ARMS = ["%s_%d" % (p, b) for p, bs in SWEEP.items() for b in bs] + EXTRA

# ------------------------------------------------------------------ answer-call usage per (arm, q)
_ledg = {}


def _ledger(path):
    if path not in _ledg:
        _ledg[path] = L.jl(path)
    return _ledg[path]


def frozen_answer_call(arm, q):
    r = L.REC[L.P][arm][q]
    calls = [c for c in L.RT.ledgers(L.TR / "runs" / L.pool(q) / arm)
             if c.get("question_id") == q and c.get("run_id") == r.get("run_id") and c.get("purpose") == "answer"
             and c.get("ok")]
    return calls[-1] if calls else None


def s6_answer_call(arm, q):
    r = L.OWN[arm][0][q]
    calls = [c for c in _ledger(L.PH / "S6_baselines" / "ledger.jsonl") if c.get("approach") == arm and
             c.get("question_id") == q and c.get("run_id") == r.get("run_id") and c.get("purpose") == "answer" and c.get("ok")]
    return calls[-1] if calls else None


def s3_answer_call(arm, q):
    calls = [c for c in _ledger(L.PH / "S3_agentic" / "ledger.jsonl") if c.get("arm") == arm and c.get("run_tag") == "test"
             and c.get("question_id") == q and c.get("purpose") == "answer" and c.get("ok", True)]
    return calls[-1] if calls else None


def usage(arm, q, depth=0):
    """-> (prompt_tokens, completion_tokens) of the answer call, or None."""
    if arm in L.FROZEN_ARMS:
        c = frozen_answer_call(arm, q)
        return (c["prompt_tokens"], c["completion_tokens"]) if c and c.get("usage_known") else None
    if arm in ("SETR_NOFILL", "LISTWISE", "CLOSEDBOOK"):
        c = s6_answer_call(arm, q)
        return (c["prompt_tokens"], c["completion_tokens"]) if c and c.get("usage_known", True) else None
    if arm in ("ITER_BM25", "GREP_AGENT"):
        c = s3_answer_call(arm, q)
        return (c["prompt_tokens"], c["completion_tokens"]) if c else None
    r = L.S2A_ANS.get((arm, q))
    if r is None:
        return None
    src = r["source"]
    if src == "call":
        return r["prompt_tokens"], r["completion_tokens"]
    if src.startswith("frozen:"):
        return usage(src.split(":", 1)[1], q)
    if src.startswith("dup:") and depth < 3:
        return usage(src.split(":", 1)[1], q, depth + 1)
    return None


rows = []
for a in ARMS:
    for q in L.QIDS:
        if a == "GOLD" and q not in L.ANS:
            continue
        m = L.messages(a, q)
        u = usage(a, q)
        if m is None:
            continue
        rows.append(dict(arm=a, q=q, lang=L.lang(q), answerable=q in L.ANS,
                         prompt_chars=sum(len(x["content"]) for x in m), evidence_chars=len(L.evidence_block(m)),
                         system_chars=len(m[0]["content"]), pt=u[0] if u else None, ct=u[1] if u else None))
by = defaultdict(list)
for r in rows:
    by[(r["arm"], r["lang"])].append(r)

res = {"cells": {}, "ratios": {}}
for a in ARMS:
    out = {}
    for lg in ("fa", "en"):
        rs = [r for r in by[(a, lg)] if r["pt"] is not None]
        out[lg] = {"n": len(rs), "n_missing_usage": len(by[(a, lg)]) - len(rs),
                   "mean_prompt_tokens": round(st.mean(r["pt"] for r in rs), 1) if rs else None,
                   "mean_completion_tokens": round(st.mean(r["ct"] for r in rs), 1) if rs else None,
                   "mean_evidence_chars": round(st.mean(r["evidence_chars"] for r in by[(a, lg)]), 1),
                   "mean_prompt_chars": round(st.mean(r["prompt_chars"] for r in by[(a, lg)]), 1)}
        qs = [q for q in L.ANS if L.lang(q) == lg]
        out[lg]["acc45"] = list(L.total(a, qs))
    out["acc90"] = list(L.total(a))
    res["cells"][a] = out

# ------------------------------------------------------------------ the ratio, four ways (retrieval-pack answer calls)
RET = ["%s_%d" % (p, b) for p, bs in SWEEP.items() for b in bs] + ["SETR_NOFILL", "LISTWISE", "GOLD"]
R = [r for r in rows if r["arm"] in RET and r["pt"] is not None]


def per_lang(f):
    return {lg: f([r for r in R if r["lang"] == lg]) for lg in ("fa", "en")}


pooled = per_lang(lambda rs: sum(r["pt"] for r in rs) / sum(r["prompt_chars"] for r in rs))
median = per_lang(lambda rs: st.median(r["pt"] / r["prompt_chars"] for r in rs))


def slope(rs):
    X = np.array([[1.0, r["evidence_chars"], r["system_chars"]] for r in rs])
    y = np.array([r["pt"] for r in rs], float)
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    return {"intercept": float(b[0]), "tokens_per_evidence_char": float(b[1]), "tokens_per_system_char": float(b[2])}


marg = per_lang(slope)
# same-budget ratio of mean prompt tokens (S2a's reading), per sweep cell
same_budget = {}
for a in RET:
    c = res["cells"][a]
    if c["fa"]["mean_prompt_tokens"] and c["en"]["mean_prompt_tokens"]:
        same_budget[a] = round(c["fa"]["mean_prompt_tokens"] / c["en"]["mean_prompt_tokens"], 3)
# question-only overhead: CLOSEDBOOK prompts (system + question, no evidence)
cb = per_lang(lambda rs: None)
cbr = [r for r in rows if r["arm"] == "CLOSEDBOOK" and r["pt"] is not None]
cb = {lg: {"mean_prompt_tokens": st.mean(r["pt"] for r in cbr if r["lang"] == lg),
           "mean_prompt_chars": st.mean(r["prompt_chars"] for r in cbr if r["lang"] == lg)} for lg in ("fa", "en")}
dev_calib = json.loads((L.WORK / "datasets_v2" / "lex10" / "calib10.json").read_text(encoding="utf-8"))["fit"]
lc = json.loads((L.PH / "S4_route_long" / "lc_results.json").read_text(encoding="utf-8"))
res["ratios"] = {
    "paper_1.66_source": {"file": "work/datasets_v2/lex10/calib10.json (eval10.py stage_calib)",
                          "what": "sum(GLM prompt tokens)/sum(prompt chars) of whole answer prompts (system+evidence+question), "
                                  "36 dev answer calls per language (HYBRID, HYBRID+rerank, LEX7.1 on 12 dev questions)",
                          "fa": dev_calib["fa"]["glm_tokens_per_char"], "en": dev_calib["en"]["glm_tokens_per_char"],
                          "ratio": dev_calib["fa"]["glm_tokens_per_char"] / dev_calib["en"]["glm_tokens_per_char"]},
    "pooled_whole_prompt_test": {"fa": pooled["fa"], "en": pooled["en"], "ratio": pooled["fa"] / pooled["en"],
                                 "n_calls": len(R)},
    "median_per_call_whole_prompt_test": {"fa": median["fa"], "en": median["en"], "ratio": median["fa"] / median["en"]},
    "marginal_evidence_test": {"fa": marg["fa"], "en": marg["en"],
                               "ratio": marg["fa"]["tokens_per_evidence_char"] / marg["en"]["tokens_per_evidence_char"]},
    "same_budget_mean_prompt_ratio": same_budget,
    "closedbook_prompt_overhead": cb,
    "s4_document_only_prompts": {a: {"median_tokens_per_char": lc["arms"][a]["prompt_tokens_per_char_median"],
                                     "ratio": lc["arms"][a]["fa_over_en_tokens_per_char"]} for a in ("LC-GOLD-DOCS", "LC-BM25-DOCS")},
    "s4_test_answer_median": json.loads((L.PH / "S4_route_long" / "pool_stats.json").read_text(encoding="utf-8"))["fa_over_en"],
}

# ------------------------------------------------------------------ token-matched view
# per language accuracy vs mean reader prompt tokens; and mixed cells that equalise tokens across languages
TM = {"TM_low": {"fa": 3000, "en": 6000}, "TM_mid": {"fa": 6000, "en": 10000}, "TM_high": {"fa": 10000, "en": 15500},
      "TM_top": {"fa": 15500, "en": 24000}}
tm = {}
for p in ("HYBRID", "HR", "FUSION"):
    for name, bud in TM.items():
        arm_of = lambda q: "%s_%d" % (p, bud[L.lang(q)])
        f = lambda q: L.score(arm_of(q), q)
        toks = {lg: res["cells"]["%s_%d" % (p, bud[lg])][lg]["mean_prompt_tokens"] for lg in ("fa", "en")}
        char_ref = "%s_%d" % (p, bud["fa"])          # same char budget as the Persian half
        tm["%s|%s" % (p, name)] = {"budgets": bud, "mean_prompt_tokens": toks,
                                   "acc90": [sum(f(q)[0] for q in L.ANS), sum(f(q)[1] for q in L.ANS)],
                                   "vs_char_matched_%s" % char_ref: L.compare(f, L.S(char_ref))}
res["token_matched"] = tm
# smallest-pack contrast
small = {}
for b in ("HYBRID_3000", "HR_3000", "FUSION_3000", "HYBRID_6000", "SETR_6000"):
    small["SETR_NOFILL vs " + b] = L.compare(L.S("SETR_NOFILL"), L.S(b))
    for lg in ("fa", "en"):
        small["SETR_NOFILL vs %s|%s" % (b, lg)] = L.compare(L.S("SETR_NOFILL"), L.S(b), [q for q in L.ANS if L.lang(q) == lg])
res["setr_nofill_small"] = small
L.wjson(L.HERE / "tokens.json", res)

# ------------------------------------------------------------------ outputs
with open(L.HERE / "fig_tokens_acc.dat", "w", encoding="utf-8") as fh:
    fh.write("# S7 task 3: mean answer-call prompt tokens vs strict accuracy per language (45 questions each); tokens.json\n")
    fh.write("pipeline budget lang mean_prompt_tokens mean_evidence_chars acc_lo acc_hi\n")
    for p, bs in SWEEP.items():
        for b in bs:
            for lg in ("fa", "en"):
                c = res["cells"]["%s_%d" % (p, b)][lg]
                fh.write("%s %d %s %.1f %.1f %d %d\n" % (p, b, lg, c["mean_prompt_tokens"], c["mean_evidence_chars"],
                                                         *c["acc45"]))
    for a in ("SETR_NOFILL", "LISTWISE", "GOLD"):
        for lg in ("fa", "en"):
            c = res["cells"][a][lg]
            fh.write("%s 0 %s %.1f %.1f %d %d\n" % (a, lg, c["mean_prompt_tokens"], c["mean_evidence_chars"], *c["acc45"]))

md = ["# S7 task 3: tokens (generated by t3_tokens.py -> tokens.json)", "",
      "Answer-call prompt tokens as reported by the provider (GLM-5.3-flash via Together), mean per question, all 96",
      "questions (48 fa / 48 en; GOLD 45/45 answerable only). Accuracy = machine strict on the 45 answerable per language.", "",
      "| cell | fa tokens | fa ev chars | fa acc /45 | en tokens | en ev chars | en acc /45 | fa/en tokens | all /90 |",
      "|---|---|---|---|---|---|---|---|---|"]
for a in ARMS:
    c = res["cells"][a]
    f, e = c["fa"], c["en"]
    rr = (f["mean_prompt_tokens"] / e["mean_prompt_tokens"]) if f["mean_prompt_tokens"] and e["mean_prompt_tokens"] else 0
    md.append("| %s | %s | %.0f | %s | %s | %.0f | %s | %.2f | %s |" % (
        a, "%.0f" % f["mean_prompt_tokens"] if f["mean_prompt_tokens"] else "-", f["mean_evidence_chars"], L.fmt(*f["acc45"]),
        "%.0f" % e["mean_prompt_tokens"] if e["mean_prompt_tokens"] else "-", e["mean_evidence_chars"], L.fmt(*e["acc45"]),
        rr, L.fmt(*c["acc90"])))
ra = res["ratios"]
md += ["", "## Persian/English tokens-per-character ratio, four ways", "",
       "| measure | fa | en | fa/en |", "|---|---|---|---|",
       "| paper 1.66: pooled whole-prompt tokens/char, 36+36 dev calls (lex10/calib10.json) | %.4f | %.4f | %.3f |" % (
           ra["paper_1.66_source"]["fa"], ra["paper_1.66_source"]["en"], ra["paper_1.66_source"]["ratio"]),
       "| same, test: pooled whole-prompt tokens/char over %d retrieval answer calls | %.4f | %.4f | %.3f |" % (
           ra["pooled_whole_prompt_test"]["n_calls"], ra["pooled_whole_prompt_test"]["fa"], ra["pooled_whole_prompt_test"]["en"],
           ra["pooled_whole_prompt_test"]["ratio"]),
       "| S4: median of per-call whole-prompt tokens/char, test | %.4f | %.4f | %.3f |" % (
           ra["median_per_call_whole_prompt_test"]["fa"], ra["median_per_call_whole_prompt_test"]["en"],
           ra["median_per_call_whole_prompt_test"]["ratio"]),
       "| marginal tokens per evidence char (OLS: tokens ~ 1 + evidence chars + system chars), test | %.4f | %.4f | %.3f |" % (
           ra["marginal_evidence_test"]["fa"]["tokens_per_evidence_char"], ra["marginal_evidence_test"]["en"]["tokens_per_evidence_char"],
           ra["marginal_evidence_test"]["ratio"]),
       "", "Same-budget ratio of mean prompt tokens (S2a's reading): " +
       ", ".join("%s %.2f" % kv for kv in ra["same_budget_mean_prompt_ratio"].items()),
       "", "Closed-book prompts (system + question only): fa %.0f tokens / %.0f chars, en %.0f / %.0f." % (
           ra["closedbook_prompt_overhead"]["fa"]["mean_prompt_tokens"], ra["closedbook_prompt_overhead"]["fa"]["mean_prompt_chars"],
           ra["closedbook_prompt_overhead"]["en"]["mean_prompt_tokens"], ra["closedbook_prompt_overhead"]["en"]["mean_prompt_chars"]),
       "S4 document-only long prompts: %s" % json.dumps(ra["s4_document_only_prompts"]), "",
       "## token-matched mixed cells (Persian at budget fa, English at budget en)", "",
       "| pipeline | fa/en budgets | mean tokens fa / en | acc /90 | vs char-matched (fa budget for both) net, b/w, p |", "|---|---|---|---|---|"]
for k, v in tm.items():
    ref = [x for x in v if x.startswith("vs_")][0]
    c = v[ref]
    md.append("| %s | %d / %d | %.0f / %.0f | %s | %s: %s (%d/%d) p %s |" % (
        k, v["budgets"]["fa"], v["budgets"]["en"], v["mean_prompt_tokens"]["fa"], v["mean_prompt_tokens"]["en"],
        L.fmt(*v["acc90"]), ref[3:], L.fmt_net(c), *c["better_worse_resolved"], L.fmt_p(c)))
md += ["", "## SETR_NOFILL (selected set only) vs the smallest packs", "",
       "| contrast | net | b/w | sign p | CI q | CI comp |", "|---|---|---|---|---|---|"]
for k, c in small.items():
    md.append("| %s | %s | %d/%d | %s | %s | %s |" % (k, L.fmt_net(c), *c["better_worse_resolved"], L.fmt_p(c), L.fmt_ci(c),
                                                    L.fmt_ci(c, "ci_pp_component")))
(L.HERE / "tokens_tables.md").write_text("\n".join(md) + "\n", encoding="utf-8")
print("\n".join(md))
