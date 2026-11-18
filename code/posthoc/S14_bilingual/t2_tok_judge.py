"""S14 task 2 (FRAMING extra analysis 4): tokenizers and judges by language. Saved data only.
-> tokjudge.json, tokjudge_tables.md

(a) Persian/English reader tokens per evidence character for each of the six readers: provider-reported prompt tokens
    of the answer call (frozen ledgers for GLM, replay ledgers for the other five) on the byte-identical frozen packs of
    the 8 frozen arms x 96 questions; per language OLS tokens ~ 1 + evidence chars + system chars (S7 t3_tokens.py
    method); ratio = fa slope / en slope; 95% CI by a question-cluster bootstrap (2,000 resamples, seed 0).
(b) judge 1 (pinned gpt-oss-120b) vs judge 2 (mistral-large-2512) agreement and Cohen's kappa per language on the 768
    GLM cells (S1); each judge vs the human on the 120 reviewed cells per language; the S7 cause labels of the 66
    disagreements by language; question-level permutation tests of the language difference."""
import json
import random
import sys
from collections import defaultdict

import numpy as np

import s14lib as L

sys.path.insert(0, str(L.PH / "S1_evaluator"))
import j2lib  # noqa: E402

RT = L.RT
FROZEN_PACKDIR = {"HYBRID_6000": "HYBRID", "HR_6000": "HYBRID_rerank-0c0612fe", "LEX5B_6000": "LEX5B",
                  "SETR_6000": "SETR", "HYBRID_15500": "HYBRID15500", "HR_15500": "HR15500_rerank-dcb91474",
                  "LEX5B_15500": "LEX5B_15500", "SETR_15500": "SETR_15500"}
QRUN = {d["id"]: d for d in L.jl(L.FZ / "questions" / "test_v2judge_all.jsonl")}


def messages(arm, q):
    f = L.TR / "packs" / FROZEN_PACKDIR[arm] / ("%s__%s.json" % (q, QRUN[q]["meta"]["setting"]))
    return json.loads(f.read_text(encoding="utf-8"))["messages"] if f.is_file() else None


def evidence_block(msgs):
    u = msgs[1]["content"]
    if not u.startswith("Evidence:\n"):
        return ""
    i = u.rfind("\n\nQuestion: ")
    return u[len("Evidence:\n"):i]


_LEDG = {}


def answer_call(rd, arm, q):
    d = RT.cell_dir(rd, q, arm)
    if d not in _LEDG:
        _LEDG[d] = RT.ledgers(d)
    r = L.REC[rd][arm].get(q)
    if r is None or r.get("error"):
        return None
    calls = [c for c in _LEDG[d] if c.get("question_id") == q and c.get("purpose") == "answer" and c.get("ok")
             and c.get("run_id") == r.get("run_id")]
    return calls[-1] if calls else None


# ------------------------------------------------------------------ (a) tokens
rows = []
MSG = {}
for arm in FROZEN_PACKDIR:
    for q in RT.QIDS:
        m = messages(arm, q)
        if m is None:
            continue
        MSG[(arm, q)] = (len(evidence_block(m)), len(m[0]["content"]), sum(len(x["content"]) for x in m))
for rd in L.READERS:
    for arm in FROZEN_PACKDIR:
        for q in RT.QIDS:
            if (arm, q) not in MSG:
                continue
            c = answer_call(rd, arm, q)
            if c is None or not c.get("usage_known", True) or not c.get("prompt_tokens"):
                continue
            ev, sc, pc = MSG[(arm, q)]
            rows.append(dict(reader=rd, arm=arm, q=q, lang=L.LANG[q], pt=c["prompt_tokens"], ev=ev, sys=sc,
                             prompt_chars=pc, ledger_prompt_chars=c.get("prompt_chars")))


def ols(rs):
    X = np.array([[1.0, r["ev"], r["sys"]] for r in rs]); y = np.array([r["pt"] for r in rs], float)
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    return b


TOK = {}
rng = random.Random(0)
for rd in L.READERS:
    R = [r for r in rows if r["reader"] == rd]
    out = {"n_calls": len(R), "n_calls_by_lang": {lg: sum(r["lang"] == lg for r in R) for lg in ("fa", "en")},
           "ledger_prompt_chars_match_pack": sum(1 for r in R if r["ledger_prompt_chars"] == r["prompt_chars"]),
           "ledger_prompt_chars_known": sum(1 for r in R if r["ledger_prompt_chars"] is not None)}
    by_q = defaultdict(list)
    for r in R:
        by_q[r["q"]].append(r)
    for lg in ("fa", "en"):
        rs = [r for r in R if r["lang"] == lg]
        b = ols(rs)
        out[lg] = {"intercept": float(b[0]), "tokens_per_evidence_char": float(b[1]), "tokens_per_system_char": float(b[2]),
                   "pooled_whole_prompt_tokens_per_char": sum(r["pt"] for r in rs) / sum(r["prompt_chars"] for r in rs),
                   "median_per_call_tokens_per_char": float(np.median([r["pt"] / r["prompt_chars"] for r in rs]))}
    out["ratio_marginal_evidence"] = out["fa"]["tokens_per_evidence_char"] / out["en"]["tokens_per_evidence_char"]
    out["ratio_pooled_whole_prompt"] = (out["fa"]["pooled_whole_prompt_tokens_per_char"] /
                                        out["en"]["pooled_whole_prompt_tokens_per_char"])
    # question-cluster bootstrap of the marginal ratio
    qfa = sorted({r["q"] for r in R if r["lang"] == "fa"}); qen = sorted({r["q"] for r in R if r["lang"] == "en"})
    sims = []
    for _ in range(2000):
        sf = [r for _k in qfa for r in by_q[qfa[rng.randrange(len(qfa))]]]
        se = [r for _k in qen for r in by_q[qen[rng.randrange(len(qen))]]]
        sims.append(ols(sf)[1] / ols(se)[1])
    sims.sort()
    out["ratio_marginal_ci95"] = [sims[50], sims[1949]]
    TOK[rd] = out

# ------------------------------------------------------------------ (b) judges
HUM, SEL = j2lib.human_sym()
ARMS8 = list(RT.ARMS)


def pairs_j1j2(arms, qs):
    out, unres = [], 0
    for a in arms:
        for q in qs:
            x, y = j2lib.judge1(a, q), j2lib.judge2(a, q)[:2]
            if x[0] != x[1] or y[0] != y[1]:
                unres += 1
                continue
            out.append((x[0], y[0], q))
    return out, unres


def agree_stats(pairs):
    p2 = [(x, y) for x, y, *_ in pairs]
    n = len(p2)
    return {"comparable": n, "agree": sum(x == y for x, y in p2), "agree_pct": 100.0 * sum(x == y for x, y in p2) / n if n else None,
            "kappa": j2lib.kappa(p2), "a1_b0": sum(x > y for x, y in p2), "a0_b1": sum(x < y for x, y in p2),
            "a_correct": sum(x for x, _ in p2), "b_correct": sum(y for _, y in p2)}


def perm_lang_diff(pairs, stat, n_perm=10000, seed=0):
    """question-level permutation of the language label: two-sided p for stat(fa) - stat(en)."""
    by_q = defaultdict(list)
    for p in pairs:
        by_q[p[2]].append(p)
    qs = sorted(by_q)
    lab = [L.LANG[q] for q in qs]
    obs_fa = stat([p for q in qs if L.LANG[q] == "fa" for p in by_q[q]])
    obs_en = stat([p for q in qs if L.LANG[q] == "en" for p in by_q[q]])
    obs = obs_fa - obs_en
    r = random.Random(seed)
    hits = 0
    for _ in range(n_perm):
        lp = lab[:]
        r.shuffle(lp)
        a = stat([p for q, lg in zip(qs, lp) if lg == "fa" for p in by_q[q]])
        b = stat([p for q, lg in zip(qs, lp) if lg == "en" for p in by_q[q]])
        if a is not None and b is not None and abs(a - b) >= abs(obs) - 1e-12:
            hits += 1
    return {"diff_fa_minus_en": obs, "p": (hits + 1) / (n_perm + 1)}


def boot_ci_diff(pairs, stat, n_boot=2000, seed=0):
    by_q = defaultdict(list)
    for p in pairs:
        by_q[p[2]].append(p)
    qf = sorted(q for q in by_q if L.LANG[q] == "fa"); qe = sorted(q for q in by_q if L.LANG[q] == "en")
    r = random.Random(seed)
    sims = []
    for _ in range(n_boot):
        a = stat([p for _k in qf for p in by_q[qf[r.randrange(len(qf))]]])
        b = stat([p for _k in qe for p in by_q[qe[r.randrange(len(qe))]]])
        if a is not None and b is not None:
            sims.append(a - b)
    sims.sort()
    return [sims[int(0.025 * len(sims))], sims[int(0.975 * len(sims)) - 1]]


kap = lambda ps: j2lib.kappa([(x, y) for x, y, *_ in ps])
dis = lambda ps: (100.0 * sum(x != y for x, y, *_ in ps) / len(ps)) if ps else None
JJ = {}
SYN_Q = sorted({c["q"] for c in json.loads((L.PH / "S7_analysis" / "errors.json").read_text(encoding="utf-8"))["a_cells"]
                if c["label"] == "synonym_fa"})
for scope, arms in (("all8", ARMS8), ("6000x4", j2lib.ARMS4)):
    for qscope, qs in (("all96", RT.QIDS), ("answerable90", L.ANS),
                       ("all96_without_synonym_question", [q for q in RT.QIDS if q not in SYN_Q])):
        pr, unres = pairs_j1j2(arms, qs)
        d = {"unresolved": unres}
        for lg in ("fa", "en"):
            d[lg] = agree_stats([p for p in pr if L.LANG[p[2]] == lg])
            d[lg]["questions"] = len({p[2] for p in pr if L.LANG[p[2]] == lg})
        d["all"] = agree_stats(pr)
        d["kappa_diff_perm"] = perm_lang_diff(pr, kap)
        d["kappa_diff_ci95_question_boot"] = boot_ci_diff(pr, kap)
        d["disagree_pct_diff_perm"] = perm_lang_diff(pr, dis)
        d["disagree_pct_diff_ci95_question_boot"] = boot_ci_diff(pr, dis)
        JJ["%s|%s" % (scope, qscope)] = d

# each judge vs the human on the 120 reviewed cells
HV = {"selected_questions_by_lang": {lg: sum(L.LANG[q] == lg for q in SEL) for lg in ("fa", "en")}}
for jn, jf in (("j1", j2lib.judge1), ("j2", lambda a, q: j2lib.judge2(a, q)[:2])):
    pr, unres = [], 0
    for a in j2lib.ARMS4:
        for q in SEL:
            m = jf(a, q)
            if m[0] != m[1]:
                unres += 1
                continue
            pr.append((m[0], HUM[(a, q)], q, a))
    d = {"unresolved": unres}
    for lg in ("fa", "en"):
        s = agree_stats([p for p in pr if L.LANG[p[2]] == lg])
        s["stricter_judge0_human1"] = s.pop("a0_b1"); s["laxer_judge1_human0"] = s.pop("a1_b0")
        s["judge_correct"] = s.pop("a_correct"); s["human_correct"] = s.pop("b_correct")
        s["by_arm_stricter"] = {a: sum(1 for p in pr if L.LANG[p[2]] == lg and p[3] == a and p[0] < p[1]) for a in j2lib.ARMS4}
        d[lg] = s
    stricter = lambda ps: (100.0 * sum(x < y for x, y, *_ in ps) / len(ps)) if ps else None
    d["stricter_pct_diff_perm"] = perm_lang_diff(pr, stricter)
    d["kappa_diff_perm"] = perm_lang_diff(pr, kap)
    # strictness relative to human positives: of the human-accepted cells, share the judge rejected
    for lg in ("fa", "en"):
        pos = [p for p in pr if L.LANG[p[2]] == lg and p[1] == 1]
        d[lg]["rejected_of_human_accepted"] = [sum(1 for p in pos if p[0] == 0), len(pos)]
    HV[jn] = d

# S7 judge-disagreement cause labels by language
E = json.loads((L.PH / "S7_analysis" / "errors.json").read_text(encoding="utf-8"))["a_cells"]
CAUSE = defaultdict(lambda: {"fa": 0, "en": 0, "fa_J1": 0, "fa_J2": 0, "en_J1": 0, "en_J2": 0})
for c in E:
    CAUSE[c["label"]][c["lang"]] += 1
    CAUSE[c["label"]]["%s_%s" % (c["lang"], c["stricter"])] += 1
tot = {lg: sum(v[lg] for v in CAUSE.values()) for lg in ("fa", "en")}
qn = {lg: len({c["q"] for c in E if c["lang"] == lg}) for lg in ("fa", "en")}
CAUSES = {"by_label": dict(CAUSE), "total_cells": tot, "questions_with_disagreement": qn, "n_cells": len(E)}
# without the single-question Persian synonym case
CAUSES["total_without_synonym_fa"] = {lg: tot[lg] - CAUSE["synonym_fa"][lg] for lg in ("fa", "en")}

RES = {"synonym_question": SYN_Q, "tokens": TOK, "judge1_vs_judge2": JJ, "judges_vs_human_120": HV, "disagreement_causes": CAUSES}
L.wjson(L.HERE / "tokjudge.json", RES)

# ------------------------------------------------------------------ markdown
md = ["# S14 task 2: tokenizers and judges by language (generated by t2_tok_judge.py -> tokjudge.json)", "",
      "## (a) Reader tokens per evidence character, Persian vs English (8 frozen arms x 96 questions, byte-identical packs)", "",
      "| reader | calls fa / en | fa tok/ev-char | en tok/ev-char | **fa/en marginal** | 95% CI (question bootstrap) | fa/en whole prompt (pooled) | pack chars = ledger chars |",
      "|---|---|---|---|---|---|---|---|"]
for rd, t in TOK.items():
    md.append("| %s | %d / %d | %.4f | %.4f | **%.3f** | [%.3f, %.3f] | %.3f | %d/%d |" % (
        rd, t["n_calls_by_lang"]["fa"], t["n_calls_by_lang"]["en"], t["fa"]["tokens_per_evidence_char"],
        t["en"]["tokens_per_evidence_char"], t["ratio_marginal_evidence"], *t["ratio_marginal_ci95"],
        t["ratio_pooled_whole_prompt"], t["ledger_prompt_chars_match_pack"], t["ledger_prompt_chars_known"]))
rs = [t["ratio_marginal_evidence"] for t in TOK.values()]
md += ["", "Range of the marginal ratio over the six readers: %.3f to %.3f." % (min(rs), max(rs)), "",
       "## (b1) Judge 1 vs judge 2, GLM cells resolved under both, by language", "",
       "| scope | lang | questions | comparable | agree % | kappa | J1=1,J2=0 | J1=0,J2=1 | J1 correct | J2 correct |",
       "|---|---|---|---|---|---|---|---|---|---|"]
for k, d in JJ.items():
    for lg in ("fa", "en", "all"):
        s = d[lg]
        md.append("| %s | %s | %s | %d | %.1f | %.3f | %d | %d | %d | %d |" % (
            k, lg, s.get("questions", "-"), s["comparable"], s["agree_pct"], s["kappa"], s["a1_b0"], s["a0_b1"],
            s["a_correct"], s["b_correct"]))
    md.append("| %s | fa - en | | | %+.1f pp disagree (p %.3f; CI [%.1f, %.1f]) | %+.3f (p %.3f; CI [%.3f, %.3f]) | | | | |" % (
        k, d["disagree_pct_diff_perm"]["diff_fa_minus_en"], d["disagree_pct_diff_perm"]["p"],
        *d["disagree_pct_diff_ci95_question_boot"], d["kappa_diff_perm"]["diff_fa_minus_en"], d["kappa_diff_perm"]["p"],
        *d["kappa_diff_ci95_question_boot"]))
md += ["", "p = question-level permutation of the language label (10,000, seed 0); CI = stratified question bootstrap (2,000).", "",
       "## (b2) Each judge vs the human on the 120 reviewed cells (4 frozen 6,000 arms x 30 questions: fa %d, en %d)" % (
           HV["selected_questions_by_lang"]["fa"], HV["selected_questions_by_lang"]["en"]), "",
       "| judge | lang | comparable | agree | kappa | stricter (J0,H1) | laxer (J1,H0) | rejected of human-accepted | stricter by arm H/HR/LEX/SETR |",
       "|---|---|---|---|---|---|---|---|---|"]
for jn in ("j1", "j2"):
    d = HV[jn]
    for lg in ("fa", "en"):
        s = d[lg]
        md.append("| %s | %s | %d | %d (%.1f%%) | %s | %d | %d | %d/%d | %s |" % (
            jn, lg, s["comparable"], s["agree"], s["agree_pct"], "%.3f" % s["kappa"] if s["kappa"] is not None else "-",
            s["stricter_judge0_human1"], s["laxer_judge1_human0"], *s["rejected_of_human_accepted"],
            "/".join(str(s["by_arm_stricter"][a]) for a in j2lib.ARMS4)))
    md.append("| %s | fa - en | | | %+.3f (p %.3f) | %+.1f pp (p %.3f) | | | |" % (
        jn, d["kappa_diff_perm"]["diff_fa_minus_en"], d["kappa_diff_perm"]["p"],
        d["stricter_pct_diff_perm"]["diff_fa_minus_en"], d["stricter_pct_diff_perm"]["p"]))
md += ["", "## (b3) Causes of the 66 judge-1/judge-2 disagreements (S7 labels, GLM, 8 cells) by language", "",
       "| cause | fa cells (J1 / J2 stricter) | en cells (J1 / J2 stricter) |", "|---|---|---|"]
for lab, v in sorted(CAUSE.items(), key=lambda kv: -(kv[1]["fa"] + kv[1]["en"])):
    md.append("| %s | %d (%d / %d) | %d (%d / %d) |" % (lab, v["fa"], v["fa_J1"], v["fa_J2"], v["en"], v["en_J1"], v["en_J2"]))
md.append("| total | %d (on %d questions) | %d (on %d questions) |" % (tot["fa"], qn["fa"], tot["en"], qn["en"]))
md.append("| total without synonym_fa | %d | %d |" % (CAUSES["total_without_synonym_fa"]["fa"], CAUSES["total_without_synonym_fa"]["en"]))
(L.HERE / "tokjudge_tables.md").write_text("\n".join(md) + "\n", encoding="utf-8")
print("\n".join(md))
