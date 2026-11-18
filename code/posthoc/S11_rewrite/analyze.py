"""S11 analysis (no model calls, no GPU): HYDE_6000 and MQ_6000 vs the frozen GLM arms on the test questions.
Loaders and statistics come from S7_analysis/s7lib.py (frozen verdicts via report_test, the revision_stats helpers run
unchanged: sign-test range over unresolved cells, bootstrap over questions and over the 20 components, 10,000 resamples,
seed 0; gold-bundle coverage = S7's gold_reach on the exact reader messages).
    python analyze.py -> results.json, tables.md"""
import json
import os
import statistics
import sys
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "S7_analysis"))
import s7lib as S7  # noqa: E402

RT, JM = S7.RT, S7.JM
NEW = ["HYDE_6000", "MQ_6000"]
FROZEN = ["HYBRID_6000", "HR_6000", "HYBRID_15500"]
ARMS = NEW + FROZEN
ANS, AB = S7.ANS, S7.AB
ALLQ = list(ANS) + list(AB)


def jl(p):
    return S7.jl(p)


REC, VER, PACK = {}, {}, {}
for a in NEW:
    REC[a] = {}
    for r in jl(HERE / "runs" / a / "test" / ("runs_%s.jsonl" % a)):
        old = REC[a].get(r["question_id"])
        if old is None or old.get("error") or not r.get("error"):
            REC[a][r["question_id"]] = r
    VER[a] = RT.final_verdicts_by_answer(HERE / "judge" / a / "test")
LEDGER = jl(HERE / "ledger.jsonl")
REWR = {(r["qid"], r["method"]): r for r in jl(HERE / "rewrites_test.jsonl") if r.get("ok")}


def status(a, q):
    if a in FROZEN:
        return RT.score(S7.REC, S7.VER, S7.P, a, q)[2]
    r = REC[a].get(q)
    if r is None:
        return "missing"
    if r.get("error"):
        return "system_failure"
    v = VER[a].get((r["meta"]["identity"]["rid"], JM.answer_sha1(r)))
    return "unjudged" if v is None else v["status"]


def score(a, q):
    if a in FROZEN:
        return S7.score(a, q)
    r = REC[a].get(q)
    if r is None or r.get("error"):
        return 0, 0
    v = VER[a].get((r["meta"]["identity"]["rid"], JM.answer_sha1(r)))
    if v is None or v["status"] == "needs_adjudication":
        return 0, 1
    return v["strict"], v["strict"]


def F(a):
    return lambda q: score(a, q)


def tot(a, qs):
    return sum(score(a, q)[0] for q in qs), sum(score(a, q)[1] for q in qs)


def fmt(t):
    return S7.fmt(*t)


def genre(q):
    return S7.pool(q).split("-")[0]


def calls(a, q):
    m = a.split("_")[0]
    return [c for c in LEDGER if c.get("question_id") == q and c.get("split") == "test" and c.get("arm") in (m, a)]


def tokens(a, q):
    if a in FROZEN:
        lo, hi, _ = RT.token_cell(S7.REC, S7.P, a, q)
        return lo, hi
    cs = calls(a, q)
    known = sum(c["prompt_tokens"] + c["completion_tokens"] for c in cs if c.get("ok") and c.get("usage_known"))
    unk = sum(c.get("max_tokens", 0) + c.get("prompt_chars", 0) for c in cs if not (c.get("ok") and c.get("usage_known")))
    return known, known + unk


def tok_parts(a, q):
    cs = calls(a, q)
    rw = [c for c in cs if str(c.get("purpose", "")).startswith("rewrite")]
    an = [c for c in cs if c.get("purpose") == "answer"]
    f = lambda L, k: sum(c.get(k) or 0 for c in L)  # noqa: E731
    return {"rewrite_prompt": f(rw, "prompt_tokens"), "rewrite_completion": f(rw, "completion_tokens"),
            "rewrite_reasoning": f(rw, "reasoning_tokens"), "answer_prompt": f(an, "prompt_tokens"),
            "answer_completion": f(an, "completion_tokens"), "n_calls": len(rw) + len(an),
            "cost": f(rw, "cost_usd") + f(an, "cost_usd")}


def messages(a, q):
    if a in FROZEN:
        return S7.messages(a, q)
    r = REC[a].get(q)
    return r and r.get("answer_messages")


def main():
    res = {"n_answerable": len(ANS), "n_unanswerable": len(AB)}
    # totals and status counts
    res["totals"] = {a: tot(a, ANS) for a in ARMS}
    res["status_counts"] = {a: {} for a in ARMS}
    for a in ARMS:
        for q in ALLQ:
            s = status(a, q)
            res["status_counts"][a][s] = res["status_counts"][a].get(s, 0) + 1
    # abstention on the 6 unanswerable: strict verdict 1 = correct abstention
    res["unanswerable"] = {a: tot(a, AB) for a in ARMS}
    res["abstained_unanswerable"] = {a: sum(bool((REC[a].get(q) or {}).get("abstained")) for q in AB) for a in NEW}
    # paired comparisons
    res["paired"] = {}
    for a in NEW:
        for b in FROZEN + [x for x in NEW if x != a]:
            res["paired"]["%s vs %s" % (a, b)] = S7.compare(F(a), F(b), ANS)
    # breakdowns
    groups = {}
    for q in ANS:
        groups.setdefault(("tier", S7.tier(q)), []).append(q)
        groups.setdefault(("lang", S7.lang(q)), []).append(q)
        groups.setdefault(("genre", genre(q)), []).append(q)
    res["breakdown"] = {}
    for (k, g), qs in sorted(groups.items()):
        row = {"n": len(qs)}
        for a in ARMS:
            row[a] = tot(a, qs)
        for a in NEW:
            c = S7.compare(F(a), F("HYBRID_6000"), qs)
            row["net_%s_vs_HYBRID_6000" % a] = c["net_bounds"]
            row["bw_%s_vs_HYBRID_6000" % a] = c["better_worse_resolved"]
        res["breakdown"]["%s=%s" % (k, g)] = row
    # tokens
    res["tokens"] = {}
    for a in ARMS:
        lo = [tokens(a, q)[0] for q in ALLQ]
        hi = [tokens(a, q)[1] for q in ALLQ]
        d = {"mean_lower": round(statistics.mean(lo), 1), "mean_upper": round(statistics.mean(hi), 1),
             "median_lower": statistics.median(lo), "n_questions": len(ALLQ)}
        if a in NEW:
            parts = [tok_parts(a, q) for q in ALLQ]
            for k in parts[0]:
                d["mean_" + k] = round(statistics.mean(p[k] for p in parts), 6 if k == "cost" else 1)
            d["total_cost_usd"] = round(sum(p["cost"] for p in parts), 5)
        else:
            d["llm_calls_mean"] = round(statistics.mean((S7.REC[S7.P][a].get(q) or {}).get("llm_calls", 0)
                                                       for q in ALLQ), 2)
        res["tokens"][a] = d
    # gold-bundle coverage (S7 method)
    res["coverage"] = {}
    per_q = {}
    for a in ARMS:
        cnt = {"complete": 0, "partial": 0, "none": 0}
        cnt_t = {}
        for q in ANS:
            m = messages(a, q)
            g = S7.gold_reach(q, m)
            cnt[g["cls"]] += 1
            key = "%s/%s" % (S7.tier(q), S7.lang(q))
            cnt_t.setdefault(S7.tier(q), {"complete": 0, "partial": 0, "none": 0})[g["cls"]] += 1
            cnt_t.setdefault(S7.lang(q), {"complete": 0, "partial": 0, "none": 0})[g["cls"]] += 1
            per_q.setdefault(q, {})[a] = g["cls"]
            _ = key
        res["coverage"][a] = {"all": cnt, "by_group": cnt_t}
    for a in NEW:
        gain = sum(per_q[q][a] == "complete" and per_q[q]["HYBRID_6000"] != "complete" for q in ANS)
        loss = sum(per_q[q][a] != "complete" and per_q[q]["HYBRID_6000"] == "complete" for q in ANS)
        res["coverage"][a]["vs_HYBRID_6000_complete_gained_lost"] = [gain, loss]
    # rewrite stats
    res["rewrites"] = {
        "hyde_chars_mean": round(statistics.mean(len(REWR[(q, "HYDE")]["rewrites"][0]) for q in ALLQ
                                                 if (q, "HYDE") in REWR), 1),
        "mq_n_rewrites": {str(n): sum(len(REWR[(q, "MQ")]["rewrites"]) == n for q in ALLQ if (q, "MQ") in REWR)
                          for n in (1, 2, 3)},
        "n_ok": {m: sum((q, m) in REWR for q in ALLQ) for m in ("HYDE", "MQ")}}
    # pack overlap with HYBRID_6000 (chunk sets in the pack)
    res["pack_identical_to_HYBRID_6000"] = {a: sum(S7.evidence_block(messages(a, q) or [{}, {"content": ""}])
                                                   == S7.evidence_block(S7.messages("HYBRID_6000", q))
                                                   for q in ALLQ) for a in NEW}
    res["repro"] = {k: sum(bool(r.get(k)) for r in jl(HERE / "repro_test.jsonl"))
                    for k in ("search_eq_channel_code", "saved_sha_equal", "bytes_equal")}
    res["repro"]["n"] = len(jl(HERE / "repro_test.jsonl"))
    res["gpu_phases"] = jl(HERE / "gpu_phases.jsonl")
    res["spent"] = {"glm_usd": round(sum(c.get("cost_usd") or 0 for c in LEDGER), 5)}
    sys.path.insert(0, str(HERE))
    import s11lib as L
    res["spent"]["judge_usd"] = round(L.judge_spent(), 5)
    res["spent"]["total_usd"] = round(res["spent"]["glm_usd"] + res["spent"]["judge_usd"], 5)
    S7.wjson(HERE / "results.json", res)
    tables(res)


def tables(res):
    o = ["# S11 tables (generated by analyze.py from results.json)", ""]
    o.append("## Accuracy, 90 answerable (strict, machine judge; a..b = bounds over unresolved cells)")
    o.append("")
    o.append("| arm | correct /90 | unanswerable correct /6 | complete gold bundle /90 | mean query tokens (lo..hi) |")
    o.append("|---|---|---|---|---|")
    for a in ARMS:
        t = res["tokens"][a]
        o.append("| %s | %s | %s | %d | %.0f..%.0f |" % (a, fmt(res["totals"][a]), fmt(res["unanswerable"][a]),
                                                        res["coverage"][a]["all"]["complete"], t["mean_lower"],
                                                        t["mean_upper"]))
    o += ["", "## Paired comparisons (90 answerable)", "",
          "| comparison | net | better/worse (resolved) | sign p | 95% CI pp (questions) | 95% CI pp (components) |",
          "|---|---|---|---|---|---|"]
    for k, c in res["paired"].items():
        o.append("| %s | %s | %d/%d | %s | %s | %s |" % (k, S7.fmt_net(c), *c["better_worse_resolved"], S7.fmt_p(c),
                                                       S7.fmt_ci(c), S7.fmt_ci(c, "ci_pp_component")))
    o += ["", "## Breakdowns (correct, bounds)", "",
          "| group | n | " + " | ".join(ARMS) + " | HYDE-HYBRID net | MQ-HYBRID net |",
          "|---" * (len(ARMS) + 4) + "|"]
    for g, r in res["breakdown"].items():
        nets = []
        for a in NEW:
            lo, hi = r["net_%s_vs_HYBRID_6000" % a]
            nets.append("%+d" % lo if lo == hi else "%+d..%+d" % (lo, hi))
        o.append("| %s | %d | %s | %s |" % (g, r["n"], " | ".join(fmt(r[a]) for a in ARMS), " | ".join(nets)))
    o += ["", "## Gold-bundle coverage (S7 method), complete / partial / none", "", "| arm | all | T1 | T2 | T3 | fa | en |",
          "|---|---|---|---|---|---|---|"]
    for a in ARMS:
        c = res["coverage"][a]
        f = lambda d: "%d/%d/%d" % (d["complete"], d["partial"], d["none"])  # noqa: E731
        o.append("| %s | %s | %s |" % (a, f(c["all"]), " | ".join(f(c["by_group"][g]) for g in
                                                                ("T1", "T2", "T3", "fa", "en"))))
    o += ["", "## Query-time tokens per question (96 questions, means)", "",
          "| arm | rewrite prompt | rewrite completion (of which reasoning) | answer prompt | answer completion | calls | cost/question $ |",
          "|---|---|---|---|---|---|---|"]
    for a in NEW:
        t = res["tokens"][a]
        o.append("| %s | %.1f | %.1f (%.1f) | %.1f | %.1f | %.2f | %.6f |" % (
            a, t["mean_rewrite_prompt"], t["mean_rewrite_completion"], t["mean_rewrite_reasoning"],
            t["mean_answer_prompt"], t["mean_answer_completion"], t["mean_n_calls"], t["mean_cost"]))
    (HERE / "tables.md").write_text("\n".join(o) + "\n", encoding="utf-8")
    print("\n".join(o))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
