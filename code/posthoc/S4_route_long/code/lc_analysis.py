"""Experiment 2 analysis: LC arms vs the frozen arms (saved verdicts), no model calls.
Writes S4/lc_results.json and S4/lc_tables.md."""
import json
import statistics as st
import common as C
import stats_common as S

JM = S.RT.JM
ARMS = ("LC-GOLD-DOCS", "LC-BM25-DOCS")
VER = S.RT.final_verdicts_by_answer(C.S4 / "lc_judge" / "test")
REC = {}
for a in ARMS:
    REC[a] = {}
    for r in C.jl(C.S4 / "lc_runs" / "test" / a / ("runs_%s.jsonl" % a)):
        old = REC[a].get(r["question_id"])
        if old is None or old.get("error") or not r.get("error"):
            REC[a][r["question_id"]] = r
PK = {(r["arm"], r["question_id"]): r for r in C.jl(C.S4 / "lc_packs" / "test" / "index.jsonl")}
TOK = json.loads((C.S4 / "frozen_cell_tokens.json").read_text(encoding="utf-8"))
FROZEN = ["HYBRID_6000", "HR_6000", "LEX5B_6000", "SETR_6000", "HYBRID_15500", "HR_15500", "LEX5B_15500",
          "SETR_15500"]


def score(a, q):
    r = REC[a].get(q)
    if r is None:
        return 0, 1, "missing"            # not run (e.g. budget stop): reported as missing, bounds [0, 1]
    if r.get("error"):
        return 0, 0, "system_failure"
    v = VER.get((r["meta"]["identity"]["rid"], JM.answer_sha1(r)))
    if v is None:
        return 0, 1, "unjudged"
    if v["status"] == "needs_adjudication":
        return 0, 1, "needs_adjudication"
    return v["strict"], v["strict"], v["status"]


def f_lc(a):
    return lambda q: score(a, q)[:2]


def f_fr(a):
    return lambda q: S.mach(a, q)


def main():
    res = {"arms": {}, "comparisons": {}, "frozen": {}}
    for a in ARMS:
        qs_run = [q for q in S.QIDS if q in REC[a]]
        status = {}
        for q in S.ANS:
            s = score(a, q)[2]
            status[s] = status.get(s, 0) + 1
        recs = [REC[a][q] for q in qs_run]
        ptok = [r["prompt_tokens"] for r in recs]
        ctok = [r["completion_tokens"] for r in recs]
        lat = [r["reader_ms"] for r in recs if not r.get("error")]
        pk = [PK[(a, q)] for q in qs_run]
        by_lang_tok = {l: round(st.mean([r["prompt_tokens"] for r in recs if r["q_lang"] == l]))
                       for l in ("fa", "en") if any(r["q_lang"] == l for r in recs)}
        cost = sum((r["trace"]["lc"].get("cost_usd") or 0) for r in recs)
        tpc = {l: round(st.median([r["prompt_tokens"] / PK[(a, r["question_id"])]["prompt_chars"]
                                   for r in recs if r["q_lang"] == l and r["prompt_tokens"]]), 4)
               for l in ("fa", "en") if any(r["q_lang"] == l for r in recs)}
        ans_pk = [p for p in pk if p["answerable"]]
        res["arms"][a] = {
            "n_run": len(qs_run), "status_answerable": status, "acc": S.by_group(f_lc(a)),
            "abstain_correct": (S.acc_bounds(f_lc(a), S.AB) + [len(S.AB)]) if a == "LC-BM25-DOCS" else None,
            "prompt_tokens_per_q_mean": round(st.mean(ptok)), "prompt_tokens_per_q_median": st.median(ptok),
            "prompt_tokens_max": max(ptok), "prompt_tokens_by_lang_mean": by_lang_tok,
            "prompt_tokens_per_char_median": tpc,
            "fa_over_en_tokens_per_char": round(tpc["fa"] / tpc["en"], 3) if len(tpc) == 2 else None,
            "completion_tokens_per_q_mean": round(st.mean(ctok)),
            "tokens_per_q_mean": round(st.mean([p + c for p, c in zip(ptok, ctok)])),
            "calls_per_q": 1.0, "reader_cost_usd_total": round(cost, 5),
            "reader_latency_s": {"median": round(st.median(lat) / 1000, 2), "mean": round(st.mean(lat) / 1000, 2),
                                 "p90": round(sorted(lat)[int(0.9 * len(lat))] / 1000, 2), "max": round(max(lat) / 1000, 2)},
            "done_reason": {k: sum(1 for r in recs if r["trace"]["lc"].get("done_reason") == k)
                            for k in {r["trace"]["lc"].get("done_reason") for r in recs}},
            "abstained_answerable": sum(1 for q in S.ANS if q in REC[a] and REC[a][q].get("abstained")),
            "evidence_chars_mean": round(st.mean([p["evidence_chars"] for p in pk])),
            "gold_units_retained": [sum(p["retained"] for p in ans_pk), sum(p["gold_units"] for p in ans_pk)],
            "questions_all_gold_units_retained": sum(p["retained"] == p["gold_units"] for p in ans_pk),
            "questions_any_gold_doc_shown": sum(p["gold_docs_shown"] > 0 for p in ans_pk),
            "packs_all_docs_whole": sum(p["whole_docs"] == p["docs"] for p in pk),
            "docs_per_pack_mean": round(st.mean([p["docs"] for p in pk]), 2)}
        # accuracy split by whether every gold unit is in the prompt (LC-BM25-DOCS: retrieval quality at doc level)
        if a == "LC-BM25-DOCS":
            full = [q for q in S.ANS if PK[(a, q)]["retained"] == PK[(a, q)]["gold_units"]]
            part = [q for q in S.ANS if q not in full]
            res["arms"][a]["acc_when_all_gold_retained"] = S.acc_bounds(f_lc(a), full) + [len(full)]
            res["arms"][a]["acc_when_not_all_gold_retained"] = S.acc_bounds(f_lc(a), part) + [len(part)]
        if a == "LC-GOLD-DOCS":
            whole = [q for q in S.ANS if PK[(a, q)]["whole_docs"] == PK[(a, q)]["docs"]]
            res["arms"][a]["acc_when_gold_docs_whole"] = S.acc_bounds(f_lc(a), whole) + [len(whole)]
    for b in FROZEN:
        res["frozen"][b] = {"acc": S.by_group(f_fr(b)),
                            "tokens_per_q": [round(sum(TOK[b][q]["tok_lo"] for q in S.QIDS) / 96),
                                             round(sum(TOK[b][q]["tok_hi"] for q in S.QIDS) / 96)]}
    for a in ARMS:
        for b in FROZEN:
            res["comparisons"]["%s vs %s" % (a, b)] = S.compare(f_lc(a), f_fr(b))
    res["comparisons"]["LC-GOLD-DOCS vs LC-BM25-DOCS"] = S.compare(f_lc("LC-GOLD-DOCS"), f_lc("LC-BM25-DOCS"))
    json.dump(res, open(C.S4 / "lc_results.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)

    L = ["# Long-context tables (generated by code/lc_analysis.py; S4 runs/verdicts + saved frozen verdicts)", "",
         "Accuracy = strict machine verdicts (pinned judge) on the 90 answerable test questions, bounds a..b over "
         "needs_adjudication / missing cells. LC prompt limit L_TOK = 32,000 tokens per call. LC-GOLD-DOCS is an "
         "ORACLE upper reference (gold documents chosen by the annotation).", "",
         "| Arm | All /90 | T1 /30 | T2 /30 | T3 /30 | fa /45 | en /45 | Abstain /6 | Query tokens/q | Calls/q |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for a in ARMS:
        x = res["arms"][a]
        g = x["acc"]
        L.append("| %s | %s | %s | %s | %s | %s | %s | %s | %d | 1 |" % (
            a, S.fmt_b(g["all"]), S.fmt_b(g["T1"]), S.fmt_b(g["T2"]), S.fmt_b(g["T3"]), S.fmt_b(g["fa"]),
            S.fmt_b(g["en"]), S.fmt_b(x["abstain_correct"]) if x["abstain_correct"] else "n/a",
            x["tokens_per_q_mean"]))
    for b in FROZEN:
        x = res["frozen"][b]
        g = x["acc"]
        tk = x["tokens_per_q"]
        L.append("| %s (frozen) | %s | %s | %s | %s | %s | %s | 6 | %s | %s |" % (
            b, S.fmt_b(g["all"]), S.fmt_b(g["T1"]), S.fmt_b(g["T2"]), S.fmt_b(g["T3"]), S.fmt_b(g["fa"]),
            S.fmt_b(g["en"]), tk[0] if tk[0] == tk[1] else "%d-%d" % tuple(tk),
            "2" if b.startswith(("LEX5B", "SETR")) else "1"))
    L += ["", "## LC arm details", ""]
    for a in ARMS:
        L.append("- **%s**: %s" % (a, json.dumps({k: v for k, v in res["arms"][a].items() if k != "acc"},
                                                 ensure_ascii=False)))
    L += ["", "## Paired comparisons (90 answerable)", "",
          "| Comparison | Net | W/L (resolved) | Sign p | CI pp (questions) | CI pp (components) |",
          "|---|---|---|---|---|---|"]
    for k, c in res["comparisons"].items():
        sp = c["sign_p_range"]
        L.append("| %s | %s | %d/%d | %s | [%.1f, %.1f] | [%.1f, %.1f] |" % (
            k, S.fmt_b(c["net_bounds"]), c["wins_losses_resolved"][0], c["wins_losses_resolved"][1],
            "%.3f" % sp[0] if sp[0] == sp[1] else "%.3f-%.3f" % tuple(sp), c["ci_pp_question"][0],
            c["ci_pp_question"][1], c["ci_pp_component"][0], c["ci_pp_component"][1]))
    (C.S4 / "lc_tables.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
