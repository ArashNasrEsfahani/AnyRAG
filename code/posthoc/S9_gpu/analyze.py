"""S9 analysis: HYBRID_AGENT (this stream) and ITER_BM25 (S3, read from its folder) vs the frozen GLM arms on the 96
test questions. No model calls. Same procedure as S3_agentic/analyze.py (which re-implements revision_stats.py:
diff bounds over unresolved cells, exact sign-test range over every assignment, bootstrap over questions and over the
connected components of source families, 10,000 resamples, seed 0), plus: the contrast without the 26
closed-book-correct questions (S7_analysis/contam.json) and GPU time per question.

    python analyze.py      -> results.json, results_tables.md
"""
import itertools
import json
import os
import random
import statistics
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
WORK = HERE.parent.parent
S3 = HERE.parent / "S3_agentic"
os.environ["ANYRAG_TEST_DIR"] = str(WORK / "datasets_v2" / "test_run")
os.environ["ANYRAG_FREEZE_DIR"] = str(WORK / "datasets_v2" / "test_freeze")
os.environ["ANYRAG_CODE_DIR"] = r"D:\KareOne\RAG-Paper-upgrades"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.path.insert(0, r"D:\KareOne\RAG-Paper-upgrades")
sys.path.insert(0, str(WORK / "datasets_v2" / "test_run"))
import report_test as RT  # noqa: E402
from ragbench.tools import judge_multi as JM  # noqa: E402

N_BOOT, SEED = 10000, 0
P = RT.PRIMARY
QS, ANS, AB, QIDS = RT.QS, RT.ANS, RT.AB, RT.QIDS
FROZEN = ["HYBRID_6000", "HR_6000", "LEX5B_6000", "SETR_6000", "HYBRID_15500", "HR_15500", "LEX5B_15500", "SETR_15500"]
COMPARE_TO = ["ITER_BM25", "HR_6000", "HR_15500", "HYBRID_6000", "LEX5B_6000"]
NEW = {"HYBRID_AGENT": HERE, "ITER_BM25": S3}
REC, VER = RT.load()
CONTAM = json.loads((HERE.parent / "S7_analysis" / "contam.json").read_text(encoding="utf-8"))["closedbook_correct"][
    "questions"]
assert len(CONTAM) == 26

MYREC, MYVER, LEDGER = {}, {}, {}
for a, d in NEW.items():
    MYREC[a] = {}
    f = d / "runs" / a / "test" / ("runs_%s.jsonl" % a)
    for line in (open(f, encoding="utf-8") if f.is_file() else []):
        if line.strip():
            r = json.loads(line)
            old = MYREC[a].get(r["question_id"])
            if old is None or old.get("error") or not r.get("error"):
                MYREC[a][r["question_id"]] = r
    MYVER[a] = RT.final_verdicts_by_answer(d / "judge" / a / "test")
    LEDGER[a] = [json.loads(l) for l in open(d / "ledger.jsonl", encoding="utf-8") if l.strip()]


def score(a, q):
    if a in FROZEN:
        lo, hi, _ = RT.score(REC, VER, P, a, q)
        return lo, hi
    r = MYREC[a].get(q)
    if r is None or r.get("error"):
        return 0, 0
    v = MYVER[a].get((r["meta"]["identity"]["rid"], JM.answer_sha1(r)))
    if v is None or v["status"] == "needs_adjudication":
        return 0, 1
    return v["strict"], v["strict"]


def status(a, q):
    if a in FROZEN:
        return RT.score(REC, VER, P, a, q)[2]
    r = MYREC[a].get(q)
    if r is None:
        return "missing"
    if r.get("error"):
        return "system_failure"
    v = MYVER[a].get((r["meta"]["identity"]["rid"], JM.answer_sha1(r)))
    return "unjudged" if v is None else v["status"]


def my_calls(a, q):
    return [c for c in LEDGER[a] if c.get("arm") == a and c.get("question_id") == q and c.get("run_tag") == "test"]


def tokens(a, q):
    if a in FROZEN:
        lo, hi, _ = RT.token_cell(REC, P, a, q)
        return lo, hi
    cs = my_calls(a, q)
    known = sum(c["prompt_tokens"] + c["completion_tokens"] for c in cs if c.get("ok") and c.get("usage_known"))
    unk = sum(c.get("max_tokens", 0) + int(c.get("prompt_chars", 0) * 1.0) for c in cs
              if not (c.get("ok") and c.get("usage_known")))
    return known, known + unk


def n_calls(a, q):
    if a in FROZEN:
        return (REC[P][a].get(q) or {}).get("llm_calls", 0)
    return len(my_calls(a, q))


def latency(a, q):
    r = (REC[P][a] if a in FROZEN else MYREC[a]).get(q)
    return None if (r is None or r.get("error")) else r.get("latency_ms")


_FL = {}


def frozen_embed_ms(a, q):
    """Wall ms of the frozen cell's Ollama query-embed call(s) for q (its run ledger; the last run's)."""
    key = (a, RT.pool_of(q))
    if key not in _FL:
        _FL[key] = RT.ledgers(RT.HERE / "runs" / RT.pool_of(q) / a)
    r = REC[P][a].get(q) or {}
    cs = [c for c in _FL[key] if c.get("kind") == "embed" and c.get("question_id") == q and c.get("phase") == "query"
          and c.get("run_id") == r.get("run_id")]
    return sum(c.get("wall_ms") or 0 for c in cs) if cs else None


def gpu_ms(a, q):
    """GPU-side time per question (ms): HYBRID_AGENT = sum over searches of Ollama embed wall + reranker ms;
    HR = query embed + rerank ms; HYBRID = query embed; LEX5B = HF bge-m3 query encode (t_encode_ms); ITER_BM25 = 0."""
    if a == "ITER_BM25":
        return 0.0
    if a == "HYBRID_AGENT":
        r = MYREC[a].get(q)
        if r is None or r.get("error"):
            return None
        return sum((g.get("embed_ms") or 0) + (g.get("rerank_ms") or 0) for g in (r["trace"].get("gpu") or []))
    r = REC[P][a].get(q)
    if r is None or r.get("error"):
        return None
    t = r.get("trace") or {}
    if a.startswith("LEX5B"):
        return (t.get("lex5b") or {}).get("t_encode_ms")
    e = frozen_embed_ms(a, q) or 0.0
    if a.startswith("HR"):
        return e + ((t.get("rerank") or {}).get("ms") or 0.0)
    if a.startswith("HYBRID") or a.startswith("SETR"):
        return e
    return None


# ---------------------------------------------------------------- components (as revision_stats.py)
QROWS = {}
for line in open(WORK / "datasets_v2" / "test_freeze" / "questions" / "test_v2judge_all.jsonl", encoding="utf-8-sig"):
    if line.strip():
        d = json.loads(line)
        QROWS[d["id"]] = d["meta"]["v2"]
constituents = {q: set(QROWS[q]["source_family"].split("+")) for q in ANS}
parent = {}


def find(x):
    parent.setdefault(x, x)
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x


for q in ANS:
    c = sorted(constituents[q])
    for o in c[1:]:
        parent[find(o)] = find(c[0])
comp = {q: find(sorted(constituents[q])[0]) for q in ANS}


def boot_mean(diffs, clusters=None):
    rnd = random.Random(SEED)
    n = len(diffs)
    sims = []
    if clusters is None:
        for _ in range(N_BOOT):
            s = [diffs[rnd.randrange(n)] for _ in range(n)]
            sims.append(100.0 * sum(s) / n)
    else:
        g = defaultdict(list)
        for d, c in zip(diffs, clusters):
            g[c].append(d)
        keys = sorted(g)
        for _ in range(N_BOOT):
            tot = cnt = 0
            for _k in keys:
                grp = g[keys[rnd.randrange(len(keys))]]
                tot += sum(grp)
                cnt += len(grp)
            sims.append(100.0 * tot / cnt)
    sims.sort()
    return sims[int(0.025 * N_BOOT)], sims[int(0.975 * N_BOOT) - 1]


def compare(a, b, qs=ANS, boot=True):
    fa, fb = (lambda q: score(a, q)), (lambda q: score(b, q))
    lo, up, cells = [], [], []
    for q in qs:
        al, ah = fa(q)
        bl, bh = fb(q)
        lo.append(al - bh)
        up.append(ah - bl)
        if al != ah:
            cells.append((q, "a"))
        if bl != bh:
            cells.append((q, "b"))
    ps, nets, bws = [], [], []
    for assign in itertools.product((0, 1), repeat=len(cells)):
        val = dict(zip(cells, assign))
        bb = ww = net = 0
        for q in qs:
            av = val.get((q, "a"), fa(q)[0])
            bv = val.get((q, "b"), fb(q)[0])
            net += av - bv
            bb += av > bv
            ww += av < bv
        ps.append(RT.sign_p(bb, ww))
        nets.append(net)
        bws.append((bb, ww))
    out = {"n": len(qs), "net": [min(nets), max(nets)], "wins_losses": [list(x) for x in sorted(set(bws))],
           "sign_p": [min(ps), max(ps)], "unresolved_cells": len(cells)}
    if boot:
        for name, cl in (("question", None), ("component", [comp[q] for q in qs])):
            a1, a2 = boot_mean(lo, cl), boot_mean(up, cl)
            out["ci_pp_" + name] = [round(min(a1[0], a2[0]), 1), round(max(a1[1], a2[1]), 1)]
    return out


# ---------------------------------------------------------------- tables
ALL = list(NEW) + FROZEN
NOC = [q for q in ANS if q not in set(CONTAM)]
res = {"n_answerable": len(ANS), "n_unanswerable": len(AB), "components": len(set(comp.values())),
       "n_answerable_without_closedbook": len(NOC), "arms": {}}
groups = {"all": ANS}
for t in ("T1", "T2", "T3"):
    groups[t] = [q for q in ANS if QS[q]["tier"] == t]
for lg in ("fa", "en"):
    groups[lg] = [q for q in ANS if QS[q]["lang"] == lg]
groups["no_closedbook"] = NOC
hyb_tok = [sum(tokens("HYBRID_6000", q)[i] for q in QIDS) / len(QIDS) for i in (0, 1)]
for a in ALL:
    row = {}
    for g, qs in groups.items():
        row[g] = [sum(score(a, q)[0] for q in qs), sum(score(a, q)[1] for q in qs), len(qs)]
    row["abstention_correct"] = [sum(score(a, q)[0] for q in AB), sum(score(a, q)[1] for q in AB), len(AB)]
    tk = [tokens(a, q) for q in QIDS]
    row["tokens_per_q"] = [sum(t[0] for t in tk) / len(QIDS), sum(t[1] for t in tk) / len(QIDS)]
    row["tokens_ratio_vs_HYBRID_6000"] = [row["tokens_per_q"][0] / hyb_tok[1], row["tokens_per_q"][1] / hyb_tok[0]]
    row["calls_per_q"] = sum(n_calls(a, q) for q in QIDS) / len(QIDS)
    lat = [x for x in (latency(a, q) for q in QIDS) if x is not None]
    row["latency_s_median"] = statistics.median(lat) / 1000.0 if lat else None
    row["latency_s_mean"] = statistics.mean(lat) / 1000.0 if lat else None
    gp = [x for x in (gpu_ms(a, q) for q in QIDS) if x is not None]
    row["gpu_ms_per_q_mean"] = statistics.mean(gp) if gp else None
    row["gpu_ms_per_q_median"] = statistics.median(gp) if gp else None
    row["gpu_n"] = len(gp)
    row["statuses"] = dict(sorted({s: sum(1 for q in QIDS if status(a, q) == s)
                                   for s in {status(a, q) for q in QIDS}}.items()))
    if a in NEW:
        cs = [c for c in LEDGER[a] if c.get("arm") == a and c.get("run_tag") == "test"]
        row["cost_usd_test"] = sum(c.get("cost_usd") or 0 for c in cs)
        row["agent_calls_per_q"] = sum(1 for c in cs if c.get("purpose") == "agent") / len(QIDS)
        tc = defaultdict(int)
        rounds, src, evc, fails = [], defaultdict(int), [], 0
        for q in QIDS:
            r = MYREC[a].get(q)
            if r is None or r.get("error"):
                fails += 1
                continue
            for k, v in (r["trace"].get("tool_calls") or {}).items():
                tc[k] += v
            rounds.append(r["trace"].get("rounds"))
            src[r["trace"].get("evidence_source")] += 1
            evc.append(r.get("evidence_chars") or 0)
        row["tool_calls_per_q"] = {k: v / len(QIDS) for k, v in tc.items()}
        row["rounds_mean"] = statistics.mean(rounds) if rounds else None
        row["rounds_hist"] = {str(k): rounds.count(k) for k in sorted(set(rounds))}
        row["evidence_source"] = dict(src)
        row["evidence_chars_median"] = statistics.median(evc) if evc else None
        row["evidence_chars_mean"] = statistics.mean(evc) if evc else None
        row["system_failures"] = fails
        row["answer_truncated"] = sum(1 for q in QIDS if (MYREC[a].get(q) or {}).get("answer_done_reason") == "length")
        if a == "HYBRID_AGENT":
            g = [x for q in QIDS for x in ((MYREC[a].get(q) or {}).get("trace", {}).get("gpu") or [])]
            row["searches"] = len(g)
            row["embed_ms_per_search_mean"] = statistics.mean(x["embed_ms"] for x in g)
            row["rerank_ms_per_search_mean"] = statistics.mean(x["rerank_ms"] for x in g)
            row["rerank_devices"] = sorted({"%s/fp16=%s" % (x["device"], x["fp16"]) for x in g})
            row["rerank_fallbacks"] = sum(1 for x in g if x.get("fallback"))
    res["arms"][a] = row
res["paired"] = {}
for b in COMPARE_TO:
    key = "HYBRID_AGENT vs %s" % b
    res["paired"][key] = {"all": compare("HYBRID_AGENT", b)}
    for g in ("T1", "T2", "T3", "fa", "en"):
        c = compare("HYBRID_AGENT", b, groups[g], boot=False)
        res["paired"][key][g] = {k: c[k] for k in ("net", "wins_losses", "sign_p")}
    res["paired"][key]["no_closedbook"] = compare("HYBRID_AGENT", b, NOC)
(HERE / "results.json").write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")


# ---------------------------------------------------------------- markdown
def b(x):
    return "%d" % x[0] if x[0] == x[1] else "%d..%d" % (x[0], x[1])


def nb(x):
    return "%+d" % x[0] if x[0] == x[1] else "%+d..%+d" % tuple(x)


def pb(x):
    return "%.3f" % x[0] if x[0] == x[1] else "%.3f-%.3f" % tuple(x)


L = ["## Accuracy (machine, strict, GLM-5.3-flash; bounds over unresolved judge cells)", "",
     "| arm | all /90 | T1 /%d | T2 /%d | T3 /%d | fa /%d | en /%d | without closed-book-correct /%d | abstention /6 |" % tuple(
         len(groups[g]) for g in ("T1", "T2", "T3", "fa", "en", "no_closedbook")), "|---|---|---|---|---|---|---|---|---|"]
for a in ALL:
    r = res["arms"][a]
    L.append("| %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
        a, b(r["all"]), b(r["T1"]), b(r["T2"]), b(r["T3"]), b(r["fa"]), b(r["en"]), b(r["no_closedbook"]),
        b(r["abstention_correct"])))
L += ["", "## Cost and time (all 96 questions; latency and GPU time descriptive)", "",
      "| arm | query LLM tokens / q | x HYBRID_6000 | LLM calls / q | GPU ms / q mean (median) | latency s median (mean) |",
      "|---|---|---|---|---|---|"]
for a in ALL:
    r = res["arms"][a]
    tq = r["tokens_per_q"]
    tr = r["tokens_ratio_vs_HYBRID_6000"]
    L.append("| %s | %s | %s | %.2f | %s | %s |" % (
        a, "%.0f" % tq[0] if abs(tq[1] - tq[0]) < 0.5 else "%.0f..%.0f" % tuple(tq),
        "%.2f" % tr[0] if abs(tr[1] - tr[0]) < 0.005 else "%.2f..%.2f" % tuple(tr), r["calls_per_q"],
        "-" if r["gpu_ms_per_q_mean"] is None else "%.0f (%.0f)" % (r["gpu_ms_per_q_mean"], r["gpu_ms_per_q_median"]),
        "%.1f (%.1f)" % (r["latency_s_median"], r["latency_s_mean"]) if r["latency_s_median"] else "-"))
L += ["", "## Agent behaviour (test)", ""]
for a in NEW:
    r = res["arms"][a]
    L.append("- **%s**: tool calls/q %s; rounds mean %.2f, histogram %s; evidence source %s; evidence chars median %s "
             "mean %.0f; agent calls/q %.2f; system failures %d; truncated answers %d; test cost $%.4f" % (
                 a, {k: round(v, 2) for k, v in r["tool_calls_per_q"].items()}, r["rounds_mean"] or 0,
                 r["rounds_hist"], r["evidence_source"], r["evidence_chars_median"], r["evidence_chars_mean"] or 0,
                 r["agent_calls_per_q"], r["system_failures"], r["answer_truncated"], r["cost_usd_test"]))
r = res["arms"]["HYBRID_AGENT"]
L.append("- **HYBRID_AGENT GPU**: %d searches; per search mean Ollama embed %.0f ms, reranker %.0f ms (50 pairs); "
         "devices %s; CPU fallbacks %d" % (r["searches"], r["embed_ms_per_search_mean"], r["rerank_ms_per_search_mean"],
                                          r["rerank_devices"], r["rerank_fallbacks"]))
L += ["", "## Paired: HYBRID_AGENT minus comparison arm, 90 answerable questions", "",
      "| comparison | net | wins/losses | sign p | 95% CI pp (questions) | 95% CI pp (components) | T1 net | T2 net | T3 net | fa net | en net |",
      "|---|---|---|---|---|---|---|---|---|---|---|"]
for k, v in res["paired"].items():
    al = v["all"]
    L.append("| %s | %s | %s | %s | [%.1f, %.1f] | [%.1f, %.1f] | %s |" % (
        k, nb(al["net"]), "; ".join("%d/%d" % tuple(x) for x in al["wins_losses"]), pb(al["sign_p"]),
        *al["ci_pp_question"], *al["ci_pp_component"], " | ".join(nb(v[g]["net"]) for g in ("T1", "T2", "T3", "fa", "en"))))
L += ["", "## Paired, without the 26 closed-book-correct questions (%d answerable left)" % len(NOC), "",
      "| comparison | net | wins/losses | sign p | 95% CI pp (questions) | 95% CI pp (components) |", "|---|---|---|---|---|---|"]
for k, v in res["paired"].items():
    al = v["no_closedbook"]
    L.append("| %s | %s | %s | %s | [%.1f, %.1f] | [%.1f, %.1f] |" % (
        k, nb(al["net"]), "; ".join("%d/%d" % tuple(x) for x in al["wins_losses"]), pb(al["sign_p"]),
        *al["ci_pp_question"], *al["ci_pp_component"]))
(HERE / "results_tables.md").write_text("\n".join(L) + "\n", encoding="utf-8")
print("\n".join(L))
