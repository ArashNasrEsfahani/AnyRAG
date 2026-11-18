"""S10 report (no model call): accuracy bounds, paired comparisons, breakdowns, gold-bundle coverage, tokens.
Frozen arms: s7lib (report_test.load/score). S10 cells: our judge shards via report_test.final_verdicts_by_answer.
Statistics: revision_stats helpers through s7lib (extracted with ast, unchanged). -> results.json, tables.md"""
import json
import sys
from collections import Counter, defaultdict

import numpy as np

sys.path.insert(0, r"D:\KareOne\RAG Paper\work\posthoc_2026-11\S7_analysis")
import s7lib as S7  # noqa: E402  (read-only: loaders, gold_reach, compare)
import s10lib as L  # noqa: E402
import judge as JG  # noqa: E402
from ragbench.tools import judge_multi as JM  # noqa: E402

RT = S7.RT
ANS, AB, QS = S7.ANS, S7.AB, S7.QS
NEW = L.ARMS
FROZ = ["HYBRID_6000", "HR_6000", "HYBRID_15500", "HR_15500", "LEX5B_6000", "LEX5B_15500"]

ANSW = {}
for l in open(L.HERE / "answers.jsonl", encoding="utf-8"):
    if l.strip():
        r = json.loads(l)
        if not r.get("error"):
            ANSW[(r["arm"], r["qid"])] = r
NV = {}
for d in sorted(p for p in (L.HERE / "judge").iterdir() if p.is_dir()):
    for k, v in RT.final_verdicts_by_answer(d).items():
        NV.setdefault(k, v)
IDX = {(r["arm"], r["qid"]): r for r in (json.loads(l) for l in open(L.HERE / "packs_index.jsonl", encoding="utf-8"))}


def status(arm, q):
    r = ANSW.get((arm, q))
    if r is None:
        return "not_run"
    if r["source"].startswith("frozen:"):
        return "frozen"
    v = NV.get((JG.identity(arm, q, r["setting"])["rid"], JM.answer_sha1({"answer_text": r["answer_text"]})))
    return "unjudged" if v is None else v["status"]


def score(arm, q):
    if arm not in NEW:
        return S7.score(arm, q)
    r = ANSW.get((arm, q))
    if r is None:
        return 0, 1
    if r["source"].startswith("frozen:"):
        lo, hi, _ = RT.score(S7.REC, S7.VER, S7.P, r["source"].split(":", 1)[1], q)
        return lo, hi
    v = NV.get((JG.identity(arm, q, r["setting"])["rid"], JM.answer_sha1({"answer_text": r["answer_text"]})))
    if v is None or v["status"] == "needs_adjudication":
        return 0, 1
    return v["strict"], v["strict"]


def F(arm):
    return lambda q: score(arm, q)


def run_qs(arm, qs):
    return [q for q in qs if arm not in NEW or (arm, q) in ANSW]


def tot(arm, qs):
    return sum(score(arm, q)[0] for q in qs), sum(score(arm, q)[1] for q in qs)


def genre(q):
    return S7.pool(q).split("-")[0]


def subtype(q):
    return S7.QV2[q].get("subtype") or "none"


GROUPS = {"tier": S7.tier, "lang": S7.lang, "genre": genre, "pool": S7.pool, "subtype": subtype}


# ------------------------------------------------------------------ tokens
def my_tokens():
    out = {}
    for l in open(L.HERE / "ledger.jsonl", encoding="utf-8"):
        if l.strip():
            c = json.loads(l)
            if c.get("purpose") == "answer" and c.get("ok") and str(c.get("approach", "")).startswith("S10:"):
                out[(c["approach"][4:], c["question_id"])] = c
    return out


def frozen_tokens(arm):
    out = {}
    for f in (L.TR / "runs").glob("*/%s/ledger_*.jsonl" % arm):
        for l in open(f, encoding="utf-8"):
            if l.strip():
                c = json.loads(l)
                if c.get("kind") == "chat" and c.get("phase") == "query" and c.get("ok") and c.get("purpose") == "answer":
                    out[c["question_id"]] = c
    return out


# ------------------------------------------------------------------ gold coverage
def msgs(arm, q):
    if arm in NEW:
        p = L.HERE / "packs" / arm / (q + ".json")
        return json.loads(p.read_text(encoding="utf-8"))["messages"]
    return S7.messages(arm, q)


def span_cov(set_name, q, chunk_ids):
    """S7 t5_channels.gold_cov: fraction of each gold span covered by these chunks' spans (same doc)."""
    hx = L.hx(set_name)
    out = {}
    for e in S7.QV2[q]["evidence"]:
        a, b = e["start"], e["end"]
        cov = np.zeros(max(1, b - a), bool)
        for c in chunk_ids:
            ch = hx.chunks[c]
            if ch["doc_id"] != e["doc_id"] or not ch.get("span"):
                continue
            s0, s1 = ch["span"]
            lo, hi = max(a, s0), min(b, s1)
            if hi > lo:
                cov[lo - a:hi - a] = True
        out[e["id"]] = float(cov.mean())
    return out


def bundle_cls(q, cov, thr=0.9):
    fr = [sum(cov[x] >= thr for x in bun) / len(bun) for bun in S7.QV2[q]["evidence_bundles"]]
    best = max(fr)
    return "complete" if best == 1.0 else ("partial" if best > 0 else "none")


def list40(arm, q):
    if arm in NEW:
        return IDX[(arm, q)]["list40"]
    return [u["meta"]["chunk"] for u in S7.REC[S7.P][arm][q]["retrieved_units"]][:40]


def main():
    res = {"family": [], "table": {}, "groups": {}, "compare": {}, "reach": {}, "list_reach": {}, "wins": {}}
    ARMS = NEW + FROZ
    mt = my_tokens()
    ft = {a: frozen_tokens(a) for a in FROZ}
    for a in ARMS:
        qs_run = run_qs(a, ANS)
        st = Counter(status(a, q) for q in ANS + AB) if a in NEW else None
        if a in NEW:
            tk = [mt.get((ANSW[(a, q)]["source"][4:] if ANSW[(a, q)]["source"].startswith("dup:") else a, q))
                  for q in qs_run if ANSW[(a, q)]["source"] == "call"]
            ch = [IDX[(a, q)]["chars"] for q in ANS]
        else:
            tk = [ft[a].get(q) for q in ANS]
            ch = [S7.REC[S7.P][a][q]["evidence_chars"] for q in ANS]
        tk = [t for t in tk if t]
        ab_run = run_qs(a, AB)
        res["table"][a] = {
            "acc90": tot(a, ANS), "answered_of_90": len(qs_run), "acc_on_answered": tot(a, qs_run),
            "unresolved": sum(score(a, q)[0] != score(a, q)[1] for q in qs_run),
            "abstain_correct": tot(a, ab_run) if ab_run else None, "abstain_run": len(ab_run),
            "status": dict(st) if st else None, "evidence_chars_mean": round(float(np.mean(ch)), 1),
            "answer_prompt_tokens_mean": round(float(np.mean([t["prompt_tokens"] for t in tk])), 1) if tk else None,
            "answer_completion_tokens_mean": round(float(np.mean([t["completion_tokens"] for t in tk])), 1) if tk else None,
            "tokens_n": len(tk), "llm_calls_per_question": 1}
        res["groups"][a] = {}
        for g, f in GROUPS.items():
            d = defaultdict(list)
            for q in ANS:
                d[f(q)].append(q)
            res["groups"][a][g] = {k: [*tot(a, v), len(v), len(run_qs(a, v))] for k, v in sorted(d.items())}
        # gold coverage of the delivered pack (S7 t1 method) and of the top-40 list (S7 t5 span method)
        rc, lc = {}, {}
        for q in ANS:
            rc[q] = S7.gold_reach(q, msgs(a, q))["cls"]
            lc[q] = bundle_cls(q, span_cov(L.query_for(q).corpus_set, q, list40(a, q)))
        res["reach"][a] = {"all": dict(Counter(rc.values()))}
        res["list_reach"][a] = {"all": dict(Counter(lc.values()))}
        for g in ("tier", "lang", "genre"):
            for k in sorted({GROUPS[g](q) for q in ANS}):
                qq = [q for q in ANS if GROUPS[g](q) == k]
                res["reach"][a][k] = dict(Counter(rc[q] for q in qq))
                res["list_reach"][a][k] = dict(Counter(lc[q] for q in qq))
        res["reach"][a]["_per_q"] = rc
        res["list_reach"][a]["_per_q"] = lc
    # ---------------------------------------------------------- paired comparisons (one pre-declared family)
    pairs = []
    for b in (6000, 15500):
        D, B = "DENSE_%d" % b, "BM25_%d" % b
        pairs += [(D, B), (D, "HYBRID_%d" % b), (B, "HYBRID_%d" % b), (D, "HR_%d" % b), (B, "HR_%d" % b)]
    pairs += [("DENSE_15500", "DENSE_6000"), ("BM25_15500", "BM25_6000")]
    res["family_size"] = len(pairs)
    res["bonferroni_alpha"] = 0.05 / len(pairs)
    for a, b in pairs:
        qs = [q for q in ANS if q in set(run_qs(a, ANS)) and q in set(run_qs(b, ANS))]
        k = sum((score(a, q)[0] != score(a, q)[1]) + (score(b, q)[0] != score(b, q)[1]) for q in qs)
        key = "%s vs %s" % (a, b)
        if k > 16:
            res["compare"][key] = {"skipped": "unresolved %d > 16" % k}
            continue
        c = S7.compare(F(a), F(b), qs)
        c["n_questions"] = len(qs)
        c["acc_a"], c["acc_b"] = tot(a, qs), tot(b, qs)
        res["compare"][key] = c
    # ---------------------------------------------------------- where each wins (resolved, paired)
    for b in (6000, 15500):
        D, B, Hy = "DENSE_%d" % b, "BM25_%d" % b, "HYBRID_%d" % b
        qs = [q for q in ANS if (D, q) in ANSW and (B, q) in ANSW]
        rows = []
        for q in qs:
            d, m = score(D, q), score(B, q)
            if d[0] != d[1] or m[0] != m[1] or d[0] == m[0]:
                continue
            rows.append({"q": q, "winner": "DENSE" if d[0] > m[0] else "BM25", "tier": S7.tier(q), "lang": S7.lang(q),
                         "genre": genre(q), "subtype": subtype(q), "hybrid": score(Hy, q),
                         "reach_dense": res["reach"][D]["_per_q"][q], "reach_bm25": res["reach"][B]["_per_q"][q],
                         "gloss": (S7.QV2[q].get("question_gloss_en") or S7.QV2[q].get("question") or "")[:140]})
        both = {"both_right": sum(score(D, q) == (1, 1) and score(B, q) == (1, 1) for q in qs),
                "both_wrong": sum(score(D, q) == (0, 0) and score(B, q) == (0, 0) for q in qs)}
        agg = {}
        for g in ("tier", "lang", "genre", "subtype"):
            c = defaultdict(lambda: [0, 0])
            for r in rows:
                c[r[g]][0 if r["winner"] == "DENSE" else 1] += 1
            agg[g] = {k: v for k, v in sorted(c.items())}
        res["wins"][str(b)] = {"n": len(qs), "dense_only": sum(r["winner"] == "DENSE" for r in rows),
                               "bm25_only": sum(r["winner"] == "BM25" for r in rows), **both, "by": agg, "rows": rows}
        # oracle of the two single channels vs the hybrid
        res["wins"][str(b)]["oracle_either"] = sum(max(score(D, q)[0], score(B, q)[0]) for q in qs)
    (L.HERE / "results.json").write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: res[k] for k in ("table", "family_size")}, indent=1))
    for k, v in res["compare"].items():
        if "skipped" in v:
            print(k, v)
            continue
        print("%-28s n=%d %s vs %s net %s p %s ci_q %s ci_c %s bw %s" % (
            k, v["n_questions"], v["acc_a"], v["acc_b"], v["net_bounds"], [round(x, 4) for x in v["sign_p_range"]],
            [round(x, 1) for x in v["ci_pp_question"]], [round(x, 1) for x in v["ci_pp_component"]],
            v["better_worse_resolved"]))
    for b, w in res["wins"].items():
        print(b, {k: v for k, v in w.items() if k != "rows"})


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
