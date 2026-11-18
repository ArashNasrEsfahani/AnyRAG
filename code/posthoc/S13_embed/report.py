"""S13 report (no model calls): accuracy bounds /90, abstention /6, paired comparisons (revision_stats helpers via
S7's s7lib), tier/language/genre, gold-bundle coverage of the reader pack (S7 t1_reach's gold_reach), recall@40 of
complete gold bundles (lists.jsonl), tokens, GPU wall time. -> results.json, tables.md"""
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "S7_analysis"))
import s7lib as L  # noqa: E402

RT = L.RT
sys.path.insert(0, str(HERE))
import llm as LL  # noqa: E402
from ragbench.tools import judge_multi as JM  # noqa: E402

NEW = ["E5_DENSE", "E5_HYBRID"]
S10 = L.PH / "S10_retrieval"

ANSW = {(r["arm"], r["qid"]): r for r in LL.done_keys("test").values()}
NV = {}
jd = HERE / "judge" / "test"
if jd.is_dir():
    NV.update(RT.final_verdicts_by_answer(jd))
IDX = {(r["arm"], r["qid"]): r for r in (json.loads(l) for l in open(HERE / "packs_index.jsonl", encoding="utf-8"))}


def score_new(arm, q):
    r = ANSW.get((arm, q))
    if r is None:
        return 0, 1
    if r["source"].startswith("frozen:"):
        return L.score(r["source"].split(":", 1)[1], q)
    rid = LL.identity(arm, q, r["setting"])["rid"]
    v = NV.get((rid, JM.answer_sha1({"answer_text": r["answer_text"]})))
    if v is None or v["status"] == "needs_adjudication":
        return 0, 1
    return v["strict"], v["strict"]


# ------------------------------------------------------------------ optional S10 bge-m3 DENSE-ONLY at 6000
S10_ARM, S10_SCORE, S10_NOTE = None, None, "S10_retrieval folder not present"
if S10.is_dir():
    S10_NOTE = "S10 folder present; see s10_hook()"
    try:
        sys.path.insert(0, str(HERE))
        import s10_hook  # noqa: E402  (written after inspecting S10's outputs)
        S10_ARM, S10_SCORE, S10_NOTE = s10_hook.load()
    except Exception as e:  # noqa: BLE001
        S10_NOTE = "S10 present but not loaded: %s" % e


def score(arm, q):
    if arm in NEW:
        return score_new(arm, q)
    if arm == S10_ARM:
        return S10_SCORE(q)
    return L.score(arm, q)


ARMS = NEW + ["HYBRID_6000", "HR_6000"] + ([S10_ARM] if S10_ARM else [])


def tot(arm, qs):
    return sum(score(arm, q)[0] for q in qs), sum(score(arm, q)[1] for q in qs)


def genre(q):
    return L.pool(q).split("-")[0]


def groups(arm):
    out = {}
    for key, f in (("tier", L.tier), ("lang", L.lang), ("genre", genre), ("pool", L.pool)):
        g = defaultdict(list)
        for q in L.ANS:
            g[f(q)].append(q)
        out[key] = {k: [*tot(arm, v), len(v)] for k, v in sorted(g.items())}
    return out


def messages(arm, q):
    if arm in NEW:
        f = HERE / "packs" / arm / (q + ".json")
        return json.loads(f.read_text(encoding="utf-8"))["messages"]
    if arm == S10_ARM:
        f = S10 / "packs" / "DENSE_6000" / (q + ".json")
        return json.loads(f.read_text(encoding="utf-8"))["messages"] if f.is_file() else None
    return L.messages(arm, q)


def reach(arm):
    c, by = Counter(), defaultdict(Counter)
    for q in L.ANS:
        m = messages(arm, q)
        if m is None:
            return None
        cls = L.gold_reach(q, m)["cls"]
        c[cls] += 1
        by["lang:" + L.lang(q)][cls] += 1
        by["tier:" + L.tier(q)][cls] += 1
        lo, hi = score(arm, q)
        by["outcome"][("unresolved" if lo != hi else ("correct_" if lo else "miss_") + cls)] += 1
    return {"all": dict(c), **{k: dict(v) for k, v in sorted(by.items())}}


def recall40():
    rows = [json.loads(l) for l in open(HERE / "lists.jsonl", encoding="utf-8")]
    rows = [r for r in rows if r["qid"] in set(L.ANS)]
    out = {}
    for a in ("M3_DENSE", "E5_DENSE", "M3_HYBRID", "E5_HYBRID"):
        k = "bundle40_" + a
        out[a] = {"all": [sum(bool(r[k]) for r in rows), len(rows)]}
        for lg in ("fa", "en"):
            rr = [r for r in rows if r["lang"] == lg]
            out[a][lg] = [sum(bool(r[k]) for r in rr), len(rr)]
        for t in ("T1", "T2", "T3"):
            rr = [r for r in rows if L.tier(r["qid"]) == t]
            out[a][t] = [sum(bool(r[k]) for r in rr), len(rr)]
    disc = {}
    for x, y in (("E5_DENSE", "M3_DENSE"), ("E5_HYBRID", "M3_HYBRID")):
        for lg in ("fa", "en", "all"):
            rr = [r for r in rows if lg == "all" or r["lang"] == lg]
            b = sum(bool(r["bundle40_" + x]) and not r["bundle40_" + y] for r in rr)
            w = sum(bool(r["bundle40_" + y]) and not r["bundle40_" + x] for r in rr)
            disc["%s vs %s|%s" % (x, y, lg)] = {"only_first": b, "only_second": w, "sign_p": L.sign_p(b, w)}
    out["paired"] = disc
    # consistency of the span-based list measure with S7's text-based pack measure: a pack is a subset of its top 40,
    # so pack reach "complete" should imply a complete bundle in the top 40
    cons = {}
    for arm, lst in (("HYBRID_6000", "M3_HYBRID"), ("E5_DENSE", "E5_DENSE"), ("E5_HYBRID", "E5_HYBRID")):
        bad = [r["qid"] for r in rows if L.gold_reach(r["qid"], messages(arm, r["qid"]))["cls"] == "complete"
               and not r["bundle40_" + lst]]
        cons[arm] = {"pack_complete_but_list_not": len(bad), "qids": bad}
    out["consistency_pack_vs_list"] = cons
    s10i = S10 / "packs_index.jsonl"
    if s10i.is_file():
        s10 = {r["qid"]: r["list40"] for r in (json.loads(l) for l in open(s10i, encoding="utf-8"))
               if r["arm"] == "DENSE_6000"}
        al = [json.loads(l) for l in open(HERE / "lists.jsonl", encoding="utf-8")]
        out["m3_dense_list_equal_s10"] = [sum(s10.get(r["qid"]) == r["M3_DENSE"] for r in al), len(al)]
    allrows = [json.loads(l) for l in open(HERE / "lists.jsonl", encoding="utf-8")]
    out["overlap40_mean"] = {k: {lg: round(sum(r[k] for r in allrows if lg == "all" or r["lang"] == lg) /
                                           sum(1 for r in allrows if lg == "all" or r["lang"] == lg), 2)
                                 for lg in ("fa", "en", "all")}
                             for k in ("overlap40_e5_m3_dense", "overlap40_e5_m3_hybrid")}
    return out


def tokens(arm):
    led = defaultdict(list)
    for l in open(HERE / "ledger.jsonl", encoding="utf-8"):
        if l.strip():
            c = json.loads(l)
            if c.get("purpose") == "answer" and c.get("ok"):
                led[(c["approach"], c["question_id"])].append(c)
    pt, alltok, src = [], [], Counter()
    for q in L.QIDS:
        r = ANSW.get((arm, q))
        if r is None:
            continue
        src[r["source"].split(":")[0]] += 1
        a = r["source"].split(":", 1)[1] if r["source"].startswith("dup:") else arm
        cs = led.get((a, q))
        if cs:
            pt.append(cs[-1]["prompt_tokens"])
            alltok.append(cs[-1]["prompt_tokens"] + cs[-1]["completion_tokens"])
    m = lambda x: round(sum(x) / len(x), 1) if x else None  # noqa: E731
    return {"answer_prompt_tokens_mean_called": m(pt), "query_tokens_mean_called": m(alltok), "n_called_cells": len(pt),
            "calls_per_question": 1, "answer_sources": dict(src)}


def main():
    res = {"s10": S10_NOTE, "table": {}, "groups": {}, "compare": {}, "reach": {}}
    for a in ARMS:
        res["table"][a] = {"acc90": tot(a, L.ANS), "unresolved": sum(score(a, q)[0] != score(a, q)[1] for q in L.ANS),
                           "abstain6": tot(a, L.AB)}
        if a in NEW:
            res["table"][a]["mean_chars"] = round(sum(IDX[(a, q)]["chars"] for q in L.QIDS) / len(L.QIDS))
            res["table"][a]["pack_identical_to_frozen"] = dict(Counter(
                (IDX[(a, q)]["same_as_frozen"] or ["none"])[0] for q in L.QIDS))
            res["table"][a]["tokens"] = tokens(a)
        res["groups"][a] = groups(a)
        res["reach"][a] = reach(a)
    pairs = [("E5_DENSE", "HYBRID_6000"), ("E5_HYBRID", "HYBRID_6000"), ("E5_DENSE", "HR_6000"),
             ("E5_HYBRID", "HR_6000"), ("E5_HYBRID", "E5_DENSE")]
    if S10_ARM:
        pairs += [("E5_DENSE", S10_ARM)]
    res["family_size"] = len(pairs)
    res["bonferroni_alpha"] = 0.05 / len(pairs)
    for a, b in pairs:
        c = L.compare(lambda q, a=a: score(a, q), lambda q, b=b: score(b, q))
        for lg in ("fa", "en"):
            qs = [q for q in L.ANS if L.lang(q) == lg]
            c["net_bounds_" + lg] = L.compare(lambda q, a=a: score(a, q), lambda q, b=b: score(b, q), qs)["net_bounds"]
        res["compare"]["%s vs %s" % (a, b)] = c
    res["recall40"] = recall40()
    res["gpu"] = json.loads((HERE / "gpu_phase.json").read_text(encoding="utf-8"))
    res["retrieve_checks"] = json.loads((HERE / "retrieve_checks.json").read_text(encoding="utf-8"))
    res["spent_usd"] = round(LL.spent(), 4)
    L.wjson(HERE / "results.json", res)
    print(json.dumps({k: res[k] for k in ("table", "compare", "recall40", "family_size", "spent_usd", "s10")},
                     indent=1, ensure_ascii=False))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
