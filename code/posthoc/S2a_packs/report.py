"""S2a report: scores, paired comparisons, tiers/languages, abstention, tokens. No model call.
Frozen cells: report_test.load/score (the test's own scoring). New cells: our judge dirs, read with
report_test.final_verdicts_by_answer (same policy). Comparison statistics: the revision_stats.py helper functions
(sign_p, diff_vectors, sign_range, boot_mean, compare, find), extracted from its source with ast and executed unchanged
(importing that module would recompute and overwrite the paper's own outputs). -> results.json, printed tables."""
import ast
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import packs as PK

TRD = str(PK.TR)
os.environ["ANYRAG_TEST_DIR"] = TRD
os.environ["ANYRAG_FREEZE_DIR"] = str(PK.FZ)
os.environ["ANYRAG_CODE_DIR"] = PK.FROZEN
sys.path.insert(0, TRD)
import report_test as RT  # noqa: E402

P = RT.PRIMARY
REC, VER = RT.load()
QS, ANS, AB = RT.QS, RT.ANS, RT.AB

# ---------------------------------------------------------------- revision_stats helpers (unchanged source)
SRC = PK.ROOT / "paper" / "analysis" / "revision_stats.py"
tree = ast.parse(SRC.read_text(encoding="utf-8"))
want = {"find", "sign_p", "diff_vectors", "sign_range", "boot_mean", "compare"}
mod = ast.Module(body=[n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in want], type_ignores=[])
NS = {"RT": RT, "itertools": __import__("itertools"), "random": __import__("random"), "defaultdict": defaultdict,
      "N_BOOT": 10000, "SEED": 0, "ANS": ANS, "parent": {}}
exec(compile(mod, str(SRC), "exec"), NS)
QROWS = {}
for line in open(PK.FZ / "questions" / "test_v2judge_all.jsonl", encoding="utf-8-sig"):
    if line.strip():
        d = json.loads(line)
        QROWS[d["id"]] = d["meta"]["v2"]
constituents = {q: set(QROWS[q]["source_family"].split("+")) for q in ANS}
for q in ANS:
    c = sorted(constituents[q])
    for o in c[1:]:
        NS["parent"][NS["find"](o)] = NS["find"](c[0])
NS["comp"] = {q: NS["find"](sorted(constituents[q])[0]) for q in ANS}
NS["block"] = {q: QROWS[q]["source_block"] for q in ANS}
compare = NS["compare"]

# ---------------------------------------------------------------- new cells
ANSW = {}
for l in open(PK.HERE / "answers.jsonl", encoding="utf-8"):
    if l.strip():
        r = json.loads(l)
        if not r.get("error"):
            ANSW[(r["arm"], r["qid"])] = r
import judge as JG  # noqa: E402

NV = {}
for d in [PK.HERE / "judge"] + sorted(p for p in (PK.HERE / "judge").iterdir() if p.is_dir()):
    for k, v in RT.final_verdicts_by_answer(d).items():
        NV.setdefault(k, v)
from ragbench.tools import judge_multi as JM  # noqa: E402

FROZEN_ARMS = RT.ARMS


def score(arm, q):
    if arm in FROZEN_ARMS:
        lo, hi, _ = RT.score(REC, VER, P, arm, q)
        return lo, hi
    r = ANSW.get((arm, q))
    if r is None:
        return 0, 1                                   # not run (reported)
    if r["source"].startswith("frozen:"):
        lo, hi, _ = RT.score(REC, VER, P, r["source"].split(":", 1)[1], q)
        return lo, hi
    rid = JG.identity(arm, q, r["setting"])["rid"]
    v = NV.get((rid, JM.answer_sha1({"answer_text": r["answer_text"]})))
    if v is None or v["status"] == "needs_adjudication":
        return 0, 1
    return v["strict"], v["strict"]


def status(arm, q):
    if arm in FROZEN_ARMS:
        return RT.score(REC, VER, P, arm, q)[2]
    r = ANSW.get((arm, q))
    if r is None:
        return "missing"
    if r["source"].startswith("frozen:"):
        return "frozen"
    rid = JG.identity(arm, q, r["setting"])["rid"]
    v = NV.get((rid, JM.answer_sha1({"answer_text": r["answer_text"]})))
    return "unjudged" if v is None else v["status"]


# ---------------------------------------------------------------- tokens
def frozen_answer_calls(arm):
    out = defaultdict(list)
    for f in (PK.TR / "runs").glob("*/%s/ledger_*.jsonl" % arm):
        for l in open(f, encoding="utf-8"):
            if l.strip():
                c = json.loads(l)
                if c.get("kind") == "chat" and c.get("phase") == "query":
                    out[c["question_id"]].append(c)
    return out


FCALLS = {a: frozen_answer_calls(a) for a in FROZEN_ARMS}
MYL = defaultdict(list)
for l in open(PK.HERE / "ledger.jsonl", encoding="utf-8"):
    if l.strip():
        c = json.loads(l)
        if c.get("purpose") == "answer" and c.get("ok"):
            MYL[(c["approach"], c["question_id"])].append(c)


def tokens(arm, q, _seen=None):
    """-> (answer prompt tokens, all query-LLM tokens, query calls) of a cell; None when unknown."""
    if arm in FROZEN_ARMS:
        cs = [c for c in FCALLS[arm].get(q, []) if c.get("ok")]
        ans = [c for c in cs if c.get("purpose") == "answer"]
        if not ans:
            return None
        return (ans[-1]["prompt_tokens"], sum((c.get("prompt_tokens") or 0) + (c.get("completion_tokens") or 0)
                                               for c in cs), len(FCALLS[arm].get(q, [])))
    r = ANSW.get((arm, q))
    if r is None:
        return None
    if r["source"].startswith("frozen:"):
        t = tokens(r["source"].split(":", 1)[1], q)
        fa = r["source"].split(":", 1)[1]
        ans = [c for c in FCALLS[fa].get(q, []) if c.get("ok") and c.get("purpose") == "answer"][-1]
        return (ans["prompt_tokens"], ans["prompt_tokens"] + ans["completion_tokens"], 1) if t else None
    src = r["source"].split(":", 1)[1] if r["source"].startswith("dup:") else arm
    cs = MYL.get((src, q))
    if not cs:
        return None
    c = cs[-1]
    return c["prompt_tokens"], c["prompt_tokens"] + c["completion_tokens"], 1


def chars(arm, q):
    if arm in FROZEN_ARMS:
        r = REC[P][arm].get(q)
        return r["evidence_chars"] if r else None
    r = ANSW.get((arm, q))
    return r["chars"] if r else None


# ---------------------------------------------------------------- tables
BUDGETS = [3000, 6000, 10000, 15500, 24000]
PIPES = ["HYBRID", "HR", "FUSION"]


def arm_name(p, b):
    return "%s_%d" % (p, b)


def tot(arm, qs=ANS):
    lo = sum(score(arm, q)[0] for q in qs)
    hi = sum(score(arm, q)[1] for q in qs)
    return lo, hi


def mean(xs):
    xs = [x for x in xs if x is not None]
    return round(sum(xs) / len(xs), 1) if xs else None


def cell_row(arm):
    t = {lg: [tokens(arm, q) for q in QS if QS[q]["lang"] == lg] for lg in ("en", "fa")}
    return {"arm": arm, "acc90": tot(arm), "n_scored": sum(1 for q in ANS if status(arm, q) != "missing"),
            "unresolved": sum(1 for q in ANS if score(arm, q)[0] != score(arm, q)[1]),
            "chars_mean": mean([chars(arm, q) for q in QS]),
            "answer_prompt_tokens_en": mean([x[0] if x else None for x in t["en"]]),
            "answer_prompt_tokens_fa": mean([x[0] if x else None for x in t["fa"]]),
            "query_tokens_all_mean": mean([x[1] if x else None for x in t["en"] + t["fa"]]),
            "query_calls_mean": mean([x[2] if x else None for x in t["en"] + t["fa"]]),
            "tokens_unknown": sum(1 for x in t["en"] + t["fa"] if x is None),
            "abstain_correct6": tot(arm, AB)}


def by_group(arm):
    out = {}
    for key, f in (("tier", lambda q: QS[q]["tier"]), ("lang", lambda q: QS[q]["lang"])):
        g = defaultdict(list)
        for q in ANS:
            g[f(q)].append(q)
        out[key] = {k: [*tot(arm, v), len(v)] for k, v in sorted(g.items())}
    return out


def main():
    res = {"table": [], "groups": {}, "compare": {}}
    arms = [arm_name(p, b) for p in PIPES for b in BUDGETS] + ["LEX5B_6000", "LEX5B_15500", "SETR_6000",
                                                              "SETR_15500", "GOLD"]
    extra = [a for a in ("KNAP_6000", "MMR_6000") if any(k[0] == a for k in ANSW)]
    arms += extra
    for a in arms:
        if a in FROZEN_ARMS or any(k[0] == a for k in ANSW):
            res["table"].append(cell_row(a))
            res["groups"][a] = by_group(a)
    pairs = [("FUSION_6000", "LEX5B_6000"), ("LEX5B_6000", "FUSION_6000"), ("FUSION_6000", "HYBRID_6000"),
             ("FUSION_6000", "HR_6000"), ("FUSION_15500", "LEX5B_15500"), ("LEX5B_15500", "FUSION_15500"),
             ("FUSION_15500", "HYBRID_15500"), ("FUSION_15500", "HR_15500"), ("GOLD", "LEX5B_6000"),
             ("GOLD", "HYBRID_15500")]
    for p in PIPES:
        for b1, b2 in zip(BUDGETS, BUDGETS[1:]):
            pairs.append((arm_name(p, b2), arm_name(p, b1)))
    for b in (3000, 10000, 24000):
        pairs += [(arm_name("FUSION", b), arm_name("HYBRID", b)), (arm_name("FUSION", b), arm_name("HR", b)),
                  (arm_name("HR", b), arm_name("HYBRID", b))]
    for a in extra:
        pairs += [(a, "FUSION_6000"), (a, "LEX5B_6000"), (a, "HYBRID_6000"), (a, "HR_6000")]
    for a, b in pairs:
        if all(x in FROZEN_ARMS or any(k[0] == x for k in ANSW) for x in (a, b)):
            k = sum((score(a, q)[0] != score(a, q)[1]) + (score(b, q)[0] != score(b, q)[1]) for q in ANS)
            if k > 16:      # 2^k enumeration; only while judging is incomplete
                res["compare"]["%s vs %s" % (a, b)] = {"skipped": "unresolved cells %d > 16" % k}
                continue
            res["compare"]["%s vs %s" % (a, b)] = compare(lambda q, a=a: score(a, q), lambda q, b=b: score(b, q))
    # the selector's effect: questions where LEX5B's pack equals the fallback (identical messages -> same answer)
    same = [q for q in QS if ANSW.get(("FUSION_6000", q), {}).get("source") == "frozen:LEX5B_6000"]
    same15 = [q for q in QS if ANSW.get(("FUSION_15500", q), {}).get("source") == "frozen:LEX5B_15500"]
    res["fusion_pack_identical_to_lex5b"] = {"6000": len(same), "15500": len(same15)}
    sel = defaultdict(int)
    for q in QS:
        st = REC[P]["LEX5B_6000"][q]["trace"]["lex5b"]["selector"]
        sel["fallback" if st.get("fallback") else ("ok_same_pack" if q in same else "ok_diff_pack")] += 1
    res["lex5b_6000_selector"] = dict(sel)
    res["status_counts"] = {a: dict(__import__("collections").Counter(status(a, q) for q in QS)) for a in arms
                            if a not in FROZEN_ARMS and any(k[0] == a for k in ANSW)}
    (PK.HERE / "results.json").write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    for r in res["table"]:
        print("%-13s acc %s/90 unres %d chars %s ptok en %s fa %s all %s calls %s unk %d abst %s" % (
            r["arm"], r["acc90"], r["unresolved"], r["chars_mean"], r["answer_prompt_tokens_en"],
            r["answer_prompt_tokens_fa"], r["query_tokens_all_mean"], r["query_calls_mean"], r["tokens_unknown"],
            r["abstain_correct6"]))
    for k, v in res["compare"].items():
        if "skipped" in v:
            print(k, v)
            continue
        print("%-28s net %s p %s ci_q %s ci_comp %s bw %s" % (
            k, v["net_bounds"], [round(x, 4) for x in v["sign_p_range"]], [round(x, 1) for x in v["ci_pp_question"]],
            [round(x, 1) for x in v["ci_pp_component"]], v["better_worse_resolved"]))
    print(json.dumps({k: res[k] for k in ("fusion_pack_identical_to_lex5b", "lex5b_6000_selector", "status_counts")}))


if __name__ == "__main__":
    main()
