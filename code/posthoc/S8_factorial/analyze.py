"""S8 analysis (no model calls) -> results.json, tables.md.
Judge 1 (pinned gpt-oss-120b) cells:
  frozen arms              report_test.load/score
  S2a arms (FUSION_B, HYBRID_B, HR_B, GOLD)   S2a's report.py score rule (copied): frozen:<arm> source -> frozen score,
                           else S2a judge dirs via report_test.final_verdicts_by_answer, keyed by S2a identity rid
  S6 / S3 arms             their judge dirs, keyed by (rid, answer_sha1) of the records they judged
  S8 arms                  LEX_3000/10000/24000 (lexbud), LEX_NODL_6000 (nodl), SELHR_6000 (selhr): reused cells take the
                           reused cell's score; new cells from <exp>/judge
Judge 2 (mistral-large-2512): frozen 8 cells from S1 (j2lib.judge2); post-hoc arms from judge2/<ARM>; FUSION_6000 cells
that are the frozen LEX5B_6000 answer take S1's verdict of that answer.
Statistics: revision_stats.py helpers extracted with ast and run unchanged (as S2a/report.py does)."""
import ast
import json
import os
import random
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
PH = REPO / "work" / "posthoc_2026-11"
TR = REPO / "work" / "datasets_v2" / "test_run"
FZ = REPO / "work" / "datasets_v2" / "test_freeze"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.path.insert(0, str(TR))
import report_test as RT  # noqa: E402
from ragbench.tools import judge_multi as JM  # noqa: E402

sys.path.insert(0, str(HERE))
import judge2_posthoc as J2P  # noqa: E402

sys.path.insert(0, str(PH / "S1_evaluator"))
import j2lib  # noqa: E402

P = RT.PRIMARY
REC, VER = RT.load()
ANS, AB, QS = RT.ANS, RT.AB, RT.QS


def jl(p):
    p = Path(p)
    return [json.loads(l) for l in open(p, encoding="utf-8-sig") if l.strip()] if p.is_file() else []


# ---------------------------------------------------------------- revision_stats helpers (unchanged source)
SRC = REPO / "paper" / "analysis" / "revision_stats.py"
tree = ast.parse(SRC.read_text(encoding="utf-8"))
want = {"find", "sign_p", "diff_vectors", "sign_range", "boot_mean", "compare"}
mod = ast.Module(body=[n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in want], type_ignores=[])
NS = {"RT": RT, "itertools": __import__("itertools"), "random": random, "defaultdict": defaultdict,
      "N_BOOT": 10000, "SEED": 0, "ANS": ANS, "parent": {}}
exec(compile(mod, str(SRC), "exec"), NS)
QROWS = {}
for line in open(FZ / "questions" / "test_v2judge_all.jsonl", encoding="utf-8-sig"):
    if line.strip():
        d = json.loads(line)
        QROWS[d["id"]] = d["meta"]["v2"]
cons = {q: set(QROWS[q]["source_family"].split("+")) for q in ANS}
for q in ANS:
    c = sorted(cons[q])
    for o in c[1:]:
        NS["parent"][NS["find"](o)] = NS["find"](c[0])
NS["comp"] = {q: NS["find"](sorted(cons[q])[0]) for q in ANS}
NS["block"] = {q: QROWS[q]["source_block"] for q in ANS}
_orig_sign_range = NS["sign_range"]


def sign_range_dp(fa, fb, qs):
    """Same output as revision_stats.sign_range, by a DP over reachable (better, worse) counts (exact; used only when
    there are more than 12 unresolved cells, where the 2^k enumeration is too slow)."""
    k = 0
    states = {(0, 0)}
    for q in qs:
        al, ah = fa(q)
        bl, bh = fb(q)
        k += (al != ah) + (bl != bh)
        opts = {(int(x > y), int(x < y)) for x in {al, ah} for y in {bl, bh}}
        states = {(b + db, w + dw) for b, w in states for db, dw in opts}
    ps = [NS["sign_p"](b, w) for b, w in states]
    nets = [b - w for b, w in states]
    return min(ps), max(ps), min(nets), max(nets), k


def sign_range_auto(fa, fb, qs):
    n = sum((fa(q)[0] != fa(q)[1]) + (fb(q)[0] != fb(q)[1]) for q in qs)
    return _orig_sign_range(fa, fb, qs) if n <= 12 else sign_range_dp(fa, fb, qs)


NS["sign_range"] = sign_range_auto
compare = NS["compare"]

# ---------------------------------------------------------------- judge 1
S2A = PH / "S2a_packs"
S2A_ROWS = J2P.s2a_rows()
S2A_NV = {}
for d in [S2A / "judge"] + sorted(p for p in (S2A / "judge").iterdir() if p.is_dir()):
    for k, v in RT.final_verdicts_by_answer(d).items():
        S2A_NV.setdefault(k, v)


def _v(v):
    if v is None or v["status"] == "needs_adjudication":
        return 0, 1
    return v["strict"], v["strict"]


def j1_frozen(arm, q):
    lo, hi, _ = RT.score(REC, VER, P, arm, q)
    return lo, hi


def j1_s2a(arm, q):
    r = S2A_ROWS.get((arm, q))
    if r is None:
        return 0, 1
    if r["source"].startswith("frozen:"):
        return j1_frozen(r["source"].split(":", 1)[1], q)
    rid = J2P.s2a_identity(arm, q, r["setting"])["rid"]
    return _v(S2A_NV.get((rid, JM.answer_sha1({"answer_text": r["answer_text"]}))))


_EXT = {}


def j1_ext(arm, q):
    if arm not in _EXT:
        nv = {}
        for d in J2P.J1_DIRS[arm]:
            for k, v in RT.final_verdicts_by_answer(d).items():
                nv.setdefault(k, v)
        _EXT[arm] = (J2P.records(arm)[1], nv)
    cell, nv = _EXT[arm]
    c = cell.get(q)
    if c is None or c[0] == "error":
        return 0, 0
    return _v(nv.get((c[1], c[2])))


_S8 = {}


def s8_assign(exp):
    if exp not in _S8:
        rows = {}
        for r in jl(HERE / exp / "assign.jsonl"):
            rows[(r["q"], r["budget"])] = r
        nv = RT.final_verdicts_by_answer(HERE / exp / "judge")
        _S8[exp] = (rows, nv)
    return _S8[exp]


def j1_s8(exp, budget, q):
    rows, nv = s8_assign(exp)
    r = rows.get((q, budget))
    if r is None:
        return 0, 1
    src = r["answer_source"]
    if src.startswith("s2a:"):
        return j1_s2a(src[4:], q)
    if src.startswith("frozen:"):
        return j1_frozen(src.split(":", 1)[1], q)
    if r.get("answer_error"):
        return 0, 0
    return _v(nv.get((r["answer_rid"], r["answer_sha1"])))


# ---------------------------------------------------------------- judge 2
_J2 = {}


def j2_ext(arm, q):
    if arm not in _J2:
        slug = JM._slug(J2P.JUDGE2)
        rows, pend = {}, {}
        d = J2P.OUT / arm
        for v in sorted(jl(d / ("verdicts_%s.jsonl" % slug)), key=lambda v: v["at"]):
            if v.get("judge") == J2P.JUDGE2 and v.get("judge_provider_pin") == J2P.PIN2:
                rows.setdefault((v["rid"], v["answer_sha1"]), v)
        for n in jl(d / ("needs_adjudication_%s.jsonl" % slug)):
            pend.setdefault((n["rid"], n["answer_sha1"]), n)
        _J2[arm] = (J2P.records(arm)[1], rows, pend)
    cell, rows, pend = _J2[arm]
    c = cell.get(q)
    if c is None:
        return 0, 1
    if c[0] == "error":
        return 0, 0
    if c[0] == "frozen":
        return j2lib.judge2(c[1], q)[:2]
    k = (c[1], c[2])
    if k in rows:
        return rows[k]["strict"], rows[k]["strict"]
    return 0, 1


def j2_frozen(arm, q):
    return j2lib.judge2(arm, q)[:2]


def j2_done(arm):
    """number of answerable cells resolved by judge 2 (to know whether an arm's judge-2 run is complete)."""
    return sum(1 for q in ANS if j2_ext(arm, q)[0] == j2_ext(arm, q)[1])


# ---------------------------------------------------------------- arms
J1 = {}
for a in RT.ARMS:
    J1[a] = (lambda a: lambda q: j1_frozen(a, q))(a)
for b in (3000, 6000, 10000, 15500, 24000):
    J1["FUSION_%d" % b] = (lambda b: lambda q: j1_s2a("FUSION_%d" % b, q))(b)
for p in ("HYBRID", "HR"):
    for b in (3000, 10000, 24000):
        J1["%s_%d" % (p, b)] = (lambda a: lambda q: j1_s2a(a, q))("%s_%d" % (p, b))
J1["GOLD"] = lambda q: j1_s2a("GOLD", q)
for a in ("SETR_NOFILL", "LISTWISE", "CLOSEDBOOK", "ITER_BM25", "GREP_AGENT"):
    J1[a] = (lambda a: lambda q: j1_ext(a, q))(a)
for b in (3000, 10000, 24000):
    J1["LEX5B_%d" % b] = (lambda b: lambda q: j1_s8("lexbud", b, q))(b)
J1["LEX5B_NODL_6000"] = lambda q: j1_s8("nodl", 6000, q)
J1["SELHR_6000"] = lambda q: j1_s8("selhr", 6000, q)

J2 = {a: (lambda a: lambda q: j2_frozen(a, q))(a) for a in RT.ARMS}
for a in ("FUSION_6000", "GOLD", "SETR_NOFILL", "LISTWISE", "CLOSEDBOOK", "ITER_BM25", "GREP_AGENT"):
    J2[a] = (lambda a: lambda q: j2_ext(a, q))(a)


def tot(f, qs):
    return [sum(f(q)[0] for q in qs), sum(f(q)[1] for q in qs)]


def groups(f):
    g = {"all": tot(f, ANS), "abstain_ok_6": tot(f, AB)}
    for t in ("T1", "T2", "T3"):
        g[t] = tot(f, [q for q in ANS if "-%s-" % t in q])
    for lang in ("fa", "en"):
        g[lang] = tot(f, [q for q in ANS if "-%s-" % lang in q])
    return g


def cmp(fa, fb):
    c = compare(fa, fb, ANS)
    for t in ("T1", "T2", "T3"):
        qs = [q for q in ANS if "-%s-" % t in q]
        lo, up, _ = NS["diff_vectors"](fa, fb, qs)
        c["net_" + t] = [sum(lo), sum(up)]
    for lang in ("fa", "en"):
        qs = [q for q in ANS if "-%s-" % lang in q]
        lo, up, _ = NS["diff_vectors"](fa, fb, qs)
        c["net_" + lang] = [sum(lo), sum(up)]
    return c


def fmt(lh):
    return "%d" % lh[0] if lh[0] == lh[1] else "%d..%d" % (lh[0], lh[1])


def fmts(lh):
    f = lambda x: ("%+d" % x) if x else "0"  # noqa: E731
    return f(lh[0]) if lh[0] == lh[1] else "%s..%s" % (f(lh[0]), f(lh[1]))


def fp(pr):
    a, b = pr
    return ("%.3g" % a) if abs(a - b) < 1e-12 else "%.3g-%.3g" % (a, b)


# ---------------------------------------------------------------- descriptives of S8 arms
def desc(exp, budget):
    rows, _ = s8_assign(exp)
    rs = [rows[(q, budget)] for q in RT.QIDS if (q, budget) in rows]
    runs = {r["meta"]["identity"]["rid"]: r for r in jl(HERE / exp / "runs.jsonl")}
    out = {"n": len(rs), "fallback": sum(r["fallback"] for r in rs),
           "status": dict(sorted(defaultdict(int, {}).items())),
           "answer_source": {}, "mean_chars": sum(r["evidence_chars"] for r in rs) / max(1, len(rs)),
           "pack_eq_lex5b6000": sum(r["pack_eq_lex5b6000"] for r in rs),
           "pack_eq_fusion": sum(r["pack_eq_fusion"] for r in rs)}
    st = defaultdict(int)
    for r in rs:
        st["%s/%s" % (r["status"], r["reason"])] += 1
        out["answer_source"][r["answer_source"]] = out["answer_source"].get(r["answer_source"], 0) + 1
    out["status"] = dict(st)
    sel = [r["selector"] for r in rs if r["selector"].get("prompt_tokens") is not None]
    if sel:
        out["selector_calls_with_usage"] = len(sel)
        out["selector_prompt_tok_mean"] = sum(s["prompt_tokens"] for s in sel) / len(sel)
        out["selector_completion_tok_mean"] = sum(s["completion_tokens"] or 0 for s in sel) / len(sel)
        ts = sorted(s["t_select_ms"] for s in sel)
        out["t_select_ms_p50"] = ts[len(ts) // 2]
        out["t_select_over_3s"] = sum(t > 3000 for t in ts)
        out["selector_failed"] = sum(1 for r in rs if r["selector"].get("failed"))
    new = [runs[r["answer_rid"]] for r in rs if r["answer_source"] == "new" and r["answer_rid"] in runs]
    if new:
        out["new_answers"] = len(new)
        out["new_answer_prompt_tok_mean"] = sum(r["answer_prompt_tokens"] or 0 for r in new) / len(new)
        out["new_answer_completion_tok_mean"] = sum(r["answer_completion_tokens"] or 0 for r in new) / len(new)
        out["new_answer_errors"] = sum(1 for r in new if r.get("error"))
    return out


def nodl_fallbacks():
    rows, _ = s8_assign("nodl")
    out = []
    for q in RT.QIDS:
        r = rows.get((q, 6000))
        if r is None:
            continue
        prim = RT.score(REC, VER, P, "LEX5B_6000", q)
        if r["saved_status"] == "fail":
            out.append({"q": q, "saved_reason": r["saved_reason"], "new_status": r["status"],
                        "new_reason": r["reason"], "t_select_ms": r["selector"].get("t_select_ms"),
                        "pack_changed": not r["pack_eq_lex5b6000"], "answer_source": r["answer_source"],
                        "j1_saved": list(prim[:2]), "j1_new": list(J1["LEX5B_NODL_6000"](q)),
                        "answerable": q in ANS})
    return out


def s2b_exp4():
    """S2b's exp4 (no-deadline re-run of the deadline fallbacks), for agreement with our nodl on those cells."""
    rows = {r["q"]: r for r in jl(PH / "S2b_variance" / "exp4" / "assign.jsonl")}
    mine, _ = s8_assign("nodl")
    agree = sum(1 for q, r in rows.items() if (q, 6000) in mine and mine[(q, 6000)]["pack_sha"] == r["pack_sha"])
    return {"n_s2b": len(rows), "same_pack_as_s8": agree}


# ---------------------------------------------------------------- dev replication
def dev():
    DS = REPO / "work" / "datasets_v2"
    d60 = json.loads((DS / "dev60" / "dev60_results.json").read_text(encoding="utf-8"))["per_q"]
    rows = {r["q"]: r for r in jl(HERE / "dev_fusion" / "assign.jsonl")}
    if not rows:
        return None
    nv = RT.final_verdicts_by_answer(HERE / "dev_fusion" / "judge")
    qs = sorted(d60["LEX5B@6000"])
    ans = [q for q in qs if d60["LEX5B@6000"][q].get("answerable", "-AB-" not in q) and "-AB-" not in q]
    abq = [q for q in qs if "-AB-" in q]

    def fus(q):
        r = rows.get(q)
        if r is None:
            return 0, 1
        if r.get("answer_error"):
            return 0, 0
        return _v(nv.get((r["answer_rid"], r["answer_sha1"])))

    def saved(cell):
        return lambda q: (int(d60[cell][q]["correct"]), int(d60[cell][q]["correct"])) if q in d60[cell] else (0, 1)
    F = {"FUSION@6000": fus}
    for c in d60:
        F[c] = saved(c)

    def dcomp(a, b, qset):
        lo, up, unres = NS["diff_vectors"](F[a], F[b], qset)
        p_lo, p_hi, n_lo, n_hi, k = NS["sign_range"](F[a], F[b], qset)
        bw = [0, 0]
        for q in qset:
            al, ah = F[a](q)
            bl, bh = F[b](q)
            if al == ah and bl == bh:
                bw[0] += al > bl
                bw[1] += al < bl
        ci = [min(NS["boot_mean"](lo)[0], NS["boot_mean"](up)[0]), max(NS["boot_mean"](lo)[1], NS["boot_mean"](up)[1])]
        return {"net": [sum(lo), sum(up)], "better_worse": bw, "sign_p": [p_lo, p_hi], "ci_pp_question": ci,
                "unresolved": k}
    out = {"n_answerable": len(ans), "acc": {c: tot(F[c], ans) for c in F},
           "abstain_ok": {c: tot(F[c], abq) for c in F},
           "pack_same_as_lex5b": sum(1 for q in ans if rows.get(q, {}).get("pack_same_as_lex5b")),
           "cmp": {}}
    for a, b in (("LEX5B@6000", "FUSION@6000"), ("FUSION@6000", "HYBRID@6000"), ("FUSION@6000", "HR@6000"),
                 ("HYBRID@15500", "HYBRID@6000"), ("HR@15500", "HR@6000"), ("LEX5B@6000", "HYBRID@6000"),
                 ("LEX5B@6000", "HR@6000")):
        out["cmp"]["%s vs %s" % (a, b)] = dcomp(a, b, ans)
    # FUSION fresh answer vs saved LEX5B on the identical-pack cells (reader nondeterminism on dev)
    same = [q for q in ans if rows.get(q, {}).get("pack_same_as_lex5b")]
    out["identical_pack_cells"] = {"n": len(same), "fusion": tot(fus, same), "lex5b_saved": tot(F["LEX5B@6000"], same)}
    return out


def main():
    res = {"acc_j1": {}, "acc_j2": {}, "cmp_j1": {}, "cmp_j2": {}, "desc": {}}
    arms1 = ["HYBRID_%d" % b for b in (3000, 6000, 10000, 15500, 24000)] + \
            ["HR_%d" % b for b in (3000, 6000, 10000, 15500, 24000)] + \
            ["FUSION_%d" % b for b in (3000, 6000, 10000, 15500, 24000)] + \
            ["LEX5B_%d" % b for b in (3000, 6000, 10000, 15500, 24000)] + \
            ["LEX5B_NODL_6000", "SELHR_6000", "SETR_6000", "SETR_15500", "SETR_NOFILL", "LISTWISE", "ITER_BM25",
             "GREP_AGENT", "GOLD", "CLOSEDBOOK"]
    for a in arms1:
        res["acc_j1"][a] = groups(J1[a])
    j2arms = [a for a in J2 if a not in RT.ARMS]
    res["j2_resolved"] = {a: j2_done(a) for a in j2arms}
    for a in list(RT.ARMS) + j2arms:
        res["acc_j2"][a] = groups(J2[a])
    C1 = []
    for b in (3000, 6000, 10000, 15500, 24000):
        C1 += [("LEX5B_%d" % b, "FUSION_%d" % b), ("LEX5B_%d" % b, "HYBRID_%d" % b), ("LEX5B_%d" % b, "HR_%d" % b)]
    C1 += [("LEX5B_3000", "LEX5B_6000"), ("LEX5B_10000", "LEX5B_6000"), ("LEX5B_15500", "LEX5B_10000"),
           ("LEX5B_24000", "LEX5B_15500"), ("LEX5B_24000", "LEX5B_6000"),
           ("SELHR_6000", "LEX5B_NODL_6000"), ("SELHR_6000", "HR_6000"), ("SELHR_6000", "LEX5B_6000"),
           ("LEX5B_NODL_6000", "LEX5B_6000"), ("LEX5B_NODL_6000", "FUSION_6000"), ("LEX5B_NODL_6000", "HYBRID_6000"),
           ("LEX5B_NODL_6000", "HR_6000")]
    for a, b in C1:
        res["cmp_j1"]["%s vs %s" % (a, b)] = cmp(J1[a], J1[b])
    KEY = [("LEX5B_6000", "FUSION_6000"), ("FUSION_6000", "HYBRID_6000"), ("FUSION_6000", "HR_6000"),
           ("SETR_NOFILL", "SETR_6000"), ("SETR_NOFILL", "HYBRID_6000"), ("SETR_NOFILL", "LEX5B_6000"),
           ("LISTWISE", "HYBRID_6000"), ("LISTWISE", "HR_6000"), ("LISTWISE", "LEX5B_6000"),
           ("ITER_BM25", "HYBRID_15500"), ("ITER_BM25", "HR_15500"), ("ITER_BM25", "LEX5B_15500"),
           ("GREP_AGENT", "HYBRID_15500"), ("GREP_AGENT", "HR_15500"), ("GREP_AGENT", "LEX5B_15500"),
           ("ITER_BM25", "GREP_AGENT"), ("GOLD", "LEX5B_6000"), ("GOLD", "HR_6000"), ("GOLD", "HYBRID_15500"),
           ("CLOSEDBOOK", "HYBRID_6000")]
    for a, b in KEY:
        k = "%s vs %s" % (a, b)
        res["cmp_j1"][k] = cmp(J1[a], J1[b])
        res["cmp_j2"][k] = cmp(J2[a], J2[b])
    for exp, b in (("lexbud", 3000), ("lexbud", 10000), ("lexbud", 24000), ("nodl", 6000), ("selhr", 6000)):
        res["desc"]["%s_%d" % (exp, b)] = desc(exp, b)
    # judge-2 partial: on the cells judge 2 resolved, agreement with judge 1 (all 96 questions incl. rule verdicts)
    res["j2_partial"] = {}
    for a in j2arms:
        pairs = [(J1[a](q)[0], J2[a](q)[0]) for q in RT.QIDS
                 if J1[a](q)[0] == J1[a](q)[1] and J2[a](q)[0] == J2[a](q)[1] and (q in ANS or a != "GOLD")]
        res["j2_partial"][a] = {"resolved_both": len(pairs), "agree": sum(x == y for x, y in pairs),
                                "j1_1_j2_0": sum(x > y for x, y in pairs), "j1_0_j2_1": sum(x < y for x, y in pairs),
                                "j1_correct": sum(x for x, _ in pairs), "j2_correct": sum(y for _, y in pairs)}
    res["nodl_fallbacks"] = nodl_fallbacks()
    res["s2b_exp4_agreement"] = s2b_exp4()
    res["dev"] = dev()
    res["step0"] = json.loads((HERE / "step0_checks.json").read_text(encoding="utf-8"))
    (HERE / "results.json").write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    # ----- tables
    T = ["# S8 tables (written by analyze.py; judge 1 = pinned gpt-oss-120b, judge 2 = mistral-large-2512)", ""]
    T += ["## Accuracy, judge 1 (/90 answerable; abstention /6)", "",
          "| arm | all | T1 | T2 | T3 | fa | en | abst /6 |", "|---|---|---|---|---|---|---|---|"]
    for a in arms1:
        g = res["acc_j1"][a]
        T.append("| %s | %s | %s | %s | %s | %s | %s | %s |" % (a, fmt(g["all"]), fmt(g["T1"]), fmt(g["T2"]),
                                                                 fmt(g["T3"]), fmt(g["fa"]), fmt(g["en"]),
                                                                 fmt(g["abstain_ok_6"])))
    T += ["", "## Accuracy, judge 2 (/90; abstention /6; resolved = answerable cells resolved by judge 2)", "",
          "| arm | all J2 | all J1 | T1 | T2 | T3 | fa | en | abst /6 | resolved /90 |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for a in list(RT.ARMS) + j2arms:
        g = res["acc_j2"][a]
        g1 = res["acc_j1"].get(a) or groups(J1[a])
        T.append("| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
            a, fmt(g["all"]), fmt(g1["all"]), fmt(g["T1"]), fmt(g["T2"]), fmt(g["T3"]), fmt(g["fa"]), fmt(g["en"]),
            fmt(g["abstain_ok_6"]), res["j2_resolved"].get(a, "S1")))

    def crow(k, c):
        return "| %s | %s | %d/%d | %s | [%.1f, %.1f] | [%.1f, %.1f] | %s / %s / %s | %s / %s |" % (
            k, fmts(c["net_bounds"]), c["better_worse_resolved"][0], c["better_worse_resolved"][1], fp(c["sign_p_range"]),
            c["ci_pp_question"][0], c["ci_pp_question"][1], c["ci_pp_component"][0], c["ci_pp_component"][1],
            fmts(c["net_T1"]), fmts(c["net_T2"]), fmts(c["net_T3"]), fmts(c["net_fa"]), fmts(c["net_en"]))
    H = ["| comparison | net /90 | b/w | sign p | CI q (pp) | CI comp (pp) | T1 / T2 / T3 | fa / en |",
         "|---|---|---|---|---|---|---|---|"]
    T += ["", "## Paired comparisons, judge 1 (S8 arms)", ""] + H
    for a, b in C1:
        k = "%s vs %s" % (a, b)
        T.append(crow(k, res["cmp_j1"][k]))
    T += ["", "## Key post-hoc contrasts under both judges", ""] + H
    for a, b in KEY:
        k = "%s vs %s" % (a, b)
        T.append(crow(k + " [J1]", res["cmp_j1"][k]))
        T.append(crow(k + " [J2]", res["cmp_j2"][k]))
    T += ["", "## Judge 2 partial (cells resolved by both judges; all 96 questions)", "",
          "| arm | resolved by both | agree | J1=1,J2=0 | J1=0,J2=1 | J1 correct | J2 correct |", "|---|---|---|---|---|---|---|"]
    for a, v in res["j2_partial"].items():
        T.append("| %s | %d | %d | %d | %d | %d | %d |" % (a, v["resolved_both"], v["agree"], v["j1_1_j2_0"],
                                                          v["j1_0_j2_1"], v["j1_correct"], v["j2_correct"]))
    T += ["", "## S8 arm descriptives", "", "```", json.dumps(res["desc"], indent=1), "```",
          "", "## No-deadline selector on the 19 saved LEX5B_6000 fallbacks", "",
          "| q | saved reason | new status | t_select ms | pack changed | answer | J1 saved | J1 new |",
          "|---|---|---|---|---|---|---|---|"]
    for r in res["nodl_fallbacks"]:
        T.append("| %s | %s | %s/%s | %s | %s | %s | %s | %s |" % (
            r["q"], r["saved_reason"], r["new_status"], r["new_reason"], r["t_select_ms"], r["pack_changed"],
            r["answer_source"], fmt(r["j1_saved"]), fmt(r["j1_new"])))
    if res["dev"]:
        d = res["dev"]
        T += ["", "## Dev replication (%d answerable dev questions; saved dev60 verdicts for the frozen dev arms)" %
              d["n_answerable"], "", "| cell | acc | abst |", "|---|---|---|"]
        for c, v in d["acc"].items():
            T.append("| %s | %s | %s |" % (c, fmt(v), fmt(d["abstain_ok"][c])))
        T += ["", "| comparison | net | b/w | sign p | CI q (pp) |", "|---|---|---|---|---|"]
        for k, c in d["cmp"].items():
            T.append("| %s | %s | %d/%d | %s | [%.1f, %.1f] |" % (k, fmts(c["net"]), c["better_worse"][0],
                                                                  c["better_worse"][1], fp(c["sign_p"]),
                                                                  c["ci_pp_question"][0], c["ci_pp_question"][1]))
        T += ["", "identical-pack cells: %s" % json.dumps(d["identical_pack_cells"])]
    (HERE / "tables.md").write_text("\n".join(T) + "\n", encoding="utf-8")
    print("\n".join(T))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
