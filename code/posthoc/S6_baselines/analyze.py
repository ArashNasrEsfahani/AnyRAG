"""S6 analysis (no model calls): scores of the S6 arms (own pinned-judge verdicts) next to the frozen GLM arms (saved
machine verdicts via report_test.load/score), paired comparisons with the revision_stats method (bounds over
unresolved cells, exact sign-test range, bootstrap over questions and over the connected components of source
families; the helper functions below are copied verbatim from paper/analysis/revision_stats.py, which cannot be
imported without rewriting its outputs). Writes analysis.json."""
import itertools
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, r"D:\KareOne\RAG Paper\paper\analysis")
import layout  # noqa: E402

LAYOUT = layout.resolve()
RT = layout.import_report_test(LAYOUT)
import s6lib as L  # noqa: E402

JM = RT.JM
N_BOOT, SEED = 10000, 0
P = RT.PRIMARY
QS, ANS, AB = RT.QS, RT.ANS, RT.AB
REC, VER = RT.load()
S6_ARMS = ["CLOSEDBOOK", "LISTWISE", "SETR_NOFILL"]
FROZEN = ["HYBRID_6000", "HR_6000", "LEX5B_6000", "SETR_6000"]

# ------------------------------------------------------------------ S6 records + verdicts
S6REC = {}
for a in S6_ARMS:
    S6REC[a] = {}
    for r in L.jl(HERE / "test" / a / ("runs_%s.jsonl" % a)):
        old = S6REC[a].get(r["question_id"])
        if old is None or old.get("error") or not r.get("error"):
            S6REC[a][r["question_id"]] = r
S6VER = {}
for a in S6_ARMS:          # one pinned-judge pass per arm (run in parallel); test/judge_stopped_partial is not used
    S6VER.update(RT.final_verdicts_by_answer(HERE / "test" / ("judge_%s" % a)))


def s6score(a, q):
    r = S6REC[a].get(q)
    if r is None:
        return 0, 0, "missing"
    if r.get("error"):
        return 0, 0, "system_failure"
    v = S6VER.get((r["meta"]["identity"]["rid"], JM.answer_sha1(r)))
    if v is None:
        return 0, 1, "unjudged"
    if v["status"] == "needs_adjudication":
        return 0, 1, "needs_adjudication"
    return v["strict"], v["strict"], v["status"]


def sc(a, q):
    if a in S6_ARMS:
        return s6score(a, q)
    return RT.score(REC, VER, P, a, q)


# ------------------------------------------------------------------ components (verbatim logic of revision_stats)
QROWS = {}
for line in open(LAYOUT["questions"], encoding="utf-8-sig"):
    if line.strip():
        d = json.loads(line)
        QROWS[d["id"]] = d["meta"]["v2"]
fam_label = {q: QROWS[q]["source_family"] for q in ANS}
block = {q: QROWS[q]["source_block"] for q in ANS}
constituents = {q: set(fam_label[q].split("+")) for q in ANS}
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


def sign_p(b, w):
    return RT.sign_p(b, w)


def diff_vectors(fa, fb, qs):
    lo, up, unres = [], [], []
    for q in qs:
        al, ah = fa(q)
        bl, bh = fb(q)
        lo.append(al - bh)
        up.append(ah - bl)
        if (al, ah) != (ah, ah) or (bl, bh) != (bh, bh):
            unres.append(q)
    return lo, up, unres


def sign_range(fa, fb, qs):
    cells = []
    for q in qs:
        for f, tag in ((fa, "a"), (fb, "b")):
            l, h = f(q)
            if l != h:
                cells.append((q, tag))
    ps, nets = [], []
    for assign in itertools.product((0, 1), repeat=len(cells)):
        val = dict(zip(cells, assign))
        b = w = net = 0
        for q in qs:
            av = val.get((q, "a"), fa(q)[0])
            bv = val.get((q, "b"), fb(q)[0])
            net += av - bv
            b += av > bv
            w += av < bv
        ps.append(sign_p(b, w))
        nets.append(net)
    return min(ps), max(ps), min(nets), max(nets), len(cells)


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


def compare(fa, fb, qs=ANS):
    lo, up, unres = diff_vectors(fa, fb, qs)
    p_lo, p_hi, n_lo, n_hi, k = sign_range(fa, fb, qs)
    out = {"net_bounds": [sum(lo), sum(up)], "pp_bounds": [100 * sum(lo) / len(qs), 100 * sum(up) / len(qs)],
           "unresolved_cells": k, "sign_p_range": [p_lo, p_hi]}
    for name, cl in (("question", None), ("component", [comp[q] for q in qs])):
        a1 = boot_mean(lo, cl)
        a2 = boot_mean(up, cl)
        out["ci_pp_" + name] = [min(a1[0], a2[0]), max(a1[1], a2[1])]
    b = w = 0
    for q in qs:
        al, ah = fa(q)
        bl, bh = fb(q)
        if al == ah and bl == bh:
            b += al > bl
            w += al < bl
    out["better_worse_resolved"] = [b, w]
    return out


def F(a):
    return lambda q: sc(a, q)[:2]


# ------------------------------------------------------------------ accuracy tables
def subset(pred):
    return [q for q in ANS if pred(q)]


GROUPS = {"all": ANS}
for t in ("T1", "T2", "T3"):
    GROUPS[t] = subset(lambda q, t=t: QS[q]["tier"] == t)
for lg in ("fa", "en"):
    GROUPS[lg] = subset(lambda q, lg=lg: QS[q]["lang"] == lg)
for dm in ("BOOK", "LAW", "RPT"):
    GROUPS[dm] = subset(lambda q, dm=dm: QS[q]["domain"] == dm)
for dm in ("BOOK", "LAW", "RPT"):
    for lg in ("fa", "en"):
        GROUPS["%s-%s" % (dm, lg)] = subset(lambda q, dm=dm, lg=lg: QS[q]["domain"] == dm and QS[q]["lang"] == lg)

res = {"n_answerable": len(ANS), "n_unanswerable": len(AB), "groups_n": {g: len(v) for g, v in GROUPS.items()}}
acc = {}
status = {}
for a in S6_ARMS + FROZEN:
    acc[a] = {}
    for g, qs in GROUPS.items():
        lo = sum(sc(a, q)[0] for q in qs)
        hi = sum(sc(a, q)[1] for q in qs)
        acc[a][g] = [lo, hi]
    st = defaultdict(int)
    for q in RT.QIDS:
        st[sc(a, q)[2]] += 1
    status[a] = dict(st)
res["accuracy"] = acc
res["cell_status"] = status

# unanswerable: strict 1 = correct abstention (judge rule)
unans = {}
for a in S6_ARMS + FROZEN:
    rows = []
    for q in AB:
        lo, hi, s = sc(a, q)
        r = S6REC[a].get(q) if a in S6_ARMS else REC[P][a].get(q)
        rows.append({"q": q, "strict": [lo, hi], "abstained": bool(r and r.get("abstained")),
                     "answer_head": (r or {}).get("answer_text", "")[:160]})
    unans[a] = {"correct": [sum(x["strict"][0] for x in rows), sum(x["strict"][1] for x in rows)],
                "abstained": sum(x["abstained"] for x in rows), "rows": rows}
res["unanswerable"] = unans

# closed-book abstention on answerable questions, and overlap with HYBRID
cb_abst = {g: sum(1 for q in qs if (S6REC["CLOSEDBOOK"].get(q) or {}).get("abstained")) for g, qs in GROUPS.items()}
res["closedbook_abstained_answerable"] = cb_abst
ov = defaultdict(int)
for q in ANS:
    c, h = sc("CLOSEDBOOK", q), sc("HYBRID_6000", q)
    if c[0] != c[1] or h[0] != h[1]:
        ov["unresolved"] += 1
    else:
        ov["cb%d_hyb%d" % (c[0], h[0])] += 1
res["closedbook_vs_hybrid_cells"] = dict(ov)
res["closedbook_correct_qids"] = [q for q in ANS if sc("CLOSEDBOOK", q)[0] == 1]

# ------------------------------------------------------------------ paired comparisons
pairs = [("CLOSEDBOOK", "HYBRID_6000"), ("LISTWISE", "HYBRID_6000"), ("LISTWISE", "HR_6000"),
         ("HR_6000", "HYBRID_6000"), ("LISTWISE", "LEX5B_6000"), ("LISTWISE", "SETR_6000"),
         ("SETR_NOFILL", "SETR_6000"), ("SETR_NOFILL", "HYBRID_6000"), ("SETR_NOFILL", "LEX5B_6000"),
         ("SETR_6000", "HYBRID_6000")]
cmp_ = {}
for a, b in pairs:
    cmp_["%s vs %s" % (a, b)] = compare(F(a), F(b))
    for g in ("T1", "T2", "T3", "fa", "en"):
        lo, up, _ = diff_vectors(F(a), F(b), GROUPS[g])
        bw = [0, 0]
        for q in GROUPS[g]:
            x, y = F(a)(q), F(b)(q)
            if x[0] == x[1] and y[0] == y[1]:
                bw[0] += x[0] > y[0]
                bw[1] += x[0] < y[0]
        cmp_["%s vs %s" % (a, b)]["net_%s" % g] = [sum(lo), sum(up)]
        cmp_["%s vs %s" % (a, b)]["bw_%s" % g] = bw
res["paired"] = cmp_


# ------------------------------------------------------------------ tokens / calls (S6 ledger) and frozen reference
def s6_calls(arm):
    return [c for c in L.jl(HERE / "ledger.jsonl") if c.get("approach") == arm and c.get("phase") == "query"
            and c.get("question_id") in QS]


tok = {}
for a in S6_ARMS:
    by = defaultdict(lambda: defaultdict(float))
    for c in s6_calls(a):
        # only the calls of the record kept (a re-run cell's failed first call is still counted: it was spent)
        p = c.get("purpose")
        for lg in (QS[c["question_id"]]["lang"], "all"):
            g = by["%s|%s" % (p, lg)]
            g["calls"] += 1
            g["ok"] += bool(c.get("ok"))
            g["prompt"] += c.get("prompt_tokens") or 0
            g["completion"] += c.get("completion_tokens") or 0
            g["reasoning"] += c.get("reasoning_tokens") or 0
            g["cost"] += c.get("cost_usd") or 0
            g["wall_ms"] += c.get("wall_total_ms") or 0
            g["unknown_usage"] += not c.get("usage_known")
            g["length"] += c.get("done_reason") == "length"
    nq = {"fa": sum(1 for q in RT.QIDS if QS[q]["lang"] == "fa"), "en": sum(1 for q in RT.QIDS if QS[q]["lang"] == "en"),
          "all": len(RT.QIDS)}
    tok[a] = {k: dict(v, per_q_prompt=v["prompt"] / nq[k.split("|")[1]], per_q_completion=v["completion"] / nq[k.split("|")[1]],
                      per_q_calls=v["calls"] / nq[k.split("|")[1]]) for k, v in by.items()}
ref = {}
for a in FROZEN:
    by = defaultdict(lambda: defaultdict(float))
    for q in RT.QIDS:
        for c in RT.primary_calls(q, a):
            p = "answer" if c.get("purpose") == "answer" else "selector"
            g = by[p]
            g["calls"] += 1
            g["prompt"] += c.get("prompt_tokens") or 0
            g["completion"] += c.get("completion_tokens") or 0
            g["reasoning"] += c.get("reasoning_tokens") or 0
            g["unknown_usage"] += not c.get("usage_known")
    ref[a] = {k: dict(v, per_q_prompt=v["prompt"] / len(RT.QIDS), per_q_completion=v["completion"] / len(RT.QIDS),
                      per_q_calls=v["calls"] / len(RT.QIDS)) for k, v in by.items()}
res["tokens_s6"] = tok
res["tokens_frozen_ref"] = ref


# ------------------------------------------------------------------ pack descriptives
def packed_chunks(r):
    n = (r.get("trace") or {}).get("evidence_items") or 0
    return [int(u["meta"]["chunk"]) for u in r["retrieved_units"][:n]]


desc = {}
lw = S6REC["LISTWISE"]
rk = [r["trace"]["s6"]["ranker"] for r in lw.values()]
desc["listwise_ranker"] = {"cells": len(rk), "fallback": sum(x["fallback"] for x in rk),
                           "status": dict(defaultdict(int, {s: sum(1 for x in rk if x["status"] == s) for s in {x["status"] for x in rk}})),
                           "full_permutation": sum(1 for x in rk if x.get("n_parsed") == x["shown"]),
                           "passages_cut_mean": sum(x["passages_cut"] for x in rk) / len(rk),
                           "prompt_chars_mean": sum(x["prompt_chars"] for x in rk) / len(rk),
                           "t_rank_ms_median": sorted(x["t_rank_ms"] for x in rk)[len(rk) // 2],
                           "finish_length": sum(1 for x in rk if x.get("finish") == "length")}
jac_hr, jac_hyb, same_hyb = [], [], 0
for q, r in lw.items():
    s = set(packed_chunks(r))
    for arm, lst in (("HR_6000", jac_hr), ("HYBRID_6000", jac_hyb)):
        o = REC[P][arm].get(q)
        if o and not o.get("error"):
            t = set(packed_chunks(o))
            lst.append(len(s & t) / max(1, len(s | t)))
    same_hyb += packed_chunks(r) == packed_chunks(REC[P]["HYBRID_6000"][q])
desc["listwise_pack_jaccard_mean"] = {"vs_HR": sum(jac_hr) / len(jac_hr), "vs_HYBRID": sum(jac_hyb) / len(jac_hyb),
                                      "identical_order_to_HYBRID": same_hyb}
nf = S6REC["SETR_NOFILL"]
desc["setr_nofill"] = {
    "cells": len(nf), "selector_fallback": sum(1 for r in nf.values() if r["trace"]["s6"]["setr_fallback"]),
    "identical_to_setr_pack": sum(1 for r in nf.values() if r["trace"]["s6"]["identical_to_setr_pack"]),
    "n_selected_mean": sum(r["trace"]["s6"]["n_selected"] for r in nf.values() if not r["trace"]["s6"]["setr_fallback"])
    / max(1, sum(1 for r in nf.values() if not r["trace"]["s6"]["setr_fallback"])),
    "n_selected_hist": dict(sorted(defaultdict(int, {k: sum(1 for r in nf.values() if not r["trace"]["s6"]["setr_fallback"]
                                                           and r["trace"]["s6"]["n_selected"] == k) for k in range(0, 21)}).items())),
    "evidence_chars_mean": sum(r["evidence_chars"] for r in nf.values()) / len(nf),
    "evidence_chars_mean_setr": sum(REC[P]["SETR_6000"][q]["evidence_chars"] for q in nf) / len(nf),
    "last_item_cut": sum(1 for r in nf.values() if r["trace"]["last_item_cut"]),
    "abstained_answerable": sum(1 for q, r in nf.items() if r.get("abstained") and q in ANS),
    "abstained_answerable_setr": sum(1 for q in ANS if REC[P]["SETR_6000"][q].get("abstained"))}
desc["abstained_answerable"] = {a: sum(1 for q in ANS if ((S6REC[a].get(q) if a in S6_ARMS else REC[P][a].get(q)) or {}).get("abstained"))
                                for a in S6_ARMS + FROZEN}
desc["errors"] = {a: sum(1 for r in S6REC[a].values() if r.get("error")) for a in S6_ARMS}
desc["answer_length_cutoff"] = {a: sum(1 for r in S6REC[a].values() if r["trace"]["s6"].get("answer_done_reason") == "length")
                                for a in S6_ARMS}
res["descriptives"] = desc

(HERE / "analysis.json").write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
print(json.dumps({k: res[k] for k in ("accuracy", "cell_status")}, indent=None)[:6000])
for k, v in cmp_.items():
    print(k, v["net_bounds"], v["better_worse_resolved"], "p", ["%.3f" % x for x in v["sign_p_range"]],
          "ciq", ["%.1f" % x for x in v["ci_pp_question"]], "cicomp", ["%.1f" % x for x in v["ci_pp_component"]])
print(json.dumps(desc, indent=1)[:4000])
