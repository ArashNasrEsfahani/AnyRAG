"""Revision R1 statistics for the paper (no model calls; saved test data only).

Reads the frozen test records/verdicts through work/datasets_v2/test_run/report_test.py (load/score) and the symmetric
human merge (test_results_human_sym.json). Writes paper/analysis/revision_stats.json and revision_stats.md.

Computes:
  1. constituent source families (labels split on '+') and connected components over the 90 answerable test questions;
  2. paired differences (percentage points of 90) with (a) identification bounds over unresolved judge cells,
     (b) exact sign-test p over every assignment of unresolved cells, (c) 95% bootstrap intervals over questions and
     a cluster bootstrap over shared-family components (and source blocks), each as the question-weighted mean of the
     resampled questions; for machine scores the interval is the union over the two extreme assignments;
  3. per-reader nets as bounds (LEX5B vs controls at both budgets, budget effects, LEX-by-budget interaction);
  4. clipping of the last packed item and rendered evidence characters (GLM cells);
  5. query-LLM tokens split by purpose (selector / answer) and language (GLM 6,000 cells).

    python paper/analysis/revision_stats.py [--root REPO] [--code DIR]
Paths are relative to the repository root (workspace or release layout; see layout.py); no absolute paths.
"""
import itertools
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import layout  # noqa: E402

LAYOUT = layout.resolve(**vars(layout.parse_args(description="Revision statistics for the paper")))
TR = LAYOUT["test"]
RT = layout.import_report_test(LAYOUT)

N_BOOT, SEED = 10000, 0
P = RT.PRIMARY
QS, ANS = RT.QS, RT.ANS
REC, VER = RT.load()
ARMS = RT.ARMS
SYM = json.loads((TR / "test_results_human_sym.json").read_text(encoding="utf-8"))
HUM = SYM["per_question_glm_6000_sym"]          # arm -> q -> [lo, hi, status]

# ------------------------------------------------------------------ 1. families and components
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
all_constituents = set().union(*constituents.values())
fam_info = {"labels_answerable": len(set(fam_label.values())),
            "labels_all96": len({QROWS[q]["source_family"] for q in QROWS}),
            "constituents_answerable": len(all_constituents),
            "constituents_all96": len(set().union(*[set(QROWS[q]["source_family"].split("+")) for q in QROWS])),
            "components_answerable": len(set(comp.values())),
            "blocks_answerable": len(set(block.values()))}


# ------------------------------------------------------------------ helpers
def mach(rd, a, q):
    lo, hi, _ = RT.score(REC, VER, rd, a, q)
    return lo, hi


def hum(a, q):
    lo, hi, _ = HUM[a][q]
    return lo, hi


def sign_p(b, w):
    return RT.sign_p(b, w)


def diff_vectors(fa, fb, qs):
    """-> (lower-extreme diffs, upper-extreme diffs, unresolved q list)."""
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
    out = {"net_bounds": [sum(lo), sum(up)], "net_enum": [n_lo, n_hi], "pp_bounds": [100 * sum(lo) / len(qs), 100 * sum(up) / len(qs)],
           "unresolved_cells": k, "sign_p_range": [p_lo, p_hi]}
    for name, cl in (("question", None), ("component", [comp[q] for q in qs]), ("block", [block[q] for q in qs])):
        a1 = boot_mean(lo, cl)
        a2 = boot_mean(up, cl)
        out["ci_pp_" + name] = [min(a1[0], a2[0]), max(a1[1], a2[1])]
    # better / worse counts at resolved pairs
    b = w = 0
    for q in qs:
        al, ah = fa(q)
        bl, bh = fb(q)
        if al == ah and bl == bh:
            b += al > bl
            w += al < bl
    out["better_worse_resolved"] = [b, w]
    return out


def M(rd, a):
    return lambda q: mach(rd, a, q)


def H(a):
    return lambda q: hum(a, q)


res = {"families": fam_info}

# ------------------------------------------------------------------ 2. primary comparisons
prim = {}
pairs = [("LEX5B_6000", "HYBRID_6000"), ("LEX5B_6000", "HR_6000"), ("SETR_6000", "LEX5B_6000"),
         ("HR_6000", "HYBRID_6000"), ("SETR_6000", "HYBRID_6000"), ("SETR_6000", "HR_6000"),
         ("LEX5B_15500", "HYBRID_15500"), ("LEX5B_15500", "HR_15500"), ("SETR_15500", "LEX5B_15500"),
         ("HYBRID_15500", "HYBRID_6000"), ("HR_15500", "HR_6000"), ("LEX5B_15500", "LEX5B_6000"),
         ("SETR_15500", "SETR_6000"), ("HYBRID_15500", "LEX5B_6000")]
for a, b in pairs:
    prim["machine %s vs %s" % (a, b)] = compare(M(P, a), M(P, b))
for a, b in pairs[:6]:
    prim["human_sym %s vs %s" % (a, b)] = compare(H(a), H(b))
res["glm"] = prim

# ------------------------------------------------------------------ 3. per-reader bounds
def inter(rd, ctrl):
    """(L6-C6)-(L15-C15) bounds."""
    lo = hi = 0
    for q in ANS:
        l6, c6 = mach(rd, "LEX5B_6000", q), mach(rd, ctrl + "_6000", q)
        l15, c15 = mach(rd, "LEX5B_15500", q), mach(rd, ctrl + "_15500", q)
        lo += l6[0] - c6[1] - l15[1] + c15[0]
        hi += l6[1] - c6[0] - l15[0] + c15[1]
    return [lo, hi]


readers = {}
for rd in RT.READERS:
    r = {}
    for a, b in [("LEX5B_6000", "HYBRID_6000"), ("LEX5B_6000", "HR_6000"), ("LEX5B_15500", "HYBRID_15500"),
                 ("LEX5B_15500", "HR_15500"), ("HYBRID_15500", "HYBRID_6000"), ("HR_15500", "HR_6000"),
                 ("LEX5B_15500", "LEX5B_6000"), ("SETR_6000", "LEX5B_6000"), ("SETR_6000", "HYBRID_6000")]:
        lo, up, _ = diff_vectors(M(rd, a), M(rd, b), ANS)
        bw = compare(M(rd, a), M(rd, b))["better_worse_resolved"] if False else None
        b_ = w_ = 0
        for q in ANS:
            x, y = mach(rd, a, q), mach(rd, b, q)
            if x[0] == x[1] and y[0] == y[1]:
                b_ += x[0] > y[0]
                w_ += x[0] < y[0]
        r["%s vs %s" % (a, b)] = {"net": [sum(lo), sum(up)], "bw": [b_, w_]}
    r["interaction vs HYBRID"] = inter(rd, "HYBRID")
    r["interaction vs HR"] = inter(rd, "HR")
    readers[RT.SHORT.get(rd.split("/", 1)[-1] if "/" in rd else rd, rd)] = r
res["readers"] = readers

# ------------------------------------------------------------------ 4. clipping and evidence chars (GLM)
clip = {}
for a in ARMS:
    n = cut = 0
    chars = []
    for q in RT.QIDS:
        r = REC[P][a].get(q)
        if r is None or r.get("error"):
            continue
        n += 1
        items = r.get("evidence_items") or []
        cut += any(it.get("cut") for it in items) or bool((r.get("trace") or {}).get("last_item_cut"))
        chars.append(r.get("evidence_chars") or 0)
    chars.sort()
    clip[a] = {"cells": n, "clipped": cut, "chars_mean": sum(chars) / len(chars), "chars_median": chars[len(chars) // 2]}
res["clipping"] = clip

# ------------------------------------------------------------------ 5. tokens by purpose and language (GLM)
tok = {}
for a in ARMS:
    agg = defaultdict(lambda: {"lower": 0, "upper": 0, "unknown": 0, "q": set(), "prompt": 0, "completion": 0, "reasoning": 0})
    for q in RT.QIDS:
        lang = QS[q]["lang"]
        for c in RT.primary_calls(q, a):
            purpose = "answer" if c.get("purpose") == "answer" else "selector"
            b = RT.TB.question_bounds([c], RT.CAL)
            for key in ((purpose, lang), (purpose, "all"), ("total", lang), ("total", "all")):
                g = agg[key]
                g["lower"] += b["lower"]
                g["upper"] += b["upper"]
                g["unknown"] += b["unknown"]
                g["q"].add(q)
                if c.get("usage_known"):
                    g["prompt"] += c.get("prompt_tokens") or 0
                    g["completion"] += c.get("completion_tokens") or 0
                    g["reasoning"] += c.get("reasoning_tokens") or 0
    nq = {"fa": sum(1 for q in RT.QIDS if QS[q]["lang"] == "fa"), "en": sum(1 for q in RT.QIDS if QS[q]["lang"] == "en"),
          "all": len(RT.QIDS)}
    tok[a] = {"%s|%s" % k: {"per_q_lower": v["lower"] / nq[k[1]], "per_q_upper": v["upper"] / nq[k[1]],
                             "unknown_calls": v["unknown"], "known_prompt": v["prompt"], "known_completion": v["completion"],
                             "known_reasoning": v["reasoning"]} for k, v in agg.items()}
res["tokens"] = tok

out = HERE / "revision_stats.json"
out.write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")

# ------------------------------------------------------------------ markdown summary
L = ["# Revision R1 statistics (generated by analysis/revision_stats.py; saved data only)", "",
     "Families: %s" % json.dumps(fam_info), "",
     "## GLM-5.3-flash paired comparisons (/90; pp = percentage points; CIs 95%%, %d resamples, seed %d)" % (N_BOOT, SEED), "",
     "| comparison | net bounds | pp bounds | b/w (resolved) | unresolved cells | sign p range | CI q (pp) | CI comp (pp) | CI block (pp) |",
     "|---|---|---|---|---|---|---|---|---|"]
for k, v in prim.items():
    L.append("| %s | %+d..%+d | %+.1f..%+.1f | %d/%d | %d | %.3f-%.3f | [%.1f, %.1f] | [%.1f, %.1f] | [%.1f, %.1f] |" % (
        k, v["net_bounds"][0], v["net_bounds"][1], v["pp_bounds"][0], v["pp_bounds"][1], *v["better_worse_resolved"],
        v["unresolved_cells"], *v["sign_p_range"], *v["ci_pp_question"], *v["ci_pp_component"], *v["ci_pp_block"]))
L += ["", "## Per-reader net bounds (machine)", ""]
for rd, r in readers.items():
    L.append("- **%s**: " % rd + "; ".join("%s %s (%s)" % (k, "%+d..%+d" % tuple(v["net"]) if isinstance(v, dict) else "%+d..%+d" % tuple(v),
                                                           "%d/%d" % tuple(v["bw"]) if isinstance(v, dict) else "") for k, v in r.items()))
L += ["", "## Clipping (GLM cells)", ""] + ["- %s: %s" % (a, json.dumps(v)) for a, v in clip.items()]
L += ["", "## Tokens per question (GLM; bounds; by purpose|language)", ""]
for a, v in tok.items():
    L.append("- %s: " % a + "; ".join("%s %.0f-%.0f (unk %d)" % (k, x["per_q_lower"], x["per_q_upper"], x["unknown_calls"]) for k, x in sorted(v.items())))
(HERE / "revision_stats.md").write_text("\n".join(L) + "\n", encoding="utf-8")
print("\n".join(L))
