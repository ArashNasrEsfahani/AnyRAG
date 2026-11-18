"""Data files for every plotted number in the paper's figures (no model calls; saved records only).

Reads
  - the frozen test records and verdicts through report_test.py (load/score: latest successful record, verdict
    matched by (rid, answer hash)), for latency and selector outcomes;
  - test_results.json `cells` (per reader x arm aggregates written by report_test.py from the same records) for
    accuracy bounds and query-LLM token envelopes;
  - test_results_human_sym.json (machine / asymmetric / symmetric totals, agreement_cells, Holm values);
  - paper/analysis/revision_stats.json (paired differences: identification bounds over unresolved judgments and
    95% source-component bootstrap envelopes; run revision_stats.py first);
  - development audit and retention rows (lex8/audit/audit_rows.jsonl, winloss/matrix.csv, lex8/rows8_*.jsonl,
    lex11/rows11.jsonl) when the workspace layout is present.
Writes paper/figures/data/*.dat (semicolon-separated; the figure that reads each file names it in its % src line).

    python paper/analysis/figure_data.py [--root REPO] [--code DIR]
"""
import csv
import json
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import layout  # noqa: E402

LAYOUT = layout.resolve(**vars(layout.parse_args(description="Figure data for the paper")))
TR = LAYOUT["test"]
RT = layout.import_report_test(LAYOUT)
OUT = HERE.parent / "figures" / "data"
OUT.mkdir(parents=True, exist_ok=True)
P = RT.PRIMARY
NANS = len(RT.ANS)              # 90 answerable
NQ = len(RT.QIDS)               # 96
PIPE = {"HYBRID": 1, "HR": 2, "LEX5B": 3, "SETR": 4}   # class ids used by figures/style.tex


def write(name, header, rows):
    with open(OUT / name, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(";".join(header) + "\n")
        for r in rows:
            fh.write(";".join(fmt(x) for x in r) + "\n")
    print("wrote", name, len(rows), "rows")


def fmt(x):
    if isinstance(x, float):
        return "%.4f" % x
    return str(x)


TRES = json.loads((TR / "test_results.json").read_text(encoding="utf-8"))
SYM = json.loads((TR / "test_results_human_sym.json").read_text(encoding="utf-8"))
RS = json.loads((HERE / "revision_stats.json").read_text(encoding="utf-8"))
C = TRES["cells"]

# ------------------------------------------------------------------ 1. paired effects (forest plot)
# y positions top-down; machine panel has the pipeline block and the budget block, symmetric panel the pipeline block.
pipe_pairs = [("HR_6000", "HYBRID_6000"), ("LEX5B_6000", "HYBRID_6000"), ("LEX5B_6000", "HR_6000"),
              ("SETR_6000", "HYBRID_6000"), ("SETR_6000", "HR_6000")]
budget_pairs = [("HYBRID_15500", "HYBRID_6000"), ("HR_15500", "HR_6000"), ("LEX5B_15500", "LEX5B_6000"),
                ("SETR_15500", "SETR_6000"), ("HYBRID_15500", "LEX5B_6000")]
def effect_rows(regime, pairs, y0):
    out, y = [], y0
    for a, b in pairs:
        y -= 1
        v = RS["glm"]["%s %s vs %s" % (regime, a, b)]
        lo, hi = v["pp_bounds"]
        clo, chi = v["ci_pp_component"]
        bw = v["better_worse_resolved"]
        gate = 1 if (a, b) in (("LEX5B_6000", "HYBRID_6000"), ("LEX5B_6000", "HR_6000")) else 0
        out.append([round(y, 2), PIPE[a.split("_")[0]], (lo + hi) / 2, lo, hi, clo, chi, "%d/%d" % tuple(bw), gate,
                    "%s-%s" % (a, b)])
    return out


hdr = ["y", "cls", "mid", "blo", "bhi", "clo", "chi", "wl", "gate", "pair"]
# left panel: pipeline effects at 6,000 (rows -1..-5), then budget effects (rows -6.7..-10.7); right panel: symmetric merge
write("effects_machine.dat", hdr, effect_rows("machine", pipe_pairs, 0) + effect_rows("machine", budget_pairs, -5.7))
write("effects_sym.dat", hdr, effect_rows("human_sym", pipe_pairs, 0))

# ------------------------------------------------------------------ 2. evaluator sensitivity
T = SYM["glm_6000_table"]
rows = []
for a in ("HYBRID_6000", "HR_6000", "LEX5B_6000", "SETR_6000"):
    m = T[a]["machine"]["answerable_90"]
    s1 = T[a]["asym_human_old"]["answerable_90"]
    s2 = T[a]["sym_human_new"]["answerable_90"]
    rows.append([PIPE[a.split("_")[0]], (m[0] + m[1]) / 2, m[0], m[1], s1[0], s2[0], T[a]["cells_human_first"],
                 T[a]["cells_human_sym_total"], a])
write("evaluator_totals.dat", ["cls", "machmid", "machlo", "machhi", "asym", "sym", "revfirst", "revsym", "arm"], rows)
# long format for the trajectory: stage 1 machine, 2 asymmetric review (withdrawn), 3 symmetric merge;
# key = 10*cls + valid (valid = 1 for stages 1 and 3); ly = label position (data units), moved apart so that direct
# labels of nearby points do not overlap (presentation only; the plotted value is y)
traj = []
for r in rows:
    cls, mm, mlo, mhi, asy, sy = r[0], r[1], r[2], r[3], r[4], r[5]
    traj += [[cls, 1, mm, mm - mlo, mhi - mm, 10 * cls + 1, "%d" % mlo if mlo == mhi else "%d--%d" % (mlo, mhi), mm],
             [cls, 2, asy, 0, 0, 10 * cls, "%d" % asy, asy],
             [cls, 3, sy, 0, 0, 10 * cls + 1, "%d" % sy, sy]]


def spread(pts, gap):
    """pts: list of rows (label y in column 7); push labels apart to at least `gap`, keeping their mean."""
    pts.sort(key=lambda t: (t[7], -t[0]))
    for _ in range(50):
        moved = False
        for a_, b_ in zip(pts, pts[1:]):
            if b_[7] - a_[7] < gap - 1e-9:
                d = (gap - (b_[7] - a_[7])) / 2
                a_[7] -= d
                b_[7] += d
                moved = True
        if not moved:
            break


for st in (1, 3):
    spread([t for t in traj if t[1] == st], 1.55)
hdr_t = ["cls", "stage", "y", "eminus", "eplus", "key", "lab", "ly"]
write("evaluator_traj.dat", hdr_t, traj)
write("evaluator_valid.dat", hdr_t, [t for t in traj if t[1] != 2])
ag = Counter()
for c in SYM["agreement_cells"]:
    m, h = c["machine"], c["human"]
    k = "unres" if m is None else ("ww" if (m, h) == (0, 0) else "wc" if (m, h) == (0, 1) else
                                   "cc" if (m, h) == (1, 1) else "cw")
    ag[(c["arm"], k)] += 1
rows = []
for i, a in enumerate(("HYBRID_6000", "HR_6000", "LEX5B_6000", "SETR_6000")):
    cc, wc, ww, cw, un = (ag[(a, k)] for k in ("cc", "wc", "ww", "cw", "unres"))
    rows.append([PIPE[a.split("_")[0]], -i, cc, wc, ww, cw, un, cc + wc + ww + cw + un, wc + cc, a])
write("evaluator_agreement.dat", ["cls", "y", "cc", "wc", "ww", "cw", "unres", "n", "humanacc", "arm"], rows)
tot = {k: sum(ag[(a, k)] for a in ("HYBRID_6000", "HR_6000", "LEX5B_6000", "SETR_6000"))
       for k in ("cc", "wc", "ww", "cw", "unres")}
hp = SYM["pairs"]
# label: Holm p on resolved pairs, and up to the worst case over every assignment of unresolved judge cells when they differ
holm = [(name, hp[name]["LEX5B_6000 vs HYBRID_6000"]["holm_p"],
         hp[name]["LEX5B_6000 vs HYBRID_6000"].get("holm_p_worst", hp[name]["LEX5B_6000 vs HYBRID_6000"]["holm_p"]),
         hp[name]["LEX5B_6000 vs HYBRID_6000"]["net_lower"],
         hp[name]["LEX5B_6000 vs HYBRID_6000"]["net_upper"]) for name in ("machine", "asym_human_old", "sym_human_new")]


def _plab(p, pw):
    a, b = ("%.3f" % p).lstrip("0"), ("%.3f" % pw).lstrip("0")
    return a if a == b else "%s--%s" % (a, b)


write("evaluator_holm.dat", ["x", "stage", "holm_lex_vs_hybrid", "lab", "netlo", "nethi"],
      [[i + 1, n, p, _plab(p, pw), lo, hi] for i, (n, p, pw, lo, hi) in enumerate(holm)])
print("agreement totals", tot)

# ------------------------------------------------------------------ 3. cost-accuracy (six readers)
# x = accounted query-LLM tokens per question (selector + answer calls, calibrated envelope for unknown usage) / 96,
# in thousands; y = correct of 90 (machine judge, bounds over unresolved replies). key = 100*reader + 10*cls + big.
rows = []
for i, rd in enumerate(RT.READERS, 1):
    for a in RT.ARMS:
        c = C["%s|%s" % (rd, a)]
        lo, hi = c["answerable"]
        tl, tu = c["tokens_lower"] / NQ / 1000.0, c["tokens_upper"] / NQ / 1000.0
        cls, big = PIPE[a.split("_")[0]], int(a.endswith("15500"))
        rows.append([i, cls, big, 100 * i + 10 * cls + big, (tl + tu) / 2, tl, tu, (lo + hi) / 2, lo, hi,
                     RT.SHORT[rd], a])
rows.sort(key=lambda r: r[3])
write("cost_accuracy.dat", ["rd", "cls", "big", "key", "tokmid", "toklo", "tokhi", "accmid", "acclo", "acchi", "reader",
                            "arm"], rows)
# highlight for the primary reader: HYBRID at 15,500 vs LEX5B at 6,000 (similar tokens, similar accuracy)
hp_ = [r for r in rows if r[0] == 1 and ((r[1], r[2]) in ((1, 1), (3, 0)))]
write("cost_highlight.dat", ["x", "y"], [[sum(r[4] for r in hp_) / 2, sum(r[7] for r in hp_) / 2]])

# ------------------------------------------------------------------ 4. tiers (primary reader, machine)
rows = []
for a in RT.ARMS:
    c = C["%s|%s" % (P, a)]
    for t in ("T1", "T2", "T3"):
        lo, hi = c[t]
        rows.append([PIPE[a.split("_")[0]], int(a.endswith("15500")), int(t[1]), (lo + hi) / 2, lo, hi, a])
write("tiers.dat", ["cls", "big", "tier", "mid", "lo", "hi", "arm"], rows)

# ------------------------------------------------------------------ 5. latency ECDF + selector outcomes (primary)
REC, VER = RT.load()
for a in RT.ARMS:
    lat = sorted(REC[P][a][q]["latency_ms"] / 1000.0 for q in RT.QIDS
                 if REC[P][a].get(q) is not None and not REC[P][a][q].get("error"))
    pts = [[lat[0], 0.0]] + [[x, (i + 1) / len(lat)] for i, x in enumerate(lat)]   # ECDF steps (log-x safe)
    write("latency_%s.dat" % a, ["s", "F"], pts)
sel = []
BSL = chr(92)
for a in ("LEX5B_6000", "LEX5B_15500", "SETR_6000", "SETR_15500"):
    k = Counter()
    for q in RT.QIDS:
        r = REC[P][a].get(q)
        if r is None or r.get("error"):
            k["error"] += 1
            continue
        t = r.get("trace") or {}
        if "lex5b" in t:
            lx = t["lex5b"]
            s = lx.get("selector") or {}
            shown, fb = lx.get("shown_chunks") or [], lx.get("fallback_chunks") or []
            if s.get("status") == "ok":
                k["ident" if shown == fb else "reord" if set(shown) == set(fb) else "changed"] += 1
            elif s.get("status") == "all_none":
                k["allnone"] += 1
            elif s.get("reason") == "deadline":
                k["deadline"] += 1
            else:
                k["rejected"] += 1
        else:
            s = t["setr"].get("selector") or {}
            chosen = s.get("selected") or []
            if s.get("fallback"):
                k["deadline" if s.get("reason") == "deadline" else "rejected"] += 1
            elif chosen == list(range(1, len(chosen) + 1)):
                k["ident"] += 1
            else:
                k["changed"] += 1
    lab = (BSL + "lexb" if a.startswith("LEX5B") else BSL + "setr") + (", 6,000" if a.endswith("_6000") else ", 15,500")
    sel.append([PIPE[a.split("_")[0]], int(a.endswith("15500"))] +
               [k[x] for x in ("deadline", "rejected", "allnone", "ident", "reord", "changed", "error")] + [lab, a])
write("selector_outcomes.dat", ["cls", "big", "deadline", "rejected", "allnone", "ident", "reord", "changed", "error",
                                "label", "arm"], sel)

# ------------------------------------------------------------------ 6./7. development (workspace layout only)
DS = LAYOUT["root"] / "work" / "datasets_v2"
if not (DS / "lex8" / "audit" / "audit_rows.jsonl").is_file():
    print("development rows not in this layout; skipping audit and development-map data")
    sys.exit(0)


def jl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


aud = jl(DS / "lex8" / "audit" / "audit_rows.jsonl")
names = ["HYBRID", "HR", "a1", "MMR-a1", "O1", "O2"]
BS = chr(92)
AUDLAB = {"HYBRID": BS + "hybrid", "HR": BS + "hr", "a1": "RRF, three channels", "MMR-a1": "same + MMR",
          "O1": "order fixer, first", "O2": "order fixer, second"}
rows = []
for i, a in enumerate(names):
    rows.append([-i, sum(r["arms"][a]["raw40"][1] for r in aud), sum(r["arms"][a]["raw24"][1] for r in aud),
                 sum(r["arms"][a]["shown"][1] for r in aud), len(aud), AUDLAB[a], a])
write("audit.dat", ["y", "top40", "top24", "pack", "n", "label", "arm"], rows)
feas = sum(1 for r in aud if r["union"]["feas"]["feasible"])
vis = {"held": 0, "visible": 0, "key_visible": 0}
for r in aud:
    for k in vis:
        vis[k] += (r["arms"]["O1"].get("visibility") or {}).get(k, 0)
write("audit_union.dat", ["feasible", "n", "held", "visible", "key_visible"],
      [[feas, len(aud), vis["held"], vis["visible"], vis["key_visible"]]])

# development retention map: every saved 54-question, 6,000-character arm, paired with HR on question id
mx = list(csv.DictReader(open(DS / "winloss" / "matrix.csv", encoding="utf-8")))
ret = {}
for col in mx[0]:
    if col.startswith("R54|") and col.endswith("|6000"):
        ret[col.split("|")[1]] = {r["id"]: float(r[col]) for r in mx if r[col] not in ("", None)}
for f in ("rows8_compact.jsonl", "rows8_synth.jsonl"):
    for r in jl(DS / "lex8" / f):
        for arm, v in r["arms"].items():
            ret.setdefault(arm, {})[r["id"]] = v["span_c"]
for r in jl(DS / "lex11" / "rows11.jsonl"):
    for arm in ("F2-2ch@6000", "F2-intact@6000", "S-c core only", "S-c core+20%"):
        ret.setdefault(arm, {})[r["id"]] = r["arms"][arm]["span_c"]
screen = {}
for col in mx[0]:
    if col.startswith("A24|") and col.endswith("|6000"):
        vals = [float(r[col]) for r in mx if r[col] not in ("", None)]
        screen[col.split("|")[1]] = (int(sum(vals)), len(vals))
# group, internal arm, screen key (24-question answer screen, GLM reader) or None
# (group, internal arm, paper label, lex = 1 if the row is a component or ablation of Lex-RAG, screen key or None)
GROUPS = {"req": "Per-requirement retrieval", "list": "Listwise LLM selection",
          "view": r"\lexb{} selector views", "m3": "bge-m3 channels", "fix": "Small LLM order fixer",
          "fus": "Score fusion and index-time units", "lrn": "Learned ranker", "abl": r"\lexb{} pack ablations"}
plan = [
    # HR is the zero line and HYBRID a reference line; HYBRID-intact is omitted because its saved retention row is
    # identical to HYBRID's on every question (an alias, not a separate measurement)
    ("req", "LEX4-A", "rule A", 0, None), ("req", "LEX4-B", "full design", 0, None),
    ("list", "LEX5 glm-full-24", r"GLM, full passages, top 24", 0, None),
    ("list", "LEX5 glm-full-40", r"GLM, full passages, top 40", 0, None),
    ("list", "LEX5 oss20b-full-24", r"gpt-oss-20b, full passages, top 24", 0, None),
    ("view", "LEX5b S-a oss smart", "gpt-oss-20b, smart excerpts", 0, None),
    ("view", "LEX5b S-b glm smart", "GLM, smart excerpts", 0, None),
    ("view", "LEX5b S-d glm full", "GLM, full top 10", 0, None),
    ("view", "LEX5b S-c glm literal", "GLM, literal excerpts (admitted)", 1, "LEX5B(selector)"),
    ("m3", "a0 sparse", "learned sparse alone", 0, None), ("m3", "colbert", "multi-vector alone", 0, None),
    ("m3", "a1 RRF3", "RRF of BM25, dense, sparse", 0, None),
    ("fix", "LEX7 O1 fixer", "first version", 0, None), ("fix", "LEX7.1 O2 fixer", "second version", 0, "LEX7.1(fixer)"),
    ("fus", "F1", "CombSUM fusion", 0, None), ("fus", "F2", "three-signal score fusion", 1, None),
    ("fus", "C1 compact", "compact units, ranked packing", 0, None),
    ("fus", "C2 compact + coverage packer", "compact units, coverage packer", 0, None),
    ("fus", "S1 C2 + synthetic-question field", "+ synthetic-question field", 0, None),
    ("fus", "S1r (post-hoc diagnostic) S1 candidates + ranked packing", "synthetic field, ranked (post hoc)", 0, None),
    ("lrn", "LEX9 learned", "LightGBM on synthetic questions", 0, None),
    ("abl", "F2-2ch@6000", "fusion without the sparse channel", 1, None),
    ("abl", "F2-intact@6000", "fusion order, whole packages", 1, None),
    ("abl", "S-c core only", "selected bundles only", 1, None),
    ("abl", "S-c core+20%", r"selected bundles + 20\% fill", 1, None),
]
hr = ret["HR"]
rows, heads = [], []
y, last = 0.0, None
for g, arm, lab, lex, sk in plan:
    if g != last:
        y -= 1.0 if last is None else 1.25
        heads.append([round(y, 2), GROUPS[g]])
    y -= 1
    last = g
    d = ret[arm]
    qs = [q for q in d if q in hr]
    diff = 100.0 * sum(d[q] - hr[q] for q in qs) / len(qs)
    full = sum(1 for q in qs if d[q] >= 1 - 1e-9)
    sc = screen.get(sk) if sk else None
    rows.append([round(y, 2), diff, full, len(qs), sc[0] if sc and sc[1] == 24 else "", lex,
                 100.0 * sum(d[q] for q in qs) / len(qs), lab, arm.replace(";", ",")])
write("devmap.dat", ["y", "dret", "full", "n", "screen", "lex", "ret", "label", "arm"], rows)
write("devmap_groups.dat", ["y", "label"], heads)
hyb = 100.0 * sum(ret["HYBRID"][q] - hr[q] for q in hr) / len(hr)
write("devmap_ref.dat", ["hr_ret", "hr_full", "hybrid_dret", "hr_screen", "hybrid_screen"],
      [[100.0 * sum(hr.values()) / len(hr), sum(1 for v in hr.values() if v >= 1 - 1e-9), hyb, screen["HR"][0],
        screen["HYBRID"][0]]])
