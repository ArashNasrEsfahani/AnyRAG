"""Data files for the post-hoc figures (saved results only; no model calls).

Reads work/posthoc_2026-11/{S1_evaluator,S2a_packs,S2b_variance,S7_analysis}/*.json and writes
paper/figures/data/{lost_arms,budget,stability}.dat (semicolon separated, `% src:` comment lines first).
    python paper/analysis/figure_data_posthoc.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PH = ROOT / "work" / "posthoc_2026-11"
OUT = ROOT / "paper" / "figures" / "data"


def load(rel):
    return json.loads((PH / rel).read_text(encoding="utf-8"))


def write(name, src, header, rows):
    with open(OUT / name, "w", encoding="utf-8", newline="\n") as fh:
        for s in src:
            fh.write("% src: " + s + "\n")
        fh.write(";".join(header) + "\n")
        for r in rows:
            fh.write(";".join(str(x) for x in r) + "\n")
    print("wrote", name, len(rows))


# ---------------------------------------------------------------- where the answers are lost (S7 reach.json)
reach = load("S7_analysis/reach.json")["arms"]
order = [("HYBRID_6000", "Hybrid", 1), ("HR_6000", "HR", 2), ("LEX5B_6000", "Lex-RAG", 3), ("SETR_6000", "SetR", 4),
         ("FUSION_6000", "Lex-RAG without selection", 5), ("HYBRID_15500", "Hybrid, 15,500", 1),
         ("HR_15500", "HR, 15,500", 2), ("ITER_BM25", "BM25 search agent", 6), ("GREP_AGENT", "Grep agent", 7)]
rows = []
for i, (k, lab, cls) in enumerate(order):
    a = reach[k]
    rows.append([i, cls, lab, a["correct_complete"], a["correct_partial"], a["correct_none"], a["unresolved"],
                 a["miss_none"], a["miss_partial"], a["miss_complete"], a["reach_complete"]])
write("lost_arms.dat",
      ["work/posthoc_2026-11/S7_analysis/reach.json arms (GLM-5.3-flash, 90 answerable, machine strict verdicts; "
       "bundle complete/partial/none = how much of the gold evidence the pack held)"],
      ["i", "cls", "lab", "hc", "hp", "hn", "un", "mn", "mp", "mc", "rc"], rows)

# ---------------------------------------------------------------- budget curves (S2a results.json table)
tab = {r["arm"]: r for r in load("S2a_packs/results.json")["table"]}
rows = []
for pipe, cls in (("HYBRID", 1), ("HR", 2), ("FUSION", 5), ("LEX5B", 3)):
    for b in (3000, 6000, 10000, 15500, 24000):
        r = tab.get("%s_%d" % (pipe, b))
        if r is None:
            continue
        lo, hi = r["acc90"]
        rows.append([pipe, cls, b, "%.2f" % ((lo + hi) / 2), lo, hi, "%.3f" % (r["query_tokens_all_mean"] / 1000.0),
                     r["query_calls_mean"]])
# Lex-RAG at 3,000 / 10,000 / 24,000 (S8 step 1: the saved 6,000 selector reply re-packed at each budget). Token position = the same budget's
# Lex-RAG-without-selection answer call (S2a FUSION_B) plus the selection call, whose cost is the 6,000-character difference LEX5B_6000 - FUSION_6000.
s8 = load("S8_factorial/results.json")["acc_j1"]
sel_tok = tab["LEX5B_6000"]["query_tokens_all_mean"] - tab["FUSION_6000"]["query_tokens_all_mean"]
for b in (3000, 10000, 24000):
    lo, hi = s8["LEX5B_%d" % b]["all"]
    rows.append(["LEX5B", 3, b, "%.2f" % ((lo + hi) / 2), lo, hi,
                 "%.3f" % ((tab["FUSION_%d" % b]["query_tokens_all_mean"] + sel_tok) / 1000.0), 2.0])
rows.sort(key=lambda r: ({"HYBRID": 0, "HR": 1, "FUSION": 2, "LEX5B": 3}[r[0]], r[2]))
write("budget.dat",
      ["work/posthoc_2026-11/S2a_packs/results.json table: acc90 = bounds over unresolved judge cells (of 90); "
       "tokens = mean query-time LLM tokens per question (all calls, thousands). Lex-RAG at 6,000 and "
       "15,500 characters are frozen cells; Lex-RAG at 3,000, 10,000, 24,000 from work/posthoc_2026-11/S8_factorial/results.json acc_j1, token position approximate (see script)"],
      ["pipe", "cls", "budget", "mid", "lo", "hi", "tok", "calls"], rows)

# ---------------------------------------------------------------- stability (S2b exp1, S1 judge2)
e1 = load("S2b_variance/results.json")["exp1"]["cells"]
j2 = load("S1_evaluator/judge2_results.json")
rk = j2["ranking_6000_ans90"]
arms = [("HYBRID_6000", 1), ("HR_6000", 2), ("LEX5B_6000", 3), ("SETR_6000", 4)]
cols = [(1, "acc_by_rep", "0"), (2, "acc_by_rep", "1"), (3, "acc_by_rep", "2"),
        (4.6, "judge2", None), (6.2, "sym_human_j1", None), (7.2, "sym_human_j2", None)]
rows = []
for k, cls in arms:
    for x, src, rep in cols:
        lo, hi = (e1[k]["acc_by_rep"][rep] if rep is not None else rk[src][k])
        rows.append([cls, x, lo, hi, "%.2f" % ((lo + hi) / 2), lo if lo == hi else "%d--%d" % (lo, hi)])
write("stability.dat",
      ["work/posthoc_2026-11/S2b_variance/results.json exp1.cells[arm].acc_by_rep (reader draws 0 = saved, 1, 2; judge 1)",
       "work/posthoc_2026-11/S1_evaluator/judge2_results.json ranking_6000_ans90 (judge2; sym_human_j1, sym_human_j2 = "
       "30 questions reviewed by the human, judge 1 or judge 2 on the other 60)"],
      ["cls", "x", "lo", "hi", "mid", "lab"], rows)
