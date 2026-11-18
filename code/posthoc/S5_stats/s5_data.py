"""S5 data extraction + reproduction check (no API calls, CPU only).

Loads the frozen test verdicts through report_test.py (via paper/analysis/layout.py, read-only), copies the
revision_stats.py helpers (families -> connected components; sign-p range; cluster bootstrap) WITHOUT importing
revision_stats.py (importing it would rewrite paper/analysis/revision_stats.json), writes cells.json and checks the
loaded data against paper/analysis/revision_stats.json and test_results.json.

    CUDA_VISIBLE_DEVICES= .venv/Scripts/python.exe work/posthoc_2026-11/S5_stats/s5_data.py
"""
import itertools, json, os, random, sys
from collections import defaultdict
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
OUT = Path(__file__).resolve().parent
ROOT = OUT.parent.parent.parent
sys.path.insert(0, str(ROOT / "paper" / "analysis"))
import layout  # noqa
LAY = layout.resolve()
TR = LAY["test"]
RT = layout.import_report_test(LAY)
REC, VER = RT.load()
ANS, QIDS, P = RT.ANS, RT.QIDS, RT.PRIMARY

# components (copied logic from revision_stats.py)
QROWS = {}
for line in open(LAY["questions"], encoding="utf-8-sig"):
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

cells = {}
for rd in RT.READERS:
    for a in RT.ARMS:
        for q in QIDS:
            lo, hi, st = RT.score(REC, VER, rd, a, q)
            cells["%s|%s|%s" % (rd, a, q)] = [lo, hi, st]
SYM = json.loads((TR / "test_results_human_sym.json").read_text(encoding="utf-8"))["per_question_glm_6000_sym"]
data = {"readers": RT.READERS, "arms": RT.ARMS, "primary": P, "qids": QIDS, "ans": ANS,
        "q": {q: {"lang": RT.QS[q]["lang"], "tier": RT.QS[q]["tier"], "answerable": RT.QS[q]["answerable"],
                  "comp": comp.get(q), "block": QROWS[q]["source_block"]} for q in QIDS},
        "cells": cells, "human_sym": SYM}
(OUT / "cells.json").write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

# ------------------------------------------------------------------ reproduction check
import s5_lib as L
D = L.load()
ref = json.loads((ROOT / "paper" / "analysis" / "revision_stats.json").read_text(encoding="utf-8"))
tres = json.loads((TR / "test_results.json").read_text(encoding="utf-8"))
log, bad = [], 0
def chk(name, got, exp):
    global bad
    ok = (got == exp) if not isinstance(exp, float) else abs(got - exp) < 1e-9
    if isinstance(exp, list) and exp and isinstance(exp[0], float):
        ok = all(abs(g - e) < 1e-9 for g, e in zip(got, exp))
    bad += not ok
    log.append("%s %s got=%s exp=%s" % ("OK " if ok else "BAD", name, got, exp))
log.append("components: %d" % len(set(comp.values())))
chk("components", len(set(comp.values())), ref["families"]["components_answerable"])
# main cells: answerable bounds per reader x arm
for rd in RT.READERS:
    for a in RT.ARMS:
        lo = sum(D.m(rd, a, q)[0] for q in ANS); hi = sum(D.m(rd, a, q)[1] for q in ANS)
        chk("cell %s %s answerable" % (rd, a), [lo, hi], tres["cells"]["%s|%s" % (rd, a)]["answerable"])
# primary comparisons
for k, v in ref["glm"].items():
    src, rest = k.split(" ", 1); a, b = rest.split(" vs ")
    fa = (lambda q, a=a: D.m(P, a, q)) if src == "machine" else (lambda q, a=a: D.h(a, q))
    fb = (lambda q, b=b: D.m(P, b, q)) if src == "machine" else (lambda q, b=b: D.h(b, q))
    r = L.sign_range(fa, fb, ANS)
    chk(k + " net_bounds", [r["net_lo"], r["net_hi"]], v["net_bounds"])
    chk(k + " sign_p_range", [r["p_lo"], r["p_hi"]], v["sign_p_range"])
    chk(k + " unresolved", r["k"], v["unresolved_cells"])
    chk(k + " bw", list(L.bw_resolved(fa, fb, ANS)), v["better_worse_resolved"])
    if k in ("machine LEX5B_6000 vs HYBRID_6000", "machine LEX5B_6000 vs HR_6000", "human_sym LEX5B_6000 vs HR_6000"):
        lo, up = L.diff_vectors(fa, fb, ANS)
        cl = [comp[q] for q in ANS]
        a1 = L.boot_ci(lo, cl, 0.95); a2 = L.boot_ci(up, cl, 0.95)
        chk(k + " ci_component", [min(a1[0], a2[0]), max(a1[1], a2[1])], v["ci_pp_component"])
for rd, r in ref["readers"].items():
    for k, v in r.items():
        if k.startswith("interaction"):
            ctrl = k.split()[-1]
            chk(rd + " " + k, list(L.inter_bounds(D, rd, ctrl)), v)
        else:
            a, b = k.split(" vs ")
            lo, up = L.diff_vectors(lambda q: D.m(rd, a, q), lambda q: D.m(rd, b, q), ANS)
            chk(rd + " " + k + " net", [sum(lo), sum(up)], v["net"])
log.append("TOTAL BAD: %d" % bad)
(OUT / "repro_check.txt").write_text("\n".join(log) + "\n", encoding="utf-8")
print("\n".join(l for l in log if not l.startswith("OK")))
print("checks:", len(log), "bad:", bad)
