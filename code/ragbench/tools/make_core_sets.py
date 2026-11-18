"""Build the CORE PLAN question files (deterministic, stratified subsets of the verified sets).

Outputs (ragbench/data/questions/):
  test_core.jsonl   split=test  A 4 reports x 10 questions (stratified), B Digikala 1401, C Divar 1400 / Snapp 1402 / Shaparak 1401
  dev_core.jsonl    split=dev   pilot questions on rpt-b and rpt-f (mini tuning set)
  dev_core_timeline.jsonl  split=dev  2 dev timeline questions (agentic tuning)
  core_sets.md      what was selected and why

Usage: python -m ragbench.tools.make_core_sets
"""
import collections
import json
from pathlib import Path

QDIR = Path(__file__).resolve().parents[1] / "data" / "questions"

A_REPORTS = ["rpt-b", "rpt-f", "rpt-c", "rpt-e"]
B_KEYS = {"digikala-1401"}
C_KEYS = {"divar_annual-report_1400", "snapp_annual-report_1402", "shaparak_annual-economic-report_1401"}
DEV_A = {"rpt-b", "rpt-f"}
DEV_TIMELINE_IDS = ["C-TL-divar-02", "C-TL-cafebazaar-01"]
A_QUOTA = [("lookup", 4), ("locate", 2), ("multi-section", 3), ("unanswerable", 1)]  # 10 per report


def key(doc_id):
    return doc_id.split(".", 1)[1].rsplit(".", 1)[0] if doc_id.count(".") >= 2 else doc_id


def keys(q):
    return {key(d) for d in (q.get("doc_scope") or [])}


def load(name):
    with open(QDIR / name, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write(name, rows):
    with open(QDIR / name, "w", encoding="utf-8", newline="\n") as f:
        for q in rows:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")


def main():
    test = load("test.jsonl")
    notes = []

    # ---- A: 10 per report, stratified; cross-report questions allowed only if both reports are core
    a_core = []
    for r in A_REPORTS:
        pool = [q for q in test if q["corpus"] == "A" and q["id"].startswith("T-A-%s-" % r) and keys(q) <= set(A_REPORTS)]
        pool.sort(key=lambda q: q["id"])
        chosen = []
        for typ, n in A_QUOTA:
            chosen += [q for q in pool if q["type"] == typ and q not in chosen][:n]
        for q in pool:  # top up in id order if a type bucket was short
            if len(chosen) >= 10:
                break
            if q not in chosen:
                chosen.append(q)
        a_core += chosen[:10]
        notes.append("A %s: %d of %d eligible (%s)" % (r, len(chosen[:10]), len(pool),
                     dict(collections.Counter(q["type"] for q in chosen[:10]))))

    b_core = sorted([q for q in test if q["corpus"] == "B" and keys(q) <= B_KEYS], key=lambda q: q["id"])
    c_core = sorted([q for q in test if q["corpus"] == "C" and keys(q) <= C_KEYS], key=lambda q: q["id"])
    notes.append("B Digikala 1401 (single-report questions): %d" % len(b_core))
    notes.append("C Divar 1400 / Snapp 1402 / Shaparak 1401 (single-report questions): %d" % len(c_core))

    core = []
    for q in a_core + b_core + c_core:
        q = json.loads(json.dumps(q))
        q.setdefault("meta", {})["core_plan"] = True
        core.append(q)
    write("test_core.jsonl", core)

    pilot = load("pilot.jsonl")
    dev_core = [q for q in pilot if q["corpus"] == "A" and keys(q) <= DEV_A]
    write("dev_core.jsonl", dev_core)
    notes.append("dev_core (pilot A on rpt-b, rpt-f): %d" % len(dev_core))

    tl_path = QDIR / "pilot_timeline.jsonl"
    if tl_path.exists():
        tl = {q["id"]: q for q in load("pilot_timeline.jsonl")}
        dev_tl = [tl[i] for i in DEV_TIMELINE_IDS if i in tl]
        write("dev_core_timeline.jsonl", dev_tl)
        notes.append("dev_core_timeline: %d (%s)" % (len(dev_tl), ", ".join(q["id"] for q in dev_tl)))

    by = collections.Counter((q["corpus"], q["type"]) for q in core)
    lines = ["# Core plan question sets", "", "Built by `python -m ragbench.tools.make_core_sets` (deterministic).", "",
             "Test core: %d questions (A %d, B %d, C %d). Timeline test questions are added separately." % (
                 len(core), len(a_core), len(b_core), len(c_core)), ""]
    lines += ["- " + n for n in notes]
    lines += ["", "| Corpus | Type | Count |", "|---|---|---|"]
    lines += ["| %s | %s | %d |" % (c, t, n) for (c, t), n in sorted(by.items())]
    (QDIR / "core_sets.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
