"""Completeness guard for the multi-document experiment (2026-09-24, P0 list): no analysis before every expected record
and every expected verdict exists. multidoc_analysis.py calls it and refuses to run on an incomplete grid.

    .venv/Scripts/python.exe -m ragbench.tools.check_grid --design work/multidoc/design.json --runs <dir> [--runs ...]
        --verdicts <verdicts_<judge>.jsonl> [--failures <judge_failures_<judge>.jsonl>]

The design is a FROZEN JSON (schema multidoc-design-v1, design_sha1 over its body; freeze_design() writes it):
  experiment_id, questions (path, relative to the project root or absolute), methods [variant names],
  readers [chat model names as in RunRecord.models.chat], replicates [ints], settings {corpus: [settings]},
  variants_version, judge, primary_contrasts [[id, method 1, method 2, metric]], secondary_contrasts [...],
  bootstrap {n, seed}.
Expected grid: every question x each of its pool ids x method x reader x replicate x declared setting of its corpus
that the question supports. A cell is complete when its latest identity record (meta.identity with the design's
experiment id and variants_version) has no error and a verdict of the design's judge exists for that record's rid, bound
to the record's current answer_sha1 and the question's current rubric_sha1. Gaps are listed by kind: missing record,
errored record, conflicting records (two rids for one cell), missing verdict, judge failure (never judged successfully).
"""
import argparse
import glob
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

from ..contracts import Question
from . import judge_multi as JM

DESIGN_SCHEMA = "multidoc-design-v1"


def design_sha1(design):
    body = {k: v for k, v in design.items() if k != "design_sha1"}
    return hashlib.sha1(json.dumps(body, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def freeze_design(design, path):
    d = dict(design, schema=DESIGN_SCHEMA)
    d["design_sha1"] = design_sha1(d)
    p = Path(path)
    if p.exists():
        old = json.loads(p.read_text(encoding="utf-8"))
        if old.get("design_sha1") != d["design_sha1"]:
            raise ValueError("design %s is frozen with another content" % p)
        return p
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(d, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return p


def load_design(path):
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    if d.get("schema") != DESIGN_SCHEMA:
        raise ValueError("%s: schema %r, expected %s" % (path, d.get("schema"), DESIGN_SCHEMA))
    if d.get("design_sha1") != design_sha1(d):
        raise ValueError("%s: design_sha1 does not match its content (edited after freezing?)" % path)
    return d


def expected_cells(design, questions):
    """-> [cell] with cell = (question_id, pool_id, method, reader, replicate, setting)."""
    cells = []
    for qid, qd in sorted(questions.items()):
        q = Question.from_dict(qd)
        sets = (design.get("settings") or {}).get(q.corpus) or []
        for pid in q.pool_ids():
            for m in design["methods"]:
                for rd in design["readers"]:
                    for rep in design.get("replicates") or [0]:
                        for s in sets:
                            if q.supports(s):
                                cells.append((qid, pid, m, rd, int(rep), s))
    return cells


def cell_of(rec):
    i = (rec.get("meta") or {}).get("identity") or {}
    return (i.get("question_id"), i.get("pool_id"), i.get("method"), (i.get("reader") or {}).get("chat"),
            int(i.get("replicate") or 0), i.get("setting"))


def read_records(run_dirs):
    """All identity record lines in file order (runs_*.jsonl of each dir)."""
    out = []
    for d in run_dirs:
        for f in sorted(glob.glob(str(Path(d) / "runs_*.jsonl"))):
            for line in open(f, encoding="utf-8"):
                if line.strip():
                    r = json.loads(line)
                    if ((r.get("meta") or {}).get("identity") or {}).get("rid"):
                        out.append(r)
    return out


def records_by_cell(design, records):
    """{cell: {rid: latest record}} for the design's experiment and variants version."""
    by = defaultdict(dict)
    for r in records:
        i = r["meta"]["identity"]
        if i.get("experiment_id") != design["experiment_id"]:
            continue
        vv = (i.get("options") or {}).get("variants_version")
        if design.get("variants_version") and vv != design["variants_version"]:
            continue
        by[cell_of(r)][i["rid"]] = r
    return by


def expected_rubric(qd, setting):
    return JM.question_rubric(qd, setting[:2])[2]


def grid_gaps(design, questions, records, verdicts, failures=()):
    """-> report dict; report["complete"] is True only with no gap of any kind. Also returns report["cells"]
    {cell: (record, verdict)} for the complete cells."""
    by_cell = records_by_cell(design, records)
    judge = design.get("judge")
    vkey = {}
    for v in verdicts:
        if judge and v.get("judge") != judge:
            continue
        vkey[(v.get("rid"), v.get("answer_sha1"), v.get("rubric_sha1"))] = v     # latest wins
    fkey = {(f.get("rid"), f.get("answer_sha1"), f.get("rubric_sha1")) for f in failures
            if not judge or f.get("judge") == judge}
    rep = {"expected": 0, "missing_record": [], "error_record": [], "conflicting_records": [], "missing_verdict": [],
           "judge_failure": [], "cells": {}}
    for cell in expected_cells(design, questions):
        rep["expected"] += 1
        recs = by_cell.get(cell) or {}
        if not recs:
            rep["missing_record"].append(cell)
            continue
        if len(recs) > 1:
            rep["conflicting_records"].append((cell, sorted(recs)))
            continue
        r = next(iter(recs.values()))
        if r.get("error"):
            rep["error_record"].append((cell, str(r.get("error"))[:120]))
            continue
        k = (r["meta"]["identity"]["rid"], JM.answer_sha1(r), expected_rubric(questions[cell[0]], cell[5]))
        v = vkey.get(k)
        if v is None:
            (rep["judge_failure"] if k in fkey else rep["missing_verdict"]).append(cell)
            continue
        rep["cells"][cell] = (r, v)
    gaps = sum(len(rep[k]) for k in ("missing_record", "error_record", "conflicting_records", "missing_verdict",
                                     "judge_failure"))
    rep["n_gaps"] = gaps
    rep["complete"] = gaps == 0 and rep["expected"] > 0
    return rep


def summary_lines(rep, limit=20):
    L = ["expected cells %d, complete %d, gaps %d" % (rep["expected"], len(rep["cells"]), rep["n_gaps"])]
    for k in ("missing_record", "error_record", "conflicting_records", "missing_verdict", "judge_failure"):
        if rep[k]:
            L.append("%s: %d, e.g. %s" % (k, len(rep[k]), rep[k][:limit]))
    return L


def resolve(path, root):
    p = Path(path)
    return p if p.is_absolute() else Path(root) / p


def load_inputs(design_path, run_dirs, verdicts_path, failures_path=None, root=None):
    from ..config import load_config
    root = root or load_config().project_root
    design = load_design(design_path)
    qs = JM.load_question_dicts(resolve(design["questions"], root))
    if design.get("question_ids"):
        qs = {k: v for k, v in qs.items() if k in set(design["question_ids"])}
    recs = read_records([resolve(d, root) for d in run_dirs])
    verdicts = JM.read_jsonl(verdicts_path)
    fails = JM.read_jsonl(failures_path) if failures_path else []
    return design, qs, recs, verdicts, fails


def main(argv=None):
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--design", required=True)
    ap.add_argument("--runs", action="append", required=True)
    ap.add_argument("--verdicts", required=True)
    ap.add_argument("--failures", default=None)
    a = ap.parse_args(argv)
    design, qs, recs, verdicts, fails = load_inputs(a.design, a.runs, a.verdicts, a.failures)
    rep = grid_gaps(design, qs, recs, verdicts, fails)
    print("\n".join(summary_lines(rep)))
    print("COMPLETE" if rep["complete"] else "INCOMPLETE: analysis refused until every gap is closed")
    return 0 if rep["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
