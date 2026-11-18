"""Export documents for question authoring: a table of contents and every unit, as plain TSV the authors can Grep.

    .venv/Scripts/python.exe -m ragbench.tools.export_units <set> <doc_id> [<doc_id> ...]
    .venv/Scripts/python.exe -m ragbench.tools.export_units A --untouched A.rpt-a.fa ...   (sections no question used)

-> work/final/authoring/docs/<doc_id>/toc.tsv  (section_id, depth, title, own units, chars)
   work/final/authoring/docs/<doc_id>/units.tsv (unit_id, section_id, kind, text with tabs/newlines as " | ")
With --untouched (corpus A), sections holding a gold unit of ANY existing question file are left out, so the authors
can only write questions on text no development question used (the "unseen sections" cell, reported separately).
"""
import argparse
import glob
import json
import sys
from pathlib import Path

from ..config import load_config
from ..corpus.registry import load_set

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "work" / "final" / "authoring" / "docs"


def used_units():
    gold = set()
    for f in glob.glob(str(ROOT / "ragbench" / "data" / "questions" / "*.jsonl")):
        if "test_fresh" in f:
            continue
        for line in open(f, encoding="utf-8"):
            q = json.loads(line)
            for us in (q.get("gold_units") or {}).values():
                gold.update(us)
    return gold


def flat(t):
    return " ".join((t or "").replace("\t", " ").replace("\r", " ").replace("\n", " | ").split())


def main(argv=None):
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("set")
    ap.add_argument("docs", nargs="+")
    ap.add_argument("--untouched", action="store_true")
    a = ap.parse_args(argv)
    cs = load_set(a.set, load_config())
    gold = used_units() if a.untouched else set()
    for did in a.docs:
        d = cs.doc(did)
        skip = set()
        if a.untouched:
            # the section (and its paired-language twin) of every unit a question already used
            for u in d.units:
                if u.id in gold or u.id.replace(".fa::", ".en::") in gold or u.id.replace(".en::", ".fa::") in gold:
                    skip.add(u.section_id)
        o = OUT / did
        o.mkdir(parents=True, exist_ok=True)
        n_units = 0
        with open(o / "toc.tsv", "w", encoding="utf-8") as fh:
            fh.write("section_id\tdepth\ttitle\tunits\tchars\n")
            for s in d.sections:
                if s.id in skip:
                    continue
                us = [d.unit(u) for u in s.unit_ids]
                fh.write("%s\t%d\t%s\t%d\t%d\n" % (s.id, s.level, flat(s.title), len(us), sum(len(u.text) for u in us)))
        with open(o / "units.tsv", "w", encoding="utf-8") as fh:
            fh.write("unit_id\tsection_id\tkind\ttext\n")
            for u in d.units:
                if u.section_id in skip or not (u.text or "").strip():
                    continue
                fh.write("%s\t%s\t%s\t%s\n" % (u.id, u.section_id, u.kind, flat(u.text)))
                n_units += 1
        print("%-50s %5d units, %4d sections%s" % (did, n_units, len(d.sections) - len(skip),
                                                  " (%d used sections left out)" % len(skip) if skip else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
