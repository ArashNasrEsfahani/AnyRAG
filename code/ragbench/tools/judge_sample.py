"""Copy a hand-picked sample of RunRecords into a separate run file, so the paid judge can be exercised on a few
records without touching the full run directory.

    .venv/Scripts/python.exe -m ragbench.tools.judge_sample --runs runs/smoke --out runs/smoke_judge_sample \
        --pick OURS:C-divar-01:fa-fa --pick B1:B-digikala-04:en-en ...
    .venv/Scripts/python.exe -m ragbench.eval.aggregate --runs runs/smoke_judge_sample --questions ... \
        --judge openrouter --judge-max-calls 8 --out runs/smoke/results_judged
"""
import argparse
import sys

from ..config import load_config
from ..eval.aggregate import load_run_records, resolve_path
from ..util import atomic_write_text

import json


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--runs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--pick", action="append", required=True, help="approach:question_id:setting (repeatable)")
    args = ap.parse_args(argv)
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass
    cfg = load_config()
    records, _ = load_run_records([args.runs], cfg)
    by = {(r["approach"], r["question_id"], r["setting"]): r for r in records}
    picked = []
    for p in args.pick:
        key = tuple(p.split(":"))
        if key not in by:
            raise SystemExit("no record %s" % (key,))
        picked.append(by[key])
    out = resolve_path(args.out, cfg)
    out.mkdir(parents=True, exist_ok=True)
    atomic_write_text(out / "runs_judge_sample.jsonl", "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in picked))
    for r in picked:
        print("%-28s %-16s %s | %s" % (r["approach"], r["question_id"], r["setting"], " ".join(r["answer_text"].split())[:90]))
    print("wrote %d records to %s" % (len(picked), out / "runs_judge_sample.jsonl"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
