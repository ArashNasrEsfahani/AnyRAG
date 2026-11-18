#!/usr/bin/env bash
# The WARM experiment, re-run after the route fix of 2026-09-20 (AGENTIC3_SPEC.md section 8, "index by use").
#
#   bash ragbench/tools/v3_warm.sh            # run it (resumable: finished ids are skipped)
#   bash ragbench/tools/v3_warm.sh --reset    # empty verified_facts and start the curve again
#
# What this is. A COPY of the knowledge base with `verified_facts` emptied, write-back on, the `warm` flag on, and
# the 28 dev records of `work/runs/v3warm/order.tsv` (seed 42) asked ONE PER PROCESS, in that order, so a question
# can only read rows written by the questions before it. Dev files only; the copy is never the database the cold
# runs use, and dev and test never share a database file.
#
# Why it is re-run. In the first warm arm `route.decide` short-circuited to FAST on ANY non-empty `warm_rows`,
# before the pool test and before the fast test, with verify=False - and those rows came back from an OR over the
# plan's expanded terms, so rows written by a DIFFERENT question about the same report answered this one.
# `route.warm_check` now requires the rows to carry every content word of the plan's metric, to number at least
# `warm_min_rows`, to agree on one value, and not to be contradicted by the database's own best candidate. This
# script measures what that is worth: `work/runs/v3warm` (before) against `work/runs/v3warm2` (after), same
# records, same order.
#
# One question per process is the point - a warm hit has to survive a cold start - and it is also why every
# latency here carries the ~0.8 s the first database query of a process costs against a 220 MB file.
set -u
cd "$(dirname "$0")/../.." || exit 1
PY=.venv/Scripts/python.exe
KB=work/kb/kb_warm2.sqlite
OUT=runs/v3warm2
DIR=work/$OUT
ORDER=work/runs/v3warm/order.tsv
CFG=$DIR/config_warm2.yaml

mkdir -p "$DIR"
if [ ! -f "$ORDER" ]; then echo "missing $ORDER (the seeded order of the first warm arm)"; exit 1; fi

if [ "${1:-}" = "--reset" ] || [ ! -f "$KB" ]; then
  rm -f "$DIR"/runs_*.jsonl "$DIR"/ledger_*.jsonl "$DIR"/agentic_calls_*.sqlite "$DIR"/warm_loop.log
  $PY -m ragbench.tools.v3_warm_setup reset --kb "$KB" --config "$CFG" || exit 1
else
  $PY -m ragbench.tools.v3_warm_setup config --kb "$KB" --config "$CFG" || exit 1
fi

export RAGBENCH_CONFIG="$CFG"
touch "$DIR/warm_loop.log"
while IFS=$'\t' read -r n qfile corpus qid setting; do
  [ -z "${qid:-}" ] && continue
  echo "--- $n $qid $setting" | tee -a "$DIR/warm_loop.log"
  $PY -m ragbench.run --approach agentic3 --ablation warm --docs-from-questions --questions "$qfile" \
    --corpus "$corpus" --ids "$qid" --settings "$setting" --out "$OUT" 2>&1 | tail -2 | tee -a "$DIR/warm_loop.log"
done < "$ORDER"

echo "records: $(cat "$DIR"/runs_*.jsonl 2>/dev/null | wc -l)"
$PY -m ragbench.tools.v3_warm_setup count --kb "$KB"
