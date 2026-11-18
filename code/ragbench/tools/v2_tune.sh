#!/usr/bin/env bash
# AGENTIC2 DEV tuning run (never test files): one tag = one option set in its own run directories.
#   bash ragbench/tools/v2_tune.sh <tag> ['<AGENTIC2_OPTIONS json>'] [ablation flags]
# -> work/runs/v2dev_<tag>/ (dev_core.jsonl, A, 4 settings) and work/runs/v2dev_<tag>_tl/ (pilot_timeline.jsonl, C fa-fa),
#    aggregated with --timeline-scoring text; journal run_start/run_stop with the duration.
set -u
cd "$(dirname "$0")/../.." || exit 1
PY=.venv/Scripts/python.exe
export PYTHONIOENCODING=utf-8
TAG=$1
OPTS=${2:-}
ABL=${3:-}
QD=ragbench/data/questions
OUT=runs/v2dev_$TAG
TL=runs/v2dev_${TAG}_tl
mkdir -p work/$OUT/logs
LOG=work/$OUT/logs/tune.log
if [ -n "$OPTS" ]; then export AGENTIC2_OPTIONS="$OPTS"; else unset AGENTIC2_OPTIONS; fi
ABLARG=()
if [ -n "$ABL" ]; then ABLARG=(--ablation "$ABL"); fi
T0=$(date +%s)
$PY -m ragbench.tools.journal add --kind run_start --name "agentic2_tune_$TAG" --data "{\"options\": ${OPTS:-null}, \"ablation\": \"$ABL\"}" >/dev/null
TELE=work/telemetry/gpu_agentic2_tune_${TAG}_$(date +%Y%m%d_%H%M%S).csv
$PY -m ragbench.tools.gpu_monitor --out "$TELE" --interval 30 &
MON=$!
rc=0
for s in fa-fa en-en fa-en en-fa; do
  echo "=== $(date '+%F %T') A $s" >> "$LOG"
  $PY -m ragbench.run --approach agentic2 "${ABLARG[@]}" --docs-from-questions --questions $QD/dev_core.jsonl --corpus A \
    --settings $s --out $OUT >> "$LOG" 2>&1 || rc=$?
done
echo "=== $(date '+%F %T') pilot_timeline" >> "$LOG"
$PY -m ragbench.run --approach agentic2 "${ABLARG[@]}" --docs-from-questions --questions $QD/pilot_timeline.jsonl --corpus C \
  --settings fa-fa --out $TL >> "$LOG" 2>&1 || rc=$?
kill $MON 2>/dev/null
$PY -m ragbench.eval.aggregate --runs $OUT --questions $QD/dev_core.jsonl --split dev --judge none --timeline-scoring text \
  --out $OUT/results >> "$LOG" 2>&1
$PY -m ragbench.eval.aggregate --runs $TL --questions $QD/pilot_timeline.jsonl --split dev --judge none --timeline-scoring text \
  --out $TL/results >> "$LOG" 2>&1
$PY -m ragbench.tools.journal add --kind run_stop --name "agentic2_tune_$TAG" --duration-s $(( $(date +%s) - T0 )) \
  --data "{\"exit_code\": $rc, \"telemetry\": \"$TELE\"}" >/dev/null
echo "done tag=$TAG rc=$rc in $(( $(date +%s) - T0 ))s" >> "$LOG"
exit $rc
