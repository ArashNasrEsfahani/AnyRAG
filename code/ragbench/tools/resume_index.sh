#!/usr/bin/env bash
# Resume the dev index builds (L3 enrichment etc.) after any stop, with full logging for the paper.
# Safe to stop at any time: every cached section is committed immediately (SQLite WAL),
# so a stop loses at most the one section in flight.
#
#   bash ragbench/tools/resume_index.sh          # check Ollama, then resume in the foreground
#   bash ragbench/tools/resume_index.sh status   # progress only
#
# Logs: work/journal/events.jsonl (run_start / run_stop with duration and exit code),
#       work/telemetry/gpu_<stamp>.csv (GPU utilisation, temperature, power, throttle flags every 30 s),
#       work/runs/dev_full/logs/index_resume_<stamp>.log and steps.tsv (per-step durations),
#       per-call ledgers work/runs/dev_full/ledger_*.jsonl (timestamps, wall time, tokens, load/wait).
set -u
cd "$(dirname "$0")/../.." || exit 1
PY=.venv/Scripts/python.exe
PIN="0.32.15"

if [ "${1:-}" = "status" ]; then
  $PY -m ragbench.tools.dev_progress 2>/dev/null | grep -E "OURS|B4|B3|----|steps done"
  exit 0
fi

v=$(curl -s -m 5 http://127.0.0.1:11434/api/version | sed -E 's/.*"version":"([^"]+)".*/\1/')
if [ -z "$v" ]; then
  echo "Ollama is not running. Start it with: ollama serve   (then re-run this script)"; exit 1
fi
if [ "$v" != "$PIN" ]; then
  $PY -m ragbench.tools.journal add --kind note --name dev_index_refused --note "Ollama $v != pinned $PIN" >/dev/null
  echo "Ollama version is $v, but runs are pinned to $PIN. Not resuming (results would mix versions)."; exit 1
fi

stamp=$(date +%Y%m%d_%H%M%S)
log="work/runs/dev_full/logs/index_resume_$stamp.log"
tele="work/telemetry/gpu_dev_index_$stamp.csv"
start=$(date +%s)
$PY -m ragbench.tools.journal add --kind run_start --name dev_index \
  --data "{\"ollama\": \"$v\", \"log\": \"$log\", \"telemetry\": \"$tele\"}" >/dev/null

$PY -m ragbench.tools.gpu_monitor --out "$tele" --interval 30 &
mon=$!
finish() {
  rc=$1
  kill "$mon" 2>/dev/null
  end=$(date +%s)
  $PY -m ragbench.tools.journal add --kind run_stop --name dev_index --duration-s $((end - start)) \
    --data "{\"exit_code\": $rc, \"log\": \"$log\"}" >/dev/null
}
trap 'finish 130; exit 130' INT TERM

echo "Ollama $v OK. Resuming index builds; log: $log; GPU telemetry: $tele"
bash ragbench/tools/dev_full.sh index 2>&1 | tee "$log"
finish "${PIPESTATUS[0]}"
