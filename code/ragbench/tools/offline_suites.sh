#!/usr/bin/env bash
# Every offline test suite of ragbench (no model calls). Prints one summary line per suite; exit 1 on any failure.
#   bash ragbench/tools/offline_suites.sh [log file]
cd "$(dirname "$0")/../.." || exit 1
PY=.venv/Scripts/python.exe
export PYTHONIOENCODING=utf-8
LOG=${1:-work/test_offline_suites.log}
: > "$LOG"
rc=0
for m in ragbench.tests.test_foundation ragbench.tests.test_baselines "ragbench.approaches.tree_nav --offline" \
         ragbench.approaches.agentic.test_agentic ragbench.approaches.agentic2.test_agentic2 ragbench.tests.test_questions \
         ragbench.eval.test_eval ragbench.eval.test_timeline ragbench.tests.test_core_integration \
         ragbench.kb.test_kb ragbench.kb.test_ingest ragbench.kb.test_labels \
         ragbench.kb.test_graph ragbench.kb.test_concepts ragbench.approaches.agentic3.test_agentic3 ragbench.approaches.agentic4.test_agentic4 \
         ragbench.approaches.lex.test_lex ragbench.tests.test_v2_sets "ragbench.tools.freeze_corpus verify"; do
  t0=$(date +%s)
  out=$($PY -m $m 2>&1)
  r=$?
  echo "===== $m (rc=$r, $(( $(date +%s) - t0 ))s)" >> "$LOG"
  echo "$out" >> "$LOG"
  fails=$(echo "$out" | grep -cE "^\s*FAIL|FAILED")
  printf "%-45s rc=%s  %s  (%s FAIL lines)\n" "$m" "$r" "$(echo "$out" | grep -iE "checks|failure|passed|failed" | tail -1 | cut -c1-90)" "$fails"
  [ $r -ne 0 ] && rc=1
done
exit $rc
