#!/usr/bin/env bash
# Full DEV evaluation (pilot.jsonl, --docs-from-questions). Sequential: one GPU. Resumable.
#   bash ragbench/tools/dev_full.sh index   # index builds only (cost logged separately)
#   bash ragbench/tools/dev_full.sh query   # every variant x corpus (A, B all settings; C fa-fa)
#   bash ragbench/tools/dev_full.sh aggregate
#   bash ragbench/tools/dev_full.sh "<phase> <phase>" [step filter regex]
# Logs: work/runs/dev_full/logs/<step>.log ; exit codes + seconds in logs/steps.tsv
cd "D:/KareOne/RAG Paper" || exit 1
export PYTHONIOENCODING=utf-8
PY=.venv/Scripts/python.exe
Q=ragbench/data/questions/pilot.jsonl
OUT=runs/dev_full
LOGS="work/${OUT}/logs"
mkdir -p "$LOGS"
PHASES=${1:-"index query aggregate"}
ONLY=${2:-.}
ALL4="fa-fa,en-en,fa-en,en-fa"

step() {   # step <name> <args...>
  local name=$1; shift
  if ! echo "$name" | grep -Eq "$ONLY"; then return 0; fi
  local t0=$(date +%s)
  echo "=== $(date '+%F %T') $name: $*" | tee -a "$LOGS/steps.log"
  $PY -m ragbench.run --questions "$Q" --out "$OUT" --docs-from-questions "$@" > "$LOGS/$name.log" 2>&1
  local rc=$?
  local t1=$(date +%s)
  printf "%s\t%d\t%ds\t%s\n" "$name" "$rc" "$((t1 - t0))" "$(date '+%F %T')" | tee -a "$LOGS/steps.tsv"
  tail -3 "$LOGS/$name.log" | tee -a "$LOGS/steps.log"
}

settings_for() { if [ "$1" = C ]; then echo fa-fa; else echo "$ALL4"; fi; }

for phase in $PHASES; do
  case $phase in
    index)
      for ap in b2 b3 b5 b4 ours; do
        for c in A B C; do step index_${ap}_$c --approach $ap --corpus $c --settings "$(settings_for $c)" --index-only; done
      done
      for ab in no_enrichment no_facts facts_bare; do
        for c in A B C; do step index_ours_${ab}_$c --approach ours --ablation $ab --corpus $c --settings "$(settings_for $c)" --index-only; done
      done
      ;;
    query)
      for c in A B C; do
        for ap in b0 b1 b2 b3 b4 b5 ours; do step q_${ap}_$c --approach $ap --corpus $c --settings "$(settings_for $c)"; done
        for ab in facts_bare no_enrichment no_facts; do
          step q_ours_${ab}_$c --approach ours --ablation $ab --corpus $c --settings "$(settings_for $c)"
        done
      done
      ;;
    aggregate)
      echo "=== $(date '+%F %T') aggregate" | tee -a "$LOGS/steps.log"
      $PY -m ragbench.eval.aggregate --runs "$OUT" --questions "$Q" --split dev --judge none --out "$OUT/results" > "$LOGS/aggregate.log" 2>&1
      echo "aggregate rc=$?" | tee -a "$LOGS/steps.log"
      $PY -m ragbench.eval.sanity --runs "$OUT" --results "$OUT/results" > "$LOGS/sanity.log" 2>&1
      echo "sanity rc=$?" | tee -a "$LOGS/steps.log"
      ;;
  esac
done
echo "=== $(date '+%F %T') phases '$PHASES' finished" | tee -a "$LOGS/steps.log"
