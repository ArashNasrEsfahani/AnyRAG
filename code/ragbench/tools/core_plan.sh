#!/usr/bin/env bash
# CORE PLAN (~4.8 GPU-hours): the reduced study that keeps all method/corpus/language variety.
#
#   bash ragbench/tools/core_plan.sh l3          # stage 1: finish L3 enrichment for the core documents only
#   bash ragbench/tools/core_plan.sh index       # stage 2: build every index the core test run uses (--final --index-only)
#   bash ragbench/tools/core_plan.sh dev         # stage 3: mini tuning run on the dev core files (runs/core_dev*)
#   bash ragbench/tools/core_plan.sh test        # stage 4: the --final core test run (runs/core_test, runs/core_test_timeline)
#   bash ragbench/tools/core_plan.sh aggregate   # stage 5: results + sanity per run directory, cost summary (no model)
#                                                #          (timelines scored from answer text; results_structured = the old rule)
#   bash ragbench/tools/core_plan.sh v2dev       # AGENTIC2 (v2) + its ablations on the dev files (runs/core_dev*)
#   bash ragbench/tools/core_plan.sh v2test      # AGENTIC2 --final on test_core + test_timeline (runs/core_test*),
#                                                #   blocks in the core test's order; added 2026-09-16 after the core test
#   bash ragbench/tools/core_plan.sh kbsync      # AGENTIC v3: ingest the core documents into work/kb/kb.sqlite
#                                                #   (no model call; writes work/runs/core/INGEST_COST.md)
#   bash ragbench/tools/core_plan.sh v3dev       # AGENTIC3 (v3) + its ablations on the dev files (runs/core_dev*)
#   bash ragbench/tools/core_plan.sh v3test      # AGENTIC3 --final on test_core + test_timeline (runs/core_test*),
#                                                #   blocks in the core test's order; added 2026-09-20
#   bash ragbench/tools/core_plan.sh v3wide      # the WIDE dev evaluation (runs/core_dev_wide): AGENTIC3, AGENTIC3
#                                                #   +vectors_always and AGENTIC2 on pilot.jsonl (36 dev questions,
#                                                #   corpora A+B+C), added 2026-09-20 because 20 dev A records could
#                                                #   not tell a real effect from decoding noise (V3_ABLATIONS.md)
#   bash ragbench/tools/core_plan.sh all         # index -> dev -> test -> aggregate
#   bash ragbench/tools/core_plan.sh status      # steps, records per variant, estimated remaining GPU time
#   bash ragbench/tools/core_plan.sh plan        # dry-run every index/dev/test step -> work/runs/core/PLAN.md
#   bash ragbench/tools/core_plan.sh index|dev|test --dry-run   # the plan of one stage (no index build, no model call)
#
# STOP ANY TIME (Ctrl+C, closing the terminal, a crash, a reboot). Everything is saved as it happens:
#   - every enriched section / summary / embedding is committed to its SQLite cache immediately (WAL),
#   - every answered question is appended to its run's JSONL (runs resume and skip finished ids),
#   - every step's exit code and duration goes to work/runs/core/logs/steps.tsv,
#   - every stage start/stop goes to work/journal/events.jsonl, GPU telemetry to work/telemetry/.
# Re-running the same stage continues where it stopped.
#   - Indexes: L1 glossaries, B4 summaries, B3 embeddings and L3 enrichment are cached per item, so an interrupted
#     build resumes; a finished index loads in seconds.
#   - Records: finished (question, variant, setting) keys are skipped, errored ones are retried. AGENTIC also caches
#     every model call of an unfinished question (work/runs/<out>/agentic_calls_*.sqlite).
#   - A stage never deletes anything. ragbench.run refuses to resume (exit 2, logged in steps.tsv) only when the config
#     hash, an index fingerprint, a question's text or the Ollama version changed under existing records: decide then.
#   - When Ollama is down or not the pinned version the stage stops before the next step (journal run_stop is written).
# Order, so that stopping halfway stays useful:
#   index  CPU-only indexes first, then B3 embeddings, L1 glossaries, B4 summaries, the OURS (L3) stores;
#   test   corpus/setting blocks outer (A fa-fa, A en-en, B fa-fa, B en-en, C fa-fa, timelines, A fa-en, A en-fa) and
#          every variant inner, so whatever has finished compares every method on the same questions.
# OURS (L3) and OURS+monolingual_patterns never run on the timeline documents (no L3 enrichment is built for them;
# config.yaml final.exclude_variants refuses it). One GPU: a GPU stage refuses to start while another model-heavy
# ragbench job runs (CORE_PLAN_FORCE=1 skips that check).
set -u
cd "$(dirname "$0")/../.." || exit 1
PY=.venv/Scripts/python.exe
PIN="0.32.15"
OUT=runs/core
LOGS=work/runs/core/logs
mkdir -p "$LOGS" work/telemetry

# ---- core documents -------------------------------------------------------------------------
A_DOCS="rpt-b,rpt-f,rpt-c,rpt-e"
B_DOCS="digikala-1401"
C_DOCS="C.divar_annual-report_1400.fa,C.snapp_annual-report_1402.fa,C.shaparak_annual-economic-report_1401.fa"
DEVQ=ragbench/data/questions/pilot.jsonl   # index-only builds use the dev file (no --final needed)

# ---- core question files and run directories (stages index, dev, test, aggregate) ------------
QD=ragbench/data/questions
TQ=$QD/test_core.jsonl                  # 61 test questions: A 40 (4 settings), B 10 (fa-fa, en-en), C 11 (fa-fa)
TT=$QD/test_timeline.jsonl              # timeline test questions (corpus C, fa-fa); skipped while the file is absent
DQ=$QD/dev_core.jsonl                   # 5 dev questions (A rpt-b, rpt-f)
DT=$QD/dev_core_timeline.jsonl          # 2 dev timeline questions
PT=$QD/pilot_timeline.jsonl             # all 8 dev timeline questions (AGENTIC, B3 as reference)
PQ=$QD/pilot.jsonl                      # 36 dev questions, corpora A (20) + B (8) + C (8): the WIDE dev set
case "${1:-} ${2:-}" in                 # CORE_TT: dry-run the timeline steps with a stand-in file (dry-run only)
  plan*|*" --dry-run") if [ -n "${CORE_TT:-}" ]; then TT=$CORE_TT; fi ;;
esac
TEST_OUT=runs/core_test
TL_OUT=runs/core_test_timeline          # own directory: the timeline C restriction differs from test_core's C documents
DEV_OUT=runs/core_dev                   # dev_core (A) + dev_core_timeline (C)
PT_OUT=runs/core_dev_pilot_timeline     # pilot_timeline (C, another restriction)
WIDE_OUT=runs/core_dev_wide             # pilot.jsonl (stage v3wide): its own directory and its own doc restriction
F="--final --docs-from-questions"
DRYLOGS=work/runs/core/dryrun
TEST_VARIANTS="b0 b1 b2 b3 b4 b5 l0 l1 ours mono agentic"
TL_VARIANTS="b0 b1 b2 b3 b4 b5 l0 l1 agentic"
V2DEV_VARIANTS=${V2DEV_VARIANTS:-"agentic2 a2noverify a2nofdb a2notree a2novec a2c32"}   # stage v2dev (dev files only)
# stage v3dev (dev files only): AGENTIC3 and the ablations of AGENTIC3_SPEC.md section 8.
# NOTE 2026-09-20: `vectors_always` became the DEFAULT after stage v3wide (defaults.py `vector_policy`), so
# `a3vecall` now re-runs the base configuration under another name and `a3nofast` switches off a path that is
# already off. Both are kept in the list so the recorded rows stay reproducible; the ablation that matters now is
# the OLD routing, which has no flag of its own because AGENTIC3_FLAGS is in a frozen file (change request in
# V3_ABLATIONS.md). Until it has one, run it as:
#   AGENTIC3_OPTIONS='{"vector_policy": "when_thin"}' ... --out runs/<its own directory>
# in a directory of its own, because the variant NAME is unchanged and records would otherwise collide.
V3DEV_VARIANTS=${V3DEV_VARIANTS:-"agentic3 a3gate a3nofast a3notyped a3novec a3vecall a3nofacet a3nosql a3notri a3noprior"}
# stage v3wide (dev only, pilot.jsonl): the three arms the vectors decision needs, in this order. agentic3 runs
# first because a3vecall SHARES ITS CALL CACHE (V3WIDE_SHARED_CACHE below), so a question on which the flag changes
# neither route nor candidates is the same run rather than an independent re-run of it.
V3WIDE_VARIANTS=${V3WIDE_VARIANTS:-"agentic3 a3vecall agentic2"}
RC=0

vargs() {  # vargs <key>: ragbench.run arguments of a core variant
  case $1 in
    b0|b1|b2|b3|b4|b5|agentic|agentic2|agentic3|agentic4) echo "--approach $1" ;;
    a4nograph) echo "--approach agentic4 --ablation no_graph" ;;  # AGENTIC4 ablations (stage v4dev only; never --final)
    a4nofam) echo "--approach agentic4 --ablation no_family" ;;
    a4noquals) echo "--approach agentic4 --ablation no_graph_quals" ;;
    a4novec) echo "--approach agentic4 --ablation no_vectors" ;;
    a4nocpt) echo "--approach agentic4 --ablation no_concepts" ;;
    lex) echo "--approach lex" ;;                                   # LEX (2026-09-22) and its ablations (stage lexdev)
    lexllmrx) echo "--approach lex --ablation llm_regex" ;;
    lexfree) echo "--approach lex --ablation free_keywords" ;;
    lexfixed) echo "--approach lex --ablation fixed_window" ;;
    lexw250) echo "--approach lex --ablation window=250" ;;
    lexw1000) echo "--approach lex --ablation window=1000" ;;
    lexopen) echo "--approach lex --ablation open_corpus" ;;
    lexopentitles) echo "--approach lex --ablation open_corpus,title_routing" ;;
    lexbm25) echo "--approach lex --ablation bm25" ;;                # LEX upgrades one at a time, LEX2 = all three
    lexrv) echo "--approach lex --ablation report_vocab" ;;
    lexyp) echo "--approach lex --ablation year_pass" ;;
    lexlite) echo "--approach lex --ablation no_labels" ;;
    lex2) echo "--approach lex2" ;;
    lex2open) echo "--approach lex2 --ablation open_corpus" ;;
    lexlearn) echo "--approach lex --ablation learned" ;;    # the window score fitted from data (stage lexlearn)
    lex2learn) echo "--approach lex2 --ablation learned" ;;
    a2novec) echo "--approach agentic2 --ablation no_vectors" ;;   # AGENTIC2 ablations (stage v2dev only; never --final)
    a2nofdb) echo "--approach agentic2 --ablation no_factdb" ;;
    a2notree) echo "--approach agentic2 --ablation no_tree" ;;
    a2noverify) echo "--approach agentic2 --ablation no_verify" ;;
    a2c32) echo "--approach agentic2 --ablation candidates=32" ;;
    a3notyped) echo "--approach agentic3 --ablation no_typed" ;;   # AGENTIC3 ablations (stage v3dev only; never --final)
    a3notri) echo "--approach agentic3 --ablation no_trigram" ;;
    a3nosql) echo "--approach agentic3 --ablation no_sql_reconcile" ;;
    a3nofast) echo "--approach agentic3 --ablation no_fastpath" ;;
    a3gate) echo "--approach agentic3 --ablation verify_gate" ;;
    a3vecall) echo "--approach agentic3 --ablation vectors_always" ;;
    a3novec) echo "--approach agentic3 --ablation no_vectors" ;;
    a3nofacet) echo "--approach agentic3 --ablation no_facets" ;;
    a3noprior) echo "--approach agentic3 --ablation no_tree_prior" ;;
    a3warm) echo "--approach agentic3 --ablation warm" ;;
    l0) echo "--approach ours --index-level L0" ;;            # OURS+index_level=L0
    l1) echo "--approach ours --index-level L1" ;;            # OURS+index_level=L1
    ours) echo "--approach ours" ;;                           # OURS (L3, full enrichment)
    mono) echo "--approach ours --ablation monolingual_patterns" ;;
    *) echo "unknown variant key $1" >&2; return 1 ;;
  esac
}

core_settings() {  # the declared settings of a corpus (config.yaml final.settings)
  case $1 in A) echo "fa-fa,en-en,fa-en,en-fa" ;; B) echo "fa-fa,en-en" ;; *) echo "fa-fa" ;; esac
}

check_ollama() {
  local v
  v=$(curl -s -m 5 http://127.0.0.1:11434/api/version | sed -E 's/.*"version":"([^"]+)".*/\1/')
  if [ -z "$v" ]; then echo "Ollama is not running: start it with 'ollama serve' and re-run."; exit 1; fi
  if [ "$v" != "$PIN" ]; then
    $PY -m ragbench.tools.journal add --kind note --name core_plan_refused --note "Ollama $v != pinned $PIN" >/dev/null
    echo "Ollama is $v but runs are pinned to $PIN; not running."; exit 1
  fi
}

step() {  # step <name> <ragbench.run args...>
  local name=$1; shift
  if [ -n "${DRY:-}" ]; then            # dry-run: the plan only, logged under work/runs/core/dryrun
    $PY -m ragbench.run "$@" --dry-run > "$DRYLOGS/$name.log" 2>&1
    local drc=$?
    printf "%s\t%s\t%s\n" "$name" "$drc" "$*" >> "$DRYLOGS/steps.tsv"
    printf "%-26s rc=%s  %s\n" "$name" "$drc" "$(grep -E ' planned \(|^FINAL|^refusing|Error' "$DRYLOGS/$name.log" | head -2 | cut -c1-170 | tr '\n' ' ')"
    return $drc
  fi
  local t0=$(date +%s)
  echo "=== $(date '+%F %T') $name: $*" | tee -a "$LOGS/steps.log"
  $PY -m ragbench.run "$@" > "$LOGS/$name.log" 2>&1
  local rc=$?
  printf "%s\t%s\t%ss\t%s\n" "$name" "$rc" "$(( $(date +%s) - t0 ))" "$(date '+%F %T')" >> "$LOGS/steps.tsv"
  tail -2 "$LOGS/$name.log"
  return $rc
}

stage_begin() {  # stage_begin <stage>
  STAGE=$1; STAGE_T0=$(date +%s); STAMP=$(date +%Y%m%d_%H%M%S)
  TELE="work/telemetry/gpu_core_${STAGE}_$STAMP.csv"
  $PY -m ragbench.tools.journal add --kind run_start --name "core_$STAGE" --data "{\"telemetry\": \"$TELE\"}" >/dev/null
  $PY -m ragbench.tools.gpu_monitor --out "$TELE" --interval 30 &
  MON=$!
  trap 'stage_end 130; exit 130' INT TERM
}

stage_end() {  # stage_end <exit code>
  kill "$MON" 2>/dev/null
  $PY -m ragbench.tools.journal add --kind run_stop --name "core_$STAGE" --duration-s $(( $(date +%s) - STAGE_T0 )) \
    --data "{\"exit_code\": $1}" >/dev/null
  trap - INT TERM
}

# ---- helpers of stages index, dev, test, aggregate ------------------------------------------

ollama_ok() {  # like check_ollama, but returns (inside a stage, so stage_end still runs)
  local v
  v=$(curl -s -m 5 http://127.0.0.1:11434/api/version | sed -E 's/.*"version":"([^"]+)".*/\1/')
  [ "$v" = "$PIN" ] && return 0
  echo "Ollama is '${v:-not running}' (pinned $PIN): stopping this stage; re-run it once Ollama $PIN is up."
  return 1
}

run_step() {  # run_step <name> <ragbench.run args...>: a step; returns 1 only when the stage must stop (Ollama down)
  if [ -z "${DRY:-}" ] && ! ollama_ok; then RC=99; return 1; fi
  step "$@"
  local r=$?
  if [ $r -ne 0 ]; then RC=$r; fi       # a failed or refused step is logged; the stage goes on with the next one
  return 0
}

pystep() {  # pystep <name> <python module> <args...>: a non-ragbench.run step with the same logging
  local name=$1; shift
  local t0=$(date +%s)
  echo "=== $(date '+%F %T') $name: $*" | tee -a "$LOGS/steps.log"
  $PY -m "$@" > "$LOGS/$name.log" 2>&1
  local rc=$?
  printf "%s\t%s\t%ss\t%s\n" "$name" "$rc" "$(( $(date +%s) - t0 ))" "$(date '+%F %T')" >> "$LOGS/steps.tsv"
  tail -2 "$LOGS/$name.log"
  return $rc
}

gpu_free() {  # no other model-heavy ragbench job may run (one GPU)
  [ -n "${CORE_PLAN_FORCE:-}" ] && return 0
  $PY -m ragbench.tools.core_plan busy
}

TL_READY=""
tl_ready() {  # the timeline test file exists and passes the core-plan rules (checked once per invocation)
  if [ -z "$TL_READY" ]; then
    if [ ! -f "$TT" ]; then
      TL_READY=no; echo "timeline steps skipped: $TT not present yet (re-run the stage when it exists)"
    elif $PY -m ragbench.tools.core_plan check-timeline --path "$TT"; then
      TL_READY=yes
    else
      TL_READY=no; echo "timeline steps skipped: $TT breaks the core-plan rules (see above)"
    fi
  fi
  [ "$TL_READY" = yes ]
}

stage_index() {  # every index the --final --docs-from-questions test run loads (copies in the test run directories)
  local v c
  for v in b0 b1 b2 b5 agentic l0 b3 l1 b4 ours mono; do        # CPU-only first, then B3, L1, B4, L3 stores
    for c in A B C; do
      run_step index_${v}_$c $(vargs $v) $F --questions "$TQ" --corpus $c --settings "$(core_settings $c)" \
        --index-only --out "$TEST_OUT" || return
    done
    case $v in ours|mono) continue ;; esac                        # no L3 on the timeline documents
    if tl_ready; then
      run_step index_${v}_TL $(vargs $v) $F --questions "$TT" --corpus C --settings fa-fa --index-only --out "$TL_OUT" || return
    fi
  done
}

stage_dev() {  # mini tuning run (dev split): every dev variant on the same questions before the next setting
  local s v
  for s in fa-fa en-en fa-en en-fa; do
    for v in b3 l0 l1 ours agentic; do
      run_step dev_${v}_A_$s $(vargs $v) --docs-from-questions --questions "$DQ" --corpus A --settings $s --out "$DEV_OUT" || return
    done
    if [ $s = fa-fa ]; then                                       # dev timelines (no L3 on timeline documents)
      for v in b3 l0 l1 agentic; do
        run_step dev_${v}_TL $(vargs $v) --docs-from-questions --questions "$DT" --corpus C --settings fa-fa --out "$DEV_OUT" || return
      done
    fi
  done
  for v in b3 agentic; do                                         # all 8 dev timeline questions: AGENTIC, B3 reference
    run_step dev_${v}_PT $(vargs $v) --docs-from-questions --questions "$PT" --corpus C --settings fa-fa --out "$PT_OUT" || return
  done
}

stage_test() {  # the --final core test: blocks outer, variants inner (CORE_TEST_BLOCKS selects a subset)
  local block c s v
  for block in ${CORE_TEST_BLOCKS:-A:fa-fa A:en-en B:fa-fa B:en-en C:fa-fa TL:fa-fa A:fa-en A:en-fa}; do
    c=${block%%:*}; s=${block#*:}
    if [ $c = TL ]; then
      tl_ready || continue
      for v in $TL_VARIANTS; do
        run_step test_${v}_TL_$s $(vargs $v) $F --questions "$TT" --corpus C --settings $s --out "$TL_OUT" || return
      done
    else
      for v in $TEST_VARIANTS; do
        run_step test_${v}_${c}_$s $(vargs $v) $F --questions "$TQ" --corpus $c --settings $s --out "$TEST_OUT" || return
      done
    fi
  done
}

# The test in three ~1-hour phases (estimates from PLAN.md; each phase is resumable on its own):
#   test1  A fa-fa, A en-en                  (~1.25 h)  monolingual AI-generated corpus
#   test2  B fa-fa, B en-en, C fa-fa, TL     (~0.95 h)  natural corpora + timelines
#   test3  A fa-en, A en-fa                  (~1.25 h)  cross-lingual
stage_v2dev() {  # AGENTIC2 (v2) on the dev files, next to B3/AGENTIC of stage dev, plus its ablations (dev only)
  # Same order as stage dev: settings outer, variants inner; timelines after fa-fa; pilot_timeline last.
  local s v
  for s in fa-fa en-en fa-en en-fa; do
    for v in $V2DEV_VARIANTS; do
      run_step v2dev_${v}_A_$s $(vargs $v) --docs-from-questions --questions "$DQ" --corpus A --settings $s --out "$DEV_OUT" || return
    done
    if [ $s = fa-fa ]; then
      for v in $V2DEV_VARIANTS; do
        run_step v2dev_${v}_TL $(vargs $v) --docs-from-questions --questions "$DT" --corpus C --settings fa-fa --out "$DEV_OUT" || return
      done
    fi
  done
  for v in $V2DEV_VARIANTS; do
    run_step v2dev_${v}_PT $(vargs $v) --docs-from-questions --questions "$PT" --corpus C --settings fa-fa --out "$PT_OUT" || return
  done
}

stage_v2test() {  # AGENTIC2 --final on the core test files, blocks in the core test's order (partial runs stay comparable)
  local block c s
  if [ -n "${AGENTIC2_OPTIONS:-}" ]; then echo "AGENTIC2_OPTIONS is set (dev tuning only): refusing the test stage"; RC=2; return; fi
  for block in ${CORE_TEST_BLOCKS:-A:fa-fa A:en-en B:fa-fa B:en-en C:fa-fa TL:fa-fa A:fa-en A:en-fa}; do
    c=${block%%:*}; s=${block#*:}
    if [ $c = TL ]; then
      tl_ready || continue
      run_step v2test_agentic2_TL_$s $(vargs agentic2) $F --questions "$TT" --corpus C --settings $s --out "$TL_OUT" || return
    else
      run_step v2test_agentic2_${c}_$s $(vargs agentic2) $F --questions "$TQ" --corpus $c --settings $s --out "$TEST_OUT" || return
    fi
  done
}

stage_kbsync() {  # AGENTIC v3 ingestion: every core document into work/kb/kb.sqlite, no model call
  # `ragbench.run --index-only` is what ingests, because _build_index is the one place the approach sees the
  # corpus and never a question (fairness rule 2); it also writes the index copies the test run reads. The same
  # documents under another corpus set are already loaded, so the second and third call are no-ops.
  local c
  for c in A B C; do
    run_step kbsync_agentic3_$c $(vargs agentic3) $F --questions "$TQ" --corpus $c --settings "$(core_settings $c)" \
      --index-only --out "$TEST_OUT" || return
  done
  if tl_ready; then
    run_step kbsync_agentic3_TL $(vargs agentic3) $F --questions "$TT" --corpus C --settings fa-fa \
      --index-only --out "$TL_OUT" || return
  fi
  for c in A C; do                    # the dev documents, so stage v3dev does not ingest inside its first question
    local q=$DQ; [ $c = C ] && q=$DT
    run_step kbsync_agentic3_dev_$c $(vargs agentic3) --docs-from-questions --questions "$q" --corpus $c \
      --settings "$(core_settings $c)" --index-only --out "$DEV_OUT" || return
  done
  run_step kbsync_agentic3_dev_PT $(vargs agentic3) --docs-from-questions --questions "$PT" --corpus C \
    --settings fa-fa --index-only --out "$PT_OUT" || return
  [ -n "${DRY:-}" ] && return
  pystep kbsync_status ragbench.kb status --md || RC=$?
  echo "ingestion cost: work/runs/core/INGEST_COST.md"
}

stage_v3dev() {  # AGENTIC3 (v3) on the dev files, next to v2's rows, plus its ablations (dev only)
  # Same order as stage dev and stage v2dev: settings outer, variants inner; timelines after fa-fa; pilot last.
  local s v
  for s in fa-fa en-en fa-en en-fa; do
    for v in $V3DEV_VARIANTS; do
      run_step v3dev_${v}_A_$s $(vargs $v) --docs-from-questions --questions "$DQ" --corpus A --settings $s --out "$DEV_OUT" || return
    done
    if [ $s = fa-fa ]; then
      for v in $V3DEV_VARIANTS; do
        run_step v3dev_${v}_TL $(vargs $v) --docs-from-questions --questions "$DT" --corpus C --settings fa-fa --out "$DEV_OUT" || return
      done
    fi
  done
  for v in $V3DEV_VARIANTS; do
    run_step v3dev_${v}_PT $(vargs $v) --docs-from-questions --questions "$PT" --corpus C --settings fa-fa --out "$PT_OUT" || return
  done
}

stage_v3test() {  # AGENTIC3 --final on the core test files, blocks in the core test's order
  local block c s
  if [ -n "${AGENTIC3_OPTIONS:-}" ]; then echo "AGENTIC3_OPTIONS is set (dev tuning only): refusing the test stage"; RC=2; return; fi
  for block in ${CORE_TEST_BLOCKS:-A:fa-fa A:en-en B:fa-fa B:en-en C:fa-fa TL:fa-fa A:fa-en A:en-fa}; do
    c=${block%%:*}; s=${block#*:}
    if [ $c = TL ]; then
      tl_ready || continue
      run_step v3test_agentic3_TL_$s $(vargs agentic3) $F --questions "$TT" --corpus C --settings $s --out "$TL_OUT" || return
    else
      run_step v3test_agentic3_${c}_$s $(vargs agentic3) $F --questions "$TQ" --corpus $c --settings $s --out "$TEST_OUT" || return
    fi
  done
}

# The WIDE dev evaluation, 2026-09-20. Why it exists: on the 20 corpus-A dev records of stage v3dev a PROVABLE
# no-op (`no_fastpath`, with FAST firing 0 times in both arms) moved dev A strict by 0.10 and rewrote 15 of 28
# answer texts, so the ablation table could not separate an effect from decoding noise (V3_ABLATIONS.md, section
# "Read this first"). pilot.jsonl is 36 dev questions that AGENTIC2 and AGENTIC3 have never been run on, and it
# is the only dev file with single-fact questions on the NATURAL corpora: stage v3dev's lookup evidence is five
# corpus-A questions from the AI-generated corpus and nothing else.
#
# Settings: A and B monolingual (fa-fa, en-en), C fa-fa. The two cross-lingual A settings are dropped on purpose -
# they re-ask the same 20 questions and buy correlated records, not new questions, and stage v3dev already holds
# all four settings for the five questions it covers.
#
# Shared call cache: the two AGENTIC3 arms are given ONE cache file, so `vectors_always` pays the model only for
# the questions where it actually changes the route or the candidate set. The cache is content-addressed on
# (model, purpose, prompt version, messages, options), so a changed prompt is always a fresh call; what it removes
# is the run-to-run decoding noise between two arms that are doing the same thing. AGENTIC2 gets its own cache:
# it shares no prompt with v3 and nothing would hit.
V3WIDE_SHARED_CACHE=${V3WIDE_SHARED_CACHE:-work/$WIDE_OUT/agentic_calls_v3wide_shared.sqlite}

wide_opts() {  # wide_opts <variant key>: the AGENTIC3_OPTIONS of one wide arm (dev tuning env; --final refuses it)
  case $1 in
    agentic3|a3vecall) echo "{\"call_cache_path\": \"$V3WIDE_SHARED_CACHE\"}" ;;
    *) echo "" ;;
  esac
}

stage_v3wide() {  # AGENTIC3, AGENTIC3+vectors_always and AGENTIC2 on pilot.jsonl (dev only, never --final)
  local v c s
  mkdir -p "work/$WIDE_OUT"
  for v in $V3WIDE_VARIANTS; do
    for c in A B C; do
      case $c in A|B) s=fa-fa,en-en ;; *) s=fa-fa ;; esac
      AGENTIC3_OPTIONS="$(wide_opts $v)" \
        run_step v3wide_${v}_$c $(vargs $v) --docs-from-questions --questions "$PQ" --corpus $c \
          --settings $s --out "$WIDE_OUT" || return
    done
  done
}

# AGENTIC4 (v4, 2026-09-21): v3 plus the graph path over typed facts. The dev replay (tools/v4_replay.py, no model
# call) set its one decision: on ONE report the graph path is off and v4 is v3 exactly, so it acts only where a
# question spans several reports - the timelines, and the cross-report questions of pilot.jsonl. This stage runs,
# on the FROZEN corpus and in fresh directories: agentic3 (the paired reference, today's defaults), agentic4,
# a4nograph (must reproduce agentic3 answer for answer - the equivalence check) and a4noquals (the graph qualifiers
# off in the reconciliation). ONE shared call cache, seeded from v3's recorded wide-dev calls: the cache is
# content-addressed, so every call whose prompt did not change replays, and the GPU is spent where v4 differs.
V4DEV_OUT=runs/core_dev_v4                 # pilot.jsonl on the frozen corpus
V4DEV_TL_OUT=runs/core_dev_v4_tl           # pilot_timeline.jsonl on the frozen corpus (its own doc restriction)
V4DEV_VARIANTS=${V4DEV_VARIANTS:-"agentic3 agentic4 a4nograph a4noquals a4nocpt"}
V4_SHARED_CACHE=${V4_SHARED_CACHE:-work/$V4DEV_OUT/agentic_calls_v4_shared.sqlite}
V4_CACHE_SEED=work/$WIDE_OUT/agentic_calls_v3wide_shared.sqlite   # where v3's wide and recall arms kept their calls

v4_opts() {  # the dev-tuning env of one v4dev arm: the shared call cache only (--final refuses the variable)
  echo "{\"call_cache_path\": \"$V4_SHARED_CACHE\"}"
}

stage_v4dev() {  # v3 vs v4 on pilot.jsonl and pilot_timeline.jsonl (dev only, never --final)
  local v c s o
  mkdir -p "work/$V4DEV_OUT" "work/$V4DEV_TL_OUT"
  if [ ! -f "$V4_SHARED_CACHE" ] && [ -f "$V4_CACHE_SEED" ]; then cp "$V4_CACHE_SEED" "$V4_SHARED_CACHE"; fi
  for v in $V4DEV_VARIANTS; do
    o="$(v4_opts)"
    for c in A B C; do
      case $c in A|B) s=fa-fa,en-en ;; *) s=fa-fa ;; esac
      AGENTIC3_OPTIONS="$o" AGENTIC4_OPTIONS="$o" \
        run_step v4dev_${v}_$c $(vargs $v) --docs-from-questions --questions "$PQ" --corpus $c \
          --settings $s --out "$V4DEV_OUT" || return
    done
    AGENTIC3_OPTIONS="$o" AGENTIC4_OPTIONS="$o" \
      run_step v4dev_${v}_TL $(vargs $v) --docs-from-questions --questions "$PT" --corpus C --settings fa-fa \
        --out "$V4DEV_TL_OUT" || return
  done
}

# LEX (2026-09-22): report cards -> LLM keywords -> the reports' own phrases -> code-compiled regex -> windows -> the
# shared answer. Dev only: pilot.jsonl and pilot_timeline.jsonl, every ablation, two more window sizes, layer 1 with the
# report list withheld (cards / titles only), and the one-call baselines B2, B3 and OURS L0 on the same files for a
# paired comparison (AGENTIC3's dev records are in core_dev_v4). The LEX arms share one call cache: the keywords call
# is the same prompt in every arm and replays.
LEXDEV_OUT=runs/lex_dev
LEXDEV_TL_OUT=runs/lex_dev_tl
LEXDEV_VARIANTS=${LEXDEV_VARIANTS:-"lex lexllmrx lexfree lexfixed lexw250 lexw1000 lexopen lexopentitles b2 b3 l0"}
LEX_SHARED_CACHE=${LEX_SHARED_CACHE:-work/$LEXDEV_OUT/lex_calls_shared.sqlite}

stage_lexdev() {  # LEX and its ablations on the dev files (never --final)
  local v c s o
  mkdir -p "work/$LEXDEV_OUT" "work/$LEXDEV_TL_OUT"
  o="{\"call_cache_path\": \"$LEX_SHARED_CACHE\"}"
  for v in $LEXDEV_VARIANTS; do
    for c in A B C; do
      case $c in A|B) s=fa-fa,en-en ;; *) s=fa-fa ;; esac
      LEX_OPTIONS="$o" run_step lexdev_${v}_$c $(vargs $v) --docs-from-questions --questions "$PQ" --corpus $c \
        --settings $s --out "$LEXDEV_OUT" || return
    done
    LEX_OPTIONS="$o" run_step lexdev_${v}_TL $(vargs $v) --docs-from-questions --questions "$PT" --corpus C \
      --settings fa-fa --out "$LEXDEV_TL_OUT" || return
  done
}

# LEX v5 + LEX2 (2026-09-22), on the dev files only, in fresh directories. The first arm (lex) runs with NO call cache
# so its latency is a fresh measurement; every other LEX arm shares one cache. Then the earlier methods that have no
# dev records on these files yet (B0, B1, B4, B5, OURS L0/L1, AGENTIC v1) go into the lexdev directories, so every
# method is compared on the same questions. OURS L3 is not run (hours of index-time model calls).
LEX5_OUT=runs/lex_dev5
LEX5_TL_OUT=runs/lex_dev5_tl
LEX5_VARIANTS=${LEX5_VARIANTS:-"lex lex2 lexbm25 lexrv lexyp lexlite lexllmrx lexfree lexfixed lexw250 lexw1000 lexopen lexopentitles lex2open"}
PREV_DEV_VARIANTS=${PREV_DEV_VARIANTS:-"b0 b1 l0 l1 b5 agentic b4"}
LEX5_SHARED_CACHE=${LEX5_SHARED_CACHE:-work/$LEX5_OUT/lex_calls_shared.sqlite}

stage_lexdev5() {  # LEX v5 variants + LEX2, then the missing earlier methods, on the dev files (never --final)
  local v c s o
  mkdir -p "work/$LEX5_OUT" "work/$LEX5_TL_OUT"
  for v in $LEX5_VARIANTS; do
    if [ "$v" = lex ]; then o='{"call_cache": "off"}'; else o="{\"call_cache_path\": \"$LEX5_SHARED_CACHE\"}"; fi
    for c in A B C; do
      case $c in A|B) s=fa-fa,en-en ;; *) s=fa-fa ;; esac
      LEX_OPTIONS="$o" run_step lexdev5_${v}_$c $(vargs $v) --docs-from-questions --questions "$PQ" --corpus $c \
        --settings $s --out "$LEX5_OUT" || return
    done
    LEX_OPTIONS="$o" run_step lexdev5_${v}_TL $(vargs $v) --docs-from-questions --questions "$PT" --corpus C \
      --settings fa-fa --out "$LEX5_TL_OUT" || return
  done
  for v in $PREV_DEV_VARIANTS; do
    for c in A B C; do
      case $c in A|B) s=fa-fa,en-en ;; *) s=fa-fa ;; esac
      run_step lexdev5_${v}_$c $(vargs $v) --docs-from-questions --questions "$PQ" --corpus $c --settings $s \
        --out "$LEXDEV_OUT" || return
    done
    case $v in agentic) continue ;; esac        # AGENTIC v1's dev timelines exist (core_dev_pilot_timeline)
    run_step lexdev5_${v}_TL $(vargs $v) --docs-from-questions --questions "$PT" --corpus C --settings fa-fa \
      --out "$LEXDEV_TL_OUT" || return
  done
}

# ------------------------------------------------------------------------------------------------------------------
# lexlearn: LEX's window score FITTED from the dev questions (approaches/lex/learn.py) instead of hand-set. The dump
# and the five fits cost no GPU time (tools/lex_learn.py dump/folds/report, run first); each arm here is five runs,
# one per fold, every question scored by weights that never saw it. Dev only: --ids and LEX_OPTIONS are both refused
# by --final, so this stage cannot touch the test set.
LEXLEARN_DIR=${LEXLEARN_DIR:-work/runs/lex_learn_1se}
LEXLEARN_OUT=runs/lex_learn_e2e
LEXLEARN_TL_OUT=runs/lex_learn_e2e_tl
LEXLEARN_K=${LEXLEARN_K:-5}
LEXLEARN_VARIANTS=${LEXLEARN_VARIANTS:-"lexlearn"}

stage_lexlearn() {  # the learned window score, out of fold, on the dev files (never --final)
  local v i c s ids o
  mkdir -p "work/$LEXLEARN_OUT" "work/$LEXLEARN_TL_OUT"
  for v in $LEXLEARN_VARIANTS; do
    for i in $(seq 0 $((LEXLEARN_K - 1))); do
      if [ ! -f "$LEXLEARN_DIR/fold_ids_$i.txt" ]; then
        echo "no $LEXLEARN_DIR/fold_ids_$i.txt: run tools/lex_learn.py dump and folds first"; RC=2; return 0
      fi
      ids=$(cat "$LEXLEARN_DIR/fold_ids_$i.txt")
      o="{\"call_cache_path\": \"$LEX5_SHARED_CACHE\", \"learned_model\": \"$LEXLEARN_DIR/model_fold$i.json\"}"
      for c in A B C; do
        case $c in A|B) s=fa-fa,en-en ;; *) s=fa-fa ;; esac
        LEX_OPTIONS="$o" run_step lexlearn_${v}_f${i}_$c $(vargs $v) --ids "$ids" --docs-from-questions \
          --questions "$PQ" --corpus $c --settings $s --out "$LEXLEARN_OUT" || return
      done
      LEX_OPTIONS="$o" run_step lexlearn_${v}_f${i}_TL $(vargs $v) --ids "$ids" --docs-from-questions \
        --questions "$PT" --corpus C --settings fa-fa --out "$LEXLEARN_TL_OUT" || return
    done
  done
}

stage_v4test() {  # AGENTIC4 --final on the core test files, blocks in the core test's order (the user gates it)
  local block c s
  if [ -n "${AGENTIC4_OPTIONS:-}" ]; then echo "AGENTIC4_OPTIONS is set (dev tuning only): refusing the test stage"; RC=2; return; fi
  for block in ${CORE_TEST_BLOCKS:-A:fa-fa A:en-en B:fa-fa B:en-en C:fa-fa TL:fa-fa A:fa-en A:en-fa}; do
    c=${block%%:*}; s=${block#*:}
    if [ $c = TL ]; then
      tl_ready || continue
      run_step v4test_agentic4_TL_$s $(vargs agentic4) $F --questions "$TT" --corpus C --settings $s --out "$TL_OUT" || return
    else
      run_step v4test_agentic4_${c}_$s $(vargs agentic4) $F --questions "$TQ" --corpus $c --settings $s --out "$TEST_OUT" || return
    fi
  done
}

stage_test1() { CORE_TEST_BLOCKS="A:fa-fa A:en-en" stage_test; }
stage_test2() { CORE_TEST_BLOCKS="B:fa-fa B:en-en C:fa-fa TL:fa-fa" stage_test; }
stage_test3() { CORE_TEST_BLOCKS="A:fa-en A:en-fa" stage_test; }

agg() {  # agg <name> <run dir> <split> <question files...>
  # Timeline answers are scored from the ANSWER TEXT of every approach (--timeline-scoring text, the fair headline,
  # decided 2026-09-16); results_structured keeps the original rule (a structured series when an approach returns one)
  # for the directories that hold timeline questions.
  local name=$1 out=$2 split=$3 f
  shift 3
  if ! ls work/$out/runs_*.jsonl > /dev/null 2>&1; then echo "skip aggregate $name: no records in work/$out yet"; return 0; fi
  local qa=() tl=""
  for f in "$@"; do qa+=(--questions "$f"); case $f in *timeline*) tl=1 ;; esac; done
  pystep aggregate_results_$name ragbench.eval.aggregate --runs "$out" "${qa[@]}" --split $split --judge none \
    --timeline-scoring text --out "$out/results" || RC=$?
  pystep aggregate_common_$name ragbench.eval.aggregate --runs "$out" "${qa[@]}" --split $split --judge none --common-only \
    --timeline-scoring text --out "$out/results_common" || RC=$?
  # Key-fact scoring (work/analysis/SCORING.md section 9): a SECOND, parallel pass into its own directory, so the
  # strict aggregation in results/ is left exactly as it is. Questions carrying meta.key_facts (locate,
  # multi-section, cross-year) are scored by their key facts; every other question is unchanged.
  pystep aggregate_keyfacts_$name ragbench.eval.aggregate --runs "$out" "${qa[@]}" --split $split --judge none \
    --timeline-scoring text --scoring keyfacts --out "$out/results_keyfacts" || RC=$?
  if [ -n "$tl" ]; then
    pystep aggregate_structured_$name ragbench.eval.aggregate --runs "$out" "${qa[@]}" --split $split --judge none \
      --timeline-scoring structured --out "$out/results_structured" || RC=$?
  fi
  pystep aggregate_sanity_$name ragbench.eval.sanity --runs "$out" --results "$out/results" "${qa[@]}" || RC=$?
}

stage_aggregate() {  # match metrics only (judge none); results_common = keys every variant answered (partial runs)
  agg core_dev "$DEV_OUT" dev "$DQ" "$DT"
  agg core_dev_pilot_timeline "$PT_OUT" dev "$PT"
  agg core_dev_wide "$WIDE_OUT" dev "$PQ"
  agg core_test "$TEST_OUT" test "$TQ"
  if [ -f "$TT" ]; then agg core_test_timeline "$TL_OUT" test "$TT"; fi
  pystep aggregate_cost ragbench.tools.core_plan cost > /dev/null || RC=$?
  echo "cost summary: work/runs/core/COST.md"
}

dry_stages() {  # dry_stages <plan file> <stage...>: every step with --dry-run, then the plan table
  local plan=$1 s
  shift
  DRY=1; RC=0
  mkdir -p "$DRYLOGS"
  : > "$DRYLOGS/steps.tsv"
  for s in "$@"; do echo "--- dry-run: stage $s"; stage_$s; done
  $PY -m ragbench.tools.core_plan plan --dry-dir "$DRYLOGS" --out "$plan" > "$DRYLOGS/plan.log" 2>&1
  local prc=$?
  grep -E "^- |^\| \*\*Total|^Estimated GPU time per block|^-> " "$DRYLOGS/plan.log"
  return $(( RC != 0 ? RC : prc ))
}

gpu_stage() {  # gpu_stage <stage>: checks, then the stage between journal start/stop and GPU telemetry
  local s=$1
  stage_begin $s
  RC=0
  stage_$s
  stage_end $RC
  if [ $RC -eq 99 ]; then exit 99; fi
}

case "${1:-}" in
  status)
    tail -n 12 "$LOGS/steps.tsv" 2>/dev/null
    $PY -m ragbench.tools.core_plan status
    ;;
  l3)
    check_ollama
    stage_begin l3
    rc=0
    step l3_A --approach ours --questions "$DEVQ" --corpus A --docs "$A_DOCS" --settings fa-fa,en-en,fa-en,en-fa --index-only --out "$OUT" || rc=$?
    step l3_B --approach ours --questions "$DEVQ" --corpus B --docs "$B_DOCS" --settings fa-fa,en-en --index-only --out "$OUT" || rc=$?
    step l3_C --approach ours --questions "$DEVQ" --corpus C --docs "$C_DOCS" --settings fa-fa --index-only --out "$OUT" || rc=$?
    stage_end $rc
    ;;
  test1|test2|test3|v2dev|v2test|v3dev|v3test|v3wide|v4dev|v4test|kbsync|lexdev|lexdev5|lexlearn)
    if [ "${2:-}" = "--dry-run" ]; then dry_stages "$DRYLOGS/PLAN_$1.md" $1; exit $?; fi
    check_ollama
    gpu_free || exit 1
    gpu_stage $1
    exit $RC
    ;;
  index|dev|test)
    if [ "${2:-}" = "--dry-run" ]; then dry_stages "$DRYLOGS/PLAN_$1.md" $1; exit $?; fi
    check_ollama
    gpu_free || exit 1
    gpu_stage $1
    exit $RC
    ;;
  aggregate)
    stage_begin aggregate
    RC=0
    stage_aggregate
    stage_end $RC
    exit $RC
    ;;
  all)
    check_ollama
    gpu_free || exit 1
    for s in index dev test; do gpu_stage $s; done
    stage_begin aggregate; RC=0; stage_aggregate; stage_end $RC
    ;;
  plan)
    dry_stages work/runs/core/PLAN.md index dev test
    exit $?
    ;;
  *)
    sed -n 2,38p "$0"; exit 2
    ;;
esac
