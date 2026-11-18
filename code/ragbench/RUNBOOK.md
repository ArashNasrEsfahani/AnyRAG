# ragbench runbook

How to reproduce the study: which retrieval method a Persian organisation should use when it can only
run a small local model. Every approach answers with the same local model, generation options, answer
prompt and evidence budget. Only retrieval differs.

All commands run from `D:/KareOne/RAG Paper` in Git Bash (PowerShell works too; replace `export X=y`
with `$env:X = "y"`). `PY` below means `.venv/Scripts/python.exe`.

```bash
cd "D:/KareOne/RAG Paper"
export PYTHONIOENCODING=utf-8          # Persian in console output
PY=.venv/Scripts/python.exe
```

## 0. Development history

Sections 0-0e of the internal runbook recorded the smoke, dev and core-plan runs on the early development corpus
(corpus A, internal reports). That corpus is not part of the public release, so those sections are omitted here.

## 1. Setup (once)

| Piece | Where / what | Check |
|---|---|---|
| Python | venv `.venv` (CPython 3.14.0), packages pinned in `ragbench/requirements.txt` (bm25s, numpy, PyYAML, regex, requests, scipy) | `$PY -m pip freeze` |
| ripgrep | `tools/bin/rg.exe` (14.1.1, `paths.rg` in config). Python does not use the `rg` on PATH | `tools/bin/rg.exe --version` |
| Ollama | 0.32.15 at `http://127.0.0.1:11434`; models `gemma4:e4b-it-qat` (chat, digest `ee665637…`) and `bge-m3:latest` (embed, digest `79076464…`) | `curl -s 127.0.0.1:11434/api/version` |
| OpenRouter key | `.env` → `OPENROUTER_API_KEY`. Never print it. Judge model `google/gemini-2.5-flash-lite` | `$PY -m ragbench.eval.judge --probe` (costs 1 call) |
| Corpora | A: early development corpus of internal reports, not released (config `corpora.A`, empty here). B/C: `corpus/extracted/<company>/<stem>/` from `tools/extract/` | `$PY -m ragbench.tests.test_foundation` |

Create the venv only if it is missing: `"C:/Users/arash/AppData/Local/Programs/Python/Python314/python.exe" -m venv .venv && .venv/Scripts/python.exe -m pip install -r ragbench/requirements.txt`.
Do not run `pip install --upgrade pip` inside the venv (it broke the first venv on a Windows file lock).

**Before real runs:**
1. **Freeze Ollama.** Turn off the Ollama auto-updater (v0.34.0 is downloaded and pending). `run.py` refuses to resume a run directory when the Ollama version changes, but a changed runtime still changes results.
2. **Wait for the extractor to finish.** B/C documents are re-read when their extraction changes. If gold unit ids moved, rebuild the questions first: `$PY -m ragbench.tools.questions build && $PY -m ragbench.tools.questions validate --strict`.
3. **Run the offline test suites** (no model calls, a few minutes in total):
   ```bash
   $PY -m ragbench.tests.test_foundation      # 20 checks
   $PY -m ragbench.tests.test_baselines       # 29
   $PY -m ragbench.approaches.tree_nav --offline   # 42
   $PY -m ragbench.tests.test_grep_agent      # 46
   $PY -m ragbench.approaches.ours.smoke_test # OURS offline part (97 checks, "0 failure(s)")
   $PY -m ragbench.approaches.ours.levels_test   # 72 (OURS index levels L0-L2)
   $PY -m ragbench.approaches.agentic.test_agentic   # 106 (AGENTIC, incl. the shared timeline answer prompt)
   $PY -m ragbench.approaches.agentic2.test_agentic2 # 53 (AGENTIC2: tree, fact DB, lazy vectors, call/time caps, repairs)
   $PY -m ragbench.tests.test_questions       # 43 (pilot + the timeline dev file)
   $PY -m ragbench.eval.test_eval             # 65
   $PY -m ragbench.eval.test_timeline         # 89 (series parser, scoring, aggregate incl. --timeline-scoring text, judge rubric, test file)
   $PY -m ragbench.tests.test_core_integration   # 68 (registry/flags incl. AGENTIC2, answer formats, --final guard, dry-run, caps)
   ```
   All 11 suites: 666 checks, 0 failures on 2026-09-16 after the CORE PLAN integration. With AGENTIC2 (section 0e):
   12 suites, 730 checks, 0 failures (`bash ragbench/tools/offline_suites.sh` runs them all).
4. **One GPU.** Never run two model-heavy jobs at once. The client serialises calls across processes
   (`work/locks/ollama.lock`), so a second job silently doubles everyone's latency. `RunRecord.lock_wait_ms`
   records the wait, and the no-load latency subtracts it.
5. **Keep other GPU/CPU work off the machine.** This laptop GPU (RTX 4060, 63 W cap) slows down under sustained
   load: enrichment ran at about 60 tokens/s at first and about 25 tokens/s after 15 minutes.

## 2. Configuration

`ragbench/config.yaml` is the single place for fairness-critical values. `cfg.hash(models, generation,
evidence, chunking, caps, answer)` (`contracts.RECORD_HASH_SECTIONS`; `answer` since 2026-09-16) is stamped on every
RunRecord.

| Key | Value | Meaning |
|---|---|---|
| `generation` | temperature 0, seed 42, top_p 1, top_k 1, num_ctx 8192, think off | every chat call of every approach |
| `generation.num_predict_answer` | 256 | the shared final answer call (every question type except timeline) |
| `answer.timeline_num_predict` | 768 | the shared answer call of timeline questions (prompt `answer-timeline-v1`, section 0d); never add keys to `generation` (caches key on it) |
| `evidence.max_chars` | 6000 | THE evidence budget (rendered block incl. `[n] title > path` headers) |
| `evidence.min_item_chars` | 200 | a cut item keeps at least this much text, else it is dropped |
| `evidence.use_doc_scope` | true | every approach receives the question's doc scope |
| `chunking` | 1000 / 200 overlap / top_k 8 | B2, B3 |
| `caps.tree_nav_max_calls` | 10 | B4, including routing, repairs and the answer call |
| `caps.grep_agent_max_steps` | 8 | B5, model calls including the final answer call |
| `caps.ours_max_calls` | 3 | OURS: plan + at most one of (language repair, widened plan) + answer |
| `caps.agentic_max_calls` | 8 | AGENTIC: plan (+1 repair) + verify batches + at most 1 gap batch + the answer call |
| `caps.agentic2_max_calls`, `caps.agentic2_max_seconds` | 4, 60 | AGENTIC2 (section 0e): plan + verify (1, at most 2) + answer; query clock per question (lazy first-touch embedding excluded). Added 2026-09-16: the records' config hash changed for AGENTIC2 records only (earlier records keep theirs) |
| `judge.*` | gemini-2.5-flash-lite, `max_calls_total` 10, `max_cost_usd` 0.10 | persisted budget in `work/judge_budget.json`; raise deliberately before real judging |
| `final.docs_from_questions` | true | the author's corpus decision; `run.py --final` requires `--docs-from-questions` |
| `final.variants` | the 11 CORE PLAN variants (section 0d) + AGENTIC2 (section 0e) | the only variants `run.py --final` accepts (see section 5a); not part of the config hash |
| `final.question_files`, `final.settings`, `final.exclude_variants`, `final.answer_prompts` | test_core/test_timeline (test_core.jsonl is not released); A 4 settings, B fa-fa+en-en, C fa-fa; no OURS/OURS+monolingual_patterns on test_timeline; answer-v2 + answer-timeline-v1 | the rest of the `--final` declaration (section 5a) |

Approach tunables that are not fairness-critical are `DEFAULTS` in each module (`approaches/baselines.py`,
`tree_nav.py`, `grep_agent.py`, `ours/defaults.py`) and are copied into each record's `trace.params`.

## 3. Approaches and variants

| CLI `--approach` | Variant | What it does | Query-time calls |
|---|---|---|---|
| `b0` | B0 | no retrieval, answers from memory | 1 |
| `b1` | B1 | gold evidence units (upper bound; the only class with `needs_gold`) | 1 |
| `b2` | B2 | BM25 (bm25s, Lucene) over 1000-char chunks, rank order | 1 |
| `b3` | B3 | bge-m3 cosine top-k over the same chunks | 1 chat + 1 embed |
| `b4` | B4 | PageIndex-style tree walk over titles, 30-word node summaries and subsection titles; rank-order evidence | 3-10 |
| `b5` | B5 | ReAct grep/read agent over the plain section store | up to 8 |
| `b5 --ablation agent_answer` | B5+agent_answer | the agent's own answer instead of the shared answer call (ablation only) | up to 8 |
| `ours` | OURS | enriched store (keywords/synonyms fa+en, questions, fact lines) + planned ripgrep | 2-3 |
| `ours --ablation no_enrichment` | OURS+no_enrichment | same planner over the plain store | 2-3 |
| `ours --ablation no_facts` | OURS+no_facts | enriched store without fact lines | 2-3 |
| `ours --ablation facts_bare` | OURS+facts_bare | fact items show the fact lines (each with <= 2 source lines), the dev_fix1 packing; full OURS shows each matched value's source text (`fact_evidence="context"`); same ranking | 2-3 |
| `ours --ablation no_normalization` | OURS+no_normalization | raw text store, patterns not normalised (enrichment still from normalised text) | 2-3 |
| `ours --ablation monolingual_patterns` | OURS+monolingual_patterns | patterns in the question language only (schema has only that list; the check requires only it) | 2-3 |
| `ours --ablation granularity=always_section` / `=always_window` | OURS+granularity=… | fixed evidence granularity | 2-3 |
| `ours --index-level L0\|L1\|L2` (= `--ablation index_level=L*`) | OURS+index_level=L* | cheap index levels (INTEGRATION_NOTES.md): L0 regex fact lines + title/path headers, no model; L1 + one bilingual glossary per document; L2 + batched keywords. L3 = OURS | 2-3 |
| `agentic` | AGENTIC | explicit budgeted pipeline: plan, ripgrep windows, batched verification with a verbatim code guard, 1 gap round, reconciliation table, shared answer (`RunRecord.structured.series`) | up to 8 |
| `agentic --ablation no_guard\|no_gap_fill\|line_windows\|no_doc_selection\|synth_answer` | AGENTIC+… | AGENTIC ablations (not in the core plan) | up to 8 |
| `agentic2` | AGENTIC2 | AGENTIC v2 (section 0e): v1 plan, lazy bge-m3 vectors of the target documents' tree leaves + tree expansion + an on-the-fly SQLite fact table (L0 regex facts), one (at most two) verify calls with the v1 guard, v1 reconciliation table + shared answer | 2-4 chat + 1 embed (+ first-touch embeds) |
| `agentic2 --ablation no_vectors\|no_factdb\|no_tree\|no_verify\|candidates=<n>` | AGENTIC2+… | AGENTIC2 dev ablations | 2-4 |

Flags combine: `--ablation no_facts,granularity=always_window`. OURS uses no BM25 and no embeddings.

**OURS plan contract (plan prompt `ours-plan-v3`, 2026-09-15).** The planner returns one pattern list per language,
question language first: `{"fa": [...], "en": [...], "scope": ..., "expect": ...}` (JSON schema). Code, not only the prompt,
enforces the languages: after post-processing (normalisation, joiner flex, ripgrep compile, too-broad filter) at least one
pattern must be in the **document** language and one in the **question** language (monolingual_patterns: the question
language only). A pattern's language is its script, not the list the model put it in, and an alternation that mixes scripts
(`price|قیمت`) is split per script. `caps.ours_patterns_max` (5) applies **per language list**, and the lists are interleaved
document language first. A cap shared by both languages would halve the terms a bilingual plan can search compared with plan-v2's
mixed-script alternations; in dev_fix1_attempt1 it cut a key term while keeping the report name. When the question has a
document scope, the plan prompt adds one sentence saying the search already runs inside the named report(s), so a pattern for the
report's own name is wasted. If the check fails, one **repair call** (`purpose=plan_repair`) shows the model its reply
and what is missing. The repaired plan keeps the first plan's patterns for the languages that passed and takes the repair's
only for the missing ones. The repair counts toward the 3-call cap, so a repaired question gets no widening call
(plan + repair + answer). The widening retry (nothing matched) is available only when no repair was used. The question-word
grep (no model call) remains the last resort. Every record's `trace.stats` has `plan_repair`, `plan_missing_lang`,
`plan_missing_lang_after_repair`, `widen`, `question_words_fallback` and `pattern_fallback` (0/1). `run_meta` sums them per
invocation (`trace_stats`), and aggregate reports their means as `ts_*` columns and in the "Approach counters" table.

**B4 (prompt `b4-2026-09-15.2`).**
- Summaries follow PageIndex. A section with subsections is summarised from its own text only (the "prefix summary").
  Groups and sections without own text are described from their children.
- Each navigation entry also lists its subsection titles, which is the tree structure PageIndex's tree search shows.
- Evidence is packed in the navigator's rank order: picks depth-first, and an opened section's own text right after its
  picked descendants.
- An item that does not fit is narrowed to its leading whole units, or line-cut when not even one unit fits. It is not
  dropped, and later smaller items still fill the remaining room.
- `trace.stats` has `repair_calls`, `fallback_steps`, `stop_steps`, `narrowed_items`, `dropped_items` and `call_cap`.
- Leaf summaries are reused from the cache (`SUMMARY_CACHE_VERSION`). Only parent, group and document summaries are
  re-generated.

## 4. Build indexes

Indexes are built automatically by `ragbench.run` before the first question, and are cached and resumable under
`work/indexes/<variant>/<set>-<fp8>/` (fingerprint = corpus documents + models + the approach's
`config_signature`). To build ahead of time (for example overnight) without answering:

```bash
Q=ragbench/data/questions/pilot.jsonl
for ap in b2 b3 b4 b5 ours; do
  $PY -m ragbench.run --approach $ap --questions $Q --corpus A --out runs/full --index-only
done
```

Shared caches make rebuilds cheap. A rebuilt index then reports few calls in `build_stats.json`, and aggregation
reports the cache-inclusive cost in the `index_full_*` columns:
- B3 vectors: `work/indexes/_embed_cache` (per text + model digest)
- B4 summaries: `work/indexes/B4/_summary_cache.sqlite`
- OURS enrichment: `work/indexes/OURS/_enrichment/<model>-<prompt>.sqlite`, shared by every OURS ablation
- OURS index levels: `work/indexes/OURS/_levels/<model>-ours-glossary-v2.sqlite` (L1) and `...-ours-kwbatch-v1.sqlite` (L2)
- AGENTIC model calls of unfinished questions: `work/runs/<out>/agentic_calls_<variant>.sqlite` (per run directory)

`--dry-run` prints, per corpus set, the index directory a run would use and whether it is complete, plus one
`DRYRUN-JSON` line (planned/todo per setting and answer format, sets with doc ids). `index_<variant>_<set>.json` copies
carry `index_cost` (the per-stage cost the OURS levels write; None for other approaches).

`--index-only` needs question-supported sets. It builds the sets for the chosen corpus: both languages for A and B, `C` for C.

## 5. Run

```bash
Q=ragbench/data/questions/pilot.jsonl
# one approach, one corpus, all four settings (question language - document language)
$PY -m ragbench.run --approach b3 --questions $Q --corpus A --settings fa-fa,en-en,fa-en,en-fa --out runs/full
# an ablation
$PY -m ragbench.run --approach ours --ablation no_facts --questions $Q --corpus A --out runs/full
# corpus C is Persian-only (fa-fa); B has all four settings
$PY -m ragbench.run --approach ours --questions $Q --corpus C --settings fa-fa --out runs/full
# plan only
$PY -m ragbench.run --approach b5 --questions $Q --corpus B --out runs/full --dry-run
```

Useful options:
- `--ids A-rpt-a-01,A-rpt-a-02`: restrict to these questions. `--limit N`: the first N questions.
- `--docs rpt-b,rpt-f`: restrict the corpus to these documents (keys, both languages). Indexes are built over
  the subset only, and questions whose doc scope is `corpus` or outside the subset are skipped. This is for smoke tests, dev
  checks and cost estimates. With a subset, BM25 IDF and B4/B5 routing see only the subset.
- `--docs-from-questions`: **the corpus restriction the author decided for the study.**
  - Every approach's corpus set is restricted to the documents the question file references: every `doc_scope` document
    and the document of every gold unit. For A and B that means both language versions of each referenced document.
  - All questions of the file count, not only `--ids`/`--limit`, so the corpus does not change with a subset of questions.
  - The restriction is applied in `run.py` before any index is built, identically for every approach.
  - Questions with `doc_scope: corpus` still run, over the restricted corpus.
  - It is stamped in `run_meta_*.json` (`docs_restriction`: mode, keys, per-set doc ids), in `index_*_*.json`, and on
    every record (`meta.docs_restriction`).
  - A comma list of other question files may be given instead (`--docs-from-questions a.jsonl,b.jsonl`), but not with
    `--final`. Combined with `--docs`, the intersection is used.
  - **Consequences to state in the paper:** BM25 IDF (B2), B4 corpus routing and B5 corpus-wide search depend on the
    restricted set. They see only the documents that have questions, and so does the corpus-wide search of OURS.
  - Index fingerprints follow the document set. A different question file (dev vs test) therefore gives different index
    directories. The per-section caches (OURS enrichment, B4 summaries, B3 embeddings) are shared, so the rebuild is cheap.
- Resume: rerun the same command. Finished `(question, variant, setting)` keys are skipped and errored ones are retried.
  Resume refuses (exit 2) when:
  - the config hash, index fingerprint or a question's text/scope changed;
  - any record or invocation in the output directory, of any variant, was answered under another Ollama version.
  
  Use a new `--out`, or `--allow-mixed` if you really mean it. With `--no-health` the version is not checked (a warning).
- `--no-warmup`: skip the untimed warm-up call, which is logged as phase `smoke` and excluded from question latency.

Outputs in `work/runs/<name>/`:
- `runs_<variant>.jsonl`: RunRecords. Each carries `split` and `meta` (`ollama_version`, `questions_file`, `final`,
  `docs_restriction`).
- `ledger_<variant>.jsonl`: every model call (phase index|query|smoke, purpose, tokens, wall/wait/load ms).
- `index_<variant>_<set>.json`: build stats plus the cache-inclusive cost.
- `run_meta_<variant>.json`, per invocation: arguments, digests, Ollama version, config hash, question-file hash, approach
  source sha1, split, final flag, docs restriction, and `trace_stats` (sums and rates of the approach's `trace.stats`
  counters).

### 5a. Dev/test protocol

- **DEV = `ragbench/data/questions/pilot.jsonl`.** All tuning, prompt changes and diagnosis happen on it.
- **TEST = `ragbench/data/questions/test_*.jsonl`** (being written), later merged into `test.jsonl`.
- The split comes from the question file name: `pilot*`/`dev*` → `dev`, `test*` → `test`, anything else → `other`. It is
  stamped on every RunRecord (`split`).
- `ragbench.eval.aggregate` groups every table and every paired comparison by split, so dev and test are never pooled.
  - Older records without a split take the split of the question file that holds their question.
  - `--split dev|test` filters.
- **Guard:** `run.py` refuses (exit 2) to run a `test*` question file unless `--final` is passed.
- `--final` refuses anything not declared in `config.yaml` `final`:
  - a variant not in `final.variants`;
  - a question file whose name is not in `final.question_files`, or a variant listed for that file in
    `final.exclude_variants`;
  - `--ids`, `--limit`, `--docs` or `--approach-class`;
  - `--allow-mixed` or `--no-health`;
  - settings outside `settings`, or outside the invocation corpus's list in `final.settings` (a corpus without an
    entry runs nothing; one invocation may run any subset of its corpus's list, which lets the core driver run one
    setting block at a time);
  - answer prompt versions in `contracts.ANSWER_PROMPT_VERSIONS` that differ from `final.answer_prompts`;
  - a config other than `ragbench/config.yaml` (RAGBENCH_CONFIG or overrides);
  - a missing `--docs-from-questions` (required because `final.docs_from_questions: true`).
- Approach tunables live in code (`DEFAULTS`, prompt versions) and are recorded per invocation by the approach source
  sha1. Freeze the code before the test run and do not change it between test invocations; resume warns when the source
  changed.

```bash
T=ragbench/data/questions/test.jsonl
for c in A B; do for ap in b0 b1 b2 b3 b4 b5 ours; do
  $PY -m ragbench.run --final --docs-from-questions --approach $ap --questions $T --corpus $c --out runs/test; done; done
$PY -m ragbench.eval.aggregate --runs runs/test --questions $T --split test --judge none
```

Full experiment on the test split, one variant at a time (sequential, one GPU; see 5a):

```bash
Q=ragbench/data/questions/test.jsonl
F="--final --docs-from-questions"
for c in A B; do for ap in b0 b1 b2 b3 b4 b5 ours; do
  $PY -m ragbench.run $F --approach $ap --questions $Q --corpus $c --out runs/full; done; done
for ap in b0 b1 b2 b3 b4 b5 ours; do $PY -m ragbench.run $F --approach $ap --questions $Q --corpus C --settings fa-fa --out runs/full; done
for ab in no_enrichment no_facts no_normalization monolingual_patterns granularity=always_section granularity=always_window; do
  for c in A B; do $PY -m ragbench.run $F --approach ours --ablation $ab --questions $Q --corpus $c --out runs/full; done
  $PY -m ragbench.run $F --approach ours --ablation $ab --questions $Q --corpus C --settings fa-fa --out runs/full
done
```

The smoke run is scripted in `ragbench/tools/smoke_run.sh` (logs in `work/runs/smoke/logs/`). To exercise the paid judge
on a few records only:

```bash
$PY -m ragbench.tools.judge_sample --runs runs/smoke --out runs/smoke_judge_sample \
  --pick B1:B-digikala-04:en-en --pick OURS:C-divar-01:fa-fa ...          # approach:question_id:setting
$PY -m ragbench.eval.aggregate --runs runs/smoke_judge_sample --questions $Q --judge openrouter \
  --judge-max-calls 8 --judge-max-cost 0.01 --out runs/smoke/results_judged
```

### 5b. Timeline questions (series answers, DEV: `pilot_timeline.jsonl`)

A timeline question asks for a metric across several reports, for example how many people worked at Divar from 1397
to 1400 and how the number changed. The gold answer is a series: one value per period, each tied to its report and to
the units that state it. This is the question type the agentic system targets. Schema and scoring are documented in
`ragbench/eval/timeline.py`, the authoring YAML in `ragbench/tools/timeline_questions.py`.

- **Schema.** A normal Question line with `type: timeline` and `answer_type: series`. The series lives in `meta`:
  - `answer_series`: period, value, unit, `value_base`, the per-period definition (fa/en), `doc`, `gold_units`, and
    `context_units` (labels of detached numbers, definition sentences);
  - `comparable` (false when the definition or scope changes);
  - `series_notes_fa` / `series_notes_en` and `comparability_cues`;
  - `reconciliation`.

  `gold_units` is the union of every period's gold and context units, so B1 and evidence recall work unchanged.
  `answer_fa` / `answer_en` hold a short prose answer.
- **Reconciliation rule.** The report OF a period gives the gold value. A later report's different number for that
  period is a *restatement*: listed, described in the notes, never gold. A period without its own report in the
  corpus may be *supplementary* (optional, taken from a later report). Example: Tipax's report named 1400 reviews 1399,
  so the period is 1399.
- **Build and validate.**
  ```bash
  $PY -m ragbench.tools.questions build --src ragbench/data/questions/pilot_timeline.src.yaml --out ragbench/data/questions/pilot_timeline.jsonl
  $PY -m ragbench.tools.questions validate --path ragbench/data/questions/pilot_timeline.jsonl --strict   # 8 questions: 0 errors, 0 warnings
  $PY -m ragbench.tools.questions summary --path ragbench/data/questions/pilot_timeline.jsonl             # series tables
  ```
  Validation checks the following:
  - periods are unique and in order;
  - each period's report is in `doc_scope`;
  - every gold unit exists, sits in that report and contains the value at the precision it is written;
  - the gold union holds;
  - comparable is a boolean, and non-comparable series have notes;
  - restatements come from another report;
  - the reference answers parse back to the gold series (and flag the change when needed);
  - no gold value leaks into the question;
  - the composition: 8 questions, at least 2 non-comparable, at least 1 with a restatement.
- **`contracts.py` lists `series` / `timeline`** (the INTEGRATION_NOTES patch is applied), so `ragbench.run` runs these
  questions. They get the shared timeline answer prompt (section 0d). Example:
  ```bash
  Q=ragbench/data/questions/pilot_timeline.jsonl
  $PY -m ragbench.run --approach ours --questions $Q --corpus C --settings fa-fa --docs-from-questions --out runs/dev_timeline
  ```
  15 of its 16 reports are not in the pilot restriction: 464 sections, about 486 OURS parts, about 1.2 GPU-hours of
  enrichment.
- **Scoring** (`eval/timeline.py`, per record; `match_correct` = `tl_series_correct`):
  - **Structured output.** When a record carries structured output (`record.structured.series`, a RunRecord field
    since 2026-09-16, or `trace.structured.series`: items with period, value, unit, definition, quote, unit_id,
    doc_id), it is scored directly.
  - **Text answers.** Otherwise the answer text is parsed into (period, value) pairs. The parser handles:
    - Persian digits, `٫` decimals, `/` decimals (`%23/17`), and `,` `٬` `،` thousands separators;
    - dot-grouped thousands: two or more dots with 3-digit groups (`4.318.701`, `۴٫۳۱۸٫۷۰۱`); one dot stays a decimal
      (fix of 2026-09-16: `4.318.701` used to read as 4.318);
    - durations: a value followed by a duration word (`3 سال`, `12 hours`) is dropped, unless the gold unit is that
      duration, so a series in hours or days scores from text (fix of 2026-09-16; it used to score zero);
    - scale words (هزار / میلیون / میلیارد / همت, thousand / million / billion, `44M`, `1.2bn`);
    - composites (`18 میلیون و 519 هزار`, `18 هزار و 800`), number words (`چهل‌میلیون`) and percent before or after;
    - Jalali and Gregorian years, and `سال 98`.

    A report's own year ("the 1400 report") is a source, not a period. Values are paired with the nearest period in
    the segment's reading direction (`1398: 35M` or `35M in 1398`, tables, "به ترتیب" lists).
  - **Metrics.**
    - `tl_coverage` and `tl_precision`. Supplementary periods count as correct. Periods outside the asked range are not
      counted.
    - `tl_f1` and `tl_series_correct`.
    - `tl_comparability_flag`, for non-comparable series. It counts a structured `comparable: false`, a flag phrase, or
      a question cue.
    - `tl_false_trend`: a trend stated without the flag.
    - `tl_restated_used`: the later report's number given for a period.
    - `tl_citation_correct`: the cited unit states the value.
  - **Tolerance.** 2%, or a rounding at the written precision when the number has at least 2 significant digits.
- **Aggregate.**
  - `timeline.csv` has the tl_* metrics per split × variant × corpus × setting, plus calls, tokens, latency p50/p95 and
    no-load p50.
  - `results.md` gets a "Timeline questions" table per split and corpus.
  - `cost_by_qtype.csv` and the "Cost by question type" table give cost and accuracy per question type (lookup,
    timeline, ...).
  - `results.csv` rows gain `n_timeline` and tl_* means.
  - `comparisons.csv` gains `tl_coverage` and `tl_series_correct`.
- **Judge.** With `--judge openrouter`, series answers use the `judge-series-v2` rubric. It returns strict JSON with
  per-period correctness, `flags_definition_change` and `false_trend`, stored as `tl_judge_*`.
  - v1 was probed once on 2026-09-16 and missed a false trend. v2 has not been probed live, because the workflow judge
    budget is at 10/10.
  - Raise `judge.max_calls_total` first, then run `$PY -m ragbench.eval.judge --probe-series`. It costs 1 call and
    exits 0 when the rubric catches the false trend.

## 6. Judge and aggregate

```bash
# match metrics only (free): results.csv, results.md, comparisons.csv, scored.jsonl, cost_by_qtype.csv
# (+ timeline.csv when timeline questions are present, see 5b)
$PY -m ragbench.eval.aggregate --runs runs/full --questions $Q --judge none
# OpenRouter judge: an explicit per-invocation cap is required
#   (raise judge.max_calls_total / max_cost_usd in config.yaml first; judge in blocks, the verdict cache keeps finished work)
$PY -m ragbench.eval.aggregate --runs runs/full --questions $Q --judge openrouter --judge-max-calls 1000 --judge-max-cost 0.08 --variants B0,B1,B2,B3,B4,B5,OURS --out runs/full/results_judged
# sanity checks over records + scored metrics (call caps, ledger consistency, budget, document language, B1 >= B0)
$PY -m ragbench.eval.sanity --runs runs/full --results runs/full/results
```

- Rule verdicts cost nothing: a run error or empty answer is wrong, and an exact NOT_FOUND is correct only for unanswerable questions.
  Verdicts are cached in `work/judge/verdicts.sqlite`, keyed by question, gold and normalised answer, so re-aggregation is free and
  identical answers across approaches are judged once.
- Records are judged in (corpus, question, setting, approach) order, so a partial budget still covers complete paired groups.
- Before real judging, raise `judge.max_calls_total` and `judge.max_cost_usd` in `config.yaml`. They cap the whole workflow and are
  persisted in `work/judge_budget.json`. Set them deliberately.
- `--common-only` keeps only (question, setting) keys every variant answered. `--corpus`, `--approaches` (a base name includes its
  ablations), `--variants` (exact variant names) and `--split dev|test` filter.
- Tables are per split and corpus (`## Corpus A, dev split`). The pooled table has an `Abstain %` column.
- The "Approach counters" table shows the means of each approach's `trace.stats`, which are rates for 0/1 counters: OURS
  `plan_repair`, `widen` and so on, B4 `narrowed_items`, `repair_calls` and so on. They are also `ts_*` columns in
  results.csv.
- Aggregate warns when records hold several Ollama versions. Sanity FAILs on more than one version (`records.single_ollama_version`)
  and WARNs on records without a split or version stamp.
- `--judge ollama` is a WEAKER, same-model fallback and is labelled as such. Do not use it as the paper's judge.

## 7. Cost estimate for the full run

Measured on this machine in the smoke run, then scaled. **Caveat:** the laptop GPU throttles under sustained load (power cap),
so per-call times varied by 2x during the smoke run (enrichment 9–20 s per call). Treat the figures as ±50%.

**Index time (one-off; cached and resumable):**

| Index | Measured unit cost | Full corpora (A 1914 sections / 4.08M chars; B 1220 / 0.64M; C 8397 / 4.76M) | GPU time |
|---|---|---|---|
| B2 BM25 | 0.7 s CPU for 0.58M chars | seconds | – |
| B3 bge-m3 | 33 s per 0.58M chars (217k embed tokens) | ~9.5M chars | ~10 min |
| B4 summaries | 1.07 calls per section, 1.27 s per call | ~9,700 calls (A ~2,050, B ~1,300, C ~6,360 with grouped flat trees) | ~3.5–5 h |
| B5 plain store | seconds (C cold-store warm-up ~76 s once) | – | – |
| OURS enrichment | 1 call per section part (max 6000 chars) + 2.4% repairs. A: 16.4 s/call, ~680 completion tokens (throttled); B/C: 9.0 s/call, ~500 tokens | A 2,013 parts, B 1,003, C 6,670, i.e. ~9,900 calls | A ~9.4 h, B ~2.5 h, C ~17–30 h, **~30–40 h in total** |

OURS ablations reuse the enrichment cache: no_facts, monolingual_patterns and granularity add 0 calls, no_enrichment needs none,
and no_normalization reuses it by default. Enrichment for C dominates the whole experiment. Decide before the full run whether C is
enriched in full or restricted to the documents questions use (report it either way).

**Query time:** ~300 questions × 4 settings = 1,200 (question, setting) pairs per variant (an upper bound, because C questions
only run fa-fa). Mean seconds per record, measured on corpus A: B0 0.2, B1 1.1, B2 1.9, B3 2.1, B4 3.4, B5 17.9, OURS 2.9,
OURS ablations ~2.3.

| Variant set | Seconds per pair | 1,200 pairs |
|---|---|---|
| B0–B5 | 26.6 | ~8.9 h (B5 alone ~6 h) |
| OURS | 2.9 | ~1 h |
| 6 OURS ablations | ~14 | ~4.7 h |
| **Total** | ~43.6 | **~15 h** (plus index loading and warm-ups) |

**Whole experiment:** ~50 GPU-hours (~35–45 h of index building, ~15 h of queries), strictly sequential.
Run it overnight in blocks, one approach per invocation. Everything resumes.

**OpenRouter judge:**
- In the smoke run, 42% of records needed a model call. The rest were rule verdicts on an exact NOT_FOUND (52%) or identical answers
  hitting the cache (6%).
- The measured cost is about $0.00007 per call.
- Full run: 13 variants × 1,200 records = 15,600 records, about 6,500 judge calls, about $0.45–0.60.
- The key had about $0.54 left before this workflow (it spent $0.00062). **That is not enough headroom to judge every
  ablation.** Either top up the credit, or judge B0–B5 and OURS only (8,400 records, about 3,500 calls, about $0.25) and report ablations
  by match accuracy.
- In either case raise `judge.max_calls_total` (e.g. 8000) and `judge.max_cost_usd` (e.g. 0.60) first. Judge in blocks with
  `--judge-max-calls`. The verdict cache keeps finished work.

## 8. Known limitations and decisions to state in the paper

Environment and reproducibility:
- **Model tag.** The installed and pinned chat model is `gemma4:e4b-it-qat` (QAT, Q4_0), not the plain `gemma4:e4b`. Name that tag in the
  paper, or pull the plain tag and rebuild every cache (all caches are keyed by model and digest).
- **Ollama updates.** An update is pending (0.34.0). Freeze the version for the whole experiment. Every RunRecord carries
  `meta.ollama_version` (since the dev_fix1 revision of 2026-09-15; smoke records do not). `run.py` refuses to add records to an output directory
  that holds records or invocations answered under another version, whatever the variant.
- **Latency.** The GPU throttles, so latency and index times vary by up to 2x with thermal state. Report medians, report the hardware and
  power mode, and never run other GPU jobs. `lock_wait_ms` shows waits behind other processes (0.7 ms max in the smoke run).
- **Extraction.** Re-running it changes B/C unit ids. It re-ran during the smoke run. The smoke questions were unaffected, but
  C-shaparak-01 needed relinking, which is done: `pilot.jsonl` was rebuilt and `validate --strict` passes. Rebuild the questions after
  any extraction run.
- **Subsets.** `--docs` subsets change BM25 IDF and B4/B5 routing. Smoke numbers are not comparable with full-corpus numbers.
- **Corpus = documents with questions.** The study indexes and searches only the documents the question file references
  (`--docs-from-questions`, required by `--final`). BM25 IDF (B2), B4 corpus routing and B5/OURS corpus-wide search all
  depend on that restricted set, so state the document counts per corpus in the paper.

Fairness and design choices reviewers will ask about:
- `doc_scope` goes to every approach (`evidence.use_doc_scope`). Every pilot question also names its report.
- Evidence packing order is each approach's own:
  - B2/B3: rank order, filling the budget beyond top_k while chunks fit whole.
  - B4: rank order, narrowing an oversized item to its leading whole units. It was fit-first before prompt b4-2026-09-15.2.
  - B5: cited-first.
  - OURS: budget-aware selection.
  
  The truncation rule is shared.
- **Dev/test split.** All tuning happened on `pilot.jsonl` (dev). Test files run only under `--final`, and results are
  reported per split.
- B5's paper number uses the shared answer call. `B5+agent_answer` is an ablation.
- `OURS+no_normalization` still uses enrichment computed from normalised text (DEFAULTS
  `no_normalization_raw_enrichment=False`).
- The OURS question-word fallback is a program-only step beyond the brief, logged as `trace.evidence_from=question_words`. It never
  fired in the smoke run: full OURS always found matches with its first plan.
- B4 keeps a document root whole when navigation fails, which gives it a free lead-of-document fallback, logged in the trace.
- Index cost for cache-reusing variants must come from the `index_full_*` columns (cache-inclusive), not from `build_stats` totals.
  The readers of those approach-specific report files (`summary_stats.json`, `index.json`, `vector_meta.json`) are not a formal contract yet.

Metrics:
- **Match accuracy is conservative.** Short or span answers must contain the whole gold string. In the smoke run all four approaches
  failed the multi-part B-digikala-04 en-en answer by match, while the judge accepted it. Make the judge the headline.
- **Numeric match is lenient.** It accepts any number in the answer that equals the gold, including rounding at the written precision
  (`81` matches 81.3), so a list of numbers can match by luck.
- **The judge is generous with `partially_correct`.** It gave 16.7% vs 16% and a 700–900 kWh/t range vs 1,088 kWh/t "partial", so use strict
  `acc_judge`.
- **Evidence recall is lenient.**
  - Chunk, window and section unit_ids count partly shown units.
  - Split tables credit any part.
  - Cut items count a unit when ≥ 60 of its characters survive (exact for verbatim items since this integration; a heuristic
    otherwise).
  - Report answer accuracy alongside recall, and consider the text-level "all facts covered" recall the questions agent proposed
    (`meta.evidence` quote terms).
- **Lexical bias.** Pilot questions echo their gold text (about 0.70 content-word overlap). `meta.lexical_overlap` is recorded so you
  can stratify by it.

Model behaviour seen in the smoke run (genuine findings, not bugs):
- **Planner.** The 4B planner misspells Persian terms (one letter dropped from a key term). For a Persian question it wrote the same Persian-only
  patterns whether the documents were fa or en, so fa-en lost the gold for A-rpt-f-01 despite the bilingual instruction.
  On 2026-09-15 (dev_fix1) the per-language plan contract and the code-enforced language check (section 3) fixed the language
  failure: every dev plan now carries document-language patterns. A-rpt-f-01 fa-en still misses gold because the plan omits
  the key term in both languages. Question-language terms are now copied correctly, but translated terms are still sometimes misspelled or
  mistranslated.
- **Answer format.** OURS answers sometimes copy the fact-line format (`24 | % | -`). One answer out of 81 non-abstaining ones used the
  wrong language (an English question answered in Persian).
- **B4 abstains** on 69% of records although gold is in its evidence for 58%: whole sections ordered fit-first bury the fact.
  After dev_fix1 (prefix summaries, subsection titles, rank-order packing), evidence hit is 92% but abstention is still 69% (4 of
  its 16 records are the unanswerable question). In 6 of the 7 abstentions on answerable records the gold unit was shown, usually
  a chapter's own text that states the fact in passing. The rest is navigation quality and the 4B answer model, not packing.
- **B5 is the slowest** (18 s per record, 3 of 22 records hit the 8-call cap).
- **B2 fails cross-lingually** (evidence hit 0% in fa-en and en-fa), as expected for lexical matching.
- **B0 never answered from memory** (all NOT_FOUND). The A reports are from 2026.
- Answers append citation markers (`[1]`); eval strips them.

Open engineering items:
- `OpenRouterClient.chat` cannot pass `response_format`, so JSON verdicts are parsed leniently (8/8 parsed in the smoke run).
- The util.FileLock GPU lock polls without a queue. That only matters if two processes compete, which the one-GPU rule forbids.
- Stale directories are safe to delete by hand:
  - `work/indexes/B2/A_fa-8d6ba074` and `A_en-cb1b806d` (tokenizer v1)
  - `work/indexes/OURS/_smoke_offline` and the 8-section chapter-subset indexes `OURS/A_fa-7458b286` and `A_en-3e84c2a6`
  - `work/test_baselines/`
  - the review run files in `work/runs/*.jsonl`
