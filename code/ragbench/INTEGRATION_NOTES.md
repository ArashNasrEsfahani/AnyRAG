# Integration notes

Patches that agents could not apply themselves because the files are frozen or owned by someone else. Each section
names the files, the exact change and why.

**Status (2026-09-16, CORE PLAN integration): every patch below is APPLIED**. The exceptions are the `final.variants`
lists proposed in "OURS Patch 3" and "AGENTIC Patch 3b". `final` now declares exactly the core plan: B0-B5,
OURS+index_level=L0, OURS+index_level=L1, OURS, AGENTIC and OURS+monolingual_patterns, with per-corpus settings,
question files and exclusions. See OWNERSHIP.md "CORE PLAN integration" and RUNBOOK section 0d.

Beyond these patches, the integration added:
- one shared timeline answer prompt and budget for every approach (`answer-timeline-v1`, `answer.timeline_num_predict`);
  the AGENTIC answer call uses `contracts.answer_num_predict`;
- two parser fixes in `eval/timeline.py`: dot-grouped thousands and duration-unit series.

## OURS index levels L0 / L1 / L2 (ours agent, 2026-09-16)

The index levels exist to compare cheap OURS indexes against the full enrichment (L3), as accuracy against index
cost. L3 does not change: its prompt, cache key and file, store format, signature, variant name (`OURS`) and index
directories are the same. Before and after the `approach.py` edit, a dump of variant, signature, fingerprints, index
dirs, cache path, plan messages and options for all 9 existing flag combinations was byte-identical. The OURS smoke
test gives 97 checks and 0 failures before and after.

**Usable now from Python:** `get_approach("OURS", flags={"index_level": "L1"})`. The only missing pieces are the CLI and
the registry.

### What each level stores and how the query side uses it

The query side is the same at every level: the plan prompt `ours-plan-v3`, the executor, `caps.ours_max_calls` = 3 and
the shared answer call.

| Level | Index-time model calls | Store per section, in the same format as L3 |
|---|---|---|
| L0 | none | `@@TITLE` holds the section title, and `@@PATH` holds the document title followed by the section path, both normalised. Deterministic fact lines `metric \| value \| unit \| period` are extracted by regex from the section's own text and tables (`lite.extract_facts`, `ours-factx-v1`), and the executor ranks, selects and packs them exactly like L3 fact lines. |
| L1 | 1 glossary per document, plus at most 1 repair | L0, plus `gl_fa:` / `gl_en:` header lines. They hold the entries of the document's bilingual glossary (`ours-glossary-v2`, up to 30 terms) whose document-language term or synonym occurs in that section, at most 8 per section, best first. An entry found in more than 40% of a document's sections is not projected (`ours-glproj-v2`), because a header hit on nearly every section carries no ranking signal. v1 of the prompt picked generic words such as "rpt-c", "technology" and "data" in the live check. |
| L2 | L1, plus keyword batches of up to 6 sections or 5000 input chars, with at most 1 repair batch for missing sections | L1, plus `kw_fa` / `kw_en` / `syn_fa` / `syn_en` header lines, as L3 writes them (without L3's questions). |
| L3 | 1 call per section part, plus repairs | unchanged |

**Why the glossary is projected onto section headers rather than given to the planner:**
- The plan prompt, its tokens and the 3-call budget stay identical across levels, so the comparison isolates index cost.
- Every pilot question is doc-scoped. If the whole glossary sat on every header of a document, every section of that
  document would get the same header hit, which gives no ranking signal.
- Projected, a pattern in the other language reaches exactly the sections that use the term, as L3's `kw_en` lines do.

**Why the document title goes on `@@PATH` rather than into a header line.** `@@PATH` is weighted 0.75, while header
lines are weighted 3. At weight 3, a pattern hitting the document title (a year, or the company name) would lift every
section of the document above sections with a real body hit.

### Patch 1: `ragbench/approaches/__init__.py` (frozen registry) — APPLIED 2026-09-16

```python
OURS_FLAGS = ("no_enrichment", "no_facts", "no_normalization", "monolingual_patterns", "granularity", "facts_bare",
              "index_level")
# index_level (2026-09-16): L0 | L1 | L2 cheap index levels (approaches/ours/levels.py); takes a value like granularity
# ("index_level=L1" -> variant "OURS+index_level=L1"); L3 = the full enrichment = the default (flag dropped, variant "OURS").
```

`ours/smoke_test.py` already accepts the registry with or without `index_level`.

### Patch 2: `ragbench/run.py` — APPLIED 2026-09-16 (a-d)

**(a) Flag parsing.** Put the constant next to `GRANULARITY`, and add the `index_level` branch to `parse_flags`. The
live check ran exactly this code, through a monkeypatch.

```python
OURS_INDEX_LEVELS = ("L0", "L1", "L2", "L3")   # OURS index levels (approaches/ours/levels.py); L3 = the full enrichment


def parse_flags(values, approach):
    flags = {}
    known = B5_FLAGS if approach == "B5" else OURS_FLAGS
    for v in values or []:
        for part in str(v).split(","):
            part = part.strip()
            if not part:
                continue
            k, val = (part.split("=", 1) + [None])[:2] if "=" in part else (part, None)
            k = k.strip()
            if k not in known:
                raise SystemExit("unknown ablation %r for %s; choose from %s" % (k, approach, known))
            if k == "granularity":
                if val not in GRANULARITY:
                    raise SystemExit("granularity must be one of %s" % (GRANULARITY,))
                flags[k] = val
            elif k == "index_level":
                lv = str(val or "").strip().upper()
                if lv not in OURS_INDEX_LEVELS:
                    raise SystemExit("index_level must be one of %s" % (OURS_INDEX_LEVELS,))
                if lv != "L3":                      # L3 is today's OURS: no flag, same variant name and index
                    flags[k] = lv
            else:
                if val not in (None, "true", "True", "1"):
                    raise SystemExit("ablation %s takes no value" % k)
                flags[k] = True
    if flags and approach not in ("OURS", "B5"):
        raise SystemExit("--ablation applies to --approach ours (and b5 agent_answer) only")
    return flags
```

**(b) Optional CLI shorthand.** Add this in `build_parser()`:

```python
    ap.add_argument("--index-level", default=None, choices=list(OURS_INDEX_LEVELS),
                    help="OURS index level, the same as --ablation index_level=<L>: L0 regex fact lines only, L1 + one "
                         "bilingual glossary per document, L2 + batched section keywords, L3 the full enrichment (default)")
```

Then add this in `main()`, right before `flags = parse_flags(args.ablation, name)`:

```python
    if args.index_level:
        args.ablation = list(args.ablation) + ["index_level=%s" % args.index_level]
```

**(c) Help text.** The `--ablation` help string gains `, index_level=L0|L1|L2|L3`.

**(d) Per-stage index cost in the copied index stats.** This is for the accuracy-vs-index-cost plot. In the
`finally:` block of the index loop, extend `st.update(...)` with:

```python
                          index_cost=(read_json(d / "index.json") or {}).get("index_cost"),
```

The key is `None` for every approach that writes no `index_cost`. The levels write the full breakdown, described under
"Index cost" below.

### Patch 3: `ragbench/config.yaml`, `final.variants` — SUPERSEDED by the core plan (L0 and L1 only; no L2, no other ablations)

Only when the levels belong to the frozen test run:

```yaml
  variants: ["B0", "B1", "B2", "B3", "B4", "B5", "OURS",
             "OURS+no_enrichment", "OURS+no_facts", "OURS+no_normalization", "OURS+monolingual_patterns",
             "OURS+granularity=always_section", "OURS+granularity=always_window",
             "OURS+index_level=L0", "OURS+index_level=L1", "OURS+index_level=L2"]
```

No new config keys are needed:
- The level settings are approach tunables. They live in `levels.LEVEL_DEFAULTS`, are not fairness-critical, and are
  part of each level's index fingerprint through `level_signature`.
- The query side and its caps do not change, so the config hash does not change either.

### Patch 4 (optional; `eval/` is owned by the eval agent)

No eval change is required:
- `eval.aggregate.cached_index_cost` already reads `index.json:enrichment.cost_all_cached_calls`, which every level
  writes.
- The `index_full_*` columns are therefore cache-inclusive for L0, L1 and L2, just as for L3.

For the plot, `load_index_stats` rows carry `index_cost` after Patch 2d. Useful columns are `index_cost.s_per_section`,
`s_per_document`, `calls_per_section` and `llm_calls`. For L3, use `index_full_model_ms / sections`.

### Index cost (per level `index.json`) and ledger

`index.json:index_cost` holds:
- **Identity:** `index_level`, `ollama_version` (this build), and `cached_result_ollama_versions` (version → count of the
  cached glossary and keyword results used).
- **Size:** `docs`, `sections`.
- **Cache-inclusive model cost:** `llm_calls`, `prompt_tokens`, `completion_tokens`, `llm_wall_ms`, `calls_per_section`,
  `s_per_section_model`, `s_per_document_model`.
- **This build:** `new_llm_calls`, `cache_hits`, `cache_misses`, `program_ms`, `build_wall_ms`,
  `s_per_section_program`, `s_per_document_program`.
- **Totals:** `s_per_section`, `s_per_document`.
- **`by_stage`:** one entry each for `facts`, `digest`, `glossary`, `projection`, `keywords` and `store_write`.
  - Each has this build's `wall_ms`, `s_per_section` and `s_per_document`.
  - The model stages also have `calls`, `prompt_tokens`, `completion_tokens`, `model_wall_ms`, `cache_hits` and
    `cache_misses`.
  - The model stages also have cache-inclusive `cached_calls`, `cached_prompt_tokens`, `cached_completion_tokens`,
    `cached_model_wall_ms`, `s_per_section_cached_model` and `s_per_document_cached_model`.

**Ledger:** each model call has its own purpose: `glossary`, `glossary_repair`, `kw_batch` or `kw_batch_repair`. Each
stage of each document also gets one line, with `kind: "stage"`, `phase: "index"`, `purpose: "index_stage"`,
`stage`, `index_level`, `doc_id`, `wall_ms`, counts, and `ok: true`. These lines carry no chat kind, so the call,
token and failure sums in `LedgerScope`, `sanity` and `dev_progress` are unchanged. Stages that finished before an
interruption stay logged.

**Crash safety:** every glossary and every section's keywords are committed to KVCache right after the call. The cache
lives at `work/indexes/OURS/_levels/<model>-ours-glossary-v2.sqlite` and `...-ours-kwbatch-v1.sqlite`, keyed by content
hash, model, digest, generation options, prompt version and caps. A glossary whose first reply was unusable is
checkpointed before its repair call. A restart skips everything cached. Stores are written into `store.partial` and
swapped in. A transport failure (`ModelError` after the client's retries, for example an Ollama restart) is never
cached.

### Driver lines (`tools/dev_full.sh` style)

```bash
# index (A 1,914 sections: L0 seconds of CPU; L1 ~20 glossary calls for A; L2 ~ sections/5 batch calls)
for lv in L0 L1 L2; do for c in A B C; do
  step index_ours_${lv}_$c --approach ours --ablation index_level=$lv --corpus $c --settings "$(settings_for $c)" --index-only
done; done
# query
for c in A B C; do for lv in L0 L1 L2; do
  step q_ours_${lv}_$c --approach ours --ablation index_level=$lv --corpus $c --settings "$(settings_for $c)"
done; done
```

### Tests and runbook

- **Offline tests.** `$PY -m ragbench.approaches.ours.levels_test` gives 72 checks and 0 failures. It covers L0 fact
  extraction in Persian and English, table rows and detached numbers; L1 and L2 JSON parsing, repair, caching and an
  interrupted build that resumes; and the approach end to end: stores, executor, `answer()`, index cost and ledger
  stage lines. Add it to RUNBOOK section 1 next to the smoke test.
- **RUNBOOK section 3.** Add a table row: `ours --ablation index_level=L0|L1|L2` gives `OURS+index_level=L*`, a cheap
  index level with 2-3 query calls.
- **RUNBOOK section 4.** Add the `_levels` caches next to `_enrichment`.

### Things to know

- **Code sha1.** `run.approach_code_sha1` hashes every non-test `.py` file in `approaches/ours/`. `lite.py`,
  `lite_llm.py` and `levels.py` changed that hash for all OURS variants, L3 included. Where earlier records of the same
  variant exist in the same `--out`, resume only warns. dev_full had not started its query phase when the files were
  added.
- **Planner prompt at L0-L2.** `ours-plan-v3` still describes headers "with keywords and synonyms ... and the questions
  it answers". At L0 there is no header, and at L1/L2 there are no questions. This was kept so the query side stays
  identical. It is a caveat for the paper, not a bug.
- **`no_normalization` combined with a level.** Fact lines, glossary matching and keyword inputs always come from
  normalised text, like L3's default `no_normalization_raw_enrichment=False`.

## TIMELINE questions: series answers (questions/eval agent, 2026-09-16)

A new question type asks for a metric across several reports ("how many people worked at Divar from 1397 to 1400,
and how did it change?"). The gold answer is a series with one value per period. Schema, parser and scoring are in
`eval/timeline.py`; authoring and validation are in `tools/timeline_questions.py`, driven by `tools/questions.py`.
The DEV set is `data/questions/pilot_timeline.{src.yaml,jsonl}` (8 questions, corpus C). Everything that needs no frozen
file is done and tested (`eval/test_timeline.py` 71 checks; `tests/test_questions.py` 43 checks). The series fields live in
`Question.meta`, so `Question.from_dict` keeps them unchanged.

### Patch 1 (required): `ragbench/contracts.py`, the two enums — APPLIED (enums before 2026-09-16 14:00, comments 2026-09-16)

```python
ANSWER_TYPES = ("numeric", "short", "span", "unanswerable", "series")
QUESTION_TYPES = ("lookup", "locate", "multi-section", "cross-year", "unanswerable", "timeline")
```

Also update the two comments in `class Question`: `answer_type: str  # numeric | short | span | unanswerable | series` and
`type: str  # lookup | locate | multi-section | cross-year | unanswerable | timeline`.

**Why this is required.** `run.py` drops every question whose `q.validate()` is non-empty. It logs
`WARNING: skipping N invalid questions`. Today that happens to all 8 timeline questions, whose problems are
`['answer_type', 'type']`. Nothing else in `Question.validate()` needs a change:
- "no gold units" already applies;
- the numeric check only applies to `numeric`.

The patch changes no config hash: `contracts.py` is not hashed. Existing question files and their resume signatures are
unaffected. `tools/questions.py validate` reports these two schema problems as `info`, with a `[note]` line, until the
patch is in.

### Patch 2 (for the agentic approach): structured output on `AnswerResult` and `RunRecord` — APPLIED 2026-09-16

The scorer reads `record["structured"]` first and `record["trace"]["structured"]` second. The agentic approach can
therefore ship today by putting its series into `trace["structured"]`. A first-class field is cleaner:

```python
@dataclass
class AnswerResult:
    answer_text: str
    evidence: EvidencePack = None
    retrieved: list = field(default_factory=list)
    trace: object = None
    structured: dict = None        # optional machine-readable answer, e.g. {"series": [...]} (eval/timeline.py)

@dataclass
class RunRecord:
    ...                            # after meta
    structured: dict = None        # copied from AnswerResult.structured; None for approaches without one
```

In `Approach.answer()`, add `structured=(res.structured if res else None),` to the `RunRecord(...)` call. Both fields
default to `None`, so old records and every existing approach are unchanged.

**Contract** (documented in `eval/timeline.py`):

```
{"series": [{"period": "1399", "value": 18.5, "unit": "million users", "definition": "...",
             "quote": "...", "unit_id": "C.divar_annual-report_1399.fa::p16_b4", "doc_id": "C.divar_annual-report_1399.fa"}],
 "comparable": false,            # optional; false counts as flagging a definition change
 "notes": "...",                 # optional; flag phrases here also count
 "trend": "..."}                 # optional; a trend statement (a false trend if not comparable and not flagged)
```

- `value` may be a number, scaled by `unit` (the scale words `thousand`/`million`/`هزار`/`میلیون`/`percent`), or a
  string such as `"18.5 میلیون"`.
- A non-empty `series` list is scored directly. Otherwise the answer text is parsed.
- `unit_id` drives `tl_citation_correct`: the cited unit must state the value. Gold units are recognised without
  loading the corpus; any other cited unit is loaded lazily from the corpus.

### No other frozen file needs a change

- **`run.py`.** The split is stamped from the file name (`pilot_timeline.jsonl` gives `dev`). `--docs-from-questions`
  works: it takes `doc_scope` plus the gold units. The resume guards are unchanged.
- **`config.yaml`.** Needs no change. The judge rubric is outside the config hash, like the existing judge.
- **`approaches/__init__.py`.** Needs no change.

### What running timeline questions costs (for the integrator's driver)

The 16 timeline reports are all in corpus C, and only `C.divar_annual-report_1400` is already in the pilot restriction. The
other 15 reports hold 464 sections, 386k characters and about 486 OURS section parts. At the measured B/C rates that
means:
- OURS enrichment: about 1.2 GPU-hours (9 s per part);
- B4: about 500 summary calls, roughly 15 minutes;
- B3: about 25 seconds of embeddings.

Two ways to run:

```bash
Q=ragbench/data/questions/pilot_timeline.jsonl
# own restriction (new index fingerprints for C):
$PY -m ragbench.run --approach ours --questions $Q --corpus C --settings fa-fa --docs-from-questions --out runs/dev_timeline
# or one C index for both dev files (comma list; not allowed with --final):
$PY -m ragbench.run --approach ours --questions $Q --corpus C --settings fa-fa \
    --docs-from-questions ragbench/data/questions/pilot.jsonl,$Q --out runs/dev_timeline
$PY -m ragbench.eval.aggregate --runs runs/dev_timeline --questions $Q --split dev --judge none
```

Aggregation writes `timeline.csv` and `cost_by_qtype.csv`. `results.md` gains "Timeline questions" and "Cost by question
type" tables. `comparisons.csv` gains `tl_coverage` and `tl_series_correct`.

### Judge: series rubric (`eval/judge.py`, `judge-series-v2`)

- **v1 live probe.** One OpenRouter call (`--probe-series`, $0.00012): strict JSON, and every field parsed. It graded
  "grew steadily from 35 million users ... to 53.1 million" across a users→phones change as `false_trend: false`,
  `correct: true`.
- **v2 changes.**
  - A WARNING block with the series notes comes before the gold series.
  - The comparability fields are asked for first, with an example.
  - The parser forces `correct` to agree with the judge's own fields; the judge's raw value is kept as `correct_raw`.
- **Not re-probed.** The persisted workflow budget is now 10/10 calls. Before judging timeline answers, raise
  `judge.max_calls_total` and run `$PY -m ragbench.eval.judge --probe-series` once. It exits 0 when the judge reports
  full coverage, `false_trend: true` and `correct: false`.
- **Reporting.** The model-free `tl_*` metrics do not depend on the judge. Report the judge's
  `tl_judge_coverage` / `tl_judge_flag` / `tl_judge_false_trend` next to them.

### RUNBOOK section 1 (already edited in RUNBOOK.md)

`$PY -m ragbench.eval.test_timeline` (71 checks). `tests/test_questions.py` now has 43 checks, including the timeline file.

## AGENTIC: explicit budgeted agentic pipeline (agentic agent, 2026-09-16)

The user's main proposed system (`ragbench/approaches/agentic/`, class `AgenticPipeline`, variant name `AGENTIC`).
Fixed control flow with the model called only at named decision points, every output checked by code:

1. **plan** (1 call, JSON schema, repair ladder: parse, extract, suffix repair, one `plan_repair` call when the budget
   allows, then a program plan). It returns entity, metric, kind (single | comparison | timeline), literal fa/en terms,
   entity names, unit kind, period range and company keys. Company keys come from an enum built from the document
   ids and titles in scope. Code then cleans the terms:
   - entity names and question words ("گروه", "تغییر") are removed;
   - a term left with no content word is dropped;
   - each content word of a phrase is added as its own term;
   - a model period is ignored unless the question names a year.

   Code also expands the terms into ripgrep patterns with Persian morphology and chooses the target documents
   (company + period overlap).
2. **search** (no model). ripgrep over the plain section store, plus term-number proximity windows built from
   neighbouring units, tables and title leads. Adjacent windows are merged, then scored and deduplicated. Selection is
   round-robin over documents in year order, with section diversity, `per_doc_cap` 4 and `max_candidates` 32.
3. **verify** (batches of 8 snippets per call, prompt `agentic-verify-v2`).
   - **Reply schema.** The JSON schema requires one key per snippet (`s1`..`sn`), each holding
     `{relevant, values: [...] (at most 3)}`. With v1's free-length `items` array, the live 4B verifier answered 1 of
     8 snippets and skipped the rest.
   - **Code guard**, on every value. The quote must occur in the snippet after canonicalisation. The value's number
     must occur in the quote (compared by reading: separators, scale words and right-to-left reversed groups may
     differ). The quote must hold a label, and for numeric questions a metric term (`quote_without_term`). The value
     must not be a bare year or a chart-axis tick (`value_is_axis_tick`). A value with several numbers is rejected.
   - A snippet that a parsed reply leaves out counts as `model_omitted`, not `unparsed`.
   - **Re-attribution.** Live, the verifier also filed values under the neighbouring key (s13 held s14's values).
     When a quote is not in its own snippet, the value is accepted for the batch snippet whose text passes the whole
     guard (`reattribute`, counted in `stats.reattributed`).
4. **gap fill** (at most 1 round, at most 1 batch). Only for target documents with no verified value. Windows are
   wider, non-weak term words are also searched alone, and already-seen units are excluded.
5. **reconcile** (code) and **answer** (1 call).
   - Reconcile builds a table. Each period's authoritative value is the report of that period. Restatements and
     conflicts are noted, and definition classes come from the verbatim quote. Values with different classes are
     not comparable, and then no trend is stated.
   - Gender-subgroup rows and "narrower" rows leave the series when a plainer row exists for the same period. A
     narrower row has at least 2 definition words that the question's terms do not explain, for example "دارای
     معلولیت" or "معاف از کمیسیون". Those rows stay in `structured.narrower`.
   - The evidence for the answer call is [1] the table, then the rows' windows, packed with `pack_evidence` (6000
     chars).
   - Default `answer_mode` is `shared`: the study's shared answer prompt and `num_predict_answer`. Flag
     `synth_answer` uses its own timeline prompt (citations, not-comparable wording) and is an ablation.

Budget: at most 8 calls. The answer call is always reserved, and one gap call is reserved when at least 3 calls are
left after planning. The ledger is the source of truth: every call carries `stage=<plan|verify|gap_fill|answer>`, and
`trace.timing` reconciles stage sums with the question's ledger scope. `trace.structured.series` rows carry exactly
`period, value, unit, definition, quote, unit_id, doc_id, verified`, plus extras. It follows the eval/timeline.py
contract (checked in the offline test with `structured_series`).

**Resumability.** run.py resume works unchanged (one RunRecord per question). Inside a question every model call is
cached in `work/runs/<out>/agentic_calls_<variant>.sqlite`, via `util.KVCache` with a commit on every put. The key is
the full request: model + digest, purpose, prompt version, messages, schema and options. A restarted question
replays finished calls into the ledger with `cache_hit: true` and the original tokens and wall time, and repeats at
most the call in flight. Record costs therefore equal a clean run. Only `latency_ms` is lower, and `trace.stats.cache_hits`
marks such records. A new `--out` starts with an empty cache.

### Patch 1: `ragbench/approaches/__init__.py` (frozen registry) — APPLIED 2026-09-16

```python
APPROACHES = {
    ...,
    "OURS": "ragbench.approaches.ours:EnrichedGrep",
    "AGENTIC": "ragbench.approaches.agentic:AgenticPipeline",
}

# ablation flags accepted by AGENTIC (value True)
AGENTIC_FLAGS = ("no_guard", "no_gap_fill", "line_windows", "no_doc_selection", "synth_answer")
```

### Patch 2: `ragbench/run.py` — APPLIED 2026-09-16

The OURS index-level patch above also rewrites `parse_flags`. Both patches together:

```python
from .approaches import AGENTIC_FLAGS, APPROACHES, OURS_FLAGS, approach_class
...
def parse_flags(values, approach):
    flags = {}
    known = B5_FLAGS if approach == "B5" else AGENTIC_FLAGS if approach == "AGENTIC" else OURS_FLAGS
    for v in values or []:
        ...                                   # unchanged loop (granularity / index_level branches only match OURS keys)
    if flags and approach not in ("OURS", "B5", "AGENTIC"):
        raise SystemExit("--ablation applies to --approach ours, agentic (and b5 agent_answer) only")
    return flags
```

Help strings: `--approach` becomes `"b0|b1|b2|b3|b4|b5|ours|agentic"`, and `--ablation` gains
`; AGENTIC: no_guard, no_gap_fill, line_windows, no_doc_selection, synth_answer`.

Nothing else in run.py changes:
- `ALIASES` is derived from `APPROACHES`, so `--approach agentic` resolves.
- Indexes build through the base class (a plain store under the index dir, no model calls).
- `approach_code_sha1` hashes every non-test `.py` file of the package. That includes `live_check.py`, so freeze
  it with the code.

### Patch 3: `ragbench/config.yaml` — (a) APPLIED 2026-09-16; (b) superseded (AGENTIC only, no AGENTIC ablations in `final`)

**(a) The call cap.** Add it at a moment when no output directory needs resuming. `caps` is part of the records'
`config_hash`, so a new key changes the hash for every approach, and resume refuses mixed hashes. Until then the
approach uses `DEFAULTS["max_calls_default"] = 8`, the same value.

```yaml
caps:
  ...
  agentic_max_calls: 8               # AGENTIC: plan (+1 repair) + verify batches + <= 1 gap batch + the answer call
```

**(b) `final.variants`** (only the variants that belong to the frozen test run):

```yaml
  variants: [..., "AGENTIC", "AGENTIC+no_guard", "AGENTIC+no_gap_fill", "AGENTIC+line_windows", "AGENTIC+synth_answer"]
```

### Patch 4: `ragbench/contracts.py` structured output — APPLIED 2026-09-16

This is the "TIMELINE questions, Patch 2" above, nothing more. `AgenticPipeline` already sets
`AnswerResult.structured` when the field exists. It always also writes `trace["structured"]`, which the scorer reads
today.

### Patch 5 (optional; `eval/` is owned by the eval agent) — APPLIED 2026-09-16 (sanity and aggregate fairness warnings)

`eval/sanity.py` call caps: add `"AGENTIC": (1, cfg.caps.get("agentic_max_calls", 8))` next to B5/OURS. Replayed
(cached) calls are ledger records with `kind: chat`, so the ledger-consistency check still holds.

### Driver lines

```bash
Q=ragbench/data/questions/pilot.jsonl; T=ragbench/data/questions/pilot_timeline.jsonl
$PY -m ragbench.approaches.agentic.test_agentic                                   # offline, no model
for c in A B; do $PY -m ragbench.run --approach agentic --questions $Q --corpus $c --docs-from-questions --out runs/dev_agentic; done
$PY -m ragbench.run --approach agentic --questions $Q --corpus C --settings fa-fa --docs-from-questions --out runs/dev_agentic
$PY -m ragbench.run --approach agentic --questions $T --corpus C --settings fa-fa --docs-from-questions --out runs/dev_timeline
for ab in no_guard no_gap_fill line_windows synth_answer; do
  $PY -m ragbench.run --approach agentic --ablation $ab --questions $T --corpus C --settings fa-fa --docs-from-questions --out runs/dev_timeline; done
```

Cost: the store and warm-up for C take about 27 s cold and under 1 s warm, with no index-time model calls. Query time
is 3 to 8 calls per question. Measured numbers are in `work/runs/agentic_smoke/` (see the live check below).

### Live check (2026-09-16, `python -m ragbench.approaches.agentic.live_check --synth`)

Outputs are in `work/runs/agentic_smoke/`, with attempts 1-3 in `attempt*/` and the current run at the top level. The GPU
was shared with the dev_full OURS index job, so most latency is `lock_wait_ms`.

**Snapp timeline** ("how many drivers, and how has that changed", corpus C, all 97 documents, scope `corpus`). The planner
chose exactly the 7 Snapp reports.

| Attempt | What changed before it | Calls | LLM s / latency s | Verified value per period |
|---|---|---|---|---|
| 1 | - | 6 | 26.9 / 191.9 | 1398 >2M ✓; 1400 "3 million club users"; 1401 6,561 (staff, wrong) |
| 2 | term cleaning, `quote_without_term`, `model_omitted` | 7 | 33.0 / 217.4 | 1398 ✓ and noise (verifier skipped 7 of 8 snippets per batch) |
| 3 | keyed verify schema v2, axis-tick guard, narrower rows | 6 | 36.9 / 146.1 | 1398 ✓, 1400 3,761,925 ✓, 1402 6,152,509 + 3,305,147 ✓ |
| 3b | re-attribution (plan and verify replayed from the cache) | 6 | answer only | also 1404 8,592,169 + 3,919,028 ✓ |

- **Still wrong.**
  - Service-scoped values: SnappBox, SnappFood and "club users" for 1399, 1401 and 1403, and extra rows in 1402 and 1404.
  - The 1403 car figures (7,417,231 / 3,692,657) were never among the 4 candidates for that report.
- **Detected.** `comparable=false` (classes: total, active, registered, unqualified). The shared answer claims no trend.
- **Not captured.** The 1403/1404 glossary rule that totals count accounts, not persons.

**A-rpt-f-02** (fa-fa, development corpus A). The gold value was verified every time.
- The shared answer call was correct in attempt 1.
- In attempts 2 and 3 it answered with a distractor range from the same evidence window, although the table
  row held the gold value.

## AGENTIC2 ("AGENTIC v2": tree + fact DB + verify; AGENTIC v2 agent, 2026-09-16, APPLIED)

Designed after the core test results were seen (disclosed); tuned on dev only. RUNBOOK section 0e has the design.
Applied by the same agent, which was allowed to edit these shared files by its brief:
- `approaches/__init__.py`: `AGENTIC2` -> `ragbench.approaches.agentic2:TreeFactVerify`, `AGENTIC2_FLAGS`
  (`no_vectors`, `no_factdb`, `no_tree`, `no_verify`, `candidates` with an integer value).
- `run.py`: `parse_flags` for AGENTIC2 (`candidates=<n>`), help texts, the untimed embed warm-up also for AGENTIC2.
- `config.yaml`: `caps.agentic2_max_calls: 4`, `caps.agentic2_max_seconds: 60` (the records' config hash of AGENTIC2
  differs from the earlier records'), `final.variants` + `AGENTIC2` (the shared answer prompts stay as declared).
- `eval/sanity.py`, `eval/aggregate.py`: the AGENTIC2 call cap (4).
- `eval/timeline.py`, `eval/metrics.py`, `eval/aggregate.py`: `--timeline-scoring auto|text|structured` (text = every
  approach from its answer text; secondary `tls_*` structured scores; `timeline_scoring` in timeline.csv and meta).
- `tools/core_plan.sh`: variant keys `agentic2`, `a2novec`, `a2nofdb`, `a2notree`, `a2noverify`, `a2c32`; stages `v2dev`
  and `v2test`; aggregate with text timeline scoring + `results_structured/`.
- `tools/answers_html.py`: `("AGENTIC2", "AGENTIC v2")`; timelines noted as text-scored.
- Tests: `tests/test_core_integration.py` (+8 checks: registry, flags, variant, caps, final refuses AGENTIC2 ablations),
  `eval/test_timeline.py` (+3 checks: text scoring). New: `approaches/agentic2/test_agentic2.py` (53),
  `tools/offline_suites.sh`, `tools/v2_tune.sh`.
- AGENTIC v1 files (`approaches/agentic/*`) and OURS files were NOT changed; AGENTIC2 imports them.
