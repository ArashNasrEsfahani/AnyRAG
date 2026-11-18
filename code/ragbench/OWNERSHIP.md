# ragbench: ownership and contracts

Package root: `D:/KareOne/RAG Paper/ragbench` (import as `ragbench`, run from `D:/KareOne/RAG Paper`).
Python: `D:/KareOne/RAG Paper/.venv/Scripts/python.exe`. ripgrep: `D:/KareOne/RAG Paper/tools/bin/rg.exe`.
Caches and outputs: `D:/KareOne/RAG Paper/work/` (corpus, stores, indexes, runs, ledger, locks, judge_budget.json).

## Who owns what

| Path (under ragbench/) | Owner | Notes |
|---|---|---|
| `config.yaml`, `config.py` | foundation (frozen) | change requests only |
| `llm.py` | foundation (frozen) | OllamaClient, OpenRouterClient, Ledger |
| `normalize.py` | foundation (frozen) | the only normalisation code allowed |
| `corpus/` (model, kareone, extracted, registry) | foundation (frozen) | |
| `store.py` | foundation (frozen) | plain section store + shared `grep` |
| `contracts.py` | foundation (frozen) | Approach, Query, Question, RunRecord, prompt, budget |
| `util.py`, `approaches/__init__.py` (registry), `tests/test_foundation.py`, `requirements.txt` | foundation (frozen) | |
| `approaches/baselines.py` (+ `tests/test_baselines.py`) | baselines agent | B0 `NoRetrieval`, B1 `GoldEvidence`, B2 `BM25Retrieval`, B3 `VectorRetrieval` |
| `approaches/tree_nav.py` | tree agent | B4 `TreeNavigation` |
| `approaches/grep_agent.py` | grep-agent agent | B5 `GrepAgent` |
| `approaches/ours/` (package; `__init__.py` exports `EnrichedGrep`) | ours agent | OURS + ablation flags |
| `data/questions/pilot.jsonl` + question-authoring tools (e.g. `tools/questions/`) | questions agent | |
| `data/questions/pilot_timeline.{src.yaml,jsonl}` + `tools/timeline_questions.py` (build / validate / composition of timeline questions, driven by `tools/questions.py`) | questions agent (timeline work, 2026-09-16) | DEV timeline set; schema in `eval/timeline.py` |
| `eval/` (metrics, judge, aggregate, report) + `run.py` CLI | evaluation agent | |
| `eval/timeline.py` (series schema accessors, period/value parser, tl_* scoring, timeline aggregation) + `eval/test_timeline.py` | evaluation agent (timeline work, 2026-09-16) | series judge rubric lives in `eval/judge.py` |
| `approaches/ours/lite.py`, `lite_llm.py`, `levels.py` (+ `levels_test.py`) | ours agent (index levels, 2026-09-16) | OURS+index_level=L0/L1/L2 |
| `approaches/agentic/` (package; `__init__.py` exports `AgenticPipeline`) + `test_agentic.py`, `live_check.py` | agentic agent (2026-09-16) | AGENTIC + ablation flags |
| `approaches/agentic2/` (package; `__init__.py` exports `TreeFactVerify`) + `test_agentic2.py`; `tools/v2_tune.sh` (dev tuning), stages `v2dev` / `v2test` of `tools/core_plan.sh`, `tools/offline_suites.sh` | AGENTIC v2 agent (2026-09-16, after the core test) | AGENTIC2 ("AGENTIC v2": tree + fact DB + verify) + ablation flags; imports the v1 planner/verifier/reconciler, OURS `lite.py` (L0 facts) and B3's embedding cache helpers without changing them |
| `data/questions/test_core.jsonl`, `dev_core.jsonl`, `dev_core_timeline.jsonl` + `tools/make_core_sets.py` | orchestrator (CORE PLAN, 2026-09-16) | `core_sets.md` |
| `data/questions/test_timeline.{src.yaml,jsonl}` | questions agent (timeline test, 2026-09-16) | 10 TEST timeline questions |
| `tools/core_plan.sh` (driver) + `tools/core_plan.py` (busy / check-timeline / plan / status / cost) + `tests/test_core_integration.py` | integration pass (CORE PLAN, 2026-09-16) | stage `l3` by the orchestrator; RUNBOOK section 0d |

Frozen means: do not edit; send a change request (file, exact change, reason) to the orchestrator.
Approach-specific tunables that are not fairness-critical live as `DEFAULTS` in the approach module;
fairness-critical values (models, generation options, evidence budget, chunking, caps, judge) are only in `config.yaml`.

## Fairness rules (enforced in code where possible)

1. Approaches receive a `Query`, never a `Question`. Only a class with `needs_gold = True` (B1) gets `query.gold_unit_ids`.
   `query.doc_scope` is given to every approach alike (`evidence.use_doc_scope: true`).
   Document/Section/Unit `.meta` is provenance (e.g. KareOne figure registry via `corpus.kareone.load_figures`) for question authoring only; it must not drive retrieval or enrichment.
2. Index-time work never sees questions. `_build_index(corpus_set, index_dir)` gets only the corpus.
3. Every model call goes through `self.client` (`OllamaClient.chat/embed`); the base class opens ledger scopes
   (`phase="index"` around `_build_index`, `phase="query"` around `_answer`), and RunRecord / build_stats numbers come from those scopes.
4. The final answer is `self.generate_answer(query, evidence)`: shared prompt (`build_answer_messages`) over an `EvidencePack` from `self.pack(items, index)` (`pack_evidence`, budget `evidence.max_chars` = 6000 chars of rendered evidence block). Intermediate calls (navigation, planning, tool steps) use other `purpose` strings.
5. One GPU: the client serialises calls across threads and processes (`work/locks/ollama.lock`). Never run two model-heavy jobs at once.
6. OURS uses no BM25 and no embeddings anywhere.
7. Judge (OpenRouter) calls go only through `OpenRouterClient.chat`, which enforces `judge.max_calls_total` (10) and `judge.max_cost_usd` persisted in `work/judge_budget.json`. Never print the key.

## Contracts (summary; the code is authoritative)

### Config — `ragbench.config`
`cfg = load_config()`; sections as attributes/dicts: `cfg.models`, `cfg.ollama`, `cfg.generation`, `cfg.embedding`, `cfg.evidence`, `cfg.chunking`, `cfg.caps`, `cfg.judge`, `cfg.corpora`, `cfg.settings`; `cfg.path(key)` -> absolute Path for `paths.<key>`; `cfg.get("caps.grep_agent_max_steps")`; `cfg.hash(*sections)`; `cfg.copy_with({...})`; `cfg.openrouter_key()`.

### LLM — `ragbench.llm`
- `OllamaClient(cfg=None, ledger=None)`
  - `.chat(messages, *, purpose, phase=None, approach=None, num_predict=None, format=None, tools=None, options=None, think=None, model=None, **ctx) -> ChatResult`
    (`format`: None | "json" | JSON-schema dict; options come from `generation`: temperature 0, seed 42, top_p 1, top_k 1, num_ctx 8192; `think=False`).
  - `.embed(texts, *, purpose, phase=None, approach=None, model=None, batch_size=None, **ctx) -> EmbedResult` (bge-m3, 1024-d, batches of 16, one ledger record per batch).
  - `.health()`, `.use_ledger(ledger)` (context manager), `.options(...)`.
  - phase must be one of `index|query|eval|smoke` (taken from the open ledger scope if not passed); `purpose` is required.
- `ChatResult{text, prompt_tokens, completion_tokens, wall_ms, load_ms, done_reason, thinking, tool_calls, model, call_id, cost_usd}`; `.json(default)` parses the first JSON object in the text. `extract_json(text, default)` is the same as a function.
- `EmbedResult{vectors: list[list[float]], prompt_tokens, wall_ms, calls, model}`.
- `Ledger(path=None, run_id=None)`; `with ledger.scope(**ctx) as sc:` -> `LedgerScope` with `.calls` (chat), `.embed_calls`, `.judge_calls`, `.prompt_tokens`, `.completion_tokens`, `.embed_tokens`, `.wall_ms`, `.load_ms`, `.failed_calls`, `.summary()`.
  Ledger JSONL record: `{ts, call_id, run_id, kind: chat|embed|judge, model, phase, approach, purpose, question_id?, setting?, corpus_set?, ok, error?, prompt_tokens, completion_tokens, wall_ms, wait_ms, load_ms, prompt_eval_ms, eval_ms, attempts, done_reason, prompt_chars, completion_chars, n_inputs (embed), cost_usd (judge)}`. Default file: `work/ledger/<yyyymmdd>_<pid>.jsonl`.
- `OpenRouterClient(cfg=None, ledger=None)`: `.chat(messages, *, purpose, model=None, max_tokens=None, **ctx) -> ChatResult` (raises `BudgetExceeded`), `.budget_state()`, `.key_info()` (free).
- Errors: `ModelError`, `BudgetExceeded`.

### Normalisation — `ragbench.normalize`
`normalize(text)` = extractor `text_norm` rule (NFKC, ي/ى→ی, ك→ک, tatweel removed, ٫٬٪→.,%, joiners→ZWNJ between Arabic letters else removed, bidi controls removed, Persian/Arabic digits→ASCII, whitespace collapsed; verified identical to `tools/extract/fa_text.normalize`).
`normalize_lines(text)` (same, per line), `to_ascii_digits`, `fold(text)` (optional extra folding for lexical matching: harakat removed, آأإٱ→ا, ة/ۀ→ه, ؤ→و, ئ→ی, ZWNJ→space, lower-case — use on both sides or not at all), `word_tokens(text, folded=True)`, `detect_lang(text) -> "fa"|"en"` (fa if ≥30% of letters are Arabic-script), `fa_ratio`, `script_counts`.

### Corpus — `ragbench.corpus`
- IDs: doc `"<corpus>.<key>.<lang>"`; section `"<doc_id>::<local>"`; unit `"<doc_id>::<local>"` (unit locals contain `#`). `swap_lang(id, lang)`, `split_id(id)`, `doc_lang(doc_id)`, `make_doc_id(corpus, key, lang)`.
- `Unit{id, doc_id, section_id, kind: paragraph|list|caption|fact|table|heading, lang (detected), text (normalised, may contain \n), text_raw, span [start,end) in Document.text, order, meta}`; `.local_id`.
- `Section{id, doc_id, title, title_raw, level (root=1), parent (id|None), children [ids], unit_ids [own units], span [title+own units], subtree_span, order (pre-order), meta}`; `.local_id`.
- `Document{id, corpus, lang, title, title_raw, source, pair_id, alignment: "unit"(A)|"document"(B)|None(C), sections (pre-order), units (doc order), text, meta}`
  methods: `section(id)`, `unit(id)`, `has_unit`, `roots()`, `children(id)`, `descendants(id)`, `leaves()`, `section_units(id, recursive=False)`, `section_text(id, recursive=False)` (== `text[span]`), `section_path(id)` (titles root→node), `units_overlapping(start, end)` (map a char span to units), `stats()`, `to_dict/from_dict/save/load`.
  Invariant: `doc.text[u.span[0]:u.span[1]] == u.text`; a section block is `title + "\n" + units joined by "\n\n"`, blocks joined by `"\n\n"`.
- `CorpusSet{name, docs, skipped}`: `doc(id)`, `has_doc`, `doc_ids()`, `unit(uid)`, `section(sid)`, `subset(ids)`, `fingerprint()`, `stats()`, `.corpus`, `.lang`, iterable.
- `load_set(name, cfg=None, rebuild=False)`; names `A, A_fa, A_en, B, B_fa, B_en, C (=C_fa)`; `set_name_for(corpus, d_lang)`; `load_document(doc_id)`.
- Corpus A tree: `_report` → chapter (`<chapter>/<section>`) → part sections → chart sections (`.../<chart_id>`, only titled charts). Unit locals e.g. `chapter/section#lede`, `#keyInsight`, `#summary`, `#caveat`, `#takeaway.<i>`, `.../<chart>#kpi.<i>` (fact), `#item.<i>`, `#insight.<i>`, `#table.<k>` (≤20 rows, header repeated), `#subtitle`/`#source` (caption), `#note`, `#axis.<i>`, `#step.<i>`, `#input.<i>`, `#output.<i>`, `#assumption.<i>`. `{{figure}}` placeholders are resolved to formatted values in both languages.
- Corpora B/C tree: PDF outline (≥3 entries) else heading blocks (≥3) else pages; `front` section titled with the doc title; unit locals are extractor block ids (`p12_b3`, groups `p3_g1` as fact, tables `p012_t1#table.<k>`). B pairs: `B.<pair>.fa` ↔ `B.<pair>.en` (pair_id), not unit-aligned.

### Store and grep — `ragbench.store`
- `build_plain_store(corpus_set, root=None, normalized=True, rebuild=False) -> Path` (default `work/stores/<set>[-raw]`).
- Files: `<root>/<doc_id>/<order:04d>.txt`: lines `@@SECTION doc=<doc_id> sec=<section_id> level=<n>`, `@@TITLE <title>`, `@@PATH <a > b > c>`, blank, then unit texts separated by one blank line. `<root>/<doc_id>/_linemap.json` `{rel_path: [[first_line, last_line, unit_id, tag], ...]}`; `<root>/manifest.json` `{variant, files: {rel_path: {doc_id, section_id, level, title}}, corpus_fingerprint, ...}`.
- `write_store(root, files: list[StoreFile], variant, extra_manifest=None)`; `StoreFile{doc_id, section_id, rel_path, level, title, blocks: list[StoreBlock{text, unit_id="", tag="body"}]}` (OURS writes its enriched store this way, with its own tags such as `header`/`fact`); `plain_files(doc, normalized=True)`.
- `grep(root, pattern | [patterns], *, doc_ids=None, paths=None, globs=None, tags=None, ignore_case=True, fixed_strings=False, pcre2=False, max_hits=None, timeout_s=60) -> list[GrepHit]` (ripgrep `--json`, multi-threaded; hits sorted in Python by target order, path, line; `doc_ids=[]` / `paths=[]` search nothing and return `[]`, `None` searches the whole store; raises ValueError on a bad regex).
  `GrepHit{path, doc_id, section_id, line, text, unit_id, tag, submatches [[start,end] char offsets]}`.
- `read_lines(root, rel_path, start, end) -> (text, first, last, unit_ids)`; `line_info(root, rel_path, line) -> (unit_id, tag)`; `section_file(root, section_id) -> rel_path`; `rg_binary()`.

### Contracts — `ragbench.contracts`
- `Setting(q_lang, d_lang)`; `.name` "fa-en"; `.cross_lingual`; `Setting.parse("fa→en"|"fa-en"|("fa","en"))`; `ALL_SETTINGS`.
- `Question{id, corpus: A|B|C, question_fa, question_en, answer_fa, answer_en, answer_type: numeric|short|span|unanswerable, gold_units: {"fa": [unit ids], "en": [unit ids]}, doc_scope: [doc ids] | "corpus", type: lookup|locate|multi-section|cross-year|unanswerable, answer_value: float|None, answer_unit: str|None, meta: dict}`
  methods `text(lang)`, `answer(lang)`, `gold(lang)`, `gold_refs(lang) -> [GoldRef{doc_id, unit_ids}]`, `scope(lang)`, `supports(setting)`, `validate() -> [problems]`, `to_dict/from_dict`. `load_questions(path)`, `save_questions(path, qs)` (JSONL).
- `Query{question_id, text, q_lang, d_lang, setting, corpus, corpus_set, doc_scope, gold_unit_ids (B1 only), answer_format ("default" | "timeline", from `answer_format(question)` = Question.type; selects the shared answer prompt and budget only, approaches must not read it)}`.
- `RetrievedItem{text, doc_id, section_id=None, unit_ids=[], kind: unit|chunk|section|window|fact|node, score=None, rank=None, span=None, meta={}}` — `unit_ids` must list every unit whose text the item contains (use `Document.units_overlapping` for chunks/windows, `store.read_lines` unit ids for line windows).
- `pack_evidence(items, corpus, budget=None, cfg=None) -> EvidencePack{items, text, chars, budget, n_offered, n_included, last_item_cut}`, `.unit_ids`. Rendering: `[n] <doc title> > <section path>\n<text>`, items separated by a blank line, headers count toward the budget; items are added in the given order while they fit; the first that does not fit is cut to the remaining room if ≥ `evidence.min_item_chars` (200) survive, otherwise dropped; packing stops there. Unit ids of a cut item: for a verbatim item (`span` given and `Document.text[span]` equals its text) every declared unit with at least 60 characters (or all of it) inside the kept span; otherwise a text heuristic (unit's first 60 chars inside the kept text). Packing order is each approach's choice (B4 rank order with unit-boundary narrowing of oversized items, B5 cited-first, OURS budget-aware selection, B2/B3 rank order) and must be stated in the paper.
- `build_answer_messages(query, evidence_or_None) -> [system, user]` (`ANSWER_PROMPT_VERSION` "answer-v2"): answer only from the numbered evidence and reply exactly `NOT_FOUND` when the answer is not there; B0 variant: "answer from your own knowledge" and `NOT_FOUND` "if you do not know the answer"; answer in the question language, answer only, numbers copied with unit and period.
  Timeline questions (`query.answer_format == "timeline"`) get `ANSWER_TIMELINE_SYSTEM` ("answer-timeline-v1": same source/abstain rules; one line per period `<period>: <value> <unit>`, `<period>: NOT_FOUND` for a missing one, a period's own report wins, one closing sentence on the change or "not directly comparable"). `ANSWER_SYSTEMS`, `ANSWER_PROMPT_VERSIONS` (declared in config `final.answer_prompts`), `answer_prompt_version(q)`, `answer_num_predict(q, cfg)` (256 = `generation.num_predict_answer`, timeline `answer.timeline_num_predict` = 768). `RECORD_HASH_SECTIONS` = models, generation, evidence, chunking, caps, answer.
- `AnswerResult{answer_text, evidence: EvidencePack|None, retrieved: [RetrievedItem], trace, structured: dict|None}` (structured e.g. `{"series": [...]}`, contract in eval/timeline.py).
- `RunRecord{question_id, approach (variant, e.g. "OURS+no_facts"), approach_base, flags, setting, q_lang, d_lang, corpus, corpus_set, answer_text, abstained, retrieved_units [RetrievedItem dicts, before the budget], evidence_units [unit ids given to the answer call], evidence_items [{doc_id, section_id, kind, chars, unit_ids, cut}], evidence_chars, llm_calls, embed_calls, prompt_tokens, completion_tokens, embed_tokens, llm_wall_ms, load_ms, latency_ms, failed_calls, error, trace, models, config_hash, index_fingerprint, ts, run_id, lock_wait_ms, split, meta, structured}`; `.to_dict()`, `.key()`. `split` ("dev" | "test" | "other") and `meta` ({ollama_version, questions_file, final, docs_restriction}) are stamped by `ragbench.run`; `Approach.answer` adds `meta.answer_format`, `answer_prompt`, `answer_num_predict` and copies `AnswerResult.structured`. `lock_wait_ms` (default 0) is the part of `latency_ms` spent waiting for the GPU lock behind another process (from `LedgerScope.summary()["lock_wait_ms"]`); eval subtracts it with `load_ms` for no-load latency.
- `RunLog(path)`: `.append(record)`, `.records()`, `.completed(include_errors=False) -> {(question_id, approach, setting)}`.
- `IndexHandle{approach, corpus_set, path, fingerprint, corpus (CorpusSet), payload, stats}`.
- `Approach(cfg=None, client=None, flags=None)` — subclass sets `name`, optional `needs_gold`, implements
  `_build_index(corpus_set, index_dir) -> payload` (resumable; cache under `index_dir`), optional `_load_index(corpus_set, index_dir)`, optional `config_signature() -> dict`, and `_answer(query, index: IndexHandle) -> AnswerResult`.
  Public (do not override): `build_index(corpus_set, ledger=None) -> IndexHandle` (writes `<index_dir>/build_stats.json` with per-attempt and total calls/tokens/ms, `complete`, `size_bytes`), `ensure_index(set_name)`, `answer(question, setting) -> RunRecord`, `make_query`, `variant`, `index_dir(corpus_set)` = `work/indexes/<variant>/<set>-<fp8>/`.
  Helpers: `docs_in_scope(query, index)`, `pack(items, index, budget=None)`, `generate_answer(query, evidence, purpose="answer")`.
- Registry `ragbench.approaches`: `get_approach(name, cfg=None, client=None, flags=None)`, `APPROACHES` (B0..B5, OURS, AGENTIC class paths above), `OURS_FLAGS` (incl. `index_level`), `AGENTIC_FLAGS`.

### Timeline questions — `ragbench.eval.timeline` (2026-09-16)
- Question lines with `type: "timeline"`, `answer_type: "series"`. The series lives in `meta` (never given to an approach):
  - `answer_series [{period, calendar, value, unit, unit_fa, value_base, percent, definition_fa, definition_en, doc, gold_units {fa, en}, context_units {fa, en}, quote}]`, ordered by period;
  - `comparable` (bool), `series_notes_fa`, `series_notes_en`, `comparability_cues {fa, en}`;
  - `reconciliation {rule: "report_of_period", rule_text, restatements [...], supplementary [...]}`;
  - `schema_version: "timeline-v1"`.
  Top-level `gold_units` = the union of every period's gold and context units; `doc_scope` = every report used;
  `answer_fa/en` = a short prose answer.
- **Rule:** the report OF a period is gold for that period. A later report's different number is a restatement (never
  gold); a period without its own report may be supplementary (optional).
- **Accessors:**
  - `is_timeline(q)`, `timeline_spec(q)`, `gold_periods(spec)`, `item_base_value(item)`;
  - `extract_series(text, known_periods, want_percent) -> {period: [NumberMention]}`, `parse_numbers(text)`;
  - `score_series(question, record, unit_resolver=None) -> tl_*`, `timeline_rows(scored)`, `render_timeline_md(rows, split, corpus)`.
- **Structured output** (read from `record["structured"]`, or today `record["trace"]["structured"]`):
  `{"series": [{period, value, unit, definition, quote, unit_id, doc_id}], "comparable"?, "notes"?, "trend"?}`.
- `eval.metrics.score_record` dispatches `answer_type == "series"` to `score_series` (`match_correct = tl_series_correct`),
  and keeps latency, calls and tokens as for every record.
- `eval.aggregate` writes `timeline.csv` and `cost_by_qtype.csv`.
- `eval.judge` uses the `judge-series-v2` rubric for series answers (`verdict["series"]`).
- **Contract patches applied** (INTEGRATION_NOTES.md, "TIMELINE questions"): `ANSWER_TYPES` + `"series"`,
  `QUESTION_TYPES` + `"timeline"`, `AnswerResult.structured` / `RunRecord.structured`.
- Parser: dot-grouped thousands (`4.318.701`) and duration-unit series (`series_duration_kinds`, `extract_series(...,
  duration_kinds)`) since 2026-09-16.

## Smoke test
`.venv/Scripts/python.exe -m ragbench.tests.test_foundation [--ollama] [--judge-key] [--scratch-extracted DIR]`

End-to-end: `bash ragbench/tools/smoke_run.sh` then `ragbench.eval.aggregate` and `ragbench.eval.sanity`; see `RUNBOOK.md`.

## Integration changes (after the component reviews)
Applied by the integration pass, which was allowed to edit frozen files:
- `contracts.py`: exact cut unit ids for verbatim items (`_units_in_cut`); B0 abstain rule; `RunRecord.lock_wait_ms`.
- `llm.py`: `LedgerScope.wait_ms` and `summary()["lock_wait_ms"]`.
- `store.py`: empty `doc_ids`/`paths` list searches nothing; ripgrep runs without `--sort path` (order restored in Python).
- `config.yaml`: comments only for caps; `judge.max_tokens` 64 -> 160 (matches eval/judge.py; judge is outside the config hash).
- `run.py`: `--docs` subset filter; `--ablation agent_answer` for B5. `eval/metrics.py`: no-load latency also subtracts `lock_wait_ms`. New `eval/sanity.py`.

## Pre-full-run fixes (2026-09-15, dev_fix1; allowed to edit frozen files by the orchestrator's brief)
- `contracts.py`: `RunRecord.split` (default None) and `RunRecord.meta` (default {}). Both are backward compatible.
- `config.yaml`: new `final` block (`docs_from_questions`, `variants`), outside the records' config hash. The
  `ours_patterns_max` comment now says the cap is per language list. No value changed, so the config hash is unchanged.
- `run.py`:
  - split from the question file name, and the test-file guard;
  - `--final` with `final_problems()`;
  - `--docs-from-questions [FILES]`;
  - the Ollama version guard over the whole output directory;
  - `split` and `meta` stamped on every record;
  - `trace_stats` in run_meta.
- `eval/aggregate.py`: rows and comparisons per split, `--split`, `Abstain %`, `ts_*` counters, and a mixed Ollama version
  warning. `eval/metrics.py` copies `split`, `meta.ollama_version` and `trace.stats` (as `ts_*`). `eval/sanity.py` has
  split and version checks.
- `approaches/ours/planner.py` + `approach.py` (plan prompt `ours-plan-v3`):
  - per-language pattern lists and a code-enforced language check;
  - one repair call inside the 3-call cap, with no widening after a repair;
  - the pattern cap applies per language;
  - `trace.stats`.
- `approaches/tree_nav.py` (prompt `b4-2026-09-15.2`): PageIndex prefix summaries, subsection titles in navigation entries,
  rank-order packing with unit-boundary narrowing (`narrow_item`, `pack_rank_order`), and `trace.stats`.
- `tools/dev_fix1.sh`: the DEV verification run.

## Dev full fixes (2026-09-15, dev_full; frozen registry edited under the orchestrator's brief)
- `approaches/__init__.py`: `OURS_FLAGS` gained `facts_bare`. `run.py` changed only its `--ablation` help text.
- `approaches/ours/` (`defaults.py` `fact_evidence`, `executor.py` `fact_source`/`fact_context_text`, `approach.py`):
  - fact items show each matched value's source text by default;
  - `facts_bare` keeps the old fact-line packing;
  - `trace.fact_evidence` and `trace.stats` `fact_items` / `fact_lines_sourced` / `fact_lines_fallback` are new;
  - the enrichment prompt, cache key and index signature are unchanged.
- `config.yaml` `final.variants` does NOT list `OURS+facts_bare`. Add it only if the frozen configuration needs it.
- New `tools/dev_full.sh` (driver), `tools/dev_analysis.py` (tables) and `tools/dev_progress.py` (ledger progress).

## CORE PLAN integration (2026-09-16; frozen files edited under the orchestrator's brief)
All edits were tested on a copy of the package, then each file was renamed into place (a new file, never a partial
write) while stage `l3` kept running. `approaches/ours/*.py` was not touched. Details: RUNBOOK section 0d,
INTEGRATION_NOTES.md (every patch marked APPLIED).
- `approaches/__init__.py`: `AGENTIC` registered, `AGENTIC_FLAGS`, `index_level` in `OURS_FLAGS`.
- `contracts.py`:
  - `Query.answer_format`, `answer_format()`;
  - the shared timeline answer prompt `ANSWER_TIMELINE_SYSTEM` (`answer-timeline-v1`), `ANSWER_SYSTEMS`,
    `ANSWER_PROMPT_VERSIONS`, `answer_prompt_version()`, `answer_num_predict()`, and `generate_answer` uses it;
  - `AnswerResult.structured`, `RunRecord.structured`, and `meta.answer_*` stamps in `Approach.answer`;
  - `RECORD_HASH_SECTIONS` (+ `answer`);
  - the Question enum comments.
- `config.yaml`:
  - `answer.timeline_num_predict: 768` (new hashed section; `generation` untouched because caches key on it);
  - `caps.agentic_max_calls: 8`;
  - `final` rewritten for the core plan: `question_files`, the 11 `variants`, per-corpus `settings`,
    `exclude_variants`, `answer_prompts`.
  - The records' config hash changed for every approach. No output directory held query records under the old hash.
- `run.py`:
  - `parse_flags` for `index_level` and AGENTIC, `--index-level`, help texts;
  - `index_cost` in the index copies;
  - `final_problems`: question files, exclusions, per-corpus settings (a list keeps the old global rule), answer
    prompt versions;
  - `--dry-run` prints the index directory and completeness per set, plus `DRYRUN-JSON`;
  - `run_meta` gets `answer_prompts`, `answer_formats` and `config_hash_sections`.
- `approaches/agentic/approach.py`: the answer call's `num_predict` comes from `contracts.answer_num_predict` (one
  line). `test_agentic.py`: the budget tests set `caps.agentic_max_calls`, which now wins over `max_calls_default`,
  plus checks for `RunRecord.structured` and the timeline prompt (106 checks).
- `eval/sanity.py`: AGENTIC call cap, `--questions` repeatable or a comma list, and an unknown-question warning.
  `eval/aggregate.py`: AGENTIC in `fairness_warnings`. `eval/test_eval.py`: the final-guard test uses a declared
  question file name.
- `eval/timeline.py` (the coordinator's request): dot-grouped thousands (`4.318.701`) and duration-unit series.
  `eval/test_timeline.py`: 15 new checks, and the test file's reference answers must score 1.0 (86 checks).
  `tools/timeline_questions.py`: validation passes the duration kinds, the same way scoring does.
- New files: `tools/core_plan.sh` stages index/dev/test/aggregate/all/status/plan, `tools/core_plan.py` and
  `tests/test_core_integration.py` (60 checks).
