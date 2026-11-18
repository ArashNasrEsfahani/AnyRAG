# AGENTIC v3 — a knowledge base that fills itself, and a budgeted agent on top of it

Status: DESIGN, written 2026-09-20. Nothing here is built yet. Build order is in section 10.
Audience: the engineer (Opus 5) who builds it, and later the method section of the paper.

Rules from the user that this design must respect:
1. Answers in under 1 minute. At most 4 model calls per question.
2. Based on the tree structure of the report and on a real database.
3. Very brief preprocessing, close to none. No model call at ingestion time.
4. Regex-style search is the core. BM25 is never used inside the method. Vectors are a helper, not the core.
5. A full pipeline: a new document is added and goes straight into the database structure, ready to be asked about.
6. Everything is timed and logged (this is a paper). Everything is saved at every stage and can be stopped and resumed.


## 1. What the evidence says (core test, 2026-09-16 to 09-20)

| Finding | Number | What it means for v3 |
|---|---|---|
| v2 equals vector search on single facts | A 39.4% vs 40.6%, not significant | retrieval is no longer the bottleneck on lookups |
| v2 is the best on timelines | 80% of years, 50% fully correct, 0 false trends (B3: 70 / 30 / 33% false trends) | verification + reconciliation is the real strength; keep it |
| Half the test cannot be scored | locate + multi-section = 0% for EVERY method, including gold evidence (B1) | fix scoring first or no v3 gain is visible |
| Lookup ceiling | B1 74/79, B3 64/79, v2 61/79 | 13 lookups are left to win |
| v2 gives up | 11 of 18 missed lookups say NOT_FOUND; in 9 of the 11 the verifier accepted nothing (119 model rejections, 18 guard rejections) | the 4B verifier is a bad GATE for single facts; 14 correct answers also had 0 accepted rows, saved only by the fallback |
| The database is barely used | one in-memory table per question, value stored as TEXT, one REGEXP filter; 3.8 of 12 candidates come from it (vectors 6.3, both 1.8); removing it costs 1 dev question | make the database the first path, with typed values and SQL doing real work |
| Verification matters on timelines | dev, no_verify: fully correct series 38% -> 0%, false trends 0 -> 25% | never skip it on timelines |
| Vectors carry recall | dev, no_vectors: A 75 -> 60, timeline coverage 74 -> 54 | keep vectors, but as the second path |
| Cost today | 3 calls per fact, 4 per timeline; p50 11.6 s, max 48 s; verify is the slow call (4-18 s) | skipping verify when safe is the main speed lever (no_verify p50 = 4.0 s) |
| Ingestion cost without a model | PDF extraction median 7.2 s (mean 16 s); regex facts median 68 ms per report (23,712 facts from 39 documents in 3.6 s); bge-m3 5.7 s per 100k characters | a new report can be queryable in about 10 s with zero model calls |


## 2. The idea in one paragraph

Every report is ingested once, without any model, into ONE persistent SQLite file: the section tree, the text units, and a
table of typed facts (number, unit, year, label, where it sits in the tree) found by regex. A question is planned by one
model call, then answered FROM THE DATABASE FIRST: regex over fact labels, section titles and unit text, narrowed by
typed filters (unit kind, year, company) and by the tree. SQL does the timeline work: group by year, find missing years,
find conflicting restatements. Vectors are computed lazily and used only when the database path is not enough. The model
verifies candidates only when the program cannot decide alone, and always for timelines. Verified facts are written back,
so the database gets better as it is used ("index by use").

What is new compared with v2: the database is persistent and typed, it is the first retrieval path instead of a side
channel, SQL replaces Python for tree expansion and reconciliation, verification is gated instead of mandatory, and
there is an ingestion pipeline.


## 3. Architecture

    corpus/inbox/  or  `python -m ragbench.kb add <file>`
          |
      INGEST (no model)   register -> extract -> Document tree -> load -> typed facts -> passages -> READY
          |                                                                                      |
          v                                                                                      v  (background / lazy)
      work/kb/kb.sqlite   documents | sections | units | facts | tri (trigram) | passages(+vec) | verified_facts | logs
          ^
      QUERY (AGENTIC3)    plan (1 call) -> DB retrieval (SQL + regex) -> sufficiency check -> [vectors if needed]
                          -> tree expansion (SQL) -> verify (0-2 calls, gated) -> SQL reconcile (+ gap fill)
                          -> answer (1 call, the SHARED prompt) -> write back verified facts

Two new packages:
- `ragbench/kb/`                     the knowledge base: schema, ingestion, typed normaliser, query helpers, CLI, watcher
- `ragbench/approaches/agentic3/`    the approach, registered as "AGENTIC3" in `ragbench/approaches/__init__.py`

Reused as is (do not fork): `corpus/model.py` (Document / Section / Unit and their ids), `corpus/extracted.py` and
`corpus/kareone.py` (adapters), `tools/extract/extract.py`, `approaches/ours/lite.py::extract_facts` (L0 regex facts),
`approaches/agentic/planner.py`, `verify.py`, `numbers.py`, `reconcile.py` (rules and texts), `approaches/agentic2/tree.py`
(passages + expansion semantics), `approaches/agentic2/lazyvec.py` and the shared embedding cache, `util.KVCache`,
the per-call cache and `CallBudget` of v1, the ledger, the journal.


## 4. Ingestion pipeline (the user's requirement 5)

### 4.1 Commands

    python -m ragbench.kb add <file-or-folder> [--company snapp --year 1403 --lang fa --corpus U --pair-of <doc_id>]
    python -m ragbench.kb watch [--inbox corpus/inbox]        # polls every 2 s, ingests whatever is dropped in
    python -m ragbench.kb sync --sets A_fa,A_en,B_fa,B_en,C   # loads the benchmark corpora from the existing Document cache
    python -m ragbench.kb embed [--doc <doc_id>]              # fill missing vectors now (normally lazy)
    python -m ragbench.kb remove <doc_id>
    python -m ragbench.kb status                              # documents, rows per table, stage timings, DB size
    python -m ragbench.kb sql "SELECT ..."                    # read-only, for analysis and for the paper

### 4.2 Stages (each one timed into `ingest_log`, each one committed, so a stop at any point loses nothing)

| # | Stage | What happens | Model? | Measured cost |
|---|---|---|---|---|
| 1 | register | sha256 of the file; same hash already ingested -> no-op. Company / year / language from the arguments, else from the file name pattern `<company>_<type>_<year>_<lang>.<ext>`, else from the title. Builds the doc id `<corpus>.<key>.<lang>` and appends a row to `corpus/natural/manifest.csv` (new files only) | no | ms |
| 2 | extract | PDF / HTML -> `corpus/extracted/<company>/<stem>/` with `tools/extract/extract.py` (already idempotent and atomic). A KareOne report folder goes through the `kareone` adapter instead | no | median 7.2 s, mean 16 s per PDF |
| 3 | model | `extracted.build_extracted` -> Document: section tree (PDF outline, else headings, else pages) and units | no | under 1 s |
| 4 | load | ONE transaction: `documents`, `sections` (with parent_id), `units`, trigram rows | no | under 1 s |
| 5 | facts | `extract_facts` per section -> typed normaliser (4.4) -> `facts` rows + trigram rows | no | median 68 ms per report |
| 6 | passages | `agentic2.tree.build_passages` -> `passages` rows, vectors still NULL | no | ms |
| 7 | ready | `documents.status = 'ready'`; the document is now answerable through the database path | - | total about 10 s for a typical PDF |
| 8 | embed | background thread of the watcher, or first question that escalates: bge-m3 for passages with NULL vec, written into `passages.vec` and into the shared embedding cache. Then `status = 'vector_ready'` | embedding model only | about 6 s per report |

Quality gate: extraction verdict `fail` -> `status = 'rejected'` with the reasons, nothing loaded. Verdict `suspect` ->
loaded with `quality = 'suspect'` (the agent may down-rank it; the benchmark keeps today's rule).

### 4.3 Guarantees

- Idempotent: identity is (doc_id, content sha1 of Document.text + versions). Same content -> no-op.
- Update: same doc_id, new content -> one transaction deletes the document's rows (ON DELETE CASCADE) and inserts the new
  ones. `verified_facts` rows keep the old `doc_content_sha1`, so they stop matching and are ignored (never trusted stale).
- Versions: `meta` stores schema, FACTX_VERSION, normaliser version, TREE_VERSION, embed model. A version change marks
  documents `stale`; `kb sync` / `kb add --refresh` rebuilds only the affected stage (facts or passages), not extraction.
- Crash safety: WAL mode, one commit per stage per document. A document is visible to queries only when `status` is
  `ready` or `vector_ready`.
- Pairing: a Persian and an English edition with the same company, type and year are linked through `pair_id`.
- Fairness: only what every approach may see goes into retrieval columns — doc id, title, text, tree, unit kind.
  `Document.meta` is never copied (OWNERSHIP rule 1). Company and year come from the doc id / title exactly as
  `agentic/docmeta.py` does today.
- Isolation: documents added by the user get corpus letter `U`. They can never enter a benchmark set by accident.
  The benchmark loads through `kb sync`, from the SAME Document objects the other approaches use, so unit ids are
  identical and evidence recall is scored the same way.

### 4.4 Typed normaliser (`ragbench/kb/typed.py`) — what makes it a database and not a text table

Input: one L0 fact (`metric | value | unit | period`, unit_id, score) + its unit text. Output columns:

- `value_num`     the primary reading as a float WITH the scale applied ("4.4 میلیون" -> 4400000; "47/8" -> 47.8;
                  "2،788،550" -> 2788550). Uses `agentic/numbers.py::find_numbers` so the readings are the same ones
                  the verbatim guard uses. `value_hi` for ranges ("150 تا 550"). `alt_readings` (JSON) for glitch tokens
                  (group-reversed thousands, Snapp 1401).
- `unit_norm`, `unit_kind`   kind in: people, count, money, percent, mass, energy, length, area, time, ratio, other.
                  `currency` (toman, rial, usd, eur, ...), `is_percent`. For money, `value_cmp` = value in rial-equivalent
                  when the currency is toman or rial, so restatements in the other unit are still compared.
- `year`, `year_to`, `period_kind` (year, half, quarter, month, date, range), `calendar` (jalali, gregorian),
  `year_source` (`own` = found in the fact's clause or table column, `doc` = inherited from the document's year, NULL).
- `label`, `label_fold` (normalised + folded, the regex target), `row_label`, `col_header` for table cells.
- `def_class`     the definition class from `reconcile.DEF_CLASSES` applied to label + quote at INGESTION time
                  (active, registered, total, ...). This is what lets SQL see a definition change without a model.
- `source_kind`   text, kpi (detached number + label), table. `char_start`, `char_end`, `quote` (240 chars), `score`.

Unit tests must cover every number format listed in the docstring of `agentic/numbers.py`, both languages.


## 5. Database schema (`ragbench/kb/schema.py`, file `work/kb/kb.sqlite`)

    PRAGMA journal_mode = WAL;  PRAGMA foreign_keys = ON;  PRAGMA synchronous = NORMAL;

    CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT);

    CREATE TABLE documents (
      doc_id TEXT PRIMARY KEY, corpus TEXT, key TEXT, lang TEXT, company TEXT,
      year_from INTEGER, year_to INTEGER, calendar TEXT, title TEXT, title_fold TEXT, pair_id TEXT,
      source_path TEXT, source_sha256 TEXT, content_sha1 TEXT, quality TEXT, status TEXT,
      n_sections INTEGER, n_units INTEGER, n_facts INTEGER, n_chars INTEGER, versions TEXT, ingested_at TEXT);

    CREATE TABLE sections (
      section_id TEXT PRIMARY KEY, doc_id TEXT NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
      parent_id TEXT, level INTEGER, ord INTEGER, title TEXT, title_fold TEXT, path TEXT, path_fold TEXT,
      span_start INTEGER, span_end INTEGER, subtree_end INTEGER, n_units INTEGER, n_facts INTEGER);

    CREATE TABLE units (
      unit_id TEXT PRIMARY KEY, doc_id TEXT NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
      section_id TEXT NOT NULL, ord INTEGER, kind TEXT, lang TEXT, text TEXT, text_fold TEXT,
      span_start INTEGER, span_end INTEGER, table_id TEXT, n_numbers INTEGER);

    CREATE TABLE facts (
      fact_id INTEGER PRIMARY KEY, doc_id TEXT NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
      section_id TEXT, unit_id TEXT, ord INTEGER,
      label TEXT, label_fold TEXT, row_label TEXT, col_header TEXT,
      value_text TEXT, value_num REAL, value_hi REAL, value_cmp REAL, alt_readings TEXT,
      unit_text TEXT, unit_norm TEXT, unit_kind TEXT, currency TEXT, is_percent INTEGER,
      period_text TEXT, year INTEGER, year_to INTEGER, period_kind TEXT, calendar TEXT, year_source TEXT,
      def_class TEXT, source_kind TEXT, char_start INTEGER, char_end INTEGER, quote TEXT, score REAL);

    CREATE TABLE passages (
      passage_id TEXT PRIMARY KEY, doc_id TEXT NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
      section_id TEXT, ord INTEGER, unit_ids TEXT, text TEXT, embed_sha1 TEXT,
      vec BLOB, dim INTEGER, model TEXT, embedded_at TEXT);

    -- trigram index: an INDEXED PREFILTER for regex, never a ranker (no bm25(), no rank column is ever read)
    CREATE VIRTUAL TABLE tri USING fts5 (text_fold, kind UNINDEXED, ref_id UNINDEXED, doc_id UNINDEXED,
                                         tokenize = 'trigram');      -- kind: fact_label | unit | section_title

    CREATE TABLE verified_facts (
      vf_id INTEGER PRIMARY KEY, doc_id TEXT, doc_content_sha1 TEXT, unit_id TEXT, fact_id INTEGER,
      entity TEXT, metric TEXT, terms TEXT, value_text TEXT, value_num REAL, unit_norm TEXT, year INTEGER,
      def_class TEXT, definition TEXT, quote TEXT, guard TEXT, model TEXT, prompt_version TEXT,
      question_hash TEXT, run_id TEXT, verified_at TEXT);

    CREATE TABLE ingest_log (id INTEGER PRIMARY KEY, doc_id TEXT, stage TEXT, started_at TEXT, ms REAL,
                             rows INTEGER, ok INTEGER, detail TEXT);

    CREATE INDEX ix_facts_doc_year ON facts (doc_id, year);
    CREATE INDEX ix_facts_kind_year ON facts (unit_kind, year);
    CREATE INDEX ix_facts_unit ON facts (unit_id);
    CREATE INDEX ix_sections_doc ON sections (doc_id, ord);
    CREATE INDEX ix_sections_parent ON sections (parent_id);
    CREATE INDEX ix_units_section ON units (section_id, ord);
    CREATE INDEX ix_docs_company ON documents (company, year_from);
    CREATE INDEX ix_vf_lookup ON verified_facts (doc_id, year, unit_norm);

Checked on this machine: Python 3.14 ships SQLite 3.50.4; the trigram tokenizer matches Persian substrings
(LIKE and MATCH both found "رانند" inside a Persian sentence). REGEXP is a Python function registered on the
connection, as in v2.

Why a trigram index and not BM25: it answers "which rows CAN match this regex" (rows containing the literal stems of the
plan terms), and the regex then decides. No term weighting, no ranking. At today's size (about 24k facts) a full REGEXP
scan takes milliseconds, so the index is for scale; it is an ablation (`no_trigram`) and gets its own scaling measurement.


## 6. Query pipeline (AGENTIC3)

Budget: `caps.agentic3_max_calls: 4`, `caps.agentic3_max_seconds: 60` (config.yaml, fairness-critical). The clock rule is
v2's: program time + recorded wall time of each call, first-touch embedding logged as stage `lazy-index` and reported
separately. NEW: also report the latency WITH first-touch embedding as a headline column, because the user's
one-minute rule is about what a person waits for.

### Stage 1 — plan (1 call)
The v1 planner unchanged in role: entity, metric, kind (lookup, timeline, comparison), terms in fa AND en, period, target
documents chosen from `documents` (company, years, title) — the listing now comes from SQL instead of `docmeta`.
One change, new PLAN_PROMPT_VERSION, tuned on dev only: the plan may return up to 3 FACETS (each facet = a few terms)
for questions that need several sections. Today multi-section evidence recall is 0.63 against 0.90 for lookups.

### Stage 2 — database retrieval (no model, target under 300 ms)
All four run on the target documents only:

a. facts         trigram prefilter on the stems of the plan terms -> `label_fold REGEXP any_term` (patterns from
                 `planner.term_patterns`, both languages, morphology) -> typed filters: `unit_kind` compatible with the
                 plan's expected kind, `year BETWEEN y0 AND y1 OR year IS NULL`. Score: distinct terms matched, all
                 terms matched, unit-kind match, `year_source = 'own'` over `'doc'`, table / kpi bonus, extractor score.
b. sections      `title_fold / path_fold REGEXP term` -> sections whose TITLE names the topic. Used to boost facts and
                 units under them (tree prior) and to answer locate questions (the path is the answer).
c. units         for non-numeric questions (answer types short and span): trigram prefilter -> `text_fold REGEXP` with a
                 term-near-term window rule -> unit windows. This is the v1 grep search, moved into the database.
d. verified      `verified_facts` rows for the same document content, entity / metric terms and year. WARM MODE ONLY
                 (section 8).

### Stage 3 — sufficiency check (no model). Decides the path; the decision and its reasons go into the trace.

| Path | When | Calls | Expected time |
|---|---|---|---|
| FAST | lookup; the top fact matches every plan term (at least 2), unit kind and year agree, and no other candidate with a DIFFERENT value scores within a margin; or a warm verified fact exists | plan + answer = 2 | about 4-5 s |
| STANDARD | the database gave enough term-matching candidates, but the FAST test failed | plan + verify + answer = 3, NO embedding call | about 8-12 s |
| ESCALATED | too few database candidates (non-numeric, locate, multi-section, extraction lost the label) | lazy vectors over the target documents, merged with the database candidates as in v2; 3-4 calls | as v2, 12-30 s |
| TIMELINE | kind is timeline or comparison | always verified, 4 calls | 20-35 s |

The FAST gate is tuned on dev for precision, not coverage: a wrong fast answer is worse than a slow right one.
Dev warning to respect: no_verify cost 2 of 12 dev lookups, so the gate must be strict.

### Stage 4 — tree expansion in SQL
Same semantics as `agentic2/tree.py` (a number gets its label, a table row gets caption + header, every candidate carries
its section path), but read from the tables: ancestors by a recursive CTE on `sections.parent_id`, neighbours by
`units.ord`, table header by `table_id`. Candidates under a section found in 2b get the tree prior. Cap 12 candidates
(24 over several documents); with facets, every facet keeps at least 2 slots.

### Stage 5 — verification, gated (0 to 2 calls)
The v1 batched verifier and the verbatim guard, with the two v2 repairs (Arabic-comma thousands, label completion).
Change of ROLE for single facts: the verifier RANKS, it no longer GATES. If it accepts nothing, the evidence pack is the
ranked candidates in the plain format every baseline uses, with no "nothing was verified" note. Reason: in 9 of the 11
NOT_FOUND misses the verifier had accepted nothing, mostly model rejections, and a note like that pushes a 4B answerer
to abstain. (Hypothesis to confirm in M0 by reading the 18 traces before changing anything.) For timelines the verifier
keeps its gate: only verified rows enter the series.

### Stage 6 — reconciliation in SQL (timelines and comparisons)
Verified rows go into a temp table `cand`, joined to `documents`:

- role        authoritative when `documents.year_from <= cand.year <= documents.year_to`, restatement when the
              document is later, projection when it is earlier (kept out) — the v1 rules, now one CASE expression.
- series      `GROUP BY year, def_class`, authoritative first, else the latest restatement.
- conflicts   same year, same def_class, same unit kind, `value_cmp` differing by more than the tolerance, from
              different documents -> flagged, the authoritative value stays.
- comparable  false when the kept rows do not share one def_class -> no trend is stated (v1 behaviour, 0 false trends).
- gaps        years of the period with no row. ONE gap-fill round without a model: the missing year's own report AND
              the next year's report (restatements live there), queried with the labels of the rows already verified
              (their wording is the best regex we have), plus vectors if present. A second verify call runs only if
              the time budget allows (v2's estimate rule).

This closes a v2 weakness: its verified table held fewer years than its final answer, so part of the answer rested on
unverified text.

### Stage 7 — answer (1 call)
The SHARED answer prompt and the shared timeline prompt, the shared 6000-character evidence budget. No change allowed
here; it is what makes the comparison fair.

### Stage 8 — write back
Every verified row is inserted into `verified_facts` with the document content hash, plan terms, definition, quote,
guard result, model and prompt version. Used for retrieval only in warm mode.


## 7. Step zero: make the other half of the test measurable

Without this, nothing v3 does for locate and multi-section questions can be seen. Two scorers, both reported:

1. Key-fact scorer (program, free, deterministic). Each multi-section and locate question gets a `key_facts` list: the
   numbers and the short strings a correct answer must contain (drafted automatically from the gold answer's numbers
   and the gold section titles, then reviewed by hand; the core test has 17 + 13 such questions, dev has its own).
   Score = share of key facts present (numeric match through `numbers.py` readings, strings through folded match);
   correct = all present. The strict matcher stays the headline for lookups.
2. LLM judge through OpenRouter for every run. Needs the user to raise the cap on the key (about $0.30-0.60 needed,
   $0.54 left under a $5 cap). Used to validate the key-fact scorer on a sample (agreement reported).

Re-score ALL existing runs with the new scorer (no GPU needed) and regenerate the four HTML pages before building v3, so
the baseline for v3 is known.


## 8. Evaluation protocol

- Tuning on dev only (dev_core, dev_core_timeline, pilot_timeline), as for v2. Test once, with `--final`.
- COLD mode is the headline: `verified_facts` is not read. Every question is independent, comparable to every baseline.
- WARM mode is a separate experiment ("index by use"): a fresh database copy per run, a fixed seeded question order,
  write-back on, and the curve of calls, latency and accuracy against the number of questions already asked. Verified
  facts come from the documents, never from gold answers, so there is no label leakage; dev and test never share a
  database file.
- Ingestion benchmark (new table in the paper): for all 115 extracted documents, time per stage, rows per table, database
  size, and "time from file to first answer" for a document added to a database that already holds the rest.
- Live-add test: ingest a held-out report while the watcher runs, ask a question immediately, record the path taken
  (database path before the vectors exist, escalated path after).
- Path report: share of questions per path (FAST, STANDARD, ESCALATED, TIMELINE), accuracy and latency per path.
- Ablations (flags): `no_typed` (v2-style text facts), `no_trigram`, `no_sql_reconcile` (v1 Python reconcile),
  `no_fastpath`, `verify_gate` (v2 behaviour: verifier gates single facts), `vectors_always` (v2 behaviour),
  `no_vectors`, `no_facets`, `no_tree_prior`, `warm`.
- Costs as before: calls, tokens, latency p50 / p95 / max, GPU-hours, plus preprocessing seconds per document.
- Disclosed limitation, carried over and extended: v2 and v3 were designed after the core test results were seen. The
  clean check is the full 316-question test, of which the core subset is 61.


## 9. Code layout

    ragbench/kb/__init__.py
    ragbench/kb/schema.py        DDL, migrations, connection factory (WAL, REGEXP, foreign keys)
    ragbench/kb/typed.py         typed normaliser (4.4)
    ragbench/kb/ingest.py        stages 1-8, idempotency, update / remove, ingest_log, journal events
    ragbench/kb/query.py         doc selection, facts / sections / units retrieval, tree CTEs, vector search over BLOBs
    ragbench/kb/watch.py         inbox polling + background embedding
    ragbench/kb/cli.py           add | watch | sync | embed | remove | status | sql
    ragbench/kb/test_kb.py       offline tests (no model): schema, typed values, idempotency, update, cascade, trigram + Persian
    ragbench/approaches/agentic3/{__init__,approach,defaults,route,retrieve,reconcile_sql,writeback,test_agentic3}.py
    ragbench/eval/keyfacts.py    the key-fact scorer + `aggregate --scoring keyfacts`
    ragbench/tools/core_plan.sh  new stages: kbsync, v3dev, v3test   (stoppable / resumable like the others)
    ragbench/tools/agentic3_report.py -> work/runs/core/AGENTIC3_COST.md and work/runs/core/INGEST_COST.md
    config.yaml                  caps.agentic3_max_calls, caps.agentic3_max_seconds, kb.path, kb.inbox, kb.embed (lazy | eager)

Write every script with the Write tool, never through a Bash heredoc (backslashes get corrupted on this machine).


## 10. Build order, with the check that closes each step

| # | Step | Done when |
|---|---|---|
| M0 | Key-fact scorer; re-score all runs; regenerate the 4 HTML pages. Read the traces of the 18 missed v2 lookups and write down the cause of each | B1 on locate + multi-section is no longer 0; the HTML shows the new columns; a short failure table exists |
| M1 | `kb` schema + typed normaliser + offline tests | tests pass; every number format of `numbers.py` round-trips in fa and en |
| M2 | Ingestion: `add`, `sync`, `remove`, `status`; then `watch` + background embedding | `kb sync` loads all benchmark sets; a second run is a no-op; killing it mid-run and restarting loses nothing; INGEST_COST.md exists |
| M3 | AGENTIC3 retrieval + paths + SQL tree expansion, with verification still as in v2 (`verify_gate`) | dev results equal or beat v2 on dev; path report printed |
| M4 | Gated verification, plain fallback, FAST path | dev: lookups up, p50 latency down, no loss on timelines |
| M5 | SQL reconcile + gap fill | dev timelines: coverage and fully-correct at or above v2, 0 false trends |
| M6 | Facets for multi-section; locate from the tree | dev multi-section and locate up under the key-fact scorer |
| M7 | Write back + warm experiment | warm curve produced on dev |
| M8 | Dev ablations, then the test ONCE (`core_plan.sh v3test`), reports, HTML pages, memory update | AGENTIC3 rows in all four pages; AGENTIC3_COST.md and INGEST_COST.md written |

Expected GPU time: dev iterations about 1.5 h, ablations about 1 h, test about 1.2 h (v2 took 1.71 h for test + tuning).


## 11. Risks and open questions

1. L0 facts miss labels that the PDF extraction detached from their numbers. The kpi rule recovers many, not all; the
   ESCALATED path exists for the rest. Measure: share of lookup gold values present in `facts` at all (a recall ceiling
   for the database path, computable without any model).
2. The FAST gate may be too rare to matter, or too loose. Dev decides; report the share honestly either way.
3. `def_class` at ingestion is regex-only and coarse. It is a prior; the verifier's definition still wins for timelines.
4. Warm mode can spread a wrong verified fact to later questions. Mitigation: the guard result is stored, only
   guard-passed rows are reused, the content hash must match, and warm is never the headline.
5. Facets change the plan prompt, so plan quality must be re-checked on the dev lookups (no regression allowed).
6. Novelty stays modest and should be written that way: the contribution is the controlled study plus a documented,
   cheap, database-first design with index-by-use; not a new retrieval principle.
7. Still open from before: keep or swap timeline question T-C-TL-snapp-05; disable the Ollama auto-update
   (runs are pinned to 0.32.15).


## 12. What the build found against what this spec predicted (written 2026-09-21, dev only)

This section is the spec's own scorecard. It was written after the dev evaluation and before the test set was run,
and it is deliberately unflattering: **three of the six design bets in section 2 are withdrawn on evidence, and
one more framing has been corrected since.** Nothing here is a test result. Every number is dev.

### The bets that paid

| bet | where it is measured | what it turned out to be worth |
|---|---|---|
| **Zero-model ingestion** | `work/runs/core/INGEST_COST.md`, AGENTIC3_COST.md | Stands as written. 129 documents, 63,388 typed facts, 26,571 passages, no model call in stages 1-7; **0.17 s median from a file on disk to answerable** when the extraction exists. Embedding is the only GPU part and is deferred to first use. |
| **Typed facts (the typed normaliser, §4.4)** | V3_ABLATIONS.md, round 2 | The only ablation outside the round-2 noise band: `no_typed` costs **-0.20** on dev A strict and -0.18 on timeline coverage, against a band of +-0.15. This is the one part of the design the measurement defends on its own. |
| **SQL reconciliation replacing Python (§6 stage 6)** | `ragbench/approaches/agentic3/test_reconcile_sql.py` | `reconcile_sql` and `agentic/reconcile.py` agree on **709 of 709** historical cases of the fixed set `reconcile-diff-2026-09-21` (comparison 150, single 322, timeline 237), on every differential field, plus the hand-built cases history has no example of. It is a faithful reimplementation, not a rewrite that changed behaviour. |
| **The key-fact scorer (§7, step zero)** | every dev table in this build | Not a retrieval bet, and it paid more than most of them. On the wide set 31 of 64 records carry key facts, and strict scores **0.00 for every method including AGENTIC2** on multi-section and cross-year questions. Without it those 20 records are unmeasurable, and the route table in particular is unreadable. |

### The bets that were withdrawn

| bet | the claim in this spec | what dev measured |
|---|---|---|
| **"The database is the first retrieval path; vectors are used only when it is not enough"** (§2) | a routing principle | **False as stated.** On the wide dev set the 22 records the database-first routing kept away from the vectors scored 0.41 strict against 0.73 for the records it sent to them, and always merging the vectors in took key-fact 0.672 -> 0.750 with 5 records won and 0 lost. `vector_policy = "always"` since 2026-09-20. The database adds reconciliation, typed filtering and SQL **on top of** a vector pipeline; it does not replace retrieval. |
| **The 2-call FAST path** ("a lookup the database settles alone is answered in 2 calls and about 4-5 s") | §6 stage 3 | **It does not exist on this data.** FAST fired **0 times in 428 dev records**. The gate was reached 103 times, and a free replay under every relaxation shows why: on the 69 gate-reaching records with a gold number, the rank-1 database candidate's own value is the gold value **2 times (0.03)**, and a gate answers from rank 1. `fast_cold_gate = False`; the gate still runs and still records its refusal, because the refusal mix is the finding. |
| **Index by use** ("verified facts are written back, so the database gets better as it is used") | §2, §6 stage 8 | **Nothing was saved.** The first warm arm's speed came from a routing bug - the route short-circuited to FAST on ANY non-empty warm rows, matched by an OR over the plan's terms, so rows written by a different question about the same report answered this one. With `warm_check` added and the arm re-run on a fresh database copy, warm makes **3.29 calls against cold's 3.29**, is 0.9 s slower, and is not distinguishable on accuracy. The gate was reached 5 times and refused all 5 with `warm_metric_mismatch`. |

### One more correction, after the spec's own bets

**The cap was not the lever, and the experiment that assumed it was said so.** v3's evidence recall trailed
AGENTIC2's (0.678 against 0.760), and the obvious reading was that the candidate cap of 12 cut the gold unit. A
free replay of the recorded plans confirmed half of that - the gold unit is in a targeted document on 59 of 59
records and in the pool on 0.960 of them, and the cap cuts it on 24 - and refuted the other half: raising the cap
to 24 changes the coverage that survives `pack_evidence`'s 6000-char budget on **exactly zero records**, because
twelve candidates already render about 5300 characters. The GPU arm that would have tested a bigger cap was
**not run**, on the strength of the replay. What worked was the vector reservation
(`vec_min_slots` 4 -> 8), which changes WHICH candidates hold the early, packable slots: evidence recall
0.698 -> 0.781 against AGENTIC2's 0.760, better on 16 records and worse on 4, paired p = 0.0118. Details in
V3_ABLATIONS.md, round 4.

### Route versions, and what the records carry

`ROUTE_VERSION` in `defaults.py` is now **`agentic3-route-v3`**, and it is stamped into every record's
`trace.versions.route`, so arms from different routings never mix. The constant is stamped at RUN time, which is
one step behind the decision each arm was run to make:

| `trace.versions.route` | what those records are |
|---|---|
| `agentic3-route-v1` | the **retired database-first routing**: vectors reached only through `escalate_check`, and the cold FAST gate acted on. `work/runs/core_dev`, `work/runs/core_dev_pilot_timeline`, `work/runs/core_dev_wide`, `work/runs/v3warm`, `work/runs/v3warm2`. The `AGENTIC3+vectors_always` rows of the wide set carry v1 too: they ran the flag before the constant was bumped. |
| `agentic3-route-v2` | vectors merged in on every question, cold FAST gate evaluated but not acted on, vector reservation still 4. `work/runs/v3recall_ctl`, `work/runs/v3recall_vec8`. |
| `agentic3-route-v3` | the same as v2 with `vec_min_slots = 8`. **No records on disk carry it.** The arm that measured it (`v3recall_vec8`) ran while the constant still said v2, and the new default was then checked against that arm on 3 records - retrieved units, evidence units and answer text all equal - so the behaviour is the arm's, and only the label differs. |

### What section 11's risks look like now

- **Risk 1 (extraction detaches a number from its label)** is real and measured: 3 of 25 dev timeline period values
  are in the raw unit text and absent from `facts`. It is 0 of 53 on lookups. See `work/runs/core/DB_CEILING_V3.md`.
- **Risk 2 (the FAST gate may be too rare to matter)** resolved as "too rare", and is reported as a negative result.
- **Risk 4 (warm mode can spread a wrong verified fact)** materialised exactly as written, through the route rather
  than through a wrong value, and both cures named in the risk were implemented before the re-run.
- **Risk 6 (keep the novelty claim modest)** is the one to hold to. After three withdrawals the contribution is the
  controlled study plus a typed, persistent, zero-model-ingestion database that **adds to** a vector pipeline. The
  words "database-first", "index by use" and "2-call fast path" should not appear in the paper as properties of the
  method.

### The disclosure that travels with all of this

AGENTIC3 was designed after the core test results of v1, v2 and the baselines had been seen, and two of its
mechanisms (`guard_escape` / `guard_escape_label`, and `rank_by_verdict`) were designed from analyses that read
test questions' gold answers - see the TEST-SET DISCLOSURE at the top of `defaults.py`. Separately, **10 of the
64 wide-dev records are questions AGENTIC2 was tuned on**: `dev_core.jsonl` is a subset of `pilot.jsonl`. No
approach saw a test question at query time, and **v3 has not been run on the test set.**
