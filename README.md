# AnyRAG

**Choose the best RAG for your project.**

Code, benchmark and full results for the paper
*What Reaches the Reader: A Comparison of Retrieval-Augmented Generation Setups in Persian and English*
(Arash Nasr Esfahani, University of Tehran; journal manuscript in `paper_journal/`, arXiv ID to be added).

Project page: https://arashnasresfahani.github.io/AnyRAG/

## What the study found

Retrieval-augmented generation offers many design choices, and most comparisons of them use English. We built a
Persian-English benchmark (156 questions over natural company reports, non-fiction books and statutes, in three tiers:
single fact, within-document integration, cross-document integration; with annotated gold evidence) and compared
retrieval channels, chunk size, query rewriting, embedding models, reranking and LLM selection, evidence budgets and
iterative search. Unless stated, numbers are for the primary reader (GLM-5.3-flash), the pinned judge (gpt-oss-120b)
and the 90 answerable test questions.

- **The confirmatory test was inconclusive.** A frozen, one-shot test of four pipelines (protocol fixed in advance)
  failed its gate: the selection pipeline Lex-RAG (`LEX5B` in the code) answered 5 to 7 more questions than a
  BM25+dense hybrid (HYBRID) and 0 to 3 fewer than a cross-encoder reranker (HR), and no pipeline separated from HR.
  Under the symmetric human/machine merge HYBRID, HR, Lex-RAG and a SetR adaptation answer 71, 76, 76 and 78 of 90;
  none of the selection pipelines is clearly better than HR.
- **Everything else is post hoc** analysis of the same 90 answerable questions (see "Post hoc experiments" below).
- **Answers are lost before the reader, not in it.** Given only the gold evidence, the reader answers 83 of 90. For 15
  to 18 questions no complete evidence bundle reaches the top 40 candidates, and for 11 to 15 more it is dropped when
  the evidence pack is built.
- **Reranking or selection helps most at tight budgets.** Its value fades as the budget grows and plain hybrid
  retrieval catches up, at several times the text. Iterative search has the best point estimate at about four times the
  tokens, within noise of the reranker.
- **Components.** Hybrid search beats either channel alone; 1,000-character chunks beat larger ones at a
  6,000-character budget; query rewriting (HyDE, multi-query) and a second embedding model (multilingual-e5-large) do
  not help.
- **Persian and English.** Persian evidence costs 1.5 to 3 times the tokens per character. When a question is asked in
  the other language, BM25 fails while dense retrieval holds.
- **Power and robustness.** With 90 questions only differences of about 11 answers are detectable, and the order of the
  middle pipelines changes with the judge, the reader draw and the review design. Every recommendation in the paper
  therefore carries a confidence label; the suggested starting point is hybrid retrieval with a cross-encoder reranker.
- An earlier human review that covered more of Lex-RAG's answers than the controls' produced a spurious significant
  advantage; it is withdrawn and kept only as a superseded record (`results/test/superseded_asymmetric_review/`).

## Repository layout

```
code/
  ragbench/            the benchmark harness and every method (frozen test commit 77924d4)
    approaches/        HYBRID, rerank, LEX* variants, SETR, AGENTIC*, baselines ...
    eval/              judge, metrics, timeline scoring
    tools/             reader replay, multi-judge, token bounds, analysis scripts
    tests/             unit tests (including the test-freeze checks)
    config.yaml        shared configuration (paths, models, generation options)
  tools/extract/       PDF/HTML -> unit extraction used to build the corpora
  corpus_build/        scripts that fetched / converted / registered the BOOK and LAW pools
  test_freeze/         the one-shot test: freeze_test.py, run_test.py (plan/run/replay/judge/status), index prebuild,
                       judge pin probe, parser replay
  posthoc/             the post hoc stream scripts (S1-S16, S5 cross-check), one folder per stream, with a README
  test_run/            helpers used during the test: report_test.py, cellstate.py, infra_cleanup.py, finish_todo.py,
                       launch.ps1, ollama_watchdog.ps1, reader_smoke.py, balance_check.py
  requirements.txt
benchmark/
  questions/           {rpt,book,law}_{dev,test}.jsonl: question, reference answer, requirements (rubric),
                       gold evidence (doc_id, unit_id, offsets, short quote, SHA-256), tier, language, pool;
                       *_review_decisions.jsonl and AUTHORING_LOG.md (question review trail)
  decisive_spans/      decisive spans for the 60 dev questions (short gold quotations, <= 390 chars)
  corpus_manifest/     manifest.csv (report pool), manifest_v2.csv + sources_v2.json + FROZEN_v2.json (book and law
                       pools), FROZEN_reports.json: titles, source/landing URLs, editions, SHA-256 of every file
  SCHEMA.json, DESIGN.md, FROZEN_questions.json (hashes of the frozen question files)
results/
  test/                the true test: TEST_RESULTS.md + test_results.json (machine judge),
                       TEST_RESULTS_HUMAN_SYM.md + test_results_human_sym.json (symmetric human/machine merge, the
                       version reported in the paper), runs/ (primary reader), replay/ (5 other readers),
                       packs/ (answer-pack metadata), judge/<reader>/ (verdicts, ledgers, adjudication queue),
                       reruns.jsonl;
                       superseded_asymmetric_review/: the first, asymmetric merge (TEST_RESULTS_HUMAN.md,
                       test_results_human.json). Historical artifact, superseded; do not cite its numbers
  protocol/            PROTOCOL.md, MANIFEST.json, judge and reader pins, token calibration, index manifests
  dev/                 development experiments: DEV60, BUDGET_MATRIX, WIN_LOSS, SETR, LEX4-LEX11 reports and their
                       per-question records
  human_review/        blinded human adjudication exports and unblinding keys (GLM reader, 6,000 characters only):
                       key_test.json + human_test_export.json (first session, 77 items),
                       key_test_sym.json + human_test_sym_export.json (symmetric completion, 39 items); both sessions
                       feed the symmetric merge. key_dev.json + human_dev_export.json (dev, asymmetric, not used
                       for claims)
  protocol/questions/  test_v2judge_all.jsonl, the frozen test question file the report scripts read
  posthoc/             the post hoc experiments of November 2026 (S1-S17, see below): RESULTS.md per stream,
                       results.json and tables, per-question runs, answers, judge verdicts, ledgers, checks
  RESULTS_REGISTRY.jsonl   one row per (set, arm, reader) result reported for the frozen test and dev
paper_journal/         the main version of the paper: the Elsevier (elsarticle) journal manuscript. main.tex +
                       preamble.tex + sections/j*.tex, supplementary.tex (supplementary material), main.pdf and
                       supplementary.pdf, refs.bib, tables/, figures/ (with figures/data); analysis/: the same
                       scripts as paper/analysis/ (revision_stats.py, fallback_table.py, figure_data.py,
                       figure_data_posthoc.py, layout.py, prose and layout checks)
paper/                 the earlier preprint version (ACL format), kept for reference: LaTeX sources and main.pdf;
                       paper/analysis/: revision_stats.py (bounds, sign-test ranges, cluster bootstrap, reader
                       nets, clipping, tokens), fallback_table.py (selector fallbacks by language and tier),
                       figure_data.py and figure_data_posthoc.py (the data in paper/figures/data), layout.py (path
                       resolution), and the prose and layout checking scripts used while editing
skills/choose-rag/     an Agent Skill for Claude and OpenAI Codex that turns the paper's findings into a RAG
                       recommendation (see "Choose your RAG with Claude or Codex"); skills/choose-rag.zip is the
                       same folder zipped for upload to the Claude apps
docs/lex_pipeline.html interactive 3D view of the selection pipeline (open in a browser)
tools/strip_corpus_text.py   the script that removed corpus text from results/ (see Data availability)
```

## Choose your RAG with Claude or Codex

`skills/choose-rag/` is an Agent Skill that reads your repository and documents locally (a read-only scanner, no
dependencies, nothing executed or sent) to infer the eight chooser answers (question types, language, evidence
budget, latency, GPU, extra LLM calls, whether the model may already know the answers, Persian spelling), asks
only what it cannot infer (or all eight when there is no codebase), applies the paper's decision rules and returns a
recommended pipeline, with file-by-file changes when it finds an existing RAG stack. Every part carries its evidence, a
confidence label (Strong / Tentative / Measured) and its source in the paper, plus what to avoid, the trade-offs and
a recipe to check the choice on your own questions. The same folder works in Claude (Claude Code and the Claude
apps) and in OpenAI Codex; see `skills/choose-rag/README.md` for details and an example conversation.

```bash
# Claude Code (macOS / Linux), personal skill for all projects
mkdir -p ~/.claude/skills && cp -r skills/choose-rag ~/.claude/skills/
```

```powershell
# Claude Code (Windows PowerShell)
New-Item -ItemType Directory -Force "$env:USERPROFILE\.claude\skills" | Out-Null
Copy-Item -Recurse skills\choose-rag "$env:USERPROFILE\.claude\skills\"
```

```bash
# OpenAI Codex (CLI, IDE extension)
mkdir -p ~/.codex/skills && cp -r skills/choose-rag ~/.codex/skills/
```

- **Claude apps (claude.ai, desktop):** turn on code execution, then upload `skills/choose-rag.zip` under
  Settings > Capabilities > Skills (menu names may differ by plan and version).
- Then open your project and ask, for example, "Which RAG setup should I use for this project?" or "Review my RAG
  setup", or invoke the skill directly (`/choose-rag` in Claude Code, `$choose-rag` in Codex).
- Check the install: `python ~/.claude/skills/choose-rag/scripts/test_choose.py` (or the Codex path) should print
  `Profiles: 33 of 33 passed` and a sweep with 0 problems; `scripts/test_scan.py` should print
  `Scan checks: 38 of 38 passed`. The scripts also run without an agent:
  `python skills/choose-rag/scripts/scan_project.py path/to/repo --recommend` and
  `python skills/choose-rag/scripts/choose.py --questions`.

## Post hoc experiments (November 2026)

After the one-shot test, the post hoc streams (S1-S17) analysed the **same, already-used test set** (96 questions,
90 answerable). They are exploratory: the test answers were known, so their p-values are descriptive. Every new arm was
fixed on dev or a priori and run once on test, with the same pinned reader and judge as the test; the rebuilt evidence
packs of the frozen arms match the frozen ones byte for byte (576 of 576). The paper's results beyond the confirmatory
test, and its post hoc appendix, are built from these files.

- `results/posthoc/COMMON.md`: the rules every stream followed. `results/posthoc/<stream>/RESULTS.md`: what was run,
  result tables with bounds over unresolved judge cells, cost, what failed, and facts for the paper with their source
  files. Alongside: `results.json` and `*tables*.md` (written by the stream's analysis script), per-question runs and
  packs (corpus text stripped), answers, judge verdicts and adjudication queues, cost and token ledgers of every call,
  and verification and check files.
- `code/posthoc/<stream>/`: the scripts, with `code/posthoc/README.md` listing what each stream does and how to rerun
  it. They hard-code the original `D:/KareOne` paths, as the test code does.
- Streams: S1 second judge and human-review packets; S2a packs, GOLD and budget sweep; S2b run-to-run variance; S3
  agents; S4 routing and long context; S5 statistics (GLMM, power); S6 closed book, listwise reranking, SetR without
  fill; S7 where answers are lost, contamination, tokens, errors; S8 Lex-RAG factorial; S9 GPU checks and a hybrid
  agent; S10 dense vs BM25; S11 query rewriting; S12 chunk size; S13 a second embedding model; S14 language analyses;
  S15 cross-lingual queries and Persian spelling variants; S16 second judge on the post hoc variants; S17 the seven
  questions the reader misses even with the gold evidence.
- **S5 cross-check** (`results/posthoc/S5_stats/crosscheck/`): the GLMM estimates of S5 re-estimated independently
  with standard statsmodels estimators (Bayesian mixed logistic GLM by variational Bayes and posterior mode under two
  priors, GEE and cluster-robust GLM, and pipeline x reader interaction tests), read straight from the S5 cells.
  No model calls.
- **S16** (`results/posthoc/S16_judge2/`): the second judge (Mistral Large) on the twelve post hoc variants that
  only the first judge had scored (hybrid-index agent, the ten component variants at 6,000 characters, the
  cross-language run); $1.22. Agreement with judge 1 is 91.8% (kappa 0.82); 30 of 1,152 cells stay unresolved and
  count as ranges. No contrast reverses; the loss of 2,000-character chunks shrinks to 6 to 13 answers and the
  hybrid-index agent's lead over HR grows to 11 to 12 (`RESULTS.md` sec. 7, `tables.md`).
- **S17** (`results/posthoc/S17_goldmiss/`): a reading of the saved packs, answers and both judges' verdicts for the
  7 of 90 questions the reader gets wrong with the gold evidence (4 insufficient evidence bundles, 2 judge
  strictness, 1 reader error). No new runs.
- Not included: the vectors and indexes these streams built (`.npz`/`.npy`, the S12 500- and 2,000-character
  indexes, the S13 e5 vectors, the S15 query vectors), the downloaded e5 model, run logs, judge SQLite caches, and the
  two new S1 review packets' HTML pages and unblinding keys (the packets themselves are included;
  `build_packets.py` regenerates all of them).

## Setup

```bash
python -m venv .venv            # the study used CPython 3.14.0 on Windows 11
.venv/Scripts/pip install -r code/requirements.txt     # (.venv/bin/pip on Linux/macOS)
```

- **Readers, selectors and judge** run through [OpenRouter](https://openrouter.ai). Put `OPENROUTER_API_KEY=...` in a
  `.env` file at the project root (see `paths.env_file` in `code/ragbench/config.yaml`). Never commit it.
- **Embeddings** (`bge-m3`) and the original local reader run through [Ollama](https://ollama.com); the model digests
  are pinned in `config.yaml`.
- **Cross-encoder reranker**: `bge-reranker-v2-m3` via `transformers` (downloaded from Hugging Face on first use).
- **ripgrep** (`rg`) must be on the PATH or at `paths.rg` for the grep-based baselines.
- Paths: `config.yaml` and the scripts in `code/test_freeze` hard-code the original Windows locations
  (`D:/KareOne/RAG Paper`, `D:/KareOne/RAG-Paper-upgrades`). Edit `paths.project_root` and the path constants at the
  top of those scripts to match your checkout. The re-scoring and analysis scripts (`code/test_run/report_test.py`,
  `code/test_run/human_merge_sym.py`, `paper/analysis/*.py`) resolve their paths from the repository layout and need
  no edits; `--root` / `ANYRAG_ROOT` and `ANYRAG_TEST_DIR`, `ANYRAG_FREEZE_DIR`, `ANYRAG_CODE_DIR` override them.

## Reproducing the results

1. **Rebuild the corpora** (not included, see below): download every document listed in
   `benchmark/corpus_manifest/` from its `source_url`, check the SHA-256, and convert/extract with
   `code/corpus_build/` and `code/tools/extract/`. The extracted-text hashes in `FROZEN_v2.json` let you confirm that
   your unit ids and offsets match the gold evidence.
2. **Re-score without any model call**: `code/test_run/report_test.py` rebuilds `TEST_RESULTS.md` and
   `test_results.json` from the records, verdicts and ledgers in `results/test/` (it overwrites them in place).
   `code/test_run/human_merge_sym.py` rebuilds the symmetric human/machine merge from both review sessions, after
   reproducing the superseded asymmetric merge cell for cell. `python paper/analysis/revision_stats.py` and
   `python paper/analysis/fallback_table.py` recompute the paper's revised statistics and the fallback table.
   All verdicts, token counts and costs are in the released records.
3. **Re-run the test** (costs money; the original run is one-shot and must not be repeated as "the" test): with the
   corpora rebuilt and indexes prebuilt (`test_freeze/prebuild_indexes_v2.py`), `run_test.py plan`, then
   `run`, `replay` and `judge` with `--confirm-one-shot` (see `results/protocol/PROTOCOL.md`).
4. **Dev experiments**: `python -m ragbench.run --help` and `code/ragbench/RUNBOOK.md`.

Unit tests live in `code/ragbench/tests` (some need the rebuilt corpora, indexes or local models).

## Data availability

The early development corpus of internal reports used while building the harness is not part of this release.
The report, book and law corpora are **not redistributed** in this repository. The books (Project Gutenberg,
Wikisource) and statutes (EUR-Lex, legislation.gov.uk, Iranian statute portals) are public, and the company reports are
published by the companies themselves; `benchmark/corpus_manifest/` lists each document's title, edition, source URL,
landing page and file hash, so the pools can be rebuilt from their origins.

Gold evidence in `benchmark/` keeps short quotations (the decisive sentence or table cell, at most a few hundred
characters), which are needed to define the gold answer. Everything in `results/` (including `results/posthoc/`) was passed through
`tools/strip_corpus_text.py`: retrieved passage text, chunk text, evidence views, excerpts, section cards and prompts
that embed evidence were replaced by `"[stripped: N chars]"`, while ids, ranks, scores, answers, verdicts, tokens,
latencies and costs are kept. `python tools/strip_corpus_text.py --check results` verifies that no record still holds a
string longer than 400 characters outside answer/judgment fields. `docs/lex_pipeline.html` keeps passage previews of at
most 140 characters.

**Available on request** (withheld because they embed corpus text or are too large): the judge SQLite caches
(`judge_multi_cache.sqlite`, which store full judge prompts), raw run logs (`answers.log`, `replay.log`, stdout logs),
retrieval indexes and embedding pickles (`.npz`/`.pkl`, about 3 GB), LLM section cards and contextual prefixes (lex6,
lex7), the lex8 synthetic-question field (46 MB), the visualisation data dumps (`viz_*.json`), and the
`review_*.html` adjudication pages.

## Use of AI assistance

This work used AI systems throughout, under the author's direction. Benchmark questions were authored by LLM agents
(Claude) in separated roles, checked by a mechanical verifier and spot-checked by the author. Methods were co-designed
with AI assistants (Claude implementing, a GPT model reviewing as a critic). Experiments, analysis scripts and reports
were executed by AI agents, with the author making every protocol decision. Evaluation used an LLM judge
(gpt-oss-120b) with blinded human adjudication by the author of all four 6,000-character pipelines on the 30 machine-discordant test questions. The paper was drafted with AI
assistance from the result files. See the paper's "Use of AI Assistance" section for details.

## Citation

```bibtex
@misc{nasresfahani2026evidence,
  title         = {What Reaches the Reader: A Comparison of Retrieval-Augmented Generation Setups in Persian and English},
  author        = {Nasr Esfahani, Arash},
  year          = {2026},
  eprint        = {XXXX.XXXXX},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CL},
  note          = {Code and data: https://github.com/ArashNasrEsfahani/AnyRAG}
}
```

See also `CITATION.cff`.

## Licences

The repository is public (since 2026-11-18). The benchmark and the per-question results are also on Hugging Face:
[ArashNasr/anyrag-benchmark](https://huggingface.co/datasets/ArashNasr/anyrag-benchmark) and
[ArashNasr/anyrag-results](https://huggingface.co/datasets/ArashNasr/anyrag-results). Licences:

- **Code** (`code/`, `tools/`, `docs/`, `skills/`): MIT, see `LICENSE`.
- **Benchmark annotations and results** (`benchmark/`, `results/`): CC BY 4.0, see `LICENSE-DATA`.
- The underlying documents keep their own terms (public domain, official legislation, or the publishing company's
  terms); the short gold quotations are included for research purposes.

Contact: Arash Nasr Esfahani, arash.nasr@ut.ac.ir
