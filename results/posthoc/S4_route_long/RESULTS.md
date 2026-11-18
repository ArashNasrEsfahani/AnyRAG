# S4: routing and long context (post-hoc, exploratory; test split already unsealed)

All numbers below come from files in this folder: `routing_tables.md` / `routing_results.json` (experiment 1),
`lc_tables.md` / `lc_results.json` (experiment 2), `pool_stats.json` (sizes), `ledger.jsonl` (every API call made
by this stream). Code is in `code/`. Nothing was tuned on test questions: the router prompt was chosen on dev, and
every test arm ran once.

## 1. What was run

### Common
- Frozen code `D:\KareOne\RAG-Paper-upgrades` @ 77924d4, imported read-only (`git rev-parse HEAD` checked). No GPU,
  no Ollama, `CUDA_VISIBLE_DEVICES=""`.
- Reader: `z-ai/glm-5.3-flash` through the frozen `OpenRouterReader` and `reader_pins.json` (Together pin,
  allow_fallbacks false, require_parameters true, same payload rule: max_tokens = num_predict + 3000, reasoning effort
  low). The only change is the money guard: the shared `work/openrouter/reader_budget.json` was replaced in-process
  by a local guard (`code/client.py`) that checks this stream's ledger against the $1.20 cap and reads
  `limit_remaining` before every 50 calls (`key_checks.jsonl`; lowest reading $5.09, floor $2.00 never reached).
- Judge: the pinned judge (`openai/gpt-oss-120b`, `deepinfra/bf16`, temperature 0, seed 42, no fallbacks), via
  `ragbench.tools.judge_multi.judge_records` with the test settings (3 judge attempts, on_malformed adjudicate,
  3 HTTP attempts), test judge question file `test_v2judge_all.jsonl`. Verdicts in `lc_judge/test/`.
- Frozen-arm scores: `report_test.load/score` (machine verdicts, first valid verdict final, needs_adjudication =
  bounds [0, 1]). Paired statistics: the `revision_stats.py` functions (sign-p range over all unresolved
  assignments, 10,000 bootstrap resamples, seed 0, over questions and over the 20 source-family components), copied
  into `code/stats_common.py` because that module writes its own outputs when imported.
- Tokens of frozen cells: `report_test.token_cell` (the frozen per-call bound rule), cached in
  `frozen_cell_tokens.json`. Totals match TEST_RESULTS.md (HYBRID_6000 2,216 tokens/q, etc.).

### Experiment 1a: oracle routing (no API)
T1 -> HYBRID_6000; T2/T3 -> one of HR_15500, LEX5B_15500, SETR_15500, HYBRID_15500. Tier = annotation tier; the 6
unanswerable questions are routed by their intended tier (5 x T1, 1 x T3). Accuracy and tokens are read from the
saved cells of whichever arm the question is routed to. Tokens are over all 96 questions.

### Experiment 1b: real router
One GLM call per question that sees only the question text (system prompt + `Question: <text>`), reply A (one fact,
-> T1) or B (combine evidence, -> T2/T3); 600-token cap including reasoning (the frozen LEX5B selector cap), reasoning
effort low, 7 HTTP attempts. Unparsable reply -> T2/T3 (none occurred).
- Development on the 60 dev questions, three prompts (`code/router.py` PROMPTS v1, v2, v3). Selection rule fixed
  before seeing v3: highest dev accuracy among prompts with at most 3 T2/T3 questions sent to the cheap path
  (sending T1 to the wide path costs only tokens; sending T2/T3 to HYBRID_6000 costs answers). Only v1 met it.
- v1 run once on the 96 test questions (`router_test_v1.jsonl`). Routed-system accuracy is the saved verdict of the
  arm chosen; its tokens are the arm's tokens plus the router call's tokens; latency adds the router wall time.
- Also reported, at no cost: a format rule that routes `answer_format == v2_single` to T1. The answer format is
  already given to every pipeline (it selects the answer prompt), so this router needs no model call.

### Experiment 2: long context
- **Context length**: OpenRouter's live endpoint listing (`glm_endpoints_live.json`, fetched 2026-11-10) gives the
  pinned Together endpoint 1,048,575 context tokens and 943,717 max completion tokens, at $0.15/M prompt and $0.50/M
  completion tokens.
- **Pool sizes** (`pool_stats.json`, frozen pools loaded through `ragbench.pools` + `corpus.registry`, `pool.verify` OK).
  Estimated tokens use the median prompt tokens per character of the test's GLM answer calls (fa 0.396, en 0.219).

  | Pool | Docs | Characters | Est. reader tokens | Largest doc (chars) | Fits 1,048,575? |
  |---|---|---|---|---|---|
  | RPT-fa | 97 | 4,760,118 | 1,882,833 | 376,360 | no |
  | RPT-en | 15 | 16,638,928 | 3,644,478 | 1,994,276 | no |
  | BOOK-fa | 9 | 5,695,311 | 2,252,742 | 1,546,028 | no |
  | BOOK-en | 11 | 6,263,617 | 1,371,940 | 1,257,331 | no |
  | LAW-fa | 13 | 1,183,142 | 467,984 (measured 464,389) | 224,893 | yes |
  | LAW-en | 10 | 3,523,474 | 771,758 | 852,926 | yes (estimate) |

  Persian/English tokens per character: 1.81 on the test's GLM answer prompts (median ratio; these prompts include an
  English system prompt), 1.97 (LC-GOLD-DOCS) and 1.94 (LC-BM25-DOCS) on long prompts that are almost all document
  text (`lc_results.json`, `prompt_tokens_per_char_median`). The paper's "about 1.66" is lower than all three
  measurements here; the figure depends on how it is computed (median of per-call ratios here).
- **Limit**: L_TOK = 32,000 prompt tokens per call for both LC arms (evidence = (32,000 - 700) / tokens-per-char:
  80,900 chars fa, 142,900 chars en; about 5-24 times the 15,500-character budget). Why: a call at L costs
  ~$0.0048 here, so the two arms cost about $0.86 together, which fits the $1.20 cap. No gold-document set at a
  100k-token limit would have fit (estimated $1.16 for LC-GOLD-DOCS alone; full gold documents with no limit: $2.92).
  The median gold-document set is about 151k tokens; only 7 of the 90 fit whole in 32k.
- **LC-GOLD-DOCS (ORACLE upper reference)**: the documents that contain the question's gold units (annotation
  `oracle_docs`). Each document gets an equal share of the limit (water-filling). A document longer than its share is
  cut to one contiguous window (snapped to line breaks) that contains all its gold units, or one window per unit if
  they lie too far apart, so 228/228 gold units are always shown. Cuts are marked `[...]`. 90 answerable questions
  (an unanswerable question has no gold document).
- **LC-BM25-DOCS**: pool documents ranked by document-level BM25 of the question (ragbench B2 tokenizer and bm25s
  k1 1.5, b 0.75, lucene; one BM25 document = title + full text; IDF over the pool's documents; CPU). Documents are
  added whole while they fit. The first that does not fit is cut to the remaining space (its head), then stop. 96
  questions.
- **LC-POOL**: the whole pool, in manifest order. Only LAW-fa and LAW-en fit the context, and a full run costs about
  $0.070 per LAW-fa call and about $0.116 per LAW-en call (about $3 for 32 questions, over the cap). So only the
  3-question cost estimate was run: LAW-fa T1-001, T2-001, T3-001 (the first question of each tier, chosen by index;
  1 HTTP attempt, so a timeout would not be billed twice).
- Prompt (all LC arms): the test's answer contract unchanged. System prompt by answer format, user message
  `Evidence:\n...\n\nQuestion: ...`, num_predict by format. For every test question the system prompt, num_predict and
  question text were asserted equal to the frozen HYBRID15500 pack. Evidence is rendered as `[k] <document title>\n<text>`.
- Smoke / cost estimate: 3 dev questions x 2 arms (RPT-fa-T2-001, BOOK-en-T1-001, LAW-en-T3-003), judged with the dev
  judge file. The estimated prompt tokens matched the measured ones within about 5%, at $0.0010-0.0050 per call.

## 2. Results

### 2.1 Routing (`routing_tables.md`; 90 answerable for accuracy, 96 for tokens)

| Policy | All /90 | T1 | T2 | T3 | fa /45 | en /45 | Tokens/q | Calls/q | Share sent to HYBRID_6000 |
|---|---|---|---|---|---|---|---|---|---|
| always HYBRID_6000 | 62..63 | 28 | 20..21 | 14 | 35..36 | 27 | 2,216 | 1.0 | 1.00 |
| always HR_6000 | 69..71 | 28 | 23..24 | 18..19 | 36 | 33..35 | 2,237 | 1.0 | 0 |
| always HR_15500 | 69..70 | 28 | 21..22 | 20 | 38 | 31..32 | 5,287 | 1.0 | 0 |
| always LEX5B_15500 | 67 | 28 | 20 | 19 | 36 | 31 | 7,542-8,155 | 2.0 | 0 |
| always SETR_15500 | 66..67 | 28 | 22..23 | 16 | 35 | 31..32 | 10,882 | 2.0 | 0 |
| always HYBRID_15500 | 68 | 28 | 22 | 18 | 35 | 33 | 5,236 | 1.0 | 0 |
| oracle -> HR_15500 | 69..70 | 28 | 21..22 | 20 | 38 | 31..32 | 4,167 | 1.0 | 0.365 |
| oracle -> LEX5B_15500 | 67 | 28 | 20 | 19 | 36 | 31 | 5,531-6,021 | 1.635 | 0.365 |
| oracle -> SETR_15500 | 66..67 | 28 | 22..23 | 16 | 35 | 31..32 | 7,753 | 1.635 | 0.365 |
| oracle -> HYBRID_15500 | 68 | 28 | 22 | 18 | 36 | 32 | 4,143 | 1.0 | 0.365 |
| GLM router v1 -> HR_15500 | 68..69 | 28 | 21..22 | 19 | 38 | 30..31 | 5,065 | 2.0 | 0.156 |
| GLM router v1 -> LEX5B_15500 | 66 | 28 | 19 | 19 | 35 | 31 | 6,977-7,508 | 2.844 | 0.156 |
| GLM router v1 -> SETR_15500 | 66..67 | 28 | 22..23 | 16 | 35 | 31..32 | 9,896 | 2.844 | 0.156 |
| GLM router v1 -> HYBRID_15500 | 66 | 27 | 21 | 18 | 34 | 32 | 5,011 | 2.0 | 0.156 |
| format rule -> HR_15500 | 67..68 | 28 | 20..21 | 19 | 37 | 30..31 | 4,153 | 1.0 | 0.365 |
| format rule -> HYBRID_15500 | 65 | 27 | 20 | 18 | 34 | 31 | 4,128 | 1.0 | 0.365 |

Abstention: 6/6 for every policy. Key paired results (all in `routing_tables.md`):
- oracle -> HR_15500 vs always HR_15500: net -1..+1 (1/1), sign p 1.000, CI [-4.4, 5.6] pp, with 21% fewer
  tokens (4,167 vs 5,287 per question).
- oracle -> HR_15500 vs always HR_6000: net -2..+1 (6/5), p 0.790-1.000, CI [-10.0, 8.9] pp. The oracle router uses
  1.86 times the tokens of HR_6000.
- oracle -> HYBRID_15500 vs always HYBRID_15500: net 0 (1/1). Tokens 4,143 vs 5,236 (-21%).
- GLM router v1 -> HR_15500 vs always HR_15500: net -2..0 (0/1), p 0.500-1.000, tokens 5,065 vs 5,287 (-4%).
- Router v1 itself (test, run once): 66/90 correct routes on answerable questions. Of the 30 T1 questions, 8 were
  sent to T1 and 22 to T2/T3; of the 60 T2/T3 questions, 58 were sent to T2/T3 and 2 to T1. Router cost: 18,520
  tokens in total (193 per question), $0.0023, 1 extra call per question. Dev accuracies (54 answerable): v1 40/54
  (T2/T3 sent to T1: 2), v2 40/54 (9), v3 44/54 (6). The zero-cost format rule scored 45/54 on dev (5) and 76/90 on
  test (30 T1 -> 23 T1 + 7 T2/T3 by format; T2/T3 -> T1 errors 7 of 60).

### 2.2 Long context (`lc_tables.md`, `lc_results.json`; L_TOK = 32,000)

| Arm | All /90 | T1 /30 | T2 /30 | T3 /30 | fa /45 | en /45 | Abstain /6 | Query tokens/q (mean) | Calls/q |
|---|---|---|---|---|---|---|---|---|---|
| LC-GOLD-DOCS (ORACLE) | 79 | 27 | 24 | 28 | 37 | 42 | n/a | 29,569 | 1 |
| LC-BM25-DOCS | 24 | 13 | 9 | 2 | 11 | 13 | 6 | 30,967 | 1 |
| HYBRID_6000 (frozen) | 62..63 | 28 | 20..21 | 14 | 35..36 | 27 | 6 | 2,216 | 1 |
| HR_6000 (frozen) | 69..71 | 28 | 23..24 | 18..19 | 36 | 33..35 | 6 | 2,237 | 1 |
| HR_15500 (frozen) | 69..70 | 28 | 21..22 | 20 | 38 | 31..32 | 6 | 5,287 | 1 |
| HYBRID_15500 (frozen) | 68 | 28 | 22 | 18 | 35 | 33 | 6 | 5,236 | 1 |

All LC cells were resolved: 0 needs_adjudication, 0 system failures, every call ended with `stop`.
- **LC-GOLD-DOCS**: mean prompt 29,413 tokens (fa 30,638, en 28,188; max 41,706). Completion tokens 156 per question.
  Reader latency median 3.1 s, p90 4.7 s, max 6.3 s. Reader cost $0.404 for 90 calls. vs HR_15500: net +9..+10
  (15/6), sign p 0.052-0.078, CI [0.0, 21.1] pp over questions and [1.4, 19.5] over components. vs HR_6000: +8..+10
  (17/8), p 0.075-0.169. vs HYBRID_6000: +16..+17 (23/7), p 0.003-0.005. vs HYBRID_15500: +11 (17/6), p 0.035. Most
  of the gain is in T3 (28/30 vs 14-20 for the frozen arms) and in English (42/45 vs 27-35). T1 is 27/30, one or two
  below the frozen arms. Only 7 of the 90 gold-document sets fit whole (4/7 correct on those). The rest are oracle
  windows around the gold units.
- **LC-BM25-DOCS**: 24/90. The reader replied NOT_FOUND on 50 of the 90 answerable questions; 6/6 correct abstentions.
  The top BM25 documents included a gold document for 82/90 questions, but the head-truncated text kept only 62/228
  gold units. All gold units were in the prompt for 21 questions, and 18 of those were correct; 6 of the other 69 were.
  Mean prompt 30,899 tokens. Reader latency median 2.9 s. Cost $0.421 for 96 calls. It lost to every frozen arm
  (for example vs HYBRID_6000 net -39..-38, 1/39, p < 0.001).
- **LC-POOL** (LAW-fa, 3 questions; descriptive only): 464,389-464,455 prompt tokens per call (the measured whole
  LAW-fa pool), 25.7 / 25.3 / 22.6 s reader latency, $0.0697-0.0699 per call. All 3 strict-correct, as were HR_6000
  and HYBRID_15500 on the same questions (LC-BM25-DOCS 0/3). Extrapolated: 16 LAW-fa questions about $1.12, 16
  LAW-en about $1.85. One LC-POOL call is about 210 times the tokens of a HYBRID_6000 question.

## 3. Cost

Spent by S4: **$1.084** of $1.20 (`ledger.jsonl`, sum of `usage.cost`), all on the pinned endpoints.

| Item | Calls | USD |
|---|---|---|
| LC-GOLD-DOCS reader (3 dev + 90 test) | 93 | 0.414 |
| LC-BM25-DOCS reader (3 dev + 96 test) | 99 | 0.435 |
| LC-POOL reader (3 test) | 3 | 0.209 |
| Router v1 (60 dev + 96 test), v2 (60 dev), v3 (60 dev) | 276 | 0.008 |
| Judge (dev smoke + test LC arms; rule verdicts make no call) | 129 | 0.017 |

## 4. Failed, skipped or limited

- **LC-POOL was run on 3 questions only.** Only LAW-fa and LAW-en fit the 1M context, and the full 32-question run
  (about $3) is over the cap. The other four pools are 1.4-3.6M estimated tokens and do not fit at all.
- **No full-document gold arm.** LC-GOLD-DOCS is limited to 32k tokens with oracle windows, because whole gold
  documents (median ~151k tokens) would cost about $2.9. So LC-GOLD-DOCS measures "the reader with ~30k tokens of the
  right documents, positioned around the gold evidence". It is not "whole documents", and it uses oracle knowledge
  twice: which documents to read, and where the gold units are.
- LC-BM25-DOCS cuts the document head, not a passage near the evidence. This is the plain "add whole top documents
  until the limit" design asked for. With documents of 0.1-2M characters it mostly measures how often the evidence
  lies in the first 80-143k characters of the top document. A smarter cut would be retrieval again.
- Router: the chosen v1 prompt is conservative and sends only 8 of 30 T1 questions to HYBRID_6000, so it saves little
  (5,065 vs 5,287 tokens/q with HR_15500). Prompt v3 had a higher dev accuracy but sent more T2/T3 questions to the cheap
  path. It was not run on test (one test run only).
- Latency is descriptive only. LC latency is the reader call (plus <1 s of CPU pack building), not end-to-end under
  the frozen harness. The frozen latencies include retrieval and the warm-up conditions of 2026-10.
- Single run of a reader that is not deterministic (the Together pin ignores the seed). There is no run-to-run variance
  estimate.

## 5. Facts for the paper

1. Routing by the gold tier does not beat the best fixed pipeline. Sending T1 to HYBRID_6000 and T2/T3 to HR_15500
   scores 69..70/90, the same as always running HR_15500 (net -1..+1), with 21% fewer tokens (4,167 vs 5,287 per
   question). Always running HR_6000 scores 69..71 at 2,237 tokens. A real one-call GLM router routed 66/90 answerable
   test questions correctly and cut tokens by only 4%. (`routing_tables.md`, `routing_results.json`)
2. The answer format the pipelines already receive is a free tier proxy: "v2_single -> T1" routes 76/90 test
   questions correctly, against 66/90 for the GLM router. (`routing_results.json`, router_test.format_rule_accuracy_answerable)
3. Only the two LAW pools fit GLM-5.3-flash's 1,048,575-token context (LAW-fa measured at 464k prompt tokens). The
   other pools need 1.4-3.6M estimated tokens. A whole-pool LAW-fa call costs $0.070 and takes 23-26 s, about 210
   times the tokens of a HYBRID_6000 question. (`pool_stats.json`, `glm_endpoints_live.json`,
   `lc_runs/test/LC-POOL/runs_LC-POOL.jsonl`)
4. Oracle long context helps. About 30k tokens of the gold documents, windowed around the gold evidence, gave 79/90
   (T3 28/30), against 69..70 for HR_15500 (net +9..+10, sign p 0.052-0.078). The cost is about 5.6 times the tokens.
   It is an upper reference that needs oracle document and position knowledge. (`lc_tables.md`)
5. Without that oracle, whole-document reading fails. The top BM25 documents, cut to 32k tokens, scored 24/90. The
   reader abstained on 50 answerable questions because the cut text kept only 62/228 gold units, although a gold
   document was in the prompt for 82/90 questions. At these document lengths (0.1-2M characters), retrieval inside
   the document is still needed. (`lc_results.json`)
6. Persian costs 1.8-2.0 times more GLM reader tokens per character than English in these measurements (1.81 on the
   test answer prompts, 1.97 / 1.94 on document-only long prompts), more than the paper's 1.66.
   (`pool_stats.json`, `lc_results.json`)
