# S3 agentic — post-hoc agent arms on the frozen test (November 2026)

Post-hoc and exploratory (COMMON.md). The test split had already been unsealed. Nothing was tuned on test questions: each arm ran once on the 96 test questions, with prompts fixed on dev beforehand.

## 1. What was run

Code: `agents.py` (the arms), `judge.py` (the frozen judge), `analyze.py` (tables and statistics), `plot_svg.py` (the figure), `merge_judge_ledger.py` (copies judge calls into `ledger.jsonl`), `keycheck.py`. The frozen `ragbench` @ 77924d4 was imported read-only. No GPU, no Ollama, no dense vectors and no reranker were used (`CUDA_VISIBLE_DEVICES=""`).

- **Model.** `z-ai/glm-5.3-flash` is both the agent and the answerer. It is pinned exactly as `reader_pins.json`: provider order `[together]`, `allow_fallbacks false`, `require_parameters true`. Sampling parameters are filtered by the pinned endpoint's parameter list (temperature 0, top_p 1, no seed), reasoning effort is low (excluded), and up to 7 HTTP attempts are allowed. This is the same rule as the frozen `OpenRouterReader._payload`. Agent turns use native tool calling with a cap of 3,000 tokens per turn, reasoning included.
- **Final answer.** The frozen shared answer call: `build_answer_messages` + `answer_num_predict`. That is the test's answer prompt and answer-format routing, with `max_tokens = num_predict + 3000` as in the test. The answerer sees only the evidence the agent handed over, rendered by the frozen `pack_evidence` at **15,500 characters**, the wide test budget. The full answer messages are saved in every record (`answer_messages`).
- **Loop limits (both arms).** At most **5 tool rounds**. One agent turn is one round, with at most 3 tool calls per round; extra calls get an error. After round 5, one forced `finish` turn follows (`tool_choice = finish`). Tool results older than the last 2 rounds are cut to 400 characters to keep the context small.
- **GREP_AGENT** — a ReAct loop over the raw `Document.text` of the question's pool only (same pool mask as the test). Documents are named by short handles D1..Dn, as in B5, so English slugs do not leak into Persian runs.
  - `grep(pattern, doc_id?)`: a case-insensitive Python `regex` (5 s timeout), run line by line. It returns at most 20 lines (one hit per line) as `D<k> @<offset>: …200-char snippet…`, plus per-document match counts, capped at 3,500 characters.
  - `read(doc_id, start, length)`: at most 2,500 characters.
  - `list_docs()`.
  - `finish(evidence=[{doc_id,start,length}])`: each span at most 4,000 characters; overlapping spans are merged.
  - Persian handling: text and pattern get the same offset-preserving fold. Arabic ي/ى/ئ→ی, ك→ک, alef variants→ا, ة/ۀ→ه, ؤ→و, and Persian/Arabic digits→ASCII. ZWNJ and other joiners, tatweel and harakat are dropped, and text is lower-cased. The pattern's non-ASCII runs first go through `ragbench.normalize.normalize` (NFKC), as `grep_agent.normalize_pattern` does; regex syntax (ASCII) is untouched. Offsets map back to the raw text.
  - Patterns that match empty text or a single letter are refused.
  - The document list (= `list_docs()`) is put in the first user message. This saves a round, as B5 did. It is a deviation from offering it only as a tool.
  - `ragbench/approaches/grep_agent.py` was **not** reused. It searches the old section store with ripgrep through a JSON-in-text protocol written for a small local model, and has a different tool set. This is a clean implementation that reuses only the normalisation rules.
- **ITER_BM25** — an iterative agent with `search(query)`. Each call returns the top 8 of the frozen 1,000-character chunks of the pool: the frozen `_hybrid` index, **BM25 channel only** (`HybridIndex.search(channels=("bm25",))`, CPU, the same chunks, tokenizer and BM25 parameters as HYBRID's BM25 channel). The tool is followed by `finish(chunk_ids)`. A chunk shown earlier is listed by header only. **This is the no-GPU stand-in for an iterative agent over the hybrid index (BM25 + bge-m3 with RRF). It is not that agent: no dense channel and no reranker.**
- **Fallbacks.** If `finish` gives no usable evidence, the arm falls back to the read spans or seen chunks. This never happened on test: every evidence set came from `finish` (96/96 in both arms).
- **Development (dev only).** The dev questions came from `work/datasets_v2/devrun/questions` (the dev adapter of `*_dev.jsonl`). Each arm used 5 distinct dev questions; 6 distinct in all, under the limit of 10. GREP_AGENT used RPT-fa-T1-001, RPT-en-T2-001, BOOK-fa-T3-001, LAW-en-T1-001 and RPT-fa-T3-001. ITER_BM25 used RPT-en-T2-001, BOOK-fa-T3-001, LAW-en-T1-001, RPT-fa-T3-001 and LAW-fa-T2-001.
  - The cost estimate came from 3 dev questions (`runs/*/dev_est3`): about $0.001 per question per arm, about $0.2 for 2 × 96.
  - One prompt change came after the estimate: "make spans generous / include every passage that may be needed (up to about 15); the budget is large". It was checked on 3 dev questions per arm (`runs/*/dev_final2`).
  - Dev machine verdicts: est3 GREP 3/3, BM25 2/3; final2 GREP 2/3, BM25 3/3 (`judge/*/dev_*`).
  - Note: in `runs/*/dev_final` the prompt edit had not yet taken effect, so those runs used the pre-change prompt.
- **Test.** Each arm ran once on all 96 questions (`runs/<arm>/test/runs_<arm>.jsonl`, logs `test_<arm>.log`). There were 0 errors.
- **Judging.** The frozen pinned judge: `ragbench.tools.judge_multi`, `openai/gpt-oss-120b`, `--judge-provider deepinfra/bf16 --require-pin --max-judge-attempts 3 --on-malformed adjudicate --http-attempts 3`, with `test_v2judge_all.jsonl`, in batches of at most 45 calls (`judge/<arm>/test/`). All 96 + 96 cells were judged; 0 are `needs_adjudication`.
- **Frozen arms.** Machine verdicts, records and token bounds come from `work/datasets_v2/test_run/report_test.py`, imported (`load`, `score`, `token_cell`, `sign_p`). The paired procedure re-implements `paper/analysis/revision_stats.py`: diff bounds and an exact sign-test range over every assignment of unresolved cells, and a bootstrap over questions and over the 20 connected components of source families (10,000 resamples, seed 0). The module was not imported, because importing it rewrites its own output files.
- **Tokens.** Query-time LLM tokens = prompt + completion (reasoning included) over every agent turn and the answer call. Every call returned usage. Cached prompt tokens count, as in the frozen resource rule; most agent prompt tokens were cache hits at Together, so billed USD is far below what the token count suggests. The frozen arms use `report_test.token_cell`, giving bounds where usage was unknown.

## 2. Results (90 answerable test questions; GLM-5.3-flash; machine strict verdicts; a..b = bounds over unresolved frozen cells)

Sources: `results.json`, `results_tables.md` (written by `analyze.py`).

## Accuracy (machine, strict, GLM-5.3-flash reader; bounds over unresolved judge cells)

| arm | all /90 | T1 /30 | T2 /30 | T3 /30 | fa /45 | en /45 | abstention /6 |
|---|---|---|---|---|---|---|---|
| GREP_AGENT | 68 | 28 | 19 | 21 | 29 | 39 | 5 |
| ITER_BM25 | 73 | 27 | 23 | 23 | 34 | 39 | 6 |
| HYBRID_6000 | 62..63 | 28 | 20..21 | 14 | 35..36 | 27 | 6 |
| HR_6000 | 69..71 | 28 | 23..24 | 18..19 | 36 | 33..35 | 6 |
| LEX5B_6000 | 68..69 | 29 | 23..24 | 16 | 36..37 | 32 | 6 |
| SETR_6000 | 63 | 27 | 18 | 18 | 33 | 30 | 6 |
| HYBRID_15500 | 68 | 28 | 22 | 18 | 35 | 33 | 6 |
| HR_15500 | 69..70 | 28 | 21..22 | 20 | 38 | 31..32 | 6 |
| LEX5B_15500 | 67 | 28 | 20 | 19 | 36 | 31 | 6 |
| SETR_15500 | 66..67 | 28 | 22..23 | 16 | 35 | 31..32 | 6 |

## Cost (all 96 questions)

| arm | query LLM tokens / q | x HYBRID_6000 | LLM calls / q | median latency s |
|---|---|---|---|---|
| GREP_AGENT | 12338 | 5.57 | 5.10 | 12.6 |
| ITER_BM25 | 9285 | 4.19 | 3.64 | 8.8 |
| HYBRID_6000 | 2216 | 1.00 | 1.00 | 1.8 |
| HR_6000 | 2237 | 1.01 | 1.00 | 3.2 |
| LEX5B_6000 | 4511..5131 | 2.04..2.32 | 2.00 | 4.0 |
| SETR_6000 | 7503..8128 | 3.39..3.67 | 2.00 | 5.5 |
| HYBRID_15500 | 5236 | 2.36 | 1.00 | 2.6 |
| HR_15500 | 5287 | 2.39 | 1.00 | 3.6 |
| LEX5B_15500 | 7542..8155 | 3.40..3.68 | 2.00 | 4.0 |
| SETR_15500 | 10882 | 4.91 | 2.00 | 5.8 |

## Agent behaviour (test)

- **GREP_AGENT**: tool calls/q {'grep': 2.47, 'read': 1.51, 'finish': 1.0}; rounds mean 3.84, histogram {'2': 5, '3': 39, '4': 18, '5': 34}; evidence source {'finish': 96}; evidence chars median 1652.0 mean 2144; agent calls/q 4.10; system failures 0; truncated answers 0; cost $0.1173
- **ITER_BM25**: tool calls/q {'search': 2.41, 'finish': 1.0}; rounds mean 2.54, histogram {'2': 68, '3': 15, '4': 2, '5': 11}; evidence source {'finish': 96}; evidence chars median 2735.0 mean 3014; agent calls/q 2.64; system failures 0; truncated answers 0; cost $0.1151

## Paired comparisons on the 90 answerable questions (new arm minus frozen arm)

| comparison | net | wins/losses | sign p | 95% CI pp (questions) | 95% CI pp (components) | T1 net | T2 net | T3 net | fa net | en net |
|---|---|---|---|---|---|---|---|---|---|---|
| GREP_AGENT vs HYBRID_6000 | +5..+6 | 16/10; 16/11 | 0.327-0.442 | [-5.6, 17.8] | [-8.8, 19.8] | +0 | -2..-1 | +7 | -7..-6 | +12 |
| GREP_AGENT vs HR_6000 | -3..-1 | 12/13; 12/14; 12/15 | 0.701-1.000 | [-14.4, 10.0] | [-16.7, 10.4] | +0 | -5..-4 | +2..+3 | -7 | +4..+6 |
| GREP_AGENT vs LEX5B_6000 | -1..+0 | 12/12; 12/13 | 1.000 | [-12.2, 11.1] | [-16.7, 14.3] | -1 | -5..-4 | +5 | -8..-7 | +7 |
| GREP_AGENT vs SETR_6000 | +5 | 13/8 | 0.383 | [-4.4, 15.6] | [-6.5, 19.4] | +1 | +1 | +3 | -4 | +9 |
| GREP_AGENT vs HYBRID_15500 | +0 | 11/11 | 1.000 | [-10.0, 10.0] | [-9.6, 10.1] | +0 | -3 | +3 | -6 | +6 |
| GREP_AGENT vs HR_15500 | -2..-1 | 12/13; 12/14 | 0.845-1.000 | [-13.3, 10.0] | [-16.7, 13.4] | +0 | -3..-2 | +1 | -9 | +7..+8 |
| GREP_AGENT vs LEX5B_15500 | +1 | 14/13 | 1.000 | [-10.0, 12.2] | [-13.0, 15.5] | +0 | -1 | +2 | -7 | +8 |
| GREP_AGENT vs SETR_15500 | +1..+2 | 12/11; 13/11 | 0.839-1.000 | [-10.0, 13.3] | [-10.2, 15.9] | +0 | -4..-3 | +5 | -6 | +7..+8 |
| GREP_AGENT vs ITER_BM25 | -5 | 4/9 | 0.267 | [-13.3, 2.2] | [-12.5, 0.0] | +1 | -4 | -2 | -5 | +0 |
| ITER_BM25 vs HYBRID_6000 | +10..+11 | 16/6; 17/6 | 0.035-0.052 | [1.1, 22.2] | [0.0, 25.0] | -1 | +2..+3 | +9 | -2..-1 | +12 |
| ITER_BM25 vs HR_6000 | +2..+4 | 11/8; 11/9; 12/8; 12/9 | 0.503-0.824 | [-7.8, 14.4] | [-9.3, 15.2] | -1 | -1..+0 | +4..+5 | -2 | +4..+6 |
| ITER_BM25 vs LEX5B_6000 | +4..+5 | 14/10; 15/10 | 0.424-0.541 | [-6.7, 16.7] | [-10.0, 19.6] | -2 | -1..+0 | +7 | -3..-2 | +7 |
| ITER_BM25 vs SETR_6000 | +10 | 15/5 | 0.041 | [1.1, 21.1] | [0.0, 25.9] | +0 | +5 | +5 | +1 | +9 |
| ITER_BM25 vs HYBRID_15500 | +5 | 13/8 | 0.383 | [-4.4, 15.6] | [-3.3, 15.8] | -1 | +1 | +5 | -1 | +6 |
| ITER_BM25 vs HR_15500 | +3..+4 | 12/8; 12/9 | 0.503-0.664 | [-6.7, 14.4] | [-9.5, 18.6] | -1 | +1..+2 | +3 | -4 | +7..+8 |
| ITER_BM25 vs LEX5B_15500 | +6 | 13/7 | 0.263 | [-3.3, 16.7] | [-6.1, 19.8] | -1 | +3 | +4 | -2 | +8 |
| ITER_BM25 vs SETR_15500 | +6..+7 | 12/6; 13/6 | 0.167-0.238 | [-2.2, 17.8] | [-3.4, 22.1] | -1 | +0..+1 | +7 | -1 | +7..+8 |

Notes on the tables:
- The sign p values are unadjusted. There are 17 post-hoc comparisons, so none survives a multiplicity correction. ITER_BM25 vs HYBRID_6000 (p 0.035–0.052) and vs SETR_6000 (p 0.041) are the only ones near 0.05, and both are exploratory.
- Abstention: GREP_AGENT answered one unanswerable question (V2-test-LAW-fa-AB-001) instead of NOT_FOUND. ITER_BM25 abstained correctly on 6/6.
- Evidence actually handed over: medians of 1,652 characters (GREP_AGENT) and 2,735 (ITER_BM25), means of 2,144 and 3,014, maxima of 8,050 and 8,323. **Neither agent came close to the 15,500 budget.** The frozen 15,500 arms fill it. Against the 15,500 arms, the agents are therefore not compared at matched evidence: they choose less.
- Cost–accuracy plane: `cost_accuracy.svg`. It plots machine accuracy (% of 90) against mean query-time LLM tokens per question over all 96 questions, for the frozen cells (circles; blue = 6,000, grey = 15,500) and the agent arms (squares). matplotlib is not installed in the venv, so the SVG is drawn by `plot_svg.py` from `results.json`.
  - ITER_BM25 (9,285 tok/q, 73/90) lies above every frozen cell. The best frozen cell is HR_6000 at 69–71 with 2,237 tok/q. ITER_BM25 is +2..+4 over HR_6000, not significant, at 4.2× HYBRID_6000's tokens.
  - GREP_AGENT (12,338 tok/q, 68/90) is the most expensive point and no better than HYBRID_15500 (68/90 at 5,236 tok/q).
- Where the gains come from: English and T3. Both agents gain +12 over HYBRID_6000 on English questions and +7 (GREP) / +9 (BM25) on T3 cross-document questions.
  - On Persian, both are at or below the single-pass arms: GREP_AGENT 29/45 vs HYBRID_6000 35–36 and HR_6000 36; ITER_BM25 34/45. The grep agent loses −6..−7 on Persian against HYBRID_6000 despite the normalisation.
  - On T2 (within-document integration), GREP_AGENT is the weakest arm (19/30).
- Wall time (descriptive; agent arms in this session through OpenRouter, frozen arms in the test session): median 12.6 s (GREP) and 8.8 s (BM25) per question, against 1.8 s (HYBRID_6000) and 3.2 s (HR_6000). That is about 3–7× slower.

## 3. Cost spent

Source: `ledger.jsonl`, 1,090 calls; every call has `usage.cost`. Judge calls were copied in from `judge/*/*/judge_ledger_*.jsonl` by `merge_judge_ledger.py`.

| part | calls | USD |
|---|---|---|
| GREP_AGENT dev (agent + answer) | 33 | 0.0071 |
| ITER_BM25 dev | 26 | 0.0096 |
| GREP_AGENT test | 490 | 0.1173 |
| ITER_BM25 test | 349 | 0.1151 |
| judge (gpt-oss-120b, dev + test) | 192 | 0.0252 |
| **total** | **1,090** | **$0.274** of the $2.00 cap |

The key had $9.61 left at the start and $5.09 at the end; other streams were spending in parallel. The global floor ($2.00) was checked every 5 questions and before every judge batch, and was never reached. The billed cost per test question is about $0.0012 for each agent arm, because Together's prompt cache absorbed most of the repeated agent context. The judge calls went through the frozen `OpenRouterReader`, so they are also counted in the shared `work/openrouter/reader_budget.json`.

## 4. Failed or skipped

- Nothing failed: 0 system failures, 0 truncated answers, 0 unresolved judge cells for the new arms.
- **Not done:** an iterative agent over the actual hybrid index (BM25 + bge-m3 dense + RRF), or with the reranker. This **needs GPU** (query embedding / cross-encoder); ITER_BM25 is the CPU stand-in.
- **Not done:** human checks of the new arms' answers. Only machine verdicts exist, so these arms cannot be compared with the paper's human-merged numbers.
- **Not done:** run-to-run variance. One run per arm; the reader has no seed on Together.
- `grep_agent.py` (B5) was not reused (see section 1), and the `list_docs` output is pre-filled in the first message.
- The 15,500 evidence budget was a cap, not a target. The agents used a median of 11–18% of it, so "the same evidence budget" holds only as an upper bound.

## 5. Facts for the paper

1. Each arm ran once on the 96 test questions. ITER_BM25, an iterative GLM-5.3-flash agent with only a BM25 search tool over the frozen 1,000-character chunks (at most 5 rounds, answer by the shared test prompt), scored **73/90** machine-strict. That is +10..+11 vs HYBRID_6000 (p 0.035–0.052, unadjusted, post-hoc) and +2..+4 vs HR_6000 (p ≥ 0.50), at **9,285 query tokens per question, 4.19× HYBRID_6000** (`results.json`, `results_tables.md`).
2. GREP_AGENT, a ReAct agent with regex grep/read over the raw pool text and Persian-aware normalisation, scored **68/90**. That is −3..−1 vs HR_6000 and 0 vs HYBRID_15500, at **12,338 tokens per question (5.57× HYBRID_6000)**, 5.1 LLM calls per question and about 7× HYBRID_6000's median latency (`results.json`).
3. Both agents' gains are on English and cross-document questions: +12 over HYBRID_6000 on English and +7/+9 on T3. On Persian they are at or below the single-pass arms (GREP_AGENT 29/45 vs HYBRID_6000 35–36/45; ITER_BM25 34/45) (`results_tables.md`).
4. Neither agent used the evidence budget it was given. The evidence handed to the answerer had a median of 1,652 (grep) and 2,735 (BM25) of the 15,500 characters allowed, and every evidence set came from the agent's own `finish` call (`results.json`: `evidence_chars_median`, `evidence_source`).
5. With prompt caching, the agents' cost shows up in tokens rather than dollars: about $0.0012 billed per question per arm, against 4–6× the query tokens of HYBRID_6000. The whole stream cost $0.274 (`ledger.jsonl`).
6. Caveat for the wording: post-hoc, one run, machine verdicts only, 17 unadjusted comparisons. The BM25 agent is a CPU stand-in for an iterative agent over the hybrid index, not that agent (`RESULTS.md` sections 1 and 4).
