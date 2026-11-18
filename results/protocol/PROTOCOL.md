# Frozen protocol: the one-shot datasets-v2 test (frozen 2026-10-01, before any test answer)

Source of the fixes: Astra's pre-test review (`work/analysis/FREEZE_ASTRA_R1.log`, from the last `tokens used`) and
`work/TODO.md` "NEXT SESSION" item 1. This file, `MANIFEST.json` and the code commit below fix every choice
**before** the first test answer. Nothing in it may change after the test starts. A change before the start needs a
new manifest, a new dated section here, and a reason.

## 1. What is frozen

| Item | Value |
|---|---|
| Code | `D:\KareOne\RAG-Paper-upgrades`, branch `upgrades`, commit **`b31cd4ca9b8ce167003db6d475ebf846589af0c7`**. The `--test-protocol` gate refuses any other HEAD and any uncommitted change under `ragbench/`. |
| Manifest | `test_freeze/MANIFEST.json`, sha256 `6db0f50a0c73b6dec19f19f0bf4e39bc7e88c78b68a49c451643026ed520724b`. `ragbench.run --test-protocol` checks it on every cell. |
| Runner | `test_freeze/run_test.py`, sha256 `f955c87ec56fd08cdf5785b3b3de015d226b6d901feb4941ba130812e1d51ac5`. |
| Freeze tool | `test_freeze/freeze_test.py`, sha256 `4634588b487cb5729f1b9116037b223cb6854cd1325083e735f061aa78adab27`. |
| Source questions | `FROZEN_questions.json` (sha256 `341c181d…`). All six source files are re-verified equal to it. The three `*_test.jsonl` files hold 96 questions: 90 answerable and 6 unanswerable (`*-AB-001`). |
| Test adapter | `ragbench/tools/verify_v2run.convert_record`, experiment id `datasets-v2-test`. Output goes to `test_freeze/questions/`: `test_v2run_all.jsonl` (sha256 `14bc5d02…`), one file per pool with 16 questions each, and the evaluator-only `test_v2judge_all.jsonl` (`a65e275a…`). `verify` reports 0 errors and 0 flags. Methods receive a `Query`, never the gold fields. |
| Pools | The six frozen manifests in `work/multidoc/pools/<pool>.json`, hashed in `MANIFEST.json`: RPT-fa `5e5947dc…`, RPT-en `4dc13cc5…`, BOOK-fa `56820197…`, BOOK-en `419a751e…`, LAW-fa `4d0078ca…`, LAW-en `70fdd85d…`. `pool.verify` is OK. No test evidence document lies outside its pool. |
| Indexes | Every file is hashed (`inputs.json`, `index_manifests/`): `_hybrid` `82216cce…`, `lex8/idx` `f23f1c4e…`, `lex8/synth` `f7101a54…`, plus the arm pointer directories. The LEX5B and LEX5B_15500 pointer indexes were rebuilt by `prebuild_indexes.py`, because the parser version is part of their configuration signature. The rebuild made 0 model calls and produced an identical `hybrid_ref.json`. |
| Dev/test separation | Dev and test share 0 evidence documents, 0 source blocks, 0 information needs and 0 source families. |
| Duplicate audit | At the freeze, no `V2-test-` id appears in any `runs_*`, `ledger_*`, `verdicts_*`, `judge_failures_*` or `needs_adjudication_*` file under `work/`. `run_test.py run` repeats this audit before the first answer and refuses to start if it is not clean. |

## 2. Arms (all six chosen now; nothing is added after outcomes are seen)

| Arm | Variant | Command | Role |
|---|---|---|---|
| HYBRID_6000 | HYBRID | `--approach hybrid` | primary baseline |
| HR_6000 | HYBRID+rerank | `--approach hybrid --ablation rerank` | primary stronger baseline |
| LEX5B_6000 | LEX5B | `--approach lex5b` + `LEX8_IDX_DIR`/`LEX8_SYNTH_DIR` | frozen candidate |
| HYBRID_15500 | HYBRID15500 | `--approach-class ragbench.approaches.lex10.control:Hybrid15500` | budget effect, baseline |
| HR_15500 | HR15500+rerank | `--approach-class ragbench.approaches.budget_matrix:HR15500` | budget effect, stronger baseline |
| LEX5B_15500 | LEX5B_15500 | `--approach-class ragbench.approaches.budget_matrix:Lex5b_15500` + LEX8 env | LEX-by-budget interaction |

- **Reader:** `openrouter:z-ai/glm-5.3-flash`, pinned through `OPENROUTER_PROVIDER_PINS` to `together`. The pin uses the dev `lex5_pilot/providers.json` choice and its parameter list. Routing is `allow_fallbacks false, require_parameters true`.
- **Reader seed:** the saved Together parameter list has no `seed`, so the configured seed 42 does **not** make the reader deterministic.
- **Selector:** the same model and pin, 1 attempt, a 3 s total deadline with cancellation, a 600-token completion cap that includes reasoning, and reasoning effort low. `LEX5B_SELECTOR_PIN` is unset, and the gate refuses it.
- **Parser:** `lex5-bundles-v1.1`. Alternatives within a part must be different passage **sets**, so `1+2 | 2+1` is rejected. Any rejection falls back to the F2-intact pack.
- **Not run:** F2-intact@6,000 and F2@15,500 are not arms. Their absence is declared now: the test cannot separate the selection mechanism from the whole LEX5B pipeline.

## 3. Execution (`run_test.py run --confirm-one-shot`)

- **Order:** the 96 questions run in `test_v2run_all.jsonl` order. For each question, all six arms run back to back, with the arm order rotated by the question index (k = i mod 6). There is one `ragbench.run --ids <q> --test-protocol MANIFEST.json` process per cell, one at a time, at below-normal priority with 2 threads. A 2 s pause separates questions and falls outside every timed cell.
- **Before starting:** start Ollama and warm bge-m3 (untimed; HYBRID uses it for the query embedding).
- **One shot:** a cell that produced an answer is never run again.
  - **Infrastructure exception:** a cell whose record has `error` and no answer gets at most one rerun. Causes are a local service down, a process crash, or all reader HTTP attempts failing. The rerun is logged in `test_run/reruns.jsonl`. A second error counts as a system failure, scores 0, and is reported.
  - **Selector deadline or failure:** the fallback pack is a **completed** method execution and is never a reason to rerun.
- **Reader retry policy:** `READER_MAX_ATTEMPTS=7` (the dev default of 6 retries, now explicit). Every ledger record carries `attempts` and `max_attempts`.
  - Retried: HTTP 429 or 5xx, network errors, an error body, or no choices.
  - Never retried: any other 4xx, a deadline, or answer content of any kind.
  - Answers are never regenerated.
- **Budget:** about $1–1.5 is expected. The runner stops before a cell once the shared reader file reaches start + $2.00. Resuming loses nothing.
- **Latency:** the warm-query end-to-end `latency_ms` is reported as a measured trade-off, not a gate (user decision of 2026-09-26). Loading indexes and models and warming connections are excluded.

## 4. Judging (`run_test.py judge --confirm-one-shot`)

- **Judge:** `openai/gpt-oss-120b`, judge-multi v2 strict rubric, pinned to endpoint **`deepinfra/bf16`** (`order [deepinfra/bf16]`, `allow_fallbacks false`, `require_parameters true`).
  - Why this endpoint: it served the most dev60 judge calls (34) with no failures, runs at bf16, supports temperature, top_p and seed, and is the cheapest bf16 endpoint.
  - Probe: 2 calls on synthetic fictional-company text, both parsed with the expected verdict, served by DeepInfra, cost $0.000056 (`judge_pin.json`).
  - **This is an evaluation amendment:** the dev judge was unpinned (13 providers).
  - Every verdict records the pin and the served provider. The pin is also part of the verdict key and the cache key.
- **Malformed judge output** (`--max-judge-attempts 3 --on-malformed adjudicate --http-attempts 3`):
  - A malformed, unparsable or incomplete reply triggers up to 3 **identical** calls.
  - If none parses, the cell goes to `needs_adjudication`. That outcome is terminal: it is never scored 0 and the system answer is never regenerated.
  - Such cells go to **blinded** human adjudication: the question, the reference packet and the answer are shown, never the method. Machine and adjudicated scores are kept apart (`judge_multi.final_verdicts`).
  - Without adjudication a cell is reported as the bounds [0, 1].
  - Call errors with no reply at all are an infrastructure failure. They go to `judge_failures` and are retried only on request.
- **Verdict precedence:** the first valid verdict for a (record, answer hash, rubric, judge, pin) key is final, and a judged answer is never judged again. The dev merge rule "later wins" does not apply to the test.
- **Blinded human audit (prespecified):** every numerical or binding answer in a primary discordant pair (LEX5B_6000 vs HYBRID_6000 or HR_6000), with methods hidden. TODO item 2's review page shows about 30–40 answers. Machine and audited scores are reported separately; the primary analysis uses the machine scores, with adjudicated `needs_adjudication` cells filled in.

## 5. Decision rules: three separate conclusions

1. **Engineering accuracy gate.**
   - On the **fixed 90 answerable** questions, LEX5B_6000 must gain at least **+5** strictly correct answers over **each** 6,000 control (HYBRID_6000 and HR_6000).
   - Derivation: k = ceil(90 × 3/54) = 5, scaling the dev gate of +3/54.
   - The denominator never shrinks: a judge failure stays in the 90, and the 6 unanswerable questions are not in it.
   - With `needs_adjudication` cells, the gate is **passed** if the lower-bound gain is at least 5 against both controls, **failed** if the upper-bound gain is below 5 against either, and otherwise **unresolved** until blinded adjudication.
2. **Resource gate.**
   - Total query-LLM tokens of LEX5B_6000 over **all 96** test questions must be at most **2.5×** the tokens of the **contemporaneous HYBRID_6000** run. The HR ratio is reported as well. The historical 5,422 figure is a reference only.
   - Tokens = prompt + completion, with reasoning included in completion, summed over every query-phase chat call (selector and reader) and every call that returned usage. Cached input tokens count; billed USD is reported separately.
   - Calls with unknown usage are bounded per call by `ragbench/tools/token_bounds.py`: lower 0; upper ceil(prompt_chars × r) + the call's max_tokens. Here r is the maximum observed tokens-per-character of that model, purpose and language in the dev ledgers × 1.10 (`token_calibration.json`, sha256 `76d80eaf…`).
   - The gate is passed if the upper ratio is at most 2.5, failed if the lower ratio exceeds 2.5, and otherwise unresolved. It is never passed by counting unknown usage as 0.
3. **Statistical superiority**, reported **separately** from the gate (passing the gate does not establish superiority):
   - Paired accuracy differences on the 90 answerable questions with 95% bootstrap CIs (10,000 resamples over questions, seed 0).
   - Two-sided exact sign tests on discordant pairs for the two primary comparisons, with **Holm** correction across the two. Superiority is claimed only where the Holm-adjusted p is below 0.05.
   - Sensitivity analysis: a cluster bootstrap by source family.
   - Descriptive only: subgroups by language, domain and tier, the 6 abstention questions, the budget effects (HYBRID and HR from 6,000 to 15,500), and the LEX-by-budget interaction (LEX5B gain at 6,000 minus gain at 15,500 against each control). The two LEX budgets run independent selector calls, so that interaction includes selector and provider variability.
   - Also reported: the selector failure rate, the all-NONE rate, and the rate of unchanged packs (final pack = fallback).
   - One-shot execution cannot estimate run-to-run model variance; this is stated.

## 6. Recorded exceptions and corrections to the dev record (declared before the test)

1. **Smoke LAW-en-T1-003.** A LEX5B/GLM smoke answer (pinned to Together) existed before dev60, and the dev60 answer is textually identical. The "36 unseen questions" wording is corrected to: "36 questions outside the 24-question answer screen; their answerable questions took part in offline development". The test's duplicate check searches raw records, smoke directories included, not just the registry.
2. **HR RPT-fa-T2-002 has a contradictory verdict.** The same HR answer, with identical answer hash, rubric hash and record identity, was scored strict **1** in `lex5_pilot` and **0** in `idea_test`, and the dev merge took the later verdict. With the earlier verdict, HR scores 33/54 answerable and LEX5B's dev lead over HR is +4, not +5. This is direct evidence of judge instability. It is the reason for the judge pin and the first-verdict-final rule.
3. **Dev60 LEX5B@15,500 RPT-fa-T2-003.** The judge reply was malformed 5 times and the cell was counted 0. Under the new policy it is `needs_adjudication`, so LEX5B@15,500 is **42–43/60**, not 42.
4. **Parser v1.1 replay** (`parser_replay.json`): 134 saved selector replies were replayed (lex5b 20, budget_matrix 14, dev60 59, lex5_pilot 21; 31 calls returned no text) plus 20 probe replies, through the old and new parsers.
   - 153 of 154 parse identically.
   - **One difference:** the lex5b LEX5B@6000 cell for `V2-dev-RPT-fa-T3-001`, reply `P1: 2+3 | 3+2`. The old parser said ok; v1.1 says fail (duplicate_alternative), so that cell would receive the fallback pack: the same 6 chunks in F2 order.
   - Impact: the saved answer was correct and all 7 arms were correct on that question, so no discordance changes. The worst case is LEX5B@6000 at 36/54 instead of 37 (+6 vs HYBRID, +4 vs HR). The cell is not rerun.
5. **Dev token figure.** report60's "conservative" 4,974 tokens/q was an imputation and added one allowance per question. Under the frozen per-call bound, dev60 LEX5B@6000 (36 interleaved questions) is **4,451–5,106** tokens/q against HYBRID's 2,169, a ratio of 2.05–2.35.
6. **Newly answered dev group.** On answerable questions LEX5B gains only **+2/30 vs HYBRID and +1/30 vs HR** (discordances 5/3 and 4/3). The advertised +3 included an abstention gain.
7. **Judge pin** (section 4): this is an amendment, and the dev verdicts remain unpinned.

## 7. Claims and their limits (paper wording)

- The method was heavily developed on a small benchmark written by an agent. That benchmark had independent agent review, native Persian wording review, and an author spot-check of 32 of the 156 questions. The test is held out from method evaluation; it is not a random sample of a population.
- Character budgets are **rendered reader-evidence budgets**, not equal resources across tokenizers. LEX5B also gets a second model call and sees candidate excerpts beyond the final pack. HR gets cross-encoder compute. HYBRID uses Ollama dense embeddings, while F2 uses the HF implementation. The test therefore compares whole pipelines, not an isolated selector.
- Retention means source-aware lexical containment of the annotated decisive spans. It is not entailment, and not proof that the reasoning context is sufficient.
- The full development history, failed variants included (LEX2–LEX11), is reported, together with the token-gate amendment (2.5×), made before the LEX5B results, and the later acceptance of latency as a trade-off.

## 8. Order of operations for the test session

1. `python freeze_test.py audit` must be clean, and `git -C D:\KareOne\RAG-Paper-upgrades rev-parse HEAD` must equal the commit in section 1.
2. Start Ollama and warm bge-m3 (untimed).
3. `python run_test.py plan`: all 6 gates must report `TEST PROTOCOL: matches`.
4. `python run_test.py run --confirm-one-shot` (resumable). It writes `work/datasets_v2/test_run/`.
5. `python run_test.py judge --confirm-one-shot`.
6. Analysis by sections 5–6. Blinded adjudication and audit follow TODO item 2.

## 9. Amendment 1 (2026-10-01, before any test answer): re-freeze with SETR and six readers

**Why.** (a) SETR, the zero-shot SetR literature baseline (`setr/SETR.md`), was the strongest dev arm (40/54 vs
LEX5B 37). Leaving out a published baseline after seeing it beat the candidate would be selective reporting; the user
approved adding it. (b) The user asked for at least five reader models of varied families, so the reader is no longer
a single model. (c) The code moved: 1cdb056 (SETR) and **77924d4** (answer packs, reader replay, SETR@15500).
Nothing about LEX5B, HYBRID or HR changed, and no test question has been answered. The b31cd4c files are kept as
`MANIFEST_b31cd4c.json`, `inputs_b31cd4c.json`, `run_test_b31cd4c.py` and `freeze_test_b31cd4c.py`.

**What is frozen now (supersedes section 1 where they differ).**

| Item | Value |
|---|---|
| Code | `upgrades` commit **`77924d45c0f0964b5657379a6da19b4ceaf8443a`** (gate refuses any other HEAD or a dirty `ragbench/`) |
| Manifest | `MANIFEST.json` schema 2, sha256 `c688c84b2ce1b252960becbb635ed2d489603a658ae096897f520fc7d26a0c86` |
| Runner | `run_test.py` (plan / run / replay / judge / status) |
| Freeze tool | `freeze_test.py` sha256 `f0f6c6cafbc9d084e91077ab63867d719fbe711524a46ef3316e276db54784a7` |
| Reader pins | `reader_pins.json` sha256 `cb60af397d6781bc0b1f2c4655792abfe0e93f4ece244a1a99a1d761d3948f46` |
| Questions, adapter, pools | unchanged (same sha256 values as section 1; re-verified: 0 errors, 0 flags, dev/test overlap 0, raw-record audit clean) |
| Indexes | all re-hashed into `inputs.json`; new pointer dirs `SETR`, `SETR_15500` built with `prebuild_indexes_v2.py` (index-only over DEV files, 0 model calls) |

**Arms: exactly 8 arm-budget cells** (k = i mod 8 rotation; the "7 arms" wording earlier counted LEX5B@15500 once):

| Arm | Variant | Command | Role |
|---|---|---|---|
| HYBRID_6000, HR_6000, LEX5B_6000 | as section 2 | as section 2 | primary gate cells |
| HYBRID_15500, HR_15500, LEX5B_15500 | as section 2 | as section 2 | budget effect, LEX-by-budget |
| **SETR_6000** | SETR | `--approach setr` | literature baseline, **no gate** |
| **SETR_15500** | SETR_15500 | `--approach-class ragbench.approaches.budget_matrix:Setr15500` | literature baseline at the wide budget |

SETR settings, as declared in `setr/SETR.md`: zero-shot SetR prompt verbatim, HYBRID top-20 full chunks (1,500-char
safety cap), GLM selector on the same Together pin, 1 attempt, 8 s total deadline, 2,000-token cap incl. reasoning,
reasoning effort low; fallback = HYBRID order; pack = selected first, then the rest of the top 40, through HYBRID's
offer + chunk packer at the cell budget. The SETR call does not depend on the budget but runs separately in each cell
(like LEX5B's two budgets).

**Readers (6).** The same answer contract (prompts, num_predict, sampling parameters filtered by the pinned endpoint's
supported list, reasoning effort low where supported, the same max_tokens rule), each pinned with
`allow_fallbacks false, require_parameters true`, `READER_MAX_ATTEMPTS=7`:

| Reader | Pinned endpoint | Role |
|---|---|---|
| z-ai/glm-5.3-flash | together | **primary** (all gates) |
| openai/gpt-oss-20b | amazon-bedrock (dev pin) | generality |
| qwen/qwen3-235b-a22b-2507 | gmicloud/fp8 (deepinfra/fp8 was degraded, status -2) | generality |
| meta-llama/llama-3.3-70b-instruct | deepinfra/turbo (fp8) | generality |
| google/gemma-3-27b-it | deepinfra/fp8 | generality |
| deepseek/deepseek-v4-flash | deepinfra/fp8 | generality |

**Shared packs.** Retrieval, HR reranking and the LEX5B / SETR selector calls run **once per question x arm**, inside
the primary (GLM) cell. The exact answer-call input (system + user messages with the rendered evidence, num_predict)
is saved as the PACK (`run.save_answer_pack`, after the timed scope; every record carries
`meta.answer_messages_sha256`). The other five readers answer that identical pack (`ragbench.tools.reader_replay`);
a replay is refused if the pack hash differs from the primary record's. The pack is therefore byte-identical across
readers. Consequence, declared now: a non-primary reader's cell is "this arm's GLM-run pack, read by reader X"; its
latency = shared pre-answer time (from the primary) + its own answer call; its tokens = shared selector tokens + its
own answer call. A primary cell that fails twice (system failure) has no pack, so all six readers score 0 there.

**Infrastructure rule for replays** = section 3: a replay whose every HTTP attempt failed gets one rerun (logged in
`test_run/reruns.jsonl`); a second error is a system failure.

**Judging.** Unchanged pinned judge and malformed policy (section 4), run as one `judge_multi` process per reader with
its own output dir `test_run/judge/<reader>/` (verdict files never shared between processes).

**Decision rules.** Section 5 unchanged and evaluated on the **primary reader (GLM)** only: LEX5B_6000 >= +5/90
answerable vs HYBRID_6000 and vs HR_6000; tokens <= 2.5x contemporaneous HYBRID_6000 (GLM cells; LEX5B tokens =
selector + reader); exact sign tests + Holm reported separately. **SETR has no gate**: SETR vs LEX5B is a secondary
paired comparison with SETR's token ratio. The other five readers are a **descriptive generality analysis** (same
nets, sign tests without a gate). The blinded human review covers the GLM 6,000 discordants: LEX5B vs HYBRID / HR and
SETR vs LEX5B.

**Unit tests at 77924d4.** All suites pass (`test_run/unit_tests_refreeze.log`; new `test_reader_replay` 3/3,
`test_setr` 5/5) except the two pre-existing failures, which touch no test arm: `test_rerank`
isolation.only_hybrid_imports_rerank is a static check that only `approaches/hybrid.py` imports the reranker (the dev
tool `tools/rerank_smoke.py` also does; the 31 functional rerank checks pass); `test_foundation` A.loaded expects 20
documents in corpus set A (it has 22); corpus A is not used by any test pool (C, D, K, L).

**Reader smoke (dev, before this freeze; `test_run/smoke/SMOKE.json`).** Three dev questions (BOOK-fa-T1-003,
LAW-en-T2-002, BOOK-en-T3-002), HYBRID@6000 pack captured with gpt-oss-20b, then replayed to the other four: 15/15
answers pass the contract check (non-empty, finish `stop`, no reasoning leak, abstain token alone, answer language =
question language); served by the pinned providers. No reader needed a fix or a replacement.

**Cost check (STEP 0, `test_run/balance_check.json`).** Account credits left $11.19 (shared account). Estimate:
readers $2.12 (768 answers each), selectors $0.33, judge $0.75 (4,608 calls), x1.3 = **$4.2**; + $3 safety = $7.2 <
$11.19, so all 8 cells run (SETR@15500 not dropped). Runner stops: reader file > start + $5.00, or account < $1.50.

**Operational note (2026-10-01 23:10; changes no method, arm, reader setting or gate).** To finish faster, the work
runs as three detached, resumable workers (`test_run/launch.ps1`): (1) the PRIMARY GLM cells stay serial and
interleaved exactly as in section 3 (their latency feeds the trade-off and token/latency report); (2) the five replay
readers run in parallel, 2 concurrent calls per reader (DeepSeek 3), sharded by arm so no two threads write one file;
429/5xx are retried by the frozen reader attempt policy (7 attempts, back-off 5-60 s); replay latencies are
descriptive only; (3) the pinned judge runs continuously (`judge --follow`, a resumable pass every 5 minutes, one
judge process per reader), with the unchanged rules of section 4. Infrastructure event: the primary cell
RPT-fa-T1-002 x LEX5B_15500 hung >15 min while loading its index (before any query call; GPU at 100%, VRAM 7.6/8.2 GB)
and was killed; no record had been written, so it ran again (logged in `test_run/reruns.jsonl`).

## 10. Amendment 2 (2026-10-04, after all answers were attempted; changes no method, arm, reader setting or gate): infrastructure clean-up

**Why.** The DeepSeek replay endpoint (deepinfra/fp8 via OpenRouter) returned HTTP 429 (provider rate limit) for
~60 cells, often twice, so under the section-3 rule they would count as "system failures" and score 0. A 429 says
nothing about the method or the reader's ability: it is a provider quota event. Scoring it as a wrong answer would
penalise one reader for its host's traffic. The same holds for a local Ollama connection outage (process died; logged in
`test_run/PROGRESS.md`).

**Rule.** Only after every cell has been attempted, `test_run/infra_cleanup.py` re-tries ONLY cells whose every record
is an infrastructure error of kind HTTP 429 or local Ollama connection error, regardless of the earlier one-rerun
counter: concurrency 1 per reader, up to 5 tries per cell, back-off 10 / 30 / 90 / 270 s before tries 2-5 (each try =
one ordinary reader call under the frozen attempt policy). Every try is logged in `test_run/reruns.jsonl` with
`kind="infra-cleanup"`. A finished (non-error) cell is never touched or re-answered. A try that ends in any other
error kind stops that cell (original rule). Cells still failing after 5 tries remain system failures (score 0) and are
listed in TEST_RESULTS.md. The 14 never-attempted DeepSeek cells left by the 2026-10-04 stop were first attempted as
ordinary replay calls (concurrency 1). Everything is reported as an operational note, not a method result.
