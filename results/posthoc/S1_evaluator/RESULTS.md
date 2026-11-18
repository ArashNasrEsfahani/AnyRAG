# S1 "evaluator": second judge + human-review packets (post-hoc, exploratory)

All numbers below come from `judge2_results.md` / `judge2_results.json` (written by `analyze_judge2.py`) and from `ledger.jsonl`. Nothing was tuned on test. The answers were not regenerated: these are the frozen GLM-5.3-flash test answers.

## 1. What was run

**Task 1: second judge.** `judge2.py --estimate` judged 3 questions × 8 cells. `judge2.py --all --shards 3` then judged all 768 GLM-5.3-flash answers (8 arm-budget cells × 96 test questions).
- **Judge 2:** `mistralai/mistral-large-2512` (Mistral family, which none of the six readers belongs to). It is pinned to the OpenRouter endpoint `mistral/zdr` with `allow_fallbacks: false` and `require_parameters: true`, at temperature 0, top_p 1 and seed 42 (the same generation settings as `judge_pin.json`). Every reply was served by Mistral, and every finish_reason was `stop`.
- **Same rubric, prompt and parser as judge 1:** the run calls `ragbench.tools.judge_multi.judge_records` from the frozen worktree @77924d4 unchanged (rubrics judge-v2-multi-v1 and judge-v2-unans-v1). That includes the same rule verdicts, up to 3 identical judge attempts, and the rule that an unparsable reply goes to needs_adjudication and is counted as a bound [0, 1].
- **Changes from the original run:** only the client, which is a thin pinned OpenRouter client with up to 6 transport retries for HTTP 429, plus a per-shard cache. max_tokens = num_predict + 64, because the model does no reasoning.
- **Records:** exactly the records `report_test.load()` scores. A record with a run error counts 0, as in the test report.
- **Statistics:** sign-p ranges are computed exactly over every assignment of unresolved cells, using a DP over the (better, worse) counts. Bootstrap: 10,000 resamples, seed 0, over questions and over the 20 source-family components. This copies the logic of `revision_stats.py`; that module was not imported because importing it runs the whole module.

**Task 2: human-review packets (no API calls).** `build_packets.py` writes `packets/A_concordant24/` and `packets/B_second_reviewer/`. `human_estimate.py` does the merge and the estimate. `test_human_estimate.py` is the synthetic test. `README.md` covers loading, reviewing and merging.

## 2. Results

### 2.1 Correct answers per cell (bounds over unresolved cells)

| cell | J1 /90 | **J2 /90** | J1 /96 | J2 /96 | unanswerable /6 (J1, J2) | J2 unresolved |
|---|---:|---:|---:|---:|---:|---:|
| HYBRID_6000 | 62-63 | **59-61** | 68-69 | 65-67 | 6, 6 | 2 |
| HR_6000 | 69-71 | **63** | 75-77 | 69 | 6, 6 | 0 |
| LEX5B_6000 | 68-69 | **65-67** | 74-75 | 71-73 | 6, 6 | 2 |
| SETR_6000 | 63 | **63-65** | 69 | 69-71 | 6, 6 | 2 |
| HYBRID_15500 | 68 | **65-68** | 74 | 71-74 | 6, 6 | 3 |
| HR_15500 | 69-70 | **66-68** | 75-76 | 72-74 | 6, 6 | 2 |
| LEX5B_15500 | 67 | **67** | 73 | 73 | 6, 6 | 0 |
| SETR_15500 | 66-67 | **64-65** | 72-73 | 70-71 | 6, 6 | 1 |

Judge 2 left 12 of 768 cells unresolved (1.6%). Every one fails for the same reason: three identical replies that used the claim status `missing`, which the frozen parser rejects. Abstention on the 6 unanswerable questions is 6/6 in every cell under both judges.

### 2.2 Judge 1 vs judge 2 agreement (cells resolved under both)

| scope | comparable | agree % | kappa | J1=1, J2=0 | J1=0, J2=1 |
|---|---:|---:|---:|---:|---:|
| all 8 cells | 750 | 91.2 | 0.760 | 43 | 23 |
| four 6,000 cells | 374 | 91.7 | 0.781 | 22 | 9 |
| per cell | 93-96 | 89.4-94.6 | 0.705-0.868 | | |

The full per-cell table is in `judge2_results.md` §2. HR_6000 has the lowest kappa (0.705); SETR_6000 is at 0.782.

### 2.3 Each judge vs the human on the 120 reviewed cells, per pipeline

"Stricter" means judge 0 and human 1; "laxer" means judge 1 and human 0.

| pipeline | J1 agree | J1 stricter / laxer | J1 kappa | J2 agree | J2 stricter / laxer | J2 kappa | human correct /30 |
|---|---:|---:|---:|---:|---:|---:|---:|
| HYBRID_6000 | 20/29 | 9 / 0 | 0.431 | 20/29 | 9 / 0 | 0.431 | 20 |
| HR_6000 | 22/28 | 6 / 0 | 0.462 | 21/30 | 9 / 0 | 0.372 | 25 |
| LEX5B_6000 | 22/29 | 7 / 0 | 0.456 | 20/29 | 9 / 0 | 0.329 | 25 |
| SETR_6000 | 15/30 | **15** / 0 | 0.138 | 17/29 | **12** / 0 | 0.194 | 26 |
| all | 79/116 | 37 / 0 | 0.362 | 78/117 | 39 / 0 | 0.344 | 96 |

Neither judge is ever laxer than the human. Judge 2 is about as strict as judge 1 overall (39 vs 37 cells). But its strictness is spread evenly over the pipelines (9/9/9/12), while judge 1 concentrates it on SETR (15 against 6-9 for the others).

### 2.4 Does the SETR reordering appear with judge 2? Partly

| scoring (correct /90 @6,000) | HYBRID | HR | LEX5B | SETR |
|---|---:|---:|---:|---:|
| judge 1 | 62-63 | 69-71 | 68-69 | 63 (tied last with HYBRID) |
| **judge 2** | 59-61 | 63 | 65-67 | **63-65** |
| symmetric human merge (judge 1 outside the 30) | 71 | 76 | 76 | **78 (first)** |
| symmetric human merge with judge 2 outside the 30 | 68-69 | 72 | 74-75 | **76-77 (first)** |

- **Not last under judge 2:** SETR comes second, behind LEX5B (65-67). It ties or beats HR (63) and beats HYBRID (59-61). So SETR's tied-last place is specific to judge 1, and it does not reproduce with a second judge.
- **Not first either:** judge 2 does not put SETR first. That happens only once human verdicts are used.
- **Robust to the choice of judge:** with the human merge, SETR is first whichever judge scores the other 60 questions (76-77 vs LEX5B 74-75).
- **Sign tests:** SETR vs LEX5B is −4..+0 under judge 2 (p 0.45-1.00) against −6..−5 under judge 1 (p 0.18-0.30). SETR vs HR is +0..+2 under judge 2 against −8..−6 under judge 1.

### 2.5 Primary paired comparisons under judge 2 (90 answerable)

| comparison | J2 net | J2 sign p | J2 95% CI pp (question / component) | J1 net (sign p) |
|---|---|---|---|---|
| LEX5B_6000 vs HYBRID_6000 | +4..+8 (10/5) | 0.096-0.455 | [−4.4, 17.8] / [−2.4, 20.9] | +5..+7 (0.118-0.332) |
| LEX5B_6000 vs HR_6000 | +2..+4 (10/7) | 0.481-0.815 | [−6.7, 13.3] / [−4.8, 15.2] | −3..+0 (0.607-1.000) |
| LEX5B_15500 vs HYBRID_15500 | −1..+2 | 0.79-1.00 | [−10.0, 10.0] / [−9.1, 10.3] | −1 (1.000) |
| LEX5B_15500 vs HR_15500 | −1..+1 | 1.000 | [−11.1, 10.0] / [−8.0, 9.7] | −3..−2 (0.63-0.80) |
| HYBRID 15500 vs 6000 | +4..+9 | 0.049-0.455 | [−4.4, 18.9] / [−4.1, 18.9] | +5..+6 (0.18-0.30) |
| HR 15500 vs 6000 | +3..+5 | 0.227-0.549 | [−3.3, 12.2] / [−2.9, 10.2] | −2..+1 (0.79-1.00) |
| LEX5B 15500 vs 6000 | +0..+2 | 0.81-1.00 | [−8.9, 11.1] / [−10.1, 8.7] | −2..−1 (0.80-1.00) |
| SETR 15500 vs 6000 | −1..+2 | 0.77-1.00 | [−8.9, 10.0] / [−8.7, 8.8] | +3..+4 (0.34-0.55) |
| HYBRID_15500 vs LEX5B_6000 | −2..+3 | 0.63-1.00 | [−11.1, 12.2] / [−14.3, 9.3] | −1..+0 (1.000) |

- **No significance:** no comparison is significant under judge 2 at every assignment of unresolved cells. HYBRID 15500 vs 6000 reaches p 0.049 only at the most favourable assignment.
- **Direction:** LEX5B ≥ both controls at 6,000 holds under judge 2. The +5/90 gate is met against HYBRID only at some assignments (+4..+8), and is not met against HR (+2..+4).
- **By tier and language (judge 2):** the LEX5B vs HYBRID gain comes entirely from English (+4..+8; Persian +0) and mostly from T3 (+3..+5; T1 +0, T2 +1..+3). Tables are in `judge2_results.md`.

### 2.6 Packets (task 2)

| packet | questions | cells | items shown (identical answers merged) | order |
|---|---:|---:|---:|---|
| A: 24 of the 60 concordant questions (seed 20261110) | 24 | 96 | 82 | questions random, answers shuffled per question (seed 20261111) |
| B: second reviewer, the 120 reviewed cells | 30 | 120 | 116 | one random order across questions (seed 20261112) |

**Formats.** Each packet comes in two formats:
- **Stand-alone HTML page:** the earlier template, with the same export JSON, plus a requirements card and a gold-evidence-quote card. The page was checked by rendering it in a headless browser.
- **review_game answer-check files:** checked by loading them with `review_game/server.py`'s own `load_questions` / `blinded_answers` functions. All 24 + 30 questions loaded, every evidence unit resolved, and no evidence errors.

**Blinding.** A grep for method names in the packet files finds no match. The keys are kept in separate files.

**Merge and estimate.** `human_estimate.py` computes:
- a census on the 30 reviewed questions, plus an estimate on the 60 concordant ones. The estimate is given three ways: naive (machine verdicts), PPI (difference estimator with finite-population correction) and Rogan-Gladen (per pipeline and pooled);
- PPI estimates of the paired nets;
- for packet B, Cohen's kappa between the two reviewers and between each judge and reviewer 2, per pipeline.

**Synthetic test.** `test_human_estimate.py` prints ALL PASS. It covers the parsers in both formats, the half-correct rule, the PPI identity, PPI 95% coverage (0.928 over 4,000 Monte Carlo populations) and hand-computed kappa values. No real human verdicts exist yet.

## 3. Cost

- **Total:** $0.8325 of the $1.00 cap, from `ledger.jsonl`. 687 successful calls: 1,136,927 prompt tokens and 197,353 completion tokens. 199 HTTP-429 transport retries were not billed.
- **Estimate first:** the 3-question run cost $0.0147 for 21 calls. Extrapolated, that was about $0.5 for the full run.
- **Why the run cost more:** longer prompts on the later questions (about 1,650 against about 1,240 prompt tokens per call) and malformed-reply re-attempts.
- **Global floor:** checked before every 50 calls (limit_remaining was $9.61 at the start).
- **Task 2:** made no API calls.

## 4. Failed or skipped

- **Unresolved cells:** 12 judge-2 cells are needs_adjudication (status `missing` used in claims), reported as bounds. Following protocol, they were neither adjudicated nor re-prompted.
- **Duplicate verdict rows:** the estimate run used 4 shards and the full run 3, so 17 of the 24 estimate cells were judged a second time. All 17 repeat verdicts match the first ones. The analysis keeps the first verdict per (rid, answer_sha1), as the frozen policy does. This cost about $0.012.
- **Second judge on other readers:** not run. The brief asked only for the primary reader.
- **Human work:** none was done, as briefed. The packets are ready.
- **Lee et al. 2025:** the method was implemented from the brief's description, not checked against the paper (no web access was used). The Rogan-Gladen variance follows the standard delta method with the calibration terms only, because C is a census for the machine.
- **Housekeeping:** a headless-browser check briefly wrote a `.playwright-mcp/` folder at the repository root. I deleted it at once.

## 5. Facts for the paper

1. A second judge from a different family, Mistral Large 3 (2512) with the same rubric, pin policy and temperature 0, agrees with the pinned gpt-oss-120b judge on 91.2% of 750 GLM cells (Cohen's kappa 0.760), and on 91.7% (kappa 0.781) of the four 6,000-character cells (`judge2_results.md` §2).
2. Like judge 1, judge 2 is never laxer than the human on the 120 reviewed cells. It is stricter on 39 of 117 (judge 1: 37 of 116), but its strictness is spread evenly over the pipelines: 9/9/9/12 for HYBRID/HR/LEX5B/SETR, against 9/6/7/15 for judge 1 (`judge2_results.md` §3).
3. SETR's tied-last place at 6,000 is specific to judge 1. Under judge 2, SETR scores 63-65/90: behind LEX5B (65-67), level with or above HR (63), and above HYBRID (59-61). SETR ranks first only once human verdicts are used, and it stays first (76-77/90) when judge 2 replaces judge 1 outside the 30 reviewed questions (`judge2_results.md` §4).
4. Under judge 2, LEX5B_6000 vs HYBRID_6000 is +4..+8/90 (sign p 0.096-0.455) and LEX5B_6000 vs HR_6000 is +2..+4/90 (p 0.481-0.815). No 6,000 or 15,500 comparison or budget effect is significant at every assignment of unresolved cells (`judge2_results.md` §5).
5. Judge 2 left 12 of 768 cells (1.6%) unresolved under the frozen parser, every one because it used the claim status "missing". Re-judging cost $0.83 (`ledger.jsonl`, `judge2/shard_*/needs_adjudication_*.jsonl`).
