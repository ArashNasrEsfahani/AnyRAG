# Budget matrix: evidence budget x arm on the idea_test 24 (2026-09-26)

**Decision (user, 2026-09-26).** The evidence budget is a study variable with two levels, 6,000 and 15,500 rendered chars, for every arm. LEX is judged only against same-budget controls.

**Fixed settings in every cell.** GLM z-ai/glm-5.3-flash reader with the lex5_pilot/providers.json pins; judge judge_multi gpt-oss-120b, strict v2; the shared chunk packer `pack_evidence` at the cell's budget. Only the pack budget differs between a 6,000 cell and its 15,500 cell.

## 1. The matrix (correct /24, strict)

| arm | @6,000 | @15,500 | Δ budget | tokens/q @6,000 | tokens/q @15,500 | × HYBRID@6,000 (15,500) | p50 / p95 s @6,000 | p50 / p95 s @15,500 | source @15,500 |
|---|---:|---:|---:|---:|---:|---:|---|---|---|
| HYBRID | **12** | **19** | +7 | 2169 | 5152 | 2.38 | 1.62 / 3.69 | 2.68 / 11.26 | lex10/b_control (reused) |
| HR | **13** | **16** | +3 | 2215 | 5229 | 2.41 | 2.74 / 5.59 | 3.61 / 5.73 | this task |
| LEX7.1 | **13** | **14** | +1 | 3152 | 6097 | 2.81 **> 2.5x gate** | 2.30 / 4.38 | 2.90 / 6.39 | this task |
| LEX10/F2 | not run | **15** | – | – | 5150 | 2.37 | – | 1.41 / 2.80 | lex10/b (reused) |
| LEX5B | **17** | **15** | -2 | 4315 (4964 cons.) | 6731 (8031 cons.) | 3.70 **> 2.5x gate** | 4.38 / 7.74 | 4.99 / 8.72 | this task |
| HYBRID+KW | **10** | not run | – | 2191 | – | – | 1.44 / 3.29 | – | – |
| DOC-3 | **11** | not run | – | 2188 | – | – | 1.52 / 3.46 | – | – |
| LEX6 | **11** | not run | – | 2141 | – | – | 1.46 / 2.52 | – | – |

By language and tier (correct):

| cell | fa | en | T1 | T2 | T3 | shown retention (100%) | mean pack chars | $ (24) |
|---|---:|---:|---:|---:|---:|---|---:|---:|
| HYBRID @6,000 | 5 | 7 | 6 | 2 | 4 | 0.728 (14) | 5941 | 0.0085 |
| HYBRID @15,500 | 9 | 10 | 7 | 7 | 5 | 0.878 (18) | 15447 | 0.0197 |
| HR @6,000 | 6 | 7 | 6 | 1 | 6 | 0.828 (17) | 5954 | 0.0088 |
| HR @15,500 | 7 | 9 | 6 | 4 | 6 | 0.863 (18) | 15415 | 0.0199 |
| LEX7.1 @6,000 | 6 | 7 | 7 | 2 | 4 | 0.781 (15) | 5982 | 0.0108 |
| LEX7.1 @15,500 | 7 | 7 | 6 | 3 | 5 | 0.877 (19) | 15439 | 0.0214 |
| LEX10/F2 @15,500 | 7 | 8 | 7 | 3 | 5 | 0.898 (19) | 15408 | 0.0197 |
| LEX5B @6,000 | 6 | 11 | 7 | 5 | 5 | 0.852 (17) | 5803 | 0.0165 |
| LEX5B @15,500 | 8 | 7 | 6 | 4 | 5 | 0.898 (19) | 15314 | 0.0255 |
| HYBRID+KW @6,000 | 6 | 4 | 6 | 1 | 3 | 0.726 (14) | 5944 | 0.0078 |
| DOC-3 @6,000 | 6 | 5 | 7 | 1 | 3 | 0.684 (14) | 5950 | 0.0064 |
| LEX6 @6,000 | 5 | 6 | 5 | 1 | 5 | 0.754 (15) | 5828 | 0.0087 |

**Token gate flags.** The amended LEX gate is 2.5x HYBRID@6,000's 2,169 tokens/q = 5,422. **LEX7.1@15,500 (6,097 =
2.81x) and LEX5B@15,500 (6,731 measured, 8,031 conservative = 3.70x) VIOLATE it.** They are informative cells, not
eligible LEX results. LEX10/F2@15,500 (2.37x) is inside. The "conservative" figure counts each selector call cancelled
at its deadline as its prompt estimate + 600, as in LEX5B.md.

**Where the new cells came from.** HR, LEX7.1 and LEX5B @15,500 were run here (`ragbench/approaches/budget_matrix.py`,
commit d859486 on `upgrades`). The arm code is unchanged; only the pack budget differs. The 24 questions were
interleaved (three arms back to back per question, arm order rotated), one process at a time. Spend: $0.068 on
answers plus $0.014 on the judge (reader file 10.1109 → 10.1925). Judge: 72 records; one malformed judge reply
(unit status "extra") was retried once with `--retry-failures` and succeeded.

**Parity checks.**
- HR15500's order40 equals the HR@6,000 record's order (40/40, smoke question).
- LEX5B_15500's selector prompt and F2 top 40 are identical to the LEX5B@6,000 record.

**Latency caveats.**
- The new cells were interleaved with each other. They were not interleaved with the reused cells: all @6,000 cells,
  HYBRID@15,500 and LEX10/F2@15,500 were measured earlier the same day.
- HYBRID@15,500's p95 of 11.26 s is contaminated: its first two questions ran during Ollama warm-up while the GPU was
  shared with the parallel LEX5b run.
- **Shared GPU during this run.** HR's bge-reranker ran on cuda (rerank p50 791 ms, against 910 ms at 6,000), and so
  did the bge-m3 query encoder of LEX7.1 and LEX5B. No other ragbench job was running. A LAYA `studio/server.py`
  process held GPU memory, and Ollama's llama-server was resident. Mild contention cannot be excluded, but nothing
  suggests it: the reranker was faster than at 6,000.

**LEX5B@15,500 is confounded by selector deadlines.**
- 10 of 24 selector calls hit the 3 s deadline, against 5 of 24 at 6,000. The prompt was identical, so the extra
  fallbacks come from provider latency (selector p50 1.58 s, p95 3.05 s).
- On those 10 questions the pack is the F2-intact fallback, not the selected bundles.
- Three of its four losses against LEX5B@6,000 (RPT-en-T2-002, BOOK-en-T2-002, BOOK-en-T3-001) are questions whose
  selector succeeded at 6,000 and fell back at 15,500.
- Its −2 budget effect therefore does not isolate budget.

## 2. Same-budget comparisons (the only comparisons LEX is judged on)

Net = LEX correct − control correct on the same 24. Token and latency ratios are LEX / control at the same budget.

| budget | LEX arm | vs | LEX | control | net | LEX-only correct | control-only correct | tokens × | p50 × | p95 × |
|---|---|---|---:|---:|---:|---|---|---:|---:|---:|
| 6,000 | LEX7.1 | HYBRID | 13 | 12 | **+1** | RPT-fa-T2-002, BOOK-en-T1-001 | RPT-en-T2-002 | 1.45 | 1.42 | 1.19 |
| 6,000 | LEX7.1 | HR | 13 | 13 | **+0** | RPT-fa-T2-002, BOOK-en-T1-001, LAW-en-T2-001 | RPT-fa-T3-002, BOOK-en-T2-002, BOOK-en-T3-001 | 1.42 | 0.84 | 0.78 |
| 6,000 | LEX10/F2 | HYBRID | not run | 12 | – | – | – | – | – | – |
| 6,000 | LEX10/F2 | HR | not run | 13 | – | – | – | – | – | – |
| 6,000 | LEX5B | HYBRID | 17 | 12 | **+5** | RPT-fa-T2-002, BOOK-en-T2-002, RPT-en-T2-001, BOOK-en-T1-001, BOOK-en-T3-001 | – | 1.99 | 2.70 | 2.10 |
| 6,000 | LEX5B | HR | 17 | 13 | **+4** | RPT-fa-T2-002, RPT-en-T2-002, RPT-en-T2-001, BOOK-en-T1-001, LAW-en-T2-001 | RPT-fa-T3-002 | 1.95 | 1.60 | 1.39 |
| 15,500 | LEX7.1 | HYBRID | 14 | 19 | **-5** | – | LAW-fa-T2-002, BOOK-en-T2-002, RPT-en-T2-001, BOOK-fa-T2-001, BOOK-en-T1-001 | 1.18 | 1.08 | 0.57 |
| 15,500 | LEX7.1 | HR | 14 | 16 | **-2** | LAW-en-T2-001 | BOOK-en-T2-002, RPT-en-T2-001, BOOK-en-T3-001 | 1.17 | 0.80 | 1.12 |
| 15,500 | LEX10/F2 | HYBRID | 15 | 19 | **-4** | – | LAW-fa-T2-002, BOOK-en-T2-002, RPT-en-T2-001, BOOK-fa-T2-001 | 1.00 | 0.52 | 0.25 |
| 15,500 | LEX10/F2 | HR | 15 | 16 | **-1** | BOOK-en-T1-001, LAW-en-T2-001 | BOOK-en-T2-002, RPT-en-T2-001, BOOK-en-T3-001 | 0.98 | 0.39 | 0.49 |
| 15,500 | LEX5B | HYBRID | 15 | 19 | **-4** | – | RPT-en-T2-002, BOOK-en-T2-002, BOOK-fa-T2-001, BOOK-en-T1-001 | 1.31 | 1.86 | 0.77 |
| 15,500 | LEX5B | HR | 15 | 16 | **-1** | LAW-fa-T2-002, LAW-en-T2-001 | RPT-en-T2-002, BOOK-en-T2-002, BOOK-en-T3-001 | 1.29 | 1.38 | 1.52 |

## 3. Budget effect within each arm (15,500 vs 6,000, same arm)

| arm | @6,000 | @15,500 | net | gained at 15,500 | lost at 15,500 |
|---|---:|---:|---:|---|---|
| HYBRID | 12 | 19 | +7 | RPT-fa-T2-002, RPT-fa-T3-002, LAW-fa-T2-002, BOOK-en-T2-002, RPT-en-T2-001, BOOK-fa-T2-001, BOOK-en-T1-001 | – |
| HR | 13 | 16 | +3 | RPT-fa-T2-002, RPT-en-T2-002, RPT-en-T2-001 | – |
| LEX7.1 | 13 | 14 | +1 | RPT-fa-T3-002, RPT-en-T2-002 | BOOK-en-T1-001 |
| LEX5B | 17 | 15 | -2 | RPT-fa-T3-002, LAW-fa-T2-002 | RPT-en-T2-002, BOOK-en-T2-002, BOOK-en-T1-001, BOOK-en-T3-001 |

## 4. Per question (1 = correct)

| id | HYBRID@6k | HYBRID@15.5k | HR@6k | HR@15.5k | LEX7.1@6k | LEX7.1@15.5k | LEX10/F2@6k | LEX10/F2@15.5k | LEX5B@6k | LEX5B@15.5k |
|---|---|---|---|---|---|---|---|---|---|---|
| RPT-fa-T1-002 | 1 | 1 | 1 | 1 | 1 | 1 | – | 1 | 1 | 1 |
| RPT-fa-T2-002 | 0 | 1 | 0 | 1 | 1 | 1 | – | 1 | 1 | 1 |
| RPT-fa-T3-002 | 0 | 1 | 1 | 1 | 0 | 1 | – | 1 | 0 | 1 |
| RPT-en-T1-002 | 1 | 1 | 1 | 1 | 1 | 1 | – | 1 | 1 | 1 |
| RPT-en-T2-002 | 1 | 1 | 0 | 1 | 0 | 1 | – | 1 | 1 | 0 |
| RPT-en-T3-002 | 1 | 1 | 1 | 1 | 1 | 1 | – | 1 | 1 | 1 |
| BOOK-fa-T1-002 | 0 | 0 | 0 | 0 | 0 | 0 | – | 0 | 0 | 0 |
| LAW-fa-T2-002 | 0 | 1 | 0 | 0 | 0 | 0 | – | 0 | 0 | 1 |
| BOOK-fa-T3-002 | 0 | 0 | 0 | 0 | 0 | 0 | – | 0 | 0 | 0 |
| LAW-en-T1-002 | 1 | 1 | 1 | 1 | 1 | 1 | – | 1 | 1 | 1 |
| BOOK-en-T2-002 | 0 | 1 | 1 | 1 | 0 | 0 | – | 0 | 1 | 0 |
| LAW-en-T3-002 | 0 | 0 | 0 | 0 | 0 | 0 | – | 0 | 0 | 0 |
| RPT-fa-T1-001 | 1 | 1 | 1 | 1 | 1 | 1 | – | 1 | 1 | 1 |
| RPT-fa-T2-001 | 0 | 0 | 0 | 0 | 0 | 0 | – | 0 | 0 | 0 |
| RPT-fa-T3-001 | 1 | 1 | 1 | 1 | 1 | 1 | – | 1 | 1 | 1 |
| RPT-en-T1-001 | 1 | 1 | 1 | 1 | 1 | 1 | – | 1 | 1 | 1 |
| RPT-en-T2-001 | 0 | 1 | 0 | 1 | 0 | 0 | – | 0 | 1 | 1 |
| RPT-en-T3-001 | 1 | 1 | 1 | 1 | 1 | 1 | – | 1 | 1 | 1 |
| BOOK-fa-T2-001 | 0 | 1 | 0 | 0 | 0 | 0 | – | 0 | 0 | 0 |
| LAW-fa-T1-001 | 1 | 1 | 1 | 1 | 1 | 1 | – | 1 | 1 | 1 |
| LAW-fa-T3-001 | 1 | 1 | 1 | 1 | 1 | 1 | – | 1 | 1 | 1 |
| BOOK-en-T1-001 | 0 | 1 | 0 | 0 | 1 | 0 | – | 1 | 1 | 0 |
| LAW-en-T2-001 | 1 | 1 | 0 | 0 | 1 | 1 | – | 1 | 1 | 1 |
| BOOK-en-T3-001 | 0 | 0 | 1 | 1 | 0 | 0 | – | 0 | 1 | 0 |

## 5. Reading (24-question screen; one question = one answer; nothing here is confirmatory)

1. **At 15,500, no LEX arm beats either same-budget control.** Against HYBRID@15,500 (19): LEX7.1 −5, LEX10/F2 −4,
   LEX5B −4, and no LEX arm is right on any question HYBRID gets wrong. Against HR@15,500 (16): −2, −1, −1.
2. **At 6,000, only LEX5B clears +3 against both controls** (+5 against HYBRID, +4 against HR). It fails the latency
   gate there (p50 2.70x HYBRID, 1.60x HR). LEX7.1 is +1 / 0. LEX10/F2 has no 6,000 answer cell.
3. **Budget is the largest single effect on this screen.** HYBRID gains +7 from 6,000 to 15,500 (7 gained, 0 lost)
   and HR gains +3 (0 lost).
   - The arms that re-order evidence gain little or lose: LEX7.1 +1; LEX5B −2, confounded by its selector deadlines.
   - Plain RRF order at 15,500 is the strongest cell measured (19/24).
   - The LEX gains measured at 6,000 look like a small-pack effect: getting the right evidence into 6,000 chars.
     A wider pack gives the same gain more cheaply.
4. **Same-budget verdict under the user's rule.**
   - At 6,000: LEX5B passes accuracy but fails latency.
   - At 15,500: no LEX arm passes accuracy, and LEX7.1 and LEX5B also break the 2.5x token gate.

## Files

- `PROGRESS.md`: the declarations and the log.
- `backfill_budget.py`: registry step 1; the backup is `RESULTS_REGISTRY.before_backfill.jsonl`.
- `answers_bm.py`: the run and judge harness. `eval/runs/<pool>/{HR15500,LEX7_15500,LEX5B_15500}` hold the records;
  `eval/judge` holds the verdicts.
- `build_matrix.py`: builds this file's tables and `budget_matrix.json` (cells, same-budget comparisons, budget
  effect, per-question values).
