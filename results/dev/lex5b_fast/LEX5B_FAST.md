# LEX5B-fast: can the LEX5B selector be made faster before the frozen test? (2026-10-01)

**Recommendation: keep LEX5B in the frozen protocol, unchanged.** No fast variant passed the rule declared before
any call, so Step B (the 60 answers) was not run. Nothing here needs to be reported as a separate test arm. The
dev-only speed results can go in the development history as a negative result.

## Where the time goes (saved data, no calls)

- The pre-selector work in dev60 LEX5B@6000 takes about 80 ms at p50:
  - encode 18 ms;
  - F2 4 ms;
  - packages 14 ms;
  - view 45 ms.
- The selector call takes 1.76 s at p50 and 3.04 s at p95. The answer call takes 1.79 s at p50.
- Selector wall time tracks completion tokens, which are mostly reasoning (r = 0.84):
  - 13 tokens take about 0.55 s;
  - about 280 tokens take about 2.3 s.
- In the saved S-c replies, completion is 49 tokens at p50, 233 at p95 and 279 at most. The 600 cap never binds.

**So two of the levers cannot move p50:**
- **Overlap (lever 1)** can save at most about 15–30 ms. It is built (`Lex5bFast24`, results-neutral), but it was not
  worth an answer run.
- **A lower completion cap (lever 2b)** cannot shorten a reply that succeeds; it only turns late deadline failures
  into earlier truncation failures. It was not tested.

## Lever 3: a faster endpoint for the same model

The probe re-measured 16 eligible (non-fp4) endpoints with 3 non-dev prompts each (`providers_probe3.json`).
- Together: median 1,811 ms, 3/3 replies parsed.
- The rule needed a median below 1,539 ms (0.85x Together's) with 3/3 replies parsed.
- The fastest others were friendli at 2,044 ms, reka at 2,467 ms and baseten/fp8 at 2,679 ms.
- **No endpoint qualified**, so the selector stays on Together and F-P was not run.
- The probe's own $0.03 guard stopped it early. The guard reads the shared budget file, so spending by Step A and the
  parallel agent counted against it. The endpoints left unprobed all had dev60 warm-up medians of at least 3.2 s.

## Step A: selector-only replay (54 answerable dev questions, arms interleaved; `TABLES_STEP_A.md`)

| arm | span retention | complete (fa/en) | fallbacks | selector tokens/q | select p50 / p95 ms |
|---|---:|---|---:|---:|---:|
| S-c = LEX5B (saved, 2026-09-26) | **0.854** | **38** (20/18) | 7 (all deadline) | 2,779 | 1,252 / 3,030 |
| F-16: glm, top-16 at the top-24 cap | 0.813 | 35 (19/16) | 9 (8 deadline, 1 duplicate alternative) | 2,008 | 1,556 / 3,024 |
| F-OSS: gpt-oss-20b on Bedrock, literal top-24 | 0.788 | 33 (18/15) | 13 (8 duplicate alternative, 4 bad ids, 1 empty) | 2,741 | 1,181 / 2,066 |

The declared rule was: retention ≥ 0.84, complete ≥ 37, tokens ≤ 2,779, and select p50 ≤ 939 ms (a 25% cut).
**Neither variant qualifies.** Both lose retention, and neither reaches the latency target:
- **F-16** cut the prompt to about 1,930 tokens, but its p50 was *slower* than S-c's. In the same window, Together's
  probe median (1,811 ms) was also slower than in September, so this is cross-session drift, and the shorter prompt
  saves almost nothing in any case.
- **F-OSS** is faster only at p95, because it has no deadline hits. Most of its 13 fallbacks are grammar failures:
  v1.1 rejects alternatives like `1 | 1`. Its retention (0.788) matches the earlier smart-view result (0.789).

## Not done, and why

- **Step B (60 answers):** not run, because no variant was picked by the rule.
- **Untested lever:** turning off GLM reasoning, or setting a minimal effort, is the only lever the measurements point
  to. The time is in reasoning tokens. It was not on the declared list and was not tried.

## Cost and artifacts

- **Spend:** $0.042 in this task's ledgers (probe and Step A).
- **Shared budget file:** 10.4957 → 10.5663. This includes the parallel agent.
- **Data:** `calls_F-*.jsonl`, `rows_F-*.jsonl`, `results_step_a.json`, `dry_check.json` (fast_view top-24 is
  byte-identical to S-c's prompts, 54/54), `providers_probe3.json` and `probe3.log`.
- **Harness, unused:** `answers_fast.py` and `report_fast.py`.
- **Registry:** 2 rows were added (`lex5b-fast-stepA`, budget_chars 6000, admitted false).
- **Code:** commit `b2bf796` on the side branch **`lex5b-fast`** (parent 1cdb056), not on `upgrades`, so it cannot
  touch the test code.

**Freeze note.** `upgrades` HEAD is already **1cdb056** (the parallel agent's SETR commit), not the frozen
**b31cd4c**. The `--test-protocol` gate refuses any other HEAD, so the test must run from a checkout of b31cd4c, or
the freeze must be redone.
