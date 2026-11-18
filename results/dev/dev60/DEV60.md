# DEV60: the finalists on all 60 dev questions (2026-09-27)

**User decision (2026-09-26): "Accept LEX5B as what it is."** LEX5B's latency is reported as a measured trade-off.
It is accurate but slower, and that is not treated as a disqualifying gate failure. The measured latency ratios are
still reported in full in section 4.

**What was run.** The 36 dev questions not in the idea_test 24 (30 answerable, 6 unanswerable) were answered for 7 cells
each (252 cells), interleaved. The saved 24 were reused, not re-run.
- Harness: `answers60.py`.
- Reader: GLM glm-5.3-flash, pinned to together.
- Judge: gpt-oss-120b, strict (v2).
- Merge: `report60.py`; full tables follow below.

**Spend.** $0.249 in total (reader file 10.2448 → 10.4932). The provider probe cost ~$0.035, the answers ~$0.17 and the
judge ~$0.04. The budget was $0.60.

**One judge gap.** LEX5B@15500 on RPT-fa-T2-003 got a malformed judge reply five times ("reply omits units R4–R7").
It is counted as 0 (unjudged), flagged here and not repaired. It is one of LEX5B@15500's losses against HYBRID@15500
and HR@15500.

## Headline

| cell | /60 | /54 answerable | AB /6 | fa /27 | en /27 |
|---|---:|---:|---:|---:|---:|
| HYBRID@6,000 | 35 | 30 | 5 | 15 | 15 |
| HR@6,000 | 38 | 32 | 6 | 17 | 15 |
| **LEX5B@6,000** | **43** | **37** | 6 | 17 | 20 |
| HYBRID@15,500 | 44 | 38 | 6 | 19 | 19 |
| HR@15,500 | 44 | 38 | 6 | 19 | 19 |
| F2@15,500 | 43 | 37 | 6 | 19 | 18 |
| LEX5B@15,500 | 42 | 36 | 6 | 19 | 17 |

- **At 6,000, LEX5B passes the accuracy and token gates against both controls.**
  - vs HYBRID: net **+7/54** (10 better, 3 worse; sign test p = 0.09).
  - vs HR: net **+5/54** (9/4; p = 0.27).
  - The gain holds on the new 36 as well: +3 vs HYBRID and +1 vs HR. The 24-question screen gave +5 and +4.
  - Tokens are 2.05x the control (2.29x conservative), so within 2.5x.
  - Latency on the 36 interleaved questions:
    - vs HYBRID: p50 2.12x, p95 2.98x. This is the accepted trade-off.
    - vs HR: p50 1.44x, p95 1.22x, both within 1.5x.
- **At 15,500, neither LEX arm beats its controls.**
  - LEX5B: −2 vs HYBRID, −2 vs HR, −1 vs F2.
  - F2 (LEX10b): −1 vs HYBRID, −1 vs HR.
  - The larger pack lets plain HYBRID and HR catch up: 38/54 each.
- **LEX5B@6,000 is about as accurate as the 15,500 controls** (37 against 38) at roughly 0.87x their tokens (4,451
  against ~5,150/q).

## Pre-run fixes and probes

**The LEX5B view fix is results-neutral** (commit 02d0e37 on `upgrades`).
- The cached literal fit gives byte-identical prompts on all 48 saved LEX5B selector prompts (`view_fix.json`).
- View-fit time, offline p50: 559 → 101 ms.
- View-fit time, online on the 36, p50: 45 ms at 6,000 and 44 ms at 15,500. The earlier online figure was ~402 ms.

**No faster eligible provider was found, so the selector stays on together** (`providers_probe.json`).
- The probe tried all 33 endpoints with 3 non-dev prompts.
- The fastest endpoints are fp4/nvfp4 quantisations, which are excluded.
- In the measured round, the eligible shortlist hit upstream 429 errors, and no endpoint parsed all 3 replies.
- Selector fallbacks on the 36: 6/36 at 6,000 (all deadline) and 7/36 at 15,500 (6 deadline, 1 call error).

**Caveat.** This is a heavily inspected dev set. The p-values are descriptive only.


# Full tables (54 answerable + 6 unanswerable)

**User decision (2026-09-26): "Accept LEX5B as what it is."** LEX5B's latency is reported as a measured trade-off (accurate but slower), not as a disqualifying gate failure. The latency ratios below are still the measured ones.

**Set.** The 24 idea_test questions are REUSED records + verdicts (RAG test once; answered in earlier sessions). The other 36 (30 answerable + 6 AB) were answered here, all seven cells per question back to back, cell order rotated (interleaved). Reader GLM z-ai/glm-5.3-flash pinned (together, lex5_pilot/providers.json); judge judge_multi gpt-oss-120b strict (v2). AB = unanswerable: correct = the judge's strict 1 (a proper abstention).

**Latency and tokens** for gates come from the 36 interleaved questions only; the 24 were measured in separate sessions. Dev set, much inspected: p-values are descriptive (exact two-sided sign test on discordant pairs).

## 1. Correct answers

| cell | budget | correct /60 | answerable /54 | AB abstain /6 | fa /27 | en /27 | T1 | T2 | T3 | saved 24 | new 36 (ans /30, AB /6) | shown retention (100%) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| HYBRID@6000 | 6000 | **35** | **30** | 5 | 15 | 15 | 16 | 7 | 7 | 12 | 23 (18, 5) | 0.720 (32) |
| HR@6000 | 6000 | **38** | **32** | 6 | 17 | 15 | 16 | 6 | 10 | 13 | 25 (19, 6) | 0.806 (35) |
| LEX5B@6000 | 6000 | **43** | **37** | 6 | 17 | 20 | 16 | 10 | 11 | 17 | 26 (20, 6) | 0.842 (36) |
| HYBRID@15500 | 15500 | **44** | **38** | 6 | 19 | 19 | 17 | 11 | 10 | 19 | 25 (19, 6) | 0.847 (39) |
| HR@15500 | 15500 | **44** | **38** | 6 | 19 | 19 | 16 | 10 | 12 | 16 | 28 (22, 6) | 0.874 (41) |
| F2@15500 | 15500 | **43** | **37** | 6 | 19 | 18 | 17 | 8 | 12 | 15 | 28 (22, 6) | 0.882 (42) |
| LEX5B@15500 | 15500 | **42** | **36** | 6 | 19 | 17 | 16 | 10 | 10 | 15 | 27 (21, 6) | 0.896 (42) |

Cells answered: HYBRID@6000 60/60 judged 60, HR@6000 60/60 judged 60, LEX5B@6000 60/60 judged 60, HYBRID@15500 60/60 judged 60, HR@15500 60/60 judged 60, F2@15500 60/60 judged 60, LEX5B@15500 60/60 judged 59

## 2. Same-budget comparisons (paired, strict correct)

| comparison | net /54 answerable (b/w) | sign p | net /60 (b/w) | sign p | net new 36 | net saved 24 |
|---|---:|---:|---:|---:|---:|---:|
| LEX5B@6000 vs HYBRID@6000 | +7 (10/3) | 0.092 | +8 (11/3) | 0.057 | +3 | +5 |
| LEX5B@6000 vs HR@6000 | +5 (9/4) | 0.267 | +5 (9/4) | 0.267 | +1 | +4 |
| LEX5B@15500 vs HYBRID@15500 | -2 (3/5) | 0.727 | -2 (3/5) | 0.727 | +2 | -4 |
| LEX5B@15500 vs HR@15500 | -2 (3/5) | 0.727 | -2 (3/5) | 0.727 | -1 | -1 |
| LEX5B@15500 vs F2@15500 | -1 (4/5) | 1.000 | -1 (4/5) | 1.000 | -1 | +0 |
| F2@15500 vs HYBRID@15500 | -1 (6/7) | 1.000 | -1 (6/7) | 1.000 | +3 | -4 |
| F2@15500 vs HR@15500 | -1 (5/6) | 1.000 | -1 (5/6) | 1.000 | +0 | -1 |

Paired lists (60; id without V2-dev-):

- **LEX5B@6000 vs HYBRID@6000**: better on RPT-fa-T2-002, BOOK-en-T2-002, RPT-en-T2-001, BOOK-en-T1-001, BOOK-en-T3-001, RPT-fa-T2-003, RPT-fa-T3-003, RPT-en-T2-003, BOOK-fa-T3-003, BOOK-en-AB-001, LAW-en-T3-001; worse on BOOK-fa-T1-001, LAW-fa-T2-003, LAW-en-T2-002
- **LEX5B@6000 vs HR@6000**: better on RPT-fa-T2-002, RPT-en-T2-002, RPT-en-T2-001, BOOK-en-T1-001, LAW-en-T2-001, RPT-fa-T3-003, RPT-en-T2-003, BOOK-fa-T3-003, BOOK-en-T3-003; worse on RPT-fa-T3-002, BOOK-fa-T1-001, BOOK-en-T2-001, LAW-fa-T3-002
- **LEX5B@15500 vs HYBRID@15500**: better on RPT-en-T2-003, BOOK-fa-T2-003, LAW-fa-T2-003; worse on RPT-en-T2-002, BOOK-en-T2-002, BOOK-fa-T2-001, BOOK-en-T1-001, RPT-fa-T2-003
- **LEX5B@15500 vs HR@15500**: better on LAW-fa-T2-002, LAW-en-T2-001, LAW-fa-T2-003; worse on RPT-en-T2-002, BOOK-en-T2-002, BOOK-en-T3-001, RPT-fa-T2-003, RPT-fa-T3-003
- **LEX5B@15500 vs F2@15500**: better on LAW-fa-T2-002, RPT-en-T2-001, BOOK-en-T2-001, LAW-fa-T3-003; worse on RPT-en-T2-002, BOOK-en-T1-001, RPT-fa-T3-003, RPT-en-T3-003, BOOK-fa-T3-003
- **F2@15500 vs HYBRID@15500**: better on RPT-fa-T3-003, RPT-en-T2-003, RPT-en-T3-003, BOOK-fa-T2-003, BOOK-fa-T3-003, LAW-fa-T2-003; worse on LAW-fa-T2-002, BOOK-en-T2-002, RPT-en-T2-001, BOOK-fa-T2-001, RPT-fa-T2-003, BOOK-en-T2-001, LAW-fa-T3-003
- **F2@15500 vs HR@15500**: better on BOOK-en-T1-001, LAW-en-T2-001, RPT-en-T3-003, BOOK-fa-T3-003, LAW-fa-T2-003; worse on BOOK-en-T2-002, RPT-en-T2-001, BOOK-en-T3-001, RPT-fa-T2-003, BOOK-en-T2-001, LAW-fa-T3-003

## 3. Tokens and latency (36 interleaved questions; the 60-q figures mix sessions and are informative)

| cell | tokens/q (36) | p50 s (36) | p95 s (36) | tokens/q (60) | p50 s (60) | p95 s (60) | $ (36) |
|---|---:|---:|---:|---:|---:|---:|---:|
| HYBRID@6000 | 2169 | 1.75 | 4.15 | 2169 | 1.69 | 3.89 | 0.0120 |
| HR@6000 | 2177 | 2.57 | 10.17 | 2192 | 2.66 | 7.37 | 0.0126 |
| LEX5B@6000 | 4451 | 3.72 | 12.37 | 4397 | 4.04 | 8.75 | 0.0229 |
| HYBRID@15500 | 5135 | 2.41 | 4.90 | 5142 | 2.54 | 5.51 | 0.0270 |
| HR@15500 | 5173 | 3.54 | 7.34 | 5196 | 3.61 | 6.73 | 0.0278 |
| F2@15500 | 5185 | 2.31 | 9.54 | 5171 | 1.90 | 8.04 | 0.0291 |
| LEX5B@15500 | 7350 | 4.85 | 8.43 | 7102 | 4.91 | 8.80 | 0.0381 |

- LEX5B@6000 selector on the 36: 6 fallbacks {"deadline": 6}; view fit p50/p95 45/76 ms (cached fitter); selector call p50/p95 1758/3043 ms; 6 calls with unknown usage (conservative tokens/q 4974)

- LEX5B@15500 selector on the 36: 7 fallbacks {"call_error": 1, "deadline": 6}; view fit p50/p95 44/67 ms (cached fitter); selector call p50/p95 1690/3040 ms; 7 calls with unknown usage (conservative tokens/q 7961)

## 4. Gate check (accuracy +3/54 vs each same-budget control; tokens <= 2.5x; latency p50 and p95 <= 1.5x)

Token interpretations as LEX11: (i) historical absolute cap 2.5 x 2,169 = 5422 tokens/q; (ii) 2.5x same-budget HYBRID on the 36 (@6,000 5423, @15,500 12839); (iii) ratio to the named control.

| arm vs control | net /54 | accuracy +3 | tokens ratio | tokens (i) / (ii) / (iii) | p50 ratio | p95 ratio | latency <= 1.5x |
|---|---:|---|---:|---|---:|---:|---|
| LEX5B@6000 vs HYBRID@6000 | +7 | yes | 2.05 | ok / ok / ok | 2.12 | 2.98 | trade-off (user decision) |
| LEX5B@6000 vs HR@6000 | +5 | yes | 2.05 | ok / ok / ok | 1.44 | 1.22 | ok |
| LEX5B@15500 vs HYBRID@15500 | -2 | no | 1.43 | FAIL / ok / ok | 2.01 | 1.72 | trade-off (user decision) |
| LEX5B@15500 vs HR@15500 | -2 | no | 1.42 | FAIL / ok / ok | 1.37 | 1.15 | ok |
| LEX5B@15500 vs F2@15500 | -1 | no | 1.42 | FAIL / ok / ok | 2.09 | 0.88 | trade-off (user decision) |
| F2@15500 vs HYBRID@15500 | -1 | no | 1.01 | ok / ok / ok | 0.96 | 1.95 | FAIL |
| F2@15500 vs HR@15500 | -1 | no | 1.00 | ok / ok / ok | 0.65 | 1.30 | ok |

## 5. Per question (1 = strict correct; AB = unanswerable)

| id | new | HYBRID@6000 | HR@6000 | LEX5B@6000 | HYBRID@15500 | HR@15500 | F2@15500 | LEX5B@15500 |
|---|---|---|---|---|---|---|---|---|
| RPT-fa-T1-002 |  | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| RPT-fa-T2-002 |  | 0 | 0 | 1 | 1 | 1 | 1 | 1 |
| RPT-fa-T3-002 |  | 0 | 1 | 0 | 1 | 1 | 1 | 1 |
| RPT-en-T1-002 |  | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| RPT-en-T2-002 |  | 1 | 0 | 1 | 1 | 1 | 1 | 0 |
| RPT-en-T3-002 |  | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| BOOK-fa-T1-002 |  | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| LAW-fa-T2-002 |  | 0 | 0 | 0 | 1 | 0 | 0 | 1 |
| BOOK-fa-T3-002 |  | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| LAW-en-T1-002 |  | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| BOOK-en-T2-002 |  | 0 | 1 | 1 | 1 | 1 | 0 | 0 |
| LAW-en-T3-002 |  | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| RPT-fa-T1-001 |  | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| RPT-fa-T2-001 |  | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| RPT-fa-T3-001 |  | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| RPT-en-T1-001 |  | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| RPT-en-T2-001 |  | 0 | 0 | 1 | 1 | 1 | 0 | 1 |
| RPT-en-T3-001 |  | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| BOOK-fa-T2-001 |  | 0 | 0 | 0 | 1 | 0 | 0 | 0 |
| LAW-fa-T1-001 |  | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| LAW-fa-T3-001 |  | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| BOOK-en-T1-001 |  | 0 | 0 | 1 | 1 | 0 | 1 | 0 |
| LAW-en-T2-001 |  | 1 | 0 | 1 | 1 | 0 | 1 | 1 |
| BOOK-en-T3-001 |  | 0 | 1 | 1 | 0 | 1 | 0 | 0 |
| RPT-fa-T1-003 | x | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| RPT-fa-T2-003 | x | 0 | 1 | 1 | 1 | 1 | 0 | 0 |
| RPT-fa-T3-003 | x | 0 | 0 | 1 | 0 | 1 | 1 | 0 |
| RPT-fa-AB-001 | x | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| RPT-en-T1-003 | x | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| RPT-en-T2-003 | x | 0 | 0 | 1 | 0 | 1 | 1 | 1 |
| RPT-en-T3-003 | x | 0 | 0 | 0 | 0 | 0 | 1 | 0 |
| RPT-en-AB-001 | x | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| BOOK-fa-T1-001 | x | 1 | 1 | 0 | 1 | 1 | 1 | 1 |
| BOOK-fa-T1-003 | x | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| BOOK-fa-T2-002 | x | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| BOOK-fa-T2-003 | x | 1 | 1 | 1 | 0 | 1 | 1 | 1 |
| BOOK-fa-T3-001 | x | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| BOOK-fa-T3-003 | x | 0 | 0 | 1 | 0 | 0 | 1 | 0 |
| BOOK-fa-AB-001 | x | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| BOOK-en-T1-002 | x | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| BOOK-en-T1-003 | x | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| BOOK-en-T2-001 | x | 0 | 1 | 0 | 1 | 1 | 0 | 1 |
| BOOK-en-T2-003 | x | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| BOOK-en-T3-002 | x | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| BOOK-en-T3-003 | x | 1 | 0 | 1 | 1 | 1 | 1 | 1 |
| BOOK-en-AB-001 | x | 0 | 1 | 1 | 1 | 1 | 1 | 1 |
| LAW-fa-T1-002 | x | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| LAW-fa-T1-003 | x | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| LAW-fa-T2-001 | x | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| LAW-fa-T2-003 | x | 1 | 0 | 0 | 0 | 0 | 1 | 1 |
| LAW-fa-T3-002 | x | 0 | 1 | 0 | 1 | 1 | 1 | 1 |
| LAW-fa-T3-003 | x | 1 | 1 | 1 | 1 | 1 | 0 | 1 |
| LAW-fa-AB-001 | x | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| LAW-en-T1-001 | x | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| LAW-en-T1-003 | x | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| LAW-en-T2-002 | x | 1 | 0 | 0 | 0 | 0 | 0 | 0 |
| LAW-en-T2-003 | x | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| LAW-en-T3-001 | x | 0 | 1 | 1 | 1 | 1 | 1 | 1 |
| LAW-en-T3-003 | x | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| LAW-en-AB-001 | x | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
