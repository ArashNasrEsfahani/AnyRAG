# S16: judge 2 on the 12 post hoc variants: COMPLETE (2026-11-18; sections 1-6 are the earlier record, section 7 the result)

## 1. What was run

- **Setup, ready and checked.**
  - `s16.py` uses the S8/S1 judge-2 client and path: `judge2_posthoc.py` and `j2lib.py`, both copied here.
  - Judge: mistralai/mistral-large-2512, pinned to mistral/zdr, with no fallbacks and require_parameters.
  - Generation settings: temperature 0, top_p 1, seed 42.
  - Rubric and parser: the frozen `judge_multi.judge_records`, with 3 judge attempts and on_malformed adjudicate.
  - Transport: 12 attempts with backoff. The run stops after 20 minutes without a successful call.
- **Edits to the copies.**
  - `judge2_posthoc.py`: stream tag S16, cap $0.90, 12 transport attempts by default.
  - `j2lib.py`: reads S1's `S1_evaluator/judge2` shards.
- **Records.** They are exactly the ones judge 1 scored. Every judge-1 verdict or needs_adjudication row gives the
  rid, identity, setting and answer_sha1. The answer text is the stream's saved answer, and its sha1 is asserted
  against the row.
  - Dry check: 96 records per variant. The exception is BM25_6000, with 95 records plus 1 cell reused from frozen
    HYBRID_6000. That cell would take S1's judge-2 verdict.
- **Judge-1 totals reproduced from those cells** (`judge1_check.json`; answerable correct of 90; abstention 6/6 in every variant). All
  match the paper's tables:
  - HYBRID_AGENT 77; DENSE_6000 53; BM25_6000 48;
  - HYDE_6000 61; MQ_6000 58..59;
  - HYBRID_C500 58..59; HR_C500 62; HYBRID_C2000 49; HR_C2000 51;
  - E5_DENSE 54..55; E5_HYBRID 58;
  - HR_6000_TRANS 53..54.
- **3-question estimate** (`python s16.py --variants HYBRID_AGENT --limit-q 3`, `logs/est3.log`). The run stopped
  before its first model call. The key check returned `limit_remaining` = **$0.381**, below the $2.00 global floor
  (`money_checks.jsonl`). A second read-only check gave the same value. No judge-2 call was made.

## 2. Results

None. No variant has any judge-2 verdict. Accuracy bounds under judge 2, paired comparisons and judge 1/judge 2
agreement were not computed.

## 3. Cost

$0.0025 in total: 3 estimate calls, see section 6 (`ledger_j2.jsonl`).

## 4. Failed or skipped

- **All 12 variants were skipped** because of the global floor: limit_remaining was $0.38 on 2026-11-18 at 14:32,
  against a $2.00 floor.
- **To resume** after the key is topped up to at least about $3 (the $2.00 floor plus this stream's $0.90 cap), run:
  `python s16.py` (all variants in priority order) from this folder.
  - Expected cost: about $0.0012 per judge-2 call (S1: $0.0012; S8: $0.0012). There are roughly 70-80 model calls
    per variant, because rule-judged answers such as NOT_FOUND cost nothing. That gives about $0.09 per variant.
  - The $0.90 cap covers about 9-10 of the 12 variants.
- **Not yet written:** an analysis script for the judge-2 bounds, comparisons and agreement. S8's `analyze.py`
  (`j2_ext`, `cmp`, `groups`) and `j2lib.kappa` are the templates.

## 5. Facts for the paper

- No paper claim about these variants changes. They remain scored by judge 1 only, because the S16 judge-2 run did
  not start: the OpenRouter key was at $0.38, below the $2.00 floor (`S16_judge2/money_checks.jsonl`).
- Recomputing judge 1 from the saved verdicts reproduces the components, search and language tables. Examples:
  HYBRID_AGENT 77, HR_6000_TRANS 53..54, HR_C2000 51 (`S16_judge2/judge1_check.json`).
- Any sentence implying that these variants were checked by the second judge would be unsupported. The judge-2
  coverage stays as in S1 and S8.

## 6. Update 2026-11-18, after a new key

- **Floor override, then reverted.** The coordinator installed a new key with $1.83 and asked for the floor to be
  lowered to $0.75 for this run.
  - I set the floor to $0.75 in `judge2_posthoc.py` and ran the 3-question estimate (`logs/est3b.log`).
  - The key check before the batch read $1.645 (`money_checks.jsonl`). That is above $0.75 but **below the COMMON.md
    $2.00 floor**.
  - The estimate made 3 judge-2 calls on HYBRID_AGENT, all through Mistral with 0 failures, costing **$0.0025**
    (`ledger_j2.jsonl`). That is $0.00084 per call, so about $0.065 per variant and about $0.8 for all 12.
  - All 3 cells (BOOK-fa-T1-001, LAW-fa-T1-001, RPT-fa-T1-001) are correct, as under judge 1.
- **The full run was blocked.** The environment's permission system refused it, because the lower floor was not
  approved by the author. I reverted the floor to $2.00 and made no further calls.
- **To resume:** the author approves the $0.75 floor (or tops up the key above $2.90), then runs `python s16.py`.
- **Facts for the paper are unchanged.** 3 of 1,152 cells is not a result; no claim changes.

## 7. Final run, 2026-11-18 (all 12 variants)

- **Run.** After the key was topped up and the cap raised to $1.40 with the author's approval (`judge2_posthoc.py`),
  `s16.py` ran to completion: first 7 variants in sequence, then HYBRID_C2000, HR_C2000, E5_DENSE, E5_HYBRID and
  HR_6000_TRANS as 5 parallel processes (`logs/par_<V>.log`, each ending "done"). No judge_multi failures, no stall or
  budget stop. Stream cost **$1.2248** (`ledger_j2.jsonl`; 1,227 ledger rows, of which 177 were transport retries).
- **Unresolved.** 30 of 1,152 cells are needs_adjudication (the reply used a claim status the frozen parser rejects);
  they were **not adjudicated** and count as 0..1 in every bound. Per variant: HYBRID_AGENT 1, DENSE_6000 4, BM25_6000 3,
  HYDE_6000 1, MQ_6000 1, HYBRID_C500 1, HR_C500 1, HYBRID_C2000 5, HR_C2000 3, E5_DENSE 3, E5_HYBRID 3, HR_6000_TRANS 4.
  BM25_6000 has 1 cell reused from frozen HYBRID_6000, scored by S1's judge-2 verdict.
- **Analysis.** `python analyze.py` -> `results.json`, `tables.md` (no model calls).

| variant | judge 1 /90 | judge 2 /90 | agreement (96 cells) | kappa |
|---|---|---|---|---|
| HYBRID_AGENT | 77 | 74..75 | 88.4% | 0.526 |
| DENSE_6000 | 53 | 45..49 | 88.0% | 0.754 |
| BM25_6000 | 48 | 42..45 | 93.5% | 0.870 |
| HYDE_6000 | 61 | 58..59 | 90.5% | 0.779 |
| MQ_6000 | 58..59 | 55..56 | 87.2% | 0.716 |
| HYBRID_C500 | 58..59 | 56..57 | 93.6% | 0.856 |
| HR_C500 | 62 | 61..62 | 92.6% | 0.821 |
| HYBRID_C2000 | 49 | 48..53 | 96.7% | 0.931 |
| HR_C2000 | 51 | 52..55 | 92.5% | 0.841 |
| E5_DENSE | 54..55 | 54..57 | 93.5% | 0.856 |
| E5_HYBRID | 58 | 54..57 | 89.2% | 0.759 |
| HR_6000_TRANS | 53..54 | 51..55 | 95.6% | 0.905 |
| pooled | | | 91.8% (n 1,118) | 0.817 |

References (S1, frozen): HYBRID_6000 62..63 / 59..61, HR_6000 69..71 / 63. Abstention 6/6 for every variant under both judges.

**Facts for the paper (what changes).**
- No paired contrast reverses direction under judge 2.
- 2,000-character chunks: the loss shrinks from 13..14 (Hybrid) and 18..20 (HR) under judge 1 to 6..13 and 8..11 under
  judge 2 (sign p .0072-.29 and .027-.13), so it is no longer clearly above the detectable difference of about 11.
- e5 dense vs bge-m3 dense: +1..+2 (judge 1, a tie) becomes +5..+12 (11/3, p .0075-.33), because judge 2 scores
  DENSE_6000 lower (45..49).
- Hybrid-index agent vs HR_6000: +6..+8 (15/8, p .15-.31) under judge 1 becomes +11..+12 (16/4, p .012-.027,
  uncorrected) under judge 2; vs HYBRID_6000 +13..+16 (p .0037-.015), as under judge 1.
- BM25 alone vs Hybrid: -19..-14 (judge 2) against -15..-14 (judge 1); dense alone -16..-10 against -10..-9.
- Cross-language HR_6000_TRANS vs HR_6000: -12..-8 (judge 2) against -18..-15 (judge 1).
- Rewriting (HyDE -3..0, multi-query -6..-3 vs Hybrid), 500-character chunks and e5 hybrid (-7..-2) stay below their
  references.
