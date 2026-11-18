# S7 "analysis": where answers are lost, contamination, tokens, errors, Persian channels, loose ends

Post-hoc and exploratory, from saved data only. **No API calls, no GPU** (`CUDA_VISIBLE_DEVICES=""`; BM25 on CPU only).
Inputs: frozen test records/packs/verdicts (`work/datasets_v2/test_run`, read through `report_test.load/score`), the saved
packs, answers and verdicts of S1 (judge 2), S2a (FUSION, GOLD, budget sweep), S3 (agents), S6 (CLOSEDBOOK, LISTWISE,
SETR_NOFILL), and the frozen question file `test_freeze/questions/test_v2judge_all.jsonl` (meta.v2: evidence offsets and
quotes, bundles, subtype). Statistics: the `revision_stats.py` functions (`compare`, `sign_range`, `boot_mean`, ...)
extracted with `ast` and run unchanged, as S2a does (10,000 resamples, seed 0, questions and 20 components). Every
per-arm total reproduces the owning stream's number (HYBRID_6000 62..63, HR_6000 69..71, LEX5B_6000 68..69, SETR_6000 63,
FUSION_6000 58..59, SETR_NOFILL 63, LISTWISE 66..67, HYBRID_15500 68, HR_15500 69..70, ITER_BM25 73, GREP_AGENT 68,
GOLD 83, CLOSEDBOOK 26). Scores are machine strict verdicts of the pinned judge on the 90 answerable questions; `a..b` =
bounds over unresolved cells.

Code: `s7lib.py` (loaders, matcher, statistics), `t1_reach.py`, `t2_contam.py`, `t3_tokens.py`, `t4a_dump.py`,
`t4_errors.py`, `t4c_examples.py`, `t5_channels.py`, `t6_numbers.py`. Each writes a `.json` and a `*_tables.md`.

## 1. Where the answer is lost (`reach.json`, `reach_tables.md`, `fig_reach_{arms,tier,lang}.dat`)

**Method.** For each (arm, question) the exact reader user message was parsed into its numbered items
(`[n] <doc title> > <section>`). A gold evidence item *reached the reader* if at least 90% of its normalised quote
(`ragbench.normalize`) occurs verbatim, in segments of at least 12 characters, inside pack items whose header names the
item's own document (so a matching string in another edition or document does not count). A bundle is complete if all its
items reached; a question is **complete** if any annotated bundle is complete, **partial** if some item of the best bundle
reached, **none** otherwise. Checks: GOLD packs come out complete on 90/90; against the record's `evidence_units` (gold
unit id delivered) the text rule agrees on 241-243 of 248 items for the six frozen arms checked.

**Result (90 answerable; miss = resolved strict 0).**

| arm | correct | reach C / P / N | miss, evidence complete | miss, partial | miss, never reached | correct without a complete bundle | unres. |
|---|---|---|---|---|---|---|---|
| HYBRID_6000 | 62 | 57 / 26 / 7 | 3 | 19 | 5 | 8 | 1 |
| HR_6000 | 69 | 63 / 23 / 4 | 3 | 14 | 2 | 9 | 2 |
| LEX5B_6000 | 68 | 64 / 23 / 3 | 3 | 16 | 2 | 8 | 1 |
| SETR_6000 | 63 | 63 / 24 / 3 | 8 | 18 | 1 | 8 | 0 |
| FUSION_6000 | 58 | 53 / 31 / 6 | 2 | 26 | 3 | 8 | 1 |
| SETR_NOFILL | 63 | 62 / 25 / 3 | 9 | 17 | 1 | 10 | 0 |
| LISTWISE | 66 | 61 / 25 / 4 | 4 | 16 | 3 | 10 | 1 |
| HYBRID_15500 | 68 | 65 / 22 / 3 | 6 | 15 | 1 | 9 | 0 |
| HR_15500 | 69 | 70 / 18 / 2 | 6 | 13 | 1 | 6 | 1 |
| ITER_BM25 | 73 | 71 / 14 / 5 | 7 | 6 | 4 | 9 | 0 |
| GREP_AGENT | 68 | 60 / 23 / 7 | 4 | 14 | 4 | 12 | 0 |
| GOLD (reference) | 83 | 90 / 0 / 0 | 7 | 0 | 0 | 0 | 0 |

- **Most misses are partial bundles.** At 6,000 characters 14-26 of each arm's misses had part of the needed evidence and
  1-5 had none of it. Only 2-9 misses had a complete bundle.
- **Complete-evidence misses are mostly judge misses.** In the four frozen 6,000 arms, every complete-evidence miss that the
  human reviewed was accepted by the human: HYBRID 3/3, HR 3/3, LEX5B 2/2 (1 not reviewed), SETR 8/8. Among reviewed
  partial-evidence misses the human accepted HYBRID 3/11, HR 2/6, LEX5B 5/8, SETR 7/10 (`reach.json` human_on_misses).
  SETR's 8 complete-evidence misses are the judge strictness S1 found on SETR.
- **By tier** (complete bundles reached, of 30): T1 27-29 for every arm; T2 13 (FUSION) to 24 (HR_15500); T3 12-17 for the
  single-pass arms and 21 for ITER_BM25. T3 misses are almost all partial: 9-12 per 6,000 arm (`fig_reach_tier.dat`).
- **By language** (complete, of 45): fa 28 (GREP_AGENT) to 38, en 22-35. English packs miss more: HYBRID_6000 fa 33 / en 24, FUSION fa 31 /
  en 22 (`fig_reach_lang.dat`).
- **The selector's effect is visible here.** LEX5B delivers a complete bundle on 64 questions and FUSION (same candidates, no
  selector) on 53. On the 12 questions LEX5B wins against FUSION, 7 go from partial to complete (task 4b).
- No 6,000 retrieval arm delivered a complete bundle on 22 questions (T3 12, T2 9, T1 1; en 13, fa 9). At least one of
  the 11 arms did on 84 of 90.
- Figure data: `fig_reach_arms.dat` has one row per arm, with columns correct_{complete,partial,none},
  miss_{complete,partial,none}, unresolved and reach_*. The tier and language files have the same columns plus a group
  column.

## 2. Contamination sensitivity (`contam.json`, `contam_tables.md`)

The 26 closed-book-correct questions (S6 CLOSEDBOOK, machine strict 1) are BOOK-en 9, LAW-en 11, BOOK-fa 3, LAW-fa 3 and RPT
0. By tier they are T1 9, T2 8, T3 9; by language en 20, fa 6.

| contrast | all 90 | without the 26 (n=64) | closed-book-correct only (n=26) |
|---|---|---|---|
| LEX5B_6000 vs HYBRID_6000 | +5..+7 (11/4), p .118-.332 | **+6..+8 (10/2), p .039-.180**, CI q [-1.6, 21.9] | -1 (1/2) |
| LEX5B_6000 vs HR_6000 | -3..+0 (6/7), p .61-1 | +0..+2 (6/4), p .75-1 | -3..-2 (0/3) |
| LEX5B_6000 vs FUSION_6000 | +9..+11 (12/2), p .007-.035 | +7..+9 (9/1), p .012-.065, CI q [1.6, 23.4] | +2 (3/1) |
| HR_6000 vs HYBRID_6000 | +6..+9 (10/4), p .049-.180 | +5..+7 (8/3), p .092-.227 | +1..+2 |
| HYBRID_15500 vs HYBRID_6000 | +5..+6 (10/4), p .18-.30 | +4..+5 (8/3), p .23-.39 | +1 |
| ITER_BM25 vs HR_6000 | +2..+4 (11/8), p .50-.82 | **-1..+0 (7/7), p 1.0** | +3..+4 (4/1) |
| GOLD vs HR_6000 | +12..+14 (18/5), p .007-.023 | +7..+8 (13/5), p .096-.167 | +5..+6 (5/0) |
| GOLD vs LEX5B_6000 | +14..+15 (17/3), p .001-.003 | +6..+7 (9/3), p .092-.146 | +8 (8/0) |
| GOLD vs HYBRID_15500 | +15 (18/3), p .001 | +9 (12/3), p .035 | +6 (6/0) |

- **The pipeline contrasts do not depend on the memorisable questions.** LEX5B's lead over HYBRID and FUSION is the same size
  or larger without them.
- **ITER_BM25's edge over HR comes from them.** Its +2..+4 over HR_6000 lies entirely in the closed-book-correct questions
  (+3..+4); on the other 64 it is -1..0.
- **About half of the GOLD gap is on closed-book-correct questions.** Of GOLD's +14..+15 over LEX5B, +8 is on the 26
  (8 better, 0 worse) and +6..+7 on the other 64. On those 26 the retrieval arms lose answers that the reader could give
  from memory, or from gold evidence.
- **By pool** (15 questions each; full table in `contam_tables.md`), no single-pool contrast is significant. LEX5B vs HYBRID
  is +4 on RPT-en (4/0), +2 on BOOK-fa and LAW-en, and -1..-2 on BOOK-en and LAW-fa. HR vs HYBRID is +3..+4 on LAW-en and
  RPT-en and -2 on LAW-fa.

## 3. Tokens, token-matched view, the 1.66 ratio, SETR_NOFILL vs 3,000 (`tokens.json`, `tokens_tables.md`, `fig_tokens_acc.dat`)

Answer-call prompt tokens are taken from the provider-reported usage in the frozen ledgers and in the S2a, S3 and S6
ledgers. Means are over 48 questions per language.

| pipeline | budget | fa tokens | en tokens | fa acc /45 | en acc /45 |
|---|---|---|---|---|---|
| HYBRID | 3,000 / 6,000 / 10,000 / 15,500 / 24,000 | 1,416 / 2,655 / 4,265 / 6,527 / 9,979 | 857 / 1,531 / 2,457 / 3,683 / 5,607 | 28..29 / 35..36 / 33 / 35 / 37 | 21..22 / 27 / 30..31 / 33 / 35 |
| HR | same | 1,421 / 2,669 / 4,308 / 6,560 / 10,024 | 850 / 1,553 / 2,490 / 3,751 / 5,672 | 29 / 36 / 37 / 38 / 33 | 27 / 33..35 / 33 / 31..32 / 33 |
| FUSION | same | 1,354 / 2,584 / 4,213 / 6,471 / 9,922 | 822 / 1,476 / 2,385 / 3,631 / 5,551 | 23..24 / 31..32 / 36 / 34 / 35 | 20 / 27 / 30 / 31 / 35 |
| LEX5B | 6,000 / 15,500 | 2,570 / 6,463 | 1,507 / 3,629 | 36..37 / 36 | 32 / 31 |
| SETR_NOFILL | ~3,056 chars | 1,449 | 892 | 31 | 32 |

**The ratio, four ways** (fa/en):

| measure | fa/en |
|---|---|
| the paper's 1.66 (`work/datasets_v2/lex10/calib10.json`): pooled tokens/char of whole answer prompts (system + evidence + question), 36 dev answer calls per language, 3 arms | 1.657 |
| same rule on all 2,106 test retrieval answer calls | 1.754 |
| S4's 1.81 (`S4_route_long/pool_stats.json`): median of per-call whole-prompt tokens/char, 384 frozen test calls per language; over the 2,106 calls here 1.78 | 1.806 |
| **marginal tokens per evidence character** (OLS, tokens ~ 1 + evidence chars + system chars): fa 0.409, en 0.227 | **1.797** |
| S4 1.94-1.97: median tokens/char of ~30k-token document-only long prompts (LC arms) | 1.94-1.97 |
| S2a "≈1.7": ratio of mean prompt tokens at the same character budget; it rises with budget | 1.65 (3,000) to 1.79 (24,000) |

- **What the numbers measure.** They differ because the fixed, English, ~200-token system prompt dilutes the whole-prompt
  ratio, more so for small packs and for the dev sample's packs. 1.66 is a small dev sample.
- **Recommendation.** For "a character budget is not a token budget" the paper should use the marginal evidence rate:
  **Persian evidence costs about 1.8× as many reader tokens per character (0.41 vs 0.23)**. A whole 6,000-character answer
  prompt costs about 1.7× (2,655 vs 1,531 tokens for HYBRID).
- **Token-matched view.** Persian at budget B costs about as many tokens as English at the next budget up: fa 3,000 ≈ en 6,000
  (~1,420 vs ~1,530 tokens); fa 6,000 ≈ en 10,000 (~2,650 vs ~2,460). Mixed cells with the Persian half at B and the
  English half at the next budget, compared with B for both:
  - HYBRID: +4..+7 at 3,000 (55..56 vs 49..51, 7/1, p .04-.34); +2..+5 at 6,000 (65..67, p .13-.69).
  - HR: +6..+8 at 3,000 (62..64 vs 56, p .02-.07); -2..0 at 6,000 (69).
  - FUSION: +6..+8 at 3,000 (p .02-.11); +2..+4 at 6,000.

  So at equal reader tokens English gets more evidence. The gain matters only at the smallest budgets, and for HR it is gone
  from 6,000 characters on.
- **SETR_NOFILL vs the smallest packs** (SETR_NOFILL ≈ 3,056 chars, 1,449 / 892 answer tokens fa / en, plus SETR's saved
  selector call):
  - vs HYBRID_3000: **+12..+14 (16/4), sign p .004-.012**, CI q [4.4, 25.6], comp [4.7, 25.4]. English +10..+11 (p .003-.006),
    Persian +2..+3.
  - vs HR_3000: +7 (14/7), p .189, CI q [-2.2, 17.8], comp [0.0, 15.7].
  - vs FUSION_3000: +19..+20 (25/6), p .001.
  - vs HYBRID_6000: +0..+1. vs SETR_6000: 0.

## 4. Error analysis (`errors.json`, `errors_tables.md`, `j12_disagreements.json`, `examples.{json,md}`)

**(a) Judge 1 vs judge 2 disagreements**, GLM reader, the 8 frozen cells: 66 cells on 29 questions. Judge 2 was the stricter
one in 43 and judge 1 in 23. The causes were labelled by reading both judges' saved units, claims and reasons
(`j12_disagreements_dump.md`); there is one label per cell, listed in `t4_errors.py`.

| cause of the stricter judge's 0 | cells | J1 / J2 stricter | questions | fa / en |
|---|---|---|---|---|
| extra_unsupported: every required unit correct under both judges, but an added (mostly true, in-pack) detail is failed as "unsupported" | 30 | 9 / 21 | 13 | 18 / 12 |
| omission_disputed: a required unit only implied or partly stated | 12 | 2 / 10 | 5 | 3 / 9 |
| judge_misread: the unit is plainly in the answer | 6 | 2 / 4 | 5 | 5 / 1 |
| synonym_fa: Persian "کلیمیان" (Kalimian) not recognised as "Jews" | 7 | 7 / 0 | 1 | 7 / 0 |
| numeric_tolerance: "9.2%" against "about 9%" | 4 | 1 / 3 | 1 | 4 / 0 |
| substantive_error: a real defect that only one judge caught | 7 | 2 / 5 | 7 | 2 / 5 |

Only 7 of the 66 disagreements (11%) are about a real error in the answer. 30 of them (45%) come from the strict rule that
fails an answer for one unsupported added claim. The judges see only the annotated excerpts, so true details from the
reader's pack count as unsupported (example 4).

**(b) Lex-RAG wins and losses at 6,000 by integration operation** (subtype; machine strict; W/L)

| operation | n | vs HYBRID (all / fa / en) | vs FUSION (all / fa / en) | vs HR (all) |
|---|---|---|---|---|
| T1 fact / table / list | 30 | 2/1 · 1/1 · 1/0 | 0/0 | 1/0 |
| T2 comparison/derivation | 13 | 4/1 (+1 unres.) · 3/0 · 1/1 | 5/0 (+1) · 4/0 · 1/0 | 1/1 (+1) |
| T2 rule + exception | 7 | 2/2 · 0/1 · 2/1 | 3/1 · 0/0 · 3/1 | 1/1 (+1) |
| T2 definition bridge | 8 | 0/0 (+1) | 1/0 · 1/0 · 0/0 | 1/1 |
| T2 bounded synthesis | 2 | 1/0 · 0/0 · 1/0 | 1/0 · 0/0 · 1/0 | 0/0 |
| T3 cross-time/version | 11 | 2/0 · 0/0 · 2/0 | 2/0 · 0/0 · 2/0 | 2/0 |
| T3 cross-subject | 9 | 0/0 | 0/1 · 0/0 · 0/1 | 0/3 (+1) |
| T3 cross-doc bridge | 7 | 0/0 | 0/0 | 0/1 |
| T3 bounded aggregation | 3 | 0/0 | 0/0 | 0/0 |
| all | 90 | 11/4 (+2) · fa 4/2 · en 7/2 | 12/2 (+1) · fa 5/0 · en 7/2 | 6/7 (+3) |

- **Where Lex-RAG gains.** Its gains over HYBRID and FUSION are in T2 within-document integration (comparison and
  derivation, rule plus exception) and in T3 cross-time/version questions, in both languages. Against FUSION, Persian is
  5/0, all in T2.
- **Where it loses.** Against HR it loses the T3 cross-subject questions (0/3, all English).
- **Reach on the decisive questions vs FUSION.** 7 of the 12 wins go from a partial FUSION pack to a complete Lex-RAG pack,
  4 are partial in both, and 1 is complete in both (`reach_transitions`). All 14 decisive packs came from a successful
  selector call, not the fallback.

**(c) Four examples** (`examples.md`: exact text, selector replies, pack item headers, verdicts and reasons; English
glosses of the Persian are this stream's own):
1. **RPT-fa-T2-005** (fa, selector rescue). The question asks for the full-time drivers' monthly income and the definition
   of full-time. The selector reply is `P1: 3 / P2: 9`, and it puts the 1403 glossary item ("کاربری که بیشتر از 40 سفر موفق
   در ماه انجام دهد", "a user who completes more than 40 successful trips a month") second. The FUSION pack never reaches
   that item. FUSION: income only, judged 0. Lex-RAG: income and definition, judged 1.
2. **LAW-en-T2-004** (en, selector rescue). The question is about the DMA notification deadline and the fine. FUSION has
   Art. 3(3) but not Art. 30(3)(b), and answers the fine part with NOT_FOUND (judged 0). The selector picks `P2: 15+21`
   (Art. 30(3) + point (b)). Correct under judge 1, judge 2 and the human.
3. **LAW-fa-T1-002** (fa, judge disagreement). All 8 GLM cells list the 5 minority seats and use "کلیمیان" (Kalimian, the
   Persian term for Jews). Judge 1 marks "Jews" missing in 7 of the 8 cells. Judge 2 and the human accept the answer.
4. **RPT-en-T2-002** (en, judge disagreement). The answer's extra sentence "Profitability lagged in the first half of 2024
   ..." is verbatim in the reader's pack, right after gold quote E1. Judge 2 fails it as unsupported in 6 of the 8 cells;
   judge 1 passes 7 of the 8.

## 5. Persian retrieval by channel (`channels.json`, `channels_tables.md`, `fig_channels.dat`)

- **BM25 is exact.** The BM25 top 40 was recomputed on CPU with the frozen index and tokenizer, and matches all 2,737 saved
  per-channel BM25 ranks.
- **Dense is a lower bound.** The dense top 40 is known only through the saved fused lists (HYBRID top 40, HR top 50). It is
  complete to a median depth of 27-28, and 405 (fa) and 423 (en) of the 1,800 dense slots per language are unknown.
- **The learned-sparse channel needs GPU.** Its lists were never saved (LEX5B keeps only the fused F2 top 40), and
  recomputing them needs a bge-m3 query encoding.

| gold recall@40 (bundle C/P/N; items) | fa | en |
|---|---|---|
| BM25 (exact) | 37/6/2; 85/106 | 29/14/2; 88/124 |
| bge-m3 dense (known part; lower bound) | 37/7/1; 95/106 | 31/13/1; 96/124 |
| BM25 ∪ dense | 39/6/0; 99/106 | 37/8/0; 105/124 |
| HYBRID RRF top 40 | 39/6/0; 98/106 | 33/11/1; 98/124 |
| Lex-RAG F2 top 40 (BM25 + dense + sparse) | 40/5/0; 100/106 | 35/9/1; 101/124 |

- **Recall by channel.** In Persian, BM25 and dense each find a complete bundle for 37 of 45 questions. Dense finds more
  gold items (95 vs 85), and their union finds 39. Persian recall@40 is higher than English for every channel.
- **Gold items found by one channel only:** fa dense-only 14, BM25-only 4, both 81, neither 7 (3 of which F2 finds, i.e.
  sparse or deep dense); en dense-only 17, BM25-only 9, neither 19.
- **What the reader received.** The gold items that reached it were carried by chunks found by both channels in most
  cases. Dense-only chunks matter for HR, where the cross-encoder lifts dense-only chunks: HR_6000 fa 12 dense-only vs 3
  BM25-only. HYBRID_6000 delivered no single-channel gold item in Persian. LEX5B_6000 fa: 85 both, 2 dense-only, 1
  BM25-only, 0 F2-only.
- **BM25 normalisation ablation** (recall only):

  | variant | fa complete / items | en complete / items |
  |---|---|---|
  | frozen tokenizer | 37 / 85 | 29 / 88 |
  | raw query on the frozen index | 35 / 80 | 27 / 86 |
  | plain Unicode tokenizer on both sides | 35 / 81 | 27 / 81 |

  Normalisation is worth about 2 complete bundles and 4-5 gold items in Persian, and about as much in English (where
  stemming, stopwords and number variants are lost). 27 of the 45 Persian questions contain Persian digits, which only the
  normalised path matches against the ASCII digits of the corpus. The corpus text was already normalised at extraction,
  so this ablation removes only the tokenizer-level normalisation.

## 6. Numeric loose ends (`numbers.json`, `numbers_tables.md`)

1. **"15 of the 27 human-accepted SetR answers" (App. F) is correct.** Machine × human on the 30 reviewed SETR cells: m1h1
   12, m0h1 15, m1h0 0, m0h0 3, no unresolved cell. The human accepted 27, and the judge rejected 15 of them.
   - The other pipelines (accepted / judge-rejected): HYBRID 20 / 9 (1 unresolved h0); HR 25 / 6, of which 24 are
     machine-resolved; LEX5B 25 / 7, of which 24 are machine-resolved. So the "20 to 24" in App. F counts machine-resolved
     cells.
   - S1's "26" counts only the 29 SETR cells that judge 2 could resolve. Source: `test_results_human_sym.json`
     per_question_glm_6000_sym × `report_test.score`.
   - Suggested fix: print this 2×2 per pipeline.
2. **Dev gold-only T2 12/18 vs test GOLD T2 30/30 are different measurements.**
   - Dev (`datasets_v2/diagnosis`, 2026-09-26): different questions; the pack held every annotated gold *unit* (whole
     units); the judge was gpt-oss-120b **without a provider pin** (no pin, no served provider recorded). T2 is 12/18, with
     17 judged; RPT-fa-T2-003 was unjudged (malformed judge reply) and counted wrong. T3 is 17/18.
   - The 5 judged dev T2 failures are BOOK-fa-T2-001 (the "sword" qualifier), LAW-en-T2-003 (article number cited as
     19(4)), LAW-fa-T2-003 (exemption inverted, a suspected question defect), RPT-en-T2-002 (one unsupported extra claim)
     and RPT-fa-T2-001 (derived value omitted).
   - Test (S2a GOLD): the first annotated bundle as quote slices, scored by the pinned judge: T1 27, T2 30, T3 26 /30.
   - Of the 6 dev T2 non-passes, one is unjudged, one (RPT-en-T2-002) is the one-unsupported-extra-claim rule of task 4a
     and one (LAW-fa-T2-003) is a suspected question defect. So 12/18 is not evidence that a third of T2 questions are
     unanswerable from their gold.
3. **Table 13 mixes two judge settings.** The dev machine HYBRID 30, HR 32 and LEX5B 37 come from `dev60/judge`: 251 rows, no
   provider pin, 2026-09-27. SETR 40..42 comes from `setr/judge`: 58 rows, pinned to deepinfra/bf16 (49 served by DeepInfra,
   9 rule verdicts), 2026-10-01. No pinned dev verdicts exist for HYBRID/HR/LEX5B, so the column cannot be put on one judge
   without new judge calls (about 162 calls, under $0.03 at the ledger price). Otherwise drop the SETR dev cell, or keep the
   caption's warning.

## 7. Cost
$0.00. No API calls, no GPU. CPU only: BM25 rebuilds for the ablation, plus the text matching.

## 8. Failed, skipped, limits
- **Learned-sparse channel:** not analysable from saved data; this **needs GPU** (bge-m3 sparse query encoding). F2-only
  items are reported as "sparse or deep dense".
- **Dense channel:** known only in part, so dense recall@40 is a lower bound (see §5).
- **Reach is a text criterion.**
  - The 90% / 12-char / same-document rule can miss evidence that is paraphrased in a table rendering, and it does not
    credit alternative evidence outside the annotated bundles. That is why 6-12 answers per arm are correct without a
    complete bundle.
  - It agrees with the unit-id criterion on 97-98% of items.
- **The task-4a cause labels are one analyst's reading,** with no second annotator. The cell-level labels are in
  `t4_errors.py`, so they can be re-coded.
- **Integration operation:** the `integration_operation` field is free text (null for T1), so the `subtype` field was used
  as the operation class.
- **Human checks:** only the 30 symmetric-review questions have human verdicts, and they cover the four frozen 6,000 arms
  only.
- **Table 13:** the mixed judges were not resolved. That needs judge calls, which are out of scope for this stream.

## 9. Facts for the paper
1. At 6,000 characters most misses are partial-evidence misses: 14-26 per arm had some but not all of a gold bundle in the
   pack, against 1-5 with none. Every human-reviewed miss that had a complete gold bundle in the pack (16 of 16 across the
   four frozen arms) was a judge error (`reach.json`, arms and human_on_misses).
2. On the same 40 candidates, the selector raises complete-bundle delivery from 53 (FUSION) to 64 (Lex-RAG) of 90. 7 of its
   12 wins over FUSION are partial → complete, all in T2 or T3, and Persian is 5/0 (`reach.json`, `errors.json`
   b_wins_losses).
3. Without the 26 closed-book-correct questions, LEX5B vs HYBRID is +6..+8 (10/2) and vs FUSION +7..+9. ITER_BM25's edge over
   HR_6000 vanishes (−1..0), and GOLD's lead over LEX5B shrinks from +14..+15 to +6..+7 (`contam.json`).
4. Per evidence character, Persian costs 1.80× the reader tokens of English on test (0.409 vs 0.227, marginal OLS). The
   paper's 1.66 is a whole-prompt ratio from 36+36 dev calls (`lex10/calib10.json`), diluted by the English system prompt.
   SETR_NOFILL (~3,056 chars) beats HYBRID_3000 by +12..+14 (p .004-.012) but not HR_3000 (+7, p .19) (`tokens.json`).
5. Of the 66 judge-1/judge-2 disagreements, only 7 involve a real error in the answer. 30 come from the rule that fails an
   answer for one "unsupported" added claim, often a true sentence from the pack. 7 come from judge 1 not recognising the
   Persian کلیمیان ("Jews") on one question (`errors.json`).
6. In Persian, BM25 and bge-m3 dense each reach a complete gold bundle in the top 40 for 37/45 questions, and their union
   for 39/45. Dense finds 14 gold items that BM25 misses, against 4 the other way. Removing the tokenizer normalisation
   costs BM25 2 Persian bundles (37 → 35) (`channels.json`). The App. F "15 of 27" is correct: 27 SetR cells were
   human-accepted, 15 of them machine-rejected (`numbers.json`).
