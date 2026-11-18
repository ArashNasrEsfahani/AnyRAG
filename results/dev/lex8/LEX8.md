# LEX8: score-aware fusion, compact source units, coverage packing, synthetic-question field (2026-09-26)

**Scope.** Astra's converged bounded experiment (`work/analysis/LEX7_ASTRA_R1.log`), run on the 54 answerable dev
questions. Metric: lex4 decisive-span retention with source identity in a 6,000-char pack.

- Every arm, constant and admission rule was declared in `PROGRESS.md` before it ran. Two amendments were logged before
  the arm they affect ran: units group short extracted lines, and the synthetic-question parser treats END as optional.
- Baselines were reused, not re-run: HYBRID, HR (HYBRID+rerank), a1, MMR-a1, O1, O2, and the 24-question answers.
- This is development and model selection on a much-inspected dev set, not confirmation. One question moves the mean
  by 0.019.

**Code.** `ragbench/approaches/lex8/` on branch `upgrades` (commits 7dfd899, e5898c4):

- `units.py`, `fusion.py`, `packer.py`, `identity.py`, `index.py`, `retrieve.py`, `synthfield.py`, and `approach.py`
  (approach LEX8, flags `ranked` and `synth`).
- Tests: `ragbench/tests/test_lex8.py`, 53 checks, all pass.

**Harness.** `audit8.py`, `build8.py`, `eval8.py`, `synth8.py`, `reg8.py`, `viz8.py`, and `eval/` (answers8 and
report8, prepared but not run).

**Spend.** $3.487 on OpenRouter: the synthetic questions, including the 43-chunk pilot. Budget file 6.4808 → 9.9678,
inside the $4.50 task cap. The answer screen was not run, so it cost nothing.

## 1. Measurement audit (from saved artifacts only, $0; `audit/AUDIT.md`)

| arm | raw top-12 | raw top-24 | expanded top-24 | raw top-40 | complete bundle present in top-40 | complete bundle that fits 6,000 chars in top-40 | shown pack |
|---|---:|---:|---:|---:|---:|---:|---:|
| HYBRID | 0.7962 (36) | 0.8858 (43) | 0.8858 | 0.9423 (46) | 46 | 46 | 0.7200 (32) |
| HR | 0.8481 (39) | 0.8964 (41) | 0.9057 | 0.9330 (45) | 45 | 45 | 0.8065 (35) |
| a1 | 0.8250 (38) | 0.9117 (44) | 0.9210 | 0.9318 (46) | 46 | 46 | 0.7632 (34) |
| MMR-a1 | 0.8204 (38) | 0.9164 (44) | 0.9256 | 0.9318 (46) | 46 | 46 | 0.7673 (34) |
| O1 / O2 | 0.8204 (38) | 0.9164 (44) | 0.9256 | 0.9318 (46) | 46 | 46 | 0.7735 (35) / 0.7652 (33) |

- **Raw versus expanded coverage.** The two differ on one question only, BOOK-en-T2-003. The old "top-24" column
  (0.9256) used expanded packages; the raw figure is 0.9164. Astra's measurement point is reproduced.
- **Channel union** (BM25 ∪ dense ∪ sparse, top-40 each, 73.5 chunks on average). Raw coverage is 0.9577.
  - A complete bundle is present in 48 questions, and **all 48 fit within 6,000 chars** (fa 26/27, en 22/27).
  - The headroom is therefore selection: every arm shows only 32–35 complete bundles.
- **Excerpt visibility** (fixer window for O1 and O2). The window chunks hold decisive spans 130 times.
  - Only 12 of these (9.2%) lie wholly inside the fixer's excerpt.
  - 24 (18.5%) show at least their key value.
  - The fixer mostly judged without seeing the evidence.
- **Diagnostic fix.** `v71/diag.py` mixed HR's intact pack with raw retention.
  - Chunk 3829 (RPT-fa-T2-003) is in HR's raw pack, as the 7th item.
  - `diag.py` is patched in place. The old `diag_rptfa.json` is kept; the corrected output is
    `audit/diag_rptfa_fixed.json`, which also reports pre-MMR a1 ranks.
  - HR's raw pack, recomputed from the saved order, matches the saved retention exactly.
- **Registry.** 25 lex_research arms were backfilled as `LEXR <name>`.

## 2. Offline arms (54 questions, 6,000-char packs; `TABLES8.md`, `results8.json`)

Cells give mean retention with complete bundles in brackets.

| arm | pack | fa | en | T1 | T2 | T3 | RPT-fa | complete fa / en | better/worse vs HR | better/worse vs HYBRID |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| HYBRID (reused) | 0.7200 (32) | 0.722 | 0.718 | 0.926 | 0.623 | 0.611 | 0.556 | 17 / 15 | 6/13 | – |
| **HR (reused)** | **0.8065 (35)** | 0.877 | 0.736 | 0.981 | 0.748 | 0.690 | 0.889 | 20 / 15 | – | 13/6 |
| a1 (reused) | 0.7632 (34) | 0.772 | 0.755 | 0.981 | 0.669 | 0.639 | 0.611 | 18 / 16 | 6/10 | 4/0 |
| O1 fixer (reused, 945 LLM tokens/q) | 0.7735 (35) | 0.790 | 0.757 | 0.981 | 0.682 | 0.657 | 0.667 | 18 / 17 | 6/9 | 6/1 |
| F1 CombSUM-norm, chunk packer | 0.7462 (31) | 0.790 | 0.702 | 0.926 | 0.669 | 0.644 | 0.667 | 18 / 13 | 6/11 | 6/2 |
| **F2 MaxMean, chunk packer** | **0.7792 (33)** | 0.796 | 0.762 | 0.981 | 0.678 | 0.678 | 0.722 | 17 / 16 | 6/8 | 7/1 |
| C1 compact units + ranked packing | 0.7692 (33) | 0.796 | 0.742 | 0.981 | 0.650 | 0.676 | 0.722 | 18 / 15 | 6/10 | 7/3 |
| C2 compact units + coverage packer (main arm) | 0.7121 (28) | 0.772 | 0.653 | 0.926 | 0.595 | 0.616 | 0.778 | 15 / 13 | 4/12 | 9/12 |
| S1 = C2 + synthetic-question field | 0.7255 (30) | 0.790 | 0.661 | 0.981 | 0.552 | 0.644 | 0.778 | 16 / 14 | 5/12 | 10/10 |
| S1r (post-hoc diagnostic, not eligible) | 0.7456 (30) | 0.784 | 0.707 | 0.981 | 0.593 | 0.662 | 0.722 | 17 / 13 | 5/13 | 7/7 |

**Stage timings** (warm CPU scoring; ms p50 / p95; the query vectors are the saved bge-m3 encodes):

| stage | p50 | p95 |
|---|---:|---:|
| chunk channels | 4.5 | 13 |
| unit channels | 5.4 | 27 |
| candidates | 0.4 | 0.8 |
| ranked pack | 0.5 | 0.9 |
| **coverage pack** | **57** | **672** |
| coverage pack, S1 | 64 | 747 |

**Index time.**

- 102,434 compact units.
- bge-m3 encoding: 6.9 min on the GPU.
- Unit BM25: about 40 s.
- Synthetic questions: 58,218 chunks, $3.48.
  - 141,306 questions accepted: 2.43 per chunk, 97.2% of replies parsed.
  - 97% of questions are linked to units at their quote's location.

## 3. Findings

1. **Fusion earns a little, but not enough.** MaxMean over real channel scores (F2) is the best arm that uses no LLM:
   - 0.7792, or +0.016 over a1, and above O1 without its 945 tokens.
   - It is still 0.027 below HR, with 33 complete bundles against HR's 35.
   - Normalized CombSUM is worse than a1 (−0.017; English 0.702). Summing min-max scores still rewards consensus, as
     Astra predicted.
2. **Compact units are roughly neutral as a representation.** With ordinary ranked packing, C1 scores 0.769, against
   0.779 for F2 on chunks only.
   - RPT-fa improves: 0.722 (C1) and 0.778 (C2), against 0.611 (a1) and 0.667 (O1).
   - English loses.
3. **The coverage packer fails, and the cause is the objective, not a bug.** F = Σu(a)² over displayed units rewards
   volume.
   - Example, RPT-en-T1-003: the decisive unit is the top unit (u = 1.00). The dF/dC greedy instead packs 19
     candidates covering many table-row units at u ≈ 0.6 (F 20.8, against 5.6 for the best single candidate), and
     never packs the decisive unit.
   - The checks confirm the packer computes F and cost exactly (53 checks). The utility is simply mis-specified for
     corpora made of many small rows.
   - This is the risk Astra named: several plausible fragments crowd out one necessary piece.
   - The coverage packer costs 0.057 against ranked packing (C2 vs C1). It is also slow: 672 ms at p95.
4. **Synthetic questions do not help.** Adding the field to C2 gives +0.013 (S1 vs C2).
   - T1 is recovered: 0.981, up from 0.926.
   - T2 falls further: 0.552.
   - RPT-fa is unchanged at 0.778.
   - With ranked packing it is worse than C1 (S1r 0.746 vs 0.769).
   - Paraphrase access through a separate BM25 question field does not move the decisive units into the pack on this
     dev set.

## 4. Admission to the 24-question answer screen (the rule declared in step 5)

The rule required: retention ≥ 0.8065, ≥ 38 complete bundles, no language regression in complete bundles against HR,
and a credible resource margin.

**No arm qualifies.**

- The best retention is 0.7792 (F2), and the most complete bundles is 33 (F2 and C1). Both are far below the bar.
- C2 and S1 also fail the resource filter: coverage-pack p95 is 672–747 ms.
- The same margin that stopped them offline would decide the answer screen. **The answer screen was not run**, so no
  GLM or judge money was spent and no registry answer row exists.
- `eval/answers8.py` and `eval/report8.py` are ready. They reuse HYBRID 12, HR 13 and LEX7.1 13, but carry no result.

## 5. What goes to the true test

**Nothing from LEX8.** No arm passed the development filter. Per Astra's closing rule, we stop claiming that this
line of LEX is approaching superiority over HYBRID+rerank on retention.

Reusable and publishable pieces:

- **Negative results.** Score-normalized CombSUM, the Σu² coverage objective, and a 141k-question synthetic field
  each fail to close a 0.03 gap on this dev set.
- **Diagnosis.** 48/54 questions have a budget-feasible complete bundle among the three channels' candidates; every
  arm shows only 32–35. The gap is selection. Neither bi-encoder scores, their fusion, nor the fixer's 9%-visible
  excerpts identify the decisive passage the way the cross-encoder does.
- **Efficiency.** F2 (0 LLM tokens, about 10 ms of CPU fusion) is the best cheap LEX order. If a prospective secondary
  efficiency claim against HR is wanted, it must be declared before any test, with its comparator and tolerance
  fixed.

**Artifacts.**

- `rows8_fusion.jsonl`, `rows8_compact.jsonl` and `rows8_synth.jsonl`: per-question traces with channel lists, real
  union scores, fusion, u, candidates, packer runs, chosen keys, and pack items with source provenance.
- `viz_lex8.json`: the same per question, in the viz key layout.
- Registry rows `LEX8 F1/F2/C1/C2/S1/S1r` (S1r marked post-hoc).
- `idx/`: unit and chunk m3 artifacts with identity manifests.
- `synth/`: generations, the question field and `COST.json`.
