# LEX9: a tiny ranker learned from synthetic questions (2026-09-26)

**Idea.** Learn chunk ranking from the 141k grounded synthetic questions produced in LEX8. The method uses no dev or
test labels, no cross-encoder, and no LLM tokens at query time. Nothing in this task called an LLM, and it cost $0.

**Declarations.** Everything was declared in `PROGRESS.md` before any computation:

- the sample and the document split;
- the leakage control, labels, features and both models;
- the selection rule and the dev protocol.

Only the selected model was run on dev.

**Code.**

- Package: `ragbench/approaches/lex9/` on branch `upgrades` (commit 8ee0481).
  - `features.py`: four channels and 41 features.
  - `model.py`: the frozen scorer and the ordering rule.
- Checks: `ragbench/tests/test_lex9.py`, 4 checks, all pass. They cover self-question exclusion, number folding,
  ties, and the frozen scorer.
- Harness: `data9.py`, `train9.py`, `eval9.py`.

## 1. Training data

- **Sample.** 3,000 questions per pool, 18,000 in all. Sampling was round-robin across documents.
- **Validation.** 20% of each pool's documents, chosen by sha1 order, are held out: 3,724 validation questions.
- **Quotes.** Quotes were recovered exactly: 0 mismatches against `questions.jsonl`. Each quote was located in its
  source chunk text for 98.3% of questions.
- **Encoding.** bge-m3 encoded the queries on the GPU in 14 s (normalize, max length 128, the same snapshot as the
  dev queries).
- **Channels.** BM25, bge-m3 dense, bge-m3 sparse, and the synthetic-question BM25 field lifted to chunks (a chunk's
  score is the maximum over its own questions). Each channel returns its top 40.
- **Leakage control.** The query's own question is removed from the field.
  - Smoke test: the source chunk's synth z drops from 1.00 to between 0.07 and 1.00. Siblings keep it high, and that
    is realistic.
- **Labels.** 2 when the chunk contains the whole quote location, 1 for a partial overlap, 0 otherwise.
- **Union recall.** 0.98–0.998 per pool. About 1.9M candidate rows.

## 2. Models (validation documents, nDCG@10)

| scorer | nDCG@10 |
|---|---:|
| **LightGBM lambdarank (selected)** | **0.8565** |
| logistic regression | 0.8146 |
| F2 MaxMean (unlearned reference) | 0.8096 |
| MaxMean over 4 channels | 0.7343 |
| dense only | 0.6342 |

**Where LightGBM's gain comes from.**

| feature | share of gain |
|---|---:|
| maxmean3 (= F2) | 0.66 |
| maxmean4 | 0.16 |
| adj_max | 0.04 |
| each of the rest | ≤ 0.01 |

The ranker mostly learned to trust F2.

## 3. Dev (54 answerable questions, unchanged chunk packer, 6,000 chars)

**Sanity check.** F2, recomputed through this harness, equals the saved LEX8 F2 exactly (maximum difference 0.0000,
0.7792 with 33 complete bundles).

| arm | pack (complete) | fa | en | T1 | T2 | T3 | RPT-fa | complete fa / en | T3 all docs /18 | better/worse vs HR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| HYBRID | 0.7200 (32) | 0.722 | 0.718 | 0.926 | 0.623 | 0.611 | 0.556 | 17 / 15 | 12 | 6/13 |
| HR | **0.8065 (35)** | 0.877 | 0.736 | 0.981 | 0.748 | 0.690 | 0.889 | 20 / 15 | 13 | – |
| F2 | 0.7792 (33) | 0.796 | 0.762 | 0.981 | 0.678 | 0.678 | 0.722 | 17 / 16 | 13 | 6/8 |
| **LEX9 (LightGBM)** | **0.7611 (32)** | 0.778 | 0.744 | 0.981 | 0.612 | 0.690 | 0.667 | 18 / 14 | 14 | 6/10 |

LEX9 is 7/3 against HYBRID.

**The selection gap is not closed.**

- A complete, budget-feasible bundle exists in 48/54 questions (the 3-channel union), and in 49 once the synth
  channel is added.
- LEX9's top 40 contains one for 46 questions, the same as F2.
- LEX9's shown pack completes 32 of the 48. F2 completes 33 and HR 35.
- Bundle chunks sit at median rank 2 under both orders. The share inside the top 10 is 0.767 for LEX9 and 0.778 for
  F2.
- Packs hold only about 6.6 chunks, so the misses are bundle chunks ranked 7th and lower.

**Latency** (ms, warm CPU, p50 / p95):

| stage | p50 | p95 |
|---|---:|---:|
| channels | 6.0 | 15.5 |
| features | 6.7 | 21.0 |
| model and order | 3.9 | 9.8 |

That makes about 18 ms in total at p50, plus the bge-m3 query encode.

## 4. Admission (declared in LEX8 and LEX9)

The rule required retention ≥ 0.8065, ≥ 38 complete bundles, and no language with fewer complete bundles than HR.

**LEX9 fails every condition:** 0.7611, 32 complete bundles, and English 14 against HR's 15.

The 24-question answer screen was **not run**: $0, no GLM or judge call, and no answer row. `viz_lex9.json` records
answer and verdict as null for that reason.

## 5. Finding

Synthetic single-chunk questions are an easy, lexical target. F2 already reaches 0.81 nDCG@10 on them, and the
learned ranker adds 0.047 on held-out documents, mostly by re-weighting F2.

That gain does not transfer to the dev questions. Those need multi-span bundles, and the ranker is −0.018 against F2
on them (T2 0.612 vs 0.678). Supervision from synthetic questions teaches "find the one source chunk". It does not
teach "which of the relevance-similar chunks completes the bundle", and that is where the gap to the cross-encoder
lies.

The negative result is publishable alongside LEX8's: learned fusion from 18k synthetic queries does not beat a fixed
MaxMean.

**Artifacts.**

- `sample.jsonl`, `qenc.pkl`, `feats/*.npz`: the training data.
- `train9.json` and `model/`: the validation results and the frozen LightGBM model.
- `rows9.jsonl`, `TABLES9.md`, `results9.json`: the dev evaluation.
- `viz_lex9.json`: per question, the pack's features, the learned top 40, the pack, the gap and the bundle ranks.
- Registry row: `LEX9 learned scorer ...`.
