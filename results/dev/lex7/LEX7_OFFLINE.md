# LEX7 offline: contextual prefixes and a small-budget order fixer (2026-09-26)

**Scope.** The 54 answerable dev questions. Metric: lex4 decisive-span retention with source identity, in the 6,000-char pack built by the **chunk packer**. This packer reproduces HYBRID 0.720 and HYBRID+rerank 0.806.
- **Rules.** Arms and rules were declared in `PROGRESS.md` before any run. No constant was tuned on results.
- **Baselines.** Reused from `lex_research/rows.jsonl`, not re-run. The a1 and MMR-a1 recomputes match exactly (max diff 0.000).
- **Caveat.** This dev set has been seen many times. One question moves the mean by 0.019.

**Code.**
- `ragbench/approaches/lex7/prefixes.py` and `orderfix.py` (branch `upgrades`).
- Harness in this folder: `encode7.py`, `evaluate7.py`, `orderfix7.py`, `report7.py`.

**Data.**
- `TABLES7.md` and `TABLES7_orderfix.md`, with per-question rows in `rows7*.jsonl`.
- Order-fixer calls: `orderfix_rows.jsonl` and `ledger_orderfix.jsonl`.
- Visualizer trace: `viz_lex7.json`.

**Spend.** $0.004 on OpenRouter: 54 calls to gpt-oss-20b. Budget file 6.4561 → 6.4600.

## 1. Results (chunk packer; cells give mean retention with questions at 100% in brackets)

| arm | pack | top-24 | fa | en | T1 | T2 | T3 | RPT-fa | T3 all gold docs in pack /18 | better/worse vs rerank | better/worse vs HYBRID | extra LLM tokens per question |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| HYBRID (reused) | 0.720 (32) | 0.886 | 0.722 | 0.718 | 0.926 | 0.623 | 0.611 | 0.556 | 12 | 6/13 | – | 0 |
| **HYBRID+rerank (reused)** | **0.806 (35)** | 0.906 | **0.877** | 0.736 | 0.981 | **0.748** | **0.690** | **0.889** | 13 | – | 13/6 | 0 |
| a1 RRF(BM25, dense, sparse) (reused) | 0.763 (34) | 0.921 | 0.772 | 0.755 | 0.981 | 0.669 | 0.639 | 0.611 | 14 | 6/10 | 4/0 | 0 |
| MMR-a1 (reused, recomputed identically) | 0.767 (34) | 0.926 | 0.790 | 0.744 | 0.981 | 0.663 | 0.657 | 0.667 | 14 | 6/9 | 5/2 | 0 |
| P0 a1 with HF bge-m3 dense (control) | 0.763 (34) | 0.921 | 0.772 | 0.755 | 0.981 | 0.669 | 0.639 | 0.611 | 14 | 6/10 | 4/0 | 0 |
| P1 a1-prefix | 0.714 (28) | 0.902 | 0.772 | 0.656 | 0.926 | 0.637 | 0.579 | 0.611 | 11 | 4/14 | 5/5 | 0 |
| P2 a1-prefix + MMR | 0.726 (31) | 0.897 | 0.796 | 0.656 | 0.926 | 0.702 | 0.551 | 0.611 | 11 | 5/12 | 5/5 | 0 |
| P3 RRF-of-4-prefix | 0.708 (27) | 0.908 | 0.753 | 0.663 | 0.926 | 0.609 | 0.590 | 0.556 | 9 | 5/15 | 6/6 | 0 |
| P4 prefix on BM25 and sparse only | 0.749 (31) | 0.898 | 0.759 | 0.739 | 0.981 | 0.637 | 0.630 | 0.611 | 10 | 5/11 | 7/3 | 0 |
| **O1 order fixer on MMR-a1** | **0.773 (35)** | 0.926 | 0.790 | **0.757** | 0.981 | 0.681 | 0.657 | 0.667 | 14 | 6/9 | **6/1** | 945 |

## 2. Findings

1. **Free contextual prefixes hurt.** P1 scores −0.049 against a1. English falls from 0.755 to 0.656; Persian and RPT-fa do not move.
   - Each prefix is the compact doc card plus the covering section card. It makes sibling chunks of one section look alike, so the channels pull in more chunks from the same section.
   - The only variant that loses little is P4, which prefixes BM25 and sparse only and keeps plain dense: 0.749.
   - P0 equals a1 exactly, because the Ollama embedder is bge-m3. The drop therefore comes from the prefixes, not from swapping the dense model.
   - Under the pre-declared rule ("helps" needs at least 0.773), **step 2 was skipped**: LLM-written prefixes were not tried and $0 was spent on them.
2. **The order fixer is cheap and fast but only slightly positive.** It adds +0.006 over its base (MMR-a1) and +0.010 over a1: 6/1 against HYBRID, 2/1 against a1.
   - **Calls:** 54 of 54 returned and none hit the 1.5 s deadline.
     - Wall time: p50 586 ms, p95 866 ms.
     - Prompt tokens: p50 851, max 887.
     - Completion tokens: p50 91.
   - **Outcomes:** 34 valid picks, 8 NONE, 12 truncated at the 120-token cap. The reasoning ran out of room, so those fell back to the base order.
   - Retention stays 0.033 below the reranker, about 2 questions. RPT-fa is still 0.667 against 0.889.
3. **Gained / lost against the reranker (O1):**
   - **Gained 6:** BOOK-en-T3-002, BOOK-en-T3-003, LAW-fa-T2-003, LAW-en-T2-001, LAW-en-T2-002, RPT-en-T3-003.
   - **Lost 9:** BOOK-fa-T2-002, BOOK-en-T2-001, BOOK-en-T2-002, BOOK-en-T3-001, LAW-fa-T3-002, RPT-fa-T2-001, RPT-fa-T2-003, RPT-fa-T3-002, RPT-en-T2-003.
   - Against HYBRID it gains 6 and loses 1 (BOOK-en-T3-002).
   - Details are in `gained_lost.json`.

## 3. Latency per stage (ms; p50 / p95 over 54 questions)

| stage | p50 | p95 |
|---|---:|---:|
| bge-m3 query encode, GPU fp16 (reused measurement) | 13.2 | 27.3 |
| bge-m3 query encode, CPU (reused measurement) | 182.6 | 254.1 |
| BM25 + dense ranking (CPU) | 2.3 | 6.7 |
| sparse channel | 1.2 | 3.8 |
| MMR over the top 40 | 9.8 | 12.3 |
| **order fixer call** (gpt-oss-20b, Bedrock, 1.5 s deadline) | **586** | **866** |
| cross-encoder rerank (saved HYBRID+rerank trace) | 3,938 | 5,545 |

**Gate estimate for O1.** These are additive estimates, not a measured end-to-end run.
- **Latency:** about HYBRID + 0.61 s at p50 (about 2.24 s), which is **1.38x HYBRID** and 0.82x the reranker. At p95 it is about 1.24x HYBRID.
- **Tokens:** 2,169 + 945 = 3,114 per question, which is **1.44x HYBRID** and 1.41x the reranker.
- Both are inside the 1.5x gates, but with little room to spare.

**Index-time cost** (separate from query time):
- Prefix build: 7 s.
- BM25 rebuild over the prefixed text: 77 s.
- bge-m3 re-encode of all 58,218 prefixed chunks: **1,102 s (18.4 min) on the RTX 4060**. Mean length is 290 tokens; 23 chunks reached the 640-token cap.
- The cards are reused from lex6 (they cost $0.99 when they were made).

## 4. Pick and verdict

- **Pick** (pre-declared rule: highest retention inside the token gate) is **O1 = MMR-a1 plus the order fixer: 0.773**, at 945 extra LLM tokens per question.
- It stays **below the 0.80 threshold**, so **the GLM answer and judge screen was not run** (by the rule). No registry answer row was added. The offline arms were appended as retention rows.
- **Verdict.** LEX7 does not beat HYBRID+rerank on retention. The gap in ordering the first few items is still about 0.03, concentrated in RPT-fa T2/T3.
- **Next cheapest lever:** a declared variant of the fixer with a completion cap of about 200. It would recover the 12 truncated calls and stay at about 1.48x HYBRID tokens.

## 5. LEX7.1 (2026-09-26): RPT-fa diagnosis, fixer v2, and the answer screen

Arms and rules were declared in `PROGRESS.md` before any run. Files: `v71/` (diag, window, orderfix71, parity, m3idx,
viz71), `v71/eval/` (answers71, report71, runs, judge), `TABLES7_1.md`. Code: approach `LEX7`, commit 522fa31 on `upgrades`.
**Spend:** $0.021 (budget file 6.4601 -> 6.4808).

### 5.1 Why the reranker wins on RPT-fa (no model calls; `v71/diag_rptfa.json`)

| question | decisive chunk | BM25 | dense | sparse | MMR-a1 / O1 rank | in rerank pack | in O1 pack |
|---|---|---:|---:|---:|---:|---|---|
| RPT-fa-T2-001 E1 "+21.8 میلیون تعداد کاربران یکتای آگهی گذار" | 3652 (stat card, starts mid-figure) | 13 | - | 18 | 22 | yes | no |
| RPT-fa-T2-001 E2 "آگهی های 86 درصد ... رایگان" | 3657 | - | 18 | **1** | 19 | no | no |
| RPT-fa-T2-003 E1 "3300 برنامه ی فروشنده، جدید" | 3829 | 34 | 20 | 23 | 13 | no | no |
| RPT-fa-T2-003 E2 "1730 بازی فروشنده، جدید" | 3838 | 32 | 16 | 13 | 10-11 | yes | no |
| RPT-fa-T3-002 E2 "نصب فعال* بازار از 53 میلیون" | 3866 | 37 | **6** | 23 | 13 | yes | no |

- **Not normalisation.** Every RPT-fa chunk already holds ASCII digits (0 Persian/Arabic digits). The bge-m3 tokenizer
  splits a ZWNJ exactly like a space (`کافه‌بازار` and `کافه بازار` give the same pieces). BM25 query tokens already
  add the ZWNJ parts. Tables are not the pattern either (only one of the chunks has pipes).
- **The pattern is paraphrase plus RRF.** The question says "تازه‌واردها" / "بدون پرداخت هزینه"; the text says "جدید" /
  "رایگان". BM25 misses those chunks (rank 32-37 or absent), one bi-encoder channel ranks them well, and equal-weight RRF
  puts them at fused rank 10-22: below the fixer's 12-candidate window and below the ~7-chunk pack. The cross-encoder
  reads question and chunk together and lifts them.
- **Fix tested (F1, domain-neutral, question-independent):** a channel-diverse fixer window (base top-8, then each
  channel's best ranks round-robin up to 12). Window coverage 0.815 vs 0.820 for the plain top-12 (RPT-fa 0.778 vs
  0.722; up 3 / down 4). **Not adopted** by the declared rule (needs +0.010 overall).

### 5.2 Fixer v2 offline (O2 = 200-token cap, MMR-a1 top-12)

- Retention **0.765** (O1 0.773, MMR-a1 0.767, rerank 0.806). Truncations fell from 12 to 3, but NONE rose to 10 and the
  picks changed (0 gained / 2 lost vs O1). gpt-oss-20b's picks vary run to run; the cap is not the lever.
- 953 tokens/q; fixer ms p50 621, p95 1,034; 1 deadline miss.

### 5.3 Answer screen, idea_test 24 (LEX7.1 = approach LEX7: RRF(BM25, bge-m3 dense, bge-m3 sparse) + MMR + fixer v2)

Parity first: the approach's no-fixer base order gives pack retention 0.7673 = offline MMR-a1 on the 54.

| arm | correct /24 | shown retention | p50 s | p95 s | LLM tokens/q |
|---|---:|---:|---:|---:|---:|
| HYBRID (reused) | 12 | 0.728 | 1.62 | 3.69 | 2,169 |
| HYBRID+rerank (reused) | 13 | 0.828 | 2.74 | 5.59 | 2,215 |
| **LEX7.1** | **13** | 0.781 | 2.30 | 4.38 | 3,152 (fixer 945) |

| vs | net | tokens | p50 | p95 |
|---|---:|---:|---:|---:|
| HYBRID | +1 | 1.45x | 1.42x | 1.19x |
| HYBRID+rerank | 0 | 1.42x | 0.84x | 0.78x |

- Resources are inside every 1.5x gate, but accuracy is not: +1 and 0 on 24, far from the +3/54 needed.
- Stage p50: encode 24 ms (GPU), channels 7 ms, MMR 14 ms, fixer 680 ms, answer 1,477 ms. Fixer: 16 picks, 7 NONE, 1 fail.
- vs rerank: gains RPT-fa-T2-002, BOOK-en-T1-001, LAW-en-T2-001; loses RPT-fa-T3-002, BOOK-en-T2-002, BOOK-en-T3-001.
- Caveat: control latencies were measured earlier the same day, not interleaved.

**Verdict.** LEX7.1 ties HYBRID+rerank on the screen at 0.84x its latency, but it does not beat it. The remaining gap is
paraphrase-level relevance that no lexical normalisation reaches. The fixer only sees the top 12, and widening that
window did not help coverage.
