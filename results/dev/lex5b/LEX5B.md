# LEX5b: LLM support-bundle selector over F2 candidates under the 2.5x token gate (2026-09-26)

**Protocol amendment (user, 2026-09-26 ~19:20, before any LEX5b result).** LEX may use up to **2.5x** HYBRID's mean
LLM tokens per question (was 1.5x): 2.5 x 2,169 = **5,422 tokens/q** (selector + answer). The latency gate is
unchanged (p50 and p95 <= 1.5x vs EACH control), and so is the accuracy target (+3 vs each control).

**Where things are.**
- Code: `RAG-Paper-upgrades`, branch `upgrades`.
  - `ragbench/approaches/lex5b/views.py` (27cb3ae).
  - Approach `LEX5B` in `lex5b/approach.py` (2a0c543).
- Harness: `lex5b.py` (dry and run), `report_lex5b.py`, `eval/answers5b.py`, `eval/report5b.py`.
- Every arm and rule was declared in `PROGRESS.md` before any selector call.
- Spend: $0.143 on the reader budget (9.9678 -> 10.1109) plus $0.004 for the judge.

**Caveat.** This is the much-inspected dev set. One question is worth 0.019 of 54-question retention and one answer
out of 24. Nothing here is a confirmatory result.

## 1. Design (as declared)

1. **Candidates.** The F2 MaxMean top-40 from lex8: fusion of BM25, bge-m3 dense and bge-m3 sparse over the real
   channel scores. Parity with lex8's saved order is 54/54.
2. **Evidence units.** lex4 intact packages, packed whole.
3. **Selector view of the top 24.** A document legend (`D<k>: title`), then a header per candidate
   (`[n] D<k> > section`), then an excerpt. One excerpt cap per question: the largest cap whose prompt estimate fits
   **2,600 tokens**.
4. **One selector call.**
   - The endpoint is pinned (no provider fallback).
   - 3 s deadline; 600-token completion cap (reasoning included); reasoning effort low.
   - Strict lex5 bundle grammar. Any failure falls back to F2-intact.
5. **Packing.** Exact bundle packing, then fill (remaining complete bundles, then F2 packages), up to 6,000 chars.
   The shared GLM answer call follows.

The views:
- **smart**: the chunk's 1-2 best lex8 compact units, ranked by R_unit (MaxMean of the same three channels).
- **literal**: the old query-word window, under the same cap rule.
- **full**: whole packages, for the top k <= 10 that fit.
  - Top-10 full views did not fit (estimate mean 3,093), so S-d fits k per question (p50 8).
  - This was declared before the run.

## 2. Offline arms (54 questions; `TABLES_LEX5B.md`)

| arm | span | complete (fa / en) | b/w vs HR | selector tok/q | select p50 / p95 ms | fallbacks | admitted |
|---|---:|---:|---:|---:|---:|---:|---|
| HYBRID (reused) | 0.7200 | 32 (17 / 15) | 6/13 | 0 | - | - | - |
| HR (reused) | 0.8065 | 35 (20 / 15) | - | 0 | - | - | - |
| F2-intact (fallback) | 0.7693 | 32 (17 / 15) | 4/9 | 0 | - | - | - |
| S-a oss20b, smart | 0.7892 | 33 (17 / 16) | 6/8 | 2,746 | 707 / 1,248 | 8 (grammar) + 8 all-NONE | no |
| S-b glm, smart | 0.8008 | 34 (17 / 17) | 7/7 | 2,828 (2,921*) | 1,689 / 3,031 | 16 (10 deadline, 6 HTTP 429) | no |
| **S-c glm, literal** | **0.8540** | **38 (20 / 18)** | **9/4** | 2,779 (2,827*) | 1,252 / 3,030 | 7 (deadline) | **yes** |
| S-d glm, full k<=10 | 0.7920 | 34 (17 / 17) | 6/7 | 2,607 | 764 / 1,517 | 2 | no |

\* The gate figure counts each cancelled call, whose usage is unknown, as its prompt estimate + 600.

**The smart excerpts did not work as hoped.**
- The literal query-word window beat them with the same model: 0.854 against 0.801.
- Much of S-b's gap is its 16 fallbacks, including a burst of upstream 429s from Together.
- On the 35 questions where both S-b and S-c parsed, literal still leads 0.894 against 0.880, better/worse 2/0.
- S-a (oss20b) often answered all-NONE when the evidence was not visible in its excerpts.
- The "paraphrase-visible" view is a negative result on this dev set.

**Admission.** Only S-c meets all four conditions:
- span 0.854 >= 0.8065;
- 38 complete >= 38;
- fa 20 >= 20 and en 18 >= 15;
- 2,827 + 2,169 = 4,996 <= 5,422 tokens.

## 3. Answer screen: S-c as approach LEX5B on the idea_test 24 (`eval/ANSWER_TABLES5B.md`)

Controls were reused: HYBRID 12, HR 13, LEX7.1 13. The reader was GLM pinned to Together; the judge was gpt-oss-120b,
strict.

| arm | correct /24 | fa / en | p50 s | p95 s | LLM tokens/q |
|---|---:|---:|---:|---:|---:|
| HYBRID | 12 | 5 / 7 | 1.62 | 3.69 | 2,169 |
| HYBRID+rerank | 13 | 6 / 7 | 2.74 | 5.59 | 2,215 |
| LEX7.1 | 13 | 6 / 7 | 2.30 | 4.38 | 3,152 |
| **LEX5B** | **17** | 6 / 11 | 4.38 | 7.74 | 4,315 (4,964 conservative) |

| vs | net correct | tokens (<= 2.5x) | p50 (<= 1.5x) | p95 (<= 1.5x) |
|---|---:|---:|---:|---:|
| HYBRID | **+5** | 1.99 (2.29 cons.) ok | **2.70 FAIL** | **2.10 FAIL** |
| HYBRID+rerank | **+4** | 1.95 (2.24) ok | **1.60 FAIL** | 1.39 ok |
| LEX7.1 | **+4** | 1.37 (1.57) ok | **1.90 FAIL** | **1.77 FAIL** |

**Honest reading.**
- **Accuracy and tokens pass.** The accuracy target (+3) is met against every control, and the amended token gate
  holds even with the conservative count.
- **Latency fails against all three controls**, on p50 in every case.
- **Where the time goes** (stage p50):
  - selector call: 1.6 s p50, 3.05 s p95 (5 of 24 hit the 3 s deadline and fell back);
  - excerpt-cap fitting: 0.40 s (the fitting loop re-renders views; a faster search is an implementation fix that has
    not been tried);
  - answer call: 2.17 s.
- **The answer call ran slower than when the controls were measured.** The controls' pilot answer p50 was about 1.4 s,
  and they were measured earlier the same day, not interleaved. Even at the controls' answer speed, retrieval plus
  selector (2.1 s p50) alone breaks 1.5x of HYBRID's 1.62 s.
- **English carries the gain**: 11 against 7. Persian ties at 6.
- **24 questions is a screen, not a result.** +4 or +5 on 24 is suggestive only.

## 4. Artifacts

- `rows_S-*.jsonl` and `calls_S-*.jsonl`: per question, candidates, excerpts, the prompt, the raw and parsed reply,
  bundles, the pack, retention, tokens and timings.
- `dry_rows.jsonl`: F2 parity and prompt sizing.
- `best_model.json`.
- `eval/runs`, `eval/judge`, `eval/answer_results5b.json`.
- `viz_lex5b.json`: per question, the offline arms and the screen records, including the controls' answers and
  verdicts.
- Registry: 5 rows, tagged `gate_tokens: 2.5x (user amendment 2026-09-26)`.
- `void/`: a harness bug (no ledger phase) refused all 54 first S-a requests locally, at $0 and with no call made.
  The rows are voided, and S-a was run once, properly.
