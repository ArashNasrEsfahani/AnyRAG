# LEX5 matched resource pilot (2026-09-26)

Spec: `work/analysis/LEX5_DESIGN_FINAL.md`, steps 1–2. Code: `RAG-Paper-upgrades` branch `upgrades` (commits d554589 + the
END-line prompt commit). The step log is in `PROGRESS.md`.

**Setup**
- **Questions.** 12 dev questions, declared before the run: the `-002` question of each chosen cell. 6 are Persian and
  6 English, 4 per tier. RPT-fa and RPT-en (the largest pools) get all three tiers.
- **Arms.** Five arms, interleaved per question, with the arm order rotated by one each question.
- **Answer reader.** Every arm is answered by `z-ai/glm-5.3-flash`, pinned to **together** with no fallback and
  identical settings.
- **Selectors.** Two selectors, each pinned with `allow_fallbacks:false`: `openai/gpt-oss-20b` on **amazon-bedrock**
  and GLM on **together**. The OpenRouter endpoints API returns null latency, so both were chosen by a live probe on a
  non-pilot question (`providers.json`).
- **Selector call settings.** 1 attempt, a 3 s total deadline with real cancellation, a 1,024-token cap that includes
  reasoning, and reasoning effort low.
- **Judge.** The v2 judge (`judge_multi`, `openai/gpt-oss-120b`), strict verdicts.
- **Spend.** $0.071 in total: the probes, 60 answer cells and 42 judge calls. The budget file moved from 5.3309 to 5.4019.

## Per arm

| arm | correct /12 | fa /6 | en /6 | p50 s | p95 s | LLM tokens/q (prompt / completion / of which reasoning) | LLM calls/q | selector ok / all-NONE / fallback | $/q | $ total (+warm-up) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| HYBRID | 5 | 1 | 4 | 1.50 | 2.63 | 2143 (2049 / 94 / 21) | 1.0 | - | 0.00035 | 0.0042 (+0.00048) |
| HYBRID+rerank | 7 | 3 | 4 | 2.75 | 6.41 | 2235 (2084 / 150 / 34) | 1.0 | - | 0.00038 | 0.0046 (+0.00096) |
| HYBRID+intact | 3 | 1 | 2 | 2.01 | 3.52 | 2101 (1981 / 120 / 16) | 1.0 | - | 0.00031 | 0.0037 (+0.00036) |
| LEX5 | 8 | 3 | 5 | 3.10 | 6.16 | 9146 (8843 / 302 / 17) | 2.0 | 11 / 0 / 1 | 0.00083 | 0.0099 (+0.00063) |
| LEX5+selector=glm | 6 | 2 | 4 | 2.91 | 5.23 | 7496 (7384 / 112 / 24) | 2.0 | 8 / 0 / 4 | 0.00105 | 0.0126 (+0.00048) |

| arm | answer call tokens/q (p / c / r) | selector call tokens/q (p / c / r) | selector p50 / p95 ms | answer p50 / p95 ms | retrieval p50 ms (incl. selector) | embed calls/q (tokens) | reranker tokens/q non-pad / padded | reranker ms/q | selector statuses |
|---|---|---|---|---|---|---|---|---|---|
| HYBRID | 2049 / 94 / 21 | - | - | 1376 / 2492 | 127 | 1.0 (40) | - | - | - |
| HYBRID+rerank | 2084 / 150 / 34 | - | - | 1762 / 5288 | 678 | 1.0 (40) | 12274 / 14250 | 944 | - |
| HYBRID+intact | 1981 / 120 / 16 | - | - | 1852 / 3227 | 131 | 1.0 (40) | - | - | - |
| LEX5 | 2035 / 120 / 17 | 6808 / 182 / 0 | 1091 / 2075 | 1739 / 4404 | 1248 | 1.0 (40) | - | - | {'ok': 11, 'fail:duplicate_alternative': 1} |
| LEX5+selector=glm | 2008 / 88 / 19 | 5375 / 25 / 5 | 1166 / 3053 | 1287 / 2414 | 1373 | 1.0 (40) | - | - | {'ok': 8, 'fail:overflow_alternatives': 1, 'fail:deadline': 3} |

| comparison | LLM-token ratio | p50 ratio | p95 ratio | resource gate (all <= 1.5x) | net correct /12 |
|---|---:|---:|---:|---|---:|
| LEX5 vs HYBRID | 4.27 | 2.07 | 2.34 | FAIL | +3 |
| LEX5 vs HYBRID+rerank | 4.09 | 1.13 | 0.96 | FAIL | +1 |
| LEX5+selector=glm vs HYBRID | 3.50 | 1.94 | 1.99 | FAIL | +1 |
| LEX5+selector=glm vs HYBRID+rerank | 3.35 | 1.06 | 0.82 | FAIL | -1 |

Selector room under the token gate (tokens/q): LEX5: selector uses 6991, answer 2155; allowed selector <= 1060 vs HYBRID, <= 1197 vs HYBRID+rerank; LEX5+selector=glm: selector uses 5400, answer 2096; allowed selector <= 1118 vs HYBRID, <= 1256 vs HYBRID+rerank

Judge: 42 calls, $0.0102.

How to read the tables:
- **LLM tokens.** This is prompt plus completion over every LLM call of the arm, with reasoning included in the
  completion. It is the gate's "LLM-token budget".
- **Reranker tokens.** These are reported, never gated: about 12.3k non-padding and 14.3k padded tokens per question,
  which matches Astra's 12,395 / 14,740.
- **Latency.** This is the whole timed query per question: retrieval, then the selector, then the answer.

**Fallbacks**

| arm | fallbacks /12 | causes |
|---|---:|---|
| LEX5 (oss20b) | 1 | 1 grammar rejection |
| LEX5+glm | 4 | 3 hit the 3 s deadline; 1 had more than 2 alternatives |

No fallback was repaired.

**Usage-reporting gaps**
- The usage of GLM's 3 cancelled calls is unknown and is recorded as unknown. GLM's selector token mean therefore
  **understates** its real cost.
- Bedrock reports 0 reasoning tokens for oss20b.

## Gate ratios (unchanged thresholds: LLM tokens, p50 and p95 each <= 1.5x of EACH control)

| LEX5 arm | vs HYBRID: tokens / p50 / p95 | vs HYBRID+rerank: tokens / p50 / p95 |
|---|---|---|
| LEX5 (oss20b) | **4.27** / 2.07 / 2.34 → FAIL | **4.09** / 1.13 / 0.96 → FAIL (tokens) |
| LEX5+selector=glm | **3.50** / 1.94 / 1.99 → FAIL | **3.35** / 1.06 / 0.82 → FAIL (tokens) |

## Verdict: the LLM-token gate is plainly infeasible. STOP. No full grid is recommended.

**The token gap is structural.**
- The answer call costs about 2.1k tokens in every arm.
- The gate therefore leaves the selector at most about 1.1k tokens per question (1,060 against HYBRID, 1,197 against
  HYBRID+rerank).
- The full-view selector reads 5.4k tokens (GLM) to 7.0k tokens (oss20b), 5–6x over that allowance.
- Even the 300-char snippet view (about 3.1–3.3k prompt tokens in REPLAY5) would be about 3x over.
- Only about 3 candidates' worth of text fits into 1.1k tokens, so no view of 24 candidates can pass.

**The latency gate fails too, against plain HYBRID.**
- LEX5 is about 2x HYBRID on p50 and p95. The selector alone takes about 1.1 s p50 even on the fastest pinned
  endpoints.
- Against HYBRID+rerank, latency passes: 0.82–1.13x.

**Quality is not a basis for continuing.**
- LEX5 (oss20b) scored 8/12, against 5 for HYBRID and 7 for HYBRID+rerank.
- With 12 questions, these differences are noise, and they cannot rescue a hard resource constraint.

**For the paper.**
- Report LEX5 as a documented negative result under the resource gate: LLM selection improves evidence retention
  (REPLAY5) at about 3.5–4.3x the LLM tokens.
- Keep the resource table above, with the reranker tokens beside it, as the quality/resource tradeoff.

**If the gate is ever amended and a full run is wanted anyway (not recommended here).**
- The full run would be 60 dev questions × the same 5 arms. Measured cost was about $0.0029 per question across all
  five arms, plus about $0.0009 per question for the judge, for roughly **$0.25** in total.
- The spec's ablations (greedy, flat, no_fill, query_hit, head, top=40, permute) would add about $0.05 per arm.
- Before any such run, GLM's deadline fallbacks (3 of 12) need a decision: its selector is not reliable within 3 s.

## Also done (step 1)
- **HYBRID-intact recomputed offline** under the new rules (exact/contained dedupe with provenance, exact cost, and
  source-identity retention). It scores 0.7292 with 32/54 full, the same as before (`intact_recompute.json`).
- **Code.**
  - lex4 fixes: the mutation fix and provenance.
  - `retention()` checks source identity.
  - OpenRouter pinning, deadline and cap options.
  - The reranker reports its token counts.
  - `lex5/bundles.py` holds the strict grammar and the exact, greedy and flat packers.
  - `lex5/approach.py` runs `--approach lex5` and the HYBRID flag `intact`.
  - `tests/test_lex5.py` passes 69/69, including Astra's greedy counterexample and a real deadline-cancellation test.
- **Analyzer and plan.** `analyze_v2.py` now lists the LEX5 arms under the unchanged LEX gates, with HYBRID+intact as a
  reference. `make_plan.py lex5` writes the GLM-reader plan.
