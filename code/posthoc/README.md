# Post hoc experiment code (November 2026)

The scripts behind `results/posthoc/`. One folder per stream, with the same layout as the original workspace
(`work/posthoc_2026-11/<stream>/`); each stream's outputs live in `results/posthoc/<stream>/`, whose `RESULTS.md`
says exactly what was run and which script wrote each file. `results/posthoc/COMMON.md` holds the rules every stream
followed (same pinned reader and judge as the test, money floor, no tuning on test questions).

All of this is **post hoc and exploratory**: it ran after the one-shot test had been unsealed, on the same 96 test
questions (90 answerable). Every new arm was fixed before its single test run.

| stream | what it does | main scripts |
|---|---|---|
| `S1_evaluator` | second judge (mistral-large-2512, same rubric and parser as the pinned judge) over all 768 GLM test answers; builds the blinded human-review packets | `judge2.py`, `analyze_judge2.py`, `build_packets.py`, `human_estimate.py` |
| `S2a_packs` | FUSION (Lex-RAG without the selector), GOLD (gold evidence only), non-LLM packers (MMR, knapsack), budget sweep 3,000 to 24,000 characters; byte-for-byte reproduction of the frozen packs | `verify.py`, `build.py`, `answer.py`, `judge.py`, `report.py` |
| `S2b_variance` | reader redraws on identical messages, selector re-runs (LEX5B, SETR), judge consistency | `exp1_reader_repeats.py`, `exp_selector.py`, `exp6_setr_selector.py`, `analyze.py` |
| `S3_agentic` | agent arms ITER_BM25 and GREP_AGENT (GLM as agent and answerer) | `agents.py`, `judge.py`, `analyze.py` |
| `S4_route_long` | LLM router over the frozen arms; long-context reading of whole documents | `code/router.py`, `code/route_analysis.py`, `code/lc.py`, `code/lc_analysis.py` |
| `S5_stats` | GLMM over readers, arms and questions; minimum detectable effects; reproduction checks against the paper's statistics | `s5_data.py`, `analyses.py`, `glmm_final.py`, `glmm_pack.py`, `tables.py` |
| `S5_stats/crosscheck` | independent re-estimate of the S5 GLMM with statsmodels (BinomialBayesMixedGLM by VB and MAP, GEE, cluster-robust GLM, pipeline x reader interaction tests); reads `../cells.json` only | `crosscheck.py`, `run_all.sh`, `combine.py` |
| `S6_baselines` | closed book, LLM listwise reranking (RankGPT style), SetR without hybrid fill | `run_arms.py`, `judge_arms.py`, `analyze.py`, `verify_repack.py` |
| `S7_analysis` | where answers are lost (candidate list vs pack), contamination, tokens, error types, Persian channels, paper examples (saved data only) | `t1_reach.py` ... `t6_numbers.py` |
| `S8_factorial` | Lex-RAG across budgets, the selector over HR's candidates, no-deadline selector, judge 2 on the post hoc arms, dev replication | `s8run.py`, `judge2_posthoc.py`, `dev_fusion.py`, `analyze.py` |
| `S9_gpu` | Ollama vs Hugging Face bge-m3 alignment; HYBRID_AGENT (agent over hybrid retrieval) | `align.py`, `hybrid_agent.py`, `judge.py`, `analyze.py` |
| `S10_retrieval` | dense-only and BM25-only arms with HYBRID's packer, at 6,000 and 15,500 characters | `build.py`, `answer.py`, `judge.py`, `report.py`, `tables.py` |
| `S11_rewrite` | query rewriting (HyDE, multi-query) on top of HYBRID at 6,000 characters (GPU for query encoding) | `rewrite.py`, `encode.py`, `run.py`, `judge.py`, `analyze.py` |
| `S12_chunks` | HYBRID and HR with 500- and 2,000-character chunks (new indexes, GPU) | `index.py`, `retrieve.py`, `packs.py`, `answer.py`, `judge.py`, `report.py` |
| `S13_embed` | multilingual-e5-large in place of bge-m3 (GPU) | `embed.py`, `retrieve.py`, `llm.py`, `report.py` |
| `S14_bilingual` | language interaction and replication from saved verdicts; tokens per character and judges by language | `t1_lang.py`, `t2_tok_judge.py`, `t3_compact.py` |
| `S15_crossling` | cross-lingual query swap (question asked in the other language) and Persian orthographic perturbation | `translate.py`, `variants.py`, `retrieve.py`, `answer.py`, `judge.py`, `analyze.py` |
| `S16_judge2` | second judge (mistral-large-2512) on the 12 post hoc variants scored only by judge 1; judge-1 totals reproduced from the saved cells. Only a 3-question estimate ran (money floor); see its `RESULTS.md` | `s16.py`, `judge2_posthoc.py`, `j2lib.py` |
| `S17_goldmiss` | reading of the 7 questions missed with gold evidence (saved data only; no scripts) | none |

## Rerunning

- **Paths are hard-coded.** Like `code/test_freeze`, these scripts hard-code the original Windows locations:
  the project root `D:/KareOne/RAG Paper` (with `work/posthoc_2026-11/<stream>`, `work/datasets_v2/test_run`,
  `work/datasets_v2/test_freeze`, `work/indexes`) and the frozen code worktree `D:/KareOne/RAG-Paper-upgrades`
  at commit 77924d4 (the code in `code/ragbench`), which they put on `sys.path` and import read-only. Edit those
  constants to match your checkout; the frozen test data correspond to `results/test`, `results/protocol` and
  `benchmark/` in this repository.
- **Analysis-only streams** (S5 incl. its cross-check, S7, S14, S17, and the `analyze.py` / `report.py` / `tables.py` scripts of the others)
  make no model calls and rebuild their tables from the released records, verdicts and ledgers.
- **Streams that call models** (S1, S2a, S2b, S3, S4, S6, S8-S13, S15, S16) need an OpenRouter key in the project `.env`
  (`OPENROUTER_API_KEY`) and cost money; they write append-only JSONL keyed per cell, so a rerun skips finished
  cells. Retrieval runs need the rebuilt corpora and indexes (see the top-level README); S9, S11-S13 and S15 also need a
  GPU and/or Ollama with bge-m3. The run order within each stream is given in its `RESULTS.md`.
- Released records have their corpus text stripped (`tools/strip_corpus_text.py`), so scripts that rebuild packs from
  passage text need the rebuilt corpora.
