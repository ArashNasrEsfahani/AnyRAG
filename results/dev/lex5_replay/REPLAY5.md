# LEX5 replay: LLM listwise selection over HYBRID candidates (dev, 54 answerable questions)

2026-09-26. Design: `work/analysis/LEX4_DESIGN_R3.md` (Proposal R3). Code: `RAG-Paper-upgrades/ragbench/approaches/lex5/select.py`
(`lex5-select-v2`). Scripts: `step1_select.py` (one select call per question per config, resumable from `calls_*.jsonl`),
`step2_report.py` (-> `report5.json`, `tables5.md`). Step log: `PROGRESS.md`.

How it works. For each question, one OpenRouter call reads the question and the top-N HYBRID RRF candidates, each shown
as a header plus a *view*. The call names, part by part, the candidates that hold each part. The selected intact chunks
are packed first (one per part, then the rest), and the pack is then filled in RRF order. The budget is 6,000 rendered
chars, with the lex4 redundancy rule; no item is cut. An unparsable or empty reply falls back to HYBRID-intact.
The metric is lex4 decisive-span retention (`../lex4_replay/REPLAY.md` §2), unchanged.

The views:
- **snippet**: the 300-char window with the highest weight of query words.
- **full**: the whole chunk.
- **head**: the first ~300 chars. This is the ablation.

Models: `openai/gpt-oss-20b` (oss20b) and `z-ai/glm-5.3-flash` (glm), with reasoning effort low.
Total spend for this task was **$0.1725**. The budget file moved from 5.1584 to 5.3309, well below the 5.7584 stop.

**Caveat.** This dev set has been seen many times, and it has 54 questions, 18 per tier and 27 per language. One question
is worth 0.019 of overall span retention and 0.037 of a language mean. Differences of 1–3 questions are noise.
No constant was tuned on these results.

## 1. Main table

| config | span ret. | 100% q | T1 | T2 | T3 | fa | en | T3 all docs (of 18) | vs rerank better/worse | parse fallbacks | select p50 / p95 ms | tokens in / out per q | $/q | go |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| oss20b-snippet-24 | 0.804 | 36/54 | 0.981 | 0.694 | 0.736 | 0.821 | 0.787 | 10 | 7/7 | 0 | 5634 / 12797 | 3064 / 160 | 0.00011 | no |
| oss20b-full-24 | 0.838 | 40/54 | 0.981 | 0.770 | 0.764 | 0.870 | 0.806 | 12 | 10/4 | 0 | 5789 / 17478 | 6773 / 195 | 0.00024 | no (fa) |
| glm-snippet-24 | 0.837 | 39/54 | 0.981 | 0.779 | 0.750 | **0.883** | 0.791 | 12 | 9/5 | 4 | 2718 / 63546 | 3306 / 571 | 0.00069 | **yes** |
| **glm-full-24** | **0.858** | **42/54** | 0.981 | 0.785 | **0.807** | 0.864 | 0.852 | 13 | 11/2 | 1 | 5758 / 26901 | 7442 / 428 | 0.00088 | no (fa) |
| glm-full-40 | 0.855 | 40/54 | 0.981 | 0.785 | 0.799 | 0.809 | **0.902** | **14** | 12/4 | 3 | 4440 / 15395 | 12297 / 94 | 0.00071 | no (fa) |
| glm-head-24 (ablation) | 0.782 | 33/54 | 0.926 | 0.685 | 0.736 | 0.827 | 0.738 | 10 | 7/7 | 3 | 2467 / 51178 | 3319 / 390 | 0.00056 | no |
| HYBRID-intact | 0.729 | 32/54 | 0.981 | 0.623 | 0.583 | 0.722 | 0.736 | 8 | 6/13 | - | - | - | - | - |
| HYBRID | 0.720 | 32/54 | 0.926 | 0.623 | 0.611 | 0.722 | 0.718 | 9 | 6/13 | - | - | - | - | - |
| HYBRID+rerank | **0.806** | 35/54 | 0.981 | 0.748 | 0.690 | **0.877** | 0.736 | 9 | - | - | - | - | - | - |

How to read the columns:
- **Candidate ceiling.** This is the share of spans present in any single top-n candidate: 0.886 at top 24 (43 full) and
  0.942 at top 40 (46 full). glm-full-24 reaches 97% of the top-24 ceiling.
- **T3 all docs.** This counts the 18 T3 questions that need two or more documents and gets every one of them into the
  pack. It measures second-document recovery. HYBRID+rerank gets 9 of 18; every LEX5 config gets 10–14.
- **Parse fallbacks.** All glm fallbacks after the v2 parser are empty replies. The model reasoned until the 3,200-token
  completion cap (`finish=length`), which took 49–75 s each. That is also why glm's p95 latency is 27–64 s. oss20b never
  failed to parse.
- **Latency.** This is the select call alone (OpenRouter wall time). It excludes retrieval.
- **Totals per config.** oss20b-snippet $0.006, oss20b-full $0.013, glm-snippet $0.037, glm-full-24 $0.048,
  glm-full-40 $0.038 and glm-head $0.030.
- **Selection size.** The selector names 1.7–2.1 parts per question and 2.3–2.9 chunks. On average 2.1–2.7 of those chunks
  fit into the pack. No config ever went over the budget or cut an item.
- **Against HYBRID-intact.** glm-full-24 and glm-full-40 are never worse on any question: 14/0 and 13/0 better/worse,
  +10 and +8 full.

## 2. Per-language check against HYBRID+rerank

| config | fa mean (rerank 0.877) | fa better/worse | fa 100% (rerank 20) | en mean (rerank 0.736) | en better/worse | en 100% (rerank 15) |
|---|---:|---:|---:|---:|---:|---:|
| oss20b-snippet-24 | 0.821 | 2/4 | 20 | 0.787 | 5/3 | 16 |
| oss20b-full-24 | 0.870 | 3/3 | 23 | 0.806 | 7/1 | 17 |
| glm-snippet-24 | 0.883 | 3/3 | 22 | 0.791 | 6/2 | 17 |
| glm-full-24 | 0.864 | 2/2 | 22 | 0.852 | 9/0 | 20 |
| glm-full-40 | 0.809 | 2/4 | 20 | 0.902 | 10/0 | 20 |
| glm-head-24 | 0.827 | 2/4 | 19 | 0.738 | 5/3 | 14 |

- **English.** Every config with a snippet or full view beats the reranker in English. glm-full-24 and glm-full-40 are
  never worse on an English question. English is where the reranker was weak, at 0.736.
- **Persian.** The reranker is strong in Persian, at 0.877, and LEX5 roughly ties it.
  - glm-full-24 misses by 0.013, which is less than half a question. It has 2 worse questions:
    - RPT-fa-T2-001 is an empty reasoning-runaway reply that fell back.
    - BOOK-fa-T2-002 is a parsed selection that picked the wrong chunks.
  - glm-full-24 still has more fully covered Persian questions than the reranker: 22 against 20.
- **More candidates hurt Persian.** At 40 candidates Persian drops to 0.809: BOOK-fa-T3-003, RPT-fa-T2-003 and
  RPT-fa-T3-002 are lost. The longer, mixed list seems to distract the selector in Persian, while English gains.

## 3. Go rule (≥ 0.806 overall and neither language below HYBRID+rerank)

- **Passes: glm-snippet-24.** It scores 0.837 with 39/54 full, fa 0.883 ≥ 0.877 and en 0.791 ≥ 0.736.
- **Best overall: glm-full-24.** It scores 0.858 with 42/54 full, the only config at +0.05 over the reranker. It fails the
  letter of the rule on Persian by 0.013, about half a question, and one of its two Persian losses is an empty reply, not
  a wrong selection.
- **Fails the rule, same reason:** oss20b-full-24, at 0.838 with fa 0.870, the cheapest strong config at $0.00024/q.
- **Fail:** oss20b-snippet-24 (0.804) and glm-head-24 (0.782).

Honest reading: the rule is met by one config, and the three best configs clear the overall bar by 1.6–2.8 questions'
worth. The per-language margins are all within 1 question. A held-out test run is needed before any of this is claimed.

## 4. Ablation: does the lexical snippet window matter?

This compares glm at 24 candidates with the same prompt and only the view changed:

| view | span ret. | 100% q |
|---|---:|---:|
| query-hit snippet | 0.837 | 39/54 |
| head of chunk | 0.782 | 33/54 |

The head view loses 0.055 overall and 6 fully covered questions. The losses come from T2 (0.779 -> 0.685), from
English (0.791 -> 0.738) and above all from RPT (0.931 -> 0.747). In reports, the answer is rarely at the top of a chunk.
Against HYBRID-intact, head is worse on 5 questions, snippet on 2 and full on 0.

So the lexical window does real work: the query-hit window beats the head of the chunk. This supports the design's
narrow lexical claim. The whole chunk (full) is still better than either 300-char view for both models:
- glm: +0.021
- oss20b: +0.034

The strongest config is therefore "LLM-selected hybrid retrieval". The lexical part is necessary only in the cheap
snippet mode.

## 5. Defects and notes

1. **Parser (fixed, post-hoc, no effect on scores).** glm sometimes answers in Persian punctuation, for example
   `…: 10، 15` with an Arabic comma, or `هیچ` for none. The v1 parser read those lines as prose. v2 accepts
   U+060C/U+061B and `هیچ`. All six configs were re-parsed from the cached replies, with no new calls. Span retention and
   100% counts are identical to the as-run v1 results, which are archived in `parse_v1/`. glm-full-24's fallbacks went
   from 3 to 1, and glm-head's from 4 to 3.
2. **glm reasoning runaway.** 8 replies across the glm configs are empty after 3,200 completion tokens. A retry, or a
   hard reasoning cap, would likely remove these fallbacks. This was not tested.
3. **Mojibake in the snippet ellipsis.** The snippet function emits `â€¦` instead of `…`, from an earlier encoding slip in
   `select.py`. It is left as is, so the snippet runs stay reproducible. It affects only the selector's view, never the
   evidence.
4. **oss20b-full-24 was resumed.** 47 of its 54 calls were reused from the interrupted run, and 7 were new. The script
   reuses cached calls by question id.
