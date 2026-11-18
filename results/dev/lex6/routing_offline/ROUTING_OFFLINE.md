# LEX6 offline document routing: results (2026-09-26)

This test made no LLM or API calls. The only model use was local bge-m3, which embedded the 155 heuristic cards in 6.5 s.
- **Questions:** the 54 answerable dev questions, with the idea_test 24 reported separately.
- **Gold documents:** the documents of each question's decisive spans. They match the evidence-unit documents and `doc_scope.oracle_docs` in 54/54 questions.
- **Where the rest is:** full tables are in `ROUTING_TABLES.md`, numbers in `results.json`, per-question rows in `rows.jsonl`.
- **Code:** `ragbench/approaches/lex6/docrank.py` on branch `upgrades`, and the runner `routing.py` in this folder.
- **Rank ties:** every signal gives a full ranking of the pool. Tied documents keep pool order, so a document with no chunk in the top 80 still gets a rank.
- **Recall@k:** a question counts as a hit only when EVERY gold document is in the top k.

## 1. Document-ranking recall (R@k = share of questions; hit count in brackets)

All 54:

| signal | R@1 | R@2 | R@3 | R@5 | R@8 | R@12 | mean worst-gold rank |
|---|---:|---:|---:|---:|---:|---:|---:|
| S1 chunk-vote | 0.56 | 0.89 | 0.93 | 0.94 | 0.96 | 1.00 | 1.9 |
| S2 chunk-max | **0.59** | **0.93** | 0.93 | 0.94 | 0.98 | 1.00 | **1.8** |
| S3 card BM25 | 0.52 | 0.76 | 0.83 | 0.93 | 0.94 | 0.96 | 2.5 |
| S4 card dense | 0.57 | 0.83 | 0.89 | 0.94 | 0.96 | 0.98 | 2.1 |
| S5 doc-centroid dense | 0.57 | 0.91 | **0.94** | **0.96** | 0.98 | 1.00 | **1.8** |
| S6 fused RRF(S1,S3,S4) | 0.57 | 0.83 | 0.91 | 0.94 | 1.00 | 1.00 | 1.9 |
| S7 keyword doc-first (24 q only) | 0.50 | 0.75 | 0.83 | 0.83 | 0.92 | 0.96 | 4.0 |

For comparison, on the same 24 idea_test questions: S5 R@2 0.92 / R@3 0.92, S2 0.88 / 0.88, S1 0.83 / 0.88, S6 0.79 / 0.83, S7 0.75 / 0.83.

The hard slices are shown below. For each slice, a cell gives R@2 / R@3 / R@5 and then the mean worst-gold rank.

| signal | RPT-fa, 97 docs (n=9) | T3 (n=18) | T3 in RPT-fa (n=3) | fa (n=27) | en (n=27) |
|---|---|---|---|---|---|
| S1 chunk-vote | .44/.67/.78; 4.2 | .78/.78/.83; 3.3 | 0/0/.33; 8.0 | .78/.85/.89; 2.5 | 1/1/1; 1.3 |
| S2 chunk-max | .67/.67/.78; 3.3 | .83/.83/.83; 2.9 | .33/.33/.33; 6.0 | .85/.85/.89; 2.2 | 1/1/1; 1.3 |
| S3 card BM25 | .44/.67/.67; 5.4 | .56/.61/.83; 4.3 | 0/.33/.33; 10.3 | .67/.78/.85; 3.2 | .85/.89/1; 1.7 |
| S4 card dense | .44/.67/.78; 4.2 | .67/.78/.83; 3.7 | 0/.33/.33; 9.0 | .70/.81/.89; 2.7 | .96/.96/1; 1.5 |
| **S5 centroid** | **.89/.89/.89; 2.7** | .78/**.89/.89; 2.9** | **.67/.67/.67; 5.0** | .85/**.89/.93; 2.1** | .96/1/1; 1.4 |
| S6 fused | .33/.56/.78; 3.7 | .67/.83/.83; 3.1 | 0/.33/.33; 6.0 | .70/.81/.89; 2.3 | .96/1/1; 1.4 |
| S7 keyword (24 q) | .50/.50/.50; 10.5 | .50/.75/.75; 7.9 | .5/.5/.5; 21.5 | .58/.67/.67; 6.5 | .92/1/1; 1.5 |

The hard cases are all Persian and mostly in RPT-fa:
- **RPT-fa-T3-002** (3 gold docs): worst gold rank S1 9, S2 9, S3 15, S4 18, S5 11, S6 7, S7 41.
- **RPT-fa-T3-003:** S1 11, S5 2.
- **BOOK-fa-T3-002:** S1 7, S4 9.

Every English question is solved at R@2 by S1 and S2.

## 2. Shortlist ceiling at M = min(pool, 12): the best an LLM router over cards could do

| signal | questions with all gold docs in the shortlist | RPT-fa | T3 |
|---|---:|---:|---:|
| S1, S2, S5, S6 | **54/54** | 9/9 | 18/18 |
| S4 card dense | 53/54 | 8/9 | 17/18 |
| S3 card BM25 | 52/54 | 7/9 | 16/18 |

Only RPT-fa has more than 12 documents, so pools of up to 15 documents are trivially 100%. The worst gold rank S6 gives on RPT-fa is 8, so its shortlist has room to spare. S1 reaches rank 11, which is close to the edge.

## 3. Decisive-span retention of the 6,000-char pack (HYBRID-intact packer, same as idea_test)

| arm | all 54 | idea 24 | RPT-fa (9) | T2 (18) | T3 (18) | T3: all gold docs in pack | fa (27) | en (27) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| HYBRID-intact (replay) | 0.729 | 0.748 | 0.556 | 0.623 | 0.583 | 12/18 | 0.722 | 0.736 |
| HYBRID+rerank (reference, not re-run) | 0.807 (saved) | 0.799 (replay) | | | | | | |
| **ORACLE: gold docs + per-doc reservation** | **0.805** | **0.821** | **0.889** | 0.725 | **0.708** | **17/18** | 0.846 | 0.764 |
| ORACLE: gold docs, no reservation | 0.782 | 0.769 | 0.833 | 0.725 | 0.639 | 13/18 | 0.827 | 0.736 |
| whole pool + gold-doc reservation | 0.762 | 0.800 | 0.667 | 0.623 | 0.681 | 17/18 | 0.759 | 0.764 |
| S6 top-2 + reservation + global top-2 escape | 0.710 | 0.692 | 0.389 | 0.580 | 0.625 | 14/18 | 0.679 | 0.741 |
| S6 top-3 + reservation + escape | 0.685 | 0.685 | 0.389 | 0.506 | 0.625 | 15/18 | 0.654 | 0.717 |
| S6 top-5 + reservation + escape | 0.670 | 0.699 | 0.444 | 0.459 | 0.625 | 15/18 | 0.630 | 0.710 |
| S6 top-12, restricted, no reservation | 0.738 | 0.769 | 0.611 | 0.651 | 0.583 | 12/18 | 0.741 | 0.736 |

Measured against the replayed HYBRID-intact on all 54, perfect routing gains:

| slice | HYBRID-intact | oracle routing | gain |
|---|---:|---:|---:|
| all 54 | 0.729 | 0.805 | +0.076, 5 more questions fully retained |
| RPT-fa | 0.556 | 0.889 | +0.33 |
| Persian | 0.722 | 0.846 | +0.12 |
| English | 0.736 | 0.764 | +0.03 |
| BOOK / LAW | | | about 0 |

In BOOK and LAW, the gold documents already rank 1–2. Losses there happen inside the document.

Even with perfect routing, 0.195 of the spans are still lost, and T2 stays at 0.725. That remainder is a within-document ranking problem, and routing cannot fix it.

## Verdict

- **Best non-LLM signal:** S5, the document centroid. It uses the prebuilt chunk vectors, so it costs nothing extra.
  - Overall it ties S2 (chunk-max).
  - It is clearly best where routing matters: RPT-fa R@2 0.89 against 0.44–0.67 for the others, T3 R@3 0.89, and mean worst-gold rank 2.7 on RPT-fa.
  - The fused signal S6 is not better than its best parts. Heuristic cards are weak alone: S3 and S4 are the worst signals.
  - Keyword doc-first (S7) is the worst signal of all.
  - Caveat: RPT-fa has only 9 questions and T3-in-RPT-fa only 3.
- **LLM-router ceiling:** a 12-card shortlist contains every gold document in 54/54 questions (18/18 T3) with S1, S2, S5 or S6. The shortlist does not limit the router, so any routing error will be the router's own.
- **Headroom:** yes, but narrow and concentrated.
  - Oracle routing with per-document reservation reaches 0.805 on all 54 and 0.821 on the idea 24. That is about HYBRID+rerank's level (0.807 / 0.799) and +0.08 over HYBRID-intact.
  - Nearly all of that gain is in RPT-fa (0.56 → 0.89) and in T3 document coverage (12 → 17 of 18).
  - Routing that does not pick exactly the gold documents loses retention. Without an LLM, S6 top-k routing with reservation (0.67–0.71) is WORSE than HYBRID, because reserving packages for wrong documents crowds out good ones.
- **What this means for LEX6:**
  - LEX6 can only beat HYBRID+rerank if the router is nearly exact on the large Persian pool.
  - Its gain is capped at about +0.08 retention. That is roughly what the reranker already delivers without an LLM call.
  - Cheaper to try first: HYBRID plus reservation for the top-2 documents by S5, which needs no call. Whole-pool gold-doc reservation shows the potential at 0.762, with T3 at 17/18.
