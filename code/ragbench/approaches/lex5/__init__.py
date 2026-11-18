"""LEX5: LLM-selected hybrid retrieval (lineage: LEX). Converged design: work/analysis/LEX5_DESIGN_FINAL.md.

select.py    candidate views (full / query-hit snippet / head) and the v2 flat replay parser (kept for the replay)
bundles.py   the bundle grammar, its strict parser, exact / greedy / flat packing, fill, HYBRID-intact fallback
approach.py  the Lex5Select approach (run.py --approach lex5) and answer_intact (HYBRID flag intact)
"""
