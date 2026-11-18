"""Evaluation: per-record metrics, the LLM judge, aggregation and the results report.

    from ragbench.eval.metrics import score_record, numeric_match, string_match, evidence_recall
    from ragbench.eval.judge import make_judge            # none | openrouter | ollama
    from ragbench.eval.aggregate import aggregate_runs    # results.csv, results.md, comparisons.csv

CLIs (run from D:/KareOne/RAG Paper):
    .venv/Scripts/python.exe -m ragbench.run --approach b2 --questions ... --corpus A --out runs/pilot
    .venv/Scripts/python.exe -m ragbench.eval.aggregate --runs runs/pilot --questions ... --judge none
    .venv/Scripts/python.exe -m ragbench.eval.judge --probe          # ONE paid judge call (auth + JSON shape)
    .venv/Scripts/python.exe -m ragbench.eval.test_eval              # offline smoke test (no model, no credit)
"""
