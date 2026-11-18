"""AGENTIC: explicit, budgeted agentic pipeline (plan -> term-number proximity search -> batched verification with
a verbatim code guard -> one gap-fill round -> reconciliation table + answer). See approach.py.

    from ragbench.approaches.agentic import AgenticPipeline
    ap = AgenticPipeline(flags={"no_guard": True})
"""
from .approach import AgenticPipeline, CallBudget, parse_flags
from .defaults import DEFAULTS, FLAGS

__all__ = ["AgenticPipeline", "CallBudget", "parse_flags", "DEFAULTS", "FLAGS"]
