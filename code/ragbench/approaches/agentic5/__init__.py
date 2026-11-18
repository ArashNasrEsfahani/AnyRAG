"""AGENTIC v5: budgeted verification and reconciliation of an explicit evidence-requirement matrix (2026-09-24).

Plan with a requirement matrix (entity x metric x period, operation) -> recoverable routing over the pool (cards +
capped best-chunk evidence, shortlist <= 8, escape candidates) -> allocation across requirements, then channels ->
typed verification (<= 12 in batch 1) -> entity/metric-aware reconciliation -> an optional second batch for missing
or conflicting requirements -> deterministic arithmetic -> a joint evidence table + snippets with citations numbered
after packing -> the shared answer call. At most 4 model calls; cold and isolated by default.

    from ragbench.approaches.agentic5 import RequirementAgent
    ap = RequirementAgent(flags={"no_escape": True})
"""
from .approach import RequirementAgent, parse_flags
from .defaults import APPROACH_VERSION, DEFAULTS, FLAGS, PLAN_PROMPT_VERSION, VALUE_FLAGS, VERIFY_PROMPT_VERSION

__all__ = ["RequirementAgent", "parse_flags", "DEFAULTS", "FLAGS", "VALUE_FLAGS", "APPROACH_VERSION",
           "PLAN_PROMPT_VERSION", "VERIFY_PROMPT_VERSION"]
