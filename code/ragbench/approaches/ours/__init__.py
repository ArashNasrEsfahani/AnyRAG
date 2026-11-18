"""OURS: enriched section index + planned grep, with ablation flags.

    from ragbench.approaches import get_approach
    ap = get_approach("OURS", flags={"no_facts": True})        # or flags="no_facts,granularity=always_window"

Modules: defaults (tunables), indexer (index-time enrichment + enriched store), planner
(query call 1 / widening call), executor (ripgrep, ranking, granularity; no LLM),
approach (EnrichedGrep). Smoke test: python -m ragbench.approaches.ours.smoke_test
"""
from .approach import EnrichedGrep, parse_flags, KNOWN_FLAGS
from .defaults import DEFAULTS, PROMPT_VERSION, PLAN_PROMPT_VERSION

__all__ = ["EnrichedGrep", "parse_flags", "KNOWN_FLAGS", "DEFAULTS", "PROMPT_VERSION", "PLAN_PROMPT_VERSION"]
