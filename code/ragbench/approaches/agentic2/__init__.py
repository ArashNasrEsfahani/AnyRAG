"""AGENTIC2 ("AGENTIC v2"): tree + fact DB + verify. Plan (v1 planner) -> lazy bge-m3 vectors of the target documents'
tree leaves + tree expansion + an on-the-fly SQLite fact table (L0 regex facts) -> one (at most two) batched verify
calls with the v1 verbatim guard -> v1 reconciliation table + the shared answer prompt. <= 4 calls, <= 60 s.

    from ragbench.approaches.agentic2 import TreeFactVerify
    ap = TreeFactVerify(flags={"no_factdb": True})
"""
from .approach import TreeFactVerify, parse_flags
from .defaults import DEFAULTS, FLAGS, VALUE_FLAGS

__all__ = ["TreeFactVerify", "parse_flags", "DEFAULTS", "FLAGS", "VALUE_FLAGS"]
