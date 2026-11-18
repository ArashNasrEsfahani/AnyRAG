"""AGENTIC v3: a persistent typed knowledge base (`ragbench.kb`) as the first retrieval path, with a budgeted
agent on top. Plan with facets (v1 planner, one added field) -> SQL + regex retrieval and tree expansion over
`kb/query.py` -> a sufficiency check that picks FAST | STANDARD | ESCALATED | TIMELINE -> the v1 batched verifier
and verbatim guard (v2's two repairs, plus an escape hatch for the synonym gap that destroyed 7 of v2's 18 missed
lookups) -> reconciliation and gap fill in SQL -> the shared answer prompt -> write-back into `verified_facts`.
At most 4 model calls and 60 s of query clock per question.

    from ragbench.approaches.agentic3 import TypedKBAgent
    ap = TypedKBAgent(flags={"no_fastpath": True})
"""
from .approach import TypedKBAgent, parse_flags
from .defaults import DEFAULTS, FLAGS, PLAN_PROMPT_VERSION, VALUE_FLAGS
from .reconcile_sql import RECONCILE_SQL_VERSION
from .writeback import WRITEBACK_VERSION

__all__ = ["TypedKBAgent", "parse_flags", "DEFAULTS", "FLAGS", "VALUE_FLAGS", "PLAN_PROMPT_VERSION",
           "RECONCILE_SQL_VERSION", "WRITEBACK_VERSION"]
