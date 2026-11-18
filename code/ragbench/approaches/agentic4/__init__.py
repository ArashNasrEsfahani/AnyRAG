"""AGENTIC v4: AGENTIC3 plus a graph retrieval path over typed facts (kb/labels.py + kb/graph.py) - facts found by
their metric and entity NAMES in every spelling, each target report's members of the seed families as the soft
cross-year hop, qualifier-aware scoring, graph slots in the merge, and graph qualifiers feeding the reconciliation.
`no_graph` is AGENTIC3 exactly. At most 4 model calls and 60 s of query clock per question.

    from ragbench.approaches.agentic4 import GraphAgent
    ap = GraphAgent(flags={"no_vectors": True})
"""
from .approach import GraphAgent, parse_flags
from .defaults import DEFAULTS, FLAGS, VALUE_FLAGS
