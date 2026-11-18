"""AGENTIC5 tunables. Layered on AGENTIC3's DEFAULTS (which carry v1/v2 and `kb/query.py` keys) so every retrieval,
expansion and guard key AGENTIC5 reuses keeps its AGENTIC3 value; the keys below are AGENTIC5's own or overrides.

Fairness-critical values stay in config.yaml: generation.*, evidence.* (the shared 6000-char answer budget),
caps.agentic5_max_calls (fallback 4) and caps.agentic5_max_seconds (fallback 60).

AGENTIC5 was designed on 2026-09-24 from the reviewer design in work/analysis/UPGRADES_ASTRA_session.log (table items
1, 5, 7) and work/analysis/UPGRADE_PLAN.md. No numeric default below was tuned on any question set: they are the
reviewer's stated dev starting points (shortlist 8, first verify batch 12, 6000-char joint budget) or plain caps.
"""
from ..agentic3.defaults import DEFAULTS as A3_DEFAULTS

APPROACH_VERSION = "agentic5-v1"
PLAN_PROMPT_VERSION = "agentic5-plan-v1"       # v3's plan prompt + entities / metrics / periods / operation
VERIFY_PROMPT_VERSION = "agentic5-verify-v1"   # typed claims against the requirement matrix
ROUTE_VERSION = "agentic5-route-v1"            # cards + capped best-chunk evidence, shortlist, escape pool
RECONCILE_VERSION = "agentic5-reconcile-v1"    # entity/metric/period/definition/unit identity
EVIDENCE_VERSION = "agentic5-evidence-v1"      # joint table + snippets, citations numbered after packing

OPERATIONS = ("lookup", "series", "comparison", "difference", "ratio", "sum", "list")

# no_hybrid: never use the shared foundation even when it is importable (the AGENTIC3 KB path only);
# no_dense: skip the dense channel; no_escape: drop the pooled escape candidates outside the shortlist;
# no_gap_fill: no requirement-triggered second batch; no_arith: no deterministic arithmetic rows;
# write_back / warm: opt back IN to AGENTIC3's persistence (both off by default: cold and isolated);
# candidates=<n>: the first verify batch size; shortlist=<n>: routed documents kept.
# doc_card / section_expand: the shared method options of ragbench/variants.py (2026-09-24). The card reaches the
# plan / plan_repair calls only; section expansion runs once, after the joint table + snippet fit, is kept only when
# the table, every cited snippet and every reserved item survive, and citations are re-numbered after it (approach.py).
FLAGS = ("no_hybrid", "no_dense", "no_escape", "no_gap_fill", "no_arith", "write_back", "warm", "candidates",
         "shortlist", "doc_card", "section_expand")
VALUE_FLAGS = ("candidates", "shortlist")

DEFAULTS = dict(A3_DEFAULTS)
DEFAULTS.update({
    # ---- budget (config caps win; these are fallbacks) ---------------------------------------------------------
    "max_calls_default": 4,
    "max_seconds_default": 60.0,
    "answer_floor_s": 8.0,              # the answer call always gets at least this read timeout
    "call_min_s": 2.0,                  # a verify call is not started with less than this left (plus the answer)

    # ---- 1. plan: the requirement matrix ----------------------------------------------------------------------
    "max_entities": 4,
    "max_metrics": 3,
    "max_metric_terms": 6,
    "max_periods": 10,                  # a series range is expanded year by year up to this many periods
    "max_requirements": 12,
    "plan_num_predict": 640,

    # ---- 2. routing over the pool -------------------------------------------------------------------------------
    "shortlist_max": 8,
    "route_hits_k": 40,                 # pooled chunk hits per channel (the question's own retrieval)
    "route_best_chunks": 2,             # a document's contribution = its best 1-2 chunk hits (never a vote sum)
    "route_chunk_weight": 1.0,
    "route_card_weight": 1.0,
    "route_entity_bonus": 3.0,          # an explicitly requested entity (company key / name) matches the document
    "route_period_bonus": 1.0,          # ... and a requested period is inside its years
    "escape_max": 4,                    # pooled candidates from OUTSIDE the shortlist kept as a recovery channel
    "expand_max_docs": 4,               # documents added for a requirement no shortlisted document can serve

    # ---- 3. allocation ----------------------------------------------------------------------------------------
    "req_pool_cap": 12,                 # candidates retrieved per requirement before allocation
    "batch1_max": 12,
    "batch2_max": 12,

    # ---- 4. verification -----------------------------------------------------------------------------------------
    "guard_escape": False,              # the model-written-definition escape is REMOVED (reviewer item 7)
    "guard_escape_label": False,
    "source_label_escape": True,        # a quote without the term passes only when the KB fact LABEL carries a
                                        # term AND that label is itself verbatim in the snippet (source text)
    "verify5_num_predict_base": 160,
    "verify5_num_predict_per_item": 150,

    # ---- 5. reconciliation ------------------------------------------------------------------------------------
    "restatement_rel_tol": 0.005,

    # ---- 6. arithmetic ------------------------------------------------------------------------------------------
    "arith": True,

    # ---- 7. evidence ------------------------------------------------------------------------------------------
    "snippet_max_chars": 700,
    "snippet_min_chars": 200,
    "table_def_chars": 60,
    "fill_unverified": True,            # spend budget left after the table on unverified candidates of MISSING reqs

    # ---- 8. cold and isolated -----------------------------------------------------------------------------------
    "write_back": False,
    "warm": False,
    "dense": "prebuilt",                # never embed documents at query time; "off" skips the channel
    "dense_query_embed": True,          # the QUESTION is embedded (one embed call) when prebuilt vectors exist
    "hybrid": True,                     # use ragbench.retrieval.hybrid when importable
    "hybrid_channels": ("bm25", "dense"),
})
