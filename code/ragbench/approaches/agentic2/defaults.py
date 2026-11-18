"""AGENTIC2 tunables that are NOT fairness-critical. The fairness-critical values stay in config.yaml: generation.*,
evidence.* (the shared 6000-char budget of the answer call), caps.agentic2_max_calls (4) and caps.agentic2_max_seconds
(60). Everything below is copied into every record's trace.params.

AGENTIC2 was designed on 2026-09-16 AFTER the core test results of AGENTIC v1, B3 and the others had been seen (a
disclosed limitation); its defaults were then set on the DEV files only (dev_core.jsonl, dev_core_timeline.jsonl,
pilot_timeline.jsonl).
"""
from ..agentic.defaults import DEFAULTS as V1_DEFAULTS

TREE_VERSION = "agentic2-tree-v1"          # passage (leaf) segmentation + embed text; part of the embedding cache key
FACTDB_VERSION = "agentic2-factdb-v1"      # L0 fact rows per document (approaches/ours/lite.py FACTX_VERSION inside)
RANK_VERSION = "agentic2-rank-v1"

FLAGS = ("no_vectors", "no_factdb", "no_tree", "no_verify", "candidates")
VALUE_FLAGS = ("candidates",)              # take a value: candidates=<int>

DEFAULTS = dict(V1_DEFAULTS)               # the v1 planner / verifier / reconciler read these keys
DEFAULTS.update({
    # ---- budget (config caps win; these are fallbacks) ----------------------------------------------------------
    "max_calls_default": 4,
    "max_seconds_default": 60.0,
    "plan_repair_call": True,               # only when plan + repair + one verify + answer still fit in the cap
    # time estimates, calibrated on the dev run t1 (2026-09-16): a verify call of 12 candidates took 3.8-18.1 s (up to
    # 1,050 output tokens), an answer call <= 2.9 s (<= 125 tokens), a plan call <= 3.3 s; a 768-token timeline answer
    # would take ~14 s at the measured ~55 tokens/s
    "answer_reserve_s": 6.0,                # seconds kept for the answer call of a single-fact plan
    "answer_reserve_timeline_s": 15.0,      # ... of a multi-period plan (timeline / comparison)
    "verify_s_per_candidate": 1.0,          # estimate of a verify call: fixed + per candidate
    "verify_s_fixed": 3.0,

    # ---- 2a. lazy embeddings (tree leaves = passages) --------------------------------------------------------------
    "passage_chars": 520,                   # consecutive units of one section are merged up to this many chars
    "passage_max_chars": 900,               # a unit longer than this is cut into windows of passage_chars
    "passage_overlap_chars": 120,
    "embed_with_path": True,                # embed "<doc title> > <section path>\n<text>" (as B3 does for chunks)

    # ---- 2b. vector search -------------------------------------------------------------------------------------------
    "vec_top_k_single": 12,                 # passages kept when one document is targeted
    "vec_top_k_per_doc": 5,                 # ... per document when several are targeted
    "vec_terms_query": True,                # a second query vector from entity + metric + terms (same embed call)
    "vec_terms_weight": 0.35,               # score = (1 - w) * sim(question) + w * sim(terms)

    # ---- 2c. tree expansion ------------------------------------------------------------------------------------------
    "window_chars": 700,                    # a candidate's text after expansion
    "neighbour_chars": 250,                 # only neighbours this short are added (labels, KPI lines, captions)
    "window_neighbours": 2,
    "table_max_chars": 900,

    # ---- 2d. fact DB -------------------------------------------------------------------------------------------------
    "factdb_max_rows": 24,                  # matching rows turned into candidates (best first)
    "factdb_rows_per_doc": 6,

    # ---- 2e. rank + cap -------------------------------------------------------------------------------------------------
    "max_candidates": 12,                   # single-fact questions: one verify call
    "max_candidates_multi": 24,             # timeline / comparison over several documents: at most two verify calls
    "verify_batch_max": 12,                 # candidates per verify call (candidates=32 -> two calls of 16)
    "max_verify_calls": 2,
    "number_bonus": 0.05,                   # a number of the expected kind in the candidate
    "factdb_bonus": 0.08,                   # a fact-DB row matched inside the candidate
    "term_bonus": 0.03,                     # a plan term in the candidate text

    # ---- 3. verify (v1 verifier + guard) --------------------------------------------------------------------------------
    "thousands_fix": True,                  # "2،788،550" (Arabic comma as thousands separator) shown as "2,788,550"
    "label_completion": True,               # a number-only quote is re-guarded with the adjacent label line added

    # ---- 4. evidence ---------------------------------------------------------------------------------------------------
    "fill_unverified": True,                # after the verified table + its source windows, the budget is filled with
                                            # the remaining candidates in rank order (marked unverified)
    "unverified_fallback_items": 8,
})
