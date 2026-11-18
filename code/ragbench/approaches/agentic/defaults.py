"""AGENTIC tunables that are NOT fairness-critical. The fairness-critical values stay in config.yaml:
generation.*, evidence.* (the 6000-char budget used for the final answer call) and the call cap
caps.agentic_max_calls (read with a fallback to DEFAULTS["max_calls_default"] until the registration
patch in ragbench/INTEGRATION_NOTES.md adds it to config.yaml).
"""

PLAN_PROMPT_VERSION = "agentic-plan-v1"
VERIFY_PROMPT_VERSION = "agentic-verify-v2"   # v2: keyed reply s1..sn (no skipped snippets), axis numbers named
SYNTH_PROMPT_VERSION = "agentic-synth-v1"

KINDS = ("single", "comparison", "timeline")
UNIT_KINDS = ("count", "percent", "currency", "quantity", "ratio", "date", "text")
ANSWER_MODES = ("shared", "synth")

DEFAULTS = {
    # ---- budget -------------------------------------------------------------------------------
    "max_calls_default": 8,            # used only when config caps.agentic_max_calls is absent

    # ---- 1. plan --------------------------------------------------------------------------------
    "plan_num_predict": 420,
    "plan_repair_call": True,          # one repair call when the plan reply is not usable JSON (budget allowing)
    "max_terms_per_lang": 6,
    "max_entity_terms": 4,
    "max_term_chars": 40,
    "max_target_docs": 12,             # documents searched per question (by year order) after metadata selection
    "max_companies_listed": 60,        # company lines shown to the planner (the rest by count)

    # ---- 2. search (no LLM) ---------------------------------------------------------------------
    "rg_max_hits": 6000,               # hit lines parsed per search
    "rg_timeout_s": 120,
    "window_chars": 700,               # a candidate window's text (neighbouring units) is at most this long
    "window_neighbours": 2,            # units added on each side while no candidate number is in the window
    "neighbour_chars": 250,            # only neighbours this short are added (labels, KPI lines); x2 in the gap round
    "clip_chars": 560,                 # a long unit is clipped to this many chars around the term hit
    "table_rows_after": 1,             # table window: header row + hit row + this many following rows
    "lead_units": 6,                   # a title hit: the section's first units (window_chars cap)
    "per_doc_cap": 4,                  # candidates verified per document when several documents are targeted
    "single_doc_cap": 12,              # ... when one document is targeted
    "max_candidates": 32,              # global candidate cap (first verify round; 4 batches of 8)
    "max_windows_per_unit": 2,         # clips per long unit

    # ---- 3. verify --------------------------------------------------------------------------------
    "verify_batch_size": 8,
    "verify_num_predict_base": 96,
    "verify_num_predict_per_item": 120,
    "min_quote_letters": 2,            # a quote must carry a label, not only a number
    "guard": True,                     # the code guard (flag no_guard turns it off: ablation only)
    "quote_term_guard": True,          # numeric questions: the quote must contain a metric term (quote_without_term)
    "reattribute": True,               # a value filed under the wrong snippet key is accepted for the batch snippet
                                       # whose text passes the whole guard (the verbatim check decides, not the key)

    # ---- 4. gap fill ------------------------------------------------------------------------------
    "gap_fill": True,
    "gap_window_chars": 1200,
    "gap_window_neighbours": 5,
    "gap_per_doc": 3,
    "gap_max_candidates": 8,
    "gap_max_batches": 1,

    # ---- 5. reconcile + answer -------------------------------------------------------------------
    "answer_mode": "shared",           # shared = Approach.generate_answer (the study's shared prompt); synth = own prompt
    "table_rows_max": 12,
    "single_rows_max": 3,              # a single-fact table shows at most this many verified rows (best first)
    "unverified_fallback": True,       # nothing verified: the answer call sees the top unverified windows
    "unverified_fallback_items": 6,
    "restatement_rel_tol": 0.005,      # a restated value differing by more than this share is a conflict
    "trace_candidates_max": 40,
    "trace_text_chars": 400,
}

FLAGS = ("no_guard", "no_gap_fill", "line_windows", "no_doc_selection", "synth_answer")
