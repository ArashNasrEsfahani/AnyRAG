"""LEX3 tunables (dev defaults, not optima; nothing here was set from a test question or score). Fairness-critical
values stay in config.yaml: generation.*, evidence.max_chars (the shared 6000-char budget), caps.lex_max_calls (3).
Everything below is copied into every record's trace.params.

LEX3 = LEX2's lexical machinery re-planned around hybrid feedback (Astra's items 3, 5, 6, 8; UPGRADE_PLAN.md):

    0 question   (code) BM25 + dense over the whole allowed pool, top `hybrid_depth` per channel, from the ORIGINAL
                 question; these candidates survive whatever the planner does
    1 interpret  (1 call) entities, explicit periods, operation, <= 3 facets (the requirements' metric side)
    2 route      (code) over > `route_min_docs` reports: card score + a capped contribution of each report's best
                 1-2 question chunks; shortlist <= `shortlist_max`; explicit entity/period matches kept; pooled
                 question candidates outside the shortlist stay in as the escape channel
    3 feedback   (code) a bounded phrase list from the retrieved evidence (headings, captions, table labels, clauses),
                 each with an ID and its source, spread over reports and facets
    4 pick       (1 call) phrase IDs per facet (+ bounded added terms, marked not observed)
    5 lexical    (code) picked phrases -> LEX's Persian-aware regexes over the shortlist (+ lower-weight morphology,
                 issuer-only name demotion)
    6 fuse       (code) equal-weight RRF over regions, one contribution per channel per region; evidence blocks
                 (ragbench.evidence.build_blocks) with one block reserved per requirement; one bounded follow-up
                 retrieval per missing requirement
    7 answer     (1 call) the shared answer prompt over the shared budget
"""

APPROACH_VERSION = "lex3-v1"
PROMPT_VERSION = "lex3-prompts-v1"

# ablations (value True):
#   open_corpus   the question's own report list is withheld: routing must find the reports among all of the set's
#   no_pick       no call 2: the call-1 facet terms are compiled directly (the "LEX without phrase selection" control)
#   no_lexical    no lexical channel at all: question hybrid + the same interpretation, blocks and follow-up
#   no_morph      no lower-weight morphological matches
#   no_followup   no follow-up retrieval for a missing requirement
#   no_escape     pooled question candidates outside the shortlist are dropped (routing without the escape channel)
#   doc_card / section_expand   ragbench/variants.py options (off by default)
FLAGS = ("open_corpus", "no_pick", "no_lexical", "no_morph", "no_followup", "no_escape", "doc_card", "section_expand")

DEGRADED_MODES = ("none", "empty_plan", "no_shortlist", "pick_failed", "pick_empty")

DEFAULTS = {
    "max_calls_default": 3,
    "max_seconds_default": 60,

    # ---- 0. question channels -----------------------------------------------------------------------------------
    "hybrid_depth": 40,                 # per-channel depth (bm25, dense) from the original question
    "hybrid_k": 80,                     # fused hits kept (every chunk in either channel's top 40)

    # ---- 1. interpret -------------------------------------------------------------------------------------------
    "interp_num_predict": 320,
    "max_facets": 3,
    "facet_terms_max": 5,
    "term_max_words": 5,
    "max_entities": 6,
    "max_periods": 12,

    # ---- 2. routing ---------------------------------------------------------------------------------------------
    "route_min_docs": 8,                # route only when the pool holds more reports than this
    "shortlist_max": 8,
    "forced_max": 12,                   # explicit entity/period matches kept even past shortlist_max, up to this
    "route_best_chunks": 2,             # a report's chunk contribution = its best 1-2 question chunks, not all
    "w_route_card": 1.0,                # card score, normalised to [0, 1] over the pool
    "w_route_chunk": 1.0,               # capped chunk contribution, normalised to [0, 1]
    "bm25_k1": 1.2,                     # LEX's card BM25 (search.route)
    "bm25_b": 0.75,
    "w_route_name": 3.0,
    "w_route_year": 2.0,
    "route_k": 8,
    "route_max": 8,

    # ---- 3. feedback vocabulary ---------------------------------------------------------------------------------
    "vocab_max": 40,
    "vocab_hits": 24,                   # question hits (in the shortlist) mined for phrases
    "vocab_per_doc_min": 3,             # each report its own share first
    "vocab_per_facet_min": 3,           # each facet its own share first
    "vocab_min_words": 1,
    "vocab_max_words": 7,
    "vocab_min_letters": 3,

    # ---- 4. pick ------------------------------------------------------------------------------------------------
    "pick_num_predict": 260,
    "pick_max": 6,                      # phrase IDs per facet
    "add_max": 2,                       # added (not observed) terms per facet

    # ---- 5. lexical ---------------------------------------------------------------------------------------------
    "w_picked": 1.0,                    # a phrase picked from the list (observed in the reports)
    "w_added": 0.7,                     # a term the model added (not observed)
    "w_keyword": 0.5,                   # a call-1 facet term
    "w_morph": 0.5,                     # a morphological (not literal) match, x the term's own weight
    "morph_min_stem": 4,
    "w_name": 0.7,                      # a window naming an entity the question names ...
    "w_issuer_name": 0.15,              # ... that is the issuer of the window's own report (boilerplate there)
    # LEX's window machinery (search.windows / compile_terms read these)
    "w_relaxed": 0.6,
    "w_weak": 0.3,
    "w_number": 0.5,
    "w_year": 0.5,
    "min_term_letters": 2,
    "window_chars": 500,
    "table_rows_around": 1,
    "table_window_factor": 1.8,
    "lex_windows": 40,                  # lexical windows entering the fusion

    # ---- 6. fusion, blocks, follow-up ---------------------------------------------------------------------------
    "rrf_k": 60,
    "region_overlap": 0.5,              # two candidates are one region when this share of the smaller overlaps
    "fused_candidates": 60,             # regions offered to build_blocks
    "max_requirements": 8,
    "followup_max": 2,                  # follow-up retrievals (one per missing requirement)
    "followup_k": 6,                    # candidates each follow-up adds
}
