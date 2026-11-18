"""LEX tunables that are NOT fairness-critical. Fairness-critical values stay in config.yaml: generation.*, evidence.*
(the shared 6000-char budget), caps.lex_max_calls (3) and caps.lex_max_seconds (60). Everything below is copied into
every record's trace.params.

LEX is the user's design of 2026-09-22 ("a really strong regex combined with an LLM"), built as one fixed pass:

    1 cards      layer 1: every report is a node - its name, its years and the keywords of everything it holds (section
                 titles, metric names; kb/graph.py doc_cards, built with no model)
    2 keywords   (1 call) the question -> search keywords in the reports' language, names, years, answer kind
    3 route      (code) the keywords against the cards -> the reports to open (only when the scope is not given or is
                 larger than `route_min_docs`; the benchmark gives every question its reports)
    4 pick       (1 call) the reports' OWN phrases that share a word with the keywords (cards + the section titles
                 around a first regex pass) are listed; the model picks the ones that name what is asked
    5 compile    (code) every picked phrase and keyword -> Persian/English-aware regexes (agentic.planner.term_patterns:
                 plural and ezafe forms, ZWNJ / space / no space, yeh and kaf variants through folding)
    6 search     (code) the regexes over the FULL text of the chosen reports, unit by unit
    7 window     (code) each hit -> the text the model reads: its unit and the neighbouring units of the same section
                 up to `window_chars` (a table: its header line and the matching rows)
    8 answer     (1 call) the SHARED answer prompt over the SHARED 6000-char evidence budget

Research notes: work/analysis/REGEX_LLM_2026.md. TEST-BLINDNESS: nothing here was set from a test question or score.
"""

APPROACH_VERSION = "lex-v6"     # v6 (second opinion 2): matched llm_regex arm (keywords kept), regex parser accepts
                                #   arrays and validates types, failure bookkeeping in the trace; v5: upgrade flags + LEX2, card-only index build; lex-v4     # v4 (second opinion): emitted-text credit, table rows ranked, measure-aware AND,
                                #   word boundaries, selection by characters with overlap and reservation; v3     # v3: stems match word starts; v2 (smoke run): AND-of-content-words terms, table rows with their header, IDF-ranked phrase list
PROMPT_VERSION = "lex-prompts-v2"    # v2: no report-furniture keywords

# ablations (value True unless noted):
#   llm_regex      the model writes the regexes itself in call 2 (the user's literal step 3) instead of picking phrases
#                  that code compiles
#   free_keywords  no call 2: the call-1 keywords are compiled directly (no grounding in the reports' vocabulary)
#   fixed_window   +-window/2 raw characters around the hit instead of the structure-aware window
#   title_routing  layer 1 sees only each report's title, company and years, not its keywords
#   open_corpus    the question's own report list is withheld: layer 1 must find the reports among all of them
#   window=<n>     the window size in characters (default `window_chars`)
#   bm25           the BM25 safety net (upgrade; on in LEX2)
#   report_vocab   per-report phrase lists with the reports' years (upgrade; on in LEX2)
#   year_pass      the missing-year pass (upgrade; on in LEX2)
#   no_labels      "lite": cards and phrase list from section titles only (no fact / label layer needed)
#   learned        the window score comes from a model fitted on dev questions (learn.py) instead of the hand-set
#                  weights; needs options["learned_model"] (LEX_OPTIONS), and is only reportable OUT OF FOLD
FLAGS = ("llm_regex", "free_keywords", "fixed_window", "title_routing", "open_corpus", "window", "bm25", "report_vocab",
         "year_pass", "no_labels", "learned",
         "doc_card", "section_expand")          # ragbench/variants.py (2026-09-23), off by default
UPGRADES = ("bm25", "report_vocab", "year_pass")
VALUE_FLAGS = ("window",)

DEFAULTS = {
    "max_calls_default": 3,
    "max_seconds_default": 60,

    # ---- call 1: keywords ------------------------------------------------------------------------------------------
    "kw_num_predict": 220,
    "kw_max": 8,                        # keywords kept from call 1
    "kw_max_words": 4,                  # a longer "keyword" is a sentence, not a search term

    # ---- layer 1: routing over the cards ----------------------------------------------------------------------------
    "route_min_docs": 6,                # route only when the scope holds more reports than this
    "route_k": 3,                       # reports kept by card score ...
    "route_max": 6,                     # ... plus the same company's reports of the question's years, up to this
    "bm25_k1": 1.2,
    "bm25_b": 0.75,
    "w_route_name": 3.0,                # a name the question gives (company, country) in the report's title
    "w_route_year": 2.0,                # a year the question gives is the report's own year

    # ---- call 2: pick the reports' own phrases ------------------------------------------------------------------------
    "vocab_max": 40,                    # phrases shown to the model
    "vocab_max_words": 7,
    "vocab_feedback_windows": 10,       # the section titles of this many first-pass windows join the list
    "pick_max": 8,
    "add_max": 3,
    "pick_num_predict": 160,
    "regex_num_predict": 260,           # llm_regex
    "regex_max": 6,
    "llm_regex_fallback": True,         # llm_regex: when no regex compiles, the call-1 keywords are compiled instead
                                        # (counted in the trace: `regex_fallback`)
    "regex_keep_keywords": True,        # llm_regex (lex-v6, MATCHED arm): the call-1 keywords stay beside the model's
                                        # regexes at w_keyword, exactly as they stay beside the picked phrases

    # ---- search and scoring ----------------------------------------------------------------------------------------
    "w_picked": 1.0,                    # a phrase the model picked (or added)
    "w_relaxed": 0.6,                   # a term matched only by all its words (AND), not as the phrase
    "w_keyword": 0.5,                   # a call-1 keyword, once phrases were picked
    "w_weak": 0.3,                      # a term made only of measure words (تعداد, share, total, ...)
    "w_number": 0.5,                    # the window holds a number and the answer is a number
    "w_year": 0.5,                      # the window names a year the question asks about
    "w_name": 0.7,                      # the window names an entity the question names
    "min_term_letters": 2,

    # ---- windows ------------------------------------------------------------------------------------------------------
    "window_chars": 500,                # structure: the budget of one window; fixed: +- half of it around the hit
    "table_rows_around": 1,             # a table hit: the matching rows +- this many
    "table_window_factor": 1.8,         # a table window may be this much longer (rows are dense)
    "max_windows": 18,                  # windows offered to the pack, chosen by the characters they cost
    "overlap_max": 0.5,                 # a window more than this share of which a chosen window shows is skipped
    "w_card_prior": 1.0,                # routed reports only: the top card's prior, falling linearly with its rank

    # ---- upgrades (plus.py) ---------------------------------------------------------------------------------------
    "bm25_top_k": 20,                   # BM25 chunks entering the fusion
    "rrf_k": 60,                        # reciprocal rank fusion constant (the usual 60)
    "bm25_max_chars": 1200,             # evidence characters for windows ONLY BM25 found (second opinion's cap)
    "rrf_prior_scale": 0.02,            # the card prior moved onto the fused score's scale
    "year_pass_labels": 8,              # labels carried from the supported years
    "year_pass_label_words": 6,         # a longer metric name is not carried

    # ---- learned window score (learn.py; flag `learned`) ----------------------------------------------------------
    "learned_model": None,              # path to the fitted weights (LEX_OPTIONS); required by the flag
    "learn_min_df": 2,                  # a phrase column is kept when it is seen in this many training windows
    "learn_rank_k": 8,                  # the penalty is chosen on gold recall in the top k windows
    "learn_dump": None,                 # tools/lex_learn.py: where to write the training windows
}
