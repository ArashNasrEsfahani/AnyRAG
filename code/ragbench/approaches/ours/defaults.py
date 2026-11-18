"""OURS tunables that are NOT fairness-critical (the evidence budget, generation options,
call cap and pattern count limits live only in config.yaml: evidence.*, generation.*,
caps.ours_max_calls, caps.ours_patterns_min/max).

Anything here that changes the enriched store is part of EnrichedGrep.config_signature()
and therefore of the index fingerprint.
"""

PROMPT_VERSION = "ours-enrich-v2"      # bump when the enrichment prompt/schema changes (invalidates the cache)
# v1 -> v2: table sections hit the 1200-token cap (plus a repair call, ~30 s/section); v2 asks for
# short keywords, fewer synonyms/facts and bounds the arrays with maxItems in the JSON schema.
PLAN_PROMPT_VERSION = "ours-plan-v3"   # v2: one term per pattern, years separate, copy the question's words
# v3: one pattern list per language in the JSON schema; the language check (document + question language) is
# enforced in code with one repair call; mixed-script alternations are split by script; caps.ours_patterns_max per
# language list; a doc-scoped question's prompt says the search already runs inside the named report(s).
# (work/runs/dev_fix1_attempt1 holds an intermediate v3 draft with the cap shared by both languages and no scope sentence.)

DEFAULTS = {
    # ---- index time: enrichment ------------------------------------------------------
    "enrich_input_chars": 6000,        # a section body longer than this is enriched in parts (unit boundaries)
    "enrich_max_parts": 6,             # parts beyond this are not enriched (still searchable as body)
    "enrich_min_chars": 40,            # sections whose own body is shorter get a title-only header (no call)
    "enrich_num_predict": 1000,        # completion cap of one enrichment call
    "enrich_attempts": 2,              # first try + one repair try on bad/truncated JSON
    "max_keywords": 8,                 # per language (also the schema maxItems)
    "max_synonyms": 6,                 # per language
    "min_questions": 3,
    "max_questions": 5,
    "max_facts_per_part": 12,
    "field_max_chars": 120,            # longer keyword/metric strings are cut
    # no_normalization: False = reuse the enrichment computed from normalised section text
    # (the store body, titles and patterns are still raw); True = enrich the raw text again
    # (a second full enrichment pass).
    "no_normalization_raw_enrichment": False,

    # ---- query time: planner ----------------------------------------------------------
    "plan_num_predict": 256,
    "flex_joiners": True,              # with normalisation on: a ZWNJ/space between two Persian letters in a
                                       # pattern matches ZWNJ, space or nothing
    "min_pattern_literal_chars": 2,    # patterns with fewer literal letters/digits are dropped as too broad

    # ---- query time: executor (no LLM) -------------------------------------------------
    "weights": {"title": 3.0, "header": 3.0, "fact": 2.0, "body": 1.0, "path": 0.75},
    "hit_count_bonus": 0.05,           # + bonus * min(hit lines, cap): tie-breaker only
    "hit_count_cap": 20,
    "max_hits_per_pattern": 5000,      # a pattern reaching this is flagged too_broad and ignored if others hit
    "scope_fallback_to_all": True,     # True: scope is a ranking tier (in-scope matches first, then any other
                                       # matched section); False: sections without an in-scope match are dropped
    "max_sections": 12,                # ranked sections considered for evidence
    "window_lines": 3,                 # +- store lines around a body match (blank separator lines count)
    "window_table_header": True,       # a window starting inside a table repeats the table's first line
    "ranking_trace_max": 12,           # ranked sections listed in the trace (retrieved_units = offered items only)
    "answer_when_empty": True,         # still make the shared answer call when nothing was found
    "question_words_fallback": True,   # after plan + widened plan both match nothing: grep the question's content
                                       # words (program only, no model call)
    # What a fact-granularity item shows (the matched fact lines still rank the sections either way):
    #   "context" (default since 2026-09-15 dev_full): for each matched fact, the verbatim source text holding its
    #             value, found by value matching in the section's own units: a table's header row + the matching
    #             row(s) (with the table's caption), a short unit whole, or the sentence(s) of a long unit. Shared
    #             sources are shown once; the bare fact line only when no unit holds the value. The section title is
    #             the last element of the item header ("[n] report > ... > section title").
    #   "bare"    (ablation facts_bare; the dev_fix1 behaviour): the fact lines "metric | value | unit | period",
    #             each followed by <= 2 verbatim source lines holding its number (fact_source_lines).
    "fact_evidence": "context",
    "fact_source_lines": True,         # bare mode: a fact line is followed by the verbatim source line(s) holding its number
    "fact_context_unit_chars": 400,    # context mode: a source unit up to this long is shown whole
    "fact_context_max_rows": 2,        # context mode: table rows per fact (best metric-word overlap first)
    "fact_context_max_sentences": 2,   # context mode: sentences per fact from a longer unit
    "fact_context_sentence_chars": 500,  # context mode: a longer sentence is clipped around the value
    "fact_context_caption_chars": 250,   # context mode: a table's caption unit up to this long is shown above it
}

FACT_EVIDENCE = ("context", "bare")

GRANULARITIES = ("auto", "always_section", "always_window")
SCOPES = ("headers", "facts", "body", "all")
EXPECTS = ("number", "text")
