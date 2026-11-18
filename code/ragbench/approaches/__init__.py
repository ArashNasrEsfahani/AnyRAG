"""Approach registry (shared, frozen). Modules are owned by the agents in OWNERSHIP.md;
class names below are part of the contract.

    from ragbench.approaches import get_approach
    ap = get_approach("OURS", flags={"no_facts": True})
    ap = get_approach("OURS", flags={"index_level": "L1"})     # variant "OURS+index_level=L1"
    ap = get_approach("AGENTIC")
    ap = get_approach("AGENTIC2", flags={"candidates": 32})  # variant "AGENTIC2+candidates=32"
    ap = get_approach("AGENTIC3", flags={"verify_gate": True})
"""
import importlib

APPROACHES = {
    "B0": "ragbench.approaches.baselines:NoRetrieval",
    "B1": "ragbench.approaches.baselines:GoldEvidence",
    "B2": "ragbench.approaches.baselines:BM25Retrieval",
    "B3": "ragbench.approaches.baselines:VectorRetrieval",
    "B4": "ragbench.approaches.tree_nav:TreeNavigation",
    "B5": "ragbench.approaches.grep_agent:GrepAgent",
    "OURS": "ragbench.approaches.ours:EnrichedGrep",
    "AGENTIC": "ragbench.approaches.agentic:AgenticPipeline",
    "AGENTIC2": "ragbench.approaches.agentic2:TreeFactVerify",
    "AGENTIC3": "ragbench.approaches.agentic3:TypedKBAgent",
    "AGENTIC4": "ragbench.approaches.agentic4:GraphAgent",
    "AGENTIC5": "ragbench.approaches.agentic5:RequirementAgent",
    "LEX": "ragbench.approaches.lex:LexAgent",
    "LEX2": "ragbench.approaches.lex:Lex2Agent",
    "HYBRID": "ragbench.approaches.hybrid:HybridRetrieval",
    "LEX3": "ragbench.approaches.lex3:Lex3Agent",
    "LEX5": "ragbench.approaches.lex5.approach:Lex5Select",
    "LEX6": "ragbench.approaches.lex6.approach:Lex6",
    "LEX7": "ragbench.approaches.lex7.approach:Lex7",
    "LEX8": "ragbench.approaches.lex8.approach:Lex8",
    "LEX5B": "ragbench.approaches.lex5b.approach:Lex5b",
    "LEX10": "ragbench.approaches.lex10.approach:Lex10",
    "SETR": "ragbench.approaches.setr.approach:SetR",
}

# ablation flags accepted by OURS (value True unless noted)
OURS_FLAGS = ("no_enrichment", "no_facts", "no_normalization", "monolingual_patterns", "granularity", "facts_bare",
              "index_level")
# facts_bare (2026-09-15, dev_full): fact items show the fact lines (dev_fix1 behaviour) instead of their source text.
# granularity takes "always_section" | "always_window" (omitted = "auto", the full method). EnrichedGrep also accepts a
# CLI-style string ("no_facts,granularity=always_window"); the variant name is "OURS+<sorted flags>".
# index_level (2026-09-16): L0 | L1 | L2 cheap index levels (approaches/ours/levels.py); takes a value like granularity
# ("index_level=L1" -> variant "OURS+index_level=L1"); L3 = the full enrichment = the default (flag dropped, variant "OURS").

# ablation flags accepted by AGENTIC (value True) (2026-09-16; approaches/agentic/defaults.py FLAGS)
AGENTIC_FLAGS = ("no_guard", "no_gap_fill", "line_windows", "no_doc_selection", "synth_answer")

# ablation flags accepted by AGENTIC2 ("AGENTIC v2", 2026-09-16; approaches/agentic2/defaults.py FLAGS); candidates takes
# an integer value ("candidates=32" -> variant "AGENTIC2+candidates=32"), the others are True
AGENTIC2_FLAGS = ("no_vectors", "no_factdb", "no_tree", "no_verify", "candidates")

# ablation flags accepted by AGENTIC3 ("AGENTIC v3", 2026-09-20; approaches/agentic3/defaults.py FLAGS), the
# experiments of AGENTIC3_SPEC.md section 8: no_typed (the typed filters and weights off = v2's text fact table),
# no_trigram (full REGEXP scan), no_sql_reconcile (v1's Python reconciler), no_fastpath (never skip verification),
# verify_gate (v2's verification exactly: the hard quote_without_term veto and no verdict-aware ordering),
# vectors_always / no_vectors, no_facets, no_tree_prior (section prior weight 0), warm (read verified_facts back).
# candidates takes an integer ("candidates=32" -> variant "AGENTIC3+candidates=32"), the others are True.
AGENTIC3_FLAGS = ("no_typed", "no_trigram", "no_sql_reconcile", "no_fastpath", "verify_gate", "vectors_always",
                  "no_vectors", "no_facets", "no_tree_prior", "warm", "candidates", "doc_card", "section_expand")
# the two optional method options of ragbench/variants.py (2026-09-23) also apply to the BM25 and vector baselines
# (doc_card has no effect there: they make no planning call)
BASELINE_FLAGS = ("doc_card", "section_expand")
# HYBRID (2026-09-24; approaches/hybrid.py): the B2+B3 RRF control; blocks = evidence from ragbench.evidence.build_blocks
# (variant "HYBRID+blocks"); doc_card / section_expand as for B2/B3
# rerank (2026-09-25; retrieval/rerank.py): the top 50 fused hits re-ordered by the bge-reranker-v2-m3 cross-encoder
# (variant "HYBRID+rerank") - a BASELINE / comparison arm only: LEX, LEX3 and AGENTIC* never use the reranker
# intact (2026-09-26; approaches/lex5/approach.py answer_intact): HYBRID-intact - the same question hybrid (top 40 U 40),
# lex4 intact packages in RRF order with exact / fully-contained dedupe and exact rendered cost (LEX5's fallback)
HYBRID_FLAGS = ("blocks", "doc_card", "section_expand", "rerank", "intact", "kw", "doc")

# LEX5 (2026-09-26; approaches/lex5, work/analysis/LEX5_DESIGN_FINAL.md): LLM-selected hybrid retrieval. Valued flags:
# selector=oss20b|glm, view=full|query_hit|head, top=<int> (24 default, 40 ablation); True flags: no_fill, flat,
# greedy, permute
LEX5_FLAGS = ("selector", "view", "top", "no_fill", "flat", "greedy", "permute")

# LEX6 (2026-09-26; approaches/lex6, work/analysis/LEX6_DESIGN_FINAL.md): document-aware local retrieval with bounded
# evidence replacement. Valued flag: rank=s2|s5 (default s2s5 names no variant); True flags: absent_only, global_only,
# no_anchor, cards, sections, router, select (LEX6+S)
LEX6_FLAGS = ("rank", "absent_only", "global_only", "no_anchor", "cards", "sections", "router", "select")

# LEX7 (2026-09-26; approaches/lex7/approach.py, LEX7.1): RRF(BM25, bge-m3 dense, bge-m3 sparse) + MMR + the gpt-oss-20b
# order fixer v2 (no cross-encoder). True flag: no_fix (the MMR-a1 base order only)
LEX7_FLAGS = ("no_fix",)

# LEX8 (2026-09-26; approaches/lex8, Astra R1): compact source units + chunks, MaxMean fusion over real channel scores,
# budgeted coverage packer; no cross-encoder, no query-time LLM. True flags: ranked (ablation C1), synth (S1)
LEX8_FLAGS = ("ranked", "synth")

# LEX5B (2026-09-26; approaches/lex5b, work/datasets_v2/lex5b): the LEX5 support-bundle selector (glm) over F2 MaxMean
# candidates with a budgeted literal-window view (the configuration admitted offline); no flags

# ablation flags accepted by AGENTIC4 ("AGENTIC v4", 2026-09-21; approaches/agentic4/defaults.py FLAGS): no_graph
# (the graph path, the graph merge and the graph qualifiers all off = AGENTIC3 exactly), no_family (no cross-report
# family walk), no_graph_quals (graph qualifiers do not reach the reconciliation), no_cards (no card routing),
# no_vectors, no_facets, no_concepts (the concept layer: metric-name vectors, kb/concepts.py); candidates takes
# an integer.
AGENTIC4_FLAGS = ("no_graph", "no_family", "no_graph_quals", "no_cards", "no_vectors", "no_facets", "no_concepts",
                  "candidates")

# ablation flags accepted by AGENTIC5 ("AGENTIC v5", 2026-09-24; approaches/agentic5/defaults.py FLAGS): the
# requirement-matrix method. no_hybrid (never the shared foundation, the KB path only), no_dense, no_escape (no pooled
# candidates from outside the shortlist), no_gap_fill (no second verify batch), no_arith; write_back / warm opt back
# IN to persistence (both off by default); candidates=<n> (first verify batch) and shortlist=<n> take integers.
AGENTIC5_FLAGS = ("no_hybrid", "no_dense", "no_escape", "no_gap_fill", "no_arith", "write_back", "warm", "candidates",
                  "shortlist", "doc_card", "section_expand")     # the last two: ragbench/variants.py options

# ablation flags accepted by LEX (2026-09-22; approaches/lex/defaults.py FLAGS): llm_regex (the model writes the regexes),
# free_keywords (no phrase-picking call), fixed_window (+-chars instead of the structure-aware window), title_routing
# (layer 1 on titles only), open_corpus (the question's report list withheld); window takes an integer (characters).
# bm25 / report_vocab / year_pass are the upgrades (all three always on in LEX2, which takes the same flags);
# no_labels = cards and phrase list from section titles only.
# learned = the window score comes from weights fitted on dev questions (lex/learn.py), given by LEX_OPTIONS.
LEX_FLAGS = ("llm_regex", "free_keywords", "fixed_window", "title_routing", "open_corpus", "window", "bm25",
             "report_vocab", "year_pass", "no_labels", "learned", "doc_card", "section_expand")

# ablation flags accepted by LEX3 (2026-09-24; approaches/lex3/defaults.py FLAGS): hybrid-grounded lexical planning.
# open_corpus (routing must find the reports), no_pick (no call 2: facet terms compiled directly - the "without phrase
# selection" control), no_lexical (question hybrid + the same interpretation, blocks and follow-up only), no_morph,
# no_followup, no_escape (no pooled question candidates outside the shortlist); doc_card / section_expand as for LEX.
LEX3_FLAGS = ("open_corpus", "no_pick", "no_lexical", "no_morph", "no_followup", "no_escape", "doc_card",
              "section_expand")


def approach_class(name):
    mod, cls = APPROACHES[name].split(":")
    return getattr(importlib.import_module(mod), cls)


def get_approach(name, cfg=None, client=None, flags=None):
    return approach_class(name)(cfg=cfg, client=client, flags=flags)
