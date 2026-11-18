"""AGENTIC4 tunables that are NOT fairness-critical. Fairness-critical values stay in config.yaml: generation.*,
evidence.* (the shared 6000-char budget), caps.agentic4_max_calls (4) and caps.agentic4_max_seconds (60).
Everything below is copied into every record's trace.params.

AGENTIC4 is AGENTIC3 with a GRAPH retrieval path added (graphret.py over kb/graph.py and kb/labels.py): v3's plan,
verification, SQL reconciliation, gap fill and the shared answer call are inherited unchanged, so every difference
between the two in a paired comparison is the graph. `no_graph` turns the path off and IS v3 (same pool, same merge).

TEST-BLINDNESS. Nothing in this file was set from a test question or a test score. The graph layer (labels.py,
graph.py) was designed from the knowledge base's own label statistics and the dev files; its defaults are set on
the dev files only (pilot.jsonl, pilot_timeline.jsonl). v4 inherits v3's defaults, and with them v3's TEST-SET
DISCLOSURE (agentic3/defaults.py): five v3 mechanisms were designed from analyses that read test gold answers.
"""
from ..agentic3.defaults import DEFAULTS as V3_DEFAULTS

APPROACH_VERSION = "agentic4-v3"     # v3: the graph works in the GAP round; v2: new windows only, typed drops

FLAGS = ("no_graph", "no_family", "no_graph_quals", "no_cards", "no_vectors", "no_facets", "no_concepts",
         "candidates")
VALUE_FLAGS = ("candidates",)

DEFAULTS = dict(V3_DEFAULTS)
DEFAULTS.update({
    # ---- merge: the graph path keeps slots, as the vector path does (a graph score and a fact score are not the
    # same quantity, and the cap of 12 - not retrieval - is where v3 lost its evidence: retrieval found the gold
    # unit in 59 of 59 wide-dev records, selection at 12 kept 0.703 of it) ------------------------------------------
    # SET ON THE DEV REPLAY (tools/v4_replay.py, 67 records, no model call), 2026-09-21. Packed evidence recall,
    # v3 0.772 against: graph 4/8 slots + vec 5 -> 0.670; graph 2/4 + vec 8 -> 0.749; graph in the pool with NO
    # slot -> 0.750. The graph path never beat v3's pool on ONE report: it re-finds the facts the raw-label path
    # finds (7 of 67 lookup gold units were new), and a vocabulary gap (a common name asked, a trade name written) is
    # one no lexical structure bridges. Without vectors it cannot replace them either: 0.540 against v3's 0.515
    # without and 0.772 with. So on ONE report the graph path is OFF and v4 is v3 exactly; it runs where a series
    # needs every report represented.
    "graph_single_doc": False,         # add the graph pool to a one-report plan at all
    "graph_min_slots": 0,              # one target report (cap 12) - unused while graph_single_doc is False
    "graph_min_slots_multi": 0,        # several reports (cap 24). v1/v2 used 4. SET TO 0 on the dev GPU run of v2
                                       # (2026-09-21): a several-report plan verifies only its first 12 candidates in
                                       # round one (the second call is held for the gap round), so 4 graph windows at
                                       # the front took 4 of v3's 12 verification seats; the graph's windows held gold
                                       # in 19% of cases against 27% for v3's, and cafebazaar-01 lost its 1399 value.
                                       # The graph now works in the gap round (graph_gap below).
    "graph_per_period": True,          # several reports: the first graph slots go one per report, in report order
    "vec_min_slots": 8,                # v3's value (tools/v3_recall_pack.py sweep), unchanged

    # ---- card routing (only when the scope is broad and no company is named) ------------------------------------
    "cards_route_min_docs": 6,
    "cards_route_k": 5,

    # ---- seeds -------------------------------------------------------------------------------------------------
    "graph_max_seeds": 60,
    "w_g_cover": 4.0,                  # x the share of the plan's content words the metric name holds
    "w_g_word": 1.0,                   # per whole-word term hit on the metric name (at most 3)
    "w_g_solid": 0.6,                  # per hit that exists only in the solid key (at most 2)
    "w_g_entity": 1.0,                 # per term hit on the entity name (at most 2)
    "w_g_qual_match": 1.5,             # the node carries every qualifier the plan asks for
    "w_g_qual_other": -1.0,            # the node carries a qualifier the plan does not ask for
    "graph_hub_docs": 20,              # a family in more documents than this is a hub: not walked, and penalised
    "w_g_hub": -0.1,                   # per document above the hub cap
    "graph_idf": True,                 # coverage weighted by each word's IDF over the metric names
    "w_g_entity_match": 2.0,           # the plan names an entity and this fact's row IS it
    "w_g_entity_other": -2.0,          # the plan names an entity and this fact's row is ANOTHER one (dev: the graph
                                       # ranked one country's row of a table above the row the question asked for)
    "graph_text_slots": 0,             # graph slots when the plan expects TEXT (locate, multi-section): a fact graph
                                       # cannot reach a paragraph, and reserving slots there cost locate 0.78 -> 0.49
                                       # on the dev replay

    # ---- concepts (kb/concepts.py): bge-m3 vectors of the metric NAMES, embedded lazily and cached ---------------
    "graph_concepts": True,            # seeds and the cross-year walk also by MEANING (flag no_concepts turns off)
    "concept_per_doc": 6,              # names per report kept for the plan's terms
    "concept_min_sim": 0.60,           # the probe: gold names 0.607-0.747, the best distractor 0.562
    "w_g_concept": 4.0,                # x the cosine of the name to the plan's terms
    "concept_walk_min_sim": 0.65,      # a seed name's closest name in another report must be at least this close

    # ---- family walk (several reports) -------------------------------------------------------------------------
    "graph_max_families": 4,           # the best seed families walked
    "family_min_overlap": 0.5,         # Jaccard of the families' words: «راننده» -> «راننده کاربر» is 0.5
    "w_g_walk": 0.5,                   # a walked fact scores its own match + this x overlap x its family's best seed

    # ---- v2, from the dev GPU run of v1 (2026-09-21, work/analysis/V4_DEV.md) -----------------------------------
    # v1 tied v3 on the wide dev set (84.4 = 84.4) and LOST on the dev timelines (years covered 0.708 -> 0.656),
    # while v1 without concepts gained (0.771). Reading the answers: graph candidates took 3-12 of the 24 multi-report
    # slots (4 were reserved), because the graph's scores (concept term included) outgrew v3's and the merge ranked
    # them together; the windows they brought were NEAR-MISS measures and restated years. Each rule below answers one
    # observed failure; each can be switched back for the replay (tools/v4_replay.py) to measure it alone.
    "graph_merge_mode": "new_windows", # v3's selection unchanged + at most the reserved slots of windows v3 did not
                                       # select ("legacy": v1's joint ranking)
    "concept_fallback_only": True,     # concept seeds only in a report whose word seeds cover less than the floor
                                       # below, and no concept score on top of a word match (C-TL-divar-03: ads for
                                       # advertisers)
    "concept_fallback_cover": 0.5,     # the word coverage that counts as "words found it"
    "graph_drop_kind_conflict": True,  # no graph window whose unit kind contradicts the plan's (C-TL-divar-02: a
                                       # count of rejected ads offered for the rejection rate)
    "graph_drop_restated": True,       # no graph window giving another target report's year (C-TL-divar-02: the 1400
                                       # report's «1399: 18%» over the 1399 report's own 20%)

    # ---- v3: the graph in the GAP round (approach.py gap_batch / graph_gap) -------------------------------------
    "graph_gap": True,                 # the years still missing after round one get the graph's candidates in the
                                       # call v3 holds back for its gap round, after v3's own gap candidates
    "graph_gap_per_year": 2,           # windows per missing year's report
    "graph_gap_max_words": 8,          # a verified row's label / definition longer than this is not used as a seed

    # ---- reconciliation ----------------------------------------------------------------------------------------
    "graph_quals_to_def_class": True,  # a verified row the verifier left 'unspecified' takes its node's qualifier
})

GRAPH_DEF_CLASSES = ("active", "registered", "total", "accounts", "fulltime", "parttime", "new", "monthly", "daily",
                     "subgroup")        # the qualifiers reconcile.DEF_CLASSES knows; only these may become a def_class
