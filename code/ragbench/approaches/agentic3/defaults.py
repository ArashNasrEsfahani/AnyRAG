"""AGENTIC3 tunables that are NOT fairness-critical. The fairness-critical values stay in config.yaml:
generation.*, evidence.* (the shared 6000-char budget of the answer call), caps.agentic3_max_calls (4) and
caps.agentic3_max_seconds (60). Everything below is copied into every record's trace.params.

The dict is layered so nothing drifts: v2's DEFAULTS (which carry v1's planner / verifier / reconciler keys and
v2's expansion and verify keys) first, then `kb/query.py`'s DEFAULTS (the retrieval scores and caps, which the
query module also reads), then the keys that are v3's own - the plan facets, the route thresholds, the guard
escape hatch and the write-back switch.

AGENTIC3 was designed on 2026-09-20 AFTER the core test results of v1, v2 and the baselines had been seen (a
disclosed limitation, the same one v2 carries). Every numeric default below was set on the DEV files.

TEST-SET DISCLOSURE. The two analyses v3 is designed against are NOT test-blind. Both read
`ragbench/data/questions/test_core.jsonl` and judge candidate values against its gold `answer_value`; both also
quote test question wording where the wording is the point:
  - work/analysis/V2_FAILURES.md diagnoses v2's 79 TEST lookup records (26 distinct test questions, split=test,
    from work/runs/core_test). Its section 2.2 decides case by case whether the verifier was RIGHT to reject, by
    checking the rejected value against the test question's gold answer, and names seven test case-settings where
    the `quote_without_term` guard destroyed a correct value. That finding is the direct reason for
    `guard_escape`, `guard_escape_label` and `rank_by_verdict` below, so the guard escape hatch and the
    verdict-aware ordering were designed against test labels.
  - work/analysis/DB_CEILING.md measures the fact table's ceiling over the same 26 test lookups plus 3 dev ones,
    again by matching rows against gold `answer_value`. Its conclusion that the typed columns outvote the label is
    why `w_kind_conflict` and `w_label_cover` were moved off `kb/query.py`'s defaults, although the SIZE of the
    move was then measured by replaying dev plans only.
What is still true: no approach saw a test question at query time, no v3 number was chosen by reading a v3 test
score, and v3 has not been run on the test set. What is not true is that these analyses touched only test records:
they read test questions' gold answers. Two v3 mechanisms therefore carry test-set information by construction,
and their test result has to be reported with that caveat attached.
"""
from ...kb.query import DEFAULTS as Q_DEFAULTS
from ..agentic2.defaults import DEFAULTS as V2_DEFAULTS

PLAN_PROMPT_VERSION = "agentic3-plan-v1"    # v1's plan prompt + the FACETS field (spec 6 stage 1)
ROUTE_VERSION = "agentic3-route-v3"         # v2 (2026-09-20, wide dev set): vectors are merged in on every question
                                            # (`vector_policy`) and the cold FAST gate is evaluated but not acted on
                                            # (`fast_cold_gate`). v1 records are the database-first routing; the
                                            # version is in every record's trace.versions, so the two never mix.
                                            # v3 (2026-09-20, same wide dev set): the vector path's reservation in
                                            # the merge goes from 4 slots to 8 (`vec_min_slots`), which is the same
                                            # decision - how much of the candidate pool the second path holds -
                                            # measured one level further down. It closed the evidence-recall gap to
                                            # AGENTIC2 (0.698 -> 0.781 against 0.760, paired p = 0.0118 against the
                                            # old default). The measurement is under `vec_min_slots` below.
APPROACH_VERSION = "agentic3-v1"

ROUTES = ("FAST", "STANDARD", "ESCALATED", "TIMELINE")

FLAGS = ("no_typed", "no_trigram", "no_sql_reconcile", "no_fastpath", "verify_gate", "vectors_always",
         "no_vectors", "no_facets", "no_tree_prior", "warm", "candidates",
         "doc_card", "section_expand")          # ragbench/variants.py (2026-09-23), off by default
VALUE_FLAGS = ("candidates",)

DEFAULTS = dict(V2_DEFAULTS)
DEFAULTS.update(Q_DEFAULTS)
DEFAULTS.update({
    # ---- budget (config caps win; these are fallbacks) ----------------------------------------------------------
    "max_calls_default": 4,
    "max_seconds_default": 60.0,

    # ---- 1. plan: facets ------------------------------------------------------------------------------------------
    "facets": True,                     # the plan may return up to `max_facets` term groups (flag no_facets)
    "max_facets": 3,
    "max_facet_terms": 4,               # terms kept per facet and per language
    # Candidate slots every facet keeps in the merge (spec 6 stage 4) AND, since round 2, the bar the route uses
    # to call a facet unserved: a facet the merge could not fill is a topic the database did not serve, and the
    # question escalates to the vectors (`route.starved_facets`). One number, one meaning - the slots a facet must
    # have are the slots whose absence sends the question to the second path. Below it on dev: three of the four
    # A-rpt-f-03 settings (the multi-section question facets exist for), each of which answered NOT_FOUND.
    "facet_min_slots": 2,
    "plan_num_predict": 520,            # v1's 420 + room for the facets array

    # ---- 2. database retrieval ------------------------------------------------------------------------------------
    "max_target_docs": 12,
    "trigram_single_doc": False,        # one targeted document: the prefilter costs ~8 ms and saves nothing
    "verified_max_rows": 12,            # warm mode only
    "facet_cap": 12,                    # candidates one facet pass may contribute before the merge
    # Vectors are the SECOND path, so their similarity is put on the fact score's scale rather than the other way
    # round: a passage of similarity 1.0 is worth two matched term concepts (2 * w_term).
    "vec_score_scale": 2.0,
    # ... but pricing is not a seat at the table. Round 1 measured what the two scales actually do on dev: a
    # passage at bge-m3's real similarity scores 1.2 - 1.5, while the WEAKEST fact row still selected in a full
    # pool scores 2.4 - 3.5, because the fact score carries large constant terms (w_kind, w_source_table,
    # w_label_len) that a cosine has no counterpart for. Over the 11 dev questions whose route asked for vectors,
    # the best passage scored below the worst SELECTED fact row in 9 of them: without a reservation those
    # passages are not ranked low, they are absent. The two scores are not commensurable, so the vector path
    # keeps slots instead of competing for them, exactly as a facet does. v2's own dev ablation is why this
    # matters: no_vectors cost A 75 -> 60 and timeline coverage 74 -> 54.
    # Round 2 re-measured the size of the reservation the same way, this time with the vector pass actually run
    # over the 30 dev records (51 wanted values: the key facts, the gold series, the gold answer). Gold values in
    # the SELECTED pool, by `vec_min_slots`: 0 -> 40, 2 -> 42, 4 -> 42, 6 -> 42, 8 -> 41, against 35 with no
    # vector pass at all. Four sits on the flat optimum, and eight already starts crowding out fact rows.
    #
    # EIGHT SINCE 2026-09-20, AND ROUND 2's MEASUREMENT IS SUPERSEDED. Round 2 counted gold VALUES over 30 records
    # of the narrow dev set; the WIDE dev set (pilot.jsonl, 64 records over corpora A, B and C) counts gold UNITS
    # over 59 gold-bearing records and disagrees. The question it was made to answer: v3 reached evidence recall
    # 0.678 against AGENTIC2's 0.760, and a paired sign test over the same 59 records separated that from chance
    # (v2 better on 20 records, v3 on 8, p = 0.036). Three analyses, all replays of the recorded plans against the
    # knowledge base with no model call (`tools/v3_recall_trace.py`, and the pack simulation it is quoted with):
    #   1. WHERE THE GOLD IS. The gold unit is in a targeted document in 59 of 59 records and on a candidate of
    #      the uncapped pool in 0.960 of them. It is never missing from retrieval; it is cut. In the pool as the
    #      run builds it the coverage is 0.866, and the cap of 12 takes it to 0.703. So the loss is selection.
    #   2. THE CAP IS NOT THE LEVER. Raising the cap to 24 raises the SELECTED coverage 0.703 -> 0.820 and leaves
    #      the coverage that survives `pack_evidence`'s 6000-char budget at 0.664, unchanged on every one of the
    #      59 records, because the extra candidates land after the budget is full (12 candidates already render
    #      about 5300 chars, and the pack stops at the first item that does not fit). Ordering the same cap-24
    #      selection gold-first - an oracle no run can have - does pack it (0.819), which is what says the cap is
    #      real but reachable only through the ORDER, never through the count. `candidates=24` was therefore NOT
    #      run on the GPU: the traces already predict it moves nothing.
    #   3. THE RESERVATION IS THE LEVER, because it changes which candidates hold the EARLY slots rather than how
    #      many there are. At the production cap, selected / packed gold coverage by `vec_min_slots`:
    #        0 -> 0.673 / 0.572   2 -> 0.701 / 0.626   4 -> 0.703 / 0.664   6 -> 0.746 / 0.706
    #        8 -> 0.812 / 0.769  10 -> 0.806 / 0.762  12 -> 0.808 / 0.762
    #      Eight is a peak, not a point on a slope: ten and twelve are flat-to-worse and cost three records.
    # Then measured on the GPU over the same 64 records, against a control at the same code version sharing the
    # wide run's call cache (`work/runs/v3recall_ctl` and `work/runs/v3recall_vec8`, 72 of 202 calls were cache
    # hits), with AGENTIC2 as the reference:
    #                     strict   key-fact   ev-recall   ret-recall   abstain   calls   p50 s   model s
    #   v3, slots 4        0.469     0.797      0.698        0.727      11/64    3.16    12.3      855
    #   v3, slots 8        0.500     0.812      0.781        0.834       6/64    3.16    13.3      915
    #   AGENTIC2           0.500     0.812      0.760        0.782       9/64    3.09    11.2      789
    # Evidence recall is what the change was aimed at and it is the axis that clears the project's noise band:
    # paired against the old default, better on 16 records and worse on 4, p = 0.0118. Against AGENTIC2 it is now
    # +0.021 with 11 wins and 9 losses, p = 0.824 - the gap is closed, not reversed. Strict (+2 records) and
    # key-fact (+1) sit INSIDE the noise band (|delta| < 6 records = 0.094 at n = 64) and are NOT claimed; what is
    # claimed is that v3 no longer retrieves less gold evidence than v2, and that it costs +8% of model time.
    "vec_min_slots": 8,                 # candidate slots the vector path keeps when the route ran it
    # VECTORS ARE ALWAYS MERGED IN SINCE 2026-09-20, and that is a correction of v3's central claim. The design
    # said "the database is the first path and vectors are the second", so a single-fact question only escalated
    # to the vector pass when `escalate_check` found the database pool thin. The WIDE dev set (pilot.jsonl, 36 dev
    # questions over corpora A, B and C, 64 records per arm - stage `v3wide`) measured that routing against always
    # merging the vectors in, on the same records and with a shared call cache so the planner was identical:
    #   strict          0.453 both ways - 0 discordant records out of 64, the two arms are the same answer;
    #   key-fact        0.672 -> 0.750; 5 records gained, 0 lost, exact sign test p = 0.062;
    #   locate          0.45 -> 0.73, multi-section 0.40 -> 0.53 (key-fact, the types the database path loses);
    #   timelines       fully correct 0.250 -> 0.375 and false trends 0.250 -> 0.000 on the 8 dev timelines;
    #   cost            p50 11.2 -> 12.1 s, model time 754 -> 798 s (+6%), and the WORST case improves,
    #                   53.1 -> 22.7 s, because a thin database pool is what drove the long verify rounds.
    # Nothing measured on dev favours the database-first routing. The honest finding, and it is the opposite of
    # the design's: the typed database adds reconciliation, typing and SQL to a vector pipeline; it does not
    # replace retrieval. `vector_policy = "when_thin"` restores the old routing for the ablation row (the
    # `AGENTIC3` rows of work/runs/core_dev_wide ARE that arm).
    "vector_policy": "always",          # "always" | "when_thin" (the database-first routing of round 1)
    # A mis-typed plan must not bury a well-labelled row. The planner typed an energy-per-tonne metric
    # (A-rpt-f-02) as `currency`, so every kWh row took w_kind_conflict and every money row took w_kind: the
    # gold row fell to pool rank 15 and was cut by the cap of 12. Replaying the recorded dev plans against the
    # knowledge base with no model call, softening the conflict to half a term concept and paying more for label
    # specificity moved the gold value into the selected pool for 12 of the 12 gold-bearing dev instances,
    # against 9 of 12 with `kb/query.py`'s own defaults (-1.5 / 0.8). The typed columns still rank; they no
    # longer outvote the label, which is what DB_CEILING says the failures are about.
    "w_kind_conflict": -0.5,            # kb/query.py default -1.5
    "w_label_cover": 1.2,               # kb/query.py default 0.8

    # ---- 3. the sufficiency check (route.py) -----------------------------------------------------------------------
    # The FAST gate is tuned for PRECISION: on dev, no_verify cost 2 of 12 lookups, so a candidate leaves the
    # verifier out only when the database agrees with the plan on every axis and nothing else claims another value.
    # THE COLD FAST GATE IS OFF SINCE 2026-09-20 AND IS REPORTED AS A NEGATIVE RESULT. It was designed, tuned for
    # precision and measured, and it fired 0 times in 428 dev records (300 of the round-2 ablation matrix plus the
    # 128 of the wide dev set). `tools/v3_fastgate.py` replays `fast_check` over the recorded candidates under
    # looser thresholds, with no model call, and gives the reason it cannot be rescued: on the 69 dev records that
    # REACH the gate and have a gold number, the rank-1 database candidate's own value is the gold value 2 times
    # (0.03). A gate answers from rank 1, so 0.03 is the ceiling of every threshold on these features, and the
    # relaxations that do fire (2 to 7 records of 44) sit at precision 0.00 to 1.00 on two-record samples. A path
    # that never fires is not a feature and an unmeasured one in the paper is worse, so the claim "a lookup the
    # database settles alone is answered in 2 calls and about 4 s" is withdrawn.
    # `fast_check` still RUNS and its verdict is still written to the trace, because the refusal mix is the
    # finding; what is off is acting on it. The FAST route survives for warm mode only (`warm_check`).
    "fast_cold_gate": False,
    "fast_min_terms": 2,                # distinct term concepts the top fact's label must match
    # ... and the label must be ABOUT them: every term of one language, or `fast_min_cover` of the label's own
    # words covered by term matches. `all_terms` alone is `kb/query.py::Terms.covers_a_language`, which wants
    # EVERY term of a language inside one fact label; the planner expands its terms ("مصرف برق" also yields
    # "مصرف" and "برق"), so a dev plan carries a median of 7 per language against labels of 5 - 6 words, and on
    # dev the top candidate carried it in 3 of 28 questions. It is a sound signal, too rare to be the only one.
    # `fast_min_cover` is the same axis measured on the label: at 0.6 and at 0.4 the gate still fires 0 times on
    # the 20 dev A questions, and at 0.3 it fires once, on a question whose rank-1 candidate is NOT the gold
    # value - which is why it stays at 0.6 rather than being tuned until the path has customers.
    "fast_need_all_terms": True,
    "fast_min_cover": 0.6,              # ... the reachable form of the same axis (`db.label_cover`)
    "fast_need_kind": True,             # unit_kind compatible with the plan's expected kind
    "fast_need_year": True,             # the fact dates itself inside the period (or the plan names no period)
    "fast_source_kinds": ("table", "kpi"),   # () = any source kind
    "fast_margin": 1.0,                 # no candidate with a DIFFERENT value within this much of the top score
    "fast_max_docs": 1,                 # several targeted documents are never a FAST question
    # WARM MODE ONLY (`route.warm_check`, 2026-09-20). The warm lookup is an OR over the plan's expanded terms,
    # so a verified row that shares ONE word with the question comes back; before these two keys existed any such
    # row short-circuited the question to FAST with verify=False, before the pool test and before the fast test.
    # That is the whole mechanism behind three of the four warm-mode losses on dev. A warm row now has to carry
    # every content word of the plan's metric, there have to be at least two of them, they have to agree on one
    # value, and the database's own best candidate must not contradict them.
    "warm_min_rows": 2,                 # one verified row is an anecdote, not a cache hit
    "warm_metric_cover": 1.0,           # share of the plan metric's content words the row's wording must carry
    "escalate_min_candidates": 6,       # fewer database candidates than this -> vectors
    # A pool is thin when its LABELS are thin, not when it is short. `merge` fills the cap from whatever the
    # regex matched, so on dev every question came back with 10 - 12 candidates and the old count-based test
    # escalated 0 of 20 - including A-rpt-f-02 fa-fa, whose twelve candidates each matched one generic term
    # ("مصرف", "تولید") and whose labels were about long-term debt and production tonnage.
    "escalate_strong_terms": 2,         # distinct term concepts that make a candidate evidence rather than noise
    "escalate_min_strong": 3,           # fewer than this many strong candidates -> vectors
    "escalate_text_min": 6,             # a text answer needs this many term-matching candidates to skip vectors
    "timeline_min_per_doc": 2,          # a targeted document with fewer candidates costs a period -> vectors
    "timeline_min_strong": 1,           # ... and one with no strong candidate costs one just as surely
    # An unserved facet escalates on the TIMELINE path too, because a multi-section question is routed by its
    # plan's `kind` and that came back `comparison` for two of the three failing dev settings. The timeline
    # trigger is NOT loosened any further: round 2 measured the alternative both ways and forcing vectors on
    # every dev timeline is a wash (`vectors_always` has the same coverage, 0.667, and replaying the recorded
    # plans with the vector pass gains one gold value on cafebazaar-01 and loses one on divar-03).

    # ---- 5. verification ------------------------------------------------------------------------------------------
    # V2_FAILURES 2.2: `quote_without_term` fired 149 times over 79 lookups and destroyed the correct value in 7 of
    # the 18 misses, always as a synonym gap (employees/staff, تعطیل‌شده/متوقف, rounds/capital raises). The escape
    # hatch keeps the guard for the case it was built for - a number labelled with a DIFFERENT metric - by asking
    # the verifier's own definition, and the fact row's label, for the term the quote does not carry.
    "guard_escape": True,               # flag verify_gate turns it off (v2's hard veto)
    "guard_escape_label": True,         # ... also accept when the fact table's own label carries a plan term
    "rank_by_verdict": True,            # the unverified pack is ordered by the verifier's own judgement first
    "verify_batch_max": 12,
    "max_verify_calls": 2,

    # ---- 6. gap fill ----------------------------------------------------------------------------------------------
    "sql_reconcile": True,              # flag no_sql_reconcile falls back to agentic/reconcile.py
    "gap_fill": True,
    "gap_max_years": 3,                 # gap queries run per question (each is one SELECT, no model call)
    "gap_rows_per_year": 8,
    "gap_verify_call": True,            # a second verify call on what the gap query found, if the clock allows
    # The reconciler's NARROW rule assumes the rows of a period are readings of one metric, so on a FACETED plan
    # whose rows carry no period at all it removes the question's second topic instead of a sub-definition (three
    # dev records, `_Session3.narrowing_applies`). True restores it; `no_facets` also restores it, which is what
    # makes that ablation isolate this change as well as the route's.
    "narrow_faceted_topics": False,

    # ---- 8. write back --------------------------------------------------------------------------------------------
    "write_back": True,                 # verified rows go into verified_facts (reading them needs the warm flag)
    "warm": False,

    # ---- trace ----------------------------------------------------------------------------------------------------
    "trace_text_chars": 400,
    "trace_candidates_max": 40,
})
