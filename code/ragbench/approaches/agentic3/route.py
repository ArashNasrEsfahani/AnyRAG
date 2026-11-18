"""Stage 3 - the sufficiency check: which path this question takes, and why.

No model call, no database access, no side effect: `decide` reads the plan, the candidates `kb/query.py` returned
and that call's `info`, and returns one of the four paths of AGENTIC3_SPEC.md section 6 together with the reason
the paper reports.

    FAST        lookup; the database alone already agrees with the plan on every axis   plan + answer      2 calls
    STANDARD    enough term-matching candidates, but the FAST test failed                plan + verify + answer  3
    ESCALATED   too few database candidates: lazy vectors are merged in as in v2         3 - 4 calls
    TIMELINE    the plan asks for several periods: always verified                       4 calls

The pool is examined before the FAST test, so a thin pool escalates rather than answering fast: two candidates
agreeing on a value is not evidence when the database only found two, and a thin pool is precisely where its
labelling failed (DB_CEILING: the gold value is in a fact row 53 times of 53, and the failures are ranking and
labelling failures). THIN MEANS THIN LABELS, not a short list: `merge` fills the cap from whatever the regex
matched, so counting the selected candidates escalated 0 of 20 dev questions in round 1, including the ones whose
twelve candidates had each matched a single generic term. `escalate_strong_terms` is the bar a candidate has to
clear to count as evidence at all.

A POOL CAN ALSO BE THIN IN ONE TOPIC ONLY, and that is what facets are for. `merge` reserves `facet_min_slots`
candidate slots for every facet of the plan; a facet it could not fill is a topic the database did not serve at
all, not a topic that was out-ranked, and `starved_facets` escalates on it - on the single path and on TIMELINE
alike, because a multi-section question is routed by its plan's `kind` and lands on either. The dev evidence is
A-rpt-f-03, the multi-section question facets exist for: in three of its four settings one facet's own
retrieval pass returned zero fact rows and zero selected candidates, all three answered NOT_FOUND, and replaying
the recorded plans with the vector pass added put the missing key fact into the selected pool in both settings
where the database had lost it (key facts in the pool 8 of 8 with vectors against 6 of 8 without). Over the whole
dev set the same replay raises the pool from 35 of 51 wanted values to 42. FAST is unaffected by construction:
`escalate_check` runs first, so a question with a starved facet never reaches the fast test.

The FAST gate is tuned for PRECISION, not coverage. On dev, `no_verify` cost 2 of 12 lookups, so a question
skips verification only when ALL of these hold, each one a column of the score breakdown `kb/query.py` already
computes (`features["db"]`):

    * one targeted document, the plan asks for a single value, and the best candidate came from the fact table;
    * its label matches at least `fast_min_terms` distinct term concepts, and the label is ABOUT them: either
      every term of one language is in it, or term matches cover `fast_min_cover` of the label's own words.
      (`all_terms` on its own is `Terms.covers_a_language`, which wants every expanded plan term - a median of
      seven per language on dev - inside one label of five or six words; it held for the top candidate of 3 of
      28 dev questions, which is too rare to be the only form of the test.)
    * its `unit_kind` is compatible with the plan's expected kind;
    * it dates itself inside the period, or the plan names no period;
    * its source is a table or a KPI line (a bare sentence is not enough);
    * no other candidate within `fast_margin` of its score carries a DIFFERENT value - and a candidate whose
      value cannot be read at all counts as different, because it cannot be ruled out.

A warm verified fact (`info["verified"]["matches"]`, warm mode only) is the other FAST entry: the row was already
verified by the model against the same document content. Until 2026-09-20 that entry was a SHORT CIRCUIT placed
before `escalate_check` and `fast_check`: any non-empty `warm_rows` answered the question in two calls with
`verify=False`. `kb/query.py::verified_lookup` selects those rows with `metric REGEXP <any plan term> OR terms
REGEXP <any plan term>` - an OR over every expanded term - so ONE shared word was enough for rows written by a
different question about the same report to answer this one. Traced on dev: `A-rpt-f-01` fa-fa, whose plan
metric is "درصد قیمت محصول الف نسبت به محصول ب", was answered from six rows whose metric is "قیمت هر تن محصول الف",
written by `A-rpt-f-03`; the two questions share "قیمت" and the product name and nothing else. Three of the four
warm-mode losses went through that door, which is why warm mode was faster AND less accurate (13/28 against
17/28). `warm_check` below replaces it: the warm rows are now examined after the pool, they have to be ABOUT the
metric this question asks for, there have to be at least `warm_min_rows` of them, they have to agree on one
value, and the database's own best candidate must not contradict them. A warm row that fails is not discarded -
it stays an ordinary evidence item and the question is verified like any other.

THE FAST GATE IS A NEGATIVE RESULT AND IS NOT TUNED FURTHER. It fired 0 times in all 300 dev records of round 1
and 0 times again in round 2; round 1 measured `fast_min_cover` at 0.6, 0.4 and 0.3, and the first question the
loosest setting admits is one whose rank-1 candidate is not the gold value. Round 2 also measured what a dev
difference is worth: `no_fastpath` changes nothing at all while FAST fires 0 times, yet against the base run it
produces a different plan for 10 of the 28 dev questions and a different answer for 15, because each variant keeps
its own call cache and the chat model is not reproducible across processes even at temperature 0 with seed 42 and
top_k 1. A threshold that moves dev accuracy by one or two questions therefore cannot be measured here at all, and
only a change with a mechanism visible in the traces is worth making.

ESCALATED is the thin-pool path. `no_vectors` turns it into STANDARD (the ablation), `vectors_always` turns every
question into it (v2's behaviour). A TIMELINE question also pulls vectors when a targeted document contributed
fewer than `timeline_min_per_doc` candidates or fewer than `timeline_min_strong` labelled ones, because a
document with no usable candidate is a period with no value, and the dev measurement behind that is
`no_vectors`: timeline coverage 74 -> 54. It is NOT loosened beyond that: forcing vectors on every dev timeline
was measured both ways in round 2 and is a wash (the `vectors_always` run has the same coverage, 0.667, and the
replay gains one gold value on one question and loses one on another), so the trigger stays a diagnosis rather
than a default.
"""
from ..agentic.numbers import find_numbers, same_number
from ...normalize import normalize, word_tokens
from .defaults import ROUTE_VERSION

WARM_MIN_WORD_CHARS = 3     # a metric word shorter than this is a function word (به, هر, تن, of, in, per)

FAST, STANDARD, ESCALATED, TIMELINE = "FAST", "STANDARD", "ESCALATED", "TIMELINE"


def fact_of(cand):
    return (cand.features or {}).get("fact") or {}


def db_features(cand):
    return (cand.features or {}).get("db") or {}


def value_readings(text):
    """-> the numeric readings of a value string, or None when it carries no number."""
    toks = [t for t in find_numbers(normalize(str(text or ""))) if not t.is_date]
    if not toks:
        return None
    out = []
    for t in toks:
        out += list(t.readings)
    return out or None


def same_value(a, b):
    """True when two value strings share a reading (the verbatim guard's own number equality)."""
    ra, rb = value_readings(a), value_readings(b)
    if ra is None or rb is None:
        return False
    return any(same_number(x, y) for x in ra for y in rb)


def strong_candidates(cands, min_terms=2):
    """Candidates the database can defend: a label (or unit window) that matched `min_terms` distinct term
    concepts. One generic term is not evidence - it is what every row of the report matches."""
    return [c for c in cands if int(db_features(c).get("terms") or 0) >= int(min_terms)]


def starved_facets(info, opts):
    """Facets the merge could not fill. -> sorted facet names.

    `merge` reserves `facet_min_slots` candidate slots for each facet before the round robin over documents, and
    reports how many selected candidates ended up carrying each facet's vocabulary (`merge.facet_slots`). A facet
    below its own reservation is not a topic that lost a ranking contest - there was nothing to rank. The same
    number decides both halves on purpose: the slots a facet must have are exactly the slots whose absence sends
    the question to the second path.

    The reservation is counted after `absorb_facets`, so a facet whose candidate is the same window the base pass
    found is credited with it; without that this test would fire on every faceted question. Absent `merge` info
    (the offline route tests, and any caller that has not merged yet) there are no facets and nothing is starved.
    """
    slots = (info.get("merge") or {}).get("facet_slots") or {}
    return sorted(n for n, k in slots.items() if int(k or 0) < int(opts["facet_min_slots"]))


def fast_check(plan, cands, info, opts):
    """-> (ok, reason, detail). Every refusal names the axis that failed, so the dev report can show the mix."""
    if plan.get("kind") != "single":
        return False, "not_single", {}
    if len(info.get("docs") or []) > int(opts["fast_max_docs"]):
        return False, "several_documents", {"docs": len(info.get("docs") or [])}
    if not cands:
        return False, "no_candidates", {}
    top = cands[0]
    f = db_features(top)
    fact = fact_of(top)
    detail = {"cid": top.cid, "score": round(top.score, 3), "db": f, "value": fact.get("value")}
    if "facts" not in ((top.features or {}).get("src") or []):
        return False, "top_not_a_fact", detail
    if int(f.get("terms") or 0) < int(opts["fast_min_terms"]):
        return False, "too_few_terms", detail
    if opts["fast_need_all_terms"] and not (f.get("all_terms")
                                            or float(f.get("label_cover") or 0.0) >= float(opts["fast_min_cover"])):
        return False, "not_all_terms", detail
    if opts["fast_need_kind"] and f.get("kind") != "match":
        return False, "unit_kind_%s" % (f.get("kind") or "none"), detail
    if opts["fast_need_year"] and f.get("year") not in ("in_period", "no_period"):
        return False, "year_%s" % (f.get("year") or "none"), detail
    kinds = tuple(opts["fast_source_kinds"] or ())
    if kinds and (f.get("source_kind") or "text") not in kinds:
        return False, "source_%s" % (f.get("source_kind") or "text"), detail
    margin = float(opts["fast_margin"])
    for c in cands[1:]:
        if top.score - c.score > margin:
            break
        other = fact_of(c).get("value")
        if other is None:
            detail["rival"] = {"cid": c.cid, "score": round(c.score, 3), "value": None}
            return False, "rival_value_unknown", detail
        if not same_value(fact.get("value"), other):
            detail["rival"] = {"cid": c.cid, "score": round(c.score, 3), "value": other}
            return False, "rival_value_differs", detail
    return True, "database_agrees", detail


def metric_words(text):
    """The content words of a metric string: folded tokens of at least WARM_MIN_WORD_CHARS characters."""
    return {t for t in word_tokens(str(text or ""), folded=True) if len(t) >= WARM_MIN_WORD_CHARS}


def warm_metric_match(plan, row, opts):
    """True when a verified row is ABOUT the metric this question asks for.

    The warm lookup is an OR over the plan's expanded terms, so a row that shares one word with the question comes
    back. This is the AND the OR is missing: `warm_metric_cover` of the plan metric's own content words have to be
    in the row's metric or in the plan terms recorded with it. At the default 1.0 the row's wording has to carry
    every content word the question's metric carries, which is what separates "درصد قیمت محصول الف نسبت به محصول ب"
    from "قیمت هر تن محصول الف" - the two metrics of the traced dev failure.
    """
    want = metric_words(plan.get("metric"))
    if not want:
        return False
    have = metric_words("%s %s" % (row.get("metric") or "", row.get("terms") or ""))
    return len(want & have) / float(len(want)) >= float(opts["warm_metric_cover"])


def warm_check(plan, warm_rows, cands, opts):
    """-> (ok, reason, detail): may these verified rows answer the question without the verifier?

    Warm mode only; `warm_rows` is empty in the headline COLD protocol. Every test below is a refusal the dev
    report can count, in the same shape as `fast_check`'s.
    """
    detail = {"warm_rows": len(warm_rows)}
    if not warm_rows:
        return False, "no_warm_rows", detail
    if plan.get("kind") != "single":
        return False, "warm_not_single", detail        # a series is always verified (spec 6 stage 5)
    kept = [r for r in warm_rows if warm_metric_match(plan, r, opts)]
    detail.update(metric_match=len(kept), metric_dropped=len(warm_rows) - len(kept))
    if len(kept) < int(opts["warm_min_rows"]):
        return False, "warm_metric_mismatch", detail
    values = [r.get("value_text") for r in kept]
    detail["values"] = sorted({str(v) for v in values})
    if any(value_readings(v) is None for v in values):
        return False, "warm_value_unreadable", detail
    if not all(same_value(values[0], v) for v in values[1:]):
        return False, "warm_values_disagree", detail
    if cands:
        other = fact_of(cands[0]).get("value")
        if other is not None and not same_value(values[0], other):
            detail["rival"] = {"cid": cands[0].cid, "value": other}
            return False, "warm_contradicted_by_db", detail
    return True, "warm_verified_fact", detail


def escalate_check(plan, cands, info, opts):
    """-> (escalate, reason, detail): is the database pool too thin to answer without vectors?

    Thin over the whole pool (too few candidates, too few whose label matched the metric) or thin in one topic
    (a facet the merge could not fill). The whole-pool tests keep round 1's order and reasons exactly, so the
    dev report's refusal mix stays comparable; `facet_unserved` is only ever returned where this function used to
    say the pool was full.
    """
    strong = strong_candidates(cands, opts["escalate_strong_terms"])
    starved = starved_facets(info, opts)
    detail = {"candidates": len(cands), "strong": len(strong), "strong_terms": int(opts["escalate_strong_terms"]),
              "facts_rows": info.get("facts_rows", 0), "units_rows": info.get("units_rows", 0),
              "facets_unserved": starved}
    if len(cands) < int(opts["escalate_min_candidates"]):
        return True, "few_candidates", detail
    text_answer = (plan.get("unit") or "count") == "text"
    if text_answer:
        if len(strong) < int(opts["escalate_text_min"]):
            return True, "text_answer_thin", detail
    elif len(strong) < int(opts["escalate_min_strong"]):
        return True, "few_term_matches", detail
    if starved:
        return True, "facet_unserved", detail
    return False, "text_answer_covered" if text_answer else "database_pool_full", detail


def timeline_needs_vectors(cands, doc_ids, opts, starved=()):
    """-> (want, detail): a targeted document is a period at risk when it contributed fewer than
    `timeline_min_per_doc` candidates, or fewer than `timeline_min_strong` whose label matched the metric. A
    document represented only by rows that matched one generic term has a value the verifier will refuse.

    `starved` are the facets the merge could not fill. A multi-period plan reaches this function instead of
    `escalate_check`, and a multi-section question whose plan came back `comparison` lands here too, so the same
    rule has to hold on both paths or it holds on neither: on dev, two of the three A-rpt-f-03 failures were
    routed TIMELINE.
    """
    strong_ids = {id(c) for c in strong_candidates(cands, opts["escalate_strong_terms"])}
    per = {d: 0 for d in doc_ids}
    strong = {d: 0 for d in doc_ids}
    for c in cands:
        if c.doc_id in per:
            per[c.doc_id] += 1
            strong[c.doc_id] += 1 if id(c) in strong_ids else 0
    thin = sorted(d for d in per if per[d] < int(opts["timeline_min_per_doc"]))
    weak = sorted(d for d in per if d not in thin and strong[d] < int(opts["timeline_min_strong"]))
    return bool(thin or weak or starved), {"per_doc": per, "strong_per_doc": strong, "thin_docs": thin,
                                           "weak_docs": weak, "facets_unserved": list(starved)}


def decide(plan, cands, info, opts, *, flags=None, warm_rows=()):
    """-> {route, reason, vectors, verify, detail} for one question (the record's trace.stages.route)."""
    flags = flags or {}
    out = {"version": ROUTE_VERSION, "route": STANDARD, "reason": "", "vectors": False, "verify": True,
           "detail": {}}
    # Since 2026-09-20 the DEFAULT is that vectors are always merged in (defaults.py `vector_policy`, with the
    # wide-dev measurement that decided it). `vector_policy = "when_thin"` is round 1's database-first routing,
    # kept so the ablation row can still be produced; the `vectors_always` flag forces the default explicitly.
    always = bool(flags.get("vectors_always") or str(opts.get("vector_policy", "always")) == "always")
    if always:
        out["detail"]["vectors_always"] = True
    if plan.get("kind") != "single":
        want, detail = timeline_needs_vectors(cands, info.get("docs") or [], opts,
                                              starved=starved_facets(info, opts))
        out.update(route=TIMELINE, reason="multi_period_plan", verify=True,
                   vectors=bool((want or always) and not flags.get("no_vectors")),
                   detail=dict(detail, thin=want))
        return out
    # the pool is examined BEFORE the FAST test: a thin pool is exactly where the database's own labelling went
    # wrong (DB_CEILING), so agreement among two rows is not evidence that the question is settled.
    esc, ereason, edetail = escalate_check(plan, cands, info, opts)
    out["detail"]["escalate"] = dict(edetail, reason=ereason)
    if esc:
        out.update(route=ESCALATED if not flags.get("no_vectors") else STANDARD,
                   reason=ereason if not flags.get("no_vectors") else "no_vectors_" + ereason,
                   vectors=not flags.get("no_vectors"))
        return out
    # The gate still runs and its verdict is still recorded - the refusal mix IS the finding - but since
    # 2026-09-20 `fast_cold_gate` is off and the route does not act on it. See defaults.py for the measurement.
    ok, reason, detail = fast_check(plan, cands, info, opts)
    out["detail"]["fast"] = {"ok": ok, "reason": reason, "acted_on": bool(opts.get("fast_cold_gate")), **detail}
    if ok and opts.get("fast_cold_gate") and not flags.get("no_fastpath"):
        out.update(route=FAST, reason=reason, verify=False, vectors=False, detail=detail)
        return out
    # warm mode only, and only here: after the pool test and after the database's own fast test have both had
    # their say. A warm row is evidence either way (the pack carries it); what it no longer is, is a bypass.
    wok, wreason, wdetail = warm_check(plan, warm_rows, cands, opts)
    if warm_rows:
        out["detail"]["warm"] = {"ok": wok, "reason": wreason, **wdetail}
    if wok and not flags.get("no_fastpath"):
        out.update(route=FAST, reason=wreason, verify=False, vectors=False, detail=wdetail)
        return out
    # Default policy: the vectors are merged in even though the pool test found the pool full. A thin pool has
    # already escalated above, and a sound warm consensus has already answered, so this is the ordinary case.
    if always:
        out.update(route=ESCALATED if not flags.get("no_vectors") else STANDARD,
                   reason="vectors_always" if not flags.get("no_vectors") else "no_vectors_vectors_always",
                   vectors=not flags.get("no_vectors"))
        return out
    out.update(route=STANDARD, reason=reason if not ok else "fastpath_disabled", vectors=False)
    return out
