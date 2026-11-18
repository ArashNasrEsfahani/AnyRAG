"""AGENTIC3 offline tests (no model, no GPU; a scripted client replaces the HTTP calls, embeddings are hashed
trigrams). The knowledge base is a throwaway file, deleted on every run.

    .venv/Scripts/python.exe -m ragbench.approaches.agentic3.test_agentic3

Covers: the plan prompt's FACETS field (schema order, cleaning, disjoint vocabulary, the no_facets ablation), the
sufficiency check (every FAST refusal reason, ESCALATED on a thin pool, TIMELINE, the flags), ingestion through
`_build_index` (idempotent, resumable, the index fingerprint following the knowledge base's component versions),
database retrieval and tree expansion (verbatim spans, facet slots, dedupe across paths), the guard escape hatch
that v2's misses call for (accepted through the definition, accepted through the fact label, still rejected when
the term is nowhere, and off under verify_gate), the verdict-aware order of the unverified pack, gap queries
restricted to the corpus set, write-back (guard-gated, idempotent), and a full RunRecord (<= 4 calls with the
answer last, the route in the trace and in `structured`, the query clock without the first-touch embedding, the
shared 6000-char evidence budget).
Writes only under work/test_agentic3/.
"""
import json
import shutil
import sys

from ...config import load_config
from ...contracts import Question, Setting
from ...corpus.registry import CorpusSet
from ...kb import query as Q
from ...llm import Ledger
from ..agentic import planner as P
from ..agentic import search as S
from ..agentic.docmeta import doc_meta
from ..agentic.test_agentic import PLAN_OBJ, _doc, make_corpus, make_responder
from ..agentic2.test_agentic2 import ScriptedClient2
from . import retrieve as RT
from . import route as RO
from . import writeback as WB
from .approach import TypedKBAgent, _Session3, plan_messages, plan_schema, validate_facets
from .defaults import DEFAULTS

FAILS = []
N = [0]


def check(cond, label, detail=""):
    N[0] += 1
    print(("  ok   " if cond else "  FAIL ") + label + ((" - " + str(detail)[:600]) if (detail and not cond) else ""))
    if not cond:
        FAILS.append(label)
    return cond


# ---------------------------------------------------------------------------------------------------------------
# corpus: the v1 test documents plus one shaped like the customs-01 miss (a synonym the plan does not have)
# ---------------------------------------------------------------------------------------------------------------

def staff_doc():
    return _doc("C.staffco_annual-report_1401.fa", "گزارش سالانه استاف‌کو ۱۴۰۱", [
        ("p1", "منابع انسانی", [("paragraph", "تعداد کارکنان گروه"), ("paragraph", "269,105 نفر"),
                                ("paragraph", "این گروه در سال 1401 فعالیت خود را گسترش داد.")]),
    ])


def corpus():
    cs = make_corpus()
    return CorpusSet("C", list(cs.docs) + [staff_doc()])


def question(kind="timeline", scope="corpus"):
    if kind == "timeline":
        return Question(id="t3-timeline", corpus="C",
                        question_fa="تعداد رانندگان تست‌کو طی سال‌ها چگونه تغییر کرده است؟",
                        question_en="How has the number of Testco drivers changed over the years?",
                        answer_fa="", answer_en="", answer_type="short",
                        gold_units={"fa": ["C.testco_annual-report_1402.fa::p1_b2"]}, doc_scope=scope,
                        type="cross-year")
    return Question(id="t3-single", corpus="C", question_fa="تعداد کاربران راننده تست‌کو در ۱۴۰۲ چند نفر بود؟",
                    question_en="How many Testco driver users were there in 1402?", answer_fa="", answer_en="",
                    answer_type="numeric", gold_units={"fa": ["C.testco_annual-report_1402.fa::p1_b2"]},
                    doc_scope=["C.testco_annual-report_1402.fa"], type="lookup")


def approach(cfg, cs, responder, *, flags=None, options=None, build=True):
    ledger = Ledger(path=None)
    client = ScriptedClient2(cfg, ledger, responder)
    ap = TypedKBAgent(cfg=cfg, client=client, flags=flags, options=options)
    index = ap.build_index(cs, ledger) if build else None
    return ap, client, index


def plan_of(cs, obj=None, opts=None):
    metas = [doc_meta(d) for d in cs.docs]
    plan = P.validate_plan(obj or PLAN_OBJ, question="تعداد رانندگان تست‌کو", q_lang="fa", metas=metas,
                           choose_companies=True, opts=opts or DEFAULTS)
    plan["facets"] = []
    return plan


# ---------------------------------------------------------------------------------------------------------------
# 1. the plan prompt
# ---------------------------------------------------------------------------------------------------------------

def test_plan(cs):
    o = dict(DEFAULTS)
    msgs = plan_messages("سوال", [doc_meta(d) for d in cs.docs], True, o)
    base = P.plan_messages("سوال", [doc_meta(d) for d in cs.docs], True, o)
    check(msgs[0]["content"].startswith(base[0]["content"]) and '"facets"' in msgs[0]["content"],
          "plan.facets_rule_appended_to_v1_system")
    check(msgs[1] == base[1], "plan.user_message_unchanged")
    schema = plan_schema(["testco"], o)
    props = list(schema["properties"])
    check(props[-1] == "facets" and "facets" in schema["required"], "plan.facets_is_the_last_field", props)
    check(props[:-1] == list(P.plan_schema(["testco"])["properties"]), "plan.other_fields_unchanged", props)
    off = plan_schema(["testco"], dict(o, facets=False))
    check("facets" not in off["properties"] and "facets" not in " ".join(plan_messages("س", [], False, dict(o, facets=False))[0]["content"].split()[-3:]),
          "plan.no_facets_is_v1_exactly")
    plan = plan_of(cs)
    facets = validate_facets({"facets": [["رانندگان فعال", "راننده فعال"], ["باشگاه رانندگان", "رانندگان"],
                                         ["سرویس خودرو"], ["چهارم"], ["پنجم"]]}, plan, o)
    check(len(facets) == int(o["max_facets"]), "plan.facets_capped_at_three", [f["name"] for f in facets])
    words = [t for f in facets for t in f["terms"]["fa"]]
    check(len(words) == len(set(words)), "plan.facet_vocabulary_is_disjoint", words)
    check(any(f["overlap_dropped"] for f in facets), "plan.overlap_is_recorded",
          [f["overlap_dropped"] for f in facets])
    check(validate_facets({"facets": [["1398"], []]}, plan, o) == [], "plan.empty_facets_dropped")


# ---------------------------------------------------------------------------------------------------------------
# 2. the sufficiency check
# ---------------------------------------------------------------------------------------------------------------

def _cand(cid, score, value, *, terms=2, all_terms=1, kind="match", year="in_period", source="kpi", src="facts"):
    c = S.Candidate("C.testco_annual-report_1402.fa", "sec", ["u1"], "تعداد کاربران راننده %s" % value, None,
                    "unit", ["u1"])
    c.cid = cid
    c.score = score
    c.features = {"src": [src], "facet": [],
                  "db": {"terms": terms, "all_terms": all_terms, "kind": kind, "year": year,
                         "source_kind": source},
                  "fact": {"label": "تعداد کاربران راننده", "value": value}}
    return c


def test_route():
    o = dict(DEFAULTS)
    single = {"kind": "single", "unit": "count", "period": (1402, 1402)}
    info = {"docs": ["C.testco_annual-report_1402.fa"], "facts_rows": 8, "units_rows": 0}
    good = [_cand("c1", 4.2, "6,152,509"), _cand("c2", 3.9, "6,152,509"), _cand("c3", 1.0, "3,305,147")]
    full = good + [_cand("c%d" % i, 0.9, "1") for i in range(4, 9)]
    # the cold gate is OFF since 2026-09-20 (a measured negative result, defaults.py): it still runs and still
    # records its verdict, but the route does not act on it. With `fast_cold_gate` on it is round 1's behaviour.
    d = RO.decide(single, full, info, o)
    check(d["route"] != RO.FAST and d["verify"] and d["detail"]["fast"]["ok"]
          and d["detail"]["fast"]["acted_on"] is False, "route.cold_fast_gate_is_recorded_but_not_acted_on", d)
    # ... and the default policy merges the vectors in even on a pool the database calls full (2026-09-20)
    check(d["route"] == RO.ESCALATED and d["reason"] == "vectors_always" and d["vectors"],
          "route.vectors_are_merged_in_by_default", d)
    dw = RO.decide(single, full, info, dict(o, vector_policy="when_thin"))
    check(dw["route"] == RO.STANDARD and not dw["vectors"],
          "route.vector_policy_when_thin_restores_the_database_first_routing", dw)
    d = RO.decide(single, full, info, dict(o, fast_cold_gate=True))
    check(d["route"] == RO.FAST and not d["vectors"] and not d["verify"], "route.fast_when_database_agrees", d)
    rival = [_cand("c1", 4.2, "6,152,509"), _cand("c2", 3.9, "3,305,147")] + \
        [_cand("c%d" % i, 0.9, "1") for i in range(3, 9)]
    d = RO.decide(single, rival, info, o)
    check(d["route"] != RO.FAST and d["detail"]["fast"]["reason"] == "rival_value_differs",
          "route.fast_refused_on_a_different_rival_value", d["detail"]["fast"]["reason"])
    for change, reason in ((dict(terms=1), "too_few_terms"), (dict(all_terms=0), "not_all_terms"),
                           (dict(kind="conflict"), "unit_kind_conflict"),
                           (dict(year="out_of_period"), "year_out_of_period"),
                           (dict(source="text"), "source_text"), (dict(src="vector"), "top_not_a_fact")):
        pool = [_cand("c1", 4.2, "6,152,509", **change)] + [_cand("c%d" % i, 0.9, "6,152,509") for i in range(2, 9)]
        ok, why, _ = RO.fast_check(single, pool, info, o)
        check(not ok and why == reason, "route.fast_refused_%s" % reason, why)
    thin = good[:2]
    d = RO.decide(single, thin, info, o)
    check(d["route"] == RO.ESCALATED and d["vectors"] and d["reason"] == "few_candidates",
          "route.escalated_on_a_thin_pool_even_when_it_agrees", d)
    d = RO.decide(single, thin, info, o, flags={"no_vectors": True})
    check(d["route"] == RO.STANDARD and not d["vectors"], "route.no_vectors_falls_back_to_standard", d)
    d = RO.decide(single, good + [_cand("c%d" % i, 0.9, "1") for i in range(4, 9)], info, o,
                  flags={"no_fastpath": True})
    check(d["route"] != RO.FAST and d["verify"], "route.no_fastpath_never_skips_verification", d)
    d = RO.decide(single, good, info, o, flags={"vectors_always": True})
    check(d["route"] == RO.ESCALATED, "route.vectors_always_escalates_everything", d)
    tl = {"kind": "timeline", "unit": "count", "period": (1398, 1402)}
    many = {"docs": ["a", "b"], "facts_rows": 20}
    cands = [_cand("c1", 4.0, "1"), _cand("c2", 3.0, "2")]
    for c in cands:
        c.doc_id = "a"
    d = RO.decide(tl, cands, many, o)
    check(d["route"] == RO.TIMELINE and d["verify"] and d["vectors"] and d["detail"]["thin_docs"] == ["b"],
          "route.timeline_always_verified_and_pulls_vectors_for_a_thin_document", d)
    # warm mode (2026-09-20): a verified row no longer short-circuits the route. It is examined after the pool
    # test and after the database's own fast test, it has to be ABOUT this question's metric, there have to be at
    # least `warm_min_rows` of them and they have to agree.
    wsingle = dict(single, metric="تعداد کاربران راننده")
    wrow = {"metric": "تعداد کاربران راننده", "terms": "{}", "value_text": "6,152,509"}
    other = {"metric": "درآمد هر سفر", "terms": "{}", "value_text": "6,152,509"}
    d = RO.decide(wsingle, rival, info, o, warm_rows=[wrow, dict(wrow)])
    check(d["route"] == RO.FAST and d["reason"] == "warm_verified_fact" and not d["verify"],
          "route.warm_two_agreeing_rows_about_the_metric_are_fast", d)
    d = RO.decide(wsingle, rival, info, o, warm_rows=[dict(other), dict(other)])
    check(d["route"] != RO.FAST and d["detail"]["warm"]["reason"] == "warm_metric_mismatch",
          "route.warm_rows_about_another_metric_are_refused", d["detail"]["warm"])
    d = RO.decide(wsingle, rival, info, o, warm_rows=[wrow])
    check(d["route"] != RO.FAST and d["detail"]["warm"]["reason"] == "warm_metric_mismatch",
          "route.warm_one_row_is_not_enough", d["detail"]["warm"])
    d = RO.decide(wsingle, rival, info, o, warm_rows=[wrow, dict(wrow, value_text="3,305,147")])
    check(d["route"] != RO.FAST and d["detail"]["warm"]["reason"] == "warm_values_disagree",
          "route.warm_rows_that_disagree_are_refused", d["detail"]["warm"])
    d = RO.decide(wsingle, thin, info, o, warm_rows=[wrow, dict(wrow)])
    check(d["route"] == RO.ESCALATED and "warm" not in d["detail"],
          "route.a_thin_pool_escalates_even_with_warm_rows", d)
    d = RO.decide(dict(tl, metric="تعداد کاربران راننده"), cands, many, o, warm_rows=[wrow, dict(wrow)])
    check(d["route"] == RO.TIMELINE and d["verify"], "route.warm_rows_never_skip_a_series", d)


# ---------------------------------------------------------------------------------------------------------------
# 3. ingestion through _build_index
# ---------------------------------------------------------------------------------------------------------------

def test_index(cfg, cs):
    ap, _client, index = approach(cfg, cs, make_responder())
    ing = index.payload["ingest"]
    check(ing["documents"] == len(cs.docs) and ing.get("loaded") == len(cs.docs),
          "index.every_document_ingested", {k: ing.get(k) for k in ("documents", "loaded", "skipped")})
    check(ing["rows"].get("facts", 0) > 0 and ing["rows"].get("units", 0) > 0 and "ingest_ms" in ing,
          "index.rows_and_time_recorded", ing["rows"])
    stats = json.loads((index.path / "build_stats.json").read_text(encoding="utf-8"))
    check(stats["complete"] and stats["totals"]["llm_calls"] == 0 and stats["attempts"][0]["build_ms"] > 0,
          "index.no_model_call_and_ingestion_time_in_build_stats", stats.get("totals"))
    again = ap._build_index(cs, index.path)
    check(again["ingest"].get("skipped") == len(cs.docs), "index.second_build_is_a_no_op",
          {k: again["ingest"].get(k) for k in ("loaded", "rebuilt", "skipped")})
    loaded = ap._load_index(cs, index.path)
    check(loaded["kb_path"] == again["kb_path"] and not RT.missing_documents(ap.kb(), cs.doc_ids()),
          "index.load_is_a_pointer_check")
    fp = ap.index_fingerprint(cs)
    sig = ap.config_signature()
    check(sig["kb"]["typed"] and sig["kb"]["schema"] and sig["query"],
          "index.fingerprint_covers_kb_component_versions", sig)
    other = TypedKBAgent(cfg=cfg.copy_with({"models": {"embed": "other-embed"}}), client=ap.client)
    check(other.index_fingerprint(cs) != fp, "index.fingerprint_changes_with_the_embed_model")
    return ap, index


# ---------------------------------------------------------------------------------------------------------------
# 4. retrieval, expansion and the facet slots
# ---------------------------------------------------------------------------------------------------------------

def test_retrieve(cfg, cs, ap):
    o = dict(DEFAULTS)
    con = ap.kb()
    plan = plan_of(cs)
    doc_ids = [d.id for d in cs.docs if d.id.startswith("C.testco")]
    cands, info = RT.retrieve_all(con, plan, doc_ids, d_lang="fa", opts=o, trigram=True, reader=Q.DocReader(con))
    check(len(cands) > 0 and info["facts_rows"] > 0, "retrieve.facts_found", (len(cands), info["facts_rows"]))
    bad = []
    for c in cands:
        if c.span is None:
            continue
        doc = cs.doc(c.doc_id)
        if doc.text[c.span[0]:c.span[1]] != c.text:
            bad.append((c.doc_id, c.span, c.text[:40]))
    check(not bad, "retrieve.every_span_is_a_verbatim_slice_of_document_text", bad[:3])
    fplan = dict(plan)
    fplan["facets"] = [{"name": "f1", "terms": {"fa": ["باشگاه"], "en": []}},
                       {"name": "f2", "terms": {"fa": ["سرویس"], "en": []}}]
    facets = RT.facet_plans(fplan, o)
    check([n for n, _p in facets] == ["f1", "f2"], "retrieve.facet_plans_named_in_order")
    pool, finfo = RT.retrieve_all(con, fplan, doc_ids, d_lang="fa", opts=o, reader=Q.DocReader(con), facets=facets)
    avail = {n: sum(1 for c in pool if n in RT.facets_of(c)) for n, _p in facets}
    sel, minfo = RT.merge(pool, doc_ids, facets, o, 8)
    check(finfo["facet_candidates"] > 0, "retrieve.facet_passes_add_candidates", finfo["facet_candidates"])
    kept = minfo["facet_slots"]
    check(all(v > 0 for v in avail.values())
          and all(kept[n] >= min(int(o["facet_min_slots"]), avail[n]) for n in kept),
          "retrieve.every_facet_keeps_its_slots", (avail, kept))
    tight, tinfo = RT.merge(pool, doc_ids, facets, o, 2)
    check(tinfo["facet_slots"] == {n: 1 for n in avail} and len(tight) == 2,
          "retrieve.facets_are_reserved_before_the_cap_bites", tinfo["facet_slots"])
    check([c.cid for c in sel] == ["c%d" % i for i in range(1, len(sel) + 1)],
          "retrieve.cids_assigned_in_pack_order", [c.cid for c in sel][:4])
    check(all(getattr(c, "verdict", "x") is None for c in sel), "retrieve.verdict_cleared_for_the_verifier")
    dup = RT.merge(pool + pool, doc_ids, facets, o, 8)[0]
    check(len(dup) == len(sel), "retrieve.merge_dedupes_across_passes", (len(dup), len(sel)))


# ---------------------------------------------------------------------------------------------------------------
# 5. the guard escape hatch (V2_FAILURES 2.2)
# ---------------------------------------------------------------------------------------------------------------

def _session(cfg, cs, ap, index, q):
    query = ap.make_query(q, Setting.parse("fa-fa"))
    return _Session3(ap, query, index)


def _batch_item(cand, definition, quote, value="269,105"):
    return {"cid": cand.cid, "doc_id": cand.doc_id, "verdict": "guard:quote_without_term", "value": value,
            "unit": "نفر", "period": "1401", "definition": definition, "quote": quote, "guard": "quote_without_term"}


def _staff_candidate(cs, label="تعداد کارکنان گروه"):
    doc = cs.doc("C.staffco_annual-report_1401.fa")
    units = [doc.unit(u) for u in doc.section("C.staffco_annual-report_1401.fa::p1").unit_ids]
    text = "\n\n".join(u.text for u in units[:2])
    c = S.Candidate(doc.id, doc.sections[0].id, [u.id for u in units[:2]], text,
                    [units[0].span[0], units[1].span[1]], "unit", [units[1].id])
    c.cid = "c1"
    c.features = {"src": ["facts"], "facet": [], "fact": {"label": label, "value": "269,105", "fact_id": 7}}
    return c


def test_guard_escape(cfg, cs, ap, index):
    sess = _session(cfg, cs, ap, index, question("single"))
    plan = P.validate_plan({"entity": "استاف‌کو", "metric": "نیروی انسانی", "kind": "single",
                            "terms_fa": ["نیروی انسانی"], "terms_en": ["workforce"], "entity_terms": ["استاف‌کو"],
                            "unit": "count", "period_from": "", "period_to": ""},
                           question="نیروی انسانی استاف‌کو چند نفر است؟", q_lang="fa",
                           metas=[doc_meta(d) for d in cs.docs], choose_companies=False, opts=sess.o)
    meta_by = {d.id: doc_meta(d) for d in cs.docs}
    quote = "تعداد کارکنان گروه\n\n269,105 نفر"

    def run(definition, label, opts=None):
        c = _staff_candidate(cs, label)
        out = {"items": [_batch_item(c, definition, quote)], "accepted": 0, "guard_rejected": 1,
               "guard_reasons": {"quote_without_term": 1}}
        rows = []
        s = sess
        if opts:
            s.o = dict(sess.o, **opts)
        s.guard_reasons = {"quote_without_term": 1}
        s._term_rx = None
        s.accept_by_definition(out, [c], plan, rows, meta_by)
        s.o = sess.o
        return out, rows, c

    out, rows, c = run("نیروی انسانی گروه", "تعداد کارکنان گروه")
    check(out["guard_escaped"] == 1 and len(rows) == 1 and c.verdict == "accepted",
          "guard.accepted_when_the_verifiers_definition_carries_the_term", (out["guard_escaped"], len(rows)))
    check(rows and rows[0]["value_num"] == 269105.0 and rows[0]["guard_escape"] == "definition",
          "guard.row_keeps_the_verbatim_value_and_records_the_escape", rows[:1])
    check(out["items"][0]["verdict"] == "accepted" and out["items"][0]["guard"] == "ok"
          and out["guard_reasons"]["quote_without_term"] == 0,
          "guard.counters_moved_from_rejected_to_accepted", out["items"][0])
    out, rows, c = run("کارکنان گروه", "مجموع نیروی انسانی گروه")
    check(out["guard_escaped"] == 1 and rows[0]["guard_escape"] == "label",
          "guard.accepted_when_the_fact_label_carries_the_term", out["guard_escaped"])
    out, rows, c = run("کارکنان گروه", "تعداد کارکنان گروه")
    check(out["guard_escaped"] == 0 and not rows and getattr(c, "verdict", None) != "accepted",
          "guard.still_rejected_when_the_term_is_nowhere", (out["guard_escaped"], len(rows)))
    out, rows, c = run("نیروی انسانی گروه", "تعداد کارکنان گروه", opts={"guard_escape_label": True})
    check(out["guard_escaped"] == 1, "guard.escape_is_on_by_default")
    gated = TypedKBAgent(cfg=cfg, client=ap.client, flags={"verify_gate": True})
    check(gated.opts["guard_escape"] is False and gated.opts["rank_by_verdict"] is False,
          "guard.verify_gate_restores_v2_exactly", gated.opts["guard_escape"])
    # a quote that is NOT in the snippet is still refused: the escape hatch only removes the TERM test
    c2 = _staff_candidate(cs)
    out2 = {"items": [_batch_item(c2, "نیروی انسانی گروه", "تعداد کارکنان گروه 999,999 نفر")], "accepted": 0,
            "guard_rejected": 1, "guard_reasons": {"quote_without_term": 1}}
    rows2 = []
    sess.guard_reasons = {"quote_without_term": 1}
    sess.accept_by_definition(out2, [c2], plan, rows2, meta_by)
    check(out2["guard_escaped"] == 0 and not rows2, "guard.verbatim_check_still_applies", out2["guard_escaped"])


def test_verdict_order(cfg, cs, ap, index):
    sess = _session(cfg, cs, ap, index, question("single"))
    cands = []
    for i, verdict in enumerate(["model_rejected", "guard:quote_without_term", None, "accepted", "model_omitted"], 1):
        c = _cand("c%d" % i, 5.0 - i, str(i))
        c.verdict = verdict
        cands.append(c)
    order = [c.cid for c in sess.by_verdict(cands)]
    check(order == ["c4", "c2", "c5", "c3", "c1"], "evidence.unverified_pack_follows_the_verifiers_judgement",
          order)


# ---------------------------------------------------------------------------------------------------------------
# 6. gap queries stay inside the corpus set
# ---------------------------------------------------------------------------------------------------------------

def test_gap_scope():
    gap = {"year": 1399, "docs": [{"doc_id": "C.a.fa"}, {"doc_id": "C.outside.fa"}],
           "doc_ids": ["C.a.fa", "C.outside.fa"], "labels": [], "pattern": "x",
           "sql": "SELECT 1", "params": {"d0": "C.a.fa", "d1": "C.outside.fa", "y": 1399, "lo": 1399, "hi": 1399,
                                          "pattern": "x", "limit": 8}, "runnable": True}
    scoped, dropped = RT.scope_gap(gap, ["C.a.fa"])
    check(dropped == 1 and scoped["doc_ids"] == ["C.a.fa"] and scoped["params"].get("d1") is None
          and ":d1" not in scoped["sql"], "gap.documents_outside_the_corpus_set_are_dropped", scoped["params"])
    check(scoped["params"]["pattern"] == "x" and scoped["params"]["limit"] == 8 and scoped["runnable"],
          "gap.other_parameters_survive_the_restriction", scoped["params"])
    empty, dropped = RT.scope_gap(gap, [])
    check(dropped == 2 and not empty["runnable"], "gap.nothing_in_scope_is_not_runnable")


def test_gap_fill(cfg, cs, ap, index):
    import re
    sess = _session(cfg, cs, ap, index, question("timeline"))
    plan = plan_of(cs)
    scope = [d.id for d in cs.docs]
    rec = {"periods_missing": ["1400"], "series": [], "restatements": []}
    gcands, info = sess.gap_fill(rec, plan, [1400], scope, [], {}, [])
    check(info["ran"] and info["queries"] and not info["queries"][0]["runnable"] and not gcands,
          "gap.nothing_verified_leaves_the_reconciler_without_a_pattern", info["queries"][:1])
    pattern = RT.plan_gap_pattern(plan, "fa", sess.o)
    check(pattern and re.compile(pattern), "gap.plan_pattern_compiles", pattern[:80])
    gcands, info = sess.gap_fill(rec, plan, [1400], scope, [], {}, [], fallback=pattern)
    check(info["pattern_from"] == "plan" and info["queries"][0]["runnable"]
          and info["queries"][0]["rows"] > 0 and gcands,
          "gap.plan_wording_makes_the_gap_query_runnable", info["queries"][:1])
    check(all(c.cid.startswith("g") and c.features["src"] == ["gap"] for c in gcands),
          "gap.candidates_are_tagged_and_numbered_apart", [c.cid for c in gcands][:3])


# ---------------------------------------------------------------------------------------------------------------
# 7. the whole pipeline
# ---------------------------------------------------------------------------------------------------------------

def test_pipeline(cfg, cs):
    ap, client, _index = approach(cfg, cs, make_responder())
    rec = ap.answer(question("timeline"), Setting.parse("fa-fa"))
    t = rec.trace
    check(not rec.error, "pipeline.no_error", rec.error)
    check(rec.llm_calls <= 4 and t["calls"][-1]["purpose"] == "answer",
          "pipeline.call_cap_with_the_answer_last", [c["purpose"] for c in t["calls"]])
    check(t["stages"]["route"]["route"] in ("TIMELINE", "ESCALATED", "STANDARD", "FAST")
          and rec.structured["route"] == t["stages"]["route"]["route"]
          and t["stages"]["route"]["reason"], "pipeline.route_and_reason_in_trace_and_structured",
          t["stages"]["route"])
    check(t["stats"]["route_timeline"] == 1 and rec.structured["kind"] == "timeline",
          "pipeline.timeline_plan_takes_the_timeline_route", t["stats"]["route_timeline"])
    check(len(rec.structured["series"]) >= 2 and all(r["verified"] for r in rec.structured["series"]),
          "pipeline.only_verified_rows_enter_the_series", rec.structured["series"][:2])
    check(rec.evidence_chars <= int(cfg.evidence["max_chars"]), "pipeline.shared_evidence_budget",
          rec.evidence_chars)
    clock = t["stats"]["query_clock_ms"]
    lazy = t["stats"]["lazy_index_ms"]
    check(clock <= ap.max_seconds * 1000 and (lazy == 0 or clock < t["timing"]["totals"]["wall_ms"]),
          "pipeline.query_clock_excludes_the_first_touch_embedding", (clock, lazy))
    check(t["timing"]["wall_with_embedding_ms"] >= clock, "pipeline.wall_clock_reported_next_to_it",
          (t["timing"]["wall_with_embedding_ms"], clock))
    check(t["versions"]["kb"]["schema"] and t["versions"]["query"] and t["versions"]["reconcile_sql"],
          "pipeline.versions_recorded", t["versions"])
    con = ap.kb()
    counts = WB.counts(con)
    check(counts["rows"] == t["stages"]["writeback"]["written"] > 0 and counts["stale"] == 0,
          "pipeline.verified_rows_written_back", (counts, t["stages"]["writeback"]))
    ap2, _c2, _i2 = approach(cfg, cs, make_responder())
    rec2 = ap2.answer(question("timeline"), Setting.parse("fa-fa"))
    check(rec2.trace["stages"]["writeback"]["written"] == 0
          and rec2.trace["stages"]["writeback"]["skipped"].get("already_written", 0) > 0,
          "pipeline.write_back_is_idempotent", rec2.trace["stages"]["writeback"])
    check(WB.read_rows(con, doc_ids=cs.doc_ids(), warm=False) == [], "pipeline.warm_reads_are_off_by_default")
    warm = WB.read_rows(con, doc_ids=cs.doc_ids(), warm=True)
    check(len(warm) > 0 and all(w["guard"] in WB.GUARD_OK for w in warm),
          "pipeline.warm_reads_only_guard_passed_rows", len(warm))
    return ap


def test_flags(cfg, cs):
    ap, _client, _index = approach(cfg, cs, make_responder(), flags={"no_facets": True, "no_trigram": True})
    check(ap.variant == "AGENTIC3+no_facets,no_trigram" and not ap.opts["facets"] and not ap.use_trigram,
          "flags.variant_name_and_switches", ap.variant)
    rec = ap.answer(question("single"), Setting.parse("fa-fa"))
    check(not rec.error and rec.llm_calls <= 4, "flags.no_trigram_answers", rec.error)
    check(rec.trace["stages"]["retrieve"]["trigram"] is False, "flags.no_trigram_reaches_the_query_module",
          rec.trace["stages"]["retrieve"].get("trigram"))
    ap2 = TypedKBAgent(cfg=cfg, client=ap.client, flags={"no_typed": True})
    check(ap2.opts["kind_filter"] == "off" and ap2.opts["w_kind"] == 0.0 and ap2.opts["year_filter"] == "off",
          "flags.no_typed_turns_off_every_typed_filter_and_weight")
    ap3 = TypedKBAgent(cfg=cfg, client=ap.client, flags={"no_tree_prior": True})
    check(ap3.opts["w_section"] == 0.0, "flags.no_tree_prior_zeroes_the_section_prior")
    ap4 = TypedKBAgent(cfg=cfg, client=ap.client, flags={"no_sql_reconcile": True})
    check(ap4.opts["sql_reconcile"] is False, "flags.no_sql_reconcile_falls_back_to_v1")
    try:
        TypedKBAgent(cfg=cfg, client=ap.client, flags={"nonsense": True})
        check(False, "flags.unknown_flag_refused")
    except ValueError as e:
        check("nonsense" in str(e), "flags.unknown_flag_refused", str(e))


def test_resume(cfg, cs):
    """A question killed after the plan call replays it from the call cache and takes the same decisions."""
    tmp = cfg.project_root / "work" / "test_agentic3" / "calls.sqlite"
    ap, client, _i = approach(cfg, cs, make_responder(), options={"call_cache_path": str(tmp)})
    client.fail_at = 2                       # the verify call of the first attempt
    rec = ap.answer(question("timeline"), Setting.parse("fa-fa"))
    check(bool(rec.error), "resume.first_attempt_fails", rec.error)
    ap2, client2, _i2 = approach(cfg, cs, make_responder(), options={"call_cache_path": str(tmp)})
    rec2 = ap2.answer(question("timeline"), Setting.parse("fa-fa"))
    plans = [p for p in client2.payloads if p["messages"][0]["content"].startswith("You plan a search")]
    check(not rec2.error and not plans, "resume.plan_replayed_from_the_cache_not_re_asked",
          (rec2.error, len(plans)))
    check(rec2.trace["stats"]["cache_hits"] >= 1, "resume.cache_hits_recorded",
          rec2.trace["stats"]["cache_hits"])


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass
    base = load_config()
    tmp = base.project_root / "work" / "test_agentic3"
    if tmp.exists():
        shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True, exist_ok=True)
    cfg = base.copy_with({"paths": {"index_root": str(tmp / "indexes")},
                          "kb": {"path": str(tmp / "kb.sqlite")}})
    cs = corpus()
    print("plan");     test_plan(cs)
    print("route");    test_route()
    print("index");    ap, index = test_index(cfg, cs)
    print("retrieve"); test_retrieve(cfg, cs, ap)
    print("guard");    test_guard_escape(cfg, cs, ap, index)
    print("evidence"); test_verdict_order(cfg, cs, ap, index)
    print("gap");      test_gap_scope(); test_gap_fill(cfg, cs, ap, index)
    print("pipeline"); test_pipeline(cfg, cs)
    print("flags");    test_flags(cfg, cs)
    print("resume");   test_resume(cfg, cs)
    print("\n%d checks, %d failure(s)" % (N[0], len(FAILS)))
    for f in FAILS:
        print("  FAILED:", f)
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
