"""AGENTIC2 offline tests (no model, no GPU; a scripted client replaces the HTTP calls, embeddings are hashed trigrams).

    .venv/Scripts/python.exe -m ragbench.approaches.agentic2.test_agentic2

Covers: tree leaves (verbatim passages inside sections, long-unit windows), tree expansion (detached Persian number
gets its label, table caption + header + row), the fact DB (rows on Persian and English detached numbers, term +
period query in SQLite), lazy embeddings (shared cache, a second process embeds nothing, a failure mid-document resumes
with only the missing vectors), the call cap (<= 4, answer last), the time cap (verify skipped, answer still given),
the replayed clock, the verbatim guard inside v2, the flags / ablations (no_verify, no_factdb, no_tree, no_vectors,
candidates=n) and a full RunRecord (structured series, evidence budget, lazy-index stage in timing and ledger).
Writes only under work/test_agentic2/.
"""
import hashlib
import json
import re
import shutil
import sys

import numpy as np

from ...config import load_config
from ...contracts import Question
from ...corpus.model import DocBuilder
from ...corpus.registry import CorpusSet
from ...llm import Ledger, ModelError
from ...normalize import normalize
from ..agentic import planner as P
from ..agentic.docmeta import doc_meta
from ..agentic.test_agentic import PLAN_OBJ, ScriptedClient, _question, make_corpus, make_responder
from . import tree as T
from .approach import TreeFactVerify, parse_flags
from .defaults import DEFAULTS
from .factdb import FactDB, doc_fact_rows
from .lazyvec import LazyVectors

FAILS = []
N = [0]


def check(cond, label, detail=""):
    N[0] += 1
    print(("  ok   " if cond else "  FAIL ") + label + ((" - " + str(detail)[:600]) if (detail and not cond) else ""))
    if not cond:
        FAILS.append(label)
    return cond


def fake_vector(text, dim=96):
    """Hashed character trigrams (deterministic, similarity follows shared substrings)."""
    v = np.zeros(dim, dtype=np.float32)
    t = normalize(text or "").replace("\n", " ")
    for i in range(max(1, len(t) - 2)):
        h = int(hashlib.md5(t[i:i + 3].encode("utf-8")).hexdigest()[:8], 16)
        v[h % dim] += 1.0
    return v.tolist()


class ScriptedClient2(ScriptedClient):
    def __init__(self, cfg, ledger, responder, fail_at=None, embed_fail_at=None, wall_ms=1.0):
        super().__init__(cfg, ledger, responder, fail_at=fail_at)
        self.embed_payloads = []
        self.embed_fail_at = embed_fail_at
        self.wall = wall_ms

    def _post(self, endpoint, payload):
        if endpoint == "/api/embed":
            self.embed_payloads.append(payload)
            if self.embed_fail_at is not None and len(self.embed_payloads) == self.embed_fail_at:
                raise ModelError("simulated embed failure")
            vecs = [fake_vector(x) for x in payload["input"]]
            return {"embeddings": vecs, "prompt_eval_count": sum(len(x) for x in payload["input"]) // 4}, 2.0, 0.0, 1
        data, _wall, wait, att = super()._post(endpoint, payload)
        return data, self.wall, wait, att


def en_doc():
    b = DocBuilder("A.testrep.en", "A", "en", "Test market report", source="test")
    b.add_section("kpi", "Key figures")
    for k, (kind, text) in enumerate([("paragraph", "Active users"), ("paragraph", "4.4M"),
                                      ("paragraph", "Registered merchants"), ("paragraph", "38,900")]):
        b.add_unit("kpi", "kpi#u%d" % k, kind, text)
    b.add_section("long", "A long chapter")
    b.add_unit("long", "long#p", "paragraph", ("The market grew steadily over the period and the structure changed. " * 30)
               + "In 2024 the company had 1,250 employees. " + ("Further remarks follow here about many things. " * 20))
    d = b.build()
    d.meta["fingerprint"] = "test-en"
    return d


# --------------------------------------------------------------------------
# tree
# --------------------------------------------------------------------------

def test_tree(cs):
    o = dict(DEFAULTS)
    for d in cs.docs + [en_doc()]:
        ps = T.build_passages(d, o)
        ok = all(d.text[p.span[0]:p.span[1]] == p.text for p in ps)
        same_sec = all(all(d.unit(u).section_id == p.section_id for u in p.unit_ids) for p in ps)
        if not (ok and same_sec):
            check(False, "tree.passages_verbatim_inside_one_section %s" % d.id)
            return
    check(True, "tree.passages_verbatim_inside_one_section")
    e = en_doc()
    ps = T.build_passages(e, o)
    long_ps = [p for p in ps if p.unit_ids == ["A.testrep.en::long#p"]]
    check(len(long_ps) >= 3 and max(len(p.text) for p in long_ps) <= o["passage_chars"]
          and any("1,250 employees" in p.text for p in long_ps), "tree.long_unit_windows", [len(p.text) for p in long_ps])
    check(ps[0].embed_text.startswith("Test market report > Key figures\n"), "tree.embed_text_has_path", ps[0].embed_text[:60])

    d1402 = cs.doc("C.testco_annual-report_1402.fa")
    plan = P.validate_plan(PLAN_OBJ, question="تعداد رانندگان", q_lang="fa", metas=[doc_meta(d) for d in cs.docs],
                           choose_companies=True)
    ctx = T.search_context(plan, "fa")
    # a vector hit on the number alone gets its label
    o_small = dict(o, passage_chars=12)
    ps2 = T.build_passages(d1402, o_small)
    num_p = [p for p in ps2 if p.text.strip() == "6,152,509 نفر"]
    cands = T.expand_passage(d1402, num_p[0], ctx, o, "count") if num_p else []
    check(cands and "تعداد کاربران راننده" in cands[0].text and "6,152,509" in cands[0].text
          and d1402.text[cands[0].span[0]:cands[0].span[1]] == cands[0].text, "tree.detached_number_gets_label",
          [c.text for c in cands])
    raw = T.expand_passage(d1402, num_p[0], ctx, o, "count", tree=False) if num_p else []
    check(raw and "تعداد کاربران راننده" not in raw[0].text, "tree.no_tree_keeps_bare_number", [c.text for c in raw])
    # a fact-DB row in a table: caption + header + the row
    tab_uid = "C.testco_annual-report_1402.fa::p2_b1"
    c = T.expand_unit(d1402, tab_uid, "1,103,992", ctx, o, "count")
    check(c is not None and c.text.startswith("جدول سرویس‌ها\n\nسرویس | تعداد راننده | سال") and "باکس | 1,103,992" in c.text,
          "tree.table_caption_header_row", c.text if c else None)
    # a fact on a label-only unit's next short unit
    c2 = T.expand_unit(d1402, "C.testco_annual-report_1402.fa::p1_b4", "3,305,147", ctx, o, "count")
    check(c2 is not None and "تعداد کاربران راننده" in c2.text and "3,305,147" in c2.text, "tree.unit_with_label_neighbour",
          c2.text if c2 else None)


# --------------------------------------------------------------------------
# fact DB
# --------------------------------------------------------------------------

def test_factdb(cfg, cs, tmp):
    d1402 = cs.doc("C.testco_annual-report_1402.fa")
    rows = doc_fact_rows(d1402, (1402, 1402))
    r = [x for x in rows if x["value"] == "6,152,509"]
    check(r and "کاربران راننده" in r[0]["label"] and r[0]["year"] == 1402 and r[0]["unit_id"].endswith("p1_b2"),
          "factdb.persian_detached_number_row", r[:1] or rows[:5])
    e = en_doc()
    rows_en = doc_fact_rows(e, None)
    r_en = [x for x in rows_en if x["value"] == "4.4"]
    check(r_en and "active users" in r_en[0]["label"].lower() and r_en[0]["year"] is None, "factdb.english_detached_number_row",
          r_en[:1] or rows_en[:5])
    db = FactDB(cfg, cache_path=tmp / "facts.sqlite")
    rows1, i1 = db.rows_for(d1402, (1402, 1402))
    db2 = FactDB(cfg, cache_path=tmp / "facts.sqlite")
    rows2, i2 = db2.rows_for(d1402, (1402, 1402))
    check(i1["source"] == "extracted" and i2["source"] == "cache" and rows1 == rows2, "factdb.cached_per_content_hash",
          (i1, i2))
    allrows = []
    for d in cs.docs:
        allrows += db.rows_for(d, doc_meta(d)["years"])[0]
    con = db.table(allrows)
    plan = P.validate_plan(PLAN_OBJ, question="تعداد رانندگان تست‌کو", q_lang="fa", metas=[doc_meta(d) for d in cs.docs],
                           choose_companies=True)
    ids = [d.id for d in cs.docs if d.id.startswith("C.testco")]
    kept, info = db.query(con, plan, "fa", ids, period=None, max_rows=24, per_doc=24)
    vals = {x["value"] for x in kept}
    check({"6,152,509", "3,305,147", "3,761,925"} <= vals and all(x["doc_id"] in ids for x in kept),
          "factdb.query_terms_in_labels_target_docs_only", (sorted(vals), info))
    kept2, _ = db.query(con, plan, "fa", ids, period=(1402, 1402), max_rows=24, per_doc=24)
    check(kept2 and all(x["year"] in (None, 1402) for x in kept2) and "3,761,925" not in {x["value"] for x in kept2},
          "factdb.period_filter", sorted((x["year"], x["value"]) for x in kept2))
    plan_en = dict(plan, terms={"fa": [], "en": ["active users"]})
    con_en = db.table(rows_en)
    k_en, _ = db.query(con_en, plan_en, "en", [e.id], max_rows=5, per_doc=5)
    check(k_en and k_en[0]["value"] == "4.4", "factdb.english_term_query", k_en)
    con.close()
    con_en.close()
    db.close()
    db2.close()


# --------------------------------------------------------------------------
# lazy vectors
# --------------------------------------------------------------------------

class _A:
    def __init__(self, cfg):
        self.cfg = cfg


def test_lazy(cfg, cs):
    d = cs.doc("C.testco_annual-report_1402.fa")
    o = dict(DEFAULTS, passage_chars=40)      # several passages -> several embed batches of 2
    cfg2 = cfg.copy_with({"embedding": {"batch_size": 2}})
    led = Ledger(path=None)
    c1 = ScriptedClient2(cfg2, led, make_responder(), embed_fail_at=2)
    lv = LazyVectors(_A(cfg2))
    with led.scope(phase="query", approach="AGENTIC2", stage="lazy-index"):
        try:
            lv.ensure(d, o, c1)
            check(False, "lazy.failure_mid_document_raises")
        except ModelError:
            check(True, "lazy.failure_mid_document_raises")
    lv.close()
    n_total = len(T.build_passages(d, o))
    c2 = ScriptedClient2(cfg2, led, make_responder())
    lv2 = LazyVectors(_A(cfg2))
    with led.scope(phase="query", approach="AGENTIC2", stage="lazy-index"):
        ps, m, info = lv2.ensure(d, o, c2)
    check(info["embedded_now"] == n_total - 2 and m.shape[0] == len(ps) == n_total, "lazy.resume_embeds_only_missing",
          (info, n_total))
    check(np.allclose(np.linalg.norm(m, axis=1), 1.0, atol=1e-4) and info["cached_cost_ms"] > 0,
          "lazy.normalised_and_cache_inclusive_cost", info)
    _ps, _m, info_mem = lv2.ensure(d, o, c2)
    check(info_mem["memory_hit"] == 1 and info_mem["embed_calls"] == 0, "lazy.memory_hit_second_question", info_mem)
    lv2.close()
    c3 = ScriptedClient2(cfg2, led, make_responder())
    lv3 = LazyVectors(_A(cfg2))
    with led.scope(phase="query", approach="AGENTIC2", stage="lazy-index"):
        ps3, m3, info3 = lv3.ensure(d, o, c3)
    check(info3["embedded_now"] == 0 and len(c3.embed_payloads) == 0 and np.allclose(m3, m), "lazy.new_process_embeds_nothing",
          info3)
    recs = [r for r in led.records if r.get("kind") == "embed" and r.get("ok")]
    check(recs and all(r.get("stage") == "lazy-index" and r.get("purpose") == "lazy-index-embed" and r.get("phase") == "query"
                       for r in recs), "lazy.ledger_stage_label", recs[:1])
    lv3.close()
    # B3's key function: the same text gives the same shared-cache key
    from ..baselines import VectorRetrieval
    b3 = VectorRetrieval(cfg=cfg2, client=c3)
    check(lv3.text_key("abc") == b3._text_key("abc") and str(lv3.cache_path()) == str(b3._cache_path(None)),
          "lazy.shares_b3_cache_and_key")


# --------------------------------------------------------------------------
# pipeline
# --------------------------------------------------------------------------

def _approach(cfg, cs, responder, options=None, flags=None, **client_kw):
    ledger = Ledger(path=None)
    client = ScriptedClient2(cfg, ledger, responder, **client_kw)
    ap = TreeFactVerify(cfg=cfg, client=client, flags=flags, options=dict({"call_cache": "off"}, **(options or {})))
    ap.build_index(cs, ledger)
    return ap, client


def _timeline_q():
    q = _question()
    q.type, q.answer_type = "timeline", "series"
    return q


def test_pipeline(cfg, cs, tmp):
    ap, client = _approach(cfg, cs, make_responder())
    rec = ap.answer(_question(), "fa-fa")
    tr = rec.trace or {}
    check(not rec.error, "pipeline.no_error", rec.error or (tr.get("traceback") if isinstance(tr, dict) else ""))
    if rec.error:
        return
    st = tr["stats"]
    purposes = [c["purpose"] for c in tr["calls"]]
    check(rec.llm_calls <= 4 and purposes[0] == "plan" and purposes[-1] == "answer" and rec.llm_calls == tr["ledger_calls"],
          "budget.calls_le_4_plan_first_answer_last", purposes)
    stages = [s["stage"] for s in tr["timing"]["stages"]]
    check(stages == ["plan", "lazy-index", "retrieve", "verify", "reconcile", "answer"] and tr["timing"]["reconciles"],
          "timing.stage_records_reconcile", (stages, tr["timing"]["check"]))
    li = tr["timing"]["stages"][1]
    check(li["embed_calls"] > 0 and li["llm_calls"] == 0 and st["lazy_docs_embedded"] == 4 and tr["timing"]["lazy_index_ms"] > 0,
          "lazy.first_touch_in_lazy_index_stage", li)
    check(st["docs_targeted"] == 4 and all(c["doc_id"].startswith("C.testco") for c in tr["stages"]["retrieve"]["candidates"]),
          "retrieve.only_target_documents", tr["stages"]["plan"]["doc_selection"])
    check(st["cand_factdb"] + st["cand_both"] > 0 and st["cand_vector"] + st["cand_both"] > 0 and st["candidates"] <= 24,
          "retrieve.union_vector_and_factdb_capped", {k: st[k] for k in ("cand_vector", "cand_factdb", "cand_both", "candidates")})
    series = {(r["period"], r["value"]) for r in rec.structured["series"]}
    check(("1402", "6,152,509") in series and ("1400", "1,896,210") in series, "pipeline.verified_series", sorted(series))
    check(rec.structured["comparable"] is False, "pipeline.not_comparable_flagged", rec.structured["definition_classes"])
    check(rec.evidence_items and rec.evidence_items[0]["kind"] == "fact" and rec.evidence_chars <= cfg.evidence["max_chars"],
          "pipeline.table_first_within_budget", rec.evidence_items[:2])
    check(st["verify_batches_planned"] == min(2, -(-st["candidates"] // 12)), "verify.batches_follow_candidates",
          (st["verify_batches_planned"], st["candidates"]))
    ret = tr["stages"]["retrieve"]["candidates"]
    nested = [(a["cid"], b["cid"]) for a in ret for b in ret if a is not b and a["doc_id"] == b["doc_id"]
              and a["section_id"] == b["section_id"] and set(b["unit_ids"]) <= set(a["unit_ids"]) and a["kind"] != "clip"
              and b["kind"] != "clip"]
    check(not nested, "retrieve.no_candidate_inside_another", nested)
    check(rec.embed_calls == li["embed_calls"] + st["query_embed_calls"], "ledger.embed_calls_in_record",
          (rec.embed_calls, li["embed_calls"], st["query_embed_calls"]))
    staged = [r for r in client.ledger.records if r.get("question_id") == "t-timeline"]
    check(all(r.get("phase") == "query" for r in staged) and {r.get("stage") for r in staged if r.get("kind") == "embed"}
          == {"lazy-index", "retrieve"}, "ledger.phase_query_stage_labels", {(r.get("kind"), r.get("stage")) for r in staged})

    # second question on the same documents: no lazy embedding
    rec_b = ap.answer(_question(), "fa-fa")
    check(not rec_b.error and rec_b.trace["stats"]["lazy_passages_embedded"] == 0, "lazy.second_question_free",
          rec_b.trace["stats"]["lazy_passages_embedded"] if not rec_b.error else rec_b.error)

    # timeline question: shared timeline answer prompt
    from ...contracts import ANSWER_TIMELINE_SYSTEM
    apt, ct = _approach(cfg, cs, make_responder())
    rect = apt.answer(_timeline_q(), "fa-fa")
    check(not rect.error and "one line per period" in ct.payloads[-1]["messages"][0]["content"]
          and rect.meta.get("answer_prompt") == "answer-timeline-v1", "pipeline.timeline_uses_shared_prompt", rect.error)

    # guard: a nearby wrong number with the right quote is rejected
    ap2, _ = _approach(cfg, cs, make_responder(wrong=True))
    rec2 = ap2.answer(_question(), "fa-fa")
    check(not rec2.error and rec2.trace["guard_reasons"].get("value_not_in_quote", 0) >= 1 and
          ("1400", "100,263") not in {(r["period"], r["value"]) for r in rec2.structured["series"]},
          "guard.rejects_nearby_wrong_number", rec2.trace.get("guard_reasons") if not rec2.error else rec2.error)

    # time cap: an answer reserve larger than the clock skips every verify batch, the answer is still given
    ap3, c3 = _approach(cfg, cs, make_responder(), options={"answer_reserve_s": 10000.0, "answer_reserve_timeline_s": 10000.0})
    rec3 = ap3.answer(_question(), "fa-fa")
    p3 = [c["purpose"] for c in rec3.trace["calls"]] if not rec3.error else []
    check(not rec3.error and p3 == ["plan", "answer"] and rec3.trace["stats"]["time_skipped_batches"] >= 1
          and rec3.answer_text, "time.cap_skips_verify_keeps_answer", (p3, rec3.error))
    # slow calls: the recorded wall time drives the clock (60 s cap, 25 s per call: plan 25 s, batch 1 fits, batch 2 not)
    ap4, c4 = _approach(cfg, cs, make_responder(), options={"verify_batch_max": 4, "answer_reserve_timeline_s": 12.0}, wall_ms=25000.0)
    rec4 = ap4.answer(_question(), "fa-fa")
    s4 = rec4.trace["stats"] if not rec4.error else {}
    check(not rec4.error and s4.get("time_skipped_batches", 0) == 1 and s4.get("verify_calls", 0) == 1
          and rec4.trace["calls"][-1]["purpose"] == "answer", "time.slow_calls_skip_second_batch",
          ({k: s4.get(k) for k in ("time_skipped_batches", "verify_calls", "query_clock_ms")}, rec4.error))
    # replay: the per-call cache replays the same decisions with the recorded wall time
    cache_path = tmp / "calls.sqlite"
    ap5, c5 = _approach(cfg, cs, make_responder(), options={"call_cache": "run", "call_cache_path": str(cache_path),
                                                              "verify_batch_max": 4, "answer_reserve_timeline_s": 12.0},
                        wall_ms=25000.0)
    rec5 = ap5.answer(_question(), "fa-fa")
    ap6, c6 = _approach(cfg, cs, make_responder(), options={"call_cache": "run", "call_cache_path": str(cache_path),
                                                              "verify_batch_max": 4, "answer_reserve_timeline_s": 12.0},
                        wall_ms=1.0)
    rec6 = ap6.answer(_question(), "fa-fa")
    check(not rec5.error and not rec6.error and len(c6.payloads) == 0 and rec6.answer_text == rec5.answer_text
          and rec6.trace["stats"]["time_skipped_batches"] == rec5.trace["stats"]["time_skipped_batches"] == 1,
          "cache.replay_same_decisions_recorded_clock", (len(c6.payloads), rec5.error, rec6.error))

    # a tight call cap still keeps the answer call
    ap7, _ = _approach(cfg.copy_with({"caps": {"agentic2_max_calls": 2}}), cs, make_responder())
    rec7 = ap7.answer(_question(), "fa-fa")
    check(not rec7.error and [c["purpose"] for c in rec7.trace["calls"]] == ["plan", "answer"], "budget.cap2_plan_answer",
          rec7.error or [c["purpose"] for c in rec7.trace["calls"]])


def test_flags(cfg, cs):
    check(parse_flags("no_verify,candidates=32") == {"no_verify": True, "candidates": 32}, "flags.parse_value_flag")
    for bad in ("candidates=x", "candidates=0", "no_guard"):
        try:
            parse_flags(bad)
            check(False, "flags.reject_%s" % bad)
        except ValueError:
            check(True, "flags.reject_%s" % bad)
    ap = TreeFactVerify(cfg=cfg, flags={"candidates": 32})
    check(ap.variant == "AGENTIC2+candidates=32" and ap.opts["max_candidates"] == 32, "flags.variant_name_and_cap", ap.variant)
    ap1, c1 = _approach(cfg, cs, make_responder(), flags={"no_verify": True})
    r1 = ap1.answer(_question(), "fa-fa")
    check(not r1.error and [c["purpose"] for c in r1.trace["calls"]] == ["plan", "answer"]
          and r1.trace["stages"]["answer"]["evidence_from"] == "ranked_candidates", "ablation.no_verify_two_calls",
          r1.error or [c["purpose"] for c in r1.trace["calls"]])
    ap2, _ = _approach(cfg, cs, make_responder(), flags={"no_factdb": True})
    r2 = ap2.answer(_question(), "fa-fa")
    check(not r2.error and r2.trace["stats"]["cand_factdb"] == 0 and r2.trace["stats"]["cand_both"] == 0
          and r2.trace["stats"]["factdb_rows"] == 0, "ablation.no_factdb", r2.error)
    ap3, _ = _approach(cfg, cs, make_responder(), flags={"no_tree": True})
    r3 = ap3.answer(_question(), "fa-fa")
    check(not r3.error and ap3.variant == "AGENTIC2+no_tree", "ablation.no_tree_runs", r3.error)
    ap4, c4 = _approach(cfg, cs, make_responder(), flags={"no_vectors": True})
    r4 = ap4.answer(_question(), "fa-fa")
    check(not r4.error and r4.trace["stats"]["cand_grep"] > 0 and len(c4.embed_payloads) == 0
          and r4.trace["stats"]["lazy_docs_embedded"] == 0, "ablation.no_vectors_uses_v1_grep",
          r4.error or r4.trace["stats"]["cand_grep"])
    ap5, _ = _approach(cfg, cs, make_responder(), flags={"candidates": 4})
    r5 = ap5.answer(_question(), "fa-fa")
    check(not r5.error and r5.trace["stats"]["candidates"] <= 4 and r5.trace["stats"]["verify_batches_planned"] == 1,
          "ablation.candidates_cap", r5.error or r5.trace["stats"]["candidates"])
    # single document, single fact, English: fact DB + vectors on detached English numbers
    e = en_doc()
    cse = CorpusSet("A_en", [e])
    q = Question(id="t-en", corpus="A", question_fa="", question_en="How many active users does the test market report give?",
                 answer_fa="", answer_en="4.4M", answer_type="numeric", gold_units={"en": ["A.testrep.en::kpi#u1"]},
                 doc_scope=["A.testrep.en"], type="lookup", answer_value=4400000)
    plan = {"entity": "test market", "metric": "active users", "kind": "single", "terms_fa": ["کاربران فعال"],
            "terms_en": ["active users"], "entity_terms": ["test market"], "unit": "count", "period_from": "", "period_to": ""}

    def respond(payload):
        system = payload["messages"][0]["content"]
        if system.startswith("You plan a search"):
            return json.dumps(plan)
        if system.startswith("You check candidate snippets"):
            user = payload["messages"][-1]["content"]
            out = {}
            for m in re.finditer(r"\[(\d+)\] \([^\n]*\)\n(.*?)(?=\n\n\[\d+\] \(|\Z)", user.split("Snippets:\n", 1)[1], re.S):
                ok = "4.4M" in m.group(2) and "Active users" in m.group(2)
                out["s" + m.group(1)] = {"relevant": ok, "values": [{"value": "4.4M", "unit": "", "period": "",
                                                                     "definition": "active users",
                                                                     "quote": "Active users\n\n4.4M"}] if ok else []}
            return json.dumps(out)
        return "4.4M"
    ape, ce = _approach(cfg, cse, respond)
    re_ = ape.answer(q, "en-en")
    st = re_.trace["stats"] if not re_.error else {}
    check(not re_.error and st.get("rows", 0) >= 1 and "A.testrep.en::kpi#u1" in re_.evidence_units
          and st.get("verify_calls") == 1, "pipeline.english_single_fact_detached_number",
          re_.error or {k: st.get(k) for k in ("rows", "candidates", "cand_factdb", "cand_both", "verify_calls")})


def test_verify_repairs(cfg):
    """Arabic-comma thousands (2،788،550) and a number-only quote whose label is on the previous line."""
    b = DocBuilder("C.shipco_annual-report_1403.fa", "C", "fa", "گزارش سالانه شیپ‌کو 1403", source="test")
    b.add_section("p1", "آمار مشتریان")
    for k, text in enumerate(["تعداد مشتریان فرستنده", "2،788،550", "تعداد مشتریان گیرنده", "12،993،237"]):
        b.add_unit("p1", "p1_b%d" % k, "paragraph", text)
    d = b.build()
    d.meta["fingerprint"] = "test-ship"
    cs = CorpusSet("C", [d])
    q = Question(id="t-ship", corpus="C", question_fa="در گزارش 1403 شیپ‌کو تعداد مشتریان فرستنده چند نفر است؟",
                 question_en="", answer_fa="2,788,550", answer_en="", answer_type="numeric",
                 gold_units={"fa": ["C.shipco_annual-report_1403.fa::p1_b1"]}, doc_scope=["C.shipco_annual-report_1403.fa"],
                 type="lookup", answer_value=2788550)
    plan = {"entity": "شیپ‌کو", "metric": "تعداد مشتریان فرستنده", "kind": "single", "terms_fa": ["مشتریان فرستنده"],
            "terms_en": ["sending customers"], "entity_terms": ["شیپ‌کو"], "unit": "count", "period_from": "1403",
            "period_to": "1403"}

    def respond(payload):
        system = payload["messages"][0]["content"]
        if system.startswith("You plan a search"):
            return json.dumps(plan, ensure_ascii=False)
        if system.startswith("You check candidate snippets"):
            user = payload["messages"][-1]["content"]
            out = {}
            for m in re.finditer(r"\[(\d+)\] \([^\n]*\)\n(.*?)(?=\n\n\[\d+\] \(|\Z)", user.split("Snippets:\n", 1)[1], re.S):
                ok = "2,788,550" in m.group(2)
                out["s" + m.group(1)] = {"relevant": ok, "values": [{"value": "2,788,550", "unit": "نفر", "period": "1403",
                                                                     "definition": "مشتریان فرستنده",
                                                                     "quote": "2,788,550"}] if ok else []}
            return json.dumps(out, ensure_ascii=False)
        return "2,788,550"
    ap, _c = _approach(cfg, cs, respond)
    rec = ap.answer(q, "fa-fa")
    st = rec.trace["stats"] if not rec.error else {}
    vals = [(r["value"], r.get("value_num")) for r in (rec.structured or {}).get("series", [])]
    check(not rec.error and st.get("thousands_fixed", 0) >= 1 and st.get("label_completed", 0) >= 1
          and any(v == 2788550.0 for _t, v in vals) and not rec.trace["guard_reasons"].get("quote_no_label"),
          "verify.thousands_fix_and_label_completion", (rec.error, {k: st.get(k) for k in ("thousands_fixed", "label_completed")},
                                                         vals, rec.trace.get("guard_reasons")))
    ap2, _c2 = _approach(cfg, cs, respond, options={"thousands_fix": False, "label_completion": False})
    rec2 = ap2.answer(q, "fa-fa")
    check(not rec2.error and not (rec2.structured or {}).get("series"), "verify.repairs_off_nothing_verified",
          (rec2.error, (rec2.structured or {}).get("series")))


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass
    base = load_config()
    tmp = base.project_root / "work" / "test_agentic2"
    if tmp.exists():
        shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True, exist_ok=True)
    cfg = base.copy_with({"paths": {"index_root": str(tmp / "indexes")}})
    cs = make_corpus()
    print("tree");     test_tree(cs)
    print("factdb");   test_factdb(cfg, cs, tmp)
    print("lazy");     test_lazy(cfg, cs)
    print("pipeline"); test_pipeline(cfg, cs, tmp)
    print("flags");    test_flags(cfg, cs)
    print("repairs");  test_verify_repairs(cfg)
    print("\n%d checks, %d failure(s)" % (N[0], len(FAILS)))
    for f in FAILS:
        print("  FAILED:", f)
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
