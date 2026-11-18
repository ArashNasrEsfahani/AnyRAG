"""AGENTIC5 offline tests (no model, no GPU, no network: a scripted client answers the plan, verify and answer calls;
embeddings, if any were asked for, are hashed trigrams). Temp knowledge base and indexes only, under
work/upgrades/agentic5/test_tmp (deleted on every run); the study knowledge base is never opened.

    .venv/Scripts/python.exe -m ragbench.approaches.agentic5.test_agentic5

Covers the reviewer's deterministic correctness list (UPGRADES_ASTRA_session.log, "Deterministic correctness
checks"): two companies with equal values in one year stay two rows; two metrics in one year; an absent requested
period stays missing; annual vs quarterly; table rows sharing one unit id; citation renumbering after packing; scope
isolation; write-back disabled; malformed plans; budget exhaustion; the 4-call cap. Plus: routing (shortlist <= 8,
explicit matches retained, escape candidates, no recency fallback), allocation (requirements before channels,
batch 1 <= 12 balanced), the removed definition escape, arithmetic (unknown never zero), the deadline at each model
request, the shared foundation (hybrid index loaded; KB fallback with no_hybrid; no query-time document embedding).
"""
import json
import re
import shutil
import sys

from ...config import load_config
from ...contracts import Question, RetrievedItem, Setting
from ...corpus.model import DocBuilder
from ...corpus.registry import CorpusSet
from ...llm import Ledger
from ..agentic import planner as P
from ..agentic import search as S
from ..agentic.docmeta import doc_meta
from ..agentic2.test_agentic2 import ScriptedClient2
from ..agentic3 import reconcile_sql as RS
from . import allocate as AL
from . import arith as AR
from . import claims as CL
from . import evidence5 as EV
from . import reconcile5 as RC
from . import requirements as RQ
from . import routing as RO5
from .approach import RequirementAgent
from .defaults import DEFAULTS

FAILS = []
N = [0]


def check(cond, label, detail=""):
    N[0] += 1
    print(("  ok   " if cond else "  FAIL ") + label + ((" - " + str(detail)[:700]) if (detail and not cond) else ""))
    if not cond:
        FAILS.append(label)
    return cond


# ---------------------------------------------------------------------------------------------------------------
# a synthetic English pool: two companies with EQUAL revenue in 2023, a second metric with the same value, a table
# row holding two years in one unit, a quarterly figure, a restating report, and fillers that out-score the
# companies on the question's words (routing must still retain the named companies)
# ---------------------------------------------------------------------------------------------------------------

FILL = ("The board reviewed the strategy, the markets and the outlook for the coming year in detail, and noted the "
        "progress of the programmes described in the other chapters of this report. ")


def _doc(doc_id, title, sections):
    b = DocBuilder(doc_id, "B", "en", title, source="test")
    for sec_local, sec_title, units in sections:
        b.add_section(sec_local, sec_title)
        for k, (kind, text) in enumerate(units):
            b.add_unit(sec_local, "%s_b%d" % (sec_local, k), kind, text)
    d = b.build()
    d.meta["fingerprint"] = "test-" + doc_id
    return d


ALPHA23 = "B.alpha_annual-report_2023.en"
BETA23 = "B.beta_annual-report_2023.en"
ALPHA22 = "B.alpha_annual-report_2022.en"
GAMMA21 = "B.gamma_annual-report_2021.en"


def make_pool(n_fillers=10):
    docs = [
        _doc(ALPHA23, "Alpha Corp annual report 2023", [
            ("p1", "Financial highlights", [
                ("paragraph", "Alpha Corp revenue in 2023 was 100 million USD, the highest in its history. " + FILL),
                ("paragraph", "Net profit in 2023 was 100 million USD after tax. " + FILL),
                ("paragraph", "Revenue in the first quarter of 2023 was 30 million USD. " + FILL)]),
            ("p2", "Five-year table", [
                ("caption", "Table 1: Alpha Corp key figures"),
                ("table", "Metric | 2023 | 2022\nRevenue (million USD) | 100 | 92\nEmployees | 5,000 | 4,800")]),
        ]),
        _doc(BETA23, "Beta Inc annual report 2023", [
            ("p1", "Results", [("paragraph", "Beta Inc revenue in 2023 was 100 million USD. " + FILL),
                               ("paragraph", "Beta Inc employs 2,000 people. " + FILL)]),
        ]),
        _doc(ALPHA22, "Alpha Corp annual report 2022", [
            ("p1", "Financial highlights", [("paragraph", "Alpha Corp revenue in 2022 was 90 million USD. " + FILL)]),
        ]),
        _doc(GAMMA21, "Gamma Ltd annual report 2021", [
            ("p1", "Results", [("paragraph", "Gamma Ltd revenue in 2021 was 55 million USD. " + FILL)]),
        ]),
    ]
    for i in range(n_fillers):
        docs.append(_doc("B.filler%02d_sector-review_2020.en" % i, "Sector review %d on revenue" % i, [
            ("p1", "Revenue outlook", [("paragraph", "Revenue, revenue growth and revenue drivers of the sector: "
                                                      "revenue revenue revenue in 2023 and 2022. " + FILL)])]))
    return CorpusSet("B_en", docs)


# ---------------------------------------------------------------------------------------------------------------
# the scripted model
# ---------------------------------------------------------------------------------------------------------------

def snippets_of(user):
    body = user.split("Snippets:\n", 1)[1]
    parts = re.split(r"\n\n(?=\[\d+\] \()", "\n\n" + body)
    out = []
    for p in parts:
        m = re.match(r"\[(\d+)\] \(([^\n]*)\)\n(.*)", p.strip(), re.S)
        if m:
            out.append((int(m.group(1)), m.group(2), m.group(3)))
    return out


def requirement_lines(user):
    block = user.split("Requirements (id: entity; metric; period):\n", 1)[1].split("\n\nSnippets:", 1)[0]
    out = []
    for ln in block.split("\n"):
        m = re.match(r"(R\d+): entity=(.*?); metric=(.*?); period=(.*)$", ln.strip())
        if m:
            out.append({"id": m.group(1), "entity": m.group(2), "metric": m.group(3), "period": m.group(4)})
    return out


# (must be in the snippet, must be in the header or "", item) - what a careful typed verifier would answer
RULES = [
    ("Alpha Corp revenue in 2023 was 100 million USD", "", {
        "entity": "Alpha Corp", "metric": "revenue", "period": "2023", "definition": "revenue",
        "value": "100", "unit": "million USD", "quote": "Alpha Corp revenue in 2023 was 100 million USD"}),
    ("Net profit in 2023 was 100 million USD", "", {
        "entity": "Alpha Corp", "metric": "net profit", "period": "2023", "definition": "net profit after tax",
        "value": "100", "unit": "million USD", "quote": "Net profit in 2023 was 100 million USD"}),
    ("Beta Inc revenue in 2023 was 100 million USD", "", {
        "entity": "Beta Inc", "metric": "revenue", "period": "2023", "definition": "revenue",
        "value": "100", "unit": "million USD", "quote": "Beta Inc revenue in 2023 was 100 million USD"}),
    ("Revenue in the first quarter of 2023 was 30 million USD", "", {
        "entity": "Alpha Corp", "metric": "revenue", "period": "Q1 2023", "definition": "quarterly revenue",
        "value": "30", "unit": "million USD", "quote": "Revenue in the first quarter of 2023 was 30 million USD"}),
    ("Alpha Corp revenue in 2022 was 90 million USD", "", {
        "entity": "Alpha Corp", "metric": "revenue", "period": "2022", "definition": "revenue",
        "value": "90", "unit": "million USD", "quote": "Alpha Corp revenue in 2022 was 90 million USD"}),
    ("Revenue (million USD) | 100 | 92", "", {
        "entity": "Alpha Corp", "metric": "revenue", "period": "2022", "definition": "revenue (table)",
        "value": "92", "unit": "million USD", "quote": "Revenue (million USD) | 100 | 92"}),
]


def pick_req(reqs, it):
    for r in reqs:
        ent_ok = (not r["entity"] or r["entity"].startswith("(")) or \
            r["entity"].lower().split()[0] in it["entity"].lower()
        met_ok = r["metric"].lower() in it["metric"].lower() or it["metric"].lower() in r["metric"].lower()
        per_ok = r["period"].startswith("(") or r["period"] == it["period"]
        if ent_ok and met_ok and per_ok:
            return r["id"]
    return ""


def verify_reply(payload, rules=RULES, force_req=None):
    user = payload["messages"][-1]["content"]
    reqs = requirement_lines(user)
    out = {}
    for i, header, text in snippets_of(user):
        found = []
        for must, head, it in rules:
            if must in text and (not head or head in header) and len(found) < 3:
                v = dict(it)
                v["req"] = force_req if force_req is not None else pick_req(reqs, it)
                found.append(v)
        out["s%d" % i] = {"relevant": bool(found), "values": found}
    return json.dumps(out, ensure_ascii=False)


class Script:
    """A responder that also keeps what the answer call was shown."""

    def __init__(self, plan=None, plan_text=None, rules=RULES, force_req=None, repair_text=None):
        self.plan, self.plan_text, self.rules, self.force_req = plan, plan_text, rules, force_req
        self.repair_text = repair_text
        self.evidence = []
        self.purposes = []

    def __call__(self, payload):
        system = payload["messages"][0]["content"]
        if system.startswith("You plan a search"):
            if len(payload["messages"]) > 2 and self.repair_text is not None:
                return self.repair_text
            return self.plan_text if self.plan_text is not None else json.dumps(self.plan, ensure_ascii=False)
        if system.startswith("You check candidate snippets"):
            return verify_reply(payload, self.rules, self.force_req)
        self.evidence.append(payload["messages"][-1]["content"])
        return "answer"


def plan_obj(entities, metrics, periods, operation, kind="comparison", unit="currency"):
    terms_en = [t for m in metrics for t in m["terms"]]
    return {"entity": entities[0] if entities else "", "metric": metrics[0]["name"], "kind": kind,
            "terms_fa": [], "terms_en": terms_en, "entity_terms": entities, "unit": unit, "period_from": "",
            "period_to": "", "companies": [], "entities": entities, "metrics": metrics, "periods": periods,
            "operation": operation}


REVENUE = {"name": "revenue", "terms": ["revenue"]}
PROFIT = {"name": "net profit", "terms": ["net profit"]}


def question(text, qid="t5", scope="corpus"):
    return Question(id=qid, corpus="B", question_fa="", question_en=text, answer_fa="", answer_en="",
                    answer_type="short", gold_units={"en": []}, doc_scope=scope, type="cross-company")


def approach(cfg, cs, script, *, flags=None, options=None, wall_ms=1.0):
    ledger = Ledger(path=None)
    client = ScriptedClient2(cfg, ledger, script, wall_ms=wall_ms)
    ap = RequirementAgent(cfg=cfg, client=client, flags=flags, options=dict({"call_cache": "off"}, **(options or {})))
    index = ap.build_index(cs, ledger)
    return ap, client, index


def run_q(cfg, cs, script, text, *, flags=None, options=None, scope="corpus", wall_ms=1.0, qid="t5"):
    ap, client, index = approach(cfg, cs, script, flags=flags, options=options, wall_ms=wall_ms)
    rec = ap.answer(question(text, qid=qid, scope=scope), Setting.parse("en-en"))
    return ap, client, rec


def req_status(rec):
    return {r["id"]: r["status"] for r in rec.structured["requirements"]}


# ---------------------------------------------------------------------------------------------------------------
# helpers for the pure-function tests
# ---------------------------------------------------------------------------------------------------------------

def matrix_of(reqs_spec, operation="comparison"):
    """[(entity, key, metric_id, metric, year | None | "2023Q1")] -> a requirement matrix."""
    reqs, ents, mets = [], {}, {}
    for i, (ent, key, mid, metric, per) in enumerate(reqs_spec, 1):
        ents.setdefault(key, "E%d" % (len(ents) + 1))
        mets.setdefault(mid, {"id": mid, "name": metric, "terms": {"en": [metric], "fa": []}})
        if per is None:
            p = None
        elif isinstance(per, str) and "Q" in per:
            p = RQ.period(int(per[:4]), "quarter", int(per[-1]))
        else:
            p = RQ.period(per)
        reqs.append({"id": "R%d" % i, "entity_id": ents[key], "entity": ent, "entity_key": key,
                     "entity_keys": [key], "entity_terms": [ent, key], "metric_id": mid, "metric": metric,
                     "terms": {"en": [metric], "fa": []}, "period": p, "period_key": p["key"] if p else "",
                     "operation": operation, "unit": "currency"})
    return {"requirements": reqs, "metrics": list(mets.values()), "entities": [], "periods": [],
            "operation": operation, "open_series": False, "fixes": [], "truncated": 0}


def claim(req, value, *, doc="B.alpha_annual-report_2023.en", years=(2023, 2023), cid="c1", unit_id="u1",
          period=None, def_class="unspecified", unit="million USD", score=1.0, kid=None):
    per = period or (req["period"]["key"] if req and req.get("period") else "")
    ptype = "quarter" if "Q" in per else "annual"
    return {"req_id": req["id"] if req else None, "entity_key": req["entity_key"] if req else "",
            "entity_name": req["entity"] if req else "", "metric_id": req["metric_id"] if req else None,
            "period_key": per, "period_type": ptype, "period_year": int(per[:4]) if per else None,
            "period": per, "def_class": def_class, "unit_family": "money:usd", "unit": unit, "unit_norm": "USD",
            "unit_status": "verified", "scale": 1e6, "currency": "usd", "value": str(value),
            "value_num": float(value), "value_norm": float(value) * 1e6, "value_raw": str(value),
            "doc_id": doc, "doc_years": list(years), "cid": cid, "unit_id": unit_id, "score": score,
            "definition": req["metric"] if req else "", "quote": "%s %s" % (req["metric"] if req else "", value),
            "claim_id": kid or "k_%s_%s" % (cid, value), "verified": True}


# ---------------------------------------------------------------------------------------------------------------
# 1. the requirement matrix and malformed plans
# ---------------------------------------------------------------------------------------------------------------

def test_requirements(cs):
    o = dict(DEFAULTS)
    metas = [doc_meta(d) for d in cs.docs]
    q = "What was the revenue of Alpha Corp and Beta Inc in 2023?"
    obj = plan_obj(["Alpha Corp", "Beta Inc"], [REVENUE], ["2023"], "comparison")
    plan = P.validate_plan(obj, question=q, q_lang="en", metas=metas, choose_companies=True, opts=o)
    m = RQ.build_requirements(obj, plan, q, metas, o)
    rs = m["requirements"]
    check(len(rs) == 2 and [r["entity_key"] for r in rs] == ["alpha", "beta"]
          and all(r["period_key"] == "2023" for r in rs) and m["operation"] == "comparison",
          "plan.two_entities_one_period_two_requirements", [(r["entity_key"], r["period_key"]) for r in rs])
    check(rs[0]["terms"] == rs[1]["terms"] and "revenue" in rs[0]["terms"]["en"],
          "plan.metric_vocabulary_shared_across_requirements", [r["terms"] for r in rs])
    obj2 = plan_obj(["Alpha Corp"], [REVENUE, PROFIT], ["2023"], "comparison")
    m2 = RQ.build_requirements(obj2, P.validate_plan(obj2, question=q, q_lang="en", metas=metas,
                                                     choose_companies=True, opts=o), q, metas, o)
    check([r["metric_id"] for r in m2["requirements"]] == ["M1", "M2"], "plan.two_metrics_two_requirements",
          [(r["metric_id"], r["metric"]) for r in m2["requirements"]])
    # required periods come from the QUESTION even when no report of that period exists in the pool
    q3 = "What was Alpha Corp revenue in 2019?"
    m3 = RQ.build_requirements({}, P.program_plan(q3, "en", metas, True, o), q3, metas, o)
    check([r["period_key"] for r in m3["requirements"]] == ["2019"], "plan.period_without_any_report_is_kept",
          [r["period_key"] for r in m3["requirements"]])
    # a plan period the question does not name is ignored; a series range is expanded
    q4 = "How did Alpha Corp revenue change from 2020 to 2023?"
    obj4 = plan_obj(["Alpha Corp"], [REVENUE], ["2019", "2020", "2023"], "series", kind="timeline")
    m4 = RQ.build_requirements(obj4, P.validate_plan(obj4, question=q4, q_lang="en", metas=metas,
                                                     choose_companies=True, opts=o), q4, metas, o)
    check([r["period_key"] for r in m4["requirements"]] == ["2020", "2021", "2022", "2023"]
          and "plan_period_not_in_question" in m4["fixes"], "plan.series_range_from_question_and_foreign_period_dropped",
          ([r["period_key"] for r in m4["requirements"]], m4["fixes"]))
    # quarters: an annual question year refined by the plan to a quarter
    q5 = "What was Alpha Corp revenue in the first three months of 2023?"
    obj5 = plan_obj(["Alpha Corp"], [REVENUE], ["Q1 2023"], "lookup", kind="single")
    m5 = RQ.build_requirements(obj5, plan, q5, metas, o)
    check([r["period_key"] for r in m5["requirements"]] == ["2023Q1"], "plan.quarter_refines_the_question_year",
          [r["period_key"] for r in m5["requirements"]])
    # malformed plans: every field junk, the question's entities and periods survive
    junk = [None, "not json", {}, {"entities": "Alpha Corp", "periods": [2023, "xyz", {"a": 1}], "operation": "boom",
                                   "metrics": "revenue"},
            {"entities": [None, 5, ""], "metrics": [{"name": None, "terms": None}, 7], "periods": None,
             "operation": None}, {"terms_fa": [], "terms_en": []}]
    ok = []
    for j in junk:
        pl = P.program_plan(q, "en", metas, True, o)
        mj = RQ.build_requirements(j, pl, q, metas, o)
        keys = sorted({r["entity_key"] for r in mj["requirements"]})
        ok.append((keys == ["alpha", "beta"] and all(r["period_key"] == "2023" for r in mj["requirements"])
                   and mj["operation"] in ("comparison", "lookup") and mj["requirements"], keys, mj["fixes"]))
    check(all(x[0] for x in ok), "plan.malformed_plans_keep_question_entities_and_periods", ok)
    check(P.parse_plan_text("garbage")[0] is None and P.plan_problem(None), "plan.garbage_is_a_problem")
    sch = RQ.plan_schema(["alpha", "beta"], o)
    check(list(sch["properties"])[-4:] == ["entities", "metrics", "periods", "operation"],
          "plan.new_fields_are_last_in_the_schema", list(sch["properties"]))
    periods, _ = RQ.parse_periods("revenue in Q1 2023 vs Q1 2022 and in 2019 and 2023")
    check([p["key"] for p in periods] == ["2023Q1", "2022Q1", "2019", "2023"],
          "plan.periods_parsed_with_type_and_no_false_range", [p["key"] for p in periods])


# ---------------------------------------------------------------------------------------------------------------
# 2. reconciliation by identity (pure)
# ---------------------------------------------------------------------------------------------------------------

def test_reconcile():
    m = matrix_of([("Alpha Corp", "alpha", "M1", "revenue", 2023), ("Beta Inc", "beta", "M1", "revenue", 2023)])
    r1, r2 = m["requirements"]
    cl = [claim(r1, 100, cid="c1"), claim(r2, 100, doc=BETA23, cid="c2", unit_id="u2")]
    rec = RC.reconcile([dict(c) for c in cl], m)
    rows = [c for c in rec["rows"] if c["req_status"] == "verified"]
    check(len(rows) == 2 and {c["entity_key"] for c in rows} == {"alpha", "beta"},
          "reconcile.two_companies_equal_values_one_year_stay_two_rows", [(c["entity_key"], c["value"]) for c in rows])
    # the same two rows through AGENTIC3's year-only reconciler collapse into one (the bug AGENTIC5 fixes)
    v3 = RS.reconcile([dict(c, role="authoritative") for c in cl], kind="comparison", target_periods=[2023])
    check(len(v3["series"]) == 1, "reconcile.agentic3_collapses_them_to_one_row_reproduced", len(v3["series"]))
    m2 = matrix_of([("Alpha Corp", "alpha", "M1", "revenue", 2023), ("Alpha Corp", "alpha", "M2", "net profit", 2023)])
    a, b = m2["requirements"]
    rec2 = RC.reconcile([claim(a, 100, cid="c1"), claim(b, 100, cid="c2", unit_id="u2")], m2)
    check(len(rec2["rows"]) == 2 and {c["metric_id"] for c in rec2["rows"]} == {"M1", "M2"}
          and not rec2["missing"], "reconcile.two_metrics_one_year_two_rows", [c["metric_id"] for c in rec2["rows"]])
    # restatement: the 2023 report restates 2022; kept = the 2022 report's own value, the difference is a note, and
    # Beta's rows are untouched
    m3 = matrix_of([("Alpha Corp", "alpha", "M1", "revenue", 2022), ("Beta Inc", "beta", "M1", "revenue", 2022)])
    a3, b3 = m3["requirements"]
    rec3 = RC.reconcile([claim(a3, 90, doc=ALPHA22, years=(2022, 2022), cid="c1"),
                         claim(a3, 92, doc=ALPHA23, years=(2023, 2023), cid="c2", unit_id="u9"),
                         claim(b3, 92, doc=BETA23, years=(2023, 2023), cid="c3", unit_id="u3")], m3)
    e1, e2 = rec3["requirements"]["R1"], rec3["requirements"]["R2"]
    check([c["value"] for c in e1["rows"]] == ["90"] and len(e1["conflicts"]) == 1
          and e1["conflicts"][0]["restated"] == "92", "reconcile.restatement_within_the_lineage_only", e1["conflicts"])
    check([c["value"] for c in e2["rows"]] == ["92"] and e2["rows"][0]["kept_role"] == "latest_restatement"
          and not e2["conflicts"], "reconcile.other_entity_not_mixed_into_the_lineage", e2["rows"])
    # an unresolved disagreement: two authoritative values in one identity
    rec4 = RC.reconcile([claim(r1, 100, cid="c1"), claim(r1, 120, cid="c5", unit_id="u5")],
                        matrix_of([("Alpha Corp", "alpha", "M1", "revenue", 2023)], "lookup"))
    e4 = rec4["requirements"]["R1"]
    check(e4["status"] == "conflict" and len(e4["rows"]) == 2, "reconcile.disagreement_preserved_not_resolved",
          (e4["status"], [c["value"] for c in e4["rows"]]))
    # a missing requirement is missing, never a zero row
    rec5 = RC.reconcile([claim(r1, 100)], m)
    check(rec5["missing"] == ["R2"] and not rec5["requirements"]["R2"]["rows"], "reconcile.missing_is_missing",
          rec5["missing"])
    # table rows sharing ONE unit id are two claims, both kept
    m6 = matrix_of([("Alpha Corp", "alpha", "M1", "revenue", 2023), ("Alpha Corp", "alpha", "M1", "revenue", 2022)],
                   "series")
    x, y = m6["requirements"]
    rec6 = RC.reconcile([claim(x, 100, cid="c7", unit_id="tbl::row1"),
                         claim(y, 92, cid="c7", unit_id="tbl::row1", years=(2023, 2023))], m6)
    check(len(rec6["rows"]) == 2 and {c["period_key"] for c in rec6["rows"]} == {"2023", "2022"},
          "reconcile.table_row_sharing_one_unit_gives_two_rows", [(c["period_key"], c["value"]) for c in rec6["rows"]])
    return rec6, m6


# ---------------------------------------------------------------------------------------------------------------
# 3. binding: annual vs quarterly, entity, metric (pure)
# ---------------------------------------------------------------------------------------------------------------

def test_binding():
    m = matrix_of([("Alpha Corp", "alpha", "M1", "revenue", 2023), ("Beta Inc", "beta", "M1", "revenue", 2023)])
    r1, r2 = m["requirements"]
    rx = {"M1": CL.term_rx({"en": ["revenue"]})}
    c = S.Candidate(ALPHA23, "sec", ["u1"], "Revenue in the first quarter of 2023 was 30 million USD.", None, "unit",
                    ["u1"])
    q = {"req": "R1", "entity": "Alpha Corp", "period": "Q1 2023", "quote": "Revenue in the first quarter of 2023 "
                                                                            "was 30 million USD"}
    cper, _how = CL.claim_period(q["period"], q["quote"], (2023, 2023))
    req, how, det = CL.bind(q, c, m["requirements"], rx, {"company": "alpha"}, cper, "")
    check(cper["type"] == "quarter" and req is None and det["checks"]["period"] == "period_type_mismatch",
          "bind.a_quarter_never_satisfies_an_annual_requirement", (cper, how, det))
    a = {"req": "R1", "entity": "Alpha Corp", "period": "2023", "quote": "Alpha Corp revenue in 2023 was 100"}
    req, how, _ = CL.bind(a, c, m["requirements"], rx, {"company": "alpha"}, RQ.period(2023), "")
    check(req is not None and req["id"] == "R1" and how == "model", "bind.model_requirement_accepted_when_checked")
    # the model names R1 (Alpha) for a value in Beta's report that names Beta: re-bound to R2
    cb = S.Candidate(BETA23, "sec", ["u2"], "Beta Inc revenue in 2023 was 100 million USD.", None, "unit", ["u2"])
    b = {"req": "R1", "entity": "Beta Inc", "period": "2023", "quote": "Beta Inc revenue in 2023 was 100 million USD"}
    req, how, _ = CL.bind(b, cb, m["requirements"], rx, {"company": "beta"}, RQ.period(2023), "")
    check(req is not None and req["id"] == "R2" and how == "rebound", "bind.wrong_entity_is_rebound_by_the_source",
          (req and req["id"], how))
    # a metric the quote does not carry does not bind
    np_ = {"req": "R1", "entity": "Alpha Corp", "period": "2023", "quote": "Net profit in 2023 was 100 million USD"}
    req, how, _ = CL.bind(np_, c, m["requirements"], rx, {"company": "alpha"}, RQ.period(2023), "")
    check(req is None, "bind.metric_must_stand_in_the_quote_or_source_label", how)
    # an entity that maps to no company key: the ONLY entity of the matrix is a label (any document binds); with two
    # such entities the document's title or the snippet must name it
    mu = matrix_of([("China", None, "M1", "revenue", 2023)])
    mu["requirements"][0]["entity_keys"] = []
    mu["requirements"][0]["entity_terms"] = ["China"]
    cm = S.Candidate("B.market_report_2023.en", "sec", ["u1"], "Total revenue in 2023 was 7 billion USD.", None,
                     "unit", ["u1"])
    it = {"req": "R1", "entity": "", "period": "2023", "quote": "Total revenue in 2023 was 7 billion USD"}
    req, how, _ = CL.bind(it, cm, mu["requirements"], rx, {"company": "market", "title": "Market report"},
                          RQ.period(2023), "")
    check(req is not None, "bind.single_unkeyed_entity_is_a_label_not_a_filter", how)
    m2 = matrix_of([("China", "cn", "M1", "revenue", 2023), ("India", "in", "M1", "revenue", 2023)])
    for r, name in zip(m2["requirements"], ("China", "India")):
        r["entity_keys"], r["entity_terms"] = [], [name]
    req, how, _ = CL.bind(dict(it, req=""), cm, m2["requirements"], rx,
                          {"company": "market", "title": "India market report 2023"}, RQ.period(2023), "")
    check(req is not None and req["id"] == "R2", "bind.two_unkeyed_entities_need_the_title_or_the_snippet",
          (req and req["id"], how))
    # the removed escape: a model DEFINITION containing the term is not evidence
    cand = S.Candidate(ALPHA23, "sec", ["u1"], "Sales proceeds 2023\n\n77 million USD", None, "unit", ["u1"])
    cand.features = {"fact": {"label": "income from operations"}}
    lab = CL.source_label(cand)
    check(lab == "", "guard.fact_label_not_in_the_snippet_is_not_source_text", lab)
    cand.features = {"fact": {"label": "Sales proceeds 2023"}}
    check(CL.source_label(cand) == "Sales proceeds 2023", "guard.fact_label_in_the_snippet_counts_as_source")
    check(DEFAULTS["guard_escape"] is False and DEFAULTS["guard_escape_label"] is False,
          "guard.definition_escape_is_off_in_agentic5")


# ---------------------------------------------------------------------------------------------------------------
# 4. arithmetic (pure)
# ---------------------------------------------------------------------------------------------------------------

def test_arith():
    m = matrix_of([("Alpha Corp", "alpha", "M1", "revenue", 2023), ("Beta Inc", "beta", "M1", "revenue", 2023)],
                  "difference")
    r1, r2 = m["requirements"]
    rec = RC.reconcile([claim(r1, 130, kid="k1"), claim(r2, 100, doc=BETA23, cid="c2", kid="k2")], m)
    res = AR.compute(m, rec)
    check(res and res[0]["status"] == "ok" and abs(res[0]["value"] - 30e6) < 1 and res[0]["operands"] == ["k1", "k2"]
          and "30" in res[0]["text"], "arith.difference_from_verified_operands_with_citations", res)
    rec2 = RC.reconcile([claim(r1, 130, kid="k1")], m)
    res2 = AR.compute(m, rec2)
    check(res2 and res2[0]["status"] == "unknown" and res2[0]["value"] is None and "missing" in res2[0]["reason"],
          "arith.a_missing_operand_is_unknown_never_zero", res2)
    bad = claim(r2, 100, doc=BETA23, cid="c2", kid="k2")
    bad["unit_family"] = "count"
    res3 = AR.compute(m, RC.reconcile([claim(r1, 130, kid="k1"), bad], m))
    check(res3[0]["status"] == "unknown" and res3[0]["reason"] == "incompatible units",
          "arith.incompatible_units_are_unknown", res3)
    mr = dict(m, operation="ratio")
    res4 = AR.compute(mr, rec)
    check(res4[0]["status"] == "ok" and abs(res4[0]["value"] - 1.3) < 1e-9, "arith.ratio", res4)
    ms = dict(m, operation="sum")
    res5 = AR.compute(ms, rec)
    check(res5[0]["status"] == "ok" and abs(res5[0]["value"] - 230e6) < 1, "arith.sum", res5)
    mc = dict(m, operation="comparison")
    res6 = AR.compute(mc, rec)
    check([x["op"] for x in res6] == ["difference", "order"] and res6[1]["operand_reqs"] == ["R1", "R2"],
          "arith.comparison_orders_and_differences", res6)
    # one entity over two periods: later minus earlier regardless of the matrix order
    mt = matrix_of([("Alpha Corp", "alpha", "M1", "revenue", 2023), ("Alpha Corp", "alpha", "M1", "revenue", 2022)],
                   "difference")
    a, b = mt["requirements"]
    rt = RC.reconcile([claim(a, 100, kid="k1"), claim(b, 90, doc=ALPHA22, years=(2022, 2022), cid="c3", kid="k3")],
                      mt)
    res7 = AR.compute(mt, rt)
    check(res7[0]["status"] == "ok" and res7[0]["value"] > 0 and res7[0]["operands"] == ["k1", "k3"],
          "arith.period_difference_is_later_minus_earlier", res7)
    unv = claim(r2, 100, doc=BETA23, cid="c2", kid="k2")
    unv["unit_status"] = "unverified"
    res8 = AR.compute(m, RC.reconcile([claim(r1, 130, kid="k1"), unv], m))
    check(res8[0]["status"] == "unknown" and "unit_unverified" in res8[0]["reason"],
          "arith.an_unverified_unit_is_never_an_operand", res8)


# ---------------------------------------------------------------------------------------------------------------
# 5. the joint evidence build: renumbering after packing, budget exhaustion, shared units
# ---------------------------------------------------------------------------------------------------------------

class _Idx:
    def __init__(self, cs):
        self.corpus = cs


def _cand_for(cs, doc_id, cid, text=None, unit=0):
    d = cs.doc(doc_id)
    u = d.units[unit]
    t = text if text is not None else d.text[u.span[0]:u.span[1]]
    c = S.Candidate(doc_id, u.section_id, [u.id], t, [u.span[0], u.span[0] + len(t)] if text is None else None,
                    "unit", [u.id])
    c.cid = cid
    return c


def cites(text):
    return [int(x) for x in re.findall(r"\[(\d+)\]", text)]


def test_evidence(cfg, cs):
    o = dict(DEFAULTS)
    # eight requirements, each with its own long source: the snippets cannot all fit
    spec = [("Company %d" % i, "c%d" % i, "M1", "revenue", 2023) for i in range(8)]
    m = matrix_of(spec)
    long = ("Revenue of the company in 2023 was %d million USD. " + FILL * 6)
    cands, cl = {}, []
    for i, r in enumerate(m["requirements"]):
        cid = "c%d" % (i + 1)
        c = _cand_for(cs, ALPHA23, cid, text=long % (100 + i))
        cands[cid] = c
        k = claim(r, 100 + i, cid=cid, unit_id="U%d" % i, kid="k%d" % i)
        k["quote"] = "Revenue of the company in 2023 was %d million USD" % (100 + i)
        cl.append(k)
    rec = RC.reconcile(cl, m)
    small = cfg.copy_with({"evidence": {"max_chars": 2600}})
    pack, info, text = EV.build(m, rec, [], cands, _Idx(cs), small, "en", o)
    check(pack.chars <= 2600 and info["exact"] and not pack.last_item_cut and pack.n_included == len(pack.items),
          "evidence.joint_build_fits_exactly_under_the_budget", info)
    check(info["dropped_sources"], "evidence.some_sources_dropped_for_budget", info["dropped_sources"])
    shown = {i + 1: it for i, it in enumerate(pack.items)}
    ok = True
    for ln in text.split("\n"):
        for n in cites(ln):
            it = shown.get(n)
            if it is None or it.meta.get("cid") is None:
                ok = False
                continue
            want = [c for c in cl if c["cid"] == it.meta["cid"]]
            if not want or want[0]["value"] not in ln:
                ok = False
    check(ok and cites(text), "evidence.every_citation_points_at_the_packed_snippet_of_its_row", text[:900])
    check(max(cites(text)) <= len(pack.items) and 1 not in cites(text),
          "evidence.citations_numbered_after_packing_never_beyond_the_pack", (cites(text), len(pack.items)))
    dropped_rows = [ln for ln in text.split("\n") if "source not shown" in ln]
    check(len(dropped_rows) == len(info["dropped_sources"]), "evidence.primary_rows_keep_their_value_marked_unsourced",
          dropped_rows)
    table = pack.items[0]
    kept_units = {c["unit_id"] for c in cl if c["cid"] not in info["dropped_sources"]}
    check(set(table.unit_ids) == kept_units, "evidence.table_units_only_from_packed_sources",
          (table.unit_ids, sorted(kept_units)))
    # budget exhaustion: a budget that holds the table and almost nothing else
    tiny = cfg.copy_with({"evidence": {"max_chars": 1200}})
    pack2, info2, text2 = EV.build(m, rec, [], cands, _Idx(cs), tiny, "en", o)
    check(pack2.chars <= 1200 and info2["exact"] and not pack2.last_item_cut
          and all(n <= len(pack2.items) for n in cites(text2)), "evidence.budget_exhaustion_stays_exact", info2)
    # table rows sharing one unit: one snippet, one citation for both rows
    m6 = matrix_of([("Alpha Corp", "alpha", "M1", "revenue", 2023), ("Alpha Corp", "alpha", "M1", "revenue", 2022)],
                   "series")
    x, y = m6["requirements"]
    tc = _cand_for(cs, ALPHA23, "c7", text="Table 1: Alpha Corp key figures\nMetric | 2023 | 2022\n"
                                           "Revenue (million USD) | 100 | 92")
    c1 = claim(x, 100, cid="c7", unit_id="tbl::row1", kid="k1")
    c2 = claim(y, 92, cid="c7", unit_id="tbl::row1", years=(2023, 2023), kid="k2")
    c1["quote"] = c2["quote"] = "Revenue (million USD) | 100 | 92"
    rec7 = RC.reconcile([c1, c2], m6)
    pack7, info7, text7 = EV.build(m6, rec7, AR.compute(m6, rec7), {"c7": tc}, _Idx(cs), cfg, "en", o)
    rows7 = [ln for ln in text7.split("\n") if "Alpha" in ln and "| revenue" in ln]
    check(len(rows7) == 2 and all("[2]" in ln for ln in rows7) and len(pack7.items) == 2,
          "evidence.two_rows_of_one_table_unit_share_one_snippet_and_citation", text7)
    check(any("change" in ln and "[2]" in ln for ln in text7.split("\n")),
          "evidence.computed_row_cites_its_operands", text7)


# ---------------------------------------------------------------------------------------------------------------
# 6. routing and allocation (pure over the pool)
# ---------------------------------------------------------------------------------------------------------------

def test_routing_allocation(cs):
    o = dict(DEFAULTS)
    metas = [doc_meta(d) for d in cs.docs]
    m = matrix_of([("Alpha Corp", "alpha", "M1", "revenue", 2023), ("Beta Inc", "beta", "M1", "revenue", 2023)])
    # pooled hits: every filler outranks the companies on the question's words
    pool = []
    for i, d in enumerate([d for d in cs.docs if "filler" in d.id] + [cs.doc(ALPHA23), cs.doc(BETA23)], 1):
        for j in range(3 if "filler" in d.id else 1):
            c = _cand_for(cs, d.id, "p%d_%d" % (i, j))
            c.features = {"src": ["pool"], "pool_rank": {"bm25": i * 3 + j}}
            pool.append(c)
    cards = {m_["doc_id"]: m_["title"] for m_ in metas}
    short, info = RO5.shortlist(metas, cards, pool, m, o)
    check(len(short) <= 8, "route.shortlist_at_most_eight", len(short))
    check(ALPHA23 in short and BETA23 in short and set(info["retained"]) >= {ALPHA23, BETA23},
          "route.explicit_entity_period_matches_retained", info["retained"])
    many = [c for c in pool if c.doc_id.startswith("B.filler")]
    per_doc = RO5.chunk_contribution(many, 2)
    check(max(per_doc.values()) <= 2.0, "route.chunk_contribution_is_capped_at_the_best_two", per_doc)
    esc = RO5.escape_candidates(pool, short, o)
    check(esc and all(c.doc_id not in short for c in esc) and len(esc) <= o["escape_max"],
          "route.escape_candidates_come_from_outside_the_shortlist", [c.doc_id for c in esc])
    nothing, ninfo = RO5.shortlist(metas, {}, [], matrix_of([("", "zz", "M1", "dividend", None)]), o)
    check(nothing == [], "route.no_recency_fallback_when_nothing_matches", ninfo)
    mr = matrix_of([("Alpha Corp", "alpha", "M1", "revenue", 2022)])
    exp = RO5.expansion_docs(mr["requirements"][0], metas, [ALPHA23], info["scores"], o)
    check(exp and exp[0] == ALPHA22, "route.requirement_triggered_expansion_finds_the_period_report", exp)
    # allocation: a facet-like flood on R1 cannot take the batch; every requirement gets a slot first
    reqs = [dict(r) for r in matrix_of([("A", "a", "M1", "revenue", 2023), ("B", "b", "M1", "revenue", 2023),
                                        ("C", "c", "M1", "revenue", 2023)])["requirements"]]
    cands = []
    for i in range(20):
        c = _cand_for(cs, "B.filler%02d_sector-review_2020.en" % (i % 10), "x%d" % i, text="flood %d" % i)
        c.score, c.features = 10.0 - i * 0.1, {"src": ["vector"], "req": ["R1"]}
        cands.append(c)
    for rid, doc in (("R2", BETA23), ("R3", ALPHA23)):
        for k in range(2):
            c = _cand_for(cs, doc, "%s_%d" % (rid, k), text="%s cand %d" % (rid, k), unit=k)
            c.score, c.features = 0.5, {"src": ["facts"] if k == 0 else ["units"], "req": [rid]}
            cands.append(c)
    batch, rest, ainfo = AL.allocate(cands, reqs, 12)
    first3 = [c.features["req"][0] for c in batch[:3]]
    check(len(batch) <= 12 and first3 == ["R1", "R2", "R3"] and ainfo["share"]["R2"] == 2
          and ainfo["share"]["R3"] == 2, "allocate.requirements_first_batch_balanced", (first3, ainfo["share"]))
    check([c.cid for c in batch] == ["c%d" % i for i in range(1, len(batch) + 1)] and rest,
          "allocate.cids_in_batch_order_rest_kept_for_batch_two", [c.cid for c in batch][:4])


# ---------------------------------------------------------------------------------------------------------------
# 7. the whole pipeline
# ---------------------------------------------------------------------------------------------------------------

def test_pipeline(cfg, cs):
    q = "What was the revenue of Alpha Corp and Beta Inc in 2023, and what is the difference?"
    script = Script(plan=plan_obj(["Alpha Corp", "Beta Inc"], [REVENUE], ["2023"], "difference"))
    ap, client, rec = run_q(cfg, cs, script, q)
    t = rec.trace
    check(not rec.error, "pipeline.no_error", rec.error or (t or {}).get("traceback"))
    if rec.error:
        return
    purposes = [c["purpose"] for c in t["calls"]]
    check(rec.llm_calls <= 4 and purposes[0] == "plan" and purposes[-1] == "answer",
          "pipeline.four_call_cap_plan_first_answer_last", purposes)
    st = req_status(rec)
    check(st == {"R1": "verified", "R2": "verified"}, "pipeline.both_companies_verified", st)
    rows = rec.structured["series"]
    check(len(rows) == 2 and {r["entity"] for r in rows} == {"Alpha Corp", "Beta Inc"}
          and all(r["value"].startswith("100") and r["value_num"] == 100e6 for r in rows),
          "pipeline.equal_values_of_two_companies_two_rows",
          [(r["entity"], r["value"]) for r in rows])
    comp = rec.structured["computed"]
    check(comp and comp[0]["status"] == "ok" and comp[0]["value"] == 0.0, "pipeline.difference_computed_zero_from_two_values",
          comp)
    ev = script.evidence[-1]
    check("company | period | metric / definition | value | status | sources" in ev
          and "Alpha Corp | 2023" in ev and "Beta Inc | 2023" in ev, "pipeline.table_has_a_company_column", ev[:600])
    check(rec.evidence_chars <= int(cfg.evidence["max_chars"]) and t["stages"]["evidence"]["exact"],
          "pipeline.evidence_exact_under_the_shared_budget", t["stages"]["evidence"])
    n_items = len(rec.evidence_items)
    check(all(n <= n_items for n in cites(t["stages"]["answer"]["table"])),
          "pipeline.table_citations_inside_the_final_pack", (cites(t["stages"]["answer"]["table"]), n_items))
    check(t["stages"]["route"]["shortlist"] and len(t["stages"]["route"]["shortlist"]) <= 8
          and ALPHA23 in t["stages"]["route"]["shortlist"] and BETA23 in t["stages"]["route"]["shortlist"],
          "pipeline.shortlist_holds_both_companies", t["stages"]["route"]["shortlist"])
    check(len(t["stages"]["retrieve"]["batch1"]) <= 12, "pipeline.first_batch_at_most_twelve",
          len(t["stages"]["retrieve"]["batch1"]))
    check(t["stages"]["retrieve"]["channels"]["hybrid"] == "loaded", "pipeline.shared_hybrid_index_used",
          t["stages"]["retrieve"]["channels"]["hybrid"])
    doc_embeds = [p for p in client.embed_payloads if any(len(x) > 300 for x in p["input"])]
    check(not doc_embeds, "pipeline.no_query_time_document_embedding",
          [len(p["input"]) for p in client.embed_payloads])
    check(t["isolation"] == {"warm": False, "write_back": False, "kb_read_only": True,
                             "query_time_document_embedding": False}, "pipeline.cold_and_isolated_by_default",
          t["isolation"])
    return ap


def test_quarter_and_missing(cfg, cs):
    # annual requirement; the pool holds Alpha's annual 2023 value and its Q1 2023 value: only the annual binds
    q = "What was Alpha Corp revenue in 2023?"
    s1 = Script(plan=plan_obj(["Alpha Corp"], [REVENUE], ["2023"], "lookup", kind="single"))
    _ap, _c, rec = run_q(cfg, cs, s1, q, qid="t5q")
    check(not rec.error, "quarter.no_error", rec.error or rec.trace.get("traceback"))
    if not rec.error:
        vals = [r["value_num"] for r in rec.structured["series"]]
        check(vals == [100e6], "quarter.annual_requirement_takes_the_annual_value_only", vals)
        items = [it for b in rec.trace["stages"]["verify"]["batches"] for it in b["items"]]
        q1 = [it for it in items if it.get("value") == "30"]
        check(q1 and all(it["verdict"].startswith("unbound") for it in q1),
              "quarter.the_quarterly_value_is_refused_for_the_annual_requirement", q1)
    # a requested period with no report: stays missing, unknown in the arithmetic, a `missing` row in the table
    q2 = "What was the difference between Alpha Corp revenue in 2019 and in 2023?"
    s2 = Script(plan=plan_obj(["Alpha Corp"], [REVENUE], ["2019", "2023"], "difference"))
    _ap, _c, rec2 = run_q(cfg, cs, s2, q2, qid="t5m")
    check(not rec2.error, "missing.no_error", rec2.error or rec2.trace.get("traceback"))
    if not rec2.error:
        st = req_status(rec2)
        check(st.get("R1") == "missing" and st.get("R2") == "verified", "missing.absent_period_stays_missing", st)
        comp = rec2.structured["computed"]
        check(comp and comp[0]["status"] == "unknown" and comp[0]["value"] is None,
              "missing.arithmetic_unknown_not_zero", comp)
        ev = s2.evidence[-1]
        check("2019 | revenue | - | missing - not found" in ev, "missing.table_row_says_missing", ev[:700])
        check(rec2.llm_calls <= 4 and rec2.trace["stages"]["gap_fill"]["targets"] == ["R1"],
              "missing.second_batch_targets_only_the_missing_requirement", rec2.trace["stages"]["gap_fill"])


def test_scope_and_writeback(cfg, cs):
    q = "What was the revenue of Alpha Corp and Beta Inc in 2023?"
    scope = [ALPHA23, GAMMA21, "B.filler00_sector-review_2020.en"]
    s = Script(plan=plan_obj(["Alpha Corp", "Beta Inc"], [REVENUE], ["2023"], "comparison"))
    ap, _c, rec = run_q(cfg, cs, s, q, scope=scope, qid="t5s")
    check(not rec.error, "scope.no_error", rec.error or rec.trace.get("traceback"))
    if rec.error:
        return
    docs = {it["doc_id"] for it in rec.evidence_items} | {u.split("::")[0] for u in rec.evidence_units}
    docs |= {it["doc_id"] for it in rec.retrieved_units}
    tr = rec.trace
    routed = set(tr["stages"]["route"]["shortlist"]) | set(tr["stages"]["retrieve"]["expanded"])
    check(docs <= set(scope) and routed <= set(scope), "scope.evidence_and_routing_stay_inside_the_scope",
          (sorted(docs - set(scope)), sorted(routed - set(scope))))
    st = req_status(rec)
    check(st.get("R2") == "missing", "scope.an_out_of_scope_company_is_missing_not_borrowed", st)
    con = ap.query_kb()
    qo = list(con.execute("PRAGMA query_only").fetchone().values())[0]
    n_vf = list(ap.kb().execute("SELECT COUNT(*) AS n FROM verified_facts").fetchone().values())[0]
    check(qo == 1 and n_vf == 0 and tr["stages"]["writeback"]["written"] == 0
          and tr["stages"]["writeback"]["skipped"].get("disabled", 0) >= 1,
          "writeback.disabled_by_default_and_the_kb_connection_is_read_only", (qo, n_vf, tr["stages"]["writeback"]))
    try:
        con.execute("CREATE TABLE t5_probe (x)")
        check(False, "writeback.read_only_connection_refuses_writes")
    except Exception as e:  # noqa: BLE001
        check("readonly" in str(e).lower() or "query_only" in str(e).lower() or "read-only" in str(e).lower(),
              "writeback.read_only_connection_refuses_writes", str(e))
    ap2, _c2, rec2 = run_q(cfg, cs, Script(plan=plan_obj(["Alpha Corp"], [REVENUE], ["2023"], "lookup",
                                                         kind="single")),
                           "What was Alpha Corp revenue in 2023?", flags={"write_back": True}, qid="t5w")
    check(not rec2.error and rec2.trace["stages"]["writeback"]["written"] >= 1,
          "writeback.opt_in_flag_writes_guard_passed_rows", rec2.error or rec2.trace["stages"]["writeback"])


def test_malformed_and_cap(cfg, cs):
    q = "What was the revenue of Alpha Corp and Beta Inc in 2023?"
    # a reply that is not JSON, then a repair that is not JSON either: the program plan, the repair consumed the
    # optional second verify slot, still <= 4 calls with the answer last
    s = Script(plan_text="I cannot answer in JSON, sorry.", repair_text="still not json")
    _ap, _c, rec = run_q(cfg, cs, s, q, qid="t5bad")
    check(not rec.error, "malformed.no_error", rec.error or rec.trace.get("traceback"))
    if rec.error:
        return
    t = rec.trace
    purposes = [c["purpose"] for c in t["calls"]]
    check(purposes[:2] == ["plan", "plan_repair"] and purposes[-1] == "answer" and len(purposes) <= 4,
          "cap.repair_counts_against_the_four_calls", purposes)
    check("verify_gap" not in purposes, "cap.repair_consumes_the_optional_second_verify", purposes)
    rq = t["stages"]["requirements"]
    check(sorted(r["entity_key"] for r in rq["requirements"]) == ["alpha", "beta"]
          and all(r["period_key"] == "2023" for r in rq["requirements"]) and "program_plan" in rq["fixes"],
          "malformed.program_plan_keeps_question_entities_and_periods", rq["fixes"])
    # a plan whose JSON is fine but every new field is junk
    junk = {"entity": "Alpha Corp", "metric": "revenue", "kind": "weird", "terms_fa": [], "terms_en": ["revenue"],
            "entity_terms": "Alpha", "unit": "??", "period_from": "x", "period_to": None, "entities": "Alpha Corp",
            "metrics": [{"name": 3}], "periods": {"y": 1}, "operation": "boom"}
    _ap, _c, rec2 = run_q(cfg, cs, Script(plan=junk), "What was Alpha Corp revenue in 2023?", qid="t5junk")
    check(not rec2.error and [r["period_key"] for r in rec2.trace["stages"]["requirements"]["requirements"]]
          == ["2023"] and rec2.llm_calls <= 4, "malformed.junk_fields_repaired_field_by_field",
          rec2.error or rec2.trace["stages"]["requirements"]["fixes"])
    # the cap itself: a config with 3 calls never spends a 4th
    cfg3 = cfg.copy_with({"caps": {"agentic5_max_calls": 3}})
    s3 = Script(plan=plan_obj(["Alpha Corp"], [REVENUE], ["2019", "2023"], "difference"))
    _ap, _c, rec3 = run_q(cfg3, cs, s3, "Alpha Corp revenue in 2019 and 2023?", qid="t5cap")
    check(not rec3.error and rec3.llm_calls <= 3 and rec3.trace["calls"][-1]["purpose"] == "answer",
          "cap.a_smaller_cap_is_honoured_with_the_answer_kept",
          rec3.error or [c["purpose"] for c in rec3.trace["calls"]])


def test_deadline(cfg, cs):
    q = "What was the revenue of Alpha Corp and Beta Inc in 2023?"
    s = Script(plan=plan_obj(["Alpha Corp", "Beta Inc"], [REVENUE], ["2023"], "comparison"))
    ap, client, index = approach(cfg, cs, s, wall_ms=9000.0)          # every call "takes" 9 s of recorded wall time
    ap.max_seconds = 12.0
    before = client.timeout
    rec = ap.answer(question(q, qid="t5dl"), Setting.parse("en-en"))
    check(not rec.error, "deadline.no_error", rec.error or rec.trace.get("traceback"))
    if rec.error:
        return
    t = rec.trace
    purposes = [c["purpose"] for c in t["calls"]]
    check(purposes == ["plan", "answer"] and t["stats"]["deadline_skips"] + int(
        t["stages"]["verify"]["batches"] == []) >= 1, "deadline.verify_skipped_when_time_is_short_answer_kept",
        (purposes, t["deadlines"]))
    ans = [d for d in t["deadlines"] if d["purpose"] == "answer"]
    check(ans and ans[0]["applied"] and ans[0]["read_timeout_s"] >= DEFAULTS["answer_floor_s"],
          "deadline.read_timeout_bounded_at_each_request", t["deadlines"])
    check(client.timeout == before, "deadline.client_timeout_restored_after_the_call", (client.timeout, before))


def test_kb_fallback(cfg, cs):
    q = "What was the revenue of Alpha Corp and Beta Inc in 2023?"
    s = Script(plan=plan_obj(["Alpha Corp", "Beta Inc"], [REVENUE], ["2023"], "comparison"))
    _ap, client, rec = run_q(cfg, cs, s, q, flags={"no_hybrid": True}, qid="t5kb")
    check(not rec.error, "fallback.no_error", rec.error or rec.trace.get("traceback"))
    if rec.error:
        return
    ch = rec.trace["stages"]["retrieve"]["channels"]
    check(ch["hybrid"] == "disabled" and ch["dense"].get("skipped") == "no_prebuilt_vectors"
          and ch["dense"].get("query_time_document_embedding") is False and not client.embed_payloads,
          "fallback.kb_path_with_dense_skipped_and_recorded_no_embedding", (ch["hybrid"], ch["dense"],
                                                                             len(client.embed_payloads)))
    check(req_status(rec) == {"R1": "verified", "R2": "verified"}, "fallback.kb_path_verifies_both_companies",
          req_status(rec))


def test_prebuilt_dense(cfg, cs):
    """Vectors embedded at INDEX time (the prebuild's job) are used at query time; the query embeds only the
    question."""
    from ...retrieval.hybrid import HybridIndex
    q = "What was the revenue of Alpha Corp and Beta Inc in 2023?"
    s = Script(plan=plan_obj(["Alpha Corp", "Beta Inc"], [REVENUE], ["2023"], "comparison"))
    ap, client, _index = approach(cfg, cs, s)
    with client.ledger.scope(phase="index", approach="AGENTIC5"):
        HybridIndex.build(cs, cfg, client=client, embed=True)      # index time: the prebuild
    ap.__dict__.pop("_hybrid_cache", None)
    n0 = len(client.embed_payloads)
    rec = ap.answer(question(q, qid="t5dense"), Setting.parse("en-en"))
    check(not rec.error, "dense.no_error", rec.error or rec.trace.get("traceback"))
    if rec.error:
        return
    during = [x for p in client.embed_payloads[n0:] for x in p["input"]]
    check(during and all(len(x) < 300 for x in during), "dense.query_time_embeds_only_the_question",
          [len(x) for x in during])
    calls = rec.trace["stages"]["retrieve"]["channels"]["calls"]
    batch = rec.trace["stages"]["retrieve"]["batch1"]
    dense_ranked = [c for c in batch if ((c["features"] or {}).get("ranks") or {}).get("dense")]
    check(any("dense" in c["channels"] for c in calls) and dense_ranked, "dense.prebuilt_vectors_rank_candidates",
          (calls[:2], len(dense_ranked)))
    check(req_status(rec) == {"R1": "verified", "R2": "verified"}, "dense.both_companies_verified", req_status(rec))


# ---------------------------------------------------------------------------------------------------------------
# 9. the ragbench/variants.py options: doc_card and section_expand
# ---------------------------------------------------------------------------------------------------------------

CARD_MARK = "Context - the document(s) being searched"


class CardSpy(Script):
    """A Script that also records, per call kind, whether the document card reached the request."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.seen = []

    def __call__(self, payload):
        system = payload["messages"][0]["content"]
        kind = "plan_repair" if system.startswith("You plan a search") and len(payload["messages"]) > 2 else \
            "plan" if system.startswith("You plan a search") else \
            "verify" if system.startswith("You check candidate snippets") else "answer"
        self.seen.append((kind, any(CARD_MARK in (m.get("content") or "") for m in payload["messages"])))
        return super().__call__(payload)


def test_variant_flags():
    from ...run import parse_flags as run_parse_flags
    from .approach import parse_flags
    f = run_parse_flags(["doc_card,section_expand"], "AGENTIC5")
    check(f == {"doc_card": True, "section_expand": True}, "variants.run_accepts_both_options_for_agentic5", f)
    check(parse_flags("doc_card,section_expand") == f, "variants.agentic5_parse_flags_accepts_both", f)


def test_doc_card(cfg, cs):
    q = "What was the revenue of Alpha Corp and Beta Inc in 2023, and what is the difference?"
    spy = CardSpy(plan=plan_obj(["Alpha Corp", "Beta Inc"], [REVENUE], ["2023"], "difference"))
    ap, client, index = approach(cfg, cs, spy, flags={"doc_card": True})
    ap.variants_version = "variants-v2"
    rec = ap.answer(question(q), Setting.parse("en-en"))
    check(not rec.error, "doc_card.no_error", rec.error or (rec.trace or {}).get("traceback"))
    if rec.error:
        return
    check(rec.approach == "AGENTIC5+doc_card" and ap.index_variant == "AGENTIC5",
          "doc_card.names_the_variant_not_the_index", (rec.approach, ap.index_variant))
    plans = [has for k, has in spy.seen if k in ("plan", "plan_repair")]
    others = [(k, has) for k, has in spy.seen if k not in ("plan", "plan_repair")]
    check(plans and all(plans), "doc_card.card_reaches_the_plan_call", spy.seen)
    check(others and not any(has for _k, has in others), "doc_card.verify_and_answer_calls_never_see_it", spy.seen)
    vt = rec.trace.get("variants") or {}
    sel = vt.get("card_selection") or {}
    check(sel.get("version") == "variants-v2" and sel.get("shown") and sel.get("routed_ids") == [],
          "doc_card.query_ranked_shortlist_logged_nothing_routed_before_the_plan", sel)
    cvr = vt.get("card_vs_route") or {}
    check(cvr.get("route_shortlist") == rec.trace["stages"]["route"]["shortlist"]
          and set(cvr.get("shortlist_shown_in_card") or []) <= set(cvr["route_shortlist"]),
          "doc_card.card_compared_with_the_routed_shortlist_after_routing", cvr)
    check(req_status(rec) == {"R1": "verified", "R2": "verified"}, "doc_card.pipeline_still_verifies", req_status(rec))
    # a malformed plan: the repair call is a planning call too
    spy2 = CardSpy(plan_text="not json at all",
                   repair_text=json.dumps(plan_obj(["Alpha Corp"], [REVENUE], ["2023"], "lookup", kind="single")))
    ap2, _c2, _i2 = approach(cfg, cs, spy2, flags={"doc_card": True})
    ap2.variants_version = "variants-v2"
    rec2 = ap2.answer(question("What was Alpha Corp revenue in 2023?", qid="t5r"), Setting.parse("en-en"))
    kinds = [k for k, _h in spy2.seen]
    check(not rec2.error and "plan_repair" in kinds and all(h for k, h in spy2.seen if k in ("plan", "plan_repair"))
          and not any(h for k, h in spy2.seen if k not in ("plan", "plan_repair")),
          "doc_card.card_reaches_plan_repair_and_nothing_else", (rec2.error, spy2.seen))
    # off: no card anywhere, the request untouched
    spy3 = CardSpy(plan=plan_obj(["Alpha Corp"], [REVENUE], ["2023"], "lookup", kind="single"))
    _ap3, _c3, rec3 = run_q(cfg, cs, spy3, "What was Alpha Corp revenue in 2023?", qid="t5o")
    check(not rec3.error and not any(h for _k, h in spy3.seen) and "variants" not in rec3.trace,
          "doc_card.off_leaves_every_request_untouched", spy3.seen)


def _expand_case(cfg, cs, q, cands, spec, *, budget=None):
    o = dict(DEFAULTS)
    m = matrix_of(spec)
    cl = []
    for (cid, c), r in zip(cands.items(), m["requirements"]):
        v = re.search(r"was (\d+) million", c.text).group(1)
        k = claim(r, v, cid=cid, doc=c.doc_id, unit_id=c.unit_ids[0], kid="k_" + cid)
        k["quote"] = re.search(r"[^.]*was \d+ million USD", c.text).group(0).strip()
        cl.append(k)
    rec = RC.reconcile(cl, m)
    c2 = cfg.copy_with({"evidence": {"max_chars": budget}}) if budget else cfg
    st = {}
    pack, info, text = EV.build(m, rec, [], cands, _Idx(cs), c2, "en", o, state=st)
    out = EV.expand_after_fit(st, cs, q, c2)
    return pack, info, text, out, cl


def test_section_expand(cfg, cs):
    q = "List the financial highlights of Alpha Corp and Beta Inc for 2023"
    ca = _cand_for(cs, ALPHA23, "c1", unit=0)          # verbatim: Alpha's revenue paragraph (Financial highlights)
    cb = _cand_for(cs, BETA23, "c2", unit=0)           # Beta's revenue paragraph (another document)
    spec = [("Alpha Corp", "alpha", "M1", "revenue", 2023), ("Beta Inc", "beta", "M1", "revenue", 2023)]
    pack, info, text, (pack2, text2, dec), cl = _expand_case(cfg, cs, q, {"c1": ca, "c2": cb}, spec)
    check(info["exact"] and cites(text) == [2, 3], "expand.unexpanded_table_cites_two_snippets", text)
    check(pack2 is not None and dec["applied"] and dec["section"] == "Financial highlights",
          "expand.list_question_expands_the_whole_section_after_the_fit", dec)
    if pack2 is None:
        return
    check(pack2.items[0].meta.get("table") and pack2.items[1].kind == "section",
          "expand.table_stays_first_section_follows", [it.kind for it in pack2.items])
    check(dec["citations_before"] == {"c1": 2, "c2": 3} and dec["citations_after"] == {"c1": 2, "c2": 3}
          and dec["renumbered"] is False and dec["dropped_as_inside"] == 1,
          "expand.snippet_inside_the_section_cites_the_section", dec)
    ok = True
    for c in cl:
        n = dec["citations_after"][c["cid"]]
        ok = ok and c["quote"] in pack2.items[n - 1].text and any(
            ("[%d]" % n) in ln and c["value"] in ln for ln in text2.split("\n"))
    check(ok and pack2.items[0].text == text2, "expand.every_row_cites_an_item_that_shows_its_quote", text2)
    check(not pack2.last_item_cut and pack2.n_included == len(pack2.items) and pack2.chars <= int(cfg.evidence["max_chars"]),
          "expand.expanded_pack_exact_under_the_budget", (pack2.n_included, pack2.chars))
    check(set(pack.unit_ids) <= set(pack2.unit_ids) and {it.doc_id for it in pack.items} <= {it.doc_id for it in pack2.items},
          "expand.no_unit_or_document_lost", (pack.unit_ids, pack2.unit_ids))
    # the Beta snippet comes first in the table order -> after expansion its number moves: re-numbered AFTER expansion
    spec_b = [("Beta Inc", "beta", "M1", "revenue", 2023), ("Alpha Corp", "alpha", "M1", "revenue", 2023)]
    ca2, cb2 = _cand_for(cs, ALPHA23, "c1", unit=0), _cand_for(cs, BETA23, "c2", unit=0)
    _p, _i, text_b, (pack_b, text_b2, dec_b), _cl = _expand_case(cfg, cs, q, {"c2": cb2, "c1": ca2}, spec_b)
    check(pack_b is not None and dec_b["citations_before"] == {"c2": 2, "c1": 3}
          and dec_b["citations_after"] == {"c2": 3, "c1": 2} and dec_b["renumbered"]
          and pack_b.items[2].meta.get("cid") == "c2" and "Beta Inc |" in text_b2
          and any("Beta Inc" in ln and "[3]" in ln for ln in text_b2.split("\n"))
          and any("Alpha Corp" in ln and "[2]" in ln for ln in text_b2.split("\n")),
          "expand.citations_renumbered_after_expansion_never_before", (dec_b, text_b2))
    # skipped: a replaced snippet that is not shown verbatim inside the section
    cx = _cand_for(cs, ALPHA23, "c1", text="Alpha Corp revenue in 2023 was 100 million USD (restated excerpt).")
    _p, _i, _t, (px, tx, dx), _cl = _expand_case(cfg, cs, q, {"c1": cx, "c2": _cand_for(cs, BETA23, "c2")}, spec)
    check(px is None and tx is None and dx["skipped"] and "verbatim" in dx["reason"],
          "expand.skipped_when_a_replaced_item_is_not_verbatim_inside", dx)
    # skipped: the expanded pack would not hold everything whole
    _p, info_t, _t, (pt, _tt, dt), _cl = _expand_case(cfg, cs, q, {"c1": _cand_for(cs, ALPHA23, "c1"),
                                                               "c2": _cand_for(cs, BETA23, "c2")}, spec, budget=1100)
    check(info_t["exact"] and pt is None and dt["skipped"] and dt["reason"],
          "expand.skipped_when_the_expansion_does_not_fit", dt)
    # skipped: no list wording
    _p, _i, _t, (pn, _tn, dn), _cl = _expand_case(cfg, cs, "What was the revenue of Alpha Corp in 2023?",
                                                  {"c1": _cand_for(cs, ALPHA23, "c1"),
                                                   "c2": _cand_for(cs, BETA23, "c2")}, spec)
    check(pn is None and dn["skipped"] and dn["list_question"] is False, "expand.skipped_without_list_wording", dn)


def test_section_expand_pipeline(cfg, cs):
    q = "List the revenue of Alpha Corp and Beta Inc in 2023"
    spy = CardSpy(plan=plan_obj(["Alpha Corp", "Beta Inc"], [REVENUE], ["2023"], "comparison"))
    ap, _client, index = approach(cfg, cs, spy, flags={"section_expand": True})
    ap.variants_version = "variants-v2"
    rec = ap.answer(question(q, qid="t5x"), Setting.parse("en-en"))
    check(not rec.error, "expand_pipe.no_error", rec.error or (rec.trace or {}).get("traceback"))
    if rec.error:
        return
    t = rec.trace
    sx = t["stages"]["evidence"].get("section_expand") or {}
    check(sx and (t.get("variants") or {}).get("section_expand") == sx and ("applied" in sx)
          and (sx["applied"] or sx.get("reason")), "expand_pipe.decision_recorded_in_the_trace", sx)
    check(not any(h for _k, h in spy.seen), "expand_pipe.no_card_without_doc_card", spy.seen)
    table = t["stages"]["answer"]["table"]
    ev = spy.evidence[-1]
    check(table and table in ev, "expand_pipe.the_answer_sees_the_table_that_was_rendered_last", table)
    n_items = len(rec.evidence_items)
    check(cites(table) and all(2 <= n <= n_items for n in cites(table)),
          "expand_pipe.table_citations_inside_the_final_pack", (cites(table), n_items))
    check(req_status(rec) == {"R1": "verified", "R2": "verified"}, "expand_pipe.requirements_still_verified",
          req_status(rec))
    if sx.get("applied"):
        check(t["stages"]["evidence"]["citations"] == sx["citations_after"],
              "expand_pipe.trace_citations_are_the_renumbered_ones", (t["stages"]["evidence"]["citations"], sx))


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass
    base = load_config()
    tmp = base.project_root / "work" / "upgrades" / "agentic5" / "test_tmp"
    if tmp.exists():
        shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True, exist_ok=True)
    cfg = base.copy_with({"paths": {"index_root": str(tmp / "indexes")}, "kb": {"path": str(tmp / "kb.sqlite")}})
    cs = make_pool()
    print("requirements"); test_requirements(cs)
    print("reconcile");    test_reconcile()
    print("binding");      test_binding()
    print("arith");        test_arith()
    print("evidence");     test_evidence(cfg, cs)
    print("routing");      test_routing_allocation(cs)
    print("pipeline");     test_pipeline(cfg, cs)
    print("quarter");      test_quarter_and_missing(cfg, cs)
    print("scope");        test_scope_and_writeback(cfg, cs)
    print("malformed");    test_malformed_and_cap(cfg, cs)
    print("deadline");     test_deadline(cfg, cs)
    print("fallback");     test_kb_fallback(cfg, cs)
    print("dense");        test_prebuilt_dense(cfg, cs)
    print("variants");     test_variant_flags()
    print("doc_card");     test_doc_card(cfg, cs)
    print("expand");       test_section_expand(cfg, cs)
    print("expand_pipe");  test_section_expand_pipeline(cfg, cs)
    print("\n%d checks, %d failure(s)" % (N[0], len(FAILS)))
    for f in FAILS:
        print("  FAILED:", f)
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
