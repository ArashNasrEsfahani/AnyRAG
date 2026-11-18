"""Offline tests for AGENTIC4 (no model: the knowledge base is read, no chat or embed call is made).

    .venv/Scripts/python.exe -m ragbench.approaches.agentic4.test_agentic4

  flags     v4's own flag list, variant names, and that v3's options / caps are not read
  wiring    registered, declared in config final.variants, own caps, own options variable refused by --final
  graph     on the LIVE knowledge base: a one-report plan gets no graph candidate (v4 is v3 there), a several-report
            plan does; candidates carry the graph features; the merge reserves one slot per report first; with no
            graph candidate the merge IS v3's merge; the qualifier annotation fills only 'unspecified' rows
"""
import os
import sys

from ...approaches import AGENTIC4_FLAGS, APPROACHES, get_approach
from ...config import load_config
from ...kb import graph as G
from ...kb import ingest as I
from ...kb import query as Q
from ...kb.schema import connect
from ..agentic3 import retrieve as RT
from . import graphret as GR
from .approach import GraphAgent, _Session4, parse_flags
from .defaults import DEFAULTS, FLAGS

FAILS = []


def check(cond, label):
    print(("  ok   " if cond else "  FAIL ") + label)
    if not cond:
        FAILS.append(label)


def flag_tests():
    print("\n[flags]")
    check(parse_flags("no_graph,no_family") == {"no_graph": True, "no_family": True}, "parse a comma list")
    check(parse_flags({"candidates": "16"}) == {"candidates": 16}, "candidates takes an integer")
    try:
        parse_flags("verify_gate")
        check(False, "a v3-only flag is refused")
    except ValueError:
        check(True, "a v3-only flag is refused")
    check(tuple(FLAGS) == tuple(AGENTIC4_FLAGS), "defaults.FLAGS and the registry's AGENTIC4_FLAGS agree")
    os.environ["AGENTIC3_OPTIONS"] = '{"max_candidates": 99}'
    try:
        ap = GraphAgent()
    finally:
        del os.environ["AGENTIC3_OPTIONS"]
    check(ap.opts["max_candidates"] != 99, "AGENTIC3_OPTIONS does not reach v4 (it has its own AGENTIC4_OPTIONS)")
    check(ap.variant == "AGENTIC4" and GraphAgent(flags="no_graph").variant == "AGENTIC4+no_graph", "variant names")
    check(GraphAgent(flags="no_graph").opts["vec_min_slots"] == 8, "no_graph keeps v3's merge budget (8 vector slots)")


def wiring_tests(cfg):
    print("\n[wiring]")
    check(APPROACHES.get("AGENTIC4") == "ragbench.approaches.agentic4:GraphAgent", "registered as AGENTIC4")
    check("AGENTIC4" in (cfg.get("final") or {}).get("variants", []), "declared in config final.variants")
    check(int(cfg.caps.get("agentic4_max_calls")) == 4 and int(cfg.caps.get("agentic4_max_seconds")) == 60,
          "own caps: 4 calls, 60 s")
    from ...run import TUNING_ENV
    check("AGENTIC4_OPTIONS" in TUNING_ENV, "a --final run refuses AGENTIC4_OPTIONS")
    sig = GraphAgent().config_signature()
    check({"graph", "labels", "graphret"} <= set(sig), "the index fingerprint covers the graph, label and graphret versions")


def graph_tests(cfg):
    print("\n[graph, live knowledge base]")
    con = connect(I.kb_path(cfg), read_only=True)
    if G.summary(con)["facts"] == 0:
        check(False, "the live knowledge base has a graph (run `python -m ragbench.kb.graph build`)")
        return
    o = dict(DEFAULTS)
    reader = Q.DocReader(con)
    snapp = [r["doc_id"] for r in con.execute(
        "SELECT doc_id FROM documents WHERE company = 'snapp' AND lang = 'fa' ORDER BY year_from")]
    plan = {"kind": "timeline", "unit": "count", "metric": "تعداد رانندگان فعال", "period": [1398, 1404],
            "terms": {"fa": ["تعداد رانندگان", "رانندگان", "راننده"], "en": ["drivers"]}, "entity_terms": [],
            "companies": ["snapp"]}
    one, info1 = GR.graph_candidates(con, reader, dict(plan, kind="single"), snapp[:1], d_lang="fa", opts=o)
    check(one == [] and info1.get("skipped") == "one_report", "a one-report plan gets no graph candidate (v4 = v3)")
    many, info = GR.graph_candidates(con, reader, plan, snapp, d_lang="fa", opts=o)
    check(len(many) > 0 and all(GR.is_graph(c) for c in many), "a several-report plan gets graph candidates (%d)" % len(many))
    check(all({"node", "family", "quals"} <= set(c.features["graph"]) for c in many), "candidates carry the graph features")
    check("active" in info["plan_quals"], "the plan's qualifier is read: %s" % info["plan_quals"])
    top_active = [c for c in many[:10] if "active" in (c.features["graph"]["quals"] or "")]
    check(top_active, "an ACTIVE-driver fact ranks in the top ten for a question about active drivers")
    RT.tag_facet(many, None)
    sel, m = GR.merge_graph(many, snapp, [], o, 24, slots=4)
    first = [c.doc_id for c in sel[:m["graph_slots"]]]
    check(m["graph_slots"] == 4 and len(set(first)) == len(first),
          "the reserved graph slots go one per report first: %s" % [d.split("_")[-1] for d in first])
    check(GR.graph_slots(o, snapp, "text") == int(o["graph_text_slots"]), "a text plan reserves graph_text_slots")
    plain = [c for c in many if False]
    base, _i = RT.merge(list(plain), snapp, [], o, 24)
    same, _j = GR.merge_graph(list(plain), snapp, [], o, 24)
    check([c.cid for c in same] == [c.cid for c in base], "with no graph candidate the merge IS v3's merge")

    class _B:                                               # a batch of two candidates, one with an active node
        def __init__(self, cid, quals):
            self.cid, self.features = cid, {"graph": {"quals": quals}}
    ap = GraphAgent()
    sess = _Session4.__new__(_Session4)
    sess.a, sess.o, sess.stats = ap, ap.opts, {"graph_quals_applied": 0}
    rows = [{"cid": "c1", "def_class": "unspecified"}, {"cid": "c2", "def_class": "registered"}]
    from ..agentic3.approach import _Session3
    orig = _Session3.annotate_rows
    _Session3.annotate_rows = lambda self, out, rows, batch: None
    try:
        sess.annotate_rows({"items": []}, rows, [_B("c1", "active,growth"), _B("c2", "active")])
    finally:
        _Session3.annotate_rows = orig
    check(rows[0]["def_class"] == "active" and rows[0].get("def_source") == "graph" and rows[1]["def_class"] == "registered",
          "the graph qualifier fills only an 'unspecified' row, and only with a DEF_CLASSES name: %s" % rows)


def _cand(doc, units, score, src, quals=None):
    from ..agentic.search import Candidate
    c = Candidate(doc, doc + "::s", [doc + "::" + u for u in units], "text " + "-".join(units), None, "unit", [])
    c.score, c.features = score, {"src": list(src), "facet": []}
    if "graph" in src:
        c.features["graph"] = {"node": "n", "family": "f", "quals": quals or "", "walk": False}
    return c


def fix_tests(cfg):
    """The v2 rules, each answering a failure of the dev GPU run of v1 (work/analysis/V4_DEV.md)."""
    print("\n[v2 fixes]")
    o = dict(DEFAULTS)
    docs = ["D.1398", "D.1399", "D.1400"]
    # v3's pool: 30 windows, 10 per report, scores 3.0-3.9. The graph: two windows v3 also has (one scored HIGH, as
    # v1's concept term made them), and new windows in every report, all scored far above v3's.
    base = [_cand(d, ["u%d" % i], 3.0 + i / 10.0, ["facts"]) for d in docs for i in range(10)]
    graph = [_cand("D.1398", ["u9"], 9.0, ["graph"], "active"), _cand("D.1399", ["u8"], 9.0, ["graph"])]
    graph += [_cand(d, ["g%d" % i], 8.0 - i / 10.0, ["graph"]) for d in docs for i in range(5)]
    v3, _i = RT.merge([_cand(c.doc_id, [u.split("::")[1] for u in c.unit_ids], c.score, ["facts"]) for c in base],
                      docs, [], o, 24 - 4)
    sel, m = GR.merge_graph(base + graph, docs, [], o, 24, slots=4)
    new = [c for c in sel if c.features.get("src") == ["graph"]]
    check(len(new) == 4 and m["graph_slots"] == 4,
          "the graph adds at most its reserved slots of NEW windows, however high it scores (%d)" % len(new))
    check({c.doc_id for c in new[:3]} == set(docs), "one new window per report first")
    rest = [c for c in sel if c not in new]
    check([(c.doc_id, c.unit_ids) for c in rest] == [(c.doc_id, c.unit_ids) for c in v3],
          "the rest IS v3's merge at the cap minus the graph's slots: same windows, same order")
    conf = [c for c in rest if (c.features.get("graph") or {}).get("confirmed")]
    check(any(c.doc_id == "D.1398" and c.unit_ids == ["D.1398::u9"] and c.features["graph"]["quals"] == "active"
              for c in conf) and all(abs(c.score - 3.9) < 1e-9 or c.score < 3.9 for c in conf),
          "a v3 window the graph also found keeps v3's score and gets the graph's qualifier (%d confirmed)"
          % len(conf))
    legacy, lm = GR.merge_graph([_cand(c.doc_id, [u.split("::")[1] for u in c.unit_ids], c.score, c.features["src"],
                                       (c.features.get("graph") or {}).get("quals")) for c in base + graph],
                                docs, [], dict(o, graph_merge_mode="legacy"), 24, slots=4)
    check(sum(1 for c in legacy if "graph" in c.features.get("src")) > 4,
          "legacy mode still lets the graph flood (the v1 failure it is kept to measure): %d graph windows"
          % sum(1 for c in legacy if "graph" in c.features.get("src")))

    years = {"D.1398": (1398, 1398, "jalali"), "D.1399": (1399, 1399, "jalali"), "D.1400": (1400, 1400, "jalali")}
    row = lambda d, y, y2=None, cal="jalali": {"doc_id": d, "year": y, "year_to": y2, "calendar": cal}  # noqa: E731
    check(GR.restated(row("D.1400", 1399), years, docs), "the 1400 report's value for 1399 is a restatement")
    check(not GR.restated(row("D.1400", 1400), years, docs), "the 1400 report's own year is not")
    check(not GR.restated(row("D.1400", 1399), years, ["D.1398", "D.1400"]),
          "a year no other target report covers is kept (the later report is the only source)")
    check(not GR.restated(row("D.1400", 1397, 1400), years, docs), "a multi-year (cumulative) value is not a restatement")
    check(not GR.restated(row("D.1400", 2020, cal="gregorian"), years, docs), "another calendar is left alone")

    con = connect(I.kb_path(cfg), read_only=True)
    divar = ["C.divar_annual-report_%d.fa" % y for y in (1398, 1399, 1400)]
    plan = {"kind": "timeline", "unit": "percent", "metric": "درصد آگهی‌های رد شده", "period": [1398, 1400],
            "terms": {"fa": ["درصد آگهی رد شده", "سهم آگهی پذیرفته نشدن", "آگهی", "پذیرفته"], "en": []},
            "entity_terms": [], "companies": ["divar"]}
    cands, info = GR.graph_candidates(con, Q.DocReader(con), plan, divar, d_lang="fa", opts=o)
    kinds = {c.features["fact"]["unit_kind"] for c in cands}
    check(info["dropped"]["kind_conflict"] > 0 and not any(c.features["db"].get("kind") == "conflict" for c in cands),
          "C-TL-divar-02's percent plan: no graph window with a conflicting unit kind (%d dropped; kinds kept %s)"
          % (info["dropped"]["kind_conflict"], sorted(k or "-" for k in kinds)))
    yrs = GR.doc_years(con, divar)
    check(not any(GR.restated(dict(doc_id=c.doc_id, year=c.features["fact"]["year"], year_to=None,
                                   calendar="jalali"), yrs, divar) for c in cands),
          "and none restating another target report's year (%d dropped)" % info["dropped"]["restated"])


def gap_tests():
    """graphret-v4: the graph works in v3's gap round, through the `gap_batch` hook."""
    print("\n[gap round]")
    ap = GraphAgent()
    s = _Session4.__new__(_Session4)
    s.a, s.o, s.stats = ap, ap.opts, {}
    n = int(ap.opts["verify_batch_max"])
    v3gap = [_cand("D.1398", ["x%d" % i], 1.0, ["gap"]) for i in range(3)]
    skipped = [_cand("D.1399", ["s%d" % i], 1.0, ["facts"]) for i in range(20)]
    graph = [_cand("D.1400", ["h%d" % i], 5.0, ["graph"]) for i in range(2)]
    s._gap_ctx = ({"periods_missing": [1400]}, {}, [])
    s.graph_gap = lambda rec, plan, cands, taken=(): (list(graph), {"candidates": len(graph)})
    gap = {}
    extra, new = s.gap_batch(v3gap, skipped, gap)
    check(extra[:3] == v3gap and extra[3:] == graph and new == extra and len(extra) <= n,
          "v3's gap candidates keep their seats; the graph's follow them in the same held-back call")
    extra, new = s.gap_batch([], skipped, {})
    check(extra[:2] == graph and extra[2:] == skipped[: n - 2] and new == graph,
          "no v3 gap candidate: the graph's first, then the skipped first-round candidates up to the batch size")
    s.graph_gap = lambda rec, plan, cands, taken=(): ([], {"candidates": 0})
    check(s.gap_batch(v3gap, skipped, {}) == (v3gap, v3gap) and s.gap_batch([], skipped, {}) == (skipped[:n], []),
          "no graph candidate: the batch IS v3's (gap candidates, else the skipped head)")
    s.a = GraphAgent(flags="no_graph")
    s.graph_gap = lambda *a, **k: (list(graph), {})
    check(s.gap_batch([], skipped, {}) == (skipped[:n], []), "no_graph: v3's batch exactly")


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    cfg = load_config()
    flag_tests()
    wiring_tests(cfg)
    graph_tests(cfg)
    fix_tests(cfg)
    gap_tests()
    print("\n%d failure(s)" % len(FAILS))
    for f in FAILS:
        print("  -", f)
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
