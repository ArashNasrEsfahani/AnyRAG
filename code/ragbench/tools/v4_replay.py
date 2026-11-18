"""AGENTIC4 against AGENTIC3 on the dev files with NO model call: does the graph put more gold evidence in the pack?

Each dev record of v3 carries its plan (the one model call retrieval depends on) and its vector hits. Replaying
them rebuilds v3's candidate pool exactly with no chat and no embedding call (tools/v3_recall_trace.py, 59/59
fidelity when it was written). This module builds that pool ONCE per record and then:

  v3   merges it with AGENTIC3's current defaults (8 vector slots of 12)
  v4   adds the graph pool (approaches/agentic4/graphret.py) and merges with the graph slots reserved

and packs both into the SHARED 6000-char evidence budget with contracts.pack_evidence, the same function the
answer call uses. Reported per group: the share of gold units in the selection and in the pack, paired better /
worse / equal records, and the graph slots used. Evidence recall is the quantity that decided v3's last round (it
explained the whole gap to v2); it is necessary for a better answer, not sufficient - the GPU run measures answers.

Records: the wide dev set (pilot.jsonl, 64 records, runs/v3recall_vec8 - v3's current routing) and the dev
timelines (pilot_timeline.jsonl, 8 records, runs/core_dev_pilot_timeline, the vectors-always arm). Dev only.

    .venv/Scripts/python.exe -m ragbench.tools.v4_replay
    .venv/Scripts/python.exe -m ragbench.tools.v4_replay --set graph_min_slots=6 --set vec_min_slots=4
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

from ..approaches.agentic3 import retrieve as RT
from ..approaches.agentic3.defaults import DEFAULTS as V3
from ..approaches.agentic4 import graphret as GR
from ..approaches.agentic4.defaults import DEFAULTS as V4
from ..config import load_config
from ..contracts import pack_evidence
from ..corpus.registry import load_set, set_name_for
from ..kb import graph as G
from ..kb import ingest as I
from ..kb import query as Q
from ..kb.schema import connect
from .v3_recall_trace import recorded_vector_candidates

ROOT = Path(__file__).resolve().parents[2]
QDIR = ROOT / "ragbench" / "data" / "questions"
SOURCES = [("wide", ROOT / "work/runs/v3recall_vec8/runs_AGENTIC3.jsonl", QDIR / "pilot.jsonl"),
           ("timeline", None, QDIR / "pilot_timeline.jsonl")]


def _timeline_file():
    d = ROOT / "work/runs/core_dev_pilot_timeline"
    for p in sorted(d.glob("runs_AGENTIC3*vectors_always*.jsonl")) + sorted(d.glob("runs_AGENTIC3.jsonl")):
        return p
    return None


def _questions(path):
    return {json.loads(l)["id"]: json.loads(l) for l in path.open(encoding="utf-8") if l.strip()}


def _pool(con, rec, opts, cap):
    tr = rec["trace"]
    plan = dict(tr["stages"]["plan"])
    plan.pop("doc_selection", None)
    plan.pop("missing_documents", None)
    doc_ids = list(tr["stages"]["retrieve"]["docs"])
    reader = Q.DocReader(con)
    facets = RT.facet_plans(plan, opts) if opts["facets"] else []
    trigram = bool(tr["stages"]["retrieve"].get("trigram_used"))
    pool, _i = RT.retrieve_all(con, plan, doc_ids, d_lang=rec["d_lang"], opts=opts, trigram=trigram, warm=False,
                               tree=True, reader=reader, facets=facets, cap=cap)
    pool += recorded_vector_candidates(con, rec, plan, rec["d_lang"], opts, reader)
    return plan, doc_ids, facets, pool, reader


def _pack(sel, corpus_set, cfg):
    items = [c.to_item(meta={"verified": None}) for c in sel]
    for i, it in enumerate(items):
        it.rank = i + 1
    pack = pack_evidence(items, corpus_set, cfg=cfg)
    return {u for it in pack.items for u in (it.unit_ids or [])}


def _graph_only(sel):
    """Windows in the selection that ONLY the graph found (v1 let these flood the cap; v2 caps them at its slots)."""
    return sum(1 for c in sel if ((c.features or {}).get("src") or []) == ["graph"])


def evaluate(overrides=None, family=True, cards=True, verbose=False, concepts=False):
    cfg = load_config()
    con = connect(I.kb_path(cfg), read_only=True)
    G.ensure_graph_schema  # the graph must exist; `kb graph build` (or kb sync) makes it
    o3 = dict(V3)
    o4 = dict(V4, **(overrides or {}))
    agent = None
    if concepts:                       # real embedding calls: names on first touch (cached), plan terms each time
        from ..approaches.agentic4 import GraphAgent
        agent = GraphAgent(cfg=cfg)
        scope = agent.client.ledger.scope(phase="eval", approach="v4_replay")
        scope.__enter__()              # every embed call of the replay is ledgered as evaluation work
    sets, rows = {}, []
    for group, path, qfile in SOURCES:
        path = path or _timeline_file()
        if path is None or not path.exists():
            continue
        qs = _questions(qfile)
        for line in path.open(encoding="utf-8"):
            rec = json.loads(line)
            q = qs.get(rec["question_id"])
            if q is None or rec.get("error"):
                continue
            gold = set((q.get("gold_units") or {}).get(rec["d_lang"]) or [])
            if not gold:
                continue
            name = set_name_for(rec["corpus"], rec["d_lang"])
            if name not in sets:
                sets[name] = load_set(name, cfg)
            multi = len(rec["trace"]["stages"]["retrieve"]["docs"]) > 1
            cap3 = int(o3["max_candidates_multi" if multi else "max_candidates"])
            cap4 = int(o4["max_candidates_multi" if multi else "max_candidates"])
            plan, doc_ids, facets, pool, _r = _pool(con, rec, o3, cap3)
            sel3, _m = RT.merge(pool, doc_ids, facets, o3, cap3)
            plan, doc_ids, facets, pool, reader = _pool(con, rec, o4, cap4)
            cpt = None
            if agent is not None and (len(doc_ids) > 1 or o4["graph_single_doc"]):
                mats, _ni = agent.names.ensure(con, doc_ids, agent.client)
                texts = GR.concept_texts(plan, rec["d_lang"])
                qv = agent.lazy.query_vectors(texts, agent.client, purpose="replay-concept-query")[0] if texts else []
                cpt = (mats, qv)
            g, ginfo = GR.graph_candidates(con, reader, plan, doc_ids, d_lang=rec["d_lang"], opts=o4,
                                           family=family, cards=cards, concepts=cpt)
            RT.tag_facet(g, None)
            sel4, m4 = GR.merge_graph(pool + g, doc_ids, facets, o4, cap4, per_period=bool(o4["graph_per_period"]),
                                      slots=GR.graph_slots(o4, doc_ids, plan.get("unit") or "count"))
            s3 = {u for c in sel3 for u in c.unit_ids}
            s4 = {u for c in sel4 for u in c.unit_ids}
            p3, p4 = _pack(sel3, sets[name], cfg), _pack(sel4, sets[name], cfg)
            g_units = {u for c in g for u in c.unit_ids}
            r = {"group": group if group == "timeline" else rec["corpus"], "qid": rec["question_id"],
                 "setting": rec["setting"], "n_gold": len(gold),
                 "sel3": len(gold & s3) / len(gold), "sel4": len(gold & s4) / len(gold),
                 "pack3": len(gold & p3) / len(gold), "pack4": len(gold & p4) / len(gold),
                 "prec3": len(gold & p3) / max(1, len(p3)), "prec4": len(gold & p4) / max(1, len(p4)),
                 "gonly": _graph_only(sel4), "multi": multi, "dropped": ginfo.get("dropped") or {},
                 "graph_pool": len(gold & g_units) / len(gold), "graph_slots": m4["graph_slots"],
                 "seeds": ginfo.get("seeds", 0), "walked": ginfo.get("walked", 0)}
            rows.append(r)
            if verbose:
                print("%-16s %-6s pack v3 %.2f v4 %.2f | graph pool %.2f | slots %d seeds %d walked %d" % (
                    r["qid"], r["setting"], r["pack3"], r["pack4"], r["graph_pool"], r["graph_slots"], r["seeds"],
                    r["walked"]))
    return rows


def report(rows, title="default"):
    groups = defaultdict(list)
    for r in rows:
        groups[r["group"]].append(r)
    groups["ALL"] = rows
    print("\n## v3 vs v4 evidence recall, no model call (%s)\n" % title)
    groups["MULTI-REPORT"] = [r for r in rows if r.get("multi")]
    print("| group | records | selected v3 | selected v4 | PACKED v3 | PACKED v4 | v4 better / worse / equal |"
          " gold in graph pool | graph slots | pack precision v3 / v4 | graph-only in selection |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for gname in ("A", "B", "C", "timeline", "MULTI-REPORT", "ALL"):
        rs = groups.get(gname) or []
        if not rs:
            continue
        n = len(rs)
        avg = lambda k: sum(x[k] for x in rs) / n            # noqa: E731
        b = sum(1 for x in rs if x["pack4"] > x["pack3"] + 1e-9)
        w = sum(1 for x in rs if x["pack4"] < x["pack3"] - 1e-9)
        print("| %s | %d | %.3f | %.3f | %.3f | **%.3f** | %d / %d / %d | %.3f | %.1f | %.3f / %.3f | %.1f |" % (
            gname, n, avg("sel3"), avg("sel4"), avg("pack3"), avg("pack4"), b, w, n - b - w, avg("graph_pool"),
            avg("graph_slots"), avg("prec3"), avg("prec4"), avg("gonly")))


def _parse_set(items):
    out = {}
    for it in items or []:
        k, _, v = it.partition("=")
        try:
            out[k] = json.loads(v)
        except ValueError:
            out[k] = v
    return out


def main(argv=None):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    ap = argparse.ArgumentParser(prog="python -m ragbench.tools.v4_replay", description=__doc__.split("\n\n")[0])
    ap.add_argument("--set", action="append", help="an AGENTIC4 option override, key=json")
    ap.add_argument("--no-family", action="store_true")
    ap.add_argument("--no-cards", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--concepts", action="store_true", help="use the concept layer (embedding calls)")
    args = ap.parse_args(argv)
    ov = _parse_set(args.set)
    rows = evaluate(ov, family=not args.no_family, cards=not args.no_cards, verbose=args.verbose,
                    concepts=args.concepts)
    report(rows, ", ".join("%s=%s" % kv for kv in ov.items()) or "v4 defaults")
    return 0


if __name__ == "__main__":
    sys.exit(main())
