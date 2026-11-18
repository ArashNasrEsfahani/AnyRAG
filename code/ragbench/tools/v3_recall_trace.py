"""Where v3's evidence recall goes: a replay of the recorded dev plans against the knowledge base, no model call.

Written 2026-09-20 by the recall agent. The wide dev set (`work/runs/core_dev_wide`, pilot.jsonl, 64 records per
arm) put AGENTIC3+vectors_always at evidence recall 0.678 against AGENTIC2's 0.760. Two hypotheses explain that,
and they have different fixes:

  * a CAP problem - the gold unit is in the merged pool and the cap of 12 (24 multi-document) cuts it;
  * a RETRIEVAL problem - the gold unit never entered the pool, so no cap would admit it.

The trace records the SELECTED candidates only, plus `merge.pre_cap` as a count, so the pool contents have to be
rebuilt. Stages 2 and 4 make NO model call: given the recorded plan, the recorded targeted documents and the
recorded options, `retrieve_all` is a pure function of the knowledge base. The vector pass needs a query
embedding, which IS a model call - so it is not re-embedded either: `trace.stages.vectors.search.top` records
every passage id the search returned with its blended similarity, and `expand_passage` over those rows rebuilds
the vector candidates exactly as the run had them.

Levels reported per record, each a superset of the next:

  reachable   the gold unit exists in the `units` table of a document the plan TARGETED
  pool        the gold unit is on a candidate of the UNCAPPED pool (every row cap and pass cap lifted)
  pre_cap     ... of the pool as the run actually built it, before `merge`'s cap
  selected    ... of the candidates the run kept (this is the record's own `ret_recall`)

    .venv/Scripts/python.exe -m ragbench.tools.v3_recall_trace
    .venv/Scripts/python.exe -m ragbench.tools.v3_recall_trace --arm AGENTIC3 --verbose
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
WIDE = ROOT / "work" / "runs" / "core_dev_wide"
QUESTIONS = ROOT / "ragbench" / "data" / "questions" / "pilot.jsonl"

ARMS = {
    "AGENTIC3": "runs_AGENTIC3.jsonl",
    "AGENTIC3+vectors_always": "runs_AGENTIC3_vectors_always-7afdfb27.jsonl",
    "AGENTIC2": "runs_AGENTIC2.jsonl",
}

BIG = 10000        # "no cap": larger than any pool these documents can produce
VEC_SLOTS = (0, 4, 6, 8)

# every knob that trims a candidate list before `merge` sees it
UNCAPPED = {
    "facts_max_rows": BIG, "facts_rows_per_doc": BIG, "single_doc_cap": BIG,
    "sections_max_rows": BIG, "units_max_rows": BIG, "units_rows_per_doc": BIG,
    "facet_cap": BIG, "max_candidates": BIG, "max_candidates_multi": BIG,
    "per_doc_cap": BIG, "factdb_max_rows": BIG, "factdb_rows_per_doc": BIG,
}


def load_questions():
    out = {}
    for line in QUESTIONS.open(encoding="utf-8"):
        q = json.loads(line)
        out[q["id"]] = q
    return out


def records(arm):
    return [json.loads(l) for l in (WIDE / ARMS[arm]).open(encoding="utf-8")]


def gold_units(q, d_lang):
    return list(dict.fromkeys((q.get("gold_units") or {}).get(d_lang) or []))


def _recorded_pre_cap(rec):
    """The pre-cap size of the pool the run FINALLY capped: the vectors stage re-merges, so its own merge info
    supersedes the database merge whenever the route ran the vector pass."""
    st = rec["trace"]["stages"]
    v = st.get("vectors")
    if v and (v.get("merge") or {}).get("pre_cap") is not None:
        return v["merge"]["pre_cap"]
    return (st["retrieve"].get("merge") or {}).get("pre_cap")


# ---------------------------------------------------------------------------------------------------------------
# replay
# ---------------------------------------------------------------------------------------------------------------

def passage_rows(con, doc_id, passage_ids):
    """The stored passage rows for these ids, in the order given (no `vec`: the similarity is recorded)."""
    by_id = {}
    for r in con.execute(
            "SELECT passage_id, doc_id, section_id, ord, unit_ids, text, embed_sha1, dim FROM passages "
            "WHERE doc_id = ?", (doc_id,)):
        by_id[r["passage_id"]] = dict(r)
    return [by_id[p] for p in passage_ids if p in by_id]


def recorded_vector_candidates(con, rec, plan, d_lang, opts, reader):
    """The run's own vector hits, rebuilt from the trace with no embedding call. -> [Candidate]."""
    from ..approaches.agentic2 import tree as T
    from ..kb import query as Q
    v = (rec["trace"]["stages"] or {}).get("vectors")
    if not v:
        return []
    top = ((v.get("search") or {}).get("top") or {})
    ctx = T.search_context(plan, d_lang)
    unit_kind = plan.get("unit") or "count"
    scale = float(opts["vec_score_scale"])
    out = []
    for doc_id, hits in top.items():
        rows = {r["passage_id"]: r for r in passage_rows(con, doc_id, [h[0] for h in hits])}
        for pid, sim in hits:
            row = rows.get(pid)
            if row is None:
                continue
            for c in Q.expand_passage(reader, row, ctx, opts, unit_kind, tree=True):
                c.score = round(scale * float(sim), 4)
                c.features = {"src": ["vector"], "vec": round(float(sim), 4), "pid": pid, "facet": []}
                out.append(c)
    return out


def replay(con, rec, opts_override=None, cap_override=None):
    """One record's retrieval, replayed. -> dict with the ranked pool and the selection."""
    from ..approaches.agentic3 import retrieve as RT
    from ..approaches.agentic3.defaults import DEFAULTS
    from ..kb import query as Q

    tr = rec["trace"]
    plan = dict(tr["stages"]["plan"])
    plan.pop("doc_selection", None)
    plan.pop("missing_documents", None)
    doc_ids = list(tr["stages"]["retrieve"]["docs"])
    opts = dict(DEFAULTS)
    opts.update(tr["params"])
    opts.update(opts_override or {})
    d_lang = rec["d_lang"]
    multi = len(doc_ids) > 1
    cap = int(cap_override if cap_override is not None
              else opts["max_candidates_multi" if multi else "max_candidates"])

    reader = Q.DocReader(con)
    facets = RT.facet_plans(plan, opts) if opts["facets"] else []
    trigram = bool(tr["stages"]["retrieve"].get("trigram_used"))
    pool, _rinfo = RT.retrieve_all(con, plan, doc_ids, d_lang=d_lang, opts=opts, trigram=trigram,
                                   warm=False, tree=True, reader=reader, facets=facets, cap=cap)
    vcands = recorded_vector_candidates(con, rec, plan, d_lang, opts, reader)
    allc = pool + vcands
    ranked = RT.absorb_facets(Q.rank_candidates(list(allc)), allc)
    sel, minfo = RT.merge(allc, doc_ids, facets, opts, cap)
    return {"ranked": ranked, "selected": sel, "merge": minfo, "doc_ids": doc_ids,
            "n_vec": len(vcands), "cap": cap}


def gold_rank(cands, gold):
    """1-based rank of the first candidate carrying ANY gold unit, and how many gold units the list covers."""
    g = set(gold)
    rank = None
    covered = set()
    for i, c in enumerate(cands, 1):
        hit = g & set(c.unit_ids)
        if hit:
            covered |= hit
            if rank is None:
                rank = i
    return rank, covered


# ---------------------------------------------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------------------------------------------

def unit_exists(con, unit_id):
    return con.execute("SELECT 1 FROM units WHERE unit_id = ? LIMIT 1", (unit_id,)).fetchone() is not None


def analyse(arm, verbose=False, caps=(12, 24, 48)):
    from ..approaches.agentic3 import retrieve as RT
    from ..config import load_config
    from ..kb import ingest as I
    from ..kb.schema import connect

    qs = load_questions()
    con = connect(I.kb_path(load_config()), read_only=True)
    rows = []
    for rec in records(arm):
        q = qs[rec["question_id"]]
        gold = gold_units(q, rec["d_lang"])
        r = {"qid": rec["question_id"], "setting": rec["setting"], "d_lang": rec["d_lang"],
             "qtype": q.get("type"), "n_gold": len(gold),
             "ret_units": {u for it in (rec.get("retrieved_units") or []) for u in (it.get("unit_ids") or [])},
             "ev_units": set(rec.get("evidence_units") or []),
             "gold": gold, "docs_targeted": list(rec["trace"]["stages"]["retrieve"]["docs"]),
             "pre_cap_recorded": _recorded_pre_cap(rec)}
        if not gold:
            rows.append(r)
            continue
        r["reachable"] = [u for u in gold if unit_exists(con, u)]
        r["in_targeted_doc"] = [u for u in gold if u.split("::")[0] in set(r["docs_targeted"])]

        # 1. the pool as the run built it
        run = replay(con, rec)
        r["replay_selected_units"] = {u for c in run["selected"] for u in c.unit_ids}
        r["pre_cap_replay"] = run["merge"]["pre_cap"]
        rank, cov = gold_rank(run["ranked"], gold)
        r["rank_pre_cap"] = rank
        r["cov_pre_cap"] = cov
        rank_s, cov_s = gold_rank(run["selected"], gold)
        r["rank_selected"] = rank_s
        r["cov_selected"] = cov_s

        # 2. the pool with every cap lifted
        full = replay(con, rec, opts_override=UNCAPPED, cap_override=BIG)
        rank_f, cov_f = gold_rank(full["ranked"], gold)
        r["rank_uncapped"] = rank_f
        r["cov_uncapped"] = cov_f
        r["pool_uncapped"] = len(full["ranked"])

        # 3. the (cap, vec_min_slots) grid, on the run's own pool. The `candidates` knob raises the pass cap and
        #    the merge cap together, exactly as `AGENTIC3_OPTIONS` would; multi-document caps go up in proportion.
        multi = len(run["doc_ids"]) > 1
        for cp in caps:
            for vms in VEC_SLOTS:
                c_eff = cp * 2 if multi else cp
                res = replay(con, rec, opts_override={"max_candidates": cp, "max_candidates_multi": cp * 2,
                                                      "vec_min_slots": vms},
                             cap_override=c_eff)
                _rk, cv = gold_rank(res["selected"], gold)
                r["cov_cap_%d_v%d" % (cp, vms)] = cv
                if vms == 4:
                    r["cov_cap_%d" % cp] = cv
                    r["sel_cap_%d" % cp] = len(res["selected"])
                    r["vecslots_cap_%d" % cp] = res["merge"]["vec_slots"]
        rows.append(r)
    con.close()
    return rows


def report(arm, rows, verbose=False, caps=(12, 24, 48)):
    scored = [r for r in rows if r["n_gold"]]
    n = len(scored)
    print("\n## %s  (%d records, %d with gold units)\n" % (arm, len(rows), n))

    def rec_recall(key):
        return sum(len(r.get(key) or ()) / r["n_gold"] for r in scored) / n

    print("| level | ev-recall (mean over records) | records with >=1 gold |")
    print("|---|---|---|")
    print("| gold unit exists in kb            | %.3f | %d |"
          % (sum(len(r["reachable"]) / r["n_gold"] for r in scored) / n,
             sum(1 for r in scored if r["reachable"])))
    print("| gold unit in a TARGETED document  | %.3f | %d |"
          % (sum(len(r["in_targeted_doc"]) / r["n_gold"] for r in scored) / n,
             sum(1 for r in scored if r["in_targeted_doc"])))
    print("| uncapped pool                     | %.3f | %d |"
          % (rec_recall("cov_uncapped"), sum(1 for r in scored if r.get("cov_uncapped"))))
    print("| pre-cap pool (run's own caps)     | %.3f | %d |"
          % (rec_recall("cov_pre_cap"), sum(1 for r in scored if r.get("cov_pre_cap"))))
    for cp in caps:
        print("| selected, cap = %-3d               | %.3f | %d |"
              % (cp, rec_recall("cov_cap_%d" % cp), sum(1 for r in scored if r.get("cov_cap_%d" % cp))))
    print("| retrieved (recorded)              | %.3f | %d |"
          % (sum(len(set(r["gold"]) & r["ret_units"]) / r["n_gold"] for r in scored) / n,
             sum(1 for r in scored if set(r["gold"]) & r["ret_units"])))
    print("| evidence   (recorded)             | %.3f | %d |"
          % (sum(len(set(r["gold"]) & r["ev_units"]) / r["n_gold"] for r in scored) / n,
             sum(1 for r in scored if set(r["gold"]) & r["ev_units"])))

    # replay fidelity
    same = sum(1 for r in scored if r.get("replay_selected_units") is not None
               and (set(r["gold"]) & r["replay_selected_units"]) == (set(r["gold"]) & r["ret_units"]))
    print("\nreplay fidelity: gold coverage of the replayed selection equals the recorded one in %d/%d records"
          % (same, n))
    pc = sum(1 for r in scored if r["pre_cap_replay"] == r["pre_cap_recorded"])
    print("replay fidelity: merge pre_cap equals the recorded pre_cap in %d/%d records" % (pc, n))

    print("\n### the (cap, vec_min_slots) grid: mean gold coverage of the SELECTED candidates\n")
    print("| cap | " + " | ".join("vec_min_slots=%d" % v for v in VEC_SLOTS) + " |")
    print("|---|" + "---|" * len(VEC_SLOTS))
    for cp in caps:
        cells = []
        for v in VEC_SLOTS:
            k = "cov_cap_%d_v%d" % (cp, v)
            cells.append("%.3f" % (sum(len(r.get(k) or ()) / r["n_gold"] for r in scored) / n))
        print("| %d | %s |" % (cp, " | ".join(cells)))

    # the cap question
    cut = [r for r in scored if r.get("cov_pre_cap") and (r["cov_pre_cap"] - (r.get("cov_cap_12") or set()))]
    never = [r for r in scored if r["in_targeted_doc"] and not r.get("cov_uncapped")]
    offdoc = [r for r in scored if not r["in_targeted_doc"]]
    print("\ncap vs retrieval, per record:")
    print("  gold in pre-cap pool but NOT selected at cap 12 : %d" % len(cut))
    print("  gold in a targeted doc but NOT in the uncapped pool: %d" % len(never))
    print("  gold not in ANY targeted document                 : %d" % len(offdoc))
    if verbose:
        for label, group in (("CUT BY CAP", cut), ("NEVER RETRIEVED", never), ("DOC NOT TARGETED", offdoc)):
            for r in group:
                print("  [%s] %-22s %-5s gold=%d rank_pre_cap=%s pre_cap=%d docs=%s"
                      % (label, r["qid"], r["setting"], r["n_gold"], r["rank_pre_cap"], r["pre_cap_replay"],
                         ",".join(r["docs_targeted"])[:60]))
    ranks = [r["rank_pre_cap"] for r in scored if r.get("rank_pre_cap")]
    if ranks:
        ranks.sort()
        print("\nrank of the first gold-bearing candidate in the pre-cap pool (n=%d): min %d, median %d, max %d"
              % (len(ranks), ranks[0], ranks[len(ranks) // 2], ranks[-1]))
        over = [x for x in ranks if x > 12]
        print("  ranks worse than 12: %s" % (over or "none"))
    return scored


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", action="append", default=None)
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--json-out", default=None)
    a = ap.parse_args(argv)
    arms = a.arm or ["AGENTIC3+vectors_always", "AGENTIC3"]
    out = {}
    for arm in arms:
        rows = analyse(arm, verbose=a.verbose)
        report(arm, rows, verbose=a.verbose)
        out[arm] = [{k: (sorted(v) if isinstance(v, set) else v) for k, v in r.items()} for r in rows]
    if a.json_out:
        Path(a.json_out).write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
        print("\nwrote %s" % a.json_out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
