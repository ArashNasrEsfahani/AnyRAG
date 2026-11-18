"""Why raising the candidate cap cannot fix v3's evidence recall, and why the vector reservation can.

Written 2026-09-20 by the recall agent, as the free half of the recall experiment: no model call, no GPU. It is
the analysis that decided NOT to spend GPU on the `candidates = 24` arm.

`tools/v3_recall_trace.py` replays the recorded dev plans and shows where the gold unit is at each level - in a
targeted document (59 of 59 records), on a candidate of the uncapped pool (0.960), of the pool the run actually
built (0.866), of the selection the cap of 12 leaves (0.703). That makes the loss a SELECTION loss and points at
the cap. This module asks the next question, which the trace alone cannot answer: does a bigger selection reach
the answer prompt? `pack_evidence` renders the items in order into a FAIRNESS-CRITICAL 6000-char budget and stops
at the first item that does not fit, and twelve candidates already render about 5300 chars.

Three passes over the same replayed selections:

  grid      selected gold coverage, and the coverage that survives the pack, over a (cap, vec_min_slots) grid
  oracle    the same with the gold-bearing items moved to the FRONT - a packing order no run can have. If the
            oracle pack at a bigger cap is no better than at cap 12, the cap cannot help under ANY ordering; if
            it is better, the cap is reachable only through the order.
  sweep     the reservation swept at the production cap, to see whether the chosen value is a peak or a slope

Calibration is printed with the grid: the same simulation at today's cap and the OLD reservation is compared with
the recorded ret-recall and ev-recall of the same records, so the reader can see how far merge order sits from
the run's own verdict order.

    .venv/Scripts/python.exe -m ragbench.tools.v3_recall_pack
    .venv/Scripts/python.exe -m ragbench.tools.v3_recall_pack --pass sweep
"""
import argparse
import json
import sys

from ..config import load_config
from ..contracts import pack_evidence
from ..corpus.registry import load_set, set_name_for
from ..kb import ingest as I
from ..kb.schema import connect
from .v3_recall_trace import ARMS, WIDE, gold_units, load_questions, replay

ARM = "AGENTIC3+vectors_always"          # the arm whose records carry today's routing
GRID = [(12, 4), (12, 8), (24, 4), (24, 8), (48, 4)]
SWEEP = [0, 2, 4, 6, 8, 10, 12]


def _open():
    cfg = load_config()
    return cfg, load_questions(), connect(I.kb_path(cfg), read_only=True)


def _records():
    return [json.loads(l) for l in (WIDE / ARMS[ARM]).open(encoding="utf-8")]


def _pack(sel, corpus_set, cfg, order=None):
    """-> (unit ids the pack carries, EvidencePack). `order` replaces the merge order."""
    items = [c.to_item(meta={"verified": None}) for c in (order if order is not None else sel)]
    for i, it in enumerate(items):
        it.rank = i + 1
    pack = pack_evidence(items, corpus_set, cfg=cfg)
    return {u for it in pack.items for u in (it.unit_ids or [])}, pack


def _selection(con, rec, cap, vms):
    multi = len(rec["trace"]["stages"]["retrieve"]["docs"]) > 1
    return replay(con, rec, opts_override={"max_candidates": cap, "max_candidates_multi": cap * 2,
                                           "vec_min_slots": vms},
                  cap_override=cap * 2 if multi else cap)


def run(which="all"):
    cfg, qs, con = _open()
    sets, rows = {}, []
    for rec in _records():
        q = qs[rec["question_id"]]
        gold = set(gold_units(q, rec["d_lang"]))
        if not gold:
            continue
        name = set_name_for(rec["corpus"], rec["d_lang"])
        if name not in sets:
            sets[name] = load_set(name, cfg)
        cs = sets[name]
        ev = set(rec.get("evidence_units") or [])
        ret = {u for it in (rec.get("retrieved_units") or []) for u in (it.get("unit_ids") or [])}
        r = {"qid": rec["question_id"], "setting": rec["setting"], "n_gold": len(gold),
             "rec_ret": len(gold & (ret | ev)) / len(gold), "rec_ev": len(gold & ev) / len(gold)}
        pairs = GRID if which in ("all", "grid", "oracle") else []
        for cap, vms in pairs:
            res = _selection(con, rec, cap, vms)
            sel = res["selected"]
            merged, mp = _pack(sel, cs, cfg)
            oracle = sorted(range(len(sel)), key=lambda i: (0 if (gold & set(sel[i].unit_ids)) else 1, i))
            oracled, _op = _pack(sel, cs, cfg, order=[sel[i] for i in oracle])
            k = "%d_%d" % (cap, vms)
            r["sel_" + k] = len(gold & {u for c in sel for u in c.unit_ids}) / len(gold)
            r["pack_" + k] = len(gold & merged) / len(gold)
            r["oracle_" + k] = len(gold & oracled) / len(gold)
            r["nin_" + k], r["noff_" + k], r["chars_" + k] = mp.n_included, mp.n_offered, mp.chars
        if which in ("all", "sweep"):
            for v in SWEEP:
                res = replay(con, rec, opts_override={"vec_min_slots": v})
                sel = res["selected"]
                packed, _p = _pack(sel, cs, cfg)
                r["ssel_%d" % v] = len(gold & {u for c in sel for u in c.unit_ids}) / len(gold)
                r["spack_%d" % v] = len(gold & packed) / len(gold)
                r["svec_%d" % v] = res["merge"]["vec_slots"]
        rows.append(r)
    con.close()
    return rows


def report(rows, which="all"):
    n = len(rows)
    avg = lambda k: sum(r[k] for r in rows) / n            # noqa: E731
    if which in ("all", "grid", "oracle"):
        print("\n## The cap and the 6000-char evidence budget (%d gold-bearing dev records)\n" % n)
        print("| cap | vec_min_slots | selected gold | packed, merge order | packed, ORACLE order |"
              " items offered | items packed | chars |")
        print("|---|---|---|---|---|---|---|---|")
        for cap, vms in GRID:
            k = "%d_%d" % (cap, vms)
            print("| %d | %d | %.3f | %.3f | %.3f | %.1f | %.1f | %.0f |"
                  % (cap, vms, avg("sel_" + k), avg("pack_" + k), avg("oracle_" + k),
                     avg("noff_" + k), avg("nin_" + k), avg("chars_" + k)))
        print("\ncalibration: recorded ret-recall %.3f and ev-recall %.3f; the same records simulated at cap 12 /"
              " slots 4 give selected %.3f and packed %.3f. Merge order is a little PESSIMISTIC against the run's"
              " own verdict order, by %.3f." % (avg("rec_ret"), avg("rec_ev"), avg("sel_12_4"), avg("pack_12_4"),
                                                avg("rec_ev") - avg("pack_12_4")))
        better = [r for r in rows if r["pack_24_4"] > r["pack_12_4"]]
        worse = [r for r in rows if r["pack_24_4"] < r["pack_12_4"]]
        ob = [r for r in rows if r["oracle_24_4"] > r["oracle_12_4"]]
        print("\ncap 24 against cap 12 at the same reservation: packed coverage better on %d records, worse on %d"
              % (len(better), len(worse)))
        print("... but with the gold-bearing items packed FIRST it is better on %d. The cap is reachable only"
              " through the ORDER of the pack, never through the size of the selection." % len(ob))
    if which in ("all", "sweep"):
        print("\n## The vector reservation swept at the production cap\n")
        print("| vec_min_slots | selected gold | packed gold | vector candidates selected |")
        print("|---|---|---|---|")
        for v in SWEEP:
            print("| %d | %.3f | %.3f | %.1f |" % (v, avg("ssel_%d" % v), avg("spack_%d" % v), avg("svec_%d" % v)))
        best = max(SWEEP, key=lambda v: avg("spack_%d" % v))
        print("\npeak packed coverage at vec_min_slots = %d" % best)
        drop = [r for r in rows if r["spack_10"] < r["spack_8"] or r["spack_12"] < r["spack_8"]]
        for r in drop:
            print("  falls above 8: %-22s %-5s  8:%.2f  10:%.2f  12:%.2f"
                  % (r["qid"], r["setting"], r["spack_8"], r["spack_10"], r["spack_12"]))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--pass", dest="which", default="all", choices=["all", "grid", "oracle", "sweep"])
    ap.add_argument("--json-out", default=None)
    a = ap.parse_args(argv)
    rows = run(a.which)
    report(rows, a.which)
    if a.json_out:
        from pathlib import Path
        Path(a.json_out).write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
        print("\nwrote %s" % a.json_out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
