"""Can the AGENTIC3 FAST path ever fire with acceptable precision, or is it a negative result?

Written 2026-09-20 by the decision agent. FAST fired 0 times in all 300 records of the v3 dev ablation matrix, so
the headline claim "a lookup the database settles alone is answered in two calls and about 4 seconds" has no
evidence behind it at all. A path that never fires is not a feature, and carrying it in the paper unmeasured is
worse than removing it.

The question is decided WITHOUT GPU TIME, by replaying the recorded route decisions. Every AGENTIC3 record's trace
holds the plan, the full candidate list with the score breakdown `kb/query.py` computed (`features.db`), the fact
row each candidate came from (`features.fact`, including its value), and the route the gate chose. `route.py`'s
own `fast_check` is a pure function of exactly those inputs and the thresholds in `defaults.py`, so re-running it
over the recorded candidates under other thresholds gives the same answer the agent would have given - with no
model call, because the FAST route's answer would have been built from the same top candidate.

PRECISION is then measurable, and this is the number the decision turns on: for every record the relaxed gate
WOULD have fired on, does the top candidate's own value equal the question's gold answer? A fast wrong answer is
worse than a slow right one (spec 6 stage 3), so a threshold is only worth having if that share is high.

DEV FILES ONLY. The replay reads `dev_core.jsonl`, `dev_core_timeline.jsonl`, `pilot.jsonl` and
`pilot_timeline.jsonl`. It never opens a test question file and it makes no model call.

    .venv/Scripts/python.exe -m ragbench.tools.v3_fastgate
    .venv/Scripts/python.exe -m ragbench.tools.v3_fastgate --runs work/runs/core_dev_wide
"""
import argparse
import json
from pathlib import Path

from ..approaches.agentic.numbers import same_number
from ..approaches.agentic3 import route as RO
from ..approaches.agentic3.defaults import DEFAULTS
from ..contracts import load_questions

ROOT = Path(__file__).resolve().parents[2]
DEV_FILES = ["dev_core.jsonl", "dev_core_timeline.jsonl", "pilot.jsonl", "pilot_timeline.jsonl"]
DEFAULT_RUNS = ["work/runs/core_dev", "work/runs/core_dev_pilot_timeline", "work/runs/core_dev_wide"]


class Cand:
    """The parts of a merged candidate `route.fast_check` reads, rebuilt from `trace.stages.retrieve.candidates`."""

    def __init__(self, d):
        self.cid = d.get("cid")
        self.score = float(d.get("score") or 0.0)
        self.features = d.get("features") or {}
        self.doc_id = d.get("doc_id")


def gold_values(files=DEV_FILES):
    """{question_id: gold numeric answer} for the dev questions that have one. Dev files only."""
    out = {}
    for name in files:
        path = ROOT / "ragbench" / "data" / "questions" / name
        if not path.is_file():
            continue
        for q in load_questions(path):
            if q.answer_value is not None:
                out[q.id] = float(q.answer_value)
    return out


def load_records(run_dirs, approach="AGENTIC3"):
    out = []
    for d in run_dirs:
        for f in sorted((ROOT / d).glob("runs_%s*.jsonl" % approach)):
            for line in f.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                r = json.loads(line)
                if r.get("error") or not r.get("trace"):
                    continue
                r["_dir"] = d
                out.append(r)
    return out


def replay(rec, opts):
    """-> (fired, reason, top) for one record under `opts`. None when the gate was never reached."""
    tr = rec["trace"]["stages"]
    plan = tr.get("plan") or {}
    route = tr.get("route") or {}
    if route.get("route") == "TIMELINE" or plan.get("kind") != "single":
        return None
    det = route.get("detail") or {}
    esc = det.get("escalate") or {}
    if esc.get("reason") not in ("database_pool_full", "text_answer_covered"):
        return None                     # the pool test sent this question to the vectors before the gate
    cands = [Cand(c) for c in (tr.get("retrieve") or {}).get("candidates") or []]
    if not cands:
        return None
    info = {"docs": (tr.get("retrieve") or {}).get("docs") or (route.get("detail", {}).get("docs") or [])}
    if not info["docs"]:
        info["docs"] = sorted({c.doc_id for c in cands})
    ok, reason, detail = RO.fast_check(plan, cands, info, opts)
    return ok, reason, cands[0]


def value_correct(cand, gold):
    """Would a FAST answer from this candidate have been right? Its fact row's value against the gold number."""
    if gold is None:
        return None
    readings = RO.value_readings((cand.features.get("fact") or {}).get("value"))
    if not readings:
        return False
    return any(same_number(x, gold) for x in readings)


def sweep(recs, golds, settings):
    print("\n| setting | gate reached | fires | share | value correct | precision |")
    print("|---|---|---|---|---|---|")
    for name, over in settings:
        opts = dict(DEFAULTS, **over)
        reached = fired = correct = scoreable = 0
        for r in recs:
            res = replay(r, opts)
            if res is None:
                continue
            reached += 1
            ok, _reason, top = res
            if not ok:
                continue
            fired += 1
            v = value_correct(top, golds.get(r["question_id"]))
            if v is not None:
                scoreable += 1
                correct += 1 if v else 0
        share = "%.1f%%" % (100.0 * fired / reached) if reached else "-"
        prec = "%.2f" % (correct / scoreable) if scoreable else "-"
        print("| %s | %d | %d | %s | %d of %d | %s |" % (name, reached, fired, share, correct, scoreable, prec))


def refusals(recs, opts):
    print("\nRefusal mix at the shipped thresholds (the axis that failed first):")
    mix = {}
    for r in recs:
        res = replay(r, opts)
        if res is None:
            continue
        ok, reason, _top = res
        mix[reason if not ok else "FIRED"] = mix.get(reason if not ok else "FIRED", 0) + 1
    for k, v in sorted(mix.items(), key=lambda x: -x[1]):
        print("  - `%s` %d" % (k, v))


def top_is_gold(recs, golds, opts):
    """The ceiling of any gate whose answer is the rank-1 candidate: how often is rank 1 already the gold value?"""
    n = hit = 0
    for r in recs:
        res = replay(r, opts)
        if res is None:
            continue
        v = value_correct(res[2], golds.get(r["question_id"]))
        if v is None:
            continue
        n += 1
        hit += 1 if v else 0
    print("\nCEILING: on the %d records that reach the gate and have a gold number, the rank-1 candidate's own "
          "value is the gold value %d times (%.2f). No threshold on these features can beat that." %
          (n, hit, hit / n if n else float("nan")))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=",".join(DEFAULT_RUNS))
    args = ap.parse_args()
    dirs = [d for d in args.runs.split(",") if (ROOT / d).is_dir()]
    recs = load_records(dirs)
    golds = gold_values()
    print("# FAST gate replay — %d AGENTIC3 records from %s, %d dev questions with a gold number"
          % (len(recs), ", ".join(dirs), len(golds)))
    base = dict(DEFAULTS)
    refusals(recs, base)
    top_is_gold(recs, golds, base)
    sweep(recs, golds, [
        ("shipped (cover 0.6, all axes)", {}),
        ("cover 0.4", {"fast_min_cover": 0.4}),
        ("cover 0.3", {"fast_min_cover": 0.3}),
        ("cover 0.3 + any source kind", {"fast_min_cover": 0.3, "fast_source_kinds": ()}),
        ("no label test (terms>=2 only)", {"fast_need_all_terms": False}),
        ("no label test + any source", {"fast_need_all_terms": False, "fast_source_kinds": ()}),
        ("no label test + terms>=3", {"fast_need_all_terms": False, "fast_min_terms": 3}),
        ("no label test + terms>=4", {"fast_need_all_terms": False, "fast_min_terms": 4}),
        ("no label, any source, terms>=4", {"fast_need_all_terms": False, "fast_min_terms": 4,
                                            "fast_source_kinds": ()}),
        ("everything off but the rivals", {"fast_need_all_terms": False, "fast_need_kind": False,
                                           "fast_need_year": False, "fast_source_kinds": (),
                                           "fast_min_terms": 1}),
    ])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
