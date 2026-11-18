#!/usr/bin/env python3
"""Check choose.py against the test profiles in references/chooser.json.

Every profile must reproduce the expected slots, the expected lists (same ids,
same order) and the exact sequence of rules fired. Also checks that every id
the rules can produce exists in the option catalogue and every evidence id
exists in evidence.json.  Run:  python test_choose.py
"""
import itertools
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import choose  # noqa: E402


def main():
    chooser = choose.load("chooser.json")
    evidence = {e["id"] for e in choose.load("evidence.json")}
    failures = 0
    for prof in chooser["tests"]:
        ans, state, fired = choose.choose(prof["answers"], chooser)
        problems = []
        for k, v in prof["expected"].items():
            if state.get(k) != v:
                problems.append("%s: got %r, expected %r" % (k, state.get(k), v))
        if "rules_fired" in prof and fired != prof["rules_fired"]:
            problems.append("rules_fired: got %r, expected %r" % (fired, prof["rules_fired"]))
        status = "PASS" if not problems else "FAIL"
        print("%s  %s" % (status, prof["name"]))
        for p in problems:
            print("      " + p)
        failures += bool(problems)

    # Exhaustive sweep: every answer combination fills every slot with a known
    # option, and every evidence id resolves.
    opts = chooser["options"]
    qopts = [[o["id"] for o in q["options"]] for q in chooser["questions"]]
    combos, bad = 0, 0
    for combo in itertools.product(*qopts):
        raw = dict(zip(choose.QIDS, combo))
        ans, state, _ = choose.choose(raw, chooser)
        combos += 1
        ids = [state[s] for s in choose.SLOTS] + [i for l in choose.LISTS for i in state[l]]
        for i in ids:
            if i is None or i not in opts or any(e not in evidence for e in opts[i].get("evidence", [])):
                bad += 1
                if bad <= 5:
                    print("BAD   %s -> %r" % (raw, i))
                break
    print("\nSweep: %d answer combinations, %d with a missing slot/option/evidence id" % (combos, bad))
    n = len(chooser["tests"])
    print("Profiles: %d of %d passed" % (n - failures, n))
    return 1 if failures or bad else 0


if __name__ == "__main__":
    sys.exit(main())
