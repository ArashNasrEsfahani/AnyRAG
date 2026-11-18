"""Infra clean-up (PROTOCOL.md section 9, amendment 2, 2026-10-04): re-try ONLY cells whose every record is an
infrastructure error of kind HTTP 429 (provider rate limit) or local Ollama connection outage, regardless of the
earlier one-rerun counter. Concurrency 1 per reader, up to 5 tries per cell with back-off 10 / 30 / 90 / 270 s before
tries 2-5 (each try = one ordinary reader call under the frozen attempt policy, 7 HTTP attempts). Every try is logged in
reruns.jsonl with kind="infra-cleanup". A finished (non-error) cell is never touched. A try that ends in any other error
kind stops that cell (original rule). Primary (GLM) cells would be re-run through the frozen run_cell_watchdog; at the
time of the amendment there were none.

    python infra_cleanup.py [--dry-run]
"""
import datetime as dt
import json
import os
import sys
import threading
import time

import cellstate as CS

RT, RR = CS.RT, CS.RR
BACKOFF = [0, 10, 30, 90, 270]


def log(row):
    row = dict(at=dt.datetime.now().isoformat(timespec="seconds"), kind="infra-cleanup", **row)
    with open(RT.TR / "reruns.jsonl", "a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def eligible(c):
    return c["state"] != "ok" and c["state"] != "todo" and c["state"] != "no_pack" and c["kinds"] and \
        all(k in ("http429", "ollama") for k in c["kinds"])


def worker(m, reader, cells, stop_usd, res):
    sys.path.insert(0, m["code_worktree"])
    from ragbench.config import load_config
    from ragbench.llm import Ledger
    from ragbench.util import safe_name
    cfg = load_config()
    ledger = Ledger(path=None)
    client = RR.make_reader(cfg, ledger, "openrouter:" + reader, int(m["reader_max_attempts"]))
    arms = {a["arm"]: a for a in m["arms"]}
    for c in cells:
        q, arm = c["qid"], arms[c["arm"]]
        st, prim = RT.primary_state(arm, q)
        out = RT.TR / "replay" / RT.slug(reader) / RT.pool_of(q) / arm["arm"]
        pk = RT.PACKS / safe_name(prim["approach"]) / ("%s__%s.json" % (q, prim["setting"]))
        final = "failed"
        for t, wait in enumerate(BACKOFF, 1):
            ok, _errs = RR.done_rids(out, prim["approach"])
            if c["rid"] in ok:                       # never re-answer a finished cell
                final = "ok"
                break
            if RT.total() >= stop_usd:
                RT.say("CLEANUP BUDGET STOP %s %.4f" % (reader, RT.total()), "replay.log")
                res[reader].append({"qid": q, "arm": arm["arm"], "final": "budget_stop"})
                return
            if wait:
                time.sleep(wait)
            ledger.path = str(out / ("ledger_%s.jsonl" % safe_name(prim["approach"])))
            rec = RR.replay_one(cfg, pk, prim, "openrouter:" + reader, out, client=client, ledger=ledger)
            k = CS.kind(rec.get("error")) if rec.get("error") else "ok"
            log({"question_id": q, "arm": arm["arm"], "reader": reader, "try": t, "backoff_s": wait,
                 "result": k, "error": (rec.get("error") or "")[:200] or None})
            RT.say("cleanup %s %s %s try %d -> %s" % (reader.split("/")[1][:14], q, arm["arm"], t, k), "replay.log")
            if k == "ok":
                final = "ok"
                break
            if k == "other":
                final = "failed_other"
                break
        res[reader].append({"qid": q, "arm": arm["arm"], "final": final})


def main(dry):
    m = RT.man()
    os.environ["OPENROUTER_PROVIDER_PINS"] = json.dumps(m["reader_pins"])
    os.environ["READER_FALLBACK_ENV"] = RT.FALLBACK_ENV
    cells = CS.census()
    todo = [c for c in cells if c["state"] == "todo"]
    if todo:
        print("REFUSED: %d cells were never attempted (the replay follower has not finished)" % len(todo))
        return 2
    el = [c for c in cells if eligible(c)]
    prim = [c for c in el if c["primary"]]
    if prim:
        print("primary infra cells present (not handled by this helper): %d" % len(prim))
        return 2
    by = {}
    for c in el:
        by.setdefault(c["reader"], []).append(c)
    print(json.dumps({r: len(v) for r, v in by.items()}))
    if dry:
        return 0
    stop = json.loads((RT.TR / "budget_start.json").read_text(encoding="utf-8"))["start_usd"] + RT.STOP_ABOVE_START_USD
    res = {r: [] for r in by}
    ths = [threading.Thread(target=worker, args=(m, r, v, stop, res), name=r) for r, v in by.items()]
    for t in ths:
        t.start()
    for t in ths:
        t.join()
    (RT.TR / "infra_cleanup_result.json").write_text(json.dumps(res, ensure_ascii=False, indent=1) + "\n",
                                                     encoding="utf-8")
    print(json.dumps({r: {k: sum(1 for x in v if x["final"] == k) for k in {x["final"] for x in v}}
                      for r, v in res.items()}))
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main("--dry-run" in sys.argv[1:]))
