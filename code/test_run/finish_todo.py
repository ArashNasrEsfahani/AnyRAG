"""Attempt the never-attempted replay cells (state 'todo' in cellstate) once each, as ordinary replay calls,
concurrency 1, with a pause after an HTTP 429 (10 s, doubling, max 120 s). Never touches an ok or errored cell
(those go to infra_cleanup.py, PROTOCOL.md section 10). Same calls as run_test._replay_worker."""
import json
import os
import sys
import time

import cellstate as CS

RT, RR = CS.RT, CS.RR


def main():
    m = RT.man()
    os.environ["OPENROUTER_PROVIDER_PINS"] = json.dumps(m["reader_pins"])
    os.environ["READER_FALLBACK_ENV"] = RT.FALLBACK_ENV
    sys.path.insert(0, m["code_worktree"])
    from ragbench.config import load_config
    from ragbench.llm import Ledger
    from ragbench.util import safe_name
    cfg = load_config()
    stop = json.loads((RT.TR / "budget_start.json").read_text(encoding="utf-8"))["start_usd"] + RT.STOP_ABOVE_START_USD
    todo = [c for c in CS.census() if c["state"] == "todo"]
    print("todo cells:", len(todo), flush=True)
    arms = {a["arm"]: a for a in m["arms"]}
    clients, pause, res = {}, 10, []
    for c in todo:
        rd, q, arm = c["reader"], c["qid"], arms[c["arm"]]
        st, prim = RT.primary_state(arm, q)
        out = RT.TR / "replay" / RT.slug(rd) / RT.pool_of(q) / arm["arm"]
        ok, errs = RR.done_rids(out, prim["approach"])
        if c["rid"] in ok or errs.get(c["rid"], 0):
            continue
        if RT.total() >= stop:
            print("BUDGET STOP", RT.total()); break
        if rd not in clients:
            led = Ledger(path=None)
            clients[rd] = (RR.make_reader(cfg, led, "openrouter:" + rd, int(m["reader_max_attempts"])), led)
        client, ledger = clients[rd]
        pk = RT.PACKS / safe_name(prim["approach"]) / ("%s__%s.json" % (q, prim["setting"]))
        out.mkdir(parents=True, exist_ok=True)
        ledger.path = str(out / ("ledger_%s.jsonl" % safe_name(prim["approach"])))
        rec = RR.replay_one(cfg, pk, prim, "openrouter:" + rd, out, client=client, ledger=ledger)
        k = CS.kind(rec.get("error")) if rec.get("error") else "ok"
        res.append({"qid": q, "arm": arm["arm"], "reader": rd, "result": k})
        RT.say("todo-finish %s %s %s -> %s" % (rd.split("/")[1][:14], q, arm["arm"], k), "replay.log")
        print(q, arm["arm"], k, flush=True)
        if k == "http429":
            time.sleep(pause); pause = min(pause * 2, 120)
        else:
            pause = 10
    print(json.dumps({k: sum(1 for r in res if r["result"] == k) for k in {r["result"] for r in res}}))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
