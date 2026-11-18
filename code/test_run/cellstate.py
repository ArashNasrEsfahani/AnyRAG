"""Read-only census of every primary and replay cell of the one-shot test (no model call, writes cellstate.json).
State per cell: ok | err1 | err2+ | todo, and the error kinds of its records (infra = HTTP 429 or local Ollama
connection outage; other = anything else). Used by infra_cleanup.py (PROTOCOL.md section 9, amendment 2)."""
import json
import re
import sys
from pathlib import Path

FZ = Path(r"D:\KareOne\RAG Paper\work\datasets_v2\test_freeze")
sys.path.insert(0, str(FZ))
import run_test as RT  # noqa: E402

sys.path.insert(0, RT.man()["code_worktree"])
from ragbench.tools import reader_replay as RR  # noqa: E402


def kind(err):
    e = str(err or "")
    if re.search(r"HTTP 429", e):
        return "http429"
    if re.search(r"(127\.0\.0\.1|localhost)", e) and re.search(r"11434|/api/embed|ConnectionError|NewConnection", e):
        return "ollama"
    return "other"


def census():
    m = RT.man()
    cells = []
    for qid, arms in RT.schedule(m):
        for arm in arms:
            rs = RT.records(RT.TR / "runs" / RT.pool_of(qid) / arm["arm"], qid)
            ok = [r for r in rs if not r.get("error")]
            errs = [r for r in rs if r.get("error")]
            cells.append({"reader": m["primary_reader"], "qid": qid, "arm": arm["arm"], "primary": True,
                          "state": "ok" if ok else ("err%d" % len(errs) if errs else "todo"),
                          "kinds": [kind(r["error"]) for r in errs], "last_error": (errs[-1]["error"][:200] if errs else None)})
            prim = ok[-1] if ok else None
            for rd in m["readers"]:
                if rd == m["primary_reader"]:
                    continue
                c = {"reader": rd, "qid": qid, "arm": arm["arm"], "primary": False}
                if prim is None:
                    c.update(state="no_pack", kinds=[], last_error=None)
                    cells.append(c)
                    continue
                out = RT.TR / "replay" / RT.slug(rd) / RT.pool_of(qid) / arm["arm"]
                rid = RR.replay_identity(prim, "openrouter/" + rd)["rid"]
                rrs = []
                f = out / ("runs_%s.jsonl" % prim["approach"].replace("/", "_"))
                for ff in out.glob("runs_*.jsonl"):
                    for line in ff.read_text(encoding="utf-8").splitlines():
                        if line.strip():
                            r = json.loads(line)
                            if ((r.get("meta") or {}).get("identity") or {}).get("rid") == rid:
                                rrs.append(r)
                ok2 = [r for r in rrs if not r.get("error")]
                e2 = [r for r in rrs if r.get("error")]
                c.update(state="ok" if ok2 else ("err%d" % len(e2) if e2 else "todo"), rid=rid,
                         kinds=[kind(r["error"]) for r in e2], last_error=(e2[-1]["error"][:200] if e2 else None))
                cells.append(c)
    return cells


def summary(cells):
    out = {}
    for c in cells:
        s = out.setdefault(c["reader"], {})
        k = c["state"]
        if c["state"].startswith("err"):
            k += "_" + ("infra" if c["kinds"] and all(x in ("http429", "ollama") for x in c["kinds"]) else "other")
        s[k] = s.get(k, 0) + 1
    return out


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    cells = census()
    (RT.TR / "cellstate.json").write_text(json.dumps({"summary": summary(cells), "non_ok": [c for c in cells if c["state"] != "ok"]},
                                                     ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(summary(cells), indent=1))
