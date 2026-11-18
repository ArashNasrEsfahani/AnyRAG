"""Project research journal: one JSON line per event, for the paper's methods and cost sections.

Every run start/stop, interruption, resume, build, question-authoring pass or decision gets an event with a
timestamp and, where known, a duration and cost. Model-call detail lives in the per-run ledgers; this journal
is the project-level timeline that ties them together.

    python -m ragbench.tools.journal add --kind run_start --name dev_index --note "resume after stop"
    python -m ragbench.tools.journal add --kind milestone --name test_set_final --duration-s 5400 --data '{"questions": 316}'
    python -m ragbench.tools.journal list [--kind run_stop]
"""
import argparse
import datetime as dt
import json
import os
import platform
import socket
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
JOURNAL = ROOT / "work" / "journal" / "events.jsonl"

KINDS = ("run_start", "run_stop", "interruption", "resume", "build", "milestone", "authoring",
         "verification", "decision", "environment", "note")


def add(kind, name, note=None, duration_s=None, data=None, ts=None):
    JOURNAL.parent.mkdir(parents=True, exist_ok=True)
    event = {"ts": ts or dt.datetime.now().isoformat(timespec="seconds"), "kind": kind, "name": name}
    if duration_s is not None:
        event["duration_s"] = round(float(duration_s), 1)
    if note:
        event["note"] = note
    if data:
        event["data"] = data
    event["host"] = socket.gethostname()
    event["pid"] = os.getpid()
    line = json.dumps(event, ensure_ascii=False)
    with open(JOURNAL, "a", encoding="utf-8", newline="\n") as f:
        f.write(line + "\n")
        f.flush()
        os.fsync(f.fileno())
    return event


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("add")
    a.add_argument("--kind", required=True, choices=KINDS)
    a.add_argument("--name", required=True)
    a.add_argument("--note")
    a.add_argument("--duration-s", type=float)
    a.add_argument("--data", help="JSON object")
    a.add_argument("--ts", help="ISO timestamp (for backfilled events)")
    l = sub.add_parser("list")
    l.add_argument("--kind")
    args = ap.parse_args()
    if args.cmd == "add":
        data = json.loads(args.data) if args.data else None
        if args.kind == "environment" and data is None:
            data = {"python": platform.python_version(), "platform": platform.platform()}
        print(json.dumps(add(args.kind, args.name, args.note, args.duration_s, data, args.ts), ensure_ascii=False))
    else:
        if not JOURNAL.exists():
            return
        for line in JOURNAL.read_text(encoding="utf-8").splitlines():
            e = json.loads(line)
            if not args.kind or e["kind"] == args.kind:
                print(line)


if __name__ == "__main__":
    main()
