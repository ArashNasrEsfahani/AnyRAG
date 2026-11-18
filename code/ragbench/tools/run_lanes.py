"""Run a saved plan of ragbench.run steps in parallel LANES, every step logged; rerunning the same plan resumes it.

    .venv/Scripts/python.exe -m ragbench.tools.run_lanes <plan.json> [--dry-run]

plan.json: {"name": ..., "env": {...}, "lanes": {"<lane>": [[ragbench.run args...], ...], ...}}
Each lane runs its steps in order in its own process tree (below-normal priority, 2 BLAS threads); lanes run side by
side. Two lanes must never share one (method, out dir) pair - each method writes its own files, and that is what makes
lanes safe. Saved as it goes: every answered question by ragbench.run itself (append-only records, a stopped step
loses at most the question in flight), plus per step a log under <plan dir>/logs/ and a line in <plan dir>/steps.tsv.
A step already complete (the runner reports 0 still missing) costs only its startup when the plan is rerun.
"""
import argparse
import datetime as dt
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PY = str(ROOT / ".venv" / "Scripts" / "python.exe")
BELOW_NORMAL = 0x00004000
_LOCK = threading.Lock()


def lane(name, steps, env, logdir, tsv, results):
    for i, args in enumerate(steps):
        step = "%s_%02d" % (name, i)
        t0 = time.time()
        with _LOCK:
            print("=== %s %s: %s" % (dt.datetime.now().strftime("%H:%M:%S"), step, " ".join(args)), flush=True)
        with open(logdir / (step + ".log"), "w", encoding="utf-8") as fh:
            rc = subprocess.run([PY, "-m", "ragbench.run"] + args, cwd=ROOT, env=env, stdout=fh,
                                stderr=subprocess.STDOUT,
                                creationflags=BELOW_NORMAL if os.name == "nt" else 0).returncode
        tail = (logdir / (step + ".log")).read_text(encoding="utf-8", errors="ignore").splitlines()
        done = next((l for l in reversed(tail) if l.startswith("done:")), tail[-1] if tail else "")
        with _LOCK:
            with open(tsv, "a", encoding="utf-8") as fh:
                fh.write("%s\t%s\t%d\t%ds\t%s\t%s\n" % (dt.datetime.now().isoformat(timespec="seconds"), step, rc,
                                                        time.time() - t0, done[:160], " ".join(args)))
            print("    %s rc %d, %ds | %s" % (step, rc, time.time() - t0, done[:120]), flush=True)
        results.append((step, rc))


def main(argv=None):
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("plan")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    plan_path = Path(a.plan)
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    logdir = plan_path.parent / "logs"
    logdir.mkdir(parents=True, exist_ok=True)
    tsv = plan_path.parent / "steps.tsv"
    env = dict(os.environ, OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2", MKL_NUM_THREADS="2", PYTHONIOENCODING="utf-8")
    env.update({k: str(v) for k, v in (plan.get("env") or {}).items()})
    if a.dry_run:
        for k, steps in plan["lanes"].items():
            for s in steps:
                print(k, " ".join(s))
        return 0
    results, threads = [], []
    for k, steps in plan["lanes"].items():
        t = threading.Thread(target=lane, args=(k, steps, env, logdir, tsv, results), daemon=False)
        t.start()
        threads.append(t)
    for t in threads:
        t.join()
    bad = [s for s, rc in results if rc]
    print("plan %s finished: %d steps, %d with rc != 0 %s" % (plan.get("name"), len(results), len(bad), bad))
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
