"""Sample GPU telemetry to CSV while experiments run (latency depends on thermal/power throttling).

Columns: timestamp, utilization %, memory used MiB, temperature C, power W, power limit W,
throttle flags (sw power cap, hw slowdown, thermal slowdown), and pstate.

    python -m ragbench.tools.gpu_monitor --out work/telemetry/gpu_dev_index.csv --interval 30 [--until-pid 1234]

Stops when --until-pid exits (if given), after --max-hours, or on Ctrl+C. Each row is flushed immediately.
"""
import argparse
import csv
import datetime as dt
import os
import subprocess
import time
from pathlib import Path

FIELDS = ["utilization.gpu", "memory.used", "temperature.gpu", "power.draw", "power.limit",
          "clocks_event_reasons.sw_power_cap", "clocks_event_reasons.hw_slowdown",
          "clocks_event_reasons.sw_thermal_slowdown", "pstate"]
HEADER = ["ts", "util_pct", "mem_used_mib", "temp_c", "power_w", "power_limit_w",
          "sw_power_cap", "hw_slowdown", "sw_thermal_slowdown", "pstate"]


def pid_alive(pid):
    if pid is None:
        return True
    try:
        out = subprocess.run(["tasklist", "/FI", "PID eq %d" % pid, "/NH"], capture_output=True, text=True, timeout=15)
        return str(pid) in out.stdout
    except Exception:
        try:
            os.kill(pid, 0)
            return True
        except OSError:
            return False


def sample():
    out = subprocess.run(["nvidia-smi", "--query-gpu=" + ",".join(FIELDS), "--format=csv,noheader,nounits"],
                         capture_output=True, text=True, timeout=20)
    vals = [v.strip() for v in out.stdout.strip().splitlines()[0].split(",")]
    return [dt.datetime.now().isoformat(timespec="seconds")] + vals


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--interval", type=float, default=30)
    ap.add_argument("--until-pid", type=int)
    ap.add_argument("--max-hours", type=float, default=48)
    args = ap.parse_args()
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    deadline = time.time() + args.max_hours * 3600
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(HEADER)
        try:
            while time.time() < deadline and pid_alive(args.until_pid):
                try:
                    w.writerow(sample())
                    f.flush()
                except Exception as e:  # keep sampling through transient nvidia-smi failures
                    w.writerow([dt.datetime.now().isoformat(timespec="seconds"), "error: %s" % e])
                    f.flush()
                time.sleep(args.interval)
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
