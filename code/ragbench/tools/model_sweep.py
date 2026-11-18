"""The READER sweep: the same retrieval methods, the same prompts, evidence budget and call caps, with the model that
plans and answers swapped for a panel of OpenRouter models. Dev files only (pilot.jsonl 64 records + pilot_timeline.jsonl
8); the test set is never touched (run.py refuses --reader with --final).

    .venv/Scripts/python.exe -m ragbench.tools.model_sweep [--phase 1|2|all] [--only <reader substring>] [--dry-run]

Phase 1 = B1 gold evidence, B2 BM25, LEX2 (the core comparison, ~5 calls a record); phase 2 = AGENTIC3, LEX with
model-written regex, B0 no retrieval. One step at a time, below-normal priority, 2 BLAS threads (the laptop froze
under parallel jobs); every step is a resumable ragbench.run, so a stopped sweep continues where it stopped.
MONEY: work/openrouter/budget_cap.json (else READER_MAX_USD) across both keys - the project key first, then READER_FALLBACK_ENV's key after a 402;
free models stop for the day at READER_FREE_PER_DAY (950). Log: work/openrouter/sweep/steps.tsv + one log per step.
"""
import argparse
import datetime as dt
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PY = str(ROOT / ".venv" / "Scripts" / "python.exe")
LOGS = ROOT / "work" / "openrouter" / "sweep"
PQ = "ragbench/data/questions/pilot.jsonl"
PT = "ragbench/data/questions/pilot_timeline.jsonl"
FALLBACK_ENV = r"D:\KareOne\Holding Dashboard\app\.env"      # the second key with budget (user, 2026-09-23)

# (OpenRouter id, what it adds to the panel)
READERS = [
    ("openai/gpt-oss-20b", "OpenAI open-weight, 21B MoE (3.6B active), reasoning"),
    ("cohere/command-r7b-12-2024", "Cohere, 7B dense, RAG-tuned, no reasoning"),
    ("mistralai/mistral-small-3.2-24b-instruct", "Mistral, 24B dense, no reasoning"),
    ("qwen/qwen3.7-flash", "Alibaba Qwen, flash tier, reasoning"),
    ("deepseek/deepseek-v4-flash", "DeepSeek, large MoE, reasoning"),
    ("meta-llama/llama-4-scout", "Meta, 109B MoE (17B active), no reasoning"),
    ("google/gemma-4-31b-it", "Google, 31B dense - same family as the local 4B reader"),
    ("nvidia/nemotron-3-super-120b-a12b", "NVIDIA, 120B hybrid Mamba-Transformer MoE (12B active)"),
    ("inclusionai/ling-3.0-flash-fin", "Ant Group, finance-tuned"),
    ("liquid/lfm-2.5-2.6b:free", "Liquid, 2.6B hybrid - smaller than the local reader"),
    # GLM 5.2 :free was rate-limited upstream (12 of 21 calls 429, 2026-09-23): replaced by the paid flash tier; its
    # partial free records stay in work/runs/sweep/z-ai__glm-5.2_free but are not part of the panel
    ("z-ai/glm-5.3-flash", "Zhipu GLM, flash tier, reasoning"),
]
PHASE1 = [("b1", ["--approach", "b1"]), ("b2", ["--approach", "b2"]), ("lex2", ["--approach", "lex2"])]
PHASE2 = [("agentic3", ["--approach", "agentic3"]), ("lexllmrx", ["--approach", "lex", "--ablation", "llm_regex"]),
          ("b0", ["--approach", "b0"])]
BLOCKS = [(PQ, "A", "fa-fa,en-en", ""), (PQ, "B", "fa-fa,en-en", ""), (PQ, "C", "fa-fa", ""), (PT, "C", "fa-fa", "_tl")]
BELOW_NORMAL = 0x00004000


def slug(model_id):
    return model_id.replace("/", "__").replace(":", "_")


def budget():
    """The shared budget file, read under the readers' own lock and retried (Windows refuses a file that another lane
    is replacing - lane L5 died on exactly that, 2026-09-23)."""
    from ..llm_openrouter import _retry
    from ..util import FileLock
    p = ROOT / "work" / "openrouter" / "reader_budget.json"
    if not p.exists():
        return {"cost_usd": 0.0, "free_calls": {}}
    with FileLock(str(p) + ".lock"):
        return _retry(lambda: json.loads(p.read_text(encoding="utf-8")))


def max_usd_now():
    from ..llm_openrouter import cap_usd
    return cap_usd(ROOT)


def free_gb():
    if os.name != "nt":
        return 99.0
    import ctypes

    class MS(ctypes.Structure):
        _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
    m = MS()
    m.dwLength = ctypes.sizeof(MS)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
    return m.ullAvailPhys / 1024 ** 3


def wait_for_ram(min_gb):
    while free_gb() < min_gb:
        print("    waiting: %.1f GB free < %.1f GB" % (free_gb(), min_gb), flush=True)
        time.sleep(30)


def snapshot():
    """work/openrouter/sweep/progress.json: records written per reader x method (wide and timeline), rewritten after
    every step by every lane - the saved state at a glance, no scoring."""
    out = {}
    for d in sorted((ROOT / "work" / "runs" / "sweep").glob("*")):
        for f in d.glob("runs_*.jsonl"):
            ok = err = 0
            for line in f.open(encoding="utf-8", errors="ignore"):
                if '"error": null' in line or '"error":null' in line:
                    ok += 1
                else:
                    err += 1
            out["%s/%s" % (d.name, f.stem[5:])] = {"ok": ok, "error_or_other": err}
    tmp = LOGS / ("progress.%d.tmp" % os.getpid())
    try:
        tmp.write_text(json.dumps({"at": dt.datetime.now().isoformat(timespec="seconds"), "budget": budget(),
                                   "files": out}, indent=1), encoding="utf-8")
        os.replace(tmp, LOGS / "progress.json")
    except OSError:
        pass                                   # another lane is writing it this instant; the next step rewrites it


def main(argv=None):
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", default="all", choices=["1", "2", "all"])
    ap.add_argument("--only", default=None)
    ap.add_argument("--readers", default=None, help="comma list of reader ids: this lane's share of the panel")
    ap.add_argument("--lane", default="main", help="lane name: its console, steps and progress files")
    ap.add_argument("--methods", default=None,
                    help="comma list of method keys (b1,b2,lex2,agentic3,lexllmrx,b0): one lane per method is safe - "
                         "each method writes its own runs/ledger/meta/cache files - while two lanes on ONE method of "
                         "one reader would race on its runs file")
    ap.add_argument("--min-free-gb", type=float, default=2.5, dest="min_free_gb",
                    help="wait before a step while free RAM is below this")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    LOGS.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2", MKL_NUM_THREADS="2",
               READER_MAX_USD=os.environ.get("READER_MAX_USD", "5.0"),
               READER_FREE_PER_DAY=os.environ.get("READER_FREE_PER_DAY", "950"),
               READER_FALLBACK_ENV=FALLBACK_ENV, PYTHONIOENCODING="utf-8")
    env.pop("LEX_OPTIONS", None)
    max_usd = float(env["READER_MAX_USD"])
    phases = {"1": [PHASE1], "2": [PHASE2], "all": [PHASE1, PHASE2]}[a.phase]
    readers = [r for r in READERS if not a.only or a.only in r[0]]
    if a.readers:
        want = [x.strip() for x in a.readers.split(",") if x.strip()]
        readers = [r for r in READERS if r[0] in want]
    for methods in phases:
        for mid, _why in readers:
            for mkey, margs in methods:
                if a.methods and mkey not in a.methods.split(","):
                    continue
                for qf, corpus, settings, suffix in BLOCKS:
                    b = budget()
                    max_usd = max_usd_now()                  # budget_cap.json can be raised while lanes run
                    if b["cost_usd"] >= max_usd:
                        print("STOP: $%.4f spent of $%.2f" % (b["cost_usd"], max_usd), flush=True)
                        return 3
                    today = dt.datetime.utcnow().strftime("%Y-%m-%d")
                    if mid.endswith(":free") and b["free_calls"].get(today, 0) >= int(env["READER_FREE_PER_DAY"]):
                        print("skip %s (free quota for %s used)" % (mid, today), flush=True)
                        continue
                    name = "%s__%s__%s%s" % (slug(mid), mkey, corpus, suffix)
                    cmd = [PY, "-m", "ragbench.run"] + margs + [
                        "--reader", "openrouter:" + mid, "--docs-from-questions", "--questions", qf, "--corpus", corpus,
                        "--settings", settings, "--out", "runs/sweep/%s%s" % (slug(mid), suffix), "--no-warmup"]
                    if a.dry_run:
                        print(" ".join(cmd[2:]))
                        continue
                    wait_for_ram(a.min_free_gb)
                    t0 = time.time()
                    print("=== %s %s" % (dt.datetime.now().strftime("%H:%M:%S"), name), flush=True)
                    with open(LOGS / (name + ".log"), "w", encoding="utf-8") as fh:
                        rc = subprocess.run(cmd, cwd=ROOT, env=env, stdout=fh, stderr=subprocess.STDOUT,
                                            creationflags=BELOW_NORMAL if os.name == "nt" else 0).returncode
                    tail = (LOGS / (name + ".log")).read_text(encoding="utf-8", errors="ignore").splitlines()
                    done = next((l for l in reversed(tail) if l.startswith("done:")), tail[-1] if tail else "")
                    with open(LOGS / ("steps_%s.tsv" % a.lane), "a", encoding="utf-8") as fh:
                        fh.write("%s\t%s\t%d\t%ds\t$%.4f\t%s\n" % (dt.datetime.now().isoformat(timespec="seconds"),
                                                                   name, rc, time.time() - t0, budget()["cost_usd"],
                                                                   done[:160]))
                    print("    rc %d, %ds, spent $%.4f | %s" % (rc, time.time() - t0, budget()["cost_usd"], done[:120]),
                          flush=True)
                    snapshot()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
