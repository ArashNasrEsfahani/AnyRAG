"""S11: judge runs with the frozen pinned judge (judge_multi, exactly run_test.py's arguments), in batches of at most
45 model calls with the global-floor key check before each batch.  python judge.py <run dir> <dev|test>"""
import json, os, subprocess, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from s11lib import _key as key, key_remaining, spent, STOP_USD
HERE = Path(__file__).resolve().parent
DS = HERE.parent.parent / "datasets_v2"
PY = r"D:\KareOne\RAG Paper\.venv\Scripts\python.exe"
J = json.loads((DS / "test_freeze" / "MANIFEST.json").read_text(encoding="utf-8"))["judge"]
QF = {"dev": DS / "devrun" / "questions" / "dev_v2judge_all.jsonl",
      "test": DS / "test_freeze" / "questions" / "test_v2judge_all.jsonl"}
run_dir, split = Path(sys.argv[1]).resolve(), sys.argv[2]
out = HERE / "judge" / run_dir.parent.name / run_dir.name
out.mkdir(parents=True, exist_ok=True)
env = dict(os.environ, PYTHONIOENCODING="utf-8", OMP_NUM_THREADS="2", CUDA_VISIBLE_DEVICES="", OPENROUTER_API_KEY=key())
env.pop("READER_FALLBACK_ENV", None); env.pop("OPENROUTER_PROVIDER_PINS", None)
for it in range(20):
    rem = key_remaining()
    if spent() >= STOP_USD:
        print("STREAM CAP", spent()); sys.exit(1)
    if rem is not None and rem < 2.0:
        print("GLOBAL FLOOR", rem); sys.exit(1)
    args = [PY, "-m", "ragbench.tools.judge_multi", "--judge", J["model"], "--judge-provider", J["provider_pin"],
            "--require-pin", "--max-judge-attempts", str(J["max_judge_attempts"]), "--on-malformed", J["on_malformed"],
            "--http-attempts", str(J["http_attempts"]), "--questions", str(QF[split]), "--out", str(out),
            "--runs", str(run_dir), "--max-model-calls", "45"]
    r = subprocess.run(args, cwd=r"D:\KareOne\RAG-Paper-upgrades", env=env, capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    line = [l for l in (r.stdout + r.stderr).splitlines() if "judge_multi:" in l or "BUDGET" in l or "Error" in l]
    print(r.returncode, line[-1][:400] if line else (r.stdout + r.stderr)[-800:])
    if "stopped_at_max_model_calls" not in (r.stdout + r.stderr):
        break
