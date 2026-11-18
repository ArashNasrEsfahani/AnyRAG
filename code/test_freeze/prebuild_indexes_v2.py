"""(2026-10-01 re-freeze) Pre-build the SETR / SETR_15500 pointer indexes (copy of prebuild_indexes.py, arms changed). under the frozen code (no model call, no answer).

Why: the bundles parser version (lex5-bundles-v1.1) is part of the LEX5B config signature, so the index fingerprint
changed and the old pointer dirs (built 2026-09-26) no longer match. These indexes are only a hybrid_ref.json pointer
to the shared work/indexes/_hybrid index (0 LLM / 0 embed calls). They are built here with `ragbench.run --index-only`
over the DEV question files (the test files are never touched), one at a time, below-normal priority, before the
index trees are hashed into MANIFEST.json.
"""
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DS = HERE.parent
PY = r"D:\KareOne\RAG Paper\.venv\Scripts\python.exe"
WT = r"D:\KareOne\RAG-Paper-upgrades"
OUT = HERE / "_index_prebuild"
POOLS = {"RPT-fa": ("C", "fa-fa"), "RPT-en": ("D", "en-en"), "BOOK-fa": ("K", "fa-fa"), "BOOK-en": ("K", "en-en"),
         "LAW-fa": ("L", "fa-fa"), "LAW-en": ("L", "en-en")}
ARMS = [("SETR", ["--approach", "setr"]),
        ("SETR_15500", ["--approach-class", "ragbench.approaches.budget_matrix:Setr15500"])]
ENV = {"PYTHONIOENCODING": "utf-8", "HF_HOME": r"D:\KareOne\RAG Paper\work\models\hf", "HF_HUB_OFFLINE": "1",
       "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2",
       "LEX8_IDX_DIR": str(DS / "lex8" / "idx"), "LEX8_SYNTH_DIR": str(DS / "lex8" / "synth")}

rc_all = 0
for arm, args in ARMS:
    for pid, (corpus, setting) in POOLS.items():
        out = OUT / arm / pid
        out.mkdir(parents=True, exist_ok=True)
        cmd = [PY, "-m", "ragbench.run"] + args + [
            "--questions", str(DS / "devrun" / "questions" / ("dev_v2run_%s.jsonl" % pid)), "--corpus", corpus,
            "--settings", setting, "--pool", pid, "--reader", "openrouter:z-ai/glm-5.3-flash",
            "--experiment", "index-prebuild", "--out", str(out), "--index-only"]
        with open(out / "run.log", "w", encoding="utf-8") as fh:
            rc = subprocess.run(cmd, cwd=WT, env=dict(os.environ, **ENV), stdout=fh, stderr=subprocess.STDOUT,
                                creationflags=0x00004000 if os.name == "nt" else 0).returncode
        ready = [l for l in (out / "run.log").read_text(encoding="utf-8", errors="ignore").splitlines()
                 if "ready in" in l or "refus" in l.lower()]
        print(arm, pid, "rc=%d" % rc, ready[-1][:200] if ready else "", flush=True)
        rc_all |= rc
sys.exit(rc_all)
