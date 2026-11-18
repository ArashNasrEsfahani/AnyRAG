"""STEP 2: reader format smoke on 3 DEV questions (never test) for the 5 new readers.

    python reader_smoke.py            # resumable; writes smoke/ and smoke/SMOKE.json

1. HYBRID@6000 on the 3 dev questions with reader gpt-oss-20b, RAGBENCH_PACK_DIR set (exercises the direct path and
   writes the packs); 2. each pack replayed (tools/reader_replay.py, the test's own path) to qwen3-235b-2507,
   llama-3.3-70b, gemma-3-27b, deepseek-v4-flash; 3. answer-contract check on every answer: non-empty, finish not
   'length', no reasoning leak (<think>), the abstain token alone when abstaining, answer language = question language.
"""
import json
import os
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DS = HERE.parent
WT = r"D:\KareOne\RAG-Paper-upgrades"
PY = r"D:\KareOne\RAG Paper\.venv\Scripts\python.exe"
SM = HERE / "smoke"
QIDS = ["V2-dev-BOOK-fa-T1-003", "V2-dev-LAW-en-T2-002", "V2-dev-BOOK-en-T3-002"]
POOLS = {"RPT-fa": ("C", "fa-fa"), "RPT-en": ("D", "en-en"), "BOOK-fa": ("K", "fa-fa"), "BOOK-en": ("K", "en-en"),
         "LAW-fa": ("L", "fa-fa"), "LAW-en": ("L", "en-en")}
DIRECT = "openai/gpt-oss-20b"
REPLAY = ["qwen/qwen3-235b-a22b-2507", "meta-llama/llama-3.3-70b-instruct", "google/gemma-3-27b-it",
          "deepseek/deepseek-v4-flash"]
PINS = json.loads((DS / "test_freeze" / "reader_pins.json").read_text(encoding="utf-8"))["pins"]
ENV = dict(os.environ, PYTHONIOENCODING="utf-8", OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2", MKL_NUM_THREADS="2",
           OPENROUTER_PROVIDER_PINS=json.dumps(PINS), READER_MAX_ATTEMPTS="7",
           READER_FALLBACK_ENV=r"D:\KareOne\Holding Dashboard\app\.env", RAGBENCH_PACK_DIR=str(SM / "packs"),
           HF_HOME=r"D:\KareOne\RAG Paper\work\models\hf", HF_HUB_OFFLINE="1")
FA = re.compile(r"[\u0600-\u06FF]")


def pool_of(q):
    return "-".join(q.split("-")[2:4])


def slug(m):
    return m.replace("/", "__")


def recs(d):
    out = []
    for f in Path(d).glob("runs_*.jsonl"):
        out += [json.loads(l) for l in open(f, encoding="utf-8") if l.strip()]
    return out


def contract(rec, done_reason, abstain="NOT_FOUND"):
    a = (rec.get("answer_text") or "").strip()
    probs = []
    if rec.get("error"):
        probs.append("error")
    if not a:
        probs.append("empty")
    if done_reason == "length":
        probs.append("truncated")
    if "<think>" in a or "</think>" in a:
        probs.append("reasoning_leak")
    if abstain in a and a != abstain and len(a) < 3 * len(abstain):
        probs.append("abstain_with_text")
    if a and a != abstain:
        fa = bool(FA.search(a))
        if rec["q_lang"] == "fa" and not fa:
            probs.append("wrong_language(en for fa)")
        if rec["q_lang"] == "en" and len(FA.findall(a)) > 0.3 * len(a):
            probs.append("wrong_language(fa for en)")
    return probs


def main():
    sys.path.insert(0, WT)
    os.environ.update({k: v for k, v in ENV.items() if k in ("OPENROUTER_PROVIDER_PINS", "READER_FALLBACK_ENV")})
    from ragbench.config import load_config
    from ragbench.tools import reader_replay as RR
    cfg = load_config()
    abstain = cfg.evidence["abstain_token"]
    rows = []
    for q in QIDS:
        pool = pool_of(q)
        corpus, setting = POOLS[pool]
        out = SM / "runs" / pool / slug(DIRECT)
        if not any(r["question_id"] == q and not r.get("error") for r in recs(out)):
            out.mkdir(parents=True, exist_ok=True)
            cmd = [PY, "-m", "ragbench.run", "--approach", "hybrid", "--questions",
                   str(DS / "devrun" / "questions" / ("dev_v2run_%s.jsonl" % pool)), "--corpus", corpus,
                   "--settings", setting, "--pool", pool, "--reader", "openrouter:" + DIRECT,
                   "--experiment", "reader-smoke-dev", "--ids", q, "--out", str(out)]
            with open(out / "run.log", "a", encoding="utf-8") as fh:
                subprocess.run(cmd, cwd=WT, env=ENV, stdout=fh, stderr=subprocess.STDOUT)
        prim = [r for r in recs(out) if r["question_id"] == q and not r.get("error")][-1]
        led = [json.loads(l) for f in out.glob("ledger_*.jsonl") for l in open(f, encoding="utf-8")]
        call = [c for c in led if c.get("question_id") == q and c.get("purpose") == "answer" and c.get("ok")][-1]
        rows.append({"q": q, "reader": DIRECT, "path": "direct", "answer": prim["answer_text"][:300],
                     "done_reason": call.get("done_reason"), "provider": call.get("provider"),
                     "latency_ms": call.get("wall_total_ms"), "ptok": call.get("prompt_tokens"),
                     "ctok": call.get("completion_tokens"), "cost": call.get("cost_usd"),
                     "contract": contract(prim, call.get("done_reason"), abstain)})
        pack = SM / "packs" / "HYBRID" / ("%s__%s.json" % (q, setting))
        for m in REPLAY:
            rout = SM / "replay" / slug(m) / pool
            have = [r for r in recs(rout) if r["question_id"] == q and not r.get("error")]
            rec = have[-1] if have else RR.replay_one(cfg, pack, prim, "openrouter:" + m, rout, attempts=7)
            rp = rec["trace"]["replay"]
            rows.append({"q": q, "reader": m, "path": "replay", "answer": rec["answer_text"][:300],
                         "done_reason": rp["done_reason"], "provider": rp["provider"], "latency_ms": rec["latency_ms"],
                         "ptok": rec["prompt_tokens"], "ctok": rec["completion_tokens"], "cost": rp["cost_usd"],
                         "error": rec.get("error"), "contract": contract(rec, rp["done_reason"], abstain)})
    summ = {}
    for r in rows:
        s = summ.setdefault(r["reader"], {"n": 0, "ok": 0, "cost": 0.0, "lat_ms": [], "ptok": 0, "ctok": 0,
                                          "providers": set()})
        s["n"] += 1
        s["ok"] += not r["contract"]
        s["cost"] += float(r["cost"] or 0)
        s["lat_ms"].append(r["latency_ms"])
        s["ptok"] += r["ptok"] or 0
        s["ctok"] += r["ctok"] or 0
        s["providers"].add(str(r["provider"]))
    for s in summ.values():
        s["providers"] = sorted(s["providers"])
        s["cost_cents"] = round(100 * s.pop("cost"), 3)
    (SM / "SMOKE.json").write_text(json.dumps({"rows": rows, "summary": summ}, ensure_ascii=False, indent=1) + "\n",
                                   encoding="utf-8")
    for m, s in summ.items():
        print(m, json.dumps(s))
    for r in rows:
        if r["contract"]:
            print("CONTRACT FAIL", r["q"], r["reader"], r["contract"], repr(r["answer"][:120]))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
