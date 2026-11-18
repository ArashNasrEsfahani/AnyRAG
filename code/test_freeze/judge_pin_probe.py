"""FREEZE fix (a): connectivity probe of the pinned judge endpoint. NON-dev/test text only (a synthetic question about
a fictional company), judge-final-v3 prompt shape, <= 2 calls, hard stop at $0.01. No dev or test record is judged.

Pin rule (declared before the probe): judge model openai/gpt-oss-120b (unchanged); endpoint = deepinfra/bf16 -
the endpoint that served the most dev60 judge calls (34 of 175) with no failed call, bf16 (not fp4), supports
temperature, top_p, seed, reasoning and response_format, cheapest bf16 endpoint ($0.037/M in, $0.17/M out on
2026-10-01). Kept if the probe's replies parse and OpenRouter reports the served provider as DeepInfra.
-> judge_pin.json, judge_pin_probe_ledger.jsonl
"""
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, r"D:\KareOne\RAG-Paper-upgrades")
from ragbench.config import load_config  # noqa: E402
from ragbench.llm import Ledger  # noqa: E402
from ragbench.llm_openrouter import OpenRouterReader, model_info  # noqa: E402
from ragbench.tools import judge_final as JF  # noqa: E402
from ragbench.tools import judge_multi as JM  # noqa: E402

JUDGE, PIN = "openai/gpt-oss-120b", "deepinfra/bf16"
CASES = [("Northwind Kettles (fictional) sold 4,200 kettles in 2031 and 5,100 in 2032. How many did it sell in 2032?",
          "5,100 kettles", "It sold 5,100 kettles in 2032.", True),
         ("Northwind Kettles (fictional) sold 4,200 kettles in 2031 and 5,100 in 2032. How many did it sell in 2031?",
          "4,200 kettles", "It sold 5,100 kettles in 2031.", False)]


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    cfg = load_config()
    budget_file = Path(cfg.project_root) / "work" / "openrouter" / "reader_budget.json"
    before = json.loads(budget_file.read_text(encoding="utf-8-sig"))["cost_usd"]
    ledger = Ledger(path=str(HERE / "judge_pin_probe_ledger.jsonl"))
    info = model_info(JUDGE, Path(cfg.project_root) / "work" / "openrouter" / "catalog.json")
    client = OpenRouterReader(cfg, ledger, JUDGE, supported=info.get("params"), provider=PIN, attempts=2)
    rows = []
    for q, gold, ans, want in CASES:
        spent = json.loads(budget_file.read_text(encoding="utf-8-sig"))["cost_usd"] - before
        if spent >= 0.01:
            break
        user = JF.JUDGE_V2_USER.format(lang="English", question=q, answer_type="short", gold=gold, gold_value="",
                                       key_facts="", answer=ans)
        msgs = [{"role": "system", "content": JF.JUDGE_V2_SYSTEM}, {"role": "user", "content": user}]
        with ledger.scope(phase="smoke", approach="judge-pin-probe", question_id="SYNTHETIC", setting="en-en"):
            res = client.chat(msgs, purpose="judge", num_predict=200)
        c, reason = JF.parse(res.text)
        rows.append({"expected": want, "parsed": c, "reason": reason, "provider": res.provider,
                     "provider_match": JM._served_matches(PIN, res.provider), "prompt_tokens": res.prompt_tokens,
                     "completion_tokens": res.completion_tokens, "cost_usd": res.cost_usd,
                     "wall_ms": round(res.wall_ms)})
    after = json.loads(budget_file.read_text(encoding="utf-8-sig"))["cost_usd"]
    ok = bool(rows) and all(r["parsed"] is not None and r["provider_match"] for r in rows)
    out = {"judge": JUDGE, "provider_pin": PIN,
           "routing": {"order": [PIN], "allow_fallbacks": False, "require_parameters": True},
           "sent_params": sorted(k for k in ("temperature", "top_p", "seed") if not client.supported
                                 or k in client.supported),
           "generation": {k: cfg.generation.get(k) for k in ("temperature", "top_p", "seed")},
           "probe_calls": rows, "probe_cost_usd": round(after - before, 6), "pin_confirmed": ok,
           "note": "synthetic fictional-company text only; no dev/test question or answer was sent"}
    (HERE / "judge_pin.json").write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({k: out[k] for k in ("pin_confirmed", "probe_cost_usd")}),
          [(r["provider"], r["parsed"], r["expected"]) for r in rows])
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
