"""STEP 0: OpenRouter account balance (shared account) + live prices of the 6 readers, the selector and the judge.
Prints numbers only; the key is read from READER_FALLBACK_ENV and never printed."""
import json, os, sys, datetime as dt
import requests
from pathlib import Path
HERE = Path(__file__).resolve().parent

def key():
    for line in open(os.environ["READER_FALLBACK_ENV"], encoding="utf-8", errors="ignore"):
        k, _, v = line.partition("=")
        if k.strip() == "OPENROUTER_API_KEY":
            return v.strip().strip('"').strip("'")

H = {"Authorization": "Bearer " + key()}
out = {"at": dt.datetime.now().isoformat(timespec="seconds")}
c = requests.get("https://openrouter.ai/api/v1/credits", headers=H, timeout=30)
out["credits_http"] = c.status_code
if c.ok:
    d = c.json().get("data", {})
    out["total_credits"] = d.get("total_credits"); out["total_usage"] = d.get("total_usage")
    out["remaining"] = round(float(d.get("total_credits") or 0) - float(d.get("total_usage") or 0), 4)
k = requests.get("https://openrouter.ai/api/v1/key", headers=H, timeout=30)
out["key_http"] = k.status_code
if k.ok:
    d = k.json().get("data", {})
    out["key"] = {x: d.get(x) for x in ("limit", "limit_remaining", "usage", "usage_daily", "is_free_tier")}
models = ["z-ai/glm-5.3-flash", "openai/gpt-oss-20b", "qwen/qwen3-235b-a22b-2507", "meta-llama/llama-3.3-70b-instruct",
          "google/gemma-3-27b-it", "deepseek/deepseek-v4-flash", "openai/gpt-oss-120b"]
out["endpoints"] = {}
for m in models:
    r = requests.get("https://openrouter.ai/api/v1/models/%s/endpoints" % m, headers=H, timeout=30)
    if not r.ok:
        out["endpoints"][m] = {"http": r.status_code}; continue
    eps = r.json()["data"]["endpoints"]
    out["endpoints"][m] = [{"tag": e.get("tag"), "provider": e.get("provider_name"), "quant": e.get("quantization"),
                            "in_per_M": round(float(e["pricing"]["prompt"]) * 1e6, 4),
                            "out_per_M": round(float(e["pricing"]["completion"]) * 1e6, 4),
                            "ctx": e.get("context_length"), "status": e.get("status"),
                            "uptime": e.get("uptime_last_30m"), "params": e.get("supported_parameters")} for e in eps]
(HERE / "balance_check.json").write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
print(json.dumps({k_: v for k_, v in out.items() if k_ != "endpoints"}))
for m, eps in out["endpoints"].items():
    if isinstance(eps, dict):
        print(m, eps); continue
    print(m, [(e["tag"], e["quant"], e["in_per_M"], e["out_per_M"], e["ctx"]) for e in eps])
