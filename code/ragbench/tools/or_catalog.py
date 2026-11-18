"""OpenRouter: the key's budget and the live model catalogue (prices), without ever printing the key.

    .venv/Scripts/python.exe -m ragbench.tools.or_catalog [--env .env]

Writes work/openrouter/catalog.json (every model: id, prices per M tokens, context, modality) and prints the key's
limit / usage / free-tier status and the free text models.
"""
import argparse
import json
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "work" / "openrouter"


def read_key(env_path):
    for line in Path(env_path).read_text(encoding="utf-8").splitlines():
        k, _, v = line.partition("=")
        if k.strip() == "OPENROUTER_API_KEY":
            return v.strip().strip('"').strip("'")
    raise SystemExit("no OPENROUTER_API_KEY in %s" % env_path)


def get(url, key=None):
    req = urllib.request.Request(url, headers={"Authorization": "Bearer %s" % key} if key else {})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read().decode("utf-8"))


def main(argv=None):
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default=str(ROOT / ".env"))
    a = ap.parse_args(argv)
    key = read_key(a.env)
    info = get("https://openrouter.ai/api/v1/key", key).get("data", {})
    safe = {k: v for k, v in info.items() if k not in ("label",) or not str(v).startswith("sk-")}
    print("key ...%s:" % key[-4:], json.dumps(safe, indent=1))
    try:
        credits = get("https://openrouter.ai/api/v1/credits", key).get("data", {})
        print("account credits:", json.dumps(credits))
    except Exception as e:  # noqa: BLE001
        print("credits endpoint:", type(e).__name__)
    models = get("https://openrouter.ai/api/v1/models").get("data", [])
    rows = []
    for m in models:
        p = m.get("pricing") or {}
        arch = m.get("architecture") or {}
        rows.append({"id": m["id"], "name": m.get("name"), "ctx": m.get("context_length"),
                     "in_per_m": float(p.get("prompt") or 0) * 1e6, "out_per_m": float(p.get("completion") or 0) * 1e6,
                     "input": arch.get("input_modalities"), "output": arch.get("output_modalities"),
                     "params": m.get("supported_parameters")})
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "catalog.json").write_text(json.dumps(rows, indent=1, ensure_ascii=False), encoding="utf-8")
    free = [r for r in rows if r["in_per_m"] == 0 and r["out_per_m"] == 0 and "text" in (r["output"] or ["text"])]
    print("\n%d models, %d free text models:" % (len(rows), len(free)))
    for r in sorted(free, key=lambda r: -(r["ctx"] or 0)):
        print("  %-60s ctx %7s" % (r["id"], r["ctx"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
