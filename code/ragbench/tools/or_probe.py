"""One call per candidate reader: a short Persian extraction question that must come back as JSON. Drops models that
are down, rate-limited to death, refuse, or cannot follow the format - before any quota is spent on them.

    .venv/Scripts/python.exe -m ragbench.tools.or_probe [model ids...]

-> work/openrouter/probe.json and a table. Uses the budgeted reader (llm_openrouter), so free calls count against the
day and paid calls against the dollar cap.
"""
import json
import sys
import time

from ..config import load_config
from ..llm import Ledger
from ..llm_openrouter import OpenRouterReader, model_info

FREE = ["thinkingmachines/inkling-small:free", "thinkingmachines/inkling:free", "nvidia/nemotron-3.5-lightning:free",
        "nvidia/nemotron-3-ultra-550b-a55b:free", "dots-studio/dots-3-note-preview:free", "nex-agi/nex-n2.5-mini:free",
        "nex-agi/nex-n2.5-pro:free", "inclusionai/ling-3.0-flash-fin:free", "qwen/qwen3.8-27b:free",
        "poolside/laguna-s-2.1:free", "poolside/laguna-xs-2.1:free", "google/gemma-4-26b-a4b-it:free",
        "google/gemma-4-31b-it:free", "nvidia/nemotron-3-super-120b-a12b:free",
        "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free", "liquid/lfm-2.5-2.6b:free", "z-ai/glm-5.2:free"]
PAID = ["openai/gpt-oss-20b", "qwen/qwen3.7-flash", "deepseek/deepseek-v4-flash"]
# left out on purpose: lyria (music), nemotron content-safety (a classifier), north-mini-code (code only),
# openrouter/free (a router to a random model - not reproducible), ling-3.0-flash-vl / -sante (vision / health variants)

EVIDENCE = ("[1] گزارش سالانه شرکت نمونه ۱۴۰۱ > عملکرد مالی\n"
            "درآمد عملیاتی شرکت در سال ۱۴۰۱ به ۱۲۰ میلیارد ریال رسید که نسبت به سال ۱۴۰۰ (۸۵ میلیارد ریال) "
            "۴۱ درصد رشد داشت. تعداد کارکنان در پایان سال ۳۴۰ نفر بود.")
QUESTION = "درآمد عملیاتی شرکت در سال ۱۴۰۱ چقدر بود و نسبت به سال قبل چند درصد رشد کرد؟"
SCHEMA = {"type": "object", "properties": {"value": {"type": "string"}, "growth_percent": {"type": "string"}},
          "required": ["value", "growth_percent"]}


def main(argv=None):
    sys.stdout.reconfigure(encoding="utf-8")
    cfg = load_config()
    models = (argv if argv else sys.argv[1:]) or FREE + PAID
    cat = cfg.project_root / "work" / "openrouter" / "catalog.json"
    ledger = Ledger(path=str(cfg.project_root / "work" / "openrouter" / "probe_ledger.jsonl"))
    msgs = [{"role": "system", "content": "Answer only from the evidence. Reply with a JSON object "
                                          "{\"value\": ..., \"growth_percent\": ...} and nothing else."},
            {"role": "user", "content": EVIDENCE + "\n\nQuestion: " + QUESTION}]
    out = []
    for mid in models:
        info = model_info(mid, cat)
        rd = OpenRouterReader(cfg, ledger, mid, supported=info.get("params"), retries=2)
        t0 = time.perf_counter()
        row = {"model": mid, "price_in": info.get("in_per_m"), "price_out": info.get("out_per_m")}
        try:
            with ledger.scope(phase="query", approach="probe"):
                r = rd.chat(msgs, purpose="probe", num_predict=120, format=SCHEMA)
            j = r.json({}) or {}
            val, gr = str(j.get("value", "")), str(j.get("growth_percent", ""))
            ok = ("120" in val or "۱۲۰" in val) and ("41" in gr or "۴۱" in gr)
            row.update(ok=ok, json=bool(j), seconds=round(time.perf_counter() - t0, 1), ptok=r.prompt_tokens,
                       ctok=r.completion_tokens, cost=r.cost_usd, finish=r.done_reason, text=r.text[:160])
        except Exception as e:  # noqa: BLE001
            row.update(ok=False, error="%s: %s" % (type(e).__name__, str(e)[:160]),
                       seconds=round(time.perf_counter() - t0, 1))
        out.append(row)
        print("%-52s %-5s %5.1fs  %s" % (mid, "OK" if row.get("ok") else ("json" if row.get("json") else "FAIL"),
                                          row["seconds"], row.get("error") or row.get("text", "").replace("\n", " ")[:90]),
              flush=True)
    p = cfg.project_root / "work" / "openrouter" / "probe.json"
    prev = json.loads(p.read_text(encoding="utf-8")) if p.exists() else []
    keep = {r["model"]: r for r in prev}
    keep.update({r["model"]: r for r in out})
    p.write_text(json.dumps(list(keep.values()), indent=1, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
