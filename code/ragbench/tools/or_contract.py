"""Choose each remote reader's structured-output contract by a live check with the benchmark's own prompts and schemas.

    .venv/Scripts/python.exe -m ragbench.tools.or_contract <model id> [...]

For each reader, in order json_schema -> json_object -> prompt: LEX's keyword call on one Persian and one English dev
question and LEX's regex call on the English one; the first contract under which all three replies parse into a
usable object (keywords present; regex shape object/array with at least one pattern) is kept. Result appended to
work/openrouter/contracts.json ({model: {contract, checked_at, details}}) - the reader client (llm_openrouter v2)
reads it. Second opinion 1 (2026-09-23): the sweep had given remote readers no structured-output contract while the
local reader had schema enforcement, and parse failures then measured fallbacks, not methods.
"""
import datetime as dt
import json
import sys
from pathlib import Path

from ..approaches.lex import prompts as P
from ..approaches.lex.defaults import DEFAULTS
from ..config import load_config
from ..llm import Ledger
from ..llm_openrouter import OpenRouterReader, model_info
from ..util import atomic_write_json, read_json

ROOT = Path(__file__).resolve().parents[2]
QFA = "بر اساس گزارش سالانهٔ ۱۴۰۱ دیجی‌کالا، مرکز تماس روزانه به چند تماس پاسخ داده است؟"
QEN = "According to Digikala's 1401 annual report, how many phone calls did the call centre answer per day?"
VOCAB = [{"text": "call centre", "feedback": 0}, {"text": "customer satisfaction", "feedback": 0},
         {"text": "daily calls answered", "feedback": 0}]


def check(cfg, ledger, mid, contract):
    info = model_info(mid, cfg.project_root / "work" / "openrouter" / "catalog.json")
    rd = OpenRouterReader(cfg, ledger, mid, supported=info.get("params"), retries=2)
    rd._contracts = {mid: {"contract": contract}}
    o = dict(DEFAULTS)
    out = {}
    with ledger.scope(phase="query", approach="contract-check"):
        for tag, q, lang in (("kw_fa", QFA, "fa"), ("kw_en", QEN, "en")):
            r = rd.chat(P.kw_messages(q, lang, o), purpose="contract", format=P.KW_SCHEMA,
                        num_predict=int(o["kw_num_predict"]))
            kw, how = P.parse_keywords(r.text, o)
            out[tag] = {"ok": bool(kw["keywords"]) and how != "failed", "how": how, "finish": r.done_reason,
                        "text": r.text[:160]}
        r = rd.chat(P.regex_messages(QEN, VOCAB, ["call centre", "calls per day"], "en", o), purpose="contract",
                    format=P.REGEX_SCHEMA, num_predict=int(o["regex_num_predict"]))
        got, how = P.parse_regex(r.text, o)
        out["regex"] = {"ok": bool(got["regex"]) and got["shape"] in ("object", "array", "alias", "object_string"),
                        "shape": got["shape"], "finish": r.done_reason, "text": r.text[:160]}
    return all(v["ok"] for v in out.values()), out


def main(argv=None):
    sys.stdout.reconfigure(encoding="utf-8")
    cfg = load_config()
    models = argv if argv else sys.argv[1:]
    path = cfg.project_root / "work" / "openrouter" / "contracts.json"
    ledger = Ledger(path=str(cfg.project_root / "work" / "openrouter" / "contract_ledger.jsonl"))
    for mid in models:
        chosen, tried = "prompt", {}
        for c in ("json_schema", "json_object", "prompt"):
            try:
                ok, det = check(cfg, ledger, mid, c)
            except Exception as e:  # noqa: BLE001
                ok, det = False, {"error": "%s: %s" % (type(e).__name__, str(e)[:160])}
            tried[c] = {"ok": ok, **det}
            if ok:
                chosen = c
                break
        data = read_json(path, {}) or {}
        data[mid] = {"contract": chosen, "checked_at": dt.datetime.now().isoformat(timespec="seconds"),
                     "all_failed": not any(v.get("ok") for v in tried.values()), "tried": tried}
        atomic_write_json(path, data, indent=1)                      # saved per reader
        print("%-45s -> %-12s %s" % (mid, chosen, {c: v.get("ok") for c, v in tried.items()}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
