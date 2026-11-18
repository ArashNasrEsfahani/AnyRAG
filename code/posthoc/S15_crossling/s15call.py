"""S15 OpenRouter clients and money guards. Every call goes through the frozen OpenRouterReader (pinned provider,
manifest params, usage included) and is logged by the ragbench Ledger to S15_crossling/ledger*.jsonl."""
import json

import requests

import s15lib as L
from ragbench.llm import Ledger
from ragbench.llm_openrouter import OpenRouterReader, model_info

CAP_USD = 0.35
FLOOR_USD = 2.00
READER = "z-ai/glm-5.3-flash"
MAN = json.loads((L.FZ / "MANIFEST.json").read_text(encoding="utf-8"))
JPIN = json.loads((L.FZ / "judge_pin.json").read_text(encoding="utf-8"))


class _NoBudget:
    def reserve(self, model):
        if spent() >= CAP_USD - 0.01:
            from ragbench.llm import BudgetExceeded
            raise BudgetExceeded("S15 cap")

    def settle(self, *a, **k):
        pass


def spent():
    tot = 0.0
    for f in sorted(L.HERE.glob("ledger*.jsonl")):
        for l in open(f, encoding="utf-8"):
            if l.strip():
                try:
                    tot += float(json.loads(l).get("cost_usd") or 0.0)
                except ValueError:
                    pass
    return tot


def limit_remaining():
    key = L.CFG.openrouter_key()
    d = requests.get("https://openrouter.ai/api/v1/key", headers={"Authorization": "Bearer " + key}, timeout=20).json()
    v = d["data"].get("limit_remaining")
    L.append(L.HERE / "floor_checks.jsonl", {"limit_remaining": v, "s15_spent": round(spent(), 6)})
    return float(v) if v is not None else None


def guard():
    """Call before every batch of <= 50 calls. -> True to proceed."""
    rem, sp = limit_remaining(), spent()
    print("guard: key limit_remaining=%s stream spent=$%.4f" % (rem, sp), flush=True)
    if rem is not None and rem < FLOOR_USD:
        print("STOP: key below floor", flush=True)
        return False
    if sp >= CAP_USD - 0.02:
        print("STOP: stream cap", flush=True)
        return False
    return True


def make_client(ledger, model, provider, attempts, **kw):
    info = model_info(model, L.CFG.project_root / "work" / "openrouter" / "catalog.json")
    c = OpenRouterReader(L.CFG, ledger, model, supported=info.get("params"), provider=provider, attempts=attempts, **kw)
    c.budget = _NoBudget()
    return c


def reader(ledger_name="ledger.jsonl"):
    led = Ledger(path=str(L.HERE / ledger_name))
    return make_client(led, READER, MAN["reader_pins"][READER], int(MAN["reader_max_attempts"])), led
