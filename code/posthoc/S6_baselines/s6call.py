"""S6 model clients: the frozen OpenRouterReader (same payload rule, pins, params) with the shared cross-stream budget
file replaced by a stream-local one (S6 cap), every call logged to S6_baselines/ledger.jsonl (ragbench Ledger: model,
provider, tokens, reasoning tokens, cost_usd, done_reason, ...). Global floor: GET /api/v1/key limit_remaining."""
import json
import os
import threading

import requests

import s6lib as L
from ragbench.llm import BudgetExceeded, Ledger
from ragbench.llm_openrouter import OpenRouterReader, model_info

CAP_USD = 0.40
FLOOR_USD = 2.00
LEDGER = Ledger(path=str(L.OUT / "ledger.jsonl"))
os.environ["OPENROUTER_PROVIDER_PINS"] = json.dumps(L.MAN["reader_pins"])
os.environ.pop("READER_FALLBACK_ENV", None)


def spent():
    tot = 0.0
    p = L.OUT / "ledger.jsonl"
    if p.is_file():
        for line in open(p, encoding="utf-8"):
            if line.strip():
                tot += float(json.loads(line).get("cost_usd") or 0.0)
    return tot


class LocalBudget:
    """reserve/settle interface of ragbench ReaderBudget; the S6 ledger is the source of truth."""
    lock = threading.Lock()

    def reserve(self, model):
        if spent() >= CAP_USD:
            raise BudgetExceeded("S6 cap reached: $%.4f of $%.2f" % (spent(), CAP_USD))

    def settle(self, model, cost, ptok, ctok):
        pass


def account_remaining():
    key = L.CFG.openrouter_key()
    d = requests.get("https://openrouter.ai/api/v1/key", headers={"Authorization": "Bearer " + key}, timeout=20).json()
    return d["data"].get("limit_remaining")


def floor_check():
    rem = account_remaining()
    with open(L.OUT / "floor_checks.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps({"limit_remaining": rem, "s6_spent": round(spent(), 6)}) + "\n")
    if rem is not None and float(rem) < FLOOR_USD:
        raise BudgetExceeded("account limit_remaining $%.2f below the $%.2f floor" % (float(rem), FLOOR_USD))
    if spent() >= CAP_USD:
        raise BudgetExceeded("S6 cap reached")
    return rem


def _catalog(mid):
    return model_info(mid, L.CFG.project_root / "work" / "openrouter" / "catalog.json")


def glm_reader():
    """The answer reader exactly as ragbench.run --reader builds it (READER_MAX_ATTEMPTS = manifest 7)."""
    mid = "z-ai/glm-5.3-flash"
    c = OpenRouterReader(L.CFG, LEDGER, mid, supported=_catalog(mid).get("params"),
                         attempts=int(L.MAN["reader_max_attempts"]))
    c.budget = LocalBudget()
    return c


def glm_ranker(max_tokens_total=2000, attempts=3, deadline_s=None):
    """The listwise ranker call: GLM via the same pinned endpoint, reasoning effort low, one cap for reasoning+output."""
    mid = "z-ai/glm-5.3-flash"
    c = OpenRouterReader(L.CFG, LEDGER, mid, supported=_catalog(mid).get("params"), attempts=attempts,
                         deadline_s=deadline_s, max_tokens_total=max_tokens_total, reasoning_effort="low")
    c.budget = LocalBudget()
    return c


def judge_client():
    j = L.MAN["judge"]
    c = OpenRouterReader(L.CFG, LEDGER, j["model"], supported=_catalog(j["model"]).get("params"),
                         provider=j["provider_pin"], attempts=j["http_attempts"])
    c.budget = LocalBudget()
    return c
