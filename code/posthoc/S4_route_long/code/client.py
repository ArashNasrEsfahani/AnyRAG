"""OpenRouter clients for S4: the frozen ragbench OpenRouterReader (same payload rule, pin routing, retries, ledger
records with usage.cost), with its shared reader_budget.json guard replaced by a LOCAL guard that writes nothing
outside S4 and refuses a call once this stream's ledger reaches the cap or the account floor is near."""
import json
import os
import common as C

CAP_USD = 1.20
FLOOR_USD = 2.00
_CHECK = {"n": 0, "remaining": None}


class LocalBudget:
    """reserve(): every 50 calls (and on the first) GET /key limit_remaining; refuse below the floor or at the cap."""

    def reserve(self, model):
        from ragbench.llm import BudgetExceeded
        if _CHECK["n"] % 50 == 0:
            _CHECK["remaining"] = C.limit_remaining()
            C.append(C.S4 / "key_checks.jsonl", {"n_calls_before": _CHECK["n"], "limit_remaining": _CHECK["remaining"]})
        _CHECK["n"] += 1
        if _CHECK["remaining"] is not None and _CHECK["remaining"] < FLOOR_USD:
            raise BudgetExceeded("account limit_remaining %.3f below floor %.2f" % (_CHECK["remaining"], FLOOR_USD))
        s = C.spent()
        if s >= CAP_USD:
            raise BudgetExceeded("S4 cap reached: $%.4f of $%.2f" % (s, CAP_USD))

    def settle(self, model, cost, ptok, ctok):
        pass


def cfg_and_ledger():
    from ragbench.config import load_config
    from ragbench.llm import Ledger
    cfg = load_config()
    ledger = Ledger(path=str(C.S4 / "ledger.jsonl"))
    return cfg, ledger


def reader(cfg, ledger, **kw):
    """The pinned GLM reader exactly as the test (reader_pins.json via OPENROUTER_PROVIDER_PINS, 7 attempts)."""
    from ragbench.tools import reader_replay as RR
    m = json.loads((C.FZ / "MANIFEST.json").read_text(encoding="utf-8"))
    os.environ["OPENROUTER_PROVIDER_PINS"] = json.dumps(m["reader_pins"])
    os.environ.pop("READER_FALLBACK_ENV", None)
    if kw:
        from ragbench.llm_openrouter import OpenRouterReader, model_info
        info = model_info("z-ai/glm-5.3-flash", cfg.project_root / "work" / "openrouter" / "catalog.json")
        cl = OpenRouterReader(cfg, ledger, "z-ai/glm-5.3-flash", supported=info.get("params"),
                              attempts=int(m["reader_max_attempts"]), **kw)
    else:
        cl = RR.make_reader(cfg, ledger, m["reader"], int(m["reader_max_attempts"]))
    cl.budget = LocalBudget()
    return cl


def judge_client(cfg, ledger):
    from ragbench.llm_openrouter import OpenRouterReader, model_info
    jp = json.loads((C.FZ / "judge_pin.json").read_text(encoding="utf-8"))
    m = json.loads((C.FZ / "MANIFEST.json").read_text(encoding="utf-8"))
    j = m["judge"]
    info = model_info(jp["judge"], cfg.project_root / "work" / "openrouter" / "catalog.json")
    cl = OpenRouterReader(cfg, ledger, jp["judge"], supported=info.get("params"), provider=jp["provider_pin"],
                          attempts=j["http_attempts"])
    cl.budget = LocalBudget()
    return cl, j
