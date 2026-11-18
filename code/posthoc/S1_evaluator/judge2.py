"""S1 task 1: re-judge the 768 primary-reader (GLM-5.3-flash) test answers with a second judge from a non-reader
family, using the IDENTICAL rubric, prompts, parser, retry and malformed-reply policy as the pinned judge
(ragbench.tools.judge_multi.judge_records from the frozen worktree @77924d4, imported read-only).

    python judge2.py --estimate          # 3 questions x 8 arms (cost estimate)
    python judge2.py --all               # every cell; resumable (verdict keys already written are skipped)

Records: exactly the records report_test.load() scores (REC[GLM][arm][q]), keyed by rid. Output (append-only):
    judge2/shard_<k>/verdicts_<judge>.jsonl, needs_adjudication_<judge>.jsonl, judge_failures_<judge>.jsonl
    ledger.jsonl (one line per API call: model, provider, tokens, usage.cost)
No GPU, no Ollama. The OpenRouter key is read from the repo .env and never printed.
"""
import argparse
import json
import os
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO / "work" / "datasets_v2" / "test_run"))
import report_test as RT  # noqa: E402  (puts the frozen worktree on sys.path, read-only)
from ragbench.llm import BudgetExceeded  # noqa: E402
from ragbench.tools import judge_multi as JM  # noqa: E402
from ragbench.util import KVCache  # noqa: E402

JUDGE2 = "mistralai/mistral-large-2512"
PIN2 = "mistral/zdr"
GEN = {"temperature": 0, "top_p": 1.0, "seed": 42}          # identical to judge_pin.json "generation"
CAP_USD = 1.00
FLOOR_USD = 2.00
LEDGER = HERE / "ledger.jsonl"
OUT = HERE / "judge2"
QFILE = REPO / "work" / "datasets_v2" / "test_freeze" / "questions" / "test_v2judge_all.jsonl"


def _key():
    for line in open(REPO / ".env", encoding="utf-8"):
        if line.startswith("OPENROUTER_API_KEY"):
            return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise RuntimeError("no OPENROUTER_API_KEY")


def key_remaining(key):
    req = urllib.request.Request("https://openrouter.ai/api/v1/key", headers={"Authorization": "Bearer " + key})
    return float(json.load(urllib.request.urlopen(req, timeout=30))["data"]["limit_remaining"])


def ledger_spent():
    if not LEDGER.exists():
        return 0.0
    return sum(float(json.loads(l).get("cost") or 0) for l in open(LEDGER, encoding="utf-8") if l.strip())


class Res:
    def __init__(self, text, provider):
        self.text, self.provider = text, provider


class Client:
    """Minimal pinned OpenRouter client with the judge_records interface: chat(messages, purpose, num_predict)."""

    def __init__(self, model, pin, cap):
        self.model, self.pin, self.cap = model, pin, cap
        self.key = _key()
        self.lock = threading.Lock()
        self.spent = ledger_spent()
        self.calls_since_check = 50                    # forces a floor check before the first call

    def _guard(self):
        with self.lock:
            if self.spent >= self.cap:
                raise BudgetExceeded("stream cap $%.2f reached (spent %.4f)" % (self.cap, self.spent))
            if self.calls_since_check >= 50:
                rem = key_remaining(self.key)
                if rem < FLOOR_USD:
                    raise BudgetExceeded("global floor: limit_remaining %.2f < %.2f" % (rem, FLOOR_USD))
                self.calls_since_check = 0
            self.calls_since_check += 1

    def chat(self, messages, *, purpose, num_predict=None, **_):
        self._guard()
        body = {"model": self.model, "messages": messages, "usage": {"include": True},
                "max_tokens": int(num_predict or 512) + 64,
                "provider": {"order": [self.pin], "allow_fallbacks": False, "require_parameters": True}, **GEN}
        data = json.dumps(body).encode("utf-8")
        last = None
        for att in range(6):                            # transport retries (429/5xx/network); not judge attempts
            t0 = time.time()
            try:
                req = urllib.request.Request("https://openrouter.ai/api/v1/chat/completions", data=data,
                                             headers={"Authorization": "Bearer " + self.key,
                                                      "Content-Type": "application/json"})
                d = json.load(urllib.request.urlopen(req, timeout=180))
            except (urllib.error.URLError, TimeoutError, OSError, ValueError) as e:
                last = e
                self._log({"ok": False, "error": "%s: %s" % (type(e).__name__, str(e)[:200]), "attempt": att + 1,
                           "wall_ms": round(1000 * (time.time() - t0))})
                time.sleep(5 * (att + 1))
                continue
            if "error" in d and not d.get("choices"):
                last = RuntimeError(str(d["error"])[:300])
                self._log({"ok": False, "error": str(d["error"])[:300], "attempt": att + 1})
                time.sleep(5 * (att + 1))
                continue
            u = d.get("usage") or {}
            ch = (d.get("choices") or [{}])[0]
            self._log({"ok": True, "provider": d.get("provider"), "prompt_tokens": u.get("prompt_tokens"),
                       "completion_tokens": u.get("completion_tokens"), "cost": u.get("cost"),
                       "finish_reason": ch.get("finish_reason"), "request_id": d.get("id"),
                       "wall_ms": round(1000 * (time.time() - t0)), "purpose": purpose})
            return Res((ch.get("message") or {}).get("content") or "", d.get("provider"))
        raise RuntimeError("call failed after 6 transport attempts: %s" % last)

    def _log(self, row):
        row = dict({"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "stream": "S1", "task": "judge2", "model": self.model,
                    "provider_pin": self.pin}, **row)
        with self.lock:
            self.spent += float(row.get("cost") or 0)
            with open(LEDGER, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(row) + "\n")


def records(qids):
    REC, _ = RT.load()
    out = {}
    for a in RT.ARMS:
        for q in qids:
            r = REC[RT.PRIMARY][a].get(q)
            if r is None or r.get("error"):
                continue                                 # scored 0 as a system failure, exactly like the test report
            out[r["meta"]["identity"]["rid"]] = r
    return out


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--estimate", action="store_true")
    g.add_argument("--all", action="store_true")
    ap.add_argument("--shards", type=int, default=6)
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    qids = RT.QIDS[:0]
    if a.estimate:
        # 3 questions spread over the three domains (first answerable of each domain)
        seen = {}
        for q in RT.ANS:
            seen.setdefault(RT.QS[q]["domain"], q)
        qids = list(seen.values())[:3]
    else:
        qids = RT.QIDS
    recs = records(qids)
    qs = JM.load_question_dicts(QFILE)
    client = Client(JUDGE2, PIN2, CAP_USD)
    print("records", len(recs), "spent so far $%.4f" % client.spent, flush=True)
    items = sorted(recs.items())
    shards = [dict(items[i::a.shards]) for i in range(a.shards)]
    errs = []

    def work(k, sub):
        out = OUT / ("shard_%d" % k)
        try:
            JM.judge_records(sub, qs, client, JUDGE2, out, cache=KVCache(out / "cache.sqlite"),
                             provider_pin=PIN2, max_attempts=3, on_malformed="adjudicate",
                             log=lambda s: print("shard", k, s, flush=True))
        except BudgetExceeded as e:
            errs.append(str(e))
            print("shard", k, "BUDGET STOP", e, flush=True)

    th = [threading.Thread(target=work, args=(k, s)) for k, s in enumerate(shards) if s]
    for t in th:
        t.start()
    for t in th:
        t.join()
    print("done; spent $%.4f; stops: %s" % (ledger_spent(), errs), flush=True)


if __name__ == "__main__":
    main()
