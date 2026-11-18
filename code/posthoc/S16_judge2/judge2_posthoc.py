"""S8 step 4: judge 2 (mistralai/mistral-large-2512, pinned mistral/zdr, no fallbacks, require_parameters,
temperature 0 / top_p 1 / seed 42, max_tokens = num_predict + 64) on the post-hoc arms, exactly as S1 ran it: the
frozen ragbench.tools.judge_multi.judge_records (same rubric/prompt/parser, max 3 judge attempts, on_malformed
adjudicate) with the S1 client (copied: S1_evaluator/judge2.py Client, ledger -> this folder).
Records are exactly those each stream fed to judge 1:
  FUSION_6000, GOLD  S2a answers.jsonl, S2a judge.identity() (cells with source frozen:<arm> are the frozen record:
                     their judge-2 verdict is S1's, read through S1 j2lib.judge2)
  SETR_NOFILL, LISTWISE, CLOSEDBOOK  JM.latest_identity_records(S6_baselines/test/<ARM>)
  ITER_BM25, GREP_AGENT              JM.latest_identity_records(S3_agentic/runs/<ARM>/test)
    python judge2_posthoc.py --arms A,B [--limit-q 3]
Out: judge2/<ARM>/verdicts_*.jsonl, needs_adjudication_*.jsonl, cache.sqlite; ledger_j2.jsonl."""
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
PH = REPO / "work" / "posthoc_2026-11"
sys.path.insert(0, str(REPO / "work" / "datasets_v2" / "test_run"))
import report_test as RT  # noqa: E402
from ragbench.llm import BudgetExceeded  # noqa: E402
from ragbench.tools import judge_multi as JM  # noqa: E402
from ragbench.util import KVCache, sha1_json  # noqa: E402

JUDGE2 = "mistralai/mistral-large-2512"
PIN2 = "mistral/zdr"
GEN = {"temperature": 0, "top_p": 1.0, "seed": 42}
CAP_USD = 0.90
TRANSPORT_ATTEMPTS = int(os.environ.get("S8_J2_TRANSPORT", "12"))   # S1 used 6; raised only for upstream 429s
FLOOR_USD = 2.00
LEDGER = HERE / "ledger_j2.jsonl"
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


def stream_spent():
    """S8 total = spend.json (GLM + judge 1, via s8lib) + this ledger."""
    s = 0.0
    p = HERE / "spend.json"
    if p.exists():
        s += json.loads(p.read_text(encoding="utf-8"))["cost_usd"]
    if LEDGER.exists():
        s += sum(float(json.loads(l).get("cost") or 0) for l in open(LEDGER, encoding="utf-8") if l.strip())
    return s


class Res:
    def __init__(self, text, provider):
        self.text, self.provider = text, provider


class Client:
    """S1_evaluator/judge2.py Client, unchanged except the ledger path, the stream tag and the cap source."""

    def __init__(self, model, pin, cap):
        self.model, self.pin, self.cap = model, pin, cap
        self.key = _key()
        self.lock = threading.Lock()
        self.calls_since_check = 50

    def _guard(self):
        with self.lock:
            sp = stream_spent()
            if sp >= self.cap:
                raise BudgetExceeded("stream cap $%.2f reached (spent %.4f)" % (self.cap, sp))
            if self.calls_since_check >= 50:
                rem = key_remaining(self.key)
                with open(HERE / "money_checks.jsonl", "a", encoding="utf-8") as fh:
                    fh.write(json.dumps({"at": time.strftime("%Y-%m-%dT%H:%M:%S"), "tag": "judge2",
                                         "limit_remaining": rem, "stream_spent": sp, "ok": rem >= FLOOR_USD}) + "\n")
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
        for att in range(TRANSPORT_ATTEMPTS):
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
                time.sleep(min(60, 5 * (att + 1)))
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
        raise RuntimeError("call failed after %d transport attempts: %s" % (TRANSPORT_ATTEMPTS, last))

    def _log(self, row):
        row = dict({"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "stream": "S16", "task": "judge2", "model": self.model,
                    "provider_pin": self.pin}, **row)
        with self.lock:
            with open(LEDGER, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(row) + "\n")


# ------------------------------------------------------------------ records per arm (as judged by judge 1)
def s2a_identity(arm, qid, setting):          # S2a_packs/judge.py identity(), copied
    ident = {"experiment_id": "posthoc-2026-11-S2a", "question_id": qid, "pool_id": "-".join(qid.split("-")[2:4]),
             "method": arm, "options": {"flags": {}, "variants_version": "variants-v2"},
             "reader": {"chat": "openrouter/z-ai/glm-5.3-flash", "chat_digest": "openrouter"}, "replicate": 0,
             "setting": setting}
    ident["rid"] = sha1_json(ident)[:20]
    return ident


def s2a_rows():
    out = {}
    for l in open(PH / "S2a_packs" / "answers.jsonl", encoding="utf-8"):
        if l.strip():
            r = json.loads(l)
            if not r.get("error"):
                out[(r["arm"], r["qid"])] = r
    return out


RUN_DIRS = {"SETR_NOFILL": PH / "S6_baselines" / "test" / "SETR_NOFILL",
            "LISTWISE": PH / "S6_baselines" / "test" / "LISTWISE",
            "CLOSEDBOOK": PH / "S6_baselines" / "test" / "CLOSEDBOOK",
            "ITER_BM25": PH / "S3_agentic" / "runs" / "ITER_BM25" / "test",
            "GREP_AGENT": PH / "S3_agentic" / "runs" / "GREP_AGENT" / "test"}
J1_DIRS = {"SETR_NOFILL": [PH / "S6_baselines" / "test" / "judge_SETR_NOFILL"],
           "LISTWISE": [PH / "S6_baselines" / "test" / "judge_LISTWISE"],
           "CLOSEDBOOK": [PH / "S6_baselines" / "test" / "judge_CLOSEDBOOK"],
           "ITER_BM25": [PH / "S3_agentic" / "judge" / "ITER_BM25" / "test"],
           "GREP_AGENT": [PH / "S3_agentic" / "judge" / "GREP_AGENT" / "test"],
           "FUSION_6000": [PH / "S2a_packs" / "judge", PH / "S2a_packs" / "judge" / "FUSION_6000"],
           "GOLD": [PH / "S2a_packs" / "judge", PH / "S2a_packs" / "judge" / "GOLD"]}


def records(arm):
    """-> ({rid: record} to judge, {qid: ('rec', rid, answer_sha1) | ('frozen', arm)})."""
    recs, cell = {}, {}
    if arm in ("FUSION_6000", "GOLD"):
        for (a, q), r in s2a_rows().items():
            if a != arm:
                continue
            if r["source"].startswith("frozen:"):
                cell[q] = ("frozen", r["source"].split(":", 1)[1])
                continue
            ident = s2a_identity(arm, q, r["setting"])
            rec = {"question_id": q, "approach": arm, "setting": r["setting"], "answer_text": r["answer_text"],
                   "error": None, "trace": {}, "meta": {"identity": ident}}
            recs[ident["rid"]] = rec
            cell[q] = ("rec", ident["rid"], JM.answer_sha1(rec))
    else:
        rr, _ = JM.latest_identity_records([RUN_DIRS[arm]])
        for rid, r in rr.items():
            q = r["question_id"]
            if r.get("error"):
                cell[q] = ("error",)
                continue
            recs[rid] = r
            cell[q] = ("rec", rid, JM.answer_sha1(r))
    return recs, cell


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", required=True)
    ap.add_argument("--limit-q", type=int, default=None)
    ap.add_argument("--retry-failures", action="store_true", help="re-try records whose calls failed in transport")
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    qs = JM.load_question_dicts(QFILE)
    client = Client(JUDGE2, PIN2, CAP_USD)
    qset = None
    if a.limit_q:
        seen = {}
        for q in RT.ANS:
            seen.setdefault(RT.QS[q]["domain"], q)
        qset = set(list(seen.values())[:a.limit_q])
    for arm in a.arms.split(","):
        recs, _ = records(arm)
        if qset:
            recs = {k: v for k, v in recs.items() if v["question_id"] in qset}
        out = OUT / arm
        print(arm, "records", len(recs), "spent $%.4f" % stream_spent(), flush=True)
        try:
            items = sorted(recs.items())
            for i in range(0, len(items), 15):        # <= 45 judge calls per chunk; floor check every 50 calls
                JM.judge_records(dict(items[i:i + 15]), qs, client, JUDGE2, out, cache=KVCache(out / "cache.sqlite"),
                                 provider_pin=PIN2, max_attempts=3, on_malformed="adjudicate", retry_failures=a.retry_failures,
                                 log=lambda s: print(arm, s, flush=True))
        except BudgetExceeded as e:
            print(arm, "BUDGET STOP", e, flush=True)
            return 3
    print("done; spent $%.4f" % stream_spent(), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
