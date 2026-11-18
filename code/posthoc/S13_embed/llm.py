"""S13 reader + judge (adapted from S2a answer.py / judge.py; frozen ragbench code, pins from the test freeze).

    python llm.py answer --split dev|test [--arms E5_DENSE,E5_HYBRID] [--limit-q 3]
    python llm.py judge  --split dev|test [--limit-q 3]

Reader: z-ai/glm-5.3-flash, MANIFEST reader pin + max attempts, exact pack messages and num_predict. A test cell whose
messages are byte-identical to the frozen HYBRID_6000 / HR_6000 pack reuses that frozen answer and verdict
(source=frozen:<arm>), as S2a does; an identical new pack reuses the earlier new answer (dup).
Judge: judge_pin.json through ragbench.tools.judge_multi.judge_records (3 identical attempts, on_malformed adjudicate).
Money: key limit_remaining >= $2.00 before every batch of <= 50 calls; stream cap $0.25 from ledger*.jsonl.
"""
import argparse
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import requests

import s13lib as S
from ragbench.llm import Ledger
from ragbench.llm_openrouter import OpenRouterReader, model_info
from ragbench.tools import judge_multi as JM
from ragbench.util import KVCache, sha1_json

PK = S.PK
CAP_USD = 0.25
FLOOR_USD = 2.00
LEDGER = S.HERE / "ledger.jsonl"
MAN = json.loads((PK.FZ / "MANIFEST.json").read_text(encoding="utf-8"))
JPIN = json.loads((PK.FZ / "judge_pin.json").read_text(encoding="utf-8"))
READER = "z-ai/glm-5.3-flash"
EXP = "posthoc-2026-11-S13"
JQ = {"test": PK.FZ / "questions" / "test_v2judge_all.jsonl",
      "dev": S.DS / "devrun" / "questions" / "dev_v2judge_all.jsonl"}


class _NoBudget:
    def reserve(self, model):
        pass

    def settle(self, *a, **k):
        pass


def spent():
    tot = 0.0
    for f in sorted(S.HERE.glob("ledger*.jsonl")):
        for l in open(f, encoding="utf-8"):
            if l.strip():
                try:
                    tot += float(json.loads(l).get("cost_usd") or 0.0)
                except ValueError:
                    pass
    return tot


def limit_remaining():
    key = PK.CFG.openrouter_key()
    d = requests.get("https://openrouter.ai/api/v1/key", headers={"Authorization": "Bearer " + key}, timeout=20).json()
    v = d["data"].get("limit_remaining")
    rem = float(v) if v is not None else None
    with open(S.HERE / "money_checks.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps({"t": time.strftime("%Y-%m-%dT%H:%M:%S"), "limit_remaining": rem,
                            "stream_spent": round(spent(), 5)}) + "\n")
    return rem


def make_client(ledger, model, provider, attempts):
    info = model_info(model, PK.CFG.project_root / "work" / "openrouter" / "catalog.json")
    c = OpenRouterReader(PK.CFG, ledger, model, supported=info.get("params"), provider=provider, attempts=attempts)
    c.budget = _NoBudget()
    return c


def ans_path(split):
    return S.HERE / ("answers_%s.jsonl" % split)


def done_keys(split):
    out = {}
    p = ans_path(split)
    if p.is_file():
        for l in open(p, encoding="utf-8"):
            if l.strip():
                r = json.loads(l)
                if not r.get("error"):
                    out[(r["arm"], r["qid"])] = r
    return out


def pick_q(rows, limit):
    qs = []
    for r in rows:
        if r["qid"] not in qs:
            qs.append(r["qid"])
    if limit:
        t2 = [q for q in qs if q.endswith("T2-001")]
        return set((t2 + qs)[:limit])
    return set(qs)


def answer(a):
    idx = S.HERE / ("packs_index%s.jsonl" % ("_dev" if a.split == "dev" else ""))
    pdir = S.HERE / ("packs_dev" if a.split == "dev" else "packs")
    rows = [json.loads(l) for l in open(idx, encoding="utf-8") if l.strip()]
    arms = a.arms.split(",")
    rows = [r for r in rows if r["arm"] in arms]
    qset = pick_q(rows, a.limit_q)
    rows = [r for r in rows if r["qid"] in qset]
    done = done_keys(a.split)
    by_sha = {r["sha"]: r for r in done.values()}
    lock = threading.Lock()
    ledger = Ledger(path=str(LEDGER))
    client = make_client(ledger, READER, MAN["reader_pins"][READER], int(MAN["reader_max_attempts"]))

    def emit(rec):
        with lock:
            with open(ans_path(a.split), "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    first, dups = {}, []
    for r in rows:
        if (r["arm"], r["qid"]) in done:
            continue
        base = {k: r[k] for k in ("arm", "qid", "setting", "sha", "chars", "num_predict")}
        if r["same_as_frozen"]:
            fa = r["same_as_frozen"][0]
            emit(dict(base, answer_text=PK.frozen_records(fa)[r["qid"]]["answer_text"], source="frozen:" + fa,
                      error=None))
            continue
        if r["sha"] in by_sha:
            d = by_sha[r["sha"]]
            emit(dict(base, answer_text=d["answer_text"], source="dup:" + d["arm"], error=None))
            continue
        if r["sha"] in first:
            dups.append((r, base))
        else:
            first[r["sha"]] = (r, base)
    print("to call:", len(first), "dups:", len(dups), "already:", len(done), flush=True)

    def call(item):
        r, base = item
        pk = json.loads((pdir / r["arm"] / (r["qid"] + ".json")).read_text(encoding="utf-8"))
        assert pk["messages_sha256"] == r["sha"]
        t0 = time.perf_counter()
        with ledger.scope(phase="query", approach=r["arm"], question_id=r["qid"], setting=r["setting"]):
            try:
                res = client.chat(pk["messages"], purpose="answer", num_predict=pk["num_predict"])
                rec = dict(base, answer_text=res.text.strip(), source="call", error=None,
                           done_reason=res.done_reason, prompt_tokens=res.prompt_tokens,
                           completion_tokens=res.completion_tokens,
                           reasoning_tokens=getattr(res, "reasoning_tokens", None), cost_usd=res.cost_usd,
                           provider=getattr(res, "provider", None))
            except Exception as e:  # noqa: BLE001
                rec = dict(base, answer_text="", source="call", error="%s: %s" % (type(e).__name__, str(e)[:300]))
        rec["latency_ms"] = round((time.perf_counter() - t0) * 1000.0, 1)
        emit(rec)
        return rec

    items = list(first.values())
    results = {}
    for i in range(0, len(items), 50):
        rem, sp = limit_remaining(), spent()
        print("batch %d: limit_remaining=%s spent=$%.4f" % (i // 50, rem, sp), flush=True)
        if rem is not None and rem < FLOOR_USD or sp >= CAP_USD - 0.03:
            print("STOP: money")
            return 1
        with ThreadPoolExecutor(a.workers) as ex:
            for rec in ex.map(call, items[i:i + 50]):
                results[rec["sha"]] = rec
                if rec.get("error"):
                    print("ERROR", rec["arm"], rec["qid"], rec["error"][:150], flush=True)
    for r, base in dups:
        d = results.get(r["sha"])
        if d and not d.get("error"):
            emit(dict(base, answer_text=d["answer_text"], source="dup:" + d["arm"], error=None))
    print("stream spent now $%.4f" % spent())
    return 0


def identity(arm, qid, setting):
    ident = {"experiment_id": EXP, "question_id": qid, "pool_id": PK.pool_of(qid), "method": arm,
             "options": {"flags": {}, "variants_version": "variants-v2"},
             "reader": {"chat": "openrouter/" + READER, "chat_digest": "openrouter"}, "replicate": 0,
             "setting": setting}
    ident["rid"] = sha1_json(ident)[:20]
    return ident


def judge(a):
    done = done_keys(a.split)
    qset = pick_q([{"qid": k[1]} for k in sorted(done, key=lambda k: k[1])], a.limit_q) if a.limit_q else None
    recs = {}
    for r in done.values():
        if r["source"].startswith("frozen:") or (qset and r["qid"] not in qset):
            continue
        ident = identity(r["arm"], r["qid"], r["setting"])
        recs[ident["rid"]] = {"question_id": r["qid"], "approach": r["arm"], "setting": r["setting"],
                              "answer_text": r["answer_text"], "error": None, "trace": {}, "meta": {"identity": ident}}
    qs = JM.load_question_dicts(JQ[a.split])
    ledger = Ledger(path=str(S.HERE / "ledger_judge.jsonl"))
    client = make_client(ledger, JPIN["judge"], JPIN["provider_pin"], 3)
    jdir = S.HERE / "judge" / a.split
    jdir.mkdir(parents=True, exist_ok=True)
    items = sorted(recs.items())
    print("records", len(items), flush=True)
    for i in range(0, len(items), 50):
        rem, sp = limit_remaining(), spent()
        if rem is not None and rem < FLOOR_USD or sp >= CAP_USD - 0.02:
            print("STOP: money", rem, sp)
            return 1
        JM.judge_records(dict(items[i:i + 50]), qs, client, JPIN["judge"], jdir,
                         cache=KVCache(jdir / "judge_cache.sqlite"), ledger=ledger, provider_pin=JPIN["provider_pin"],
                         max_attempts=3, on_malformed="adjudicate")
    print("spent now $%.4f" % spent())
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["answer", "judge"])
    ap.add_argument("--split", required=True)
    ap.add_argument("--arms", default="E5_DENSE,E5_HYBRID")
    ap.add_argument("--limit-q", type=int, default=None)
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    raise SystemExit(answer(a) if a.cmd == "answer" else judge(a))
