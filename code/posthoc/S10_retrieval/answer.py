"""Answer the S10 packs with the pinned primary reader (GLM-5.3-flash, MANIFEST reader pin, the frozen OpenRouterReader
payload rule and attempts) - exactly the pack messages and num_predict. (Adapted from S2a_packs/answer.py.)

    python answer.py --split test|dev [--arms A,B] [--ids q1,q2] [--workers 6]

A cell byte-identical to a frozen HYBRID/HR pack reuses the frozen answer (source frozen:<arm>); a cell identical to an
already answered S10 cell reuses it (dup:<arm>). Append-only answers.jsonl (dev: dev_answers.jsonl) keyed (arm, qid).
Money: before each batch of <= 50 calls the key's limit_remaining must be >= $2.00; stream cap $0.25 from the ledgers.
"""
import argparse
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import requests

import s10lib as L
from ragbench.llm import Ledger
from ragbench.llm_openrouter import OpenRouterReader, model_info

CAP_USD = 0.45          # raised from 0.25 by the coordinator (2026-11-10)
FLOOR_USD = 2.00
LEDGER = L.HERE / "ledger.jsonl"
MAN = json.loads((L.FZ / "MANIFEST.json").read_text(encoding="utf-8"))
READER = "z-ai/glm-5.3-flash"


class _NoBudget:
    def reserve(self, model):
        pass

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
    return float(v) if v is not None else None


def make_client(ledger, model, provider, attempts):
    info = model_info(model, L.CFG.project_root / "work" / "openrouter" / "catalog.json")
    c = OpenRouterReader(L.CFG, ledger, model, supported=info.get("params"), provider=provider, attempts=attempts)
    c.budget = _NoBudget()
    return c


def ans_path(split):
    return L.HERE / ("answers.jsonl" if split == "test" else "dev_answers.jsonl")


def done_keys(split="test"):
    out = {}
    p = ans_path(split)
    if p.is_file():
        for l in open(p, encoding="utf-8"):
            if l.strip():
                r = json.loads(l)
                if not r.get("error"):
                    out[(r["arm"], r["qid"])] = r
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--arms", default=",".join(L.ARMS))
    ap.add_argument("--ids", default=None)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--reader-cap", type=float, default=0.20, help="no new reader call once stream spend >= this")
    ap.add_argument("--answerable-only", action="store_true")
    ap.add_argument("--unanswerable-only", action="store_true")
    a = ap.parse_args()
    arms = a.arms.split(",")
    idx = L.HERE / ("packs_index.jsonl" if a.split == "test" else "dev_packs_index.jsonl")
    pdir = L.HERE / ("packs" if a.split == "test" else "dev_packs")
    rows = [json.loads(l) for l in open(idx, encoding="utf-8") if l.strip()]
    qset = set(a.ids.split(",")) if a.ids else None
    rows = [r for r in rows if r["arm"] in arms and (qset is None or r["qid"] in qset)]
    if a.answerable_only:
        rows = [r for r in rows if "-AB-" not in r["qid"]]
    if a.unanswerable_only:
        rows = [r for r in rows if "-AB-" in r["qid"]]
    rows.sort(key=lambda r: (arms.index(r["arm"]),))
    done = done_keys(a.split)
    by_sha = {r["sha"]: r for r in done.values()}
    lock = threading.Lock()
    ledger = Ledger(path=str(LEDGER))
    client = make_client(ledger, READER, MAN["reader_pins"][READER], int(MAN["reader_max_attempts"]))
    AP = ans_path(a.split)

    def emit(rec):
        with lock:
            with open(AP, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    todo = []
    for r in rows:
        if (r["arm"], r["qid"]) in done:
            continue
        base = {k: r[k] for k in ("arm", "qid", "setting", "sha", "chars", "num_predict")}
        if r.get("same_as_frozen"):
            fa = r["same_as_frozen"][0]
            fr = L.frozen_records(fa)[r["qid"]]
            emit(dict(base, answer_text=fr["answer_text"], source="frozen:" + fa, error=None))
            continue
        if r["sha"] in by_sha:
            d = by_sha[r["sha"]]
            emit(dict(base, answer_text=d["answer_text"], source="dup:" + d["arm"], error=None))
            continue
        todo.append((r, base))
    first, dups = {}, []
    for r, base in todo:
        (dups.append((r, base)) if r["sha"] in first else first.__setitem__(r["sha"], (r, base)))
    print("cells:", len(rows), "to call:", len(first), "dups:", len(dups), "already:", len(done), flush=True)

    def call(item):
        r, base = item
        pk = json.loads((pdir / r["arm"] / (r["qid"] + ".json")).read_text(encoding="utf-8"))
        assert pk["messages_sha256"] == r["sha"]
        if spent() >= a.reader_cap:
            return dict(base, answer_text="", source="call", error="skipped: reader cap", skipped=True)
        t0 = time.perf_counter()
        with ledger.scope(phase="query", approach="S10:" + r["arm"], question_id=r["qid"], setting=r["setting"]):
            try:
                res = client.chat(pk["messages"], purpose="answer", num_predict=pk["num_predict"])
                rec = dict(base, answer_text=res.text.strip(), source="call", error=None,
                           done_reason=res.done_reason, prompt_tokens=res.prompt_tokens,
                           completion_tokens=res.completion_tokens, cost_usd=res.cost_usd,
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
        print("batch %d: key limit_remaining=%s stream spent=$%.4f" % (i // 50, rem, sp), flush=True)
        if rem is not None and rem < FLOOR_USD:
            print("STOP: key limit_remaining below floor")
            break
        if sp >= CAP_USD - 0.03:
            print("STOP: stream cap")
            break
        with ThreadPoolExecutor(a.workers) as ex:
            for rec in ex.map(call, items[i:i + 50]):
                if rec.get("skipped"):
                    continue
                results[rec["sha"]] = rec
                if rec.get("error"):
                    print("ERROR", rec["arm"], rec["qid"], rec["error"][:120], flush=True)
    for r, base in dups:
        d = results.get(r["sha"])
        if d and not d.get("error"):
            emit(dict(base, answer_text=d["answer_text"], source="dup:" + d["arm"], error=None))
    print("stream spent now $%.4f" % spent())


if __name__ == "__main__":
    main()
