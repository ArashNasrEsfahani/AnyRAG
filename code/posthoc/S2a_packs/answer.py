"""Answer the new packs with the pinned primary reader (GLM-5.3-flash, together pin, the frozen OpenRouterReader payload
rule, 7 HTTP attempts), exactly the pack messages and num_predict.

    python answer.py --arms FUSION_6000,FUSION_15500,... [--limit-q 3] [--workers 6]

A cell whose messages are byte-identical to a frozen GLM pack reuses that frozen answer (source=frozen:<arm>); a cell
identical to an already-answered new cell reuses it (source=dup:<arm>). Append-only answers.jsonl keyed (arm, qid).
Money: before each batch of <= 50 calls the key's limit_remaining must be >= $2.00; stream cap from ledger.jsonl.
"""
import argparse
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import requests

import packs as PK
from ragbench.llm import Ledger
from ragbench.llm_openrouter import OpenRouterReader, model_info

CAP_USD = 1.50
FLOOR_USD = 2.00
LEDGER = PK.HERE / "ledger.jsonl"
ANS = PK.HERE / "answers.jsonl"
MAN = json.loads((PK.FZ / "MANIFEST.json").read_text(encoding="utf-8"))
READER = "z-ai/glm-5.3-flash"


class _NoBudget:          # the shared work/openrouter/reader_budget.json is not touched; spend is capped from LEDGER
    def reserve(self, model):
        pass

    def settle(self, *a, **k):
        pass


def spent():
    tot = 0.0
    for f in sorted(PK.HERE.glob("ledger*.jsonl")):          # ledger.jsonl + one ledger_judge_<shard>.jsonl per shard
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
    return float(v) if v is not None else None


def make_client(ledger, model, provider, attempts):
    info = model_info(model, PK.CFG.project_root / "work" / "openrouter" / "catalog.json")
    c = OpenRouterReader(PK.CFG, ledger, model, supported=info.get("params"), provider=provider, attempts=attempts)
    c.budget = _NoBudget()
    return c


def reader_client(ledger):
    return make_client(ledger, READER, MAN["reader_pins"][READER], int(MAN["reader_max_attempts"]))


def done_keys():
    out = {}
    if ANS.is_file():
        for l in open(ANS, encoding="utf-8"):
            if l.strip():
                r = json.loads(l)
                if not r.get("error"):
                    out[(r["arm"], r["qid"])] = r
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", required=True)
    ap.add_argument("--index", default="packs_index.jsonl")
    ap.add_argument("--limit-q", type=int, default=None)
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    arms = a.arms.split(",")
    rows = [json.loads(l) for l in open(PK.HERE / a.index, encoding="utf-8") if l.strip()]
    qorder = PK.qids_all()
    if a.limit_q:
        # 3 questions spread over pools/languages: first T1 of BOOK-en, RPT-fa, LAW-fa ... (deterministic)
        pick = [q for q in qorder if q.endswith("T2-001")][: a.limit_q]
        qset = set(pick)
    else:
        qset = set(qorder)
    rows = [r for r in rows if r["arm"] in arms and r["qid"] in qset]
    done = done_keys()
    by_sha = {r["sha"]: r for r in done.values()}
    lock = threading.Lock()
    ledger = Ledger(path=str(LEDGER))
    client = reader_client(ledger)

    def emit(rec):
        with lock:
            with open(ANS, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    todo = []
    for r in rows:
        if (r["arm"], r["qid"]) in done:
            continue
        base = {k: r[k] for k in ("arm", "qid", "setting", "sha", "chars", "num_predict")}
        if r["same_as_frozen"]:
            fa = r["same_as_frozen"][0]
            fr = PK.frozen_records(fa)[r["qid"]]
            emit(dict(base, answer_text=fr["answer_text"], source="frozen:" + fa, error=None))
            continue
        if r["sha"] in by_sha:
            d = by_sha[r["sha"]]
            emit(dict(base, answer_text=d["answer_text"], source="dup:" + d["arm"], error=None,
                      done_reason=d.get("done_reason")))
            continue
        todo.append((r, base))
    # in-batch duplicates: answer the first, copy later
    first, dups = {}, []
    for r, base in todo:
        if r["sha"] in first:
            dups.append((r, base))
        else:
            first[r["sha"]] = (r, base)
    print("cells:", len(rows), "to call:", len(first), "dups:", len(dups), "already:", len(done), flush=True)

    def call(item):
        r, base = item
        pk = json.loads((PK.HERE / "packs" / r["arm"] / (r["qid"] + ".json")).read_text(encoding="utf-8"))
        assert pk["messages_sha256"] == r["sha"]
        t0 = time.perf_counter()
        with ledger.scope(phase="query", approach=r["arm"], question_id=r["qid"], setting=r["setting"]):
            try:
                res = client.chat(pk["messages"], purpose="answer", num_predict=pk["num_predict"])
                rec = dict(base, answer_text=res.text.strip(), source="call", error=None,
                           done_reason=res.done_reason, prompt_tokens=res.prompt_tokens,
                           completion_tokens=res.completion_tokens, reasoning_tokens=getattr(res, "reasoning_tokens", None),
                           cost_usd=res.cost_usd, provider=getattr(res, "provider", None))
            except Exception as e:  # noqa: BLE001
                rec = dict(base, answer_text="", source="call", error="%s: %s" % (type(e).__name__, str(e)[:300]))
        rec["latency_ms"] = round((time.perf_counter() - t0) * 1000.0, 1)
        emit(rec)
        return rec

    items = list(first.values())
    results = {}
    for i in range(0, len(items), 50):
        rem = limit_remaining()
        sp = spent()
        print("batch %d: key limit_remaining=%s stream spent=$%.4f" % (i // 50, rem, sp), flush=True)
        if rem is not None and rem < FLOOR_USD:
            print("STOP: key limit_remaining below floor")
            break
        if sp >= CAP_USD - 0.05:
            print("STOP: stream cap")
            break
        with ThreadPoolExecutor(a.workers) as ex:
            for rec in ex.map(call, items[i:i + 50]):
                results[rec["sha"]] = rec
                if rec.get("error"):
                    print("ERROR", rec["arm"], rec["qid"], rec["error"][:120], flush=True)
    for r, base in dups:
        d = results.get(r["sha"])
        if d and not d.get("error"):
            emit(dict(base, answer_text=d["answer_text"], source="dup:" + d["arm"], error=None,
                      done_reason=d.get("done_reason")))
    print("stream spent now $%.4f" % spent())


if __name__ == "__main__":
    main()
