"""Judge the S10 answers with the pinned judge (judge_pin.json: gpt-oss-120b, deepinfra/bf16, no fallbacks,
temperature 0 / top_p 1 / seed 42, 3 HTTP attempts, 3 identical judge attempts, on_malformed adjudicate) through the
frozen ragbench.tools.judge_multi.judge_records and the frozen judge question file. (Adapted from S2a_packs/judge.py.)
Cells reusing a frozen answer are scored from the frozen verdicts (report.py).

    python judge.py --split test|dev [--arms A,B] [--shard NAME]
"""
import argparse
import json

import answer as AN
import s10lib as L
from ragbench.llm import Ledger
from ragbench.tools import judge_multi as JM
from ragbench.util import KVCache, sha1_json

JPIN = json.loads((L.FZ / "judge_pin.json").read_text(encoding="utf-8"))
EXP = "posthoc-2026-11-S10"
JQ = {"test": L.FZ / "questions" / "test_v2judge_all.jsonl", "dev": L.DEVQ / "dev_v2judge_all.jsonl"}


def identity(arm, qid, setting):
    ident = {"experiment_id": EXP, "question_id": qid, "pool_id": L.pool_of(qid), "method": arm,
             "options": {"flags": {}, "variants_version": "variants-v2"},
             "reader": {"chat": "openrouter/" + AN.READER, "chat_digest": "openrouter"}, "replicate": 0,
             "setting": setting}
    ident["rid"] = sha1_json(ident)[:20]
    return ident


def records(split, arms=None):
    out = {}
    for r in AN.done_keys(split).values():
        if r["source"].startswith("frozen:") or (arms and r["arm"] not in arms):
            continue
        ident = identity(r["arm"], r["qid"], r["setting"])
        out[ident["rid"]] = {"question_id": r["qid"], "approach": r["arm"], "setting": r["setting"],
                             "answer_text": r["answer_text"], "error": None, "trace": {},
                             "meta": {"identity": ident}}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--arms", default=None)
    ap.add_argument("--shard", default=None)
    a = ap.parse_args()
    jdir = L.HERE / ("judge" if a.split == "test" else "dev_judge")
    if a.shard:
        jdir = jdir / a.shard
    recs = records(a.split, set(a.arms.split(",")) if a.arms else None)
    rem, sp = AN.limit_remaining(), AN.spent()
    print("records", len(recs), "limit_remaining", rem, "spent", round(sp, 4), flush=True)
    qs = JM.load_question_dicts(JQ[a.split])
    ledger = Ledger(path=str(L.HERE / ("ledger_judge_%s_%s.jsonl" % (a.split, a.shard or "all"))))
    client = AN.make_client(ledger, JPIN["judge"], JPIN["provider_pin"], 3)
    jdir.mkdir(parents=True, exist_ok=True)
    items = sorted(recs.items())
    for i in range(0, len(items), 50):
        rem, sp = AN.limit_remaining(), AN.spent()
        if rem is not None and rem < AN.FLOOR_USD or sp >= AN.CAP_USD - 0.02:
            print("STOP: money", rem, sp)
            return 1
        JM.judge_records(dict(items[i:i + 50]), qs, client, JPIN["judge"], jdir,
                         cache=KVCache(jdir / "judge_cache.sqlite"), ledger=ledger, provider_pin=JPIN["provider_pin"],
                         max_attempts=3, on_malformed="adjudicate")
    print("spent now $%.4f" % AN.spent())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
