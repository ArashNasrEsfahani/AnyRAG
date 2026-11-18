"""Judge the new answers with the pinned judge (judge_pin.json: openai/gpt-oss-120b, deepinfra/bf16, no fallbacks,
temperature 0 / top_p 1 / seed 42 from config.generation, 3 HTTP attempts, 3 identical judge attempts, on_malformed
adjudicate) through the frozen ragbench.tools.judge_multi.judge_records and the frozen judge question file.
Cells that reuse a frozen answer (source frozen:<arm>) are scored from the frozen verdicts instead (report.py).

    python judge.py [--arms A,B] [--limit-q 3]
"""
import argparse
import json

import answer as AN
import packs as PK
from ragbench.llm import Ledger
from ragbench.tools import judge_multi as JM
from ragbench.util import KVCache, sha1_json

JPIN = json.loads((PK.FZ / "judge_pin.json").read_text(encoding="utf-8"))
JDIR = PK.HERE / "judge"
EXP = "posthoc-2026-11-S2a"


def identity(arm, qid, setting):
    ident = {"experiment_id": EXP, "question_id": qid, "pool_id": PK.pool_of(qid), "method": arm,
             "options": {"flags": {}, "variants_version": "variants-v2"},
             "reader": {"chat": "openrouter/" + AN.READER, "chat_digest": "openrouter"}, "replicate": 0,
             "setting": setting}
    ident["rid"] = sha1_json(ident)[:20]
    return ident


def records(arms=None, qset=None):
    out = {}
    for r in AN.done_keys().values():
        if r["source"].startswith("frozen:"):
            continue
        if arms and r["arm"] not in arms or qset and r["qid"] not in qset:
            continue
        ident = identity(r["arm"], r["qid"], r["setting"])
        out[ident["rid"]] = {"question_id": r["qid"], "approach": r["arm"], "setting": r["setting"],
                             "answer_text": r["answer_text"], "error": None, "trace": {},
                             "meta": {"identity": ident}}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default=None)
    ap.add_argument("--limit-q", type=int, default=None)
    ap.add_argument("--shard", default=None, help="out dir judge/<shard> (one process per arm group)")
    a = ap.parse_args()
    global JDIR
    if a.shard:
        JDIR = PK.HERE / "judge" / a.shard
    qset = None
    if a.limit_q:
        qset = set([q for q in PK.qids_all() if q.endswith("T2-001")][: a.limit_q])
    recs = records(set(a.arms.split(",")) if a.arms else None, qset)
    rem, sp = AN.limit_remaining(), AN.spent()
    print("records", len(recs), "limit_remaining", rem, "spent", round(sp, 4), flush=True)
    if rem is not None and rem < AN.FLOOR_USD or sp >= AN.CAP_USD - 0.03:
        print("STOP: money")
        return 1
    qs = JM.load_question_dicts(PK.FZ / "questions" / "test_v2judge_all.jsonl")
    ledger = Ledger(path=str(AN.LEDGER if not a.shard else PK.HERE / ("ledger_judge_%s.jsonl" % a.shard)))
    client = AN.make_client(ledger, JPIN["judge"], JPIN["provider_pin"], 3)
    JDIR.mkdir(parents=True, exist_ok=True)
    if a.shard:     # seed the shard with the root dir's finished rows of these records (3-question estimate run)
        for name in ("verdicts_", "needs_adjudication_"):
            src = PK.HERE / "judge" / (name + JM._slug(JPIN["judge"]) + ".jsonl")
            dst = JDIR / src.name
            if src.is_file() and not dst.exists():
                rows = [l for l in open(src, encoding="utf-8") if l.strip() and json.loads(l)["rid"] in recs]
                dst.write_text("".join(rows), encoding="utf-8")
    # batches of <= 50 model calls with the floor check before each
    items = sorted(recs.items())
    for i in range(0, len(items), 50):
        rem, sp = AN.limit_remaining(), AN.spent()
        if rem is not None and rem < AN.FLOOR_USD or sp >= AN.CAP_USD - 0.03:
            print("STOP: money", rem, sp)
            return 1
        JM.judge_records(dict(items[i:i + 50]), qs, client, JPIN["judge"], JDIR,
                         cache=KVCache(JDIR / "judge_cache.sqlite"), ledger=ledger, provider_pin=JPIN["provider_pin"],
                         max_attempts=3, on_malformed="adjudicate")
    print("spent now $%.4f" % AN.spent())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
