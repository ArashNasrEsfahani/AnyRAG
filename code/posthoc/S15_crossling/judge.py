"""Step 5: judge S15 answers with the pinned judge through the frozen ragbench.tools.judge_multi.judge_records and
the frozen judge question file (as S10/judge.py). The record's setting is the question's ORIGINAL setting (fa-fa or
en-en), so the judge sees the original question text and the same reference packet as for the frozen arms; only the
candidate answer differs (the v2 rubric accepts either language).

    python judge.py [--split test|dev] [--arms A,B]
"""
import argparse
import json

import s15lib as L
import s15call as C
from ragbench.llm import Ledger
from ragbench.tools import judge_multi as JM
from ragbench.util import KVCache, sha1_json

EXP = "posthoc-2026-11-S15"
JQ = {"test": L.FZ / "questions" / "test_v2judge_all.jsonl", "dev": L.DEVQ / "dev_v2judge_all.jsonl"}


def identity(arm, qid, setting):
    ident = {"experiment_id": EXP, "question_id": qid, "pool_id": L.S10.pool_of(qid), "method": arm,
             "options": {"flags": {}, "variants_version": "variants-v2"},
             "reader": {"chat": "openrouter/" + C.READER, "chat_digest": "openrouter"}, "replicate": 0,
             "setting": setting}
    ident["rid"] = sha1_json(ident)[:20]
    return ident


def records(split, arms=None):
    out = {}
    p = L.HERE / ("answers.jsonl" if split == "test" else "dev_answers.jsonl")
    for r in L.jl(p):
        if r.get("error") or (arms and r["arm"] not in arms):
            continue
        st = L.orig_query(r["qid"], split).setting
        ident = identity(r["arm"], r["qid"], st)
        out[ident["rid"]] = {"question_id": r["qid"], "approach": r["arm"], "setting": st,
                             "answer_text": r["answer_text"], "error": None, "trace": {},
                             "meta": {"identity": ident}}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--arms", default=None)
    a = ap.parse_args()
    jdir = L.HERE / ("judge" if a.split == "test" else "dev_judge")
    recs = records(a.split, set(a.arms.split(",")) if a.arms else None)
    print("records", len(recs), flush=True)
    qs = JM.load_question_dicts(JQ[a.split])
    ledger = Ledger(path=str(L.HERE / ("ledger_judge_%s.jsonl" % a.split)))
    client = C.make_client(ledger, C.JPIN["judge"], C.JPIN["provider_pin"], 3)
    jdir.mkdir(parents=True, exist_ok=True)
    items = sorted(recs.items())
    for i in range(0, len(items), 45):
        if not C.guard():
            return 1
        JM.judge_records(dict(items[i:i + 45]), qs, client, C.JPIN["judge"], jdir,
                         cache=KVCache(jdir / "judge_cache.sqlite"), ledger=ledger, provider_pin=C.JPIN["provider_pin"],
                         max_attempts=3, on_malformed="adjudicate")
    print("spent now $%.4f" % C.spent())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
