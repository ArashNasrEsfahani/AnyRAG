"""Judge S6 records with the pinned test judge (judge_pin.json / MANIFEST judge: gpt-oss-120b, deepinfra/bf16, no
fallbacks, temperature 0 / top_p 1 / seed 42 from the frozen config, max 3 identical judge attempts, on_malformed
adjudicate, 3 HTTP attempts) through the frozen ragbench.tools.judge_multi.judge_records (same rubric/prompt code as
`run_test.py judge`). Only the client's budget file is stream-local (s6call.LocalBudget).

    python judge_arms.py --split dev|test --arms CLOSEDBOOK,LISTWISE,SETR_NOFILL
Out: <split>/judge/verdicts_*.jsonl, needs_adjudication_*.jsonl, judge_multi_cache.sqlite, judge_runs.jsonl."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import s6lib as L  # noqa: E402
import s6call as C  # noqa: E402
from ragbench.llm import BudgetExceeded  # noqa: E402
from ragbench.tools import judge_multi as JM  # noqa: E402
from ragbench.util import KVCache  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=("dev", "test"))
    ap.add_argument("--arms", required=True)
    ap.add_argument("--out", default="judge", help="out dir name under <split>/ (one per parallel process)")
    a = ap.parse_args()
    j = L.MAN["judge"]
    qf = (L.FZ / "questions" / "test_v2judge_all.jsonl") if a.split == "test" else \
        (L.DS / "devrun" / "questions" / "dev_v2judge_all.jsonl")
    qs = JM.load_question_dicts(qf)
    recs, skipped = JM.latest_identity_records([L.OUT / a.split / arm for arm in a.arms.split(",")])
    out = L.OUT / a.split / a.out
    out.mkdir(parents=True, exist_ok=True)
    client = C.judge_client()
    with open(out / "judge_runs.jsonl", "a", encoding="utf-8") as fh:
        fh.write(json.dumps({"judge": j["model"], "judge_provider_pin": j["provider_pin"], "arms": a.arms,
                             "records": len(recs), "http_attempts": client.or_retries + 1,
                             "max_judge_attempts": j["max_judge_attempts"], "on_malformed": j["on_malformed"],
                             "generation": {k: L.CFG.generation.get(k) for k in ("temperature", "top_p", "seed")},
                             "sent_params": sorted(k for k in ("temperature", "top_p", "seed")
                                                   if not client.supported or k in client.supported),
                             "questions": str(qf)}) + "\n")
    items = sorted(recs.items())
    tot = {}
    try:
        for i in range(0, len(items), 16):           # <= 48 judge calls (3 attempts max) per floor check
            rem = C.floor_check()
            print("floor ok %s; S6 spent $%.4f" % (rem, C.spent()), flush=True)
            s = JM.judge_records(dict(items[i:i + 16]), qs, client, j["model"], out,
                                 cache=KVCache(out / "judge_multi_cache.sqlite"), ledger=C.LEDGER,
                                 provider_pin=j["provider_pin"], max_attempts=j["max_judge_attempts"],
                                 on_malformed=j["on_malformed"], log=lambda m: print(m, flush=True))
            for k, v in s.items():
                if isinstance(v, int):
                    tot[k] = tot.get(k, 0) + v
    except BudgetExceeded as e:
        print("BUDGET STOP:", e)
        return 3
    print(json.dumps(tot))
    print("S6 spent $%.4f" % C.spent())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
