"""Retry the judge cells whose calls failed at the network level (no reply; judge_failures_*.jsonl) with the frozen
judge_records(retry_failures=True); same pin/settings as judge.py."""
import json
import answer as AN
import judge as JG
import packs as PK
from ragbench.llm import Ledger
from ragbench.tools import judge_multi as JM
from ragbench.util import KVCache

fails = {json.loads(l)["rid"] for l in open(JG.JDIR / "judge_failures_openai__gpt-oss-120b.jsonl", encoding="utf-8") if l.strip()}
recs = {k: v for k, v in JG.records().items() if k in fails}
print("retry", len(recs), "limit_remaining", AN.limit_remaining(), "spent", round(AN.spent(), 4))
qs = JM.load_question_dicts(PK.FZ / "questions" / "test_v2judge_all.jsonl")
ledger = Ledger(path=str(AN.LEDGER))
client = AN.make_client(ledger, JG.JPIN["judge"], JG.JPIN["provider_pin"], 3)
JM.judge_records(recs, qs, client, JG.JPIN["judge"], JG.JDIR, retry_failures=True,
                 cache=KVCache(JG.JDIR / "judge_cache.sqlite"), ledger=ledger, provider_pin=JG.JPIN["provider_pin"],
                 max_attempts=3, on_malformed="adjudicate")
