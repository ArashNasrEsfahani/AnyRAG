"""Read-only scorer for S10's bge-m3 DENSE_6000 cells (S10_retrieval/answers.jsonl + judge/<shard>/ verdicts), using
S10's own identity rule (judge.py: experiment posthoc-2026-11-S10) and report_test.final_verdicts_by_answer, as S10's
report.py does. Frozen-answer cells are scored from the frozen verdicts."""
import json
from pathlib import Path

import s7lib as L
from ragbench.tools import judge_multi as JM
from ragbench.util import sha1_json

S10 = L.PH / "S10_retrieval"
ARM = "DENSE_6000"


def _rid(arm, qid, setting):
    ident = {"experiment_id": "posthoc-2026-11-S10", "question_id": qid, "pool_id": L.RT.pool_of(qid), "method": arm,
             "options": {"flags": {}, "variants_version": "variants-v2"},
             "reader": {"chat": "openrouter/z-ai/glm-5.3-flash", "chat_digest": "openrouter"}, "replicate": 0,
             "setting": setting}
    return sha1_json(ident)[:20]


def load():
    ans = {}
    for l in open(S10 / "answers.jsonl", encoding="utf-8"):
        if l.strip():
            r = json.loads(l)
            if not r.get("error") and r["arm"] == ARM:
                ans[r["qid"]] = r
    nv = {}
    for d in sorted(p for p in (S10 / "judge").iterdir() if p.is_dir()):
        for k, v in L.RT.final_verdicts_by_answer(d).items():
            nv.setdefault(k, v)
    if len([q for q in L.ANS if q in ans]) < 90:
        raise RuntimeError("S10 DENSE_6000 incomplete (%d answered)" % len(ans))

    def score(q):
        r = ans.get(q)
        if r is None:
            return 0, 1
        if r["source"].startswith("frozen:"):
            return L.score(r["source"].split(":", 1)[1], q)
        v = nv.get((_rid(ARM, q, r["setting"]), JM.answer_sha1({"answer_text": r["answer_text"]})))
        if v is None or v["status"] == "needs_adjudication":
            return 0, 1
        return v["strict"], v["strict"]

    return "S10_DENSE_6000", score, "S10 bge-m3 DENSE_6000 loaded from %s (%d answers)" % (S10, len(ans))
