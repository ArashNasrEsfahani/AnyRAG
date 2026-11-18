"""S8 step 5 (dev replication): FUSION_6000 (the Lex-RAG fallback for every question = hybrid_intact over the saved
F2 top 40, frozen code, no selector call) on the 60 dev questions, from the saved dev LEX5B@6000 traces
(lex5b/eval/runs for the idea_test 24, dev60/runs for the other 36: the records dev60 scored).
Gate (no calls): the rebuilt fallback chunks == the saved trace's fallback_chunks, and in the saved selector-fallback
cells the rebuilt shown chunks / evidence chars == the saved ones. Then a fresh GLM answer per question (frozen pin)
and the pinned judge 1 with the dev judge question file.
    python dev_fusion.py gate|answer|judge
"""
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, r"D:\KareOne\RAG Paper\work\posthoc_2026-11\S8_factorial")
import s8lib as L  # noqa: E402

DS = L.WORK / "datasets_v2"
OUT = L.HERE / "dev_fusion"
DEVQ = DS / "devrun" / "questions"


def dev_records():
    out = {}
    for f in sorted(DS.glob("lex5b/eval/runs/*/LEX5B/runs_*.jsonl")) + sorted(DS.glob("dev60/runs/*/LEX5B_6000/runs_*.jsonl")):
        for r in L.jl(f):
            if not r.get("error"):
                out[r["question_id"]] = r
    return out


def load_dev_questions():
    from ragbench.contracts import load_questions
    L.question_obj(L.QIDS[0])                     # fill the test cache first (it is filled only when empty)
    for f in DEVQ.glob("dev_v2run_*-*.jsonl"):
        for q in load_questions(f):
            L._QOBJ[q.id] = q


def gate():
    load_dev_questions()
    R = dev_records()
    rows = []
    for q, r in sorted(R.items()):
        st = r["trace"]["lex5b"]
        rp = L.lex5b_repack(r, 6000, None, None, call_failed=True)
        fb = st["selector"].get("fallback", False)
        rows.append({"q": q, "fallback_match": rp["fallback_chunks"] == st["fallback_chunks"], "saved_fallback": fb,
                     "shown_match": (rp["shown_chunks"] == st["shown_chunks"]) if fb else None,
                     "chars_match": (rp["evidence_chars"] == r["evidence_chars"]) if fb else None,
                     "pack_same_as_lex5b": rp["shown_chunks"] == st["shown_chunks"]})
    s = {"n": len(rows), "fallback_match": sum(x["fallback_match"] for x in rows),
         "saved_fallback": sum(bool(x["saved_fallback"]) for x in rows),
         "fallback_cells_shown_match": sum(bool(x["shown_match"]) for x in rows),
         "fallback_cells_chars_match": sum(bool(x["chars_match"]) for x in rows),
         "pack_same_as_lex5b": sum(x["pack_same_as_lex5b"] for x in rows), "rows": rows}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "gate.json").write_text(json.dumps(s, indent=1), encoding="utf-8")
    print({k: v for k, v in s.items() if k != "rows"})


def answer():
    load_dev_questions()
    R = dev_records()
    done = {x["q"] for x in L.jl(OUT / "assign.jsonl")}
    todo = [q for q in sorted(R) if q not in done]

    def f(q):
        r = R[q]
        rp = L.lex5b_repack(r, 6000, None, None, call_failed=True)
        rec = L.answer_messages(r, rp["messages"], rp["num_predict"], 0, OUT / "runs.jsonl", "devfusion",
                                method_suffix="s8_devfusion_b6000", extra={"pack_sha256": rp["sha"]})
        from ragbench.tools import judge_multi as JM
        L.append(OUT / "assign.jsonl", {"q": q, "pack_sha": rp["sha"], "shown_chunks": rp["shown_chunks"],
                                        "evidence_chars": rp["evidence_chars"],
                                        "pack_same_as_lex5b": rp["shown_chunks"] == r["trace"]["lex5b"]["shown_chunks"],
                                        "answer_rid": rec["meta"]["identity"]["rid"], "answer_sha1": JM.answer_sha1(rec),
                                        "answer_error": rec.get("error")})
    for i in range(0, len(todo), 30):
        if not L.money_check("devfusion %d" % i):
            print("MONEY STOP")
            return 1
        with ThreadPoolExecutor(3) as ex:
            list(ex.map(f, todo[i:i + 30]))
        print("batch", i, "spent %.4f" % L.spent(), flush=True)


def judge():
    from ragbench.tools import judge_multi as JM
    from ragbench.util import KVCache
    j = L.MAN["judge"]
    qs = JM.load_question_dicts(DEVQ / "dev_v2judge_all.jsonl")
    recs = L.records_by_rid(OUT)
    jd = OUT / "judge"
    while True:
        if not L.money_check("devfusion judge"):
            print("MONEY STOP")
            return 1
        s = JM.judge_records(recs, qs, L.judge_client(), j["model"], jd, cache=KVCache(jd / "judge_multi_cache.sqlite"),
                             ledger=L.LEDGER, provider_pin=j["provider_pin"], max_attempts=j["max_judge_attempts"],
                             on_malformed=j["on_malformed"], max_model_calls=45)
        print({k: v for k, v in s.items() if isinstance(v, int)}, flush=True)
        if not s.get("stopped_at_max_model_calls"):
            return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit({"gate": gate, "answer": answer, "judge": judge}[sys.argv[1]]() or 0)
