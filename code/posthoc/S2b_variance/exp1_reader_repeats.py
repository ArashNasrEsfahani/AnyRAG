"""Experiment 1: reader repeats. The identical saved pack messages of every primary cell (8 arms x 96 questions) are
sent again to GLM-5.3-flash (frozen pin + params, reader_replay.make_reader), replicates 1 and 2; then judged with the
pinned judge. Resumable: a (arm, q, replicate) with a non-error record is never sent again; an error may be re-sent
once.
    python exp1_reader_repeats.py answer [--qs N] [--reps 1,2] [--workers 6]
    python exp1_reader_repeats.py judge [--workers 8]
"""
import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, r"D:\KareOne\RAG Paper\work\posthoc_2026-11\S2b_variance")
import lib as L  # noqa: E402

OUT = L.HERE / "exp1_reader"


def done_state(arm):
    f = OUT / arm / "runs.jsonl"
    st = {}
    for r in L.jl(f):
        k = (r["question_id"], r["meta"]["identity"]["replicate"])
        ok, n = st.get(k, (False, 0))
        st[k] = (ok or not r.get("error"), n + 1)
    return st


def task(arm, q, rep):
    prim = L.primary(arm, q)
    pk = json.loads(L.pack_path(arm, prim).read_text(encoding="utf-8"))
    assert pk["messages_sha256"] == prim["meta"]["answer_messages_sha256"] == L.msha(pk["messages"])
    (OUT / arm).mkdir(parents=True, exist_ok=True)
    r = L.answer_messages(prim, pk["messages"], pk["num_predict"], rep, OUT / arm / "runs.jsonl", "exp1",
                          extra={"pack_sha256": pk["messages_sha256"]})
    return arm, q, rep, r.get("error")


def answer(n_q, reps, workers):
    qs = L.QIDS[:n_q] if n_q else L.QIDS
    todo = []
    for arm in L.ARMS:
        st = done_state(arm)
        for q in qs:
            for rep in reps:
                ok, n = st.get((q, rep), (False, 0))
                if not ok and n < 2:
                    todo.append((arm, q, rep))
    print("todo", len(todo), flush=True)
    for i in range(0, len(todo), 48):
        if not L.money_check("exp1 answer batch %d" % i):
            print("MONEY STOP", L.spent())
            return 1
        with ThreadPoolExecutor(workers) as ex:
            for arm, q, rep, err in ex.map(lambda t: task(*t), todo[i:i + 48]):
                if err:
                    print("ERR", arm, q, rep, err[:100], flush=True)
        print("batch", i, "spent %.4f" % L.spent(), flush=True)
    return 0


def judge(workers, max_calls=6, dirs=None):
    """rounds of at most max_calls judge-model calls per dir (8 dirs x 6 = 48 <= 50), a money check before each."""
    dirs = dirs or [OUT / a for a in L.ARMS]
    while True:
        if not L.money_check("judge round"):
            print("MONEY STOP")
            return 1
        with ThreadPoolExecutor(workers) as ex:
            ss = list(ex.map(lambda d: L.judge_dir(d, max_model_calls=max_calls), dirs))
        n = sum(s["model"] for s in ss)
        print("judge round: model calls %d, needs_adj %d, failures %d, spent %.4f" % (
            n, sum(s["needs_adjudication"] for s in ss), sum(s["failures"] for s in ss), L.spent()), flush=True)
        if n == 0:
            return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd")
    ap.add_argument("--qs", type=int, default=None)
    ap.add_argument("--reps", default="1,2")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--max-calls", type=int, default=None)
    a = ap.parse_args()
    if a.cmd == "answer":
        raise SystemExit(answer(a.qs, [int(x) for x in a.reps.split(",")], a.workers))
    raise SystemExit(judge(a.workers, a.max_calls or 6))
