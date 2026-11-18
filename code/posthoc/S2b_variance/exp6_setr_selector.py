"""Experiment 2b (SETR selector repeats, SETR_6000): the saved SETR selector prompt (rebuilt with the frozen
build_prompt from the saved order40; checked against the saved prompt_chars) re-sent to GLM with the frozen SETR call
settings (pinned, 1 attempt, 8 s deadline, 2000-token cap, effort low); re-packed with the frozen SETR packer.
Unchanged pack -> the exp1 reader-replicate-k answer of SETR_6000; changed -> a new GLM answer. Rows: exp6/assign.jsonl.
    python exp6_setr_selector.py --reps 1 [--qs N]
"""
import argparse, sys
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, r"D:\KareOne\RAG Paper\work\posthoc_2026-11\S2b_variance")
import lib as L, setr_lib as S, exp_selector as ES  # noqa: E401,E402
from ragbench.approaches.setr.approach import SETR_DEFAULTS  # noqa: E402
from ragbench.tools import judge_multi as JM  # noqa: E402

EXP = "exp6"


def exp1_rec(q, rep):
    for r in L.jl(L.HERE / "exp1_reader" / "SETR_6000" / "runs.jsonl"):
        if r["question_id"] == q and r["meta"]["identity"]["replicate"] == rep and not r.get("error"):
            return r


def job(q, rep):
    prim = L.primary("SETR_6000", q)
    s0 = prim["trace"]["setr"]["selector"]
    base = S.setr_repack(prim, 6000, s0.get("raw"), call_failed=s0.get("raw") is None)
    msgs = base["selector_messages"]
    assert sum(len(m["content"]) for m in msgs) == s0["prompt_chars"], q
    c = L.selector(SETR_DEFAULTS["selector"], deadline_s=SETR_DEFAULTS["deadline_s"],
                   max_tokens_total=SETR_DEFAULTS["max_tokens_total"])
    with L.LEDGER.scope(phase="query", approach="SETR|" + EXP, question_id=q, setting=prim["setting"]):
        sel = L.call_selector(c, msgs, "setr_select", SETR_DEFAULTS["max_tokens_total"])
    rp = S.setr_repack(prim, 6000, sel.get("text"), call_failed=sel.get("failed", False))
    same = rp["sha"] == prim["meta"]["answer_messages_sha256"]
    row = dict(q=q, rep=rep, arm="SETR_6000", budget=6000, pack_sha=rp["sha"], pack_unchanged=same,
               fallback=rp["fallback"], status=rp["status"], reason=sel.get("reason") or (None if rp["selected"] else rp["status"]),
               selected=rp["selected"], saved_selected=s0.get("selected") or [], offered=rp["offered"],
               saved_offered=prim["trace"]["setr"]["offered"], shown_chunks=rp["offered"], saved_shown=prim["trace"]["setr"]["offered"],
               selector={k: v for k, v in sel.items() if k != "text"}, raw=(sel.get("text") or "")[:4000])
    e1 = exp1_rec(q, rep)
    if same and e1 is not None:
        row.update(answer_source="exp1_rep%d" % rep, answer_rid=e1["meta"]["identity"]["rid"], answer_sha1=JM.answer_sha1(e1))
    else:
        r = L.answer_messages(prim, rp["messages"], rp["num_predict"], rep, L.HERE / EXP / "runs.jsonl", EXP,
                              method_suffix="%s_b6000" % EXP, extra={"pack_sha256": rp["sha"]})
        row.update(answer_source="new", answer_rid=r["meta"]["identity"]["rid"], answer_error=r.get("error"))
    L.append(L.HERE / EXP / "assign.jsonl", row)
    return row


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", default="1")
    ap.add_argument("--qs", type=int, default=None)
    a = ap.parse_args()
    (L.HERE / EXP).mkdir(exist_ok=True)
    done = {(r["q"], r["rep"]) for r in L.jl(L.HERE / EXP / "assign.jsonl")}
    qs = L.QIDS[:a.qs] if a.qs else L.QIDS
    js = [(q, int(rep)) for rep in a.reps.split(",") for q in qs if (q, int(rep)) not in done]
    print("jobs", len(js), flush=True)
    for i in range(0, len(js), 24):
        if not L.money_check("exp6 batch %d" % i):
            print("MONEY STOP"); raise SystemExit(1)
        with ThreadPoolExecutor(3) as ex:
            for row in ex.map(lambda t: ES._safe(lambda: job(*t)), js[i:i + 24]):
                if isinstance(row, str):
                    print("JOB ERROR", row, flush=True)
        print("batch", i, "spent %.4f" % L.spent(), flush=True)
