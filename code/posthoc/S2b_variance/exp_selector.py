"""Experiments 2-5 on LEX5B (selector side). GLM answers every new pack (frozen pin/params); the pinned judge scores.
A re-packed pack whose messages sha256 equals an already-answered pack reuses that answer (and its verdict):
  exp2 (selector repeats, replicate k=1,2): GLM selector re-sent the saved prompt -> re-pack at 6000; unchanged pack ->
       the exp1 reader-replicate-k answer of LEX5B_6000 (an independent GLM draw on the identical messages); changed ->
       a new GLM answer. So replicate k re-runs everything after retrieval.
  exp3 (selector swap): another selector model (same prompt, 3 s deadline, 600-token cap, 1 attempt, effort low);
       unchanged pack -> the saved (replicate 0) answer.
  exp4 (relaxed deadline): the saved LEX5B_6000 deadline fallbacks, GLM selector with NO deadline (600 cap kept);
       unchanged pack -> saved answer.
  exp5 (shared selection): the saved 6000 reply packed at 15500, and the saved 15500 reply packed at 6000; unchanged
       pack -> the saved answer of the arm with that budget.
Every finished question writes one row to <exp>/assign.jsonl (resumable).
    python exp_selector.py exp2|exp3|exp4|exp5 [--qs N] [--workers 3] [--model M]
    python exp_selector.py judge exp2|exp3|exp4|exp5
"""
import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, r"D:\KareOne\RAG Paper\work\posthoc_2026-11\S2b_variance")
import lib as L  # noqa: E402

GLM = L.MAN["selector"]["model"]
SWAP_MODEL = "google/gemini-3.1-flash-lite"


def saved_pack_sha(arm, q):
    return L.primary(arm, q)["meta"]["answer_messages_sha256"]


def exp1_record(q, rep):
    for r in L.jl(L.HERE / "exp1_reader" / "LEX5B_6000" / "runs.jsonl"):
        if r["question_id"] == q and r["meta"]["identity"]["replicate"] == rep and not r.get("error"):
            return r
    return None


def selector_call(rec, model, deadline_s, tag):
    prompt = rec["trace"]["prompt"]
    c = L.selector(model, deadline_s=deadline_s, max_tokens_total=600)
    with L.LEDGER.scope(phase="query", approach="LEX5B|" + tag, question_id=rec["question_id"],
                        setting=rec["setting"]):
        return L.call_selector(c, [{"role": "user", "content": prompt}], "lex5b_select", 600)


def finish_one(exp, q, rep, prim_arm, budget, sel, src_reuse, extra):
    """Re-pack with the selector outcome, answer when the pack is new; write the assign row."""
    prim = L.primary(prim_arm, q)
    rp = L.lex5b_repack(prim, budget, sel.get("text"), sel.get("finish"), call_failed=sel.get("failed", False),
                        fail_reason=sel.get("reason") or "deadline")
    same = rp["sha"] == prim["meta"]["answer_messages_sha256"]
    row = dict(q=q, rep=rep, arm=prim_arm, budget=budget, pack_sha=rp["sha"], pack_unchanged=same,
               fallback=rp["fallback"], status=rp["status"], reason=rp["reason"] or sel.get("reason"),
               parts=rp["parts"], shown_chunks=rp["shown_chunks"],
               saved_shown=prim["trace"]["lex5b"]["shown_chunks"], evidence_chars=rp["evidence_chars"],
               selector={k: v for k, v in sel.items() if k != "text"}, raw=(sel.get("text") or "")[:2000], **extra)
    if same and src_reuse is not None:
        row["answer_source"] = src_reuse[0]
        row["answer_rid"] = src_reuse[1]["meta"]["identity"]["rid"]
        row["answer_sha1"] = __import__("ragbench.tools.judge_multi", fromlist=["x"]).answer_sha1(src_reuse[1])
    else:
        out = L.HERE / exp / "runs.jsonl"
        r = L.answer_messages(prim, rp["messages"], rp["num_predict"], rep, out, exp,
                              method_suffix="%s_b%d" % (exp, budget), extra={"pack_sha256": rp["sha"]})
        row["answer_source"] = "new"
        row["answer_rid"] = r["meta"]["identity"]["rid"]
        row["answer_error"] = r.get("error")
    L.append(L.HERE / exp / "assign.jsonl", row)
    return row


def done(exp):
    return {(r["q"], r["rep"], r["budget"]) for r in L.jl(L.HERE / exp / "assign.jsonl")}


def jobs(exp, qs, model):
    """-> list of callables (one per question x replicate / budget), each a full step."""
    out = []
    d = done(exp)
    if exp == "exp2":
        for rep in (1, 2):
            for q in qs:
                if (q, rep, 6000) in d:
                    continue

                def f(q=q, rep=rep):
                    rec = L.primary("LEX5B_6000", q)
                    sel = selector_call(rec, GLM, 3.0, "exp2")
                    e1 = exp1_record(q, rep)
                    return finish_one("exp2", q, rep, "LEX5B_6000", 6000, sel,
                                      ("exp1_rep%d" % rep, e1) if e1 else None, {"selector_model": GLM})
                out.append(f)
    elif exp in ("exp3", "exp4"):
        for q in qs:
            if (q, 0, 6000) in d:
                continue
            rec = L.primary("LEX5B_6000", q)
            s = rec["trace"]["lex5b"]["selector"]
            if exp == "exp4" and s.get("reason") != "deadline":
                continue
            m, dl = (model, 3.0) if exp == "exp3" else (GLM, None)

            def f(q=q, m=m, dl=dl):
                rec = L.primary("LEX5B_6000", q)
                sel = selector_call(rec, m, dl, exp)
                return finish_one(exp, q, 0, "LEX5B_6000", 6000, sel, ("r0", rec),
                                  {"selector_model": m, "deadline_s": dl})
            out.append(f)
    elif exp == "exp5":
        for q in qs:
            for src_arm, dst_arm, budget in (("LEX5B_6000", "LEX5B_15500", 15500), ("LEX5B_15500", "LEX5B_6000", 6000)):
                if (q, 0, budget) in d:
                    continue

                def f(q=q, src_arm=src_arm, dst_arm=dst_arm, budget=budget):
                    src = L.primary(src_arm, q)
                    s = src["trace"]["lex5b"]["selector"]
                    failed = s.get("status") == "fail" and s.get("raw") is None
                    sel = {"text": s.get("raw"), "finish": s.get("finish"), "failed": failed,
                           "reason": s.get("reason"), "saved_from": src_arm}
                    return finish_one("exp5", q, 0, dst_arm, budget, sel, ("r0", L.primary(dst_arm, q)),
                                      {"reply_from": src_arm})
                out.append(f)
    return out


def _safe(f):
    import traceback
    try:
        return f()
    except L.BudgetExceeded:
        raise
    except Exception:  # noqa: BLE001
        return traceback.format_exc()[-1500:]


def run(exp, n_q, workers, model):
    (L.HERE / exp).mkdir(parents=True, exist_ok=True)
    qs = L.QIDS[:n_q] if n_q else L.QIDS
    js = jobs(exp, qs, model)
    print(exp, "jobs", len(js), flush=True)
    # each job makes at most 2 calls -> 24 jobs per money check
    for i in range(0, len(js), 24):
        if not L.money_check("%s batch %d" % (exp, i)):
            print("MONEY STOP", L.spent())
            return 1
        with ThreadPoolExecutor(workers) as ex:
            for row in ex.map(_safe, js[i:i + 24]):
                if isinstance(row, str):
                    print("JOB ERROR", row, flush=True)
        print("batch", i, "spent %.4f" % L.spent(), flush=True)
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd")
    ap.add_argument("exp", nargs="?")
    ap.add_argument("--qs", type=int, default=None)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--model", default=SWAP_MODEL)
    a = ap.parse_args()
    if a.cmd == "judge":
        import exp1_reader_repeats as E1
        raise SystemExit(E1.judge(1, 40, [L.HERE / a.exp]))
    raise SystemExit(run(a.cmd, a.qs, a.workers, a.model))
