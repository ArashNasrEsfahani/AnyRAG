"""S8 runs (GLM selector / reader / pinned judge 1). Every finished question writes one row to <exp>/assign.jsonl
(resumable); new answers go to <exp>/runs.jsonl (records copied from the LEX5B_6000 primary with a new identity).

  lexbud  Lex-RAG at 3000 / 10000 / 24000: the SAVED LEX5B_6000 selector reply (the selector prompt is budget-free,
          step0_checks.json) re-packed by the frozen code at B (budget_matrix pattern, only budget_chars changed).
          Pack sha == S2a FUSION_B pack -> reuse S2a's answer + verdict (all fallbacks, by step0 check d).
  nodl    Lex-RAG at 6000, the saved selector prompt re-sent to the pinned GLM selector with NO deadline (600-token cap,
          1 attempt, effort low kept); pack unchanged -> the frozen LEX5B_6000 answer + verdict.
  selhr   the Lex-RAG selector over HR_6000's saved reranked list (chunk ids in rerank order, 40 kept): frozen literal
          view of the top 24, the same selector call (NO deadline, as nodl, so the two compare candidate sources only),
          strict grammar, exact bundle packing + fill from the HR order; fallback = hybrid_intact over the HR order.
    python s8run.py lexbud|nodl|selhr [--qs N]      python s8run.py judge <exp>
"""
import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, r"D:\KareOne\RAG Paper\work\posthoc_2026-11\S8_factorial")
import s8lib as L  # noqa: E402
from ragbench.tools import judge_multi as JM  # noqa: E402

GLM = L.MAN["selector"]["model"]
S2A = L.WORK / "posthoc_2026-11" / "S2a_packs"
S2A_SHA = {}
for _r in L.jl(S2A / "packs_index.jsonl"):
    S2A_SHA[(_r["arm"], _r["qid"])] = _r["sha"]


def done(exp):
    return {(r["q"], r["budget"]) for r in L.jl(L.HERE / exp / "assign.jsonl")}


def hr_order(q):
    rec = L.primary("HR_6000", q)
    return [int(u["meta"]["chunk"]) for u in sorted(rec["retrieved_units"], key=lambda u: u["rank"])]


def sel_call(prompt, rec, tag):
    c = L.selector(GLM, deadline_s=None, max_tokens_total=600)
    with L.LEDGER.scope(phase="query", approach="LEX5B|" + tag, question_id=rec["question_id"],
                        setting=rec["setting"]):
        return L.call_selector(c, [{"role": "user", "content": prompt}], "lex5b_select", 600)


_SHA_LOCK = L._th.Lock()


def finish(exp, q, budget, rp, prim, sel, reuse, extra):
    row = dict(q=q, budget=budget, pack_sha=rp["sha"], fallback=rp["fallback"], status=rp["status"],
               reason=rp["reason"] or (sel or {}).get("reason"), parts=rp["parts"], shown_chunks=rp["shown_chunks"],
               saved_shown=prim["trace"]["lex5b"]["shown_chunks"], evidence_chars=rp["evidence_chars"],
               pack_eq_lex5b6000=rp["sha"] == prim["meta"]["answer_messages_sha256"],
               pack_eq_fusion=rp["sha"] == S2A_SHA.get(("FUSION_%d" % budget, q)),
               selector={k: v for k, v in (sel or {}).items() if k != "text"},
               raw=((sel or {}).get("text") or "")[:2000], **extra)
    if reuse is not None:
        row["answer_source"] = reuse
    else:
        r = L.answer_messages(prim, rp["messages"], rp["num_predict"], 0, L.HERE / exp / "runs.jsonl", exp,
                              method_suffix="s8_%s_b%d" % (exp, budget), extra={"pack_sha256": rp["sha"]})
        row["answer_source"] = "new"
        row["answer_rid"] = r["meta"]["identity"]["rid"]
        row["answer_sha1"] = JM.answer_sha1(r)
        row["answer_error"] = r.get("error")
    L.append(L.HERE / exp / "assign.jsonl", row)
    return row


def jobs(exp, qs):
    d, out = done(exp), []
    for q in qs:
        if exp == "lexbud":
            for b in (3000, 10000, 24000):
                if (q, b) in d:
                    continue

                def f(q=q, b=b):
                    prim = L.primary("LEX5B_6000", q)
                    s = prim["trace"]["lex5b"]["selector"]
                    failed = s.get("status") == "fail" and s.get("raw") is None
                    rp = L.lex5b_repack(prim, b, s.get("raw"), s.get("finish"), call_failed=failed,
                                        fail_reason=s.get("reason") or "deadline")
                    reuse = ("s2a:FUSION_%d" % b) if rp["sha"] == S2A_SHA.get(("FUSION_%d" % b, q)) else None
                    sel = {"saved_reply_from": "LEX5B_6000", "saved_status": s.get("status"),
                           "saved_reason": s.get("reason"), "text": s.get("raw")}
                    return finish("lexbud", q, b, rp, prim, sel, reuse, {})
                out.append(f)
        elif exp in ("nodl", "selhr"):
            if (q, 6000) in d:
                continue

            def f(q=q):
                prim = L.primary("LEX5B_6000", q)
                if exp == "nodl":
                    prompt = prim["trace"]["prompt"]
                    sel = sel_call(prompt, prim, exp)
                    rp = L.lex5b_repack(prim, 6000, sel.get("text"), sel.get("finish"),
                                        call_failed=sel.get("failed", False), fail_reason=sel.get("reason") or "x")
                    reuse = "frozen:LEX5B_6000" if rp["sha"] == prim["meta"]["answer_messages_sha256"] else None
                    extra = {"saved_status": prim["trace"]["lex5b"]["selector"].get("status"),
                             "saved_reason": prim["trace"]["lex5b"]["selector"].get("reason")}
                else:
                    order = hr_order(q)
                    prompt, meta, cap, est = L.view_prompt(prim, order)
                    sel = sel_call(prompt, prim, exp)
                    rp = L.lex5b_repack_order(prim, order, 6000, sel.get("text"), sel.get("finish"),
                                              call_failed=sel.get("failed", False),
                                              fail_reason=sel.get("reason") or "x")
                    reuse = None
                    extra = {"hr_order": order, "view_cap": cap, "view_est_tokens": round(est),
                             "prompt_chars": len(prompt)}
                return finish(exp, q, 6000, rp, prim, sel, reuse, extra)
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


def dedupe_and_run(exp, js, workers):
    for i in range(0, len(js), 20):        # <= 2 calls per job -> <= 40 calls per money check
        if not L.money_check("%s batch %d" % (exp, i)):
            print("MONEY STOP", L.spent(), flush=True)
            return 1
        with ThreadPoolExecutor(workers) as ex:
            for row in ex.map(_safe, js[i:i + 20]):
                if isinstance(row, str):
                    print("JOB ERROR", row, flush=True)
        print(exp, "batch", i, "spent %.4f" % L.spent(), flush=True)
    return 0


def judge(exp):
    d = L.HERE / exp
    while True:
        if not L.money_check("judge %s" % exp):
            print("MONEY STOP")
            return 1
        s = L.judge_dir(d, max_model_calls=45)
        print(exp, {k: v for k, v in s.items() if isinstance(v, (int, str))}, "spent %.4f" % L.spent(), flush=True)
        if not s.get("stopped_at_max_model_calls"):
            return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd")
    ap.add_argument("exp", nargs="?")
    ap.add_argument("--qs", type=int, default=None)
    ap.add_argument("--workers", type=int, default=3)
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    if a.cmd == "judge":
        raise SystemExit(judge(a.exp))
    (L.HERE / a.cmd).mkdir(parents=True, exist_ok=True)
    qs = L.QIDS[:a.qs] if a.qs else L.QIDS
    js = jobs(a.cmd, qs)
    print(a.cmd, "jobs", len(js), flush=True)
    raise SystemExit(dedupe_and_run(a.cmd, js, a.workers))
