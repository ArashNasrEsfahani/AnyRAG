"""Step 4: answer calls with the pinned GLM-5.3-flash reader (frozen OpenRouterReader, manifest pin and attempts),
exact messages and num_predict as built by the frozen build_answer_messages / answer_num_predict.

    HR_6000_TRANS     packs/HR_6000_trans/<qid>.json (translated query; answer language = the language asked)
    CLOSEDBOOK_TRANS  build_answer_messages(swapped_query, None) - S6's CLOSEDBOOK prompt with the translated question
    HYB_6000_P:<v>    (only if experiment P moves recall by >= 5 bundles) the HYB pack of variant v

    python answer.py --arms HR_6000_TRANS,CLOSEDBOOK_TRANS [--split test|dev] [--ids ...]
Append-only answers.jsonl (dev: dev_answers.jsonl) keyed (arm, qid).
"""
import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor

import s15lib as L
import s15call as C
from ragbench.contracts import answer_num_predict, build_answer_messages


def cells(arm, split, ids):
    out = []
    if split == "test":
        tr = {r["qid"]: r for r in L.jl(L.HERE / "translations.jsonl") if r["direction"] == "forward" and not r.get("error")}
    else:
        tr = {r["qid"]: r for r in L.jl(L.HERE / "dev_translations.jsonl") if r["direction"] == "forward" and not r.get("error")}
    for qid in ids:
        if qid not in tr:
            continue
        q = L.swapped_query(qid, tr[qid]["translation"], split)
        if arm == "HR_6000_TRANS":
            p = L.HERE / ("packs" if split == "test" else "dev_packs") / "HR_6000_trans" / (qid + ".json")
            if not p.is_file():
                continue
            pk = json.loads(p.read_text(encoding="utf-8"))
            msgs, npred = pk["messages"], pk["num_predict"]
            assert L.S10.sha256_msgs(msgs) == pk["messages_sha256"]
        elif arm == "CLOSEDBOOK_TRANS":
            msgs, npred = build_answer_messages(q, None, L.CFG), answer_num_predict(q, L.CFG)
        elif arm.startswith("HYB_6000_P:"):
            v = arm.split(":", 1)[1]
            r = [x for x in L.jl(L.HERE / "rows.jsonl") if x["exp"] == "P" and x["cond"] == v and x["arm"] == "HYB"
                 and x["qid"] == qid]
            if not r:
                continue
            q = L.text_query(qid, r[0]["query"], split)
            items = None  # rebuilt from the saved list order
            from retrieve import items_of
            hits = L.search(q, ("bm25", "dense"), L.vec(L.normalize(q.text)))
            items = items_of(hits)
            assert [it.meta["chunk"] for it in items][:40] == r[0]["list40"]
            ev, msgs, npred = L.pack6000(items, q)
            assert L.S10.sha256_msgs(msgs) == r[0]["sha"]
        else:
            raise ValueError(arm)
        out.append({"arm": arm, "qid": qid, "setting_asked": q.setting, "msgs": msgs, "num_predict": npred,
                    "sha": L.S10.sha256_msgs(msgs)})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--ids", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    out = L.HERE / ("answers.jsonl" if a.split == "test" else "dev_answers.jsonl")
    ids = a.ids.split(",") if a.ids else L.qids(a.split)
    done = {(r["arm"], r["qid"]) for r in L.jl(out) if not r.get("error")}
    client, led = C.reader()
    todo = []
    for arm in a.arms.split(","):
        todo += [c for c in cells(arm, a.split, ids) if (c["arm"], c["qid"]) not in done]
    if a.limit:
        todo = todo[:a.limit]
    print("to call:", len(todo), flush=True)

    def call(c):
        t0 = time.perf_counter()
        base = {k: c[k] for k in ("arm", "qid", "setting_asked", "sha", "num_predict")}
        with led.scope(phase="query", approach="S15:" + c["arm"], question_id=c["qid"], setting=c["setting_asked"]):
            try:
                r = client.chat(c["msgs"], purpose="answer", num_predict=c["num_predict"])
                rec = dict(base, answer_text=r.text.strip(), error=None, done_reason=r.done_reason,
                           prompt_tokens=r.prompt_tokens, completion_tokens=r.completion_tokens,
                           reasoning_tokens=getattr(r, "reasoning_tokens", None), cost_usd=r.cost_usd,
                           provider=getattr(r, "provider", None))
            except Exception as e:  # noqa: BLE001
                rec = dict(base, answer_text="", error="%s: %s" % (type(e).__name__, str(e)[:300]))
        rec["latency_ms"] = round((time.perf_counter() - t0) * 1000.0, 1)
        L.append(out, rec)
        return rec

    for i in range(0, len(todo), 50):
        if not C.guard():
            return 1
        with ThreadPoolExecutor(a.workers) as ex:
            for rec in ex.map(call, todo[i:i + 50]):
                if rec.get("error"):
                    print("ERROR", rec["arm"], rec["qid"], rec["error"][:150], flush=True)
    print("spent $%.4f" % C.spent())
    return 0


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
