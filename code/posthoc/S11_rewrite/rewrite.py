"""S11 step 1 (API only, no GPU): GLM-5.3-flash writes the query rewrites.
    python rewrite.py dev  --ids A,B,...           (at most 5 dev questions)
    python rewrite.py test --confirm-test          (once, all 96)
-> rewrites_<split>.jsonl, one line per (question, method) with the raw text and the parsed rewrite(s)."""
import argparse
import json
import sys

import s11lib as L

DEV5 = ["V2-dev-RPT-fa-T2-001", "V2-dev-RPT-en-T3-002", "V2-dev-BOOK-fa-T3-001", "V2-dev-BOOK-en-T3-002",
        "V2-dev-LAW-fa-T2-002"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("split", choices=("dev", "test"))
    ap.add_argument("--ids", default=None)
    ap.add_argument("--confirm-test", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    if a.split == "test" and not a.confirm_test:
        raise SystemExit("REFUSED: test needs --confirm-test")
    ids = set(a.ids.split(",")) if a.ids else (set(DEV5) if a.split == "dev" else None)
    if a.split == "dev" and len(ids) > 5:
        raise SystemExit("dev: at most 5 questions")
    out = L.HERE / ("rewrites_%s.jsonl" % a.split)
    done = {(r["qid"], r["method"]) for r in L.jl(out) if r.get("ok")}
    plan = [(q, m) for q in L.queries(a.split, ids) for m in ("HYDE", "MQ") if (q.question_id, m) not in done]
    if a.limit:
        plan = plan[:a.limit]
    for n, (q, m) in enumerate(plan):
        if n % 40 == 0:
            rem = L.key_remaining()
            print("key remaining $%.2f, stream spent $%.4f" % (rem, L.spent()), flush=True)
            if rem is not None and rem < L.FLOOR_USD:
                print("GLOBAL FLOOR")
                return 1
        msgs = L.rewrite_messages(m, q)
        try:
            res = L.glm(msgs, purpose="rewrite-" + m, max_tokens=L.REWRITE_MAX_TOKENS,
                        ctx={"arm": m, "question_id": q.question_id, "split": a.split})
        except L.Stop as e:
            print("STOP", e)
            return 1
        except Exception as e:  # noqa: BLE001
            print(q.question_id, m, "ERROR", e)
            continue
        txt = res["text"]
        rw = [txt.strip()] if m == "HYDE" else L.parse_mq(txt)
        ok = bool(rw and all(rw)) and res["finish"] == "stop"
        row = {"qid": q.question_id, "method": m, "split": a.split, "pool": q.pool_id, "lang": q.q_lang,
               "question": q.text, "prompt": msgs[0]["content"], "raw": txt, "rewrites": rw, "finish": res["finish"],
               "ok": ok, "prompt_tokens": res["row"]["prompt_tokens"],
               "completion_tokens": res["row"]["completion_tokens"], "cost_usd": res["row"]["cost_usd"]}
        with open(out, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(q.question_id, m, ok, res["row"]["completion_tokens"], "|", " || ".join(x[:80] for x in rw), flush=True)
    print("spent $%.4f" % L.spent())
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
