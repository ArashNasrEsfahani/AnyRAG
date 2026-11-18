"""Step 1: translate questions into the other language with GLM-5.3-flash (pinned as the reader), plain faithful
translation prompt, no answer hints; then back-translate (check only, not used for retrieval).

    python translate.py --split dev --ids a,b,c      -> dev_translations.jsonl (prompt fixing)
    python translate.py --split test                 -> translations.jsonl (96 questions, forward + back)
Append-only, keyed (qid, direction); a rerun skips finished cells.
"""
import argparse
import re
import time
from concurrent.futures import ThreadPoolExecutor

import s15lib as L
import s15call as C

PROMPT_VERSION = "s15-translate-v1"
SYSTEM = "You are a careful professional translator between Persian and English."
USER = ("Translate this {src} question into {tgt}.\n"
        "Rules: translate faithfully and completely, as a native {tgt} speaker would ask it. Keep every name, number, "
        "date, year and unit as given (do not convert calendars, currencies or units). Do not answer the question and "
        "do not add, remove or explain anything. Reply with the translation only.\n\n"
        "Question: {text}")
NUM_PREDICT = 400


def messages(text, src, tgt):
    return [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": USER.format(src=L.LANG_NAME[src], tgt=L.LANG_NAME[tgt], text=text.strip())}]


def clean(t):
    t = (t or "").strip()
    t = re.sub(r"^(question|translation|پرسش|سؤال|سوال|ترجمه)\s*[:：]\s*", "", t, flags=re.I).strip()
    if len(t) >= 2 and t[0] in "\"'«“" and t[-1] in "\"'»”":
        t = t[1:-1].strip()
    return t


def numbers(t):
    return sorted(re.findall(r"\d+(?:[.,/]\d+)*", L.normalize(t or "")))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--ids", default=None)
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    out = L.HERE / ("translations.jsonl" if a.split == "test" else "dev_translations.jsonl")
    ids = a.ids.split(",") if a.ids else L.qids(a.split)
    done = {(r["qid"], r["direction"]): r for r in L.jl(out) if not r.get("error")}
    client, led = C.reader()

    def call(item):
        qid, direction, text, src, tgt = item
        msgs = messages(text, src, tgt)
        t0 = time.perf_counter()
        with led.scope(phase="query", approach="S15:translate:" + direction, question_id=qid, setting=src + "-" + tgt):
            try:
                r = client.chat(msgs, purpose="translate", num_predict=NUM_PREDICT)
                rec = dict(qid=qid, direction=direction, src_lang=src, tgt_lang=tgt, source=text, raw=r.text,
                           translation=clean(r.text), prompt_version=PROMPT_VERSION, done_reason=r.done_reason,
                           prompt_tokens=r.prompt_tokens, completion_tokens=r.completion_tokens, cost_usd=r.cost_usd,
                           provider=getattr(r, "provider", None), error=None)
            except Exception as e:  # noqa: BLE001
                rec = dict(qid=qid, direction=direction, src_lang=src, tgt_lang=tgt, source=text,
                           error="%s: %s" % (type(e).__name__, str(e)[:300]))
        rec["latency_ms"] = round((time.perf_counter() - t0) * 1000.0, 1)
        L.append(out, rec)
        return rec

    for direction in ("forward", "back"):
        todo = []
        for qid in ids:
            if (qid, direction) in done:
                continue
            src = L.lang_of(qid)
            if direction == "forward":
                todo.append((qid, direction, L.orig_query(qid, a.split).text, src, L.OTHER[src]))
            else:
                f = done.get((qid, "forward"))
                if f:
                    todo.append((qid, direction, f["translation"], L.OTHER[src], src))
        for i in range(0, len(todo), 50):
            if not C.guard():
                return 1
            with ThreadPoolExecutor(a.workers) as ex:
                for rec in ex.map(call, todo[i:i + 50]):
                    if rec.get("error"):
                        print("ERROR", rec["qid"], rec["error"][:150], flush=True)
                    else:
                        done[(rec["qid"], direction)] = rec
    for qid in ids:
        f, b = done.get((qid, "forward")), done.get((qid, "back"))
        if f:
            print("%s\n  SRC %s\n  TGT %s\n  BCK %s\n  nums src=%s tgt=%s" % (
                qid, f["source"], f["translation"], b["translation"] if b else None, numbers(f["source"]),
                numbers(f["translation"])))
    print("spent $%.4f" % C.spent())
    return 0


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
