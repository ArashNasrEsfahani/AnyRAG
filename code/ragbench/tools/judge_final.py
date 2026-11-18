"""Blinded semantic adjudication of the fresh-holdout final run (pre-registered: work/final/PREREG.md). Frozen with it.

    .venv/Scripts/python.exe -m ragbench.tools.judge_final --judge openai/gpt-oss-120b --runs runs/final_fresh
        [--runs runs/final_fresh_A] [--questions ragbench/data/questions/test_fresh.jsonl] [--out work/final]

For every (method, question, setting) - the LATEST record of each - one verdict, appended to
<out>/verdicts_<judge slug>.jsonl the moment it exists (resumable: keys already there are skipped). The judge never
sees the method's name, its evidence or its trace: only the question, the gold answer(s), the numeric gold and the
key facts, and the candidate answer.

Rules applied before any model call (identical for every method):
  run error, empty answer             -> incorrect
  truncated answer (finish = length)  -> incorrect   (lex-v6 trace.failures.answer_truncated; else the answer call's
                                                      done_reason in the record's trace or ledger)
  the answer is exactly the abstain token -> correct iff the question is unanswerable
Rubric JUDGE_V2 (below) otherwise; the reply must be strict JSON {"correct": bool, "reason": str}.
"""
import argparse
import datetime as dt
import glob
import json
import re
import sys
from pathlib import Path

from ..config import load_config
from ..contracts import load_questions
from ..llm import Ledger
from ..llm_openrouter import OpenRouterReader, model_info
from ..util import KVCache, sha1_json

ROOT = Path(__file__).resolve().parents[2]
PROMPT_VERSION = "judge-final-v3"   # v3 (dev calibration): Persian number conventions stated in rule 1
ABSTAIN = "NOT_FOUND"

JUDGE_V2_SYSTEM = (
    "You are a strict, impartial grader for a question-answering benchmark over company reports written in Persian "
    "and English. You compare a candidate answer with the gold answer taken from the report. You do not use outside "
    "knowledge: the gold answer is the truth. Reply with one JSON object only.")

JUDGE_V2_USER = """Question ({lang}): {question}
Answer type: {answer_type}
Gold answer: {gold}
{gold_value}{key_facts}
Candidate answer: {answer}

The candidate is CORRECT only if all of these hold:
1. It states the gold answer's key fact(s): the same number (rounding at the precision the gold uses is fine) with a
   consistent unit and scale (12,500 million = 12.5 billion; 1 toman = 10 rials), and the same entity, period/year
   and measure the question asks about. Either language is fine. Number WRITING conventions never make an answer
   wrong: Persian writes the decimal point as a slash (78/3 = 78.3; 14/92 = 14.92), may use Persian digits
   (۱۲۳ = 123) and Persian scale words (هزار thousand, میلیون million, میلیارد billion), and "61 thousand" and
   "61,000" are the same number.
2. It does not contradict the gold answer anywhere (e.g. a second, different value for the same thing, or the right
   number attached to the wrong year or entity).
3. It does not add an unsupported claim that changes the answer (e.g. a wrong reason, a wrong trend). Harmless extra
   detail is fine. Ignore citation markers such as [2].
4. For an unanswerable question: correct only if the candidate declines (for example NOT_FOUND or "the report does
   not say"). An answer with a made-up value is incorrect.
Everything else is incorrect (partly right, missing part of a multi-part answer, a declined answerable question).
Reply with JSON only: {{"correct": true or false, "reason": "at most 20 words"}}"""

UNANS_GOLD = "(unanswerable: the report does not contain this; the correct behaviour is to decline)"
LANG = {"fa": "Persian", "en": "English"}


def latest_records(run_dirs):
    out = {}
    for d in run_dirs:
        for f in glob.glob(str(ROOT / "work" / d / "runs_*.jsonl")):
            for line in open(f, encoding="utf-8"):
                r = json.loads(line)
                out[(r["approach"], r["question_id"], r["setting"])] = r
    return out


def truncated(r):
    t = r.get("trace") if isinstance(r.get("trace"), dict) else {}
    fl = t.get("failures") or {}
    if "answer_truncated" in fl:
        return bool(fl["answer_truncated"])
    calls = t.get("calls") if isinstance(t.get("calls"), list) else []
    ans = [c for c in calls if isinstance(c, dict) and c.get("purpose") == "answer"]
    return bool(ans) and ans[-1].get("done_reason") == "length"


def rule(q, r):
    ans = (r.get("answer_text") or "").strip()
    if r.get("error") or not ans:
        return False, "rule: run error or empty answer"
    if truncated(r):
        return False, "rule: truncated answer"
    if re.sub(r"[\s\W_]+", "", re.sub(r"\[\s*\d+\s*\]", "", ans)).upper() == ABSTAIN.replace("_", ""):
        return q.answer_type == "unanswerable", "rule: abstained"
    return None


def messages(q, r):
    lang = r["setting"][:2]
    unans = q.answer_type == "unanswerable"
    gold = UNANS_GOLD if unans else ((q.answer_fa if lang == "fa" else q.answer_en) or q.answer_en or q.answer_fa or "")
    gv = ""
    if q.answer_type == "numeric" and q.answer_value is not None:
        gv = "Gold numeric value: %s %s\n" % (q.answer_value, q.answer_unit or "")
    kf = [k for k in ((q.meta or {}).get("key_facts") or []) if isinstance(k, dict)]
    kft = ("Key facts a correct answer contains: " + "; ".join(
        "%s%s" % (k.get("value"), (" " + k["unit"]) if k.get("unit") else "") for k in kf) + "\n") if kf and not unans else ""
    user = JUDGE_V2_USER.format(lang=LANG.get(lang, lang), question=(q.text(lang) or "").strip(),
                                answer_type=q.answer_type, gold=gold.strip(), gold_value=gv, key_facts=kft,
                                answer=(r.get("answer_text") or "").strip())
    return [{"role": "system", "content": JUDGE_V2_SYSTEM}, {"role": "user", "content": user}]


def parse(text):
    t = (text or "").strip()
    m = re.search(r"\{.*\}", t, re.S)
    try:
        obj = json.loads(m.group(0) if m else t)
    except (ValueError, AttributeError):
        obj = None
    if isinstance(obj, dict) and isinstance(obj.get("correct"), (bool, str)):
        c = obj["correct"]
        c = c if isinstance(c, bool) else str(c).strip().lower() == "true"
        return c, str(obj.get("reason") or "")[:300]
    m = re.search(r'"correct"\s*:\s*"?(true|false)', t, re.I)
    return (m.group(1).lower() == "true", "regex-parsed") if m else (None, "unparsable")


def main(argv=None):
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge", required=True)
    ap.add_argument("--runs", action="append", required=True, help="run dir(s) under work/, e.g. runs/final_fresh")
    ap.add_argument("--questions", default=str(ROOT / "ragbench" / "data" / "questions" / "test_fresh.jsonl"))
    ap.add_argument("--out", default=str(ROOT / "work" / "final"))
    ap.add_argument("--methods", default=None, help="comma list (default: every method found)")
    a = ap.parse_args(argv)
    cfg = load_config()
    qs = {q.id: q for q in load_questions(a.questions)}
    recs = latest_records(a.runs)
    want = set(a.methods.split(",")) if a.methods else None
    slug = a.judge.replace("/", "__").replace(":", "_")
    out = Path(a.out) / ("verdicts_%s.jsonl" % slug)
    done = set()
    if out.exists():
        for line in open(out, encoding="utf-8"):
            v = json.loads(line)
            done.add((v["method"], v["question_id"], v["setting"]))
    ledger = Ledger(path=str(Path(a.out) / ("judge_ledger_%s.jsonl" % slug)))
    info = model_info(a.judge, cfg.project_root / "work" / "openrouter" / "catalog.json")
    client = OpenRouterReader(cfg, ledger, a.judge, supported=info.get("params"))
    cache = KVCache(Path(a.out) / "judge_cache.sqlite")
    todo = [(k, r) for k, r in sorted(recs.items()) if k[1] in qs and k not in done and (not want or k[0] in want)]
    print("%d records, %d already judged, %d to judge with %s" % (len(recs), len(done), len(todo), a.judge), flush=True)
    n_model = 0
    for (method, qid, setting), r in todo:
        q = qs[qid]
        rv = rule(q, r)
        if rv is not None:
            correct, reason, source = rv[0], rv[1], "rule"
        else:
            msgs = messages(q, r)
            key = sha1_json({"v": PROMPT_VERSION, "judge": a.judge, "m": msgs})
            hit = cache.get(key)
            if hit is not None:
                correct, reason, source = hit["correct"], hit["reason"], "cache"
            else:
                with ledger.scope(phase="eval", approach="judge-final", question_id=qid, setting=setting):
                    res = client.chat(msgs, purpose="judge", num_predict=200)
                correct, reason = parse(res.text)
                source = "judge"
                n_model += 1
                if correct is not None:
                    cache.put(key, {"correct": correct, "reason": reason})
        line = {"method": method, "question_id": qid, "setting": setting, "correct": None if correct is None else int(correct),
                "reason": reason, "source": source, "judge": a.judge, "prompt": PROMPT_VERSION,
                "at": dt.datetime.now().isoformat(timespec="seconds")}
        with open(out, "a", encoding="utf-8") as fh:              # saved at once: a stop loses nothing judged
            fh.write(json.dumps(line, ensure_ascii=False) + "\n")
    print("judged %d with the model; verdicts in %s" % (n_model, out), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
