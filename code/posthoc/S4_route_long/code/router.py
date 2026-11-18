"""Real router (experiment 1b): one GLM-5.3-flash call that sees ONLY the question text and predicts T1 (one fact)
versus T2/T3 (combine evidence). Same pinned endpoint as the reader; reasoning effort low; 600-token cap incl.
reasoning (the frozen LEX5B selector's cap). Unparsable reply -> T2/T3 (the safe default: the wide path).

    python router.py dev  [--version v1]      # the 60 dev questions (prompt development)
    python router.py test --version vN        # once, on the 96 test questions
Writes S4/router_<split>_<version>.jsonl (append-only, keyed by question id; reruns skip finished questions)."""
import re
import sys
import time
import common as C
import client as CL

PROMPTS = {
    "v1": ("You route questions for a question-answering system over documents such as company reports, books and "
           "laws, in Persian and English. Decide what the question needs.\n"
           "A = it asks for one fact (a value, name, date, definition or short statement) that one short passage "
           "of one document can answer.\n"
           "B = the answer must combine several pieces of evidence: for example a rule and its exception or "
           "condition, a cause and its effect, several steps of one account, values from several periods, "
           "documents, editions or versions, or a comparison or calculation.\n"
           "Reply with exactly one letter: A or B."),
    "v2": ("You route questions for a question-answering system over documents such as company reports, books and "
           "laws, in Persian and English. Decide where the answer is likely to be found.\n"
           "A = one place: a single passage, table or article of one document answers the whole question. A question "
           "with two or three parts is still A when all parts concern the same subject, period and document (for "
           "example a figure and its unit or ratio, a rule and its deadline, a fact and its stated reason).\n"
           "B = several places: the answer must be assembled from different passages, articles or chapters, or from "
           "different periods, years, documents, editions or versions; or it needs a comparison, a calculation "
           "across separate values, a rule applied together with its exceptions found elsewhere, or a multi-step "
           "account.\n"
           "Reply with exactly one letter: A or B."),
    "v3": ("You route questions for a question-answering system over documents such as company reports, books and "
           "laws, in Persian and English. Decide what the question needs.\n"
           "A = it asks about one fact or one statement that one short passage of one document can answer. A question "
           "with two parts is still A when both parts are about the same subject in the same period and document "
           "(for example a figure and how it was counted, a rule and its deadline, an event and its stated reason).\n"
           "B = the answer must combine several pieces of evidence: for example a rule and its exception or "
           "condition, a cause and its effect, several steps of one account, values from several periods, "
           "documents, editions or versions, or a comparison or calculation.\n"
           "If you are unsure, answer B.\n"
           "Reply with exactly one letter: A or B."),
}


def dev_questions():
    out = []
    for f in ("rpt", "book", "law"):
        for d in C.jl(C.QDIR / ("%s_dev.jsonl" % f)):
            out.append({"id": d["id"], "question": d["question"], "tier": d.get("tier"),
                        "intended_tier": d.get("intended_tier"), "lang": d["language"], "answerable": d["answerable"]})
    return out


def test_questions():
    out = []
    for qid, d in C.test_questions().items():
        v2 = d["meta"]["v2"]
        out.append({"id": qid, "question": v2["question"], "tier": v2.get("tier"),
                    "intended_tier": v2.get("intended_tier"), "lang": v2["language"], "answerable": v2["answerable"]})
    return out


def parse(text):
    t = (text or "").strip().upper()
    m = re.match(r"^\W*([AB])\b", t) or re.search(r"\b([AB])\b\W*$", t)
    return m.group(1) if m else None


def main():
    split = sys.argv[1]
    version = sys.argv[sys.argv.index("--version") + 1] if "--version" in sys.argv else "v1"
    qs = dev_questions() if split == "dev" else test_questions()
    out = C.S4 / ("router_%s_%s.jsonl" % (split, version))
    done = {r["id"] for r in C.jl(out)}
    cfg, ledger = CL.cfg_and_ledger()
    cl = CL.reader(cfg, ledger, max_tokens_total=600, reasoning_effort="low")
    system = PROMPTS[version]
    for q in qs:
        if q["id"] in done:
            continue
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": "Question: " + q["question"].strip()}]
        t0 = time.perf_counter()
        err, res = None, None
        with ledger.scope(phase="query", approach="S4-router-" + version, question_id=q["id"],
                          setting="%s-%s" % (q["lang"], q["lang"])):
            try:
                res = cl.chat(msgs, purpose="route", num_predict=8)
            except Exception as e:  # noqa: BLE001
                if type(e).__name__ == "BudgetExceeded":
                    raise
                err = "%s: %s" % (type(e).__name__, str(e)[:300])
        lab = parse(res.text if res else None)
        pred = "T1" if lab == "A" else "T23"
        gold = q["tier"] or q["intended_tier"]
        C.append(out, {"id": q["id"], "version": version, "lang": q["lang"], "answerable": q["answerable"],
                       "tier_label": gold, "gold": "T1" if gold == "T1" else "T23", "reply": res.text if res else None,
                       "parsed": lab, "pred": pred, "error": err,
                       "prompt_tokens": res.prompt_tokens if res else None,
                       "completion_tokens": res.completion_tokens if res else None,
                       "cost_usd": res.cost_usd if res else None, "done_reason": res.done_reason if res else None,
                       "wall_ms": round((time.perf_counter() - t0) * 1000, 1)})
    rows = C.jl(out)
    ans = [r for r in rows if r["answerable"]]
    acc = sum(r["pred"] == r["gold"] for r in ans)
    print("%s %s: routing accuracy (answerable) %d/%d; all incl. abstain-by-intended-tier %d/%d; unparsed %d; "
          "spent so far $%.5f" % (split, version, acc, len(ans), sum(r["pred"] == r["gold"] for r in rows), len(rows),
                                  sum(r["parsed"] is None for r in rows), C.spent()))
    for g in ("T1", "T23"):
        s = [r for r in ans if r["gold"] == g]
        print("  gold %s: predicted T1 %d, T23 %d" % (g, sum(r["pred"] == "T1" for r in s), sum(r["pred"] == "T23" for r in s)))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
