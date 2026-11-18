"""Write a very simple HTML page with every core test question, the true answer, and each method's answer.

Usage: python -m ragbench.tools.answers_html [--out experiment_answers.html] [--results results_keyfacts]
Reads the aggregated scored records of work/runs/core_test and work/runs/core_test_timeline.

--results picks which aggregation directory to read. `results_keyfacts` (the default) is the one
written by `ragbench.eval.aggregate --scoring keyfacts`: there `match_correct` is the key-fact
verdict for a question that carries meta.key_facts and the strict verdict for everything else, and
`acc_match_strict` keeps the strict verdict either way. The page shows both marks, the key facts of
the question, and how many of them each answer contains.
"""
import argparse
import html
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
QDIR = ROOT / "ragbench" / "data" / "questions"
RUN_DIRS = [(ROOT / "work/runs/core_test", QDIR / "test_core.jsonl"),
            (ROOT / "work/runs/core_test_timeline", QDIR / "test_timeline.jsonl")]

VARIANTS = [("B0", "B0 no retrieval"), ("B1", "B1 gold evidence"), ("B2", "B2 BM25"), ("B3", "B3 vector"),
            ("B4", "B4 tree"), ("B5", "B5 grep agent"), ("OURS+index_level=L0", "OURS L0"),
            ("OURS+index_level=L1", "OURS L1"), ("OURS", "OURS L3"),
            ("OURS+monolingual_patterns", "OURS L3 monolingual"), ("AGENTIC", "AGENTIC"), ("AGENTIC2", "AGENTIC v2"),
            ("AGENTIC3", "AGENTIC v3")]
SETTINGS = ["fa-fa", "en-en", "fa-en", "en-fa"]
CORPUS_NAME = {"A": "Corpus A — development reports (Persian + English; not released)",
               "B": "Corpus B — human-written Persian/English pair (Digikala 1401)",
               "C": "Corpus C — human-written Persian-only reports",
               "TL": "Timelines — multi-year questions (Persian-only reports)"}

CSS = """
body { font-family: "Segoe UI", Tahoma, system-ui, sans-serif; max-width: 1100px; margin: 24px auto; padding: 0 16px; color: #1d2733; }
h1 { font-size: 1.4rem; margin-bottom: 4px; } h2 { font-size: 1.15rem; margin-top: 36px; border-bottom: 2px solid #1d2733; }
.q { border: 1px solid #d9dee4; border-radius: 6px; padding: 10px 14px; margin: 16px 0; }
.qid { font-family: monospace; color: #5b6773; font-size: 0.85rem; }
.qtext { margin: 4px 0; } .gold { background: #eef7ee; padding: 6px 10px; border-radius: 4px; margin: 6px 0; font-size: 0.92rem; }
.kf { background: #f3f0fb; padding: 6px 10px; border-radius: 4px; margin: 6px 0; font-size: 0.86rem; }
.kf ul { margin: 4px 0 0 0; padding-left: 20px; } .kf code { background: #e7e1f7; padding: 0 3px; border-radius: 2px; }
table { border-collapse: collapse; width: 100%; font-size: 0.85rem; margin-top: 6px; }
th, td { border: 1px solid #e3e7eb; padding: 4px 6px; vertical-align: top; text-align: left; }
th { background: #f4f6f8; } td.m { white-space: nowrap; font-weight: 600; }
.ok { background: #e8f5e9; } .no { background: #fdecea; } .na { color: #9aa4ad; }
.ans { white-space: pre-wrap; } .mark { font-weight: 700; margin-left: 4px; }
.kfbadge { display: inline-block; margin-left: 4px; font-size: 0.78rem; color: #4a3f77; background: #ece7f8; border-radius: 3px; padding: 0 4px; }
.smark { font-size: 0.78rem; color: #5b6773; margin-left: 3px; }
.sum { color: #5b6773; font-size: 0.85rem; } .legend span { padding: 1px 6px; border-radius: 3px; margin-right: 6px; }
"""


def esc(s):
    return html.escape("" if s is None else str(s))


def fact_label(f):
    v = f.get("value")
    if f.get("kind") == "number" and isinstance(v, float) and v.is_integer():
        v = int(v)
    txt = "<code>%s</code>" % esc(v)
    if f.get("unit"):
        txt += " %s" % esc(f["unit"])
    if f.get("aliases"):
        txt += " <span class='sum'>(also: %s)</span>" % esc(", ".join(str(a) for a in f["aliases"]))
    if f.get("why"):
        txt += " — %s" % esc(f["why"])
    return txt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "experiment_answers.html"))
    ap.add_argument("--results", default="results_keyfacts",
                    help="aggregation directory under each run dir (results_keyfacts | results | ...)")
    ap.add_argument("--settings", default=",".join(SETTINGS),
                    help="comma list of settings to include, e.g. fa-fa,en-en for same-language only")
    ap.add_argument("--title", default="Core test: every question, the true answer, and each method's answer")
    args = ap.parse_args()
    keep = [s for s in SETTINGS if s in args.settings.split(",")]

    questions, answers = {}, defaultdict(dict)
    for run_dir, qfile in RUN_DIRS:
        for line in qfile.read_text(encoding="utf-8").splitlines():
            if line.strip():
                q = json.loads(line)
                questions[q["id"]] = q
        scored = run_dir / args.results / "scored.jsonl"
        if not scored.exists():
            scored = run_dir / "results" / "scored.jsonl"
        if not scored.exists():
            continue
        for line in scored.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                if r["setting"] in keep:
                    answers[r["question_id"]][(r["approach"], r["setting"])] = r

    groups = defaultdict(list)
    for qid, q in questions.items():
        groups["TL" if q.get("type") == "timeline" else q["corpus"]].append(q)

    n_kf_q = sum(1 for q in questions.values() if (q.get("meta") or {}).get("key_facts"))
    parts = ["<!doctype html><html lang='en'><head><meta charset='utf-8'><title>Core Test Answers</title>",
             "<style>%s</style></head><body>" % CSS,
             "<h1>%s</h1>" % esc(args.title),
             "<p class='sum'>Answer model gemma4:e4b-it-qat · no LLM judge · settings shown: %s "
             "(question language → document language).</p>" % esc(", ".join(keep)),
             "<p class='sum'>Scoring: lookup and unanswerable questions keep the <b>strict</b> exact/numeric matcher. "
             "The %d locate, multi-section and cross-year questions that carry key facts are scored by "
             "<b>key facts</b>: the small set of numbers and short strings a correct answer must contain "
             "(numbers matched on readings, so 4.4 million = 4,400,000 = ۴٬۴۰۰٬۰۰۰; strings on folded containment). "
             "Timelines are scored as fully correct series from each method's answer text. "
             "Both verdicts are shown: the ✓/✗ is the headline, <span class='smark'>s✓</span> means the strict "
             "matcher also counted it, and the badge is how many key facts the answer contains.</p>" % n_kf_q,
             "<p class='legend'><span class='ok'>✓ counted correct</span><span class='no'>✗ counted wrong</span>"
             "<span class='na'>– not run</span></p>"]

    total_ok = total_n = 0
    for g in ["A", "B", "C", "TL"]:
        qs = sorted(groups.get(g, []), key=lambda q: q["id"])
        if not qs:
            continue
        parts.append("<h2>%s (%d questions)</h2>" % (esc(CORPUS_NAME[g]), len(qs)))
        for q in qs:
            got = answers.get(q["id"], {})
            settings = [s for s in SETTINGS if any((v, s) in got for v, _ in VARIANTS)]
            n_ok = sum(1 for r in got.values() if r.get("match_correct"))
            total_ok += n_ok
            total_n += len(got)
            facts = (q.get("meta") or {}).get("key_facts") or []
            parts.append("<div class='q'>")
            parts.append("<div class='qid'>%s · %s · %s</div>" % (esc(q["id"]), esc(q.get("type")), esc(", ".join(q.get("doc_scope") or []))))
            parts.append("<div class='qtext' dir='rtl'><b>FA:</b> %s</div>" % esc(q.get("question_fa")))
            parts.append("<div class='qtext'><b>EN:</b> %s</div>" % esc(q.get("question_en")))
            parts.append("<div class='gold'><b>True answer:</b><div dir='rtl'>%s</div><div>%s</div></div>" % (
                esc(q.get("answer_fa")), esc(q.get("answer_en"))))
            if facts:
                parts.append("<div class='kf'><b>Key facts a correct answer must contain (%d):</b><ul>%s</ul></div>" % (
                    len(facts), "".join("<li dir='auto'>%s</li>" % fact_label(f) for f in facts)))
            parts.append("<div class='sum'>%d of %d answers counted correct</div>" % (n_ok, len(got)))
            parts.append("<table><tr><th>Method</th>%s</tr>" % "".join("<th>%s</th>" % s for s in settings))
            for key, label in VARIANTS:
                cells = []
                for s in settings:
                    r = got.get((key, s))
                    if r is None:
                        cells.append("<td class='na'>–</td>")
                        continue
                    ok = bool(r.get("match_correct"))
                    text = r.get("answer_text") or ("ERROR: %s" % r.get("error") if r.get("errored") else "")
                    badge = ("<span class='kfbadge'>key facts %d/%d</span>" % (r["kf_found"], r["kf_total"])
                             if r.get("kf_total") else "")
                    strict = ("<span class='smark'>s%s</span>" % ("✓" if r.get("acc_match_strict") else "✗")
                              if r.get("acc_match_strict") is not None else "")
                    cells.append("<td class='%s'><span class='ans' dir='auto'>%s</span><span class='mark'>%s</span>%s%s</td>" % (
                        "ok" if ok else "no", esc(text), "✓" if ok else "✗", strict, badge))
                if any("class='na'" not in c for c in cells):
                    parts.append("<tr><td class='m'>%s</td>%s</tr>" % (esc(label), "".join(cells)))
            parts.append("</table></div>")

    parts.insert(4, "<p class='sum'>%d questions · %d answers · %d counted correct</p>" % (len(questions), total_n, total_ok))
    parts.append("</body></html>")
    Path(args.out).write_text("\n".join(parts), encoding="utf-8")
    print("wrote %s (%d questions, %d answers, %d with key facts, from %s/)" % (
        args.out, len(questions), total_n, n_kf_q, args.results))


if __name__ == "__main__":
    main()
