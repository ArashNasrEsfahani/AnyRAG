"""S1 task 2 (a, b): blinded human-review packets. No model calls.

Packet A  (packets/A_concordant24/): a seeded random sample of 24 of the 60 machine-concordant answerable test questions
          (the 90 answerable minus the 30 human-reviewed ones) x the four GLM @6,000 pipelines = 96 cells.
Packet B  (packets/B_second_reviewer/): the 120 human-reviewed cells (30 questions x 4 pipelines @6,000) in one random
          order, for an independent second reviewer.

Each packet is written in TWO formats that share one key:
  review_<X>.html        the earlier stand-alone page (human_check/review_dev.template.html; same localStorage page, same
                         Export JSON {"set", "seconds", "results": [{"item", "verdict", "note"}]}), now with a card for
                         the requirement list and one for the gold evidence quotes.
  game/<x>_review.jsonl  question-v2 rows (meta.v2 of the frozen test questions) +
  game/<x>_review.answers.jsonl  {"qid", "answer_id", "answer"} rows (no method field) for review_game/server.py
                         answer-check mode (load with --dir); its verdicts land in game/<x>_review.answercheck.jsonl.
  key_<X>.json           item / answer_id -> question id and the pipeline(s) that produced it. Do not open before review.
Identical answers (answer_text.strip() byte-identical within a question) are shown once; the key lists every cell.
"""
import hashlib
import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
TR = REPO / "work" / "datasets_v2" / "test_run"
HC = REPO / "work" / "datasets_v2" / "human_check"
sys.path.insert(0, str(TR))
import report_test as RT  # noqa: E402

ARMS4 = ["HYBRID_6000", "HR_6000", "LEX5B_6000", "SETR_6000"]
SEED_SAMPLE, SEED_A, SEED_B = 20261110, 20261111, 20261112
P = RT.PRIMARY
REC, _ = RT.load()
SYM = json.loads((TR / "test_results_human_sym.json").read_text(encoding="utf-8"))
REVIEWED = list(SYM["selected_questions"])
CONCORDANT = [q for q in RT.ANS if q not in REVIEWED]
V2 = {}
for line in open(REPO / "work" / "datasets_v2" / "test_freeze" / "questions" / "test_v2judge_all.jsonl",
                 encoding="utf-8-sig"):
    if line.strip():
        d = json.loads(line)
        V2[d["id"]] = d["meta"]["v2"]
assert len(REVIEWED) == 30 and len(CONCORDANT) == 60


def ans(a, q):
    return (((REC[P][a].get(q)) or {}).get("answer_text") or "").strip()


def requirements(q):
    out = []
    for r in V2[q].get("requirements") or []:
        val = r.get("value")
        val = "; ".join(map(str, val)) if isinstance(val, list) else val
        out.append({"id": r["id"], "required": bool(r.get("required")), "kind": r.get("kind"),
                    "text": "%s%s" % (r.get("proposition") or "",
                                      (" [%s%s]" % (val, (" " + r["unit"]) if r.get("unit") else "")) if val not in (None, "") else "")})
    return out


def evidence(q):
    return [{"id": e["id"], "doc": e.get("doc_title") or e.get("doc_id"), "role": e.get("role"),
             "supports": e.get("supports") or [], "quote": e.get("quote") or ""} for e in V2[q].get("evidence") or []]


def page_item(iid, q, text):
    qd = RT.QS[q]
    return {"item": iid, "lang": qd["lang"], "question": qd["question"], "question_gloss": qd["gloss"],
            "reference": qd["reference"], "reference_gloss": qd["reference_gloss"], "answer": text,
            "requirements": requirements(q), "evidence": evidence(q)}


EXTRA_CARDS = ('<div class="card"><div class="lab">Requirements (what a correct answer must state; optional ones are '
               'marked)</div><ol id="req" class="small"></ol></div>\n'
               '<div class="card"><div class="lab">Gold evidence quotes (from the source documents)</div>'
               '<div id="ev" class="small"></div></div>\n')
EXTRA_CSS = (".small{font-size:16px;line-height:1.6}.small li{margin:4px 0}.opt{color:var(--mute)}"
             ".evq{border-inline-start:3px solid var(--line);padding:4px 10px;margin:8px 0;white-space:pre-wrap}"
             ".evm{color:var(--mute);font-size:13px}\n</style>")
EXTRA_JS = r"""
  var ol = $("req"); ol.innerHTML = "";
  (it.requirements||[]).forEach(function(r){ var li = document.createElement("li"); li.textContent = r.id + (r.required ? "" : " (optional)") + ": " + r.text; if (!r.required) li.className = "opt"; var f = /[؀-ۿ]/.test(r.text); li.dir = f ? "rtl" : "ltr"; ol.appendChild(li); });
  var ev = $("ev"); ev.innerHTML = "";
  (it.evidence||[]).forEach(function(e){ var m = document.createElement("div"); m.className = "evm"; m.textContent = e.id + " · " + (e.doc||"") + " · " + (e.role||"") + " · supports " + (e.supports||[]).join(", "); var d = document.createElement("div"); var f = /[؀-ۿ]/.test(e.quote||""); d.className = "evq" + (f ? " fa" : ""); d.dir = f ? "rtl" : "ltr"; d.textContent = e.quote; ev.appendChild(m); ev.appendChild(d); });
"""
RULE = ('<div class="card" style="border-color:var(--bad);font-size:17px"><b>Strict rule:</b> correct only if every '
        'REQUIRED requirement matches the reference; half correct = wrong; extra unsupported claims that change the '
        'answer = wrong. Optional requirements may be missing. If unsure, press 3 and write why.</div>\n')


def write_html(path, items, set_name, store_key, title):
    tpl = (HC / "review_dev.template.html").read_text(encoding="utf-8")
    reps = [('var KEY = "dev_review_v1";', 'var KEY = "%s";' % store_key),
            ("<title>Dev Answer Check</title>", "<title>%s</title>" % title),
            ('set:"dev60_discordant@6000"', 'set:"%s"' % set_name),
            ('<div class="wrap">\n', '<div class="wrap">\n' + RULE),
            ('<div class="btns">', EXTRA_CARDS + '<div class="btns">'),
            ("</style>", EXTRA_CSS),
            ('  $("note").value = r.note || "";', EXTRA_JS + '  $("note").value = r.note || "";'),
            ("wording and extra harmless detail do not matter", "wording does not matter")]
    for a, b in reps:
        assert tpl.count(a) == 1, a
        tpl = tpl.replace(a, b)
    data = json.dumps(items, ensure_ascii=False).replace("</", "<" + chr(92) + "/")
    path.write_text(tpl.replace("/*__DATA__*/[]", data), encoding="utf-8")


def build(name, qids, cells_of, seed, order, set_name, store_key, title, extra_key):
    """order: 'grouped' = questions in random order, answers shuffled within each question; 'flat' = one random
    order over all items."""
    rnd = random.Random(seed)
    out = HERE / "packets" / name
    (out / "game").mkdir(parents=True, exist_ok=True)
    groups = []
    for q in qids:
        items = {}
        for a in cells_of(q):
            items.setdefault(ans(a, q), []).append(a)
        g = [{"qid": q, "answer": t, "arms": arms} for t, arms in items.items()]
        rnd.shuffle(g)
        groups.append(g)
    rnd.shuffle(groups)
    flat = [x for g in groups for x in g]
    if order == "flat":
        rnd.shuffle(flat)
    key, page, game_ans = [], [], []
    prefix = name[0]
    for i, it in enumerate(flat, 1):
        iid = "%s%03d" % (prefix, i)
        aid = hashlib.sha1(("%s|%s|%d" % (name, iid, seed)).encode()).hexdigest()[:10]
        key.append({"item": iid, "answer_id": aid, "qid": it["qid"], "arms": it["arms"]})
        page.append(page_item(iid, it["qid"], it["answer"]))
        game_ans.append({"qid": it["qid"], "answer_id": aid, "answer": it["answer"]})
    write_html(out / ("review_%s.html" % name), page, set_name, store_key, title)
    qorder = list(dict.fromkeys(it["qid"] for it in flat))
    stem = "%s_review" % name.split("_")[0]
    with open(out / "game" / (stem + ".jsonl"), "w", encoding="utf-8") as fh:
        for q in qorder:
            fh.write(json.dumps(V2[q], ensure_ascii=False) + "\n")
    with open(out / "game" / (stem + ".answers.jsonl"), "w", encoding="utf-8") as fh:
        for r in game_ans:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    n_cells = sum(len(k["arms"]) for k in key)
    (out / ("key_%s.json" % name)).write_text(json.dumps(dict({
        "set": set_name, "seed": seed, "order": order, "questions": len(qids), "cells": n_cells, "items": len(key),
        "dedupe_rule": "answer_text.strip() byte-identical within the same question",
        "game_files": ["game/%s.jsonl" % stem, "game/%s.answers.jsonl" % stem], "items_key": key}, **extra_key),
        ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(name, "questions", len(qids), "cells", n_cells, "items", len(key))


sample = sorted(random.Random(SEED_SAMPLE).sample(sorted(CONCORDANT), 24))
build("A_concordant24", sample, lambda q: ARMS4, SEED_A, "grouped", "test24_concordant_sample@6000",
      "s1_concordant24_v1", "Concordant Sample Check",
      {"sample_seed": SEED_SAMPLE, "population": "60 machine-concordant answerable test questions (90 answerable minus "
       "the 30 human-reviewed questions of test_results_human_sym.json)", "population_qids": sorted(CONCORDANT),
       "sample_qids": sample})
build("B_second_reviewer", REVIEWED, lambda q: ARMS4, SEED_B, "flat", "test30_second_reviewer@6000",
      "s1_second_reviewer_v1", "Second Reviewer Check",
      {"population": "the 30 human-reviewed questions x 4 pipelines @6000 (120 cells)", "sample_qids": REVIEWED})
