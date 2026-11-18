"""Symmetric human check (Astra R1 s2): review the GLM @6000 answers of all four arms on the same 30 test questions.
Cells already reviewed in review_test.html are skipped; unreviewed cells whose answer is byte-identical (after strip)
to a reviewed answer of the same question inherit that verdict. No model calls."""
import json, random, sys
from pathlib import Path
HC = Path(__file__).resolve().parent
sys.path.insert(0, str(HC.parent / "test_run"))
import report_test as RT  # noqa: E402

ARMS4 = ["HYBRID_6000", "HR_6000", "LEX5B_6000", "SETR_6000"]
REC, _ = RT.load()
P = RT.PRIMARY
key_old = json.loads((HC / "key_test.json").read_text(encoding="utf-8"))["items"]
exp = {r["item"]: r for r in json.loads((HC / "human_test_export.json").read_text(encoding="utf-8-sig"))["results"]}

def ans(a, q):
    return (((REC[P][a].get(q)) or {}).get("answer_text") or "").strip()

reviewed = {}  # (q, arm) -> (item, verdict)
for it in key_old:
    for a in it["arms"]:
        reviewed[(it["qid"], a)] = (it["item"], exp.get(it["item"], {}).get("verdict"))
qids = [q for q in RT.QIDS if any(k[0] == q for k in reviewed)]
inherited, items = [], {}
for q in qids:
    rev_txt = {}
    for a in ARMS4:
        if (q, a) in reviewed:
            rev_txt.setdefault(ans(a, q), reviewed[(q, a)])
    for a in ARMS4:
        if (q, a) in reviewed:
            continue
        t = ans(a, q)
        if t in rev_txt:
            inherited.append({"qid": q, "arm": a, "from_item": rev_txt[t][0], "verdict": rev_txt[t][1]})
            continue
        it = items.setdefault((q, t), {"qid": q, "answer": t, "arms": []})
        it["arms"].append(a)
items = list(items.values())
random.Random(11).shuffle(items)
key, page = [], []
for i, it in enumerate(items, 1):
    iid = "S%02d" % i
    qd = RT.QS[it["qid"]]
    key.append({"item": iid, "qid": it["qid"], "arms": it["arms"]})
    page.append({"item": iid, "lang": qd["lang"], "question": qd["question"], "question_gloss": qd["gloss"],
                 "reference": qd["reference"] if qd["answerable"] else
                 "(This question is UNANSWERABLE from the corpus; a correct answer abstains / says not found.)",
                 "reference_gloss": qd["reference_gloss"] if qd["answerable"] else None, "answer": it["answer"]})
n_cells = sum(len(it["arms"]) for it in items)
(HC / "key_test_sym.json").write_text(json.dumps({
    "seed": 11, "source": "test_run GLM @6000, all four arms (HYBRID/HR/LEX5B/SETR) on the %d questions of "
    "key_test.json; cells not yet reviewed there" % len(qids),
    "counts": {"questions": len(qids), "cells_total": 4 * len(qids), "cells_reviewed": len(reviewed),
               "cells_unreviewed": 4 * len(qids) - len(reviewed), "cells_inherited": len(inherited),
               "cells_shown": n_cells, "items_shown": len(items)},
    "dedupe_rule": "answer_text.strip() byte-identical within the same question",
    "inherited": inherited, "items": key}, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
RULE = ('<div class="card" style="border-color:var(--bad);font-size:17px"><b>Strict rule:</b> correct only if every '
        'required part matches the reference; half correct = wrong; extra unsupported claims = wrong.</div>\n')
tpl = (HC / "review_dev.template.html").read_text(encoding="utf-8")
tpl = (tpl.replace('var KEY = "dev_review_v1";', 'var KEY = "test_sym_review_v1";')
          .replace("<title>Dev Answer Check</title>", "<title>Test Answer Check</title>")
          .replace('set:"dev60_discordant@6000"', 'set:"test30_symmetric@6000"')
          .replace('<div class="wrap">\n', '<div class="wrap">\n' + RULE, 1)
          .replace("wording and extra harmless detail do not matter", "wording does not matter"))
assert "test_sym_review_v1" in tpl and "Strict rule" in tpl and "test30_symmetric" in tpl
data = json.dumps(page, ensure_ascii=False).replace("</", "<" + chr(92) + "/")
(HC / "review_test_sym.html").write_text(tpl.replace("/*__DATA__*/[]", data), encoding="utf-8")
print("questions", len(qids), "reviewed", len(reviewed), "inherited", len(inherited),
      "shown cells", n_cells, "items", len(items))
print([ (x["qid"], x["arm"], x["verdict"]) for x in inherited])
