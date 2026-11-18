"""Selector fallbacks of LEX5B by language and tier (paper, Table 'fallbacks'; descriptive, no model calls).

Reads the saved primary-reader (GLM) test records runs/<pool>/<arm>/runs_*.jsonl and the frozen question file, using
the same record choice as report_test.load(): the latest non-error record of a question, else its last error record.
A fallback is a selector status other than 'ok' and 'all_none' (report_test.selector_stats); a deadline fallback is
one whose reason is 'deadline'. All-NONE replies (every part NONE) are counted separately; they are valid replies.

    python paper/analysis/fallback_table.py [--root REPO]        (writes paper/analysis/fallback_table.{json,md})
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import layout  # noqa: E402

ARMS = ("LEX5B_6000", "LEX5B_15500")


def jl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8-sig") if l.strip()]


def pool_of(q):
    return "-".join(q.split("-")[2:4])


def main():
    P = layout.resolve(**vars(layout.parse_args(description=__doc__.splitlines()[0])))
    M = json.loads((P["freeze"] / "MANIFEST.json").read_text(encoding="utf-8"))
    primary = M["primary_reader"]
    QS = {}
    for d in jl(P["questions"]):
        v2 = d["meta"]["v2"]
        QS[d["id"]] = {"lang": v2["language"], "tier": v2.get("tier") or v2.get("intended_tier"),
                       "answerable": d["answer_type"] != "unanswerable"}
    res = {"source": "primary reader %s, saved test records; layout %s" % (primary, P["layout"]), "arms": {}}
    for arm in ARMS:
        rec = {}
        for pool in sorted({pool_of(q) for q in QS}):
            for f in sorted((P["test"] / "runs" / pool / arm).glob("runs_*.jsonl")):
                for r in jl(f):
                    q, old = r["question_id"], rec.get(r["question_id"])
                    if old is None or old.get("error") or not r.get("error"):
                        rec[q] = r
        groups = {"all": lambda q: True,
                  "Persian": lambda q: QS[q]["lang"] == "fa", "English": lambda q: QS[q]["lang"] == "en",
                  "Answerable T1": lambda q: QS[q]["answerable"] and QS[q]["tier"] == "T1",
                  "Answerable T2": lambda q: QS[q]["answerable"] and QS[q]["tier"] == "T2",
                  "Answerable T3": lambda q: QS[q]["answerable"] and QS[q]["tier"] == "T3",
                  "Unanswerable": lambda q: not QS[q]["answerable"]}
        out = {}
        for g, f in groups.items():
            row = {"n": 0, "deadline": 0, "fallback": 0, "all_none": 0, "reasons": defaultdict(int), "missing": 0}
            for q in QS:
                if not f(q):
                    continue
                row["n"] += 1
                r = rec.get(q)
                if r is None or r.get("error"):
                    row["missing"] += 1
                    continue
                s = ((r.get("trace") or {}).get("lex5b") or {}).get("selector") or {}
                if s.get("status") == "all_none":
                    row["all_none"] += 1
                elif s.get("status") != "ok":
                    row["fallback"] += 1
                    row["reasons"][str(s.get("reason"))] += 1
                    row["deadline"] += s.get("reason") == "deadline"
            row["reasons"] = dict(row["reasons"])
            out[g] = row
        res["arms"][arm] = out
    (HERE / "fallback_table.json").write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8")
    L = ["# LEX5B selector fallbacks by group (GLM, test; descriptive)", "", res["source"], ""]
    for arm, out in res["arms"].items():
        L += ["## " + arm, "", "| group | n | deadline | all fallbacks | all-NONE | reasons |", "|---|---:|---:|---:|---:|---|"]
        L += ["| %s | %d | %d | %d | %d | %s |" % (g, r["n"], r["deadline"], r["fallback"], r["all_none"], r["reasons"])
              for g, r in out.items()]
        L.append("")
    (HERE / "fallback_table.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
