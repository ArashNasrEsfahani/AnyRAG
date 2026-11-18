"""Build the semantic-review pack of the fresh holdout: every question that passed the mechanical checks, with the
text of its gold units (both languages for corpus A), for an independent reviewer who never sees the full report.

    .venv/Scripts/python.exe -m ragbench.tools.review_pack
-> work/final/authoring/review_pack.jsonl (one question per line) and review_pack_<part>.jsonl (split for reviewers)
"""
import json
import sys
from pathlib import Path

from ..config import load_config
from ..corpus.registry import load_set

ROOT = Path(__file__).resolve().parents[2]
DIR = ROOT / "work" / "final" / "authoring"


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    round2 = "--round2" in sys.argv          # only the questions verify_fresh lists as pending (fix / never reviewed)
    rows = json.loads((DIR / "verify.json").read_text(encoding="utf-8"))
    ok_ids = {(r["batch"], r["id"]) for r in rows if r["ok"]}
    if round2:
        pend = set(json.loads((DIR / "pending_review.json").read_text(encoding="utf-8")))
        ok_ids = {k for k in ok_ids if k[1] in pend}
    units = {}
    for name in ("A", "C", "D"):
        for d in load_set(name, load_config()):
            for u in d.units:
                units[u.id] = u.text
    pack = []
    for f in sorted((DIR / "out").glob("*.jsonl")):
        last = {}
        for line in open(f, encoding="utf-8"):
            line = line.strip().lstrip("﻿")
            if line:
                q = json.loads(line)
                last[q["id"]] = q
        for qid, q in last.items():
            if (f.stem, qid) not in ok_ids:
                continue
            gold = {l: [{"unit": u, "text": units.get(u, "")[:2500]} for u in (q.get("gold_units") or {}).get(l) or []]
                    for l in ("fa", "en")}
            pack.append({"id": qid, "batch": f.stem, "corpus": q["corpus"], "type": q["type"],
                         "answer_type": q["answer_type"], "report": (q.get("doc_scope") or [""])[0],
                         "question_fa": q.get("question_fa"), "question_en": q.get("question_en"),
                         "answer_fa": q.get("answer_fa"), "answer_en": q.get("answer_en"),
                         "answer_value": q.get("answer_value"), "answer_unit": q.get("answer_unit"),
                         "key_facts": (q.get("meta") or {}).get("key_facts"),
                         "absent_search": (q.get("meta") or {}).get("absent_search"), "gold_text": gold})
    tag = "review_pack_round2" if round2 else "review_pack"
    (DIR / (tag + ".jsonl")).write_text("".join(json.dumps(p, ensure_ascii=False) + "\n" for p in pack),
                                           encoding="utf-8")
    parts = {"fa": [p for p in pack if p["corpus"] == "C"], "en": [p for p in pack if p["corpus"] == "D"],
             "ai": [p for p in pack if p["corpus"] == "A"]}
    for k, ps in parts.items():
        (DIR / ("%s_%s.jsonl" % (tag, k))).write_text("".join(json.dumps(p, ensure_ascii=False) + "\n" for p in ps),
                                                         encoding="utf-8")
    print("review pack: %d questions (%s)" % (len(pack), ", ".join("%s %d" % (k, len(v)) for k, v in parts.items())))


if __name__ == "__main__":
    main()
