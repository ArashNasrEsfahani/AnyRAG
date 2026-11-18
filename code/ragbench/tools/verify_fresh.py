"""Verify the authored fresh-holdout questions against their source units and merge the passing ones.

    .venv/Scripts/python.exe -m ragbench.tools.verify_fresh            (report only)
    .venv/Scripts/python.exe -m ragbench.tools.verify_fresh --write    (also writes ragbench/data/questions/test_fresh.jsonl)

Mechanical, source-grounded checks per question (any failure drops it, with the reason, from the merged file):
  schema      id / corpus / type / answer_type / gold_units / doc_scope present and valid (contracts.Question.validate)
  placement   the question's corpus and company belong to the batch file it is in (agents shared a scratch file)
  unique      id and question text not duplicated across all batches
  scope       exactly one report in doc_scope; every gold unit belongs to it (both languages for corpus A)
  gold        every gold unit id exists in the corpus; answerable -> at least one gold unit per required language
  key facts   answerable -> 1+ key facts, EACH found in the text of the question's gold units with the scorer's own
              matcher (eval/keyfacts.score_key_facts: number readings, string containment) - per language for A
  unanswerable no gold units, answer NOT_FOUND-like, meta.absent_search non-empty
  language    corpus C: question_fa; corpus D: question_en; corpus A: both
Report: work/final/authoring/VERIFY.md (+ verify.json with every reason). The merged file keeps the authoring order.
"""
import argparse
import glob
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

from ..config import load_config
from ..contracts import Question
from ..corpus.registry import load_set
from ..eval.keyfacts import score_key_facts

ROOT = Path(__file__).resolve().parents[2]
OUTDIR = ROOT / "work" / "final" / "authoring" / "out"
BATCH = {  # batch file -> (corpus, companies)
    "fa1": ("C", {"bitestan", "deema", "derakcloud", "dezhino", "flytoday", "meligold"}),
    "fa2": ("C", {"metrix", "myket", "najva", "okala", "payman", "podro"}),
    "fa3": ("C", {"sheypoor", "taline", "tetherland", "trendix", "triboon"}),
    "en1": ("D", {"mowi", "zalando", "arcelormittal"}),
    "en2": ("D", {"riotinto", "turkcell", "deliveryhero"}),
    "en3": ("D", {"vodafone", "wise"}),
    "en4": ("D", {"leroy", "maersk"}),
    "en5": ("D", {"tatasteel"}),
    "ai1": ("A", {"rpt-h", "rpt-i", "rpt-j", "rpt-b", "rpt-c", "rpt-e",
                  "rpt-d", "rpt-a", "rpt-k", "rpt-f", "rpt-g"}),
}
LANGS = {"C": ("fa",), "D": ("en",), "A": ("fa", "en")}


def family_of(q):
    return ((q.get("meta") or {}).get("family") or q["id"].split("-")[2]).lower()


def main(argv=None):
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args(argv)
    cfg = load_config()
    units = {}
    for name in ("A", "C", "D"):
        for d in load_set(name, cfg):
            for u in d.units:
                units[u.id] = u.text
    rows, seen_ids, seen_text = [], Counter(), Counter()
    raw = []
    for f in sorted(glob.glob(str(OUTDIR / "*.jsonl"))):
        batch = Path(f).stem
        for i, line in enumerate(open(f, encoding="utf-8")):
            line = line.strip().lstrip("﻿")
            if not line:
                continue
            try:
                q = json.loads(line)
            except ValueError as e:
                rows.append({"batch": batch, "line": i + 1, "id": None, "ok": False, "why": ["json: %s" % e]})
                continue
            raw.append((batch, i + 1, q))
    # a corrected question is APPENDED with the same id (the files stay append-only): within a batch the last
    # version of an id wins; the same id in two batches is still a duplicate
    last = {}
    for batch, ln, q in raw:
        last[(batch, q.get("id"))] = (batch, ln, q)
    raw = sorted(last.values(), key=lambda t: (t[0], t[1]))
    for batch, ln, q in raw:
        seen_ids[q.get("id")] += 1
        seen_text[(q.get("question_fa") or "") + "|" + (q.get("question_en") or "")] += 1
    keep = []
    for batch, ln, q in raw:
        why = []
        corpus = q.get("corpus")
        exp = BATCH.get(batch)
        if exp and (corpus != exp[0] or family_of(q) not in exp[1]):
            why.append("placement: %s/%s is not a %s question" % (corpus, family_of(q), batch))
        if seen_ids[q.get("id")] > 1:
            why.append("duplicate id")
        if seen_text[(q.get("question_fa") or "") + "|" + (q.get("question_en") or "")] > 1:
            why.append("duplicate question text")
        try:
            qq = Question.from_dict(q) if hasattr(Question, "from_dict") else Question(**{k: q.get(k) for k in (
                "id", "corpus", "question_fa", "question_en", "answer_fa", "answer_en", "answer_type", "gold_units",
                "doc_scope", "type", "answer_value", "answer_unit", "meta")})
            bad = qq.validate()
            if bad:
                why.append("schema: %s" % bad)
        except Exception as e:  # noqa: BLE001
            why.append("schema: %s" % str(e)[:120])
        langs = LANGS.get(corpus, ())
        for l in langs:
            if not (q.get("question_%s" % l) or "").strip():
                why.append("language: question_%s missing" % l)
        scope = q.get("doc_scope") or []
        if len(scope) != 1:
            why.append("scope: %d reports" % len(scope))
        base = scope[0][:-3] if scope and scope[0][-3:] in (".fa", ".en") else (scope[0] if scope else "")
        gold = q.get("gold_units") or {}
        unans = q.get("answer_type") == "unanswerable" or q.get("type") == "unanswerable"
        if unans:
            if any(gold.get(l) for l in ("fa", "en")):
                why.append("unanswerable with gold units")
            if not (q.get("meta") or {}).get("absent_search"):
                why.append("unanswerable without absent_search")
        else:
            kf = (q.get("meta") or {}).get("key_facts") or []
            if not kf:
                why.append("no key facts")
            for l in langs:
                gl = gold.get(l) or []
                if not gl:
                    why.append("gold: no %s units" % l)
                    continue
                missing = [u for u in gl if u not in units]
                if missing:
                    why.append("gold: unknown unit %s" % missing[0])
                    continue
                if any(not u.startswith(base + ".") for u in gl):
                    why.append("gold: unit outside the report in scope")
                if kf:
                    text = "\n".join(units[u] for u in gl)
                    s = score_key_facts({"meta": {"key_facts": kf}}, text)
                    if s["kf_correct"] != 1.0:
                        why.append("key facts not in %s gold text: %s" % (l, s.get("kf_missing")))
        rows.append({"batch": batch, "line": ln, "id": q.get("id"), "corpus": corpus, "family": family_of(q),
                     "type": q.get("type"), "ok": not why, "why": why})
        if not why:
            keep.append(q)
    by = defaultdict(lambda: [0, 0])
    for r in rows:
        by[(r["batch"], r.get("family"))][0 if r["ok"] else 1] += 1
    L = ["# Fresh-holdout verification", "", "Passing %d of %d questions." % (len(keep), len(rows)), "",
         "| batch | company | pass | fail |", "|---|---|---|---|"]
    for (b, fam), (ok, bad) in sorted(by.items(), key=lambda kv: (kv[0][0], str(kv[0][1]))):
        L.append("| %s | %s | %d | %d |" % (b, fam, ok, bad))
    L += ["", "## Failures", ""] + ["- %s %s (line %s): %s" % (r["batch"], r["id"], r["line"], "; ".join(r["why"]))
                                    for r in rows if not r["ok"]]
    cells = Counter((q["corpus"], q["type"]) for q in keep)
    L += ["", "## Passing questions by corpus and type", ""] + ["- %s %s: %d" % (c, t, n) for (c, t), n in sorted(cells.items())]
    rep = ROOT / "work" / "final" / "authoring"
    (rep / "VERIFY.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    (rep / "verify.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    print("\n".join(L[:40]))
    # the semantic review decides admission: round 2 (review/rev2_*.jsonl, for fixed, replaced and never-reviewed
    # questions) overrides round 1 (review/rev_*.jsonl); only "keep" enters the test file
    decision = {}
    for rnd in ("rev_", "rev2_"):
        for f in sorted(glob.glob(str(ROOT / "work" / "final" / "authoring" / "review" / (rnd + "*.jsonl")))):
            for line in open(f, encoding="utf-8"):
                line = line.strip().lstrip("﻿")
                if line:
                    r = json.loads(line)
                    decision[r["id"]] = (r.get("verdict"), rnd.rstrip("_"))
    admitted = [q for q in keep if decision.get(q["id"], (None,))[0] == "keep"]
    pending = [q["id"] for q in keep if decision.get(q["id"], (None,))[0] in (None, "fix")]
    dropped = [q["id"] for q in keep if decision.get(q["id"], (None,))[0] == "drop"]
    print("review: admitted %d, pending review/fix %d, dropped %d" % (len(admitted), len(pending), len(dropped)))
    (ROOT / "work" / "final" / "authoring" / "pending_review.json").write_text(
        json.dumps(pending, ensure_ascii=False, indent=1), encoding="utf-8")
    if a.write:
        if pending:
            print("NOT writing: %d questions still await review: %s" % (len(pending), pending[:10]))
            return 1
        keep = admitted
        out = ROOT / "ragbench" / "data" / "questions" / "test_fresh.jsonl"
        out.write_text("".join(json.dumps(q, ensure_ascii=False) + "\n" for q in keep), encoding="utf-8")
        print("wrote %d questions to %s" % (len(keep), out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
