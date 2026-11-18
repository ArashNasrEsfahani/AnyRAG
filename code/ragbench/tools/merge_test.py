"""Merge the verified test shards into ragbench/data/questions/test.jsonl and report statistics.

- stamps meta.source (the shard's .src.yaml; questions.py hard-codes "pilot.src.yaml"), meta.shard and meta.split
- refuses duplicate ids across shards or with the pilot
- flags possible overlap with the dev (pilot) set: shared gold units, or near-identical question text
- writes test.jsonl and test_stats.md next to the shards

Usage: python -m ragbench.tools.merge_test
"""
import collections
import json
import re
import statistics
import sys
from pathlib import Path

QDIR = Path(__file__).resolve().parents[1] / "data" / "questions"
SHARDS = ["A1", "A2", "B", "C1", "C2"]


def load(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def tokens(text):
    return set(t for t in re.findall(r"\w+", (text or "").lower()) if len(t) > 2)


def jaccard(a, b):
    return len(a & b) / len(a | b) if a and b else 0.0


def all_gold(q):
    out = set()
    for units in (q.get("gold_units") or {}).values():
        out.update(units)
    return out


def main():
    pilot = load(QDIR / "pilot.jsonl")
    pilot_ids = {q["id"] for q in pilot}
    merged, seen = [], {}
    for shard in SHARDS:
        for q in load(QDIR / ("test_%s.jsonl" % shard)):
            if q["id"] in seen:
                sys.exit("duplicate id %s in %s and %s" % (q["id"], seen[q["id"]], shard))
            if q["id"] in pilot_ids:
                sys.exit("id %s is also a pilot (dev) id" % q["id"])
            seen[q["id"]] = shard
            meta = q.setdefault("meta", {})
            meta["source"] = "test_%s.src.yaml" % shard
            meta["shard"] = shard
            meta["split"] = "test"
            merged.append(q)

    # overlap with dev
    flags = []
    for q in merged:
        g = all_gold(q)
        for p in pilot:
            shared = g & all_gold(p)
            sim = max(jaccard(tokens(q.get("question_en")), tokens(p.get("question_en"))),
                      jaccard(tokens(q.get("question_fa")), tokens(p.get("question_fa"))))
            if shared or sim >= 0.6:
                flags.append((q["id"], p["id"], sorted(shared)[:3], round(sim, 2)))

    out = QDIR / "test.jsonl"
    with open(out, "w", encoding="utf-8", newline="\n") as f:
        for q in merged:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")

    # statistics
    by_corpus = collections.Counter(q["corpus"] for q in merged)
    by_type = collections.Counter((q["corpus"], q.get("type")) for q in merged)
    by_answer = collections.Counter(q.get("answer_type") for q in merged)
    memo = sum(1 for q in merged if q.get("meta", {}).get("memorisable"))
    docs = collections.Counter()
    for q in merged:
        for d in q.get("doc_scope") or []:
            docs[d] += 1
    gold_sizes = [len(all_gold(q)) for q in merged if q.get("type") != "unanswerable"]
    ov = collections.defaultdict(list)
    for q in merged:
        for lang, v in (q.get("meta", {}).get("lexical_overlap") or {}).items():
            if isinstance(v, dict) and v.get("content") is not None:
                ov[(q["corpus"], lang, "content")].append(v["content"])
                ov[(q["corpus"], lang, "longest")].append(v.get("longest") or 0)

    lines = ["# Test set statistics", "", "Built by `python -m ragbench.tools.merge_test` from shards %s." % ", ".join(SHARDS), "",
             "| | Count |", "|---|---|", "| Questions | %d |" % len(merged)]
    for c in sorted(by_corpus):
        lines.append("| Corpus %s | %d |" % (c, by_corpus[c]))
    lines += ["| Memorisable | %d |" % memo, "| Documents in scope | %d |" % len(docs),
              "| Gold units per answerable question (mean) | %.2f |" % statistics.mean(gold_sizes), "",
              "## Question types", "", "| Corpus | Type | Count |", "|---|---|---|"]
    for (c, t), n in sorted(by_type.items()):
        lines.append("| %s | %s | %d |" % (c, t, n))
    lines += ["", "## Answer types", "", "| Answer type | Count |", "|---|---|"]
    for t, n in sorted(by_answer.items()):
        lines.append("| %s | %d |" % (t, n))
    lines += ["", "## Question-to-gold lexical overlap (answerable questions)", "",
              "| Corpus | Language | Mean content-word share | Mean longest copied run | Max content share |", "|---|---|---|---|---|"]
    for c in sorted(by_corpus):
        for lang in ("fa", "en"):
            vals = ov.get((c, lang, "content"))
            if vals:
                lines.append("| %s | %s | %.3f | %.2f | %.3f |" % (c, lang, statistics.mean(vals),
                                                                   statistics.mean(ov[(c, lang, "longest")]), max(vals)))
    lines += ["", "## Possible overlap with the dev (pilot) set", ""]
    if flags:
        lines += ["| Test id | Pilot id | Shared gold units | Question similarity |", "|---|---|---|---|"]
        for f in flags:
            lines.append("| %s | %s | %s | %s |" % (f[0], f[1], ", ".join(f[2]) or "-", f[3]))
    else:
        lines.append("None: no shared gold units and no question pair with token Jaccard >= 0.6.")
    lines += ["", "## Documents in scope", "", "| Document | Questions |", "|---|---|"]
    for d, n in sorted(docs.items()):
        lines.append("| %s | %d |" % (d, n))
    (QDIR / "test_stats.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("wrote %s: %d questions %s" % (out, len(merged), dict(by_corpus)))
    print("dev-overlap flags: %d" % len(flags))
    for f in flags[:20]:
        print("  ", f)


if __name__ == "__main__":
    main()
