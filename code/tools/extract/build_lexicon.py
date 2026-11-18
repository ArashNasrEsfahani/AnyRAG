"""Build fa_lexicon.txt: a Persian word list used only to confirm glyph-recovery guesses
(which letter an unmapped glyph stands for). Sources are clean Unicode Persian text:
Markdown outputs of documents that extracted without unmapped glyphs, and/or JSON
files whose "fa" fields hold Persian prose.

  python build_lexicon.py --md <document.md> [...] --json-glob "<report root>/*/chapters/*.json" --out fa_lexicon.txt
"""
import argparse
import glob
import json
import os
import re
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fa_text as F  # noqa: E402

TOKEN = re.compile("[\u0621-\u063a\u0641-\u064a\u067e\u0686\u0698\u06a9\u06af\u06cc\u06c0\u0622\u0623\u0624\u0626\u200c]+")


def words(text):
    for t in TOKEN.findall(F.normalize(text)):
        t = t.replace("\u200c", "")
        if 2 <= len(t) <= 14:
            yield t


def fa_strings(obj):
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == "fa" and isinstance(v, str):
                yield v
            else:
                yield from fa_strings(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from fa_strings(v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--md", nargs="*", default=[])
    ap.add_argument("--json-glob", nargs="*", default=[])
    ap.add_argument("--out", default=os.path.join(HERE, "fa_lexicon.txt"))
    ap.add_argument("--min-count", type=int, default=2)
    a = ap.parse_args()
    c = Counter()
    sources = []
    for p in a.md:
        c.update(words(open(p, encoding="utf-8").read()))
        sources.append(p)
    for g in a.json_glob:
        for p in glob.glob(g):
            try:
                obj = json.load(open(p, encoding="utf-8"))
            except Exception:  # noqa: BLE001
                continue
            for s in fa_strings(obj):
                c.update(words(s))
            sources.append(p)
    with open(a.out, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("# Persian word list for glyph-recovery confirmation; %d source files\n" % len(sources))
        for w, n in sorted(c.items(), key=lambda x: (-x[1], x[0])):
            if n >= a.min_count:
                fh.write("%s\t%d\n" % (w, n))
    print("words", sum(1 for n in c.values() if n >= a.min_count), "sources", len(sources))


if __name__ == "__main__":
    main()
