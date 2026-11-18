#!/usr/bin/env python3
"""Paired comparison of two RAG setups on the same questions: exact sign test.

This is the analysis the study used (two-sided exact sign test on discordant
pairs, the exact form of McNemar's test). Standard library only.

Input, either:
  python sign_test.py results.csv        # CSV with header: question_id,a,b
                                         # a, b = 1 if that setup answered correctly, else 0
                                         # (optional column 'exclude' = 1 drops a row,
                                         #  e.g. questions the reader answers closed-book)
  python sign_test.py --b 14 --w 6       # counts of discordant pairs directly
Output: n, A-only wins (b), B-only wins (w), net difference, exact two-sided p.
"""
import argparse
import csv
import sys
from math import comb


def sign_test(b, w):
    """Two-sided exact sign test p-value for b wins against w losses."""
    n = b + w
    if n == 0:
        return 1.0
    k = min(b, w)
    tail = sum(comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def truthy(x):
    return str(x).strip().lower() in ("1", "true", "yes", "y", "correct")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("csv", nargs="?", help="CSV with columns question_id,a,b[,exclude]")
    p.add_argument("--b", type=int, help="questions only setup A answered correctly")
    p.add_argument("--w", type=int, help="questions only setup B answered correctly")
    a = p.parse_args(argv)
    if a.csv:
        total = a_ok = b_ok = b = w = 0
        with open(a.csv, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if truthy(row.get("exclude", "0")):
                    continue
                ra, rb = truthy(row["a"]), truthy(row["b"])
                total += 1
                a_ok += ra
                b_ok += rb
                b += ra and not rb
                w += rb and not ra
        print("questions: %d   A correct: %d   B correct: %d" % (total, a_ok, b_ok))
    elif a.b is not None and a.w is not None:
        b, w = a.b, a.w
    else:
        p.print_help()
        return 2
    pval = sign_test(b, w)
    print("discordant pairs: %d   A only: %d   B only: %d   net (A - B): %+d" % (b + w, b, w, b - w))
    print("exact two-sided sign test: p = %.4f" % pval)
    if pval >= 0.05:
        print("Not significant at .05: treat the two setups as not distinguished on these questions.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
