"""Apply the final number check's open wording/rounding items (2026-11-18). Each replacement must match exactly once."""
import pathlib

R = pathlib.Path(__file__).resolve().parents[1]
EDITS = [
    ("sections/j00_front.tex", "misses its required margin over both baselines",
     "misses its required margin over the reranking baseline"),
    ("sections/j2_methodology.tex", "A half of the benchmark with $N=45$ detects approximately 8.",
     "A half of the benchmark with $N=45$ detects approximately 7 to 8."),
    ("tables/guide.tex", "and 5 to 10 at 6,000", "and about 4 to 11 at 6,000, depending on the reader run"),
    ("sections/j3_results.tex", "cost two to four times the tokens", "cost three to four times the tokens"),
    ("sections/j3_results.tex", "22 questions received a complete bundle from none of the single-pass pipelines",
     "22 questions received a complete bundle from none of the single-pass pipelines at 6,000 characters"),
    ("tables/composition.tex", "Requirements per question & mostly 2--3",
     "Requirements per question & 2--3 for 65 of 90; 4--6 for 24"),
    ("tables/negative.tex", "(3--3.5 calls)", "(3.2--3.5 calls)"),
    ("tables/posthoc_log.tex", "1.5 to 3.0 times higher", "about 1.5 to 3.0 times higher"),
]
for rel, old, new in EDITS:
    p = R / rel
    s = p.read_bytes().decode("utf-8")
    n = s.count(old)
    if n != 1:
        print("SKIP", rel, n, old[:50])
        continue
    p.write_bytes(s.replace(old, new).encode("utf-8"))
    print("ok  ", rel)
