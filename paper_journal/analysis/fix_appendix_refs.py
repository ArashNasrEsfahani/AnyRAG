"""elsarticle's \\ref to an appendix unit already prints 'Appendix A.2', so 'Appendix~\\ref{...}' printed
'Appendix Appendix A.2'. Drop the leading word in the live sources; also set the agreed title."""
import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parents[1]
FILES = [ROOT / "sections" / f for f in ("j00_front.tex", "j1_introduction.tex", "j2_methodology.tex", "j3_results.tex",
                                         "j4_conclusion.tex", "jA_appendix.tex")]
FILES += [ROOT / "tables" / f for f in ("pools.tex", "posthoc_log.tex", "settings_full.tex")]

total = 0
for p in FILES:
    s = p.read_text(encoding="utf-8")
    n0 = s
    # "Appendices~\ref{a} and~\ref{b}" -> "\ref{a} and~\ref{b}"; "Appendix~\ref{a}" / "Appendix \ref{a}" -> "\ref{a}"
    s, k1 = re.subn(r"Appendices~(\\ref\{)", r"\1", s)
    s, k2 = re.subn(r"Appendix[~ ](\\ref\{)", r"\1", s)
    total += k1 + k2
    if p.name == "j00_front.tex":
        s, k3 = re.subn(r"\\title\{[^}]*\}",
                        lambda m: r"\title{What Reaches the Reader: A Comparison of Retrieval-Augmented Generation "
                                  r"Setups in Persian and English}", s, count=1)
    if s != n0:
        p.write_bytes(s.encode("utf-8"))
        print(p.name, k1 + k2)
print("total", total)
