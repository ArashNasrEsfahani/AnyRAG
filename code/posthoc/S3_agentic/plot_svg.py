"""cost_accuracy.svg from results.json (matplotlib is not installed in the venv)."""
import json, math
from pathlib import Path
H = Path(__file__).resolve().parent
R = json.loads((H / "results.json").read_text(encoding="utf-8"))["arms"]
W, Hh, ml, mr, mt, mb = 760, 460, 70, 30, 40, 55
xs = [sum(r["tokens_per_q"]) / 2 for r in R.values()]
x0, x1 = math.log10(1500), math.log10(20000)
y0, y1 = 60, 85
X = lambda v: ml + (math.log10(v) - x0) / (x1 - x0) * (W - ml - mr)
Y = lambda v: Hh - mb - (v - y0) / (y1 - y0) * (Hh - mt - mb)
s = ['<svg xmlns="http://www.w3.org/2000/svg" width="%d" height="%d" font-family="sans-serif" font-size="11">' % (W, Hh),
     '<rect width="100%" height="100%" fill="white"/>',
     '<text x="%d" y="22" font-size="13">Machine strict accuracy (of 90) vs query-time LLM tokens per question (96 q, log x)</text>' % ml]
for t in (2000, 3000, 5000, 10000, 20000):
    s.append('<line x1="%.1f" y1="%d" x2="%.1f" y2="%d" stroke="#ddd"/><text x="%.1f" y="%d" text-anchor="middle">%d</text>' % (X(t), mt, X(t), Hh - mb, X(t), Hh - mb + 16, t))
for v in range(60, 86, 5):
    s.append('<line x1="%d" y1="%.1f" x2="%d" y2="%.1f" stroke="#ddd"/><text x="%d" y="%.1f" text-anchor="end">%d%%</text>' % (ml, Y(v), W - mr, Y(v), ml - 6, Y(v) + 4, v))
s.append('<text x="%d" y="%d" text-anchor="middle">tokens / question</text>' % ((W + ml) / 2, Hh - 12))
for a, r in R.items():
    x = X(sum(r["tokens_per_q"]) / 2)
    lo, hi = 100 * r["all"][0] / 90, 100 * r["all"][1] / 90
    col = "#c2410c" if a in ("GREP_AGENT", "ITER_BM25") else ("#1d4ed8" if a.endswith("6000") else "#64748b")
    if hi > lo:
        s.append('<line x1="%.1f" y1="%.1f" x2="%.1f" y2="%.1f" stroke="%s" stroke-width="2"/>' % (x, Y(lo), x, Y(hi), col))
    ym = Y((lo + hi) / 2)
    if a in ("GREP_AGENT", "ITER_BM25"):
        s.append('<rect x="%.1f" y="%.1f" width="10" height="10" fill="%s"/>' % (x - 5, ym - 5, col))
    else:
        s.append('<circle cx="%.1f" cy="%.1f" r="5" fill="%s"/>' % (x, ym, col))
    s.append('<text x="%.1f" y="%.1f" fill="%s">%s</text>' % (x + 8, ym - 6, col, a))
s.append("</svg>")
(H / "cost_accuracy.svg").write_text("\n".join(s), encoding="utf-8")
print("wrote cost_accuracy.svg")
