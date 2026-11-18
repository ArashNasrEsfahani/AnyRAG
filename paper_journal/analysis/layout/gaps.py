r"""Stricter layout check (white space), A4 ACL two-column at 50 dpi.

usage: python gaps.py main.pdf outdir [--appendix-page N] [--verbose]

Flags
  TAIL  a column whose content ends with a blank block taller than TAIL_FRAC of the text height
        while the next column / page still continues (exempt: last page, the page before the
        Appendix heading). Left column of a page counts as continued if the right column has ink.
  BAND  an internal blank band (between floats and text, or text and text) taller than BAND_PT,
        i.e. larger than the normal float separation (\textfloatsep / \dbltextfloatsep + caption skip).
Geometry: A4, 2.5 cm margins, columns 219 pt with a 0.6 cm gutter (acl.sty).
"""
import sys, glob, os, subprocess, re
import numpy as np
from PIL import Image

DPI = 50
PT = DPI / 72.0
TAIL_FRAC = 0.08
BAND_PT = 44.0          # normal float/text separation and section-head skips measure 22-40 pt here; taller is a hole

args = [a for a in sys.argv[1:] if not a.startswith('--')]
pdf, out = args[0], args[1]
verbose = '--verbose' in sys.argv
app_page = None
for i, a in enumerate(sys.argv):
    if a == '--appendix-page':
        app_page = int(sys.argv[i + 1])

os.makedirs(out, exist_ok=True)
for f in glob.glob(out + '/p-*'):
    os.remove(f)
subprocess.run(['pdftoppm', '-r', str(DPI), '-gray', pdf, out + '/p'], check=True)
files = sorted(glob.glob(out + '/p-*.pgm'))
n = len(files)

if app_page is None:
    # page where the Appendix heading sits: find via pdftotext
    txt = subprocess.run(['pdftotext', '-layout', pdf, '-'], capture_output=True, text=True,
                         encoding='utf-8', errors='replace').stdout.split('\f')
    for i, t in enumerate(txt, 1):
        if re.match(r'\s*Appendix\s*$', t.strip().split('\n')[0] if t.strip() else ''):
            app_page = i
            break

L = 70.87; TOP = 70.87; TH = 842 - 2 * 70.87
COLW = 219.08; GUT = 17.0
T = int(round(TOP * PT)); B = int(round((TOP + TH) * PT)); H = B - T
cols = [(int(L * PT) - 1, int((L + COLW) * PT) + 1),
        (int((L + COLW + GUT) * PT) - 1, int((L + 2 * COLW + GUT) * PT) + 2)]
band_px = BAND_PT * PT

flags = []
for i, f in enumerate(files, 1):
    a = np.array(Image.open(f)) < 200
    # full-width float region: rows where the gutter has ink (table* rules, figure* art, captions)
    g0, g1 = cols[0][1] + 1, cols[1][0] - 1
    grow = np.where(a[T:B + 2, g0:g1].any(1))[0]
    wide = np.zeros(B + 2 - T, bool)
    if len(grow):
        # top floats: from the first gutter row down to the last gutter row in the top block
        wide[grow[0]:grow[-1] + 1] = True
    inked = []
    for x0, x1 in cols:
        inked.append(a[T:B + 2, x0:x1].any(1))
    exempt_tail = (i == n) or (app_page is not None and i == app_page - 1)
    msgs = []
    for c, rows in enumerate(inked):
        name = 'LR'[c]
        idx = np.where(rows)[0]
        if len(idx) == 0:
            if not exempt_tail:
                msgs.append(f'{name} EMPTY column')
            continue
        tail = (H - 1 - idx[-1]) / H
        if c == 0:
            cont = inked[1].any()
        else:
            cont = i < n
        if tail > TAIL_FRAC and cont and not exempt_tail:
            msgs.append(f'{name} TAIL {tail*100:.0f}% blank at bottom')
        # internal bands
        runs = []
        cur = 0; start = None
        for y in range(idx[0], idx[-1] + 1):
            if not rows[y]:
                if cur == 0:
                    start = y
                cur += 1
            else:
                if cur:
                    runs.append((start, cur))
                cur = 0
        for s, ln in runs:
            if ln > band_px and not wide[s:s + ln].all():
                msgs.append(f'{name} BAND {ln/PT:.0f}pt at y={(s+T)/PT/842*100:.0f}%')
        if verbose:
            big = sorted([ln / PT for s, ln in runs if ln / PT > 14], reverse=True)[:5]
            print(f'  p{i:02d}{name} tail {tail*100:.0f}% gaps(pt) {[round(g) for g in big]}')
    for m in msgs:
        flags.append((i, m))
        print(f'p{i:02d} {m}')
print(f'pages {n}; appendix heading page {app_page}; flags {len(flags)}')
