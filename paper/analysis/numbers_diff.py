"""Number-pass invariant (N1): every numeric token removed from the prose of a section must still exist somewhere
in the paper -- a table, a figure (incl. figures/data), a caption or the appendix (a number that survives only in
other body prose is reported separately). Prose numbers are counted with readability.py's rules.

Usage: python analysis/numbers_diff.py [base dir under analysis/, default numbers_orig]
A removed number x with k decimals is matched by any number y in a home with round(y, k) == x (so .18 matches .180);
signs and thousands separators are ignored. Prints, per file, the removed numbers and where each one lives now.
"""
import os, re, sys, glob, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from readability import number_blocks, strip_comments

P = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = os.path.join(P, 'analysis', sys.argv[1] if len(sys.argv) > 1 else 'numbers_orig')
NUM = re.compile(r'\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?|\.\d+')


def val(x):
    return float(x.replace(',', ''))


def decimals(x):
    return len(x.split('.')[1]) if '.' in x else 0


def numbers_in(path, comments=False):
    s = open(path, encoding='utf8', errors='ignore').read()
    if not comments and path.endswith('.tex'):
        s = strip_comments(s)
    return [val(x) for x in NUM.findall(s)]


homes = {}
for f in glob.glob(P + '/tables/*.tex') + glob.glob(P + '/figures/*.tex') + glob.glob(P + '/figures/data/*'):
    homes[os.path.relpath(f, P).replace(os.sep, '/')] = numbers_in(f)
for f in glob.glob(P + '/sections/*.tex'):
    name = os.path.basename(f)
    key = 'sections/' + name
    raw = open(f, encoding='utf8').read()
    if name[:2].isdigit():     # body/end matter: prose numbers are not a home, captions/tables inside are
        prose = set(map(val, sum((b[1] for b in number_blocks(raw)), [])))
        homes[key + ' (non-prose)'] = [v for v in numbers_in(f) if v not in prose]
        homes[key + ' (prose)'] = list(prose)
    else:
        homes[key] = numbers_in(f)


def where(x):
    k, v = decimals(x), val(x)
    hit = [h for h, vals in homes.items() if any(round(y, k) == v for y in vals)]
    good = [h for h in hit if '(prose)' not in h]
    return good, hit


unmatched = []
for f in sorted(glob.glob(P + '/sections/*.tex')):
    name = os.path.basename(f)
    o = os.path.join(BASE, name)
    if not os.path.exists(o) or not name[:2].isdigit():
        continue
    old = collections.Counter(sum((b[1] for b in number_blocks(open(o, encoding='utf8').read())), []))
    new = collections.Counter(sum((b[1] for b in number_blocks(open(f, encoding='utf8').read())), []))
    gone = old - new
    if not gone:
        continue
    print(f'\n{name}: {sum(gone.values())} prose numbers removed ({sum(old.values())} -> {sum(new.values())})')
    for x in sorted(gone, key=lambda t: (val(t), t)):
        good, hit = where(x)
        tag = ', '.join(sorted(good)[:4]) if good else ('ONLY BODY PROSE: ' + ', '.join(hit) if hit else 'NOT FOUND')
        if not good:
            unmatched.append((name, x, tag))
        print(f'  {x:>8s} x{gone[x]}  -> {tag}')
print('\nRemoved numbers without a table/figure/appendix home:', len(unmatched))
for u in unmatched:
    print('  ', *u)
