"""Invariant check for the prose pass: numbers, \\cite/\\ref/\\label and % comment lines per file must match
the pre-pass copies in analysis/readability_orig/ (or another copy: python prose_diff.py <dir under analysis/>,
e.g. readability_p1 for the rhythm pass P2). Prints only files that differ."""
import re, glob, os, collections, sys
P = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NUM = re.compile(r'\d[\d,.]*\d|\d')
CMD = re.compile(r'\\(?:citep|citet|cite|ref|label)\{[^}]*\}')
BASE = sys.argv[1] if len(sys.argv) > 1 else 'readability_orig'
bad = 0
for f in sorted(glob.glob(P + '/sections/*.tex') + glob.glob(P + '/tables/*.tex') + glob.glob(P + '/figures/*.tex')):
    f = f.replace(os.sep, '/')
    rel = f[len(P) + 1:]
    o = P + '/analysis/' + BASE + '/' + (rel.split('/', 1)[1] if rel.startswith('sections') else rel)
    if not os.path.exists(o):
        continue
    a, b = open(o, encoding='utf8').read(), open(f, encoding='utf8').read()
    na, nb = collections.Counter(NUM.findall(a)), collections.Counter(NUM.findall(b))
    ca, cb = collections.Counter(CMD.findall(a)), collections.Counter(CMD.findall(b))
    sa = {l.strip() for l in a.split('\n') if l.lstrip().startswith('%')}
    sb = {l.strip() for l in b.split('\n') if l.lstrip().startswith('%')}
    if na != nb or ca != cb or sa != sb:
        bad += 1
        print(rel, 'nums-', dict(na - nb), 'nums+', dict(nb - na), 'cmd-', dict(ca - cb), 'cmd+', dict(cb - ca),
              'comments-', sa - sb, 'comments+', sb - sa)
print('files differing:', bad)
