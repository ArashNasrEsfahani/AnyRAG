"""Readability check for the paper prose (WRITING_GUIDE.md rules 1 and 12).

Extracts prose from the .tex sources (comments, math, floats' bodies, tables, TikZ and
listings removed; captions reported as their own group), splits it into sentences and
reports, per section file: sentence count, mean words per sentence, the share of sentences
in <12, 12-25, 25-35 and >35 words, and every sentence longer than 30 words.

Usage:
  python analysis/readability.py                 # current sources (paper/)
  python analysis/readability.py --root DIR      # another copy (e.g. analysis/readability_orig)
  python analysis/readability.py --long          # also print the long sentences
  python analysis/readability.py --json OUT      # write the numbers as JSON
  python analysis/readability.py --numbers       # number density per section (WRITING_GUIDE rule 16)

Number density (--numbers) follows the counting rules of work/analysis/NUMBERS_ASTRA_R1.log: prose only
(tables, captions, headings, citations, cross-reference numbers, comments and the bibliography excluded);
numeric tokens are counts, decimals, percentages and budget subscripts; a range contributes two endpoints
(a W/L pair such as 7/2 also two); model/version identifiers and tier labels (T1, GLM-5.3-flash, bge-m3,
BM25) are excluded; spelled-out quantities are words; list items count as separate paragraph blocks.
"""
import re, sys, os, json, argparse

HERE = os.path.dirname(os.path.abspath(__file__))
PAPER = os.path.dirname(HERE)

MACROS = {r'\lexb': 'Lex-RAG', r'\hybrid': 'Hybrid', r'\hr': 'HR', r'\setr': 'SetR-p+fill',
          r'\ftwo': 'three-signal score fusion', r'\LaTeX': 'LaTeX', r'\TeX': 'TeX'}
KEEP_ARG = ['textbf', 'textit', 'emph', 'textsc', 'texttt', 'textrm', 'textsf', 'mbox', 'text',
            'underline', 'textup', 'textnormal', 'hbox', 'url', 'footnote', 'small', 'enquote']
DROP_ARG = ['label', 'input', 'include', 'vspace', 'hspace', 'addcontentsline', 'phantomsection',
            'setlength', 'renewcommand', 'newcommand', 'bibliographystyle', 'bibliography',
            'FloatBarrier', 'clearpage', 'newpage', 'noindent', 'centering', 'pagebreak']
ABBR = ['e.g.', 'i.e.', 'et al.', 'cf.', 'vs.', 'Fig.', 'Eq.', 'approx.', 'etc.', 'No.', 'resp.']

BODY = ['01', '02', '03', '04', '05', '06', '07', '08', '09']
ABSTRACT = ['00']
END = ['10', '11', '12']


def strip_comments(s):
    return '\n'.join(re.sub(r'(?<!\\)%.*', '', ln) for ln in s.split('\n'))


def balanced(s, i):
    """s[i] == '{'; return index after the matching '}'."""
    d = 0
    for j in range(i, len(s)):
        if s[j] == '{' and (j == 0 or s[j - 1] != '\\'):
            d += 1
        elif s[j] == '}' and s[j - 1] != '\\':
            d -= 1
            if d == 0:
                return j + 1
    return len(s)


def extract_cmd_args(s, name):
    out = []
    for m in re.finditer(r'\\' + name + r'\*?(\[[^\]]*\])?\s*\{', s):
        a = m.end() - 1
        b = balanced(s, a)
        out.append(s[a + 1:b - 1])
    return out


def remove_env(s, env):
    return re.sub(r'\\begin\{' + env + r'\}.*?\\end\{' + env + r'\}', ' ', s, flags=re.S)


def captions_of(s, root):
    caps = []
    for c in extract_cmd_args(s, 'caption'):
        caps.append(c)
    return caps


def resolve_inputs(s, root):
    """Return captions from \\input{tables/..} / \\input{figures/..} files."""
    caps = []
    for inc in re.findall(r'\\input\{((?:tables|figures)/[^}]*)\}', s):
        p = os.path.join(root, inc + ('' if inc.endswith('.tex') else '.tex'))
        if os.path.exists(p):
            caps += captions_of(strip_comments(open(p, encoding='utf8').read()), root)
    return caps


def cmd_with_arg_replace(s, name, repl):
    out, i = [], 0
    pat = re.compile(r'\\' + name + r'\*?(\[[^\]]*\])?\s*\{')
    while True:
        m = pat.search(s, i)
        if not m:
            out.append(s[i:]); break
        out.append(s[i:m.start()])
        b = balanced(s, m.end() - 1)
        inner = s[m.end():b - 1]
        out.append(repl(inner))
        i = b
    return ''.join(out)


def detex(s, num=False):
    if num:   # drop whole comment lines so they do not split a paragraph
        s = '\n'.join(ln for ln in s.split('\n') if not ln.lstrip().startswith('%'))
    s = strip_comments(s)
    for env in ['table', r'table\*', 'figure', r'figure\*', 'tabular', 'tabularx', 'tikzpicture',
                'lstlisting', 'equation', r'equation\*', 'align', r'align\*', 'verbatim', 'tcolorbox']:
        s = remove_env(s, env)
    s = re.sub(r'\\\[.*?\\\]', ' X ', s, flags=re.S)
    if num:   # keep the numbers of inline math ($-3$, $p=.18$), drop its commands
        s = re.sub(r'\$([^$]*)\$', lambda m: ' ' + re.sub(r'\\[A-Za-z]+', ' ', m.group(1)) + ' ', s)
        s = cmd_with_arg_replace(s, 'textsubscript', lambda x: ' ' + x + ' ')
    s = re.sub(r'\$[^$]*\$', ' X ', s)
    # headings are not sentences: end them with a break marker
    for h in ['section', 'subsection', 'subsubsection', 'paragraph']:
        s = cmd_with_arg_replace(s, h, lambda x: '\n\n')
    s = re.sub(r'\\item(\[[^\]]*\])?', '\n\n', s)
    s = re.sub(r'\\begin\{[^}]*\}(\[[^\]]*\])?|\\end\{[^}]*\}', '\n\n', s)
    s = re.sub(r'\\(cite[pt]?|citealp|citeauthor|citeyear)\*?(\[[^\]]*\])*\{[^}]*\}', ' CITE ' if num else '[1]', s)
    s = re.sub(r'\\(ref|eqref|autoref|cref|Cref|pageref)\{[^}]*\}', ' REF ' if num else '1', s)
    s = cmd_with_arg_replace(s, 'fa', lambda x: 'Persian')
    s = cmd_with_arg_replace(s, 'foreignlanguage', lambda x: 'Persian')
    for name in DROP_ARG:
        s = cmd_with_arg_replace(s, name, lambda x: ' ')
    for name in KEEP_ARG:
        s = cmd_with_arg_replace(s, name, lambda x: x)
    for k in sorted(MACROS, key=len, reverse=True):
        s = re.sub(re.escape(k) + r'(\{\})?(?![A-Za-z])', MACROS[k], s)
    s = s.replace('~', ' ').replace('\\,', ' ').replace('\\ ', ' ').replace('---', ' -- ')
    s = s.replace('``', '"').replace("''", '"').replace('\\%', '%').replace('\\&', '&')
    s = re.sub(r'\\[A-Za-z]+\*?(\[[^\]]*\])?', ' ', s)   # any other command name
    s = s.replace('{', '').replace('}', '').replace('\\', ' ')
    return s


def sentences(text):
    out = []
    for para in re.split(r'\n\s*\n', text):
        p = ' '.join(para.split())
        if not p:
            continue
        for a in ABBR:
            p = p.replace(a, a.replace('.', '<DOT>'))
        p = re.sub(r'(\d)\.(\d)', r'\1<DOT>\2', p)
        parts = re.split(r'(?<=[.!?])["\)]?\s+(?=[A-Z0-9\[\("(])|(?<=[.!?:;])\s*$', p)
        for x in parts:
            x = (x or '').replace('<DOT>', '.').strip()
            if len(words(x)) >= 3:
                out.append(x)
    return out


def words(s):
    return re.findall(r"[A-Za-z0-9][A-Za-z0-9'’\-\.,/+%]*", s)


NUMTOK = re.compile(r'(?<![\d.])(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?|(?<![\d.])\.\d+')
QTY = re.compile(r'[+\-\u2212]?(?:\d[\d,]*(?:\.\d+)?|\.\d+)(?:[kKBM%]|pp)?')


def numeric_tokens(text):
    """Numeric tokens of a prose block (Astra's rules); identifiers such as T1, GLM-5.3-flash, bge-m3 are skipped."""
    t = re.sub(r'\b(Opus|Sonnet|Haiku|GPT|BY)\s+\d+(\.\d+)?', r'\1', text)   # Claude Opus 5.5, CC BY 4.0
    t = re.sub(r'[=()\[\];:]', ' ', t)
    out = []
    for tok in t.split():
        core = tok.rstrip('.,;:!?"\'').lstrip(',;:!?"\'')   # keep the leading dot of p-values (.118)
        if not re.search(r'\d', core):
            continue
        parts = re.split(r'--|\.\.|\u2013|/', core)
        if re.search(r'[A-Za-z]', core) and not all(QTY.fullmatch(x) or not x for x in parts):
            continue   # model/version identifier or tier label
        for x in parts:
            out += NUMTOK.findall(x)
    return out


def number_blocks(raw):
    """Prose paragraph blocks (list items separate) of one .tex source with their numeric tokens."""
    text = detex(raw, num=True)
    blocks = []
    for para in re.split(r'\n\s*\n', text):
        p = ' '.join(para.split())
        if len(words(p)) < 3:
            continue
        blocks.append((p, numeric_tokens(p), len(words(p))))
    return blocks


def number_report(root, show=6, as_json=None):
    main = strip_comments(open(os.path.join(root, 'main.tex'), encoding='utf8').read())
    files = re.findall(r'\\input\{sections/([^}]*)\}', main)
    res = {}
    print(f'{"section":26s} {"words":>6s} {"nums":>5s} {"blocks":>6s} {"/100w":>6s} {"/block":>6s} {"max":>4s}')
    tot = [0, 0, 0]
    heavy = []
    for f in files:
        p = os.path.join(root, 'sections', f + '.tex')
        if not os.path.exists(p):
            p = os.path.join(root, f + '.tex')
        b = number_blocks(open(p, encoding='utf8').read())
        w = sum(x[2] for x in b); n = sum(len(x[1]) for x in b); k = len(b)
        mx = max([len(x[1]) for x in b] or [0])
        res[f] = {'words': w, 'numbers': n, 'blocks': k, 'per100': round(100 * n / w, 2) if w else 0,
                  'per_block': round(n / k, 2) if k else 0, 'max': mx}
        if f[:2] != '00' and f[:2].isdigit():
            tot = [tot[0] + w, tot[1] + n, tot[2] + k]
        print(f'{f:26s} {w:6d} {n:5d} {k:6d} {res[f]["per100"]:6.2f} {res[f]["per_block"]:6.2f} {mx:4d}')
        heavy += [(len(x[1]), f, x[0][:70]) for x in b if len(x[1]) > show]
    print(f'{"body+end (01-12)":26s} {tot[0]:6d} {tot[1]:5d} {tot[2]:6d} {100 * tot[1] / max(tot[0], 1):6.2f} '
          f'{tot[1] / max(tot[2], 1):6.2f}')
    res['_body_end'] = {'words': tot[0], 'numbers': tot[1], 'blocks': tot[2]}
    if heavy:
        print(f'\nParagraph blocks with more than {show} numbers:')
        for n, f, t in sorted(heavy, reverse=True):
            print(f'  {n:3d}  {f}: {t}...')
    if as_json:
        json.dump(res, open(as_json, 'w', encoding='utf8'), indent=1)
    return res


def analyse(root):
    main = strip_comments(open(os.path.join(root, 'main.tex'), encoding='utf8').read())
    files = re.findall(r'\\input\{sections/([^}]*)\}', main)
    rows, caps_all = [], []
    for f in files:
        p = os.path.join(root, 'sections', f + '.tex')
        if not os.path.exists(p):
            p = os.path.join(root, f + '.tex')
        raw = open(p, encoding='utf8').read()
        nc = strip_comments(raw)
        caps = captions_of(nc, root) + resolve_inputs(nc, root)
        caps_all += [(f, c) for c in caps]
        sents = sentences(detex(raw))
        rows.append((f, sents))
    capsents = []
    for f, c in caps_all:
        capsents += sentences(detex(c))
    return rows, capsents


def stats(sents):
    n = len(sents)
    lens = [len(words(s)) for s in sents]
    mean = sum(lens) / n if n else 0
    longs = [(l, s) for l, s in zip(lens, sents) if l > 30]
    return n, mean, longs


def dist(sents):
    """Share of sentences (%) by length: <12, 12-25, 25-35, >35 words (25 and 35 go to the lower bin)."""
    lens = [len(words(x)) for x in sents]
    n = len(lens) or 1
    b = [sum(l < 12 for l in lens), sum(12 <= l <= 25 for l in lens),
         sum(25 < l <= 35 for l in lens), sum(l > 35 for l in lens)]
    return [round(100 * x / n) for x in b]


def group(f):
    k = f[:2]
    if k in ABSTRACT: return 'abstract'
    if k in BODY: return 'body'
    if k in END: return 'end matter'
    return 'appendix'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default=PAPER)
    ap.add_argument('--long', action='store_true')
    ap.add_argument('--json')
    ap.add_argument('--numbers', action='store_true')
    a = ap.parse_args()
    if a.numbers:
        number_report(a.root, as_json=a.json)
        return
    rows, capsents = analyse(a.root)
    res = {'files': {}, 'groups': {}}
    print(f'{"section":26s} {"sent":>5s} {"mean":>6s} {">30":>4s}   %<12 12-25 25-35  >35')
    groups = {}
    for f, s in rows:
        n, m, L = stats(s)
        groups.setdefault(group(f), []).extend(s)
        d = dist(s)
        res['files'][f] = {'sentences': n, 'mean': round(m, 1), 'over30': len(L), 'dist': d,
                           'long': [x for _, x in L]}
        print(f'{f:26s} {n:5d} {m:6.1f} {len(L):4d}   ' + ' '.join(f'{x:5d}' for x in d))
    groups['captions'] = capsents
    print('-' * 44)
    for g in ['abstract', 'body', 'end matter', 'appendix', 'captions']:
        n, m, L = stats(groups.get(g, []))
        d = dist(groups.get(g, []))
        res['groups'][g] = {'sentences': n, 'mean': round(m, 1), 'over30': len(L), 'dist': d,
                            'long': [x for _, x in L]}
        print(f'{g:26s} {n:5d} {m:6.1f} {len(L):4d}   ' + ' '.join(f'{x:5d}' for x in d))
    if a.long:
        for f, s in rows + [('captions', capsents)]:
            _, _, L = stats(s)
            for l, x in L:
                print(f'\n[{f}] ({l} words) {x}')
    if a.json:
        json.dump(res, open(a.json, 'w', encoding='utf8'), indent=1, ensure_ascii=False)


if __name__ == '__main__':
    main()
