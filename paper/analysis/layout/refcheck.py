"""Float placement check: each fig/tab label's page vs its first \\ref inside the float's home section.
Home section = section whose source file contains (or \\inputs) the float."""
import re, sys, subprocess, shutil, os
paper = r'D:/KareOne/RAG Paper/paper'
here = os.path.dirname(os.path.abspath(__file__)).replace('\\', '/')
if '--nobuild' not in sys.argv:
    shutil.copy(paper + '/main.bbl', paper + '/refcheck.bbl')
    pre = r'\def\RCREF#1#2#3{}\def\RCREFF#1#2#3{}\def\RCLBL#1#2#3{}'
    cmd = ['xelatex', '-jobname=refcheck', '-interaction=batchmode', '-no-pdf',
           pre + r'\AddToHook{begindocument/end}{\input{' + here + r'/rc.tex}}\input{main}']
    for _ in range(2):
        subprocess.run(cmd, cwd=paper, stdout=subprocess.DEVNULL)
aux = open(paper + '/refcheck.aux', encoding='utf8', errors='replace').read()

# home section of each float label (from the ORIGINAL source layout)
origdir = here + '/orig'
main = open(here + '/main.tex.orig', encoding='utf8').read()
files = re.findall(r'^\\input\{(sections/[^}]*)\}', main, re.M)
home = {}
num = 0; app = False; letter = 0
cur = None
for f in files:
    src = open(origdir + '/' + f + '.tex', encoding='utf8').read()
    src = re.sub(r'(?<!\\)%.*', '', src)
    if re.search(r'\\section\{', src):
        if f.split('/')[1][0].isalpha():  # appendix files start with a letter
            cur = chr(ord('A') + letter); letter += 1
        else:
            num += 1; cur = str(num)
    text = src
    for inc in re.findall(r'\\input\{((?:tables|figures)/[^}]*)\}', src):
        text += open(origdir + '/' + inc + '.tex', encoding='utf8').read()
    for lab in re.findall(r'\\label\{((?:fig|tab)[^}]*)\}', text):
        home[lab] = cur

refs = {}
for k, s, p in re.findall(r'\\RCREF\{([^}]*)\}\{([^}]*)\}\{(\d+)\}', aux):
    refs.setdefault((k, s), int(p))
    refs.setdefault((k, '*'), int(p))
lbl = {}
for k, p in re.findall(r'\\newlabel\{((?:fig|tab)[^}]*)\}\{\{[^}]*\}\{(\d+)\}', aux):
    lbl[k] = int(p)
secs = []
for s, p in re.findall(r'contentsline \{section\}\{\\numberline \{([^}]*)\}.*?\}\{(\d+)\}\{(?:section|appendix)', aux):
    secs.append((s, int(p)))
secstart = dict(secs)
order = [s for s, _ in secs]
print('section starts:', ' '.join(f'{s}:p{p}' for s, p in secs))
bad = 0
for k, fp in sorted(lbl.items(), key=lambda x: x[1]):
    hs = home.get(k, '?')
    rp = refs.get((k, hs))
    where = 'home'
    if rp is None:
        rp = refs.get((k, '*')); where = 'elsewhere'
    nxt = None
    if hs in order and order.index(hs) + 1 < len(order):
        nxt = secstart[order[order.index(hs) + 1]]
    start = secstart.get(hs)
    note = ''
    if rp is None:
        note = ' NO REF'
    elif fp - rp > 1 or fp < rp:
        note += ' FAR(%+d)' % (fp - rp); bad += 1
    if nxt is not None and fp > nxt:
        note += ' AFTER-NEXT-SECTION(p%d)' % nxt; bad += 1
    if start is not None and fp < start:
        note += ' BEFORE-OWN-SECTION'; bad += 1
    if nxt is not None and fp == nxt:
        note += ' (on page where next section starts: check visually)'
    print(f'{k:22s} sec {hs:2s} float p{fp:2d}  first ref p{rp} ({where}){note}')
print('problems', bad)
