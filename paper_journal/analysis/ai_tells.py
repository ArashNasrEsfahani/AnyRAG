"""Flag AI-writing tells in the paper prose (NATURAL_STYLE.md).

Scans sections/*.tex (prose; tabular, tikzpicture, lstlisting, verbatim and comments skipped) and the
captions of tables/*.tex and figures/*.tex. Reports, per file, the count of hits by category and every
hit as file:line. Categories:
  word       AI-tell vocabulary (delve, crucial, robust, leverage, notably, ...)
  phrase     stock phrases (it is worth noting, plays a key role, sheds light, in order to, ...)
  signpost   over-signposting (In this section we, Overall, In summary, As mentioned, ...)
  negpar     negative parallelism (not only ... but also; "not X but Y"; ", not X" reveal; rather than)
  reveal     colon-then-reveal set-ups and rhetorical questions
  hedge      hedge-and-boost pairs and stacked hedges
  ingtail    superficial -ing tails (", highlighting ...", ", reflecting ...")
  emdash     em dashes (---); flagged when a file has more than 3
  scare      scare quotes (``one or two words'')
  figure     stock metaphors and drama (levers, pays off, the picture, headline, load-bearing, proved fragile, ...)
  opener     formulaic sentence openers (Yet, Thus, Indeed, Crucially, Instead, That said, ...)
  colon      colon after a lower-case word followed by lower-case text (set-up then reveal; lists are fine
             but each one is worth a look)
  triad      soft: "a, b(,) and c" with three single-word items (review by hand; not counted in the total)
The title (main.tex) is not scanned, so its "comprehensive" is exempt.

Usage: python analysis/ai_tells.py [--root DIR] [--summary] [--soft]
  --root DIR   scan another copy laid out like analysis/readability_orig (sections flattened)
  --summary    totals per file and category only
  --soft       also list the soft triad hits
"""
import re, os, sys, glob, argparse, collections

HERE = os.path.dirname(os.path.abspath(__file__))
PAPER = os.path.dirname(HERE)

WORDS = r'''delv\w* crucial\w* pivotal\w* robust\w* leverag\w* underscor\w* notabl\w* comprehensive\w*
landscape\w* intricat\w* nuanc\w* showcas\w* foster\w* seamless\w* realm\w* tapestr\w* testament\w*
multifaceted paradigm\w* utiliz\w* utilis\w* facilitat\w* bolster\w* streamlin\w* meticulous\w*
vital\w* vibrant groundbreaking novel moreover furthermore additionally importantly interestingly
remarkabl\w* intriguing\w* holistic\w* synerg\w* unveil\w* harness\w* empower\w* elevat\w* embark\w*
noteworthy invaluable profound\w* paramount commendabl\w* garner\w* enduring underpin\w* navigat\w*
ever-evolving cutting-edge state-of-the-art game-chang\w* highlight\w* emphasiz\w* enhanc\w*'''.split()
WORD_RE = re.compile(r'\b(' + '|'.join(WORDS) + r')\b', re.I)

PHRASES = [r'it is (worth|important to) not(e|ing)', r'it should be noted', r'worth noting',
           r'plays? an? (key|crucial|pivotal|central|vital|important|significant) role',
           r'sheds? light', r'paves? the way', r'valuable insight', r'in order to', r'due to the fact',
           r'a (wide|broad|diverse) (range|array|variety) of', r'serves? as', r'stands? as',
           r'functions? as', r'a testament', r'key (insight|takeaway|finding|factor|role|aspect)s?',
           r'at its core', r'the (crux|heart) of', r'in the context of', r'a growing body',
           r'(rich|deep|deeper) understanding', r'nuanced', r'it is clear that', r'clearly shows?',
           r'in today', r'when it comes to', r'first and foremost', r'goes beyond', r'beyond (mere|just)']
PHRASE_RE = re.compile(r'\b(' + '|'.join(PHRASES) + r')', re.I)

SIGNPOSTS = [r'in this (section|paper|appendix|work),? we', r'this (section|appendix|paper) (describes|presents|discusses|reports|introduces|details|outlines)',
             r'we now (turn|describe|present)', r'as (discussed|mentioned|noted|shown|described) (above|earlier|before|previously)',
             r'as mentioned', r'(two|three|four|five)fold', r'(that|which) follows? (show|describe|present)', r'(^|[.!?]\s+)(overall|in summary|in short|to summari[sz]e|taken together|in conclusion|ultimately|in essence|all in all|to sum up),',
             r'the (following|next) (section|paragraphs?) (describes?|presents?|shows?)', r'let us', r'(we|this) (will|shall) (now )?(show|describe|discuss)']
SIGN_RE = re.compile('(' + '|'.join(SIGNPOSTS) + ')', re.I)

NEGPAR_RE = re.compile(r'\bnot (only|just|merely|simply)\b[^.;]{0,80}\bbut\b|\bnot\b[^.;,]{1,40},? but (rather )?\b|'
                       r',\s+not\s+(?!only|just|all|every|significant|enough|yet|least|always|be|been|the\s+answer)\w+[^.;,]{0,40}[.;]|'
                       r'\brather than\b|\bno longer\b[^.;]{0,40}\bbut\b', re.I)
REVEAL_RE = re.compile(r'\b(the|one|a) (answer|result|reason|lesson|point|catch|upshot|takeaway|twist|key|verdict|pattern|picture|story) (is|was)?\s*(simple|clear|this)?\s*:|'
                       r'\b(here is|here\'s) (why|how|what)|\b(simply|simple) put\b|\?\s', re.I)
HEDGE_RE = re.compile(r'\b(may|might|could|can)\s+(significantly|substantially|greatly|dramatically|considerably|drastically)\b|'
                      r'\b(could|may|might)\s+potentially\b|\barguably\s+(the\s+)?(most|best|largest)\b|'
                      r'\b(potentially|possibly)\s+(significant|substantial|major)\b', re.I)
ING_RE = re.compile(r',\s+(highlighting|underscoring|emphasizing|emphasising|reflecting|showcasing|demonstrating|illustrating|signaling|signalling|suggesting|indicating|revealing|confirming|making it|ensuring|contributing to|paving)\b', re.I)
EMDASH_RE = re.compile(r'(?<!-)---(?!-)|\u2014')
SCARE_RE = re.compile(r"``[A-Za-z-]+(\s[A-Za-z-]+)?''|\"[A-Za-z-]+(\s[A-Za-z-]+)?\"")
FIGURE_RE = re.compile(r'\b(levers?|pays? (off|for itself)|paid off|the (big |full |whole )?picture|headline\w*|the story|load-bearing|'
                       r'hinges? on|at play|boils? down|sweet spot|double-edged|under the hood|the bottom line|the needle|'
                       r'manufactur\w+|proved? (fragile|decisive|costly)|pressing|stark\w*|striking\w*|telling|'
                       r'game|toolkit|journey|recipe|north star|silver bullet|blind spot|lens|unlock\w*|'
                       r'(the )?heavy lifting|front and center|takes? center stage)\b', re.I)
OPENER_RE = re.compile(r'(^|[.!?]\s+)(Yet|Thus|Hence|Indeed|Crucially|Strikingly|Notably|Still|Instead|Conversely|Put simply|In other words|That said|Of course|Here),', re.I)
COLON_RE = re.compile(r'(?<![A-Za-z]:)\b[a-z]+:\s+(?=[a-z])')
TRIAD_RE = re.compile(r'\b([a-z]{3,}), ([a-z]{3,}),? (and|or) ([a-z]{3,})\b')

SKIP_ENVS = ['tabular', 'tabular*', 'tabularx', 'tikzpicture', 'axis', 'lstlisting', 'verbatim', 'Verbatim',
             'filecontents', 'equation', 'align', 'math']


def clean_line(ln):
    ln = re.sub(r'(?<!\\)%.*', '', ln)
    ln = re.sub(r'\\(cite[pt]?|ref|label|input|url|href|eqref|autoref|cref|Cref)\*?(\[[^\]]*\])?\{[^}]*\}', ' ', ln)
    ln = re.sub(r'\$[^$]*\$', ' X ', ln)
    return ln


def prose_lines(text):
    """Yield (lineno, text) for prose lines, skipping listed environments."""
    depth = 0
    for i, ln in enumerate(text.split('\n'), 1):
        s = clean_line(ln)
        opens = len(re.findall(r'\\begin\{(' + '|'.join(re.escape(e) for e in SKIP_ENVS) + r')\}', s))
        closes = len(re.findall(r'\\end\{(' + '|'.join(re.escape(e) for e in SKIP_ENVS) + r')\}', s))
        if depth == 0 and opens == 0:
            yield i, s
        depth += opens - closes
        depth = max(depth, 0)


def caption_lines(text):
    """Yield (lineno, text) for the lines of \\caption{...} arguments."""
    lines = text.split('\n')
    flat = '\n'.join(lines)
    for m in re.finditer(r'\\caption(\[[^\]]*\])?\{', flat):
        d, j = 0, m.end() - 1
        for j in range(m.end() - 1, len(flat)):
            if flat[j] == '{' and flat[j - 1] != '\\':
                d += 1
            elif flat[j] == '}' and flat[j - 1] != '\\':
                d -= 1
                if d == 0:
                    break
        start = flat.count('\n', 0, m.start()) + 1
        for k, ln in enumerate(flat[m.end():j].split('\n')):
            yield start + k, clean_line(ln)


def scan(path, text, captions_only):
    hits = []
    src = caption_lines(text) if captions_only else prose_lines(text)
    if not captions_only:   # captions inside section files are covered by prose_lines (figure envs not skipped)
        pass
    for ln, s in src:
        for cat, rx in [('word', WORD_RE), ('phrase', PHRASE_RE), ('signpost', SIGN_RE), ('negpar', NEGPAR_RE),
                        ('reveal', REVEAL_RE), ('hedge', HEDGE_RE), ('ingtail', ING_RE), ('emdash', EMDASH_RE),
                        ('scare', SCARE_RE), ('figure', FIGURE_RE), ('opener', OPENER_RE), ('colon', COLON_RE),
                        ('triad', TRIAD_RE)]:
            for m in rx.finditer(s):
                hits.append((cat, ln, m.group(0).strip()))
    em = [h for h in hits if h[0] == 'emdash']
    if len(em) <= 3:
        hits = [h for h in hits if h[0] != 'emdash']
    return hits


def files(root):
    out = []
    sec = os.path.join(root, 'sections')
    if os.path.isdir(sec):
        out += [(f, False) for f in sorted(glob.glob(sec + '/*.tex'))]
    else:
        out += [(f, False) for f in sorted(glob.glob(root + '/[0-9A-E]*.tex'))]
    out += [(f, True) for f in sorted(glob.glob(root + '/tables/*.tex') + glob.glob(root + '/figures/*.tex'))]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default=PAPER)
    ap.add_argument('--summary', action='store_true')
    ap.add_argument('--soft', action='store_true')
    a = ap.parse_args()
    total = collections.Counter()
    nfiles = 0
    for f, cap in files(a.root):
        text = open(f, encoding='utf8').read()
        hits = scan(f, text, cap)
        hard = [h for h in hits if h[0] != 'triad']
        soft = [h for h in hits if h[0] == 'triad']
        rel = os.path.relpath(f, a.root).replace(os.sep, '/')
        c = collections.Counter(h[0] for h in hard)
        total.update(c)
        total['triad(soft)'] += len(soft)
        if not hard and not (a.soft and soft):
            continue
        nfiles += 1
        print(f'{rel}: {len(hard)} ' + ' '.join(f'{k}={v}' for k, v in sorted(c.items())) +
              (f' [triad(soft)={len(soft)}]' if soft else ''))
        if not a.summary:
            for cat, ln, m in hard + (soft if a.soft else []):
                print(f'  {rel}:{ln}  {cat:8s} {m}')
    hard_total = sum(v for k, v in total.items() if k != 'triad(soft)')
    print(f'TOTAL {hard_total} hard tells in {nfiles} files; by category: ' +
          ', '.join(f'{k}={v}' for k, v in sorted(total.items())))


if __name__ == '__main__':
    main()
