"""LEX5 select step: one LLM call reads the question and N numbered HYBRID candidates and names, per requested part,
the candidates that contain it. Then the selected intact chunks are packed first.

Views (what the selector sees of each candidate; the EVIDENCE is always the intact lex4 package):
    snippet  header ('[n] <doc title> > <section path>') + the ~SNIPPET_CHARS window of the package text with the
             highest weight of question words (weight = log(1 + N/df) over the N candidates, words folded)
    full     header + the whole package text
    head     header + the first ~SNIPPET_CHARS of the package text (ablation: is the query-hit window what matters?)
Prompt: domain-neutral (no company / metric / period wording). Output: one line per part, 'part: ids'.
Packing: one id per part first (the first listed id of each part, part order), then the other selected ids (listed
order), then fill in HYBRID's RRF order; lex4 redundancy rule (contained or > 50% overlap) and exact rendered budget;
no item is cut. A reply without any parsable 'part: ids' line -> fallback to plain HYBRID-intact packing.
"""
import math
import re

from ...contracts import item_header
from ...normalize import fold
from ..lex4 import core as K

SELECT_VERSION = "lex5-select-v2"                  # v2: parser accepts Arabic comma/semicolon and Persian 'هیچ' (none)
SNIPPET_CHARS = 300
_WORD = re.compile(r"\w+", re.U)

PROMPT = """You are selecting evidence passages for a question. Read the question and the numbered passages.

Question: {question}

Passages:
{passages}

Task: split the question into the parts it asks for (one part per separate piece of information needed to answer it; a simple question has one part). For each part, list the numbers of the passages that directly contain that information, best first. Do not list passages that are only on a related topic. If no passage contains a part, write none.

Reply only with lines of this form, nothing else:
<short name of the part>: <passage numbers separated by commas>"""


def _words(text):
    return [(m.start(), m.end(), fold(m.group(0))) for m in _WORD.finditer(text)]


def question_hits(text, qwords):
    """The (start, folded word) of every question word in text.strip() - the width-independent part of snippet()."""
    return [(a, w) for a, _b, w in _words(text.strip()) if w in qwords]


def snippet(text, qwords, weights, width=SNIPPET_CHARS, hits=None):
    """The width-char window of text holding the highest total weight of distinct question words.
    hits: optional precomputed question_hits(text, qwords) (a cache for callers trying several widths)."""
    t = text.strip()
    if len(t) <= width + 40:
        return t
    if hits is None:
        hits = question_hits(t, qwords)
    if not hits:
        start = 0
    else:
        best, start = -1.0, hits[0][0]
        for i, (a, _w) in enumerate(hits):
            seen, s = set(), 0.0
            for a2, w2 in hits[i:]:
                if a2 - a >= width - 20:
                    break
                if w2 not in seen:
                    seen.add(w2)
                    s += weights.get(w2, 0.0)
            if s > best + 1e-9:
                best, start = s, a
        start = max(0, start - 40)                 # a little left context
        while start > 0 and not t[start - 1].isspace():
            start -= 1
    end = min(len(t), start + width)
    while end < len(t) and not t[end].isspace() and end - start < width + 30:
        end += 1
    body = t[start:end].strip()
    return ("… " if start > 0 else "") + body + (" …" if end < len(t) else "")


def views(P, pks, question, mode):
    """-> list of passage strings (1-based numbering in the header), one per package."""
    qwords = {w for _a, _b, w in _words(question) if len(w) > 1}
    df = {}
    for pk in pks:
        for w in {w for _a, _b, w in _words(pk["text"])} & qwords:
            df[w] = df.get(w, 0) + 1
    n = len(pks) or 1
    weights = {w: math.log(1.0 + n / df[w]) for w in df}
    out = []
    for i, pk in enumerate(pks, 1):
        head = item_header(i, P.item(pk), P.corpus)
        if mode == "full":
            body = pk["text"].strip()
        elif mode == "head":                           # ablation: first ~SNIPPET_CHARS, no query-hit window
            body = snippet(pk["text"], set(), {})
        else:
            body = snippet(pk["text"], qwords, weights)
        out.append(head + "\n" + body)
    return out


def prompt(question, passages):
    return PROMPT.format(question=question.strip(), passages="\n\n".join(passages))


_LINE = re.compile(r"^\s*(?:[-*\d.)\s]*)?(.*?)\s*:\s*(.*)$")


def parse(text, n):
    """-> (parts [(name, [ids 1..n])], ok). ok = at least one line of the form 'part: ids|none'."""
    parts, ok = [], False
    for line in (text or "").splitlines():
        line = line.strip().strip("`").replace("**", "")
        if ":" not in line:
            continue
        name, rhs = line.rsplit(":", 1)
        rhs = rhs.strip()
        ids = [int(x) for x in re.findall(r"\d+", rhs)]
        if not ids and not re.search(r"\bnone\b|هیچ", rhs, re.I):
            continue
        if ids and re.sub(r"[\d\s,;،؛\[\]()#and&.]|passages?|none", "", rhs, flags=re.I).strip():
            continue                                   # prose, not an id list
        ok = True
        ids = [i for i in dict.fromkeys(ids) if 1 <= i <= n]
        parts.append((name.strip(), ids))
    return parts, ok


def pack(Qn, cands, parts, budget=None):
    """Select-first packing -> (lex4 Packer, info)."""
    B = K.Packer(Qn.P, budget or K.C["budget"])
    order = []
    for _name, ids in parts:                           # one per part first
        if ids:
            order.append(cands[ids[0] - 1])
    for _name, ids in parts:                           # then the rest of the selection
        order += [cands[i - 1] for i in ids]
    order = list(dict.fromkeys(order))
    n_sel = len(order)
    placed = 0
    for j, c in enumerate(order + [c for c in Qn.q_fused() if c not in set(order)]):
        p = Qn.pk[c]
        if not K.redundant_b(p, B.sel, K.C["b_overlap"]) and B.fits(p):
            B.add(p)
            placed += int(j < n_sel)
    return B, {"n_selected": n_sel, "selected_placed": placed}
