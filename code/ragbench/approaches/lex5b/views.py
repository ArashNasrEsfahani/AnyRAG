"""LEX5b selector views under a fixed selector token budget (work/datasets_v2/lex5b/PROGRESS.md, declared 2026-09-26).

Candidates are the F2 MaxMean order (lex8 fusion of BM25 + bge-m3 dense + bge-m3 sparse over REAL channel scores). The
selector sees, per candidate, a short header and an excerpt; the EVIDENCE is always the intact lex4 package.

    legend      'D<k>: <document title>' once per document (<= TITLE chars), so titles are not repeated per candidate
    header      '[n] D<k> > <last section title>' (<= SECTION chars)
    smart       the candidate chunk's 1-2 highest-scoring compact source units (lex8 units overlapping the chunk,
                scored by R_unit = MaxMean of the same three channels against the question), clipped to the chunk,
                shown in document order; a table-row unit carries its column-heading line (<= DEP chars) when that
                line lies outside the chunk text shown. The best unit is shown first up to the cap; a second unit only
                when >= MIN_SECOND chars of the cap remain. No units -> the head of the package (counted, rare).
    literal     the old query-word window (lex5 select.snippet: highest weight of distinct question words), same cap
    full        the whole package text (no cap)
Budget: one excerpt cap E per question (the same for every candidate), the largest E in CAPS whose prompt estimate
fits PROMPT_TOKENS. Estimate = Arabic-script chars / 2.55 + other chars / 4.2 (the GLM rates measured in lex5_replay,
the more expensive of the two selectors); it is only a sizing rule - the gate uses the providers' reported usage.
"""
import bisect
import re

from ..lex5 import select as S5

VIEWS_VERSION = "lex5b-views-v1"
PROMPT_TOKENS = 2600
CAPS = tuple(range(420, 59, -20))
TITLE, SECTION, DEP, MIN_SECOND = 80, 50, 80, 60
_AR = re.compile(r"[؀-ۿݐ-ݿﭐ-﷿ﹰ-﻿]")

PROMPT = """You are selecting evidence passages for a question. Read the question and the numbered passages. Each passage is shown by its document, its section and a short excerpt; the full passage is longer.

Question: {question}

Documents:
{legend}

Passages:
{passages}

Task:
1. Split the question into the separate parts it asks for: one part per separate piece of information needed to answer it. A simple question has one part. Use at most 8 parts.
2. For each part, give the passages that together fully support its answer. Include every passage the answer depends on: each value or operand, each condition, qualifier, exception or definition - not only the passage that states the final answer. Join passages that are needed together with +. If a different set of passages would also fully support the part on its own, give it after | as an alternative (at most 2 alternatives, best first). Do not list passages that are only on a related topic. If no passage supports a part, write NONE.

Reply with exactly these lines and nothing else (numbers are passage numbers). The first line gives the number of parts, then one line per part, and the last line is the word END. An alternative must be a different set of passages.
parts=<number of parts>
P1: <passages joined by +> | <alternative passages joined by +>
P2: NONE
END
Always finish your reply with the line END."""


def est_tokens(text):
    ar = len(_AR.findall(text))
    return ar / 2.55 + (len(text) - ar) / 4.2


def _clip(s, n):
    s = " ".join(s.split())
    if len(s) <= n:
        return s
    cut = s[:n]
    sp = cut.rfind(" ")
    if sp > n * 0.6:
        cut = cut[:sp]
    return cut.rstrip() + " …"


def _short(s, n):
    s = " ".join((s or "").split())
    return s if len(s) <= n else s[:n - 1].rstrip() + "…"


class Viewer:
    """Per question: candidates (package dicts, shown order), unit scores Ru (array over the set's units)."""

    def __init__(self, six, corpus, question, pks, Ru=None):
        self.six, self.corpus, self.q, self.pks, self.Ru = six, corpus, question, pks, Ru
        self.docs = list(dict.fromkeys(p["doc_id"] for p in pks))
        self.dkey = {d: "D%d" % (i + 1) for i, d in enumerate(self.docs)}
        self._units = {}

    # -- pieces ---------------------------------------------------------------------------------------------------
    def legend(self):
        return "\n".join("%s: %s" % (self.dkey[d], _short(self.corpus.doc(d).title, TITLE)) for d in self.docs)

    def header(self, n, pk):
        doc = self.corpus.doc(pk["doc_id"])
        path = [t for t in (doc.section_path(pk["section_id"]) if pk.get("section_id") else []) if t]
        sec = (" > " + _short(path[-1], SECTION)) if path else ""
        return "[%d] %s%s" % (n, self.dkey[pk["doc_id"]], sec)

    def units_of(self, pk):
        """lex8 units overlapping the package's body interval, best R_unit first -> [(i, a, b)] clipped."""
        key = pk["chunk"]
        if key not in self._units:
            six, d = self.six, pk["doc_id"]
            a, b = pk["span"]
            ids = six.by_doc.get(d, [])
            starts = six._starts[d]
            j = bisect.bisect_left(starts, b)
            out = []
            for k in range(j - 1, -1, -1):
                i = ids[k]
                ua, ub = int(six.ua[i]), int(six.ub[i])
                if ub <= a:
                    if a - ua > 20000:
                        break
                    continue
                x, y = max(ua, a), min(ub, b)
                if y - x >= 20:
                    out.append((i, x, y))
            out.sort(key=lambda t: (-float(self.Ru[t[0]]), t[1]))
            self._units[key] = out
        return self._units[key]

    def smart(self, pk, cap):
        doc = self.corpus.doc(pk["doc_id"])
        us = self.units_of(pk)
        if not us:
            return _clip(pk["text"], cap), {"units": [], "fallback_head": True}
        chosen, left = [], cap
        for rank, (i, x, y) in enumerate(us[:2]):
            if rank == 1 and left < MIN_SECOND:
                break
            u = self.six.units[i]
            dep = ""
            if u["kind"] == "row" and u["deps"]:
                dep = " ".join(doc.text[p:q].strip() for p, q in u["deps"] if doc.text[p:q].strip())
                dep = _short(dep, DEP) + " | " if dep else ""
            body = doc.text[x:y]
            s = _clip(dep + body, left)
            chosen.append((x, s, i))
            left -= len(s) + 3
        chosen.sort()
        return " … ".join(s for _x, s, _i in chosen), {"units": [i for _x, _s, i in chosen], "fallback_head": False}

    def literal(self, pk, cap, qwords, weights, hits=None):
        s = S5.snippet(pk["text"], qwords, weights, width=cap, hits=hits)
        return s.replace("â€¦", "…").replace("Ã¢â‚¬Â¦", "…"), {}

    # -- whole prompt ---------------------------------------------------------------------------------------------
    def _lit_weights(self):
        import math
        qwords = {w for _a, _b, w in S5._words(self.q) if len(w) > 1}
        df = {}
        for pk in self.pks:
            for w in {w for _a, _b, w in S5._words(pk["text"])} & qwords:
                df[w] = df.get(w, 0) + 1
        n = len(self.pks) or 1
        return qwords, {w: math.log(1.0 + n / df[w]) for w in df}

    def render(self, mode, cap):
        lw = self._lit_weights() if mode == "literal" else None
        blocks, meta = [], []
        for n, pk in enumerate(self.pks, 1):
            if mode == "smart":
                body, m = self.smart(pk, cap)
            elif mode == "literal":
                body, m = self.literal(pk, cap, *lw)
            else:
                body, m = pk["text"].strip(), {}
            blocks.append(self.header(n, pk) + "\n" + body)
            meta.append(dict(m, chunk=pk["chunk"], chars=len(body)))
        return PROMPT.format(question=self.q.strip(), legend=self.legend(), passages="\n\n".join(blocks)), meta

    def fit(self, mode):
        """-> (prompt, meta, cap, est_tokens): the largest cap whose estimate fits PROMPT_TOKENS (smallest cap else).
        'literal' uses the cached fitter (fit_literal: byte-identical output, invariant work done once)."""
        if mode == "literal":
            return self.fit_literal()
        return self.fit_reference(mode)

    def fit_literal(self):
        """Cached literal fit (dev60 fix, results-neutral). Per question once: literal weights, legend, headers, the
        question-word hits of every package. Per cap: only the snippet windows; the estimate is summed from per-part
        character counts (Arabic-script count and length are additive over concatenation, so the float equals
        est_tokens(prompt) exactly); the prompt string is built once, for the chosen cap."""
        qwords, weights = self._lit_weights()
        hits = [S5.question_hits(pk["text"], qwords) for pk in self.pks]
        heads = [self.header(n, pk) + "\n" for n, pk in enumerate(self.pks, 1)]
        shell = PROMPT.format(question=self.q.strip(), legend=self.legend(), passages="")
        n_ar = lambda s: len(_AR.findall(s))  # noqa: E731
        base_ar = n_ar(shell) + sum(n_ar(h) for h in heads)
        base_len = len(shell) + sum(len(h) for h in heads) + 2 * max(0, len(self.pks) - 1)
        memo = {}
        for cap in CAPS:
            bodies, ar, ln = [], base_ar, base_len
            for i, pk in enumerate(self.pks):
                body = self.literal(pk, cap, qwords, weights, hits=hits[i])[0]
                k = (i, body)
                if k not in memo:
                    memo[k] = n_ar(body)
                ar += memo[k]
                ln += len(body)
                bodies.append(body)
            e = ar / 2.55 + (ln - ar) / 4.2
            if e <= PROMPT_TOKENS:
                break
        blocks = [h + b for h, b in zip(heads, bodies)]
        meta = [{"chunk": pk["chunk"], "chars": len(b)} for pk, b in zip(self.pks, bodies)]
        p = PROMPT.format(question=self.q.strip(), legend=self.legend(), passages="\n\n".join(blocks))
        return p, meta, cap, e

    def fit_reference(self, mode):
        """The original fitting loop (re-renders the whole prompt per cap); kept as the reference for the checks."""
        if mode == "full":
            p, m = self.render("full", None)
            return p, m, None, est_tokens(p)
        for cap in CAPS:
            p, m = self.render(mode, cap)
            e = est_tokens(p)
            if e <= PROMPT_TOKENS:
                return p, m, cap, e
        return p, m, cap, e


def attach_starts(six):
    six._starts = {d: [int(six.ua[i]) for i in ids] for d, ids in six.by_doc.items()}
    for d, ids in six.by_doc.items():               # by_doc lists are in unit order = (a, b) order per doc
        assert all(six.ua[ids[k]] <= six.ua[ids[k + 1]] for k in range(len(ids) - 1))
    return six
