"""The document tree as AGENTIC2 reads it (no model): leaves for lazy embedding, and tree expansion of hits.

Leaves (passages). Every section's OWN units, in document order, are merged into passages of at most passage_chars
(never across a section boundary); a unit longer than passage_max_chars is cut into overlapping windows. A passage is
a verbatim slice of Document.text ([start, end) span) and lists the units it overlaps. Its embed text is
"<doc title> > <section path>\\n<text>" (as B3 embeds its chunks), so a short KPI block still carries its heading.

Tree expansion (expand_passage / expand_unit), the candidate a verifier reads:
  * table unit       caption (the previous short caption/heading unit) + header row + the rows that carry a plan term
                     or the matched value, then the following rows while table_max_chars allows
  * detached number  a passage/unit whose edge unit is a number without a label gets the neighbouring short unit
                     (the label) on that side; a label without a number gets the next short unit
  * long unit        a clip of window_chars around the matched value / the passage window
  * the section path is shown to the verifier and to the answer call in every item header (doc title > path)
Every candidate is a v1 search.Candidate, so the v1 verifier, guard and reconciler read it unchanged.
"""
import re

from ...normalize import normalize
from ..agentic import search as S
from ..agentic.numbers import candidate_numbers, find_numbers

_LETTER = re.compile(r"[^\W\d_]", re.UNICODE)


class Passage:
    __slots__ = ("pid", "doc_id", "section_id", "unit_ids", "span", "text", "embed_text")

    def __init__(self, pid, doc_id, section_id, unit_ids, span, text, embed_text):
        self.pid, self.doc_id, self.section_id = pid, doc_id, section_id
        self.unit_ids, self.span, self.text, self.embed_text = list(unit_ids), list(span), text, embed_text


def _breadcrumb(doc, section_id):
    parts = [doc.title] + [t for t in doc.section_path(section_id) if t]
    return " > ".join(p for p in parts if p)


def _cut_windows(text, size, overlap):
    """[(a, b)] windows of at most `size` chars over `text`, breaking at a blank line / newline / space."""
    out, start, n = [], 0, len(text)
    while start < n:
        if n - start <= size:
            out.append((start, n))
            break
        hi = start + size
        lo = start + size // 2
        cut = max(text.rfind("\n", lo, hi), text.rfind(" ", lo, hi))
        end = cut if cut > lo else hi
        out.append((start, end))
        nxt = max(end - overlap, start + 1)
        while nxt < end and not text[nxt - 1].isspace():
            nxt += 1
        while nxt < n and text[nxt].isspace():
            nxt += 1
        start = nxt if nxt > start else end
    return out


def build_passages(doc, opts):
    """-> [Passage] of one document (deterministic)."""
    size = int(opts["passage_chars"])
    max_unit = int(opts["passage_max_chars"])
    overlap = int(opts["passage_overlap_chars"])
    with_path = bool(opts.get("embed_with_path", True))
    out = []

    def add(section, uids, a, b):
        text = doc.text[a:b]
        if not text.strip():
            return
        crumb = _breadcrumb(doc, section.id)
        out.append(Passage(len(out), doc.id, section.id, uids, [a, b], text,
                           (crumb + "\n" + text) if (with_path and crumb) else text))

    for s in doc.sections:
        units = [doc.unit(u) for u in s.unit_ids]
        cur = []
        for u in units:
            ulen = u.span[1] - u.span[0]
            if ulen > max_unit:
                if cur:
                    add(s, [x.id for x in cur], cur[0].span[0], cur[-1].span[1])
                    cur = []
                for a, b in _cut_windows(u.text, size, overlap):
                    add(s, [u.id], u.span[0] + a, u.span[0] + b)
                continue
            if cur and u.span[1] - cur[0].span[0] > size:
                add(s, [x.id for x in cur], cur[0].span[0], cur[-1].span[1])
                cur = []
            cur.append(u)
        if cur:
            add(s, [x.id for x in cur], cur[0].span[0], cur[-1].span[1])
    return out


# --------------------------------------------------------------------------
# expansion
# --------------------------------------------------------------------------

def _letters(t):
    return len(_LETTER.findall(t or ""))


def _is_table(unit):
    return unit.kind == "table" or (unit.text.count("\n") >= 1 and unit.text.count(" | ") >= 2)


def _number_only(text, unit_kind):
    return bool(candidate_numbers(find_numbers(text), unit_kind)) and _letters(text) <= 3


def _label_only(text, unit_kind):
    return _letters(text) >= 2 and not candidate_numbers(find_numbers(text), unit_kind)


def table_candidate(doc, section, idx, ctx, opts, value_tokens=(), round_no=1, tree=True):
    """Caption + header + matching rows (+ following rows while they fit) of the table unit section.unit_ids[idx]."""
    sec_uids = list(section.unit_ids)
    unit = doc.unit(sec_uids[idx])
    max_chars = int(opts["table_max_chars"])
    lines = unit.text.split("\n")
    if not tree:
        text = unit.text[:max_chars]
        return S.Candidate(doc.id, section.id, [unit.id], text, [unit.span[0], unit.span[0] + len(text)], "table",
                           [unit.id], round_no)
    want = set()
    for i, ln in enumerate(lines[1:], 1):
        if ctx is not None and ctx.has_term(ln):
            want.add(i)
        if value_tokens and any(v and v in ln for v in value_tokens):
            want.add(i)
    rows = [0] + sorted(want)
    used = sum(len(lines[i]) + 1 for i in rows)
    for i in range(1, len(lines)):          # fill with the remaining rows in order while they fit
        if i in want:
            continue
        if used + len(lines[i]) + 1 > max_chars:
            continue
        rows.append(i)
        used += len(lines[i]) + 1
    rows = sorted(set(rows))
    while len(rows) > 1 and sum(len(lines[i]) + 1 for i in rows) > max_chars:
        drop = next((i for i in reversed(rows) if i not in want and i != 0), None) or rows[-1]
        rows.remove(drop)
    body = "\n".join(lines[i] for i in rows)
    uids, start, text = [unit.id], unit.span[0], body
    if idx > 0:
        prev = doc.unit(sec_uids[idx - 1])
        if prev.kind in ("caption", "heading") or (len(prev.text) <= int(opts["neighbour_chars"]) and not _is_table(prev)):
            if len(prev.text) + len(body) + 2 <= max_chars + 250:
                text = prev.text + "\n\n" + body
                uids = [prev.id, unit.id]
                start = prev.span[0]
    span = [start, start + len(text)] if doc.text[start:start + len(text)] == text else None
    return S.Candidate(doc.id, section.id, uids, text, span, "table", [unit.id], round_no)


def _grow(doc, units, lo, hi, unit_kind, opts):
    """Detached numbers / labels at the edges of units[lo..hi]: add the short neighbour that completes them."""
    nchars = int(opts["neighbour_chars"])
    max_chars = int(opts["window_chars"])
    for _ in range(int(opts["window_neighbours"])):
        grew = False
        first, last = units[lo], units[hi]
        if lo > 0 and _number_only(first.text, unit_kind):
            prev = units[lo - 1]
            if len(prev.text) <= nchars and not _is_table(prev) and last.span[1] - prev.span[0] <= max_chars:
                lo -= 1
                grew = True
        # a trailing label waits for its number; a lone number (no label before it in the window) may be labelled after
        wants_next = _label_only(last.text, unit_kind) or (lo == hi and _number_only(last.text, unit_kind))
        if hi + 1 < len(units) and wants_next:
            nxt = units[hi + 1]
            if len(nxt.text) <= nchars and not _is_table(nxt) and nxt.span[1] - units[lo].span[0] <= max_chars:
                hi += 1
                grew = True
        if not grew:
            break
    return lo, hi


def expand_passage(doc, passage, ctx, opts, unit_kind, tree=True):
    """A vector hit (passage) -> [Candidate]."""
    section = doc.section(passage.section_id)
    sec_uids = list(section.unit_ids)
    units = [doc.unit(u) for u in sec_uids]
    pos = {u: i for i, u in enumerate(sec_uids)}
    out = []
    idxs = [pos[u] for u in passage.unit_ids if u in pos]
    if not idxs:
        return out
    table_idx = [i for i in idxs if _is_table(units[i])]
    if table_idx:
        for i in table_idx:
            out.append(table_candidate(doc, section, i, ctx, opts, tree=tree))
        idxs = [i for i in idxs if i not in table_idx]
        if not idxs:
            return out
    lo, hi = min(idxs), max(idxs)
    a, b = passage.span
    partial = len(idxs) == 1 and (a > units[lo].span[0] or b < units[hi].span[1])
    if partial or not tree:
        text = doc.text[a:b]
        out.append(S.Candidate(doc.id, section.id, [sec_uids[i] for i in range(lo, hi + 1)], text, [a, b],
                               "clip" if partial else "passage", [sec_uids[lo]]))
        return out
    # a passage is <= passage_chars; growth adds at most window_neighbours short units per side
    lo, hi = _grow(doc, units, lo, hi, unit_kind, dict(opts, window_chars=max(int(opts["window_chars"]), b - a)))
    a, b = units[lo].span[0], units[hi].span[1]
    text = doc.text[a:b]
    out.append(S.Candidate(doc.id, section.id, sec_uids[lo:hi + 1], text, [a, b], "passage",
                           [sec_uids[i] for i in idxs]))
    return out


def expand_unit(doc, unit_id, value, ctx, opts, unit_kind, tree=True):
    """A fact-DB row (its unit and value) -> Candidate: the unit + short label neighbours; a clip of a long unit around
    the value; the table rows holding the value."""
    unit = doc.unit(unit_id)
    section = doc.section(unit.section_id)
    sec_uids = list(section.unit_ids)
    if unit_id not in sec_uids:
        return None
    idx = sec_uids.index(unit_id)
    max_chars = int(opts["window_chars"])
    if _is_table(unit):
        return table_candidate(doc, section, idx, ctx, opts, value_tokens=[value], tree=tree)
    if len(unit.text) > max_chars:
        p = unit.text.find(value) if value else -1
        if p < 0:
            p = 0
        clips = S.clip_windows(doc, unit, [p], max_chars, 1)
        if clips:
            text, span = clips[0]
            return S.Candidate(doc.id, section.id, [unit_id], text, span, "clip", [unit_id])
    if not tree:
        return S.Candidate(doc.id, section.id, [unit_id], unit.text, list(unit.span), "unit", [unit_id])
    units = [doc.unit(u) for u in sec_uids]
    lo, hi = idx, idx
    lo, hi = _grow(doc, units, lo, hi, unit_kind, opts)
    if lo == hi == idx and _letters(unit.text) < 12 and idx > 0:
        prev = units[idx - 1]           # a value whose label sits in the previous short unit (KPI line)
        if len(prev.text) <= int(opts["neighbour_chars"]) and not _is_table(prev):
            lo = idx - 1
    a, b = units[lo].span[0], units[hi].span[1]
    return S.Candidate(doc.id, section.id, sec_uids[lo:hi + 1], doc.text[a:b], [a, b], "unit", [unit_id])


def search_context(plan, d_lang):
    """The v1 term matcher (morphology-aware regexes of the plan's terms) for table rows and term bonuses."""
    try:
        return S.SearchContext(plan, d_lang)
    except Exception:  # noqa: BLE001
        return None


def plain(text):
    return normalize(text or "")
