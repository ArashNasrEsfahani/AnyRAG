"""LEX8 compact source units (INDEX time; question-independent; the original chunks are kept unchanged).

Per document, per section, in document order (heading units skipped; units never cross a section):
  prose pieces   paragraph / list / fact / caption units, and table units without a " | " line (chart / numeric
                 blocks). A block <= MAX chars is an atomic piece (never cut). A longer block is split into sentence
                 segments (lex4 spans.segments: sentence ends and line breaks); a single segment > MAX chars is NO
                 piece (it stays reachable through its original chunk) and breaks the run.
  prose units    consecutive pieces grouped: the next piece joins while the group is < TARGET chars and the grouped
                 span stays <= MAX chars (the corpus's "paragraph" units are often extracted lines, e.g. 63 chars mean
                 in set C, so a unit is a run of them, not one line).
  table rows     a table unit with " | " lines: each non-empty row line (evidence.table_structure) is one unit; its
                 dependencies = the table caption and column-heading line when outside the row.
Unit record: {i, doc_id, section_id, a, b, kind (prose | row), deps: [[a, b]], parents: [chunk ids]}; a:b are
offsets into Document.text, so Document.text[a:b] is verbatim source.
Compact candidates: one unit, or two adjacent units of the same document and section with (b2 - a1) <= MAX.
"""
import bisect

from ...evidence import _DocView, table_structure
from ..lex4.spans import segments

UNITS_VERSION = "lex8-units-v1"
TARGET, MAX = 450, 900
PROSE_KINDS = ("paragraph", "list", "fact", "caption")


def _trim(text, a, b):
    while a < b and text[a].isspace():
        a += 1
    while b > a and text[b - 1].isspace():
        b -= 1
    return a, b


def _pieces(doc, u):
    """atomic prose pieces [(a, b)] of one block, or None for a break (an oversized segment)."""
    a, b = _trim(doc.text, u.span[0], u.span[1])
    if b <= a:
        return []
    if b - a <= MAX:
        return [(a, b)]
    out = []
    for sa, sb in segments(doc.text[a:b]):
        if sb - sa > MAX:
            out.append(None)
        else:
            out.append((a + sa, a + sb))
    return out


def _group(pieces):
    """group consecutive pieces toward TARGET without exceeding MAX; None breaks a run."""
    units, cur = [], None
    for p in pieces:
        if p is None:
            if cur:
                units.append(cur)
            cur = None
            continue
        if cur is None:
            cur = [p[0], p[1]]
        elif (cur[1] - cur[0]) < TARGET and (p[1] - cur[0]) <= MAX:
            cur[1] = p[1]
        else:
            units.append(cur)
            cur = [p[0], p[1]]
    if cur:
        units.append(cur)
    return units


def is_structured_table(u):
    return u.kind == "table" and any(" | " in ln for ln in u.text.split("\n"))


def doc_units(doc, view=None):
    """-> [{doc_id, section_id, a, b, kind, deps}] of one document, document order."""
    view = view or _DocView(doc)
    out = []
    for sec in doc.sections:
        pieces = []

        def flush():
            for a, b in _group(pieces):
                out.append({"doc_id": doc.id, "section_id": sec.id, "a": a, "b": b, "kind": "prose", "deps": []})
            pieces.clear()

        for uid in sec.unit_ids:
            u = doc.unit(uid)
            if u.kind == "heading":
                continue
            if is_structured_table(u):
                flush()
                ts = table_structure(doc, view, u)
                deps = [list(x[1]) for x in ts["caption"]] + ([list(ts["header"][1])] if ts["header"] else [])
                for _j, ln, sp in ts["rows"].get(u.id, []):
                    a, b = _trim(doc.text, sp[0], sp[1])
                    if b <= a or not ln.strip():
                        continue
                    d = [x for x in deps if not (a <= x[0] and x[1] <= b) and doc.text[x[0]:x[1]].strip()]
                    out.append({"doc_id": doc.id, "section_id": sec.id, "a": a, "b": b, "kind": "row", "deps": d})
                continue
            pieces.extend(_pieces(doc, u))
        flush()
    out.sort(key=lambda x: (x["a"], x["b"]))
    return out


def attach_parents(units, chunks):
    """parents = chunks of the same doc whose span contains the unit body, else those overlapping it."""
    by_doc = {}
    for ci, c in enumerate(chunks):
        by_doc.setdefault(c["doc_id"], []).append((int(c["span"][0]), int(c["span"][1]), ci))
    for v in by_doc.values():
        v.sort()
    starts = {d: [x[0] for x in v] for d, v in by_doc.items()}
    for u in units:
        v = by_doc.get(u["doc_id"], [])
        hi = bisect.bisect_right(starts.get(u["doc_id"], []), u["a"])
        cont, over = [], []
        j = hi - 1
        while j >= 0:
            a, b, ci = v[j]
            if b > u["a"]:
                (cont if (a <= u["a"] and u["b"] <= b) else over).append(ci)
            elif u["a"] - a > 20000:
                break
            j -= 1
        k = hi
        while k < len(v) and v[k][0] < u["b"]:
            over.append(v[k][2])
            k += 1
        u["parents"] = sorted(cont) if cont else sorted(set(over))
    return units


def build_set_units(corpus_set, chunks):
    units = []
    for doc in corpus_set.docs:
        units.extend(doc_units(doc))
    for i, u in enumerate(units):
        u["i"] = i
    return attach_parents(units, chunks)


def breadcrumb(doc, section_id):
    parts = [doc.title] + (doc.section_path(section_id) if section_id else [])
    return " > ".join(p for p in parts if p)


def dep_text(doc, deps):
    return "\n".join(doc.text[a:b].strip() for a, b in deps if doc.text[a:b].strip())


def index_text(doc, u):
    """breadcrumb + dependency lines + body (the chunk index-text convention: '<breadcrumb>\\n<text>')."""
    d = dep_text(doc, u["deps"])
    body = doc.text[u["a"]:u["b"]]
    return breadcrumb(doc, u["section_id"]) + "\n" + ((d + "\n") if d else "") + body


def pairs(units):
    """adjacent same-doc, same-section unit pairs within MAX body chars -> [(i, i+1)]."""
    out = []
    for i in range(len(units) - 1):
        x, y = units[i], units[i + 1]
        if x["doc_id"] == y["doc_id"] and x["section_id"] == y["section_id"] and y["b"] - x["a"] <= MAX:
            out.append((i, i + 1))
    return out
