"""Source-grounded evidence blocks (2026-09-24 upgrade foundation; Astra's item 2).

    items, report = build_blocks(candidates, corpus, budget, requirements=None, question=None, cfg=None)
    ev = pack_evidence(items, corpus, budget, cfg)      # every item fits whole: selection used the exact rendering

Candidates are ranked support, best first: hybrid Hits, RetrievedItems or dicts with doc_id and a span (Document.text
offsets) or unit_ids; optional ranks {"bm25": r, "dense": r}, rrf / score. Ranking is theirs; this module only turns
support into readable, source-grounded text and selects it against the budget.

Steps
  1. support: every candidate is mapped to the corpus units it overlaps, per unit (so a chunk that spans several
     sections becomes one piece per section, never text across unrelated headings);
  2. merge BEFORE selection: overlapping or directly adjacent prose pieces of one section (only whitespace between
     them) become one support while the union stays within prose_max; table pieces of one table merge by ROW
     (the rows they show), not by table id;
  3. blocks:
       prose  700-1,000 chars: the support snapped outward to sentence / line boundaries, then grown sentence by
              sentence (forward and backward in turn) inside its section's run of non-table units until >= 700 chars,
              never past 1,000 and never across a section boundary or into a table. Verbatim slices (span set).
       tables caption (a caption line inside the table unit, or the caption unit directly before the table in the
              same section) + the column-heading line (when the table repeats one) + the relevant rows (rows of the
              support that match the question / requirement terms; all support rows when none match), <= 1,200 chars;
              rows that do not fit go to a further block of the same table (disjoint rows, heading repeated).
     A block that overlaps an earlier (better) block of the same section is merged into it when the union fits,
     else clipped to what it adds; a block that adds < min_piece chars is dropped as redundant.
  4. selection against the EXACT rendered cost (contracts.item_header + "\\n" + text, "\\n\\n" between items):
     one sufficient block per requirement first (requirement order; a block covers a requirement when its header
     path + text match every field given: terms, entity, period), then the rest by relevance: up to max_blocks (6)
     for ordinary questions, up to max_blocks_req (8) when the reserved blocks need it. A block that does not fit
     is trimmed (prose to a sentence boundary, tables by rows) when >= min_trim chars of it fit, else dropped.
  5. order: single-fact questions by relevance; comparisons (requirements naming > 1 entity) grouped by entity in
     requirement order; series (> 1 period in the requirements, or > 1 year in the question) by period.

report: per block doc, section, spans, rows, channel ranks, candidates, covered requirement ids, reserved_for,
trimmed; what was dropped and why (budget | cap | redundant); per requirement covered | partial | missing.
"""
import re

from .contracts import RetrievedItem, item_header, pack_evidence
from .normalize import fold

EVIDENCE_VERSION = 1

DEFAULTS = {
    "prose_min": 700,
    "prose_max": 1000,
    "table_max": 1200,
    "max_blocks": 6,           # ordinary questions (4-6 in practice: the budget is 6,000 chars)
    "max_blocks_req": 8,       # when one block per requirement needs more
    "min_trim": 300,           # a trimmed block keeps at least this many chars of text
    "min_piece": 150,          # a clipped block adding fewer new chars is redundant
    "order": "auto",           # auto | relevance | entity | period
}

_SENT_END = re.compile(r"[.!?؟؛;](?=\s)")       # . ! ? ؟ ؛ ; followed by whitespace
_YEAR = re.compile(r"(?<!\d)(1[34]\d\d|19\d\d|20\d\d)(?!\d)")


# --------------------------------------------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------------------------------------------

def _get(c, k, default=None):
    if isinstance(c, dict):
        return c.get(k, default)
    return getattr(c, k, default)


def _tokens(text):
    from .approaches.baselines import tokenize
    return set(tokenize(text or ""))


def _as_list(v):
    if v is None:
        return []
    if isinstance(v, (list, tuple, set)):
        return [str(x) for x in v if str(x).strip()]
    return [str(v)] if str(v).strip() else []


def _norm_req(r, i):
    terms = _as_list(r.get("terms")) + _as_list(r.get("metric")) + _as_list(r.get("keywords")) + _as_list(r.get("facet"))
    return {"id": str(r.get("id") if r.get("id") is not None else "r%d" % (i + 1)), "terms": terms,
            "entity": _as_list(r.get("entity")) + _as_list(r.get("entities")),
            "period": _as_list(r.get("period")) + _as_list(r.get("periods"))}


def _period_num(p):
    m = re.search(r"\d+", str(p or ""))
    return int(m.group(0)) if m else None


def span_overlap(a, b):
    """Share of span a covered by span b ([start, end) lists); 0 when either is missing."""
    if not a or not b:
        return 0.0
    inter = min(a[1], b[1]) - max(a[0], b[0])
    return max(0, inter) / max(1, a[1] - a[0])


def row_overlap(rows_a, rows_b):
    """Share of the rows of a ([unit_id, row] pairs) that b also shows. Two blocks of one table overlap only through
    the rows they share, never merely because they come from the same table."""
    a = {tuple(r) for r in rows_a or []}
    if not a:
        return 0.0
    return len(a & {tuple(r) for r in rows_b or []}) / len(a)


def block_overlap(a, b):
    """Overlap of evidence block a with block b (dicts or RetrievedItems with meta spans/rows): row overlap for
    tables, span overlap for prose, 0 across documents."""
    if _get(a, "doc_id") != _get(b, "doc_id"):
        return 0.0
    ma, mb = _get(a, "meta") or a, _get(b, "meta") or b
    ra, rb = _get(ma, "rows"), _get(mb, "rows")
    if ra and rb:
        return row_overlap(ra, rb)
    sa = _get(ma, "spans") or ([_get(a, "span")] if _get(a, "span") else [])
    sb = _get(mb, "spans") or ([_get(b, "span")] if _get(b, "span") else [])
    if not sa or not sb:
        return 0.0
    tot = sum(max(1, s[1] - s[0]) for s in sa)
    cov = 0
    for s in sa:
        for t in sb:
            cov += max(0, min(s[1], t[1]) - max(s[0], t[0]))
    return min(1.0, cov / tot)


# --------------------------------------------------------------------------------------------------------------
# document structure
# --------------------------------------------------------------------------------------------------------------

class _DocView:
    """Per-document caches: section unit lists, prose runs, table groups."""

    def __init__(self, doc):
        self.doc = doc
        self._runs = {}

    def section_units(self, sid):
        try:
            s = self.doc.section(sid)
        except KeyError:
            return []
        return [self.doc.unit(u) for u in s.unit_ids if self.doc.has_unit(u)]

    def runs(self, sid):
        """[(run index, [units])] maximal runs of consecutive non-table units of a section."""
        if sid not in self._runs:
            out, cur = [], []
            for u in self.section_units(sid):
                if u.kind == "table":
                    if cur:
                        out.append(cur)
                    cur = []
                else:
                    cur.append(u)
            if cur:
                out.append(cur)
            self._runs[sid] = out
        return self._runs[sid]

    def run_of(self, u):
        for i, run in enumerate(self.runs(u.section_id)):
            if any(x.id == u.id for x in run):
                return i, run[0].span[0], run[-1].span[1]
        return None, u.span[0], u.span[1]


def table_key(unit_id):
    """'<doc>::<block>#table.<k>' -> '<doc>::<block>' (the table); other ids unchanged."""
    return unit_id.rsplit("#table.", 1)[0] if "#table." in unit_id else unit_id


def table_structure(doc, view, u):
    """-> {"caption": [(text, [a, b], unit_id)], "header": (text, [a, b], unit_id) | None,
           "rows": {unit_id: [(row index, text, [a, b])]}, "siblings": [units]} for the table of unit u."""
    key = table_key(u.id)
    sec_units = view.section_units(u.section_id)
    sibs = [x for x in sec_units if x.kind == "table" and table_key(x.id) == key] or [u]

    def lines(x):
        out, off = [], 0
        for ln in x.text.split("\n"):
            out.append((ln, [x.span[0] + off, x.span[0] + off + len(ln)]))
            off += len(ln) + 1
        return out

    caption = []
    first = sibs[0]
    idx = next((i for i, x in enumerate(sec_units) if x.id == first.id), None)
    if idx is not None and idx > 0 and sec_units[idx - 1].kind == "caption":
        cu = sec_units[idx - 1]
        caption.append((cu.text, list(cu.span), cu.id))
    per = {x.id: lines(x) for x in sibs}
    skip = {x.id: 0 for x in sibs}
    fl = per[first.id]
    if len(fl) > 1 and " | " not in fl[0][0] and (first.id.endswith("#table.0") or "#table." not in first.id):
        caption.append((fl[0][0], fl[0][1], first.id))
        skip[first.id] = 1
    # a column-heading line: the first " | " line, when every unit of the table repeats it (or the table is one unit)
    heads = {}
    for x in sibs:
        rest = per[x.id][skip[x.id]:]
        heads[x.id] = rest[0] if rest and " | " in rest[0][0] else None
    header = None
    texts = {h[0] for h in heads.values() if h}
    if all(heads.values()) and len(texts) == 1:
        body_lines = sum(len(per[x.id]) - skip[x.id] for x in sibs)
        if len(sibs) > 1 or body_lines >= 3:
            h = heads[first.id]
            header = (h[0], h[1], first.id)
            for x in sibs:
                skip[x.id] += 1
    rows = {x.id: [(j, ln, sp) for j, (ln, sp) in enumerate(per[x.id][skip[x.id]:])] for x in sibs}
    return {"caption": caption, "header": header, "rows": rows, "siblings": sibs, "key": key}


def _boundaries(text, lo, hi):
    """Sentence / line boundaries in [lo, hi] (positions where a block may start or end)."""
    b = {lo, hi}
    for m in _SENT_END.finditer(text, lo, hi):
        b.add(m.end())
    i = text.find("\n", lo, hi)
    while i != -1:
        b.add(i)
        b.add(i + 1)
        i = text.find("\n", i + 1, hi)
    return sorted(b)


def _strip_span(text, a, b):
    while a < b and text[a].isspace():
        a += 1
    while b > a and text[b - 1].isspace():
        b -= 1
    return a, b


def _expand(text, a, b, lo, hi, o, qtok):
    """Support [a, b) -> block span inside [lo, hi): sentence-aligned, prose_min..prose_max chars when possible."""
    mn, mx = int(o["prose_min"]), int(o["prose_max"])
    B = _boundaries(text, lo, hi)
    a2 = max([x for x in B if x <= a] or [lo])
    b2 = min([x for x in B if x >= b] or [hi])
    if b2 - a2 > mx:
        best = None
        for x in [p for p in B if a2 <= p < b2]:
            ys = [y for y in B if x < y <= min(x + mx, b2)]
            if not ys:
                continue
            y = ys[-1]
            sc = len(qtok & _tokens(text[x:y])) if qtok else 0
            key = (sc, y - x, -x)
            if best is None or key > best[0]:
                best = (key, x, y)
        if best is None:                                  # one sentence longer than prose_max: cut at a space
            y = text.rfind(" ", a2, a2 + mx)
            a2, b2 = a2, (y if y > a2 else a2 + mx)
        else:
            a2, b2 = best[1], best[2]
    else:
        turn = 0
        while b2 - a2 < mn:
            grew = False
            for side in ((0, 1) if turn == 0 else (1, 0)):
                if side == 0:
                    nxt = [y for y in B if y > b2 and y - a2 <= mx and text[b2:y].strip()]
                    if nxt:
                        b2 = nxt[0]
                        grew = True
                        break
                else:
                    prv = [x for x in B if x < a2 and b2 - x <= mx and text[x:a2].strip()]
                    if prv:
                        a2 = prv[-1]
                        grew = True
                        break
            turn ^= 1
            if not grew:
                break
    return _strip_span(text, a2, b2)


def _trim_prose(text, a, b, room):
    """Longest sentence-aligned prefix of [a, b) of at most `room` chars -> (a, b') or None."""
    if b - a <= room:
        return a, b
    B = [x for x in _boundaries(text, a, b) if a < x <= a + room]
    for x in reversed(B):
        a2, b2 = _strip_span(text, a, x)
        if b2 - a2 > 0:
            return a2, b2
    return None


# --------------------------------------------------------------------------------------------------------------
# the constructor
# --------------------------------------------------------------------------------------------------------------

def _candidates(candidates):
    out = []
    for pos, c in enumerate(candidates or []):
        meta = _get(c, "meta") or {}
        ranks = _get(c, "ranks") or meta.get("ranks") or {}
        rrf = _get(c, "rrf")
        if rrf is None:
            rrf = meta.get("rrf")
        if rrf is None:
            rrf = _get(c, "score")
        span = _get(c, "span")
        out.append({"pos": pos, "doc_id": _get(c, "doc_id"), "span": list(span) if span else None,
                    "unit_ids": list(_get(c, "unit_ids") or []), "ranks": dict(ranks),
                    "rrf": float(rrf) if rrf is not None else None})
    return out


def _merge_ranks(dst, src):
    for ch, r in (src or {}).items():
        if r is None:
            dst.setdefault(ch, None)
        elif dst.get(ch) is None or r < dst[ch]:
            dst[ch] = r


def build_blocks(candidates, corpus, budget=None, requirements=None, question=None, cfg=None, opts=None):
    """-> (items: list[RetrievedItem] in final order, report: dict). See the module docstring."""
    if cfg is None:
        from .config import load_config
        cfg = load_config()
    o = dict(DEFAULTS, **(opts or {}))
    budget = int(budget or cfg.evidence["max_chars"])
    reqs = [_norm_req(r, i) for i, r in enumerate(requirements or [])]
    qtok = _tokens(question or "")
    for r in reqs:
        for t in r["terms"]:
            qtok |= _tokens(t)
    report = {"version": EVIDENCE_VERSION, "budget": budget, "n_candidates": len(candidates or []), "unmapped": [],
              "warnings": []}
    views = {}

    def view(doc_id):
        if doc_id not in views:
            views[doc_id] = _DocView(corpus.doc(doc_id))
        return views[doc_id]

    # 1. support pieces ------------------------------------------------------------------------------------------
    cands = _candidates(candidates)
    prose, tables = {}, {}          # group key -> list of pieces
    n_pieces = 0
    for c in cands:
        try:
            v = view(c["doc_id"])
        except KeyError:
            report["unmapped"].append({"pos": c["pos"], "doc_id": c["doc_id"], "why": "unknown document"})
            continue
        doc = v.doc
        if c["span"]:
            a, b = c["span"]
            units = doc.units_overlapping(a, b)
        else:
            units = [doc.unit(u) for u in c["unit_ids"] if doc.has_unit(u)]
            a = b = None
        if not units:
            report["unmapped"].append({"pos": c["pos"], "doc_id": c["doc_id"], "why": "no corpus unit"})
            continue
        for u in units:
            ia, ib = (max(a, u.span[0]), min(b, u.span[1])) if a is not None else (u.span[0], u.span[1])
            if ib <= ia or not doc.text[ia:ib].strip():
                continue
            piece = {"doc_id": doc.id, "section_id": u.section_id, "unit": u, "a": ia, "b": ib, "pos": c["pos"],
                     "ranks": dict(c["ranks"]), "rrf": c["rrf"]}
            n_pieces += 1
            if u.kind == "table":
                tables.setdefault((doc.id, u.section_id, table_key(u.id)), []).append(piece)
            else:
                run_i, lo, hi = v.run_of(u)
                piece.update(lo=lo, hi=hi)
                prose.setdefault((doc.id, u.section_id, run_i), []).append(piece)
    report["n_supports"] = n_pieces

    # 2 + 3. merge support, build blocks -----------------------------------------------------------------------------
    blocks = []
    n_merged = 0
    for (doc_id, sid, run_i), pieces in prose.items():
        text = view(doc_id).doc.text
        pieces.sort(key=lambda p: (p["a"], p["b"]))
        groups = []
        for p in pieces:
            g = groups[-1] if groups else None
            if g and (p["a"] <= g["b"] or not text[g["b"]:p["a"]].strip()) and max(g["b"], p["b"]) - g["a"] <= int(o["prose_max"]):
                g["b"] = max(g["b"], p["b"])
                g["pos"] = min(g["pos"], p["pos"])
                g["cands"].add(p["pos"])
                _merge_ranks(g["ranks"], p["ranks"])
                if p["rrf"] is not None and (g["rrf"] is None or p["rrf"] > g["rrf"]):
                    g["rrf"] = p["rrf"]
                n_merged += 1
            else:
                groups.append({"a": p["a"], "b": p["b"], "pos": p["pos"], "cands": {p["pos"]}, "ranks": dict(p["ranks"]),
                               "rrf": p["rrf"], "lo": p["lo"], "hi": p["hi"]})
        for g in groups:
            x, y = _expand(text, g["a"], g["b"], g["lo"], g["hi"], o, qtok)
            if y <= x:
                continue
            blocks.append({"kind": "prose", "doc_id": doc_id, "section_id": sid, "a": x, "b": y, "lo": g["lo"],
                           "hi": g["hi"], "run": run_i, "pos": g["pos"], "cands": set(g["cands"]),
                           "ranks": dict(g["ranks"]), "rrf": g["rrf"], "sub": 0})
    for (doc_id, sid, key), pieces in tables.items():
        v = view(doc_id)
        doc = v.doc
        ts = table_structure(doc, v, pieces[0]["unit"])
        sup = {}                                       # (unit_id, row) -> best piece
        for p in pieces:
            for j, ln, sp in ts["rows"].get(p["unit"].id, []):
                if sp[1] > p["a"] and sp[0] < p["b"]:
                    k = (p["unit"].id, j)
                    if k not in sup or p["pos"] < sup[k]["pos"]:
                        sup[k] = {"pos": p["pos"], "text": ln, "span": sp}
        if not sup:                                    # the support touched only the caption / heading lines
            for p in pieces:
                for j, ln, sp in ts["rows"].get(p["unit"].id, []):
                    sup.setdefault((p["unit"].id, j), {"pos": p["pos"], "text": ln, "span": sp})
        if not sup:
            continue
        ranks, rrf, cset = {}, None, set()
        for p in pieces:
            _merge_ranks(ranks, p["ranks"])
            cset.add(p["pos"])
            if p["rrf"] is not None and (rrf is None or p["rrf"] > rrf):
                rrf = p["rrf"]
        n_merged += max(0, len(pieces) - 1)
        for k, r in sup.items():
            r["lex"] = len(qtok & _tokens(r["text"])) if qtok else 0
        matched = [k for k in sup if sup[k]["lex"] > 0]
        relevant = matched or list(sup)
        order_doc = {x.id: i for i, x in enumerate(ts["siblings"])}
        fixed = [(t, sp, uid) for t, sp, uid in ts["caption"]] + ([ts["header"]] if ts["header"] else [])
        fixed_len = sum(len(t) + 1 for t, _sp, _u in fixed)
        if fixed_len > int(o["table_max"]) // 2:       # an outsized caption: keep the heading line only
            fixed = [ts["header"]] if ts["header"] else []
            fixed_len = sum(len(t) + 1 for t, _sp, _u in fixed)
        remaining = sorted(relevant, key=lambda k: (-sup[k]["lex"], sup[k]["pos"], order_doc[k[0]], k[1]))
        sub = 0
        while remaining:
            take, used = [], fixed_len
            for k in remaining:
                ln = len(sup[k]["text"]) + 1
                if used + ln - 1 <= int(o["table_max"]) or not take:     # an outsized single row still goes alone
                    take.append(k)
                    used += ln
            remaining = [k for k in remaining if k not in take]
            take.sort(key=lambda k: (order_doc[k[0]], k[1]))
            lines = [(t, sp, uid) for t, sp, uid in fixed] + [(sup[k]["text"], sup[k]["span"], k[0]) for k in take]
            blocks.append({"kind": "table", "doc_id": doc_id, "section_id": sid, "lines": lines,
                           "rows": [[k[0], k[1]] for k in take], "table": key, "fixed": len(fixed),
                           "pos": min(sup[k]["pos"] for k in take), "cands": set(cset), "ranks": dict(ranks),
                           "rrf": rrf, "sub": sub,
                           "caption_unit": next((uid for t, sp, uid in ts["caption"] if not uid.startswith(key)), None)})
            sub += 1
    report["merged_supports"] = n_merged

    # block text / identity -------------------------------------------------------------------------------------------
    def render(b):
        doc = view(b["doc_id"]).doc
        if b["kind"] == "prose":
            b["text"] = doc.text[b["a"]:b["b"]]
            b["span"] = [b["a"], b["b"]]
            b["spans"] = [[b["a"], b["b"]]]
            b["unit_ids"] = [u.id for u in doc.units_overlapping(b["a"], b["b"])]
        else:
            b["text"] = "\n".join(t for t, _sp, _u in b["lines"]).strip()
            b["spans"] = [list(sp) for _t, sp, _u in b["lines"]]
            s0, s1 = b["spans"][0][0], b["spans"][-1][1]
            b["span"] = [s0, s1] if doc.text[s0:s1] == b["text"] else None
            uids = []
            for _t, _sp, u in b["lines"]:
                if u not in uids:
                    uids.append(u)
            b["unit_ids"] = uids
        try:
            path = " ".join([doc.title] + [t for t in doc.section_path(b["section_id"]) if t])
        except KeyError:
            path = doc.title
        b["ctx"] = fold(path + "\n" + b["text"])
        b["lex"] = len(qtok & _tokens(b["text"])) if qtok else 0
        b["item"] = RetrievedItem(text=b["text"], doc_id=b["doc_id"], section_id=b["section_id"], unit_ids=list(b["unit_ids"]),
                                  kind="block", score=round(b["rrf"], 6) if b["rrf"] is not None else None, span=b["span"])
        b["hlen"] = len(item_header(1, b["item"], corpus)) + 1
        return b

    for b in blocks:
        render(b)
    # weak: a piece of a multi-section candidate that shares no term with the question while its siblings do
    by_cand = {}
    for b in blocks:
        by_cand.setdefault(b["pos"], []).append(b)
    for sibs in by_cand.values():
        if len(sibs) > 1 and qtok and any(x["lex"] for x in sibs):
            for x in sibs:
                x["weak"] = x["lex"] == 0
    for b in blocks:
        b.setdefault("weak", False)
    blocks.sort(key=lambda b: (b["weak"], b["pos"], -b["lex"], b["sub"], b["doc_id"], b["spans"][0][0]))

    # redundancy between prose blocks of one section run: merge when the union fits, else clip, else drop
    kept, dropped = [], []
    for b in blocks:
        if b["kind"] != "prose":
            kept.append(b)
            continue
        text = view(b["doc_id"]).doc.text
        red = False
        for k in kept:
            if k["kind"] != "prose" or k["doc_id"] != b["doc_id"] or k["section_id"] != b["section_id"] or k.get("run") != b["run"]:
                continue
            if b["b"] <= k["a"] or b["a"] >= k["b"]:
                continue
            ua, ub = min(k["a"], b["a"]), max(k["b"], b["b"])
            if ub - ua <= int(o["prose_max"]):
                k["a"], k["b"] = ua, ub
                k["cands"] |= b["cands"]
                _merge_ranks(k["ranks"], b["ranks"])
                render(k)
                red = True
                break
            left = (b["a"], k["a"]) if b["a"] < k["a"] else None
            right = (k["b"], b["b"]) if b["b"] > k["b"] else None
            part = max([p for p in (left, right) if p], key=lambda p: p[1] - p[0], default=None)
            if part is None:
                red = True
                break
            x, y = _strip_span(text, *part)
            B = _boundaries(text, b["lo"], b["hi"])
            x = min([p for p in B if p >= x] or [y])
            y = max([p for p in B if p <= y] or [x])
            x, y = _strip_span(text, x, y)
            if y - x < int(o["min_piece"]):
                red = True
                break
            b["a"], b["b"] = x, y
            render(b)
        if red:
            dropped.append((b, "redundant"))
        else:
            kept.append(b)
    blocks = kept

    # 4. selection ------------------------------------------------------------------------------------------------------
    def covers(b, r):
        fields = []
        for f in ("terms", "entity", "period"):
            if r[f]:
                fields.append((f, any(fold(x) and fold(x) in b["ctx"] for x in r[f])))
        return fields

    for b in blocks:
        b["covers"] = {}
        for r in reqs:
            fs = covers(b, r)
            b["covers"][r["id"]] = {"full": bool(fs) and all(ok for _f, ok in fs), "n": sum(1 for _f, ok in fs if ok),
                                    "fields": [f for f, ok in fs if ok]}

    chosen, used = [], 0
    trimmed = set()

    def cost(b):
        return b["hlen"] + len(b["text"]) + (2 if chosen else 0)

    def trim(b, room_text, req=None):
        if room_text < int(o["min_trim"]):
            return False
        if b["kind"] == "prose":
            t = _trim_prose(view(b["doc_id"]).doc.text, b["a"], b["b"], room_text)
            if not t or t[1] - t[0] < int(o["min_trim"]):
                return False
            nb = dict(b, a=t[0], b=t[1])
        else:
            nb = dict(b)
            rows = list(b["rows"])
            lines = list(b["lines"])
            while len(lines) > b["fixed"] + 1 and len("\n".join(x[0] for x in lines)) > room_text:
                # drop the least relevant row (fewest matching terms, latest)
                body = lines[b["fixed"]:]
                worst = min(range(len(body)), key=lambda i: (len(qtok & _tokens(body[i][0])) if qtok else 0, -i))
                del lines[b["fixed"] + worst]
                del rows[worst]
            if len("\n".join(x[0] for x in lines)) > room_text:
                return False
            nb["lines"], nb["rows"] = lines, rows
        render(nb)
        nb["covers"] = b["covers"]
        if req is not None:
            fs = covers(nb, req)
            nb["covers"] = dict(b["covers"])
            nb["covers"][req["id"]] = {"full": bool(fs) and all(ok for _f, ok in fs), "n": sum(1 for _f, ok in fs if ok),
                                       "fields": [f for f, ok in fs if ok]}
            if nb["covers"][req["id"]]["n"] < b["covers"][req["id"]]["n"]:
                return False
        b.clear()
        b.update(nb)
        trimmed.add(id(b))
        return True

    def try_add(b, req=None):
        nonlocal used
        c = cost(b)
        if used + c <= budget:
            chosen.append(b)
            used += c
            return True
        room_text = budget - used - (c - len(b["text"]))
        if trim(b, room_text, req):
            chosen.append(b)
            used = sum(x["hlen"] + len(x["text"]) for x in chosen) + 2 * (len(chosen) - 1)
            return True
        return False

    cap_hard = int(o["max_blocks_req"]) if reqs else int(o["max_blocks"])
    rstat = {}
    for r in reqs:
        done = next((x for x in chosen if x["covers"][r["id"]]["full"]), None)
        if done is not None:
            rstat[r["id"]] = {"status": "covered", "shared": True}
            done.setdefault("reserved_for", []).append(r["id"])
            continue
        if len(chosen) >= cap_hard:
            rstat[r["id"]] = {"status": "missing", "why": "block cap"}
            continue
        full = [b for b in blocks if b not in chosen and b["covers"][r["id"]]["full"]]
        ok = None
        for b in full:
            if try_add(b, r):
                ok = b
                break
        if ok is not None:
            ok.setdefault("reserved_for", []).append(r["id"])
            rstat[r["id"]] = {"status": "covered"}
            continue
        part = sorted([b for b in blocks if b not in chosen and b["covers"][r["id"]]["n"] > 0],
                      key=lambda b: -b["covers"][r["id"]]["n"])
        for b in part:
            if try_add(b, r):
                ok = b
                break
        if ok is not None:
            ok.setdefault("reserved_for", []).append(r["id"])
            rstat[r["id"]] = {"status": "partial", "fields": ok["covers"][r["id"]]["fields"]}
        else:
            rstat[r["id"]] = {"status": "missing", "why": "no block matches" if not full and not part else "budget"}
    n_res = len(chosen)
    cap = max(int(o["max_blocks"]), min(int(o["max_blocks_req"]), n_res)) if reqs else int(o["max_blocks"])
    for b in blocks:
        if b in chosen:
            continue
        if len(chosen) >= cap:
            dropped.append((b, "cap"))
            continue
        if not try_add(b):
            dropped.append((b, "budget"))

    # 5. order ------------------------------------------------------------------------------------------------------------
    mode = o["order"]
    qyears = []
    for m in _YEAR.finditer(question or ""):
        if m.group(1) not in qyears:
            qyears.append(m.group(1))
    if mode == "auto":
        ents = {fold(r["entity"][0]) for r in reqs if r["entity"]}
        pers = {fold(r["period"][0]) for r in reqs if r["period"]}
        if len(ents) > 1:
            mode = "entity"
        elif len(pers) > 1 or (not reqs and len(qyears) > 1):
            mode = "period"
        else:
            mode = "relevance"
    prio = {id(b): i for i, b in enumerate(chosen)}
    rel_key = lambda b: (b["weak"], b["pos"], -b["lex"], b["sub"])  # noqa: E731

    def period_of(b):
        for rid in b.get("reserved_for") or []:
            r = next(x for x in reqs if x["id"] == rid)
            if r["period"]:
                return _period_num(r["period"][0])
        for r in reqs:
            if r["period"] and b["covers"][r["id"]]["full"]:
                return _period_num(r["period"][0])
        for y in qyears or []:
            if y in b["ctx"]:
                return int(y)
        m = _YEAR.search(b["ctx"])
        return int(m.group(1)) if (m and not reqs and not qyears) else None

    if mode == "entity":
        ent_order = []
        for r in reqs:
            e = fold(r["entity"][0]) if r["entity"] else None
            if e and e not in ent_order:
                ent_order.append(e)

        def ent_of(b):
            for rid in b.get("reserved_for") or []:
                r = next(x for x in reqs if x["id"] == rid)
                if r["entity"]:
                    return ent_order.index(fold(r["entity"][0]))
            for i, e in enumerate(ent_order):
                if e in b["ctx"]:
                    return i
            return len(ent_order)
        chosen.sort(key=lambda b: (ent_of(b), period_of(b) if period_of(b) is not None else 10 ** 9, rel_key(b)))
    elif mode == "period":
        chosen.sort(key=lambda b: (period_of(b) if period_of(b) is not None else 10 ** 9, rel_key(b)))
    else:
        chosen.sort(key=rel_key)

    # exact rendering check ------------------------------------------------------------------------------------------
    items = []
    for n, b in enumerate(chosen, 1):
        it = b["item"]
        it.rank = n - 1
        it.meta = {"block": b["kind"], "spans": b["spans"], "rows": b.get("rows"), "ranks": b["ranks"],
                   "rrf": round(b["rrf"], 6) if b["rrf"] is not None else None, "candidates": sorted(b["cands"]),
                   "requirements": [rid for rid, cv in b["covers"].items() if cv["full"]],
                   "reserved_for": b.get("reserved_for") or [], "trimmed": id(b) in trimmed,
                   "caption_unit": b.get("caption_unit"), "table": b.get("table")}
        items.append(it)
    pk = pack_evidence(items, corpus, budget=budget, cfg=cfg)
    while items and (pk.n_included < len(items) or pk.last_item_cut):      # never expected; keep the contract anyway
        report["warnings"].append("rendered evidence exceeded the budget; dropped the last block")
        b = chosen.pop()
        items.pop()
        dropped.append((b, "budget"))
        pk = pack_evidence(items, corpus, budget=budget, cfg=cfg)

    def brief(b):
        return {"doc_id": b["doc_id"], "section_id": b["section_id"], "kind": b["kind"], "spans": b["spans"],
                "rows": b.get("rows"), "chars": len(b["text"]), "ranks": b["ranks"],
                "rrf": round(b["rrf"], 6) if b["rrf"] is not None else None, "candidates": sorted(b["cands"])}

    report.update({
        "mode": mode, "rendered_chars": pk.chars, "n_blocks_built": len(blocks) + sum(1 for _b, w in dropped if w == "redundant"),
        "n_blocks": len(items), "cap": cap,
        "blocks": [dict(brief(b), n=i + 1, rendered_chars=b["hlen"] + len(b["text"]),
                        requirements=items[i].meta["requirements"], reserved_for=items[i].meta["reserved_for"],
                        trimmed=items[i].meta["trimmed"]) for i, b in enumerate(chosen)],
        "dropped": [dict(brief(b), reason=why) for b, why in dropped][:30],
        "n_dropped": {w: sum(1 for _b, x in dropped if x == w) for w in ("budget", "cap", "redundant")},
        "requirements": {r["id"]: dict(rstat.get(r["id"], {"status": "missing"}),
                                       block=next((i + 1 for i, b in enumerate(chosen) if r["id"] in (b.get("reserved_for") or [])), None))
                         for r in reqs},
    })
    return items, report
