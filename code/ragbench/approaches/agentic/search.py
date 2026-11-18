"""Stage 2 (and the widened search of stage 4) - no model calls.

ripgrep finds the lines of the target documents that carry a metric term (the plan's literal terms turned
into morphology-aware patterns, see planner.term_patterns) or one of their content words. Each hit becomes
a WINDOW of neighbouring units of its section, not a single line: report extraction often puts a number
on its own line next to (before or after) its label ("تعداد کاربران راننده" / "6,152,509 نفر"), or
scatters infographic labels and values ("راننده های فعال" / "رانندگان مرد" / "1,896,210" / "3,661,662").

    unit hit, short unit     the unit plus neighbours on both sides, widening (up to window_neighbours per side)
                             while the window has no candidate number, within window_chars
    unit hit, long unit      a clip of clip_chars around the term (at most max_windows_per_unit clips)
    table unit               caption (if short) + header row + matching row(s) + table_rows_after rows
    @@TITLE hit              the section's first units (lead_units, window_chars)

Windows are scored (term phrases, content words, a candidate number, term-number character distance, the
expected kind of number, title match, the document's year mentioned, table), deduplicated (a window that
adds no new number-bearing or term-bearing unit to the kept ones is dropped), and selected round-robin over
the target documents in year order with a per-document cap and a global cap, so every period gets its best
windows verified before any period gets its fifth.
"""
import re
import subprocess
import time

from ...contracts import RetrievedItem
from ...normalize import fold, normalize
from ...store import grep
from .numbers import candidate_numbers, find_numbers, unit_kind_match
from .planner import WEAK_WORDS, component_words, question_terms, score_words, term_patterns, word_regex


class Candidate:
    def __init__(self, doc_id, section_id, unit_ids, text, span, kind, hit_unit_ids, round_no=1):
        self.doc_id = doc_id
        self.section_id = section_id
        self.unit_ids = list(unit_ids)
        self.text = text
        self.span = span
        self.kind = kind
        self.hit_unit_ids = list(hit_unit_ids)
        self.round = round_no
        self.score = 0.0
        self.features = {}
        self.cid = None
        self.number_unit_ids = []

    def key(self):
        return (self.doc_id, tuple(self.unit_ids), tuple(self.span) if self.span else None, self.text[:80])

    def to_item(self, rank=None, meta=None):
        return RetrievedItem(text=self.text, doc_id=self.doc_id, section_id=self.section_id,
                             unit_ids=list(self.unit_ids), kind="window", score=round(self.score, 3), rank=rank,
                             span=list(self.span) if self.span else None,
                             meta=dict({"cid": self.cid, "window_kind": self.kind, "round": self.round}, **(meta or {})))

    def trace(self, chars=400):
        return {"cid": self.cid, "doc_id": self.doc_id, "section_id": self.section_id, "unit_ids": self.unit_ids,
                "kind": self.kind, "round": self.round, "score": round(self.score, 3), "features": self.features,
                "chars": len(self.text), "text": self.text[:chars]}


# --------------------------------------------------------------------------
# search context: patterns and scorers built from the plan
# --------------------------------------------------------------------------

_W = r"(?<![^\s‌(|:«])%s(?![^\s‌)|:»])"
MEASURE_CUES = {     # words that label a value of the expected kind when they precede a term
    "count": re.compile("|".join([_W % "تعداد", _W % "شمار", _W % "کل", _W % "مجموع", r"number of", r"\btotal\b"]), re.I),
    "percent": re.compile("|".join([_W % "درصد", _W % "سهم", _W % "نرخ", r"\bshare\b", r"\brate\b", r"percent"]), re.I),
    "currency": re.compile("|".join([_W % "ارزش", _W % "درآمد", _W % "مبلغ", _W % "فروش", r"\bvalue\b", r"\brevenue\b",
                                     r"\bamount\b"]), re.I),
    "other": re.compile("|".join([_W % "تعداد", _W % "میزان", _W % "مقدار", _W % "کل", _W % "مجموع", r"\btotal\b",
                                  r"\bamount\b", r"number of"]), re.I),
}


class SearchContext:
    def __init__(self, plan, d_lang, question="", widened=False):
        self.plan = plan
        self.d_lang = d_lang
        self.unit_kind = plan.get("unit") or "count"
        terms = plan.get("terms") or {"fa": [], "en": []}
        langs = [d_lang] if terms.get(d_lang) else [l for l in ("fa", "en") if terms.get(l)]
        self.search_langs = langs
        self.terms_used = {l: list(terms.get(l) or []) for l in langs}
        ents = plan.get("entity_terms") or []
        if widened:
            extra = [w for w in question_terms(question, ents) if _script(w) in langs]
            for w in extra:
                lang = _script(w)
                if fold(w) not in {fold(t) for t in self.terms_used[lang]} and fold(w) not in {fold(x) for x in WEAK_WORDS}:
                    self.terms_used[lang].append(w)
        self.term_rx = []          # [(concept index, compiled)]: terms whose patterns coincide share a concept
        self.patterns = []         # ripgrep patterns (deduplicated)
        self.pattern_info = []
        concept_of = {}
        n_concepts = 0
        for lang in langs:
            for t in self.terms_used[lang]:
                pats = term_patterns(t, lang)
                ti = next((concept_of[p] for p in pats if p in concept_of), None)
                if ti is None:
                    ti = n_concepts
                    n_concepts += 1
                for p in pats:
                    if p in concept_of:
                        continue
                    concept_of[p] = ti
                    self.patterns.append(p)
                    self.pattern_info.append({"pattern": p, "term": t, "kind": "term", "concept": ti})
                    self.term_rx.append((ti, re.compile(p, re.I)))
        self.words = score_words(self.terms_used, ents)
        # widened (gap) round: every non-weak content word of the terms is also searched on its own
        if widened:
            for lang, w, weak in component_words(self.terms_used, ents):
                p = word_regex(w, lang)
                if weak or p in concept_of or p in self.patterns:
                    continue
                self.patterns.append(p)
                self.pattern_info.append({"pattern": p, "term": w, "kind": "word"})
        self.weak_rx = [rx for _w, rx, weak in self.words if weak] + [MEASURE_CUES.get(self.unit_kind, MEASURE_CUES["other"])]
        self.entity_rx = [re.compile(re.escape(normalize(e)), re.I) for e in ents if len(e) >= 3]

    def term_positions(self, text):
        out = []
        for ti, rx in self.term_rx:
            for m in rx.finditer(text):
                out.append((ti, m.start(), m.end()))
        for w, rx, weak in self.words:
            if weak:
                continue
            for m in rx.finditer(text):
                out.append((None, m.start(), m.end()))
        return out

    def has_term(self, text):
        return any(rx.search(text) for _ti, rx in self.term_rx) or any(
            rx.search(text) for _w, rx, weak in self.words if not weak)


def _script(w):
    from ...normalize import script_counts
    fa, en = script_counts(w)
    return "fa" if fa >= en else "en"


# --------------------------------------------------------------------------
# windows
# --------------------------------------------------------------------------

def _has_candidate_number(text, unit_kind):
    return bool(candidate_numbers(find_numbers(text), unit_kind))


def _verbatim(doc, start, text):
    return [start, start + len(text)] if doc.text[start:start + len(text)] == text else None


def unit_window(doc, sec_unit_ids, idx, *, unit_kind, neighbours, max_chars, line_only=False, neighbour_chars=250):
    """Consecutive units around sec_unit_ids[idx]: +-1 short neighbour always, then widening while no
    candidate number is inside, within max_chars. Only short neighbours (labels, KPI lines, captions: at most
    neighbour_chars) are added; a long paragraph next to the hit is its own window. -> (unit ids, text, span)."""
    units = [doc.unit(u) for u in sec_unit_ids]
    lo = hi = idx
    if not line_only:
        def length(a, b):
            return units[b].span[1] - units[a].span[0]

        def short(i):
            return units[i].span[1] - units[i].span[0] <= neighbour_chars
        for step in range(1, neighbours + 1):
            text_now = doc.text[units[lo].span[0]:units[hi].span[1]]
            if step > 1 and _has_candidate_number(text_now, unit_kind):
                break
            grew = False
            if hi + 1 < len(units) and short(hi + 1) and length(lo, hi + 1) <= max_chars:
                hi += 1
                grew = True
            if lo - 1 >= 0 and short(lo - 1) and length(lo - 1, hi) <= max_chars:
                lo -= 1
                grew = True
            if not grew:
                break
    text = doc.text[units[lo].span[0]:units[hi].span[1]]
    return [u.id for u in units[lo:hi + 1]], text, [units[lo].span[0], units[hi].span[1]]


def clip_windows(doc, unit, positions, clip_chars, max_windows):
    """Clips of a long unit around term positions -> [(text, span)] (verbatim)."""
    t = unit.text
    out = []
    covered = []
    for p in sorted(positions):
        if any(a <= p < b for a, b in covered):
            continue
        a = max(0, p - clip_chars // 2)
        b = min(len(t), a + clip_chars)
        a = max(0, b - clip_chars)
        # snap to whitespace so words are not cut
        if a > 0:
            sp = t.find(" ", a)
            a = sp + 1 if 0 <= sp < p else a
        if b < len(t):
            sp = t.rfind(" ", p, b)
            b = sp if sp > p else b
        covered.append((a, b))
        piece = t[a:b]
        out.append((piece, [unit.span[0] + a, unit.span[0] + b]))
        if len(out) >= max_windows:
            break
    return out


def table_window(doc, sec_unit_ids, idx, ctx, rows_after, max_chars):
    unit = doc.unit(sec_unit_ids[idx])
    lines = unit.text.split("\n")
    hit_rows = [i for i, ln in enumerate(lines) if ctx.has_term(ln)]
    rows = {0}
    for i in hit_rows or [0]:
        for j in range(i, min(len(lines), i + rows_after + 1)):
            rows.add(j)
    rows = sorted(rows)
    body = "\n".join(lines[i] for i in rows)
    if len(body) > max_chars:
        keep = [0]
        for i in rows[1:]:
            if len("\n".join(lines[k] for k in keep + [i])) > max_chars:
                break
            keep.append(i)
        rows = keep
        body = "\n".join(lines[i] for i in rows)
    uids = [unit.id]
    start = unit.span[0]
    text = body
    if idx > 0:
        prev = doc.unit(sec_unit_ids[idx - 1])
        if prev.kind in ("caption", "heading") and len(prev.text) <= 250 and len(prev.text) + len(body) + 2 <= max_chars:
            text = prev.text + "\n\n" + body
            uids = [prev.id, unit.id]
            start = prev.span[0]
    return uids, text, _verbatim(doc, start, text)


def build_windows(doc, section, hit_units, title_hit, ctx, opts, round_no, line_only=False):
    """Windows for one section's hits -> [Candidate]."""
    out = []
    sec_uids = list(section.unit_ids)
    if not sec_uids:
        return out
    widened = round_no > 1
    max_chars = int(opts["gap_window_chars"] if widened else opts["window_chars"])
    neighbours = int(opts["gap_window_neighbours"] if widened else opts["window_neighbours"])
    if title_hit:
        units = [doc.unit(u) for u in sec_uids[: int(opts["lead_units"]) * (2 if widened else 1)]]
        hi = 0
        while hi + 1 < len(units) and units[hi + 1].span[1] - units[0].span[0] <= max_chars:
            hi += 1
        if units[0].span[1] - units[0].span[0] <= max_chars:
            text = doc.text[units[0].span[0]:units[hi].span[1]]
            out.append(Candidate(doc.id, section.id, [u.id for u in units[:hi + 1]], text,
                                 [units[0].span[0], units[hi].span[1]], "lead", [], round_no))
    for uid in hit_units:
        if uid not in sec_uids:
            continue
        idx = sec_uids.index(uid)
        unit = doc.unit(uid)
        if unit.kind == "table" or (unit.text.count("\n") >= 2 and unit.text.count(" | ") >= 2):
            uids, text, span = table_window(doc, sec_uids, idx, ctx, int(opts["table_rows_after"]) + (2 if widened else 0),
                                            max_chars)
            out.append(Candidate(doc.id, section.id, uids, text, span, "table", [uid], round_no))
            continue
        if len(unit.text) > max_chars:
            pos = [p for _ti, p, _e in ctx.term_positions(unit.text)]
            for text, span in clip_windows(doc, unit, pos, int(opts["clip_chars"]) * (2 if widened else 1),
                                           int(opts["max_windows_per_unit"])):
                out.append(Candidate(doc.id, section.id, [uid], text, span, "clip", [uid], round_no))
            continue
        uids, text, span = unit_window(doc, sec_uids, idx, unit_kind=ctx.unit_kind, neighbours=neighbours,
                                       max_chars=max_chars, line_only=line_only,
                                       neighbour_chars=int(opts["neighbour_chars"]) * (2 if widened else 1))
        out.append(Candidate(doc.id, section.id, uids, text, span, "window", [uid], round_no))
    return out


# --------------------------------------------------------------------------
# scoring, dedupe, selection
# --------------------------------------------------------------------------

def _gap(tok, a, b):
    return max(0, (tok.start - b) if tok.start >= b else (a - tok.end) if a >= tok.end else 0)


_COUNT_UNITS = re.compile(r"^(?:نفر|کاربر|حساب|عضو|people|persons?|users?|members?|accounts?|employees|staff)", re.I)
_CONNECTORS = set("""و در از به تا را که با برای یا هم نیز and in of to for from or the a by as at on""".split())
_WORD_AFTER = re.compile(r"^[\s+*]*([^\W\d_][^\W\d_‌]*(?:‌[^\W\d_]+)*)", re.UNICODE)


def _unit_word(text, tok):
    m = _WORD_AFTER.match(text[tok.end:tok.end + 40])
    return m.group(1) if m else ""


def score_candidate(c, doc, section, meta, ctx):
    """Lexical evidence that the window states the metric (no model):
      + 2.5 when a term matches, + 1 when a multi-word term matches, + 0.5 per content word (max 2)
      + 2 when a candidate number is present
      + proximity of the nearest term-number pair: 2.5 (<= 25 chars) | 1.5 (<= 60) | 0.75 (<= 150)
      + 1.5 when a measure word (تعداد, کل, total ...) directly precedes a term ("تعداد کاربران راننده"), not
        another noun ("تعداد سفرهای کاربران راننده" counts trips)
      + 1 when the word after that number echoes a term or is a people/account unit (نفر, کاربر); - 1 when it is
        another noun (سفر, تومان, سیم‌کارت)
      + 1 when the number has the expected kind, - 1.5 when it clearly has another (currency, percent)
      + 0.5 each: term in the section title/path, entity name, the document's year, a table
      - 1 for more than 8 numbers, - length/800"""
    text = c.text
    fea = {}
    tpos = [(ti, a, b) for ti, a, b in ctx.term_positions(text) if ti is not None]
    wpos = [(a, b) for _ti, a, b in ctx.term_positions(text) if _ti is None]
    phrase = any(" " in text[a:b] or "‌" in text[a:b] for _ti, a, b in tpos)
    words = {w for w, rx, weak in ctx.words if not weak and rx.search(text)}
    toks = find_numbers(text)
    nums = candidate_numbers(toks, ctx.unit_kind)
    s = (2.5 if tpos else 0.0) + (1.0 if phrase else 0.0) + 0.5 * min(len(words), 2)
    fea.update(terms=len({ti for ti, _a, _b in tpos}), phrase=int(phrase), words=len(words), numbers=len(nums))
    if nums:
        s += 2.0
    anchors = tpos or [(None, a, b) for a, b in wpos]
    best = None
    for t in nums:
        wrong_kind = ctx.unit_kind not in ("text", None) and not unit_kind_match(text, t, ctx.unit_kind)
        for _ti, a, b in anchors:
            d = _gap(t, a, b)
            key = d + (40 if wrong_kind else 0)        # prefer a number of the expected kind nearby
            if best is None or key < best[0]:
                best = (key, d, t)
    if best is not None:
        _key, dist, tok = best
        fea["term_number_chars"] = dist
        s += 2.5 if dist <= 25 else 1.5 if dist <= 60 else 0.75 if dist <= 150 else 0.0
        if ctx.unit_kind not in ("text", None):
            if unit_kind_match(text, tok, ctx.unit_kind):
                s += 1.0
                fea["unit_kind_match"] = 1
            elif (ctx.unit_kind != "percent" and tok.percent) or (
                    ctx.unit_kind != "currency" and unit_kind_match(text, tok, "currency")):
                s -= 1.5
                fea["unit_kind_mismatch"] = 1
            uw = _unit_word(text, tok)
            if uw:
                if ctx.has_term(uw) or (ctx.unit_kind == "count" and _COUNT_UNITS.match(uw)):
                    s += 1.0
                    fea["unit_word"] = "echo"
                elif uw.lower() not in _CONNECTORS and len(uw) >= 2 and not fea.get("unit_kind_mismatch"):
                    s -= 1.0
                    fea["unit_word"] = "other"
    for _ti, a, _b in tpos:
        before = text[max(0, a - 40):a]
        m = None
        for m in ctx.weak_rx[-1].finditer(before):
            pass
        if m is None:
            continue
        between = before[m.end():]
        if not re.sub(r"[\s‌]+", "", ctx.weak_rx[-1].sub("", between)):
            s += 1.5
            fea["measure_word"] = 1
            break
    path = " ".join(doc.section_path(section.id))
    if ctx.has_term(path):
        s += 0.5
        fea["title_term"] = 1
    if any(rx.search(text) for rx in ctx.entity_rx):
        s += 0.5
        fea["entity"] = 1
    years = meta.get("years")
    if years:
        ys = {str(y) for y in range(years[0], years[1] + 1)}
        if any(y in text for y in ys) or any(re.search(r"سال\s?%s(?![0-9])" % y[-2:], text) for y in ys):
            s += 0.5
            fea["doc_year"] = 1
    if c.kind == "table":
        s += 0.5
    if len(nums) > 10:
        s -= 1.0
    s -= len(text) / 800.0
    c.score = round(s, 3)
    c.features = fea
    # units holding a candidate number (for dedupe and the gap round's exclusion)
    num_units = []
    if c.span and len(c.unit_ids) > 0:
        for uid in c.unit_ids:
            u = doc.unit(uid)
            if _has_candidate_number(u.text, ctx.unit_kind) and uid not in num_units:
                num_units.append(uid)
    else:
        num_units = list(c.unit_ids) if nums else []
    c.number_unit_ids = num_units
    return c


def merge_adjacent(cands, corpus, metas, ctx, max_chars):
    """Windows of consecutive units in the same section that overlap or touch are merged into one window
    (re-scored) while the merged text fits max_chars: a KPI block with several labelled values becomes one
    snippet, verified once, instead of competing overlapping windows."""
    out, groups = [], {}
    for c in cands:
        if c.kind in ("window", "lead") and c.span:
            groups.setdefault((c.doc_id, c.section_id), []).append(c)
        else:
            out.append(c)
    for (doc_id, sec_id), cs in groups.items():
        doc = corpus.doc(doc_id)
        section = doc.section(sec_id)
        pos = {u: i for i, u in enumerate(section.unit_ids)}
        cs.sort(key=lambda c: (pos.get(c.unit_ids[0], 0), pos.get(c.unit_ids[-1], 0)))
        cur = None
        for c in cs:
            if cur is None:
                cur = c
                continue
            a0, a1 = pos.get(cur.unit_ids[0], 0), pos.get(cur.unit_ids[-1], 0)
            b0, b1 = pos.get(c.unit_ids[0], 0), pos.get(c.unit_ids[-1], 0)
            if b0 <= a1 + 1:
                lo, hi = min(a0, b0), max(a1, b1)
                uids = section.unit_ids[lo:hi + 1]
                start, end = doc.unit(uids[0]).span[0], doc.unit(uids[-1]).span[1]
                if end - start <= max_chars:
                    m = Candidate(doc_id, sec_id, uids, doc.text[start:end], [start, end], "window",
                                  list(dict.fromkeys(cur.hit_unit_ids + c.hit_unit_ids)), c.round)
                    score_candidate(m, doc, section, metas.get(doc_id, {}), ctx)
                    m.features["merged"] = int(cur.features.get("merged", 1)) + 1
                    cur = m
                    continue
            out.append(cur)
            cur = c
        if cur is not None:
            out.append(cur)
    return out


def dedupe(cands):
    """Best first; drop a window whose number-bearing and hit units are all inside windows already kept in
    the same section (it adds nothing to verify)."""
    kept, seen_keys = [], set()
    covered = {}
    for c in sorted(cands, key=lambda c: (-c.score, c.doc_id, c.unit_ids)):
        k = c.key()
        if k in seen_keys:
            continue
        cov = covered.setdefault(c.section_id, set())
        new_units = set(c.number_unit_ids) | set(c.hit_unit_ids)
        if c.kind != "clip" and new_units and new_units <= cov:
            continue
        seen_keys.add(k)
        kept.append(c)
        cov.update(c.number_unit_ids)
        cov.update(c.hit_unit_ids)
    return kept


def select_round_robin(cands, doc_order, per_doc_cap, global_cap):
    """Documents take turns (year order); inside a document the best window of each section comes before a
    second window of any section (diversity), then by score."""
    by = {}
    for c in cands:
        by.setdefault(c.doc_id, []).append(c)
    for d in by:
        ranked = sorted(by[d], key=lambda c: -c.score)
        seen_sec = {}
        for c in ranked:
            seen_sec[c.section_id] = seen_sec.get(c.section_id, 0) + 1
            c._sec_rank = seen_sec[c.section_id]
        by[d] = sorted(ranked, key=lambda c: (c._sec_rank > 1, -c.score))
    order = [d for d in doc_order if d in by] + [d for d in by if d not in doc_order]
    out = []
    taken = {d: 0 for d in order}
    progress = True
    while progress and len(out) < global_cap:
        progress = False
        for d in order:
            if len(out) >= global_cap:
                break
            if taken[d] < min(per_doc_cap, len(by[d])):
                out.append(by[d][taken[d]])
                taken[d] += 1
                progress = True
    return out


def search(root, corpus, plan, target_metas, *, d_lang, question, opts, round_no=1, per_doc_cap=None,
           global_cap=None, exclude_units=None, only_docs=None, line_only=False, cfg=None):
    """-> (selected candidates best-first in round-robin order, info)."""
    t0 = time.perf_counter()
    widened = round_no > 1
    ctx = SearchContext(plan, d_lang, question=question, widened=widened)
    metas = [m for m in target_metas if (only_docs is None or m["doc_id"] in only_docs)]
    meta_by = {m["doc_id"]: m for m in metas}
    doc_ids = [m["doc_id"] for m in metas if corpus.has_doc(m["doc_id"])]
    info = {"round": round_no, "patterns": ctx.pattern_info, "search_langs": ctx.search_langs, "docs": doc_ids,
            "unit_kind": ctx.unit_kind}
    if not doc_ids or not ctx.patterns:
        info.update(hit_lines=0, windows=0, kept=0, selected=0, rg_ms=0.0, rg_invocations=0, windows_ms=0.0,
                    ms=round((time.perf_counter() - t0) * 1000, 1))
        return [], info
    hits, bad = [], []
    t_rg = time.perf_counter()
    n_rg = 1
    try:
        hits = grep(root, ctx.patterns, doc_ids=doc_ids, ignore_case=True, max_hits=int(opts["rg_max_hits"]),
                    timeout_s=float(opts["rg_timeout_s"]), cfg=cfg)
    except ValueError:
        good = []
        for p in ctx.patterns:
            n_rg += 1
            try:
                grep(root, p, doc_ids=doc_ids[:1], max_hits=1, cfg=cfg)
                good.append(p)
            except ValueError:
                bad.append(p)
        n_rg += 1
        hits = grep(root, good, doc_ids=doc_ids, ignore_case=True, max_hits=int(opts["rg_max_hits"]),
                    timeout_s=float(opts["rg_timeout_s"]), cfg=cfg) if good else []
    except subprocess.TimeoutExpired:
        info["timeout"] = True
    info["rg_ms"] = round((time.perf_counter() - t_rg) * 1000.0, 1)
    info["rg_invocations"] = n_rg
    t_win = time.perf_counter()
    info["bad_patterns"] = bad
    info["hit_lines"] = len(hits)
    info["hit_lines_capped"] = len(hits) >= int(opts["rg_max_hits"])
    groups = {}
    for h in hits:
        if h.tag == "meta" and not h.text.startswith("@@TITLE "):
            continue
        g = groups.setdefault((h.doc_id, h.section_id), {"units": [], "title": False})
        if h.text.startswith("@@TITLE "):
            g["title"] = True
        elif h.unit_id and h.unit_id not in g["units"]:
            g["units"].append(h.unit_id)
    windows = []
    for (doc_id, sec_id), g in groups.items():
        try:
            doc = corpus.doc(doc_id)
            section = doc.section(sec_id)
        except KeyError:
            continue
        for c in build_windows(doc, section, g["units"], g["title"], ctx, opts, round_no, line_only=line_only):
            score_candidate(c, doc, section, meta_by.get(doc_id, {}), ctx)
            windows.append(c)
    info["windows"] = len(windows)
    if ctx.unit_kind != "text":
        before = len(windows)
        windows = [c for c in windows if c.features.get("numbers")]
        info["windows_without_number"] = before - len(windows)
    before = len(windows)
    windows = merge_adjacent(windows, corpus, meta_by, ctx,
                             int(opts["gap_window_chars"] if widened else opts["window_chars"]))
    info["merged_away"] = before - len(windows)
    if exclude_units:
        before = len(windows)
        ex = set(exclude_units)
        windows = [c for c in windows if not ((set(c.number_unit_ids) | set(c.hit_unit_ids)) <= ex)]
        info["excluded_seen"] = before - len(windows)
    kept = dedupe(windows)
    info["kept"] = len(kept)
    per_doc = per_doc_cap or (int(opts["single_doc_cap"]) if len(doc_ids) == 1 else int(opts["per_doc_cap"]))
    cap = global_cap or int(opts["max_candidates"])
    sel = select_round_robin(kept, doc_ids, per_doc, cap)
    info.update(per_doc_cap=per_doc, global_cap=cap, selected=len(sel),
                per_doc_kept={d: sum(1 for c in kept if c.doc_id == d) for d in doc_ids},
                per_doc_selected={d: sum(1 for c in sel if c.doc_id == d) for d in doc_ids},
                windows_ms=round((time.perf_counter() - t_win) * 1000, 1),
                ms=round((time.perf_counter() - t0) * 1000, 1))
    return sel, info
