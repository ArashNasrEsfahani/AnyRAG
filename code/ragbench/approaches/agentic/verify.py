"""Stage 3 - batched verification (one model call per batch of candidate snippets) and the code guard.

The model reads several numbered snippets and returns, per snippet, strict JSON
    {"id", "relevant", "value", "unit", "period", "definition", "quote"}
Then the CODE GUARD re-checks every relevant item against the snippet the model was shown (a 4B model picks
nearby wrong numbers, paraphrases labels and "fixes" digits):
    quote_empty / quote_no_label     the quote must hold a label (letters), not only a number
    quote_not_in_snippet             canon(quote) must be a substring of canon(snippet); a quote with an
                                     ellipsis must match piece by piece, in order
    value_not_numeric                a numeric question needs a number in the value
    value_not_in_snippet             every number of the value must have a reading equal to a number token of
                                     the snippet (separators/scale words may differ, digits may not)
    value_not_in_quote               ... and of the quote (the label the model quoted is the one next to it)
    value_is_year                    the only matching token is a bare year (e.g. 1402) for a non-date question
    text_value_not_in_quote          a text answer must be a substring of the quote
canon() = normalize + fold (yeh/kaf/alef variants, ZWNJ as space, lower case), digit-group separators
removed, other punctuation as spaces, whitespace collapsed. Nothing is fuzzy-matched.
"""
import json
import re

from ...llm import extract_json
from ...normalize import fold, normalize
from .defaults import VERIFY_PROMPT_VERSION
from .numbers import find_numbers, same_number

VERIFY_SYSTEM = """You check candidate snippets from reports for one question. The snippets were found by a keyword search and most of them do NOT answer it.

For EVERY snippet decide whether it states the value the question asks for: the requested metric, for the requested entity. Reply with one JSON object with one key per snippet, "s1" for snippet [1], "s2" for snippet [2], and so on, none skipped:
{"s1": {"relevant": true or false, "values": [{"value": "...", "unit": "...", "period": "...", "definition": "...", "quote": "..."}]}, "s2": {...}, ...}
"values" is [] when relevant is false. When a snippet states several values of the metric with different definitions (for example a total and an active count), give one entry per value (at most 3).

For each value:
- value: the value exactly as it is written in the snippet (same digits, separators and words). Never compute, convert, round, add up or translate numbers.
- unit: the unit written next to the value (for example نفر, کاربر, تن, %, تومان, USD), or "".
- period: the year, date or period the value refers to, as written in the snippet; "" if the snippet does not say.
- definition: what exactly is counted or measured, in a few words of the snippet's language, keeping every qualifier the snippet gives (for example active, registered, total, a service or product name, a city, a month).
- quote: the exact text from the snippet that contains both the label and the value, copied character by character. It may run across line breaks. Do not shorten words inside it.

A snippet is NOT relevant when it gives a different metric, a different entity, only a part or breakdown of the metric (for example only women, only one city, only one month, only one service, only the users of one programme) while the question asks for the whole, numbers on a chart axis, or a number whose label is unclear. Several snippets may be relevant."""

VERIFY_USER = """Question: {question}
Looking for: {metric}{entity}{unit}{period}

Snippets:
{snippets}"""


MAX_ITEMS_PER_SNIPPET = 3


def verify_schema(n):
    """One required key per snippet (s1..sn), so the grammar-constrained reply cannot skip a snippet (the live
    4B verifier answered 1 of 8 snippets with a free-length items array)."""
    value = {"type": "object",
             "properties": {"value": {"type": "string"}, "unit": {"type": "string"}, "period": {"type": "string"},
                            "definition": {"type": "string"}, "quote": {"type": "string"}},
             "required": ["value", "unit", "period", "definition", "quote"]}
    entry = {"type": "object",
             "properties": {"relevant": {"type": "boolean"},
                            "values": {"type": "array", "items": value, "maxItems": MAX_ITEMS_PER_SNIPPET}},
             "required": ["relevant", "values"]}
    keys = ["s%d" % i for i in range(1, n + 1)]
    return {"type": "object", "properties": {k: entry for k in keys}, "required": keys}


def snippet_header(c, corpus):
    try:
        doc = corpus.doc(c.doc_id)
        path = [t for t in doc.section_path(c.section_id) if t][-2:]
        return "%s | %s" % (doc.title[:90], " > ".join(p[:60] for p in path))
    except KeyError:
        return c.doc_id


def verify_messages(question, plan, batch, corpus):
    parts = []
    for i, c in enumerate(batch, 1):
        parts.append("[%d] (%s)\n%s" % (i, snippet_header(c, corpus), c.text.strip()))
    ent = plan.get("entity") or ""
    p0, p1 = (plan.get("period") or (None, None))
    period = ""
    if p0 is not None or p1 is not None:
        period = "; period: %s" % (str(p0) if p0 == p1 or p1 is None else "%s to %s" % (p0 if p0 is not None else "", p1))
    user = VERIFY_USER.format(question=normalize(question), metric=plan.get("metric") or "the value asked",
                              entity=("; entity: %s" % ent) if ent else "",
                              unit=("; expected value type: %s" % plan.get("unit")) if plan.get("unit") else "",
                              period=period, snippets="\n\n".join(parts))
    return [{"role": "system", "content": VERIFY_SYSTEM}, {"role": "user", "content": user}]


def _keyed_items(obj, n):
    """{"s1": {"relevant", "values": [...]}, ...} -> {i: [flat item dicts]}"""
    out = {}
    for k, v in obj.items():
        m = re.fullmatch(r"\s*s?\s*(\d+)\s*", str(k))
        if not m or not isinstance(v, dict):
            continue
        i = int(m.group(1))
        if not 1 <= i <= n:
            continue
        rel = v.get("relevant")
        vals = [x for x in (v.get("values") or []) if isinstance(x, dict)][:MAX_ITEMS_PER_SNIPPET]
        if vals:
            out[i] = [dict(x, id=i, relevant=rel) for x in vals]
        else:
            out[i] = [{"id": i, "relevant": rel, "value": "", "unit": "", "period": "", "definition": "", "quote": ""}]
    return out


def parse_items(text, n):
    """-> ({id: [item dicts]}, how). Reads the keyed reply {"s1": {...}, ...} (and the older {"items": [...]} list).
    Tolerates chatter and a reply cut by the token limit (every complete snippet entry / item object is kept).
    At most MAX_ITEMS_PER_SNIPPET values per snippet."""
    obj = None
    how = None
    try:
        obj = json.loads(text)
        how = "json"
    except (ValueError, TypeError):
        obj = extract_json(text)
        how = "extract" if obj is not None else None
    if isinstance(obj, dict) and any(re.fullmatch(r"\s*s\s*\d+\s*", str(k)) for k in obj):
        return _keyed_items(obj, n), how
    if text and re.search(r"\"s\d+\"\s*:\s*\{", text):
        dec = json.JSONDecoder()
        salvaged = {}
        for m in re.finditer(r"\"(s\d+)\"\s*:\s*\{", text):
            try:
                o, _e = dec.raw_decode(text[m.end() - 1:])
                salvaged[m.group(1)] = o
            except ValueError:
                continue
        if salvaged:
            return _keyed_items(salvaged, n), "salvaged"
    items = None
    if isinstance(obj, dict) and isinstance(obj.get("items"), list):
        items = obj["items"]
    elif isinstance(obj, list):
        items = obj
    if items is None:
        items = []
        dec = json.JSONDecoder()
        for m in re.finditer(r"\{\s*\"id\"", text or ""):
            try:
                o, _e = dec.raw_decode(text[m.start():])
                if isinstance(o, dict):
                    items.append(o)
            except ValueError:
                continue
        how = "salvaged" if items else None
    out = {}
    for pos, it in enumerate(items, 1):
        if not isinstance(it, dict):
            continue
        try:
            i = int(it.get("id", pos))
        except (TypeError, ValueError):
            i = pos
        if 1 <= i <= n and len(out.get(i, [])) < MAX_ITEMS_PER_SNIPPET:
            out.setdefault(i, []).append(it)
    return out, how


# --------------------------------------------------------------------------
# the guard
# --------------------------------------------------------------------------

_DIGIT_SEP = re.compile(r"(?<=[0-9])\s?[,„‚٬]\s?\+?(?=[0-9])")
_PUNCT = re.compile(r"[^\w\s%.]|_", re.UNICODE)


def canon(text):
    t = normalize(text or "")
    t = _DIGIT_SEP.sub("", t)
    t = fold(t)
    t = _PUNCT.sub(" ", t)
    t = re.sub(r"(?<![0-9])\.|\.(?![0-9])", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def _in_order(hay, pieces):
    pos = 0
    for p in pieces:
        k = hay.find(p, pos)
        if k < 0:
            return False
        pos = k + len(p)
    return True


def _letters(t):
    return sum(1 for ch in t if ch.isalpha())


def guard(item, snippet, *, unit_kind, min_quote_letters=2, term_rx=None):
    """-> (ok, reason, info). info: value_num, value_token (raw token of the quote), reading.
    term_rx: compiled regexes of the metric's words; when given, the quote must contain one of them
    (quote_without_term: the model quoted a number labelled with something else, e.g. staff for drivers)."""
    quote = str(item.get("quote") or "")
    value = str(item.get("value") or "")
    cs = canon(snippet)
    pieces = [canon(p) for p in re.split(r"…|\.\.\.", quote)]
    pieces = [p for p in pieces if p]
    if not pieces:
        return False, "quote_empty", {}
    cq = " ".join(pieces)
    if _letters(cq) < min_quote_letters:
        return False, "quote_no_label", {}
    if not _in_order(cs, pieces):
        return False, "quote_not_in_snippet", {}
    if term_rx:
        qn = normalize(quote)
        if not any(rx.search(qn) for rx in term_rx):
            return False, "quote_without_term", {}
    vnorm = normalize(value)
    vtoks = [t for t in find_numbers(vnorm) if not t.is_date]
    numeric = unit_kind != "text" or bool(vtoks)
    if not numeric:
        cv = canon(value)
        if not cv:
            return False, "value_empty", {}
        if cv not in cq:
            return False, ("text_value_not_in_quote" if cv in cs else "text_value_not_in_snippet"), {}
        return True, "ok", {"value_num": None, "value_token": value}
    if not vtoks:
        return False, "value_not_numeric", {}
    qtoks = []
    for p in re.split(r"…|\.\.\.", quote):
        qtoks += find_numbers(normalize(p))
    stoks = find_numbers(normalize(snippet))
    # the value's years count only when the value is nothing but a year (then value_is_year may apply)
    non_year = [t for t in vtoks if not t.is_year]
    if len({round(t.main or 0, 6) for t in non_year}) > 1:
        return False, "value_has_several_numbers", {"value_raw": value[:80]}
    check = non_year or vtoks
    matched = []
    for vt in check:
        in_quote = [qt for qt in qtoks if any(same_number(a, b) for a in vt.readings for b in qt.readings)]
        if not in_quote:
            in_snip = any(any(same_number(a, b) for a in vt.readings for b in st.readings) for st in stoks)
            return False, ("value_not_in_quote" if in_snip else "value_not_in_snippet"), {"value_raw": vt.raw}
        matched.append((vt, in_quote[0]))
    vt, qt = max(matched, key=lambda p: max(p[1].readings) if p[1].readings else 0)
    if unit_kind != "date" and qt.is_year and all(q.is_year for _v, q in matched):
        return False, "value_is_year", {"value_raw": vt.raw}
    if qt.main is not None and is_axis_tick(stoks, qt.main):
        return False, "value_is_axis_tick", {"value_raw": vt.raw}
    # the source token governs the value: its scale word and, for a right-to-left glitch token, its
    # reversed thousands groups (the model's own spelling only has to share one reading with it)
    main = qt.main
    how = "reversed_groups" if (qt.glitch and qt.reversed_reading is not None and main is not None
                                and same_number(main, qt.reversed_reading)) else (
        "scaled" if qt.scale != 1.0 else "as_written")
    return True, "ok", {"value_num": main, "value_token": qt.raw, "reading": how, "percent": qt.percent}


def is_axis_tick(tokens, value, min_step=5):
    """True when `value` sits in a run of 3 consecutive numbers of the snippet with one constant step that divides
    them all (400,000,000 / 350,000,000 / 300,000,000: a chart axis extracted as text, not a statement)."""
    vals = [t.main for t in tokens if t.readings and not t.is_date and not t.percent and not t.is_year]
    for i, v in enumerate(vals):
        if v is None or not same_number(v, value):
            continue
        for a in (i - 2, i - 1, i):
            w = vals[a:a + 3] if a >= 0 else []
            if len(w) != 3 or any(x is None for x in w):
                continue
            d1, d2 = w[1] - w[0], w[2] - w[1]
            if abs(d1) >= min_step and same_number(d1, d2) and all(abs(x) % abs(d1) == 0 for x in w):
                return True
    return False


def value_unit_id(candidate, corpus, token_raw, quote):
    """The unit of the window holding the value token (prefer a unit overlapping the quote)."""
    cq = canon(quote)
    best = None
    for uid in candidate.unit_ids:
        try:
            u = corpus.unit(uid)
        except KeyError:
            continue
        if token_raw and token_raw in u.text:
            cu = canon(u.text)
            if cu and (cu in cq or any(w in cu for w in cq.split()[:3])):
                return uid
            best = best or uid
    return best or (candidate.hit_unit_ids[0] if candidate.hit_unit_ids else (candidate.unit_ids[0] if candidate.unit_ids else None))


__all__ = ["VERIFY_PROMPT_VERSION", "verify_messages", "verify_schema", "parse_items", "guard", "canon",
           "value_unit_id"]
