"""OURS cheap index levels, the program-only parts (no model calls).

L0 fact lines (FACTX_VERSION): numbers found by regex in a section's own units, each with its unit/scale word and the
nearest label words, written as the same "metric | value | unit | period" lines the L3 enrichment writes, so the
executor ranks, selects and packs them exactly like L3 fact lines (fact_evidence="context" shows the verbatim source
text of each matched value).
  * text is the normalised unit text (ragbench.normalize: Persian/Arabic digits -> ASCII, ٫ -> ".", ٬ -> ",", ٪ -> "%")
  * a number: 82,663 | 12.5 | 47/8 (Persian slash decimal) | a range "150 تا 550", "22 to 40"
  * its unit: attached or following unit/scale words (هزار میلیارد تومان, درصد/درصدی, million tonnes, %, $B), a currency
    symbol or a detached percent sign before it ("$156.8B", "% 47/8" from right-to-left PDF extraction)
  * period: a year / year range / date in the same clause (tables: the year column, a year header cell)
  * metric (label words): words of the number's clause (<= l0_label_words_before before, <= l0_label_words_after after,
    edge stopwords stripped); fewer than l0_min_label_words -> the neighbouring clauses of the line (the "label:" of a
    KPI line), then the previous / next line of the unit, the previous short unit, the section title. Table cells: the
    row's text cells + the column header (its "(unit)" moved to the unit field).
  * skipped: years without a unit (they become periods), dates, chapter/figure/page references, list enumerators
    "1)", identifiers ("5G", "3D", "B2B"), URLs.
Facts are scored (unit +2, table cell +1, >= 2 label words +1, small integer without unit -1), those scoring >= 0 are
kept, capped per section by score, and written in document order.

L1 / L2 helpers: the per-document digest the glossary call reads (section titles + most frequent content terms), the
projection of a document glossary onto the sections whose text contains a glossary term (folded token match with
light plural stripping), and the per-section input block of the L2 keyword batches.
"""
import re
from collections import Counter

from ...normalize import fold, normalize, script_counts

FACTX_VERSION = "ours-factx-v1"
PROJECTION_VERSION = "ours-glproj-v2"   # v2: entries found in more than l1_max_section_share of the sections are not projected

L0_DEFAULTS = {
    "l0_max_facts_per_section": 60,
    "l0_label_words_before": 6,
    "l0_label_words_after": 5,
    "l0_min_label_words": 2,
    "l0_label_unit_chars": 160,         # a previous unit up to this long can label a detached number
    "field_max_chars": 120,
}

# ---------------------------------------------------------------------------------------------------------------
# vocabularies (normalised spelling)
# ---------------------------------------------------------------------------------------------------------------

FA_UNIT_WORDS = {
    "هزار", "میلیون", "میلیارد", "بیلیون", "تریلیون", "همت",
    "تومان", "ریال", "دلار", "یورو", "درهم", "یوان", "پوند", "لیر", "روبل", "روپیه",
    "درصد",
    "تن", "کیلوگرم", "گرم", "لیتر", "بشکه", "متر", "مترمربع", "کیلومتر", "هکتار", "مگاوات", "گیگاوات", "کیلووات",
    "کیلووات‌ساعت", "مگاوات‌ساعت", "ساعت", "دقیقه", "روز", "هفته", "ماه", "سال", "نفر", "برابر", "واحد", "امتیاز",
}
EN_UNIT_WORDS = {
    "%", "percent", "pct", "pp", "bps", "thousand", "thousands", "million", "millions", "billion", "billions",
    "trillion", "trillions", "bn", "mn", "tn", "t", "tonnes", "tonne", "tons", "ton", "kt", "mt", "kg", "km", "ha",
    "hectares", "mw", "gw", "kw", "kwh", "mwh", "gwh", "twh", "kwh/t", "barrels", "bbl", "x", "times", "usd",
    "dollars", "dollar", "eur", "euros", "euro", "gbp", "pounds", "rials", "rial", "tomans", "toman", "irr", "yuan",
    "rmb", "aed", "dirhams", "years", "months", "days", "hours", "minutes", "people",
}
EN_TWO_WORD_UNITS = re.compile(r"\s{0,2}(per cent|percentage points?|basis points?)\b", re.I)
EN_ATTACHED = {"k": "k", "m": "M", "b": "B", "bn": "bn", "mn": "mn", "tn": "tn", "t": "t", "kt": "kt", "mt": "Mt",
               "x": "x", "kg": "kg", "km": "km", "kwh": "kWh", "mwh": "MWh", "gwh": "GWh", "twh": "TWh", "mw": "MW",
               "gw": "GW", "kw": "kW", "h": "h", "pp": "pp", "bps": "bps", "yr": "yr", "yrs": "yrs", "tr": "tr"}
REF_WORDS = {"فصل", "بخش", "شکل", "جدول", "نمودار", "صفحه", "ماده", "بند", "تبصره", "پیوست", "ردیف", "شماره",
             "chapter", "chapters", "section", "sections", "figure", "fig", "table", "page", "pages", "p", "pp",
             "article", "annex", "appendix", "note", "footnote", "step", "no", "item", "rank", "ranked", "top"}
YEAR_WORDS = {"سال", "year", "years", "cohort", "fiscal year", "سال مالی", "دوره", "period"}
INDEX_HEADERS = {"#", "ردیف", "rank", "رتبه", "no", "no.", "n", "ش"}

STOP = {
    "en": set("""a an the of in on at to for from by with and or is are was were be been being as than that this these
    those it its their there which who whom whose what when where why how into about over under up down out per via
    has have had do does did not no nor but so if then also only just more most less least some any all each every
    other such same both either neither can could would should will shall may might must our your his her they we
    you i he she them us my me after before between during since until while within without across against around
    among near nearly almost approximately roughly""".split()),
    "fa": set("""از به با در که را این آن و یا تا بر برای هم نیز است هست بود شد شده شود می نمی های ها ای یک اما ولی
    اگر چون پس بین طی درباره همین همان کل کلی بوده باشد دارد دارند داشت داشته کرد کرده کند کنند شده‌اند بوده‌اند
    حدود بیش بیشتر کمتر تنها فقط نزدیک هر همه چند چه چگونه کدام آیا خود خویش او ما شما آنها ایشان وی دیگر سایر
    نسبت طبق اساس مورد پیش پس‌از پیش‌از روی زیر بالای میان سوی جای ضمن توسط وجود دارای بدون حتی نه هیچ بسیار
    است؛ باید نیست نبود شدند می‌شود می‌کند می‌دهد داده دهد رسید رسیده رسد""".split()),
}
_ALL_STOP = STOP["en"] | STOP["fa"]

# ---------------------------------------------------------------------------------------------------------------
# regexes
# ---------------------------------------------------------------------------------------------------------------

_NUM_RE = re.compile(r"(?<![\w.,/])(\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:[./]\d+)?)(?!\d)")
_WORD_RE = re.compile(r"(?:[^\W\d_]|[ً-ٰٟ])+(?:[‌'’\-](?:[^\W\d_]|[ً-ٰٟ])+)*")
_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_DATE_RE = re.compile(r"(?<![\d/])\d{1,4}/\d{1,2}/\d{1,4}(?![\d/])")
_YEAR_RANGE_RE = re.compile(r"(?<![\d,.])((?:13|14|19|20)\d{2})\s*[-–—]\s*((?:13|14|19|20)?\d{2})(?![\d%])")
_PERIOD_PREFIX_RE = re.compile(r"(?:\b(?:H[12]|Q[1-4]|FY)\s?)$")
_CLAUSE_RE = re.compile(r"[;؛:|()\[\]«»\"“”،]|,(?=\s)|\s[-–—]\s|[.!?؟…](?=\s|$)")
_CURRENCY_BEFORE = re.compile(r"(US\$|\$|€|£|¥|₹)\s?$")
_CURRENCY_WORD_BEFORE = re.compile(r"(?:^|\s)(USD|EUR|GBP|IRR|AED)\s$")
_PERCENT_BEFORE = re.compile(r"%\s?$")
_RANGE_JOIN = re.compile(r"\s*(تا|to|[-–—])\s*")
_FA_LETTERS = re.compile(r"[؀-ۿ‌]+")
_EN_LETTERS = re.compile(r"[A-Za-z]+")
_UNIT_TOKEN = re.compile(r"\s{0,2}(%|[^\s\d|:;،؛,()«»\[\]\"“”!?؟]+)")
_PAREN_UNIT = re.compile(r"\(([^()]{1,30})\)")


def _canon_int(tok):
    t = tok.replace(",", "")
    return int(t) if t.isdigit() else None


def is_year_token(tok, commas=True):
    """A plain 4-digit Persian (13xx/14xx) or Gregorian (19xx/20xx) year; commas tolerated only when `commas`
    (table cells print years as "2,015"; in running text "1,340" is a count)."""
    if "." in tok or "/" in tok or (not commas and "," in tok):
        return False
    v = _canon_int(tok)
    return v is not None and len(str(v)) == 4 and (1300 <= v <= 1499 or 1900 <= v <= 2100)


def _fa_unit(word):
    """Canonical unit word for a Persian token (adjective -ی and ZWNJ-less forms tolerated), else None."""
    w = word.strip("‌")
    for cand in (w, w[:-1] if w.endswith("ی") else None, w.replace("‌", "")):
        if cand and cand in FA_UNIT_WORDS:
            return cand
    return None


def _en_unit(word):
    w = word.lower().rstrip(".")
    return w if w in EN_UNIT_WORDS else None


def _words(text, start=0):
    """[(start, end, word)] letter words (Latin single letters dropped)."""
    out = []
    for m in _WORD_RE.finditer(text):
        w = m.group(0)
        if len(w) == 1 and w.isascii():
            continue
        out.append((start + m.start(), start + m.end(), w))
    return out


def _strip_edges(words):
    ws = list(words)
    while ws and ws[0].lower() in _ALL_STOP:
        ws.pop(0)
    while ws and ws[-1].lower() in _ALL_STOP:
        ws.pop()
    return ws


def _clauses(line):
    """[(start, end)] clause spans of one line."""
    spans, last = [], 0
    for m in _CLAUSE_RE.finditer(line):
        if m.start() > last:
            spans.append((last, m.start()))
        last = m.end()
    if last < len(line):
        spans.append((last, len(line)))
    return spans or [(0, len(line))]


def _clean_field(s, max_chars):
    s = " ".join((s or "").replace("|", "/").replace(";", ",").split()).strip(" -,:")
    return s[:max_chars].strip()


# ---------------------------------------------------------------------------------------------------------------
# numbers of one line
# ---------------------------------------------------------------------------------------------------------------

def scan_numbers(line, table_cell=False):
    """-> (numbers, periods). numbers: [{start, end, num_start, value, unit, has_unit, small}] (start/end cover the
    currency prefix, the number(s) and the unit words); periods: [(start, end, text)] years, year ranges and dates.
    table_cell: a comma-grouped 4-digit number ("2,015") may be a year."""
    masked = []
    periods = []
    for m in _URL_RE.finditer(line):
        masked.append((m.start(), m.end()))
    for m in _DATE_RE.finditer(line):
        masked.append((m.start(), m.end()))
        periods.append((m.start(), m.end(), m.group(0)))
    for m in _YEAR_RANGE_RE.finditer(line):
        if any(a <= m.start() < b for a, b in masked):
            continue
        masked.append((m.start(), m.end()))
        periods.append((m.start(), m.end(), m.group(0).replace(" ", "")))

    def in_mask(p):
        return any(a <= p < b for a, b in masked)

    nums = []
    consumed_until = -1
    for m in _NUM_RE.finditer(line):
        a, b = m.span()
        if a < consumed_until or in_mask(a):
            continue
        tok = m.group(1)
        value = tok
        units = []
        pos = b
        # attached letters
        ma = _EN_LETTERS.match(line, pos)
        if ma:
            w = ma.group(0).lower()
            if w in EN_ATTACHED:
                units.append(EN_ATTACHED[w])
                pos = ma.end()
            else:
                continue                                    # identifier / ordinal: 5G, 3D, 1st
        else:
            mf = _FA_LETTERS.match(line, pos)
            if mf:
                u = _fa_unit(mf.group(0))
                if u:
                    units.append(mf.group(0).strip("‌"))
                    pos = mf.end()
        if not units:
            if line.startswith("%", pos):
                units.append("%")
                pos += 1
            else:
                # a range: "150 تا 550 هزار تومان", "22 to 40", "700–900"
                mr = _RANGE_JOIN.match(line, pos)
                if mr:
                    m2 = _NUM_RE.match(line, mr.end())
                    if m2 and not is_year_token(tok, table_cell) and not in_mask(m2.start()):
                        value = line[a:m2.end()]
                        pos = m2.end()
                        consumed_until = m2.end()
                        ma2 = _EN_LETTERS.match(line, pos)
                        if ma2 and ma2.group(0).lower() in EN_ATTACHED:
                            units.append(EN_ATTACHED[ma2.group(0).lower()])
                            pos = ma2.end()
                        elif line.startswith("%", pos):
                            units.append("%")
                            pos += 1
        # following unit words
        for _ in range(4):
            m2 = EN_TWO_WORD_UNITS.match(line, pos)
            if m2:
                units.append(m2.group(1).lower())
                pos = m2.end()
                continue
            mw = _UNIT_TOKEN.match(line, pos)
            if not mw:
                break
            t = mw.group(1)
            if t == "%":
                units.append("%")
                pos = mw.end()
                continue
            t_clean = t.rstrip(".")
            if _en_unit(t_clean) or _fa_unit(t_clean):
                units.append(t_clean)
                pos = mw.start(1) + len(t_clean)
                continue
            break
        start = a
        before = line[:a]
        prefix = ""
        mc = _CURRENCY_BEFORE.search(before)
        if mc:
            prefix = mc.group(1)
            start = mc.start()
        else:
            mcw = _CURRENCY_WORD_BEFORE.search(before)
            if mcw:
                prefix = mcw.group(1)
                start = mcw.start(1)
        if not units and not prefix and _PERCENT_BEFORE.search(before):
            units.append("%")
            start = _PERCENT_BEFORE.search(before).start()
        has_unit = bool(units or prefix)
        # skip rules
        after = line[pos:pos + 2]
        prev_words = _words(before[-40:])
        prev_word = prev_words[-1][2].lower() if prev_words and before[-40:].rstrip().endswith(prev_words[-1][2]) else ""
        if not has_unit:
            if is_year_token(tok, table_cell) and value == tok:
                pm = _PERIOD_PREFIX_RE.search(before)
                periods.append(((pm.start() if pm else a), b, ((pm.group(0) if pm else "") + tok).replace(",", "")))
                continue
            if prev_word in REF_WORDS:
                continue
            if after[:1] == ")" or (after[:1] == "." and after[1:2] in (" ", "")) and not re.search(r"\d", line[pos:pos + 3]):
                if re.fullmatch(r"\d{1,2}", tok):
                    continue                                # list enumerator "1)" / "2."
        if prefix:
            u = " ".join(units)
            unit = prefix + (u if units and len(units[0]) <= 2 else (" " + u if u else ""))
        else:
            unit = " ".join(units)
        v_int = _canon_int(tok)
        small = (not has_unit) and v_int is not None and v_int <= 10 and value == tok
        nums.append({"start": start, "end": pos, "num_start": a, "value": value, "unit": unit.strip(),
                     "has_unit": has_unit, "small": small})
    return nums, periods


def _period_for(num, periods, clause):
    ca, cb = clause
    best, dist = "", None
    for a, b, t in periods:
        if a >= ca and b <= cb and not (num["start"] <= a < num["end"]):
            d = abs(a - num["start"])
            if dist is None or d < dist:
                best, dist = t, d
    return best


# ---------------------------------------------------------------------------------------------------------------
# facts of a section
# ---------------------------------------------------------------------------------------------------------------

def _label_from_line(line, num, clause, nums, opts):
    """(before words, after words, source) for one number of a line.
    Window: the words of the number's clause, bounded by the neighbouring numbers of that clause. For a number that
    follows another number in its clause, the words in between belong to the previous number when this one has words
    after it ("8 declaration suppliers and 6 risk-management suppliers"); the clause subject (the words before its
    first number, "Annual capital has fallen from $38.9B ... to $10.6B ...") is added instead.
    No words at all -> the neighbouring clauses of the line; fewer than l0_min_label_words and the clause follows a
    colon -> the label before the colon ("Disclosed capital: $156.8B")."""
    kb, ka = int(opts["l0_label_words_before"]), int(opts["l0_label_words_after"])
    ca, cb = clause
    ws = _words(line[ca:cb], ca)
    same = sorted((n for n in nums if ca <= n["num_start"] < cb), key=lambda n: n["num_start"])
    prevs = [n for n in same if n["num_start"] < num["num_start"]]
    nexts = [n for n in same if n["num_start"] > num["num_start"]]
    lo = prevs[-1]["end"] if prevs else ca
    hi = nexts[0]["start"] if nexts else cb
    after = _strip_edges([w for s, e, w in ws if s >= num["end"] and e <= hi][:ka])
    between = _strip_edges([w for s, e, w in ws if s >= lo and e <= num["start"]][-kb:])
    if prevs:
        subject = _strip_edges([w for s, e, w in ws if e <= same[0]["start"]][-kb:])
        before = subject + ([] if after else between)
    else:
        before = between
    src = "clause"
    own = len(before) + len(after)
    if own < int(opts["l0_min_label_words"]):
        clauses = _clauses(line)
        k = next((i for i, (a, b) in enumerate(clauses) if a <= num["num_start"] < b), None)
        if k is not None and k > 0:
            a, b = clauses[k - 1]
            colon = ":" in line[b:clauses[k][0]]
            if own == 0 or colon:
                before = _strip_edges([w for _, _, w in _words(line[a:b])][-kb:]) + before
                src = "line"
        if own == 0 and k is not None and k + 1 < len(clauses):
            a, b = clauses[k + 1]
            after = after + _strip_edges([w for _, _, w in _words(line[a:b])][: max(0, ka - len(after))])
            src = "line"
    return before, after, src


def _text_unit_facts(uid, text, ctx, opts):
    lines = text.split("\n")
    out = []
    for li, line in enumerate(lines):
        if not line.strip():
            continue
        nums, periods = scan_numbers(line)
        if not nums:
            continue
        clauses = _clauses(line)
        for num in nums:
            clause = next(((a, b) for a, b in clauses if a <= num["num_start"] < b), (0, len(line)))
            before, after, src = _label_from_line(line, num, clause, nums, opts)
            label = before + after
            penalty = 0
            if not label:
                for cand, why in _detached_label_sources(lines, li, ctx, opts):
                    words = _strip_edges([w for _, _, w in _words(cand)][-int(opts["l0_label_words_before"]):])
                    if words:
                        label = words
                        src = why
                        penalty = 2 if why == "section_title" else 0     # e.g. chart axis ticks
                        break
            period = _period_for(num, periods, clause)
            score = (2 if num["has_unit"] else 0) + (1 if len(label) >= int(opts["l0_min_label_words"]) else 0) \
                - (1 if num["small"] else 0) - penalty
            out.append({"metric": " ".join(label), "value": num["value"], "unit": num["unit"], "period": period,
                        "unit_id": uid, "score": score, "src": "text", "label_from": src})
    return out


def _detached_label_sources(lines, li, ctx, opts):
    """Label candidates for a number with no label words on its own line: the previous / next line of the unit, the
    previous unit when short (a caption or a KPI label), the section title."""
    for j in (li - 1, li + 1):
        if 0 <= j < len(lines) and lines[j].strip() and _words(lines[j]):
            yield lines[j], "prev_line" if j < li else "next_line"
    prev_unit = ctx.get("prev_unit_text") or ""
    if prev_unit and len(prev_unit) <= int(opts["l0_label_unit_chars"]):
        yield prev_unit.split("\n")[-1], "prev_unit"
    if ctx.get("section_title"):
        yield ctx["section_title"], "section_title"


def _cells(line):
    return [c.strip() for c in line.split("|")]


def _cell_number(cell):
    """(number dict, is_numeric_cell) for a table cell: numeric = nothing but numbers, units, signs left."""
    if not cell or not re.search(r"\d", cell):
        return None, False
    nums, periods = scan_numbers(cell, table_cell=True)
    rest = cell
    for n in sorted(nums, key=lambda n: -n["start"]):
        rest = rest[:n["start"]] + " " + rest[n["end"]:]
    letters = sum(1 for ch in rest if ch.isalpha())
    numeric = letters <= 1
    if nums:
        return nums[0], numeric
    if periods and numeric:
        raw = cell.strip()
        value = raw if re.fullmatch(r"\d[\d,]*", raw) else periods[0][2]     # "1,991" as written
        return {"start": 0, "end": len(cell), "num_start": 0, "value": value, "unit": "", "has_unit": False,
                "small": False, "year": True}, True
    return None, numeric


def _header_parts(h):
    """(label, unit, year) of a header cell: "(unit)" moved to the unit, a year header is a period."""
    unit = ""
    m = _PAREN_UNIT.search(h or "")
    if m:
        inner = m.group(1).strip()
        toks = inner.replace("/", " ").split()
        if any(ch in inner for ch in "$%€£") or any(_en_unit(t) or _fa_unit(t) for t in toks):
            unit = inner
            h = (h[:m.start()] + h[m.end():]).strip()
    year = ""
    hs = (h or "").strip()
    if re.fullmatch(r"\d[\d,]*", hs) and is_year_token(hs):
        year = hs.replace(",", "")
        hs = ""
    return " ".join(hs.split()), unit, year


def _table_unit_facts(uid, text, opts):
    lines = [l for l in text.split("\n") if l.strip()]
    header = _cells(lines[0])
    rows = [_cells(l) for l in lines[1:]]
    ncol = max([len(header)] + [len(r) for r in rows])
    hparts = [_header_parts(header[j]) if j < len(header) else ("", "", "") for j in range(ncol)]
    cells = [[(_cell_number(r[j]) if j < len(r) else (None, False)) for j in range(ncol)] for r in rows]
    skip_col, year_col = set(), set()
    for j in range(ncol):
        hl = hparts[j][0].lower()
        vals = [cells[i][j][0] for i in range(len(rows)) if cells[i][j][0] is not None and cells[i][j][1]]
        if hl in INDEX_HEADERS:
            skip_col.add(j)
            continue
        ints = [_canon_int(v["value"]) for v in vals]
        if vals and all(x is not None for x in ints) and ints == list(range(ints[0], ints[0] + len(ints))) \
                and ints[0] in (0, 1) and len(ints) >= 3 and not any(v["has_unit"] for v in vals):
            skip_col.add(j)                                 # 1, 2, 3, ... row numbers
            continue
        years = [v for v in vals if v.get("year") or (not v["has_unit"] and is_year_token(v["value"]))]
        # the row key is a period only in the first column or under a year header; a year-valued column elsewhere
        # ("median founding year") holds values
        if vals and len(years) == len(vals) and (hl in YEAR_WORDS or j == min(set(range(ncol)) - skip_col)):
            year_col.add(j)
    out = []
    for i, r in enumerate(rows):
        texts = [r[j] for j in range(min(len(r), ncol)) if r[j] and not cells[i][j][1] and j not in skip_col]
        row_label = [_clean_field(t, 60) for t in texts[:2]]
        row_year = ""
        for j in sorted(year_col):
            v = cells[i][j][0]
            if v is not None:
                row_year = v["value"].replace(",", "")
                break
        if not row_year:
            for t in texts[:2]:
                ys = [p[2] for p in scan_numbers(t, table_cell=True)[1]]
                if ys:
                    row_year = ys[0]
                    break
        for j in range(ncol):
            num, numeric = cells[i][j]
            if num is None or not numeric or j in skip_col or j in year_col:
                continue                                    # a year-like cell outside the period column is a value
            hlabel, hunit, hyear = hparts[j]
            if not hyear and hlabel:
                ys = [p[2] for p in scan_numbers(hlabel, table_cell=True)[1]]
                hyear = ys[0] if ys else ""
            metric_words = " ".join([x for x in row_label if x] + ([hlabel] if hlabel else []))
            unit = num["unit"] or hunit
            period = hyear or row_year
            n_words = len(_words(metric_words))
            score = 1 + (2 if (num["has_unit"] or hunit) else 0) + (1 if n_words >= int(opts["l0_min_label_words"]) else 0) \
                - (1 if (num["small"] and not hunit) else 0)
            out.append({"metric": metric_words, "value": num["value"], "unit": unit, "period": period, "unit_id": uid,
                        "score": score, "src": "table", "label_from": "row+header"})
    return out


def is_table_text(text):
    lines = [l for l in (text or "").split("\n") if l.strip()]
    return len(lines) >= 2 and "|" in lines[0] and sum(1 for l in lines[1:] if "|" in l) >= 1


def extract_facts(units, section_title="", opts=None):
    """units: [(unit_id, normalised text)] of ONE section (its own units, document order).
    -> (facts [{metric, value, unit, period, unit_id, score, src, label_from}] in document order, stats)"""
    o = dict(L0_DEFAULTS, **(opts or {}))
    mf = int(o["field_max_chars"])
    cands = []
    prev_text = ""
    for k, (uid, text) in enumerate(units):
        if text and re.search(r"\d", text):
            if is_table_text(text):
                fs = _table_unit_facts(uid, text, o)
            else:
                fs = _text_unit_facts(uid, text, {"prev_unit_text": prev_text, "section_title": section_title}, o)
            for f in fs:
                f["order"] = (k, len(cands))
                cands.append(f)
        prev_text = text or ""
    stats = {"candidates": len(cands), "dropped_score": 0, "dropped_cap": 0, "dropped_empty": 0, "duplicates": 0}
    kept, seen = [], set()
    for f in cands:
        f["metric"] = _clean_field(f["metric"], mf)
        f["value"] = _clean_field(f["value"], mf)
        f["unit"] = _clean_field(f["unit"], mf)
        f["period"] = _clean_field(f["period"], mf)
        if not f["value"] or not f["metric"]:
            stats["dropped_empty"] += 1
            continue
        if f["score"] < 0:
            stats["dropped_score"] += 1
            continue
        sig = (f["metric"].lower(), f["value"], f["unit"].lower(), f["period"])
        if sig in seen:
            stats["duplicates"] += 1
            continue
        seen.add(sig)
        kept.append(f)
    cap = int(o["l0_max_facts_per_section"])
    if len(kept) > cap:
        best = sorted(kept, key=lambda f: (-f["score"], f["order"]))[:cap]
        stats["dropped_cap"] = len(kept) - cap
        ids = {id(f) for f in best}
        kept = [f for f in kept if id(f) in ids]
    stats["kept"] = len(kept)
    return kept, stats


# ---------------------------------------------------------------------------------------------------------------
# L1: document digest and glossary projection
# ---------------------------------------------------------------------------------------------------------------

_SUFFIX_EN = (("ies", "y"), ("es", ""), ("s", ""))
_SUFFIX_FA = ("هایی", "های", "ها")


def stem(tok):
    """Light plural stripping on a folded token (matching only, both sides)."""
    if tok.isascii():
        if len(tok) > 4 and tok.endswith("ies"):
            return tok[:-3] + "y"
        if len(tok) > 4 and tok.endswith("sses"):
            return tok[:-2]
        if len(tok) > 3 and tok.endswith("s") and not tok.endswith("ss"):
            return tok[:-1]
        return tok
    for suf in _SUFFIX_FA:
        if tok.endswith(suf) and len(tok) - len(suf) >= 2:
            return tok[: -len(suf)]
    return tok


_FOLD_TOKEN_RE = re.compile(r"\w+")


def folded_stems(text):
    return [stem(t) for t in _FOLD_TOKEN_RE.findall(fold(text or ""))]


def term_stems(term):
    return tuple(s for s in folded_stems(term) if s)


def count_sequence(stems, seq, index=None):
    """Occurrences of the stem sequence `seq` in `stems` (index: {stem: [positions]} for speed)."""
    if not seq:
        return 0
    n = len(seq)
    starts = index.get(seq[0], []) if index is not None else [i for i, s in enumerate(stems) if s == seq[0]]
    return sum(1 for i in starts if tuple(stems[i:i + n]) == seq)


def stem_index(stems):
    idx = {}
    for i, s in enumerate(stems):
        idx.setdefault(s, []).append(i)
    return idx


def content_terms(texts, top_n=120, min_count=2):
    """Most frequent content unigrams and bigrams of a document's texts (normalised), in their most frequent
    surface spelling: [(surface, count)] (bigrams need >= min_count + 1 occurrences)."""
    uni, bi, surf = Counter(), Counter(), {}
    for text in texts:
        for line in (text or "").split("\n"):
            prev = None
            for m in _WORD_RE.finditer(line):
                w = m.group(0)
                low = w.lower()
                key = fold(low)
                if len(key.replace(" ", "")) < 3 or low in _ALL_STOP or key in _ALL_STOP:
                    prev = None
                    continue
                uni[key] += 1
                surf.setdefault(key, Counter())[low if w.isascii() else w] += 1
                if prev is not None and line[prev[1]:m.start()].strip() == "":
                    bkey = prev[0] + " " + key
                    bi[bkey] += 1
                    surf.setdefault(bkey, Counter())[prev[2] + " " + (low if w.isascii() else w)] += 1
                prev = (key, m.end(), low if w.isascii() else w)
    items = [(k, c) for k, c in uni.items() if c >= min_count] + [(k, c) for k, c in bi.items() if c >= min_count + 1]
    items.sort(key=lambda kc: (-kc[1], kc[0]))
    out, seen = [], set()
    for k, c in items:
        if k in seen:
            continue
        seen.add(k)
        out.append((surf[k].most_common(1)[0][0], c))
        if len(out) >= top_n:
            break
    return out


def doc_digest(doc, max_title_chars=3000, top_terms=120, min_count=2):
    """The compact document digest the glossary call reads: title, section titles (document order, unique, until
    max_title_chars) and the most frequent content terms with counts. -> (text, info)"""
    titles, seen, used = [], set(), 0
    n_titles = 0
    for s in doc.sections:
        t = " ".join((s.title or "").split())
        if not t or t.lower() in seen:
            continue
        seen.add(t.lower())
        n_titles += 1
        if used + len(t) + 3 > max_title_chars:
            continue
        titles.append(t[:120])
        used += len(t) + 3
    terms = content_terms([u.text for u in doc.units], top_n=top_terms, min_count=min_count)
    text = "Section titles:\n" + "\n".join("- " + t for t in titles) + "\n\nMost frequent terms (count): " + \
        "; ".join("%s (%d)" % (t, c) for t, c in terms)
    return text, {"titles_shown": len(titles), "titles_total": n_titles, "terms": len(terms), "chars": len(text)}


def section_match_text(doc, section):
    """Title + own unit texts (what a glossary term must occur in to be projected onto the section)."""
    return (section.title or "") + "\n" + "\n".join(doc.unit(u).text for u in section.unit_ids)


def project_glossary(entries, doc, doc_lang, max_per_section=8, sections=None, max_share=1.0, too_common=None):
    """Glossary entries -> {section_id: [entry indexes]} for the sections whose title or own text contains the entry's
    document-language term or synonym (folded, plural-tolerant), best first (title match, then count, then glossary
    order), at most max_per_section. Also -> {entry index: number of sections it was projected onto}.
    max_share: an entry found in more than this share of the document's sections with text is not projected at all
    (a header hit on nearly every section carries no ranking signal); its index is appended to `too_common`."""
    forms = []
    for i, e in enumerate(entries):
        fs = [e.get(doc_lang) or ""] + list(e.get(doc_lang + "_syn") or [])
        forms.append([term_stems(f) for f in fs if term_stems(f)])
    secs = list(sections if sections is not None else doc.sections)
    found, df = [], Counter()
    for s in secs:
        body = folded_stems(section_match_text(doc, s))
        title = folded_stems(s.title or "")
        bidx, tidx = stem_index(body), stem_index(title)
        scored = []
        for i, fs in enumerate(forms):
            c = sum(count_sequence(body, f, bidx) for f in fs)
            if c <= 0:
                continue
            t = sum(count_sequence(title, f, tidx) for f in fs)
            scored.append((-(1 if t else 0), -c, i))
            df[i] += 1
        found.append((s, scored))
    n_text = sum(1 for s in secs if any(doc.unit(u).text.strip() for u in s.unit_ids)) or len(secs) or 1
    common = {i for i, n in df.items() if n > max_share * n_text}
    if too_common is not None:
        too_common.extend(sorted(common))
    out, hits = {}, Counter()
    for s, scored in found:
        scored = sorted(x for x in scored if x[2] not in common)
        if not scored:
            continue
        chosen = [i for _, _, i in scored[:max_per_section]]
        out[s.id] = chosen
        for i in chosen:
            hits[i] += 1
    return out, dict(hits)


def glossary_header_lines(entries, idxs, max_chars=120):
    """Header lines for one section from its projected glossary entries: gl_fa / gl_en, each "term ; synonym ; ..."."""
    fa, en, seen = [], [], set()
    for i in idxs:
        e = entries[i]
        for lst, key in ((fa, "fa"), (en, "en")):
            for t in [e.get(key) or ""] + list(e.get(key + "_syn") or []):
                t = _clean_field(t, max_chars)
                if t and t.lower() not in seen:
                    seen.add(t.lower())
                    lst.append(t)
    lines = []
    if fa:
        lines.append("gl_fa: " + " ; ".join(fa))
    if en:
        lines.append("gl_en: " + " ; ".join(en))
    return lines


# ---------------------------------------------------------------------------------------------------------------
# L2: per-section input block
# ---------------------------------------------------------------------------------------------------------------

def section_block_text(doc, section, max_chars=700):
    """Section path + the start of its own text, cut at a word boundary (the L2 batch input of one section)."""
    path = " > ".join(t for t in doc.section_path(section.id) if t)
    body = "\n".join(doc.unit(u).text for u in section.unit_ids if doc.unit(u).text.strip())
    body = " ".join(body.split())
    if len(body) > max_chars:
        cut = body[:max_chars]
        sp = cut.rfind(" ")
        body = (cut[:sp] if sp > max_chars * 0.6 else cut) + " …"
    return path, body


def text_script(s):
    """'fa' | 'en' | None by letters (a Persian letter share >= 30% = fa, as normalize.detect_lang)."""
    fa, en = script_counts(s or "")
    if fa + en == 0:
        return None
    return "fa" if fa / (fa + en) >= 0.30 else "en"


def clean_term(s, max_chars=60, max_words=5):
    if isinstance(s, (list, tuple)):
        s = s[0] if s else ""
    if not isinstance(s, str):
        return ""
    s = normalize(s).strip().strip("\"'«»“”`.,;:!؟?()[]{}-–—").strip()
    s = " ".join(s.replace("|", "/").replace(";", ",").split())
    if not s or len(s) > max_chars or len(s.split()) > max_words:
        return ""
    return s
