"""TIMELINE questions: schema accessors, a Persian/English (period -> value) parser, and scoring.
No model calls here (the series judge rubric lives in eval/judge.py).

A timeline question asks for a metric across several reports ("how many people work at Divar and how
has that changed?"). The gold answer is a series with one value per period, each tied to the report
of that period and to the corpus units that state it.

Schema (a normal contracts.Question JSONL line; the series lives in Question.meta)
---------------------------------------------------------------------------------
    type: "timeline"              answer_type: "series"
    answer_fa / answer_en         short prose answer (the reference; also parsed by `validate`)
    answer_value / answer_unit    null
    doc_scope                     every report the series (and its restatements) uses
    gold_units {fa, en}           union of every period's gold_units and context_units
    meta.schema_version           "timeline-v1"
    meta.answer_series            [ordered by period, periods unique]
        period          "1398" (Jalali) or "2021" (Gregorian); the year the value describes
        calendar        "jalali" | "gregorian"
        value           number (or a short string for non-numeric facts)
        unit            neutral unit string, scale words included ("people", "million users", "percent")
        unit_fa         the unit as the Persian report writes it
        value_base      value x scale of unit (percent: the percentage itself); what scoring compares
        percent         true when the unit is a percentage
        definition_fa / definition_en   what exactly the report counts in this period
        doc             doc id (fa id, like doc_scope) of the report the value is taken from
        gold_units      {fa: [...], en: [...]}: units that state the value (each contains it);
                        en only where an English edition exists (corpora A/B)
        context_units   {fa: [...], en: [...]}: label or definition units needed to read the value (the heading
                        of a detached number, the sentence saying what is counted); they need not contain it
        quote           the authoring quote terms per language
    meta.comparable               false when the metric's definition or scope changes across periods
    meta.series_notes_fa / _en    definition changes, restatements, conflicts inside a report
    meta.comparability_cues       {fa: [...], en: [...]} extra phrases that count as flagging the change
    meta.reconciliation
        rule          "report_of_period"
        rule_text     THE RULE: the value published in the report OF a period is the gold value for that
                      period. When a later report restates that period with a different number, the
                      restated number is NOT gold: it is listed under `restatements` and described in
                      series_notes. A period without a report of its own in the corpus may be listed under
                      `supplementary` (taken from a later report); supplementary periods are optional.
        restatements  [{period, value, unit, value_base, percent, doc, gold_units, note}]
        supplementary [{period, value, unit, value_base, percent, doc, gold_units, note}]

Accessors below read these fields from top-level Question attributes when contracts.Question ever gains
them, else from Question.meta, so the data and scoring keep working either way.

Structured output (RunRecord)
-----------------------------
An approach that assembles a timeline may return, in `record["structured"]` (proposed RunRecord field, see
ragbench/INTEGRATION_NOTES.md) or today in `record["trace"]["structured"]`:
    {"series": [{"period", "value", "unit", "definition", "quote", "unit_id", "doc_id"}, ...],
     "comparable": bool (optional), "notes": str | [str] (optional), "trend": str (optional)}
`value` may be a number (scaled by `unit`, e.g. 18.5 + "million users") or a string ("18.5 میلیون").
When `series` is a non-empty list it is scored directly; otherwise the answer text is parsed.

Metrics (per record; all None for non-timeline questions)
---------------------------------------------------------
    tl_coverage            gold periods given a correct value / gold periods
    tl_precision           correct predicted periods / predicted periods (a supplementary period with a
                           correct value counts as correct; another period inside the asked range counts as
                           wrong; a period outside [first, last gold period] is not counted: tl_n_out_of_range)
    tl_f1                  harmonic mean of the two
    tl_series_correct      coverage 1, precision 1, and (comparable or the change was flagged), no false trend
    tl_comparability_flag  comparable=false only: the answer flags the definition change (structured
                           comparable=false, or a flag phrase in the notes / answer text)
    tl_false_trend         comparable=false only: the answer states a trend (grew, fell, doubled, رشد, ...)
                           without flagging the change
    tl_restated_used       share of gold periods answered with a later report's restated value
    tl_citation_correct    structured items with a unit_id whose unit states the value / such items
Value match: |pred - gold| <= rel_tol x |gold| (2%), or a rounding at the written precision when the written
number has >= 2 significant digits; a number without a scale word is also tried x the gold unit's scale.
"""
import math
import re
from collections import defaultdict
from dataclasses import asdict, dataclass

from ..normalize import fold, normalize_lines
from .metrics import _RE_REFS, _RE_SCALE_WORD, _scale_of, parse_unit

TIMELINE_TYPE = "timeline"
SERIES_ANSWER_TYPE = "series"
SCHEMA_VERSION = "timeline-v1"
RECONCILIATION_RULE = "report_of_period"
RECONCILIATION_RULE_TEXT = (
    "The value published in the report OF a period is the gold value for that period. When a later report "
    "restates that period with a different number, the restated number is not gold: it is listed under "
    "restatements and described in series_notes. A period with no report of its own in the corpus may be listed "
    "under supplementary (taken from a later report); supplementary periods are optional for coverage.")
TIMELINE_FIELDS = ("answer_series", "comparable", "series_notes_fa", "series_notes_en", "reconciliation")

# --------------------------------------------------------------------------
# schema accessors
# --------------------------------------------------------------------------


def _qget(question, key, default=None):
    if isinstance(question, dict):
        if key in question:
            return question[key]
        return (question.get("meta") or {}).get(key, default)
    v = getattr(question, key, None)
    if v is not None:
        return v
    return (getattr(question, "meta", None) or {}).get(key, default)


def is_timeline(question):
    return _qget(question, "type") == TIMELINE_TYPE or _qget(question, "answer_type") == SERIES_ANSWER_TYPE


def timeline_spec(question):
    """{answer_series, comparable, series_notes_fa, series_notes_en, reconciliation, comparability_cues}"""
    rec = _qget(question, "reconciliation") or {}
    return {
        "answer_series": list(_qget(question, "answer_series") or []),
        "comparable": _qget(question, "comparable", True),
        "series_notes_fa": _qget(question, "series_notes_fa") or "",
        "series_notes_en": _qget(question, "series_notes_en") or "",
        "reconciliation": {"rule": rec.get("rule", RECONCILIATION_RULE), "restatements": list(rec.get("restatements") or []),
                           "supplementary": list(rec.get("supplementary") or [])},
        "comparability_cues": _qget(question, "comparability_cues") or {},
    }


def gold_periods(spec):
    return [str(it["period"]) for it in spec["answer_series"]]


def period_key(period):
    """Sort key: the leading 4-digit year, then the rest ("1398", "1398-H1")."""
    m = re.match(r"\s*(\d{4})", str(period))
    return (int(m.group(1)) if m else 10 ** 6, str(period))


def item_base_value(item):
    """(value_base, percent) of a gold/restated/structured item: value x scale of its unit."""
    if item.get("value_base") is not None:
        return float(item["value_base"]), bool(item.get("percent"))
    val = item.get("value")
    unit = item.get("unit") or ""
    if isinstance(val, bool) or val is None:
        return None, False
    if isinstance(val, (int, float)):
        u = parse_unit(unit)
        return float(val) * (1.0 if u.percent else u.scale), bool(u.percent)
    ms = parse_numbers("%s %s" % (val, unit))
    if not ms:
        return None, False
    return ms[0].value, ms[0].percent


# --------------------------------------------------------------------------
# numbers
# --------------------------------------------------------------------------

@dataclass
class NumberMention:
    start: int
    end: int
    text: str
    raw: float                 # the number as written (decimals applied, no scale)
    value: float               # raw x scale
    scale: float = 1.0
    explicit_scale: bool = False
    percent: bool = False
    sig: int = 1               # significant digits written
    precision: float = 0.5     # half the last written digit, in value units
    plain_int: bool = False    # an integer written without separators, decimals, scale or percent
    kind: str = "number"       # number | composite | words

    def to_dict(self):
        return asdict(self)


_RE_AR_COMMA_THOUS = re.compile(r"(?<=\d)[،٬](?=\d{3}(?!\d))")
# dot-grouped thousands: two or more dots, each followed by exactly 3 digits ("4.318.701"; also Persian "۴٫۳۱۸٫۷۰۱",
# which normalize_lines turns into dots). One dot stays a decimal point ("4.318"). A sentence-final dot is not a group.
_RE_DOT_THOUS = re.compile(r"(?<![\d/])(?<!\d[.,])\d{1,3}(?:\.\d{3}){2,}(?!\d|[.,]\d)")
_RE_DATE = re.compile(r"(?<!\d)\d{1,4}/\d{1,2}/\d{1,4}(?!\d)")
_RE_NUM = re.compile(r"(?<![\d.,/])(?P<num>\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)(?:/(?P<frac>\d{1,3}))?(?![\d/])")
_RE_SUFFIX = re.compile(r"(K|k|M|B|bn|mn)(?![A-Za-z])")
_RE_PCT_AFTER = re.compile(r"[\s‌]*(?:%|درصد|percent\b|per[\s-]?cent\b|pct\b)", re.I)
_RE_PCT_BEFORE = re.compile(r"%[\s‌]?$")
_RE_AND = re.compile(r"[\s‌]*(?:و|and)[\s‌]*", re.I)
_RE_SCALE_AT = re.compile(r"[\s‌+]*(%s)" % _RE_SCALE_WORD.pattern, re.I)     # "40+ میلیون"
_RE_SCALE_LEAD = re.compile(r"(?P<sc>هزار|میلیون|میلیارد)[\s‌]*و[\s‌]*(?P<n>\d+)(?![\d.,/])")
_WORD_NUMS = {"یک": 1, "دو": 2, "سه": 3, "چهار": 4, "پنج": 5, "شش": 6, "هفت": 7, "هشت": 8, "نه": 9, "ده": 10,
              "بیست": 20, "سی": 30, "چهل": 40, "پنجاه": 50, "شصت": 60, "هفتاد": 70, "هشتاد": 80, "نود": 90,
              "صد": 100, "دویست": 200, "سیصد": 300, "پانصد": 500,
              "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
              "ten": 10, "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70,
              "eighty": 80, "ninety": 90, "hundred": 100}
_RE_WORD_NUM = re.compile(r"(?<![\w‌])(?P<w>%s)[\s‌]*(?P<sc>%s)" % (
    "|".join(sorted(_WORD_NUMS, key=len, reverse=True)), _RE_SCALE_WORD.pattern), re.I)


def prep_lines(text):
    """normalize_lines (Persian digits -> ASCII, ٫ -> '.', ٬ -> ','), citation markers removed, an Arabic comma
    between digit groups read as a thousands separator ("4،575"), dot-grouped thousands read as thousands
    ("4.318.701", "۴٫۳۱۸٫۷۰۱" -> 4,318,701), dates blanked. Line structure is kept."""
    out = []
    for line in normalize_lines(text or "").split("\n"):
        line = _RE_REFS.sub(" ", line)
        line = _RE_DOT_THOUS.sub(lambda m: m.group(0).replace(".", ","), line)
        prev = None
        while prev != line:
            prev = line
            line = _RE_AR_COMMA_THOUS.sub(",", line)
        line = _RE_DATE.sub(lambda m: " " * len(m.group(0)), line)
        out.append(line)
    return out


def _sig_digits(s):
    d = s.replace(",", "")
    if "." in d:
        return len(d.replace(".", "").lstrip("0")) or 1
    return len(d.lstrip("0").rstrip("0")) or 1


def _read_scales(line, pos):
    """Scale words (and attached K/M/B suffixes) starting at pos -> (scale, explicit, end)."""
    suf = _RE_SUFFIX.match(line, pos)
    if suf:
        return {"k": 1e3, "m": 1e6, "b": 1e9, "bn": 1e9, "mn": 1e6}[suf.group(1).lower()], True, suf.end()
    scale, explicit, end = 1.0, False, pos
    while True:
        m = _RE_SCALE_AT.match(line, end)
        if not m:
            break
        scale *= _scale_of(m.group(1))
        explicit = True
        end = m.end()
    return scale, explicit, end


def parse_numbers_line(line):
    """Every number mention in one prepared line, left to right."""
    mentions, taken = [], []

    def free(a, b):
        return not any(a < y and x < b for x, y in taken)

    # "هزار و 546" (a scale word that starts the number): 1546
    for m in _RE_SCALE_LEAD.finditer(line):
        if line[:m.start()].rstrip()[-1:].isdigit():
            continue
        base = {"هزار": 1e3, "میلیون": 1e6, "میلیارد": 1e9}[m.group("sc")]
        n = float(m.group("n"))
        mentions.append(NumberMention(m.start(), m.end(), m.group(0), base + n, base + n, 1.0, True, False,
                                      len(m.group("n")) + 1, 0.5, False, "composite"))
        taken.append((m.start(), m.end()))
    # "چهل میلیون", "forty million"
    for m in _RE_WORD_NUM.finditer(line):
        if not free(m.start(), m.end()):
            continue
        raw = float(_WORD_NUMS[m.group("w").lower()])
        scale, _, end = _read_scales(line, m.start("sc"))
        pct = bool(_RE_PCT_AFTER.match(line, end))
        mentions.append(NumberMention(m.start(), end, line[m.start():end], raw, raw * scale, scale, True, pct,
                                      _sig_digits(str(int(raw))), 0.5 * scale, False, "words"))
        taken.append((m.start(), end))
    pos = 0
    while True:
        m = _RE_NUM.search(line, pos)
        if not m:
            break
        pos = m.end()
        if not free(m.start(), m.end()):
            continue
        num, frac = m.group("num"), m.group("frac")
        digits = num.replace(",", "")
        end = m.end()
        text_num = digits
        if frac is not None:
            if "," not in num and "." not in num and len(digits) <= 3:
                text_num = digits + "." + frac          # Persian typography: 23/17 = 23.17
            else:
                end = m.start("frac") - 1                # 1398/99: keep 1398, drop the tail
        try:
            raw = float(text_num)
        except ValueError:
            continue
        decimals = len(text_num.split(".", 1)[1]) if "." in text_num else 0
        scale, explicit, s_end = _read_scales(line, end)
        pct_after = _RE_PCT_AFTER.match(line, s_end)
        pct = bool(pct_after) or bool(_RE_PCT_BEFORE.search(line[max(0, m.start() - 2):m.start()]))
        stop = pct_after.end() if pct_after else s_end
        value = raw * scale
        precision = 0.5 * 10 ** (-decimals) * scale
        sig = _sig_digits(text_num)
        kind = "number"
        # composite Persian numbers: "18 میلیون و 519 هزار", "18 هزار و 800"
        while explicit and not pct and scale >= 1e3:
            a = _RE_AND.match(line, stop)
            if not a:
                break
            m2 = _RE_NUM.match(line, a.end())
            if not m2 or m2.group("frac"):
                break
            raw2 = float(m2.group("num").replace(",", ""))
            sc2, ex2, e2 = _read_scales(line, m2.end())
            if sc2 >= scale or (not ex2 and raw2 >= scale):
                break
            value += raw2 * sc2
            precision = 0.5 * sc2
            sig = len(str(int(value / sc2)))
            stop = e2
            scale = sc2
            kind = "composite"
            if not ex2:
                break
        plain = (kind == "number" and not explicit and not pct and "," not in num and "." not in text_num
                 and frac is None)
        mentions.append(NumberMention(m.start(), stop, line[m.start():stop], raw, value, scale if kind == "number" else 1.0,
                                      explicit, pct, sig, precision, plain, kind))
        taken.append((m.start(), stop))
        pos = max(pos, stop)
    mentions.sort(key=lambda x: x.start)
    return mentions


def parse_numbers(text):
    """All number mentions of a text (line offsets are per line; use extract_series for pairing)."""
    out = []
    for line in prep_lines(text):
        out += parse_numbers_line(line)
    return out


# --------------------------------------------------------------------------
# periods and pairing
# --------------------------------------------------------------------------

_RE_YEAR_CTX = re.compile(r"(?:سال(?:[\s‌]*های)?|سالهای|year|years|fy|in|during|since|from|to|until|by|end of|"
                          r"start of|در|از|تا|طی|پایان|ابتدای|انتهای|اواخر|اوایل|و|and)[\s‌:,،]*$", re.I)
_RE_SOURCE_BEFORE = re.compile(r"(?:گزارش(?:[\s‌]*(?:سالانه|عملکرد|سال|پایداری|اقتصادی|نهایی)[ًٔ-ٟ]?"
                               r"[\s‌]*(?:ی|ه)?)*|report|edition)[\s‌:]*$", re.I)
_RE_SOURCE_AFTER = re.compile(r"[\s‌]*(?:annual|performance|sustainability|year[\s-]?end)?[\s‌]*(?:report|edition)\b", re.I)
_RE_LIST_AFTER = re.compile(r"[\s‌]*(?:[:：|)\]]|[-–—→]\s|=)")
_RE_VALUE_BEFORE_YEAR = re.compile(r"[\s‌]*(?:in|during|در|سال|for)[\s‌]*(?:سال[\s‌]*)?\d{4}(?!\d)", re.I)
_RE_UNIT_AFTER = re.compile(r"[\s‌]*(?:نفر|کاربر|آگهی|برنامه|بازی|تراکنش|خودرو|مشتری|نمایندگی|عدد|مورد|بار|"
                            r"people|persons?|users?|employees|staff|ads?\b|apps?\b|games?\b|transactions?|cars?\b|"
                            r"customers?|agencies|units?\b|times\b)", re.I)
_RE_DURATION_AFTER = re.compile(r"[\s‌]*(?:سال(?!\s*\d)|ساله|ماه|روز|هفته|ساعت|دقیقه|years?\b|months?\b|days?\b|weeks?\b|hours?\b|minutes?\b)", re.I)
# the same duration words by kind: a value followed by one of them is dropped unless the gold unit is that duration
_DURATION_KINDS = (("year", r"سال(?!\s*\d)|ساله|years?\b"), ("month", r"ماه|months?\b"), ("week", r"هفته|weeks?\b"),
                   ("day", r"روز|days?\b"), ("hour", r"ساعت|hours?\b"), ("minute", r"دقیقه|minutes?\b"))
_RE_DURATION_KIND = re.compile(r"[\s‌]*(?:%s)" % "|".join("(?P<%s>%s)" % kp for kp in _DURATION_KINDS), re.I)


def duration_kind_after(text):
    """'hour' | 'day' | ... when text starts (after spaces) with a duration word, else None."""
    m = _RE_DURATION_KIND.match(text or "")
    return m.lastgroup if m else None


def unit_duration_kind(unit):
    """The duration a unit string counts ("hours", "ساعت", "million hours" -> 'hour'), else None. Only a unit whose
    head word (after scale words) is a duration counts; "rides per day" is not a duration."""
    t = normalize_lines(str(unit or "")).strip()
    t = re.sub(r"^(?:[\s‌]*(?:%s))+" % _RE_SCALE_WORD.pattern, "", t, flags=re.I)
    m = _RE_DURATION_KIND.match(t)
    if not m or t[m.end():m.end() + 1].isalpha():
        return None
    return m.lastgroup


def series_duration_kinds(items):
    """Duration kinds of a gold series' units (unit and unit_fa of every item), e.g. {'hour'}; empty for other units."""
    kinds = set()
    for it in items or []:
        for key in ("unit", "unit_fa"):
            k = unit_duration_kind(it.get(key)) if isinstance(it, dict) else None
            if k:
                kinds.add(k)
    return kinds
_RE_TWO_DIGIT_YEAR = re.compile(r"(?:سال|سالهای|سال‌های)[\s‌]*(?P<y>\d{2})(?![\d.,/%])")
_RE_SEGMENT = re.compile(r"(?<=[.!?؟;؛])\s+")
_RE_RESPECTIVELY = re.compile(r"به[\s‌]*ترتیب|respectively", re.I)


def _year_value(m, known_years=()):
    """The year a plain integer can denote. Jalali 1300-1499 overlaps real values (1,340 people), so when the
    expected periods are known a year must lie within 15 years of them."""
    if not m.plain_int:
        return None
    y = int(m.raw)
    if not (1300 <= y <= 1499 or 1900 <= y <= 2099):
        return None
    if known_years:
        near = [k for k in known_years if (k < 1900) == (y < 1900)]
        if not near or not (min(near) - 15 <= y <= max(near) + 15):
            return None
    return y


def classify_mentions(line, mentions, known_periods=(), duration_kinds=()):
    """Split a line's number mentions into periods, report-year sources and values.
    A value followed by a duration word ("3 سال", "12 hours") is dropped, unless the gold unit is that duration
    (duration_kinds, from series_duration_kinds): then "1402: 48 ساعت" keeps 48. A year-like integer followed by a
    duration word is never a period.
    -> (periods [(mention, "1398")], sources [(mention, "1400")], values [mention])"""
    known = {str(p) for p in known_periods}
    known_years = [int(k[:4]) for k in known if re.match(r"\d{4}", k)]
    periods, sources, values = [], [], []
    for m in mentions:
        y = _year_value(m, known_years)
        before, after = line[:m.start], line[m.end:]
        if (y is not None and not _RE_UNIT_AFTER.match(after) and not _RE_DURATION_AFTER.match(after)
                and not (_RE_VALUE_BEFORE_YEAR.match(after) and str(y) not in known)):
            src = bool(_RE_SOURCE_BEFORE.search(before)) or bool(_RE_SOURCE_AFTER.match(after))
            ctx = (bool(_RE_YEAR_CTX.search(before)) or bool(_RE_LIST_AFTER.match(after)) or str(y) in known)
            if src and not re.search(r"(?:سال|year|in|در)[\s‌]*$", before, re.I):
                sources.append((m, str(y)))
                continue
            if ctx:
                periods.append((m, str(y)))
                continue
        values.append(m)
    # two-digit Jalali years after "سال": "سال 98" -> 1398
    for tm in _RE_TWO_DIGIT_YEAR.finditer(line):
        yy = int(tm.group("y"))
        full = str(1300 + yy) if yy >= 50 else str(1400 + yy)
        hit = [v for v in values if v.start == tm.start("y") and v.plain_int]
        if hit:
            values.remove(hit[0])
            periods.append((hit[0], full))
    periods.sort(key=lambda pm: pm[0].start)
    keep = set(duration_kinds or ())
    values = [v for v in values if duration_kind_after(line[v.end:]) in keep | {None}]
    return periods, sources, values


def _pair_segment(seg, periods, values, max_gap=160):
    """[(period, mention)] inside one segment (see module docstring for the rules)."""
    if not periods or not values:
        return []
    k = len(periods)
    # "1397 و 1398 به ترتیب 350 و 795 نفر" / "in 2021 and 2022: 5 and 7": every period before every value
    if k >= 2 and len(values) == k and periods[-1][0].end <= values[0].start:
        return [(p, v) for (_, p), v in zip(periods, values)]

    def assign(direction):
        out, dist = [], 0
        for v in values:
            if direction == "YV":
                cands = [pm for pm in periods if pm[0].end <= v.start]
                pm = cands[-1] if cands else None
                d = (v.start - pm[0].end) if pm else None
            else:
                cands = [pm for pm in periods if pm[0].start >= v.end]
                pm = cands[0] if cands else None
                d = (pm[0].start - v.end) if pm else None
            if pm is not None and d <= max_gap:
                out.append([pm, v, d])
                dist += d
        return len(out), -dist, out

    a, b = assign("YV"), assign("VY")
    best, other = (a, b) if (a[0], a[1]) >= (b[0], b[1]) else (b, a)
    pairs = best[2]
    # mixed orientation ("250 هزار ... (1401) و در سال 1402، 70 هزار"): a value the chosen direction left
    # unpaired may still pair the other way with a period that got nothing
    used_v = {id(pr[1]) for pr in pairs}
    used_p = {pr[0][0].start for pr in pairs}
    for pr in other[2]:
        if id(pr[1]) not in used_v and pr[0][0].start not in used_p:
            pairs.append(pr)
            used_v.add(id(pr[1]))
            used_p.add(pr[0][0].start)
    # rebalance: a period left without a value takes a closer value from a period that got several
    by_p = defaultdict(list)
    for pr in pairs:
        by_p[pr[0][0].start].append(pr)
    for pm in periods:
        if by_p.get(pm[0].start):
            continue
        best_move = None
        for plist in by_p.values():
            if len(plist) < 2:
                continue
            for pr in plist:
                v = pr[1]
                d_new = (v.start - pm[0].end) if pm[0].end <= v.start else (pm[0].start - v.end)
                if d_new < pr[2] and (best_move is None or d_new < best_move[1]):
                    best_move = (pr, d_new, plist)
        if best_move:
            pr, d_new, plist = best_move
            plist.remove(pr)
            pr[0], pr[2] = pm, d_new
            by_p[pm[0].start].append(pr)
    out = [(pr[0][1], pr[1]) for plist in by_p.values() for pr in plist]
    out.sort(key=lambda x: x[1].start)
    return out


def extract_series(text, known_periods=(), want_percent=None, duration_kinds=()):
    """(period -> [NumberMention, ...] in text order) from a prose, list or table answer.
    want_percent: True keeps only percentages, False drops them, None keeps everything.
    duration_kinds: the gold unit's duration kinds (series_duration_kinds); values in that duration are kept."""
    series = defaultdict(list)
    for line in prep_lines(text):
        for seg in _RE_SEGMENT.split(line):
            if not seg.strip():
                continue
            mentions = parse_numbers_line(seg)
            periods, sources, values = classify_mentions(seg, mentions, known_periods, duration_kinds)
            if not periods and sources:
                periods = sources
            if want_percent is True:
                values = [v for v in values if v.percent]
            elif want_percent is False:
                values = [v for v in values if not v.percent]
            for period, v in _pair_segment(seg, periods, values):
                series[period].append(v)
    return dict(series)


# --------------------------------------------------------------------------
# value matching
# --------------------------------------------------------------------------

@dataclass
class SeriesMatchConfig:
    rel_tol: float = 0.02           # |pred - gold| <= rel_tol * |gold|
    allow_rounding: bool = True     # "1.3 thousand" matches 1,340 (rounding at the written precision) ...
    min_sig_digits: int = 2         # ... when the written number has >= 2 significant digits
    scale_fallback: bool = True     # "35" in a table headed "million" is also tried x the gold unit's scale

    def to_dict(self):
        return asdict(self)


DEFAULT_SERIES_MATCH = SeriesMatchConfig()


def _close(a, precision, sig, gold, cfg):
    if abs(a - gold) <= max(1e-9, cfg.rel_tol * abs(gold)):
        return True
    return bool(cfg.allow_rounding and sig >= cfg.min_sig_digits and abs(a - gold) <= precision * (1 + 1e-9))


def mention_matches(m, gold_base, gold_percent=False, gold_scale=1.0, cfg=DEFAULT_SERIES_MATCH, percent_strict=True):
    """percent_strict=False lets a bare number match a percentage (an extracted unit "20" whose "درصد" sits in a
    neighbouring heading); scoring answers keeps it strict."""
    if gold_base is None or m is None:
        return False
    if percent_strict and bool(gold_percent) != bool(m.percent):
        return False
    if _close(m.value, m.precision, m.sig, gold_base, cfg):
        return True
    if cfg.scale_fallback and not m.explicit_scale and gold_scale != 1.0:
        return _close(m.value * gold_scale, m.precision * gold_scale, m.sig, gold_base, cfg)
    return False


def _gold_scale(item):
    u = parse_unit(item.get("unit") or "")
    return 1.0 if u.percent else u.scale


def written_precision(item):
    """Half the last digit the item's value is written with, in base units (18.5 million -> 50,000; 350 -> 0.5)."""
    v = item.get("value")
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        s = str(int(v)) if float(v).is_integer() else repr(float(v))
        dec = len(s.split(".", 1)[1]) if "." in s else 0
        return 0.5 * 10 ** (-dec) * _gold_scale(item)
    return 0.5


def unit_text_contains_value(unit_text, item, rel_tol=0.0, percent_strict=False):
    """True when a unit's text states the item's value: some number in the text equals the value at the precision the
    value is written with (the gold 18.5 million is in "18 میلیون و 519 هزار"; 351 is not in "350 نفر"), or within
    rel_tol. Used by validation (rel_tol 0) and citation checks. A number written without its scale word ("13/3" next to
    a separate "میلیون" unit) counts, as does a percentage without its sign unless percent_strict."""
    base, pct = item_base_value(item)
    if base is None:
        v = item.get("value")
        return bool(v) and fold(str(v)) in fold(unit_text or "")
    prec = written_precision(item)
    tol = max(prec, rel_tol * abs(base)) * (1 + 1e-9) + 1e-9
    scale = _gold_scale(item)
    for m in parse_numbers(unit_text):
        if percent_strict and bool(pct) != bool(m.percent):
            continue
        cands = [m.value] + ([m.value * scale] if (not m.explicit_scale and scale != 1.0) else [])
        if any(abs(c - base) <= tol for c in cands):
            return True
    return False


# --------------------------------------------------------------------------
# comparability and trend language
# --------------------------------------------------------------------------

FLAG_PATTERNS = [
    r"تعریف (?:\S+ )?(?:متفاوت|تغییر|عوض|دیگر)", r"تغییر (?:در )?تعریف", r"قابل مقایسه (?:\S+ )?نیست", r"مقایسه ?پذیر نیست",
    r"قیاس ?پذیر نیست", r"یکسان نیست", r"مبنای (?:متفاوت|دیگری)", r"(?:مبنا|دامنه|تعریف|روش)(?: \S+)? متفاوت",
    r"دامنه (?:\S+ )?(?:تغییر|گسترده ?تر|وسیع ?تر)", r"تجمعی", r"انباشته", r"روش شمارش", r"به جای (?:کاربر|نفر|شمار)",
    r"مستقیما (?:قابل )?مقایسه", r"نباید مقایسه", r"احتیاط",
    r"definitions? (?:chang|differ|is different|are different|shift)", r"defin(?:ed|es) differently",
    r"(?:changed|different|new|narrower|broader|wider) definition", r"not (?:directly |strictly |fully )?comparable",
    r"comparability", r"like[- ]for[- ]like", r"different (?:basis|scope|measure|metric|methodology|count|thing|population)",
    r"scope (?:changed|differs|is different|widened|narrowed)", r"(?:broader|wider|narrower) scope", r"cumulative",
    r"caveat", r"(?:counts?|measures?) (?:a )?different", r"instead of (?:users|people|staff)", r"rather than (?:users|people|staff)",
]
TREND_PATTERNS = [
    r"رشد", r"افزایش", r"کاهش", r"افت", r"برابر شد", r"دو برابر", r"سه برابر", r"روند", r"صعودی", r"نزولی",
    r"بیشتر شد", r"کمتر شد", r"جهش", r"سقوط", r"(?:به|تا) .{0,40}رسید",
    r"\bgr[eo]w", r"\bgrowth", r"\bincreas", r"\bdecreas", r"\bdeclin", r"\bdoubl", r"\btripl", r"\brose\b", r"\brise[sn]?\b",
    r"\bfell\b", r"\bfall(?:s|en|ing)?\b", r"\bdrop", r"\bjump", r"\bsurg", r"\bshr[ai]nk", r"\btrend", r"\breach(?:ed|es)\b",
    r"\bclimb", r"\bplung", r"\bcollaps", r"\bexpand", r"\bup from\b", r"\bdown from\b",
]
_RE_FLAG = re.compile("|".join("(?:%s)" % p for p in FLAG_PATTERNS), re.I)
_RE_TREND = re.compile("|".join("(?:%s)" % p for p in TREND_PATTERNS), re.I)


def _texts(x):
    if x is None:
        return []
    if isinstance(x, str):
        return [x]
    if isinstance(x, (list, tuple)):
        return [str(t) for t in x if t]
    if isinstance(x, dict):
        return [str(t) for t in x.values() if t]
    return [str(x)]


def flags_definition_change(text, cues=None, structured=None):
    """True when the answer (or structured output) says the figures are not directly comparable."""
    if isinstance(structured, dict):
        if structured.get("comparable") is False:
            return True
        for key in ("notes", "definition_changes", "caveats", "comparability"):
            for t in _texts(structured.get(key)):
                if _RE_FLAG.search(fold(t)) or _RE_FLAG.search(t):
                    return True
    f = fold(text or "")
    if _RE_FLAG.search(f):
        return True
    for lang_cues in (cues or {}).values():
        for c in lang_cues or []:
            if c and fold(c) in f:
                return True
    return False


def states_trend(text, structured=None):
    if isinstance(structured, dict) and _texts(structured.get("trend")):
        return True
    f = fold(text or "")
    return bool(_RE_TREND.search(f))


# --------------------------------------------------------------------------
# structured output
# --------------------------------------------------------------------------

def structured_of(record):
    s = record.get("structured") if isinstance(record, dict) else None
    if not isinstance(s, dict):
        tr = record.get("trace") if isinstance(record, dict) else None
        s = tr.get("structured") if isinstance(tr, dict) else None
    return s if isinstance(s, dict) else None


def _norm_period(p, known=()):
    t = normalize_lines(str(p if p is not None else "")).strip()
    m = re.search(r"\d{4}", t)
    if m:
        return m.group(0)
    m = re.fullmatch(r"'?(\d{2})", t)
    if m:
        for k in known:
            if str(k).endswith(m.group(1)):
                return str(k)
    return t


def structured_series(structured, known_periods=()):
    """(period -> [NumberMention]) and the raw items (with normalised period) from structured output."""
    series, items = defaultdict(list), []
    for it in structured.get("series") or []:
        if not isinstance(it, dict):
            continue
        period = _norm_period(it.get("period"), known_periods)
        val, unit = it.get("value"), it.get("unit") or ""
        m = None
        if isinstance(val, (int, float)) and not isinstance(val, bool):
            u = parse_unit(unit)
            scale = 1.0 if u.percent else u.scale
            s = repr(float(val)).rstrip("0").rstrip(".") if float(val) != int(val) else str(int(val))
            dec = len(s.split(".", 1)[1]) if "." in s else 0
            m = NumberMention(0, 0, "%s %s" % (val, unit), float(val), float(val) * scale, scale, scale != 1.0,
                              bool(u.percent), _sig_digits(s), 0.5 * 10 ** (-dec) * scale, False, "structured")
        elif val is not None:
            ms = parse_numbers("%s %s" % (val, unit))
            m = ms[0] if ms else None
        entry = dict(it, period=period, _mention=m)
        items.append(entry)
        if m is not None and period:
            series[period].append(m)
    return dict(series), items


# --------------------------------------------------------------------------
# scoring
# --------------------------------------------------------------------------

TIMELINE_METRICS = ["tl_coverage", "tl_precision", "tl_f1", "tl_series_correct", "tl_comparability_flag",
                    "tl_false_trend", "tl_restated_used", "tl_citation_correct"]
TIMELINE_JUDGE_METRICS = ["tl_judge_coverage", "tl_judge_flag", "tl_judge_false_trend"]


def empty_scores():
    out = {k: None for k in TIMELINE_METRICS}
    out.update(tl_source=None, tl_scoring=None, tl_n_gold_periods=None, tl_n_pred_periods=None, tl_n_out_of_range=None, tl_n_citations=None,
               tl_comparable=None, tl_period_hits=None)
    return out


TIMELINE_SCORING_MODES = ("auto", "text", "structured")
SECONDARY_METRICS = ("tl_coverage", "tl_precision", "tl_f1", "tl_series_correct", "tl_comparability_flag", "tl_false_trend")


def score_series(question, record, unit_resolver=None, cfg=DEFAULT_SERIES_MATCH, abstained=False, scoring="auto"):
    """tl_* metrics for one RunRecord dict against a timeline Question (see module docstring).

    scoring: "auto" (default, the original rule: a record's structured series when present, else its answer text),
    "text" (the answer text only, for every approach alike: structured output is ignored for the values, the
    definition-change flag and the trend; no citation score) or "structured" (the same as auto; named for reports).
    The fair headline comparison (2026-09-16) uses "text": every approach is then scored from what it answered."""
    if scoring not in TIMELINE_SCORING_MODES:
        raise ValueError("timeline scoring must be one of %s" % (TIMELINE_SCORING_MODES,))
    spec = timeline_spec(question)
    gold = spec["answer_series"]
    out = empty_scores()
    if not gold:
        return out
    rec = spec["reconciliation"]
    periods = gold_periods(spec)
    known = periods + [str(r.get("period")) for r in rec["restatements"] + rec["supplementary"]]
    want_percent = all(item_base_value(it)[1] for it in gold) if gold else None
    if not want_percent and any(item_base_value(it)[1] for it in gold):
        want_percent = None
    text = record.get("answer_text") or ""
    structured = structured_of(record) if scoring != "text" else None
    items = []
    if structured and isinstance(structured.get("series"), list) and structured["series"]:
        pred, items = structured_series(structured, known)
        source = "structured"
    else:
        durations = series_duration_kinds(gold + rec["restatements"] + rec["supplementary"])
        pred = {} if (abstained or record.get("error")) else extract_series(text, known, want_percent, durations)
        source = "text"
    by_period = {str(it["period"]): it for it in gold}
    hits, restated = {}, 0
    for p, it in by_period.items():
        base, pct = item_base_value(it)
        cands = pred.get(p) or []
        first = cands[0] if cands else None
        ok = False
        if base is not None:
            ok = mention_matches(first, base, pct, _gold_scale(it), cfg)
        elif first is None and it.get("value") is not None and source == "text":
            ok = False
        hits[p] = bool(ok)
        if first is not None and not ok:
            for r in rec["restatements"]:
                if str(r.get("period")) == p:
                    rb, rp = item_base_value(r)
                    if mention_matches(first, rb, rp, _gold_scale(r), cfg):
                        restated += 1
                        break
    supp = defaultdict(list)
    for s in rec["supplementary"]:
        supp[str(s.get("period"))].append(s)
    # precision counts predicted periods inside the asked range [first, last gold period] plus supplementary ones;
    # a period outside the range was not asked for and is neither right nor wrong (tl_n_out_of_range)
    lo, hi = period_key(periods[0])[0], period_key(periods[-1])[0]
    counted = [p for p, c in pred.items() if c and (p in supp or p in by_period or lo <= period_key(p)[0] <= hi)]
    n_pred = len(counted)
    out["tl_n_out_of_range"] = len([p for p, c in pred.items() if c]) - n_pred
    n_ok_pred = sum(1 for p in periods if hits.get(p))
    for p in counted:
        c = pred[p]
        if p in supp and any(mention_matches(c[0], *item_base_value(s), _gold_scale(s), cfg) for s in supp[p]):
            n_ok_pred += 1
    coverage = sum(1 for v in hits.values() if v) / float(len(periods))
    precision = (n_ok_pred / float(n_pred)) if n_pred else None
    f1 = (2 * coverage * precision / (coverage + precision)) if precision and coverage else (0.0 if n_pred or coverage == 0 else None)
    comparable = spec["comparable"] is not False
    flag = flags_definition_change(text, spec["comparability_cues"], structured)
    trend = states_trend(text, structured)
    out.update(tl_source=source, tl_scoring=scoring, tl_n_gold_periods=len(periods), tl_n_pred_periods=n_pred,
               tl_comparable=comparable,
               tl_coverage=round(coverage, 6), tl_precision=None if precision is None else round(precision, 6),
               tl_f1=None if f1 is None else round(f1, 6), tl_restated_used=round(restated / float(len(periods)), 6),
               tl_period_hits=hits)
    if not comparable:
        out["tl_comparability_flag"] = float(flag)
        out["tl_false_trend"] = float(trend and not flag)
    false_trend = bool(out["tl_false_trend"])
    out["tl_series_correct"] = float(coverage == 1.0 and precision == 1.0 and (comparable or flag) and not false_trend
                                     and not record.get("error"))
    # citations: a structured item's unit states its value
    checked = correct = 0
    for it in items:
        uid = it.get("unit_id")
        m = it.get("_mention")
        if not uid or m is None:
            continue
        g = by_period.get(it["period"])
        gold_units = set()
        if g:
            for l_units in (g.get("gold_units") or {}).values():
                gold_units.update(l_units or [])
        if uid in gold_units and g is not None and mention_matches(m, *item_base_value(g), _gold_scale(g), cfg):
            checked += 1
            correct += 1
            continue
        utext = unit_resolver(uid) if unit_resolver else None
        if utext is None:
            continue
        checked += 1
        probe = {"value": m.raw, "unit": it.get("unit") or "", "value_base": m.value, "percent": m.percent}
        if unit_text_contains_value(utext, probe, rel_tol=cfg.rel_tol):
            correct += 1
    if checked:
        out["tl_citation_correct"] = round(correct / float(checked), 6)
        out["tl_n_citations"] = checked
    return out


class CorpusUnitResolver:
    """unit_id -> unit text, loading a corpus set only when a citation needs it."""

    def __init__(self, cfg=None):
        self.cfg = cfg
        self._sets = {}

    def __call__(self, unit_id):
        try:
            from ..corpus.model import doc_lang, split_id
            from ..corpus.registry import load_set, set_name_for
            doc_id = split_id(unit_id)[0]
            name = set_name_for(doc_id.split(".", 1)[0], doc_lang(doc_id))
            if name not in self._sets:
                self._sets[name] = load_set(name, self.cfg)
            cs = self._sets[name]
            if not cs.has_doc(doc_id):
                return None
            d = cs.doc(doc_id)
            return d.unit(unit_id).text if d.has_unit(unit_id) else None
        except Exception:  # noqa: BLE001
            return None


# --------------------------------------------------------------------------
# aggregation and report
# --------------------------------------------------------------------------

def _mean(vals):
    v = [float(x) for x in vals if x is not None and not (isinstance(x, float) and math.isnan(x))]
    return (sum(v) / len(v)) if v else None


def timeline_rows(scored, setting_order=("fa-fa", "en-en", "fa-en", "en-fa")):
    """One row per split x variant x corpus x setting (plus setting ALL) over timeline records only."""
    groups = defaultdict(list)
    for r in scored:
        if r.get("answer_type") != SERIES_ANSWER_TYPE:
            continue
        sp = r.get("split") or "unknown"
        groups[(sp, r["approach"], r["corpus"], r["setting"])].append(r)
        groups[(sp, r["approach"], r["corpus"], "ALL")].append(r)
    rows = []
    for (split, approach, corpus, setting), rs in groups.items():
        nc = [r for r in rs if r.get("tl_comparable") is False]
        row = {"split": split, "approach": approach, "approach_base": rs[0].get("approach_base"), "corpus": corpus,
               "setting": setting, "n": len(rs), "n_questions": len({r["question_id"] for r in rs}),
               "n_non_comparable": len(nc), "n_structured": sum(1 for r in rs if r.get("tl_source") == "structured"),
               "n_errors": sum(1 for r in rs if r.get("errored"))}
        for k in TIMELINE_METRICS + TIMELINE_JUDGE_METRICS:
            row[k] = _mean(r.get(k) for r in rs)
        row["timeline_scoring"] = ",".join(sorted({str(r.get("tl_scoring")) for r in rs if r.get("tl_scoring")})) or None
        # secondary: the structured-output scores (records with a structured series, text scoring mode only)
        row["n_structured_scored"] = sum(1 for r in rs if r.get("tls_source") == "structured")
        for k in SECONDARY_METRICS:
            row["tls_" + k[3:]] = _mean(r.get("tls_" + k[3:]) for r in rs if r.get("tls_source") == "structured")
        row["acc_judge"] = _mean(r.get("acc_judge") for r in rs)
        row["n_judged"] = sum(1 for r in rs if r.get("acc_judge") is not None)
        row["ev_hit"] = _mean(r.get("ev_hit") for r in rs)
        row["ev_recall"] = _mean(r.get("ev_recall") for r in rs)
        # cost per timeline record (errored records excluded), as for every other result
        ok = [r for r in rs if not r.get("errored")]
        for k in ("llm_calls", "embed_calls", "prompt_tokens", "completion_tokens", "evidence_chars", "latency_ms"):
            row[k + "_mean"] = _mean(r.get(k) for r in ok)
        row["llm_calls_max"] = max([r.get("llm_calls") or 0 for r in ok], default=None)
        row["latency_p50_ms"] = _percentile([r.get("latency_ms") for r in ok], 50)
        row["latency_p95_ms"] = _percentile([r.get("latency_ms") for r in ok], 95)
        row["latency_noload_p50_ms"] = _percentile([r.get("latency_noload_ms") for r in ok], 50)
        rows.append(row)
    order = ["ALL"] + list(setting_order)
    rows.sort(key=lambda r: (r["split"], r["corpus"], order.index(r["setting"]) if r["setting"] in order else 9,
                             r["approach"]))
    return rows


def _percentile(vals, q):
    """Linear interpolation between closest ranks (numpy's default method)."""
    v = sorted(float(x) for x in vals if x is not None)
    if not v:
        return None
    pos = (len(v) - 1) * q / 100.0
    lo = int(math.floor(pos))
    hi = min(lo + 1, len(v) - 1)
    return v[lo] + (v[hi] - v[lo]) * (pos - lo)


def _p(x):
    return "–" if x is None else "%.1f" % (100.0 * x)


def render_timeline_md(rows, split, corpus):
    """Markdown lines for one split x corpus (empty list when it has no timeline records)."""
    sel = [r for r in rows if r["split"] == split and r["corpus"] == corpus]
    if not sel:
        return []
    modes = sorted({m for r in sel for m in str(r.get("timeline_scoring") or "auto").split(",")})
    text_mode = modes == ["text"]
    L = ["### Timeline questions (series answers)", "",
         "Coverage = gold periods with a correct value; precision = correct predicted periods; flag and false trend "
         "only over non-comparable series; citation = structured items whose cited unit states the value. "
         "Match: %s." % DEFAULT_SERIES_MATCH.to_dict(), "",
         ("Scoring: **answer text for every approach** (structured output ignored); the last columns score the "
          "structured series of the approaches that return one (secondary)." if text_mode else
          "Scoring: %s (a record's structured series when present, else its answer text)." % ", ".join(modes)), ""]
    hdr = ["Approach", "Setting", "n (non-comp.)", "Coverage %", "Precision %", "F1 %", "Series correct %",
           "Flags change %", "False trend %", "Restated used %", "Citation ok %", "Judge acc % (n)", "Structured",
           "Calls", "Prompt/compl. tok", "Latency p50/p95 s"]
    if text_mode:
        hdr += ["Struct. coverage %", "Struct. correct %", "Struct. flags %", "Struct. false trend %"]
    out = ["| " + " | ".join(hdr) + " |", "|" + "|".join("---" for _ in hdr) + "|"]

    def secs(ms):
        return "–" if ms is None else "%.1f" % (ms / 1000.0)
    for r in sel:
        out.append("| " + " | ".join(str(c) for c in [
            r["approach"], r["setting"], "%d (%d)" % (r["n"], r["n_non_comparable"]), _p(r["tl_coverage"]),
            _p(r["tl_precision"]), _p(r["tl_f1"]), _p(r["tl_series_correct"]), _p(r["tl_comparability_flag"]),
            _p(r["tl_false_trend"]), _p(r["tl_restated_used"]), _p(r["tl_citation_correct"]),
            "%s (%d)" % (_p(r["acc_judge"]), r["n_judged"]), r["n_structured"],
            "–" if r["llm_calls_mean"] is None else "%.2f" % r["llm_calls_mean"],
            "–" if r["prompt_tokens_mean"] is None else "%.0f / %.0f" % (r["prompt_tokens_mean"], r["completion_tokens_mean"] or 0),
            "%s / %s" % (secs(r["latency_p50_ms"]), secs(r["latency_p95_ms"]))] +
            ([_p(r.get("tls_coverage")), _p(r.get("tls_series_correct")), _p(r.get("tls_comparability_flag")),
              _p(r.get("tls_false_trend"))] if text_mode else [])) + " |")
    return L + out + [""]
