"""The typed normaliser: one L0 regex fact -> the typed columns of `facts`. No model, no I/O, pure functions.

This is what makes the knowledge base a database instead of a text table. The input is one dict from
`approaches/ours/lite.py::extract_facts` (`metric | value | unit | period`, `unit_id`, `score`, `src`, `label_from`),
plus the text of the unit it came from and the document's own years. The output is a dict whose keys are exactly the
typed columns of `facts` (`schema.FACT_COLUMNS` minus `doc_id`, `section_id`, `unit_id`, `ord`).

    value_num     the primary reading as a float WITH the scale applied, the scale taken from the value token AND
                  from the unit field ("4.4" + "میلیون تومان" -> 4400000.0; "47/8" -> 47.8; "2،788،550" -> 2788550.0).
                  Readings come from `agentic/numbers.py::find_numbers`, so the database and the verbatim guard read
                  a token the same way. `value_hi` is the upper end of a range ("150 تا 550"). `alt_readings` is the
                  JSON list of the token's other readings (the as-written reading of a group-reversed glitch token,
                  the unscaled reading), i.e. every value the guard would also accept.
    unit_norm     an ENGLISH canonical unit, whatever the language of the report: percent, toman, rial, usd, person,
                  tonne, kwh, mw, metre, hectare, hour, year, times, unit, point, litre, barrel ... A Persian and an
                  English edition of the same table therefore compare in SQL without a translation step. Scale words
                  are NOT part of unit_norm: they are applied to value_num, so value_num is always in base units of
                  unit_norm ("150 هزار تن" -> 150000.0 tonne).
    unit_kind     people, count, money, percent, mass, energy, length, area, time, ratio, other. The kind of the
                  FIRST recognised non-scale token of the unit field ("kwh/t" is energy, "میلیون تومان" is money).
                  No unit field at all -> count (a bare number is a count); a unit that is not in the vocabulary ->
                  other. Power (mw, kw) is filed under energy and volume (litre, barrel) under other; the kind list
                  is the spec's.
    currency      toman, rial, usd, eur, gbp, aed, cny, try, rub, inr. `value_cmp` is the rial equivalent for toman
                  (x10) and rial (x1) and NULL otherwise, so a toman restatement and a rial one still compare.
    year/year_to  the period's START year and its end (year_to is NULL unless the period is a range), so a year
                  filter reads `year IS NULL OR (year <= :hi AND COALESCE(year_to, year) >= :lo)`.
    period_kind   year, half, quarter, month, date, range.  calendar: jalali for 13xx/14xx, gregorian for 19xx/20xx.
    year_source   "own" when the year came from the fact's own clause or its table column (or from a year in its
                  quote), "doc" when it was inherited from the document's years, NULL when there is none. The year
                  itself comes from `agentic/reconcile.py::parse_period`, so the facts table and the v1 reconciler
                  date a fact identically.
    def_class     `reconcile.DEF_CLASSES` over label + quote, joined by `reconcile.definition_class`: active,
                  registered, total, accounts, fulltime, parttime, new, monthly, daily, subgroup, unspecified. A
                  coarse prior computed with no model; the verifier's definition still wins for timelines.
    source_kind   text | kpi (a detached number whose label came from another line or unit) | table.
    char_start/char_end/quote   offsets of the value inside the fact's unit text as stored in `units.text` (add
                  `units.span_start` for document offsets); quote is a 240-character window around the value.

Separator repairs (`repair_separators`): a Persian thousands separator written as the Arabic comma ("2،788،550") and
dot-grouped thousands ("4.318.701") are read as thousands, the same two repairs AGENTIC2's verifier and
`eval/timeline.py` apply. INGESTION MUST apply `repair_separators` to a unit's text BEFORE `extract_facts`: the L0
number regex splits "2،788،550" into three numbers, so the repair has to happen upstream of it as well as here.

Vocabularies are imported, never copied: `lite.FA_UNIT_WORDS` / `lite.EN_UNIT_WORDS` decide whether an unmapped
token is a unit word at all, and `numbers.SCALES` is the scale-word list.
"""
import json
import re

from ..normalize import fold, normalize
from ..approaches.agentic.numbers import SCALES, find_numbers, same_number
from ..approaches.agentic.reconcile import DEF_CLASSES, definition_class, parse_period
from ..approaches.ours.lite import EN_UNIT_WORDS, FA_UNIT_WORDS

TYPED_VERSION = "kb-typed-v1"

UNIT_KINDS = ("people", "count", "money", "percent", "mass", "energy", "length", "area", "time", "ratio", "other")
PERIOD_KINDS = ("year", "half", "quarter", "month", "date", "range")

# ---------------------------------------------------------------------------------------------------------------
# separator repairs
# ---------------------------------------------------------------------------------------------------------------

_AR_COMMA_THOUS = re.compile(r"(?<=\d)[،٬](?=\d{3}(?!\d))")
# two or more dot-separated groups of exactly 3 digits; ONE dot stays a decimal point ("4.318")
_DOT_THOUS = re.compile(r"(?<![\d/])(?<!\d[.,])\d{1,3}(?:\.\d{3}){2,}(?!\d|[.,]\d)")


def repair_separators(text):
    """"2،788،550" -> "2,788,550"; "4.318.701" -> "4,318,701". Everything else is left alone."""
    t = _DOT_THOUS.sub(lambda m: m.group(0).replace(".", ","), text or "")
    prev = None
    while prev != t:
        prev = t
        t = _AR_COMMA_THOUS.sub(",", t)
    return t


# ---------------------------------------------------------------------------------------------------------------
# units
# ---------------------------------------------------------------------------------------------------------------

_UNITS = {}


def _add(canon, kind, words, scale=1.0, currency=None):
    """`words` are "|"-separated spellings (folded); a spelling of two words is matched as a pair ("per cent")."""
    for w in words.split("|"):
        key = tuple(w.split())
        if key:
            _UNITS[key] = (canon, kind, scale, currency)


_add("percent", "percent", "درصد|%|percent|percents|per cent|percentage")
_add("pp", "percent", "pp|percentage point|percentage points")
_add("bps", "percent", "bps|basis point|basis points")

_add("toman", "money", "تومان|tomans|toman", currency="toman")
_add("toman", "money", "همت", scale=1e12, currency="toman")
_add("rial", "money", "ریال|rials|rial|irr", currency="rial")
_add("usd", "money", "دلار|dollars|dollar|usd|$|us$", currency="usd")
_add("eur", "money", "یورو|euros|euro|eur|€", currency="eur")
_add("gbp", "money", "پوند|pounds|gbp|£", currency="gbp")
_add("aed", "money", "درهم|dirhams|dirham|aed", currency="aed")
_add("cny", "money", "یوان|yuan|rmb|cny|¥", currency="cny")
_add("try", "money", "لیر|lira", currency="try")
_add("rub", "money", "روبل|rubles|rouble", currency="rub")
_add("inr", "money", "روپیه|rupees|rupee|₹", currency="inr")

_add("person", "people", "نفر|people|person|persons")

_add("tonne", "mass", "تن|tons|ton|tonnes|tonne|t")
_add("tonne", "mass", "kt", scale=1e3)
_add("tonne", "mass", "mt", scale=1e6)
_add("kg", "mass", "کیلوگرم|kg|kilogram|kilograms")
_add("gram", "mass", "گرم|grams")

_add("kwh", "energy", "کیلووات ساعت|kwh")
_add("kwh", "energy", "مگاوات ساعت|mwh", scale=1e3)
_add("kwh", "energy", "گیگاوات ساعت|gwh", scale=1e6)
_add("kwh", "energy", "twh", scale=1e9)
_add("mw", "energy", "مگاوات|mw|megawatt|megawatts")
_add("mw", "energy", "گیگاوات|gw", scale=1e3)
_add("mw", "energy", "کیلووات|kw", scale=1e-3)

_add("metre", "length", "متر|meter|metre|meters|metres")
_add("metre", "length", "کیلومتر|km|kilometer|kilometre|kilometers|kilometres", scale=1e3)
_add("sq_metre", "area", "مترمربع|متر مربع")
_add("hectare", "area", "هکتار|ha|hectare|hectares")

_add("hour", "time", "ساعت|hours|hour")
_add("minute", "time", "دقیقه|minutes|minute")
_add("day", "time", "روز|days|day")
_add("week", "time", "هفته|weeks|week")
_add("month", "time", "months|month")
_add("year", "time", "سال|years|year|yr|yrs")

_add("times", "ratio", "برابر|times|x")
_add("unit", "count", "واحد|units")
_add("point", "count", "امتیاز|points|point")
_add("litre", "other", "لیتر|litres|liters|litre|liter")
_add("barrel", "other", "بشکه|barrels|barrel|bbl")

_SYMBOLS = re.compile(r"([$€£¥₹%])")
_UNIT_SPLIT = re.compile(r"[\s/,()\[\]«»]+")


def _unit_tokens(unit_text):
    t = fold(normalize(unit_text or ""))
    t = _SYMBOLS.sub(r" \1 ", t)
    return [w for w in _UNIT_SPLIT.split(t) if w]


def _known_unit_word(tok):
    """A unit word of the OURS vocabularies that this module has no canonical form for."""
    bare = tok.replace("‌", "")
    for cand in (tok, bare, tok[:-1] if tok.endswith("ی") else "", bare[:-1] if bare.endswith("ی") else ""):
        if cand and (cand in FA_UNIT_WORDS or cand in EN_UNIT_WORDS):
            return True
    return False


def _lookup(tok):
    for cand in (tok, tok[:-1] if tok.endswith("ی") else ""):
        if cand and (cand,) in _UNITS:
            return _UNITS[(cand,)]
    return None


def parse_unit(unit_text):
    """-> {unit_norm, unit_kind, currency, is_percent, scale}. `scale` is the product of the unit's scale words."""
    toks = _unit_tokens(unit_text)
    parts, scale, i, unknown = [], 1.0, 0, False
    while i < len(toks):
        pair = tuple(toks[i:i + 2])
        entry = _UNITS.get(pair) if len(pair) == 2 else None
        if entry is not None:
            i += 2
        else:
            entry = _lookup(toks[i])
            i += 1
        if entry is not None:
            canon, kind, esc, cur = entry
            scale *= esc
            parts.append((canon, kind, cur))
            continue
        tok = toks[i - 1]
        sc = SCALES.get(tok, SCALES.get(tok.lower()))
        if sc is not None:
            scale *= sc
        elif _known_unit_word(tok):
            parts.append((tok, "other", None))
        else:
            unknown = True
    sep = "/" if "/" in (unit_text or "") and len(parts) > 1 else " "
    unit_norm = sep.join(p[0] for p in parts)
    if parts:
        kind = parts[0][1]
    else:
        kind = "other" if unknown else "count"     # only scale words, or none at all: a bare count
    currency = next((p[2] for p in parts if p[2]), None)
    return {"unit_norm": unit_norm, "unit_kind": kind, "currency": currency,
            "is_percent": any(p[0] == "percent" for p in parts), "scale": scale}


# ---------------------------------------------------------------------------------------------------------------
# values
# ---------------------------------------------------------------------------------------------------------------

_RANGE_JOIN = re.compile(r"^\s*(?:تا|to|[-–—])\s*$")


def parse_value(value_text, unit_scale=1.0):
    """-> {value_num, value_hi, alt_readings, percent, glitch, raw}. `unit_scale` is `parse_unit(...)["scale"]`.

    A scale word that stands in BOTH the value and the unit ("4.4 میلیون" + "میلیون تومان") is applied once.
    """
    raw = repair_separators(normalize(value_text or ""))
    toks = [t for t in find_numbers(raw) if t.readings]
    out = {"value_num": None, "value_hi": None, "alt_readings": None, "percent": False, "glitch": False, "raw": raw}
    if not toks:
        return out
    tok = toks[0]
    us = unit_scale
    if tok.scale != 1.0 and abs(us - tok.scale) < 1e-9:
        us = 1.0
    value_num = tok.main * us
    out.update(value_num=value_num, percent=tok.percent, glitch=tok.glitch)
    if len(toks) > 1 and _RANGE_JOIN.match(raw[tok.end:toks[1].start]):
        out["value_hi"] = toks[1].main * us
    alt = [r * us for r in tok.readings if not same_number(r * us, value_num)]
    if alt:
        out["alt_readings"] = alt
    return out


# ---------------------------------------------------------------------------------------------------------------
# periods
# ---------------------------------------------------------------------------------------------------------------

_YEAR = re.compile(r"(?<![0-9])(1[34][0-9]{2}|19[0-9]{2}|20[0-9]{2})(?![0-9])")
_RANGE = re.compile(r"(?<![0-9])(1[34][0-9]{2}|19[0-9]{2}|20[0-9]{2})\s*(?:-|–|—|/|تا|to)\s*"
                    r"(1[34][0-9]{2}|19[0-9]{2}|20[0-9]{2})(?![0-9])")
_DATE = re.compile(r"(?<![0-9])(\d{2,4})/(\d{1,2})/(\d{1,4})(?![0-9])")
_HALF = re.compile(r"نیمه[\s‌]?(?:اول|دوم|نخست)|شش[\s‌]?ماهه|(?<![a-z])h[12](?![a-z0-9])|first half|second half", re.I)
_QUARTER = re.compile(r"سه[\s‌]?ماهه|فصل[\s‌]|(?<![a-z])q[1-4](?![a-z0-9])|quarter", re.I)
_MONTH = re.compile(r"فروردین|اردیبهشت|خرداد|تیر|مرداد|شهریور|مهر|آبان|اذر|آذر|دی|بهمن|اسفند|"
                    r"january|february|march|april|may|june|july|august|september|october|november|december", re.I)
_OWN = ("period", "period_range_end", "period_short_year", "quote_year", "quote_short_year")


def calendar_of(year):
    if year is None:
        return None
    if 1300 <= year <= 1499:
        return "jalali"
    if 1900 <= year <= 2100:
        return "gregorian"
    return None


def parse_period_fields(period_text, quote=None, doc_years=None):
    """-> {year, year_to, period_kind, calendar, year_source}.

    The year comes from `reconcile.parse_period` (the fact's period field, else a year of its quote inside the
    document's years, else the document's own year), so a fact is dated here exactly as the v1 reconciler dates it.
    A range is the one departure: `parse_period` returns its end year, while the table keeps start and end.
    """
    t = normalize(period_text or "")
    kind = None
    if _DATE.search(t):
        kind = "date"
    elif _HALF.search(t):
        kind = "half"
    elif _QUARTER.search(t):
        kind = "quarter"
    elif _MONTH.search(t):
        kind = "month"
    year_to = None
    m = _RANGE.search(t)
    if m:
        year, source = int(m.group(1)), "own"
        year_to = int(m.group(2))
        kind = kind or "range"
    else:
        year, how = parse_period(period_text, quote, tuple(doc_years) if doc_years else None)
        source = "own" if how in _OWN else ("doc" if how == "doc_year" else None)
        if year is None:
            source = None
    if kind is None and year is not None:
        kind = "year"
    return {"year": year, "year_to": year_to, "period_kind": kind, "calendar": calendar_of(year),
            "year_source": source}


# ---------------------------------------------------------------------------------------------------------------
# labels and definition classes
# ---------------------------------------------------------------------------------------------------------------

def label_fold(label):
    return fold(normalize(label or ""))


def def_class_of(label, quote=""):
    """The definition class of a fact, from its label and its quote (both verbatim report text)."""
    text = normalize(label or "") + " " + normalize(quote or "")
    return definition_class([name for name, rx in DEF_CLASSES if rx.search(text)])


_KPI_LABELS = ("prev_line", "next_line", "prev_unit", "section_title")


def source_kind_of(fact):
    if (fact.get("src") or "") == "table":
        return "table"
    return "kpi" if (fact.get("label_from") or "") in _KPI_LABELS else "text"


# ---------------------------------------------------------------------------------------------------------------
# the normaliser
# ---------------------------------------------------------------------------------------------------------------

def locate_value(value_text, unit_text):
    """(char_start, char_end, quote) of the value inside the unit text; (None, None, first 240 chars) if not found.

    The unit text is searched as stored (`units.text`, already normalised). The repaired spelling is tried as well,
    so a value read through `repair_separators` still finds its place in the untouched text.
    """
    text = normalize(unit_text or "")
    raw = normalize(value_text or "")
    pos = text.find(raw) if raw else -1
    if pos < 0 and raw:
        rep = repair_separators(raw)
        toks = find_numbers(rep)
        if toks:
            pos = text.find(toks[0].raw)
            raw = toks[0].raw
    if pos < 0:
        return None, None, text[:240]
    start = max(0, pos - 120)
    return pos, pos + len(raw), text[start:start + 240]


def normalise_fact(fact, unit_text="", doc_years=None):
    """One L0 fact -> the typed columns of `facts` (see the module docstring).

    `fact`: a dict from `lite.extract_facts` (`metric`, `value`, `unit`, `period`, `unit_id`, `score`, `src`,
    `label_from`; optional `row_label` and `col_header` when the ingester can split a table cell's label).
    `unit_text`: the text of the fact's unit, as stored in `units.text`.  `doc_years`: (year_from, year_to) or None.
    The ingester adds `doc_id`, `section_id`, `unit_id` and `ord` to make a full `schema.FACT_COLUMNS` row.
    """
    label = fact.get("metric") or ""
    unit_raw = fact.get("unit") or ""
    value_text = fact.get("value") or ""
    period_text = fact.get("period") or ""
    u = parse_unit(unit_raw)
    v = parse_value(value_text, u["scale"])
    char_start, char_end, quote = locate_value(value_text, unit_text)
    period = parse_period_fields(period_text, quote, doc_years)
    is_percent = bool(u["is_percent"] or v["percent"])
    unit_norm, unit_kind = u["unit_norm"], u["unit_kind"]
    if is_percent and not unit_norm:
        unit_norm, unit_kind = "percent", "percent"
    value_cmp = None
    if v["value_num"] is not None and u["currency"] in ("toman", "rial"):
        value_cmp = v["value_num"] * (10.0 if u["currency"] == "toman" else 1.0)
    return {
        "label": label,
        "label_fold": label_fold(label),
        "row_label": fact.get("row_label"),
        "col_header": fact.get("col_header"),
        "value_text": value_text,
        "value_num": v["value_num"],
        "value_hi": v["value_hi"],
        "value_cmp": value_cmp,
        "alt_readings": json.dumps(v["alt_readings"]) if v["alt_readings"] else None,
        "unit_text": unit_raw,
        "unit_norm": unit_norm,
        "unit_kind": unit_kind,
        "currency": u["currency"],
        "is_percent": 1 if is_percent else 0,
        "period_text": period_text,
        "year": period["year"],
        "year_to": period["year_to"],
        "period_kind": period["period_kind"],
        "calendar": period["calendar"],
        "year_source": period["year_source"],
        "def_class": def_class_of(label, quote),
        "source_kind": source_kind_of(fact),
        "char_start": char_start,
        "char_end": char_end,
        "quote": quote,
        "score": float(fact.get("score") or 0.0),
    }
