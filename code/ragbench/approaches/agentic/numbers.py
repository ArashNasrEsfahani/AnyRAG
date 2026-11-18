"""Number tokens in normalised report text, and the readings the verbatim guard compares.

Text is expected to be normalised with ragbench.normalize.normalize (ASCII digits, Arabic separators
mapped to "," and "."). Formats seen in corpora A, B and C:
    6,152,509   1,088   97.4   %97.4   97.4%   4/9 (Persian decimal slash)   2میلیون   4.4 میلیون
    $38.9B   4.4M   700 هزار تن   38.9 billion
    979„379„4   567„ 150   000„+18      (right-to-left extraction glitch: thousands groups reversed,
                                         separators U+201E / U+201A; Snapp 1401 report)
A token has one or more readings (floats). A glitch token reads both as written and group-reversed;
a scale word adds the scaled reading (2میلیون -> 2 and 2,000,000). Reading-level equality is what the
guard means by "the value occurs verbatim": the digits are in the text, only separators/scale differ.
"""
import re

SCALES = {
    "هزار": 1e3, "میلیون": 1e6, "ملیون": 1e6, "میلیارد": 1e9, "ملیارد": 1e9, "بیلیون": 1e9, "تریلیون": 1e12,
    "همت": 1e12,
    "thousand": 1e3, "thousands": 1e3, "k": 1e3, "million": 1e6, "millions": 1e6, "mn": 1e6, "m": 1e6,
    "billion": 1e9, "billions": 1e9, "bn": 1e9, "b": 1e9, "trillion": 1e12, "tn": 1e12,
}
_FA_SCALE = "هزار|میلیون|ملیون|میلیارد|ملیارد|بیلیون|تریلیون|همت"
_EN_SCALE = "thousands?|millions?|billions?|trillion|mn|bn|tn|[kKmMbB](?![A-Za-z])"

_GLITCH_SEP = "„‚"      # „ ‚

NUM_RE = re.compile(
    r"(?P<date>(?<![0-9])\d{2,4}/\d{1,2}/\d{1,4}(?![0-9]))"
    r"|(?<![A-Za-z0-9.,/])(?P<sign>[+\-−])?(?P<body>"
    r"\d{1,3}(?:\s?[" + _GLITCH_SEP + r"]\s?\+?\d{1,3})+"
    r"|\d{1,3}(?:,\d{3})+(?:\.\d+)?"
    r"|\d+/\d{1,3}(?![0-9/])"
    r"|\d+(?:\.\d+)?"
    r")(?![0-9])"
    r"(?P<pct>\s?%)?"
    r"(?P<scale>\s?(?:" + _FA_SCALE + r")(?:\s?(?:" + _FA_SCALE + r"))?|\s?(?:" + _EN_SCALE + r")(?![A-Za-z]))?"
)


class NumToken:
    __slots__ = ("raw", "start", "end", "body", "readings", "percent", "scale", "glitch", "is_year", "is_date",
                 "reversed_reading")

    def __init__(self, raw, start, end, body="", readings=(), percent=False, scale=1.0, glitch=False,
                 is_year=False, is_date=False, reversed_reading=None):
        self.raw, self.start, self.end, self.body = raw, start, end, body
        self.readings = list(readings)
        self.percent, self.scale, self.glitch = percent, scale, glitch
        self.is_year, self.is_date = is_year, is_date
        self.reversed_reading = reversed_reading

    @property
    def main(self):
        """The reading a table shows: the scaled value, or for a glitch token its reversed-group reading
        when only that one is a well-formed thousands grouping."""
        if self.glitch and self.reversed_reading is not None:
            return self.reversed_reading
        return self.readings[-1] if self.readings else None

    def to_dict(self):
        return {"raw": self.raw, "readings": self.readings, "percent": self.percent, "glitch": self.glitch,
                "is_year": self.is_year}

    def __repr__(self):
        return "NumToken(%r, %s)" % (self.raw, self.readings)


def _glitch_readings(body):
    """'979„379„4' -> ([4379979.0], 4379979.0); '567„ 150' -> ([567150.0, 150567.0], 150567.0).
    A reading is kept when its groups form a valid thousands grouping (every group after the first has
    3 digits); the reversed reading is the main one whenever it is valid (the glitch reverses groups)."""
    groups = [g.lstrip("+") for g in re.split(r"\s?[" + _GLITCH_SEP + r"]\s?\+?", body) if g.strip("+") != ""]
    rev = list(reversed(groups))
    written_ok = all(len(g) == 3 for g in groups[1:])
    reversed_ok = all(len(g) == 3 for g in rev[1:])
    as_written = float(int("".join(groups)))
    rev_val = float(int("".join(rev)))
    out = []
    if written_ok or not reversed_ok:
        out.append(as_written)
    if reversed_ok and rev_val not in out:
        out.append(rev_val)
    return out, (rev_val if reversed_ok else None)


def _scale_value(scale_text):
    if not scale_text:
        return 1.0
    words = re.findall(_FA_SCALE + "|" + r"thousands?|millions?|billions?|trillion|mn|bn|tn|[kKmMbB]", scale_text)
    f = 1.0
    for w in words:
        f *= SCALES.get(w.lower(), SCALES.get(w, 1.0))
    return f


def find_numbers(text):
    """All number tokens in `text` (normalised), in order."""
    out = []
    for m in NUM_RE.finditer(text or ""):
        if m.group("date"):
            out.append(NumToken(m.group(0), m.start(), m.end(), body=m.group("date"), is_date=True))
            continue
        body = m.group("body")
        if body is None:
            continue
        glitch = any(c in body for c in _GLITCH_SEP)
        rev = None
        if glitch:
            base, rev = _glitch_readings(body)
        elif "/" in body:
            a, b = body.split("/", 1)
            base = [float(a + "." + b)]
        else:
            try:
                base = [float(body.replace(",", ""))]
            except ValueError:
                continue
        scale_txt = (m.group("scale") or "").strip()
        # a bare English letter after a number is a scale only when glued to it (4.4M, 38.9B)
        if scale_txt and len(scale_txt) == 1 and m.group("scale") != scale_txt:
            scale_txt = ""
        scale = _scale_value(scale_txt)
        end = m.end() if scale_txt else (m.end("pct") if m.group("pct") else m.end("body"))
        readings = list(base)
        if scale != 1.0:
            readings += [b * scale for b in base]
            if rev is not None:
                rev = rev * scale
        pct = bool(m.group("pct")) or (m.start() > 0 and text[m.start() - 1] == "%") or (
            m.start("body") > 1 and text[m.start("body") - 1] == "%")
        is_year = (not glitch and "," not in body and "." not in body and "/" not in body and scale == 1.0
                   and not pct and len(body) == 4 and (1300 <= int(body) <= 1499 or 1900 <= int(body) <= 2100))
        raw = text[m.start():end].strip()
        out.append(NumToken(raw, m.start(), end, body=body, readings=readings, percent=pct, scale=scale,
                            glitch=glitch, is_year=is_year, reversed_reading=rev))
    return out


def same_number(a, b):
    return abs(a - b) <= 1e-9 * max(1.0, abs(a), abs(b))


def readings_match(tok_a, tok_b):
    return any(same_number(x, y) for x in tok_a.readings for y in tok_b.readings)


def candidate_numbers(tokens, unit_kind=None):
    """Tokens that can be a metric value: not dates, not bare years (unless a date is asked for)."""
    out = []
    for t in tokens:
        if t.is_date and unit_kind != "date":
            continue
        if t.is_year and unit_kind != "date":
            continue
        if not t.readings and not t.is_date:
            continue
        out.append(t)
    return out


_CURRENCY = re.compile(r"تومان|ریال|دلار|یورو|درهم|\$|€|usd|eur|dollar|rial|toman", re.I)


def unit_kind_match(text, tok, unit_kind):
    """Does the token look like the expected kind of value?"""
    if not unit_kind or unit_kind in ("text",):
        return False
    after = text[tok.end:tok.end + 14]
    before = text[max(0, tok.start - 3):tok.start]
    if unit_kind == "percent":
        return tok.percent
    if unit_kind == "currency":
        return bool(_CURRENCY.search(after) or _CURRENCY.search(before))
    if unit_kind == "count":
        return (not tok.percent and not _CURRENCY.search(after) and not _CURRENCY.search(before)
                and any(r >= 10 for r in tok.readings) and all(float(r).is_integer() for r in tok.readings[:1]))
    if unit_kind == "date":
        return tok.is_date or tok.is_year
    if unit_kind in ("quantity", "ratio"):
        return not tok.percent
    return False


def fmt_number(x):
    """6152509.0 -> '6,152,509'; 97.4 -> '97.4'."""
    if x is None:
        return ""
    if float(x).is_integer():
        return "{:,}".format(int(x))
    return ("%.6f" % x).rstrip("0").rstrip(".")
