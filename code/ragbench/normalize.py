"""Shared Persian/English text normalisation and a script detector.

`normalize()` is a verbatim port of the corpus extractor's `text_norm` rule
(tools/extract/fa_text.py::normalize): NFKC; Arabic yeh/alef maksura -> Farsi yeh;
Arabic kaf -> keheh; tatweel removed; Arabic decimal/thousands/percent signs to
ASCII; every joiner between two Arabic letters unified to ZWNJ and removed
elsewhere; bidi controls removed; Persian and Arabic-Indic digits -> ASCII;
whitespace collapsed. tests/test_foundation.py checks it against the extractor.

Every approach and every index must use these functions; nothing else in the
package may normalise text on its own.
"""
import re
import unicodedata

ZWNJ = "\u200c"

# --------------------------------------------------------------------------
# the extractor rule (keep in lockstep with tools/extract/fa_text.py)
# --------------------------------------------------------------------------

_DIGIT_MAP = {}
for _i in range(10):
    _DIGIT_MAP[0x06F0 + _i] = str(_i)
    _DIGIT_MAP[0x0660 + _i] = str(_i)
_CHAR_MAP = {
    0x064A: "\u06cc",  # ARABIC YEH -> FARSI YEH
    0x0649: "\u06cc",  # ALEF MAKSURA -> FARSI YEH
    0x0643: "\u06a9",  # ARABIC KAF -> KEHEH
    0x0640: None,      # tatweel removed
    0x066B: ".",       # Arabic decimal separator
    0x066C: ",",       # Arabic thousands separator
    0x066A: "%",       # Arabic percent sign
}
_CHAR_MAP.update(_DIGIT_MAP)

_AR = "\u0600-\u06ff\u0750-\u077f\u08a0-\u08ff"
_RE_JOINER_BETWEEN = re.compile("(?<=[%s])[\u200c\u200b\u200f\u00ac]+(?=[%s\u064b-\u065f\u0670])" % (_AR, _AR))
_RE_JOINER_ANY = re.compile("[\u200c\u200d\u200b\u200e\u200f\u202a-\u202e\u2066-\u2069\u2060\ufeff\u061c\u00ad]")
_RE_SOFT = re.compile("[\u00ad\u200d\u2060]")
_RE_WS = re.compile(r"\s+")
_RE_CTRL = re.compile("[\x00-\x08\x0b\x0c\x0e-\x1f\x7f\ufffd\ue000-\uf8ff]")


def normalize(text):
    """The extractor's text_norm. Collapses ALL whitespace, newlines included."""
    if not text:
        return ""
    t = unicodedata.normalize("NFKC", text)
    t = t.translate(_CHAR_MAP)
    t = _RE_SOFT.sub("", t)
    t = _RE_JOINER_BETWEEN.sub(ZWNJ, t)
    parts = t.split(ZWNJ)
    t = ZWNJ.join(_RE_JOINER_ANY.sub("", p) for p in parts)
    t = re.sub("\u200c+", ZWNJ, t)
    t = re.sub("\u200c(?=[\\s]|$)|(?:(?<=\\s)|^)\u200c", "", t)
    t = _RE_CTRL.sub("", t)
    t = _RE_WS.sub(" ", t).strip()
    return t


def normalize_lines(text):
    """normalize() applied line by line: keeps line structure (tables, lists),
    drops empty lines at the ends and collapses runs of blank lines to one."""
    if not text:
        return ""
    out = []
    blank = False
    for line in str(text).replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        n = normalize(line)
        if not n:
            if out and not blank:
                out.append("")
            blank = True
            continue
        out.append(n)
        blank = False
    while out and out[-1] == "":
        out.pop()
    return "\n".join(out)


def to_ascii_digits(text):
    return (text or "").translate(_DIGIT_MAP)


# --------------------------------------------------------------------------
# optional extra folding for lexical matching (NOT part of the extractor rule)
# --------------------------------------------------------------------------

_FOLD_MAP = {
    0x0622: "\u0627", 0x0623: "\u0627", 0x0625: "\u0627", 0x0671: "\u0627",  # alef variants -> alef
    0x0629: "\u0647", 0x06C0: "\u0647", 0x06D5: "\u0647",                    # teh marbuta, heh with yeh -> heh
    0x0624: "\u0648",                                                        # waw with hamza -> waw
    0x0626: "\u06cc",                                                        # yeh with hamza -> yeh
}
_RE_HARAKAT = re.compile("[\u064b-\u065f\u0670\u06d6-\u06ed]")


def fold(text):
    """normalize() + matching-oriented folding: harakat removed, alef/heh/waw/yeh
    variants folded, ZWNJ -> space, lower-cased. Use it on BOTH sides (index and
    query) or not at all."""
    t = normalize(text)
    t = _RE_HARAKAT.sub("", t)
    t = t.translate(_FOLD_MAP)
    t = t.replace(ZWNJ, " ")
    return _RE_WS.sub(" ", t).strip().lower()


_RE_WORD = re.compile(r"[0-9]+(?:[.,/][0-9]+)*%?|[^\W\d_]+(?:[\u200c'][^\W\d_]+)*", re.UNICODE)


def word_tokens(text, folded=True):
    """Simple shared tokenizer (numbers keep their separators, Persian words keep
    internal ZWNJ unless folded). Lower-cased."""
    t = fold(text) if folded else normalize(text).lower()
    return _RE_WORD.findall(t)


# --------------------------------------------------------------------------
# script detection
# --------------------------------------------------------------------------

def _is_arabic_letter(ch):
    cp = ord(ch)
    return (0x0600 <= cp <= 0x06FF or 0x0750 <= cp <= 0x077F or 0x08A0 <= cp <= 0x08FF
            or 0xFB50 <= cp <= 0xFDFF or 0xFE70 <= cp <= 0xFEFE) and ch.isalpha()


def script_counts(text):
    """(arabic-script letters, other letters)."""
    fa = en = 0
    for ch in text or "":
        if not ch.isalpha():
            continue
        if _is_arabic_letter(ch):
            fa += 1
        else:
            en += 1
    return fa, en


def fa_ratio(text):
    fa, en = script_counts(text)
    return fa / (fa + en) if (fa + en) else 0.0


def detect_lang(text, default="en"):
    """'fa' when Arabic-script letters are at least 30% of letters, else 'en'.
    Persian prose with Latin brand names stays 'fa'; English with a Persian
    word stays 'en'. Text with no letters returns `default`."""
    fa, en = script_counts(text)
    if fa + en == 0:
        return default
    return "fa" if fa / (fa + en) >= 0.30 else "en"
