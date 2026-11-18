"""Persian/Arabic-script text helpers for the corpus extractor.

Everything that knows about right-to-left text lives here:
  * character classes and tokenisation of a PDF line (chars in content-stream order)
  * reconstruction of logical order from glyph geometry (works for both
    logical-order and visual-order PDFs, as long as glyph x positions are real)
  * the objective order tests (common function words and their reversed forms,
    common bigrams and their swapped forms)
  * text_norm normalisation
"""
import re
import unicodedata

# --------------------------------------------------------------------------
# character classes
# --------------------------------------------------------------------------

ZWNJ = "\u200c"
TATWEEL = "\u0640"
BIDI_CONTROLS = set("\u200e\u200f\u202a\u202b\u202c\u202d\u202e\u2066\u2067\u2068\u2069\ufeff\u061c")
JOINERS = set("\u200c\u200d\u200b\u00ac\u00ad\u2060")  # U+00AC: some legacy Persian fonts map ZWNJ to NOT SIGN
# U+060D (Arabic date separator) and U+201A/U+201E (low-9 quotes) are used by some
# producers as decimal / thousands separators inside numbers (Jabama, Irancell, Snapp)
NUM_CONNECTORS = set(".,:/\u066b\u066c\u060c-\u2212\u2013\u060d\u201a\u201e")
NUM_AFFIXES = set("%\u066a$\u20ac+")
LTR_CONNECTORS = set(".,:/-_@&'\u2019+#")
BULLETS = set("\u2022\u25cf\u25aa\u25a0\u25e6\u25cb\u25c6\u25c7\u25ba\u25b6\u27a2\u2713\u2714\u2023\u2043\u25fe\u25fd\u2219\u00b7\u25a1\u2610\u2611\u27a4\u279c\u2192\uf0b7\uf0a7\uf0d8\uf076\uf0fc")


def is_arabic_cp(cp):
    return (0x0600 <= cp <= 0x06FF or 0x0750 <= cp <= 0x077F or 0x08A0 <= cp <= 0x08FF
            or 0xFB50 <= cp <= 0xFDFF or 0xFE70 <= cp <= 0xFEFE)


def is_presentation_form(ch):
    cp = ord(ch)
    return 0xFB50 <= cp <= 0xFDFF or 0xFE70 <= cp <= 0xFEFF


def char_class(ch):
    """R: Arabic-script letter, L: other letter, D: digit, M: combining mark,
    Z: joiner (ZWNJ-like), X: bidi control, S: space, U: unmapped/garbage, N: neutral.
    A multi-codepoint entry (one glyph whose ToUnicode maps to several chars, e.g. a
    lam-alef ligature) is classed by its first codepoint."""
    if len(ch) != 1:
        if not ch:
            return "U"
        ch = ch[0]
    if ch in JOINERS:
        return "Z"
    if ch in BIDI_CONTROLS:
        return "X"
    if ch.isspace():
        return "S"
    if ch == "\ufffd":
        return "U"
    cp = ord(ch)
    # code points that never occur in Persian/English report text but are produced by
    # broken TrueType cmaps (variation selectors, combining half marks, CJK punctuation)
    if 0xFE00 <= cp <= 0xFE0F or 0xFE20 <= cp <= 0xFE2F or 0x3000 <= cp <= 0x303F or 0xFFF0 <= cp <= 0xFFFF:
        return "U"
    cat = unicodedata.category(ch)
    if cat in ("Mn", "Me", "Mc"):
        return "M"
    if cat == "Nd":
        return "D"
    if cat[0] == "L":
        return "R" if is_arabic_cp(cp) else "L"
    if cat in ("Cc", "Co", "Cs", "Cn"):
        return "U"
    return "N"


def is_rtl_letter(ch):
    return char_class(ch) == "R"


# --------------------------------------------------------------------------
# normalisation
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
    0x06BE: "\u0647",  # HEH DOACHASHMEE -> HEH (IRVC, Tipax: «شرکت ھای»)
    0x06C1: "\u0647",  # HEH GOAL -> HEH
    0x06D2: "\u06cc",  # YEH BARREE -> FARSI YEH (Tipax: «بین‌شهرے»)
}
_CHAR_MAP.update(_DIGIT_MAP)

_AR = "\u0600-\u06ff\u0750-\u077f\u08a0-\u08ff"
# between two Arabic letters: ZWNJ, ZWSP, RLM and the legacy NOT SIGN all mean "no join here"
_RE_JOINER_BETWEEN = re.compile("(?<=[%s])[\u200c\u200b\u200f\u00ac]+(?=[%s\u064b-\u065f\u0670])" % (_AR, _AR))
# removed everywhere else: ZWJ, soft hyphen, word joiner, bidi controls, BOM
_RE_JOINER_ANY = re.compile("[\u200c\u200d\u200b\u200e\u200f\u202a-\u202e\u2066-\u2069\u2060\ufeff\u061c\u00ad]")
_RE_SOFT = re.compile("[\u00ad\u200d\u2060]")
_RE_WS = re.compile(r"\s+")
_RE_CTRL = re.compile("[\x00-\x08\x0b\x0c\x0e-\x1f\x7f\ufffd\ue000-\uf8ff]")
_RE_DEC_SEP = re.compile("(?<=\\d)\u060d(?=\\d)")
_RE_THOU_SEP = re.compile("(?<=\\d)[\u201a\u201e](?=\\d)")
# a minus typed after the number in RTL text ("2.3-%", "%13-") is displayed in front of it
_RE_NEG_A = re.compile(r"(?<![\w.,/%\-])(\d+(?:[.,/]\d+)*)(?:-%|%-)(?![\w%\-])")
_RE_NEG_B = re.compile(r"(?<![\w.,/%\-])%(\d+(?:[.,/]\d+)*)-(?![\w%\-])")


def normalize(text):
    """text_norm: NFKC, Persian yeh/keheh/heh variants, no tatweel, ZWNJ unified,
    whitespace collapsed, Persian/Arabic-Indic digits to ASCII, U+060D between digits
    to '.', U+201A/U+201E between digits to ',', a trailing percent minus moved in front."""
    if not text:
        return ""
    # NFKC first (presentation forms -> base letters; joiners are untouched by it),
    # then joiners between two Arabic letters become ZWNJ. U+00AC counts only
    # between Arabic letters (elsewhere it is a real NOT SIGN and is kept).
    t = unicodedata.normalize("NFKC", text)
    t = t.translate(_CHAR_MAP)
    t = _RE_SOFT.sub("", t)
    t = _RE_JOINER_BETWEEN.sub(ZWNJ, t)
    # any joiner/bidi control not sitting between Arabic letters is removed
    parts = t.split(ZWNJ)
    t = ZWNJ.join(_RE_JOINER_ANY.sub("", p) for p in parts)
    t = re.sub("\u200c+", ZWNJ, t)
    t = re.sub("\u200c(?=[\\s]|$)|(?:(?<=\\s)|^)\u200c", "", t)
    t = _RE_CTRL.sub("", t)
    t = _RE_WS.sub(" ", t).strip()
    if "\u060d" in t:
        t = _RE_DEC_SEP.sub(".", t)
    if "\u201a" in t or "\u201e" in t:
        t = _RE_THOU_SEP.sub(",", t)
    if "-" in t and "%" in t:
        t = _RE_NEG_A.sub(r"-\1%", t)
        t = _RE_NEG_B.sub(r"-\1%", t)
    return t


# --------------------------------------------------------------------------
# objective order tests
# --------------------------------------------------------------------------

FUNCTION_WORDS = ["و", "در", "به", "از", "که", "این", "با", "را", "است", "برای", "می", "یک", "تا", "هم", "شده"]
_FW = set(FUNCTION_WORDS)
# reversed forms counted as evidence of letter reversal; single letters are
# palindromes, and reversed forms that are themselves common words are left out:
# 'کی' (یک), 'نیا' (این), 'رد' (در: "rejection") and 'مه' (هم: "fog")
_REV = {w[::-1]: w for w in FUNCTION_WORDS if len(w) > 1 and w[::-1] != w and w not in ("یک", "این", "در", "هم")}

# bigrams whose swapped form is (close to) ungrammatical; pairs whose reverse is a
# normal phrase ("این در حالی", "به نسبت", "از یکی از", "کشور در") are left out
BIGRAMS = [
    ("به", "عنوان"), ("است", "که"), ("بر", "اساس"), ("با", "توجه"), ("توجه", "به"),
    ("شده", "است"), ("کرده", "است"), ("بوده", "است"), ("داشته", "است"), ("خواهد", "شد"),
    ("میلیون", "تومان"), ("میلیارد", "تومان"), ("هزار", "تومان"), ("میلیون", "نفر"),
    ("سال", "گذشته"), ("مربوط", "به"), ("قرار", "گرفته"), ("هر", "یک"), ("از", "سوی"),
    ("حال", "حاضر"), ("در", "حالی"), ("قابل", "توجه"), ("افزایش", "یافته"), ("کاهش", "یافته"),
    ("مورد", "استفاده"), ("به", "منظور"), ("از", "جمله"), ("گزارش", "سالانه"), ("در", "طول"),
    ("است", "و"), ("شده", "اند"), ("می‌شود", "که"), ("نشان", "می‌دهد"), ("رشد", "داشته"),
]
_BIGRAM_SET = set(BIGRAMS)
_BIGRAM_REV = {(b, a) for (a, b) in BIGRAMS} - _BIGRAM_SET

_RE_FA_TOKEN = re.compile("[\u0600-\u06ff\u200c]+")


def fa_tokens(norm_text):
    return [t.strip(ZWNJ) for t in _RE_FA_TOKEN.findall(norm_text) if t.strip(ZWNJ)]


def order_hits(norm_text):
    """(forward word hits, reversed word hits, forward bigram hits, swapped bigram hits)."""
    toks = fa_tokens(norm_text)
    fwd = rev = 0
    for t in toks:
        if t in _FW:
            fwd += 1
        elif t in _REV:
            rev += 1
    fb = rb = 0
    words = norm_text.split()
    for a, b in zip(words, words[1:]):
        if (a, b) in _BIGRAM_SET:
            fb += 1
        elif (a, b) in _BIGRAM_REV:
            rb += 1
    return fwd, rev, fb, rb


def order_score(norm_text):
    f, r, fb, rb = order_hits(norm_text)
    return (f - r) + 2 * (fb - rb)


# --------------------------------------------------------------------------
# tokens and line ordering
# --------------------------------------------------------------------------

class Tok:
    __slots__ = ("kind", "chars", "text", "x0", "x1", "visual", "synthetic", "fonts", "ctx")

    def __init__(self, kind):
        self.kind = kind          # R, L (LTR letters/alnum), D (number), S, N, U
        self.chars = []
        self.text = ""
        self.x0 = 0.0
        self.x1 = 0.0
        self.visual = False       # stored in visual order and reversed by us
        self.synthetic = False
        self.ctx = None           # "rtl" / "ltr": direction of the run the token sits in

    def __repr__(self):
        return "Tok(%s,%r,%.1f-%.1f)" % (self.kind, self.text, self.x0, self.x1)


def _cw(c):
    return c["x1"] - c["x0"]


def _nearest_gap(tchars, ch):
    """(horizontal gap, glyph) to the nearest base glyph of a token; (None, None) if none."""
    best = None
    bc = None
    for c in tchars:
        if char_class(c["c"]) in ("M", "Z", "X") or _cw(c) <= 0.01:
            continue
        d = max(ch["x0"] - c["x1"], c["x0"] - ch["x1"])
        if best is None or d < best:
            best, bc = d, c
    return best, bc


def _span_gap_break(a, b, gap):
    """Two Arabic glyphs from different text runs with a visible gap (>= 0.08 em)
    between their advance boxes: a word boundary without a space glyph. Glyphs
    inside one word touch (gap about 0) even when a word is split into runs."""
    if a.get("span") is None or b.get("span") is None or a.get("span") == b.get("span"):
        return False
    if not (a.get("spaceless") and b.get("spaceless")):
        # only on lines set without space glyphs; elsewhere a hole between two runs
        # is a dropped glyph, not a word boundary
        return False
    size = max(min(a.get("size", 10.0), b.get("size", 10.0)), 1.0)
    return gap >= 0.08 * size


def _size_break(a, b):
    """Digits/letters of clearly different type sizes side by side (a 24 pt rank
    numeral next to a 14 pt count) are separate labels, not one number."""
    if a["c"][:1] in NUM_AFFIXES or b["c"][:1] in NUM_AFFIXES:
        return False
    if char_class(a["c"]) != "D" or char_class(b["c"]) != "D":
        # letters: small caps set a capital and lowercase-as-caps in two sizes
        return False
    sa, sb = max(a.get("size", 10.0), 0.1), max(b.get("size", 10.0), 0.1)
    return max(sa, sb) >= 1.35 * min(sa, sb)


def tokenize(chars):
    """chars: list of dicts {c,x0,x1,size,...} in content-stream order."""
    toks = []
    cur = None
    n = len(chars)
    for i, ch in enumerate(chars):
        c = ch["c"]
        cls = char_class(c)
        if cls == "X":
            # RLM used as a ZWNJ between Arabic letters is kept as joiner; other controls dropped
            if c == "\u200f" and cur is not None and cur.kind == "R" and i + 1 < n and char_class(chars[i + 1]["c"]) == "R":
                cur.chars.append(ch)
            continue
        if cls in ("M", "Z"):
            if cur is not None and cur.kind in ("R", "L", "D"):
                cur.chars.append(ch)
                continue
            if cls == "Z":
                continue
            # visual-order streams paint a mark before its base letter: the mark
            # opens that letter's token instead of standing alone
            nxt = char_class(chars[i + 1]["c"]) if i + 1 < n else ""
            kind = "R" if nxt == "R" else "N"
        elif cls == "R":
            kind = "R"
        elif cls == "L":
            kind = "L"
        elif cls == "D":
            kind = "D"
        elif cls == "S":
            kind = "S"
        elif cls == "U":
            kind = "U"
            if c[:1] == "�":
                # an unmapped glyph inside an Arabic word stays inside that word
                nxt = char_class(chars[i + 1]["c"]) if i + 1 < n else ""
                if (cur is not None and cur.kind == "R") or nxt == "R":
                    kind = "R"
        else:
            kind = "N"
            nxt = char_class(chars[i + 1]["c"]) if i + 1 < n else ""
            if cur is not None and cur.kind in ("D", "L"):
                if cur.kind == "D" and c in NUM_CONNECTORS and nxt == "D":
                    kind = cur.kind
                elif cur.kind == "L" and c in LTR_CONNECTORS and nxt in ("L", "D"):
                    kind = "L"
                elif c in NUM_AFFIXES and cur.kind == "D":
                    kind = "D"
            elif c in NUM_AFFIXES and nxt == "D":
                kind = "D"
        # continue current token?
        if cur is not None:
            same = False
            if kind == cur.kind and kind in ("R", "L", "D", "S", "U"):
                same = True
            elif kind in ("L", "D") and cur.kind in ("L", "D"):
                same = True
                kind = "L" if (kind == "L" or cur.kind == "L") else "D"
            if same and kind == "S":
                # stream-consecutive spaces far apart (end of one word run, start of
                # another) must stay separate tokens or they land in the wrong place
                last = cur.chars[-1]
                size = max(last.get("size", 10.0), ch.get("size", 10.0), 1.0)
                if max(ch["x0"] - last["x1"], last["x0"] - ch["x1"]) > 0.3 * size:
                    same = False
            if same and kind != "S" and _cw(ch) > 0.01:
                prev = None
                for pc in reversed(cur.chars):
                    if char_class(pc["c"]) not in ("M", "Z", "X") and _cw(pc) > 0.01:
                        prev = pc
                        break
                # a jump away from the stream-previous glyph starts a new word, unless
                # the glyph touches a glyph already in the token: Word exports write one
                # word as several runs in mixed order, whose glyphs touch although the
                # stream jumps around
                gap_near, near = _nearest_gap(cur.chars, ch)
                if prev is not None and near is not None:
                    size = max(prev.get("size", 10.0), ch.get("size", 10.0), 1.0)
                    gap = max(ch["x0"] - prev["x1"], prev["x0"] - ch["x1"])
                    if gap > 0.6 * size and gap_near > 0.1 * size:
                        same = False
                    elif kind == "R" and _span_gap_break(near, ch, gap_near):
                        # words set as separate text objects with no space glyph
                        same = False
                    elif kind in ("L", "D") and _size_break(prev, ch):
                        same = False
            if same:
                cur.kind = kind if cur.kind != kind else cur.kind
                cur.chars.append(ch)
                continue
        cur = Tok(kind)
        cur.chars.append(ch)
        toks.append(cur)
    for t in toks:
        _finish_token(t)
    return toks


def _finish_token(t, allow_alef_fix=True):
    chars = t.chars
    xs0 = [c["x0"] for c in chars]
    xs1 = [c["x1"] for c in chars]
    t.x0 = min(xs0)
    t.x1 = max(xs1)
    inc = dec = 0
    if t.kind in ("R", "L", "D") and len(chars) > 1:
        bases = [c for c in chars if char_class(c["c"]) in ("R", "L", "D") and _cw(c) > 0.05 * max(c.get("size", 10.0), 1.0)]
        inc = dec = 0
        for a, b in zip(bases, bases[1:]):
            d = (b["x0"] + b["x1"]) / 2.0 - (a["x0"] + a["x1"]) / 2.0
            tol = 0.15 * max(a.get("size", 10.0), 1.0)
            if d > tol:
                inc += 1
            elif d < -tol:
                dec += 1
        if t.kind == "R":
            t.visual = inc > dec
        else:
            t.visual = dec > inc
    kept = [c for c in chars if c["c"] not in BIDI_CONTROLS or c["c"] == "\u200f"]
    if t.kind == "R" and len(kept) > 1:
        _order_rtl_token(t, kept, inc, dec)
        return
    if t.kind in ("L", "D") and len(kept) > 1 and not any(c.get("blob") for c in kept) \
            and len(set(round(c["x0"], 1) for c in kept)) > 1:
        # a left-to-right run reads in visual order; Word stores numbers like 1/95 in
        # RTL text as "95" then "/1", so sort glyphs by position (stable for ties)
        order = sorted(range(len(kept)), key=lambda i: ((kept[i]["x0"] + kept[i]["x1"]) / 2.0, i))
        t.visual = order != list(range(len(kept)))
        entries = [kept[i]["c"] for i in order]
    else:
        entries = [c["c"] for c in kept]
        if t.visual:
            # reverse glyph entries, not codepoints: a ligature entry keeps its own order
            entries = entries[::-1]
    t.text = "".join(entries)


def _order_rtl_token(t, kept, inc, dec):
    """Arabic-script token -> text in logical order.

    Glyph entries are grouped into units: a base glyph plus the combining marks that
    belong to it. A mark belongs to the base whose advance box contains the mark's
    x position; failing that, to the stream-adjacent base (previous one in a logical
    stream, next one in a visual stream, where marks are painted before their base).
    Units are ordered by the stream direction measured on the base glyphs: kept
    (logical stream), reversed (visual stream), or, when the stream runs both ways
    (one word written as several runs in mixed order, typical of Word exports),
    sorted right to left by glyph position."""
    visual = inc > dec
    units = []
    marks = []
    for idx, c in enumerate(kept):
        if char_class(c["c"]) == "M":
            marks.append((idx, c))
        else:
            units.append({"c": c, "idx": idx, "marks": []})
    if not units:
        t.text = "".join(c["c"] for c in kept)
        t.visual = False
        return
    for idx, m in marks:
        cx = (m["x0"] + m["x1"]) / 2.0
        eps = 0.02 * max(m.get("size", 10.0), 1.0)
        home = None
        for u in units:
            b = u["c"]
            if char_class(b["c"]) == "R" and _cw(b) > 0.05 * max(b.get("size", 10.0), 1.0) \
                    and b["x0"] + eps < cx < b["x1"] - eps:
                home = u
                break
        if home is None:
            before = [u for u in units if u["idx"] < idx]
            after = [u for u in units if u["idx"] > idx]
            if visual:
                home = after[0] if after else before[-1]
            else:
                home = before[-1] if before else after[0]
        home["marks"].append((idx, m))
    mixed = inc > 0 and dec > 0 and not any(c.get("blob") for c in kept)
    if mixed:
        ordered = sorted(units, key=lambda u: (-(u["c"]["x0"] + u["c"]["x1"]) / 2.0, u["idx"]))
    elif visual:
        ordered = units[::-1]
    else:
        ordered = units
    t.visual = [u["idx"] for u in ordered] != [u["idx"] for u in units]
    t.text = "".join(u["c"]["c"] + "".join(m["c"] for _, m in sorted(u["marks"], key=lambda x: x[0]))
                     for u in ordered)


MIRROR = {"(": ")", ")": "(", "[": "]", "]": "[", "{": "}", "}": "{",
          "«": "»", "»": "«", "‹": "›", "›": "‹"}
OPENERS = set("([{«‹")
BRACKET_FAMILY = {"(": "paren", ")": "paren", "[": "square", "]": "square", "{": "curly", "}": "curly",
                  "«": "guil", "»": "guil", "‹": "guil1", "›": "guil1"}


def fix_brackets(seq, coding):
    """Paired punctuation in logical order. A bracket glyph's code is either its shape
    ('S': the glyph drawn as '(' is coded '(' -- standard cmaps) or its logical
    meaning in right-to-left use ('L': producer wrote the mirrored character).
    coding(glyph dict, content class) returns 'S', 'L' or None (unknown: left as
    coded); the content class is 'L' when the bracket touches a Latin word, else 'R'.
    In a right-to-left run a shape-coded bracket is mirrored; in a left-to-right run
    a logically-coded one is."""
    n = 0
    m = len(seq)
    for i, t in enumerate(seq):
        if t.kind != "N" or len(t.chars) != 1:
            continue
        ch = t.chars[0]["c"]
        if ch not in MIRROR:
            continue
        cls = "R"
        for j in (i - 1, i + 1):
            if 0 <= j < m and seq[j].kind == "L":
                cls = "L"
        k = coding(t.chars[0], cls)
        if (t.ctx == "rtl" and k == "S") or (t.ctx == "ltr" and k == "L"):
            t.text = MIRROR[ch]
            n += 1
    return n


def geometry_degenerate(chars):
    """True when glyph x positions carry no ordering information (e.g. Type3 fonts
    whose glyphs all report the same origin)."""
    base = [c for c in chars if not c["c"].isspace()]
    if len(base) < 4:
        return False
    xs = set(round(c["x0"] * 2) for c in base)
    return len(xs) <= max(1, int(0.25 * len(base)))


def line_direction(chars, default_rtl):
    r = l = 0
    for c in chars:
        k = char_class(c["c"])
        if k == "R":
            r += 1
        elif k == "L":
            l += 1
    if r == 0 and l == 0:
        return default_rtl
    return r >= l


def visual_tokens(toks, degenerate):
    """Tokens sorted left-to-right with explicit space tokens where glyphs are apart."""
    if degenerate:
        return None
    seq = sorted(toks, key=lambda t: ((t.x0 + t.x1) / 2.0))
    out = []
    prev = None
    for t in seq:
        if t.kind == "S":
            if out and out[-1].kind == "S":
                continue
            out.append(t)
            continue
        if prev is not None and out and out[-1].kind != "S":
            size = max(prev.chars[0].get("size", 10.0), t.chars[0].get("size", 10.0), 1.0)
            gap = t.x0 - prev.x1
            boundary = False
            if gap > -0.05 * size and prev.kind in ("R", "L", "D") and t.kind in ("R", "L", "D"):
                a = max((c for c in prev.chars if _cw(c) > 0.01), key=lambda c: c["x1"], default=None)
                b = min((c for c in t.chars if _cw(c) > 0.01), key=lambda c: c["x0"], default=None)
                if a is not None and b is not None:
                    if prev.kind == "R" and t.kind == "R":
                        boundary = _span_gap_break(a, b, b["x0"] - a["x1"])
                    elif prev.kind in ("L", "D") and t.kind in ("L", "D"):
                        boundary = _size_break(a, b)
            if gap > 0.18 * size or boundary:
                sp = Tok("S")
                sp.text = " "
                sp.x0, sp.x1 = prev.x1, t.x0
                sp.synthetic = True
                out.append(sp)
        out.append(t)
        prev = t
    return out


def merge_letterspaced_year(vis):
    """Four single-digit tokens separated only by spaces that read as a Solar Hijri
    year (13xx/14xx) left-to-right are a letter-spaced year set by a designer."""
    out = []
    i = 0
    n = len(vis)
    while i < n:
        t = vis[i]
        if t.kind == "D" and len(t.text) == 1:
            digits = [t]
            j = i
            k = i + 1
            while k < n and len(digits) < 5:
                if vis[k].kind == "S":
                    k += 1
                    continue
                if vis[k].kind == "D" and len(vis[k].text) == 1:
                    digits.append(vis[k])
                    j = k
                    k += 1
                else:
                    break
            if len(digits) == 4:
                s = "".join(d.text for d in digits)
                if re.fullmatch(r"1[34]\d\d", s.translate(_DIGIT_MAP)):
                    nt = Tok("D")
                    nt.chars = [c for d in digits for c in d.chars]
                    nt.text = s
                    nt.x0 = min(d.x0 for d in digits)
                    nt.x1 = max(d.x1 for d in digits)
                    out.append(nt)
                    i = j + 1
                    continue
        out.append(t)
        i += 1
    return out


def split_visual_on_gaps(vis, factor=3.0):
    """Split a visual token list where a horizontal gap is larger than factor*size."""
    groups = [[]]
    prev = None
    for t in vis:
        if t.kind == "S":
            groups[-1].append(t)
            continue
        if prev is not None:
            size = max(prev.chars[0].get("size", 10.0), t.chars[0].get("size", 10.0), 1.0)
            # large display type: labels 1.5em apart are separate labels, not one line
            if t.x0 - prev.x1 > min(factor * size, max(1.2 * size, 36.0)):
                groups.append([])
        groups[-1].append(t)
        prev = t
    out = []
    for g in groups:
        while g and g[0].kind == "S":
            g = g[1:]
        while g and g[-1].kind == "S":
            g = g[:-1]
        if g:
            out.append(g)
    return out


def _is_ltr_tok(t):
    return t.kind in ("L", "D")


def logical_from_visual(vis, rtl):
    """Inverse bidi on token level: visual left-to-right token list -> logical order."""
    if not rtl:
        seq = list(vis)
        for t in seq:
            t.ctx = "ltr"
        i = 0
        n = len(seq)
        while i < n:
            if seq[i].kind == "R":
                j = i
                k = i + 1
                while k < n and seq[k].kind != "L":
                    if seq[k].kind == "R":
                        j = k
                    k += 1
                for t in seq[i:j + 1]:
                    t.ctx = "rtl"
                if j > i:
                    seq[i:j + 1] = seq[i:j + 1][::-1]
                i = j + 1
            else:
                i += 1
        return seq
    seq = list(reversed(vis))
    for t in seq:
        t.ctx = "rtl"
    n = len(seq)
    i = 0
    while i < n:
        if _is_ltr_tok(seq[i]):
            j = i
            has_l = seq[i].kind == "L"
            k = i + 1
            while k < n and seq[k].kind != "R":
                if _is_ltr_tok(seq[k]):
                    if seq[k].kind == "D" and not has_l and seq[j].kind == "D" and k > j + 1:
                        # number, space, number in RTL context stay separate runs (UBA N1)
                        break
                    j = k
                    if seq[k].kind == "L":
                        has_l = True
                k += 1
            for t in seq[i:j + 1]:
                t.ctx = "ltr"
            if j > i:
                seq[i:j + 1] = seq[i:j + 1][::-1]
            i = j + 1
        else:
            i += 1
    return seq


def tokens_text(seq):
    out = []
    for t in seq:
        out.append(" " if t.kind == "S" else t.text)
    return re.sub(r" {2,}", " ", "".join(out)).strip()


def reverse_rtl_chars(seq):
    """Candidate fix: reverse characters inside every RTL token."""
    out = []
    for t in seq:
        if t.kind == "R":
            nt = Tok("R")
            nt.chars, nt.x0, nt.x1, nt.visual = t.chars, t.x0, t.x1, not t.visual
            if len(t.chars) > 1:
                nt.text = "".join(c["c"] for c in (t.chars if t.visual else t.chars[::-1]))
            else:
                nt.text = t.text[::-1]
            out.append(nt)
        else:
            out.append(t)
    return out


def reverse_word_order(seq):
    """Candidate fix: treat the current logical sequence as visual and re-run RTL
    inverse bidi (reverses word order, keeps LTR runs readable)."""
    return logical_from_visual(seq, True)


def best_fix(seqs_norm_texts):
    """Given candidate names -> normalised texts, return (name, scores)."""
    scores = {k: order_score(v) for k, v in seqs_norm_texts.items()}
    return scores
