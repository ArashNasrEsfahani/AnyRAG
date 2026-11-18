"""Deterministic orthographic variants of Persian questions (FRAMING item 3)."""
FA_DIG = "۰۱۲۳۴۵۶۷۸۹"
AR_DIG = "٠١٢٣٤٥٦٧٨٩"
LA_DIG = "0123456789"
ZWNJ = "‌"


def _digits(t, target):
    m = {}
    for src in (FA_DIG, AR_DIG, LA_DIG):
        for i, c in enumerate(src):
            m[ord(c)] = target[i]
    return t.translate(m)


def arabic_yk(t):
    return t.replace("ی", "ي").replace("ک", "ك")


VARIANTS = {
    "orig": lambda t: t,
    "arabic_yk": arabic_yk,                                   # Persian ی/ک -> Arabic ي/ك
    "zwnj_drop": lambda t: t.replace(ZWNJ, ""),
    "zwnj_space": lambda t: t.replace(ZWNJ, " "),
    "digits_fa": lambda t: _digits(t, FA_DIG),
    "digits_ar": lambda t: _digits(t, AR_DIG),
    "digits_latin": lambda t: _digits(t, LA_DIG),
    "combined": lambda t: _digits(arabic_yk(t).replace(ZWNJ, ""), AR_DIG),   # arabic_yk + zwnj_drop + digits_ar
}
