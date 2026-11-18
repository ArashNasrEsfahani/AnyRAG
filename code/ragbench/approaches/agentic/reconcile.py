"""Stage 5a - reconciliation (code, no model): verified items -> rows -> one table per question.

Rules, applied in this order:
  period      the item's period field (a 4-digit year, a range's end year, or "سال 98" -> 1398); else a year in
              the quote that lies inside the document's own years; else the document's year (period_inferred)
  role        authoritative   the row comes from the report OF that period (the document's years contain it)
              restatement     a later report states the value for an earlier period
              projection      an earlier report states a later period (kept out of the series)
  choice      per period the authoritative rows are kept; with none, the rows of the latest restating report
              (role latest_restatement). A restatement whose value differs (> restatement_rel_tol) from a kept row
              with the same definition class is a CONFLICT; the kept (authoritative) value stays in the series.
  definition  a class from the quote first (it is verbatim text), else from the model's definition:
              active, registered, total, accounts, fulltime, parttime, new, monthly, daily, subgroup, unspecified
  comparable  timeline/comparison with >= 2 periods: True only when every kept row (subgroups aside) has the
              same class. When False no trend is stated; per-class first/last values are listed as facts.
Single-fact questions get the same table with the best rows (authoritative first, then verification rank).
"""
import re

from ...normalize import fold, normalize
from .numbers import fmt_number, same_number

_YEAR = re.compile(r"(?<![0-9])(1[34][0-9]{2}|19[0-9]{2}|20[0-9]{2})(?![0-9])")
_RANGE = re.compile(r"(?<![0-9])(1[34][0-9]{2}|19[0-9]{2}|20[0-9]{2})\s*(?:-|–|—|/|تا|to)\s*(1[34][0-9]{2}|19[0-9]{2}|20[0-9]{2})(?![0-9])")
_SHORT_YEAR = re.compile(r"سال\s?([0-9]{2})(?![0-9])|(?<![0-9])([0-9]{2})\s?(?:خورشیدی|شمسی)")

DEF_CLASSES = (
    ("active", re.compile(r"(?<!غیر)(?<!غیر‌)(?<!غیر )فعال|(?<!in)active", re.I)),
    ("registered", re.compile(r"ثبت[‌ ]?نام|ثبت[‌ ]?شده|registered|sign[- ]?ups?|signed up|enrolled", re.I)),
    ("total", re.compile(r"(?<![^\s‌|:(])کل(?![^\s‌|:)])|مجموع|total|overall|cumulative|all[- ]time|تا ?کنون|تاکنون", re.I)),
    ("accounts", re.compile(r"حساب(?:[‌ ]?های)?[‌ ]?کاربری|(?<![a-z])accounts?(?![a-z])", re.I)),
    ("fulltime", re.compile(r"تمام[‌ ]?وقت|full[- ]time", re.I)),
    ("parttime", re.compile(r"پاره[‌ ]?وقت|part[- ]time", re.I)),
    ("new", re.compile(r"جدید|جذب|(?<![a-z])new(?![a-z])|newly|joined", re.I)),
    ("monthly", re.compile(r"ماهانه|در (?:یک|هر) ماه|per month|monthly|a month", re.I)),
    ("daily", re.compile(r"روزانه|در (?:یک|هر) روز|per day|daily", re.I)),
    ("subgroup", re.compile(r"(?<![^\s‌|:(])(?:زن|زنان|مرد|مردان)(?![^\s‌|:)])|women|female|(?<![a-z])men(?![a-z])|(?<![fe])male", re.I)),
)

CLASS_LABELS = {
    "fa": {"active": "فعال", "registered": "ثبت‌نام‌شده", "total": "کل", "accounts": "حساب کاربری",
           "fulltime": "تمام‌وقت", "parttime": "پاره‌وقت", "new": "جدید", "monthly": "ماهانه", "daily": "روزانه",
           "subgroup": "زیرگروه", "unspecified": "بدون قید"},
    "en": {"active": "active", "registered": "registered", "total": "total", "accounts": "accounts",
           "fulltime": "full-time", "parttime": "part-time", "new": "new", "monthly": "monthly", "daily": "daily",
           "subgroup": "subgroup", "unspecified": "unqualified"},
}

TEXT = {
    "fa": {
        "title": "جدول مقادیر تأییدشده (هر مقدار و نقل‌قولش با متن منبع مقایسه شده است)",
        "header": "دوره | مقدار | واحد | تعریف | منبع",
        "rule": "قاعده: مقدار هر دوره از گزارشِ همان دوره گرفته شده است؛ بازگویی‌های گزارش‌های بعدی فقط یادداشت شده‌اند.",
        "not_comparable": "تعریف‌ها در دوره‌ها یکسان نیست ({classes})؛ این اعداد یک سری قابل مقایسه نیستند و نباید از آن‌ها روند نتیجه گرفت.",
        "same_label": "با برچسب یکسان «{label}»: {points}.",
        "comparable": "تعریف در همهٔ دوره‌ها یکسان است ({label}).",
        "missing": "برای این دوره‌ها مقدار تأییدشده‌ای پیدا نشد: {periods}.",
        "restated": "گزارش {doc_year} مقدار دورهٔ {period} را {value} آورده است (مقدار گزارش همان دوره: {kept}).",
        "several": "دورهٔ {period} چند مقدار با تعریف یا دامنهٔ متفاوت دارد.",
        "latest": "برای دورهٔ {period} گزارش همان دوره مقداری نداشت؛ مقدار از گزارش {doc_year} آمده است.",
        "inferred": "دوره از سال گزارش گرفته شده است",
    },
    "en": {
        "title": "Table of verified values (each value and its quote were checked against the source text)",
        "header": "period | value | unit | definition | source",
        "rule": "Rule: each period's value comes from that period's own report; restatements in later reports are only noted.",
        "not_comparable": "Definitions differ across periods ({classes}); these values are not one comparable series and no trend should be drawn from them.",
        "same_label": "With the same label \"{label}\": {points}.",
        "comparable": "The definition is the same in every period ({label}).",
        "missing": "No verified value was found for: {periods}.",
        "restated": "The {doc_year} report restates {period} as {value} (the {period} report's own value: {kept}).",
        "several": "Period {period} has several values with different definitions or scopes.",
        "latest": "For {period} the report of that period gave no value; the value comes from the {doc_year} report.",
        "inferred": "period taken from the report's year",
    },
}


def definition_flags(quote, definition):
    q = normalize(quote or "")
    flags = [name for name, rx in DEF_CLASSES if rx.search(q)]
    source = "quote"
    if not flags:
        d = normalize(definition or "")
        flags = [name for name, rx in DEF_CLASSES if rx.search(d)]
        source = "definition" if flags else "none"
    return flags, source


def definition_class(flags):
    main = [f for f in flags if f != "subgroup"]
    if not main:
        return "subgroup" if "subgroup" in flags else "unspecified"
    return "+".join(sorted(main))


def parse_period(period_text, quote, doc_years):
    """-> (year | None, how)"""
    t = normalize(period_text or "")
    m = None
    for m in _RANGE.finditer(t):
        pass
    if m:
        return int(m.group(2)), "period_range_end"
    ys = _YEAR.findall(t)
    if ys:
        return int(ys[-1]), "period"
    m = _SHORT_YEAR.search(t)
    if m:
        yy = int(m.group(1) or m.group(2))
        return 1300 + yy, "period_short_year"
    if doc_years:
        qy = [int(y) for y in _YEAR.findall(normalize(quote or "")) if doc_years[0] <= int(y) <= doc_years[1]]
        if qy:
            return qy[-1], "quote_year"
        m = _SHORT_YEAR.search(normalize(quote or ""))
        if m:
            yy = 1300 + int(m.group(1) or m.group(2))
            if doc_years[0] <= yy <= doc_years[1]:
                return yy, "quote_short_year"
        return doc_years[1], "doc_year"
    return None, "none"


def make_row(item, cand, ginfo, doc_meta, unit_id, verified=True):
    years = doc_meta.get("years") if doc_meta else None
    py, how = parse_period(item.get("period"), item.get("quote"), years)
    flags, fsrc = definition_flags(item.get("quote"), item.get("definition"))
    if years is None:
        role = "authoritative"          # a document without a year (corpus A): its own statement
    elif py is None:
        role = "unknown_period"
    elif years[0] <= py <= years[1]:
        role = "authoritative"
    elif py < years[0]:
        role = "restatement"
    else:
        role = "projection"
    vnum = ginfo.get("value_num") if ginfo else None
    token = (ginfo or {}).get("value_token") or str(item.get("value") or "")
    if vnum is not None and (ginfo or {}).get("reading") == "reversed_groups":
        value = fmt_number(vnum)
    else:
        value = normalize(token)
    return {
        "period": str(py) if py is not None else "",
        "value": value,
        "unit": normalize(str(item.get("unit") or "")),
        "definition": normalize(str(item.get("definition") or "")),
        "quote": normalize(str(item.get("quote") or "")),
        "unit_id": unit_id,
        "doc_id": cand.doc_id,
        "verified": bool(verified),
        # extras
        "value_num": vnum,
        "value_raw": token,
        "value_reading": (ginfo or {}).get("reading"),
        "model_value": str(item.get("value") or ""),
        "model_period": str(item.get("period") or ""),
        "period_year": py,
        "period_how": how,
        "period_inferred": how == "doc_year",
        "doc_years": list(years) if years else None,
        "role": role,
        "def_flags": flags,
        "def_flags_from": fsrc,
        "def_class": definition_class(flags),
        "section_id": cand.section_id,
        "unit_ids": list(cand.unit_ids),
        "cid": cand.cid,
        "score": cand.score,
        "round": cand.round,
    }


_QUAL_SPLIT = re.compile(r"[\s‌،,.:;()«»|/\-*]+")
_QUAL_STOP = set("""از به با در که را این آن برای تا و یا هم نیز بر ای های ها است هستند بودند بود شد شده شدند می کرده کردند
داشتند دارد دارند پایان ابتدای انتهای سال سالانه تاکنون کنون کل همه مجموع تعداد شمار نفر نفری کاربر کاربران حساب
کاربری the of in and for with from to by at total number count all users user people persons accounts""".split())


def qualifiers(definition, known_stems):
    """Content words of a row's definition that the question's terms, the entity and generic words do not explain:
    "راننده دارای معلولیت" -> [دارای, معلولیت]; "کاربر راننده ی فعال" -> [] (فعال is a modifier word the
    definition classes handle)."""
    out = []
    for w in _QUAL_SPLIT.split(normalize(definition or "")):
        f = fold(w)
        if len(w) < 3 or re.search(r"[0-9]", w) or f in _QUAL_STOP:
            continue
        if any(s and (s in f or f in s) for s in known_stems):
            continue
        out.append(w)
    return out


def _pkey(p):
    return (p is None, p or 0)


def _pstr(p):
    return "" if p is None else str(p)


def _differs(a, b, tol):
    if a is None or b is None:
        return False
    return not same_number(a, b) and abs(a - b) > tol * max(abs(a), abs(b))


def reconcile(rows, *, kind, target_periods=(), opts=None, q_lang="fa", known_stems=None):
    """known_stems: folded stems of the question's metric terms, entity names and modifier words; when given, each
    row gets `qualifiers` (definition words they do not explain). Per period a row with >= 2 qualifiers and more
    than the plainest row is NARROWER (a programme's users, a subgroup, one service): it moves to out["narrower"]
    instead of the series; a single-fact table orders rows by fewest qualifiers."""
    opts = opts or {}
    tol = float(opts.get("restatement_rel_tol", 0.005))
    out = {"kind": kind, "series": [], "restatements": [], "conflicts": [], "projections": [],
           "unknown_period": [], "notes": [], "latest_restatement_periods": [], "narrower": []}
    if known_stems is not None:
        for r in rows:
            r["qualifiers"] = qualifiers(r.get("definition"), known_stems)
    by = {}
    for r in rows:
        if r["role"] == "projection":
            out["projections"].append(r)
        elif r["role"] == "unknown_period":
            out["unknown_period"].append(r)
        else:
            by.setdefault(r["period_year"], []).append(r)
    kept_all = []
    for p in sorted(by, key=_pkey):
        rs = by[p]
        auth = [r for r in rs if r["role"] == "authoritative"]
        rest = [r for r in rs if r["role"] == "restatement"]
        if auth:
            kept = auth
        else:
            latest = max((r["doc_years"] or [0, 0])[1] for r in rest)
            kept = [dict(r, role="latest_restatement") for r in rest if (r["doc_years"] or [0, 0])[1] == latest]
            rest = [r for r in rest if (r["doc_years"] or [0, 0])[1] != latest]
            out["latest_restatement_periods"].append(p)
        # dedupe equal values within a class (keep the best-ranked row)
        uniq = []
        for r in sorted(kept, key=lambda r: -(r.get("score") or 0)):
            if any(u["def_class"] == r["def_class"] and u["value_num"] is not None and r["value_num"] is not None
                   and same_number(u["value_num"], r["value_num"]) for u in uniq):
                continue
            if r["value_num"] is None and any(u["value"] == r["value"] for u in uniq):
                continue
            uniq.append(r)
        for r in rest:
            out["restatements"].append(r)
            same = [u for u in uniq if u["def_class"] == r["def_class"]]
            if same and all(_differs(u["value_num"], r["value_num"], tol) for u in same):
                out["conflicts"].append({"period": str(p), "kept": [u["value"] for u in same], "kept_doc": same[0]["doc_id"],
                                         "restated": r["value"], "restated_doc": r["doc_id"],
                                         "def_class": r["def_class"]})
        kept_all += uniq
    # subgroup rows only when nothing else exists for that period; narrower rows out when a plainer one exists
    series = []
    for p in sorted({r["period_year"] for r in kept_all}, key=_pkey):
        rs = [r for r in kept_all if r["period_year"] == p]
        main = [r for r in rs if r["def_class"] != "subgroup"]
        rs = main if main else rs
        if known_stems is not None and kind != "single" and len(rs) > 1:
            plainest = min(len(r["qualifiers"]) for r in rs)
            narrow = [r for r in rs if len(r["qualifiers"]) >= 2 and len(r["qualifiers"]) > plainest]
            out["narrower"] += narrow
            rs = [r for r in rs if r not in narrow]
        series += rs
    if kind == "single":
        series.sort(key=lambda r: (r["role"] != "authoritative", len(r.get("qualifiers") or []), -(r.get("score") or 0)))
        uniq = []          # one row per distinct value (the same number quoted twice with other wording)
        for r in series:
            if r["value_num"] is not None and any(u["value_num"] is not None and same_number(u["value_num"], r["value_num"])
                                                  for u in uniq):
                continue
            uniq.append(r)
        series = uniq[: int(opts.get("single_rows_max", 3))]
    out["series"] = series
    periods = sorted({r["period_year"] for r in series}, key=_pkey)
    out["periods_found"] = [_pstr(p) for p in periods]
    out["periods_missing"] = [str(p) for p in target_periods if p not in set(periods)]
    classes = {}
    for r in series:
        if r["def_class"] == "subgroup":
            continue
        classes.setdefault(r["def_class"], [])
        if _pstr(r["period_year"]) not in classes[r["def_class"]]:
            classes[r["def_class"]].append(_pstr(r["period_year"]))
    out["definition_classes"] = classes
    out["several_values_periods"] = [_pstr(p) for p in periods if p is not None and kind != "single"
                                     and sum(1 for r in series if r["period_year"] == p) > 1]
    if kind in ("timeline", "comparison") and len([p for p in periods if p is not None]) >= 2:
        out["comparable"] = len(classes) == 1
    else:
        out["comparable"] = None
    main_class = max(classes, key=lambda c: (len(classes[c]), c != "unspecified")) if classes else None
    out["main_class"] = main_class
    class_points = {}
    for c, ps in classes.items():
        pts = []
        for p in ps:
            rr = [r for r in series if r["def_class"] == c and _pstr(r["period_year"]) == p]
            if len(rr) == 1:
                pts.append((p, rr[0]["value"]))
        class_points[c] = pts
    out["class_points"] = class_points
    out["trend"] = None
    if out["comparable"] and main_class and len(class_points.get(main_class, [])) >= 2:
        first = [r for r in series if _pstr(r["period_year"]) == class_points[main_class][0][0]][0]
        last = [r for r in series if _pstr(r["period_year"]) == class_points[main_class][-1][0]][0]
        if first["value_num"] is not None and last["value_num"] is not None:
            direction = "up" if last["value_num"] > first["value_num"] else "down" if last["value_num"] < first["value_num"] else "flat"
            out["trend"] = {"class": main_class, "from": [first["period"], first["value"]],
                            "to": [last["period"], last["value"]], "direction": direction}
    return out


def table_text(rec, q_lang, source_numbers):
    """The evidence item [1]: a verified-values table with the reconciliation notes, in the question language.
    source_numbers: {cid: evidence item number}."""
    L = TEXT.get(q_lang, TEXT["en"])
    C = CLASS_LABELS.get(q_lang, CLASS_LABELS["en"])
    sep, sep2 = ("، ", "؛ ") if q_lang == "fa" else (", ", "; ")

    def label(c):
        return " + ".join(C.get(x, x) for x in c.split("+"))

    lines = [L["title"], L["header"]]
    for r in rec["series"]:
        src = "[%d]" % source_numbers[r["cid"]] if r["cid"] in source_numbers else "-"
        period = (r["period"] or "-") + ("*" if r.get("period_inferred") else "")
        definition = r["definition"] or label(r["def_class"])
        lines.append("%s | %s | %s | %s (%s) | %s" % (period, r["value"], r["unit"] or "-", definition[:90],
                                                    label(r["def_class"]), src))
    return "\n".join(lines + table_notes(rec, q_lang))


def table_notes(rec, q_lang):
    """The reconciliation notes under the table (also RunRecord structured.notes)."""
    L = TEXT.get(q_lang, TEXT["en"])
    C = CLASS_LABELS.get(q_lang, CLASS_LABELS["en"])
    sep, sep2 = ("، ", "؛ ") if q_lang == "fa" else (", ", "; ")

    def label(c):
        return " + ".join(C.get(x, x) for x in c.split("+"))

    notes = []
    if any(r.get("period_inferred") for r in rec["series"]):
        notes.append("* " + L["inferred"])
    if rec.get("comparable") is False:
        cl = sep2.join("%s: %s" % (label(c), sep.join(ps)) for c, ps in rec["definition_classes"].items())
        notes.append(L["not_comparable"].format(classes=cl))
        for c, pts in rec["class_points"].items():
            if len(pts) >= 2:
                notes.append(L["same_label"].format(label=label(c), points=sep.join("%s: %s" % (p, v) for p, v in pts)))
    elif rec.get("comparable") is True and rec.get("main_class"):
        notes.append(L["comparable"].format(label=label(rec["main_class"])))
    for p in rec.get("several_values_periods") or []:
        notes.append(L["several"].format(period=p))
    for p in rec.get("latest_restatement_periods") or []:
        rr = [r for r in rec["series"] if str(r["period_year"]) == str(p)]
        if rr:
            notes.append(L["latest"].format(period=p, doc_year=(rr[0]["doc_years"] or ["?", "?"])[1]))
    for c in rec.get("conflicts") or []:
        restated_year = next((r["doc_years"][1] for r in rec["restatements"] if r["doc_id"] == c["restated_doc"]
                              and r.get("doc_years")), "?")
        notes.append(L["restated"].format(doc_year=restated_year, period=c["period"], value=c["restated"],
                                          kept=", ".join(c["kept"])))
    if rec.get("periods_missing"):
        notes.append(L["missing"].format(periods=", ".join(rec["periods_missing"])))
    if rec["kind"] in ("timeline", "comparison"):
        notes.append(L["rule"])
    return notes
