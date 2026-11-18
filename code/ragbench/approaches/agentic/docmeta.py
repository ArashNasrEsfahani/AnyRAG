"""Document metadata the planner chooses target documents from (no search, no model).

Only what every approach already sees is used: the document id (its key encodes the publisher and the
year for corpora B and C, e.g. "C.snapp_annual-report_1402.fa") and the document title. Document.meta is
provenance for question authoring and is never read (OWNERSHIP fairness rule 1).

    metas = [doc_meta(d) for d in docs]           # {doc_id, key, company, years (y0, y1) | None, title}
    lines = company_listing(metas)                # "snapp: 1398, 1399, ... | گزارش عملکرد سالانه گروه اسنپ 1402"
    chosen, info = select_docs(metas, companies=["snapp"], period=(1398, 1404), kind="timeline", question=q)
"""
import re

from ...normalize import fold, normalize

_YEAR = re.compile(r"(?<![0-9])(1[34][0-9]{2}|19[0-9]{2}|20[0-9]{2})(?![0-9])")
_RANGE = re.compile(r"(?<![0-9])(1[34][0-9]{2}|19[0-9]{2}|20[0-9]{2})\s*[-_–]\s*(1[34][0-9]{2}|19[0-9]{2}|20[0-9]{2})(?![0-9])")
_GENERIC_TITLE = set("""گزارش عملکرد سالانه سالیانه گروه شرکت در سال و از به با برای the report annual performance group
of and in for year half h1 h2 fa en سال‌های نیمه اول دوم ویژه edition web""".split())


def doc_key(doc_id):
    parts = doc_id.split(".")
    return ".".join(parts[1:-1]) if len(parts) >= 3 else doc_id


def doc_years(text):
    """(y0, y1) from a key or title: a range 1398-1400, else the last year mentioned, else None."""
    t = normalize(text or "")
    m = None
    for m in _RANGE.finditer(t):
        pass
    if m:
        a, b = int(m.group(1)), int(m.group(2))
        return (min(a, b), max(a, b))
    ys = [int(y) for y in _YEAR.findall(t)]
    if ys:
        return (ys[-1], ys[-1])
    return None


def doc_meta(doc):
    key = doc_key(doc.id)
    company = re.split(r"[_\-]", key)[0] if key else key
    years = doc_years(key) or doc_years(doc.title)
    return {"doc_id": doc.id, "key": key, "company": company, "years": years, "title": normalize(doc.title or "")}


def companies(metas):
    """Ordered company keys with their documents."""
    out = {}
    for m in metas:
        out.setdefault(m["company"], []).append(m)
    return out


def company_listing(metas, max_lines=60):
    """One line per company: key, years, one title (the latest document's)."""
    lines = []
    comps = companies(metas)
    for i, (c, ms) in enumerate(comps.items()):
        if i >= max_lines:
            lines.append("... and %d more companies" % (len(comps) - max_lines))
            break
        ys = sorted({y for m in ms if m["years"] for y in range(m["years"][0], m["years"][1] + 1)})
        ms_sorted = sorted(ms, key=lambda m: (m["years"] or (0, 0)))
        title = ms_sorted[-1]["title"][:110]
        span = ", ".join(str(y) for y in ys) if ys else "no year"
        lines.append("%s: %s | %d document%s | %s" % (c, span, len(ms), "" if len(ms) == 1 else "s", title))
    return lines


def _tokens(text):
    return {w for w in re.split(r"[\s‌\-_.,:;()«»\"'|/]+", fold(text)) if len(w) >= 3}


def match_companies(question, metas):
    """Program fallback: companies whose key or distinctive title words occur in the question."""
    comps = companies(metas)
    q = _tokens(question)
    qf = fold(question)
    word_df = {}
    for c, ms in comps.items():
        for w in set().union(*[_tokens(m["title"]) for m in ms]):
            word_df[w] = word_df.get(w, 0) + 1
    hits = []
    for c, ms in comps.items():
        if len(c) >= 3 and c.lower() in qf:
            hits.append(c)
            continue
        words = set().union(*[_tokens(m["title"]) for m in ms]) - {fold(w) for w in _GENERIC_TITLE}
        distinctive = {w for w in words if word_df.get(w, 0) <= 2 and not w.isdigit()}
        if distinctive & q:
            hits.append(c)
    return hits


def _overlaps(years, period):
    if years is None or period is None:
        return True
    lo, hi = period
    if lo is None and hi is None:
        return True
    lo = lo if lo is not None else -10 ** 9
    hi = hi if hi is not None else 10 ** 9
    return years[0] <= hi and years[1] >= lo


def select_docs(metas, *, companies_chosen=None, period=None, kind="single", question="", max_docs=12,
                selection=True):
    """-> (chosen metas in year order, info). With one document in scope it is always chosen."""
    info = {"n_scope": len(metas), "companies_model": list(companies_chosen or []), "period": period}
    if len(metas) <= 1 or not selection:
        info["rule"] = "single_doc_scope" if len(metas) <= 1 else "selection_off"
        chosen = list(metas)
    else:
        comps = companies(metas)
        valid = [c for c in (companies_chosen or []) if c in comps]
        info["companies_valid"] = valid
        if not valid:
            valid = match_companies(question, metas)
            info["companies_fallback"] = valid
        pool = [m for m in metas if m["company"] in valid] if valid else list(metas)
        info["rule"] = "companies" if valid else "all_in_scope"
        in_period = [m for m in pool if _overlaps(m["years"], period)]
        if period and any(v is not None for v in period):
            if in_period:
                pool = in_period
                info["rule"] += "+period"
            else:
                info["period_matched_none"] = True
        chosen = pool
    chosen = sorted(chosen, key=lambda m: ((m["years"] or (9999, 9999)), m["doc_id"]))
    if len(chosen) > max_docs:
        info["truncated"] = [m["doc_id"] for m in chosen[max_docs:]]
        # keep the latest documents: later reports also restate earlier years
        chosen = chosen[-max_docs:]
    info["chosen"] = [m["doc_id"] for m in chosen]
    return chosen, info
