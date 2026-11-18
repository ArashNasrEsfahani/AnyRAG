"""Stage 4 of AGENTIC5: TYPED verification against the requirement matrix.

The verifier reads a batch of snippets and the list of requirements, and returns per snippet the values it states
with their REQUIREMENT ID, entity, metric, period, definition/scope, value, unit/scale and a verbatim quote. Code
then decides what each value is, and the model's words are never taken as proof:

  quote / value     v1's verbatim guard (`agentic/verify.py::guard`), unchanged: the quote is in the snippet, the
                    value is in the quote, not a bare year, not an axis tick.
  metric            the quote must carry a term of the BOUND requirement's metric. The AGENTIC3 escape that
                    accepted a quote without the term when the verifier's own DEFINITION contained it is REMOVED
                    (reviewer, item 7: a model-written definition is not evidence the source expresses the metric).
                    What remains is source text only: the knowledge base's fact LABEL may carry the term, and only
                    when that label itself stands verbatim in the snippet; and v2's label completion, which adds the
                    label LINE of the snippet to a number-only quote and re-runs the whole guard.
  entity            the value belongs to the requirement's entity when the snippet's document is that company's
                    report, or the quote / snippet names the entity; otherwise it is re-bound to the requirement
                    whose entity it does name, or left UNBOUND (kept, never silently attributed).
  period            parsed with its TYPE: an annual requirement is never satisfied by a quarter or a half.
  unit / scale      the unit's words must stand in the snippet, otherwise the unit is `unverified` and the value
                    can be shown but never used as an arithmetic operand.

A claim is AGENTIC3's row (`reconcile.make_row`) plus the typed fields; the reconciler (`reconcile5.py`) partitions
claims by (entity, metric, period, definition, unit family) and never by year alone.
"""
import json
import re

from ...kb.typed import parse_unit
from ...normalize import fold, normalize
from ..agentic import planner as P
from ..agentic import reconcile as R
from ..agentic import verify as V
from ..agentic.numbers import find_numbers
from ..agentic2 import tree as T
from .requirements import parse_periods

VERIFY5_SYSTEM = """You check candidate snippets from reports against the REQUIREMENTS of one question. The snippets were found by a search and most of them do NOT state a required value.

Reply with one JSON object with one key per snippet, "s1" for snippet [1], "s2" for snippet [2], and so on, none skipped:
{"s1": {"relevant": true or false, "values": [{"req": "R1", "entity": "...", "metric": "...", "period": "...", "definition": "...", "value": "...", "unit": "...", "quote": "..."}]}, "s2": {...}, ...}
"values" is [] when the snippet states no required value. Give one entry per value (at most 3 per snippet).

For each value:
- req: the id of the requirement the value answers, exactly as listed; "" when it answers none of them exactly.
- entity: the company, subsidiary or unit the snippet says the value belongs to, as written in the snippet.
- metric: the name of what is measured, as written in the snippet.
- period: the year, quarter, half or date the value refers to, as written in the snippet; "" if the snippet does not say.
- definition: what exactly is counted or measured, keeping every qualifier the snippet gives (scope, segment, product, region, active or total).
- value: the value exactly as written in the snippet (same digits, separators and words). Never compute, convert, round, add up or translate numbers.
- unit: the unit and scale words written next to the value (for example "million USD", "میلیارد ریال", "%", "tonnes"), or "".
- quote: the exact text from the snippet that contains both the label and the value, copied character by character. It may run across line breaks.

A value is NOT a required value when it belongs to a different entity, measures a different metric, is only a part or segment while the whole is asked, covers a different kind of period (a quarter or a half when a year is asked), or is a number on a chart axis."""

VERIFY5_USER = """Question: {question}

Requirements (id: entity; metric; period):
{requirements}

Snippets:
{snippets}"""

VALUE_FIELDS = ("req", "entity", "metric", "period", "definition", "value", "unit", "quote")
MAX_ITEMS_PER_SNIPPET = 3
_AR_THOUSANDS = re.compile("(?<=[0-9])،(?=[0-9]{3}(?![0-9]))")


def verify_schema(n):
    value = {"type": "object", "properties": {k: {"type": "string"} for k in VALUE_FIELDS},
             "required": list(VALUE_FIELDS)}
    entry = {"type": "object",
             "properties": {"relevant": {"type": "boolean"},
                            "values": {"type": "array", "items": value, "maxItems": MAX_ITEMS_PER_SNIPPET}},
             "required": ["relevant", "values"]}
    keys = ["s%d" % i for i in range(1, n + 1)]
    return {"type": "object", "properties": {k: entry for k in keys}, "required": keys}


def verify_messages(question, req_lines, batch, corpus):
    parts = ["[%d] (%s)\n%s" % (i, V.snippet_header(c, corpus), c.text.strip()) for i, c in enumerate(batch, 1)]
    user = VERIFY5_USER.format(question=normalize(question), requirements="\n".join(req_lines) or "(none)",
                               snippets="\n\n".join(parts))
    return [{"role": "system", "content": VERIFY5_SYSTEM}, {"role": "user", "content": user}]


def parse_items(text, n):
    """The keyed reply -> ({i: [item dicts]}, how). v1's tolerant parser (it passes every extra field through)."""
    return V.parse_items(text, n)


def fix_thousands(batch):
    """v2's repair: a Persian thousands separator written as the Arabic comma is shown as a comma."""
    fixed = []
    for c in batch:
        t = _AR_THOUSANDS.sub(",", c.text)
        if t != c.text:
            c.text, c.span = t, None
            fixed.append(c.cid)
    return fixed


# ---------------------------------------------------------------------------------------------------------------
# term regexes per metric
# ---------------------------------------------------------------------------------------------------------------

def term_rx(terms):
    rx = []
    for lang in ("fa", "en"):
        for t in (terms or {}).get(lang) or []:
            for p in P.term_patterns(t, lang):
                try:
                    rx.append(re.compile(p, re.I))
                except re.error:
                    continue
    return rx


def has_term(text, rxs):
    t = normalize(text or "")
    return any(rx.search(t) for rx in rxs)


def source_label(cand):
    """The KB fact label of a candidate, only when it stands verbatim in the snippet (then it is source text)."""
    label = str(((cand.features or {}).get("fact") or {}).get("label") or "")
    if label and V.canon(label) and V.canon(label) in V.canon(cand.text):
        return label
    return ""


def complete_label(item, cand, unit_kind, rxs, min_letters):
    """v2's label completion: a number-only quote plus the nearest label LINE of the snippet, the whole guard re-run.
    -> (quote, ginfo) or (None, None)."""
    lines = cand.text.split("\n")
    cq = V.canon(item.get("quote") or "")
    idx = next((i for i, ln in enumerate(lines) if cq and cq in V.canon(ln)), None)
    if idx is None:
        return None, None
    for j in (idx - 1, idx - 2, idx + 1, idx + 2):
        if not (0 <= j < len(lines)) or not lines[j].strip() or not T._label_only(lines[j], unit_kind):
            continue
        quote = (lines[j] + "\n" + lines[idx]) if j < idx else (lines[idx] + "\n" + lines[j])
        it = dict(item, quote=quote)
        ok, _reason, ginfo = V.guard(it, cand.text, unit_kind=unit_kind, min_quote_letters=min_letters,
                                     term_rx=rxs or None)
        if ok:
            return quote, ginfo
    return None, None


# ---------------------------------------------------------------------------------------------------------------
# typed fields
# ---------------------------------------------------------------------------------------------------------------

def claim_period(item_period, quote, doc_years):
    """-> (period dict | None, how). The period text first (with its type), then v1's rules (a year in the quote
    inside the document's years, else the document's year, marked inferred)."""
    ps, _ = parse_periods(item_period or "", 4)
    if ps:
        return ps[-1], "period"
    qs, _ = parse_periods(quote or "", 6)
    sub = [p for p in qs if p["type"] != "annual"]
    if sub and (not doc_years or doc_years[0] <= sub[-1]["year"] <= doc_years[1]):
        return sub[-1], "quote_sub_annual"
    y, how = R.parse_period(item_period, quote, doc_years)
    if y is None:
        return None, how
    return {"key": str(y), "year": y, "type": "annual", "n": None}, how


def unit_fields(unit_text, quote, snippet):
    """-> {unit_kind, unit_norm, currency, scale, unit_status}: `verified` when every word of the unit stands in
    the quote or the snippet, `empty` for no unit, `unverified` otherwise (shown, never an operand)."""
    u = parse_unit(unit_text or "")
    cu = V.canon(unit_text or "")
    if not cu:
        status = "empty"
    else:
        hay = V.canon(quote or "") + " " + V.canon(snippet or "")
        status = "verified" if all(w in hay for w in cu.split()) else "unverified"
    return dict(u, unit_status=status)


def value_norm(ginfo, unit):
    """The value in base units: the source token's own reading (its scale word already applied by the guard) times
    the scale words of the unit field when the token carried none; toman -> rial. None for a text value."""
    v = (ginfo or {}).get("value_num")
    if v is None:
        return None
    if (ginfo or {}).get("reading") != "scaled":
        v = v * float(unit.get("scale") or 1.0)
    if unit.get("currency") == "toman":
        v = v * 10.0
    return v


def unit_family(unit):
    if unit.get("is_percent"):
        return "percent"
    kind = unit.get("unit_kind") or "count"
    cur = unit.get("currency")
    if cur in ("toman", "rial"):
        cur = "rial"
    return kind + (":" + cur if cur else "")


# ---------------------------------------------------------------------------------------------------------------
# binding a value to a requirement
# ---------------------------------------------------------------------------------------------------------------

def entity_matches(req, doc_meta, item, snippet, single_entity=False):
    """-> (ok, how). A requirement with no entity accepts any document; so does the ONLY entity of a matrix when it
    maps to no company key of the listing (then the entity is a label of the question, not a discriminator between
    requirements - a market report's "China" is rarely repeated next to each of its numbers)."""
    keys = [k for k in (req.get("entity_keys") or []) if k]
    terms = [t for t in (req.get("entity_terms") or []) if t and len(t) >= 2]
    if not keys and not terms:
        return True, "no_entity_named"
    comp = (doc_meta or {}).get("company")
    if comp and comp in keys:
        return True, "document_company"
    if single_entity and not keys:
        return True, "single_unkeyed_entity"
    title = fold(normalize((doc_meta or {}).get("title") or ""))
    if not keys and title and any(len(fold(normalize(t))) >= 3 and fold(normalize(t)) in title for t in terms):
        return True, "document_title"
    hay = fold(normalize((item.get("quote") or "") + " " + (snippet or "")))
    ent = fold(normalize(item.get("entity") or ""))
    for t in terms:
        ft = fold(normalize(t))
        if len(ft) >= 3 and ft in hay:
            return True, "named_in_snippet"
    # the verifier's entity string counts only when it is itself in the snippet
    if ent and ent in hay:
        for t in terms:
            ft = fold(normalize(t))
            if len(ft) >= 3 and (ft in ent or ent in ft):
                return True, "entity_field_in_snippet"
    if comp and keys:
        return False, "other_company_document"
    return False, "entity_not_in_snippet"


def period_matches(req, cper):
    """-> (ok, how). A requirement without a period accepts any; an annual requirement needs an annual value."""
    rp = req.get("period")
    if not rp:
        return True, "no_period_asked"
    if cper is None:
        return False, "claim_without_period"
    if rp["type"] != cper["type"]:
        return False, "period_type_mismatch"
    if rp["key"] == cper["key"]:
        return True, "same_period"
    return False, "other_period"


def bind(item, cand, reqs, rx_by_metric, doc_meta, cper, label):
    """-> (req or None, how, detail). The model's `req` first, checked; else the requirement whose entity, period
    and metric the SOURCE supports, when exactly one does. Metric evidence is the quote or the source label, never
    the model's definition."""
    by_id = {r["id"]: r for r in reqs}
    quote = item.get("quote") or ""
    single = len({r.get("entity_id") for r in reqs}) == 1

    def metric_ok(r):
        rxs = rx_by_metric.get(r["metric_id"]) or []
        return (not rxs) or has_term(quote, rxs) or (label and has_term(label, rxs))

    def fits(r):
        e_ok, e_how = entity_matches(r, doc_meta, item, cand.text, single_entity=single)
        p_ok, p_how = period_matches(r, cper)
        m_ok = metric_ok(r)
        return e_ok and p_ok and m_ok, {"entity": e_how, "period": p_how, "metric": m_ok}

    want = str(item.get("req") or "").strip().upper()
    detail = {"model_req": want}
    if want in by_id:
        ok, why = fits(by_id[want])
        detail["checks"] = why
        if ok:
            return by_id[want], "model", detail
    fitting = [r for r in reqs if fits(r)[0]]
    if want in by_id:
        same_metric = [r for r in fitting if r["metric_id"] == by_id[want]["metric_id"]]
        if len(same_metric) == 1:
            return same_metric[0], "rebound", detail
    if len(fitting) == 1:
        return fitting[0], ("program" if want not in by_id else "rebound"), detail
    if len(fitting) > 1:
        detail["ambiguous"] = [r["id"] for r in fitting]
        return None, "ambiguous", detail
    return None, (detail.get("checks") and "checks_failed") or "no_requirement_fits", detail


def make_claim(item, cand, ginfo, doc_meta, unit_id, req, how, detail, cper, per_how, corpus):
    """A typed claim: AGENTIC3's row (so every v1 field downstream code reads is there) plus the typed fields."""
    row = R.make_row(item, cand, ginfo, doc_meta or {}, unit_id, verified=True)
    unit = unit_fields(item.get("unit"), item.get("quote"), cand.text)
    ent_key = (req or {}).get("entity_key") or (doc_meta or {}).get("company") or ""
    row.update({
        "req_id": req["id"] if req else None, "bind": how, "bind_detail": detail,
        "entity_key": ent_key, "entity_id": (req or {}).get("entity_id"),
        "entity_name": (req or {}).get("entity") or (doc_meta or {}).get("company") or "",
        "entity_text": normalize(str(item.get("entity") or ""))[:120],
        "doc_company": (doc_meta or {}).get("company"),
        "metric_id": (req or {}).get("metric_id"), "metric_name": (req or {}).get("metric") or "",
        "metric_text": normalize(str(item.get("metric") or ""))[:120],
        "period_key": cper["key"] if cper else "", "period_type": cper["type"] if cper else None,
        "period_how": per_how, "period_inferred": per_how == "doc_year",
        "unit_kind": unit["unit_kind"], "unit_norm": unit["unit_norm"], "currency": unit["currency"],
        "scale": unit["scale"], "unit_status": unit["unit_status"], "unit_family": unit_family(unit),
        "value_norm": value_norm(ginfo, unit),
        "quote_span": quote_span(cand, item.get("quote"), corpus),
        "hit_unit_ids": list(cand.hit_unit_ids or []),
    })
    if cper is not None and cper["type"] == "annual":
        row["period_year"] = cper["year"]
        row["period"] = str(cper["year"])
    elif cper is not None:
        row["period"] = cper["key"]
    return row


def quote_span(cand, quote, corpus):
    """[start, end) of the quote's first line-piece in Document.text when the candidate is a verbatim slice."""
    if not cand.span or not quote:
        return None
    try:
        doc = corpus.doc(cand.doc_id)
    except KeyError:
        return None
    if doc.text[cand.span[0]:cand.span[1]] != cand.text:
        return None
    piece = normalize(quote).split("\n")[0].strip()
    k = cand.text.find(piece) if piece else -1
    if k < 0:
        return None
    return [cand.span[0] + k, cand.span[0] + k + len(piece)]


def dumps(obj):
    return json.dumps(obj, ensure_ascii=False)


def numeric_question(reqs):
    return any((r.get("unit") or "count") != "text" for r in reqs)


def text_value_tokens(value):
    return [t for t in find_numbers(normalize(str(value or ""))) if not t.is_date]
