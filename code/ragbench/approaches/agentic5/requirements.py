"""Stage 1 of AGENTIC5: the REQUIREMENT MATRIX.

A requirement is one requested component of the answer - "company A, revenue, 1402" - with an id (R1, R2, ...), an
entity, a metric (with its own search vocabulary), a period (or None when the question names none) and the
requested operation (lookup | series | comparison | difference | ratio | sum | list). Verification, reconciliation,
gap filling, arithmetic and the evidence table are all organised by these ids.

The two rules the reviewer asked for, and how they hold here:

  * REQUIRED PERIODS COME FROM THE QUESTION AND THE PLAN, NEVER ONLY FROM THE SELECTED DOCUMENTS. AGENTIC3's
    `target_periods` read the years of the documents it had chosen, so an omitted report removed its year from the
    target itself. Here the periods are parsed from the question text (years, ranges, quarters, halves, "سال 98"),
    the plan may add periods only when they are consistent with the question, and nothing about documents enters.
    A requirement whose report is not in scope stays in the matrix and ends as `missing`.
  * THE PROGRAM FALLBACK IS A FIRST-CLASS MODE. A malformed, empty, cut or wrongly typed plan never removes the
    question's own entities and periods: `build_requirements` works from `{}` and from any junk the model returns,
    field by field, and records every repair in `fixes`.

Vocabulary is SHARED across requirements (the reviewer: do not copy AGENTIC3's rule that removes a term already used
by an earlier facet). Every requirement of one metric carries the same terms; entities and periods differ.
"""
import itertools
import re

from ...normalize import fold, normalize
from ..agentic import planner as P
from ..agentic.docmeta import companies as company_groups
from ..agentic.docmeta import match_companies
from ..agentic3 import approach as A3
from .defaults import OPERATIONS

_YEAR_RX = r"(?:1[34][0-9]{2}|19[0-9]{2}|20[0-9]{2})"
_YEAR = re.compile(r"(?<![0-9])(%s)(?![0-9])" % _YEAR_RX)
_RANGE = re.compile(r"(?:(?:from|از(?:\s+سال)?)\s*)?(?<![0-9])(%s)(?![0-9])\s*(?:-|–|—|to|until|through|تا|الی)"
                    r"\s*(?:سال\s*)?(?<![0-9])(%s)(?![0-9])|\bbetween\s+(%s)\s+and\s+(%s)\b"
                    % (_YEAR_RX, _YEAR_RX, _YEAR_RX, _YEAR_RX), re.I)
_SHORT_YEAR = re.compile(r"سال\s?([0-9]{2})(?![0-9])")

_ORD = {"first": 1, "1st": 1, "second": 2, "2nd": 2, "third": 3, "3rd": 3, "fourth": 4, "4th": 4,
        "اول": 1, "نخست": 1, "دوم": 2, "سوم": 3, "چهارم": 4}
_Q_EN = [re.compile(r"\bQ([1-4])\s*(?:FY\s*)?'?(%s)\b" % _YEAR_RX, re.I),
         re.compile(r"\b(%s)\s*Q([1-4])\b" % _YEAR_RX, re.I),
         re.compile(r"\b(first|1st|second|2nd|third|3rd|fourth|4th)\s+quarter\s+(?:of\s+)?(?:FY\s*)?(%s)\b" % _YEAR_RX,
                    re.I)]
_Q_FA = re.compile(r"(?:سه[‌ ]?ماهه|فصل)\s*(اول|نخست|دوم|سوم|چهارم)\s*(?:سال\s*)?(%s)" % _YEAR_RX)
_H_EN = [re.compile(r"\bH([12])\s*(?:FY\s*)?'?(%s)\b" % _YEAR_RX, re.I),
         re.compile(r"\b(first|1st|second|2nd)\s+half\s+(?:of\s+)?(?:FY\s*)?(%s)\b" % _YEAR_RX, re.I)]
_H_FA = re.compile(r"(?:نیمه|شش[‌ ]?ماهه)\s*(اول|نخست|دوم)\s*(?:سال\s*)?(%s)" % _YEAR_RX)

_OP_CUES = (
    ("ratio", re.compile(r"\bratio\b|how many times|\btimes (?:as|larger|higher|bigger)|as a (?:share|percentage) of|"
                         r"نسبت|چند برابر", re.I)),
    ("difference", re.compile(r"\bdifference\b|how much (?:more|less|higher|lower|larger|smaller)|\bgap between\b|"
                              r"\bby how much\b|\bincrease[ds]? (?:from|by)\b|\bdecrease[ds]? (?:from|by)\b|"
                              r"تفاوت|اختلاف|چقدر (?:بیشتر|کمتر)|چه میزان (?:افزایش|کاهش)|چقدر (?:افزایش|کاهش)", re.I)),
    ("sum", re.compile(r"\bcombined\b|\bin total\b|\btotal of\b|\bsum of\b|\btogether\b|جمع|مجموعاً|روی هم|در مجموع", re.I)),
    ("list", re.compile(r"\blist\b|\bwhich (?:companies|products|projects|countries)\b|\bname the\b|"
                        r"کدام (?:شرکت|محصول|پروژه|کشور)|نام ببرید|فهرست", re.I)),
    ("comparison", re.compile(r"compare|comparison|versus|\bvs\.?\b|higher|lower|larger|smaller|more than|less than|"
                              r"مقایسه|بیشتر|کمتر|در برابر", re.I)),
)

PLAN_RULES = (
    '- "entities": every company, organisation, subsidiary, plant, region or business unit the question names, one '
    'string each, as named in the question (keep the words that tell two of them apart). [] when it names none.\n'
    '- "metrics": each DIFFERENT thing to be measured, one object each {"name": "...", "terms": [2 to 6 short '
    'search terms in Persian and English]}. Usually one.\n'
    '- "periods": every period the question names, one string each ("1402", "2023", "Q1 2023", "first half '
    '1401"); for a range give the first and the last year. [] when it names none.\n'
    '- "operation": "lookup" (one value), "series" (one value over several periods), "comparison" (values of '
    'several entities or periods side by side), "difference", "ratio" or "sum" (a result computed from values), or '
    '"list" (several named items).')


# ---------------------------------------------------------------------------------------------------------------
# the plan prompt and schema (AGENTIC3's, the four fields appended LAST so earlier fields decode unchanged)
# ---------------------------------------------------------------------------------------------------------------

def plan_messages(question, metas, choose_companies, opts):
    msgs = A3.plan_messages(question, metas, choose_companies, opts)
    msgs[0]["content"] = msgs[0]["content"] + "\n" + PLAN_RULES
    return msgs


def plan_schema(company_keys, opts):
    schema = A3.plan_schema(company_keys, opts)
    props = schema["properties"]
    props["entities"] = {"type": "array", "items": {"type": "string"}, "maxItems": int(opts["max_entities"])}
    props["metrics"] = {"type": "array", "maxItems": int(opts["max_metrics"]),
                        "items": {"type": "object",
                                  "properties": {"name": {"type": "string"},
                                                 "terms": {"type": "array", "items": {"type": "string"},
                                                           "maxItems": int(opts["max_metric_terms"])}},
                                  "required": ["name", "terms"]}}
    props["periods"] = {"type": "array", "items": {"type": "string"}, "maxItems": int(opts["max_periods"])}
    props["operation"] = {"type": "string", "enum": list(OPERATIONS)}
    schema["required"] += ["entities", "metrics", "periods", "operation"]
    return schema


# ---------------------------------------------------------------------------------------------------------------
# periods
# ---------------------------------------------------------------------------------------------------------------

def period(year, ptype="annual", n=None):
    year = int(year)
    if ptype == "quarter":
        return {"key": "%dQ%d" % (year, n), "year": year, "type": "quarter", "n": int(n)}
    if ptype == "half":
        return {"key": "%dH%d" % (year, n), "year": year, "type": "half", "n": int(n)}
    return {"key": str(year), "year": year, "type": "annual", "n": None}


def _ordn(s):
    return _ORD.get(str(s).lower(), None) if not str(s).isdigit() else int(s)


def parse_periods(text, max_periods=10):
    """Periods named in a text -> (list of period dicts in order, info). A range becomes its years (inclusive, at most
    `max_periods`; a longer range keeps its two ends). Sub-annual periods (quarters, halves) are kept as such: an
    annual requirement is never satisfied by a quarterly value."""
    t = normalize(text or "")
    out, info = [], {"ranges": [], "sub_annual": []}
    used = []

    def add(p):
        if p["key"] not in {x["key"] for x in out}:
            out.append(p)

    for rx_list, ptype in ((_Q_EN, "quarter"), (_H_EN, "half")):
        for rx in rx_list:
            for m in rx.finditer(t):
                a, b = m.group(1), m.group(2)
                n, y = (_ordn(a), b) if not _YEAR.fullmatch(a) else (_ordn(b), a)
                if n:
                    add(period(y, ptype, n))
                    used.append(m.span())
                    info["sub_annual"].append(m.group(0))
    for rx, ptype in ((_Q_FA, "quarter"), (_H_FA, "half")):
        for m in rx.finditer(t):
            n = _ordn(m.group(1))
            if n:
                add(period(m.group(2), ptype, n))
                used.append(m.span())
                info["sub_annual"].append(m.group(0))

    def free(span):
        return not any(span[0] < b and a < span[1] for a, b in used)

    for m in _RANGE.finditer(t):
        if not free(m.span()):
            continue
        a, b = sorted((int(m.group(1) or m.group(3)), int(m.group(2) or m.group(4))))
        if abs(a - b) > 60:                   # two calendars (1402 and 2023) are not a range
            continue
        years = list(range(a, b + 1)) if (b - a + 1) <= int(max_periods) else [a, b]
        info["ranges"].append([a, b])
        for y in years:
            add(period(y))
        used.append(m.span())
    for m in _YEAR.finditer(t):
        if free(m.span()):
            add(period(m.group(1)))
    for m in _SHORT_YEAR.finditer(t):
        if free(m.span()):
            add(period(1300 + int(m.group(1))))
    return out[: max(int(max_periods), 2)], info


# ---------------------------------------------------------------------------------------------------------------
# operation
# ---------------------------------------------------------------------------------------------------------------

def infer_operation(question, kind=None, n_periods=0, n_entities=1):
    qn = normalize(question or "")
    for op, rx in _OP_CUES:
        if rx.search(qn):
            if op == "comparison" and n_entities <= 1 and n_periods <= 1:
                continue
            return op
    if kind == "timeline" or P._TIMELINE_CUES.search(qn):
        return "series"
    if kind == "comparison" or n_entities > 1:
        return "comparison"
    if n_periods > 1:
        return "series"
    return "lookup"


# ---------------------------------------------------------------------------------------------------------------
# entities and metrics
# ---------------------------------------------------------------------------------------------------------------

def _strings(v):
    if isinstance(v, str):
        return [v]
    if isinstance(v, list):
        return [x for x in v if isinstance(x, str)]
    return []


def entity_keys(name, metas):
    """Company keys of the listing that an entity name designates (key in the name, or distinctive title words)."""
    if not name or not metas:
        return []
    return match_companies(name, metas)


def build_entities(obj, plan, question, metas, opts, fixes):
    names, seen = [], set()
    raw = _strings(obj.get("entities")) if isinstance(obj, dict) else []
    if isinstance(obj, dict) and obj.get("entities") is not None and not isinstance(obj.get("entities"), list):
        fixes.append("entities_not_a_list")
    for n in raw + ([plan.get("entity")] if plan.get("entity") else []):
        n2 = normalize(n).strip()
        n2 = re.sub(r"\s+", " ", re.sub(r"[\"'`«»{}\[\]]+", " ", n2)).strip()
        if len(n2) < 2 or fold(n2) in seen:
            continue
        seen.add(fold(n2))
        names.append(n2)
    out = []
    comps = company_groups(metas)
    for n in names:
        keys = entity_keys(n, metas)
        out.append({"name": n, "keys": keys})
    # company keys the planner chose (or the question itself names) with no entity string yet
    named_keys = {k for e in out for k in e["keys"]}
    extra = [k for k in (plan.get("companies") or []) if k in comps and k not in named_keys]
    if not out:
        extra += [k for k in match_companies(question, metas) if k not in named_keys and k not in extra]
        if extra:
            fixes.append("entities_from_listing")
    for k in extra:
        out.append({"name": k, "keys": [k]})
    # one entity per company key: two strings naming the same company are one entity
    merged, by_key = [], {}
    for e in out:
        k = tuple(e["keys"]) if len(e["keys"]) == 1 else None
        if k and k in by_key:
            by_key[k]["aliases"].append(e["name"])
            continue
        e = dict(e, aliases=[])
        merged.append(e)
        if k:
            by_key[k] = e
    if len(merged) > int(opts["max_entities"]):
        fixes.append("entities_capped")
        merged = merged[: int(opts["max_entities"])]
    if not merged:
        merged = [{"name": "", "keys": [], "aliases": []}]
        fixes.append("entity_unspecified")
    for i, e in enumerate(merged, 1):
        e["id"] = "E%d" % i
        e["key"] = e["keys"][0] if len(e["keys"]) == 1 else (fold(e["name"]) or None)
        terms = [e["name"]] + e["aliases"] + list(e["keys"])
        e["terms"] = [t for t in dict.fromkeys(t for t in terms if t and len(t) >= 2)]
    return merged


def build_metrics(obj, plan, opts, fixes):
    raw = obj.get("metrics") if isinstance(obj, dict) else None
    if raw is not None and not isinstance(raw, list):
        fixes.append("metrics_not_a_list")
        raw = []
    ent_all = plan.get("entity_terms") or []
    out = []
    for m in (raw or [])[: int(opts["max_metrics"])]:
        if isinstance(m, str):
            m = {"name": m, "terms": [m]}
        if not isinstance(m, dict):
            fixes.append("metric_not_an_object")
            continue
        name = normalize(str(m.get("name") or "")).strip()
        values = _strings(m.get("terms")) + ([name] if name else [])
        terms, _dropped = P.clean_terms(values, int(opts["max_metric_terms"]), int(opts["max_term_chars"]),
                                        stop_extra=ent_all)
        terms, _rep = P.refine_terms(terms, ent_all, int(opts["max_metric_terms"]))
        if not (terms["fa"] or terms["en"]) and not name:
            fixes.append("metric_empty_dropped")
            continue
        out.append({"name": name, "terms": terms})
    if not out:
        out = [{"name": plan.get("metric") or "", "terms": {k: list(v) for k, v in (plan.get("terms") or {}).items()}}]
        if raw:
            fixes.append("metrics_from_plan_terms")
    # the plan's own terms are the question's vocabulary: with ONE metric they are shared by it (never removed)
    if len(out) == 1:
        for lang in ("fa", "en"):
            have = {fold(t) for t in out[0]["terms"].get(lang) or []}
            out[0]["terms"].setdefault(lang, [])
            out[0]["terms"][lang] += [t for t in (plan.get("terms") or {}).get(lang) or [] if fold(t) not in have]
    for i, m in enumerate(out, 1):
        m["id"] = "M%d" % i
        if not (m["terms"].get("fa") or m["terms"].get("en")):
            m["terms"] = {k: list(v) for k, v in (plan.get("terms") or {}).items()}
            fixes.append("metric_terms_from_plan")
        if not m["name"]:
            m["name"] = plan.get("metric") or " ".join((m["terms"].get("en") or m["terms"].get("fa") or [""])[:1])
    return out


def build_periods(obj, question, opts, fixes):
    """The QUESTION's periods first; the plan may add a period only when it agrees with the question (it may resolve
    a quarter or a half the question words loosely); a plan period the question does not name is ignored (the
    planner copies the listing's years)."""
    qp, qinfo = parse_periods(question, int(opts["max_periods"]))
    raw = obj.get("periods") if isinstance(obj, dict) else None
    if raw is not None and not isinstance(raw, list):
        fixes.append("periods_not_a_list")
        raw = [raw] if isinstance(raw, (str, int)) else []
    plan_p = []
    for v in raw or []:
        if not isinstance(v, (str, int)):
            fixes.append("period_not_a_string")
            continue
        ps, _ = parse_periods(str(v), int(opts["max_periods"]))
        if not ps:
            fixes.append("period_unparsed")
        plan_p += ps
    q_years = {p["year"] for p in qp}
    lo_hi = [r for r in qinfo["ranges"]]
    out = list(qp)
    for p in plan_p:
        in_q = p["year"] in q_years or any(a <= p["year"] <= b for a, b in lo_hi)
        if not in_q:
            fixes.append("plan_period_not_in_question")
            continue
        if p["key"] in {x["key"] for x in out}:
            continue
        # a sub-annual plan period refines the question's bare year of the same year ("the first three months of
        # 2023" -> Q1 2023), unless the plan asks for that annual period as well
        if p["type"] != "annual" and not any(x["type"] == "annual" and x["year"] == p["year"] for x in plan_p):
            before = len(out)
            out = [x for x in out if not (x["type"] == "annual" and x["year"] == p["year"])]
            if len(out) < before:
                fixes.append("annual_refined_to_%s" % p["type"])
        out.append(p)
    return out[: int(opts["max_periods"])], qinfo


# ---------------------------------------------------------------------------------------------------------------
# the matrix
# ---------------------------------------------------------------------------------------------------------------

def build_requirements(obj, plan, question, metas, opts):
    """Model object (anything, possibly junk) + the validated v1 plan -> the requirement matrix.

    -> {"entities", "metrics", "periods", "operation", "requirements", "open_series", "fixes", "truncated"}
    """
    fixes = []
    obj = obj if isinstance(obj, dict) else {}
    entities = build_entities(obj, plan, question, metas, opts, fixes)
    metrics = build_metrics(obj, plan, opts, fixes)
    periods, pinfo = build_periods(obj, question, opts, fixes)
    op = obj.get("operation")
    named_entities = [e for e in entities if e["name"] or e["keys"]]
    if op not in OPERATIONS:
        if op is not None:
            fixes.append("operation_invalid")
        op = infer_operation(question, plan.get("kind"), len(periods), len(named_entities))
        fixes.append("operation_inferred")
    n_cells = max(1, len(entities)) * max(1, len(metrics)) * max(1, len(periods))
    if op in ("difference", "ratio", "comparison") and n_cells < 2:
        # a computed or side-by-side answer needs two operands; with one named cell it is a lookup (the other
        # operand is often "the previous year" or "the market", which the question states only implicitly)
        if op != "comparison" or len(periods) <= 1:
            fixes.append("operation_%s_with_one_operand" % op)
    if op == "series" and len(periods) == 1:
        fixes.append("series_with_one_period")
    open_series = op == "series" and not periods
    cells = []
    per = periods or [None]
    for e, m, p in itertools.product(entities, metrics, per):
        cells.append((e, m, p))
    cap = int(opts["max_requirements"])
    truncated = max(0, len(cells) - cap)
    if truncated:
        fixes.append("requirements_capped")
    reqs = []
    for i, (e, m, p) in enumerate(cells[:cap], 1):
        reqs.append({"id": "R%d" % i, "entity_id": e["id"], "entity": e["name"], "entity_key": e["key"],
                     "entity_keys": list(e["keys"]), "entity_terms": list(e["terms"]),
                     "metric_id": m["id"], "metric": m["name"], "terms": {k: list(v) for k, v in m["terms"].items()},
                     "period": dict(p) if p else None, "period_key": p["key"] if p else "",
                     "operation": op, "unit": plan.get("unit") or "count"})
    return {"entities": entities, "metrics": metrics, "periods": periods, "period_info": pinfo, "operation": op,
            "requirements": reqs, "open_series": open_series, "fixes": fixes, "truncated": truncated}


def requirement_plan(plan, req):
    """A v1-shaped plan for ONE requirement: its metric vocabulary, its entity, its period - the input of the
    database retrieval and of the gap query (the requirement's own wording, not the question's)."""
    per = req.get("period")
    return dict(plan, terms={k: list(v) for k, v in (req.get("terms") or {}).items()},
                entity=req.get("entity") or plan.get("entity"),
                entity_terms=list(dict.fromkeys((req.get("entity_terms") or []) + list(plan.get("entity_terms") or []))),
                metric=req.get("metric") or plan.get("metric"),
                period=(per["year"], per["year"]) if per else (None, None), facets=[],
                companies=list(req.get("entity_keys") or []))


def summary(matrix):
    """One line per requirement, for the trace and the verify prompt."""
    out = []
    for r in matrix["requirements"]:
        out.append("%s: entity=%s; metric=%s; period=%s" % (r["id"], r["entity"] or "(as in the question)",
                                                           r["metric"] or "(as in the question)",
                                                           r["period_key"] or "(any / latest)"))
    return out
