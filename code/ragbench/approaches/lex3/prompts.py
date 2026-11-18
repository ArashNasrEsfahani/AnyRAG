"""LEX3's two retrieval prompts (the answer prompt is the shared one, contracts.build_answer_messages).

    call 1  interpret   entities, explicit periods, operation, <= 3 facets (strict schema + validation)
    call 2  pick        phrase IDs per facet from a list of phrases that really occur in the retrieved evidence,
                        plus a bounded number of added terms (kept, but marked as NOT observed)

Both are parsed tolerantly (prose around the JSON, a bare list) and validated strictly: nothing is trusted as a
phrase ID until it names a listed phrase, nothing as a facet until it has a usable term. Every problem found is
returned, so the trace says what the model got wrong instead of silently dropping it.
"""
import re

from ..lex.prompts import LANG, parse_json

_FA_DIGITS = str.maketrans("۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩", "01234567890123456789")

OPERATIONS = ("lookup", "compare", "series", "list", "aggregate", "explain", "other")

INTERPRET_SYSTEM = ("You help search company, market and sector reports. You read a question and say precisely what "
                    "it asks for, so that code can find every part of the answer. Reply with JSON only.")

INTERPRET_USER = """Question: {question}

The reports are written in {dlang}. Return:
- "entities": every company, subsidiary, plant, mine, region, business unit, product or country the question names,
  as the report would write it (in {dlang}). Keep distinguishing names exactly: a subsidiary or a plant is not its
  parent company.
- "periods": the years or periods the question states explicitly (e.g. 1399, 2023, 2021-2023). None if it states none.
- "operation": one of {ops}.
- "facets": at most {fmax} things the question asks to be found (a measure or topic, e.g. revenue, number of
  employees, steel output). Each is {{"name": short label, "terms": 1 to {tmax} short search terms in {dlang} (1 to 3
  words each) - the report's likely wording, synonyms, the formal term}}. No company names, no years and no words
  every report uses about itself (report, annual, company, data) in the terms.
- "answer": "number" if the answer is a number or a series of numbers, else "text"."""

INTERPRET_SCHEMA = {"type": "object", "properties": {
    "entities": {"type": "array", "items": {"type": "string"}},
    "periods": {"type": "array", "items": {"type": "string"}},
    "operation": {"type": "string", "enum": list(OPERATIONS)},
    "facets": {"type": "array", "items": {"type": "object", "properties": {
        "name": {"type": "string"},
        "terms": {"type": "array", "items": {"type": "string"}}},
        "required": ["name", "terms"]}},
    "answer": {"type": "string", "enum": ["number", "text"]}},
    "required": ["entities", "periods", "operation", "facets", "answer"]}

PICK_SYSTEM = ("You help search company reports. You are shown phrases that really occur in passages retrieved for a "
               "question, each with an ID. For each thing the question asks for, you choose the IDs of the phrases "
               "that name exactly that thing. Reply with JSON only.")

PICK_USER = """Question: {question}

What the question asks for (facet number: name - search terms):
{facets}

Phrases found in the retrieved report passages (ID: phrase [where]):
{listing}

Return "facets": one entry per facet, {{"facet": facet number, "phrases": the IDs (e.g. "P3") of at most {pmax}
phrases that name what that facet asks for, "add": at most {amax} extra short search terms in {dlang}, only if no
listed phrase uses the report's wording}}. Prefer the report's own name for the measure, even if the question words it
differently. Leave out phrases about something else (another measure, another company, another year)."""

PICK_SCHEMA = {"type": "object", "properties": {
    "facets": {"type": "array", "items": {"type": "object", "properties": {
        "facet": {"type": "integer"},
        "phrases": {"type": "array", "items": {"type": "string"}},
        "add": {"type": "array", "items": {"type": "string"}}},
        "required": ["facet", "phrases", "add"]}}},
    "required": ["facets"]}


# ---------------------------------------------------------------------------------------------------------------
# messages
# ---------------------------------------------------------------------------------------------------------------

def interpret_messages(question, d_lang, o):
    return [{"role": "system", "content": INTERPRET_SYSTEM},
            {"role": "user", "content": INTERPRET_USER.format(question=question.strip(), dlang=LANG[d_lang],
                                                              ops=", ".join(OPERATIONS), fmax=int(o["max_facets"]),
                                                              tmax=int(o["facet_terms_max"]))}]


def facet_listing(facets):
    return "\n".join("%d: %s - %s" % (i, f["name"], ", ".join(f["terms"])) for i, f in enumerate(facets, 1))


def phrase_listing(vocab):
    return "\n".join("%s: %s [%s]" % (v["id"], v["text"], v["where"]) for v in vocab) or "(none)"


def pick_messages(question, facets, vocab, d_lang, o):
    return [{"role": "system", "content": PICK_SYSTEM},
            {"role": "user", "content": PICK_USER.format(question=question.strip(), facets=facet_listing(facets),
                                                         listing=phrase_listing(vocab), pmax=int(o["pick_max"]),
                                                         amax=int(o["add_max"]), dlang=LANG[d_lang])}]


# ---------------------------------------------------------------------------------------------------------------
# validation
# ---------------------------------------------------------------------------------------------------------------

def _strings(x, max_n=None, max_words=None, problems=None, what=""):
    out = []
    items = x if isinstance(x, list) else [x] if isinstance(x, str) else []
    if x is not None and not isinstance(x, (list, str)) and problems is not None:
        problems.append("%s: not a list" % what)
    for v in items:
        if not isinstance(v, (str, int, float)) or isinstance(v, bool):
            if problems is not None:
                problems.append("%s: non-string item dropped" % what)
            continue
        s = " ".join(str(v).split())
        if not s:
            continue
        if max_words and len(s.split()) > max_words:
            if problems is not None:
                problems.append("%s: %r longer than %d words dropped" % (what, s[:40], max_words))
            continue
        if s in out:
            continue
        out.append(s)
        if max_n and len(out) >= max_n:
            break
    return out


def years_of(text):
    """Explicit 4-digit years (Persian or Latin digits) in a period string, ranges expanded by the caller."""
    t = str(text or "").translate(_FA_DIGITS)
    return [m for m in re.findall(r"(?<!\d)(1[34]\d\d|19\d\d|20\d\d)(?!\d)", t)]


def validate_interpretation(text, o):
    """-> (interp, problems, how). interp = {entities, periods (years as strings), periods_raw, operation, facets:
    [{name, terms}], answer, empty}. `empty` is True when there is nothing to plan on (no facet and no entity): the
    session then runs in degraded mode `empty_plan` (question channels only, no call 2)."""
    obj, how = parse_json(text)
    problems = []
    if how == "failed":
        problems.append("no JSON")
    if isinstance(obj, list):
        problems.append("a list where an object was asked")
        obj = next((x for x in obj if isinstance(x, dict)), {})
    if not isinstance(obj, dict):
        obj = {}
    tmw = int(o["term_max_words"])
    entities = _strings(obj.get("entities") if "entities" in obj else obj.get("names"), int(o["max_entities"]), 8,
                        problems, "entities")
    periods_raw = _strings(obj.get("periods") if "periods" in obj else obj.get("years"), None, 6, problems, "periods")
    periods = []
    for p in periods_raw:
        for y in years_of(p):
            if y not in periods:
                periods.append(y)
    periods = periods[: int(o["max_periods"])]
    op = str(obj.get("operation") or "").strip().lower()
    if op not in OPERATIONS:
        if op:
            problems.append("operation %r not one of %s" % (op[:20], "/".join(OPERATIONS)))
        op = "other"
    facets = []
    raw = obj.get("facets")
    if raw is None and obj.get("keywords"):
        raw = [{"name": "", "terms": obj.get("keywords")}]
        problems.append("keywords given instead of facets")
    if raw is not None and not isinstance(raw, list):
        problems.append("facets: not a list")
        raw = []
    for i, f in enumerate(raw or []):
        if isinstance(f, str):
            f = {"name": f, "terms": [f]}
        elif isinstance(f, list):
            f = {"name": "", "terms": f}
        if not isinstance(f, dict):
            problems.append("facet %d: not an object" % (i + 1))
            continue
        terms = _strings(f.get("terms"), int(o["facet_terms_max"]), tmw, problems, "facet %d terms" % (i + 1))
        name = " ".join(str(f.get("name") or "").split())[:60]
        if not terms and name and len(name.split()) <= tmw:
            terms = [name]
        if not terms:
            problems.append("facet %d: no usable term, dropped" % (i + 1))
            continue
        facets.append({"name": name or terms[0], "terms": terms})
        if len(facets) >= int(o["max_facets"]):
            if len(raw) > len(facets):
                problems.append("facets capped at %d" % int(o["max_facets"]))
            break
    ans = str(obj.get("answer") or "").strip().lower()
    interp = {"entities": entities, "periods": periods, "periods_raw": periods_raw, "operation": op, "facets": facets,
              "answer": ans if ans in ("number", "text") else "number"}
    interp["empty"] = not facets and not entities
    return interp, problems, how


def _phrase_id(v, ids):
    s = str(v).strip().upper()
    if re.fullmatch(r"\d+", s):
        s = "P" + s
    return s if s in ids else None


def validate_pick(text, n_facets, vocab, o):
    """-> (got, problems, how). got = {"by_facet": {facet index 0-based: {"phrases": [ids], "add": [terms]}},
    "shape"}. IDs must name a listed phrase; facets must be in range. A flat {"phrases"/"pick": [...]} reply is applied
    to every facet (shape "flat") - the model named phrases but not whose they are."""
    obj, how = parse_json(text)
    problems = []
    ids = {v["id"] for v in vocab}
    pmax, amax, tmw = int(o["pick_max"]), int(o["add_max"]), int(o["term_max_words"])
    by = {}
    shape = "none"

    def take(fi, phrases, add, tag):
        slot = by.setdefault(fi, {"phrases": [], "add": []})
        for v in phrases if isinstance(phrases, list) else [phrases] if phrases is not None else []:
            pid = _phrase_id(v, ids)
            if pid is None:
                problems.append("%s: unknown phrase id %r" % (tag, str(v)[:12]))
                continue
            if pid not in slot["phrases"] and len(slot["phrases"]) < pmax:
                slot["phrases"].append(pid)
        for a in _strings(add, amax, tmw, problems, "%s add" % tag):
            if a not in slot["add"] and len(slot["add"]) < amax:
                slot["add"].append(a)

    if isinstance(obj, dict) and isinstance(obj.get("facets"), list):
        shape = "facets"
        for j, f in enumerate(obj["facets"]):
            if not isinstance(f, dict):
                problems.append("entry %d: not an object" % (j + 1))
                continue
            try:
                fi = int(f.get("facet", j + 1)) - 1
            except (TypeError, ValueError):
                problems.append("entry %d: facet number %r" % (j + 1, f.get("facet")))
                continue
            if not 0 <= fi < n_facets:
                problems.append("entry %d: facet %d out of range" % (j + 1, fi + 1))
                continue
            take(fi, f.get("phrases", f.get("ids", f.get("pick"))), f.get("add"), "facet %d" % (fi + 1))
    elif isinstance(obj, dict) and any(k in obj for k in ("phrases", "pick", "ids")):
        shape = "flat"
        val = next(obj[k] for k in ("phrases", "pick", "ids") if k in obj)
        for fi in range(n_facets):
            take(fi, val, obj.get("add"), "flat")
    elif isinstance(obj, list):
        shape = "flat"
        for fi in range(n_facets):
            take(fi, obj, None, "flat")
    else:
        problems.append("no facets in the reply")
    return {"by_facet": by, "shape": shape}, problems, how
