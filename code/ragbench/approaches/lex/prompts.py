"""LEX's two retrieval prompts (the answer prompt is the shared one, contracts.build_answer_messages).

Both ask for JSON under an Ollama schema, and both are parsed tolerantly: a model that returns prose around the JSON,
or a list where an object was asked, still yields what it said. Nothing a prompt returns is trusted as a regex until it
compiles; nothing is trusted as a phrase number until it is in range.
"""
import json
import re

LANG = {"fa": "Persian", "en": "English"}

KW_SYSTEM = ("You help search company, market and sector reports. You write the words a report itself would use, "
             "so a plain text search finds the passage that answers a question. Reply with JSON only.")

KW_USER = """Question: {question}

The reports are written in {dlang}. Give, in {dlang}:
- "keywords": {kmin} to {kmax} short search terms (1 to 3 words each) for WHAT the question asks about - the measure,
  the topic, and the other words a report might use for it (synonyms, the formal term, the everyday term).
  Most specific first. No company names, no years, no question words, and no words every report uses about itself
  (report, annual, company, data, statistics).
- "names": the companies, countries, products or people the question names, as the report would write them.
- "years": the years or periods the question asks about, as written in the question (e.g. 1399, 2023).
- "answer": "number" if the answer is a number or a series of numbers, else "text"."""

KW_SCHEMA = {"type": "object", "properties": {
    "keywords": {"type": "array", "items": {"type": "string"}},
    "names": {"type": "array", "items": {"type": "string"}},
    "years": {"type": "array", "items": {"type": "string"}},
    "answer": {"type": "string", "enum": ["number", "text"]}},
    "required": ["keywords", "names", "years", "answer"]}

PICK_SYSTEM = ("You help search company reports. You are shown phrases that really occur in the reports. You choose "
               "the ones that name exactly what a question asks about. Reply with JSON only.")

PICK_USER = """Question: {question}

Phrases found in the reports (number: phrase):
{listing}

- "pick": the numbers of the phrases that name what the question asks about (at most {pmax}). Prefer the report's own
  name for the measure, even if the question words it differently. Leave out phrases about something else.
- "add": at most {amax} more short search terms in {dlang}, only if the list misses the report's wording."""

PICK_SCHEMA = {"type": "object", "properties": {
    "pick": {"type": "array", "items": {"type": "integer"}},
    "add": {"type": "array", "items": {"type": "string"}}},
    "required": ["pick", "add"]}

REGEX_SYSTEM = ("You write regular expressions that find, in company reports, the passages that answer a question. "
                "Reply with JSON only.")

REGEX_USER = """Question: {question}

The reports are in {dlang}. Their text is normalised before searching: Arabic ي and ك are written ی and ک, the
zero-width non-joiner is a space, diacritics are removed, Latin letters are lower case, digits are 0-9.

Phrases found in the reports (you may use them):
{listing}

Search keywords from a first step: {keywords}

- "regex": at most {rmax} Python regular expressions that match the passages answering the question. Keep each one
  short and general enough to match the report's wording (allow plural forms and spacing variants)."""

REGEX_SCHEMA = {"type": "object", "properties": {
    "regex": {"type": "array", "items": {"type": "string"}}},
    "required": ["regex"]}


def kw_messages(question, d_lang, o):
    return [{"role": "system", "content": KW_SYSTEM},
            {"role": "user", "content": KW_USER.format(question=question.strip(), dlang=LANG[d_lang], kmin=3,
                                                       kmax=int(o["kw_max"]))}]


def listing(vocab):
    """"n: phrase", plus " (1398, 1400)" - the years of the reports it occurs in - when the list carries them
    (report_vocab over several reports: the provenance lets the model pick each year's own name for the metric)."""
    return "\n".join("%d: %s%s" % (i, v["text"], " (%s)" % v["where"] if v.get("where") else "")
                     for i, v in enumerate(vocab, 1)) or "(none)"


def pick_messages(question, vocab, d_lang, o):
    return [{"role": "system", "content": PICK_SYSTEM},
            {"role": "user", "content": PICK_USER.format(question=question.strip(), listing=listing(vocab),
                                                         pmax=int(o["pick_max"]), amax=int(o["add_max"]),
                                                         dlang=LANG[d_lang])}]


def regex_messages(question, vocab, keywords, d_lang, o):
    return [{"role": "system", "content": REGEX_SYSTEM},
            {"role": "user", "content": REGEX_USER.format(question=question.strip(), listing=listing(vocab),
                                                          keywords=", ".join(keywords) or "(none)",
                                                          rmax=int(o["regex_max"]), dlang=LANG[d_lang])}]


def parse_json(text):
    """-> (object or None, how). Direct parse, else the first {...} block, else the first [...] block in the text."""
    t = (text or "").strip()
    try:
        return json.loads(t), "direct"
    except ValueError:
        pass
    for pat in (r"\{.*\}", r"\[.*\]"):
        m = re.search(pat, t, re.S)
        if m:
            try:
                return json.loads(m.group(0)), "extract"
            except ValueError:
                pass
    return None, "failed"


def _strings(x, max_n=None, max_words=None):
    out = []
    for v in (x if isinstance(x, list) else [x] if isinstance(x, str) else []):
        s = " ".join(str(v).split()) if v is not None else ""
        if not s or (max_words and len(s.split()) > max_words) or s in out:
            continue
        out.append(s)
        if max_n and len(out) >= max_n:
            break
    return out


def parse_keywords(text, o):
    """-> ({keywords, names, years, answer}, how)."""
    obj, how = parse_json(text)
    obj = obj if isinstance(obj, dict) else {}
    years = []
    for y in _strings(obj.get("years")):
        for m in re.findall(r"\d{4}", y.translate(str.maketrans("۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩", "01234567890123456789"))):
            if m not in years:
                years.append(m)
    ans = str(obj.get("answer") or "").strip().lower()
    return {"keywords": _strings(obj.get("keywords"), int(o["kw_max"]), int(o["kw_max_words"])),
            "names": _strings(obj.get("names"), 6, 5), "years": years,
            "answer": ans if ans in ("number", "text") else "number"}, how


def parse_pick(text, n_vocab, o):
    """-> ({pick: [0-based indexes in range], add: [terms]}, how)."""
    obj, how = parse_json(text)
    obj = obj if isinstance(obj, dict) else {}
    pick = []
    for v in obj.get("pick") or []:
        try:
            i = int(v) - 1
        except (TypeError, ValueError):
            continue
        if 0 <= i < n_vocab and i not in pick:
            pick.append(i)
        if len(pick) >= int(o["pick_max"]):
            break
    return {"pick": pick, "add": _strings(obj.get("add"), int(o["add_max"]), int(o["kw_max_words"]))}, how


REGEX_KEYS = ("regex", "regexes", "patterns", "regular_expressions")


def parse_regex(text, o):
    """-> ({regex: [strings as given], shape}, how). Compilation is the caller's (it folds each pattern first).

    Accepted shapes (second opinion, 2026-09-23: a bare array was silently dropped and {"regex": "revenue"} was
    iterated into seven one-letter patterns, both logged as a clean parse):
      {"regex": [..]}  -> shape "object"          {"regex": "one"}  -> "object_string" (one pattern)
      ["a", "b"]       -> "array"                 {"patterns": [..]} and the other REGEX_KEYS -> "alias"
    Anything else parses to no patterns with shape "wrong_type" (JSON but not a pattern list) or "none" (no JSON).
    Only strings count as patterns; a non-string item is dropped."""
    obj, how = parse_json(text)
    shape, items = "none", []
    if isinstance(obj, list):
        shape, items = "array", obj
    elif isinstance(obj, dict):
        key = next((k for k in REGEX_KEYS if k in obj), None)
        val = obj.get(key) if key else None
        if isinstance(val, list):
            shape, items = ("object" if key == "regex" else "alias"), val
        elif isinstance(val, str):
            shape, items = "object_string", [val]
        else:
            shape = "wrong_type"
    elif obj is not None:
        shape = "wrong_type"
    pats = [r for r in items if isinstance(r, str) and r.strip()][: int(o["regex_max"])]
    return {"regex": pats, "shape": shape}, how
