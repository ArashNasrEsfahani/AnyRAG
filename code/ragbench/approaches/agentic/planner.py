"""Stage 1 - plan (one model call, strict JSON with repair).

The planner reads the question and a compact listing of the documents in scope (company key, years, one
title per company; see docmeta) and returns:
    {"entity", "metric", "kind": single|comparison|timeline, "terms_fa": [...], "terms_en": [...],
     "entity_terms": [...], "unit": count|percent|currency|quantity|ratio|date|text,
     "period_from": "", "period_to": "", "companies": [company keys]}
Repair ladder: Ollama JSON-schema output -> extract_json -> suffix repair of a cut reply -> (budget allowing)
one repair call that shows the model its reply -> a program plan (question words, cue-word kind, companies
matched from titles). Every step is logged in the plan's "repair" record.

Terms are literals, not regexes (a 4B model writes bad regexes); code turns them into ripgrep patterns
with Persian morphology: joiner flex (ZWNJ | space | nothing, optional ezafe hamza), the plural stem of a
final heh (راننده <-> رانندگان), the plural suffixes ها/های/ان stripped (substring matching then covers
every suffixed form), English plural -s stripped.
"""
import json
import re

from ...llm import extract_json
from ...normalize import ZWNJ, detect_lang, fold, normalize, script_counts
from .defaults import DEFAULTS, KINDS, PLAN_PROMPT_VERSION, UNIT_KINDS
from .docmeta import company_listing, match_companies

LANG_NAMES = {"fa": "Persian", "en": "English"}

PLAN_SYSTEM = """You plan a search for the answer to a question in a collection of company, market and sector reports. You do not answer the question. Reply with one JSON object with these fields:
- "entity": the company, organisation, market or product the question is about, as named in the question.
- "metric": what is measured or asked, in a few words.
- "kind": "timeline" when the question asks how a value changed over several years or periods; "comparison" when it compares entities or two named periods; otherwise "single".
- "terms_fa": 2 to 6 short Persian search terms for the metric (one or two words each): the words a Persian report would use, synonyms, singular and plural forms. No entity name, no year, no number.
- "terms_en": the same terms in English.
- "entity_terms": the entity's name in Persian and in English (1 to 4 strings).
- "unit": count, percent, currency, quantity, ratio, date or text (text when the answer is not a number).
- "period_from", "period_to": the first and last year the question covers, as a 4-digit year (Persian solar years such as 1398, or Gregorian years such as 2021); "" when the question names no year.
{companies_rule}"""

COMPANIES_RULE = ('- "companies": the company keys, from the document list in the user message, whose reports can '
                  'hold the answer (usually one company).')
COMPANIES_RULE_NONE = '- "companies": [] (the documents are already chosen).'

PLAN_USER = "Question: {question}\n\nDocuments ({n_docs}), one line per company (key: years | documents | a title):\n{listing}"
PLAN_USER_SCOPED = "Question: {question}\n\nThe search runs inside: {titles}"

REPAIR_USER = ("Your reply could not be used ({problem}). Reply again with only the JSON object for the same question, "
               "with every field.")

_STOP = set("""a an the of in on at to for from by with and or is are was were be been what which who how many much when
where why does do did has have had this that these those it its as than per into about according report reports year
years over change changed has have their there over time
از به با در که را این آن برای تا چه چند چقدر چگونه کدام کجا چرا آیا است هست بود شد شده می و یا هم نیز بر اساس طبق
گزارش سال های ها ای یک دو اما ولی اگر چون پس بین طی درباره همین همان کل کلی بوده باشد دارد داشت کرد کرده کند شود
دارند کرده‌اند تغییر تغییرات طول سال‌ها سالها روند گذشته""".split())

_TIMELINE_CUES = re.compile(
    r"over the years|each year|every year|trend|year by year|year-on-year|over time|how has .* changed|"
    r"from (?:1[34]|19|20)\d\d to (?:1[34]|19|20)\d\d|"
    r"طی سال|در طول سال|سال.?به.?سال|روند|چگونه تغییر|چه تغییری|تغییر کرده|سال.?های مختلف|در سال.?های|"
    r"از سال \d{2,4} تا", re.I)
_COMPARE_CUES = re.compile(r"compare|comparison|versus|\bvs\b|higher|lower|مقایسه|بیشتر|کمتر|در برابر", re.I)
_YEAR = re.compile(r"(?<![0-9])(1[34][0-9]{2}|19[0-9]{2}|20[0-9]{2})(?![0-9])")


def plan_schema(company_keys):
    props = {
        "entity": {"type": "string"},
        "metric": {"type": "string"},
        "kind": {"type": "string", "enum": list(KINDS)},
        "terms_fa": {"type": "array", "items": {"type": "string"}},
        "terms_en": {"type": "array", "items": {"type": "string"}},
        "entity_terms": {"type": "array", "items": {"type": "string"}},
        "unit": {"type": "string", "enum": list(UNIT_KINDS)},
        "period_from": {"type": "string"},
        "period_to": {"type": "string"},
    }
    req = list(props)
    if company_keys:
        props["companies"] = {"type": "array", "items": {"type": "string", "enum": list(company_keys)}}
        req.append("companies")
    return {"type": "object", "properties": props, "required": req}


def plan_messages(question, metas, choose_companies, opts=None):
    opts = dict(DEFAULTS, **(opts or {}))
    system = PLAN_SYSTEM.replace("{companies_rule}", COMPANIES_RULE if choose_companies else COMPANIES_RULE_NONE)
    q = normalize(question)
    if choose_companies:
        listing = "\n".join(company_listing(metas, int(opts["max_companies_listed"])))
        user = PLAN_USER.format(question=q, n_docs=len(metas), listing=listing)
    else:
        titles = "; ".join(m["title"][:120] for m in metas[:6]) or "(no documents)"
        user = PLAN_USER_SCOPED.format(question=q, titles=titles)
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


# --------------------------------------------------------------------------
# parsing and repair
# --------------------------------------------------------------------------

def parse_plan_text(text):
    """-> (obj | None, how): json | extract | suffix_repair | None."""
    if not text:
        return None, None
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj, "json"
    except (ValueError, TypeError):
        pass
    obj = extract_json(text)
    if isinstance(obj, dict):
        return obj, "extract"
    if "{" in text:
        base = text.rstrip().rstrip(",")
        for suffix in ('"}', "}", '"]}', "]}", '"]"}', '""}', '"}}', "}}"):
            obj = extract_json(base + suffix)
            if isinstance(obj, dict):
                return obj, "suffix_repair"
    return None, None


def plan_problem(obj):
    """None when a parsed plan is usable, else a short reason."""
    if not isinstance(obj, dict):
        return "the reply was not a JSON object"
    terms = _as_list(obj.get("terms_fa")) + _as_list(obj.get("terms_en"))
    if not any(isinstance(t, str) and t.strip() for t in terms):
        return "terms_fa and terms_en are both empty"
    return None


def _as_list(v):
    if isinstance(v, str):
        return [v]
    return list(v) if isinstance(v, list) else []


# --------------------------------------------------------------------------
# validation
# --------------------------------------------------------------------------

def _lang_of(term):
    fa, en = script_counts(term)
    if fa + en == 0:
        return None
    return "fa" if fa / (fa + en) >= 0.3 else "en"


def clean_terms(values, max_terms, max_chars, stop_extra=()):
    """Literal terms: normalised, quotes/regex syntax removed, no digits-only or year terms, deduped,
    split by script. -> {"fa": [...], "en": [...]}, dropped."""
    out = {"fa": [], "en": []}
    dropped = []
    seen = set()
    stop_extra = {fold(s) for s in stop_extra if s}
    for v in values:
        if not isinstance(v, str):
            continue
        t = normalize(v)
        t = re.sub(r"[\\^$|?*+()\[\]{}\"'`«»/:;،,.!؟]+", " ", t)
        t = re.sub(r"\s+", " ", t).strip()
        if not t:
            continue
        if len(t) > max_chars:
            dropped.append({"term": v, "reason": "too_long"})
            continue
        if re.fullmatch(r"[\d\s%.,-]+", t) or _YEAR.search(t):
            dropped.append({"term": v, "reason": "number_or_year"})
            continue
        letters = sum(1 for ch in t if ch.isalpha())
        if letters < 2:
            dropped.append({"term": v, "reason": "too_short"})
            continue
        f = fold(t)
        if f in seen or f in stop_extra or f in _STOP:
            dropped.append({"term": v, "reason": "duplicate_or_stopword"})
            continue
        lang = _lang_of(t)
        if lang is None:
            continue
        if len(out[lang]) >= max_terms:
            dropped.append({"term": v, "reason": "cap"})
            continue
        seen.add(f)
        out[lang].append(t)
    return out, dropped


def _norm_year(s):
    m = _YEAR.search(normalize(str(s or "")))
    return int(m.group(1)) if m else None


# words that never identify a metric on their own: question words the planner copies into terms ("change",
# "group") and qualifiers that say which variant of a metric is meant (kept for scoring and definitions)
GENERIC_TERM_WORDS = set("""تغییر تغییرات روند گروه شرکت سال سالانه سالیانه سالها سال‌ها گزارش طول مدت چگونه
change changes changed trend group company year years annual report over time how""".split())
MODIFIER_WORDS = set("""فعال غیرفعال جدید ثبت نام ثبت‌نام ثبت‌نامی ماهانه روزانه هفتگی تمام وقت پاره
active inactive new registered monthly daily weekly total full part time""".split())


def refine_terms(terms, entity_terms, max_terms):
    """Code-side term hygiene (a 4B planner writes "تعداد رانندگان اسنپ", "گروه اسنپ", "Snapp driver change"):
      * words of the entity's names and generic question words are removed from every term;
      * a term left with only measure/modifier/generic words (or nothing) is dropped;
      * each remaining content word of a multi-word term is added as its own single-word term (after the
        phrases), so the metric's noun is searched even when the phrase never occurs verbatim.
    -> (terms by lang, report)"""
    ent_words = set()
    for e in entity_terms:
        for w in re.split(r"[\s‌]+", normalize(e)):
            if len(w) >= 2:
                ent_words.add(fold(w))
    weak = {fold(w) for w in WEAK_WORDS}
    generic = {fold(w) for w in GENERIC_TERM_WORDS}
    modifiers = {fold(w) for w in MODIFIER_WORDS}
    out = {"fa": [], "en": []}
    report = {"removed_words": [], "dropped": [], "added_words": []}
    for lang in ("fa", "en"):
        phrases, singles = [], []
        for t in terms.get(lang) or []:
            words = [w for w in re.split(r"[\s‌]+", normalize(t)) if w]
            kept = []
            for w in words:
                f = fold(w)
                if f in ent_words or f in generic or (lang == "en" and f.rstrip("s") in ent_words):
                    report["removed_words"].append(w)
                    continue
                kept.append(w)
            content = [w for w in kept if fold(w) not in weak and fold(w) not in modifiers and fold(w) not in _STOP
                       and len(w) >= 3]
            if not content:
                report["dropped"].append(t)
                continue
            phrase = " ".join(kept)
            if phrase not in phrases:
                phrases.append(phrase)
            if len(kept) > 1:
                for w in content:
                    if w not in singles:
                        singles.append(w)
        seen = {fold(p) for p in phrases}
        for w in singles:
            if fold(w) not in seen:
                phrases.append(w)
                seen.add(fold(w))
                report["added_words"].append(w)
        out[lang] = phrases[: max_terms + 3]
    return out, report


def question_has_year(question):
    qn = normalize(question)
    return bool(_YEAR.search(qn) or re.search(r"سال\s?[0-9]{2}(?![0-9])|(?<![0-9])[0-9]{2}\s?(?:خورشیدی|شمسی)", qn))


def validate_plan(obj, *, question, q_lang, metas, choose_companies, opts=None):
    """Parsed model object (or {}) -> normalised plan dict (program fills what is missing)."""
    opts = dict(DEFAULTS, **(opts or {}))
    obj = obj if isinstance(obj, dict) else {}
    qn = normalize(question)
    fixes = []
    kind = obj.get("kind") if obj.get("kind") in KINDS else None
    if kind is None:
        kind = infer_kind(qn)
        fixes.append("kind_inferred")
    unit = obj.get("unit") if obj.get("unit") in UNIT_KINDS else None
    if unit is None:
        unit = "count" if re.search(r"how many|number of|چند نفر|تعداد|چند", qn, re.I) else "text"
        fixes.append("unit_inferred")
    entity = normalize(str(obj.get("entity") or "")).strip()
    metric = normalize(str(obj.get("metric") or "")).strip()
    ent_terms, ent_dropped = clean_terms(_as_list(obj.get("entity_terms")) + ([entity] if entity else []),
                                         int(opts["max_entity_terms"]), int(opts["max_term_chars"]))
    ent_all = ent_terms["fa"] + ent_terms["en"]
    raw_terms = _as_list(obj.get("terms_fa")) + _as_list(obj.get("terms_en"))
    terms, dropped = clean_terms(raw_terms, int(opts["max_terms_per_lang"]), int(opts["max_term_chars"]),
                                 stop_extra=ent_all)
    model_terms = terms
    terms, refine = refine_terms(terms, ent_all, int(opts["max_terms_per_lang"]))
    if not terms["fa"] and not terms["en"]:
        fb = question_terms(qn, ent_all)
        terms, d2 = clean_terms(fb, int(opts["max_terms_per_lang"]), int(opts["max_term_chars"]), stop_extra=ent_all)
        terms, _r2 = refine_terms(terms, ent_all, int(opts["max_terms_per_lang"]))
        dropped += d2
        fixes.append("terms_from_question")
    p_from, p_to = _norm_year(obj.get("period_from")), _norm_year(obj.get("period_to"))
    if (p_from is not None or p_to is not None) and not question_has_year(qn):
        # a period the question does not name (the planner copies the years of the document listing) would only
        # narrow the documents; the question's own scope decides
        p_from = p_to = None
        fixes.append("period_ignored_not_in_question")
    q_years = sorted({int(y) for y in _YEAR.findall(qn)})
    if p_from is None and p_to is None and q_years:
        p_from, p_to = q_years[0], q_years[-1]
        fixes.append("period_from_question")
    if p_from is not None and p_to is not None and p_from > p_to:
        p_from, p_to = p_to, p_from
    if kind == "single" and p_from is not None and p_to is not None and p_from != p_to and len(q_years) >= 2:
        kind = "comparison"
        fixes.append("kind_comparison_from_years")
    comps = [c for c in _as_list(obj.get("companies")) if isinstance(c, str)]
    valid_keys = {m["company"] for m in metas}
    comps = [c for c in dict.fromkeys(comps) if c in valid_keys]
    return {"entity": entity, "metric": metric, "kind": kind, "unit": unit, "terms": terms,
            "model_terms": model_terms, "term_refinement": refine,
            "entity_terms": ent_all, "period": (p_from, p_to), "companies": comps if choose_companies else [],
            "dropped_terms": dropped + ent_dropped, "fixes": fixes}


def infer_kind(question):
    if _TIMELINE_CUES.search(question):
        return "timeline"
    if _COMPARE_CUES.search(question):
        return "comparison"
    return "single"


def question_terms(question, entity_terms=()):
    """Content words of the question (program fallback)."""
    ents = {fold(e) for e in entity_terms}
    out = []
    for w in re.split(r"[\s‌،,.?!؟:;()«»\"']+", normalize(question)):
        f = fold(w)
        if len(w) < 3 or f in _STOP or f in ents or re.fullmatch(r"[\d.,%/]+", w):
            continue
        if w not in out:
            out.append(w)
    out.sort(key=lambda w: -len(w))
    return out[:6]


def program_plan(question, q_lang, metas, choose_companies, opts=None):
    """No usable model plan: everything from the question and the document titles."""
    plan = validate_plan({}, question=question, q_lang=q_lang, metas=metas, choose_companies=False, opts=opts)
    if choose_companies:
        plan["companies"] = match_companies(question, metas)
    plan["fixes"].append("program_plan")
    return plan


# --------------------------------------------------------------------------
# terms -> ripgrep patterns (Persian morphology in code)
# --------------------------------------------------------------------------

_RX_META = re.compile(r"([\\.^$|?*+()\[\]{}])")
_FA_PLURALS = ("‌های", "‌ها", " های", " ها", "هایی", "های", "ها")


def _esc(s):
    return _RX_META.sub(r"\\\1", s)


def fa_word_variants(w):
    """Stems of one Persian word such that a substring match covers its inflected forms."""
    w = w.replace(ZWNJ, "")
    out = [w]
    for suf in ("هایی", "های", "ها"):
        if w.endswith(suf) and len(w) - len(suf) >= 3:
            out.append(w[: -len(suf)])
            break
    base = out[-1]
    if base.endswith("گان") and len(base) >= 6:          # رانندگان -> راننده
        out.append(base[:-3] + "ه")
    elif base.endswith("ه") and len(base) >= 4:          # راننده -> رانندگان
        out.append(base[:-1] + "گان")
    if base.endswith("ان") and not base.endswith("گان") and len(base) - 2 >= 4:   # کاربران -> کاربر
        out.append(base[:-2])
    uniq = []
    for x in out:
        if x not in uniq:
            uniq.append(x)
    # the shortest stems cover the longer forms by substring; keep only minimal ones
    minimal = [x for x in uniq if not any(y != x and y in x for y in uniq)]
    return minimal


def fa_word_forms(w):
    """Every form of a non-final word of a phrase (no substring shortcut inside a phrase), longest first."""
    w = w.replace(ZWNJ, "")
    forms = {w}
    for suf in ("هایی", "های", "ها"):
        if w.endswith(suf) and len(w) - len(suf) >= 3:
            forms.add(w[: -len(suf)])
    for f in list(forms):
        if f.endswith("گان") and len(f) >= 6:
            forms.add(f[:-3] + "ه")
        elif f.endswith("ه") and len(f) >= 4:
            forms.add(f[:-1] + "گان")
        if f.endswith("ان") and not f.endswith("گان") and len(f) - 2 >= 4:
            forms.add(f[:-2])
    return sorted(forms, key=lambda x: (-len(x), x))


def _fa_word_rx(w):
    return _esc(w)


_FA_JOIN = "ٔ?[‌ ]?"      # optional ezafe hamza, then ZWNJ | space | nothing


def term_patterns(term, lang):
    """One literal term -> list of ripgrep-compatible regexes (also valid Python re)."""
    t = normalize(term)
    if lang == "fa":
        words = [x for x in re.split(r"[\s‌]+", t) if x]
        if not words:
            return []
        heads = []
        for w in words[:-1]:
            forms = fa_word_forms(w)
            heads.append(_esc(forms[0]) if len(forms) == 1 else "(?:%s)" % "|".join(_esc(f) for f in forms))
        head = _FA_JOIN.join(heads)
        pats = []
        for last in fa_word_variants(words[-1]):
            pats.append((head + _FA_JOIN if head else "") + _fa_word_rx(last))
        return pats
    words = [x for x in re.split(r"\s+", t.lower()) if x]
    if not words:
        return []
    last = words[-1]
    if last.endswith("ies") and len(last) > 5:
        last = last[:-3]
    elif last.endswith("s") and not last.endswith("ss") and len(last) > 4:
        last = last[:-1]
    words[-1] = last
    return ["[ -]?".join(_esc(w) for w in words)]


# measure words: scored when present, never searched on their own (they occur next to every number)
WEAK_WORDS = set("""تعداد شمار میزان مقدار کل مجموع رشد سهم درصد نرخ ارزش رقم آمار number total share growth amount value
rate count figure percent percentage level""".split())


def component_words(terms_by_lang, entity_terms=()):
    """-> [(lang, word, weak)] component words of every term (>= 3 letters, not stopwords/entity names)."""
    ents = {fold(e) for e in entity_terms}
    out, seen = [], set()
    for lang, terms in terms_by_lang.items():
        for t in terms:
            for w in re.split(r"[\s‌]+", normalize(t)):
                f = fold(w)
                if len(w) < 3 or f in _STOP or f in ents or f in seen:
                    continue
                seen.add(f)
                out.append((lang, w, f in {fold(x) for x in WEAK_WORDS}))
    return out


def word_regex(word, lang):
    if lang == "fa":
        return "|".join(_fa_word_rx(v) for v in fa_word_variants(word))
    ww = word.lower()
    if ww.endswith("ies") and len(ww) > 5:
        ww = ww[:-3]
    elif ww.endswith("s") and not ww.endswith("ss") and len(ww) > 4:
        ww = ww[:-1]
    return _esc(ww)


def score_words(terms_by_lang, entity_terms=()):
    """Component words of every term as compiled Python regexes, for window scoring.
    -> [(word, compiled regex, weak)]"""
    return [(w, re.compile(word_regex(w, lang), re.I), weak)
            for lang, w, weak in component_words(terms_by_lang, entity_terms)]
