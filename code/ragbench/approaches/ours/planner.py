"""OURS query call 1 (plus the optional repair or widening call): question -> search plan.

Plan JSON (Ollama structured output), patterns per language:
    {"<q_lang>": [regexes], "<other>": [regexes], "scope": headers|facts|body|all, "expect": number|text}
The question-language list comes first in the schema (copy the question's words), then the other
language (their translation). With the monolingual_patterns ablation the schema has only the
question-language list. The planner sees only the question text, its language and the document
language (all part of the Query every approach gets).

Language check (enforced in code, not only asked for in the prompt): after post-processing there must
be at least one compiling pattern in the DOCUMENT language and one in the QUESTION language
(monolingual_patterns: the question language only). A pattern's language is its script, not the list
the model put it in; a top-level alternation mixing scripts ("billet|شمش") is split into one pattern
per script. When the check fails, the approach may spend ONE repair call (Planner.repair) that shows
the model its reply and what is missing; the repair counts toward caps.ours_max_calls, and a question
that used the repair gets no widening call (plan + repair + answer = 3). The repaired plan keeps the
first plan's patterns for every language that already passed and takes the repair's patterns only
for the missing ones.

Pattern post-processing (no LLM):
  1. strip quotes/slashes and inline flags; drop empty ones; split mixed-script alternations
  2. with normalisation on: ragbench.normalize.normalize() (the extractor rule: Arabic yeh/kaf,
     Persian digits -> ASCII, joiners unified ...), then "joiner flex": a ZWNJ or a single space
     between two Persian letters becomes [ZWNJ ]? so "سرمایه گذاری" matches "سرمایه‌گذاری", and the
     ezafe hamza (هٔ / ۀ) becomes optional so "میانهٔ راند" matches "میانه راند"
  3. monolingual_patterns: drop patterns whose letters are not in the question language
  4. drop too-broad patterns (fewer than 2 literal letters/digits, or matching the empty string)
  5. compile with ripgrep itself (Rust regex; look-arounds etc. are rejected) and drop failures
  6. dedupe, cap at caps.ours_patterns_max PER LANGUAGE, interleave the languages (document language
     first); if fewer than caps.ours_patterns_min survive (after the repair), add content words of
     the question (program fallback, logged)
When the question has a document scope (a list of documents, as every Query carries), the prompt says the
search already runs inside those reports, so patterns for the report's own name are wasted.
"""
import re
import subprocess

from ...normalize import normalize, word_tokens, detect_lang, script_counts
from ...store import rg_binary
from .defaults import DEFAULTS, SCOPES, EXPECTS

LANG_NAMES = {"fa": "Persian", "en": "English"}
LANGS = ("fa", "en")
ZWNJ = "\u200c"


def other_lang(lang):
    return "en" if lang == "fa" else "fa"


def plan_schema(langs):
    """JSON schema with one pattern list per language, in the given order, then scope and expect."""
    props = {l: {"type": "array", "items": {"type": "string"}} for l in langs}
    props["scope"] = {"type": "string", "enum": list(SCOPES)}
    props["expect"] = {"type": "string", "enum": list(EXPECTS)}
    return {"type": "object", "properties": props, "required": list(langs) + ["scope", "expect"]}


PLAN_SYSTEM = (
    "You plan a grep search over report sections written in {doc_lang}. Every section has a header with "
    "keywords and synonyms in Persian and English and the questions it answers, numeric fact lines "
    "\"metric | value | unit | period\", and its body text.\n"
    "Return JSON: {shape}.\n"
    "Patterns are regular expressions (ripgrep syntax, case-insensitive, matched against single lines) for the key "
    "terms of the question: names, topics, metrics, most important first. Each pattern is one term of one or two "
    "words, as a short stem (\"export\", not \"total exports of fish\"; \"صادرات\", not \"کل صادرات ماهی\"). "
    "Put a year in its own pattern, never glued to a term. Alternation such as (salmon|trout) is allowed inside one "
    "language. Do not use \\b, look-arounds or back-references. Never put the answer or a guessed number in a pattern.\n"
    "{lang_rule}{scope_rule}\n"
    "scope: headers when the question is about a topic or where something is discussed; facts when it asks for one "
    "number; body when it quotes specific wording; all when unsure.\n"
    "expect: number when the answer is a number, otherwise text."
)
SHAPE_LIST = "\"{code}\": [{name} patterns]"
SHAPE_TAIL = "\"scope\": \"headers\"|\"facts\"|\"body\"|\"all\", \"expect\": \"number\"|\"text\""
LANG_RULE_BI = (
    "Write {per_min} to {per_max} patterns in each language list. \"{q}\" ({Q}, the language of the question): copy "
    "the key words exactly as they are spelled in the question. \"{o}\" ({O}): the same terms translated into {O}, "
    "as a report written in {O} would spell them. The sections are written in {D}, so the \"{d}\" list must never be "
    "empty. Never leave a list empty."
)
LANG_RULE_MONO = ("Write {pmin} to {pmax} patterns in the \"{q}\" list, in {Q} only, the language of the question, "
                  "copying the key words exactly as they are spelled in the question.")

SCOPE_RULE_DOCS = (" The search already runs only inside the report(s) the question names, so a pattern for the name "
                   "or title of that report finds nothing useful: spend the patterns on what the question asks about.")
SCOPE_RULE_CORPUS = ""

PLAN_USER = "Question: {question}"

REPAIR_USER = (
    "This plan cannot be used yet: {problems}. The sections are written in {D}. "
    "Reply again with the complete JSON plan for the same question{need}."
)

WIDEN_USER = (
    "Question: {question}\n\n"
    "The search found no matching line for these patterns: {previous}\n"
    "Write a broader plan: shorter stems (drop prefixes and suffixes), synonyms, alternative spellings, "
    "and fewer words per pattern. {lang_rule} Same JSON format."
)

_STOP = {
    "en": set("""a an the of in on at to for from by with and or is are was were be been being what which who whom whose
    how many much when where why does do did has have had this that these those it its as than per into about
    according report reports section there their they our your can could would should will shall may might
    between during over under up down out more most less least not no yes""".split()),
    "fa": set("""از به با در که را این آن برای تا چه چند چقدر چگونه کدام کجا چرا آیا است هست بود شد شده می و یا
    هم نیز بر اساس طبق گزارش سال های ها ای یک دو اما ولی اگر چون پس بین طی درباره دربارهٔ همین همان کل کلی
    بوده باشد دارد داشت کرد کرده کند شود""".split()),
}


def _fill(template, **kw):
    out = template
    for k, v in kw.items():
        out = out.replace("{%s}" % k, str(v))
    return out


# --------------------------------------------------------------------------
# pattern utilities
# --------------------------------------------------------------------------

def _is_fa_letter(ch):
    return ("\u0600" <= ch <= "\u06ff" or "\u0750" <= ch <= "\u077f" or "\ufb50" <= ch <= "\ufdff"
            or "\ufe70" <= ch <= "\ufefe") and ch.isalpha()


HAMZA_ABOVE = "ٔ"
HEH_YEH = "ۀ"      # ۀ


def flex_joiners(p):
    """Outside character classes (pattern-side robustness for Persian, applied only with normalisation on):
      * a ZWNJ or one space between two Persian letters -> [ZWNJ ]?   ("سرمایه گذاری" ~ "سرمایه‌گذاری")
      * after heh, that joiner also allows an ezafe hamza  -> هٔ?[ZWNJ ]?
      * an explicit ezafe hamza (U+0654) becomes optional, and ۀ matches ۀ / هٔ / ه
    so "میانهٔ راند" matches "میانه راند" and the reverse."""
    out = []
    in_class = False
    i = 0
    n = len(p)
    while i < n:
        ch = p[i]
        if ch == "\\" and i + 1 < n:
            out.append(p[i:i + 2])
            i += 2
            continue
        if in_class:
            if ch == "]":
                in_class = False
            out.append(ch)
            i += 1
            continue
        if ch == "[":
            in_class = True
            out.append(ch)
            i += 1
            continue
        prev = p[i - 1] if i > 0 else ""
        nxt = p[i + 1] if i + 1 < n else ""
        if ch == HAMZA_ABOVE and prev == "ه":
            if nxt in (ZWNJ, " ") and i + 2 < n and _is_fa_letter(p[i + 2]):
                out.append(HAMZA_ABOVE + "?[" + ZWNJ + " ]?")
                i += 2
                continue
            out.append(HAMZA_ABOVE + "?")
            i += 1
            continue
        if ch == HEH_YEH:
            out.append("(?:" + HEH_YEH + "|ه" + HAMZA_ABOVE + "?)")
            i += 1
            continue
        if ch in (ZWNJ, " ") and prev and nxt and _is_fa_letter(prev) and _is_fa_letter(nxt):
            out.append((HAMZA_ABOVE + "?" if prev == "ه" else "") + "[" + ZWNJ + " ]?")
            i += 1
            continue
        out.append(ch)
        i += 1
    return "".join(out)


_ESC = re.compile(r"\\(?:[pP]\{[^}]*\}|x\{[^}]*\}|u\{[^}]*\}|.)")
_META = re.compile(r"[\[\]\(\)\{\}\|\?\*\+\.\^\$\\]")


def literal_chars(p):
    """Letters/digits that are literal in the pattern (escapes and classes like \\w removed)."""
    t = _ESC.sub("", p)
    t = re.sub(r"\[[^\]]*\]", "", t)
    t = re.sub(r"\{\d*,?\d*\}", "", t)
    t = t.replace("?:", "")
    return sum(1 for ch in t if ch.isalnum())


def pattern_lang(p):
    fa, en = script_counts(_ESC.sub("", p))
    if fa + en == 0:
        return None
    return "fa" if fa / (fa + en) >= 0.30 else "en"


def clean_pattern(p):
    if not isinstance(p, str):
        return ""
    p = p.strip()
    if len(p) >= 2 and p[0] == p[-1] and p[0] in "\"'/`":
        p = p[1:-1].strip()
    p = re.sub(r"^\(\?[imsxu]+\)", "", p)
    return p


def _top_level_split(p):
    """Split a pattern at '|' outside groups and character classes; None if it has no such '|'."""
    parts, depth, in_class, cur, i = [], 0, False, [], 0
    while i < len(p):
        ch = p[i]
        if ch == "\\" and i + 1 < len(p):
            cur.append(p[i:i + 2])
            i += 2
            continue
        if in_class:
            in_class = ch != "]"
        elif ch == "[":
            in_class = True
        elif ch == "(":
            depth += 1
        elif ch == ")":
            depth = max(0, depth - 1)
        elif ch == "|" and depth == 0:
            parts.append("".join(cur))
            cur = []
            i += 1
            continue
        cur.append(ch)
        i += 1
    parts.append("".join(cur))
    return parts if len(parts) > 1 else None


def _unwrap_group(p):
    """'(a|b)' / '(?:a|b)' spanning the whole pattern -> 'a|b'; otherwise None."""
    m = re.match(r"^\((\?:)?", p)
    if not m or not p.endswith(")"):
        return None
    depth, i = 0, 0
    while i < len(p):
        ch = p[i]
        if ch == "\\":
            i += 2
            continue
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0 and i != len(p) - 1:
                return None          # the first group closes before the end: "(a)(b)"
        i += 1
    return p[m.end():-1]


def split_by_script(p):
    """[(lang|None, pattern)]: a top-level alternation whose alternatives are in different scripts is
    split into one alternation per script ("iron ore|سنگ آهن" -> en "iron ore", fa "سنگ آهن");
    alternatives without letters (numbers) stay with the first lettered group. Anything else is
    returned whole with its script language."""
    body = p
    parts = _top_level_split(body)
    if parts is None:
        inner = _unwrap_group(body)
        parts = _top_level_split(inner) if inner else None
    if not parts:
        return [(pattern_lang(p), p)]
    langs = [pattern_lang(x) for x in parts]
    lettered = [l for l in langs if l]
    if len(set(lettered)) <= 1:
        return [(pattern_lang(p), p)]
    groups, order = {}, []
    first = lettered[0]
    for x, l in zip(parts, langs):
        key = l or first
        if key not in groups:
            groups[key] = []
            order.append(key)
        if x.strip():
            groups[key].append(x.strip())
    return [(k, "|".join(groups[k])) for k in order if groups[k]]


def rg_check(patterns, cfg=None):
    """{pattern: None | error} using ripgrep's own parser (empty stdin)."""
    rg = rg_binary(cfg)

    def run(pats):
        cmd = [rg, "--no-config", "-i"]
        for q in pats:
            cmd += ["-e", q]
        cmd.append("-")
        return subprocess.run(cmd, input=b"", capture_output=True, timeout=30)

    if not patterns:
        return {}
    try:
        res = run(patterns)
    except subprocess.TimeoutExpired:
        return {q: None for q in patterns}      # undecided: the executor's grep reports a bad pattern per pattern
    if res.returncode != 2:
        return {q: None for q in patterns}
    out = {}
    for q in patterns:
        try:
            r = run([q])
        except subprocess.TimeoutExpired:
            out[q] = None
            continue
        err = r.stderr.decode("utf-8", "replace").strip().splitlines()
        out[q] = None if r.returncode != 2 else ((err[-1] if err else "ripgrep error")[:160])
    return out


def escape_literal(s):
    return _META.sub(lambda m: "\\" + m.group(0), s)


def fallback_patterns(question, lang, k, normalized=True, flex=True):
    """Program fallback: the longest content words of the question (no model). With normalisation
    on, the same Persian joiner flex as the model's patterns (a word written with ZWNJ in the
    question also matches the space / no-joiner spelling in the store)."""
    toks = word_tokens(question, folded=False) if normalized else [t for t in re.findall(r"\w+", question or "")]
    stop = _STOP.get(lang, set()) | _STOP["en"] | _STOP["fa"]
    seen, cands = set(), []
    for t in toks:
        tl = t.lower()
        if tl in stop or len(t) < 3 or tl in seen or re.fullmatch(r"[\d.,/%]+", t):
            continue
        seen.add(tl)
        cands.append(t)
    cands.sort(key=lambda t: -len(t))
    out = [escape_literal(t) for t in cands[:k]]
    if normalized and flex:
        out = [flex_joiners(p) for p in out]
    return out


_FA_PLURAL = ("‌هایی", "‌های", "‌ها", "هایی", "های", "ها")


def word_stems(patterns):
    """Program widening: the words of plain (metacharacter-free) patterns as short stems, most
    important pattern first. English: lower-case, plural -ies/-es/-s removed, >= 4 letters; Persian:
    split on spaces and ZWNJ, plural ها/های removed, >= 3 letters; stopwords skipped."""
    out, seen = [], set()
    stop = _STOP["en"] | _STOP["fa"]
    for p in patterns or []:
        c = clean_pattern(p)
        if not c or _META.search(c):
            continue
        for w in re.split(r"[\s‌]+", c):
            w = w.strip(".,;:!?()«»\"'")
            if not w or w.lower() in stop:
                continue
            if all(ch.isascii() for ch in w):
                w = w.lower()
                if w.endswith("ies") and len(w) > 5:
                    w = w[:-3]
                elif w.endswith("es") and len(w) > 5 and w[-3] in "sxz":
                    w = w[:-2]
                elif w.endswith("s") and not w.endswith("ss") and len(w) > 4:
                    w = w[:-1]
                if len(w) < 4 or not w.isalpha():
                    continue
            else:
                for suf in _FA_PLURAL:
                    if w.endswith(suf) and len(w) - len(suf) >= 3:
                        w = w[: -len(suf)]
                        break
                if len(w) < 3 or w in stop:
                    continue
            if w not in seen:
                seen.add(w)
                out.append(w)
    return out


def _interleave(final, lang_order):
    """Round-robin by script, languages in `lang_order` first (then any other group in first-appearance order)."""
    by = {}
    for c in final:
        by.setdefault(pattern_lang(c) or "none", []).append(c)
    order = [l for l in lang_order if l in by]
    order += sorted([l for l in by if l not in order], key=lambda l: min(final.index(c) for c in by[l]))
    inter = []
    while any(by[l] for l in order):
        for l in order:
            if by[l]:
                inter.append(by[l].pop(0))
    return inter


def fill_fallback(final, report, question, q_lang, *, pmin, pmax, normalized=True, flex=True, cfg=None):
    """Add question content words while fewer than pmin patterns survive (program fallback, logged)."""
    report.setdefault("fallback_added", [])
    if len(final) >= pmin or not question:
        return final
    final = list(final)
    for fb in fallback_patterns(question, q_lang, pmax, normalized=normalized, flex=flex):
        if len(final) >= pmin:
            break
        if fb.lower() in {k.lower() for k in final}:
            continue
        if not rg_check([fb], cfg).get(fb):
            final.append(fb)
            report["fallback_added"].append(fb)
    report["langs"] = sorted({pattern_lang(c) or "none" for c in final})
    return final


def validate_patterns(raw_patterns, *, q_lang, monolingual=False, normalized=True, flex=True, pmin=2, pmax=5,
                      min_literal=2, question=None, cfg=None, lang_order=None):
    """-> (patterns, report) where report lists every drop with its reason. With `question`, content
    words of the question are added while fewer than pmin patterns survive; with `lang_order`, the
    pmax truncation takes languages round-robin in that order."""
    report = {"raw": list(raw_patterns) if isinstance(raw_patterns, list) else raw_patterns, "dropped": [],
              "fallback_added": []}
    cands = []
    for p in raw_patterns if isinstance(raw_patterns, list) else []:
        c = clean_pattern(p)
        if not c:
            report["dropped"].append({"pattern": p, "reason": "empty"})
            continue
        if normalized:
            c = normalize(c)
            if flex:
                c = flex_joiners(c)
        cands.append((p, c))
    kept = []
    for p, c in cands:
        if monolingual:
            pl = pattern_lang(c)
            if pl is not None and pl != q_lang:
                report["dropped"].append({"pattern": p, "reason": "wrong_language"})
                continue
        if literal_chars(c) < min_literal:
            report["dropped"].append({"pattern": p, "reason": "too_broad"})
            continue
        try:
            if re.search(c, "") is not None:
                report["dropped"].append({"pattern": p, "reason": "matches_empty"})
                continue
        except re.error:
            pass  # Python and Rust syntax differ; ripgrep decides below
        if c.lower() in {k.lower() for k in kept}:
            continue
        kept.append(c)
    errs = rg_check(kept, cfg)
    final = []
    for c in kept:
        if errs.get(c):
            report["dropped"].append({"pattern": c, "reason": "invalid_regex", "error": errs[c]})
        else:
            final.append(c)
    if len(final) > pmax:
        # keep both languages: take patterns round-robin by script, each language in model order
        inter = _interleave(final, list(lang_order or []))
        report["truncated"] = [c for c in inter[pmax:]]
        final = [c for c in final if c in inter[:pmax]]
    report["langs"] = sorted({pattern_lang(c) or "none" for c in final})
    if question:
        final = fill_fallback(final, report, question, q_lang, pmin=pmin, pmax=pmax, normalized=normalized, flex=flex,
                              cfg=cfg)
    c2p = {}
    for p, c in cands:
        c2p.setdefault(c, clean_pattern(p))
    report["kept_raw"] = [c2p.get(c, c) for c in final]      # model spelling of each searched pattern
    return final, report


def plan_lists(obj, langs):
    """{lang: [strings]} from a parsed plan object; a lone string is one pattern; a legacy flat
    "patterns" list is accepted and split by script."""
    out = {}
    for l in langs:
        v = obj.get(l) if isinstance(obj, dict) else None
        if isinstance(v, str):
            v = [v]
        out[l] = [x for x in v if isinstance(x, str)] if isinstance(v, list) else []
    legacy = obj.get("patterns") if isinstance(obj, dict) else None
    if isinstance(legacy, str):
        legacy = [legacy]
    if isinstance(legacy, list) and not any(out.values()):
        for x in legacy:
            if isinstance(x, str):
                out.setdefault(pattern_lang(clean_pattern(x)) or langs[0], []).append(x)
    return out


def script_candidates(lists, list_order):
    """Model lists -> [(lang, pattern, declared_list)], alternations split by script; a pattern without
    letters keeps the language of the list it was written in."""
    out = []
    for decl in list_order:
        for p in lists.get(decl) or []:
            c = clean_pattern(p)
            if not c:
                out.append((decl, p, decl))
                continue
            for lang, part in split_by_script(c):
                out.append((lang or decl, part, decl))
    return out


def interleave_candidates(cands, lang_order):
    by, order = {}, []
    for lang, p, decl in cands:
        if lang not in by:
            by[lang] = []
        by[lang].append((lang, p, decl))
    order = [l for l in lang_order if l in by] + [l for l in by if l not in lang_order]
    out = []
    while any(by[l] for l in order):
        for l in order:
            if by[l]:
                out.append(by[l].pop(0))
    return out


# --------------------------------------------------------------------------
# the planner
# --------------------------------------------------------------------------

class Planner:
    def __init__(self, cfg, client, *, monolingual=False, normalized=True, options=None):
        self.cfg = cfg
        self.client = client
        self.opts = dict(DEFAULTS, **(options or {}))
        self.monolingual = monolingual
        self.normalized = normalized
        caps = cfg.caps
        self.pmin, self.pmax = int(caps["ours_patterns_min"]), int(caps["ours_patterns_max"])

    # ---- contract -------------------------------------------------------------
    def list_langs(self, query):
        """Pattern lists in schema order: question language first."""
        return [query.q_lang] if self.monolingual else [query.q_lang, other_lang(query.q_lang)]

    def required_langs(self, query):
        """Languages that need at least one compiling pattern."""
        if self.monolingual:
            return [query.q_lang]
        return list(dict.fromkeys([query.d_lang, query.q_lang]))

    def lang_order(self, query):
        """Truncation / interleave priority: document language first."""
        return [query.q_lang] if self.monolingual else list(dict.fromkeys([query.d_lang, query.q_lang,
                                                                           other_lang(query.q_lang)]))

    def per_lang(self):
        """(min, max) patterns asked for per language list; the max is also the per-language cap."""
        return self.pmin, self.pmax

    def _lang_rule(self, query):
        q, d = query.q_lang, query.d_lang
        if self.monolingual:
            return _fill(LANG_RULE_MONO, pmin=self.pmin, pmax=self.pmax, q=q, Q=LANG_NAMES[q])
        o = other_lang(q)
        lo, hi = self.per_lang()
        return _fill(LANG_RULE_BI, per_min=lo, per_max=hi, q=q, Q=LANG_NAMES[q], o=o, O=LANG_NAMES[o], d=d,
                     D=LANG_NAMES[d])

    def _shape(self, query):
        lists = ", ".join(_fill(SHAPE_LIST, code=l, name=LANG_NAMES[l]) for l in self.list_langs(query))
        return "{" + lists + ", " + SHAPE_TAIL + "}"

    def _question(self, query):
        return normalize(query.text) if self.normalized else query.text.strip()

    def _system(self, query):
        scoped = isinstance(query.doc_scope, (list, tuple)) and len(query.doc_scope) > 0
        return _fill(PLAN_SYSTEM, doc_lang=LANG_NAMES[query.d_lang], shape=self._shape(query),
                     lang_rule=self._lang_rule(query), scope_rule=SCOPE_RULE_DOCS if scoped else SCOPE_RULE_CORPUS)

    def messages(self, query):
        return [{"role": "system", "content": self._system(query)},
                {"role": "user", "content": PLAN_USER.format(question=self._question(query))}]

    def widen_messages(self, query, previous):
        user = _fill(WIDEN_USER, question=self._question(query), previous=" ; ".join(previous) or "(none)",
                     lang_rule=self._lang_rule(query))
        return [{"role": "system", "content": self._system(query)}, {"role": "user", "content": user}]

    # ---- one model call -> validated plan ---------------------------------------
    def _chat(self, msgs, purpose, query):
        r = self.client.chat(msgs, purpose=purpose, format=plan_schema(self.list_langs(query)),
                             num_predict=self.opts["plan_num_predict"])
        obj = r.json(None)
        parse_ok = isinstance(obj, dict)
        return r, (obj if parse_ok else {}), parse_ok

    def _validate(self, cands, query):
        """[(lang, pattern, declared)] -> (patterns, report, check) without the question-word fallback.
        caps.ours_patterns_max applies PER LANGUAGE (plan-v3): each language's patterns are validated and capped
        on their own, then interleaved document language first. (A cap shared by both languages would halve the
        number of terms a bilingual plan can search, compared with plan-v2's mixed-script alternations.)"""
        order = self.lang_order(query)
        by = {}
        for lang, p, decl in cands:
            by.setdefault(lang, []).append(p)
        rep = {"raw": [p for _, p, _ in cands], "dropped": [], "fallback_added": [], "truncated": [], "kept_raw": []}
        finals = {}
        for lang in order + [l for l in by if l not in order]:
            if lang not in by:
                continue
            pats_l, rep_l = validate_patterns(by[lang], q_lang=query.q_lang, monolingual=self.monolingual,
                                              normalized=self.normalized, flex=self.opts["flex_joiners"],
                                              pmin=self.pmin, pmax=self.pmax,
                                              min_literal=self.opts["min_pattern_literal_chars"], question=None,
                                              cfg=self.cfg)
            seen = {c.lower() for fl in finals.values() for c in fl}
            keep = [(c, r) for c, r in zip(pats_l, rep_l["kept_raw"]) if c.lower() not in seen]
            finals[lang] = [c for c, _ in keep]
            rep["dropped"] += rep_l["dropped"]
            rep["truncated"] += rep_l.get("truncated") or []
            rep["kept_raw"] += [r for _, r in keep]
        pats = []
        kept_raw = dict(zip([c for fl in finals.values() for c in fl], rep["kept_raw"]))
        lists = [finals[l] for l in finals]
        while any(lists):
            for fl in lists:
                if fl:
                    pats.append(fl.pop(0))
        rep["kept_raw"] = [kept_raw[c] for c in pats]
        rep["langs"] = sorted({pattern_lang(c) or "none" for c in pats})
        rep["moved"] = [{"pattern": p, "list": decl, "script": lang} for lang, p, decl in cands
                        if lang != decl and pattern_lang(p) is not None]
        return pats, rep, self._check(pats, query)

    def _check(self, pats, query):
        present = sorted({pattern_lang(c) for c in pats} - {None})
        req = self.required_langs(query)
        return {"required": req, "present": present, "missing": [l for l in req if l not in present]}

    def _package(self, r, obj, parse_ok, lists, cands, pats, rep, check):
        return {"patterns": pats, "scope": obj.get("scope") if obj.get("scope") in SCOPES else "all",
                "expect": obj.get("expect") if obj.get("expect") in EXPECTS else "text", "parse_ok": parse_ok,
                "lists": lists, "candidates": cands, "check": check,
                "model_output": r.text[:1000], "done_reason": r.done_reason, "validation": rep,
                "prompt_tokens": r.prompt_tokens, "completion_tokens": r.completion_tokens}

    def plan(self, query):
        """Call 1. The plan's "check" names the required languages still missing (no fallback added yet)."""
        msgs = self.messages(query)
        r, obj, parse_ok = self._chat(msgs, "plan", query)
        lists = plan_lists(obj, self.list_langs(query))
        cands = script_candidates(lists, self.list_langs(query))
        pats, rep, check = self._validate(cands, query)
        p = self._package(r, obj, parse_ok, lists, cands, pats, rep, check)
        p["_messages"] = msgs
        return p

    def repair_problems(self, plan, query):
        probs = []
        rep = plan.get("validation") or {}
        for l in plan["check"]["missing"]:
            wrote = [p for lang, p, decl in plan.get("candidates") or [] if lang == l]
            if not plan.get("parse_ok"):
                probs.append("the reply was not a JSON plan")
                break
            if not wrote:
                in_list = (plan.get("lists") or {}).get(l) or []
                if in_list:
                    probs.append("the \"%s\" list has no %s pattern (what it contains is not written in %s)" % (
                        l, LANG_NAMES[l], LANG_NAMES[l]))
                else:
                    probs.append("the \"%s\" list (%s) is empty" % (l, LANG_NAMES[l]))
            else:
                reasons = sorted({d.get("reason") for d in rep.get("dropped") or []
                                  if pattern_lang(clean_pattern(str(d.get("pattern")))) == l} - {None})
                probs.append("every %s pattern was rejected (%s)" % (
                    LANG_NAMES[l], ", ".join(r.replace("_", " ") for r in reasons) or "unusable"))
        return "; ".join(probs) or "a required language has no usable pattern"

    def repair(self, query, plan):
        """The one repair call: shows the model its reply and what is missing. The result keeps the first
        plan's patterns for languages that passed and takes the repair's for the missing ones."""
        missing = list(plan["check"]["missing"])
        lo, hi = self.per_lang()
        need = ", with %s" % " and ".join("%d to %d usable %s patterns in the \"%s\" list" % (
            (lo if not self.monolingual else self.pmin), (hi if not self.monolingual else self.pmax), LANG_NAMES[l], l)
            for l in missing)
        user = _fill(REPAIR_USER, problems=self.repair_problems(plan, query), D=LANG_NAMES[query.d_lang], need=need)
        msgs = list(plan.get("_messages") or self.messages(query)) + [
            {"role": "assistant", "content": plan.get("model_output") or ""}, {"role": "user", "content": user}]
        r, obj, parse_ok = self._chat(msgs, "plan_repair", query)
        lists = plan_lists(obj, self.list_langs(query))
        rcands = script_candidates(lists, self.list_langs(query))
        merged = [c for c in plan["candidates"] if c[0] not in missing] + [c for c in rcands if c[0] in missing]
        pats, rep, check = self._validate(merged, query)
        out = dict(plan)
        out.update(patterns=pats, validation=rep, check=check, candidates=merged,
                   scope=plan["scope"] if plan.get("parse_ok") else (obj.get("scope") if obj.get("scope") in SCOPES else "all"),
                   expect=plan["expect"] if plan.get("parse_ok") else (obj.get("expect") if obj.get("expect") in EXPECTS else "text"))
        out["repair"] = {"missing_before": missing, "missing_after": check["missing"], "message": user,
                         "parse_ok": parse_ok, "lists": lists, "model_output": r.text[:1000],
                         "done_reason": r.done_reason, "taken": [p for lang, p, _ in rcands if lang in missing]}
        return out

    def finalize(self, query, plan):
        """Question-word fallback when fewer than pmin patterns survive (after any repair)."""
        rep = plan["validation"]
        pats = fill_fallback(plan["patterns"], rep, self._question(query), query.q_lang, pmin=self.pmin,
                             pmax=self.pmax, normalized=self.normalized, flex=self.opts["flex_joiners"], cfg=self.cfg)
        plan["patterns"] = pats
        rep["kept_raw"] = [x for x in rep.get("kept_raw") or []] + [x for x in rep.get("fallback_added") or []
                                                                     if x not in (rep.get("kept_raw") or [])]
        return plan

    def widen(self, query, previous_plan):
        # show the model the patterns that were searched, in its own spelling (before joiner flex)
        val = previous_plan.get("validation") or {}
        searched = [x for x in (val.get("kept_raw") or previous_plan.get("patterns") or []) if x]
        raw_all = [clean_pattern(p) for _, p, _ in previous_plan.get("candidates") or []]
        msgs = self.widen_messages(query, searched)
        r, obj, parse_ok = self._chat(msgs, "plan_widen", query)
        lists = plan_lists(obj, self.list_langs(query))
        cands = script_candidates(lists, self.list_langs(query))
        # patterns that already matched nothing are useless; add word stems of every pattern so far
        # (the model often repeats its first plan verbatim at temperature 0)
        prev = {clean_pattern(x).lower() for x in searched}
        new = [c for c in cands if clean_pattern(c[1]).lower() not in prev]
        stems = [s for s in word_stems([c[1] for c in new] + list(searched) + raw_all) if s.lower() not in prev]
        widen_info = {"model_new": [c[1] for c in new], "model_repeated": len(cands) - len(new), "stems": stems}
        allc = new + [(pattern_lang(s) or query.q_lang, s, "stem") for s in stems]
        pats, rep, check = self._validate(allc, query)
        rep["widen"] = widen_info
        p = self._package(r, obj, parse_ok, lists, allc, pats, rep, check)
        p = self.finalize(query, p)
        p["scope_model"] = p["scope"]
        p["scope"] = "all"          # the first search already fell back to every line kind
        return p
