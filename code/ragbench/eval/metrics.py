"""Per-record metrics that need no model: evidence recall, numeric match, normalised
string match, abstention detection. Pure functions; all text goes through
ragbench.normalize (the extractor rule) so Persian digits, the Arabic decimal sign and
yeh/kaf variants are handled exactly as in the corpus.

Definitions (also printed in results.md):
  ev_hit        1 if ANY gold unit of the setting's document language is in the evidence
                given to the answer call (RunRecord.evidence_units, i.e. after the budget)
  ev_recall     fraction of those gold units in the evidence (recall@budget)
  ret_hit/ret_recall  the same over everything retrieved before the budget cut
  numeric_match the answer contains a number equal to answer_value (scale words such as
                million/billion/هزار/میلیون/میلیارد/همت and rial<->toman converted; tolerance
                and rounding rules in MatchConfig)
  string_match  normalised exact match or gold-token containment (short/span answers)
  abstain_correct  unanswerable questions: 1 if the model abstained
  match_correct numeric_match | string_match | abstain_correct by answer_type, and 0 when
                the run errored (an abstention on an answerable question scores 0 because it
                contains no gold number/string)
  kf_*          key-fact scores (eval/keyfacts.py) for the questions that carry
                meta.key_facts: kf_total, kf_found, kf_recall, kf_correct. They are reported
                alongside match_correct and never change it here; eval/aggregate.py
                --scoring keyfacts is what promotes kf_correct to the headline.
"""
import math
import re
import unicodedata
from dataclasses import dataclass, asdict

from ..normalize import normalize, fold

# --------------------------------------------------------------------------
# configuration
# --------------------------------------------------------------------------


@dataclass
class MatchConfig:
    rel_tol: float = 0.005          # |answer - gold| <= rel_tol * |gold| counts as equal
    abs_tol: float = 1e-9
    allow_rounding: bool = True     # "12.3 million" matches 12,345,678 (a correct rounding at the written precision)
    min_sig_digits: int = 3         # ... but only when the written number has at least this many significant digits ("12 million" does not match 12.4M)
    sign_insensitive: bool = True   # "fell 3.5%" matches gold -3.5
    slash_decimal: bool = True      # Persian typography "3/5" = 3.5 is tried as an extra candidate
    abstain_max_chars: int = 160    # an abstention phrase only counts in a short answer

    def to_dict(self):
        return asdict(self)


DEFAULT_MATCH = MatchConfig()

ABSTAIN_TOKEN = "NOT_FOUND"

# --------------------------------------------------------------------------
# cleaning
# --------------------------------------------------------------------------

# bracketed citation markers the model adds despite the answer-only instruction: [2], [1, 3], [۲]، (NOT_FOUND stays)
_RE_REFS = re.compile(r"\[\s*\d+(?:\s*[,،;\-–]\s*\d+)*\s*\]")
_RE_THOUSANDS = re.compile(r"(?<=\d)[,٬](?=\d{3}(?!\d))")
_ZWNJ = "‌"


def strip_refs(text):
    return _RE_REFS.sub(" ", normalize(text or ""))


def _is_punct(ch):
    cat = unicodedata.category(ch)
    return cat[0] in ("P", "S") and ch != "%"


def norm_text(text):
    """Matching form: extractor normalisation, citation markers removed, fold() (alef/yeh/heh
    variants, ZWNJ->space, lower-case), thousands separators removed, punctuation -> space
    (a '.' between digits is kept)."""
    t = fold(strip_refs(text))
    prev = None
    while prev != t:
        prev = t
        t = _RE_THOUSANDS.sub("", t)
    out = []
    for i, ch in enumerate(t):
        if ch == "." and 0 < i < len(t) - 1 and t[i - 1].isdigit() and t[i + 1].isdigit():
            out.append(ch)
        elif _is_punct(ch):
            out.append(" ")
        else:
            out.append(ch)
    return re.sub(r"\s+", " ", "".join(out)).strip()


def tokens(text):
    return norm_text(text).split()


# --------------------------------------------------------------------------
# abstention
# --------------------------------------------------------------------------

_ABSTAIN_PHRASES = [
    r"not[\s_]*found", r"not (?:mentioned|stated|specified|provided|given|available|included|reported)",
    r"(?:does|do|did) not (?:mention|state|specify|provide|contain|include|say|report|give)",
    r"no (?:information|data|mention|evidence)", r"cannot be (?:found|determined|answered)",
    r"(?:is|are) not (?:in|present in) the (?:evidence|passages|documents?|reports?)",
    r"unable to (?:find|determine|answer)", r"insufficient (?:information|evidence)",
    "اطلاعاتی (?:در|درباره|دربار)", "اطلاعات[ی]? (?:کافی )?(?:وجود ندارد|موجود نیست|یافت نشد|ارائه نشده)",
    "ذکر نشده", "اشاره(?:ای)? نشده", "یافت نشد", "موجود نیست", "وجود ندارد", "مشخص نشده", "نیامده",
    "(?:نمی ?توان|نمیتوان).{0,20}(?:پاسخ|یافت|تعیین)", "ارائه نشده", "گزارش نشده",
]
_RE_ABSTAIN = re.compile("|".join("(?:%s)" % p for p in _ABSTAIN_PHRASES), re.I)


def detect_abstention(text, token=ABSTAIN_TOKEN, cfg=DEFAULT_MATCH):
    """True when the answer declines. Exactly the abstain token (any case, punctuation,
    citation markers) always counts; the token or a decline phrase inside a short answer
    without a number also counts. RunRecord.abstained is the strict exact-token form."""
    raw = (text or "").strip()
    if not raw:
        return False
    t = strip_refs(raw)
    core = re.sub(r"[\s\W_]+", "", t, flags=re.U).upper()
    if core == token.replace("_", "").upper():
        return True
    if len(t) > cfg.abstain_max_chars:
        return False
    has_number = bool(re.search(r"\d", t))
    if token.upper() in t.upper() and not has_number:
        return True
    return bool(_RE_ABSTAIN.search(fold(t)) or _RE_ABSTAIN.search(t)) and not has_number


# --------------------------------------------------------------------------
# numbers
# --------------------------------------------------------------------------

_SCALES = [
    # (pattern on normalised text, factor); longer phrases first
    (r"هزار[\s‌]*میلیارد", 1e12), (r"هزار[\s‌]*میلیون", 1e9), (r"میلیارد", 1e9), (r"میلیون", 1e6),
    (r"تریلیون", 1e12), (r"بیلیون", 1e9), (r"هزار", 1e3), (r"همت", 1e12),
    # English words/abbreviations must start a word: 'mn' inside 'column' or 'autumn' is not a scale
    (r"(?<![A-Za-z])trillions?\b", 1e12), (r"(?<![A-Za-z])tn\b", 1e12), (r"(?<![A-Za-z])billions?\b", 1e9),
    (r"(?<![A-Za-z])bn\b", 1e9), (r"(?<![A-Za-z])bln\b", 1e9), (r"(?<![A-Za-z])millions?\b", 1e6),
    (r"(?<![A-Za-z])mn\b", 1e6), (r"(?<![A-Za-z])mln\b", 1e6), (r"(?<![A-Za-z])thousands?\b", 1e3),
]
_RE_SCALE_WORD = re.compile("|".join("(?:%s)" % p for p, _ in _SCALES), re.I)
# suffix letters attached to the digits: 12.5bn handled above; 12.5B / 12.5M / 12K
_RE_SCALE_SUFFIX = re.compile(r"([KkMmBb])(?![A-Za-z])")
_SUFFIX_FACTOR = {"k": 1e3, "m": 1e6, "b": 1e9}

_CURRENCIES = [
    (r"همت", "irr", 10.0),                     # hezar-e miliard toman
    (r"ریال|rials?\b|irr\b", "irr", 1.0),
    (r"تومان|tomans?\b", "irr", 10.0),
    (r"دلار|dollars?\b|usd\b|us\$|\$", "usd", 1.0),
    (r"یورو|euros?\b|eur\b|€", "eur", 1.0),
]
_RE_PERCENT = re.compile(r"%|درصد|percent\b|per[\s-]?cent\b|pct\b", re.I)

_RE_DATE = re.compile(r"(?<!\d)\d{1,4}/\d{1,2}/\d{1,4}(?!\d)")
_RE_NUMBER = re.compile(
    r"(?<![\d.])(?P<sign>[-−])?"
    r"(?P<num>\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)"
    r"(?:/(?P<frac>\d+))?(?![\d])")


@dataclass
class UnitInfo:
    scale: float = 1.0
    currency: str = None
    cur_factor: float = 1.0
    percent: bool = False
    has_scale_word: bool = False


def parse_unit(text):
    """Scale, currency and percent flag from a unit string such as 'billion rials',
    'میلیارد تومان', 'همت', '%', 'tonnes'."""
    t = normalize(text or "")
    info = UnitInfo()
    rest = t
    for m in _RE_SCALE_WORD.finditer(t):
        info.scale *= _scale_of(m.group(0))
        info.has_scale_word = True
    for pat, cur, fac in _CURRENCIES:
        if re.search(pat, rest, re.I):
            info.currency, info.cur_factor = cur, fac
            break
    info.percent = bool(_RE_PERCENT.search(t))
    return info


def _scale_of(word):
    w = normalize(word).lower()
    for pat, fac in _SCALES:
        if re.fullmatch(pat, w, re.I):
            return fac
    return 1.0


def _sig_digits(num_str):
    d = num_str.replace(",", "")
    if "." in d:
        return len(d.replace(".", "").lstrip("0")) or 1
    return len(d.lstrip("0").rstrip("0")) or 1


def _decimals(num_str):
    return len(num_str.split(".", 1)[1]) if "." in num_str else 0


@dataclass
class NumberCandidate:
    text: str
    value: float
    precision: float        # half-width of the rounding interval at the written precision, in raw units
    sig_digits: int
    scale: float = 1.0
    currency: str = None
    cur_factor: float = 1.0
    percent: bool = False
    explicit_scale: bool = False


def extract_numbers(text, cfg=DEFAULT_MATCH):
    """Every number in the text with the scale word / currency / percent that follows it
    (currency symbols may also precede it). Dates like 1401/05/12 are ignored."""
    t = strip_refs(text)
    t = _RE_DATE.sub(lambda m: " " * len(m.group(0)), t)
    out = []
    for m in _RE_NUMBER.finditer(t):
        num = m.group("num")
        sign = -1.0 if m.group("sign") else 1.0
        # a '-' directly after a digit/letter is a range dash, not a sign
        if m.group("sign") and m.start() > 0 and (t[m.start() - 1].isalnum()):
            sign = 1.0
        after = t[m.end():m.end() + 40]
        before = t[max(0, m.start() - 6):m.start()]
        scale, explicit = 1.0, False
        suf = _RE_SCALE_SUFFIX.match(after)
        if suf and not re.match(r"[KkMmBb]\s*(?:[²³]|[23]\b)", after):   # 5m2 / 5m³ are areas/volumes
            scale, explicit = _SUFFIX_FACTOR[suf.group(1).lower()], True
            after = after[suf.end():]
        else:
            pos = 0
            while True:
                sm = re.match(r"[\s‌]*(%s)" % _RE_SCALE_WORD.pattern, after[pos:], re.I)
                if not sm:
                    break
                scale *= _scale_of(sm.group(1))
                explicit = True
                pos += sm.end()
            after = after[pos:]
        currency, cur_factor = None, 1.0
        for pat, cur, fac in _CURRENCIES:
            if re.match(r"[\s‌]*(?:%s)" % pat, after, re.I) or re.search(r"(?:%s)\s*$" % pat, before, re.I):
                currency, cur_factor = cur, fac
                break
        if not currency and re.search(r"همت", t[m.end():m.end() + 25]):
            currency, cur_factor = "irr", 10.0
        percent = bool(re.match(r"[\s‌]*(?:%s)" % _RE_PERCENT.pattern, after, re.I))
        base_num = num.replace(",", "")
        try:
            val = float(base_num)
        except ValueError:
            continue
        prec = 0.5 * 10 ** (-_decimals(base_num))
        out.append(NumberCandidate(num, sign * val, prec, _sig_digits(num), scale, currency, cur_factor, percent, explicit))
        frac = m.group("frac")
        if frac and cfg.slash_decimal and "," not in num:
            try:
                v2 = float(base_num + "." + frac)
                out.append(NumberCandidate(num + "/" + frac, sign * v2, 0.5 * 10 ** (-len(frac)),
                                           _sig_digits(base_num + "." + frac), scale, currency, cur_factor, percent,
                                           explicit))
            except ValueError:
                pass
            try:
                out.append(NumberCandidate(frac, float(frac), 0.5, _sig_digits(frac)))
            except ValueError:
                pass
    return out


def _close(a, g, precision, sig, cfg):
    tol = max(cfg.abs_tol, cfg.rel_tol * abs(g))
    pairs = [(a, g)] + ([(abs(a), abs(g))] if cfg.sign_insensitive else [])
    for x, y in pairs:
        if abs(x - y) <= tol:
            return True
        if cfg.allow_rounding and sig >= cfg.min_sig_digits and abs(x - y) <= precision * (1 + 1e-9):
            return True
    return False


def numeric_match(answer_text, gold_value, gold_unit=None, gold_text=None, cfg=DEFAULT_MATCH):
    """{'match': bool, 'matched': text|None, 'n_candidates': int}.

    Convention: answer_value is the number as written together with answer_unit
    (e.g. 12.5 + 'billion rials'). A written number without a scale word is compared to
    answer_value and to answer_value x unit scale; a number with a scale word (or a rial/toman
    currency different from the gold's) is converted and compared to answer_value x scale.
    When answer_unit is empty, the unit is read from the gold answer text next to the gold number."""
    if gold_value is None:
        return {"match": None, "matched": None, "n_candidates": 0}
    g = float(gold_value)
    gu = parse_unit(gold_unit) if gold_unit else None
    if gu is None or (not gu.has_scale_word and not gu.currency and not gu.percent and gold_text):
        gu2 = _unit_from_gold_text(g, gold_text, cfg) if gold_text else None
        gu = gu2 or gu or UnitInfo()
    cands = extract_numbers(answer_text, cfg)
    gold_base = g * gu.scale
    for c in cands:
        conv = 1.0
        if c.currency == "irr" and gu.currency == "irr" and c.cur_factor != gu.cur_factor:
            conv = c.cur_factor / gu.cur_factor
        mult = c.scale * conv
        if mult != 1.0:
            tries = [(c.value * mult, gold_base, c.precision * mult)]
            if not gu.has_scale_word:
                tries.append((c.value * mult, g, c.precision * mult))
        else:
            tries = [(c.value, g, c.precision), (c.value, gold_base, c.precision)]
        for a, target, prec in tries:
            if _close(a, target, prec, c.sig_digits, cfg):
                return {"match": True, "matched": c.text, "n_candidates": len(cands)}
    return {"match": False, "matched": None, "n_candidates": len(cands)}


def _unit_from_gold_text(g, gold_text, cfg):
    for c in extract_numbers(gold_text, cfg):
        if _close(c.value, g, c.precision, 99, cfg) or math.isclose(abs(c.value), abs(g)):
            return UnitInfo(scale=c.scale, currency=c.currency, cur_factor=c.cur_factor, percent=c.percent,
                            has_scale_word=c.explicit_scale)
    return None


# --------------------------------------------------------------------------
# strings
# --------------------------------------------------------------------------

def token_f1(pred, gold):
    p, g = tokens(pred), tokens(gold)
    if not p or not g:
        return 0.0
    from collections import Counter
    common = sum((Counter(p) & Counter(g)).values())
    if not common:
        return 0.0
    prec, rec = common / len(p), common / len(g)
    return 2 * prec * rec / (prec + rec)


def _contains(hay, needle):
    n = len(needle)
    if not n:
        return False
    return any(hay[i:i + n] == needle for i in range(len(hay) - n + 1))


def _contains_joined(hay, needle):
    """Token-span containment that ignores where the spaces/ZWNJs fall inside the span:
    Persian writes one word as 'دیجی‌کالا', 'دیجی کالا' or 'دیجیکالا' (norm_text turns the ZWNJ
    into a space), so a contiguous run of answer tokens matches when its concatenation equals
    the concatenated gold tokens. Span boundaries stay on token boundaries, so 'Snapp' does not
    match inside 'Snappfood'."""
    target = "".join(needle)
    if not target:
        return False
    for i in range(len(hay)):
        acc = ""
        for j in range(i, len(hay)):
            acc += hay[j]
            if len(acc) >= len(target):
                if acc == target:
                    return True
                break
            if not target.startswith(acc):
                break
    return False


def string_match(answer_text, gold_texts):
    """gold_texts: gold answers to try, best first (question language, then the other).
    {'exact': bool, 'contains': bool, 'match': bool, 'f1': float (vs the first gold)}
    Exact/containment are insensitive to spaces vs ZWNJ vs no space inside a Persian word."""
    golds = [g for g in gold_texts if g and g.strip()]
    if not golds:
        return {"exact": None, "contains": None, "match": None, "f1": None}
    a = tokens(answer_text)
    exact = contains = False
    for gt in golds:
        gtok = tokens(gt)
        if gtok and (a == gtok or "".join(a) == "".join(gtok)):
            exact = True
        if gtok and (_contains(a, gtok) or _contains_joined(a, gtok)):
            contains = True
    return {"exact": exact, "contains": contains, "match": exact or contains, "f1": round(token_f1(answer_text, golds[0]), 4)}


# --------------------------------------------------------------------------
# evidence recall
# --------------------------------------------------------------------------

def _retrieved_unit_ids(record):
    out = []
    for it in record.get("retrieved_units") or []:
        for u in (it.get("unit_ids") or []) if isinstance(it, dict) else []:
            out.append(u)
    return out


def evidence_recall(record, gold_units):
    """{'n_gold', 'ev_hit', 'ev_recall', 'ret_hit', 'ret_recall'}; None values when there is no gold."""
    gold = list(dict.fromkeys(gold_units or []))
    if not gold:
        return {"n_gold": 0, "ev_hit": None, "ev_recall": None, "ret_hit": None, "ret_recall": None}
    ev = set(record.get("evidence_units") or [])
    ret = set(_retrieved_unit_ids(record)) | ev
    ev_n = sum(1 for u in gold if u in ev)
    ret_n = sum(1 for u in gold if u in ret)
    return {"n_gold": len(gold), "ev_hit": float(ev_n > 0), "ev_recall": ev_n / len(gold),
            "ret_hit": float(ret_n > 0), "ret_recall": ret_n / len(gold)}


# --------------------------------------------------------------------------
# one record
# --------------------------------------------------------------------------

NO_EVIDENCE_APPROACHES = ("B0",)


def _b(x):
    return None if x is None else float(bool(x))


def score_record(record, question, cfg=DEFAULT_MATCH, abstain_token=ABSTAIN_TOKEN, unit_resolver=None, series_cfg=None,
                 timeline_scoring="auto"):
    """Model-free scores for one RunRecord dict against its Question. Errors and
    abstentions on answerable questions count as incorrect.
    Timeline questions (answer_type "series") get the tl_* metrics of eval/timeline.py and
    match_correct = tl_series_correct; unit_resolver (unit id -> text) checks structured citations."""
    q_lang, d_lang = record["q_lang"], record["d_lang"]
    other = "en" if q_lang == "fa" else "fa"
    ans = record.get("answer_text") or ""
    errored = bool(record.get("error"))
    abst = detect_abstention(ans, abstain_token, cfg)
    at = question.answer_type
    out = {
        "question_id": record["question_id"], "approach": record["approach"], "approach_base": record.get("approach_base"),
        "flags": record.get("flags") or {}, "corpus": record["corpus"], "corpus_set": record.get("corpus_set"),
        "setting": record["setting"], "q_lang": q_lang, "d_lang": d_lang, "answer_type": at, "qtype": question.type,
        "answer_text": ans, "error": record.get("error"), "errored": float(errored),
        "abstained": float(abst), "abstained_exact": float(bool(record.get("abstained"))),
        "numeric_match": None, "string_exact": None, "string_contains": None, "string_match": None, "token_f1": None,
        "abstain_correct": None, "false_abstain": None,
    }
    if at == "unanswerable":
        out["abstain_correct"] = float(abst and not errored)
        out["match_correct"] = out["abstain_correct"]
    elif at == "series":
        from . import timeline as TL
        out["false_abstain"] = float(abst)
        tl = TL.score_series(question, record, unit_resolver=unit_resolver, cfg=series_cfg or TL.DEFAULT_SERIES_MATCH,
                             abstained=abst, scoring=timeline_scoring)
        out.update(tl)
        if timeline_scoring == "text" and TL.structured_of(record):
            # secondary: the same record scored from its structured series (tls_*), never the headline
            ts = TL.score_series(question, record, unit_resolver=unit_resolver, cfg=series_cfg or TL.DEFAULT_SERIES_MATCH,
                                 abstained=abst, scoring="structured")
            out.update({"tls_" + k[3:]: ts.get(k) for k in TL.SECONDARY_METRICS})
            out["tls_source"] = ts.get("tl_source")
        out["match_correct"] = float(bool(tl.get("tl_series_correct")) and not errored)
    else:
        out["false_abstain"] = float(abst)
        if at == "numeric" and question.answer_value is not None:
            nm = numeric_match(ans, question.answer_value, question.answer_unit, question.answer(q_lang) or question.answer(other), cfg)
            out["numeric_match"] = _b(nm["match"])
            out["numeric_matched_text"] = nm["matched"]
            ok = nm["match"]
        else:
            sm = string_match(ans, [question.answer(q_lang), question.answer(other)])
            out.update(string_exact=_b(sm["exact"]), string_contains=_b(sm["contains"]),
                       string_match=_b(sm["match"]), token_f1=sm["f1"])
            ok = sm["match"]
        # an abstention cannot contain the gold number/string, so it scores 0 through `ok`;
        # not gating on the (heuristic) abstention detector keeps answers like "وجود ندارد" matchable
        out["match_correct"] = float(bool(ok) and not errored)
    # key facts (locate / multi-section / cross-year questions): reported beside the strict
    # matcher, which keeps its numbers. Imported here to keep keyfacts.py free to import this module.
    from .keyfacts import score_key_facts
    kf = score_key_facts(question, ans, q_lang)
    if kf["kf_total"]:
        out.update(kf_total=kf["kf_total"], kf_found=kf["kf_found"], kf_recall=kf["kf_recall"],
                   kf_correct=(0.0 if errored else kf["kf_correct"]), kf_missing=kf["kf_missing"])
    if record.get("approach_base") in NO_EVIDENCE_APPROACHES:
        out.update({"n_gold": len(question.gold(d_lang)), "ev_hit": None, "ev_recall": None, "ret_hit": None, "ret_recall": None})
    else:
        out.update(evidence_recall(record, question.gold(d_lang) if at != "unanswerable" else []))
    lat = record.get("latency_ms")
    load = record.get("load_ms") or 0.0
    wait = record.get("lock_wait_ms") or 0.0      # GPU-lock wait behind other processes (records since answer-v2)
    out.update({
        "llm_calls": record.get("llm_calls"), "embed_calls": record.get("embed_calls"),
        "prompt_tokens": record.get("prompt_tokens"), "completion_tokens": record.get("completion_tokens"),
        "embed_tokens": record.get("embed_tokens"), "failed_calls": record.get("failed_calls"),
        "latency_ms": lat, "load_ms": load, "lock_wait_ms": wait,
        "latency_noload_ms": (max(0.0, lat - load - wait) if lat is not None else None),
        "evidence_chars": record.get("evidence_chars"), "config_hash": record.get("config_hash"),
        "index_fingerprint": record.get("index_fingerprint"), "run_id": record.get("run_id"),
        "split": record.get("split"), "ollama_version": (record.get("meta") or {}).get("ollama_version"),
    })
    # approach-reported per-question counters (trace.stats: OURS plan_repair/widen, B4 narrowed_items, ...) -> ts_<name>
    ts = (record.get("trace") or {}).get("stats") if isinstance(record.get("trace"), dict) else None
    if isinstance(ts, dict) and not errored:
        for k, v in ts.items():
            if isinstance(v, (int, float)):
                out["ts_" + str(k)] = float(v)
    return out
