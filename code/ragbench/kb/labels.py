"""The label layer: a short, typed name for every fact - the surface an agent regexes into. No model, no I/O beyond
the knowledge base, derived from `facts` and `units` and rebuilt in seconds.

WHY. `facts.label` holds the extractor's label WORDS, not a metric name. On the live database (2026-09-21) 83% of
distinct labels occur exactly once, the 90th percentile is 43 characters, and many are sentence fragments
("chairman says members now receive"). A table cell's label is its row cells and its column header glued together
- «ایران سهم از واردات» is Iran + share of imports - so the measure is hidden inside a string unique to one row, and
regex across documents finds nothing. `facts.def_class` is 'unspecified' for 87% of facts, because DEF_CLASSES
only knows qualifiers of people counts.

ADDITIVE, ON PURPOSE. Nothing here changes a column of `facts`, the `tri` index, SCHEMA_VERSION or TYPED_VERSION,
so `schema.component_versions` - and with it AGENTIC3's index fingerprint and every v3 record - is unchanged, and
no document is marked stale. The layer carries its own LABELS_VERSION in `meta` (`labels.version`): `build`
rebuilds every document when it moves and only the missing ones otherwise. Ingestion calls `build_doc` right after
a document's facts stage, so a report added with `kb add` has its labels in the same pass.

TABLES
  fact_labels   one row per fact; fact_id -> facts(fact_id) ON DELETE CASCADE
    metric_name     WHAT is measured, short; NULL when the label is not a metric name (label_quality says why)
    entity_name     WHO it is measured for, when the extractor glued an entity to the measure (the table row
                    «ایران» under the column «سهم از واردات»); NULL otherwise
    label_context   the label words left once the metric and entity names are taken out
    metric_fold     fold(normalize(metric_name)), the regex target; metric_solid, the same with spaces removed
    entity_fold / entity_solid   likewise
    label_quality   ok | junk | prose | url | too_long | numeric | empty
    split           table (row and column located by offset) | table_row (a period column: the row is the
                    measure) | cleaned (text and KPI labels) | table_fallback (a table cell whose header could not
                    be matched to its label; cleaned like text)
    def_tags        ',' + the sorted tags + ',' so that `def_tags LIKE '%,net,%'` is exact; ',' when none
  labels_tri    fts5 trigram over the SOLID forms of metric names, entity names and section titles - a PREFILTER,
                never a ranker, exactly like `tri`. kind: metric | entity | section; ref_id: fact_id / section_id.
  A trigger on `documents` removes a deleted document's labels_tri rows; fact_labels goes with its facts.

1. SPLITTING (split_fact). Table cells are split EXACTLY. The fact's char offsets locate its line and its '|' cell
   in the unit text (checked on 4,000 table facts: every offset lands on the value). The column header comes from
   lite._header_parts - the same function that built the label - so when the label ends with that header, the
   header is the measure and the row cells before it are the entity. A header that is a year was turned into a
   period by the extractor, so the label is then the row alone and the row is the measure. Text and KPI labels are
   cleaned: lite._words keeps letter words only (numbers go), edge stopwords go, and TRAILING unit, scale and
   period words go (they are already in unit_norm and year). Leading unit words stay: «ساعت کاری» is a metric.
   The quality gate, first failing rule wins:
     url       a www / http / com / html token
     empty     nothing left after cleaning (no letter word at all, or only stopwords)
     junk      every remaining word is a unit, scale, period, month or page-furniture word (JUNK): the hubs
               «صفحه», «سال», «ثانیه», «میلیون» that connect every report to every other
     entity    every content word is a place (PLACE: countries, Iran's provinces and main cities) or a people
               category («مرد», «زن»). The label names WHO or WHERE, so it becomes entity_name, not a metric. On the
               first build these were eleven of the fifteen most-shared "metric names".
     generic   every word is a bare quantity word with no object - «رشد», «تعداد», «سهم», "growth", "number". A
               measure OBJECT (users, orders, employees, revenue) is not generic: "users" across fourteen
               companies is the cross-company node a graph wants
     prose     a finite verb or a relative marker (VERB) - a sentence fragment, not a name. Participles that
               double as nouns or adjectives are NOT verbs here: «ثبت‌شده» is "registered", a first version that
               counted it lost exactly the registered-vs-active labels a driver timeline turns on
     too_long  more than MAX_METRIC_WORDS words
     ok
   Qualifiers are read into def_tags BEFORE cleaning, because the extractor's stopword list holds «کل» (total)
   and «نسبت» (ratio): a node is the pair (metric_name, def_tags), so «درآمد کل» and «درآمد» share a name and
   differ in a tag instead of being two unrelated strings.

2. SOLID MATCH KEY (solid, term_forms). fold() maps a zero-width non-joiner to a space, so «نرم‌افزار» and
   «نرم افزار» fold alike but the solid spelling «نرمافزار» does not, and a LIKE over folded text misses it: on the
   live `tri` (2026-09-21) «نرم‌افزار» returns 281 rows and «نرمافزار» returns 0; «کسب‌وکار» 1,309 against
   «کسب و کار» 509. solid() removes every space, which makes all three spellings one key. Removing spaces also
   removes word boundaries, so `match` reports whether a hit is a WORD hit (the folded term matched on word
   boundaries) or only a SOLID hit, and ranks word hits first. Scope: the short-name surfaces (metrics, entities,
   section titles). The unit full text and `tri` are not changed - v3 was measured on them as they are.

3. DEFINITION TAGS (def_tags_of). DEF_TAGS keeps the ten DEF_CLASSES qualifiers and adds what financial reports
   turn on: basis (consolidated / standalone), margin level (gross / net / operating), accumulation (cumulative),
   period basis (annual / half / quarterly / monthly / daily), statistic (average / median / max / min), scope
   (domestic / export / import / per_capita / share / ratio), change (growth), status (target / actual),
   adjustment (nominal / adjusted), and the measure families that get confused with each other (capacity,
   production, sales, revenue, profit, cost, users). Multi-valued: "consolidated net sales per year" carries four.
   These are the canonicaliser's protected vocabulary - two names that differ in a tag are never one node. Keywords
   are written in plain Persian and English and passed through the SAME fold() as the text at compile time, so the
   two sides cannot drift; each keyword is bounded on both sides, which is what keeps «خالص» (net) out of «ناخالص»
   (gross), «فعال» out of «غیرفعال», and "net" out of "internet". Read from the label and a WINDOW of
   TAG_WINDOW_CHARS of the unit text on each side of the value.
"""
import json
import re
import sqlite3
import sys
import time

from ..normalize import fold, normalize
from ..approaches.agentic.numbers import SCALES
from ..approaches.ours.lite import (EN_UNIT_WORDS, FA_UNIT_WORDS, STOP, _cells, _header_parts, _words)

LABELS_VERSION = "kb-labels-v2"      # v2: the adjectival «ثبت‌نامی» is a registered keyword
MAX_METRIC_WORDS = 6
TAG_WINDOW_CHARS = 40

DDL = """
CREATE TABLE IF NOT EXISTS fact_labels (
  fact_id INTEGER PRIMARY KEY REFERENCES facts(fact_id) ON DELETE CASCADE,
  doc_id TEXT NOT NULL,
  metric_name TEXT, metric_fold TEXT, metric_solid TEXT,
  entity_name TEXT, entity_fold TEXT, entity_solid TEXT,
  label_context TEXT, label_quality TEXT, split TEXT, def_tags TEXT);
CREATE INDEX IF NOT EXISTS ix_fact_labels_doc ON fact_labels (doc_id);
CREATE INDEX IF NOT EXISTS ix_fact_labels_metric ON fact_labels (metric_fold);
CREATE INDEX IF NOT EXISTS ix_fact_labels_quality ON fact_labels (label_quality);
CREATE VIRTUAL TABLE IF NOT EXISTS labels_tri USING fts5 (
  text, kind UNINDEXED, ref_id UNINDEXED, doc_id UNINDEXED, tokenize = 'trigram');
CREATE TRIGGER IF NOT EXISTS labels_tri_doc_delete AFTER DELETE ON documents BEGIN
  DELETE FROM labels_tri WHERE doc_id = old.doc_id;
END;
"""


def _f(s):
    return " ".join(fold(normalize(s or "")).split())


def solid(folded):
    """The ZWNJ-proof match key of an ALREADY folded string: every space removed."""
    return (folded or "").replace(" ", "")


def term_forms(term):
    """-> (folded, solid) for one query term, the two forms `match` tries."""
    f = _f(term)
    return f, solid(f)


# ---------------------------------------------------------------------------------------------------------------
# vocabularies (written plainly, folded once at import so they cannot drift from the text side)
# ---------------------------------------------------------------------------------------------------------------

_MONTHS = """فروردین اردیبهشت خرداد تیر مرداد شهریور مهر آبان آذر دی بهمن اسفند
january february march april may june july august september october november december
jan feb mar apr jun jul aug sep sept oct nov dec"""
_FURNITURE = """صفحه اسلاید شکل نمودار جدول تصویر منبع پیوست ضمیمه بخش فصل شماره ردیف
page pages slide slides figure fig table chart source appendix annex section chapter"""
_TIME = """سال سالها ماه ماهها روز روزها هفته ساعت دقیقه ثانیه بار دفعه مورد عدد رقم تاریخ
پایان ابتدا آغاز گذشته جاری قبل قبلی آینده بعد اخیر
year years month months day days week weeks hour hours minute minutes second seconds time times date
end beginning start last previous current next recent"""
# Finite verbs and relative markers only. Participles that double as nouns or adjectives are deliberately NOT here:
# «ثبت‌شده» (registered), «تعدیل‌شده» (adjusted), «به دست آمده» (obtained), «داده» (data), «یافته» (finding),
# "reported revenue" - a first version rejected those as prose and so dropped the registered-vs-active labels a
# driver-count timeline turns on.
_VERB_FA = """است هست هستند بود بودند بوده شد شدند شود شوند نشد می نمی باشد باشند دارد دارند داشت داشتند
داشته کرد کردند کند کنند کنیم نمود نماید گفت گفته گوید یافت یابد رسید رسد گرفت گیرد دهد داد دادند خواهد
خواهند توان تواند میتوان امد اید ایند گردید که"""
_VERB_EN = """is are was were be been being has have had says said say will would can could should may might must
did does do fell rose grew grows reached shows show showed according which who whom"""
_DISCOURSE = """یعنی مثلا مثلاً البته همچنین ضمنا ضمناً سپس بنابراین زیرا اینکه چنانکه همانطور درواقع واقعا
معادل حدودا حدوداً تقریبا تقریباً
ie eg etc also however therefore thus hence namely equivalent approximately about"""
# A label made only of these names WHERE or WHO, not WHAT: it becomes the entity, not the metric. Without this the
# most-shared "metric names" were Iran's provinces and «مرد» / «زن».
_PLACES = """ایران چین روسیه ترکیه هند امارات عراق افغانستان پاکستان امریکا آمریکا ایالات متحده اروپا آلمان انگلیس
انگلستان بریتانیا فرانسه ایتالیا اسپانیا ژاپن کره عربستان سعودی قطر کویت عمان بحرین برزیل اندونزی مالزی
ازبکستان قزاقستان ترکمنستان تاجیکستان قرقیزستان ارمنستان گرجستان آذربایجان سوریه لبنان اردن مصر کانادا
استرالیا سوئیس سوئد هلند بلژیک اتریش لهستان اوکراین آفریقا آسیا خاورمیانه
تهران اصفهان فارس خراسان رضوی خوزستان مازندران کرمان البرز گیلان سیستان بلوچستان کرمانشاه هرمزگان همدان یزد
لرستان اردبیل مرکزی قزوین گلستان کردستان زنجان بوشهر قم سمنان چهارمحال بختیاری ایلام کهگیلویه بویراحمد
مشهد شیراز تبریز کرج اهواز رشت ارومیه زاهدان اراک بندرعباس سنندج خرم‌آباد گرگان ساری بیرجند بجنورد شهرکرد
یاسوج کاشان کیش شرقی غربی شمالی جنوبی
شهر شهرها استان استانها کشور کشورها منطقه مناطق جهان دنیا
iran china russia turkey india uae emirates iraq afghanistan pakistan america united states europe germany uk
britain england france italy spain japan korea saudi arabia qatar kuwait oman bahrain brazil indonesia malaysia
uzbekistan kazakhstan turkmenistan tajikistan armenia georgia azerbaijan syria lebanon jordan egypt canada
australia switzerland sweden netherlands belgium austria poland ukraine africa asia middle east
city cities province provinces country countries region regions world global"""
_DIRECTION = {"شرقی", "غربی", "شمالی", "جنوبی", "رضوی", "east", "west", "north", "south"}
_PEOPLE = """مرد مردان زن زنان آقا آقایان خانم خانمها خانم‌ها بانوان پسر دختر پسران دختران
male female men women boys girls gentlemen ladies"""
# A label made only of these has no OBJECT: "growth" of what? Measure objects - users, orders, employees, revenue -
# are not here: "users" across fourteen companies is exactly the cross-company node a graph wants.
_GENERIC = """رشد افزایش کاهش تغییر تغییرات تعداد شمار سهم درصد نسبت میزان مقدار مبلغ رقم ارزش حجم سطح شاخص نرخ
کل جمع مجموع متوسط میانگین حداکثر حداقل بیشترین کمترین
growth increase decrease change changes number count share percent percentage ratio amount value volume level
index rate total sum average mean median max min"""
_URL = {"www", "http", "https", "com", "html"}

_UNIT = {_f(w) for w in FA_UNIT_WORDS | EN_UNIT_WORDS} | {_f(w) for w in SCALES}
_PERIOD = {_f(w) for w in (_MONTHS + " " + _TIME).split()}
_FURN = {_f(w) for w in _FURNITURE.split()}
JUNK = _UNIT | _PERIOD | _FURN
VERB = {_f(w) for w in (_VERB_FA + " " + _VERB_EN).split()}
_STOP = {_f(w) for w in STOP["fa"] | STOP["en"]} | {_f(w) for w in _DISCOURSE.split()}
PLACE = {t for w in _PLACES.split() for t in _f(w).split()}
PEOPLE = {_f(w) for w in _PEOPLE.split()}
GENERIC = {_f(w) for w in _GENERIC.split()}
_TRAIL = _UNIT | _PERIOD | _STOP | _FURN

# tag -> keywords. The first ten are reconcile.DEF_CLASSES; keep them first so the two agree where they overlap.
DEF_TAGS = {
    "active": "فعال | active",
    "inactive": "غیرفعال | غیر فعال | inactive | dormant",
    # the adjectival «ثبت‌نامی» ("registered", as in «راننده‌های ثبت‌نامی») is its own keyword: the right boundary
    # rejects a trailing «ی», and a first version missed exactly Snapp's registered-driver counts that way
    "registered": ("ثبت نام | ثبت شده | ثبت‌نام | ثبت‌نامی | ثبت نامی | ثبت‌نام‌شده | ثبت نام شده | registered | "
                   "sign up | signups | enrolled"),
    "total": "کل | مجموع | کلیه | جمع کل | total | overall | aggregate",
    "accounts": "حساب کاربری | حساب های کاربری | account | accounts",
    "fulltime": "تمام وقت | تمام‌وقت | full time | full-time",
    "parttime": "پاره وقت | پاره‌وقت | part time | part-time",
    "new": "جدید | جذب | new | newly | joined",
    "monthly": "ماهانه | در ماه | monthly | per month | a month",
    "daily": "روزانه | در روز | daily | per day | a day",
    "subgroup": "زن | زنان | مرد | مردان | women | female | men | male",
    "consolidated": "تلفیقی | consolidated",
    "standalone": "شرکت اصلی | جداگانه | standalone | separate | parent company",
    "gross": "ناخالص | gross",
    "net": "خالص | net",
    "operating": "عملیاتی | operating | operational",
    "cumulative": "تجمعی | انباشته | تاکنون | تا کنون | cumulative | accumulated | to date | year to date | ytd",
    "annual": "سالانه | سالیانه | در سال | annual | annually | yearly | per year | per annum",
    "half": "شش ماهه | شش‌ماهه | نیمه اول | نیمه دوم | نیم سال | semi annual | half year | h1 | h2",
    "quarterly": "فصلی | سه ماهه | سه‌ماهه | quarterly | quarter | q1 | q2 | q3 | q4",
    "average": "میانگین | متوسط | average | avg | mean",
    "median": "میانه | median",
    "max": "حداکثر | بیشترین | بالاترین | اوج | max | maximum | peak | highest",
    "min": "حداقل | کمترین | پایین ترین | min | minimum | lowest",
    "share": "سهم | درصد از | share | proportion | percentage of | portion",
    "ratio": "نسبت | ratio",
    "growth": "رشد | افزایش | کاهش | تغییر | growth | increase | decrease | change | yoy | year on year | cagr",
    "per_capita": "سرانه | به ازای هر | per capita | per user | per customer | per employee | per head",
    "domestic": "داخلی | domestic | local",
    "export": "صادرات | صادراتی | export | exports | exported",
    "import": "واردات | وارداتی | import | imports | imported",
    "target": ("هدف | هدفگذاری | برنامه | بودجه | پیش بینی | پیش‌بینی | برآورد | چشم انداز | target | budget | "
               "forecast | projected | projection | estimate | estimated | expected | plan | planned | guidance"),
    "actual": "واقعی | محقق | تحقق | actual | realised | realized | achieved",
    "nominal": "اسمی | nominal",
    "adjusted": "تعدیل شده | تعدیلی | adjusted | underlying | normalised | normalized",
    "capacity": "ظرفیت | capacity | nameplate | installed",
    "production": "تولید | production | output | produced",
    "sales": "فروش | sales | sold",
    "revenue": "درآمد | revenue | revenues | turnover",
    "profit": "سود | profit | profits | earnings | ebitda | ebit | net income",
    "cost": "هزینه | بهای تمام شده | cost | costs | expense | expenses | cogs",
    "users": "کاربر | کاربران | مشتری | مشتریان | user | users | customer | customers",
}
_B0 = r"(?:^|(?<=[\s(/,:;\-]))"         # a keyword starts the string or follows a space or light punctuation
_B1 = r"(?=$|[\s)/,:;.\-])"             # ... and ends it the same way


def _compile_tags():
    out = {}
    for tag, spec in DEF_TAGS.items():
        kws = sorted({_f(k) for k in spec.split("|") if _f(k)}, key=len, reverse=True)
        body = "|".join(re.escape(k) for k in kws)
        pre = r"(?<!غیر )(?<!غیر)" if tag == "active" else ""       # «غیر فعال» is inactive
        out[tag] = re.compile(_B0 + pre + "(?:" + body + ")" + _B1)
    return out


_TAG_RX = _compile_tags()


def def_tags_of(text):
    """-> sorted list of DEF_TAGS present in `text` (any language, any spelling: it is folded here)."""
    f = _f(text)
    return sorted(t for t, rx in _TAG_RX.items() if rx.search(f)) if f else []


def tags_field(tags):
    return "," + ",".join(tags) + "," if tags else ","


# ---------------------------------------------------------------------------------------------------------------
# splitting and cleaning
# ---------------------------------------------------------------------------------------------------------------

def _clean_words(text):
    """-> the metric words of `text`: letter words, edge stopwords off, trailing unit / scale / period words off."""
    ws = [w for _s, _e, w in _words(normalize(text or ""))]
    while ws and _f(ws[0]) in _STOP:
        ws.pop(0)
    while ws and _f(ws[-1]) in _TRAIL:
        ws.pop()
    return ws


_VERB_PREFIX = {"می", "نمی"}                         # «می‌کنند», «نمی‌شود»: the present-tense prefix
_VERB_SUFFIX = {"اند", "است", "ایم", "اید", "شد"}      # «ثبت‌شد», «شده‌اند»: a finite ending on a joined word


def _is_verb(folded_word):
    """A folded WORD (its pieces are the parts a ZWNJ joined) is a finite verb when the whole word is one, when it
    starts with the present-tense prefix, or when it ends in a finite ending. «ثبت‌شده» ends in the participle
    «شده» and is not a verb; «ثبت‌شد» ends in the finite «شد» and is."""
    parts = folded_word.split()
    if folded_word in VERB:
        return True
    return len(parts) > 1 and (parts[0] in _VERB_PREFIX or parts[-1] in _VERB_SUFFIX)


def quality_of(words, raw_words=None):
    """The label-quality code of already cleaned words (first failing rule wins). `raw_words` are the label's
    letter words BEFORE trimming: a label that was nothing but page furniture («صفحه», "Slide") trims to nothing,
    and it is reported as junk rather than empty because that is the number worth knowing.

    Membership is checked per WORD, not per space-split piece, because fold() turns the ZWNJ inside «ثبت‌شده»
    into a space: the word is one unit of meaning and must be judged as one."""
    fw = [_f(w) for w in words]
    toks = [t for w in fw for t in w.split()]
    raw = [t for w in (raw_words or []) for t in _f(w).split()]
    if any(t in _URL for t in toks + raw):
        return "url"
    if not words:
        return "junk" if raw and all(t in JUNK or t in _STOP for t in raw) else "empty"
    if all(t in JUNK for t in toks):
        return "junk"
    content = [w for w in fw if w not in _STOP]
    if content and all(all(t in PLACE or t in _STOP for t in w.split()) for w in content) \
            and any(t not in _DIRECTION for w in content for t in w.split() if t in PLACE):
        return "entity"
    if content and all(w in PEOPLE for w in content):
        return "entity"
    if all(w in GENERIC or w in _STOP for w in fw):
        return "generic"
    if any(_is_verb(w) for w in fw):
        return "prose"
    if len(words) > MAX_METRIC_WORDS:
        return "too_long"
    return "ok"


def _table_cell(unit_text, char_start):
    """-> (line index, cell index, header label) of the value at `char_start`, or None."""
    lines = unit_text.split("\n")
    pos, li = 0, None
    for i, line in enumerate(lines):
        if pos <= char_start <= pos + len(line):
            li = i
            break
        pos += len(line) + 1
    if li is None:
        return None
    off, acc, col = char_start - pos, 0, None
    for j, cell in enumerate(lines[li].split("|")):
        if acc <= off <= acc + len(cell):
            col = j
            break
        acc += len(cell) + 1
    header_line = next((l for l in lines if l.strip()), "")
    header = _cells(header_line)
    if col is None or col >= len(header) or li == lines.index(header_line):
        return None
    return li, col, _header_parts(header[col])[0]


def split_fact(label, source_kind, unit_text="", char_start=None):
    """-> (metric words, entity words, split kind). The metric and entity words are cleaned; quality is separate."""
    lab = " ".join(normalize(label or "").split())
    if source_kind == "table":
        cell = _table_cell(unit_text, char_start) if unit_text and char_start is not None else None
        if cell is not None:
            header = " ".join(normalize(cell[2] or "").split())
            if not header:
                return _clean_words(lab), [], "table_row"          # a year column: the row is the measure
            if lab == header:
                return _clean_words(header), [], "table"
            if lab.endswith(" " + header):
                return _clean_words(header), _clean_words(lab[:-len(header) - 1]), "table"
        return _clean_words(lab), [], "table_fallback"
    return _clean_words(lab), [], "cleaned"


def _context(label, metric, entity):
    rest = " " + " ".join(normalize(label or "").split()) + " "
    for part in (metric, entity):
        if part:
            rest = rest.replace(" " + part + " ", " ", 1)
    return " ".join(rest.split()) or None


def label_row(fact, unit_text=""):
    """One `fact_labels` row for one `facts` row (a dict with fact_id, doc_id, label, source_kind, char_start,
    char_end) and the text of its unit."""
    label = fact.get("label") or ""
    mw, ew, how = split_fact(label, fact.get("source_kind"), unit_text, fact.get("char_start"))
    q = quality_of(mw, [w for _s, _e, w in _words(normalize(label))])
    cs, ce = fact.get("char_start"), fact.get("char_end")
    window = ""
    if unit_text and cs is not None and not how.startswith("table"):
        # a table's label already holds its row and its header; a character window would cross into the
        # neighbouring cells and rows (a "Founded" column picked up "active" from the next column)
        window = unit_text[max(0, cs - TAG_WINDOW_CHARS):(ce if ce is not None else cs) + TAG_WINDOW_CHARS]
    tags = def_tags_of(label + " \n " + window)
    metric = " ".join(mw) if q == "ok" else None
    if q == "entity" and not ew:
        ew, q = mw, "entity"                                  # the label names who / where: it is the entity
    entity = " ".join(ew) if ew and quality_of(ew) not in ("url", "empty", "junk") else None
    mf, ef = _f(metric), _f(entity)
    return {"fact_id": fact["fact_id"], "doc_id": fact["doc_id"],
            "metric_name": metric, "metric_fold": mf or None, "metric_solid": solid(mf) or None,
            "entity_name": entity, "entity_fold": ef or None, "entity_solid": solid(ef) or None,
            "label_context": _context(label, " ".join(mw), " ".join(ew)), "label_quality": q, "split": how,
            "def_tags": tags_field(tags)}


# ---------------------------------------------------------------------------------------------------------------
# building
# ---------------------------------------------------------------------------------------------------------------

_COLS = ("fact_id", "doc_id", "metric_name", "metric_fold", "metric_solid", "entity_name", "entity_fold",
         "entity_solid", "label_context", "label_quality", "split", "def_tags")


def ensure_labels_schema(con):
    con.executescript(DDL)


def build_doc(con, doc_id, commit=True, clear=True):
    """(Re)build the label layer of ONE document from its facts, units and sections. Idempotent. -> stats.

    `clear=False` skips deleting the document's old rows; `build` uses it after wiping both tables at once, because
    `labels_tri.doc_id` is UNINDEXED and a per-document DELETE scans the whole index (a full rebuild went from 17 s
    to 42 s that way)."""
    t0 = time.perf_counter()
    ensure_labels_schema(con)
    if clear:
        con.execute("DELETE FROM fact_labels WHERE doc_id = ?", (doc_id,))
        con.execute("DELETE FROM labels_tri WHERE doc_id = ?", (doc_id,))
    units = {r["unit_id"]: r["text"] for r in con.execute(
        "SELECT unit_id, text FROM units WHERE doc_id = ?", (doc_id,))}
    rows, tri = [], []
    for f in con.execute("SELECT fact_id, doc_id, label, source_kind, char_start, char_end, unit_id "
                         "FROM facts WHERE doc_id = ?", (doc_id,)).fetchall():
        r = label_row(f, units.get(f["unit_id"]) or "")
        rows.append(tuple(r[c] for c in _COLS))
        if r["metric_solid"]:
            tri.append((r["metric_solid"], "metric", str(r["fact_id"]), doc_id))
        if r["entity_solid"]:
            tri.append((r["entity_solid"], "entity", str(r["fact_id"]), doc_id))
    for s in con.execute("SELECT section_id, title_fold FROM sections WHERE doc_id = ?", (doc_id,)):
        if s["title_fold"]:
            tri.append((solid(s["title_fold"]), "section", s["section_id"], doc_id))
    con.executemany("INSERT INTO fact_labels (%s) VALUES (%s)" % (", ".join(_COLS), ",".join("?" * len(_COLS))),
                    rows)
    con.executemany("INSERT INTO labels_tri (text, kind, ref_id, doc_id) VALUES (?, ?, ?, ?)", tri)
    if commit:
        con.commit()
    return {"doc_id": doc_id, "facts": len(rows), "tri": len(tri), "ms": round((time.perf_counter() - t0) * 1000, 1)}


def build(con, doc_ids=None, rebuild=False, log=None):
    """Build the layer for every document that lacks it, or all of them when LABELS_VERSION moved or `rebuild`.
    Each document commits on its own, so a stop loses at most one. -> summary."""
    from .schema import get_meta, set_meta
    ensure_labels_schema(con)
    t0 = time.perf_counter()
    stale = rebuild or get_meta(con, "labels.version") != LABELS_VERSION
    docs = doc_ids or [r["doc_id"] for r in con.execute(
        "SELECT doc_id FROM documents WHERE status IN ('ready', 'vector_ready') ORDER BY doc_id")]
    wipe = stale and not doc_ids
    if wipe:
        con.execute("DELETE FROM fact_labels")
        con.execute("DELETE FROM labels_tri")
        con.execute("DELETE FROM meta WHERE key = 'labels.version'")    # a stop now leaves the layer marked stale
        con.commit()
    elif not doc_ids:
        have = {r["doc_id"] for r in con.execute("SELECT DISTINCT doc_id FROM fact_labels")}
        docs = [d for d in docs if d not in have]
    out = [build_doc(con, d, clear=not wipe) for d in docs]
    if stale and not doc_ids:
        set_meta(con, "labels.version", LABELS_VERSION)
        con.commit()
    if log:
        log("labels: %d documents, %d facts, %.1f s" % (len(out), sum(r["facts"] for r in out),
                                                         time.perf_counter() - t0))
    return {"documents": len(out), "facts": sum(r["facts"] for r in out), "tri": sum(r["tri"] for r in out),
            "seconds": round(time.perf_counter() - t0, 2), "version": LABELS_VERSION}


# ---------------------------------------------------------------------------------------------------------------
# querying
# ---------------------------------------------------------------------------------------------------------------

def _word_hit(folded_term, folded_text):
    """True when the term covers WHOLE words of the text once spacing is ignored: «نرمافزار» against
    «نرم افزار و کسب و کار» is the same word spelled solid (a word hit), while «زاروکس» starts inside «افزار» (an
    accident of removing spaces, a solid-only hit). A plain word-boundary match of the folded term is the easy case."""
    text = folded_text or ""
    if not folded_term or not text:
        return False
    if re.search(r"(?:^|\s)" + re.escape(folded_term) + r"(?:\s|$)", text):
        return True
    st = solid(folded_term)
    pos = [i for i, ch in enumerate(text) if ch != " "]
    flat = "".join(text[i] for i in pos)
    k = flat.find(st)
    while k != -1:
        a, b = pos[k], pos[k + len(st) - 1]
        if (a == 0 or text[a - 1] == " ") and (b == len(text) - 1 or text[b + 1] == " "):
            return True
        k = flat.find(st, k + 1)
    return False


def match(con, term, kinds=("metric", "entity", "section"), doc_ids=None, limit=500):
    """Rows of the label layer whose SOLID text contains the term's solid form, each marked `word` (the folded term
    matched on word boundaries) or `solid` (only once spaces were removed), word hits first.
    -> [{kind, ref_id, doc_id, text, hit}]"""
    f, s = term_forms(term)
    if not s:
        return []
    sql = "SELECT text, kind, ref_id, doc_id FROM labels_tri WHERE text LIKE ? AND kind IN (%s)" % \
        ",".join("?" * len(kinds))
    args = ["%" + s + "%", *kinds]
    if doc_ids:
        sql += " AND doc_id IN (%s)" % ",".join("?" * len(doc_ids))
        args += list(doc_ids)
    rows = con.execute(sql + " LIMIT ?", (*args, limit)).fetchall()
    if not rows:
        return []
    folded = {}
    fids = [int(r["ref_id"]) for r in rows if r["kind"] in ("metric", "entity")]
    for i in range(0, len(fids), 900):
        chunk = fids[i:i + 900]
        for x in con.execute("SELECT fact_id, metric_fold, entity_fold FROM fact_labels WHERE fact_id IN (%s)" %
                             ",".join("?" * len(chunk)), chunk):
            folded[("metric", str(x["fact_id"]))] = x["metric_fold"]
            folded[("entity", str(x["fact_id"]))] = x["entity_fold"]
    sids = [r["ref_id"] for r in rows if r["kind"] == "section"]
    for i in range(0, len(sids), 900):
        chunk = sids[i:i + 900]
        for x in con.execute("SELECT section_id, title_fold FROM sections WHERE section_id IN (%s)" %
                             ",".join("?" * len(chunk)), chunk):
            folded[("section", x["section_id"])] = x["title_fold"]
    out = [{"kind": r["kind"], "ref_id": r["ref_id"], "doc_id": r["doc_id"], "text": r["text"],
            "hit": "word" if _word_hit(f, folded.get((r["kind"], r["ref_id"]))) else "solid"} for r in rows]
    out.sort(key=lambda x: x["hit"] != "word")
    return out


def status(con):
    """-> the layer's measurements: coverage, quality mix, split mix, node sharing and tag coverage."""
    ensure_labels_schema(con)
    n = con.execute("SELECT count(*) AS n FROM fact_labels").fetchone()["n"]
    if not n:
        return {"facts": 0}
    q = {r["label_quality"]: r["c"] for r in con.execute(
        "SELECT label_quality, count(*) AS c FROM fact_labels GROUP BY label_quality")}
    sp = {r["split"]: r["c"] for r in con.execute("SELECT split, count(*) AS c FROM fact_labels GROUP BY split")}
    nodes = con.execute("""SELECT count(*) AS names, sum(CASE WHEN d = 1 THEN 1 ELSE 0 END) AS single_doc,
                                  sum(CASE WHEN d >= 2 THEN 1 ELSE 0 END) AS multi_doc,
                                  sum(CASE WHEN d >= 2 THEN c ELSE 0 END) AS facts_on_multi
                           FROM (SELECT metric_fold, count(DISTINCT doc_id) AS d, count(*) AS c
                                 FROM fact_labels WHERE metric_fold IS NOT NULL GROUP BY metric_fold)""").fetchone()
    tagged = con.execute("SELECT count(*) AS n FROM fact_labels WHERE def_tags != ','").fetchone()["n"]
    unspec = con.execute("SELECT count(*) AS n FROM facts WHERE def_class = 'unspecified'").fetchone()["n"]
    return {"version": LABELS_VERSION, "facts": n, "quality": q, "split": sp,
            "metric_names": nodes["names"], "names_in_1_doc": nodes["single_doc"],
            "names_in_2plus_docs": nodes["multi_doc"], "facts_on_shared_names": nodes["facts_on_multi"],
            "facts_with_a_tag": tagged, "facts_def_class_unspecified": unspec}


def main(argv=None):
    import argparse
    from .ingest import open_kb
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    ap = argparse.ArgumentParser(prog="python -m ragbench.kb.labels", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="build the label layer for documents that lack it (all, with --rebuild)")
    b.add_argument("--rebuild", action="store_true")
    b.add_argument("--doc", action="append")
    sub.add_parser("status", help="coverage, quality mix, shared names, tag coverage")
    m = sub.add_parser("match", help="look a term up in the metric / entity / section surface")
    m.add_argument("term")
    m.add_argument("--kind", action="append")
    m.add_argument("--limit", type=int, default=20)
    args = ap.parse_args(argv)
    con = open_kb(read_only=args.cmd != "build")
    if args.cmd == "build":
        print(json.dumps(build(con, doc_ids=args.doc, rebuild=args.rebuild, log=print), ensure_ascii=False))
    elif args.cmd == "status":
        print(json.dumps(status(con), ensure_ascii=False, indent=1))
    else:
        rows = match(con, args.term, kinds=tuple(args.kind or ("metric", "entity", "section")), limit=5000)
        print("%d rows (%d word hits)" % (len(rows), sum(1 for r in rows if r["hit"] == "word")))
        for r in rows[:args.limit]:
            print("  %-7s %-5s %-34s %s" % (r["kind"], r["hit"], r["doc_id"][:34], r["text"][:60]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
