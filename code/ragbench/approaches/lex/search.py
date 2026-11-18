"""LEX's model-free machinery: compile terms to regexes, route over the report cards, list the reports' own phrases,
search every unit of the chosen reports, and cut windows around the hits. No model call anywhere in this file.

Text is matched FOLDED on both sides (normalize.fold: yeh / kaf / alef variants, harakat, ZWNJ -> space, lower case),
so «کاركنان» and «کارکنان», «نرم‌افزار» and «نرم افزار» are one string before any regex runs. Terms become regexes
through the agentic planner's `term_patterns` (Persian plural and ezafe forms, English plurals) and `fold_pattern`.
"""
import math
import re
import time
from collections import defaultdict

from ...kb import labels as L
from ...kb.query import fold_pattern
from ...normalize import fold
from ..agentic.planner import WEAK_WORDS, fa_word_variants, term_patterns

_WEAK = {fold(w) for w in WEAK_WORDS}
_DIGIT = re.compile(r"\d")
_ESC = re.compile(r"(\\.)")
# words that say nothing about WHAT is asked: stop words, measure words (تعداد, share), periods, units, and the words
# every report uses about itself (a first smoke run spent a keyword on «گزارش سالانه», "annual report")
_DOC_WORDS = {fold(w) for w in """گزارش گزارشها گزارش‌ها سالانه سالیانه شرکت داده داده‌ها آمار اطلاعات سال
report reports annual yearly company data statistics figures information year""".split()}
NONCONTENT = L._STOP | L.GENERIC | L._PERIOD | L._UNIT | _WEAK | _DOC_WORDS


def content_words(text):
    """The folded words of a term that carry its meaning (see NONCONTENT)."""
    return [w for w in fold(text).split() if len(w) >= 2 and w not in NONCONTENT and not w.isdigit()]


# The relaxed (AND) match drops COUNT words - a report writes «کارکنان: 350» without «تعداد» - but keeps MEASURE-KIND
# words: "share of imports" is not "imports", a growth rate is not a level (second opinion, 2026-09-22: dropping every
# measure word let the AND match reduce «سهم واردات» to «واردات»).
_COUNT_WORDS = {fold(w) for w in """تعداد شمار کل جمع مجموع میزان مقدار رقم number count total sum amount figure
level""".split()}
_AND_DROP = L._STOP | L._PERIOD | L._UNIT | _DOC_WORDS | _COUNT_WORDS
_MEASURE = (L.GENERIC | _WEAK) - _COUNT_WORDS


def and_words(text):
    """-> the folded words a relaxed match requires, or [] when the term has no word beyond measure words (an AND of
    «نرخ» and «رشد» alone would match any growth rate of anything)."""
    ws = [w for w in fold(text).split() if len(w) >= 2 and w not in _AND_DROP and not w.isdigit()]
    return ws if any(w not in _MEASURE for w in ws) else []


# Word boundaries. `term_patterns` emits UNANCHORED substrings (a minimal stem covers its inflected forms), so «کار»
# (work) matched inside «کاربر» (user) and «وکار» of «کسب‌وکار» (business). Every pattern must now START a word; a short
# last word (< 4 letters) must also END one, allowing the common suffixes. Model-written regexes are left as written.
_LEFT = r"(?<![^\W\d_])"
_RIGHT = {"fa": r"(?:ها|های|ی|ان|ات)?(?![^\W\d_])", "en": r"(?:s|es)?(?![^\W\d_])"}


def bounded(pattern, text, lang):
    words = fold(text).split()
    right = _RIGHT.get(lang, _RIGHT["en"]) if words and len(words[-1]) < 4 else ""
    return _LEFT + "(?:%s)" % pattern + right


def _compile_word_patterns(text, lang):
    out = []
    for p in term_patterns(text, lang):
        try:
            out.append(re.compile(bounded(fold_pattern(p), text, lang), re.I))
        except re.error:
            continue
    return out


def _ms(t0):
    return round((time.perf_counter() - t0) * 1000.0, 1)


# ---------------------------------------------------------------------------------------------------------------
# terms -> regexes
# ---------------------------------------------------------------------------------------------------------------

class Term:
    """One search term: its text, where it came from, its weight, and its regexes over folded text.

    A term matches a text as a PHRASE (its words adjacent, in any spelling `term_patterns` allows), or - a term of
    several words - when every one of its CONTENT words occurs in the same text (`words`, AND). The second is not a
    loosening for its own sake: the reports' metric names are often assembled from a table's header and row label
    («تولید سالانه وزن کل»), so the phrase itself never occurs, and a smoke run found most picked phrases with
    no phrase match at all."""

    def __init__(self, text, source, weight, patterns, weak=False, words=None):
        self.text, self.source, self.weight, self.weak = text, source, float(weight), weak
        self.patterns = patterns                      # [compiled regex] over folded text
        self.words = words or []                      # [[compiled regex per variant]] per content word, AND
        self.df = 0
        self.idf = 1.0

    def phrase(self, folded):
        for p in self.patterns:
            m = p.search(folded)
            if m:
                return m.start()
        return -1

    def match(self, folded):
        """-> (offset, relaxed): the phrase first (relaxed False); else every AND word (relaxed True); (-1, None)."""
        off = self.phrase(folded)
        if off >= 0 or not self.words:
            return off, (False if off >= 0 else None)
        first = -1
        for alts in self.words:
            hit = min((m.start() for m in (a.search(folded) for a in alts) if m), default=-1)
            if hit < 0:
                return -1, None
            first = hit if first < 0 else min(first, hit)
        return first, True

    def search(self, folded):
        return self.match(folded)[0]

    def trace(self):
        return {"text": self.text, "source": self.source, "weight": self.weight, "weak": self.weak, "df": self.df,
                "idf": round(self.idf, 3), "patterns": [p.pattern for p in self.patterns][:4],
                "and_words": len(self.words)}


def compile_terms(texts, lang, source, weight, o):
    """Terms -> [Term] through the Persian-aware compiler. A term of measure words only (تعداد, total) is weak."""
    out, seen = [], set()
    for t in texts:
        f = fold(t)
        if not f or f in seen or sum(ch.isalpha() for ch in f) < int(o["min_term_letters"]):
            continue
        seen.add(f)
        pats = _compile_word_patterns(t, lang)
        if pats:
            weak = all(w in _WEAK or w in L._STOP for w in f.split())
            words = []
            if len(f.split()) >= 2:
                for w in and_words(t):
                    alts = _compile_word_patterns(w, lang)
                    if alts:
                        words.append(alts)
            out.append(Term(t, source, weight, pats, weak=weak, words=words))
    return out


def match_text(t, folded, table=False):
    """-> (offset, relaxed) of term t in a folded text, or (-1, None). A table is matched ROW BY ROW with its header
    line in front (a row «ایران | 31.4» under «کشور | سهم از واردات» answers "Iran's share of imports"), never across
    two rows; a match in the header alone counts at offset 0."""
    if not table:
        return t.match(folded)
    lines = folded.split("\n")
    head = lines[0] if lines else ""
    pos = len(head) + 1
    for ln in lines[1:]:
        off, rel = t.match(ln)
        if off < 0:
            off, rel = t.match(head + " " + ln)
        if off >= 0:
            return pos, rel
        pos += len(ln) + 1
    off, rel = t.match(head)
    return (0, rel) if off >= 0 else (-1, None)


def match_unit(t, u):
    return match_text(t, u.fold, table=(u.kind == "table"))


def row_match(t, head, ln):
    """-> relaxed flag when a table row is a match of its OWN (the term matches the row, or the row with its header
    while the header alone does not), else None."""
    off, rel = t.match(ln)
    if off >= 0:
        return rel
    off, rel = t.match(head + " " + ln)
    if off >= 0 and t.match(head)[0] < 0:
        return rel
    return None


def fold_regex(p):
    """A regex written by the model over NORMALISED text -> the same regex over FOLDED text. Escape sequences are
    kept as written (folding lower-cases, and `\\D` must not become `\\d`); the literal runs between them are folded
    with `fold_pattern`, keeping a space the fold would strip at a run's edge."""
    out = []
    for part in _ESC.split(p):
        if not part:
            continue
        if _ESC.fullmatch(part):
            out.append(part)
            continue
        core = fold_pattern(part)
        lead = " " if part[:1].isspace() and core else ""
        tail = " " if part[-1:].isspace() and core else ""
        out.append(lead + core + tail if core else (" " if part.strip() == "" else ""))
    return "".join(out)


def compile_regexes(patterns, source="llm_regex", weight=1.0):
    """Model-written regexes -> ([Term], [(pattern, error)]). One Term per regex; a regex that does not compile, or
    matches the empty string (it would hit every unit), is reported, not used."""
    terms, bad = [], []
    for p in patterns:
        try:
            rx = re.compile(fold_regex(p), re.I)
        except re.error as e:
            bad.append((p, "compile: %s" % str(e)[:80]))
            continue
        if rx.search(""):
            bad.append((p, "matches the empty string"))
            continue
        terms.append(Term(p, source, weight, [rx]))
    return terms, bad


# ---------------------------------------------------------------------------------------------------------------
# the reports: units by section, folded once
# ---------------------------------------------------------------------------------------------------------------

class Unit:
    __slots__ = ("id", "doc_id", "section_id", "ord", "kind", "text", "fold", "span", "table_id", "idx")

    def __init__(self, r):
        self.id, self.doc_id, self.section_id = r["unit_id"], r["doc_id"], r["section_id"]
        self.ord, self.kind, self.text = r["ord"], r["kind"], r["text"] or ""
        self.span = [r["span_start"], r["span_end"]]
        self.table_id = r["table_id"]
        self.fold = fold(self.text) if self.kind != "table" else "\n".join(fold(x) for x in self.text.split("\n"))
        self.idx = 0


class Reports:
    """The chosen reports' units, grouped by section in document order, plus section titles and fact labels."""

    def __init__(self, con):
        self.con = con
        self.units = {}          # doc_id -> [Unit] in ord order
        self.by_section = {}     # (doc_id, section_id) -> [Unit]
        self.sections = {}       # (doc_id, section_id) -> {title, path}
        self.titles = {}

    def load(self, doc_id):
        if doc_id in self.units:
            return
        us = []
        for r in self.con.execute("SELECT unit_id, doc_id, section_id, ord, kind, text, span_start, span_end, table_id "
                                  "FROM units WHERE doc_id = ? ORDER BY ord", (doc_id,)):
            us.append(Unit(r))
        self.units[doc_id] = us
        for u in us:
            lst = self.by_section.setdefault((doc_id, u.section_id), [])
            u.idx = len(lst)
            lst.append(u)
        for r in self.con.execute("SELECT section_id, title, path FROM sections WHERE doc_id = ?", (doc_id,)):
            self.sections[(doc_id, r["section_id"])] = {"title": r["title"] or "", "path": r["path"] or ""}
        row = self.con.execute("SELECT title FROM documents WHERE doc_id = ?", (doc_id,)).fetchone()
        self.titles[doc_id] = (row["title"] if row else None) or doc_id

    def section_units(self, u):
        return self.by_section[(u.doc_id, u.section_id)]

    def breadcrumb(self, doc_id, section_id):
        s = self.sections.get((doc_id, section_id)) or {}
        return " > ".join(x for x in (self.titles.get(doc_id), s.get("path")) if x)


# ---------------------------------------------------------------------------------------------------------------
# layer 1: the cards
# ---------------------------------------------------------------------------------------------------------------

def doc_rows(con, doc_ids):
    ids = list(doc_ids)
    if not ids:
        return {}
    return {r["doc_id"]: dict(r) for r in con.execute(
        "SELECT doc_id, company, year_from, year_to, title, title_fold, lang FROM documents WHERE doc_id IN (%s)"
        % ",".join("?" * len(ids)), ids)}


def card_tokens(con, doc_ids, titles_only=False, no_labels=False):
    """-> {doc_id: [folded tokens]}: the card (title, company, years, section titles, metric and entity names); with
    `titles_only` just the title, company and years; with `no_labels` the title, company, years and SECTION TITLES
    only - what a card is without the fact and label layers (LEX "lite")."""
    rows = doc_rows(con, doc_ids)
    cards = {}
    ids = list(rows)
    if no_labels and ids:
        secs = defaultdict(list)
        for r in con.execute("SELECT doc_id, title_fold FROM sections WHERE doc_id IN (%s) AND title_fold IS NOT NULL"
                             % ",".join("?" * len(ids)), ids):
            secs[r["doc_id"]].append(r["title_fold"])
        for d, r in rows.items():
            cards[d] = " | ".join([r["title_fold"] or "", r["company"] or "", str(r["year_from"] or "")] + secs[d])
    elif not titles_only and ids:
        for r in con.execute("SELECT doc_id, card_fold FROM doc_cards WHERE doc_id IN (%s)" % ",".join("?" * len(ids)),
                             ids):
            cards[r["doc_id"]] = r["card_fold"] or ""
    out = {}
    for d, r in rows.items():
        text = cards.get(d) if d in cards else " ".join(str(x) for x in (r["title_fold"] or "", r["company"] or "",
                                                                         r["year_from"] or "", r["year_to"] or ""))
        out[d] = [t for t in re.split(r"[\s|]+", text) if t]
    return out


def stem_hit(stem, word):
    """A query stem matches a word it begins (کاربر -> کاربران, کاربری); a stem shorter than 4 letters must BE the word:
    as a substring «کار» (work) matched «کاربر» (user) and «کسب‌وکار» (business) in the smoke run."""
    return word == stem if len(stem) < 4 else word.startswith(stem)


def query_stems(texts, lang):
    """Folded stems of the query words (Persian plural and -ان/-گان forms reduced), stop words and 1-letter words out."""
    out = []
    for t in texts:
        for w in fold(t).split():
            if len(w) < 2 or w in L._STOP:
                continue
            stems = fa_word_variants(w) if lang == "fa" else [w[:-1] if w.endswith("s") and len(w) > 4 else w]
            for s in stems:
                s = fold(s)
                if len(s) >= 2 and s not in out:
                    out.append(s)
    return out


def question_years(years, max_span=12):
    """The years a question asks about. Two or more years are read as the range they bound («از 1397 تا 1400» comes
    back from call 1 as its two ends, and a series needs 1398 and 1399 as well), up to `max_span` years."""
    ys = sorted({int(y) for y in years if str(y).isdigit()})
    if len(ys) >= 2 and ys[-1] - ys[0] <= max_span:
        return set(range(ys[0], ys[-1] + 1))
    return set(ys)


def route(con, doc_ids, kw, lang, o, titles_only=False, no_labels=False):
    """Layer 1. BM25 of the query stems over the cards (a stem matches a card token that contains it; the card's
    length normalises, as CORI / ReDDE warn it must when reports differ in size), a name the question gives in the
    report's title, and the report's own year among the question's. -> (routed doc ids, [(doc, score)])"""
    toks = card_tokens(con, doc_ids, titles_only, no_labels)
    rows = doc_rows(con, doc_ids)
    stems = query_stems(kw["keywords"] + kw["names"], lang)
    n = len(toks) or 1
    avg = sum(len(t) for t in toks.values()) / n or 1.0
    tf = {d: {s: sum(1 for t in ts if stem_hit(s, t)) for s in stems} for d, ts in toks.items()}
    df = {s: sum(1 for d in toks if tf[d][s] > 0) for s in stems}
    k1, b = float(o["bm25_k1"]), float(o["bm25_b"])
    names = [fold(x) for x in kw["names"] if fold(x)]
    years = question_years(kw["years"])
    scored = []
    for d, ts in toks.items():
        s = 0.0
        for q in stems:
            f = tf[d][q]
            if f:
                idf = math.log(1.0 + (n - df[q] + 0.5) / (df[q] + 0.5))
                s += idf * f * (k1 + 1.0) / (f + k1 * (1.0 - b + b * len(ts) / avg))
        r = rows.get(d) or {}
        head = "%s %s" % (r.get("title_fold") or "", r.get("company") or "")
        if any(nm in head or any(w in head for w in nm.split() if len(w) >= 3) for nm in names):
            s += float(o["w_route_name"])
        y0, y1 = r.get("year_from"), r.get("year_to") or r.get("year_from")
        if y0 is not None and any(y0 <= y <= y1 for y in years):
            s += float(o["w_route_year"])
        scored.append((d, round(s, 4)))
    scored.sort(key=lambda x: (-x[1], x[0]))
    keep = [d for d, _s in scored[: int(o["route_k"])]]
    if scored and len(years) >= 2:                   # a series: the top report's company in every year asked
        comp = (rows.get(scored[0][0]) or {}).get("company")
        for d, _s in scored:
            r = rows.get(d) or {}
            if len(keep) >= int(o["route_max"]):
                break
            if d not in keep and comp and r.get("company") == comp and r.get("year_from") in years:
                keep.append(d)
    return keep, scored


# ---------------------------------------------------------------------------------------------------------------
# call 2's list: the reports' own phrases
# ---------------------------------------------------------------------------------------------------------------

def vocabulary(con, reports, doc_ids, kw, lang, first_windows, o):
    """The phrases call 2 chooses from, all taken from the reports themselves:
      - section titles and clean metric names of the chosen reports that hold a stem of a keyword;
      - the section titles and metric names inside the best windows of a first search with the keywords (they are
        what the reports call the thing near where the keywords matched - pseudo-relevance feedback).
    Ranked by how many keywords they share, first-pass phrases first on a tie; at most `vocab_max`."""
    stems = [s for s in query_stems(kw["keywords"], lang) if s not in NONCONTENT and len(s) >= 3]
    maxw = int(o["vocab_max_words"])
    labels = not o.get("no_labels")
    cand = {}

    def add(text, where, doc):
        t = " ".join((text or "").split())
        f = fold(t)
        if not f or len(f.split()) > maxw or sum(ch.isalpha() for ch in f) < 3:
            return
        c = cand.setdefault(f, {"text": t, "stems": set(), "feedback": False, "docs": set()})
        words = f.split()
        c["stems"].update(s for s in stems if any(stem_hit(s, w) for w in words))
        c["feedback"] = c["feedback"] or where == "feedback"
        c["docs"].add(doc)

    ids = list(doc_ids)
    if ids:
        ph = ",".join("?" * len(ids))
        for r in con.execute("SELECT DISTINCT doc_id, title FROM sections WHERE doc_id IN (%s) AND title IS NOT NULL"
                             % ph, ids):
            add(r["title"], "card", r["doc_id"])
        if labels:
            for r in con.execute("SELECT DISTINCT doc_id, metric_name FROM fact_labels WHERE doc_id IN (%s) "
                                 "AND label_quality = 'ok' AND metric_name IS NOT NULL" % ph, ids):
                add(r["metric_name"], "card", r["doc_id"])
    for w in first_windows[: int(o["vocab_feedback_windows"])]:
        s = reports.sections.get((w["doc_id"], w["section_id"])) or {}
        add(s.get("title"), "feedback", w["doc_id"])
        uids = w["unit_ids"]
        if uids and labels:
            for r in con.execute("SELECT DISTINCT metric_name FROM fact_labels fl JOIN facts f ON f.fact_id = fl.fact_id "
                                 "WHERE f.unit_id IN (%s) AND fl.label_quality = 'ok' AND fl.metric_name IS NOT NULL"
                                 % ",".join("?" * len(uids)), uids):
                add(r["metric_name"], "feedback", w["doc_id"])
    # a stem shared by many phrases («کاربر» in a user-heavy report) says less than a rare one («شاغل»)
    n = len(cand) or 1
    sdf = {s: sum(1 for c in cand.values() if s in c["stems"]) for s in stems}
    for c in cand.values():
        c["score"] = sum(math.log(1.0 + n / sdf[s]) for s in c["stems"])
    keep = [c for c in cand.values() if c["stems"] or c["feedback"]]
    keep.sort(key=lambda c: (-c["score"], not c["feedback"], len(c["text"]), c["text"]))
    cap = int(o["vocab_max"])
    if o.get("report_vocab") and len(ids) > 1:
        # PER-REPORT allocation (second opinion): the global shortlist favours the report whose wording matches the
        # keywords best, so a year that renames the metric gets no phrase at all. Every report first gets its own
        # best `cap / reports` phrases (at least 3), then the rest of the list is filled by score.
        quota = max(3, cap // len(ids))
        chosen, seen = [], set()
        for d in ids:
            mine = [c for c in keep if d in c["docs"] and c["text"] not in seen][:quota]
            for c in mine:
                chosen.append(c)
                seen.add(c["text"])
        for c in keep:
            if len(chosen) >= cap:
                break
            if c["text"] not in seen:
                chosen.append(c)
                seen.add(c["text"])
        keep = sorted(chosen[:cap], key=lambda c: (-c["score"], not c["feedback"], len(c["text"]), c["text"]))
    return [{"text": c["text"], "hits": len(c["stems"]), "score": round(c["score"], 3), "feedback": c["feedback"],
             "docs": sorted(c["docs"])} for c in keep[:cap]]


# ---------------------------------------------------------------------------------------------------------------
# search and windows
# ---------------------------------------------------------------------------------------------------------------

def search(reports, doc_ids, terms):
    """Every term over every unit of the chosen reports. -> ({unit_id: {term index: (folded offset, relaxed)}}, units by
    id). Sets each term's `df` and `idf` (log(1 + N / df) over the units searched)."""
    hits, by_id, n = {}, {}, 0
    for d in doc_ids:
        reports.load(d)
        for u in reports.units[d]:
            n += 1
            by_id[u.id] = u
            for i, t in enumerate(terms):
                off, rel = match_unit(t, u)
                if off >= 0:
                    hits.setdefault(u.id, {})[i] = (off, rel)
    for i, t in enumerate(terms):
        t.df = sum(1 for h in hits.values() if i in h)
        t.idf = math.log(1.0 + n / t.df) if t.df else 0.0
    return hits, by_id


def term_weight(t, o, relaxed=False):
    """A term's weight in a window: its source weight x its IDF, less for a term of measure words only, and less for a
    relaxed (AND) match than for the phrase itself."""
    return (t.weight * t.idf * (float(o["w_weak"]) if t.weak else 1.0)
            * (float(o["w_relaxed"]) if relaxed else 1.0))


def credit(terms, folded, o, table=False):
    """-> ({term index: relaxed}, score) of the terms that match the text the window actually SHOWS. Scoring from the
    emitted text, not from the units behind it: a table window that keeps five rows of a long table must not be
    credited with a term that matched a row it left out (second opinion, 2026-09-22)."""
    got = {}
    for i, t in enumerate(terms):
        off, rel = match_text(t, folded, table=table)
        if off >= 0:
            got[i] = rel
    return got, sum(term_weight(terms[i], o, rel) for i, rel in got.items())


def _table_window(u, terms, o, cap):
    """A table hit -> (text, span, kind, rows kept). The header line always; then the MATCHING rows, best first (a
    phrase match before a relaxed one, then more terms), each with `table_rows_around` neighbours, while `cap` allows;
    printed in document order. Before, rows were taken in document order up to the cap, so a matching row late in a
    long table could be displaced by the neighbours of earlier ones."""
    raw = u.text.split("\n")
    if len(u.text) <= cap:
        return u.text, list(u.span), "unit", list(range(len(raw)))
    folded = u.fold.split("\n")
    head = folded[0] if folded else ""
    ranked, offs = [], {}
    for i in range(1, min(len(raw), len(folded))):
        s = 0.0
        for t in terms:
            rel = row_match(t, head, folded[i])
            if rel is not None:
                s += term_weight(t, o, rel)
                off = t.match(folded[i])[0]
                offs.setdefault(i, max(off, 0))
        if s > 0:
            ranked.append((-s, i))
    ranked.sort()
    around = int(o["table_rows_around"])
    shown, total = {0: raw[0]}, len(raw[0]) + 1
    matched = [i for _s, i in ranked]
    order = matched or list(range(1, min(len(raw), 4)))
    for i in order:
        for j in [i] + [k for d in range(1, around + 1) for k in (i - d, i + d)]:
            if not (1 <= j < len(raw)) or j in shown:
                continue
            if total + len(raw[j]) + 1 <= cap:
                shown[j] = raw[j]
                total += len(raw[j]) + 1
            elif j == i and i in offs and cap - total - 1 >= 60:
                # a matching row longer than what is left (text cells run to hundreds of characters): the part of it
                # around the match, rather than no row at all
                room = cap - total - 1 - 2
                pos = int(offs[i] * len(raw[i]) / max(1, len(folded[i])))
                a = max(0, min(pos - room // 2, len(raw[i]) - room))
                piece = ("…" if a > 0 else "") + raw[i][a:a + room] + ("…" if a + room < len(raw[i]) else "")
                shown[i] = piece
                total += len(piece) + 1
    keep = sorted(shown)
    return "\n".join(shown[i] for i in keep), None, "window", keep


def structure_window(reports, u, terms, o):
    """The hit's own unit and its neighbours IN THE SAME SECTION, alternating before / after (a short hit - a label or
    a bare number - looks back first), until `window_chars`. A table hit: its header line and the matching rows."""
    cap = int(o["window_chars"])
    if u.kind == "table":
        text, span, kind, rows = _table_window(u, terms, o, int(cap * float(o["table_window_factor"])))
        return {"units": [u], "text": text, "span": span, "kind": kind, "table": True,
                "rows": rows, "rows_total": len(u.text.split("\n"))}
    sec = reports.section_units(u)
    lo = hi = u.idx
    total = len(u.text)
    back_first = sum(ch.isalpha() for ch in u.text) < 30 or u.kind in ("fact", "caption")
    order = ["prev", "next"] if back_first else ["next", "prev"]
    if u.kind == "heading":
        order = ["next"]
    stuck = set()
    while len(stuck) < len(order):
        for side in order:
            if side in stuck:
                continue
            j = lo - 1 if side == "prev" else hi + 1
            if j < 0 or j >= len(sec) or sec[j].kind == "table" or total + len(sec[j].text) + 2 > cap:
                stuck.add(side)
                continue
            total += len(sec[j].text) + 2
            if side == "prev":
                lo = j
            else:
                hi = j
    us = sec[lo:hi + 1]
    return {"units": us, "text": "\n\n".join(x.text for x in us), "span": [us[0].span[0], us[-1].span[1]],
            "kind": "window", "table": False}


def fixed_window(doc_text, reports, u, offset, o):
    """+- window_chars/2 raw characters around the hit. The folded offset is mapped into the unit's raw text by
    proportion (folding can shorten a string); the window is a verbatim slice of the report text."""
    half = int(o["window_chars"]) // 2
    raw_len = max(1, u.span[1] - u.span[0])
    c = u.span[0] + int(round(offset * raw_len / max(1, len(u.fold)))) if offset >= 0 else (u.span[0] + u.span[1]) // 2
    a, b = max(0, c - half), min(len(doc_text), c + half)
    us = [x for x in reports.units[u.doc_id] if x.span[1] > a and x.span[0] < b]
    return {"units": us or [u], "text": doc_text[a:b], "span": [a, b], "kind": "window", "table": False}


def windows(reports, hits, by_id, terms, kw, o, mode="structure", doc_texts=None, score_fn=None):
    """Hits -> scored windows, best first; a hit already inside a better window adds nothing.

    A window's score is the weighted coverage of the DISTINCT terms that match the text it shows (each term once,
    weighted by its source weight and IDF, less when weak or matched only relaxed), plus a number bonus when the
    answer is a number, a year bonus when the window names a year asked about, and a name bonus when it names an
    entity the question names."""
    t0 = time.perf_counter()
    unit_score = {uid: sum(term_weight(terms[i], o, rel) for i, (_off, rel) in h.items()) for uid, h in hits.items()}
    order = sorted(hits, key=lambda uid: (-unit_score[uid], by_id[uid].doc_id, by_id[uid].ord))
    years = [str(y) for y in sorted(question_years(kw.get("years") or []))]
    names = [fold(x) for x in kw.get("names") or [] if len(fold(x)) >= 2]
    covered, out = set(), []
    for uid in order:
        if uid in covered:
            continue
        u = by_id[uid]
        if mode == "fixed":
            w = fixed_window(doc_texts[u.doc_id], reports, u, min(off for off, _r in hits[uid].values()), o)
        else:
            w = structure_window(reports, u, terms, o)
        uids = [x.id for x in w["units"]]
        f = "\n".join(fold(x) for x in w["text"].split("\n")) if w["table"] else fold(w["text"])
        got, score = credit(terms, f, o, table=w["table"])
        if not got:                   # the hit's term fell outside what the window shows (a cut table, a fixed slice)
            covered.add(uid)
            continue
        if kw.get("answer") == "number" and _DIGIT.search(w["text"]):
            score += float(o["w_number"])
        if years and any(y in w["text"] for y in years):
            score += float(o["w_year"])
        if names and any(nm in f for nm in names):
            score += float(o["w_name"])
        covered.update(uids)
        out.append({"doc_id": u.doc_id, "section_id": u.section_id, "hit": uid, "unit_ids": uids, "text": w["text"],
                    "span": w["span"], "kind": w["kind"], "score": round(score, 4),
                    "terms": sorted(terms[i].text + ("~" if rel else "") for i, rel in got.items()),
                    "rows": w.get("rows"), "rows_total": w.get("rows_total")})
        # learn.py: the same candidate window, scored by a fitted model instead of the hand-set weights. The callback
        # may also hang the feature vector on the window (the training dump); it keeps the hand-set score.
        if score_fn is not None:
            out[-1]["hand_score"] = round(score, 4)
            out[-1]["table"] = w["table"]
            out[-1]["score"] = round(score_fn(out[-1], got, terms, kw, years, names), 4)
    out.sort(key=lambda w: (-w["score"], w["doc_id"], w["hit"]))
    return out, _ms(t0)


def overlap(a, b):
    """Share of window a that window b already shows: by span when both are verbatim slices, else 1.0 when they share
    a unit (two windows of one table)."""
    if a["doc_id"] != b["doc_id"]:
        return 0.0
    if a["span"] and b["span"]:
        inter = min(a["span"][1], b["span"][1]) - max(a["span"][0], b["span"][0])
        return max(0, inter) / max(1, a["span"][1] - a["span"][0])
    return 1.0 if set(a["unit_ids"]) & set(b["unit_ids"]) else 0.0


def select(wins, doc_order, o, *, reserve=False, cost=None, budget=None, prior=None, source_caps=None):
    """The windows that go to the pack, chosen against the CHARACTERS they cost (header included), so the shared
    packer - which cuts the first item that does not fit and stops there - never has to cut or drop one.

      1. `reserve` (a question over several reports: a series, a comparison): each report's best window first, in
         report order, so every year / company is in the pack;
      2. then every window by score (+ the report's card prior, when layer 1 routed), skipping one that repeats a
         chosen window (`overlap_max`) or does not fit what is left of the budget.

    Before, the windows were a round robin over EVERY report by COUNT: in open-corpus mode a one-report question got
    the two wrong routed reports as many windows as the right one, and the count ignored their length.
    -> (chosen windows in pack order, info)"""
    prior = prior or {}
    cost = cost or (lambda w: len(w["text"]) + 80)
    budget = int(budget or 6000)
    caps = dict(source_caps or {})
    by_source = defaultdict(int)
    ranked = sorted(wins, key=lambda w: (-(w["score"] + prior.get(w["doc_id"], 0.0)), w["doc_id"], w["hit"]))
    chosen, used = [], 0
    info = {"reserved": 0, "skipped_overlap": 0, "skipped_budget": 0, "skipped_source_cap": 0}

    def fits(w):
        nonlocal used
        if any(overlap(w, k) > float(o["overlap_max"]) for k in chosen):
            info["skipped_overlap"] += 1
            return False
        c = cost(w)
        if used + c > budget:
            info["skipped_budget"] += 1
            return False
        src = w.get("source")
        if src in caps and by_source[src] + c > caps[src]:
            info["skipped_source_cap"] += 1
            return False
        chosen.append(w)
        used += c
        by_source[src] += c
        return True

    if reserve:
        for d in doc_order:
            for w in ranked:
                if w["doc_id"] == d and fits(w):
                    info["reserved"] += 1
                    break
    for w in ranked:
        if len(chosen) >= int(o["max_windows"]):
            break
        if w not in chosen:
            fits(w)
    info["chars"] = used
    info["by_source"] = dict(by_source)
    return chosen, info
