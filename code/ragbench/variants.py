"""Two method options added 2026-09-23 from the demo experiments (demo/), OFF by default, so every existing variant and
result is unchanged. Turned on with the flags `doc_card` and `section_expand` (e.g. --ablation doc_card,section_expand
-> variant "LEX2+doc_card,section_expand"), for LEX / LEX2, AGENTIC3 and B2. Revised after the second-opinion review
of 2026-09-23 (work/analysis/DEMO_ASTRA_session.log): section expansion is list-intent only, keeps the retrieved
support, expands only what fits, and runs once at the final evidence boundary (Approach.final_pack).

doc_card        A description of each document in the question's scope, at most CARD_MAX characters, built from the
                document alone (title | frequent terms | years | top-level headings). Prepended to the user message of
                the PLANNING calls only (LEX: keywords, pick, regex; AGENTIC3: plan, plan_repair). It changes planning
                and therefore retrieval; the answer call itself never sees it. B2 has no planning call (no effect).
section_expand  When the question asks for several points / a list / a summary (list_intent), the best-supported
                section among the retrieved items - or a section whose heading states the question's count and was
                retrieved - is given WHOLE, placed after any synthetic leading items and before the method's other
                items, provided it fits in the evidence budget together with the retrieved support
                (the best retrieved item outside it, up to SUPPORT_RESERVE characters, must still fit). Items it already contains are
                dropped as duplicates. The answer call of such a question asks for a numbered list with a larger token
                allowance. Evidence, answer instruction and token allowance change together: evaluate as one package.

variants-v2 (2026-09-24, item 9 of the upgrade plan / P0 list of the multi-document design). The SAME two flags, with
the semantics Approach.variants_version selects (ragbench.run sets "variants-v2" for multi-document experiments; the
default "variants-v1" keeps the finished study's behaviour above bit for bit, for reproducibility):
  doc_card        card_shortlist(): cards are ranked by content-word overlap with the question (a method's routed
                  document ids, if it passes any, lead in their own order), the best CARD_SHORTLIST are shown, and the
                  selected ids, scores and order are logged (trace.variants.card_selection). v1 showed the first 12
                  cards in corpus order. Cards are built inside the measured query time (Approach.answer).
  section_expand  expand_sections_safe(): the same section choice, but the expansion is kept only when the ACTUAL
                  packed evidence still shows every document and every reserved requirement (synthetic lead items and
                  items with meta.reserved / meta.requirement) that the unexpanded pack showed; otherwise it is
                  abandoned. The answer instruction and allowance never change: they come from the question's
                  answer format (contracts.answer_format), identical for every arm. (v1's coupling - list rule and
                  LIST_NUM_PREDICT only when the expansion succeeded - survives only under variants-v1 and only for
                  the old default/timeline formats.)
Request identity: effective_request() is the exact (messages, num_predict) the wrapped client receives after the card
and list-rule injection; effective_request_sig() is its hash, or None when nothing was injected. The LEX and AGENTIC
call caches add it to their keys, so a revised card or rule never replays a stale reply.
"""
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

from .contracts import RetrievedItem
from .normalize import fold

VARIANTS_VERSION = "variants-v1"
VARIANTS_V2_VERSION = "variants-v2"
CARD_SHORTLIST = 12             # v2: cards shown after ranking (v1 showed the first 12 in corpus order)
VARIANTS_SHA1 = hashlib.sha1(Path(__file__).read_bytes()).hexdigest()[:12]
FLAGS = ("doc_card", "section_expand")
CARD_MAX = 300
PLANNING_PURPOSES = {"keywords", "pick", "regex", "plan", "plan_repair"}      # never "answer" or "verify"
ANSWER_PURPOSES = {"answer"}
LIST_NUM_PREDICT = 900
SUPPORT_RESERVE = 800           # the best retrieved item OUTSIDE the section must still fit next to it (up to this)
HEADER_ALLOWANCE = 250          # item headers and separators pack_evidence adds
MAX_SECTION_CHARS = 5200        # a larger section is not expanded (it would crowd the retrieved support out)

_STOP_EN = set("""the a an and or of to in on for with by from at as is are was were be been this that these those it its
he she his her they their we our you your i not but which who whom what when where how than then there here also into
over under about after before between during without within upon all any each more most other some such only own same
so too very can will just should now one two three may might must would could has have had do does did no nor if while
page pages chapter section table figure see note notes p pp ed eds vol tell me""".split())
_STOP_FA = set("""و در به از که این آن با برای را تا بر هم نیز یا اما اگر هر یک دو سه می شود شده است بود هست کرد کند
کرده های ها ای خود او ما شما آنها ایشان بین پس پیش روی زیر بالا چه چون همه دیگر باید نباید شد چیست کدام چیزی""".split())


# ---------------------------------------------------------------------------------------------------------------
# document card
# ---------------------------------------------------------------------------------------------------------------

def document_card(doc):
    fa = doc.lang == "fa"
    stop = _STOP_FA if fa else _STOP_EN
    text = doc.text
    # the whole document, sampled evenly (the first 400,000 characters alone missed later topics - review 2026-09-23)
    if len(text) > 400000:
        step = len(text) // 8
        text = "\n".join(text[i:i + 50000] for i in range(0, len(text), step))
    years = Counter(re.findall(r"\b(1[34]\d\d|19\d\d|20\d\d)\b", text))
    top_years = sorted(y for y, _ in years.most_common(3))
    heads = []
    for s in doc.sections:
        t = (s.title or "").strip()
        if s.level == 1 and t and t != doc.title and len(t) <= 70 and t not in heads \
                and t.lower() not in ("contents", "table of contents", "فهرست", "فهرست مطالب"):
            heads.append(t)
    words = re.findall(r"[^\W\d_]+(?:[‌'’-][^\W\d_]+)*", text)
    grams = Counter()
    for a, b in zip(words, words[1:]):
        if a.lower() in stop or b.lower() in stop or len(a) < 3 or len(b) < 3:
            continue
        if not fa and not (a[0].isupper() and b[0].isupper()):
            continue                                     # English: names and titled terms only
        grams[a + " " + b] += 1
    terms = [g for g, n in grams.most_common(12) if n >= 3][:6]
    parts = [doc.title or doc.id]
    if terms:
        parts.append(("واژه‌های پرتکرار: " if fa else "frequent terms: ") + ", ".join(terms))
    if top_years:
        parts.append(("سال‌های پرتکرار: " if fa else "most mentioned years: ") + ", ".join(top_years))
    if heads:
        room = CARD_MAX - len(" | ".join(parts)) - 12
        keep = []
        for h in heads[:6]:
            h = h if len(h) <= 50 else h[:49] + "…"
            if len("; ".join(keep + [h])) > room:
                break
            keep.append(h)
        if keep:
            parts.append(("بخش‌ها: " if fa else "parts: ") + "; ".join(keep))
    card = " | ".join(parts)
    return card if len(card) <= CARD_MAX else card[:CARD_MAX - 1].rstrip() + "…"


def card_context(cards, max_docs=12):
    cards = list(cards)
    more = len(cards) - max_docs
    body = "\n".join("- " + c for c in cards[:max_docs]) + ("\n- ... and %d more document(s)" % more if more > 0 else "")
    return ("Context - the document(s) being searched, described from the documents themselves. Use it to read names "
            "and terms in the question correctly; it is not an answer source.\n" + body)


_CARD_MEMO = {}


def _card_of(doc):
    """document_card, memoised per process by (doc id, source fingerprint): the card depends on the document alone."""
    key = (doc.id, (getattr(doc, "meta", None) or {}).get("fingerprint"), len(doc.text or ""))
    if key not in _CARD_MEMO:
        _CARD_MEMO[key] = document_card(doc)
    return _CARD_MEMO[key]


def card_shortlist(docs, question, routed_ids=None, max_docs=CARD_SHORTLIST):
    """variants-v2 cards: -> (context text, selection log). Routed ids (a method's own routing, best first) lead in
    their order; the other documents follow by the number of the question's content words their card shares (ties:
    corpus order). Only the best max_docs are shown; the log names every shown id with its score and rank."""
    docs = list(docs)
    qwords = _content_words(question)
    routed = [d for d in (routed_ids or [])]
    pos = {d: i for i, d in enumerate(routed)}
    rows = []
    for order, d in enumerate(docs):
        card = _card_of(d)
        score = len(qwords & _content_words(card))
        rows.append({"doc_id": d.id, "score": score, "routed": d.id in pos, "order": order, "card": card})
    rows.sort(key=lambda r: (0 if r["routed"] else 1, pos.get(r["doc_id"], 0), -r["score"], r["order"]))
    shown = rows[:max_docs]
    more = len(rows) - len(shown)
    body = "\n".join("- " + r["card"] for r in shown) + ("\n- ... and %d more document(s)" % more if more > 0 else "")
    text = ("Context - the document(s) being searched, described from the documents themselves, the ones closest to "
            "the question first. Use it to read names and terms in the question correctly; it is not an answer "
            "source.\n" + body)
    sel = {"version": VARIANTS_V2_VERSION, "n_candidates": len(rows), "n_shown": len(shown),
           "routed_ids": routed, "shown": [{"doc_id": r["doc_id"], "score": r["score"], "routed": r["routed"],
                                            "rank": i + 1} for i, r in enumerate(shown)]}
    return text, sel


# ---------------------------------------------------------------------------------------------------------------
# list intent
# ---------------------------------------------------------------------------------------------------------------

_NUM_FA = "(?:\\d+|[۰-۹]+|دو|سه|چهار|پنج|شش|هفت|هشت|ده|یازده|دوازده|پانزده|بیست)"
_NUM_EN = r"(?:\d+|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|fifteen|twenty)"
_LIST_FA = re.compile(_NUM_FA + r"\s*(?:نکته|نکات|مورد|دلیل|گام|اصل|توصیه|چیز|ویژگی|یافته|درس|پیشنهاد|گزاره)"
                      r"|خلاصه|نکات\s*(?:کلیدی|مهم|اصلی)|مهم[‌ ]?ترین\s*(?:نکات|یافته|موارد)"
                      r"|یافته[‌ ]?های\s*(?:اصلی|کلیدی)|فهرست\s*(?:کن|کنید|بده|بدهید)")
# "does the report list X" asks for one fact - only an imperative / plural-points wording counts (review 2026-09-23)
_LIST_EN = re.compile(r"\b" + _NUM_EN + r"\s+(?:\w+\s+)?(?:points|things|reasons|takeaways|findings|steps|lessons|"
                      r"recommendations|insights|principles)\b|\bsummar(?:y|ise|ize)\b|^\s*list\b|\blist (?:all|every)\b"
                      r"|\bkey (?:points|findings|takeaways)\b|\bmain (?:points|findings|ideas)\b|\btakeaways\b",
                      re.IGNORECASE)
_NUMWORDS = {"دو": 2, "سه": 3, "چهار": 4, "پنج": 5, "شش": 6, "هفت": 7, "هشت": 8, "ده": 10, "یازده": 11,
             "دوازده": 12, "پانزده": 15, "بیست": 20, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
             "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "fifteen": 15, "twenty": 20}


def list_intent(question):
    """True when the question asks for several points, a list or a summary (Persian or English)."""
    return bool(_LIST_FA.search(question or "") or _LIST_EN.search(question or ""))


def numbers_in(text):
    t = (text or "").translate(str.maketrans("۰۱۲۳۴۵۶۷۸۹", "0123456789"))
    out = {int(x) for x in re.findall(r"(?<!\d)(\d{1,2})(?!\d)", t)}
    for w in re.findall(r"[^\W\d_]+", t.lower()):
        if w in _NUMWORDS:
            out.add(_NUMWORDS[w])
    return out


def _content_words(text):
    ws = re.findall(r"[^\W\d_]{3,}", fold(text or "").lower())
    return {w for w in ws if w not in _STOP_EN and w not in _STOP_FA}


# ---------------------------------------------------------------------------------------------------------------
# section expansion (the final evidence boundary)
# ---------------------------------------------------------------------------------------------------------------

def _synthetic(it):
    """Items a method builds itself and expects first (AGENTIC3's reconciled fact table)."""
    return it.section_id is None and it.kind in ("fact", "table")


def expand_sections(items, corpus, question, budget):
    """-> (items, decision). Never removes retrieved support: the expanded section must fit next to SUPPORT_RESERVE
    characters of the top retrieved items outside it, or nothing changes."""
    dec = {"version": VARIANTS_VERSION, "applied": False, "list_question": list_intent(question)}
    if not dec["list_question"]:
        dec["reason"] = "no list/summary wording in the question"
        return items, dec
    qwords, qnums = _content_words(question), numbers_in(question)
    weight, support = Counter(), Counter()
    for rank, it in enumerate(items[:16]):
        if it.section_id and corpus.has_doc(it.doc_id):
            weight[it.section_id] += 1.0 / (1.0 + 0.3 * rank)
            support[it.section_id] += 1
    scored = {}
    for sid, w in weight.items():
        sec = corpus.doc(sid.split("::", 1)[0]).section(sid)
        count = bool(qnums and numbers_in(sec.title) & qnums)
        overlap = sorted(qwords & _content_words(sec.title))
        # a candidate needs its own reason: the question's count in its heading, a heading word in common, or at
        # least two retrieved items inside it (the global winner alone is not enough - review 2026-09-23)
        if count or overlap or support[sid] >= 2:
            scored[sid] = (w + (3.0 if count else 0.0) + (1.0 if overlap else 0.0), count, overlap)
    if not scored:
        dec["reason"] = "no retrieved section qualifies (count in heading, heading word, or 2+ retrieved items)"
        return items, dec
    best = max(scored, key=lambda s: scored[s][0])
    doc = corpus.doc(best.split("::", 1)[0])
    sec = doc.section(best)
    a, b = sec.subtree_span
    chosen = [best]
    # the same-titled next section continues it only when it is adjacent in document order (repeated headings)
    nxt = next((s for s in doc.sections if s.order == sec.order + 1), None)
    if nxt is not None and nxt.title == sec.title and nxt.parent == sec.parent:
        a2, b2 = nxt.subtree_span
        if (b - a) + (b2 - a2) <= MAX_SECTION_CHARS:
            chosen.append(nxt.id)
    spans = [doc.section(s).subtree_span for s in chosen]
    sec_chars = sum(y - x for x, y in spans)
    lead = [it for it in items if _synthetic(it)]
    inside = set()
    for s in chosen:
        inside |= {u.id for u in doc.section_units(s, recursive=True)}
    rest = [it for it in items if not _synthetic(it) and not (it.unit_ids and set(it.unit_ids) <= inside)]
    support_chars = min(SUPPORT_RESERVE, len(rest[0].text or "")) if rest else 0
    lead_chars = sum(len(it.text or "") for it in lead)
    need = lead_chars + sec_chars + support_chars + HEADER_ALLOWANCE
    dec.update(section=sec.title, sections=chosen, section_chars=sec_chars, count_match=scored[best][1],
               heading_words=scored[best][2])
    if sec_chars > MAX_SECTION_CHARS or need > budget:
        dec["reason"] = "section too large to fit with the retrieved support (%d chars, budget %d)" % (sec_chars, budget)
        return items, dec
    new = [RetrievedItem(text=doc.text[x:y], doc_id=doc.id, section_id=s, kind="section",
                         unit_ids=[u.id for u in doc.section_units(s, recursive=True)], span=[x, y],
                         meta={"section_expand": VARIANTS_VERSION})
           for s, (x, y) in zip(chosen, spans)]
    dec.update(applied=True, dropped_as_inside=len(items) - len(lead) - len(rest))
    return lead + new + rest, dec


def _reserved(it):
    """Items whose presence the expansion must not cost: synthetic lead items and items a method marks reserved."""
    m = it.meta or {}
    return _synthetic(it) or bool(m.get("reserved")) or m.get("requirement") is not None


def _shown_key(it):
    return (it.doc_id, it.section_id, it.kind, (it.text or "").strip()[:80])


def expand_sections_safe(items, corpus, question, budget, pack):
    """variants-v2 section expansion -> (items, decision). `pack` renders items exactly as the answer call will get
    them (Approach.pack). The expansion is kept only if the packed result after it shows every document and every
    reserved requirement that the packed result without it showed (a cut reserved item that loses units counts as
    lost); otherwise the original items are returned and the decision says why."""
    items2, dec = expand_sections(items, corpus, question, budget)
    dec["version"] = VARIANTS_V2_VERSION
    if not dec.get("applied"):
        return items, dec
    before, after = pack(items), pack(items2)
    docs_before = [d for d in dict.fromkeys(it.doc_id for it in before.items)]
    docs_after = set(it.doc_id for it in after.items)
    lost_docs = [d for d in docs_before if d not in docs_after]
    after_units = set(after.unit_ids)
    after_keys = {_shown_key(it) for it in after.items}
    lost_req = []
    for it in before.items:
        if not _reserved(it):
            continue
        name = (it.meta or {}).get("requirement") or "%s/%s" % (it.doc_id, it.section_id or it.kind)
        if it.unit_ids:
            if not set(it.unit_ids) <= after_units:
                lost_req.append(name)
        elif _shown_key(it) not in after_keys:
            lost_req.append(name)
    lost_units = [u for u in before.unit_ids if u not in after_units]
    dec.update(docs_before=docs_before, docs_after=sorted(docs_after), lost_units=len(lost_units),
               packed_chars_before=before.chars, packed_chars_after=after.chars)
    if lost_docs or lost_req:
        dec.update(applied=False, abandoned=True, lost_docs=lost_docs, lost_requirements=lost_req,
                   reason="abandoned: the packed result would lose %d document(s) and %d reserved requirement(s)" % (
                       len(lost_docs), len(lost_req)))
        return items, dec
    return items2, dec


LIST_ANSWER_RULE = ("Reply with the answer only. The question asks for several points: give them as a numbered list, "
                    "one short line each, in the order the evidence gives them, using only the evidence.")


def list_answer_messages(messages):
    out = [dict(m) for m in messages]
    for m in out:
        if m.get("role") == "system":
            c = m.get("content") or ""
            c2 = re.sub(r"Reply with the answer only, in one or two short sentences\.", LIST_ANSWER_RULE, c)
            m["content"] = c2 if c2 != c else c + "\n" + LIST_ANSWER_RULE
    return out


# ---------------------------------------------------------------------------------------------------------------
# the client wrapper
# ---------------------------------------------------------------------------------------------------------------

class VariantClient:
    """Wraps an approach's model client: the card on planning calls, the list rule on the answer call. Everything
    else (embed, ledger, health ...) is the wrapped client's."""

    def __init__(self, inner):
        self.__dict__["inner"] = inner
        self.__dict__["card"] = None
        self.__dict__["list_answer"] = False

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def __setattr__(self, name, value):
        if name in ("card", "list_answer", "inner"):
            self.__dict__[name] = value
        else:
            setattr(self.inner, name, value)

    def effective_request(self, messages, purpose, num_predict=None):
        """-> (messages, num_predict, changed): exactly what chat() hands the wrapped client."""
        changed = False
        if self.card and purpose in PLANNING_PURPOSES:
            messages = [dict(m) for m in messages]
            last = max(i for i, m in enumerate(messages) if m.get("role") == "user")
            messages[last]["content"] = self.card + "\n\n" + (messages[last].get("content") or "")
            changed = True
        if self.list_answer and purpose in ANSWER_PURPOSES:
            messages = list_answer_messages(messages)
            num_predict = max(int(num_predict or 0), LIST_NUM_PREDICT)
            changed = True
        return messages, num_predict, changed

    def effective_request_sig(self, messages, purpose, num_predict=None):
        """sha1 of the effective request after injection, or None when chat() would pass the request unchanged
        (then a call-cache key stays exactly what it was before variants existed)."""
        msgs, n, changed = self.effective_request(messages, purpose, num_predict)
        if not changed:
            return None
        blob = json.dumps({"messages": msgs, "num_predict": n}, sort_keys=True, ensure_ascii=False)
        return hashlib.sha1(blob.encode("utf-8")).hexdigest()

    def chat(self, messages, *, purpose, **kw):
        messages, n, changed = self.effective_request(messages, purpose, kw.get("num_predict"))
        if changed and (n is not None or "num_predict" in kw):
            kw["num_predict"] = n
        return self.inner.chat(messages, purpose=purpose, **kw)
