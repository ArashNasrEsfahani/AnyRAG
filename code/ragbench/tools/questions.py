"""Pilot question set: author, link gold evidence, validate, summarise, export a review sheet.

The source of truth is a human-written YAML file, ragbench/data/questions/pilot.src.yaml.
`build` resolves every evidence quote to corpus unit ids and writes pilot.jsonl in the
foundation's Question schema (ragbench.contracts.Question). Do not hand-edit pilot.jsonl:
unit ids in corpora B and C are extractor block ids and change when the PDF extractor
re-runs, so `build` re-links them from the quotes.

    python -m ragbench.tools.questions build              # link quotes -> unit ids, write pilot.jsonl
    python -m ragbench.tools.questions validate           # schema + gold-support checks (exit 1 on errors)
    python -m ragbench.tools.questions summary            # per corpus / type / answer type / setting
    python -m ragbench.tools.questions sheet              # review sheet CSV (one row per gold unit)
    python -m ragbench.tools.questions find 71,319 Saudi --set A_en --both
    python -m ragbench.tools.questions show A-rpt-a-01

Evidence model (YAML, per question):
    evidence:
      - fact: "what this item proves"      # one required fact; every item is needed for a full answer
        doc: A.rpt-a                        # "<corpus>.<key>" (language added per side); default: the only doc in scope
        en: ["71,319", "saudi"]            # quote terms: a unit is gold when its text contains ALL terms
        fa: ["71,319", "عربستان"]          # or alternatives: [["60,000", "sellers"], ["60 thousand", "sellers"]]
        only: [local ids]                  # optional: restrict candidates to these unit locals
        exclude: [local ids]               # optional: drop matches that do not support the fact on their own
        langs: [fa]                        # optional: sides the fact exists on (B editions differ)
        needs_linking: true                # optional: the doc exists but the quote is not in the extraction yet
        page: {fa: 29, en: 28}             # optional provenance (PDF page)
        pdf_text: {fa: "...", en: "..."}   # optional: what the PDF says when the doc is not extracted yet

For corpus A (unit-aligned) the gold locals are the union of the fa and en matches, so the
two gold lists are always swap_lang images of each other; one-sided matches are reported.

Matching key, applied to both the quote and the unit text: normalize() -> fold() -> thousands
commas between digits dropped -> "/" between digits read as a decimal point (Persian editions
write 93/9%) -> whitespace collapsed. Terms match on digit boundaries wherever they start or end
with a digit, on a word boundary where they start with a letter, and a term written with "%"
needs a percent marker (%, درصد, percent) on either side of the number (see term_rx).

Question-set metadata is authoring data. It is never given to an approach (the base class
builds a Query from a Question), except gold unit ids for B1.
"""
import argparse
import csv
import json
import re
import sys
from collections import Counter, OrderedDict, defaultdict
from pathlib import Path

import yaml

from ..config import load_config
from ..contracts import ALL_SETTINGS, ANSWER_TYPES, LANGS, QUESTION_TYPES, Question, load_questions, save_questions
from ..corpus import load_set
from ..corpus.model import split_id, swap_lang
from ..normalize import fold
from ..util import atomic_write_text, now_iso

SRC_NAME = "pilot.src.yaml"
OUT_NAME = "pilot.jsonl"
SHEET_NAME = "pilot_review.csv"

# composition the pilot was specified with (checked by `validate`, reported by `summary`)
TARGETS = {
    "A": {"total": 20, "min_reports": 4,
          "cells": {("lookup", "numeric"): 8, ("locate", None): 4, ("multi-section", None): 4,
                    ("cross-report", None): 2, ("unanswerable", None): 2}},
    "B": {"total": 8, "cells": {("cross-year", None): 2}},
    "C": {"total": 8, "cells": {}},
}

AUTHOR = "questions agent (Claude Opus 5), written by reading corpus Document unit text; no local model involved"


# --------------------------------------------------------------------------
# matching
# --------------------------------------------------------------------------

_RE_THOUS = re.compile(r"(?<=\d),(?=\d{3}(?!\d))")
_RE_DEC_SLASH = re.compile(r"(?<=\d)/(?=\d)")
_RE_WS = re.compile(r"\s+")
_RE_NUMERIC_TERM = re.compile(r"^[\s$€%+\-]*\d[\d.,/]*[\s%]*$")
_RE_NUM = re.compile(r"(?<![\d.])\d+(?:\.\d+)?")


def match_key(text):
    t = fold(text or "")
    t = _RE_THOUS.sub("", t)
    t = _RE_THOUS.sub("", t)      # "1,418,000": the second pass catches overlapping groups
    t = _RE_DEC_SLASH.sub(".", t)
    return _RE_WS.sub(" ", t).strip()


_TERM_CACHE = {}


_NUM_L = r"(?<![\d.])"          # a number term may not continue a longer number on the left ...
_NUM_R = r"(?!\d|\.\d)"         # ... or on the right ("5.19" must not match "15.19" or "5.191")
_PCT_WORDS = r"(?:%|درصد|percent|per cent)"


def term_rx(term):
    """Regex source for one quote term, applied to a match_key()ed text.

    * numeric terms ("71,319", "0.775", "16%") match on digit boundaries; a term written with "%"
      also needs a percent marker next to the number: "16 %", "16 درصد", "16 percent", or "%16"
      (RTL PDF extraction often puts the sign before the number);
    * any other term is literal, with a digit boundary at an end that is a digit ("700 kt" must
      not match "1,700 kt") and a word boundary at a start that is a letter ("رد" must not match
      inside "عملکرد"; suffixes stay allowed: "خانه" matches "خانه ها")."""
    t = str(term)
    k = match_key(t)
    if _RE_NUMERIC_TERM.match(t):
        core = re.escape(k.replace("%", "").replace("$", "").replace("€", "").replace("+", "").strip())
        num = _NUM_L + core + _NUM_R
        if "%" in t:
            return r"(?:%s\s?%s|%%\s?%s)" % (num, _PCT_WORDS, num)
        return num
    src = re.escape(k)
    if k[:1].isdigit():
        src = _NUM_L + src
    elif k[:1].isalpha():
        src = r"(?<!\w)" + src
    if k[-1:].isdigit():
        src = src + _NUM_R
    return src


def term_matcher(term):
    """Callable(key) -> bool for one quote term."""
    m = _TERM_CACHE.get(term)
    if m is not None:
        return m
    rx = re.compile(term_rx(term))
    m = lambda key, rx=rx: rx.search(key) is not None  # noqa: E731
    _TERM_CACHE[term] = m
    return m


def matches_all(key, terms):
    """terms: [t1, t2] (all must match) or [[t1, t2], [t3]] (alternatives: any list matching in full)."""
    if not terms:
        return False
    if isinstance(terms[0], (list, tuple)):
        return any(matches_all(key, alt) for alt in terms)
    return all(term_matcher(t)(key) for t in terms)


def terms_text(terms):
    if terms and isinstance(terms[0], (list, tuple)):
        return " | ".join(" + ".join(map(str, alt)) for alt in terms)
    return " + ".join(map(str, terms or []))


def numbers_in(text):
    return [float(x) for x in _RE_NUM.findall(match_key(text))]


def _close(a, b):
    return abs(a - b) <= max(1e-9, 1e-6 * abs(b))


# --------------------------------------------------------------------------
# corpora
# --------------------------------------------------------------------------

class Corpora:
    """Every loaded Document of A, B and C by id, with cached matching keys."""

    def __init__(self, cfg, rebuild=False):
        self.docs = {}
        self.skipped = {}
        # a gold unit longer than the shared evidence block can never be shown whole, and any chunk or
        # window taken from anywhere inside it would be scored as a gold hit (see link_item)
        self.max_chars = int(cfg.evidence["max_chars"])
        for name in ("A", "B", "C", "D"):
            cs = load_set(name, cfg, rebuild=rebuild)
            for d in cs:
                self.docs[d.id] = d
            self.skipped[name] = cs.skipped
        self._keys = {}

    def has(self, doc_id):
        return doc_id in self.docs

    def doc(self, doc_id):
        return self.docs.get(doc_id)

    def unit(self, uid):
        d = self.docs.get(split_id(uid)[0])
        return d.unit(uid) if d is not None and d.has_unit(uid) else None

    def keyed_units(self, doc_id):
        if doc_id not in self._keys:
            d = self.docs[doc_id]
            self._keys[doc_id] = [(u, match_key(u.text)) for u in d.units]
        return self._keys[doc_id]

    def search(self, doc_id, terms):
        return [u for u, k in self.keyed_units(doc_id) if matches_all(k, terms)]

    def section_path(self, uid):
        u = self.unit(uid)
        if u is None:
            return ""
        return " > ".join(t for t in self.docs[u.doc_id].section_path(u.section_id) if t)


def _paths(cfg, args=None):
    root = cfg.path("questions_root")
    src = Path(getattr(args, "src", None) or root / SRC_NAME)
    if getattr(args, "cmd", None) == "build":
        jsonl = Path(getattr(args, "out", None) or root / OUT_NAME)
    else:
        jsonl = Path(getattr(args, "path", None) or root / OUT_NAME)
    return root, src, jsonl


def required_langs(corpus):
    return ("fa",) if corpus == "C" else ("en",) if corpus == "D" else ("fa", "en")


def doc_scope_ids(spec):
    """YAML doc_scope entries are '<corpus>.<key>' or full doc ids; written as fa doc ids."""
    out = []
    for d in spec.get("doc_scope") or []:
        base = d.rsplit(".", 1)[0] if d.endswith((".fa", ".en")) else d
        did = base + ".fa"
        if did not in out:
            out.append(did)
    return out


# --------------------------------------------------------------------------
# build: YAML -> linked Question JSONL
# --------------------------------------------------------------------------

def _item_doc_base(item, spec):
    if item.get("doc"):
        d = item["doc"]
        return d.rsplit(".", 1)[0] if d.endswith((".fa", ".en")) else d
    scope = doc_scope_ids(spec)
    bases = sorted({s.rsplit(".", 1)[0] for s in scope})
    if len(bases) != 1:
        raise ValueError("%s: evidence item without doc and doc_scope has %d docs" % (spec.get("id"), len(bases)))
    return bases[0]


def _drop_oversize(locals_by_lang, docs_by_lang, max_chars):
    """Remove unit locals whose text (in any language given) is longer than max_chars, unless that would
    leave no gold for the fact. -> (kept locals per lang, dropped locals)"""
    def too_long(x):
        for l, d in docs_by_lang.items():
            uid = "%s::%s" % (d.id, x)
            if d.has_unit(uid) and len(d.unit(uid).text) > max_chars:
                return True
        return False
    every = []
    for locs in locals_by_lang.values():
        every += [x for x in locs if x not in every]
    dropped = [x for x in every if too_long(x)]
    kept = {l: [x for x in locs if x not in dropped] for l, locs in locals_by_lang.items()}
    if dropped and all(kept.values()):
        return kept, dropped
    return locals_by_lang, []


def link_item(item, spec, corp):
    """-> {lang: {"doc_id", "units", "status", "one_sided", "oversize"}} for the sides this item exists on.

    Gold units longer than evidence.max_chars (method-ledger tables of 13-19k chars in corpus A) are left
    out of gold whenever the fact has other gold units: no approach can put them in the evidence whole,
    and unit-level evidence recall would credit a chunk from anywhere inside them. They are listed under
    "oversize" (meta.evidence[i].oversize) for evaluation that checks evidence text instead of unit ids."""
    max_chars = getattr(corp, "max_chars", None)
    corpus = spec["corpus"]
    base = _item_doc_base(item, spec)
    langs = [l for l in (item.get("langs") or required_langs(corpus)) if l in required_langs(corpus)]
    only = set(item.get("only") or [])
    exclude = set(item.get("exclude") or [])
    res = OrderedDict()
    matched_locals = {}
    for l in langs:
        did = "%s.%s" % (base, l)
        terms = item.get(l) or []
        if not corp.has(did):
            res[l] = {"doc_id": did, "units": [], "status": "doc_missing", "one_sided": []}
            continue
        if not terms:
            raise ValueError("%s: evidence item %r has no %s quote terms" % (spec["id"], item.get("fact"), l))
        hits = [u for u in corp.search(did, terms) if u.kind != "heading"]
        locs = [split_id(u.id)[1] for u in hits]
        locs = [x for x in locs if (not only or x in only) and x not in exclude]
        matched_locals[l] = locs
        res[l] = {"doc_id": did, "units": [], "status": "linked", "one_sided": [], "oversize": []}
    if corpus == "A" and set(matched_locals) == {"fa", "en"}:
        union = [x for x in matched_locals["en"]]
        union += [x for x in matched_locals["fa"] if x not in union]
        for l in ("fa", "en"):
            other = "en" if l == "fa" else "fa"
            res[l]["one_sided"] = [x for x in matched_locals[l] if x not in matched_locals[other]]
        docs = {l: corp.doc(res[l]["doc_id"]) for l in ("fa", "en")}
        dropped = []
        if max_chars:
            kept, dropped = _drop_oversize({"all": union}, docs, max_chars)
            union = kept["all"]
        for l in ("fa", "en"):
            d = docs[l]
            ids = ["%s::%s" % (d.id, x) for x in union if d.has_unit("%s::%s" % (d.id, x))]
            res[l]["units"] = sorted(ids, key=lambda u: d.unit(u).order)
            res[l]["oversize"] = ["%s::%s" % (d.id, x) for x in dropped]
    else:
        for l, locs in matched_locals.items():
            d = corp.doc(res[l]["doc_id"])
            dropped = []
            if max_chars:
                kept, dropped = _drop_oversize({l: locs}, {l: d}, max_chars)
                locs = kept[l]
            res[l]["units"] = sorted(["%s::%s" % (d.id, x) for x in locs], key=lambda u: d.unit(u).order)
            res[l]["oversize"] = ["%s::%s" % (d.id, x) for x in dropped]
    for l, r in res.items():
        if r["status"] == "linked" and not r["units"]:
            r["status"] = "no_match"
    return res


def order_gold(groups, corp, lang=None, corpus=None):
    """Round-robin over groups; inside a group shortest text first (ties: document order). For A the
    ordering is computed on the English twin so fa and en gold lists stay in the same local order."""
    def key(u):
        un = corp.unit(u)
        if corpus == "A":
            tw = corp.unit(swap_lang(u, "en"))
            un = tw or un
        return (len(un.text) if un else 10 ** 9, un.order if un else 0)
    queues = [sorted(g, key=key) for g in groups if g]
    out = []
    i = 0
    while any(queues):
        q = queues[i % len(queues)]
        if q:
            u = q.pop(0)
            if u not in out:
                out.append(u)
        i += 1
    return out


class _PackCorpus:
    """Minimal CorpusSet stand-in for contracts.pack_evidence over documents of several sets."""

    def __init__(self, corp):
        self.corp = corp

    def doc(self, doc_id):
        d = self.corp.doc(doc_id)
        if d is None:
            raise KeyError(doc_id)
        return d

    def unit(self, uid):
        u = self.corp.unit(uid)
        if u is None:
            raise KeyError(uid)
        return u


def gold_pack_coverage(q, lang, corp, cfg):
    """Pack gold_units (in order, one RetrievedItem per unit) under the shared evidence budget.
    -> (covered groups, total groups, pack chars, n units included)"""
    from ..contracts import RetrievedItem, pack_evidence
    items = []
    for u in q.gold(lang):
        un = corp.unit(u)
        if un is None:
            continue
        items.append(RetrievedItem(text=un.text, doc_id=un.doc_id, section_id=un.section_id, unit_ids=[u],
                                   kind="unit", span=list(un.span)))
    pack = pack_evidence(items, _PackCorpus(corp), cfg=cfg)
    got = set(pack.unit_ids)
    groups = ((q.meta or {}).get("gold_groups") or {}).get(lang) or []
    covered = sum(1 for g in groups if got & set(g))
    return covered, len([g for g in groups if g]), pack.chars, pack.n_included


META_PASS = ("subtype", "memorisable", "answer_values", "aliases", "notes", "absent", "absent_reviewed")

_OVERLAP_STOP = set("""
the a an of in on at to for and or by from with what which how many much is are was were did does do its it
this that according report reports s
در از به و که را با این آن برای بر تا یا چه چند کدام است بود بوده شده می اساس گزارش گزارش های
""".split())


def lexical_overlap(question, gold_ids, corp):
    """How closely a same-language question copies its gold text (questions were written by reading the
    units, which favours lexical retrieval). Over folded word tokens of the question, stopwords and the
    report-naming boilerplate removed:
      content: share of content words that occur in some gold unit;
      bigram:  share of the question's word bigrams that occur verbatim in some gold unit;
      longest: the longest run of question words found verbatim in one gold unit.
    Written to meta.lexical_overlap[lang] so evaluation can stratify results by it."""
    from ..normalize import word_tokens
    q = word_tokens(question or "")
    if not q:
        return None
    units = [word_tokens(corp.unit(u).text) for u in gold_ids if corp.unit(u) is not None]
    if not units:
        return None
    vocab = set(t for ut in units for t in ut)
    content = [t for t in q if t not in _OVERLAP_STOP]
    bigrams = [(a, b) for a, b in zip(q, q[1:]) if not (a in _OVERLAP_STOP and b in _OVERLAP_STOP)]
    ubig = set(p for ut in units for p in zip(ut, ut[1:]))
    longest = 0
    for ut in units:
        joined = " " + " ".join(ut) + " "
        n = len(q)
        for i in range(n):
            if n - i <= longest:
                break
            j = i + longest + 1
            while j <= n and (" " + " ".join(q[i:j]) + " ") in joined:
                longest = j - i
                j += 1
    return {"content": round(sum(1 for t in content if t in vocab) / float(len(content)), 3) if content else 0.0,
            "bigram": round(sum(1 for p in bigrams if p in ubig) / float(len(bigrams)), 3) if bigrams else 0.0,
            "longest": longest}


def build_question(spec, corp, source=None):
    """-> (question dict, problems[list of (level, msg)])"""
    if spec.get("type") == "timeline" or spec.get("answer_type") == "series":
        from .timeline_questions import build_timeline_question
        return build_timeline_question(spec, corp, source=source or "pilot_timeline.src.yaml")
    probs = []
    qid = spec["id"]
    corpus = spec["corpus"]
    langs = required_langs(corpus)
    evidence_out = []
    gold = {"fa": [], "en": []}
    groups = {"fa": [], "en": []}
    needs_linking = False
    for item in spec.get("evidence") or []:
        linked = link_item(item, spec, corp)
        e = OrderedDict(fact=item.get("fact"), doc=_item_doc_base(item, spec))
        e["quote"] = {l: item.get(l) for l in langs if item.get(l)}
        e["units"] = {l: r["units"] for l, r in linked.items()}
        e["status"] = {l: r["status"] for l, r in linked.items()}
        for k in ("only", "exclude", "langs", "page", "pdf_text"):
            if item.get(k):
                e[k] = item[k]
        oversize = {l: r["oversize"] for l, r in linked.items() if r.get("oversize")}
        if oversize:
            e["oversize"] = oversize
            probs.append(("info", "%s: left out of gold as longer than the evidence budget: %s" % (qid, oversize)))
        for l, r in linked.items():
            if r["status"] == "doc_missing":
                needs_linking = True
                probs.append(("warn", "%s: %s doc %s not extracted yet (fact: %s)" % (qid, l, r["doc_id"], item.get("fact"))))
            elif r["status"] == "no_match":
                if item.get("needs_linking"):
                    needs_linking = True
                    probs.append(("warn", "%s: %s quote %s not found in %s (flagged needs_linking)" % (qid, l, item.get(l), r["doc_id"])))
                else:
                    probs.append(("error", "%s: %s quote %s matches no unit in %s" % (qid, l, item.get(l), r["doc_id"])))
            if r.get("one_sided"):
                probs.append(("info", "%s: %s-only matches (kept via the other language): %s" % (qid, l, r["one_sided"])))
            for u in r["units"]:
                if u not in gold[l]:
                    gold[l].append(u)
            groups[l].append(list(r["units"]))
        evidence_out.append(e)
    # order gold for a budget-limited reader (B1 packs gold_unit_ids in this order): round-robin over the
    # evidence groups, shortest unit first inside a group, so the first units cover every required fact
    for l in gold:
        gold[l] = order_gold(groups[l], corp, lang=l, corpus=corpus)
    meta = OrderedDict()
    meta["reports"] = sorted({e["doc"] for e in evidence_out} | {d.rsplit(".", 1)[0] for d in doc_scope_ids(spec)})
    for k in META_PASS:
        if k in spec:
            meta[k] = spec[k]
    meta["memorisable"] = bool(spec.get("memorisable", False))
    meta["needs_unit_linking"] = needs_linking
    meta["evidence"] = evidence_out
    meta["gold_groups"] = {l: groups[l] for l in langs}
    meta["lexical_overlap"] = {l: lexical_overlap((spec.get("question") or {}).get(l, ""), gold[l], corp)
                               for l in langs if gold[l]}
    meta["author"] = AUTHOR
    meta["source"] = SRC_NAME
    q = OrderedDict(
        id=qid, corpus=corpus,
        question_fa=(spec.get("question") or {}).get("fa", ""),
        question_en=(spec.get("question") or {}).get("en", ""),
        answer_fa=(spec.get("answer") or {}).get("fa", ""),
        answer_en=(spec.get("answer") or {}).get("en", ""),
        answer_type=spec["answer_type"],
        gold_units={"fa": gold["fa"], "en": gold["en"] if corpus != "C" else []},
        doc_scope=doc_scope_ids(spec),
        type=spec["type"],
        answer_value=spec.get("answer_value"),
        answer_unit=spec.get("answer_unit"),
        meta=meta)
    return q, probs


def cmd_build(args, cfg):
    _root, src, out = _paths(cfg, args)
    with open(src, encoding="utf-8") as fh:
        specs = yaml.safe_load(fh)
    specs = specs.get("questions") if isinstance(specs, dict) else specs
    corp = Corpora(cfg, rebuild=args.rebuild)
    qs, all_probs = [], []
    for spec in specs:
        q, probs = build_question(spec, corp, source=Path(src).name)
        qs.append(q)
        all_probs += probs
    for level, msg in all_probs:
        if level != "info" or args.verbose:
            print("[%s] %s" % (level, msg))
    n_err = sum(1 for lv, _ in all_probs if lv == "error")
    print("built %d questions; %d errors, %d warnings" % (len(qs), n_err, sum(1 for lv, _ in all_probs if lv == "warn")))
    if n_err and not args.force:
        print("not written (fix errors or pass --force)")
        return 1
    if args.check_only:
        return 0
    save_questions(out, [Question.from_dict(q) for q in qs])
    print("wrote %s" % out)
    return 0


# --------------------------------------------------------------------------
# validate
# --------------------------------------------------------------------------

class Report:
    def __init__(self):
        self.items = []

    def add(self, level, qid, msg):
        self.items.append((level, qid, msg))

    def by_q(self):
        d = defaultdict(list)
        for lv, qid, msg in self.items:
            d[qid].append("%s: %s" % (lv, msg))
        return d

    def count(self, level):
        return sum(1 for lv, _, _ in self.items if lv == level)


def validate_questions(qs, raw, corp, rep, cfg=None):
    cfg = cfg or load_config()
    ids = Counter(q.id for q in qs)
    for qid, n in ids.items():
        if n > 1:
            rep.add("error", qid, "duplicate id (%d times)" % n)
    for q, r in zip(qs, raw):
        meta = q.meta or {}
        linking = bool(meta.get("needs_unit_linking"))
        langs = required_langs(q.corpus)
        timeline = q.type == "timeline" or q.answer_type == "series"
        unknown = set(r) - set(Question.__dataclass_fields__)
        if unknown:
            rep.add("error", q.id, "top-level keys outside the Question schema are dropped by from_dict: %s" % sorted(unknown))
        for p in q.validate():
            if timeline and p in ("answer_type", "type"):
                # contracts.py does not list timeline/series yet (patch in ragbench/INTEGRATION_NOTES.md)
                rep.add("info", q.id, "schema: %s not in contracts.py yet (timeline)" % p)
                continue
            rep.add("warn" if (p == "no gold units" and linking) else "error", q.id, "schema: %s" % p)
        if timeline:
            from .timeline_questions import validate_timeline
            validate_timeline(q, corp, rep)
        for l in LANGS:
            if not (q.text(l) or "").strip():
                rep.add("error", q.id, "empty question_%s" % l)
            if not (q.answer(l) or "").strip():
                rep.add("error", q.id, "empty answer_%s" % l)
        if (q.answer_type == "unanswerable") != (q.type == "unanswerable"):
            rep.add("error", q.id, "answer_type %s with type %s" % (q.answer_type, q.type))
        if q.doc_scope in (None, "corpus") or not isinstance(q.doc_scope, list) or not q.doc_scope:
            rep.add("warn", q.id, "doc_scope is not a document list")
            scope_docs = []
        else:
            scope_docs = list(q.doc_scope)
        for d in scope_docs:
            if not d.startswith(q.corpus + "."):
                rep.add("error", q.id, "doc_scope %s is not in corpus %s" % (d, q.corpus))
            for l in langs:
                did = swap_lang(d, l)
                if not corp.has(did):
                    rep.add("warn" if linking else "error", q.id, "scope doc %s not loaded" % did)
        # leakage / hygiene
        for l in LANGS:
            t = q.text(l) or ""
            if "::" in t or re.search(r"\b[ABC]\.[a-z0-9_-]+\.(fa|en)\b", t):
                rep.add("error", q.id, "question_%s contains an id" % l)
        # gold ids exist and sit inside the scope
        for l in LANGS:
            gl = q.gold(l)
            if q.corpus == "C" and l == "en" and gl:
                rep.add("error", q.id, "corpus C must not have English gold units")
            for u in gl:
                did = split_id(u)[0]
                if not corp.has(did):
                    rep.add("warn" if linking else "error", q.id, "gold unit doc %s not loaded" % did)
                    continue
                if corp.unit(u) is None:
                    rep.add("error", q.id, "gold unit %s does not exist" % u)
                elif len(corp.unit(u).text) > int(cfg.evidence["max_chars"]):
                    rep.add("warn", q.id, "gold unit %s (%d chars) is longer than the evidence budget; any chunk of it "
                            "counts as a gold hit" % (u, len(corp.unit(u).text)))
                if scope_docs and swap_lang(did, "fa") not in scope_docs:
                    rep.add("error", q.id, "gold unit %s outside doc_scope" % u)
        if q.corpus == "A":
            fa = {swap_lang(u, "en") for u in q.gold("fa")}
            en = set(q.gold("en"))
            if fa != en:
                rep.add("error", q.id, "A gold not aligned: fa-only %s en-only %s" % (sorted(fa - en)[:3], sorted(en - fa)[:3]))
        if q.answer_type == "unanswerable":
            if q.gold("fa") or q.gold("en"):
                rep.add("error", q.id, "unanswerable question with gold units")
        else:
            for l in langs:
                if not q.gold(l) and not linking:
                    rep.add("error", q.id, "no %s gold units" % l)
        # evidence consistency and quote support
        ev = meta.get("evidence") or []
        if q.answer_type != "unanswerable" and not ev:
            rep.add("error", q.id, "meta.evidence missing")
        union = {l: set() for l in LANGS}
        for i, e in enumerate(ev):
            for l, units in (e.get("units") or {}).items():
                terms = (e.get("quote") or {}).get(l) or []
                st = (e.get("status") or {}).get(l)
                if st in ("doc_missing", "no_match") and not linking:
                    rep.add("error", q.id, "evidence %d (%s) %s status %s but needs_unit_linking is false" % (i, e.get("fact"), l, st))
                for u in units:
                    union[l].add(u)
                    un = corp.unit(u)
                    if un is None:
                        continue
                    if not matches_all(match_key(un.text), terms):
                        # A: one-sided matches are allowed only when the other language matched the same local
                        other = "en" if l == "fa" else "fa"
                        ok = False
                        if q.corpus == "A":
                            ou = corp.unit(swap_lang(u, other))
                            ok = ou is not None and matches_all(match_key(ou.text), (e.get("quote") or {}).get(other) or [])
                        if not ok:
                            rep.add("error", q.id, "gold unit %s does not contain its quote %s" % (u, terms))
                        else:
                            rep.add("info", q.id, "gold unit %s supported through its %s twin" % (u, other))
        for l in LANGS:
            if union[l] != set(q.gold(l)):
                rep.add("error", q.id, "%s gold_units differ from the union of meta.evidence units" % l)
        # numeric answers: value present in the answer text and in a gold unit
        if q.answer_type == "numeric":
            for l in LANGS:
                av = ((meta.get("answer_values") or {}).get(l) or {}).get("value", q.answer_value)
                if av is None:
                    continue
                if not any(_close(x, float(av)) for x in numbers_in(q.answer(l))):
                    rep.add("warn", q.id, "answer_%s does not contain answer value %s" % (l, av))
                if l in langs and q.gold(l):
                    if not any(any(_close(x, float(av)) for x in numbers_in(corp.unit(u).text))
                               for u in q.gold(l) if corp.unit(u) is not None):
                        rep.add("warn", q.id, "no %s gold unit contains the number %s" % (l, av))
                qt = q.text(l) or ""
                if any(_close(x, float(av)) for x in numbers_in(qt)):
                    rep.add("warn", q.id, "question_%s contains the answer value %s (leak?)" % (l, av))
        # unanswerable: the absent patterns must not match inside the scope
        if q.answer_type == "unanswerable":
            absent = meta.get("absent") or {}
            if not absent:
                rep.add("warn", q.id, "unanswerable question without meta.absent checks")
            reviewed = set(meta.get("absent_reviewed") or [])
            for l, pats in absent.items():
                for terms in pats:
                    terms = terms if isinstance(terms, list) else [terms]
                    for d in scope_docs:
                        did = swap_lang(d, l)
                        if not corp.has(did):
                            continue
                        hits = [u for u in corp.search(did, terms) if split_id(u.id)[1] not in reviewed]
                        if hits:
                            rep.add("warn", q.id, "absent-check %s %s hits %d units in %s, e.g. %s" % (l, terms, len(hits), did, hits[0].id))
        # the gold evidence must be answerable within the shared evidence budget (what B1 sees)
        if q.answer_type != "unanswerable":
            for l in langs:
                if not q.gold(l) or any(corp.unit(u) is None for u in q.gold(l)):
                    continue
                cov, tot, chars, n_in = gold_pack_coverage(q, l, corp, cfg)
                if cov < tot:
                    rep.add("warn", q.id, "%s gold packed under the evidence budget covers %d of %d facts (%d chars, %d units)"
                            % (l, cov, tot, chars, n_in))
        # settings
        sup = [s.name for s in ALL_SETTINGS if q.supports(s)]
        want = {"A": 4, "B": 4, "C": 2}[q.corpus]
        if len(sup) < want and not linking:
            rep.add("warn", q.id, "supports only %s" % sup)
        if not isinstance(meta.get("memorisable"), bool):
            rep.add("warn", q.id, "meta.memorisable not set")


def composition(qs):
    rows = []
    for c, tgt in TARGETS.items():
        cq = [q for q in qs if q.corpus == c]
        rows.append((c, "total", len(cq), tgt["total"]))
        for (t, at), n in tgt["cells"].items():
            if t == "cross-report":
                have = sum(1 for q in cq if (q.meta or {}).get("subtype") == "cross-report")
            else:
                have = sum(1 for q in cq if q.type == t and (at is None or q.answer_type == at)
                           and (q.meta or {}).get("subtype") != "cross-report")
            rows.append((c, "%s%s" % (t, "/" + at if at else ""), have, n))
        if tgt.get("min_reports"):
            reps = {r for q in cq for r in (q.meta or {}).get("reports", [])}
            rows.append((c, "reports", len(reps), tgt["min_reports"]))
    return rows


def composition_rows(qs):
    """[(corpus|TL, cell, have, want, ok)]: the timeline targets for a file of timeline questions, else the pilot's."""
    if qs and all(q.type == "timeline" or q.answer_type == "series" for q in qs):
        from .timeline_questions import timeline_composition
        return timeline_composition(qs)
    return [(c, cell, have, want, (have >= want if cell == "reports" else have == want))
            for c, cell, have, want in composition(qs)]


def cmd_validate(args, cfg):
    _root, _src, path = _paths(cfg, args)
    raw = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    qs = load_questions(path)
    corp = Corpora(cfg)
    rep = Report()
    validate_questions(qs, raw, corp, rep)
    for c, cell, have, want, ok in composition_rows(qs):
        if not ok:
            rep.add("warn", "(set)", "composition %s %s: %d, specified %d" % (c, cell, have, want))
    shown = ("error", "warn") + (("info",) if args.verbose else ())
    for lv, qid, msg in rep.items:
        if lv in shown:
            print("[%s] %s: %s" % (lv, qid, msg))
    if any(q.type == "timeline" for q in qs):
        from .timeline_questions import contracts_support_timeline
        if not contracts_support_timeline():
            print("[note] contracts.py does not list answer_type 'series' / type 'timeline' yet, so ragbench.run skips "
                  "these questions as invalid until the patch in ragbench/INTEGRATION_NOTES.md is applied")
    print("%d questions: %d errors, %d warnings" % (len(qs), rep.count("error"), rep.count("warn")))
    if args.strict and rep.count("warn"):
        return 1
    return 1 if rep.count("error") else 0


# --------------------------------------------------------------------------
# summary
# --------------------------------------------------------------------------

def cmd_summary(args, cfg):
    _root, _src, path = _paths(cfg, args)
    qs = load_questions(path)
    print("%d questions in %s\n" % (len(qs), path))
    tab = Counter((q.corpus, q.type, q.answer_type, (q.meta or {}).get("subtype") or "") for q in qs)
    print("%-4s %-14s %-13s %-13s %s" % ("corp", "type", "answer_type", "subtype", "n"))
    for (c, t, at, st), n in sorted(tab.items()):
        print("%-4s %-14s %-13s %-13s %d" % (c, t, at, st, n))
    print()
    print("%-4s %5s %6s %11s %12s %8s %s" % ("corp", "n", "memo", "needs_link", "gold/q(fa)", "reports", "settings supported (fa-fa en-en fa-en en-fa)"))
    for c in ("A", "B", "C"):
        cq = [q for q in qs if q.corpus == c]
        if not cq:
            continue
        memo = sum(1 for q in cq if (q.meta or {}).get("memorisable"))
        link = sum(1 for q in cq if (q.meta or {}).get("needs_unit_linking"))
        gold = sum(len(q.gold("fa")) for q in cq) / float(len(cq))
        reps = len({r for q in cq for r in (q.meta or {}).get("reports", [])})
        sup = " ".join("%d" % sum(1 for q in cq if q.supports(s)) for s in ALL_SETTINGS)
        print("%-4s %5d %6d %11d %12.1f %8d %s" % (c, len(cq), memo, link, gold, reps, sup))
    print()
    print("question-to-gold lexical overlap, same language (mean content-word share / bigram share / longest verbatim run):")
    for c in ("A", "B", "C"):
        for l in required_langs(c):
            ov = [(q.meta or {}).get("lexical_overlap", {}).get(l) for q in qs if q.corpus == c]
            ov = [o for o in ov if o]
            if ov:
                print("  %s %s  n=%-3d %.2f / %.2f / %.1f words" % (c, l, len(ov), sum(o["content"] for o in ov) / len(ov),
                                                                sum(o["bigram"] for o in ov) / len(ov),
                                                                sum(o["longest"] for o in ov) / float(len(ov))))
    print()
    print("composition against the specification:")
    for c, cell, have, want, ok in composition_rows(qs):
        print("  %s %-24s %3d / %-3d %s" % (c, cell, have, want, "ok" if ok else "DIFFERS"))
    if any(q.type == "timeline" for q in qs):
        from .timeline_questions import series_table
        print("\ntimeline series:")
        for q in qs:
            if q.type == "timeline":
                print("  %s  %s" % (q.id, "\n    ".join(series_table(q))))
    reports = Counter(r for q in qs for r in (q.meta or {}).get("reports", []))
    print("\ndocuments referenced:")
    for r, n in sorted(reports.items()):
        print("  %-45s %d" % (r, n))
    return 0


# --------------------------------------------------------------------------
# review sheet
# --------------------------------------------------------------------------

def _excerpt(text, terms, width=700):
    t = (text or "").replace("\n", " / ")
    if len(t) <= width:
        return t
    key = t.lower()
    pos = 0
    for term in terms or []:
        i = key.find(str(term).lower())
        if i >= 0:
            pos = i
            break
    start = max(0, pos - width // 3)
    return ("…" if start else "") + t[start:start + width] + "…"


def cmd_sheet(args, cfg):
    root, _src, path = _paths(cfg, args)
    out = Path(args.out or root / SHEET_NAME)
    qs = load_questions(path)
    raw = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    corp = Corpora(cfg)
    rep = Report()
    validate_questions(qs, raw, corp, rep)
    probs = rep.by_q()
    cols = ["id", "corpus", "type", "subtype", "answer_type", "lang", "question", "answer", "answer_value", "answer_unit",
            "doc_scope", "memorisable", "needs_unit_linking", "fact", "quote", "unit_id", "unit_kind", "section_path",
            "unit_text", "pdf_page", "pdf_text", "checks", "reviewer_ok", "reviewer_note"]
    rows = []
    for q in qs:
        meta = q.meta or {}
        for l in required_langs(q.corpus) + (("en",) if q.corpus == "C" else ()):
            base = OrderedDict(id=q.id, corpus=q.corpus, type=q.type, subtype=meta.get("subtype", ""),
                               answer_type=q.answer_type, lang=l, question=q.text(l), answer=q.answer(l),
                               answer_value="" if q.answer_value is None else q.answer_value,
                               answer_unit=q.answer_unit or "", doc_scope=" ".join(q.doc_scope) if isinstance(q.doc_scope, list) else q.doc_scope,
                               memorisable=meta.get("memorisable"), needs_unit_linking=meta.get("needs_unit_linking"),
                               checks=" | ".join(x for x in probs.get(q.id, []) if not x.startswith("info")))
            wrote = False
            if not (q.corpus == "C" and l == "en"):
                for e in meta.get("evidence") or []:
                    units = (e.get("units") or {}).get(l) or []
                    terms = (e.get("quote") or {}).get(l) or []
                    common = dict(fact=e.get("fact"), quote=terms_text(terms),
                                  pdf_page=(e.get("page") or {}).get(l, ""), pdf_text=(e.get("pdf_text") or {}).get(l, ""))
                    if not units:
                        rows.append(dict(base, unit_id="(%s)" % ((e.get("status") or {}).get(l) or "not on this side"), **common))
                        wrote = True
                    for u in units:
                        un = corp.unit(u)
                        rows.append(dict(base, unit_id=u, unit_kind=un.kind if un else "", section_path=corp.section_path(u),
                                         unit_text=_excerpt(un.text if un else "", terms), **common))
                        wrote = True
            if not wrote:
                rows.append(dict(base, unit_id="(none: question language row)" if q.corpus == "C" and l == "en" else "(no gold)"))
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print("wrote %d rows to %s" % (len(rows), out))
    return 0


# --------------------------------------------------------------------------
# authoring helpers
# --------------------------------------------------------------------------

def cmd_find(args, cfg):
    corp = Corpora(cfg)
    if args.doc:
        doc_ids = [d for d in corp.docs if d.startswith(args.doc)]
    else:
        base = args.set.split("_")[0]
        lang = args.set.split("_")[1] if "_" in args.set else None
        doc_ids = [d for d, doc in corp.docs.items() if doc.corpus == base and (lang is None or doc.lang == lang)]
    n = 0
    for did in sorted(doc_ids):
        for u in corp.search(did, args.terms):
            if args.kind and u.kind != args.kind:
                continue
            n += 1
            if n > args.limit:
                break
            print("%s [%s] %s" % (u.id, u.kind, corp.section_path(u.id)))
            print("    " + _excerpt(u.text, args.terms, args.width))
            if args.both and corp.doc(did).alignment == "unit":
                tw = corp.unit(swap_lang(u.id, "fa" if u.lang == "en" or did.endswith(".en") else "en"))
                if tw is not None:
                    print("    <> " + _excerpt(tw.text, [], args.width))
        if n > args.limit:
            print("... more than %d hits" % args.limit)
            break
    print("%d hits" % min(n, args.limit))
    return 0


def cmd_show(args, cfg):
    _root, _src, path = _paths(cfg, args)
    qs = {q.id: q for q in load_questions(path)}
    corp = Corpora(cfg)
    for qid in args.ids:
        q = qs[qid]
        print("=" * 100)
        print("%s  corpus %s  type %s  answer_type %s  value %s %s" % (q.id, q.corpus, q.type, q.answer_type, q.answer_value, q.answer_unit or ""))
        print("scope: %s" % q.doc_scope)
        for l in LANGS:
            print("Q[%s] %s\nA[%s] %s" % (l, q.text(l), l, q.answer(l)))
        if q.type == "timeline":
            from .timeline_questions import series_table
            print("\n".join(series_table(q)))
        for l in required_langs(q.corpus):
            print("-- gold %s (%d)" % (l, len(q.gold(l))))
            for u in q.gold(l):
                un = corp.unit(u)
                print("   %s [%s] %s\n      %s" % (u, un.kind if un else "?", corp.section_path(u), _excerpt(un.text if un else "(missing)", [], 400)))
        print("meta: " + json.dumps({k: v for k, v in (q.meta or {}).items() if k not in ("evidence", "gold_groups")}, ensure_ascii=False))
    return 0


def main(argv=None):
    cfg = load_config()
    ap = argparse.ArgumentParser(prog="python -m ragbench.tools.questions", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="link evidence quotes to unit ids and write pilot.jsonl")
    b.add_argument("--src")
    b.add_argument("--out")
    b.add_argument("--check-only", action="store_true")
    b.add_argument("--force", action="store_true", help="write even with linking errors")
    b.add_argument("--rebuild", action="store_true", help="rebuild the corpus document cache first")
    b.add_argument("-v", "--verbose", action="store_true")
    v = sub.add_parser("validate", help="schema, gold existence, quote support, alignment, composition")
    v.add_argument("--path")
    v.add_argument("--strict", action="store_true", help="exit 1 on warnings too")
    v.add_argument("-v", "--verbose", action="store_true")
    s = sub.add_parser("summary")
    s.add_argument("--path")
    sh = sub.add_parser("sheet", help="export the review sheet CSV")
    sh.add_argument("--path")
    sh.add_argument("--out")
    f = sub.add_parser("find", help="search unit texts with quote terms (all must match)")
    f.add_argument("terms", nargs="+")
    f.add_argument("--set", default="A_en")
    f.add_argument("--doc", help="doc id prefix, e.g. A.rpt-a.en or C.snapp")
    f.add_argument("--kind")
    f.add_argument("--limit", type=int, default=30)
    f.add_argument("--width", type=int, default=300)
    f.add_argument("--both", action="store_true", help="corpus A: also print the other-language twin")
    w = sub.add_parser("show")
    w.add_argument("ids", nargs="+")
    w.add_argument("--path")
    args = ap.parse_args(argv)
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    return {"build": cmd_build, "validate": cmd_validate, "summary": cmd_summary, "sheet": cmd_sheet,
            "find": cmd_find, "show": cmd_show}[args.cmd](args, cfg)


if __name__ == "__main__":
    sys.exit(main())
