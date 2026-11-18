"""Contracts every approach implements, the question schema, the shared answer
prompt and the shared evidence budget.

Fairness is enforced here, not by convention:
  * an approach never receives a Question: Approach.answer() builds a Query (question
    text in the setting's question language, doc scope, ids) and passes only that to
    _answer(); gold unit ids are added to the Query only when the class sets
    needs_gold = True (B1).
  * LLM calls, tokens and wall time in RunRecord / build stats come from the ledger
    scope around _answer() / _build_index(), not from the approach.
  * the final answer must be produced with build_answer_messages() over an EvidencePack
    from pack_evidence() (Approach.generate_answer does both); the evidence block is cut
    to evidence.max_chars identically for everyone.
  * the answer prompt and its token budget depend only on the question's answer format
    (Query.answer_format, set by make_query from the question: answer_format(question)): "timeline"
    questions get ANSWER_TIMELINE_SYSTEM and answer.timeline_num_predict, every other old question the
    unchanged answer-v2 prompt and generation.num_predict_answer (answer_num_predict()).
  * multi-document questions (2026-09-24, P0 list) declare their format explicitly - Question.answer_format,
    or Question.task_family mapped by TASK_FAMILY_FORMATS - as one of MULTIDOC_FORMATS (single, series,
    comparison, list), each with its own prompt and output allowance, identical for every method and every
    option arm (never switched on or off by retrieval success or a method flag; see variants.py).
  * datasets-v2 questions (2026-09-25) use the domain-neutral V2_FORMATS (v2_single, v2_list, v2_comparison,
    v2_series, v2_explanation; v2_runner_format maps the schema's answer_format), same rules as above.
"""
import json
import time
import traceback
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from pathlib import Path

from .config import load_config
from .corpus.model import doc_lang, swap_lang
from .corpus.registry import CORPUS_LETTERS, load_set, set_name_for
from .llm import OllamaClient
from .util import append_jsonl, atomic_write_json, dir_size_bytes, now_iso, read_json, read_jsonl, safe_name, sha1_json

LANGS = ("fa", "en")
LANG_NAMES = {"fa": "Persian (Farsi)", "en": "English"}
ANSWER_TYPES = ("numeric", "short", "span", "unanswerable", "series", "multi")   # multi: multidoc content units
QUESTION_TYPES = ("lookup", "locate", "multi-section", "cross-year", "unanswerable", "timeline", "multidoc")
MULTIDOC_SCHEMA = "multidoc-q-v1"   # the versioned multi-document question schema (tools/verify_multidoc.py)


# --------------------------------------------------------------------------
# settings
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Setting:
    q_lang: str
    d_lang: str

    @property
    def name(self):
        return "%s-%s" % (self.q_lang, self.d_lang)

    @property
    def cross_lingual(self):
        return self.q_lang != self.d_lang

    @classmethod
    def parse(cls, s):
        if isinstance(s, Setting):
            return s
        if isinstance(s, (tuple, list)):
            q, d = s
        else:
            t = str(s).replace("→", "-").replace("2", "-").replace("_", "-").replace(">", "-")
            q, d = [x for x in t.split("-") if x]
        if q not in LANGS or d not in LANGS:
            raise ValueError("bad setting %r" % (s,))
        return cls(q, d)

    def __str__(self):
        return self.name


ALL_SETTINGS = [Setting("fa", "fa"), Setting("en", "en"), Setting("fa", "en"), Setting("en", "fa")]


# --------------------------------------------------------------------------
# questions
# --------------------------------------------------------------------------

@dataclass
class GoldRef:
    doc_id: str
    unit_ids: list


@dataclass
class Question:
    """One line of data/questions/*.jsonl.

    gold_units: {"fa": [unit ids in fa docs], "en": [unit ids in en docs]}. For corpus A
    the two lists are swap_lang images of each other; for B they are authored per
    language (document-aligned only); for C only "fa". Empty for unanswerable.
    doc_scope: list of doc ids (either language; mapped with swap_lang) or "corpus".
    answer_value / answer_unit: the numeric gold for answer_type == "numeric".
    """
    id: str
    corpus: str                    # "A" | "B" | "C" | "D" | "K" (BOOK, datasets v2) | "L" (LAW, datasets v2)
    question_fa: str
    question_en: str
    answer_fa: str
    answer_en: str
    answer_type: str               # numeric | short | span | unanswerable | series
    gold_units: dict
    doc_scope: object              # list[str] | "corpus"
    type: str                      # lookup | locate | multi-section | cross-year | unanswerable | timeline
    answer_value: float = None
    answer_unit: str = None
    meta: dict = field(default_factory=dict)
    # multi-document experiment (2026-09-24); None for every older question, and then left out of to_dict()
    schema: str = None             # "multidoc-q-v1" marks a multi-document question file (MULTIDOC_SCHEMA)
    task_family: str = None        # single_fact | series | comparison | aggregation | list | unanswerable (TASK_FAMILY_FORMATS)
    answer_format: str = None      # explicit answer format (MULTIDOC_FORMATS); wins over task_family
    pool_id: object = None         # str or [str]: the frozen pool manifest(s) this question runs in (ragbench/pools.py)

    def text(self, lang):
        return self.question_fa if lang == "fa" else self.question_en

    def answer(self, lang):
        return self.answer_fa if lang == "fa" else self.answer_en

    def gold(self, lang):
        return list((self.gold_units or {}).get(lang) or [])

    def gold_refs(self, lang):
        by = {}
        for u in self.gold(lang):
            by.setdefault(u.split("::", 1)[0], []).append(u)
        return [GoldRef(d, us) for d, us in by.items()]

    def scope(self, lang):
        if self.doc_scope in (None, "corpus"):
            return "corpus"
        return [swap_lang(d, lang) if doc_lang(d) != lang else d for d in self.doc_scope]

    def supports(self, setting):
        s = Setting.parse(setting)
        if not (self.text(s.q_lang) or "").strip():
            return False
        if (self.corpus == "C" and s.d_lang != "fa") or (self.corpus == "D" and s.d_lang != "en"):
            return False
        if self.answer_type != "unanswerable" and not self.gold(s.d_lang):
            return False
        return True

    def validate(self):
        p = []
        if self.corpus not in CORPUS_LETTERS:
            p.append("corpus")
        if self.answer_type not in ANSWER_TYPES:
            p.append("answer_type")
        if self.type not in QUESTION_TYPES:
            p.append("type")
        if self.answer_type == "numeric" and self.answer_value is None:
            p.append("numeric answer without answer_value")
        if self.answer_type != "unanswerable" and not any(self.gold(l) for l in LANGS):
            p.append("no gold units")
        if self.answer_format is not None and self.answer_format not in ANSWER_SYSTEMS:
            p.append("answer_format")
        if self.task_family is not None and self.task_family not in TASK_FAMILY_FORMATS:
            p.append("task_family")
        return p

    def is_multidoc(self):
        """A question of the multi-document experiment: the schema marker or meta.experiment == "multidoc"."""
        return str(self.schema or "").startswith("multidoc") or (self.meta or {}).get("experiment") == "multidoc"

    def pool_ids(self):
        p = self.pool_id
        return [] if p in (None, "") else [p] if isinstance(p, str) else list(p)

    def to_dict(self):
        d = asdict(self)
        for k in NEW_QUESTION_FIELDS:                 # older question files keep their exact shape
            if d.get(k) is None:
                d.pop(k, None)
        return d

    @classmethod
    def from_dict(cls, d):
        known = {k: d[k] for k in cls.__dataclass_fields__ if k in d}
        return cls(**known)


def load_questions(path):
    return [Question.from_dict(d) for d in read_jsonl(path)]


def save_questions(path, questions):
    from .util import atomic_write_text
    atomic_write_text(path, "".join(json.dumps(q.to_dict(), ensure_ascii=False) + "\n" for q in questions))


@dataclass
class Query:
    """What an approach sees of a question."""
    question_id: str
    text: str                      # question in q_lang
    q_lang: str
    d_lang: str
    setting: str                   # "fa-en"
    corpus: str
    corpus_set: str                # e.g. "A_en"
    doc_scope: object              # list of doc ids in d_lang, or "corpus"
    gold_unit_ids: list = None     # only for approaches with needs_gold = True (B1)
    answer_format: str = "default"  # "default" | "timeline" | single | series | comparison | list
                                    # (answer_format(question)); selects the shared answer prompt and its
                                    # num_predict ONLY (build_answer_messages / answer_num_predict);
                                    # approaches must not use it for retrieval
    pool_id: str = None             # the frozen pool (ragbench/pools.py) whose documents form doc_scope, or None


# task family of a multi-document question -> its answer format (Question.answer_format wins when given)
TASK_FAMILY_FORMATS = {"single_fact": "single", "routing": "single", "unanswerable": "single",
                       "series": "series", "cross_year": "series",
                       "comparison": "comparison", "cross_company": "comparison", "aggregation": "comparison",
                       "list": "list", "within_report_list": "list"}
NEW_QUESTION_FIELDS = ("schema", "task_family", "answer_format", "pool_id")


def answer_format(question):
    """The answer format of a question. Older questions: "timeline" for type timeline (series answers), else
    "default" (unchanged). A question with an explicit answer_format, or a task_family, gets that format
    (TASK_FAMILY_FORMATS) - the same for every method and option arm."""
    explicit = getattr(question, "answer_format", None)
    if isinstance(explicit, str) and explicit:
        return explicit
    fam = getattr(question, "task_family", None)
    if fam:
        if fam not in TASK_FAMILY_FORMATS:
            raise ValueError("unknown task_family %r; known: %s" % (fam, sorted(TASK_FAMILY_FORMATS)))
        return TASK_FAMILY_FORMATS[fam]
    return "timeline" if getattr(question, "type", None) == "timeline" else "default"


# --------------------------------------------------------------------------
# retrieval results and the evidence budget
# --------------------------------------------------------------------------

@dataclass
class RetrievedItem:
    """One piece of retrieved context. unit_ids = every corpus unit whose text the item
    contains (a chunk/window/section maps to several; a fact line to its source unit)."""
    text: str
    doc_id: str
    section_id: str = None
    unit_ids: list = field(default_factory=list)
    kind: str = "unit"             # unit | chunk | section | window | fact | node
    score: float = None
    rank: int = None
    span: list = None              # [start, end) in Document.text when the text is a verbatim slice
    meta: dict = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)


@dataclass
class EvidencePack:
    items: list                    # RetrievedItems actually given to the model (last one possibly cut)
    text: str                      # the rendered evidence block
    chars: int                     # len(text) <= budget
    budget: int
    n_offered: int
    n_included: int
    last_item_cut: bool

    @property
    def unit_ids(self):
        out = []
        for it in self.items:
            for u in it.unit_ids:
                if u not in out:
                    out.append(u)
        return out


def item_header(n, item, corpus):
    """'[n] <doc title> > <section path>' - part of the budget."""
    try:
        doc = corpus.doc(item.doc_id)
        parts = [doc.title]
        if item.section_id:
            parts += [t for t in doc.section_path(item.section_id) if t]
    except KeyError:
        parts = [item.doc_id]
    return "[%d] %s" % (n, " > ".join(p for p in parts if p))


UNIT_MIN_SHOWN_CHARS = 60      # a cut item still "contains" a unit when this many of its chars (or all) survive


def _verbatim_start(item, corpus):
    """Document offset of item.text.strip() when the item is a verbatim slice at item.span, else None."""
    if not item.span:
        return None
    try:
        doc = corpus.doc(item.doc_id)
    except KeyError:
        return None
    raw = item.text or ""
    body = raw.strip()
    start = int(item.span[0]) + (len(raw) - len(raw.lstrip()))
    if not body or doc.text[start:start + len(body)] != body:
        return None
    return start


def _units_in_cut(item, cut, corpus):
    """unit_ids of `item` still present in its cut text `cut` (ending in " …").
    Verbatim items (span given and matching Document.text): exact character overlap of the kept
    span with each unit, a unit counting when at least UNIT_MIN_SHOWN_CHARS of it (or all of it)
    survive. Other items: text heuristic (a unit's first 60 chars appear in the kept text, or the
    kept text starts inside the unit)."""
    start = _verbatim_start(item, corpus)
    if start is not None:
        kept_len = len(cut[:-2].rstrip()) if cut.endswith(" …") else len(cut)
        end = start + kept_len
        doc = corpus.doc(item.doc_id)
        want = set(item.unit_ids)
        keep = []
        for u in doc.units_overlapping(start, end):
            if u.id not in want:
                continue
            shown = min(u.span[1], end) - max(u.span[0], start)
            if shown >= min(UNIT_MIN_SHOWN_CHARS, u.span[1] - u.span[0]):
                keep.append(u.id)
        # ids the document does not know (e.g. synthetic) are kept as the item declared them
        keep += [uid for uid in item.unit_ids if not doc.has_unit(uid) and uid not in keep]
        return [uid for uid in item.unit_ids if uid in set(keep)]
    keep = []
    head = cut[:60]
    for uid in item.unit_ids:
        try:
            ut = corpus.unit(uid).text
        except KeyError:
            keep.append(uid)
            continue
        if not ut or ut[:60] in cut or head in ut:
            keep.append(uid)
    return keep


def pack_evidence(items, corpus, budget=None, cfg=None):
    """Render items in the given order into one evidence block of at most `budget`
    characters (default evidence.max_chars). Items are added whole while they fit; the
    first item that does not fit is cut to the remaining space if at least
    evidence.min_item_chars of its text survive (unit_ids re-derived from the kept text),
    otherwise dropped; packing stops there. Identical for every approach."""
    cfg = cfg or load_config()
    budget = int(budget or cfg.evidence["max_chars"])
    min_chars = int(cfg.evidence["min_item_chars"])
    parts, included = [], []
    used = 0
    cut = False
    for it in items:
        sep = "\n\n" if parts else ""
        header = item_header(len(included) + 1, it, corpus)
        body = (it.text or "").strip()
        if not body:
            continue
        block = header + "\n" + body
        if used + len(sep) + len(block) <= budget:
            parts.append(sep + block)
            used += len(sep) + len(block)
            included.append(it)
            continue
        room = budget - used - len(sep) - len(header) - 1 - 2   # 2 = " …"
        if room >= min_chars:
            kept = body[:room].rstrip() + " …"
            vstart = _verbatim_start(it, corpus)
            new = RetrievedItem(text=kept, doc_id=it.doc_id, section_id=it.section_id,
                                unit_ids=_units_in_cut(it, kept, corpus), kind=it.kind, score=it.score, rank=it.rank,
                                span=[vstart, vstart + len(kept) - 2] if vstart is not None else None,
                                meta=dict(it.meta, cut_from_chars=len(body)))
            block = header + "\n" + kept
            parts.append(sep + block)
            used += len(sep) + len(block)
            included.append(new)
            cut = True
        break
    text = "".join(parts)
    return EvidencePack(items=included, text=text, chars=len(text), budget=budget, n_offered=len(items),
                        n_included=len(included), last_item_cut=cut)


# --------------------------------------------------------------------------
# the shared answer prompt
# --------------------------------------------------------------------------

ANSWER_SYSTEM = (
    "You answer questions about company, market and sector reports.\n"
    "{source_rule}\n"
    "{abstain_rule}\n"
    "Answer in {answer_lang}. Reply with the answer only, in one or two short sentences. "
    "Copy numbers exactly as they are written, with their unit and period. Do not explain your reasoning."
)
SOURCE_RULE_EVIDENCE = "Use only the numbered evidence passages in the user message."
SOURCE_RULE_NONE = "No documents are provided; answer from your own knowledge."
ABSTAIN_RULE_EVIDENCE = "If the answer is not there, reply with exactly {abstain} and nothing else."
ABSTAIN_RULE_NONE = "If you do not know the answer, reply with exactly {abstain} and nothing else."
ANSWER_PROMPT_VERSION = "answer-v2"   # v2: the no-evidence (B0) variant got its own abstain rule
# Timeline questions (type "timeline", series answers scored by eval/timeline.py) ask for one figure over several
# periods. answer-v2's "one or two short sentences" and its 256 tokens cut such answers short, so they get this
# prompt and answer.timeline_num_predict instead (2026-09-16, CORE PLAN). Same source and abstain rules; the same
# prompt for every approach that makes the shared answer call (B0-B5, OURS at every index level, AGENTIC).
ANSWER_TIMELINE_SYSTEM = (
    "You answer questions about company, market and sector reports.\n"
    "{source_rule}\n"
    "{abstain_rule}\n"
    "Answer in {answer_lang}. The question asks for one figure over several periods. Reply with one line per period, "
    "earliest first, in the form <period>: <value> <unit>, and write <period>: {abstain} for a period whose value you "
    "cannot give. When reports give different numbers for the same period, use the number from that period's own "
    "report. Copy numbers exactly as they are written. End with one short sentence on how the figure changed; if what "
    "is counted differs between periods (a different definition or scope), say instead that the values are not "
    "directly comparable. Do not explain your reasoning."
)
ANSWER_TIMELINE_PROMPT_VERSION = "answer-timeline-v1"
# the formats of the finished study: their prompt versions are what config final.answer_prompts must declare
ANSWER_PROMPT_VERSIONS = {"default": ANSWER_PROMPT_VERSION, "timeline": ANSWER_TIMELINE_PROMPT_VERSION}
LEGACY_FORMATS = ("default", "timeline")

# Multi-document answer formats (2026-09-24, P0 list of the multi-document design). Chosen by the QUESTION
# (answer_format / task_family), never by retrieval success or a method option: every method and every option arm
# gets the same prompt and the same output allowance for the same question. Allowances: MULTIDOC_NUM_PREDICT, each
# overridable by config answer.<format>_num_predict (absent from config.yaml today, so no record hash changes).
ANSWER_SINGLE_SYSTEM = (
    "You answer questions about company, market and sector reports.\n"
    "{source_rule}\n"
    "{abstain_rule}\n"
    "Answer in {answer_lang}. The question asks for one fact. Reply with the answer only, in one or two short "
    "sentences, naming the company and period it belongs to. Copy numbers exactly as they are written, with their "
    "unit and period. Do not explain your reasoning."
)
ANSWER_SERIES_SYSTEM = (
    "You answer questions about company, market and sector reports.\n"
    "{source_rule}\n"
    "{abstain_rule}\n"
    "Answer in {answer_lang}. The question asks for one figure over several periods. Reply with one line per period, "
    "earliest first, in the form <period>: <value> <unit>, and write <period>: {abstain} for a period whose value you "
    "cannot give. When reports give different numbers for the same period, use the number from that period's own "
    "report. Copy numbers exactly as they are written. End with one short sentence on how the figure changed; if what "
    "is counted differs between periods (a different definition or scope), say instead that the values are not "
    "directly comparable. Do not explain your reasoning."
)
ANSWER_COMPARISON_SYSTEM = (
    "You answer questions about company, market and sector reports.\n"
    "{source_rule}\n"
    "{abstain_rule}\n"
    "Answer in {answer_lang}. The question compares values across companies or periods. First give each value the "
    "comparison needs on its own line, in the form <company> - <period>: <value> <unit>, and write {abstain} as the "
    "value of one you cannot give. Then give the comparison or calculation the question asks for in one short "
    "sentence, with the arithmetic in brief. If the definitions, periods or units differ, say that the values are not "
    "directly comparable instead of comparing them. Copy numbers exactly as they are written. Do not explain your "
    "reasoning."
)
ANSWER_LIST_SYSTEM = (
    "You answer questions about company, market and sector reports.\n"
    "{source_rule}\n"
    "{abstain_rule}\n"
    "Answer in {answer_lang}. The question asks for several items. Give them as a numbered list, one short line each, "
    "naming the company and period of an item whenever the question spans more than one. List each item once and "
    "only items the evidence states. Copy numbers exactly as they are written. Do not explain your reasoning."
)
MULTIDOC_SYSTEMS = {"single": ANSWER_SINGLE_SYSTEM, "series": ANSWER_SERIES_SYSTEM,
                    "comparison": ANSWER_COMPARISON_SYSTEM, "list": ANSWER_LIST_SYSTEM}
MULTIDOC_PROMPT_VERSIONS = {"single": "answer-single-v1", "series": "answer-series-v1",
                            "comparison": "answer-comparison-v1", "list": "answer-list-v1"}
MULTIDOC_NUM_PREDICT = {"single": 256, "series": 768, "comparison": 768, "list": 900}
MULTIDOC_FORMATS = tuple(MULTIDOC_SYSTEMS)

# Datasets v2 answer contract (2026-09-25, work/datasets_v2/SCHEMA.json answer_format). Three domains (company
# reports, non-fiction books, laws) in two languages, so the wording is domain-neutral (subject; aspect; time, version
# or condition) instead of "company and period". Used by every v2 question of every domain alike (no domain
# confound), identical for every method and option arm. The multidoc-v1 formats above are unchanged.
# Schema format -> runner format: single/list/comparison/series/explanation -> v2_<same>; "abstain" (unanswerable)
# is never shown as such (methods must not learn answerability): it maps to the format of the task the question
# attempts (V2_ABSTAIN_FORMAT by intended_tier).
_V2_HEAD = ("You answer questions about documents such as company reports, books and laws.\n"
            "{source_rule}\n"
            "{abstain_rule}\n"
            "Answer in {answer_lang}. ")
ANSWER_V2_SINGLE_SYSTEM = _V2_HEAD + (
    "The question asks for one fact. Reply with the answer only, in one or two short sentences, naming what it "
    "applies to (the subject and, where the question fixes them, the period, version or condition). Copy numbers "
    "exactly as they are written, with their unit. Do not explain your reasoning.")
ANSWER_V2_LIST_SYSTEM = _V2_HEAD + (
    "The question asks for several items. Give them as a numbered list, one short line each, naming the subject, "
    "period or version of an item whenever the question spans more than one. List each item once. Copy numbers "
    "exactly as they are written. Do not explain your reasoning.")
ANSWER_V2_COMPARISON_SYSTEM = _V2_HEAD + (
    "The question compares things across subjects, documents, periods or versions. First give each value or "
    "statement the comparison needs on its own line, in the form <subject> - <period or version>: <value or "
    "statement>, and write {abstain} for one you cannot give. Then give the comparison, difference or calculation the "
    "question asks for in one or two short sentences, with any arithmetic in brief. If the definitions, periods or "
    "units differ, say that the values are not directly comparable instead of comparing them. Copy numbers exactly as "
    "they are written. Do not explain your reasoning.")
ANSWER_V2_SERIES_SYSTEM = _V2_HEAD + (
    "The question asks for one value over several periods or versions. Reply with one line per period or version, "
    "earliest first, in the form <period or version>: <value> <unit>, and write <period or version>: {abstain} for one "
    "whose value you cannot give. Copy numbers exactly as they are written. End with one short sentence on how the "
    "value changed; if what is counted differs between periods (a different definition or scope), say instead that "
    "the values are not directly comparable. Do not explain your reasoning.")
ANSWER_V2_EXPLANATION_SYSTEM = _V2_HEAD + (
    "The question needs an answer that combines several points (for example a rule and its exception or condition, a "
    "definition applied to a case, a cause and its effect, or several steps of one account). Give the direct answer "
    "first, then each point it rests on, in at most six short sentences, naming the article, section, subject or "
    "period each point comes from. Answer every part of the question. Copy numbers exactly as they are written. Do "
    "not add points you cannot support.")
V2_SYSTEMS = {"v2_single": ANSWER_V2_SINGLE_SYSTEM, "v2_list": ANSWER_V2_LIST_SYSTEM,
              "v2_comparison": ANSWER_V2_COMPARISON_SYSTEM, "v2_series": ANSWER_V2_SERIES_SYSTEM,
              "v2_explanation": ANSWER_V2_EXPLANATION_SYSTEM}
V2_PROMPT_VERSIONS = {f: "answer-%s-v1" % f.replace("_", "-") for f in V2_SYSTEMS}
V2_NUM_PREDICT = {"v2_single": 256, "v2_list": 900, "v2_comparison": 768, "v2_series": 768, "v2_explanation": 640}
V2_FORMATS = tuple(V2_SYSTEMS)
V2_SCHEMA_FORMATS = {"single": "v2_single", "list": "v2_list", "comparison": "v2_comparison", "series": "v2_series",
                     "explanation": "v2_explanation"}
V2_ABSTAIN_FORMAT = {"T1": "v2_single", "T2": "v2_explanation", "T3": "v2_comparison", None: "v2_single"}


def v2_runner_format(schema_format, intended_tier=None):
    """The runner answer format of a datasets-v2 question (schema answer_format; intended_tier for "abstain")."""
    if schema_format == "abstain":
        return V2_ABSTAIN_FORMAT[intended_tier]
    return V2_SCHEMA_FORMATS[schema_format]


ANSWER_SYSTEMS = {"default": ANSWER_SYSTEM, "timeline": ANSWER_TIMELINE_SYSTEM, **MULTIDOC_SYSTEMS, **V2_SYSTEMS}
ALL_ANSWER_PROMPT_VERSIONS = dict(ANSWER_PROMPT_VERSIONS, **MULTIDOC_PROMPT_VERSIONS, **V2_PROMPT_VERSIONS)
ANSWER_FORMATS = tuple(ANSWER_SYSTEMS)
ANSWER_USER_EVIDENCE = "Evidence:\n{evidence}\n\nQuestion: {question}"
ANSWER_USER_NONE = "Question: {question}"
# config sections hashed into every RunRecord.config_hash (resume refuses mixing hashes); "answer" since 2026-09-16
RECORD_HASH_SECTIONS = ("models", "generation", "evidence", "chunking", "caps", "answer")


def _answer_format_of(query_or_format):
    fmt = query_or_format if isinstance(query_or_format, str) else getattr(query_or_format, "answer_format", None)
    fmt = fmt or "default"
    if fmt not in ANSWER_SYSTEMS:
        raise ValueError("unknown answer format %r; known: %s" % (fmt, ANSWER_FORMATS))
    return fmt


def answer_prompt_version(query_or_format):
    """Prompt version of the shared answer call for a Query (or an answer format name)."""
    return ALL_ANSWER_PROMPT_VERSIONS[_answer_format_of(query_or_format)]


def answer_num_predict(query_or_format, cfg=None):
    """num_predict of the shared answer call: generation.num_predict_answer for the default format,
    answer.timeline_num_predict for timeline questions (a config without it is an error, never a silent 256);
    a multi-document format: config answer.<format>_num_predict when present, else MULTIDOC_NUM_PREDICT."""
    cfg = cfg or load_config()
    fmt = _answer_format_of(query_or_format)
    if fmt in MULTIDOC_NUM_PREDICT or fmt in V2_NUM_PREDICT:
        n = cfg.get("answer.%s_num_predict" % fmt)
        return int(n if n is not None else {**MULTIDOC_NUM_PREDICT, **V2_NUM_PREDICT}[fmt])
    if fmt == "timeline":
        n = cfg.get("answer.timeline_num_predict")
        if n is None:
            raise KeyError("config %s has no answer.timeline_num_predict (the timeline answer budget)" % cfg.source)
        return int(n)
    return int(cfg.generation["num_predict_answer"])


def build_answer_messages(query, evidence, cfg=None):
    """messages for the final answer call. evidence: EvidencePack, or None for B0 only.
    The system prompt follows query.answer_format (ANSWER_SYSTEMS); the user message is the same for every format."""
    cfg = cfg or load_config()
    has_ev = evidence is not None
    abstain = cfg.evidence["abstain_token"]
    template = ANSWER_SYSTEMS[_answer_format_of(query)]
    system = template.format(source_rule=SOURCE_RULE_EVIDENCE if has_ev else SOURCE_RULE_NONE,
                             abstain_rule=(ABSTAIN_RULE_EVIDENCE if has_ev else ABSTAIN_RULE_NONE).format(abstain=abstain),
                             answer_lang=LANG_NAMES[query.q_lang], abstain=abstain)
    if has_ev:
        user = ANSWER_USER_EVIDENCE.format(evidence=evidence.text if evidence.text else "(no passages found)",
                                           question=query.text.strip())
    else:
        user = ANSWER_USER_NONE.format(question=query.text.strip())
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


# --------------------------------------------------------------------------
# records
# --------------------------------------------------------------------------

@dataclass
class AnswerResult:
    """What _answer() returns. evidence must come from pack_evidence (None only for B0).
    retrieved: everything the approach retrieved, best first, before the budget cut.
    structured: optional machine-readable answer, e.g. {"series": [...]} (contract in eval/timeline.py)."""
    answer_text: str
    evidence: EvidencePack = None
    retrieved: list = field(default_factory=list)
    trace: object = None
    structured: dict = None


@dataclass
class RunRecord:
    question_id: str
    approach: str                  # variant, e.g. "OURS+no_facts"
    approach_base: str             # "B0".."B5" | "OURS"
    flags: dict
    setting: str                   # "fa-en"
    q_lang: str
    d_lang: str
    corpus: str
    corpus_set: str
    answer_text: str
    abstained: bool
    retrieved_units: list          # RetrievedItem dicts (all retrieved, best first)
    evidence_units: list           # unit ids inside the evidence given to the answer call
    evidence_items: list           # [{doc_id, section_id, kind, chars, unit_ids, cut}]
    evidence_chars: int
    llm_calls: int                 # chat calls to the local model during this question
    embed_calls: int
    prompt_tokens: int
    completion_tokens: int
    embed_tokens: int
    llm_wall_ms: float
    load_ms: float
    latency_ms: float              # wall time of the whole answer() (retrieval + calls)
    failed_calls: int
    error: str
    trace: object
    models: dict
    config_hash: str
    index_fingerprint: str
    ts: str
    run_id: str
    lock_wait_ms: float = 0.0      # part of latency_ms spent waiting for the shared GPU lock (other processes)
    split: str = None              # "dev" | "test" | "other", stamped by ragbench.run from the question file name
    meta: dict = field(default_factory=dict)   # run provenance stamped by ragbench.run: ollama_version,
                                               # questions_file, final, docs_restriction; Approach.answer adds
                                               # answer_format, answer_prompt, answer_num_predict
    structured: dict = None        # copied from AnswerResult.structured; None for approaches without one

    def to_dict(self):
        return asdict(self)

    def key(self):
        return (self.question_id, self.approach, self.setting)


class RunLog:
    """Append-only JSONL of RunRecords; completed() gives the keys to skip on resume."""

    def __init__(self, path):
        self.path = Path(path)

    def append(self, record):
        append_jsonl(self.path, record.to_dict() if hasattr(record, "to_dict") else record)

    def records(self):
        return read_jsonl(self.path)

    def completed(self, include_errors=False):
        return {(r["question_id"], r["approach"], r["setting"]) for r in self.records()
                if include_errors or not r.get("error")}


# --------------------------------------------------------------------------
# approaches
# --------------------------------------------------------------------------

@dataclass
class IndexHandle:
    approach: str
    corpus_set: str
    path: Path
    fingerprint: str
    corpus: object                 # the CorpusSet the index was built over (text lookups allowed)
    payload: object                # whatever _build_index/_load_index returned
    stats: dict                    # build_stats.json: totals over all build attempts + size_bytes


class Approach(ABC):
    """Subclass contract:
        name = "B3"                                   # class attribute
        needs_gold = False                            # True only for B1
        def config_signature(self) -> dict            # params that change the index (optional)
        def _build_index(self, corpus_set, index_dir) -> payload     # resumable; index-time calls allowed
        def _load_index(self, corpus_set, index_dir) -> payload      # optional; default re-runs _build_index
        def _answer(self, query, index) -> AnswerResult              # query-time
    Call the model only through self.client (chat/embed need a `purpose`; phase/approach
    come from the ledger scope opened by the base class). Produce the final answer with
    self.generate_answer(query, evidence) so the shared prompt and budget apply.
    """
    name = "BASE"
    needs_gold = False

    def __init__(self, cfg=None, client=None, flags=None):
        self.cfg = cfg or load_config()
        self.client = client or OllamaClient(self.cfg)
        self.flags = dict(flags or {})
        self._indexes = {}
        # a frozen pool (ragbench/pools.py Pool) masks the doc scope of every query; None = the question's own scope
        self.pool = None
        # the variants.py semantics of the doc_card / section_expand flags: "variants-v1" (the finished study's, kept
        # for reproducibility) or "variants-v2" (query-ranked cards built inside the timed query, expansion that must
        # preserve every represented document; set by ragbench.run for multi-document experiments)
        self.variants_version = "variants-v1"
        # the two optional method options of ragbench/variants.py (off unless flagged; no effect otherwise)
        self._variant_trace = None
        if self.flags.get("doc_card") or self.flags.get("section_expand"):
            from .variants import VariantClient
            self.client = VariantClient(self.client)

    # identity -----------------------------------------------------------------
    @property
    def variant(self):
        if not self.flags:
            return self.name
        parts = [k if v is True else "%s=%s" % (k, v) for k, v in sorted(self.flags.items()) if v not in (False, None)]
        return self.name + ("+" + ",".join(parts) if parts else "")

    # flags that act at query time only: they name the run variant but not the index (fingerprint / index_dir)
    index_neutral_flags = ()

    @property
    def index_variant(self):
        """The variant name without index_neutral_flags: what identifies the index."""
        if not self.index_neutral_flags:
            return self.variant
        kept = {k: v for k, v in self.flags.items() if k not in self.index_neutral_flags}
        parts = [k if v is True else "%s=%s" % (k, v) for k, v in sorted(kept.items()) if v not in (False, None)]
        return self.name + ("+" + ",".join(parts) if parts else "")

    def config_signature(self):
        return {}

    def index_fingerprint(self, corpus_set):
        return sha1_json({"variant": self.index_variant, "corpus": corpus_set.fingerprint(),
                          "models": self.cfg.models, "sig": self.config_signature()})

    def index_dir(self, corpus_set):
        return self.cfg.path("index_root") / safe_name(self.index_variant) / ("%s-%s" % (corpus_set.name, self.index_fingerprint(corpus_set)[:8]))

    # index --------------------------------------------------------------------
    @abstractmethod
    def _build_index(self, corpus_set, index_dir):
        ...

    def _load_index(self, corpus_set, index_dir):
        return self._build_index(corpus_set, index_dir)

    def build_index(self, corpus_set, ledger=None):
        """Build (or load) the index for a CorpusSet; index-time calls are ledgered with
        phase="index" and summed into <index_dir>/build_stats.json across resumed attempts."""
        d = self.index_dir(corpus_set)
        d.mkdir(parents=True, exist_ok=True)
        stats_path = d / "build_stats.json"
        stats = read_json(stats_path) or {"approach": self.variant, "corpus_set": corpus_set.name,
                                          "fingerprint": self.index_fingerprint(corpus_set), "complete": False,
                                          "attempts": []}
        ledger = ledger or self.client.ledger
        if stats.get("complete"):
            payload = self._load_index(corpus_set, d)
        else:
            with self.client.use_ledger(ledger), ledger.scope(phase="index", approach=self.variant,
                                                             corpus_set=corpus_set.name) as sc:
                t0 = time.perf_counter()
                attempt = {"started": now_iso()}
                try:
                    payload = self._build_index(corpus_set, d)
                    attempt["ok"] = True
                except BaseException as e:
                    attempt.update(ok=False, error="%s: %s" % (type(e).__name__, str(e)[:300]))
                    raise
                finally:
                    attempt["build_ms"] = round((time.perf_counter() - t0) * 1000.0, 1)
                    attempt.update(sc.summary())
                    stats["attempts"].append(attempt)
                    tot = {}
                    for a in stats["attempts"]:
                        for k in ("build_ms", "llm_calls", "embed_calls", "prompt_tokens", "completion_tokens",
                                  "embed_tokens", "llm_wall_ms", "failed_calls"):
                            tot[k] = round(tot.get(k, 0) + (a.get(k) or 0), 1)
                    stats["totals"] = tot
                    stats["complete"] = bool(attempt.get("ok"))
                    if stats["complete"]:
                        stats["completed_at"] = now_iso()
                    stats["size_bytes"] = dir_size_bytes(d)
                    atomic_write_json(stats_path, stats, indent=1)
        stats["size_bytes"] = dir_size_bytes(d)
        h = IndexHandle(self.variant, corpus_set.name, d, stats["fingerprint"], corpus_set, payload, stats)
        self._indexes[corpus_set.name] = h
        return h

    def ensure_index(self, set_name):
        if set_name not in self._indexes:
            self.build_index(load_set(set_name, self.cfg))
        return self._indexes[set_name]

    # query --------------------------------------------------------------------
    @abstractmethod
    def _answer(self, query, index):
        ...

    def make_query(self, question, setting):
        s = Setting.parse(setting)
        set_name = set_name_for(question.corpus, s.d_lang)
        scope = question.scope(s.d_lang) if self.cfg.evidence.get("use_doc_scope", True) else "corpus"
        pool_id = None
        if self.pool is not None:           # the pool is a scope mask; the index stays that of the full frozen set
            scope = self.pool.mask(scope, set_name)
            pool_id = self.pool.pool_id
        return Query(question_id=question.id, text=question.text(s.q_lang), q_lang=s.q_lang, d_lang=s.d_lang,
                     setting=s.name, corpus=question.corpus, corpus_set=set_name, doc_scope=scope,
                     gold_unit_ids=question.gold(s.d_lang) if self.needs_gold else None,
                     answer_format=answer_format(question), pool_id=pool_id)

    def answer(self, question, setting):
        query = self.make_query(question, setting)
        index = self.ensure_index(query.corpus_set)
        ledger = self.client.ledger
        err, res, trace = None, None, None
        self._answer_call = None
        timed_variants = self.variants_version != "variants-v1"
        if not timed_variants:
            self._begin_variants(query, index)          # v1: before the timer (the finished study's measurement)
        with ledger.scope(phase="query", approach=self.variant, question_id=query.question_id,
                          setting=query.setting, corpus_set=query.corpus_set) as sc:
            t0 = time.perf_counter()
            try:
                if timed_variants:
                    self._begin_variants(query, index)  # v2: card construction is part of the measured query time
                res = self._answer(query, index)
                if self._variant_trace is not None and isinstance(res.trace, dict):
                    res.trace["variants"] = self._variant_trace
                if res.evidence is not None and not isinstance(res.evidence, EvidencePack):
                    raise TypeError("AnswerResult.evidence must come from pack_evidence()")
                trace = res.trace
            except Exception as e:  # noqa: BLE001
                err = "%s: %s" % (type(e).__name__, str(e)[:500])
                trace = {"traceback": traceback.format_exc()[-2000:]}
            latency = (time.perf_counter() - t0) * 1000.0
        ev = res.evidence if res else None
        answer_text = (res.answer_text if res else "") or ""
        summ = sc.summary()
        return RunRecord(
            question_id=query.question_id, approach=self.variant, approach_base=self.name, flags=self.flags,
            setting=query.setting, q_lang=query.q_lang, d_lang=query.d_lang, corpus=query.corpus,
            corpus_set=query.corpus_set, answer_text=answer_text,
            abstained=answer_text.strip() == self.cfg.evidence["abstain_token"],
            retrieved_units=[it.to_dict() for it in (res.retrieved if res else [])],
            evidence_units=ev.unit_ids if ev else [],
            evidence_items=[{"doc_id": it.doc_id, "section_id": it.section_id, "kind": it.kind, "chars": len(it.text),
                             "unit_ids": it.unit_ids, "cut": "cut_from_chars" in (it.meta or {})} for it in (ev.items if ev else [])],
            evidence_chars=ev.chars if ev else 0,
            llm_calls=summ["llm_calls"], embed_calls=summ["embed_calls"], prompt_tokens=summ["prompt_tokens"],
            completion_tokens=summ["completion_tokens"], embed_tokens=summ["embed_tokens"],
            llm_wall_ms=summ["llm_wall_ms"], load_ms=summ["load_ms"], latency_ms=round(latency, 1),
            failed_calls=summ["failed_calls"], error=err, trace=trace,
            models={"chat": self.cfg.models["chat"], "embed": self.cfg.models["embed"]},
            config_hash=self.cfg.hash(*RECORD_HASH_SECTIONS),
            index_fingerprint=index.fingerprint, ts=now_iso(), run_id=ledger.run_id,
            lock_wait_ms=summ.get("lock_wait_ms", 0.0),
            meta={"answer_format": query.answer_format, "answer_prompt": answer_prompt_version(query),
                  "answer_num_predict": answer_num_predict(query, self.cfg)},
            structured=(getattr(res, "structured", None) if res else None))

    # helpers for subclasses ----------------------------------------------------
    def docs_in_scope(self, query, index):
        if query.doc_scope == "corpus":
            return list(index.corpus.docs)
        return [index.corpus.doc(d) for d in query.doc_scope if index.corpus.has_doc(d)]

    def pack(self, items, index, budget=None):
        return pack_evidence(items, index.corpus, budget=budget, cfg=self.cfg)

    # the optional options of ragbench/variants.py --------------------------------------------------------------
    def _begin_variants(self, query, index):
        """Per question: the document card of the question's scope (doc_card) and a fresh trace entry."""
        if not (self.flags.get("doc_card") or self.flags.get("section_expand")):
            self._variant_trace = None
            return
        from . import variants as V
        v2 = self.variants_version != "variants-v1"
        self._variant_trace = {"version": V.VARIANTS_V2_VERSION if v2 else V.VARIANTS_VERSION,
                               "source_sha1": V.VARIANTS_SHA1,
                               "flags": sorted(k for k in V.FLAGS if self.flags.get(k))}
        self.client.list_answer = False
        self.client.card = None
        if self.flags.get("doc_card"):
            docs = self.docs_in_scope(query, index)
            if v2:      # query-ranked shortlist (routed ids first), selected ids and order logged
                card, sel = V.card_shortlist(docs, query.text, routed_ids=self.variant_routed_ids(query, index))
                self.client.card = card
                self._variant_trace["card_selection"] = sel
            else:       # v1: the first 12 cards in corpus order (the finished study)
                cards = [V.document_card(d) for d in docs]
                self.client.card = V.card_context(cards)
            self._variant_trace["card"] = self.client.card

    def variant_routed_ids(self, query, index):
        """Optional hook for variants-v2 card ranking: document ids a method has already routed to, best first (they
        lead the card shortlist). None = rank by the question alone."""
        return None

    def final_pack(self, items, index, query):
        """The evidence the answer call gets. Identical to pack() unless the section_expand option is on; approaches
        call it once, at their final evidence boundary (provisional packs keep using pack()).
        variants-v1 (the finished study, kept for reproducibility): a successful list expansion also switches the
        answer call to the list rule and LIST_NUM_PREDICT - only for the old formats (default / timeline).
        variants-v2: the expansion is kept only when the ACTUAL packed result still shows every document and every
        reserved requirement the unexpanded pack showed; the answer instruction and allowance never change (they come
        from the question's answer format)."""
        if not self.flags.get("section_expand"):
            return self.pack(items, index)
        from . import variants as V
        budget = int(self.cfg.evidence["max_chars"])
        if self.variants_version != "variants-v1":
            items2, dec = V.expand_sections_safe(items, index.corpus, query.text, budget,
                                                 pack=lambda its: self.pack(its, index))
            if self._variant_trace is not None:
                self._variant_trace["section_expand"] = dec
            return self.pack(items2, index)
        items2, dec = V.expand_sections(items, index.corpus, query.text, budget)
        if self._variant_trace is not None:
            self._variant_trace["section_expand"] = dec
        if dec.get("applied") and dec.get("list_question") and query.answer_format in LEGACY_FORMATS:
            self.client.list_answer = True
        return self.pack(items2, index)

    def generate_answer(self, query, evidence, purpose="answer"):
        """The shared final answer call. evidence: EvidencePack (None only for B0). Prompt and num_predict follow
        query.answer_format (answer-v2 + generation.num_predict_answer, or the timeline prompt + its budget)."""
        msgs = build_answer_messages(query, evidence, self.cfg)
        npred = answer_num_predict(query, self.cfg)
        # the exact answer-call input (2026-10-01, multi-reader test): kept so the SAME pack can be replayed to other
        # readers (tools/reader_replay.py); written by answer() after the timed scope, only when RAGBENCH_PACK_DIR is set
        self._answer_call = {"messages": msgs, "num_predict": npred, "purpose": purpose}
        r = self.client.chat(msgs, purpose=purpose, num_predict=npred)
        return r.text.strip()
