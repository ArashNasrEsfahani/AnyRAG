"""LLM judge for answer accuracy.

    judge = make_judge("openrouter", cfg, ledger, max_calls=10, max_cost_usd=0.02)
    v = judge.verdict(question, record)   # dict or None (judge "none", or budget used up)

Verdict dict: {correct: bool|None, partially_correct: bool|None, reason: str, source: rule|cache|judge|budget|error,
               judge: label, cost_usd, weaker: bool}

* provider "openrouter": OpenRouter chat completions through llm.OpenRouterClient, which
  reads OPENROUTER_API_KEY from the environment or D:/KareOne/RAG Paper/.env on demand
  (never printed or logged here) and enforces the persisted workflow budget
  (judge.max_calls_total / judge.max_cost_usd in work/judge_budget.json). On top of that
  this module enforces a per-invocation call cap and an ESTIMATED cost cap checked before
  each call (prompt chars / 3 tokens + max_tokens at the model's list price).
* provider "ollama": the local model as a fallback judge. It is WEAKER and, with the
  default model, the same model that produced the answers; results are labelled so.
* provider "none": no judge; aggregation reports match-based accuracy only.

Rules decided without a model (identical for every provider, never cached as judge output):
  run error or empty answer -> incorrect; answer is exactly the abstain token -> correct iff
  the question is unanswerable.
Verdicts are cached on disk by a hash of (prompt version, judge label, question text as asked,
gold answers, normalised model answer) in work/judge/verdicts.sqlite, so re-aggregation is free.

Timeline questions (answer_type "series") use a second rubric (SERIES_PROMPT_VERSION): the judge sees the gold
series with each period's definition and source report, the restated values that must NOT count, the
comparability flag and notes, and returns strict JSON
  {"period_correct": {"1398": true, ...}, "flags_definition_change": bool, "false_trend": bool,
   "correct": bool, "partially_correct": bool, "reason": str}
The verdict dict then carries `series` = {period_correct, coverage, flags_definition_change, false_trend}.
Same caps, budget, cache and rule verdicts as the answer rubric; a larger max_tokens (SERIES_MAX_TOKENS).
"""
import argparse
import re
import sys
import time

from ..config import load_config
from ..llm import BudgetExceeded, ModelError, OllamaClient, OpenRouterClient, extract_json, default_ledger
from ..normalize import normalize
from ..util import KVCache, sha1_json

PROMPT_VERSION = "judge-v1"
DEFAULT_MAX_TOKENS = 160
# USD per 1M tokens (input, output), list prices used only for the pre-call estimate.
PRICES = {
    "google/gemini-2.5-flash-lite": (0.10, 0.40),
    "google/gemini-2.0-flash-lite-001": (0.075, 0.30),
    "google/gemini-2.5-flash": (0.30, 2.50),
    "openai/gpt-4o-mini": (0.15, 0.60),
    "openai/gpt-4.1-nano": (0.10, 0.40),
    "deepseek/deepseek-chat": (0.30, 0.85),
}
UNKNOWN_PRICE = (1.0, 4.0)   # conservative when the model is not in the table

LANG_NAMES = {"fa": "Persian", "en": "English"}

JUDGE_SYSTEM = (
    "You are a strict, impartial grader for a question-answering benchmark over company, market and sector "
    "reports written in Persian and English. You compare a model's answer with the gold answer. "
    "You do not use outside knowledge about the facts; the gold answer is the truth. "
    "Reply with one JSON object only."
)

JUDGE_USER = """Question as asked ({q_lang_name}): {question}
{question_other_line}Answer type: {answer_type}
Gold answer (Persian): {gold_fa}
Gold answer (English): {gold_en}
{gold_value_line}
Model answer: {answer}

Grading rules:
- "correct" is true when the model answer states the gold answer's key fact: the same number (a rounding at the precision written is fine), with a consistent unit/scale (e.g. 12,500 million = 12.5 billion; 1 toman = 10 rials) and, when the question asks for it, the same entity and period. Either language is acceptable. Extra harmless detail is fine. Ignore bracketed citation markers such as [2].
- "partially_correct" is true when the answer is incomplete or has a minor error (for example the right number with a wrong or missing unit or period, or only part of a multi-part answer). It is false whenever "correct" is true.
- If the question is unanswerable, "correct" is true only when the model declines to answer (for example NOT_FOUND or "the documents do not say").
- If the question is answerable and the model declines or gives a different fact, both are false.
Reply with JSON only: {{"correct": true or false, "partially_correct": true or false, "reason": "at most 20 words"}}"""

UNANSWERABLE_GOLD = "(unanswerable: the documents do not contain this information; the model should decline)"

# v1 (2026-09-16) was probed once on OpenRouter (gemini-2.5-flash-lite): strict JSON and every field parsed, but it
# graded "grew steadily" across a users -> phones change as false_trend=false, correct=true. v2 puts the
# comparability warning before the gold series, asks for the two comparability fields first, gives an example, and
# the parser derives "correct" from the fields. v2 has not been probed live (the workflow judge budget was used up).
SERIES_PROMPT_VERSION = "judge-series-v2"
SERIES_MAX_TOKENS = 320

SERIES_JUDGE_USER = """Question as asked ({q_lang_name}): {question}
{question_other_line}Answer type: series (one value per period, assembled from several reports)
{warning_block}
Gold series (the value published in the report of each period):
{gold_lines}
{restated_lines}{optional_lines}Reference answer (English): {reference}

Model answer: {answer}

Grading rules, in this order:
1. "flags_definition_change": true when the model answer says the figures are not directly comparable, or that what is counted (definition, scope, cumulative vs annual) changes between periods. Listing the values alone is not a flag.
2. "false_trend": {false_trend_rule}
3. "period_correct": for each gold period, true when the model answer gives that period a value equal to the gold value: a rounding at the precision written is fine, the unit and scale must be consistent (1.34 thousand = 1,340; 13.3 million = 13,300,000). A restated value listed above, a value for another period, or no value is false. Either language is acceptable. Ignore bracketed citation markers such as [2].
4. "correct": true when every gold period is correct{correct_rule}. Extra correct periods are fine.
5. "partially_correct": true when at least one gold period is correct but "correct" is false.
Reply with JSON only, keys in this order: {{"flags_definition_change": true or false, "false_trend": true or false, "period_correct": {{{period_template}}}, "correct": true or false, "partially_correct": true or false, "reason": "at most 25 words"}}"""

SERIES_WARNING = """WARNING: the figures of this series are NOT comparable across periods. {notes}
An answer that describes growth, decline, doubling or any trend across these periods without saying that the figures count different things states a FALSE TREND (for example "grew steadily from 35 million users to 53.1 million" when the first figure counts users and the last counts phones).
"""


def _series_spec(question):
    from .timeline import timeline_spec
    return timeline_spec(question)


def _fmt_value(item):
    v = item.get("value")
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        v = ("{:,}".format(int(v)) if float(v).is_integer() else repr(float(v)))
    return "%s %s" % (v, item.get("unit") or "")


def build_series_judge_messages(question, record):
    q_lang = record["q_lang"]
    other = "en" if q_lang == "fa" else "fa"
    spec = _series_spec(question)
    q_other = question.text(other) if question.text(other) and question.text(other) != question.text(q_lang) else ""
    gold_lines = "\n".join("- %s: %s (report: %s; counts: %s)" % (it["period"], _fmt_value(it).strip(), it.get("doc"),
                                                                  it.get("definition_en") or "-")
                           for it in spec["answer_series"])
    rec = spec["reconciliation"]
    restated = "; ".join("%s: %s (by %s)" % (r["period"], _fmt_value(r).strip(), r.get("doc")) for r in rec["restatements"])
    optional = "; ".join("%s: %s" % (s["period"], _fmt_value(s).strip()) for s in rec["supplementary"])
    comparable = spec["comparable"] is not False
    notes = (spec["series_notes_en"] or "").strip()
    user = SERIES_JUDGE_USER.format(
        q_lang_name=LANG_NAMES.get(q_lang, q_lang), question=(question.text(q_lang) or "").strip(),
        question_other_line=("Same question in %s: %s\n" % (LANG_NAMES.get(other, other), q_other.strip())) if q_other else "",
        warning_block=("\n" + SERIES_WARNING.format(notes=notes)) if not comparable else
        ("The figures are comparable across periods.%s\n" % ((" Notes: " + notes) if notes else "")),
        gold_lines=gold_lines,
        restated_lines=("Restated values that are NOT correct for their period: %s\n" % restated) if restated else "",
        optional_lines=("Optional extra periods (correct if stated, not required): %s\n" % optional) if optional else "",
        false_trend_rule=("true when the model answer asserts growth, decline or any other trend across these non-comparable "
                          "periods and does not flag the change (see the WARNING above)." if not comparable else
                          "false (the series is comparable)."),
        correct_rule=(", the answer flags the change, and false_trend is false" if not comparable else ""),
        reference=(question.answer_en or "").strip(), answer=(record.get("answer_text") or "").strip(),
        period_template=", ".join('"%s": true or false' % it["period"] for it in spec["answer_series"]))
    return [{"role": "system", "content": JUDGE_SYSTEM}, {"role": "user", "content": user}]


_RE_PERIOD_BOOL = re.compile(r'"(\d{4}[^"]{0,12})"\s*:\s*"?(true|false)\b', re.I)


def parse_series_verdict(text, periods, comparable=None):
    """Series JSON verdict -> dict with `series`, or None without a usable verdict. Missing periods count false.
    "correct" is kept consistent with the judge's own fields: false unless every period is correct and, for a
    non-comparable series (comparable=False), the change is flagged without a false trend; a missing "correct" is
    derived from the fields. The judge's raw value is kept as series.correct_raw."""
    obj = extract_json(text or "", None)
    periods = [str(p) for p in periods]
    pc, strict = {}, False
    if isinstance(obj, dict) and isinstance(obj.get("period_correct"), dict):
        raw = {str(k): _as_bool(v) for k, v in obj["period_correct"].items()}
        pc = {p: bool(raw.get(p)) for p in periods}
        flag, trend = _as_bool(obj.get("flags_definition_change")), _as_bool(obj.get("false_trend"))
        correct, partial = _as_bool(obj.get("correct")), _as_bool(obj.get("partially_correct"))
        reason = str(obj.get("reason") or "")[:300]
        strict = all(isinstance(v, bool) for v in obj["period_correct"].values()) and isinstance(obj.get("correct"), bool)
    else:
        found = {m.group(1): m.group(2).lower() == "true" for m in _RE_PERIOD_BOOL.finditer(text or "")}
        if not found:
            return None
        pc = {p: bool(found.get(p)) for p in periods}
        mf = re.search(r'"flags_definition_change"\s*:\s*"?(true|false)', text or "", re.I)
        mt = re.search(r'"false_trend"\s*:\s*"?(true|false)', text or "", re.I)
        mc = _RE_BOOL["correct"].search(text or "")
        mp = _RE_BOOL["partially_correct"].search(text or "")
        flag = (mf.group(1).lower() == "true") if mf else None
        trend = (mt.group(1).lower() == "true") if mt else None
        correct = (mc.group(1).lower() == "true") if mc else None
        partial = (mp.group(1).lower() == "true") if mp else None
        mr = _RE_REASON.search(text or "")
        reason = (mr.group(1) if mr else "")[:300]
    coverage = (sum(1 for v in pc.values() if v) / float(len(periods))) if periods else None
    raw_correct = correct
    derived = bool(coverage == 1.0 and not trend and (comparable is not False or flag))
    correct = derived if correct is None else bool(correct and derived)
    if partial is None or (raw_correct and not correct):
        partial = bool(coverage)
    return {"correct": bool(correct), "partially_correct": bool(partial and not correct), "reason": reason,
            "strict_json": strict,
            "series": {"period_correct": pc, "coverage": coverage, "flags_definition_change": flag, "false_trend": trend,
                       "correct_raw": raw_correct}}


def _abstain_token(cfg):
    return cfg.evidence.get("abstain_token", "NOT_FOUND")


def build_judge_messages(question, record):
    q_lang = record["q_lang"]
    other = "en" if q_lang == "fa" else "fa"
    unans = question.answer_type == "unanswerable"
    q_other = question.text(other) if question.text(other) and question.text(other) != question.text(q_lang) else ""
    gv = ""
    if question.answer_type == "numeric" and question.answer_value is not None:
        gv = "Gold numeric value: %s %s" % (_fmt_num(question.answer_value), question.answer_unit or "")
    user = JUDGE_USER.format(
        q_lang_name=LANG_NAMES.get(q_lang, q_lang), question=(question.text(q_lang) or "").strip(),
        question_other_line=("Same question in %s: %s\n" % (LANG_NAMES.get(other, other), q_other.strip())) if q_other else "",
        answer_type=question.answer_type,
        gold_fa=UNANSWERABLE_GOLD if unans else (question.answer_fa or "(none)").strip(),
        gold_en=UNANSWERABLE_GOLD if unans else (question.answer_en or "(none)").strip(),
        gold_value_line=gv.strip(), answer=(record.get("answer_text") or "").strip())
    user = re.sub(r"\n{2,}(?=Model answer)", "\n", user)
    return [{"role": "system", "content": JUDGE_SYSTEM}, {"role": "user", "content": user}]


def _fmt_num(v):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return str(v)
    return ("%d" % f) if f.is_integer() else repr(f)


_RE_BOOL = {k: re.compile(r'"%s"\s*:\s*"?(true|false)\b' % k, re.I) for k in ("correct", "partially_correct")}
_RE_REASON = re.compile(r'"reason"\s*:\s*"((?:[^"\\]|\\.)*)', re.S)


def _as_bool(x):
    if isinstance(x, bool):
        return x
    if isinstance(x, str) and x.strip().lower() in ("true", "false"):
        return x.strip().lower() == "true"
    return None


def parse_verdict(text):
    """Strict JSON verdict -> dict, or None when the reply has no usable 'correct' boolean.
    Tolerates code fences, "true"/"false" strings and a reply truncated after the booleans."""
    obj = extract_json(text or "", None)
    out = {}
    if isinstance(obj, dict) and _as_bool(obj.get("correct")) is not None:
        out["correct"] = _as_bool(obj["correct"])
        pc = _as_bool(obj.get("partially_correct"))
        out["partially_correct"] = bool(pc)
        out["reason"] = str(obj.get("reason") or "")[:300]
        out["strict_json"] = isinstance(obj.get("correct"), bool)
    else:
        m = _RE_BOOL["correct"].search(text or "")
        if not m:
            return None
        out["correct"] = m.group(1).lower() == "true"
        mp = _RE_BOOL["partially_correct"].search(text or "")
        out["partially_correct"] = bool(mp and mp.group(1).lower() == "true")
        mr = _RE_REASON.search(text or "")
        out["reason"] = (mr.group(1) if mr else "")[:300]
        out["strict_json"] = False
    if out["correct"]:
        out["partially_correct"] = False
    return out


class JudgeBudgetExhausted(RuntimeError):
    pass


class Judge:
    """Base: rules, cache, budget bookkeeping. Subclasses implement _call(messages, ctx) -> (text, cost)."""
    provider = "none"
    weaker = False

    def __init__(self, cfg=None, ledger=None, model=None, max_calls=None, max_cost_usd=None, cache_path=None,
                 use_cache=True):
        self.cfg = cfg or load_config()
        self.ledger = ledger or default_ledger(self.cfg)
        self.model = model
        self.max_calls = max_calls
        self.max_cost_usd = max_cost_usd
        self.calls = 0
        self.cost_usd = 0.0
        self.est_cost_usd = 0.0
        self.exhausted = None
        self.consecutive_errors = 0
        self.max_consecutive_errors = 3
        # verdict sources: rule (no call), cache, judged (a model call), budget (stopped), error, unparsable
        self.stats = {"rule": 0, "cache": 0, "judged": 0, "budget": 0, "error": 0, "unparsable": 0}
        self.use_cache = use_cache
        self._series_call = False       # set per verdict: the series rubric needs a larger max_tokens
        self._cache = None
        self._cache_path = cache_path or (self.cfg.path("work_root") / "judge" / "verdicts.sqlite")

    @property
    def label(self):
        return "none"

    @property
    def cache(self):
        if self._cache is None:
            self._cache = KVCache(self._cache_path)
        return self._cache

    def cache_key(self, question, record):
        key = {"v": PROMPT_VERSION, "judge": self.label, "q": normalize(question.text(record["q_lang"])),
               "type": question.answer_type, "gold": [normalize(question.answer_fa), normalize(question.answer_en),
                                                      question.answer_value, question.answer_unit],
               "answer": normalize(record.get("answer_text") or "")}
        if question.answer_type == "series":
            spec = _series_spec(question)
            key.update(v=SERIES_PROMPT_VERSION, comparable=spec["comparable"],
                       series=[[it["period"], it.get("value"), it.get("unit")] for it in spec["answer_series"]],
                       restated=[[r["period"], r.get("value")] for r in spec["reconciliation"]["restatements"]],
                       optional=[[s["period"], s.get("value")] for s in spec["reconciliation"]["supplementary"]])
        return sha1_json(key)

    def rule_verdict(self, question, record):
        ans = (record.get("answer_text") or "").strip()
        tok = _abstain_token(self.cfg)
        if record.get("error") or not ans:
            return {"correct": False, "partially_correct": False, "reason": "run error or empty answer", "source": "rule"}
        if re.sub(r"[\s\W_]+", "", re.sub(r"\[\s*\d+\s*\]", "", ans)).upper() == tok.replace("_", "").upper():
            unans = question.answer_type == "unanswerable"
            return {"correct": unans, "partially_correct": False,
                    "reason": "abstained on an %s question" % ("unanswerable" if unans else "answerable"), "source": "rule"}
        return None

    def _base(self, v):
        v = dict(v)
        v.setdefault("judge", self.label)
        v.setdefault("weaker", self.weaker)
        return v

    def verdict(self, question, record):
        if self.provider == "none":
            return None
        rv = self.rule_verdict(question, record)
        if rv is not None:
            self.stats["rule"] += 1
            return self._base(rv)
        key = self.cache_key(question, record)
        if self.use_cache:
            hit = self.cache.get(key)
            if hit is not None:
                self.stats["cache"] += 1
                return self._base(dict(hit, source="cache"))
        if self.exhausted:
            self.stats["budget"] += 1
            return self._base({"correct": None, "partially_correct": None, "reason": self.exhausted, "source": "budget"})
        series = question.answer_type == "series"
        msgs = build_series_judge_messages(question, record) if series else build_judge_messages(question, record)
        self._series_call = series
        why = self._check_budget(msgs)
        if why:
            self.exhausted = why
            self.stats["budget"] += 1
            return self._base({"correct": None, "partially_correct": None, "reason": why, "source": "budget"})
        ctx = {"question_id": record.get("question_id"), "setting": record.get("setting"),
               "judged_approach": record.get("approach")}
        try:
            text, cost = self._call(msgs, ctx)
        except BudgetExceeded as e:
            self.exhausted = "workflow judge budget: %s" % e
            self.stats["budget"] += 1
            return self._base({"correct": None, "partially_correct": None, "reason": self.exhausted, "source": "budget"})
        except (ModelError, OSError, ValueError) as e:
            # a failed OpenRouter request still used one unit of the persisted workflow budget
            # (OpenRouterClient reserves before posting), so it counts toward the per-invocation cap,
            # and a run of failures (bad key, model id, outage) stops judging instead of burning the budget
            self.calls += 1
            self.stats["error"] += 1
            self.consecutive_errors += 1
            if self.consecutive_errors >= self.max_consecutive_errors:
                self.exhausted = "stopped after %d consecutive judge errors (last: %s)" % (
                    self.consecutive_errors, str(e)[:160])
            return self._base({"correct": None, "partially_correct": None, "reason": "judge error: %s" % str(e)[:200],
                               "source": "error"})
        self.calls += 1
        self.consecutive_errors = 0
        self.cost_usd += float(cost or 0.0)
        if series:
            spec = _series_spec(question)
            v = parse_series_verdict(text, [it["period"] for it in spec["answer_series"]], comparable=spec["comparable"])
        else:
            v = parse_verdict(text)
        if v is None:
            self.stats["unparsable"] += 1
            return self._base({"correct": None, "partially_correct": None, "reason": "unparsable judge reply",
                               "source": "error", "raw": (text or "")[:200]})
        v.update(source="judge", cost_usd=cost)
        self.stats["judged"] += 1
        if self.use_cache:
            self.cache.put(key, {k: v[k] for k in ("correct", "partially_correct", "reason", "strict_json", "series") if k in v})
        return self._base(v)

    def _check_budget(self, msgs):
        if self.max_calls is not None and self.calls >= self.max_calls:
            return "per-invocation judge call cap reached (%d)" % self.max_calls
        return None

    def _call(self, msgs, ctx):
        raise NotImplementedError

    def summary(self):
        return {"judge": self.label, "provider": self.provider, "weaker": self.weaker, "calls": self.calls,
                "cost_usd": round(self.cost_usd, 6), "est_cost_usd": round(self.est_cost_usd, 6),
                "max_calls": self.max_calls, "max_cost_usd": self.max_cost_usd, "exhausted": self.exhausted,
                **self.stats}


class NoJudge(Judge):
    provider = "none"


class OpenRouterJudge(Judge):
    provider = "openrouter"

    def __init__(self, cfg=None, ledger=None, model=None, max_calls=None, max_cost_usd=None, max_tokens=DEFAULT_MAX_TOKENS,
                 **kw):
        super().__init__(cfg, ledger, model, max_calls, max_cost_usd, **kw)
        self.model = model or self.cfg.judge["model"]
        self.max_tokens = int(max_tokens)
        self.client = OpenRouterClient(self.cfg, self.ledger)

    @property
    def label(self):
        return "openrouter:%s" % self.model

    def _call_max_tokens(self):
        return max(self.max_tokens, SERIES_MAX_TOKENS) if self._series_call else self.max_tokens

    def estimate_cost(self, msgs):
        pin, pout = PRICES.get(self.model, UNKNOWN_PRICE)
        chars = sum(len(m["content"]) for m in msgs)
        return (chars / 3.0) * pin / 1e6 + self._call_max_tokens() * pout / 1e6

    def _check_budget(self, msgs):
        why = super()._check_budget(msgs)
        if why:
            return why
        st = self.client.budget_state()
        if st["calls"] >= st["max_calls_total"] or st["cost_usd"] >= st["max_cost_usd"]:
            return "workflow judge budget used up (%d/%d calls, $%.4f/$%.2f)" % (
                st["calls"], st["max_calls_total"], st["cost_usd"], st["max_cost_usd"])
        est = self.estimate_cost(msgs)
        spent = max(self.cost_usd, self.est_cost_usd)
        if self.max_cost_usd is not None and spent + est > self.max_cost_usd:
            return "per-invocation estimated cost cap reached ($%.5f + $%.5f > $%.4f)" % (spent, est, self.max_cost_usd)
        self.est_cost_usd += est
        return None

    def _call(self, msgs, ctx):
        r = self.client.chat(msgs, purpose="judge_series" if self._series_call else "judge", model=self.model,
                             max_tokens=self._call_max_tokens(), **ctx)
        return r.text, r.cost_usd


class OllamaJudge(Judge):
    """Local fallback judge. WEAKER than a hosted judge; with the default model it is the
    same model that produced the answers (self-grading bias). Labelled in every output."""
    provider = "ollama"
    weaker = True
    SCHEMA = {"type": "object", "properties": {"correct": {"type": "boolean"},
                                               "partially_correct": {"type": "boolean"},
                                               "reason": {"type": "string"}},
              "required": ["correct", "partially_correct", "reason"]}

    def __init__(self, cfg=None, ledger=None, model=None, max_calls=None, max_cost_usd=None, client=None, **kw):
        super().__init__(cfg, ledger, model, max_calls, max_cost_usd, **kw)
        self.model = model or self.cfg.models["chat"]
        self.client = client or OllamaClient(self.cfg, self.ledger)

    @property
    def label(self):
        same = " same-model-as-answerer" if self.model == self.cfg.models["chat"] else ""
        return "ollama:%s (WEAKER local fallback%s)" % (self.model, same)

    SERIES_SCHEMA = {"type": "object", "properties": {"period_correct": {"type": "object"},
                                                      "flags_definition_change": {"type": "boolean"},
                                                      "false_trend": {"type": "boolean"}, "correct": {"type": "boolean"},
                                                      "partially_correct": {"type": "boolean"}, "reason": {"type": "string"}},
                     "required": ["period_correct", "flags_definition_change", "false_trend", "correct",
                                  "partially_correct", "reason"]}

    def _call(self, msgs, ctx):
        series = self._series_call
        r = self.client.chat(msgs, purpose="judge_series" if series else "judge", phase="eval", approach="judge",
                             model=self.model, num_predict=SERIES_MAX_TOKENS if series else DEFAULT_MAX_TOKENS,
                             format=self.SERIES_SCHEMA if series else self.SCHEMA, **ctx)
        return r.text, 0.0


def make_judge(kind, cfg=None, ledger=None, model=None, max_calls=None, max_cost_usd=None, **kw):
    kind = (kind or "none").lower()
    if kind == "none":
        return NoJudge(cfg, ledger)
    if kind == "openrouter":
        return OpenRouterJudge(cfg, ledger, model=model, max_calls=max_calls, max_cost_usd=max_cost_usd, **kw)
    if kind == "ollama":
        return OllamaJudge(cfg, ledger, model=model, max_calls=max_calls, max_cost_usd=max_cost_usd, **kw)
    raise ValueError("judge must be none|openrouter|ollama, got %r" % kind)


# --------------------------------------------------------------------------
# CLI: --probe makes ONE judge call on a built-in example (auth + JSON shape check)
# --------------------------------------------------------------------------

def _probe(args):
    from ..contracts import Question
    cfg = load_config()
    q = Question(id="probe-sample", corpus="A", question_fa="در سال ۲۰۲۴ چند تن محصول نمونه تولید شد؟",
                 question_en="How many tonnes of the sample product were made in 2024?",
                 answer_fa="۲۶۴٬۹۲۶ تن", answer_en="264,926 t", answer_type="numeric",
                 gold_units={"fa": [], "en": ["A.rpt-a.en::x"]}, doc_scope=["A.rpt-a.en"], type="lookup",
                 answer_value=264926, answer_unit="t")
    rec = {"question_id": q.id, "approach": "PROBE", "setting": "en-en", "q_lang": "en", "d_lang": "en",
           "answer_text": "264,926 t of the sample product were made in 2024 [2].", "error": None}
    ledger = default_ledger(cfg)
    j = make_judge(args.judge, cfg, ledger, model=args.judge_model, max_calls=1, max_cost_usd=args.judge_max_cost,
                   use_cache=False)
    if args.judge == "openrouter":
        print("budget before:", j.client.budget_state())
    t0 = time.time()
    v = j.verdict(q, rec)
    print("verdict:", {k: v.get(k) for k in ("correct", "partially_correct", "reason", "source", "strict_json", "cost_usd", "judge")})
    print("summary:", j.summary(), "%.1fs" % (time.time() - t0))
    if args.judge == "openrouter":
        print("budget after:", j.client.budget_state())
    return 0 if v and v.get("source") == "judge" and v.get("correct") is True else 1


def series_probe_example():
    """A built-in non-comparable series question and a model answer that states a false trend (for --probe-series and
    the offline tests)."""
    from ..contracts import Question
    series = [
        {"period": "1398", "value": 35, "unit": "million users", "value_base": 35e6, "percent": False,
         "definition_en": "active users", "doc": "C.divar_annual-report_1398.fa", "gold_units": {"fa": ["C.divar_annual-report_1398.fa::p5_g1"]}},
        {"period": "1399", "value": 44, "unit": "million phones", "value_base": 44e6, "percent": False,
         "definition_en": "unique phones that opened the app", "doc": "C.divar_annual-report_1399.fa",
         "gold_units": {"fa": ["C.divar_annual-report_1399.fa::p7_b6"]}},
        {"period": "1400", "value": 53.1, "unit": "million phones", "value_base": 53.1e6, "percent": False,
         "definition_en": "unique phones that opened the app", "doc": "C.divar_annual-report_1400.fa",
         "gold_units": {"fa": ["C.divar_annual-report_1400.fa::p5_g1"]}},
    ]
    q = Question(id="probe-series", corpus="C", question_fa="شمار کاربران دیوار از 1398 تا 1400 چگونه تغییر کرد؟",
                 question_en="How did the number of Divar users change from 1398 to 1400?",
                 answer_fa="1398: 35 میلیون کاربر؛ 1399: 44 میلیون گوشی؛ 1400: 53.1 میلیون گوشی (تعریف از کاربر به گوشی تغییر کرده).",
                 answer_en="1398: 35 million users; 1399: 44 million phones; 1400: 53.1 million phones (the count changes from users to phones).",
                 answer_type="series", gold_units={"fa": [s["gold_units"]["fa"][0] for s in series]},
                 doc_scope=[s["doc"] for s in series], type="timeline",
                 meta={"answer_series": series, "comparable": False,
                       "series_notes_en": "1398 counts active users; 1399 and 1400 count unique phones that opened the app.",
                       "reconciliation": {"rule": "report_of_period",
                                          "restatements": [{"period": "1399", "value": 44.1, "unit": "million phones",
                                                            "doc": "C.divar_annual-report_1400.fa"}]}})
    rec = {"question_id": q.id, "approach": "PROBE", "setting": "en-fa", "q_lang": "en", "d_lang": "fa",
           "answer_text": "Divar grew steadily: 35 million users in 1398, 44 million in 1399 and 53.1 million in 1400.",
           "error": None}
    return q, rec


def _probe_series(args):
    cfg = load_config()
    q, rec = series_probe_example()
    ledger = default_ledger(cfg)
    j = make_judge(args.judge, cfg, ledger, model=args.judge_model, max_calls=1, max_cost_usd=args.judge_max_cost,
                   use_cache=False)
    if args.judge == "openrouter":
        print("budget before:", j.client.budget_state())
    t0 = time.time()
    v = j.verdict(q, rec)
    print("verdict:", {k: v.get(k) for k in ("correct", "partially_correct", "reason", "source", "strict_json", "cost_usd",
                                             "series", "judge")})
    print("summary:", j.summary(), "%.1fs" % (time.time() - t0))
    if args.judge == "openrouter":
        print("budget after:", j.client.budget_state())
    s = (v or {}).get("series") or {}
    # expected: every period right, no flag, a false trend, so not correct
    ok = v and v.get("source") == "judge" and s.get("coverage") == 1.0 and s.get("false_trend") is True and v.get("correct") is False
    return 0 if ok else 1


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--probe", action="store_true", help="one judge call on a built-in example")
    ap.add_argument("--probe-series", action="store_true", help="one judge call on a built-in timeline (series) example")
    ap.add_argument("--judge", default="openrouter", choices=["openrouter", "ollama"])
    ap.add_argument("--judge-model", default=None)
    ap.add_argument("--judge-max-cost", type=float, default=0.01)
    args = ap.parse_args(argv)
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass
    if args.probe:
        return _probe(args)
    if args.probe_series:
        return _probe_series(args)
    ap.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
