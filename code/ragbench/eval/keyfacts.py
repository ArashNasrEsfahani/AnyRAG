"""Key-fact scoring: does an answer CONTAIN the few facts a correct answer must contain?

Why it exists: the strict matcher of eval/metrics.py compares an answer with the gold answer
(exact match, gold-token containment, numeric equality). That works for a lookup, whose gold
answer is one number or one short string. It cannot score a locate or multi-section question,
whose gold answer is a sentence: a correct answer says the same things in other words and the
strict matcher returns 0. On the core test B1 (which is handed the gold evidence) scored
locate 0/39 and multi-section 0/55 at token overlap 0.46 / 0.52 -- the answers were not wrong,
they were unscorable.

A key fact is the smallest unit a correct answer cannot omit: a number it must quote, or a
section title / term it must name. They live in `question.meta.key_facts` (meta is never given
to an approach and is excluded from run.py's question_sig, so adding them invalidates no run).

Format of one entry in meta.key_facts:

    {"kind": "number", "value": 4400000, "unit": "rial", "aliases": ["4.4 million"], "why": "..."}
    {"kind": "string", "value": "supply chain", "aliases": ["زنجیره تامین"],           "why": "..."}

  kind     "number" | "string"
  value    the float to match (number), or the surface form to find (string)
  unit     optional, informational only; the numeric match is on readings, not on units
  aliases  other acceptable surface forms, in either language. A string key fact is found when
           value OR any alias is contained; a number key fact may also be found through an
           alias when the answer writes it as a word ("چهار و چهار دهم میلیون").
  why      a short note for the human reading the question file; never used in scoring.

Matching rules:
  number   through ragbench.approaches.agentic.numbers.find_numbers, comparing READINGS, so
           "4.4 million" == "4,400,000" == "4.4 میلیون" == "۴٬۴۰۰٬۰۰۰" all match 4400000.
           A reading counts as equal within a relative tolerance (1e-9 by default, i.e. exact);
           the token's scaled and unscaled readings are both tried, as are the glitch readings
           of a right-to-left extraction artefact.
  string   containment on eval/metrics.py::norm_text (fold, citation markers and thousands
           separators removed, punctuation to space), tried BOTH WAYS: the answer contains the
           key fact, or the key fact contains the answer (a one-word answer "Logistics" satisfies
           the key fact "Logistics chapter"). Persian ZWNJ and space are the same character here,
           so "زنجیره‌تامین", "زنجیره تامین" and "زنجیرهتامین" all match.

Output of score_key_facts:

    {"kf_total": int, "kf_found": int, "kf_recall": float, "kf_correct": 0.0 | 1.0,
     "kf_missing": [values of the key facts not found]}

kf_correct is 1.0 only when every key fact is found. A question with no key facts scores
{"kf_total": 0, ...} and kf_correct None, so aggregation can leave it to the strict matcher.

Honest limit (repeated in work/analysis/SCORING.md): this scorer rewards CONTAINMENT. A long,
mostly wrong answer that happens to mention every key fact passes. It measures whether the
right material reached the answer -- which is what a retrieval study wants -- not whether the
sentence around it is true.
"""
import re

from ..approaches.agentic import numbers as N
from ..normalize import normalize
from .metrics import norm_text

NUM_REL_TOL = 1e-9
MIN_REVERSE_CHARS = 3      # the shortest answer that may count as naming a longer key fact


def key_facts_of(question):
    """The key_facts list of a Question or question dict, or [] when it has none."""
    meta = getattr(question, "meta", None)
    if meta is None and isinstance(question, dict):
        meta = question.get("meta")
    kf = (meta or {}).get("key_facts") or []
    return [k for k in kf if isinstance(k, dict) and k.get("kind") in ("number", "string")]


def _fold_ws(text):
    """The matching form of eval/metrics.py (fold, citation markers and thousands separators
    removed, punctuation to space), so a key fact and an answer are compared exactly as the
    strict string matcher compares two answers."""
    return norm_text(text or "")


def _found_string(answer_folded, fact):
    if not answer_folded:
        return False                       # an empty answer contains nothing; "" is a substring of every key fact
    for surface in [fact.get("value")] + list(fact.get("aliases") or []):
        if not isinstance(surface, str) or not surface.strip():
            continue
        s = _fold_ws(surface)
        if not s:
            continue
        if s in answer_folded:
            return True
        # a shorter answer may still name the fact ("trade" for "market > trade"), but a token that short is
        # evidence of nothing, so the reverse direction needs a real answer behind it
        if len(answer_folded) >= MIN_REVERSE_CHARS and answer_folded in s:
            return True
        # Persian writes one word with a space, a ZWNJ or nothing at all; compare unspaced too
        if s.replace(" ", "") in answer_folded.replace(" ", ""):
            return True
    return False


# "308 K", "4.4 M": a one-letter scale after a space. The shared reader (also the approaches' verbatim guard, so it is
# not changed here) reads "308K" and "308 thousand" but not the spaced letter; the scorer joins it before reading.
# Dev, 2026-09-21: «308 K Sellers» scored as missing 308,000. Both readings stay (308 and 308,000).
_SPACED_SCALE = re.compile(r"(\d)\s+([kKmMbB])(?![A-Za-z])")


def _readings(text):
    out = []
    for tok in N.find_numbers(_SPACED_SCALE.sub(r"\1\2", normalize(text or ""))):
        out += list(tok.readings)
        if tok.reversed_reading is not None:
            out.append(tok.reversed_reading)
    return out


def _found_number(answer_readings, answer_folded, fact):
    try:
        target = float(fact.get("value"))
    except (TypeError, ValueError):
        return False
    for r in answer_readings:
        if abs(r - target) <= NUM_REL_TOL * max(1.0, abs(r), abs(target)):
            return True
    # a number written out in words ("four and a half million", "چهار و نیم میلیون")
    return _found_string(answer_folded, {"value": None, "aliases": fact.get("aliases") or []})


def score_key_facts(question, answer_text, q_lang=None):
    """Key-fact scores for one answer. q_lang is accepted for symmetry with the other scorers
    and is not used: key facts carry surface forms in both languages, because a correct Persian
    answer may quote an English section title and the other way round."""
    facts = key_facts_of(question)
    if not facts:
        return {"kf_total": 0, "kf_found": 0, "kf_recall": None, "kf_correct": None, "kf_missing": []}
    folded = _fold_ws(answer_text)
    readings = _readings(answer_text)
    found, missing = 0, []
    for f in facts:
        ok = _found_number(readings, folded, f) if f["kind"] == "number" else _found_string(folded, f)
        if ok:
            found += 1
        else:
            missing.append(f.get("value"))
    return {"kf_total": len(facts), "kf_found": found, "kf_recall": found / len(facts),
            "kf_correct": float(found == len(facts)), "kf_missing": missing}
