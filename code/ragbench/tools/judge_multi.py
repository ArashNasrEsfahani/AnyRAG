"""Adjudication for the multi-document experiment (2026-09-24, P0 list; design: work/analysis/MULTIDOC_ASTRA_session.log
Q3). judge_final.py (judge-final-v3) stays frozen with the finished study; this tool IMPORTS its rule, prompt and parser
unchanged for single-fact questions and adds a content-unit rubric for multi-part answers.

    .venv/Scripts/python.exe -m ragbench.tools.judge_multi --judge <openrouter model id> --runs <run dir> [--runs ...]
        --questions <multidoc-q-v1 file> [--out work/multidoc/judge] [--methods M1,M2] [--retry-failures]

Records: only identity records (meta.identity from ragbench.run identity runs); the latest line of each rid wins.
Verdicts are keyed by the record identity (experiment, question, pool, method, options, reader, replicate, setting =
rid) and bound to the exact answer (answer_sha1) and rubric (rubric_sha1; for the multi rubric it covers the reference
packet too). A verdict counts as done only for the same rid, answer_sha1, rubric_sha1 and judge; a changed answer or
rubric is judged again.

Rubrics
  judge-final-v3 (single facts: answer format single/default and at most one content unit, or families single_fact,
  routing, unanswerable) - judge_final.rule / messages / parse, unchanged; strict = recall = precision = F1 = correct.
  judge-multi-v1 (series, comparison, list, multi-unit answers): the judge sees the question, a METHOD-INDEPENDENT
  reference packet (content units with claim ids as tuples (company, period, metric, definition, value, unit), their
  source excerpts, accepted alternatives, definitions, calculation rules) and the candidate answer - never the method,
  its evidence or its trace. It must return every claim of the answer with a status and every unit id with a status;
  the metrics are computed here, deterministically:
    strict      every required unit correct and no contradiction, wrong value/binding or unsupported addition
    recall      correct units / required units (a right number on the wrong year/company earns nothing)
    precision   correct claims / distinct substantive claims (duplicates and non-substantive text excluded)
    F1          2PR/(P+R); omissions, contradictions, unsupported additions, wrong values/bindings and duplicates are
                reported separately; operands and derived comparisons are scored separately (by unit role)
  Rules before any call (both rubrics): run error, empty or truncated answer -> all 0; the abstain token -> correct
  abstention iff the question is unanswerable (false_abstention flagged otherwise).
Judge failures (a call error, an unparsable or incomplete reply) are NOT verdicts. Since 2026-10-01 (FREEZE_ASTRA_R1):
  - a record gets up to --max-judge-attempts (3) IDENTICAL calls in one run; the system answer is never regenerated;
  - no parsable reply but at least one reply -> needs_adjudication_<judge>.jsonl (terminal: never re-called, never
    scored 0; blinded adjudication, else the score bounds [0, 1] - see final_verdicts);
  - only call errors (no reply at all) -> judge_failures_<judge>.jsonl, retried only with --retry-failures;
  - --judge-provider pins ONE endpoint (allow_fallbacks false); the pin is part of every verdict, its key and the
    cache key; the served provider is recorded; judge_runs.jsonl logs each run's judge configuration.
Per-question averaging across settings/replicates happens in multidoc_analysis.py.
"""
import argparse
import datetime as dt
import glob
import json
import re
import sys
from pathlib import Path

from ..contracts import TASK_FAMILY_FORMATS, Question, answer_format
from ..llm import BudgetExceeded
from ..util import KVCache, sha1_json, sha1_text
from . import judge_final as JF

MULTI_VERSION = "judge-multi-v1"
SINGLE_VERSION = JF.PROMPT_VERSION               # judge-final-v3, unchanged
SINGLE_FAMILIES = {"single_fact", "routing", "unanswerable"}
CLAIM_STATUSES = ("correct", "wrong_value", "wrong_binding", "contradiction", "unsupported", "duplicate",
                  "non_substantive")
UNIT_STATUSES = ("correct", "missing", "wrong", "contradicted")

MULTI_SYSTEM = (
    "You are a strict, impartial grader for a question-answering benchmark over company reports written in Persian "
    "and English. You compare a candidate answer with a reference packet built from the reports: required content "
    "units (each a claim with an id), the source excerpts behind them, accepted alternative wordings, definitions and "
    "calculation rules. The packet is the truth; do not use outside knowledge. Reply with one JSON object only.")

MULTI_USER = """Question ({lang}): {question}
Answer format: {fmt}

Reference packet:
{packet}

Candidate answer:
{answer}

Grade in two lists.
1. "claims": split the candidate into its distinct factual claims. For each give {{"text": <the claim, short>, "unit":
   <the claim id it expresses, or null>, "status": one of correct | wrong_value | wrong_binding | contradiction |
   unsupported | duplicate | non_substantive}}.
   correct: states a required unit's value (rounding at the unit's precision is fine; Persian decimal slash 78/3 =
   78.3, Persian digits and scale words are fine) bound to the SAME company, period, metric and definition.
   wrong_value: the right company/period/metric with a different value. wrong_binding: a right value attached to the
   wrong company, period or metric. contradiction: conflicts with a unit or with another claim of the answer.
   unsupported: a substantive claim no unit or excerpt supports. duplicate: repeats an earlier claim.
   non_substantive: framing, hedging or citation markers such as [2].
2. "units": for EVERY claim id in the packet give one of correct | missing | wrong | contradicted.
A derived unit (a comparison, difference or ranking) is correct only if the candidate states it; its operands are
scored as their own units. For a list, each item is a unit.
Reply with JSON only: {{"claims": [...], "units": {{"<claim id>": "<status>", ...}}, "reason": "at most 25 words"}}"""

UNITS_KEYS = ("claim_id", "company", "period", "metric", "definition", "value", "unit", "role")


def rubric_sha1(kind, packet_sha1=None):
    if kind == "single":
        return sha1_json({"v": SINGLE_VERSION, "s": JF.JUDGE_V2_SYSTEM, "u": JF.JUDGE_V2_USER})[:16]
    return sha1_json({"v": MULTI_VERSION, "s": MULTI_SYSTEM, "u": MULTI_USER, "packet": packet_sha1,
                      "claims": CLAIM_STATUSES, "units": UNIT_STATUSES})[:16]


def answer_sha1(rec):
    return sha1_text(rec.get("answer_text") or "")[:16]


def rubric_kind(qd):
    fam = qd.get("task_family")
    fmt = qd.get("answer_format") or TASK_FAMILY_FORMATS.get(fam) or "default"
    if fam in SINGLE_FAMILIES:
        return "single"
    if fmt in ("single", "default") and len(qd.get("content_units") or []) <= 1:
        return "single"
    return "multi"


def reference_packet(qd, lang, corpus=None):
    """The method-independent reference for one question (from the question file only; corpus fills excerpts that
    carry no quote)."""
    alts = qd.get("accepted_alternatives") or {}
    units = []
    for u in qd.get("content_units") or []:
        ex = []
        for sp in u.get("source_spans") or []:
            t = sp.get("quote")
            if t is None and corpus is not None and sp.get("doc_id") and sp.get("start") is not None:
                try:
                    doc = corpus.doc(sp["doc_id"]) if hasattr(corpus, "doc") else corpus[sp["doc_id"]]
                    t = doc.text[int(sp["start"]):int(sp["end"])]
                except (KeyError, ValueError):
                    t = None
            if t:
                ex.append({"doc_id": sp.get("doc_id"), "excerpt": t})
        units.append(dict({k: u.get(k) for k in UNITS_KEYS}, accepted_alternatives=list(alts.get(u.get("claim_id")) or []),
                          excerpts=ex))
    return {"question_id": qd.get("id"), "task_family": qd.get("task_family"),
            "answer_format": qd.get("answer_format") or answer_format(Question.from_dict(qd)), "lang": lang,
            "reference_answer": (qd.get("answer_%s" % lang) or qd.get("answer_en") or qd.get("answer_fa") or ""),
            "units": units, "definitions": qd.get("definitions") or {},
            "calculation_rules": qd.get("calculation_rules") or ""}


def multi_messages(qd, rec, packet):
    lang = rec["setting"][:2]
    q = Question.from_dict(qd)
    user = MULTI_USER.format(lang=JF.LANG.get(lang, lang), question=(q.text(lang) or "").strip(),
                             fmt=packet["answer_format"], packet=json.dumps(packet, ensure_ascii=False, indent=1),
                             answer=(rec.get("answer_text") or "").strip())
    return [{"role": "system", "content": MULTI_SYSTEM}, {"role": "user", "content": user}]


def parse_multi(text, unit_ids, statuses=CLAIM_STATUSES):
    """-> (parsed dict, None) or (None, why). Every unit id must be graded with a known status; claims must carry
    known statuses and unit ids from the packet (or null)."""
    t = (text or "").strip()
    m = re.search(r"\{.*\}", t, re.S)
    try:
        obj = json.loads(m.group(0) if m else t)
    except (ValueError, AttributeError):
        return None, "unparsable reply"
    if not isinstance(obj, dict) or not isinstance(obj.get("claims"), list) or not isinstance(obj.get("units"), dict):
        return None, "reply lacks claims/units"
    units = {str(k): str(v).strip().lower() for k, v in obj["units"].items()}
    missing = [u for u in unit_ids if u not in units]
    if missing:
        return None, "reply omits unit(s) %s" % missing[:5]
    extra = [u for u in units if u not in unit_ids]
    if extra:
        return None, "reply grades unknown unit(s) %s" % extra[:5]
    bad = {u: s for u, s in units.items() if s not in UNIT_STATUSES}
    if bad:
        return None, "unknown unit status %s" % bad
    claims = []
    for c in obj["claims"]:
        if not isinstance(c, dict):
            return None, "claim is not an object"
        st = str(c.get("status") or "").strip().lower()
        if st not in statuses:
            return None, "unknown claim status %r" % st
        u = c.get("unit")
        if u is not None and str(u) not in unit_ids:
            return None, "claim cites unknown unit %r" % u
        claims.append({"text": str(c.get("text") or "")[:200], "unit": None if u is None else str(u), "status": st})
    return {"claims": claims, "units": units, "reason": str(obj.get("reason") or "")[:300]}, None


def score_multi(parsed, packet):
    """Deterministic metrics from a parsed multi-rubric reply."""
    units, claims = parsed["units"], parsed["claims"]
    req = [u["claim_id"] for u in packet["units"]]
    role = {u["claim_id"]: (u.get("role") or "fact") for u in packet["units"]}
    correct_units = [u for u in req if units.get(u) == "correct"]
    subst = [c for c in claims if c["status"] not in ("duplicate", "non_substantive")]
    ok_claims = [c for c in subst if c["status"] == "correct"]
    recall = len(correct_units) / len(req) if req else 0.0
    precision = len(ok_claims) / len(subst) if subst else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    n = lambda st: sum(1 for c in claims if c["status"] == st)   # noqa: E731
    contradictions = n("contradiction") + sum(1 for u in req if units.get(u) == "contradicted")
    wrong = n("wrong_value") + n("wrong_binding")
    unsupported = n("unsupported")
    by_role = {}
    for r in sorted(set(role.values())):
        ids = [u for u in req if role[u] == r]
        by_role[r] = round(sum(1 for u in ids if units.get(u) == "correct") / len(ids), 4)
    strict = int(len(correct_units) == len(req) and bool(req) and contradictions == 0 and wrong == 0 and unsupported == 0)
    return {"strict": strict, "recall": round(recall, 4), "precision": round(precision, 4), "f1": round(f1, 4),
            "omissions": sum(1 for u in req if units.get(u) == "missing"), "contradictions": contradictions,
            "unsupported": unsupported, "wrong": wrong, "duplicates": n("duplicate"), "recall_by_role": by_role,
            "n_units": len(req), "n_claims": len(subst)}


# ---------------------------------------------------------------------------------------------------------------------
# datasets v2 (2026-09-25): questions converted by tools/verify_v2run.py (schema question-v2-run, the full source record
# in meta.v2). Three domains (reports, books, laws), so the rubric is domain-neutral: units are the source's
# requirements (subject; aspect; constraints), each marked required or optional. Harmless extra detail the packet does
# not cover is "extra" (it never fails an answer); "unsupported" is reserved for a claim that changes the answer.
#   strict     every REQUIRED requirement correct and no contradiction, wrong value/binding or unsupported claim
#   recall     correct required requirements / required requirements
#   precision  correct claims / (correct + wrong + contradiction + unsupported claims)
# Unanswerable v2 questions: the abstain token is ruled correct before any call; any other answer is judged with
# V2_UNANS_* (correct only if it declines to give the missing fact).
V2_RUN_SCHEMA = "question-v2-run"
V2_MULTI_VERSION = "judge-v2-multi-v1"
V2_UNANS_VERSION = "judge-v2-unans-v1"
V2_CLAIM_STATUSES = ("correct", "wrong_value", "wrong_binding", "contradiction", "unsupported", "extra", "duplicate",
                     "non_substantive")

V2_MULTI_SYSTEM = (
    "You are a strict, impartial grader for a question-answering benchmark over documents (company reports, "
    "non-fiction books and laws) written in Persian and English. You compare a candidate answer with a reference "
    "packet built from the documents: the reference answer, the answer requirements (each with an id, marked required "
    "or optional), the source excerpts behind them and accepted alternatives. The packet is the truth; do not use "
    "outside knowledge. Reply with one JSON object only.")

V2_MULTI_USER = """Question ({lang}): {question}
Answer format: {fmt}

Reference packet:
{packet}

Candidate answer:
{answer}

Grade in two lists.
1. "claims": split the candidate into its distinct claims. For each give {{"text": <the claim, short>, "unit":
   <the requirement id it expresses, or null>, "status": one of correct | wrong_value | wrong_binding | contradiction |
   unsupported | extra | duplicate | non_substantive}}.
   correct: states a requirement with the same meaning, bound to the same subject, aspect and constraints (time,
   version, place, population, condition). Paraphrase, either language, rounding at the requirement's precision,
   Persian digits, the Persian decimal slash (78/3 = 78.3) and scale words are all fine.
   wrong_value: the right subject and aspect with a different value or statement. wrong_binding: a right value or
   statement attached to the wrong subject, time, version or condition. contradiction: conflicts with a requirement,
   an excerpt or another claim of the answer. unsupported: a substantive claim not in the packet that changes the
   answer (a different conclusion, a wrong reason, rule, exception or trend). extra: harmless additional detail not in
   the packet that does not change the answer. duplicate: repeats an earlier claim. non_substantive: framing, hedging or
   citation markers such as [2].
2. "units": for EVERY requirement id in the packet give one of correct | missing | wrong | contradicted.
A derived requirement (a comparison, difference, ranking or conclusion) is correct only if the candidate states it;
its inputs are graded as their own requirements. For a list, each required item is its own requirement.
Reply with JSON only: {{"claims": [...], "units": {{"<requirement id>": "<status>", ...}}, "reason": "at most 25 words"}}"""

V2_UNANS_SYSTEM = (
    "You are a strict, impartial grader for a question-answering benchmark over documents (company reports, "
    "non-fiction books and laws) written in Persian and English. The question asks for something the documents do "
    "not contain. Reply with one JSON object only.")

V2_UNANS_USER = """Question ({lang}): {question}
What the documents lack: {missing}
Nearby facts that are NOT the answer: {near}

Candidate answer:
{answer}

The candidate is CORRECT only if it declines to give the missing fact (for example NOT_FOUND, "the documents do not
say", or an explanation that the value is not given), possibly while mentioning the nearby facts as such. It is
INCORRECT if it states a value, date, name or conclusion for the missing fact, including one taken from a nearby fact.
Reply with JSON only: {{"correct": true or false, "reason": "at most 20 words"}}"""


def is_v2(qd):
    return qd.get("schema") == V2_RUN_SCHEMA


def _v2_constraints(c):
    out = {}
    for k, v in (c or {}).items():
        if v in (None, "", {}, []):
            continue
        out[k] = ("%s (%s)" % (v.get("value"), v.get("kind"))) if isinstance(v, dict) else v
    return out


def reference_packet_v2(qd, lang):
    """The method-independent reference of a v2 question: requirements (required and optional), their excerpts and
    accepted alternatives, from meta.v2 only."""
    r = (qd.get("meta") or {}).get("v2") or {}
    ev = {e["id"]: e for e in r.get("evidence") or []}
    alts = r.get("accepted_alternatives") or {}
    units = []
    for q in r.get("requirements") or []:
        exs = [{"doc": ev[e].get("doc_title") or ev[e]["doc_id"], "role": ev[e].get("role"), "excerpt": ev[e]["quote"]}
               for e in ev if q["id"] in (ev[e].get("supports") or [])]
        der = q.get("derivation") or None
        asp = q.get("aspect") or {}
        units.append({"claim_id": q["id"], "required": bool(q.get("required")), "role": q.get("kind"),
                      "proposition": q.get("proposition"), "subject": q.get("subject"),
                      "aspect": asp.get("label") + ((" -> %s" % asp["object"]) if asp.get("object") else ""),
                      "constraints": _v2_constraints(q.get("constraints")), "value": q.get("value"),
                      "unit": q.get("unit"), "comparator": q.get("comparator"), "precision": q.get("precision"),
                      "accepted_alternatives": list(q.get("aliases") or []) + list(alts.get(q["id"]) or []),
                      "derivation": ({"operation": der.get("operation"), "inputs": der.get("inputs")} if der else None),
                      "excerpts": exs})
    return {"question_id": qd.get("id"), "tier": r.get("tier"), "answer_format": r.get("answer_format"),
            "integration_operation": r.get("integration_operation"), "lang": lang,
            "reference_answer": r.get("expected_answer") or "", "units": units}


def v2_multi_messages(qd, rec, packet):
    lang = rec["setting"][:2]
    q = Question.from_dict(qd)
    user = V2_MULTI_USER.format(lang=JF.LANG.get(lang, lang), question=(q.text(lang) or "").strip(),
                                fmt=packet["answer_format"], packet=json.dumps(packet, ensure_ascii=False, indent=1),
                                answer=(rec.get("answer_text") or "").strip())
    return [{"role": "system", "content": V2_MULTI_SYSTEM}, {"role": "user", "content": user}]


def v2_unans_messages(qd, rec):
    lang = rec["setting"][:2]
    q = Question.from_dict(qd)
    a = ((qd.get("meta") or {}).get("v2") or {}).get("absence_audit") or {}
    user = V2_UNANS_USER.format(lang=JF.LANG.get(lang, lang), question=(q.text(lang) or "").strip(),
                                missing=a.get("missing_obligation") or "(see question)",
                                near="; ".join(a.get("neighbouring_facts") or []) or "(none recorded)",
                                answer=(rec.get("answer_text") or "").strip())
    return [{"role": "system", "content": V2_UNANS_SYSTEM}, {"role": "user", "content": user}]


def score_v2(parsed, packet):
    units, claims = parsed["units"], parsed["claims"]
    req = [u["claim_id"] for u in packet["units"] if u.get("required")]
    opt = [u["claim_id"] for u in packet["units"] if not u.get("required")]
    role = {u["claim_id"]: (u.get("role") or "fact") for u in packet["units"]}
    correct_req = [u for u in req if units.get(u) == "correct"]
    n = lambda st: sum(1 for c in claims if c["status"] == st)   # noqa: E731
    scored = [c for c in claims if c["status"] in ("correct", "wrong_value", "wrong_binding", "contradiction",
                                                   "unsupported")]
    recall = len(correct_req) / len(req) if req else 0.0
    precision = n("correct") / len(scored) if scored else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    contradictions = n("contradiction") + sum(1 for u in req + opt if units.get(u) == "contradicted")
    wrong = n("wrong_value") + n("wrong_binding")
    by_role = {}
    for rl in sorted({role[u] for u in req}):
        ids = [u for u in req if role[u] == rl]
        by_role[rl] = round(sum(1 for u in ids if units.get(u) == "correct") / len(ids), 4)
    strict = int(bool(req) and len(correct_req) == len(req) and contradictions == 0 and wrong == 0
                 and n("unsupported") == 0)
    return {"strict": strict, "recall": round(recall, 4), "precision": round(precision, 4), "f1": round(f1, 4),
            "omissions": sum(1 for u in req if units.get(u) == "missing"), "contradictions": contradictions,
            "unsupported": n("unsupported"), "extra": n("extra"), "wrong": wrong, "duplicates": n("duplicate"),
            "optional_correct": sum(1 for u in opt if units.get(u) == "correct"), "recall_by_role": by_role,
            "n_units": len(req), "n_optional": len(opt), "n_claims": len(scored)}


def question_rubric(qd, lang, corpus=None):
    """-> (kind, packet or None, rubric_sha1, rubric version) for a question dict in a language; the one place judging
    and check_grid agree on a question's rubric. kind: single | multi | v2 | v2_unans."""
    if is_v2(qd):
        if qd.get("answer_type") == "unanswerable":
            return "v2_unans", None, sha1_json({"v": V2_UNANS_VERSION, "s": V2_UNANS_SYSTEM,
                                                "u": V2_UNANS_USER})[:16], V2_UNANS_VERSION
        packet = reference_packet_v2(qd, lang)
        psha = sha1_json(packet)[:16]
        return "v2", packet, sha1_json({"v": V2_MULTI_VERSION, "s": V2_MULTI_SYSTEM, "u": V2_MULTI_USER, "packet": psha,
                                        "claims": V2_CLAIM_STATUSES, "units": UNIT_STATUSES})[:16], V2_MULTI_VERSION
    kind = rubric_kind(qd)
    packet = reference_packet(qd, lang, corpus) if kind == "multi" else None
    psha = sha1_json(packet)[:16] if packet is not None else None
    return kind, packet, rubric_sha1(kind, psha), (SINGLE_VERSION if kind == "single" else MULTI_VERSION)


def _flat(correct):
    x = int(bool(correct))
    return {"strict": x, "recall": float(x), "precision": float(x), "f1": float(x)}


def latest_identity_records(paths):
    """{rid: record} over runs_*.jsonl in the given run dirs (latest line wins); records without meta.identity are
    counted and skipped."""
    out, skipped = {}, 0
    for d in paths:
        for f in sorted(glob.glob(str(Path(d) / "runs_*.jsonl"))):
            for line in open(f, encoding="utf-8"):
                if not line.strip():
                    continue
                r = json.loads(line)
                rid = ((r.get("meta") or {}).get("identity") or {}).get("rid")
                if not rid:
                    skipped += 1
                    continue
                out[rid] = r
    return out, skipped


def verdict_key(v):
    """A verdict is bound to the record, the exact answer, the rubric, the judge model AND the judge endpoint pin
    (None for the unpinned verdicts made before 2026-10-01: a pinned run never reuses them)."""
    return (v.get("rid"), v.get("answer_sha1"), v.get("rubric_sha1"), v.get("judge"), v.get("judge_provider_pin"))


def _slug(judge):
    return judge.replace("/", "__").replace(":", "_")


def _served_matches(pin, served):
    """Does the provider name OpenRouter reports match the pinned tag ('deepinfra/bf16' ~ 'DeepInfra')?"""
    if not pin or not served:
        return None
    norm = lambda s: re.sub(r"[^a-z0-9]", "", s.lower())       # noqa: E731
    return norm(pin.split("/")[0]) == norm(served)


def final_verdicts(out_dir, judge, provider_pin=None):
    """The reporting view of one judge run (FREEZE_ASTRA_R1 malformed-judge policy). -> {rid: row}, row = dict(status,
    strict, strict_lower, strict_upper, machine_strict, answer_sha1, ...):
      judged              the machine verdict (strict = machine_strict)
      adjudicated         a needs_adjudication cell resolved by a BLINDED human adjudication (adjudications_<judge>.jsonl
                          rows {rid, answer_sha1, strict, adjudicator, blinded: true}); machine_strict stays None
      needs_adjudication  no valid machine verdict after the bounded identical retries and no adjudication yet:
                          strict None, reported as the bounds [0, 1] - NEVER counted as 0
    Only rows of this judge model and this endpoint pin count. The FIRST valid verdict of a key is final (a judged
    answer is never re-judged; there is no later-wins precedence)."""
    out_dir, slug = Path(out_dir), _slug(judge)
    rows = {}
    for v in read_jsonl(out_dir / ("verdicts_%s.jsonl" % slug)):
        if v.get("judge") != judge or v.get("judge_provider_pin") != provider_pin or v["rid"] in rows:
            continue
        rows[v["rid"]] = dict(status="judged", strict=v["strict"], strict_lower=v["strict"], strict_upper=v["strict"],
                              machine_strict=v["strict"], answer_sha1=v["answer_sha1"], verdict=v)
    adj = {}
    for a in read_jsonl(out_dir / ("adjudications_%s.jsonl" % slug)):
        if a.get("blinded") is True and a.get("strict") in (0, 1):
            adj.setdefault((a["rid"], a["answer_sha1"]), a)
    for n in read_jsonl(out_dir / ("needs_adjudication_%s.jsonl" % slug)):
        if n.get("judge") != judge or n.get("judge_provider_pin") != provider_pin or n["rid"] in rows:
            continue
        a = adj.get((n["rid"], n["answer_sha1"]))
        if a is not None:
            rows[n["rid"]] = dict(status="adjudicated", strict=a["strict"], strict_lower=a["strict"],
                                  strict_upper=a["strict"], machine_strict=None, answer_sha1=n["answer_sha1"],
                                  adjudication=a, pending=n)
        else:
            rows[n["rid"]] = dict(status="needs_adjudication", strict=None, strict_lower=0, strict_upper=1,
                                  machine_strict=None, answer_sha1=n["answer_sha1"], pending=n)
    return rows


def read_jsonl(path):
    p = Path(path)
    if not p.is_file():
        return []
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def judge_records(records, questions, client, judge, out_dir, retry_failures=False, corpus=None, methods=None,
                  cache=None, ledger=None, log=print, max_model_calls=None, provider_pin=None, max_attempts=3,
                  on_malformed="adjudicate"):
    """Judge every identity record not yet judged under its current answer and rubric. -> summary dict.
    records {rid: record}; questions {id: question dict}; client.chat(messages, purpose=..., num_predict=...).

    provider_pin  the OpenRouter endpoint tag the client is pinned to (allow_fallbacks false); recorded in every
                  verdict and part of the verdict key and the cache key (FREEZE_ASTRA_R1 fix a). None = unpinned.
    max_attempts  bounded IDENTICAL judge calls per record in one run (same messages, same parameters): a malformed /
                  unparsable / incomplete reply or a call error is followed by the next identical attempt, up to
                  max_attempts (FREEZE_ASTRA_R1 fix b). The system answer is NEVER regenerated.
    on_malformed  'adjudicate' (default): when no attempt parses and at least one attempt returned a reply, the record
                  goes to needs_adjudication_<judge>.jsonl - terminal (never re-called, never counted as 0; resolved
                  only by a blinded adjudication, see final_verdicts). When every attempt was a call error (no reply)
                  it is an infrastructure failure: judge_failures_<judge>.jsonl, retried only with --retry-failures.
                  'failure': the legacy rule (every failure -> judge_failures)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    slug = _slug(judge)
    vpath, fpath = out_dir / ("verdicts_%s.jsonl" % slug), out_dir / ("judge_failures_%s.jsonl" % slug)
    npath = out_dir / ("needs_adjudication_%s.jsonl" % slug)
    max_attempts = max(1, int(max_attempts))
    done = {verdict_key(v) for v in read_jsonl(vpath)}
    failed = {verdict_key(v) for v in read_jsonl(fpath)}
    pending = {verdict_key(v) for v in read_jsonl(npath)}
    summ = {"records": len(records), "judged": 0, "rule": 0, "cache": 0, "model": 0, "failures": 0,
            "needs_adjudication": 0, "retried_records": 0, "skipped_done": 0, "skipped_failed": 0,
            "skipped_needs_adjudication": 0, "unknown_question": 0, "provider_pin": provider_pin,
            "max_attempts": max_attempts, "served_provider_mismatch": 0}
    for rid, r in sorted(records.items()):
        if max_model_calls is not None and summ["model"] >= max_model_calls:
            summ["stopped_at_max_model_calls"] = max_model_calls
            break
        ident = (r.get("meta") or {}).get("identity") or {}
        qd = questions.get(r.get("question_id"))
        if qd is None:
            summ["unknown_question"] += 1
            continue
        if methods and ident.get("method") not in methods:
            continue
        lang = r["setting"][:2]
        kind, packet, rsha, rversion = question_rubric(qd, lang, corpus)
        psha = sha1_json(packet)[:16] if packet is not None else None
        key = (rid, answer_sha1(r), rsha, judge, provider_pin)
        if key in done:
            summ["skipped_done"] += 1
            continue
        if key in pending:
            summ["skipped_needs_adjudication"] += 1
            continue
        if key in failed and not retry_failures:
            summ["skipped_failed"] += 1
            continue
        q = Question.from_dict(qd)
        base = {"rid": rid, "identity": ident, "experiment_id": ident.get("experiment_id"), "question_id": q.id,
                "pool_id": ident.get("pool_id"), "method": ident.get("method") or r.get("approach"),
                "options": ident.get("options"), "reader": ident.get("reader"), "replicate": ident.get("replicate"),
                "setting": r.get("setting"), "task_family": qd.get("task_family"),
                "rubric": rversion, "rubric_sha1": rsha, "tier": ((qd.get("meta") or {}).get("tier")),
                "packet_sha1": psha, "answer_sha1": key[1], "judge": judge, "judge_provider_pin": provider_pin}
        rv = JF.rule(q, r)
        res, why, raw, source = None, None, None, None
        attempts_log, served = [], None
        if rv is not None:
            source = "rule"
            res = dict(_flat(rv[0]), reason=rv[1])
            if rv[1] == "rule: abstained":
                res["abstained"] = True
                res["false_abstention"] = q.answer_type != "unanswerable"
        else:
            msgs = (JF.messages(q, r) if kind == "single" else v2_unans_messages(qd, r) if kind == "v2_unans"
                    else v2_multi_messages(qd, r, packet) if kind == "v2" else multi_messages(qd, r, packet))
            npred = 200 if kind in ("single", "v2_unans") else 1500
            ck = {"v": base["rubric"], "judge": judge, "m": msgs}
            if provider_pin:                                  # unpinned keys stay as before (old caches still hit)
                ck["pin"] = provider_pin
            ckey = sha1_json(ck)
            hit = cache.get(ckey) if cache is not None else None
            for attempt in range(1, (1 if hit is not None else max_attempts) + 1):
                why, raw, res = None, None, None
                if hit is not None:
                    raw, source = hit.get("raw"), "cache"
                    served = hit.get("provider")
                else:
                    if max_model_calls is not None and summ["model"] >= max_model_calls:
                        why = "stopped at max_model_calls"
                        break
                    try:
                        if ledger is not None:
                            with ledger.scope(phase="eval", approach="judge-multi", question_id=q.id,
                                              setting=r["setting"]):
                                cres = client.chat(msgs, purpose="judge", num_predict=npred)
                        else:
                            cres = client.chat(msgs, purpose="judge", num_predict=npred)
                        raw, served = cres.text, getattr(cres, "provider", None)
                        source = "judge"
                        summ["model"] += 1
                    except BudgetExceeded:
                        log("judge_multi: STOPPED by the budget guard (nothing written for this record): %s" % summ)
                        raise
                    except Exception as e:  # noqa: BLE001
                        why = "call failed: %s: %s" % (type(e).__name__, str(e)[:200])
                if why is None:
                    if kind in ("single", "v2_unans"):
                        c, reason = JF.parse(raw)
                        if c is None:
                            why = "unparsable single-rubric reply"
                        else:
                            res = dict(_flat(c), reason=reason)
                    elif kind == "v2":
                        parsed, why = parse_multi(raw, [u["claim_id"] for u in packet["units"]], V2_CLAIM_STATUSES)
                        if parsed is not None:
                            res = dict(score_v2(parsed, packet), units=parsed["units"], claims=parsed["claims"],
                                       reason=parsed["reason"])
                    else:
                        parsed, why = parse_multi(raw, [u["claim_id"] for u in packet["units"]])
                        if parsed is not None:
                            res = dict(score_multi(parsed, packet), units=parsed["units"], claims=parsed["claims"],
                                       reason=parsed["reason"])
                attempts_log.append({"attempt": attempt, "ok": res is not None, "error": why,
                                     "reply": raw is not None and source != "cache" and why is not None
                                     and not str(why).startswith("call failed"),
                                     "provider": served, "raw": None if res is not None else (raw or "")[:2000]})
                if res is not None:
                    if source == "judge" and cache is not None:
                        cache.put(ckey, {"raw": raw, "provider": served})
                    break
            if len(attempts_log) > 1:
                summ["retried_records"] += 1
        now = dt.datetime.now().isoformat(timespec="seconds")
        if res is None:
            replied = any(a["reply"] for a in attempts_log)
            if on_malformed == "adjudicate" and replied:
                summ["needs_adjudication"] += 1
                with open(npath, "a", encoding="utf-8") as fh:
                    fh.write(json.dumps(dict(base, status="needs_adjudication", error=why, attempts=attempts_log,
                                             n_attempts=len(attempts_log), at=now), ensure_ascii=False) + "\n")
                continue
            summ["failures"] += 1
            with open(fpath, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(dict(base, error=why, raw=(raw or "")[:2000], attempts=attempts_log, at=now),
                                    ensure_ascii=False) + "\n")
            continue
        summ["judged"] += 1
        if source in ("rule", "cache"):
            summ[source] += 1
        extra = {}
        if source in ("judge", "cache"):
            m = _served_matches(provider_pin, served)
            extra = {"judge_provider_served": served, "judge_attempts": len(attempts_log),
                     "judge_provider_match": m}
            if m is False:
                summ["served_provider_mismatch"] += 1
        with open(vpath, "a", encoding="utf-8") as fh:        # saved at once: a stop loses nothing judged
            fh.write(json.dumps(dict(base, source=source, at=now, **extra, **res), ensure_ascii=False) + "\n")
    log("judge_multi: %s" % summ)
    return summ


def load_question_dicts(path):
    return {d["id"]: d for d in (json.loads(l) for l in open(path, encoding="utf-8-sig") if l.strip())}


def main(argv=None):
    sys.stdout.reconfigure(encoding="utf-8")
    from ..config import load_config
    from ..llm import Ledger
    from ..llm_openrouter import OpenRouterReader, model_info
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge", required=True)
    ap.add_argument("--runs", action="append", required=True, help="run dir(s); relative ones resolve under work/")
    ap.add_argument("--questions", required=True)
    ap.add_argument("--out", default=None, help="default work/multidoc/judge")
    ap.add_argument("--methods", default=None)
    ap.add_argument("--retry-failures", action="store_true", help="judge again the records whose last attempt failed")
    ap.add_argument("--max-model-calls", type=int, default=None,
                    help="stop after this many judge-model calls (smoke tests); rule verdicts do not count")
    ap.add_argument("--judge-provider", default=None,
                    help="pin the judge to ONE OpenRouter endpoint tag (e.g. deepinfra/bf16): provider order [tag], "
                         "allow_fallbacks false, require_parameters true; recorded in every verdict and its key")
    ap.add_argument("--require-pin", action="store_true", help="refuse to run without --judge-provider (test runs)")
    ap.add_argument("--max-judge-attempts", type=int, default=3,
                    help="bounded identical judge calls per record (malformed reply / call error), default 3")
    ap.add_argument("--on-malformed", choices=("adjudicate", "failure"), default="adjudicate",
                    help="adjudicate (default): no parsable reply after the attempts -> needs_adjudication (terminal, "
                         "never 0); failure: the legacy rule")
    ap.add_argument("--http-attempts", type=int, default=None,
                    help="HTTP attempts per judge call for transport errors (429/5xx/network); default the reader's 7")
    a = ap.parse_args(argv)
    if a.require_pin and not a.judge_provider:
        print("REFUSED: --require-pin given without --judge-provider")
        return 2
    cfg = load_config()
    root = Path(cfg.project_root)
    runs = [Path(d) if Path(d).is_absolute() else root / "work" / d for d in a.runs]
    recs, skipped = latest_identity_records(runs)
    if skipped:
        print("skipping %d record line(s) without meta.identity (not an identity run)" % skipped)
    qs = load_question_dicts(a.questions)
    out = Path(a.out) if a.out else root / "work" / "multidoc" / "judge"
    out.mkdir(parents=True, exist_ok=True)
    slug = a.judge.replace("/", "__").replace(":", "_")
    ledger = Ledger(path=str(out / ("judge_ledger_%s.jsonl" % slug)))
    info = model_info(a.judge, root / "work" / "openrouter" / "catalog.json")
    client = OpenRouterReader(cfg, ledger, a.judge, supported=info.get("params"), provider=a.judge_provider,
                              attempts=a.http_attempts)
    run_cfg = {"at": dt.datetime.now().isoformat(timespec="seconds"), "judge": a.judge,
               "judge_provider_pin": a.judge_provider,
               "provider_routing": ({"order": [a.judge_provider], "allow_fallbacks": False, "require_parameters": True}
                                    if a.judge_provider else None),
               "http_attempts": client.or_retries + 1, "max_judge_attempts": a.max_judge_attempts,
               "on_malformed": a.on_malformed,
               "sent_params": sorted(k for k in ("temperature", "top_p", "seed")
                                     if not client.supported or k in client.supported),
               "generation": {k: cfg.generation.get(k) for k in ("temperature", "top_p", "seed")},
               "rubrics": [SINGLE_VERSION, MULTI_VERSION], "questions": a.questions, "runs": [str(d) for d in runs]}
    with open(out / "judge_runs.jsonl", "a", encoding="utf-8") as fh:      # every run's judge configuration
        fh.write(json.dumps(run_cfg, ensure_ascii=False) + "\n")
    try:
        judge_records(recs, qs, client, a.judge, out, retry_failures=a.retry_failures,
                      methods=set(a.methods.split(",")) if a.methods else None,
                      cache=KVCache(out / "judge_multi_cache.sqlite"), ledger=ledger,
                      max_model_calls=a.max_model_calls, provider_pin=a.judge_provider,
                      max_attempts=a.max_judge_attempts, on_malformed=a.on_malformed)
    except BudgetExceeded as e:
        print("BUDGET STOP: %s (verdicts so far are saved; rerun resumes)" % e)
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
