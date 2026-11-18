"""TIMELINE questions: build series questions from a .src.yaml, validate them, check the set's composition.
Driven by ragbench.tools.questions (build / validate / summary dispatch here for type: timeline).

    python -m ragbench.tools.questions build --src ragbench/data/questions/pilot_timeline.src.yaml
        --out ragbench/data/questions/pilot_timeline.jsonl          (one command line)
    python -m ragbench.tools.questions validate --path ragbench/data/questions/pilot_timeline.jsonl --strict

The schema (fields, the reconciliation rule, structured output, metrics) is documented in
ragbench/eval/timeline.py. YAML per question:

  id, corpus, type: timeline, answer_type: series, doc_scope: [every report used], memorisable, notes
  question {fa, en}, answer {fa, en}            (answer: a short prose answer stating every period)
  comparable: true | false                      (false: the metric's definition or scope changes)
  series_notes {fa, en}                         (definition changes, restatements, conflicts)
  comparability_cues {fa: [...], en: [...]}     (phrases that count as flagging the change in an answer)
  series:                                       (chronological; one item per period)
    - period: "1398"
      calendar: jalali                          (default: jalali for 13xx/14xx, gregorian for 19xx/20xx)
      value: 795
      unit: people                              (scale words included: "million users"; "percent")
      unit_fa: نفر
      definition: {fa: ..., en: ...}
      doc: C.divar_annual-report_1398           (the report OF the period, see the reconciliation rule)
      fa: ["795 نفر"]                           (quote terms, as in pilot.src.yaml; en: only for A/B)
      only / exclude / langs / needs_linking    (as in pilot.src.yaml)
      context: [{fa: [...], only: [...]}]       (label units for a detached number; headings allowed)
      note: "..."                               (required when the doc id does not name the period)
  reconciliation:
    restatements:  [{period, value, unit, doc, fa: [...], only, note}]   (a later report's different number)
    supplementary: [{period, value, unit, doc, fa: [...], only, note}]   (a period with no report of its own)
"""
from collections import Counter, OrderedDict

from ..contracts import LANGS
from ..corpus.model import split_id, swap_lang
from ..eval.timeline import (RECONCILIATION_RULE, RECONCILIATION_RULE_TEXT, SCHEMA_VERSION, SERIES_ANSWER_TYPE,
                             TIMELINE_TYPE, extract_series, flags_definition_change, gold_periods, is_timeline,
                             item_base_value, parse_numbers, period_key, timeline_spec, unit_text_contains_value)
from .questions import (AUTHOR, doc_scope_ids, lexical_overlap, link_item, order_gold, required_langs)

TIMELINE_TARGETS = {"total": 8, "min_non_comparable": 2, "min_restated": 1, "min_reports": 3}
SERIES_ITEM_FIELDS = ("period", "value", "unit", "definition_fa", "definition_en", "doc", "gold_units")


def _doc_base(d):
    return d.rsplit(".", 1)[0] if str(d).endswith((".fa", ".en")) else str(d)


def _calendar(period):
    k = period_key(period)[0]
    return "gregorian" if 1900 <= k <= 2099 else "jalali"


def _link_context(entry, base, spec, corp):
    """Label units for a detached number: like link_item, but headings are allowed."""
    corpus = spec["corpus"]
    langs = [l for l in (entry.get("langs") or required_langs(corpus)) if l in required_langs(corpus)]
    only, exclude = set(entry.get("only") or []), set(entry.get("exclude") or [])
    res = OrderedDict()
    for l in langs:
        did = "%s.%s" % (base, l)
        terms = entry.get(l) or []
        if not corp.has(did):
            res[l] = {"doc_id": did, "units": [], "status": "doc_missing"}
            continue
        if not terms:
            raise ValueError("%s: context entry without %s quote terms" % (spec["id"], l))
        d = corp.doc(did)
        hits = [u for u in corp.search(did, terms)
                if (not only or split_id(u.id)[1] in only) and split_id(u.id)[1] not in exclude]
        res[l] = {"doc_id": did, "units": sorted([u.id for u in hits], key=lambda x: d.unit(x).order),
                  "status": "linked" if hits else "no_match"}
    return res


def _link_side_item(kind, it, spec, corp, probs, qid):
    """restatement / supplementary: link its quote (optional) -> dict for meta.reconciliation."""
    base = _doc_base(it["doc"])
    langs = required_langs(spec["corpus"])
    vb, pct = item_base_value({"value": it.get("value"), "unit": it.get("unit") or ""})
    out = OrderedDict(period=str(it["period"]), value=it.get("value"), unit=it.get("unit") or "",
                      unit_fa=it.get("unit_fa") or "", value_base=vb, percent=pct, doc=base + ".fa",
                      units={l: [] for l in langs})
    if any(it.get(l) for l in langs):
        linked = link_item(dict(it, fact="%s %s: %s" % (kind, it["period"], it.get("value")), doc=base), spec, corp)
        for l, r in linked.items():
            out["units"][l] = r["units"]
            if r["status"] != "linked":
                probs.append(("error", "%s: %s %s quote %s: %s in %s" % (qid, kind, it["period"], it.get(l), r["status"],
                                                                         r["doc_id"])))
        out["quote"] = {l: it.get(l) for l in langs if it.get(l)}
    for k in ("only", "exclude", "note"):
        if it.get(k):
            out[k] = it[k]
    return out


def build_timeline_question(spec, corp, source="pilot_timeline.src.yaml"):
    """-> (question dict, problems[list of (level, msg)]) for a type: timeline spec."""
    probs = []
    qid, corpus = spec["id"], spec["corpus"]
    langs = required_langs(corpus)
    series_out, evidence_out = [], []
    order_groups = {l: [] for l in LANGS}
    value_groups = {l: [] for l in LANGS}
    needs_linking = False
    for it in spec.get("series") or []:
        period = str(it["period"])
        base = _doc_base(it["doc"])
        fact = "%s: %s %s" % (period, it.get("value"), it.get("unit") or "")
        linked = link_item(dict(it, fact=fact, doc=base), spec, corp)
        vb, pct = item_base_value({"value": it.get("value"), "unit": it.get("unit") or ""})
        entry = OrderedDict(period=period, calendar=it.get("calendar") or _calendar(period), value=it.get("value"),
                            unit=it.get("unit") or "", unit_fa=it.get("unit_fa") or "", value_base=vb, percent=pct,
                            definition_fa=(it.get("definition") or {}).get("fa", ""),
                            definition_en=(it.get("definition") or {}).get("en", ""),
                            doc=base + ".fa", gold_units={l: [] for l in langs}, context_units={l: [] for l in langs},
                            quote={l: it.get(l) for l in langs if it.get(l)})
        ev = OrderedDict(fact=fact, doc=base, period=period, role="value", quote=dict(entry["quote"]), units={}, status={})
        for l, r in linked.items():
            entry["gold_units"][l] = list(r["units"])
            ev["units"][l] = list(r["units"])
            ev["status"][l] = r["status"]
            if r["status"] == "doc_missing":
                needs_linking = True
                probs.append(("warn", "%s: %s %s doc %s not extracted yet" % (qid, period, l, r["doc_id"])))
            elif r["status"] == "no_match":
                if it.get("needs_linking"):
                    needs_linking = True
                    probs.append(("warn", "%s: %s %s quote %s not found (flagged needs_linking)" % (qid, period, l, it.get(l))))
                else:
                    probs.append(("error", "%s: period %s %s quote %s matches no unit in %s" % (qid, period, l, it.get(l),
                                                                                               r["doc_id"])))
            if r.get("oversize"):
                ev["oversize"] = {l: r["oversize"]}
        evidence_out.append(ev)
        ctx_all = {l: [] for l in langs}
        for c in it.get("context") or []:
            cl = _link_context(c, base, spec, corp)
            cev = OrderedDict(fact="%s: label for the value" % period, doc=base, period=period, role="context",
                              quote={l: c.get(l) for l in langs if c.get(l)}, units={}, status={})
            if c.get("only"):
                cev["only"] = c["only"]
            for l, r in cl.items():
                cev["units"][l] = list(r["units"])
                cev["status"][l] = r["status"]
                if r["status"] != "linked":
                    probs.append(("error", "%s: period %s context quote %s: %s in %s" % (qid, period, c.get(l), r["status"],
                                                                                         r["doc_id"])))
                for u in r["units"]:
                    if u not in ctx_all[l] and u not in entry["gold_units"][l]:
                        ctx_all[l].append(u)
            cev["units"] = {l: [u for u in cev["units"][l] if u in ctx_all[l]] for l in cev["units"]}
            evidence_out.append(cev)
        entry["context_units"] = ctx_all
        for k in ("only", "exclude", "langs", "note", "page"):
            if it.get(k):
                entry[k] = it[k]
        series_out.append(entry)
        for l in langs:
            value_groups[l].append(list(entry["gold_units"][l]))
            order_groups[l].append(list(entry["gold_units"][l]) + list(ctx_all[l]))
    gold = {l: order_gold(order_groups[l], corp, lang=l, corpus=corpus) if order_groups[l] else [] for l in LANGS}
    rec_spec = spec.get("reconciliation") or {}
    reconciliation = OrderedDict(
        rule=RECONCILIATION_RULE, rule_text=RECONCILIATION_RULE_TEXT,
        restatements=[_link_side_item("restatement", r, spec, corp, probs, qid) for r in rec_spec.get("restatements") or []],
        supplementary=[_link_side_item("supplementary", s, spec, corp, probs, qid) for s in rec_spec.get("supplementary") or []])
    notes = spec.get("series_notes") or {}
    meta = OrderedDict()
    meta["reports"] = sorted({_doc_base(e["doc"]) for e in series_out} | {d.rsplit(".", 1)[0] for d in doc_scope_ids(spec)})
    meta["subtype"] = spec.get("subtype") or ("non-comparable" if spec.get("comparable") is False else "comparable")
    meta["memorisable"] = bool(spec.get("memorisable", False))
    if spec.get("notes"):
        meta["notes"] = spec["notes"]
    meta["needs_unit_linking"] = needs_linking
    meta["schema_version"] = SCHEMA_VERSION
    meta["answer_series"] = series_out
    meta["comparable"] = spec.get("comparable")
    meta["series_notes_fa"] = notes.get("fa", "")
    meta["series_notes_en"] = notes.get("en", "")
    meta["comparability_cues"] = spec.get("comparability_cues") or {}
    meta["reconciliation"] = reconciliation
    meta["evidence"] = evidence_out
    meta["gold_groups"] = {l: value_groups[l] for l in langs}
    meta["lexical_overlap"] = {l: lexical_overlap((spec.get("question") or {}).get(l, ""), gold[l], corp)
                               for l in langs if gold[l]}
    meta["author"] = AUTHOR
    meta["source"] = source
    q = OrderedDict(
        id=qid, corpus=corpus,
        question_fa=(spec.get("question") or {}).get("fa", ""),
        question_en=(spec.get("question") or {}).get("en", ""),
        answer_fa=(spec.get("answer") or {}).get("fa", ""),
        answer_en=(spec.get("answer") or {}).get("en", ""),
        answer_type=SERIES_ANSWER_TYPE,
        gold_units={"fa": gold["fa"], "en": gold["en"] if corpus != "C" else []},
        doc_scope=doc_scope_ids(spec),
        type=TIMELINE_TYPE, answer_value=None, answer_unit=None, meta=meta)
    if spec.get("type") != TIMELINE_TYPE or spec.get("answer_type") not in (None, SERIES_ANSWER_TYPE):
        probs.append(("error", "%s: a series spec needs type timeline and answer_type series" % qid))
    return q, probs


# --------------------------------------------------------------------------
# validation
# --------------------------------------------------------------------------

def contracts_support_timeline():
    from ..contracts import ANSWER_TYPES, QUESTION_TYPES
    return SERIES_ANSWER_TYPE in ANSWER_TYPES and TIMELINE_TYPE in QUESTION_TYPES


def validate_timeline(q, corp, rep):
    """Timeline-specific checks (the generic ones run in questions.validate_questions)."""
    meta = q.meta or {}
    linking = bool(meta.get("needs_unit_linking"))
    spec = timeline_spec(q)
    if q.type != TIMELINE_TYPE or q.answer_type != SERIES_ANSWER_TYPE:
        rep.add("error", q.id, "a timeline question needs type %s and answer_type %s (has %s / %s)" % (
            TIMELINE_TYPE, SERIES_ANSWER_TYPE, q.type, q.answer_type))
    if q.answer_value is not None or q.answer_unit:
        rep.add("warn", q.id, "answer_value/answer_unit are unused for series answers")
    if meta.get("schema_version") != SCHEMA_VERSION:
        rep.add("warn", q.id, "meta.schema_version is %r, expected %r" % (meta.get("schema_version"), SCHEMA_VERSION))
    series = spec["answer_series"]
    if not series:
        rep.add("error", q.id, "meta.answer_series is empty")
        return
    if len(series) < 2:
        rep.add("error", q.id, "a timeline needs at least two periods")
    periods = [str(it.get("period")) for it in series]
    dups = [p for p, n in Counter(periods).items() if n > 1]
    if dups:
        rep.add("error", q.id, "periods are not unique: %s" % dups)
    if [period_key(p) for p in periods] != sorted(period_key(p) for p in periods):
        rep.add("error", q.id, "periods are not in chronological order: %s" % periods)
    if len({(it.get("calendar") or _calendar(str(it.get("period")))) for it in series}) > 1:
        rep.add("warn", q.id, "the series mixes Jalali and Gregorian periods")
    scope = [d for d in q.doc_scope] if isinstance(q.doc_scope, list) else []
    scope_fa = {swap_lang(d, "fa") for d in scope}
    langs = required_langs(q.corpus)
    used_docs = set()
    union = {l: set() for l in LANGS}
    for it in series:
        p = str(it.get("period"))
        missing = [k for k in SERIES_ITEM_FIELDS if it.get(k) in (None, "", {})]
        if missing:
            rep.add("error", q.id, "period %s lacks %s" % (p, missing))
        doc = it.get("doc") or ""
        doc_fa = swap_lang(doc, "fa") if doc.endswith((".fa", ".en")) else doc + ".fa"
        used_docs.add(doc_fa)
        if doc_fa not in scope_fa:
            rep.add("error", q.id, "period %s: its report %s is not in doc_scope" % (p, doc_fa))
        if p[:4] not in doc and not it.get("note"):
            rep.add("warn", q.id, "period %s comes from %s, whose id does not name the period; say why in the item's note "
                    "(e.g. the 1400 report reviews 1399)" % (p, doc))
        base, _pct = item_base_value(dict(it, value_base=None))
        if isinstance(it.get("value"), (int, float)) and base is None:
            rep.add("error", q.id, "period %s: value %r with unit %r has no numeric base" % (p, it.get("value"), it.get("unit")))
        if it.get("value_base") is not None and base is not None and abs(float(it["value_base"]) - base) > 1e-6 * max(1.0, abs(base)):
            rep.add("error", q.id, "period %s: value_base %s differs from value x unit scale %s (rebuild)" % (p, it["value_base"], base))
        for l in LANGS:
            gl = list((it.get("gold_units") or {}).get(l) or [])
            cl = list((it.get("context_units") or {}).get(l) or [])
            union[l].update(gl)
            union[l].update(cl)
            if q.corpus == "C" and l == "en" and (gl or cl):
                rep.add("error", q.id, "period %s: corpus C must not have English gold units" % p)
            if l in langs and not gl and not linking:
                rep.add("error", q.id, "period %s has no %s gold units" % (p, l))
            for u in gl + cl:
                un = corp.unit(u)
                if un is None:
                    rep.add("warn" if linking else "error", q.id, "period %s: unit %s does not exist" % (p, u))
                    continue
                if split_id(u)[0] != swap_lang(doc_fa, l):
                    rep.add("error", q.id, "period %s: unit %s is not in the period's report %s" % (p, u, swap_lang(doc_fa, l)))
            for u in gl:
                un = corp.unit(u)
                if un is not None and not unit_text_contains_value(un.text, it):
                    rep.add("error", q.id, "period %s: gold unit %s does not contain the value %s %s" % (
                        p, u, it.get("value"), it.get("unit")))
    for l in LANGS:
        if union[l] != set(q.gold(l)):
            rep.add("error", q.id, "%s gold_units differ from the union of the periods' gold and context units" % l)
    comparable = spec["comparable"]
    if not isinstance(comparable, bool):
        rep.add("error", q.id, "meta.comparable must be true or false (is %r)" % (comparable,))
    if comparable is False:
        for l in LANGS:
            if not (spec["series_notes_%s" % l] or "").strip():
                rep.add("error", q.id, "comparable=false needs series_notes_%s describing the change" % l)
        if not any(spec["comparability_cues"].get(l) for l in LANGS):
            rep.add("warn", q.id, "comparable=false without comparability_cues (only generic flag phrases will count)")
        for l in LANGS:
            if not flags_definition_change(q.answer(l), spec["comparability_cues"]):
                rep.add("warn", q.id, "answer_%s does not flag the definition change in words the scorer recognises" % l)
    rec = (meta.get("reconciliation") or {})
    if rec.get("rule") != RECONCILIATION_RULE:
        rep.add("error", q.id, "meta.reconciliation.rule must be %r" % RECONCILIATION_RULE)
    by_p = {str(it.get("period")): it for it in series}
    for kind in ("restatements", "supplementary"):
        for r in rec.get(kind) or []:
            p = str(r.get("period"))
            doc_fa = swap_lang(r.get("doc") or "", "fa") if str(r.get("doc")).endswith((".fa", ".en")) else "%s.fa" % r.get("doc")
            used_docs.add(doc_fa)
            if doc_fa not in scope_fa:
                rep.add("error", q.id, "%s %s: report %s is not in doc_scope" % (kind, p, doc_fa))
            if kind == "restatements":
                if p not in by_p:
                    rep.add("error", q.id, "restatement of %s, which is not a gold period" % p)
                    continue
                if swap_lang(by_p[p]["doc"], "fa") == doc_fa:
                    rep.add("error", q.id, "restatement of %s comes from the period's own report; the rule makes that value gold" % p)
                gb, _ = item_base_value(by_p[p])
                rb, _ = item_base_value(r)
                if gb is not None and rb is not None and abs(gb - rb) <= 1e-9 * max(1.0, abs(gb)):
                    rep.add("warn", q.id, "restatement of %s equals the gold value" % p)
            elif p in by_p:
                rep.add("error", q.id, "supplementary period %s is already a gold period" % p)
            for l, units in (r.get("units") or {}).items():
                for u in units:
                    un = corp.unit(u)
                    if un is None:
                        rep.add("error", q.id, "%s %s: unit %s does not exist" % (kind, p, u))
                    elif not unit_text_contains_value(un.text, r):
                        rep.add("error", q.id, "%s %s: unit %s does not contain %s %s" % (kind, p, u, r.get("value"), r.get("unit")))
    for d in sorted(scope_fa - used_docs):
        rep.add("warn", q.id, "doc_scope lists %s but no period, restatement or supplementary value uses it" % d)
    # the reference answer must state the series in a form the text scorer reads
    known = periods + [str(r.get("period")) for r in (rec.get("restatements") or []) + (rec.get("supplementary") or [])]
    pcts = {bool(item_base_value(it)[1]) for it in series}
    want_percent = pcts.pop() if len(pcts) == 1 else None
    from ..eval.timeline import DEFAULT_SERIES_MATCH, _gold_scale, mention_matches, series_duration_kinds
    durations = series_duration_kinds(list(series) + (rec.get("restatements") or []) + (rec.get("supplementary") or []))
    for l in LANGS:
        parsed = extract_series(q.answer(l), known, want_percent, durations)   # as eval.timeline.score_series reads it
        missed = [p for p in periods if not (parsed.get(p) and mention_matches(parsed[p][0], *item_base_value(by_p[p]),
                                                                               _gold_scale(by_p[p]), DEFAULT_SERIES_MATCH))]
        if missed:
            rep.add("warn", q.id, "answer_%s: the series parser does not read the gold value for %s (reads %s)" % (
                l, missed, {k: [m.text for m in v] for k, v in parsed.items()}))
        vals = [m for m in parse_numbers(q.text(l)) if not m.plain_int or not (1300 <= m.raw <= 1499 or 1900 <= m.raw <= 2099)]
        for it in series:
            b, pct = item_base_value(it)
            if b is not None and any(abs(m.value - b) <= 1e-9 * max(1.0, abs(b)) for m in vals):
                rep.add("warn", q.id, "question_%s contains the gold value of %s (leak?)" % (l, it.get("period")))


def timeline_composition(qs):
    """[(label, cell, have, want, ok)] for a file of timeline questions."""
    tq = [q for q in qs if is_timeline(q)]
    nc = sum(1 for q in tq if timeline_spec(q)["comparable"] is False)
    rs = sum(1 for q in tq if timeline_spec(q)["reconciliation"]["restatements"])
    wide = sum(1 for q in tq if len({it["doc"] for it in timeline_spec(q)["answer_series"]}) >= TIMELINE_TARGETS["min_reports"])
    t = TIMELINE_TARGETS
    return [("TL", "total", len(tq), t["total"], len(tq) == t["total"]),
            ("TL", "comparable=false", nc, t["min_non_comparable"], nc >= t["min_non_comparable"]),
            ("TL", "with a restatement", rs, t["min_restated"], rs >= t["min_restated"]),
            ("TL", "spanning >= %d reports (where possible)" % t["min_reports"], wide, len(tq), True)]


def series_table(q):
    """Human-readable lines for `show` / `summary`."""
    spec = timeline_spec(q)
    lines = ["comparable=%s  periods=%s" % (spec["comparable"], gold_periods(spec))]
    for it in spec["answer_series"]:
        lines.append("  %s  %s %s  (%s)  gold %s  context %s" % (it["period"], it.get("value"), it.get("unit"), it.get("doc"),
                                                              (it.get("gold_units") or {}).get("fa"),
                                                              (it.get("context_units") or {}).get("fa")))
    for r in spec["reconciliation"]["restatements"]:
        lines.append("  restated %s: %s %s by %s" % (r["period"], r.get("value"), r.get("unit"), r.get("doc")))
    for s in spec["reconciliation"]["supplementary"]:
        lines.append("  supplementary %s: %s %s from %s" % (s["period"], s.get("value"), s.get("unit"), s.get("doc")))
    return lines
