#!/usr/bin/env python3
"""choose-rag: turn eight interview answers into a RAG setup recommendation.

Deterministic and dependency-free (Python 3.8+ standard library only).
It implements the rule semantics in references/chooser.json -> notes.semantics:

  * Unanswered questions count as 'unsure'. If q2 is 'en', q8 is hidden and
    counts as 'na'.
  * Rules are applied in array order. A rule fires when every key in 'when'
    matches: a list means any-of; a key starting with '!' means none-of.
    Keys q1..q8 refer to answers; any other key is a slot name and refers to
    the current value of that slot.
  * 'set' overwrites a slot; 'add' appends ids to a list slot unless already
    present; 'remove' deletes ids from a list slot.
  * Display order: a one-sentence summary (notes.summary); the slots channels,
    chunking, reranking_or_selection, budget, agent, normalisation; then
    tradeoffs, avoid, extras, with the items in notes.universal (advice that
    holds for every setup) listed last under one heading.
  * Sources cite the journal manuscript's tables and sections by title, not by
    number, because the numbers may change before publication; numbers cut from
    the manuscript cite their folder of the public data release.

Usage examples
  python choose.py --q1 t3 --q2 fa --q5 yes
  python choose.py --json '{"q1": "t3", "q2": "fa"}'
  python choose.py --json answers.json --format json
  python choose.py --questions            # print the interview
  python choose.py --q1 t2 --lang fa      # Persian output
"""
import argparse
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REF = os.path.normpath(os.path.join(HERE, "..", "references"))

SLOTS = ["channels", "chunking", "reranking_or_selection", "budget", "agent", "normalisation"]
LISTS = ["extras", "avoid", "tradeoffs"]
QIDS = ["q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8"]

def load(name):
    with open(os.path.join(REF, name), encoding="utf-8") as f:
        return json.load(f)


def normalise_answers(raw, chooser):
    """Fill defaults, validate ids, apply the q8 visibility rule."""
    valid = {q["id"]: [o["id"] for o in q["options"]] for q in chooser["questions"]}
    ans, errors = {}, []
    for q in QIDS:
        v = raw.get(q)
        if v is None or v == "":
            v = "unsure"
        v = str(v).strip().lower()
        if v not in valid[q] and not (q == "q8" and v == "na"):
            errors.append("%s: '%s' is not one of %s" % (q, v, ", ".join(valid[q])))
        ans[q] = v
    extra = [k for k in raw if k not in QIDS]
    if extra:
        errors.append("unknown keys: " + ", ".join(sorted(extra)))
    # q8 is shown only when Persian may be involved (questions[q8].show_if).
    q8 = next(q for q in chooser["questions"] if q["id"] == "q8")
    show_if = q8.get("show_if", {"q2": ["fa", "both", "cross", "unsure"]})
    if not all(ans[k] in v for k, v in show_if.items()):
        ans["q8"] = "na"
    return ans, errors


def matches(when, ans, state):
    for key, values in when.items():
        neg = key.startswith("!")
        k = key[1:] if neg else key
        current = ans[k] if k in QIDS else state.get(k)
        hit = current in values
        if hit == neg:
            return False
    return True


def run_rules(ans, chooser):
    state = {s: None for s in SLOTS}
    for l in LISTS:
        state[l] = []
    fired = []
    for rule in chooser["rules"]:
        if not matches(rule.get("when", {}), ans, state):
            continue
        fired.append(rule["id"])
        for slot, oid in rule.get("set", {}).items():
            state[slot] = oid
        for slot, ids in rule.get("add", {}).items():
            for i in ids:
                if i not in state[slot]:
                    state[slot].append(i)
        for slot, ids in rule.get("remove", {}).items():
            state[slot] = [i for i in state[slot] if i not in ids]
    return state, fired


def choose(raw, chooser=None):
    chooser = chooser or load("chooser.json")
    ans, errors = normalise_answers(raw, chooser)
    if errors:
        raise ValueError("; ".join(errors))
    state, fired = run_rules(ans, chooser)
    return ans, state, fired


# ----------------------------------------------------------------- output
LABELS = {
    "pipeline": {"en": "Pipeline", "fa": "پایپ‌لاین"},
    "confidence": {"en": "Confidence", "fa": "میزان اطمینان"},
    "evidence": {"en": "evidence", "fa": "شواهد"},
    "source": {"en": "Source", "fa": "منبع"},
    "summary": {"en": "In one sentence", "fa": "در یک جمله"},
    "universal": {"en": "Holds for every setup", "fa": "برای هر چیدمانی برقرار است"},
    "key": {"en": "key card", "fa": "کارت اصلی"},
    "scope": {"en": "Scope", "fa": "دامنهٔ اعتبار"},
    "none": {"en": "(none)", "fa": "(هیچ)"},
    "fired": {"en": "Rules fired", "fa": "قاعده‌های اجراشده"},
}


def t(obj, lang):
    if isinstance(obj, dict):
        return obj.get(lang) or obj.get("en", "")
    return obj or ""


def conf_text(option, chooser, lang):
    ui = chooser["ui"]
    out = []
    for c in option.get("confidence", []):
        label = t(ui.get(c["level"], {"en": c["level"].title()}), lang)
        scope = t(c.get("scope", {}), lang)
        out.append(label + (" (" + scope + ")" if scope else ""))
    return "; ".join(out)


def matches_any(conds, ans, state):
    return any(matches(c, ans, state) for c in conds)


def summary_sentence(state, chooser, lang):
    """notes.summary filled with the 'summary' phrase of each slot option."""
    tpl = t(chooser["notes"].get("summary", {}), lang)
    if not tpl:
        return ""
    for slot in ("channels", "reranking_or_selection", "budget", "agent"):
        ph = chooser["options"][state[slot]].get("summary", {})
        tpl = tpl.replace("{" + slot + "}", t(ph, lang) if ph else "")
    tpl = re.sub(r"\s+([,.،؛])", r"\1", tpl)
    tpl = re.sub(r"\s{2,}", " ", tpl).strip()
    return tpl[:1].upper() + tpl[1:]


def key_cards(state, chooser, n=3):
    """The cards shown first on the site: each slot option's 'key', then trade-offs."""
    opts, out = chooser["options"], []
    for slot in ("reranking_or_selection", "agent", "channels", "budget", "normalisation", "chunking"):
        o = opts[state[slot]]
        k = o["key"] if "key" in o else (o.get("evidence") or [None])[0]
        if k and k not in out and len(out) < n:
            out.append(k)
    for oid in state["tradeoffs"]:
        ev = opts[oid].get("evidence") or []
        if ev and ev[0] not in out and len(out) < n:
            out.append(ev[0])
    return out


def as_markdown(ans, state, fired, chooser, evidence, lang="en", show_evidence=True):
    ui, opts, notes = chooser["ui"], chooser["options"], chooser["notes"]
    qs = {q["id"]: q for q in chooser["questions"]}
    L = []
    L.append("# " + t(ui.get("result_title", {"en": "Recommended setup"}), lang))
    L.append("")
    L.append("**" + t(ui.get("your_answers", {"en": "Your answers"}), lang) + ":** " + ", ".join(
        "%s=%s" % (q, ans[q]) for q in QIDS))
    unsure = [q for q in QIDS if ans[q] == "unsure"]
    if unsure:
        shown = [q for q in QIDS if ans[q] != "na"]
        L.append("")
        L.append("_" + t(ui["unsure_note"], lang).replace("{n}", str(len(unsure))).replace(
            "{m}", str(len(shown))) + " (" + ", ".join(unsure) + ")_")
    L.append("")
    L.append("**" + t(LABELS["summary"], lang) + ":** " + summary_sentence(state, chooser, lang))
    L.append("")
    L.append("## " + t(LABELS["pipeline"], lang))
    used = []
    for slot in SLOTS:
        oid = state[slot]
        o = opts[oid]
        L.append("- **%s** (`%s`): %s" % (t(ui.get("slot_" + slot, {"en": slot}), lang), oid, t(o["text"], lang)))
        if o.get("note"):
            L.append("  - " + t(o["note"], lang))
        L.append("  - " + t(LABELS["confidence"], lang) + ": " + (conf_text(o, chooser, lang) or "-") +
                 ("; " + t(LABELS["evidence"], lang) + ": " + ", ".join(o["evidence"]) if o.get("evidence") else ""))
        used += o.get("evidence", [])
    titles = {"extras": "extras_title", "avoid": "avoid_title", "tradeoffs": "tradeoffs_title"}
    universal = notes.get("universal", [])
    groups = [(titles[l], [i for i in state[l] if i not in universal]) for l in ("tradeoffs", "avoid", "extras")]
    groups.append((None, [i for l in ("avoid", "extras") for i in state[l] if i in universal]))
    for title, ids in groups:
        if not ids and title is None:
            continue
        L.append("")
        L.append("## " + (t(ui.get(title, {"en": title}), lang) if title else t(LABELS["universal"], lang)))
        if not ids:
            L.append("- " + t(LABELS["none"], lang))
        for oid in ids:
            o = opts[oid]
            ev = ", ".join(o.get("evidence", []))
            L.append("- %s (`%s`). %s: %s%s" % (t(o["text"], lang), oid, t(LABELS["confidence"], lang),
                                                 conf_text(o, chooser, lang) or "-",
                                                 "; " + t(LABELS["evidence"], lang) + ": " + ev if ev else ""))
            used += o.get("evidence", [])
    if show_evidence:
        byid = {e["id"]: e for e in evidence}
        L.append("")
        L.append("## " + t(ui.get("evidence_title", {"en": "Evidence from the paper"}), lang))
        seen = []
        for eid in used:
            if eid not in seen:
                seen.append(eid)
        keys = key_cards(state, chooser)
        for eid in sorted(seen, key=lambda x: (x not in keys, seen.index(x))):
            e = byid[eid]
            L.append("- `%s` [%s]%s %s" % (eid, t(ui.get(e["confidence"], {"en": e["confidence"]}), lang),
                                          " (" + t(LABELS["key"], lang) + ")" if eid in keys else "",
                                          t(e["claim"], lang)))
            L.append("  - %s: %s" % (t(LABELS["source"], lang), t(e["source"]["paper"], lang)))
            cav = t(e.get("caveat", {}), lang)
            if cav:
                L.append("  - " + t(ui.get("caveat", {"en": "Caveat"}), lang) + ": " + cav)
    L.append("")
    L.append("## " + t(LABELS["scope"], lang))
    L.append("- " + t(notes["permanent"], lang))
    L.append("- " + t(notes["scope"], lang))
    if matches_any(notes.get("no_default_when", [{"reranking_or_selection": ["rr.ce"]}]), ans, state):
        L.append("- " + t(notes["no_default"], lang))
    if notes.get("source"):
        L.append("- " + t(notes["source"], lang))
    L.append("")
    L.append(t(LABELS["fired"], lang) + ": " + ", ".join(fired))
    return "\n".join(L)


def as_json(ans, state, fired, chooser, evidence):
    opts = chooser["options"]
    ids = [state[s] for s in SLOTS] + [i for l in LISTS for i in state[l]]
    ev = []
    for i in ids:
        for e in opts[i].get("evidence", []):
            if e not in ev:
                ev.append(e)
    byid = {e["id"]: e for e in evidence}
    return {
        "answers": ans,
        "summary": summary_sentence(state, chooser, "en"),
        "recommendation": {k: state[k] for k in SLOTS + LISTS},
        "universal": [i for l in ("avoid", "extras") for i in state[l] if i in chooser["notes"].get("universal", [])],
        "key_cards": key_cards(state, chooser),
        "cards": {i: {"text": opts[i]["text"]["en"], "note": opts[i].get("note", {}).get("en"),
                      "confidence": opts[i].get("confidence", []), "evidence": opts[i].get("evidence", [])}
                  for i in ids},
        "evidence": {e: {"claim": byid[e]["claim"]["en"], "confidence": byid[e]["confidence"],
                         "source": byid[e]["source"]["paper"]["en"],
                         "caveat": byid[e].get("caveat", {}).get("en", "")} for e in ev},
        "rules_fired": fired,
        "caveat": chooser["notes"]["permanent"]["en"],
    }


def print_questions(chooser, lang):
    for q in chooser["questions"]:
        cond = ""
        if q.get("show_if"):
            cond = "  [ask only if " + "; ".join("%s in %s" % (k, "/".join(v)) for k, v in q["show_if"].items()) + "]"
        print("%s. %s%s" % (q["id"], t(q["prompt"], lang), cond))
        if q.get("help"):
            print("    " + t(q["help"], lang))
        for o in q["options"]:
            print("    --%s %-8s %s" % (q["id"], o["id"], t(o["label"], lang)))
        print()


def main(argv=None):
    p = argparse.ArgumentParser(description="Recommend a RAG setup from the study's decision rules.")
    for q in QIDS:
        p.add_argument("--" + q, help="answer id for " + q)
    p.add_argument("--json", help="answers as a JSON object, or a path to a JSON file")
    p.add_argument("--format", choices=["markdown", "json"], default="markdown")
    p.add_argument("--lang", choices=["en", "fa"], default="en")
    p.add_argument("--no-evidence", action="store_true", help="omit the evidence details section")
    p.add_argument("--questions", action="store_true", help="print the eight interview questions and exit")
    a = p.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    chooser = load("chooser.json")
    if a.questions:
        print_questions(chooser, a.lang)
        return 0
    raw = {}
    if a.json:
        src = a.json
        if os.path.exists(src):
            with open(src, encoding="utf-8") as f:
                src = f.read()
        raw.update(json.loads(src))
        raw = raw.get("answers", raw)
    for q in QIDS:
        if getattr(a, q) is not None:
            raw[q] = getattr(a, q)
    try:
        ans, state, fired = choose(raw, chooser)
    except ValueError as err:
        print("error: " + str(err), file=sys.stderr)
        return 2
    evidence = load("evidence.json")
    if a.format == "json":
        print(json.dumps(as_json(ans, state, fired, chooser, evidence), ensure_ascii=False, indent=2))
    else:
        print(as_markdown(ans, state, fired, chooser, evidence, a.lang, not a.no_evidence))
    return 0


if __name__ == "__main__":
    sys.exit(main())
