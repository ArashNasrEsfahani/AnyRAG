"""v4's packing question, answered by replay: how much of the evidence budget is spent on redundant information?

Written 2026-09-21 by the packing agent. NO model call, NO GPU: every number comes from replaying the recorded
wide-dev candidate pools (`v3_recall_trace.replay`, 59/59 fidelity) and from the real corpus text, through the
real `contracts.pack_evidence` and the real 6000-char budget.

THE QUESTION. v3's evidence recall is limited by what survives packing, not by what retrieval finds: the gold unit
is in a targeted document on 59 of 59 gold-bearing dev records and on a selected candidate on 0.812 of them (at
today's production cap 12 and `vec_min_slots` 8), but only 0.769 survives `pack_evidence`. `pack_evidence` renders
items in the given order and STOPS at the first item that does not fit, so the loss is an ORDER and an ITEM-SIZE
loss. Packing order and item text are each approach's own choice (contracts.py, `pack_evidence` docstring), so a
method that knows which candidates carry the same typed fact, and where inside a unit that fact sits, may spend the
same budget on more distinct information. That is the v4 claim, and this module bounds it before anyone builds it.

WHAT IS SIMULATED (strategies, each at cap 12 and cap 24, always `vec_min_slots` = 8):

  base      today: merge order, whole items, pack_evidence stops at the first item that does not fit
  fill      merge order, but an item that does not fit is narrowed to the remaining room instead of stopping the
            pack (exactly what B4 does in `tree_nav.pack_rank_order`; a control, it uses no typed knowledge)
  dedupe    drop a candidate whose typed facts (value_num, unit_norm, year, def_class) are all already carried by
            a kept candidate. `safe` also requires it to add no new unit; `strict` drops it anyway
  narrow    a long candidate is cut to a verbatim window around the facts it carries (tables keep their caption and
            header and lose the rows below the last matching one), unit ids re-derived by the SHARED rule
            (`contracts._units_in_cut`, a unit counts when 60 of its characters or all of it survive)
  greedy    coverage-greedy order: the next item is the one adding the most new units / typed facts / sections,
            not the one with the best score (maximal marginal relevance in typed space)
  oracle    the bound: gold-bearing items first. No run can have it; it says what the order is worth at most

REPORTED per strategy: packed gold-unit coverage, distinct units and distinct typed facts in the pack, items
offered and packed, characters used, and records improved / regressed against `base` at the same cap.

    .venv/Scripts/python.exe -m ragbench.tools.v4_pack_sim
    .venv/Scripts/python.exe -m ragbench.tools.v4_pack_sim --sweep      # the narrowing window sweep
    .venv/Scripts/python.exe -m ragbench.tools.v4_pack_sim --audit 12   # windows and dropped candidates, to read
"""
import argparse
import json
import sys

from ..config import load_config
from ..contracts import RetrievedItem, UNIT_MIN_SHOWN_CHARS, item_header, pack_evidence
from ..corpus.registry import load_set, set_name_for
from ..kb import ingest as I
from ..kb.schema import connect
from ..approaches.tree_nav import narrow_item
from .v3_recall_trace import ARMS, WIDE, gold_units, load_questions, replay

ARM = "AGENTIC3+vectors_always"
CAPS = (12, 24)
VMS = 8                      # production reservation since ROUTE_VERSION agentic3-route-v3
WINDOW = 260                 # default narrowing window, characters each side of the fact block
SWEEP = (120, 180, 260, 400, 600)

FACT_SQL = ("SELECT fact_id, doc_id, unit_id, ord, label, value_text, value_num, unit_norm, unit_kind, year, "
            "def_class, source_kind, char_start, char_end, score FROM facts WHERE unit_id IN (%s) ORDER BY ord")

DIAG = {}        # what narrowing had to work with, filled by the `narrow` strategy at cap 12
LOC = {}         # how well facts.char_start locates a value inside the candidate's own text
LAB = {}         # whether a narrowed window keeps the label of the number it carries
REDUN = {}       # typed redundancy in the selected pool, per cap


# ---------------------------------------------------------------------------------------------------------------
# typed facts on a candidate
# ---------------------------------------------------------------------------------------------------------------

def fact_rows(con, unit_ids, cache):
    """The typed fact rows of these units, cached per unit id."""
    want = [u for u in unit_ids if u not in cache]
    if want:
        for u in want:
            cache[u] = []
        rows = con.execute(FACT_SQL % ",".join("?" * len(want)), want).fetchall()
        for r in rows:
            cache[r["unit_id"]].append(dict(r))
    out = []
    for u in unit_ids:
        out += cache.get(u, [])
    return out


def fact_key(r):
    """The typed identity of a fact: same value, same unit, same period, same definition class."""
    return (r["value_num"], r["unit_norm"] or "", r["year"], r["def_class"] or "")


def verbatim_start(cand, doc):
    """Document offset of the candidate's text when it is a verbatim slice, else None (same test as contracts)."""
    if not cand.span:
        return None
    body = (cand.text or "").strip()
    raw = cand.text or ""
    start = int(cand.span[0]) + (len(raw) - len(raw.lstrip()))
    if not body or doc.text[start:start + len(body)] != body:
        return None
    return start


def fact_positions(cand, doc, rows):
    """-> [(start, end, row)] of each fact inside the candidate's own text.

    Offsets come from `facts.char_start` (an offset into `units.text`, which `repair_separators` keeps
    length-preserving) mapped through the unit's document span; the value string is then confirmed at that
    position, and searched for nearby when the extraction shifted it. A fact whose value cannot be located in the
    candidate's text is dropped rather than guessed."""
    body = cand.text or ""
    start = verbatim_start(cand, doc)
    out = []
    for r in rows:
        val = r["value_text"] or ""
        if not val:
            continue
        guess = None
        if start is not None and r["char_start"] is not None:
            try:
                u = doc.unit(r["unit_id"])
            except KeyError:
                u = None
            if u is not None:
                guess = u.span[0] + int(r["char_start"]) - start
        LOC["rows"] = LOC.get("rows", 0) + 1
        pos = -1
        if guess is not None and 0 <= guess < len(body):
            lo, hi = max(0, guess - 6), min(len(body), guess + len(val) + 6)
            k = body.find(val, lo, hi)
            pos = k
            if k == guess:
                LOC["exact"] = LOC.get("exact", 0) + 1
            elif k >= 0:
                LOC["drifted"] = LOC.get("drifted", 0) + 1
        if pos < 0:
            pos = body.find(val)
            if pos >= 0:
                LOC["by_search"] = LOC.get("by_search", 0) + 1
        if pos < 0:
            LOC["lost"] = LOC.get("lost", 0) + 1
            continue
        out.append((pos, pos + len(val), r))
    return out


class Cand:
    """A replayed candidate with its typed facts located inside its own text."""

    def __init__(self, c, doc, rows):
        self.c = c
        self.doc = doc
        self.rows = rows
        self.pos = fact_positions(c, doc, rows)
        self.keys = {fact_key(r) for _s, _e, r in self.pos}
        self.units = set(c.unit_ids)
        self.chars = len((c.text or "").strip())
        self.verbatim = verbatim_start(c, doc) is not None
        self.is_table = c.kind == "table"


# ---------------------------------------------------------------------------------------------------------------
# strategy 1: fact dedupe
# ---------------------------------------------------------------------------------------------------------------

def dedupe(cands, mode):
    """Drop a candidate whose typed facts are all carried by a candidate already kept.
    `safe` keeps it when it still reaches a unit no kept candidate reaches. -> (kept, dropped)."""
    kept, dropped, keys, units = [], [], set(), set()
    for x in cands:
        if x.keys and x.keys <= keys and (mode == "strict" or x.units <= units):
            dropped.append(x)
            continue
        kept.append(x)
        keys |= x.keys
        units |= x.units
    return kept, dropped


# ---------------------------------------------------------------------------------------------------------------
# strategy 2: narrowing to the fact window
# ---------------------------------------------------------------------------------------------------------------

def _snap_left(body, i):
    """Back to the start of the line, or of the word, so a window never starts inside a token."""
    nl = body.rfind("\n", 0, i)
    if nl >= 0 and i - nl <= 400:
        return nl + 1
    sp = body.rfind(" ", max(0, i - 40), i)
    return sp + 1 if sp >= 0 else i


def _snap_right(body, i):
    nl = body.find("\n", i)
    if 0 <= nl - i <= 400:
        return nl
    sp = body.find(" ", i, i + 40)
    return sp if sp >= 0 else i


def narrow_to_facts(x, window, min_chars):
    """The candidate cut to a verbatim window that still carries every typed fact it was selected for.

    A TABLE keeps its head (caption, header row, the rows above the last matching one) and loses what is below:
    a table row without its header is a number without a column label. Any other candidate keeps a window of
    `window` characters each side of the block its facts occupy. Only verbatim candidates are narrowed, so the
    kept text is still a slice of `Document.text` and the unit ids can be re-derived exactly, by the same rule
    `pack_evidence` uses for a cut item. -> RetrievedItem or None (nothing to save)."""
    c = x.c
    body = (c.text or "").strip()
    if not x.verbatim or not x.pos or len(body) <= min_chars:
        return None
    lo = min(s for s, _e, _r in x.pos)
    hi = max(e for _s, e, _r in x.pos)
    if x.is_table:
        a, b = 0, _snap_right(body, min(len(body), hi + 40))
    else:
        a = _snap_left(body, max(0, lo - window))
        b = _snap_right(body, min(len(body), hi + window))
    if b - a >= len(body) - 24 or b - a < min_chars:
        return None
    _label_audit(x, body, a, b)
    return slice_item(c, x.doc, body, a, b)


def _label_audit(x, body, a, b):
    """Does the window still carry the LABEL that makes its number readable? A number whose label was cut is
    evidence the model cannot use, and the unit-id metric would not notice."""
    from ..normalize import fold
    win = fold(body[a:b])
    for s, e, r in x.pos:
        if not (a <= s and e <= b):
            LAB["fact_dropped"] = LAB.get("fact_dropped", 0) + 1
            continue
        # `facts.label` is a bag of the label's content words, not a contiguous phrase (the L0 extractor
        # assembles it), so the test is how many of those words the window still shows.
        words = [w for w in fold(r["label"] or "").split() if len(w) > 1]
        LAB["fact_kept"] = LAB.get("fact_kept", 0) + 1
        if not words:
            LAB["no_label"] = LAB.get("no_label", 0) + 1
            return
        have = sum(1 for w in words if w in win) / len(words)
        if have >= 0.8:
            LAB["label_kept"] = LAB.get("label_kept", 0) + 1
        elif have > 0:
            LAB["label_partial"] = LAB.get("label_partial", 0) + 1
        else:
            LAB["label_lost"] = LAB.get("label_lost", 0) + 1
            if len(LAB.setdefault("examples", [])) < 8:
                LAB["examples"].append((r["value_text"], (r["label"] or "")[:60]))


def slice_item(c, doc, body, a, b):
    """`body[a:b]` as a verbatim RetrievedItem with honest unit ids (contracts' own rule for a cut)."""
    start = verbatim_start(c, doc)
    text = body[a:b].strip()
    if not text:
        return None
    off = start + a + (len(body[a:b]) - len(body[a:b].lstrip()))
    keep = []
    want = set(c.unit_ids)
    end = off + len(text)
    for u in doc.units_overlapping(off, end):
        if u.id not in want:
            continue
        shown = min(u.span[1], end) - max(u.span[0], off)
        if shown >= min(UNIT_MIN_SHOWN_CHARS, u.span[1] - u.span[0]):
            keep.append(u.id)
    keep += [uid for uid in c.unit_ids if not doc.has_unit(uid) and uid not in keep]
    unit_ids = [uid for uid in c.unit_ids if uid in set(keep)]
    return RetrievedItem(text=text, doc_id=c.doc_id, section_id=c.section_id, unit_ids=unit_ids, kind="window",
                         score=c.score, span=[off, off + len(text)],
                         meta={"cid": c.cid, "narrowed": "fact_window", "from_chars": len(body)})


# ---------------------------------------------------------------------------------------------------------------
# strategy 3: coverage-greedy order
# ---------------------------------------------------------------------------------------------------------------

def greedy_order(cands, per_char=False):
    """Order by marginal NEW typed information: units, facts and sections nothing chosen so far reaches."""
    rest = list(cands)
    units, keys, secs = set(), set(), set()
    out = []
    while rest:
        best, bi = None, 0
        for i, x in enumerate(rest):
            gain = 2.0 * len(x.units - units) + 1.0 * len(x.keys - keys) + 1.0 * (x.c.section_id not in secs)
            if per_char:
                gain = gain / max(1.0, x.chars / 300.0)
            if best is None or gain > best:
                best, bi = gain, i
        x = rest.pop(bi)
        out.append(x)
        units |= x.units
        keys |= x.keys
        secs.add(x.c.section_id)
    return out


# ---------------------------------------------------------------------------------------------------------------
# packing
# ---------------------------------------------------------------------------------------------------------------

def to_item(x):
    it = x.c.to_item(meta={"verified": None})
    return it


def build_items(cands, cs, cfg, *, narrow=None, fill=False, min_chars=200, budget=None, hardcap=None, diag=None):
    """The item list an approach would hand to `pack_evidence`, in this order.

    `narrow` applies the typed fact window first. `fill` then walks the list with the budget arithmetic
    `pack_evidence` itself uses and shrinks an item that does not fit to the remaining room (B4's
    `pack_rank_order`), so `pack_evidence` includes every item unchanged instead of stopping at the first
    one that does not fit."""
    budget = int(budget or cfg.evidence["max_chars"])
    items = []
    for x in cands:
        it = None
        if narrow:
            it = narrow_to_facts(x, narrow, min_chars)
            if diag is not None:
                diag["n"] = diag.get("n", 0) + 1
                diag["chars_before"] = diag.get("chars_before", 0) + x.chars
                diag["chars_after"] = diag.get("chars_after", 0) + (len(it.text) if it is not None else x.chars)
                if it is not None:
                    diag["shrunk"] = diag.get("shrunk", 0) + 1
                    diag["saved"] = diag.get("saved", 0) + (x.chars - len(it.text))
                elif not x.verbatim:
                    diag["not_verbatim"] = diag.get("not_verbatim", 0) + 1
                elif not x.pos:
                    diag["no_located_fact"] = diag.get("no_located_fact", 0) + 1
                else:
                    diag["nothing_to_save"] = diag.get("nothing_to_save", 0) + 1
        it = it if it is not None else to_item(x)
        if hardcap and len((it.text or "").strip()) > hardcap:
            sm = narrow_item(it, cs, hardcap, min_chars)
            if sm is not None:
                it = sm
        items.append(it)
    if not fill:
        return items
    out, used = [], 0
    for it in items:
        body = (it.text or "").strip()
        if not body:
            continue
        sep = 2 if out else 0
        room = budget - used - sep - len(item_header(len(out) + 1, it, cs)) - 1
        if room < min_chars:
            break
        new = it if len(body) <= room else narrow_item(it, cs, room, min_chars)
        if new is None:
            continue
        nb = (new.text or "").strip()
        used += sep + len(item_header(len(out) + 1, new, cs)) + 1 + len(nb)
        out.append(new)
    return out


def pack(items, cs, cfg):
    for i, it in enumerate(items):
        it.rank = i + 1
    p = pack_evidence(items, cs, cfg=cfg)
    return p


# ---------------------------------------------------------------------------------------------------------------
# strategies
# ---------------------------------------------------------------------------------------------------------------

def strategies(window):
    return [
        ("base", dict()),
        ("fill", dict(fill=True)),
        ("dedupe_safe", dict(dd="safe")),
        ("dedupe_strict", dict(dd="strict")),
        ("narrow", dict(narrow=window)),
        ("greedy", dict(order="greedy")),
        ("narrow+fill", dict(narrow=window, fill=True)),
        ("dedupe+narrow+fill", dict(dd="safe", narrow=window, fill=True)),
        ("dedupe+narrow+greedy+fill", dict(dd="safe", narrow=window, order="greedy", fill=True)),
        ("greedy+fill", dict(order="greedy", fill=True)),
        ("greedy_pc+narrow+fill", dict(order="greedy_pc", narrow=window, fill=True)),
        ("hardcap 250+fill", dict(hardcap=250, fill=True)),
        ("hardcap 350+fill", dict(hardcap=350, fill=True)),
        ("hardcap 500+fill", dict(hardcap=500, fill=True)),
        ("dedupe+narrow+hardcap 350+fill", dict(dd="safe", narrow=window, hardcap=350, fill=True)),
        ("ORACLE order", dict(order="oracle")),
        ("ORACLE+narrow+fill", dict(order="oracle", narrow=window, fill=True)),
    ]


def apply_strategy(cands, gold, cs, cfg, spec, window=WINDOW, diag=None):
    xs = list(cands)
    dropped = []
    if spec.get("dd"):
        xs, dropped = dedupe(xs, spec["dd"])
    order = spec.get("order")
    if order == "greedy":
        xs = greedy_order(xs)
    elif order == "greedy_pc":
        xs = greedy_order(xs, per_char=True)
    elif order == "oracle":
        xs = [x for _k, _i, x in sorted((0 if (gold & x.units) else 1, i, x)
                                        for i, x in enumerate(xs))]
    items = build_items(xs, cs, cfg, narrow=spec.get("narrow"), fill=spec.get("fill", False), budget=None,
                        hardcap=spec.get("hardcap"), diag=diag)
    p = pack(items, cs, cfg)
    return items, p, dropped


# ---------------------------------------------------------------------------------------------------------------
# the run
# ---------------------------------------------------------------------------------------------------------------

def gold_of_pack(p, gold):
    got = {u for it in p.items for u in (it.unit_ids or [])}
    return len(gold & got) / len(gold), got


def answer_in_pack(p, q, d_lang, gold=None):
    """Is the gold ANSWER itself inside the rendered evidence block?

    Evidence recall counts a unit id. A truncated item can still declare a unit (60 of its characters survive,
    the shared rule) while the sentence that carries the number is gone, so a strategy that buys unit ids by
    cutting items could buy nothing at all. This is the necessary condition the unit id stands for, scored with
    the SAME matchers the answer is scored with: for a numeric answer, a number in the block equal to
    `answer_value`; otherwise the gold answer string. It is a ceiling, not an accuracy: one of the 6000
    characters matching does not mean the model will read it."""
    from ..eval.metrics import numeric_match, string_match
    if gold is None:
        txt = p.text or ""
    else:                       # only the items that DECLARE a gold unit: does the counted unit carry the answer?
        txt = "\n".join((it.text or "") for it in p.items if gold & set(it.unit_ids or []))
    at = q.get("answer_type")
    other = "en" if d_lang == "fa" else "fa"
    gold_texts = [q.get("answer_" + d_lang), q.get("answer_" + other)]
    if at == "numeric" and q.get("answer_value") is not None:
        return 1.0 if numeric_match(txt, q["answer_value"], q.get("answer_unit"),
                                    gold_texts[0] or gold_texts[1])["match"] else 0.0
    if at == "unanswerable":
        return None
    m = string_match(txt, gold_texts)
    return 1.0 if m.get("contains") else 0.0


def gold_shown(p, gold, cs):
    """-> [(unit id, characters of it the pack shows, its length)] for the gold units the pack declares."""
    out = []
    for it in p.items:
        g = gold & set(it.unit_ids or [])
        if not g:
            continue
        for uid in sorted(g):
            try:
                u = cs.unit(uid)
            except KeyError:
                continue
            n = len(u.text)
            if it.span:
                shown = max(0, min(u.span[1], int(it.span[1])) - max(u.span[0], int(it.span[0])))
            else:
                shown = n if (u.text in (it.text or "")) else min(n, len(it.text or ""))
            out.append((uid, shown, n))
    return out


def run(caps=CAPS, window=WINDOW, sweep=False, audit=0, limit=None):
    cfg = load_config()
    qs = load_questions()
    con = connect(I.kb_path(cfg), read_only=True)
    recs = [json.loads(l) for l in (WIDE / ARMS[ARM]).open(encoding="utf-8")]
    sets, fcache = {}, {}
    rows = []
    audited = 0
    for rec in recs:
        q = qs[rec["question_id"]]
        gold = set(gold_units(q, rec["d_lang"]))
        if not gold:
            continue
        name = set_name_for(rec["corpus"], rec["d_lang"])
        if name not in sets:
            sets[name] = load_set(name, cfg)
        cs = sets[name]
        multi = len(rec["trace"]["stages"]["retrieve"]["docs"]) > 1
        r = {"qid": rec["question_id"], "setting": rec["setting"], "n_gold": len(gold), "qtype": q.get("type")}
        for cap in caps:
            res = replay(con, rec, opts_override={"max_candidates": cap, "max_candidates_multi": cap * 2,
                                                  "vec_min_slots": VMS},
                         cap_override=cap * 2 if multi else cap)
            sel = res["selected"]
            cands = []
            for c in sel:
                try:
                    doc = cs.doc(c.doc_id)
                except KeyError:
                    continue
                cands.append(Cand(c, doc, fact_rows(con, c.unit_ids, fcache)))
            r["sel_%d" % cap] = len(gold & {u for x in cands for u in x.units}) / len(gold)
            r["ncand_%d" % cap] = len(cands)
            R = REDUN.setdefault(cap, {})
            R["facts"] = R.get("facts", 0) + sum(len(x.pos) for x in cands)
            R["distinct"] = R.get("distinct", 0) + len(set().union(*[x.keys for x in cands]) if cands else set())
            seen, rep = set(), 0
            for x in cands:                       # a fact key a LATER candidate repeats
                rep += len(x.keys & seen)
                seen |= x.keys
            R["repeated_across"] = R.get("repeated_across", 0) + rep
            R["records"] = R.get("records", 0) + 1
            specs = strategies(window)
            if sweep:
                specs = [("base", dict())] + [("narrow+fill w=%d" % w, dict(narrow=w, fill=True)) for w in SWEEP] \
                    + [("dedupe+narrow+fill w=%d" % w, dict(dd="safe", narrow=w, fill=True)) for w in SWEEP]
            for nm, spec in specs:
                d = DIAG if (nm == "narrow" and cap == 12) else None
                items, p, dropped = apply_strategy(cands, gold, cs, cfg, spec, window=window, diag=d)
                cov, got = gold_of_pack(p, gold)
                k = "%s@%d" % (nm, cap)
                keys = set()
                for it in p.items:
                    for x in cands:
                        if (it.meta or {}).get("cid") == x.c.cid:
                            keys |= x.keys
                r[k] = cov
                a = answer_in_pack(p, q, rec["d_lang"])
                if a is not None:
                    r[k + "|ans"] = a
                    r[k + "|ansg"] = answer_in_pack(p, q, rec["d_lang"], gold=gold)
                    if q.get("answer_type") == "numeric":
                        r[k + "|ansnum"] = a
                sh = gold_shown(p, gold, cs)
                r[k + "|thin"] = sum(1 for _u, s, ln in sh if s < min(150, ln)) / len(gold)
                r[k + "|units"] = len(got)
                r[k + "|keys"] = len(keys)
                r[k + "|off"] = p.n_offered
                r[k + "|in"] = p.n_included
                r[k + "|chars"] = p.chars
                if nm.startswith("dedupe"):
                    r[k + "|dropped"] = len(dropped)
                    r[k + "|dropped_gold"] = sum(1 for d in dropped if gold & d.units)
                    r[k + "|dropped_gold_only"] = sum(
                        1 for d in dropped
                        if (gold & d.units) and not any((gold & d.units) & y.units for y in cands
                                                        if y is not d and y not in dropped))
            if audit and audited < audit and cap == 12:
                audited += 1
                r["_audit"] = audit_record(cands, gold, cs, cfg, window)
        rows.append(r)
        if limit and len(rows) >= limit:
            break
    con.close()
    return rows


def audit_record(cands, gold, cs, cfg, window):
    """What narrowing actually does to the text, for reading by a human."""
    out = []
    for x in cands:
        it = narrow_to_facts(x, window, 200)
        if it is None:
            continue
        body = (x.c.text or "").strip()
        out.append({"cid": x.c.cid, "kind": x.c.kind, "gold": bool(gold & x.units),
                    "from": len(body), "to": len(it.text), "units_before": len(x.units),
                    "units_after": len(it.unit_ids),
                    "gold_lost": sorted((gold & x.units) - set(it.unit_ids)),
                    "text": it.text})
    return out


# ---------------------------------------------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------------------------------------------

def report(rows, caps=CAPS, window=WINDOW, sweep=False):
    n = len(rows)
    avg = lambda k: sum(r.get(k, 0) for r in rows) / n                     # noqa: E731
    specs = strategies(window)
    if sweep:
        specs = [("base", dict())] + [("narrow+fill w=%d" % w, dict()) for w in SWEEP] \
            + [("dedupe+narrow+fill w=%d" % w, dict()) for w in SWEEP]
    print("\n# v4 packing simulation - %d gold-bearing wide-dev records, vec_min_slots = %d, budget 6000 chars\n" % (n, VMS))
    nans = sum(1 for r in rows if ("base@%d|ans" % caps[0]) in r)
    nnum = sum(1 for r in rows if ("base@%d|ansnum" % caps[0]) in r)
    print("answer-in-pack is scored on %d records (%d of them numeric); it asks only whether the gold answer is"
          " SOMEWHERE in the 6000 characters, which a pack holding more numbers passes more easily, so read it"
          " next to 'typed facts'. The GOLD-items column removes that confound: it asks the same question of the"
          " packed items that declare a gold unit." % (nans, nnum))
    for cap in caps:
        print("\n## cap = %d   (selected gold coverage %.3f, the ceiling any packing can reach)\n"
              % (cap, avg("sel_%d" % cap)))
        print("| strategy | packed gold | of headroom | answer in pack | ... in the GOLD items | numeric only |"
              " distinct units | typed facts | offered | packed | chars | better | worse |")
        print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        b = "base@%d" % cap
        ceil = avg("sel_%d" % cap)
        for nm, _spec in specs:
            k = "%s@%d" % (nm, cap)
            better = sum(1 for r in rows if r.get(k, 0) > r.get(b, 0) + 1e-9)
            worse = sum(1 for r in rows if r.get(k, 0) < r.get(b, 0) - 1e-9)
            head = ceil - avg(b)
            frac = "-" if nm == "base" or head <= 1e-9 else "%.0f%%" % (100.0 * (avg(k) - avg(b)) / head)
            mean = lambda s: (sum(v for v in s) / len(s)) if s else float("nan")        # noqa: E731
            na = [r[k + "|ans"] for r in rows if (k + "|ans") in r]
            ng = [r[k + "|ansg"] for r in rows if (k + "|ansg") in r]
            nn = [r[k + "|ansnum"] for r in rows if (k + "|ansnum") in r]
            print("| %s | %.3f | %s | %.3f | %.3f | %.3f | %.1f | %.1f | %.1f | %.1f | %.0f | %d | %d |"
                  % (nm, avg(k), frac, mean(na), mean(ng), mean(nn),
                     avg(k + "|units"), avg(k + "|keys"), avg(k + "|off"), avg(k + "|in"),
                     avg(k + "|chars"), better, worse))
        for nm in ("dedupe_safe", "dedupe_strict"):
            k = "%s@%d" % (nm, cap)
            if (k + "|dropped") in (rows[0] if rows else {}):
                print("\n%s: %.1f candidates dropped per record (%d records with >=1), gold-bearing dropped on %d"
                      " records, and the LAST copy of a gold unit dropped on %d"
                      % (nm, avg(k + "|dropped"), sum(1 for r in rows if r.get(k + "|dropped")),
                         sum(1 for r in rows if r.get(k + "|dropped_gold")),
                         sum(1 for r in rows if r.get(k + "|dropped_gold_only"))))
    if DIAG.get("n"):
        d = DIAG
        print("\n## what narrowing had to work with (every candidate of every record, cap 12)\n")
        print("candidates %d; the fact window shrank %d of them (%.0f%%), saving %.0f chars each on average;"
              " %d were not verbatim (a split table), %d carried no locatable typed fact, and on %d the window"
              " around the facts was already the whole candidate."
              % (d["n"], d.get("shrunk", 0), 100.0 * d.get("shrunk", 0) / d["n"],
                 (d.get("saved", 0) / max(1, d.get("shrunk", 0))), d.get("not_verbatim", 0),
                 d.get("no_located_fact", 0), d.get("nothing_to_save", 0)))
        print("mean candidate %.0f chars before narrowing, %.0f after: the whole strategy frees %.0f chars of the"
              " 6000-char budget per record (%.1f%% of it)."
              % (d["chars_before"] / d["n"], d["chars_after"] / d["n"],
                 (d["chars_before"] - d["chars_after"]) / max(1, n), 100.0 * (d["chars_before"] - d["chars_after"])
                 / max(1, n) / 6000.0))
    for cap, R in sorted(REDUN.items()):
        if not R.get("records"):
            continue
        print("\nredundancy in the selection at cap %d: %.1f located typed facts per record, %.1f of them distinct"
              " (value, unit, period, definition class); %.1f are a key a candidate earlier in the same selection"
              " already carried (%.0f%%)."
              % (cap, R["facts"] / R["records"], R["distinct"] / R["records"],
                 R["repeated_across"] / R["records"], 100.0 * R["repeated_across"] / max(1, R["facts"])))
    if LAB.get("fact_kept"):
        tot = LAB["fact_kept"] + LAB.get("fact_dropped", 0)
        print("\nnarrowing and the label: %d typed facts sat in the narrowed candidates; the window kept %d of them"
              " (%.3f) and cut %d out. Of the kept ones, %d still show their own label inside the window (%.3f),"
              " %d have no label at all in the fact row, and %d lost it."
              % (tot, LAB["fact_kept"], LAB["fact_kept"] / tot, LAB.get("fact_dropped", 0),
                 LAB.get("label_kept", 0), LAB.get("label_kept", 0) / max(1, LAB["fact_kept"]),
                 LAB.get("no_label", 0), LAB.get("label_lost", 0)))
        print("   (label words shown: >=80%% on %d, some on %d, none on %d)"
              % (LAB.get("label_kept", 0), LAB.get("label_partial", 0), LAB.get("label_lost", 0)))
        for v, l in (LAB.get("examples") or [])[:5]:
            print("   label cut: value %-14s label %r" % (v, l))
    if LOC.get("rows"):
        t = LOC["rows"]
        print("\nfact offsets: %d fact rows read; the stored char_start landed on the value %d times (%.3f),"
              " drifted by a few characters %d, needed a text search %d, and could not be located at all %d."
              % (t, LOC.get("exact", 0), LOC.get("exact", 0) / t, LOC.get("drifted", 0), LOC.get("by_search", 0),
                 LOC.get("lost", 0)))
    # where the damage is
    print("\n## by number of gold units (packed coverage, cap 12)\n")
    print("| gold units | records | base | narrow+fill | dedupe+narrow+greedy+fill | ORACLE order |")
    print("|---|---|---|---|---|---|")
    for lo, hi, lab in ((1, 1, "1"), (2, 3, "2-3"), (4, 8, "4-8"), (9, 99, "9+")):
        g = [r for r in rows if lo <= r["n_gold"] <= hi]
        if not g:
            continue
        m = lambda k: sum(r.get(k, 0) for r in g) / len(g)                 # noqa: E731
        print("| %s | %d | %.3f | %.3f | %.3f | %.3f |"
              % (lab, len(g), m("base@12"), m("narrow+fill@12"), m("dedupe+narrow+greedy+fill@12"),
                 m("ORACLE order@12")))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", type=int, default=WINDOW)
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--audit", type=int, default=0)
    ap.add_argument("--caps", default=",".join(str(c) for c in CAPS))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--json-out", default=None)
    a = ap.parse_args(argv)
    caps = tuple(int(x) for x in a.caps.split(","))
    rows = run(caps=caps, window=a.window, sweep=a.sweep, audit=a.audit, limit=a.limit)
    report(rows, caps=caps, window=a.window, sweep=a.sweep)
    if a.audit:
        print("\n## narrowing audit (real windows, cap 12)\n")
        for r in rows:
            for e in (r.get("_audit") or []):
                print("\n--- %s %s %s cid=%s kind=%s %d -> %d chars, units %d -> %d%s"
                      % (r["qid"], r["setting"], "GOLD" if e["gold"] else "", e["cid"], e["kind"], e["from"],
                         e["to"], e["units_before"], e["units_after"],
                         (" LOST GOLD %s" % e["gold_lost"]) if e["gold_lost"] else ""))
                print(e["text"])
    if a.json_out:
        from pathlib import Path
        Path(a.json_out).write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
        print("\nwrote %s" % a.json_out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
