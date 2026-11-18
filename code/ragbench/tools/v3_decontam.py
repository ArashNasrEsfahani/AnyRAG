"""Decontamination: what the five test-informed v3 mechanisms are worth ON DEV, by free replay.

Written 2026-09-21 by the decontamination agent. No model call, no GPU, DEV FILES ONLY
(`pilot.jsonl` via `work/runs/v3recall_vec8` and `work/runs/core_dev_wide`, `dev_core.jsonl` /
`dev_core_timeline.jsonl` via `work/runs/core_dev*`). Nothing here opens a test question file.

The five mechanisms disclosed in `approaches/agentic3/defaults.py` under "TEST-SET DISCLOSURE":

  guard_escape, guard_escape_label, rank_by_verdict   designed from work/analysis/V2_FAILURES.md 2.2,
                                                      which judged test rejections against test gold answers
  w_kind_conflict = -0.5, w_label_cover = 1.2         framed by work/analysis/DB_CEILING.md over 26 test lookups

Passes:

  guard    every guard escape recorded on dev, split by source (`definition` / `label`), scored against the DEV
           gold values of its own question: does the escape recover a value the run would otherwise have lost,
           and does any non-escaped accepted row already carry that value?
  order    `rank_by_verdict` replayed: it only bites when the pack is the unverified-candidate one, so the
           13 such dev records are packed both ways through the real `pack_evidence` and the gold-unit coverage
           of the two packs compared.
  weights  `w_kind_conflict` / `w_label_cover` replayed over all 64 wide-dev records at v3's values and at
           `kb/query.py`'s, measuring selected-pool and packed gold-unit coverage.
  gate     the recorded `verify_gate` ablation (guard escape + verdict order off together) on the narrow dev set.

    .venv/Scripts/python.exe -m ragbench.tools.v3_decontam --pass guard
    .venv/Scripts/python.exe -m ragbench.tools.v3_decontam --pass all
"""
import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "work" / "runs"
QDIR = ROOT / "ragbench" / "data" / "questions"

# (label, run directory, run file, question file). DEV ONLY.
WIDE = ("wide", RUNS / "v3recall_vec8", "runs_AGENTIC3.jsonl", "pilot.jsonl")
NARROW = [
    ("core_dev", RUNS / "core_dev", "runs_AGENTIC3.jsonl", ("dev_core.jsonl", "dev_core_timeline.jsonl")),
    ("core_dev_tl", RUNS / "core_dev_pilot_timeline", "runs_AGENTIC3.jsonl", "pilot_timeline.jsonl"),
]
GATE = [
    ("core_dev", RUNS / "core_dev", "runs_AGENTIC3.jsonl", "runs_AGENTIC3_verify_gate-180d21d2.jsonl"),
    ("core_dev_tl", RUNS / "core_dev_pilot_timeline", "runs_AGENTIC3.jsonl",
     "runs_AGENTIC3_verify_gate-180d21d2.jsonl"),
]

SCALES = [10.0 ** k for k in (0, 3, -3, 6, -6, 9, -9, 12, -12)]
REL = 5e-4


# ---------------------------------------------------------------------------------------------------------------
# gold values (DEV questions only)
# ---------------------------------------------------------------------------------------------------------------

def load_questions(name):
    """`name` may be one dev question file or a tuple of them. DEV FILES ONLY."""
    names = (name,) if isinstance(name, str) else tuple(name)
    out = {}
    for n in names:
        if "test" in n:
            raise SystemExit("refusing to open a test question file: %s" % n)
        for line in (QDIR / n).open(encoding="utf-8"):
            q = json.loads(line)
            out[q["id"]] = q
    return out


def gold_values(q):
    """Every numeric value the DEV question calls correct: the answer value, the per-language answer values,
    the numeric key facts and (timeline) every period's value. -> set of floats."""
    vals = set()
    if q.get("answer_value") is not None:
        vals.add(float(q["answer_value"]))
    m = q.get("meta") or {}
    for v in (m.get("answer_values") or {}).values():
        if isinstance(v, dict) and v.get("value") is not None:
            try:
                vals.add(float(v["value"]))
            except (TypeError, ValueError):
                pass
    for kf in (m.get("key_facts") or []):
        if isinstance(kf, dict) and kf.get("kind") == "number" and kf.get("value") is not None:
            try:
                vals.add(float(kf["value"]))
            except (TypeError, ValueError):
                pass
    for it in (m.get("answer_series") or []):
        for key in ("value", "value_base"):
            if it.get(key) is not None:
                try:
                    vals.add(float(it[key]))
                except (TypeError, ValueError):
                    pass
    return vals


def matches_gold(value, golds):
    """Scale-aware: value == gold * 10^k, k in {0, +-3, +-6, +-9, +-12}, within 5e-4 relative (DB_CEILING rule)."""
    if value is None:
        return False
    for g in golds:
        if g == 0:
            continue
        for s in SCALES:
            t = g * s
            if abs(value - t) <= REL * max(abs(t), 1e-12):
                return True
    return False


def gold_units(q, d_lang):
    return set((q.get("gold_units") or {}).get(d_lang) or [])


# ---------------------------------------------------------------------------------------------------------------
# records
# ---------------------------------------------------------------------------------------------------------------

def read_runs(d, name):
    p = d / name
    return [json.loads(l) for l in p.open(encoding="utf-8")]


def escaped_items(rec):
    """-> [(verify item, batch)] for every row the guard escape rescued."""
    out = []
    for b in ((rec["trace"]["stages"].get("verify") or {}).get("batches") or []):
        for it in (b.get("items") or []):
            if it.get("guard_escape"):
                out.append(it)
    return out


def accepted_items(rec):
    out = []
    for b in ((rec["trace"]["stages"].get("verify") or {}).get("batches") or []):
        for it in (b.get("items") or []):
            if it.get("verdict") == "accepted":
                out.append(it)
    return out


def cand_index(rec):
    return {c["cid"]: c for c in (rec["trace"]["stages"]["retrieve"].get("candidates") or [])}


# ---------------------------------------------------------------------------------------------------------------
# pass: guard
# ---------------------------------------------------------------------------------------------------------------

def pass_guard(sets):
    print("\n# guard_escape / guard_escape_label on DEV\n")
    grand = Counter()
    detail = []
    for label, d, runfile, qfile in sets:
        qs = load_questions(qfile)
        recs = read_runs(d, runfile)
        n_rec = len(recs)
        c = Counter()
        for rec in recs:
            q = qs.get(rec["question_id"])
            if q is None:
                continue
            golds = gold_values(q)
            gunits = gold_units(q, rec["d_lang"])
            esc = escaped_items(rec)
            acc = accepted_items(rec)
            plain_acc = [a for a in acc if not a.get("guard_escape")]
            plain_vals = [a.get("value_num") for a in plain_acc if a.get("value_num") is not None]
            cands = cand_index(rec)
            if esc:
                c["records_with_escape"] += 1
                if not plain_acc:
                    c["records_escape_is_only_acceptance"] += 1
            c["records"] += 1
            for it in esc:
                c["escapes"] += 1
                c["by_%s" % it["guard_escape"]] += 1
                v = it.get("value_num")
                is_gold = matches_gold(v, golds)
                dup = any(v is not None and pv is not None and abs(pv - v) <= REL * max(abs(v), 1e-12)
                          for pv in plain_vals)
                cu = set((cands.get(it["cid"]) or {}).get("unit_ids") or [])
                at_gold_unit = bool(cu & gunits)
                c["gold_value" if is_gold else "other_value"] += 1
                if is_gold:
                    c["gold_%s" % it["guard_escape"]] += 1
                    c["gold_dup" if dup else "gold_new"] += 1
                else:
                    c["other_dup" if dup else "other_new"] += 1
                if at_gold_unit:
                    c["at_gold_unit"] += 1
                detail.append({"set": label, "qid": rec["question_id"], "setting": rec["setting"],
                               "src": it["guard_escape"], "value": v, "gold": is_gold, "dup": dup,
                               "at_gold_unit": at_gold_unit,
                               "definition": (it.get("definition") or "")[:60],
                               "quote": (it.get("quote") or "")[:70]})
        print("## %-12s  %d records" % (label, n_rec))
        print("   records with an escape          %d / %d" % (c["records_with_escape"], c["records"]))
        print("   ... where it is the ONLY accept %d" % c["records_escape_is_only_acceptance"])
        print("   escapes                         %d   (definition %d, label %d)"
              % (c["escapes"], c["by_definition"], c["by_label"]))
        print("   escaped value IS a dev gold val %d   (definition %d, label %d)"
              % (c["gold_value"], c["gold_definition"], c["gold_label"]))
        print("   ... and NO plain accepted row has it  %d" % c["gold_new"])
        print("   escaped value is NOT gold       %d   (already accepted elsewhere %d)"
              % (c["other_value"], c["other_dup"]))
        print("   escaped candidate sits on a gold unit %d" % c["at_gold_unit"])
        print()
        grand.update(c)
    print("## all dev sets pooled")
    print("   escapes %d over %d records; gold %d (new %d, duplicate %d), non-gold %d (duplicate %d)"
          % (grand["escapes"], grand["records_with_escape"], grand["gold_value"], grand["gold_new"],
             grand["gold_dup"], grand["other_value"], grand["other_dup"]))
    print("   by source: definition %d (gold %d), label %d (gold %d)"
          % (grand["by_definition"], grand["gold_definition"], grand["by_label"], grand["gold_label"]))
    print("   records where the escape is the only acceptance: %d" % grand["records_escape_is_only_acceptance"])
    print("\n### every escape that recovered a gold value no other accepted row carried\n")
    for r in detail:
        if r["gold"] and not r["dup"]:
            print("  %-10s %-22s %-6s %-10s value=%-14s unit_gold=%s  def=%s"
                  % (r["set"], r["qid"], r["setting"], r["src"], r["value"], r["at_gold_unit"], r["definition"]))
    print("\n### escapes whose value is NOT a dev gold value and is new to the table\n")
    for r in detail:
        if not r["gold"] and not r["dup"]:
            print("  %-10s %-22s %-6s %-10s value=%-14s def=%s | %s"
                  % (r["set"], r["qid"], r["setting"], r["src"], r["value"], r["definition"], r["quote"]))
    return detail


# ---------------------------------------------------------------------------------------------------------------
# pass: order  (rank_by_verdict)
# ---------------------------------------------------------------------------------------------------------------

VERDICT_ORDER = {"accepted": 0, "relevant": 1, "rejected": 2, "guard": 3}


def by_verdict(cands, verdict_of):
    order = {c.cid: i for i, c in enumerate(cands)}

    def rank(c):
        v = verdict_of.get(c.cid)
        return VERDICT_ORDER.get(v.split(":")[0] if isinstance(v, str) else v, 4)
    return sorted(cands, key=lambda c: (rank(c), order[c.cid]))


def pass_order():
    from ..config import load_config
    from ..contracts import pack_evidence
    from ..corpus.registry import load_set
    from ..kb import ingest as I
    from ..kb.schema import connect
    from .v3_recall_trace import replay

    label, d, runfile, qfile = WIDE
    qs = load_questions(qfile)
    recs = read_runs(d, runfile)
    cfg = load_config()
    con = connect(I.kb_path(cfg), read_only=True)
    sets = {}

    print("\n# rank_by_verdict on DEV (wide set, %d records)\n" % len(recs))
    rows = []
    for rec in recs:
        ev_from = (rec["trace"]["stages"].get("answer") or {}).get("evidence_from")
        if ev_from != "unverified_candidates":
            continue
        q = qs[rec["question_id"]]
        gold = gold_units(q, rec["d_lang"])
        if not gold:
            continue
        verdict_of = {}
        for b in ((rec["trace"]["stages"].get("verify") or {}).get("batches") or []):
            for it in (b.get("items") or []):
                verdict_of[it["cid"]] = it.get("verdict")
        run = replay(con, rec)
        sel = run["selected"]
        cs_name = rec["corpus_set"]
        if cs_name not in sets:
            sets[cs_name] = load_set(cs_name, cfg)
        cs = sets[cs_name]

        def cover(order_):
            items = [c.to_item(meta={"verified": False}) for c in order_]
            for i, it in enumerate(items):
                it.rank = i + 1
            pack = pack_evidence(items, cs, cfg=cfg)
            got = {u for it in pack.items for u in (it.unit_ids or [])}
            return len(got & gold) / len(gold), pack.n_included

        c_on, n_on = cover(by_verdict(sel, verdict_of))
        c_off, n_off = cover(sel)
        rows.append((rec["question_id"], rec["setting"], c_on, c_off, n_on, n_off))
    con.close()
    if not rows:
        print("  no gold-bearing record uses the unverified pack")
        return rows
    on = sum(r[2] for r in rows) / len(rows)
    off = sum(r[3] for r in rows) / len(rows)
    better = sum(1 for r in rows if r[2] > r[3] + 1e-9)
    worse = sum(1 for r in rows if r[3] > r[2] + 1e-9)
    print("  %d gold-bearing records take the unverified pack (of %d)" % (len(rows), len(recs)))
    print("  packed gold-unit coverage   verdict order %.3f   merge order %.3f   (delta %+.3f)"
          % (on, off, on - off))
    print("  records better with the verdict order %d, worse %d, same %d"
          % (better, worse, len(rows) - better - worse))
    print()
    for qid, st, a, b, na, nb in rows:
        flag = "  <-" if abs(a - b) > 1e-9 else ""
        print("   %-22s %-6s verdict %.2f (%d items)   merge %.2f (%d items)%s" % (qid, st, a, na, b, nb, flag))
    print("\n  Over all %d wide-dev records the mechanism can move at most %d of them, so the pooled effect is"
          % (len(recs), len(rows)))
    print("  %.4f of evidence recall (%d records x %+.3f / %d)."
          % ((on - off) * len(rows) / len(recs), len(rows), on - off, len(recs)))
    return rows


# ---------------------------------------------------------------------------------------------------------------
# pass: weights  (w_kind_conflict / w_label_cover)
# ---------------------------------------------------------------------------------------------------------------

ARMS_W = [("v3   (-0.5 / 1.2)", {"w_kind_conflict": -0.5, "w_label_cover": 1.2}),
          ("kb   (-1.5 / 0.8)", {"w_kind_conflict": -1.5, "w_label_cover": 0.8}),
          ("conflict only (-0.5 / 0.8)", {"w_kind_conflict": -0.5, "w_label_cover": 0.8}),
          ("cover only    (-1.5 / 1.2)", {"w_kind_conflict": -1.5, "w_label_cover": 1.2})]


def pass_weights(which=None):
    from ..config import load_config
    from ..contracts import pack_evidence
    from ..corpus.registry import load_set
    from ..kb import ingest as I
    from ..kb.schema import connect
    from .v3_recall_trace import replay

    label, d, runfile, qfile = which or WIDE
    qs = load_questions(qfile)
    recs = read_runs(d, runfile)
    cfg = load_config()
    con = connect(I.kb_path(cfg), read_only=True)
    sets = {}
    per_arm = defaultdict(list)
    per_rec = defaultdict(dict)

    for rec in recs:
        q = qs[rec["question_id"]]
        gold = gold_units(q, rec["d_lang"])
        if not gold:
            continue
        cs_name = rec["corpus_set"]
        if cs_name not in sets:
            sets[cs_name] = load_set(cs_name, cfg)
        cs = sets[cs_name]
        key = (rec["question_id"], rec["setting"])
        for name, over in ARMS_W:
            run = replay(con, rec, opts_override=dict(over))
            sel = run["selected"]
            sel_units = {u for c in sel for u in c.unit_ids}
            items = [c.to_item(meta={"verified": None}) for c in sel]
            for i, it in enumerate(items):
                it.rank = i + 1
            pack = pack_evidence(items, cs, cfg=cfg)
            packed = {u for it in pack.items for u in (it.unit_ids or [])}
            sc = len(sel_units & gold) / len(gold)
            pc = len(packed & gold) / len(gold)
            per_arm[name].append((sc, pc))
            per_rec[key][name] = (sc, pc)
    con.close()
    n = len(per_arm[ARMS_W[0][0]])
    print("\n# w_kind_conflict / w_label_cover on DEV (wide set, %d gold-bearing records)\n" % n)
    print("  %-28s %-14s %-14s" % ("arm", "selected", "packed"))
    for name, _ in ARMS_W:
        v = per_arm[name]
        print("  %-28s %.3f          %.3f" % (name, sum(a for a, _b in v) / n, sum(b for _a, b in v) / n))
    base = ARMS_W[0][0]
    for name, _ in ARMS_W[1:]:
        better = sum(1 for k in per_rec if per_rec[k][base][1] > per_rec[k][name][1] + 1e-9)
        worse = sum(1 for k in per_rec if per_rec[k][name][1] > per_rec[k][base][1] + 1e-9)
        print("  paired on the packed coverage, v3 against %-26s better %2d, worse %2d, same %2d   %s"
              % (name, better, worse, n - better - worse, sign_p(better, worse)))
    print("\n  records where v3's values change the packed coverage (against kb defaults)\n")
    kb = ARMS_W[1][0]
    for k in sorted(per_rec):
        a, b = per_rec[k][base][1], per_rec[k][kb][1]
        if abs(a - b) > 1e-9:
            print("   %-22s %-6s  v3 %.2f   kb %.2f   %s" % (k[0], k[1], a, b, "v3 better" if a > b else "kb better"))
    return per_rec


def sign_p(better, worse):
    n = better + worse
    if n == 0:
        return "p = 1.0"
    k = min(better, worse)
    p = sum(math.comb(n, i) for i in range(0, k + 1)) / (2.0 ** n) * 2
    return "exact sign p = %.3f" % min(1.0, p)


# ---------------------------------------------------------------------------------------------------------------
# pass: gate  (the recorded verify_gate ablation)
# ---------------------------------------------------------------------------------------------------------------

# ---------------------------------------------------------------------------------------------------------------
# pass: v2guard  -- can DEV alone justify the escape rule? Replay v2's own hard-veto rejections.
# ---------------------------------------------------------------------------------------------------------------

V2_ARMS = [("AGENTIC2 wide dev", RUNS / "core_dev_wide", "runs_AGENTIC2.jsonl", "pilot.jsonl"),
           ("AGENTIC3+verify_gate", RUNS / "core_dev", "runs_AGENTIC3_verify_gate-180d21d2.jsonl",
            ("dev_core.jsonl", "dev_core_timeline.jsonl")),
           ("AGENTIC3+verify_gate tl", RUNS / "core_dev_pilot_timeline",
            "runs_AGENTIC3_verify_gate-180d21d2.jsonl", "pilot_timeline.jsonl")]


def term_rx_of(plan):
    import re
    from ..approaches.agentic import planner as P
    rx = []
    for lang in ("fa", "en"):
        for t in (plan.get("terms") or {}).get(lang) or []:
            for p in P.term_patterns(t, lang):
                rx.append(re.compile(p, re.I))
    return rx


def pass_v2guard():
    """The dev-only case for (or against) the escape: over every `quote_without_term` rejection a HARD-VETO arm
    made on dev, does the verifier's own definition carrying a plan term separate correct values from wrong ones?
    Nothing here reads a test file and nothing here reads the v3 analyses."""
    from ..approaches.agentic import verify as V
    from ..normalize import normalize

    print("\n# Can DEV alone justify the guard escape? v2's own hard-veto rejections, replayed\n")
    grand = Counter()
    for label, d, runfile, qfile in V2_ARMS:
        qs = load_questions(qfile)
        recs = read_runs(d, runfile)
        c = Counter()
        rows = []
        for rec in recs:
            q = qs.get(rec["question_id"])
            if q is None:
                continue
            golds = gold_values(q)
            plan = rec["trace"]["stages"].get("plan") or {}
            rx = term_rx_of(plan)
            unit_kind = plan.get("unit") or "count"
            text_by_cid = {}
            for it in (rec.get("retrieved_units") or []):
                cid = (it.get("meta") or {}).get("cid")
                if cid:
                    text_by_cid[cid] = it.get("text") or ""
            fact_by_cid = {cc["cid"]: (((cc.get("features") or {}).get("fact")) or {})
                           for cc in (rec["trace"]["stages"]["retrieve"].get("candidates") or [])}
            for b in ((rec["trace"]["stages"].get("verify") or {}).get("batches") or []):
                for it in (b.get("items") or []):
                    if it.get("verdict") == "accepted":
                        c["accepted"] += 1
                        if matches_gold(it.get("value_num"), golds):
                            c["accepted_gold"] += 1
                    if it.get("verdict") != "guard:quote_without_term":
                        continue
                    c["qwt"] += 1
                    snippet = text_by_cid.get(it["cid"], "")
                    if not snippet:
                        c["no_snippet"] += 1
                        continue
                    ok, _reason, info = V.guard(it, snippet, unit_kind=unit_kind, term_rx=None)
                    if not ok:
                        c["verbatim_fails_anyway"] += 1
                        continue
                    c["verbatim_ok"] += 1
                    v = info.get("value_num")
                    is_gold = matches_gold(v, golds)
                    dfn = normalize(str(it.get("definition") or ""))
                    in_def = bool(dfn) and any(r.search(dfn) for r in rx)
                    lbl = normalize(str(fact_by_cid.get(it["cid"], {}).get("label") or ""))
                    in_lbl = bool(lbl) and any(r.search(lbl) for r in rx)
                    key = ("def" if in_def else ("lbl" if in_lbl else "neither"))
                    c["%s_%s" % (key, "gold" if is_gold else "wrong")] += 1
                    rows.append((rec["question_id"], rec["setting"], key, is_gold, v,
                                 (it.get("definition") or "")[:64]))
        print("## %s  (%d records)" % (label, len(recs)))
        print("   accepted rows %d, of which carry a dev gold value %d" % (c["accepted"], c["accepted_gold"]))
        print("   quote_without_term rejections %d; snippet missing %d; still fail the verbatim guard %d;"
              " eligible for an escape %d" % (c["qwt"], c["no_snippet"], c["verbatim_fails_anyway"],
                                              c["verbatim_ok"]))
        print("   %-34s %6s %6s %s" % ("term stands in", "gold", "wrong", "precision"))
        for key, name in (("def", "the verifier's own definition"), ("lbl", "the fact row's label (def failed)"),
                          ("neither", "nowhere (the guard's own case)")):
            g, w = c["%s_gold" % key], c["%s_wrong" % key]
            p = ("%.2f" % (g / (g + w))) if (g + w) else "-"
            print("   %-34s %6d %6d %s" % (name, g, w, p))
        print()
        for r in rows:
            if r[3]:
                print("     gold recovered: %-22s %-6s via %-8s value=%-12s def=%s" % (r[0], r[1], r[2], r[4], r[5]))
        print()
        grand.update(c)
    g, w = grand["def_gold"], grand["def_wrong"]
    n, nw = grand["neither_gold"], grand["neither_wrong"]
    print("## pooled over every dev hard-veto arm")
    print("   escape-eligible rejections %d: definition carries a term %d (gold %d, precision %s);"
          % (grand["verbatim_ok"], g + w, g, ("%.2f" % (g / (g + w))) if (g + w) else "-"))
    print("   term nowhere %d (gold %d, precision %s) -- this is what the guard is FOR"
          % (n + nw, n, ("%.2f" % (n / (n + nw))) if (n + nw) else "-"))
    print("   base rate: accepted rows carrying a dev gold value %d / %d = %.2f"
          % (grand["accepted_gold"], grand["accepted"],
             grand["accepted_gold"] / max(1, grand["accepted"])))
    return grand


def pass_gate():
    print("\n# the recorded verify_gate ablation (guard escape + verdict order OFF together), narrow dev\n")
    for label, d, base, gated in GATE:
        b = {(r["question_id"], r["setting"]): r for r in read_runs(d, base)}
        g = {(r["question_id"], r["setting"]): r for r in read_runs(d, gated)}
        keys = sorted(set(b) & set(g))
        diff = [k for k in keys if (b[k]["answer_text"] or "").strip() != (g[k]["answer_text"] or "").strip()]
        ab = sum(1 for k in keys if b[k]["abstained"])
        ag = sum(1 for k in keys if g[k]["abstained"])
        esc = sum(b[k]["trace"]["stats"].get("guard_escaped", 0) for k in keys)
        print("  %-12s %d paired records; escapes in the default arm %d; answers that differ %d;"
              " abstains %d -> %d" % (label, len(keys), esc, len(diff), ab, ag))
        for k in diff:
            print("     %-22s %-6s  default: %-40s gated: %s"
                  % (k[0], k[1], (b[k]["answer_text"] or "")[:40].replace("\n", " "),
                     (g[k]["answer_text"] or "")[:40].replace("\n", " ")))


def pass_extras():
    """Two things the escape's cost estimate turns on, both free.

    1. On dev, is the verified table actually the better pack? (V2_FAILURES measured 88% against 57% on TEST.)
    2. On the records the escape fires on, is the pack already truncated? If it is, a row demoted out of the
       table is at risk of being cut rather than merely moved.
    3. The weights on the coarsest measure of all: is ANY gold unit in the selected pool at either setting?
    """
    from ..config import load_config
    from ..kb import ingest as I
    from ..kb.schema import connect
    from .v3_recall_trace import replay

    label, d, runfile, qfile = WIDE
    recs = {(r["question_id"], r["setting"]): r for r in read_runs(d, runfile)}
    kf = {}
    for l in (d / "results_keyfacts" / "scored.jsonl").open(encoding="utf-8"):
        x = json.loads(l)
        kf[(x["question_id"], x["setting"])] = x
    c = Counter()
    trunc = Counter()
    for k, r in recs.items():
        a = r["trace"]["stages"]["answer"]
        ev = a.get("evidence_from")
        c[(ev, bool(kf[k].get("match_correct")))] += 1
        if r["trace"]["stats"].get("guard_escaped", 0) > 0:
            trunc["truncated" if a["n_included"] < a["n_offered"] else "complete"] += 1
    print("\n# extras (wide dev)\n")
    print("  key-fact accuracy by evidence source")
    for ev in ("verified_table", "unverified_candidates"):
        ok, no = c[(ev, True)], c[(ev, False)]
        print("    %-24s %d / %d = %.2f" % (ev, ok, ok + no, ok / max(1, ok + no)))
    print("  packs on the records where the guard escape fired: %s" % dict(trunc))

    qs = load_questions(qfile)
    cfg = load_config()
    con = connect(I.kb_path(cfg), read_only=True)
    print("\n  ANY gold unit in the selected pool, v3 weights against kb/query.py's own")
    for which in [WIDE] + NARROW:
        qq = load_questions(which[3])
        rr = read_runs(which[1], which[2])
        out = {}
        for name, ov in (("v3", {"w_kind_conflict": -0.5, "w_label_cover": 1.2}),
                         ("kb", {"w_kind_conflict": -1.5, "w_label_cover": 0.8})):
            hit = n = 0
            for rec in rr:
                g = gold_units(qq[rec["question_id"]], rec["d_lang"])
                if not g:
                    continue
                n += 1
                sel = replay(con, rec, opts_override=dict(ov))["selected"]
                hit += bool({u for cc in sel for u in cc.unit_ids} & g)
            out[name] = (hit, n)
        print("    %-14s v3 %d/%d   kb %d/%d" % (which[0], out["v3"][0], out["v3"][1], out["kb"][0], out["kb"][1]))
    con.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pass", dest="pass_", default="all",
                    choices=["guard", "v2guard", "order", "weights", "gate", "extras", "all"])
    a = ap.parse_args()
    if a.pass_ in ("guard", "all"):
        pass_guard([WIDE] + NARROW)
    if a.pass_ in ("v2guard", "all"):
        pass_v2guard()
    if a.pass_ in ("order", "all"):
        pass_order()
    if a.pass_ in ("weights", "all"):
        for which in [WIDE] + NARROW:
            pass_weights(which)
    if a.pass_ in ("gate", "all"):
        pass_gate()


if __name__ == "__main__":
    sys.exit(main())
