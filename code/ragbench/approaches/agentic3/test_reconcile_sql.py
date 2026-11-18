"""Offline tests for the SQL reconciler, the gap queries and the write-back (no model, throwaway database).

    .venv/Scripts/python.exe -m ragbench.approaches.agentic3.test_reconcile_sql

  differential  the acceptance bar, over a FIXED and named case set: `reconcile-diff-2026-09-21`, 709 recoverable
                cases (comparison 150, single 322, timeline 237) in 50 pinned run files. The set is a manifest,
                not a glob, so the number behind the claim cannot drift with whatever runs are on the machine;
                `CASE_SET` above lists every file and the cases it must contribute, and the suite fails if any
                one of them moves. Every timeline, comparison and single-fact question of those files is
                replayed: `trace.structured.rows_all` holds the rows that went into reconciliation
                and `trace.stages.plan` holds the plan the qualifier rule needs, so the real verified row set of
                each historical run can be rebuilt. `reconcile_sql.reconcile` must return the SAME series, roles,
                restatements, projections, narrower rows, conflicts, definition classes, comparable flag, trend and
                notes as `agentic/reconcile.py` on every one of them. Two extra checks run alongside: that the
                rebuilt rows reproduce the run's OWN stored output (proving the recovery is faithful, not just
                self-consistent - `differential_tests` says exactly which part of that output the recovery can
                and cannot reach, and why), and that reading the document years from the live knowledge base
                instead of from the rows changes nothing.
  hand          ground truth built by hand where history has no example: a definition change (registered -> active)
                that must make the series not comparable and state no trend, a restatement conflict, a restatement
                inside the tolerance that is not one, a toman/rial restatement that only the typed columns can see
                is the same number, the latest-restatement fallback, and a projection kept out of the series.
  gaps          a throwaway knowledge base with three yearly reports: the missing year must come back as a query
                naming that year's own report AND the next year's, with a regex built from the verified labels,
                and the query must actually find the seeded fact.
  labels        `label_pattern` folds the literals, not the finished pattern, and the trap it avoids.
  writeback     only guard-passed rows are written, a re-run writes nothing new, warm mode is off by default, and a
                document whose content hash has moved on becomes unreadable rather than silently trusted. The warm
                READ path (`kb/query.py::verified_lookup`) is exercised on the same rows: what the writer accepts
                is what the reader finds, `ok_label_completed` included, and a row with no guard at all is a miss.

No question file is read anywhere here; the historical cases are run OUTPUTS and are used only as fixtures for an
equality assertion, never to choose a threshold.
"""
import collections
import glob
import json
import os
import shutil
import sys

from ...config import load_config
from ...kb import query as KQ
from ...kb import schema as S
from ..agentic import reconcile as R
from ..agentic.approach import _Session
from . import reconcile_sql as Q
from . import writeback as W

FAILS = []

# ---------------------------------------------------------------------------------------------------------------
# THE DIFFERENTIAL CASE SET IS FIXED AND NAMED (2026-09-21, truth pass).
#
# It used to be three globs - core_test_timeline, core_dev*, v2dev_* - and the number of cases behind the claim
# "reconcile_sql agrees with agentic/reconcile.py on every historical case" therefore grew with whatever run
# records happened to be on the machine. It was 555 when this file's docstring was written, 709 once the wide dev
# set landed, and it would have moved again on the next run. A claim whose denominator moves is not a claim, so
# the set is pinned: these files, these counts, this total, everywhere.
#
# CASE_SET_NAME is what a write-up should cite. To extend the set, run the suite, paste the new manifest it prints
# on failure, bump CASE_SET_NAME, and update every write-up that quotes the old number in the same commit.
CASE_SET_NAME = "reconcile-diff-2026-09-21"
CASE_SET_TOTAL = 709                     # recoverable cases: comparison 150, single 322, timeline 237
CASE_SET_KINDS = {"comparison": 150, "single": 322, "timeline": 237}
CASE_SET = (
    ("work/runs/core_dev/runs_AGENTIC.jsonl", 15),
    ("work/runs/core_dev/runs_AGENTIC2.jsonl", 19),
    ("work/runs/core_dev/runs_AGENTIC2_candidates_32-4d15082f.jsonl", 19),
    ("work/runs/core_dev/runs_AGENTIC2_no_factdb-ff863e47.jsonl", 16),
    ("work/runs/core_dev/runs_AGENTIC2_no_tree-0e3c238a.jsonl", 18),
    ("work/runs/core_dev/runs_AGENTIC2_no_vectors-f2749e7b.jsonl", 17),
    ("work/runs/core_dev/runs_AGENTIC3.jsonl", 18),
    ("work/runs/core_dev/runs_AGENTIC3_no_facets-3559a662.jsonl", 16),
    ("work/runs/core_dev/runs_AGENTIC3_no_fastpath-84e03f84.jsonl", 18),
    ("work/runs/core_dev/runs_AGENTIC3_no_sql_reconcile-14137a75.jsonl", 18),
    ("work/runs/core_dev/runs_AGENTIC3_no_tree_prior-d92851a6.jsonl", 17),
    ("work/runs/core_dev/runs_AGENTIC3_no_trigram-c8438eb4.jsonl", 18),
    ("work/runs/core_dev/runs_AGENTIC3_no_typed-a1b5f05f.jsonl", 16),
    ("work/runs/core_dev/runs_AGENTIC3_no_vectors-8b4d3773.jsonl", 18),
    ("work/runs/core_dev/runs_AGENTIC3_vectors_always-7afdfb27.jsonl", 18),
    ("work/runs/core_dev/runs_AGENTIC3_verify_gate-180d21d2.jsonl", 18),
    ("work/runs/core_dev_pilot_timeline/runs_AGENTIC.jsonl", 7),
    ("work/runs/core_dev_pilot_timeline/runs_AGENTIC2.jsonl", 7),
    ("work/runs/core_dev_pilot_timeline/runs_AGENTIC2_candidates_32-4d15082f.jsonl", 7),
    ("work/runs/core_dev_pilot_timeline/runs_AGENTIC2_no_factdb-ff863e47.jsonl", 7),
    ("work/runs/core_dev_pilot_timeline/runs_AGENTIC2_no_tree-0e3c238a.jsonl", 8),
    ("work/runs/core_dev_pilot_timeline/runs_AGENTIC2_no_vectors-f2749e7b.jsonl", 7),
    ("work/runs/core_dev_pilot_timeline/runs_AGENTIC3.jsonl", 7),
    ("work/runs/core_dev_pilot_timeline/runs_AGENTIC3_no_facets-3559a662.jsonl", 8),
    ("work/runs/core_dev_pilot_timeline/runs_AGENTIC3_no_fastpath-84e03f84.jsonl", 7),
    ("work/runs/core_dev_pilot_timeline/runs_AGENTIC3_no_sql_reconcile-14137a75.jsonl", 7),
    ("work/runs/core_dev_pilot_timeline/runs_AGENTIC3_no_tree_prior-d92851a6.jsonl", 7),
    ("work/runs/core_dev_pilot_timeline/runs_AGENTIC3_no_trigram-c8438eb4.jsonl", 7),
    ("work/runs/core_dev_pilot_timeline/runs_AGENTIC3_no_typed-a1b5f05f.jsonl", 6),
    ("work/runs/core_dev_pilot_timeline/runs_AGENTIC3_no_vectors-8b4d3773.jsonl", 7),
    ("work/runs/core_dev_pilot_timeline/runs_AGENTIC3_vectors_always-7afdfb27.jsonl", 7),
    ("work/runs/core_dev_pilot_timeline/runs_AGENTIC3_verify_gate-180d21d2.jsonl", 7),
    ("work/runs/core_dev_wide/runs_AGENTIC2.jsonl", 52),
    ("work/runs/core_dev_wide/runs_AGENTIC3.jsonl", 51),
    ("work/runs/core_dev_wide/runs_AGENTIC3_vectors_always-7afdfb27.jsonl", 51),
    ("work/runs/core_test_timeline/runs_AGENTIC.jsonl", 9),
    ("work/runs/core_test_timeline/runs_AGENTIC2.jsonl", 9),
    ("work/runs/v2dev_smoke/runs_AGENTIC2.jsonl", 1),
    ("work/runs/v2dev_t1/runs_AGENTIC2.jsonl", 17),
    ("work/runs/v2dev_t1_tl/runs_AGENTIC2.jsonl", 6),
    ("work/runs/v2dev_t2/runs_AGENTIC2.jsonl", 17),
    ("work/runs/v2dev_t2_tl/runs_AGENTIC2.jsonl", 7),
    ("work/runs/v2dev_t3/runs_AGENTIC2.jsonl", 17),
    ("work/runs/v2dev_t3_tl/runs_AGENTIC2.jsonl", 7),
    ("work/runs/v2dev_t4/runs_AGENTIC2.jsonl", 17),
    ("work/runs/v2dev_t4_tl/runs_AGENTIC2.jsonl", 7),
    ("work/runs/v2dev_t5/runs_AGENTIC2.jsonl", 17),
    ("work/runs/v2dev_t5_tl/runs_AGENTIC2.jsonl", 8),
    ("work/runs/v2dev_t6/runs_AGENTIC2.jsonl", 17),
    ("work/runs/v2dev_t6_tl/runs_AGENTIC2.jsonl", 7),
)
# The globs the set was FROZEN from, kept so a reader can see what it covers and what it does not. Directories
# that match them but are not in CASE_SET (a later dev run, a recall arm) are deliberately outside the set.
CASE_SET_FROM = ("work/runs/core_test_timeline/runs_AGENTIC*.jsonl",
                 "work/runs/core_dev*/runs_AGENTIC*.jsonl",
                 "work/runs/v2dev_*/runs_AGENTIC*.jsonl")

COMPARE_KEYS = ("kind", "restatements", "conflicts", "projections", "unknown_period",
                "latest_restatement_periods", "narrower", "periods_found", "periods_missing",
                "definition_classes", "class_points", "several_values_periods", "comparable", "main_class", "trend")
ROW_KEYS = ("period", "value", "unit", "definition", "quote", "doc_id", "value_num", "def_class", "role",
            "period_year", "period_inferred", "cid", "qualifiers")


def check(cond, label):
    print(("  ok   " if cond else "  FAIL ") + label)
    if not cond:
        FAILS.append(label)


def rowkeys(rows):
    return [tuple(json.dumps(r.get(k), ensure_ascii=False, sort_keys=True) for k in ROW_KEYS) for r in rows]


def compare(a, b):
    """-> list of (key, from v1, from SQL) for everything that is not identical."""
    diff = []
    for k in COMPARE_KEYS:
        x, y = a.get(k), b.get(k)
        if k in ("restatements", "projections", "unknown_period", "narrower"):
            x, y = rowkeys(x), rowkeys(y)
        if x != y:
            diff.append((k, x, y))
    if rowkeys(a["series"]) != rowkeys(b["series"]):
        diff.append(("series", rowkeys(a["series"]), rowkeys(b["series"])))
    return diff


# ---------------------------------------------------------------------------------------------------------------
# differential: every recoverable historical case
# ---------------------------------------------------------------------------------------------------------------

def run_files(root):
    """The files of the FIXED case set, in manifest order. Missing files are reported, never skipped silently."""
    out, missing = [], []
    for rel, _n in CASE_SET:
        p = os.path.join(root, rel)
        (out if os.path.exists(p) else missing).append(p if os.path.exists(p) else rel)
    check(not missing, "every file of case set %s is on disk (%d missing%s)"
          % (CASE_SET_NAME, len(missing), (": " + ", ".join(missing[:3])) if missing else ""))
    return out


def case_set_drift(root):
    """Run files that the frozen globs match but the case set does not hold. Information, not a failure."""
    known = {os.path.normpath(os.path.join(root, rel)) for rel, _n in CASE_SET}
    seen = {os.path.normpath(p) for g in CASE_SET_FROM for p in glob.glob(os.path.join(root, g))}
    return sorted(os.path.relpath(p, root).replace(os.sep, "/") for p in seen - known)


def recover(record):
    """One run record -> (rows, kind, target_periods, opts, q_lang, known_stems, stored structured).

    `rows_all` is `approach._series_row` of the rows reconciliation was given, so every field the reconciler reads
    is there except `period_year`, which is the period string back as an int, and `score`, the verification rank -
    see `WHAT THE RECOVERY LOSES` in `differential_tests`.

    `known_stems` is the run's own switch, not a fixed recipe. AGENTIC3's `_Session3.narrowing_applies` passes
    `known_stems=None` to the reconciler on a FACETED plan whose rows carry no period at all, which turns the
    narrow rule off; the run records that as `trace.stats.narrow_suppressed`. Replaying such a record with the v1
    recipe rebuilds a DIFFERENT input and silently loses the question's second topic (38 of the 709 cases of
    `reconcile-diff-2026-09-21`, all of them AGENTIC3 records).
    """
    st = record["trace"]["structured"]
    rows = []
    for d in st["rows_all"]:
        r = dict(d)
        r["period_year"] = int(r["period"]) if r.get("period") else None
        r["doc_years"] = list(r["doc_years"]) if r.get("doc_years") else None
        r.setdefault("score", None)
        rows.append(r)
    stems = _Session.known_stems(None, record["trace"]["stages"]["plan"])
    if int(((record.get("trace") or {}).get("stats") or {}).get("narrow_suppressed") or 0):
        stems = None
    return (rows, st["kind"], [int(p) for p in st["target_periods"]], record["trace"]["params"],
            record["q_lang"], stems, st)


def rank_decides(rows):
    """True when the missing `score` can change the output: the reconciler sorts a period's rows by `score` before
    it dedupes them, so a period holding two or more rows has a representative and an order the record does not
    keep. A period with at most one row has neither."""
    seen = collections.Counter(r.get("period_year") for r in rows)
    return any(v > 1 for v in seen.values())


def period_values(rows):
    """{period: {value identified by its number when it has one}} - the part of a series that does not depend on
    which of several equal readings was picked, nor on the order they were listed in."""
    out = collections.defaultdict(set)
    for r in rows:
        v = r.get("value_num")
        out[r.get("period")].add(("num", round(float(v), 9)) if v is not None else ("text", r.get("value")))
    return out


def differential_tests(root, kb):
    """WHAT THE RECOVERY LOSES, and what is therefore asserted.

    A run record keeps every field the reconciler reads except one: `score`, the verification rank. The reconciler
    uses it in exactly one place - `sorted(kept, key=-score)` before the per-period dedupe - and that decides two
    things: WHICH of several rows carrying the same number represents its period (they differ in wording, and
    therefore in `qualifiers`, and therefore in whether the narrow rule then removes the row), and in WHAT ORDER a
    period's rows are listed, which the "several values" note repeats verbatim. It is not stored anywhere in the
    record - `trace.candidates` is null on these runs - so it is not recoverable, and 34 of the 387 timeline and
    comparison cases of `reconcile-diff-2026-09-21` reproduce the run's own output only up to that choice.

    So the faithfulness of the recovery is asserted in the two places where it is well defined:

      1. exact, on the cases where the rank cannot decide anything (no period holds two rows): the rebuilt rows
         must give back the run's own series AND notes, character for character.
      2. rank-free, on ALL of them: every value the run itself reported comes back for the same period, and the
         periods found, the periods missing, the (period, role, definition class) set, `comparable` and `trend`
         are identical.

    Both hold on every case of the fixed set (82/82 and 387/387 on 2026-09-21); the exact count over all
    timeline and comparison cases, 353/387, is printed as information, not asserted. What IS the acceptance bar
    is the line above it: `reconcile_sql` and `agentic/reconcile.py` agree on all 709 cases of
    `reconcile-diff-2026-09-21`, rank or no rank, because both see the same rows.

    THE COUNT USED TO DRIFT. This docstring said 555 while the suite printed 709, because the set was three globs
    and grew with every new run directory. It is now the pinned manifest `CASE_SET`, and 709 is the one number.
    """
    files = run_files(root)
    n = same = faithful = faithful_n = kb_same = kb_n = 0
    exact_det = exact_det_n = rankfree = 0
    by_kind, first, loose, per_file = {}, None, [], {}
    for f in files:
        for line in open(f, encoding="utf-8"):
            rec = json.loads(line)
            st = (rec.get("trace") or {}).get("structured") or {}
            if not st.get("rows_all"):
                continue
            n += 1
            per_file[f] = per_file.get(f, 0) + 1
            rows, kind, tp, opts, q_lang, stems, stored = recover(rec)
            by_kind[kind] = by_kind.get(kind, 0) + 1
            v1 = R.reconcile([dict(r) for r in rows], kind=kind, target_periods=tp, opts=opts, q_lang=q_lang,
                             known_stems=stems)
            sq = Q.reconcile([dict(r) for r in rows], kind=kind, target_periods=tp, opts=opts, q_lang=q_lang,
                             known_stems=stems)
            diff = compare(v1, sq)
            if R.table_notes(v1, q_lang) != R.table_notes(sq, q_lang):
                diff.append(("notes", R.table_notes(v1, q_lang), R.table_notes(sq, q_lang)))
            if diff:
                first = first or (os.path.basename(f), rec["question_id"], diff[:2])
            else:
                same += 1
            # the recovery itself: does v1 on the rebuilt rows reproduce what the run wrote down?
            if kind != "single":          # a single-fact table is ordered by `score`, which rows_all does not keep
                faithful_n += 1
                got = [(r["period"], r["value"], r["role"], r["def_class"]) for r in v1["series"]]
                want = [(r["period"], r["value"], r["role"], r["def_class"]) for r in stored["series"]]
                exact = got == want and R.table_notes(v1, q_lang) == stored["notes"]
                faithful += exact
                if not rank_decides(rows):
                    exact_det_n += 1
                    exact_det += exact
                    if not exact:
                        loose.append((os.path.basename(f), rec["question_id"], rec.get("setting"), "exact",
                                      got, want))
                gv, wv = period_values(v1["series"]), period_values(stored["series"])
                ok = (set(wv) <= set(gv) and all(wv[p] <= gv[p] for p in wv)
                      and {(r[0], r[2], r[3]) for r in got} == {(r[0], r[2], r[3]) for r in want}
                      and v1["periods_found"] == stored["periods_found"]
                      and v1["periods_missing"] == stored["periods_missing"]
                      and v1["comparable"] == stored["comparable"] and v1["trend"] == stored["trend"])
                rankfree += ok
                if not ok:
                    loose.append((os.path.basename(f), rec["question_id"], rec.get("setting"), "rank-free",
                                  got, want))
            if kb is not None:
                kb_n += 1
                kbq = Q.reconcile([dict(r) for r in rows], kind=kind, target_periods=tp, opts=opts, q_lang=q_lang,
                                  known_stems=stems, kb=kb)
                if not compare(sq, kbq):
                    kb_same += 1
    print("  case set %s: %d run files, %d recoverable cases (%s)"
          % (CASE_SET_NAME, len(files), n, ", ".join("%s %d" % kv for kv in sorted(by_kind.items()))))
    moved = [(rel, want, per_file.get(os.path.join(root, rel), 0))
             for rel, want in CASE_SET if per_file.get(os.path.join(root, rel), 0) != want]
    check(not moved, "every file of the case set contributes the number of cases it is pinned at (%d moved%s)"
          % (len(moved), (": " + ", ".join("%s %d->%d" % m for m in moved[:3])) if moved else ""))
    check(n == CASE_SET_TOTAL, "the case set is exactly %d cases, the number every write-up quotes (%d)"
          % (CASE_SET_TOTAL, n))
    check(by_kind == CASE_SET_KINDS, "and they are %s (%s)" % (CASE_SET_KINDS, by_kind))
    drift = case_set_drift(root)
    if drift:
        print("  ..     %d run file(s) match the frozen globs but are OUTSIDE the case set, by design: %s"
              % (len(drift), ", ".join(drift[:4])))
    check(same == n, "reconcile_sql matches agentic/reconcile.py on every case of %s (%d/%d)"
          % (CASE_SET_NAME, same, n))
    if first:
        print("    first difference:", json.dumps(first, ensure_ascii=False, default=str)[:900])
    print("  recovery: %d/%d timeline and comparison cases reproduce the run's own series and notes character for "
          "character; the rest differ only where the unrecorded verification rank decides" % (faithful, faithful_n))
    check(exact_det == exact_det_n,
          "where the verification rank cannot decide anything, the rebuilt rows reproduce the run's own series and "
          "notes exactly (%d/%d cases with at most one row per period)" % (exact_det, exact_det_n))
    check(rankfree == faithful_n,
          "and on every case the rank-free part is identical: each value the run reported comes back for its "
          "period, with the same roles, classes, periods found and missing, comparable and trend (%d/%d)"
          % (rankfree, faithful_n))
    for x in loose[:3]:
        print("    recovery difference:", json.dumps(x, ensure_ascii=False, default=str)[:700])
    if kb is None:
        print("  ..     no knowledge base on disk: the documents-table source of the role CASE was not cross-checked")
    else:
        check(kb_same == kb_n,
              "documents.year_from/year_to gives the same roles as the rows' own doc_years (%d/%d)" % (kb_same, kb_n))
    return n


# ---------------------------------------------------------------------------------------------------------------
# hand-built ground truth
# ---------------------------------------------------------------------------------------------------------------

DOCS = {1400: "U.acme_annual-report_1400.fa", 1401: "U.acme_annual-report_1401.fa",
        1402: "U.acme_annual-report_1402.fa", 1403: "U.acme_annual-report_1403.fa"}


class _Cand:
    def __init__(self, doc_id, cid, score):
        self.doc_id, self.cid, self.score = doc_id, cid, score
        self.section_id, self.unit_ids, self.round = doc_id + "::s1", [doc_id + "::u1"], 1


def row(doc_year, period, value, value_num, definition, *, unit="", cid="c1", score=3.0):
    """One verified row exactly as `reconcile.make_row` builds it, from the report of `doc_year`."""
    doc_id = DOCS[doc_year]
    item = {"value": value, "unit": unit, "period": str(period), "definition": definition, "quote": definition}
    ginfo = {"value_num": value_num, "value_token": value, "reading": "as_written"}
    return R.make_row(item, _Cand(doc_id, cid, score), ginfo, {"years": [doc_year, doc_year]}, doc_id + "::u1")


def both(rows, *, kind="timeline", target_periods=(), q_lang="fa", opts=None):
    """Reconcile the same rows twice and fail if the two disagree; return the SQL result."""
    opts = opts or {}
    v1 = R.reconcile([dict(r) for r in rows], kind=kind, target_periods=list(target_periods), opts=opts,
                     q_lang=q_lang, known_stems=None)
    sq = Q.reconcile([dict(r) for r in rows], kind=kind, target_periods=list(target_periods), opts=opts,
                     q_lang=q_lang, known_stems=None)
    diff = compare(v1, sq)
    if R.table_notes(v1, q_lang) != R.table_notes(sq, q_lang):
        diff.append(("notes", None, None))
    check(not diff, "hand case: SQL and v1 agree (%s)" % ([d[0] for d in diff] or "identical"))
    return sq


def hand_tests():
    print("\ndefinition change (registered -> active)")
    rec = both([row(1400, 1400, "1,000,000", 1000000.0, "کاربران "
                    "ثبت‌نام‌شده", cid="c1"),
                row(1401, 1401, "1,200,000", 1200000.0, "کاربران "
                    "فعال", cid="c2")],
               target_periods=(1400, 1401))
    check([r["def_class"] for r in rec["series"]] == ["registered", "active"],
          "the two rows are classed registered and active")
    check(rec["definition_classes"] == {"registered": ["1400"], "active": ["1401"]},
          "definition_classes shows the change (%s)" % rec["definition_classes"])
    check(rec["comparable"] is False, "comparable is False")
    check(rec["trend"] is None, "no trend is stated")
    notes = R.table_notes(rec, "fa")
    check(any(n.startswith(R.TEXT["fa"]["not_comparable"][:30]) for n in notes),
          "the Persian not-comparable note is stated verbatim")

    print("\nrestatement conflict")
    rec = both([row(1400, 1400, "500", 500.0, "کاربران فعال",
                    unit="نفر", cid="c1"),
                row(1402, 1400, "560", 560.0, "کاربران فعال",
                    unit="نفر", cid="c2")],
               target_periods=(1400, 1402))
    check([r["value"] for r in rec["series"]] == ["500"], "the authoritative 1400 value stays in the series")
    check([r["role"] for r in rec["series"]] == ["authoritative"], "its role is authoritative")
    check(len(rec["conflicts"]) == 1 and rec["conflicts"][0]["kept"] == ["500"]
          and rec["conflicts"][0]["restated"] == "560", "v1's conflict is flagged (%s)" % rec["conflicts"])
    d = rec["conflict_detail"]
    check(len(d) == 1 and d[0]["reason"] == "conflict" and d[0]["kept_kind"] == "people"
          and d[0]["kept_doc"] != d[0]["restated_doc"] and d[0]["in_v1_conflicts"],
          "the refined rule flags it too, with the unit kind and the two documents (%s)" % d)
    check(rec["periods_missing"] == ["1402"], "1402 has no value and is missing (%s)" % rec["periods_missing"])

    print("\na restatement inside the tolerance is not a conflict")
    rec = both([row(1400, 1400, "500", 500.0, "کاربران فعال",
                    unit="نفر", cid="c1"),
                row(1402, 1400, "501", 501.0, "کاربران فعال",
                    unit="نفر", cid="c2")])
    check(rec["conflicts"] == [] and rec["conflict_detail"] == [], "nothing is flagged")

    print("\na toman restatement of a rial figure: only the typed columns see they agree")
    rec = both([row(1400, 1400, "500", 500.0, "درآمد",
                    unit="میلیون ریال", cid="c1"),
                row(1402, 1400, "50", 50.0, "درآمد",
                    unit="میلیون تومان", cid="c2")])
    check(len(rec["conflicts"]) == 1, "v1 still flags it (the two value_num differ by ten times)")
    check(rec["conflict_detail"] == [], "the refined rule does not: 500 million rial IS 50 million toman")
    check(len(rec["conflicts_resolved"]) == 1 and rec["conflicts_resolved"][0]["reason"] == "values_agree"
          and rec["conflicts_resolved"][0]["kept_cmp"] == rec["conflicts_resolved"][0]["restated_cmp"],
          "and says why, with both rial equivalents (%s)" % rec["conflicts_resolved"])

    print("\nlatest restatement, projection, and a period with no report of its own")
    rec = both([row(1401, 1399, "300", 300.0, "کاربران", cid="c1"),
                row(1403, 1399, "310", 310.0, "کاربران", cid="c2"),
                row(1401, 1403, "900", 900.0, "کاربران", cid="c3")],
               target_periods=(1399, 1403))
    check([(r["value"], r["role"]) for r in rec["series"]] == [("310", "latest_restatement")],
          "1399 takes the value of the LATEST restating report (%s)"
          % [(r["value"], r["role"]) for r in rec["series"]])
    check(rec["latest_restatement_periods"] == [1399], "and the period is recorded as such")
    check([r["value"] for r in rec["projections"]] == ["900"],
          "the 1401 report's statement about 1403 is a projection and is kept out of the series")
    check([r["value"] for r in rec["restatements"]] == ["300"], "the older restatement is only noted")
    check(len(rec["conflicts"]) == 1, "and it conflicts with the kept 310")
    check(rec["periods_missing"] == ["1403"], "1403 is missing: a projection is not a value")


# ---------------------------------------------------------------------------------------------------------------
# gap queries and label patterns, against a throwaway knowledge base
# ---------------------------------------------------------------------------------------------------------------

LABEL = "تعداد کاربران فعال"


def seed_kb(path, cfg):
    con = S.connect(path)
    S.ensure_schema(con, cfg)
    for y, doc_id in DOCS.items():
        con.execute("INSERT INTO documents (doc_id, corpus, key, lang, company, year_from, year_to, calendar, "
                    "title, title_fold, content_sha1, quality, status) "
                    "VALUES (?, 'U', ?, 'fa', 'acme', ?, ?, 'jalali', ?, ?, ?, 'ok', 'ready')",
                    (doc_id, doc_id.split(".")[1], y, y, "acme %d" % y, "acme %d" % y, "sha-%d" % y))
    con.execute("INSERT INTO documents (doc_id, corpus, key, lang, company, year_from, year_to, calendar, title, "
                "title_fold, content_sha1, quality, status) VALUES "
                "('U.other_annual-report_1401.fa', 'U', 'other_annual-report_1401', 'fa', 'other', 1401, 1401, "
                "'jalali', 'other 1401', 'other 1401', 'sha-o', 'ok', 'ready')")
    from ...normalize import fold, normalize
    con.execute("INSERT INTO facts (doc_id, section_id, unit_id, ord, label, label_fold, value_text, value_num, "
                "unit_norm, unit_kind, year, year_source, def_class, source_kind, quote, score) "
                "VALUES (?, ?, ?, 0, ?, ?, '1,100,000', 1100000.0, 'person', 'people', 1401, 'own', 'active', "
                "'text', ?, 4.0)",
                (DOCS[1401], DOCS[1401] + "::s1", DOCS[1401] + "::u1", LABEL, fold(normalize(LABEL)),
                 LABEL + " 1,100,000"))
    con.commit()
    return con


def gap_tests(con):
    rows = [row(1400, 1400, "1,000,000", 1000000.0, LABEL, cid="c1"),
            row(1402, 1402, "1,300,000", 1300000.0, LABEL, cid="c2")]
    rec = Q.reconcile(rows, kind="timeline", target_periods=[1400, 1401, 1402], q_lang="fa", kb=con)
    check(rec["periods_missing"] == ["1401"] and rec["periods_covered"] == ["1400", "1402"],
          "1401 is the gap (%s missing, %s covered)" % (rec["periods_missing"], rec["periods_covered"]))
    gaps = Q.gap_queries(rec, kb=con)
    check(len(gaps) == 1 and gaps[0]["year"] == 1401, "one follow-up query, for 1401")
    g = gaps[0]
    roles = {d["doc_id"]: d["gap_role"] for d in g["docs"]}
    check(roles == {DOCS[1401]: "own", DOCS[1402]: "next"},
          "it names the 1401 report (own) and the 1402 report, where a restatement would live (%s)" % roles)
    check("U.other_annual-report_1401.fa" not in roles, "another company's 1401 report is not in it")
    check(bool(g["pattern"]) and g["labels"] and g["labels"][0]["text"] == LABEL,
          "the regex is built from the wording of the verified rows (%s)" % g["pattern"])
    check(g["runnable"], "the query is runnable")
    hits = Q.run_gap_query(con, g)
    check(len(hits) == 1 and hits[0]["value_text"] == "1,100,000" and hits[0]["year"] == 1401,
          "and it finds the 1401 fact the series was missing (%s)" % [h["value_text"] for h in hits])
    check(Q.run_gap_query(con, dict(g, runnable=False)) == [],
          "a query with no document or no label runs nothing")
    empty = Q.gap_queries(Q.reconcile(rows, kind="timeline", target_periods=[1400, 1402], q_lang="fa", kb=con),
                          kb=con)
    check(empty == [], "a complete series asks no follow-up question")


def label_tests():
    import re

    from ..agentic import planner as P
    from ...normalize import fold
    phrase = "شهرهای فعال"            # active cities
    pat = Q.label_pattern(phrase)
    check(bool(re.search(pat, fold(phrase))), "label_pattern matches its own folded phrase")
    check(not re.fullmatch(pat, ""), "and does not match the empty string")
    check(not re.search(pat, fold("فعال")), "nor the last word on its own")
    broken = fold(P.term_patterns(phrase, "fa")[0])
    check(bool(re.search(broken, fold("فعال"))),
          "the trap it avoids: folding a FINISHED pattern turns the first word into an optional group")
    check(Q.label_pattern(LABEL) == Q.label_pattern("کاربران فعال"),
          "a counter word is dropped but the metric word is not (%s)" % Q.label_pattern(LABEL))
    check(Q.label_pattern("تعداد کل") != "",
          "a phrase made only of counter words keeps them rather than returning nothing")
    check(Q.label_pattern("  ") == "", "an empty label makes no pattern")


# ---------------------------------------------------------------------------------------------------------------
# write-back
# ---------------------------------------------------------------------------------------------------------------

def writeback_tests(con):
    plan = {"entity": "acme", "metric": LABEL, "kind": "timeline", "terms": {"fa": [LABEL], "en": ["active users"]}}
    h = W.question_hash("acme users?", plan)
    check(h == W.question_hash("acme users?", plan) and len(h) == 12, "question_hash is stable and short")
    check(h != W.question_hash("acme revenue?", plan), "and depends on the question")
    rows = [dict(row(1400, 1400, "1,000,000", 1000000.0, LABEL, cid="c1"), guard="ok"),
            dict(row(1401, 1401, "1,100,000", 1100000.0, LABEL, cid="c2"), guard="ok_label_completed"),
            dict(row(1402, 1402, "1,300,000", 1300000.0, LABEL, cid="c3"), guard="quote_not_found"),
            dict(row(1403, 1403, "1,400,000", 1400000.0, LABEL, cid="c4"))]
    rows[1]["unit"] = "نفر"
    w = W.write_rows(con, rows, plan=plan, run_id="t1", model="m", prompt_version="v", question_hash=h)
    check(w["written"] == 2, "only the two guard-passed rows are written (%s)" % w)
    check(w["skipped"] == {"guard_rejected": 1, "no_guard": 1},
          "a rejected guard and a missing guard are both skipped, and counted apart (%s)" % w["skipped"])
    again = W.write_rows(con, rows, plan=plan, run_id="t2", model="m", prompt_version="v", question_hash=h)
    check(again["written"] == 0 and again["skipped"].get("already_written") == 2,
          "a second run of the same question writes nothing new (%s)" % again)
    check(W.read_rows(con, doc_ids=list(DOCS.values())) == [], "warm mode is OFF by default")
    warm = W.read_rows(con, doc_ids=list(DOCS.values()), warm=True)
    check(len(warm) == 2 and {r["value_text"] for r in warm} == {"1,000,000", "1,100,000"},
          "with warm=True the two rows come back (%s)" % [r["value_text"] for r in warm])
    check([r["unit_norm"] for r in warm if r["value_text"] == "1,100,000"] == ["person"],
          "the unit was typed on the way in")
    check(len(W.read_rows(con, doc_ids=list(DOCS.values()), years=(1401, 1402), warm=True)) == 1,
          "the year filter works")
    check(len(W.read_rows(con, doc_ids=[DOCS[1400]], warm=True)) == 1, "so does the document filter")
    check(W.stale_rows(con) == 0, "nothing is stale yet")
    con.execute("UPDATE documents SET content_sha1 = 'sha-new' WHERE doc_id = ?", (DOCS[1400],))
    con.commit()
    check(W.stale_rows(con) == 1, "re-ingesting a document makes its verified row stale")
    warm = W.read_rows(con, doc_ids=list(DOCS.values()), warm=True)
    check([r["value_text"] for r in warm] == ["1,100,000"],
          "and the stale row is unreadable, not silently trusted (%s)" % [r["value_text"] for r in warm])
    check(W.counts(con) == {"rows": 2, "live": 1, "stale": 1, "questions": 1},
          "the counts for the warm-mode report (%s)" % W.counts(con))
    fresh = W.write_rows(con, rows[:1], plan=plan, run_id="t3", model="m", prompt_version="v", question_hash=h)
    check(fresh["written"] == 1, "the same row verified again under the NEW hash is written as a new row")
    check(W.write_rows(con, [dict(rows[0], doc_id="U.nosuch.fa")], plan=plan, question_hash=h)["skipped"]
          == {"unknown_document": 1}, "a row from a document the knowledge base does not hold is skipped")

    # the warm READ path the agent actually calls (kb/query.py), on the rows just written -----------------------
    terms = KQ.Terms(plan)
    check(tuple(KQ.guard_ok_values()) == W.GUARD_OK,
          "the read path takes its guard rule from writeback.GUARD_OK (%s)" % (KQ.guard_ok_values(),))
    off, offinfo = KQ.verified_lookup(con, list(DOCS.values()), terms)
    check(off == [] and offinfo["enabled"] is False, "verified_lookup is OFF unless the warm flag is set")
    got, info = KQ.verified_lookup(con, list(DOCS.values()), terms, enabled=True)
    vals = sorted(r["value_text"] for r in got)
    check(vals == ["1,000,000", "1,100,000"],
          "warm: the ok row and the ok_label_completed row both come back (%s)" % vals)
    check(any(r["guard"] == "ok_label_completed" for r in got),
          "a label-completed fact is readable, not written-then-lost (guards %s)"
          % sorted({r["guard"] for r in got}))
    con.execute("INSERT INTO verified_facts (doc_id, doc_content_sha1, unit_id, metric, value_text, guard, "
                "verified_at) SELECT ?, content_sha1, ?, ?, ?, NULL, ? FROM documents WHERE doc_id = ?",
                (DOCS[1401], DOCS[1401] + "::u9", LABEL, "9,999", "2026-09-20T00:00:00", DOCS[1401]))
    con.commit()
    got2, _ = KQ.verified_lookup(con, list(DOCS.values()), terms, enabled=True)
    check(sorted(r["value_text"] for r in got2) == vals,
          "a row whose guard was never recorded is a miss, not a silent pass (%s)"
          % sorted(r["value_text"] for r in got2))
    back = [r["value_text"] for r in W.read_rows(con, doc_ids=list(DOCS.values()), warm=True)]
    check("9,999" not in back and sorted(back) == vals,
          "writeback.read_rows applies the same rule to the same rows (%s)" % sorted(back))


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    cfg = load_config()
    root = str(cfg.path("root")) if "root" in getattr(cfg, "paths", {}) else os.getcwd()
    print("differential (historical runs under %s/work/runs)" % root)
    kb_path = cfg.get("kb.path", "work/kb/kb.sqlite")
    kb_file = os.path.join(root, kb_path)
    kb = S.connect(kb_file, read_only=True) if os.path.exists(kb_file) else None
    n_cases = differential_tests(root, kb)
    if kb is not None:
        kb.close()
    hand_tests()
    test_dir = os.path.join(root, "work", "kb", "_test_v3")
    shutil.rmtree(test_dir, ignore_errors=True)
    os.makedirs(test_dir, exist_ok=True)
    con = seed_kb(os.path.join(test_dir, "kb_test.sqlite"), cfg)
    print("\ngap queries")
    gap_tests(con)
    print("\nlabel patterns")
    label_tests()
    print("\nwrite-back")
    writeback_tests(con)
    con.close()
    print("\n%d historical cases covered by the differential test (case set %s)" % (n_cases, CASE_SET_NAME))
    print("%d failure(s)" % len(FAILS))
    for f in FAILS:
        print("  -", f)
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
