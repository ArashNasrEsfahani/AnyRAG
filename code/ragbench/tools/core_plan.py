"""CORE PLAN helper for ragbench/tools/core_plan.sh. No model calls.

    python -m ragbench.tools.core_plan busy             exit 1 (and list them) while a model-heavy ragbench job runs
    python -m ragbench.tools.core_plan check-timeline   exit 1 when test_timeline.jsonl breaks the core-plan rules
    python -m ragbench.tools.core_plan plan [--dry-dir work/runs/core/dryrun] [--out work/runs/core/PLAN.md]
    python -m ragbench.tools.core_plan status           steps, records per variant, estimated remaining GPU time
    python -m ragbench.tools.core_plan cost             GPU-hours per stage (steps.tsv, journal, ledgers, telemetry)

`plan` reads the dry-run logs that `bash ragbench/tools/core_plan.sh plan` writes (every index/dev/test step run with
--dry-run: the real plan of ragbench.run, the index directories it would use and whether they are complete) and writes
PLAN.md + PLAN.json. `status` counts finished records against PLAN.json and re-estimates the remaining GPU time from the
measured per-record times. Estimates use the measured costs named in COSTS below (RUNBOOK section 7, the smoke/dev runs
and the component reports); the laptop GPU throttles, so treat them as +-50%.
"""
import argparse
import collections
import csv
import datetime as dt
import json
import re
import statistics
import subprocess
import sys
from pathlib import Path

from ..config import load_config

ROOT = Path(load_config().project_root)
WORK = ROOT / "work"
CORE = WORK / "runs" / "core"
LOGS = CORE / "logs"
DRY_DIR = CORE / "dryrun"
QDIR = ROOT / "ragbench" / "data" / "questions"

# measured costs (seconds). Query: mean seconds per record on corpus A in the smoke run (RUNBOOK section 7); OURS index
# levels share the OURS query side (plan + answer); AGENTIC from its live check (~16 s model time for a single fact,
# ~37 s for a timeline). Timeline answers are longer (answer-timeline-v1, up to 768 tokens): +8 s per record for every
# other approach (about 300 more answer tokens at the throttled ~35 tokens/s). Index: B4 ~0.9 s per section (summaries),
# B3 33 s per 0.58M chars (bge-m3), L1 one glossary per document 11-31 s (21 s used), L3 12.6 s (A) / 9 s (B, C) per
# section part. INVOCATION: warm-up call + index load per ragbench.run invocation.
COSTS = {
    "query_s": {"B0": 0.2, "B1": 1.1, "B2": 1.9, "B3": 2.1, "B4": 3.4, "B5": 17.9, "OURS": 2.9,
                "OURS+index_level=L0": 2.9, "OURS+index_level=L1": 2.9, "OURS+monolingual_patterns": 2.0, "AGENTIC": 16.0},
    "timeline_query_s": {"AGENTIC": 37.0},
    "timeline_extra_s": 8.0,
    "invocation_s": 10.0,
    "b4_s_per_section": 0.9,
    "b3_s_per_mchar": 33.0 / 0.58,
    "l1_s_per_doc": 21.0,
    "l3_s_per_section": {"A": 12.6, "B": 9.0, "C": 9.0},
}
STAGES = ("l3", "index", "dev", "test", "aggregate")
TEST_BLOCKS = ["A fa-fa", "A en-en", "B fa-fa", "B en-en", "C fa-fa", "TL fa-fa", "A fa-en", "A en-fa"]
DEV_BLOCKS = ["A fa-fa", "TL fa-fa", "A en-en", "A fa-en", "A en-fa", "PT fa-fa"]
CORE_VARIANTS = ["B0", "B1", "B2", "B3", "B4", "B5", "OURS+index_level=L0", "OURS+index_level=L1", "OURS",
                 "OURS+monolingual_patterns", "AGENTIC"]
TIMELINE_EXCLUDED = ("OURS", "OURS+monolingual_patterns")
DEV_VARIANTS = {"A": ["B3", "OURS+index_level=L0", "OURS+index_level=L1", "OURS", "AGENTIC"],
                "TL": ["B3", "OURS+index_level=L0", "OURS+index_level=L1", "AGENTIC"], "PT": ["B3", "AGENTIC"]}
SETTINGS = {"A": ["fa-fa", "en-en", "fa-en", "en-fa"], "B": ["fa-fa", "en-en"], "C": ["fa-fa"]}


def _stdout_utf8():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass


def _read_jsonl(path):
    out = []
    try:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    try:
                        out.append(json.loads(line))
                    except ValueError:
                        pass
    except OSError:
        pass
    return out


def _hours(s):
    return "%.2f h" % (s / 3600.0)


def _mins(s):
    return "%.0f min" % (s / 60.0) if s >= 60 else "%.0f s" % s


# --------------------------------------------------------------------------
# busy / check-timeline
# --------------------------------------------------------------------------

def running_jobs():
    """Command lines of running python processes that call the model: ragbench.run without --dry-run, live checks.
    None when the process list cannot be read."""
    ps = ("Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
          "ForEach-Object { $_.ProcessId.ToString() + ' ' + $_.CommandLine }")
    try:
        out = subprocess.run(["powershell.exe", "-NoProfile", "-Command", ps], capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.SubprocessError):
        return None
    jobs = {}
    for line in out.stdout.splitlines():
        pid, _, cmd = line.strip().partition(" ")
        if (("ragbench.run" in cmd and "--dry-run" not in cmd) or "live_check" in cmd) and "core_plan" not in cmd:
            jobs.setdefault(cmd.replace('"', "").strip(), []).append(pid)
    return jobs


def cmd_busy(args):
    jobs = running_jobs()
    if jobs is None:
        print("WARNING: cannot list processes (powershell); not checked")
        return 0
    if not jobs:
        print("no model-heavy ragbench job running")
        return 0
    print("a model-heavy ragbench job is running (one GPU: wait for it, or stop it, then re-run):")
    for cmd, pids in jobs.items():
        print("  pid %s: %s" % (",".join(pids), cmd[:300]))
    return 1


def _doc_key(doc_id):
    return doc_id.split(".", 1)[1].rsplit(".", 1)[0] if doc_id.count(".") >= 2 else doc_id


def cmd_check_timeline(args):
    """test_timeline.jsonl: timeline questions of corpus C only; no dev timeline question (same id or text); the Snapp
    drivers series is reserved as the AGENTIC tuning example (never in test)."""
    path = Path(args.path) if args.path else QDIR / "test_timeline.jsonl"
    if not path.is_file():
        print("timeline test file %s not present" % path)
        return 0
    tq = _read_jsonl(path)
    dev = _read_jsonl(QDIR / "pilot_timeline.jsonl") + _read_jsonl(QDIR / "dev_core_timeline.jsonl")
    dev_ids = {q.get("id") for q in dev}
    dev_texts = {(q.get("question_fa") or "").strip() for q in dev} | {(q.get("question_en") or "").strip() for q in dev}
    probs = []
    for q in tq:
        qid = q.get("id")
        if q.get("type") != "timeline" or q.get("corpus") != "C":
            probs.append("%s: type %s corpus %s (want timeline, C)" % (qid, q.get("type"), q.get("corpus")))
        if qid in dev_ids or (q.get("question_fa") or "").strip() in dev_texts or (q.get("question_en") or "").strip() in dev_texts:
            probs.append("%s: is a dev timeline question (same id or text)" % qid)
        keys = {_doc_key(d) for d in (q.get("doc_scope") or []) if isinstance(d, str)}
        units = " ".join("%s %s" % (it.get("unit"), it.get("unit_fa")) for it in ((q.get("meta") or {}).get("answer_series") or [])
                         if isinstance(it, dict))
        counts_drivers = (re.search(r"driver|راننده", units, re.I)
                          or re.search(r"(?:how many|number of|count of)\s+(?:\w+\s+){0,2}drivers", q.get("question_en") or "", re.I)
                          or re.search(r"(?:تعداد|شمار|چند)\s+(?:\S+\s+){0,2}رانند", q.get("question_fa") or ""))
        if any(k.startswith("snapp_") for k in keys) and counts_drivers:
            probs.append("%s: the Snapp drivers (count) series is reserved as a DEV example (AGENTIC was tuned on it)" % qid)
    keys = sorted({_doc_key(d) for q in tq for d in (q.get("doc_scope") or []) if isinstance(d, str)})
    print("%s: %d questions over %d documents: %s" % (path.name, len(tq), len(keys), ", ".join(keys)))
    for p in probs:
        print("PROBLEM " + p)
    return 1 if probs else 0


# --------------------------------------------------------------------------
# plan
# --------------------------------------------------------------------------

def _dry_steps(dry_dir):
    rows = []
    tsv = Path(dry_dir) / "steps.tsv"
    if not tsv.is_file():
        return rows
    for line in tsv.read_text(encoding="utf-8").splitlines():
        parts = line.split("\t")
        if len(parts) < 3:
            continue
        name, rc, argv = parts[0], parts[1], parts[2]
        log = Path(dry_dir) / (name + ".log")
        js, msgs = None, []
        if log.is_file():
            for l in log.read_text(encoding="utf-8", errors="replace").splitlines():
                if l.startswith("DRYRUN-JSON "):
                    js = json.loads(l[len("DRYRUN-JSON "):])
                elif l.startswith(("FINAL:", "refusing", "WARNING", "Traceback")) or "Error" in l[:40]:
                    msgs.append(l)
        rows.append({"name": name, "rc": int(rc) if rc.lstrip("-").isdigit() else rc, "argv": argv, "dry": js,
                     "messages": msgs, "stage": name.split("_", 1)[0]})
    return rows


def _block_of(step):
    """'A fa-fa' / 'TL fa-fa' / 'PT fa-fa' for a query step, from its name (<stage>_<v>_<A|B|C|TL|PT>[_<setting>])."""
    parts = step["name"].split("_")
    corpus = parts[2] if len(parts) > 2 else "?"
    js = step.get("dry") or {}
    setting = ",".join(js.get("settings") or []) or (parts[3] if len(parts) > 3 else "?")
    return "%s %s" % (corpus, setting)


def record_seconds(variant, fmt, rates=None):
    rates = rates or {}
    if (variant, fmt) in rates:
        return rates[(variant, fmt)]
    if fmt == "timeline":
        if variant in COSTS["timeline_query_s"]:
            return COSTS["timeline_query_s"][variant]
        return COSTS["query_s"].get(variant, 5.0) + COSTS["timeline_extra_s"]
    return COSTS["query_s"].get(variant, 5.0)


def query_estimate(js, rates=None, todo=True):
    """GPU seconds for the records of one dry-run step (todo records, or all planned)."""
    key = "todo" if todo else "planned"
    n = sum(v[key] for v in (js.get("by_format") or {}).values())
    s = sum(v[key] * record_seconds(js["variant"], fmt, rates) for fmt, v in (js.get("by_format") or {}).items())
    return s + (COSTS["invocation_s"] if n else 0.0)


class _IndexCost:
    """Per-document index cost with cache knowledge: documents of the dev_full pilot restriction have warm B3/B4 caches
    (its B3/B4 indexes are complete); L1 glossaries are looked up in the glossary cache itself."""

    def __init__(self):
        from ..contracts import load_questions
        from ..run import referenced_doc_keys
        self.cfg = load_config()
        pilot = load_questions(QDIR / "pilot.jsonl")
        self.warm = {c: referenced_doc_keys(pilot, c) for c in "ABC"}
        self.covered = collections.defaultdict(set)     # variant -> doc ids whose per-item cache stage index fills
        self._sets = {}
        self._gb = None

    def doc(self, set_name, doc_id):
        from ..corpus.registry import load_set
        if set_name not in self._sets:
            self._sets[set_name] = load_set(set_name, self.cfg)
        return self._sets[set_name].doc(doc_id)

    def glossary_cached(self, doc):
        try:
            if self._gb is None:
                from ..approaches.ours.levels import cache_paths, level_options
                from ..approaches.ours.lite_llm import GlossaryBuilder
                self._gb = GlossaryBuilder(self.cfg, None, cache_paths(self.cfg)["glossary"], level_options({}))
            msgs, _ = self._gb.messages(doc)
            return self._gb.cache_key(doc, msgs) in self._gb.cache
        except Exception:  # noqa: BLE001
            return False

    def estimate(self, variant, set_info):
        """(GPU seconds, note) to build one index that is not complete."""
        if set_info.get("complete"):
            return 0.0, "complete"
        corpus = set_info["set"][0]
        docs = [self.doc(set_info["set"], d) for d in set_info.get("doc_ids") or []]
        covered = self.covered.get(variant, set())
        cold = [d for d in docs if _doc_key(d.id) not in self.warm.get(corpus, set()) and d.id not in covered]
        if variant == "B3":
            chars = sum(len(d.text) for d in cold)
            return chars / 1e6 * COSTS["b3_s_per_mchar"], "%d of %d docs uncached (%d chars)" % (len(cold), len(docs), chars)
        if variant == "B4":
            secs = sum(len(d.sections) for d in cold)
            return secs * COSTS["b4_s_per_section"], "%d of %d docs uncached (%d sections)" % (len(cold), len(docs), secs)
        if variant == "OURS+index_level=L1":
            miss = [d for d in docs if d.id not in covered and not self.glossary_cached(d)]
            return len(miss) * COSTS["l1_s_per_doc"], "%d of %d glossaries uncached" % (len(miss), len(docs))
        if variant in ("OURS", "OURS+monolingual_patterns"):
            return 0.0, "store from the stage-l3 enrichment cache (0 once stage l3 has finished)"
        return 0.0, "no model calls"


def expected_records():
    """{(stage, block): {variant: n}} from the question files: questions supporting the block's setting x variants."""
    from ..contracts import Setting, load_questions
    exp = {}
    files = {"test": {"A": "test_core.jsonl", "B": "test_core.jsonl", "C": "test_core.jsonl", "TL": "test_timeline.jsonl"},
             "dev": {"A": "dev_core.jsonl", "TL": "dev_core_timeline.jsonl", "PT": "pilot_timeline.jsonl"}}
    for stage, blocks in (("test", TEST_BLOCKS), ("dev", DEV_BLOCKS)):
        for block in blocks:
            c, s = block.split()
            path = QDIR / files[stage][c]
            if not path.is_file():
                continue
            corpus = "C" if c in ("TL", "PT") else c
            n = sum(1 for q in load_questions(path) if q.corpus == corpus and q.supports(Setting.parse(s)))
            if stage == "test":
                variants = [v for v in CORE_VARIANTS if not (c == "TL" and v in TIMELINE_EXCLUDED)]
            else:
                variants = DEV_VARIANTS[c]
            exp[(stage, block)] = {v: n for v in variants}
    return exp


def cmd_plan(args):
    dry_dir = Path(args.dry_dir)
    steps = _dry_steps(dry_dir)
    if not steps:
        print("no dry-run steps in %s (run: bash ragbench/tools/core_plan.sh plan)" % dry_dir)
        return 1
    ic = _IndexCost()
    refused = [s for s in steps if s["rc"] != 0 or s["dry"] is None]
    idx_rows, q_rows = [], []
    for s in steps:
        js = s["dry"]
        if js is None:
            continue
        if s["stage"] == "index":
            for st in js["sets"]:
                est, note = ic.estimate(js["variant"], st)
                idx_rows.append({"step": s["name"], "variant": js["variant"], "set": st["set"], "docs": st["docs"],
                                 "sections": st["sections"], "chars": st["chars"], "complete": st["complete"],
                                 "index_dir": st["index_dir"], "questions_file": js["questions_file"], "out": js["out"],
                                 "doc_ids": st.get("doc_ids"), "est_s": est, "note": note})
        else:
            q_rows.append({"step": s["name"], "stage": s["stage"], "block": _block_of(s), "variant": js["variant"],
                           "corpus": js["corpus"], "settings": js["settings"], "questions_file": js["questions_file"],
                           "out": js["out"], "planned": js["planned"], "done": js["done"], "todo": js["todo"],
                           "by_format": js["by_format"], "final": js["final"], "split": js["split"],
                           "index_dirs": [(st["set"], st["index_dir"], st["complete"]) for st in js["sets"]],
                           "est_s": query_estimate(js), "est_all_s": query_estimate(js, todo=False)})
    # verification: records per (stage, block, variant) against the question files; index reuse
    exp = expected_records()
    got = collections.defaultdict(dict)
    for r in q_rows:
        got[(r["stage"], r["block"])][r["variant"]] = got[(r["stage"], r["block"])].get(r["variant"], 0) + r["planned"]
    mismatches = []
    dry_stages = {s["stage"] for s in steps}
    tl_ok = cmd_check_timeline(argparse.Namespace(path=None)) == 0 if (QDIR / "test_timeline.jsonl").is_file() else False
    for key in sorted(set(exp) | set(got)):
        if key[0] not in dry_stages or (key == ("test", "TL fa-fa") and not tl_ok):
            continue                      # stage not dry-run here / timeline steps skipped by the driver's rule check
        e, g = exp.get(key, {}), got.get(key, {})
        if e != g:
            mismatches.append("%s %s: expected %s, planned %s" % (key[0], key[1], e, g))
    built = {r["index_dir"] for r in idx_rows}
    test_dirs = {(d, r["variant"], s_) for r in q_rows if r["stage"] == "test" for s_, d, _ in r["index_dirs"]}
    not_built = sorted({(v, s_) for d, v, s_ in test_dirs if d not in built})
    # indexes a dev/test step builds itself (not built by stage index and not complete): estimated once per directory;
    # documents whose summaries/embeddings/glossaries stage index caches count as cached (stage index runs first)
    for r in idx_rows:
        ic.covered[r["variant"]].update(r["doc_ids"] or [])
    fly_rows, seen = [], set(built)
    for s in steps:
        js = s["dry"]
        if js is None or s["stage"] == "index":
            continue
        for st in js["sets"]:
            if st["complete"] or st["index_dir"] in seen:
                continue
            seen.add(st["index_dir"])
            est, note = ic.estimate(js["variant"], st)
            if js["variant"] in ("OURS", "OURS+monolingual_patterns") and not st["complete"]:
                note = "store from the enrichment cache (documents enriched by stage l3 / dev_full)"
            fly_rows.append({"stage": s["stage"], "step": s["name"], "variant": js["variant"], "set": st["set"],
                             "docs": st["docs"], "sections": st["sections"], "index_dir": st["index_dir"], "est_s": est,
                             "note": note})
    fly_s = collections.Counter()
    for r in fly_rows:
        fly_s[r["stage"]] += r["est_s"]
    tl_present = (QDIR / "test_timeline.jsonl").is_file()

    L = ["# CORE PLAN: planned records and GPU estimate", "",
         "Generated %s by `python -m ragbench.tools.core_plan plan` from the dry-run of every index, dev and test step "
         "(`bash ragbench/tools/core_plan.sh plan`: each `ragbench.run` invocation with `--dry-run`, so the plan, the "
         "`--final` guard and the index directories are the real ones; no index build, no model call)." % (
             dt.datetime.now().isoformat(timespec="seconds")), "",
         "- Steps dry-run: %d (%d refused or without a plan)%s" % (
             len(steps), len(refused), "".join("\n  - %s rc=%s %s" % (s["name"], s["rc"], " | ".join(s["messages"][:2]))
                                                for s in refused)),
         "- Plan equals questions x settings x variants from the question files: **%s**%s" % (
             "yes" if not mismatches else "NO", "".join("\n  - " + m for m in mismatches)),
         "- Test index directories not built by stage `index`: %s" % (
             "none (the test run loads exactly the directories stage index builds)" if not not_built else not_built),
         "- Timeline test file `test_timeline.jsonl`: %s" % ("present" if tl_present else "NOT present (timeline steps skipped)"),
         ""]

    def table(stage, blocks, variants):
        rows = [r for r in q_rows if r["stage"] == stage]
        if not rows:
            return []
        cols = [b for b in blocks if any(r["block"] == b for r in rows)]
        out = ["| Variant | " + " | ".join(cols) + " | Records | Todo | Est. GPU |", "|---" * (len(cols) + 4) + "|"]
        tot_cols = collections.Counter()
        tot_n = tot_todo = tot_s = 0
        for v in variants:
            vr = [r for r in rows if r["variant"] == v]
            if not vr:
                continue
            cells = []
            for b in cols:
                n = sum(r["planned"] for r in vr if r["block"] == b)
                tot_cols[b] += n
                cells.append(str(n) if any(r["block"] == b for r in vr) else "–")
            n, todo, s = sum(r["planned"] for r in vr), sum(r["todo"] for r in vr), sum(r["est_s"] for r in vr)
            tot_n, tot_todo, tot_s = tot_n + n, tot_todo + todo, tot_s + s
            out.append("| %s | %s | %d | %d | %s |" % (v, " | ".join(cells), n, todo, _mins(s)))
        out.append("| **Total** | %s | **%d** | **%d** | **%s** |" % (" | ".join(str(tot_cols[b]) for b in cols), tot_n, tot_todo,
                                                                     _hours(tot_s)))
        return out

    L += ["## Test stage (`runs/core_test`, timelines `runs/core_test_timeline`; `--final --docs-from-questions`)", "",
          "Blocks in run order (outer loop), every variant inside a block (inner loop). Cells = planned records "
          "(questions x setting).", ""]
    L += table("test", TEST_BLOCKS, CORE_VARIANTS) + [""]
    blk_s = collections.OrderedDict((b, 0.0) for b in TEST_BLOCKS)
    for r in q_rows:
        if r["stage"] == "test":
            blk_s[r["block"]] = blk_s.get(r["block"], 0.0) + r["est_s"]
    cum, cum_parts = 0.0, []
    for b, s in blk_s.items():
        if s:
            cum += s
            cum_parts.append("%s %s (cum. %s)" % (b, _mins(s), _hours(cum)))
    L += ["Estimated GPU time per block: " + "; ".join(cum_parts) + ".", ""]
    L += ["## Dev stage (`runs/core_dev`: dev_core A + dev_core_timeline; `runs/core_dev_pilot_timeline`: pilot_timeline)", ""]
    L += table("dev", DEV_BLOCKS, CORE_VARIANTS) + [""]
    L += ["## Index stage (`--final --docs-from-questions --index-only`, copies in the test run directories)", "",
          "| Step | Variant | Set | Docs | Sections | Chars | Complete now | Est. GPU | Cache note |", "|---|---|---|---|---|---|---|---|---|"]
    for r in idx_rows:
        L.append("| %s | %s | %s | %d | %d | %d | %s | %s | %s |" % (r["step"], r["variant"], r["set"], r["docs"], r["sections"],
                                                              r["chars"], "yes" if r["complete"] else "no", _mins(r["est_s"]), r["note"]))
    idx_s = sum(r["est_s"] for r in idx_rows)
    by_var = collections.Counter()
    for r in idx_rows:
        by_var[r["variant"]] += r["est_s"]
    L += ["", "Index GPU estimate by variant: " + ", ".join("%s %s" % (v, _mins(s)) for v, s in by_var.most_common() if s) + ".", ""]
    if fly_rows:
        L += ["Indexes that a dev or test step builds itself (not part of stage index, not complete now):", "",
              "| Stage | Step | Variant | Set | Docs | Sections | Est. GPU | Note |", "|---|---|---|---|---|---|---|---|"]
        L += ["| %s | %s | %s | %s | %d | %d | %s | %s |" % (r["stage"], r["step"], r["variant"], r["set"], r["docs"], r["sections"],
                                                         _mins(r["est_s"]), r["note"]) for r in fly_rows]
        L += [""]
    dev_s = sum(r["est_s"] for r in q_rows if r["stage"] == "dev") + fly_s["dev"]
    test_s = sum(r["est_s"] for r in q_rows if r["stage"] == "test") + fly_s["test"]
    l3 = l3_progress(idx_rows)
    L += ["## GPU time per stage (estimate, todo work only)", "",
          "| Stage | Est. GPU | What |", "|---|---|---|",
          "| l3 (running) | %s | %s |" % (_mins(l3["remaining_s"]), l3["note"]),
          "| index | %s | B4 summaries, L1 glossaries and B3 embeddings for uncached documents; the rest is CPU |" % _mins(idx_s),
          "| dev | %s | %d records + per-invocation warm-up + its own indexes (%s) |" % (
              _mins(dev_s), sum(r["todo"] for r in q_rows if r["stage"] == "dev"), _mins(fly_s["dev"])),
          "| test | %s | %d records + per-invocation warm-up%s |" % (
              _hours(test_s), sum(r["todo"] for r in q_rows if r["stage"] == "test"),
              " + indexes not built by stage index (%s)" % _mins(fly_s["test"]) if fly_s["test"] else ""),
          "| aggregate | 0 | CPU only (judge none) |",
          "| **Total** | **%s** | |" % _hours(l3["remaining_s"] + idx_s + dev_s + test_s), "",
          "Assumptions (seconds; measured, see `ragbench/tools/core_plan.py` COSTS): per record %s; timeline records: AGENTIC %s, "
          "others +%s; +%s per invocation; index: B4 %s s/section, B3 %.0f s per M chars, L1 %s s/document (11-31), "
          "L3 %s s/section. Documents referenced by pilot.jsonl count as cached for B3/B4 (dev_full built those indexes); "
          "L1 glossaries are looked up in the glossary cache. The laptop GPU throttles: +-50%%." % (
              ", ".join("%s %s" % (k, v) for k, v in COSTS["query_s"].items()), COSTS["timeline_query_s"]["AGENTIC"],
              COSTS["timeline_extra_s"], COSTS["invocation_s"], COSTS["b4_s_per_section"], COSTS["b3_s_per_mchar"],
              COSTS["l1_s_per_doc"], COSTS["l3_s_per_section"]), "",
          "OURS (L3) and OURS+monolingual_patterns do not run on the timeline documents (L3 enrichment too expensive; "
          "config final.exclude_variants refuses them for test_timeline.jsonl). L0/L1 do.", "",
          "## Commands", "", "```bash",
          "bash ragbench/tools/core_plan.sh index      # stage 2: every index the test run uses",
          "bash ragbench/tools/core_plan.sh dev        # stage 3: mini dev run (runs/core_dev, runs/core_dev_pilot_timeline)",
          "bash ragbench/tools/core_plan.sh test       # stage 4: --final core test (runs/core_test, runs/core_test_timeline)",
          "bash ragbench/tools/core_plan.sh aggregate  # stage 5: results + sanity + cost summary",
          "bash ragbench/tools/core_plan.sh all        # index -> dev -> test -> aggregate",
          "bash ragbench/tools/core_plan.sh status     # progress and remaining GPU estimate",
          "bash ragbench/tools/core_plan.sh plan       # re-run every dry-run and rewrite this file", "```", ""]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(L), encoding="utf-8")
    plan_json = {"generated": dt.datetime.now().isoformat(timespec="seconds"), "costs": COSTS, "index": idx_rows,
                 "queries": q_rows, "step_indexes": fly_rows, "mismatches": mismatches, "refused": [s["name"] for s in refused],
                 "test_index_dirs_not_built": not_built, "timeline_file_present": tl_present,
                 "estimate_s": {"l3": l3["remaining_s"], "index": idx_s, "dev": dev_s, "test": test_s}}
    out.with_suffix(".json").write_text(json.dumps(plan_json, ensure_ascii=False, indent=1), encoding="utf-8")
    print("\n".join(L))
    print("-> %s (+ %s)" % (out, out.with_suffix(".json").name))
    return 0 if not mismatches and not refused and not not_built else 1


# --------------------------------------------------------------------------
# status
# --------------------------------------------------------------------------

def _ledger_sections_since(step, corpus):
    """Distinct sections of `corpus` enriched by new OURS index calls (work/runs/core ledger) since the last start of
    `step` (steps.log '=== <date> <step>:')."""
    start = None
    log = LOGS / "steps.log"
    if log.is_file():
        for m in re.finditer(r"^=== (\d{4}-\d\d-\d\d \d\d:\d\d:\d\d) %s:" % re.escape(step),
                             log.read_text(encoding="utf-8", errors="replace"), re.M):
            start = m.group(1).replace(" ", "T")
    secs = set()
    for l in _read_jsonl(CORE / "ledger_OURS.jsonl"):
        if (l.get("phase") == "index" and str(l.get("corpus_set", "")).startswith(corpus) and l.get("section_id")
                and (start is None or str(l.get("ts", "")) >= start)):
            secs.add(l["section_id"])
    return len(secs)


def l3_progress(index_rows=None):
    """Remaining OURS L3 enrichment of stage l3 from its step logs ("[OURS index] A_fa: 350/384 sections"); a step with
    no progress line yet counts the sections of its not-complete OURS index rows (upper bound: cached parts are free)."""
    totals = collections.Counter()
    for r in index_rows or []:
        if r.get("variant") == "OURS" and not r.get("complete") and r.get("step", "").startswith("index_ours_"):
            totals[r["set"][0]] += r.get("sections") or 0
    done_steps = set()
    tsv = LOGS / "steps.tsv"
    if tsv.is_file():
        for line in tsv.read_text(encoding="utf-8").splitlines():
            p = line.split("\t")
            if len(p) >= 2 and p[0].startswith("l3_") and p[1] == "0":
                done_steps.add(p[0])
    remaining, parts = 0.0, []
    for step, corpus in (("l3_A", "A"), ("l3_B", "B"), ("l3_C", "C")):
        if step in done_steps:
            parts.append("%s done" % step)
            continue
        log = LOGS / (step + ".log")
        prog = {}
        if log.is_file():
            for m in re.finditer(r"\[OURS index\] (\w+): (\d+)/(\d+) sections", log.read_text(encoding="utf-8", errors="replace")):
                prog[m.group(1)] = (int(m.group(2)), int(m.group(3)))
        if prog:
            left = sum(t - d for d, t in prog.values())
            what = ", ".join("%s %d/%d" % (k, d, t) for k, (d, t) in prog.items())
        elif log.is_file():                # progress lines are buffered: count the step's enriched sections in the ledger
            new = _ledger_sections_since(step, corpus)
            left = max(totals.get(corpus, 0) - new, 0)
            what = "running: %d sections enriched by new calls so far, <= %d left" % (new, left)
        else:
            left, what = totals.get(corpus, 0), "not started"
        remaining += left * COSTS["l3_s_per_section"][corpus]
        parts.append("%s %s" % (step, what))
    return {"remaining_s": remaining, "note": "; ".join(parts) + " (upper bound: sections already cached cost nothing)"}


def _question_ids(name):
    return {q.get("id") for q in _read_jsonl(QDIR / name)}


def measured_rates(out_dirs):
    """{(variant, answer format): median no-load seconds per record} over finished records in the run directories."""
    vals = collections.defaultdict(list)
    for d in out_dirs:
        for f in Path(d).glob("runs_*.jsonl"):
            for r in _read_jsonl(f):
                if r.get("error") or not r.get("latency_ms"):
                    continue
                fmt = (r.get("meta") or {}).get("answer_format") or "default"
                s = (r["latency_ms"] - (r.get("lock_wait_ms") or 0) - (r.get("load_ms") or 0)) / 1000.0
                vals[(r.get("approach"), fmt)].append(max(s, 0.0))
    return {k: statistics.median(v) for k, v in vals.items() if len(v) >= 3}


def cmd_status(args):
    from ..util import safe_name
    plan_path = CORE / "PLAN.json"
    tsv = LOGS / "steps.tsv"
    lines = tsv.read_text(encoding="utf-8").splitlines() if tsv.is_file() else []
    by_stage = collections.defaultdict(lambda: [0, 0, 0])
    for line in lines:
        p = line.split("\t")
        if len(p) < 3:
            continue
        st = p[0].split("_", 1)[0]
        by_stage[st][0] += 1
        by_stage[st][1] += p[1] != "0"
        by_stage[st][2] += int(p[2].rstrip("s") or 0)
    print("== steps (work/runs/core/logs/steps.tsv): " + (", ".join("%s %d steps (%d failed, %s)" % (
        k, v[0], v[1], _hours(v[2])) for k, v in by_stage.items()) or "none yet"))
    for line in lines[-5:]:
        print("   " + line)
    plan = json.loads(plan_path.read_text(encoding="utf-8")) if plan_path.is_file() else None
    l3 = l3_progress((plan or {}).get("index"))
    print("== stage l3: %s; remaining ~%s" % (l3["note"], _mins(l3["remaining_s"])))
    if plan is None:
        print("no PLAN.json: run `bash ragbench/tools/core_plan.sh plan` for planned counts and estimates")
        return 0
    out_dirs = sorted({r["out"] for r in plan["queries"]})
    rates = measured_rates(out_dirs)
    qids = {}
    recs = {}
    per_var = collections.OrderedDict()
    remaining_s = collections.Counter()
    for r in plan["queries"]:
        key = (r["out"], r["variant"])
        if key not in recs:
            recs[key] = [x for x in _read_jsonl(Path(r["out"]) / ("runs_%s.jsonl" % safe_name(r["variant"]))) if not x.get("error")]
        if r["questions_file"] not in qids:
            qids[r["questions_file"]] = _question_ids(r["questions_file"])
        done = {(x["question_id"], x["setting"]) for x in recs[key]
                if x["question_id"] in qids[r["questions_file"]] and x["setting"] in r["settings"] and x.get("corpus") == r["corpus"]}
        n_done = min(len(done), r["planned"])
        left = r["planned"] - n_done
        fmts = r["by_format"]
        tl_share = (fmts.get("timeline", {}).get("planned", 0) / float(r["planned"])) if r["planned"] else 0.0
        s = left * ((1 - tl_share) * record_seconds(r["variant"], "default", rates)
                    + tl_share * record_seconds(r["variant"], "timeline", rates)) + (COSTS["invocation_s"] if left else 0)
        pv = per_var.setdefault((r["stage"], r["variant"]), [0, 0, 0.0])
        pv[0] += r["planned"]
        pv[1] += n_done
        pv[2] += s
        remaining_s[r["stage"]] += s
    print("== records (planned / done / remaining est.)")
    print("   %-6s %-28s %8s %8s %10s %s" % ("stage", "variant", "planned", "done", "remaining", "s/record (measured)"))
    for (stage, v), (n, d, s) in per_var.items():
        m = rates.get((v, "default"))
        mt = rates.get((v, "timeline"))
        print("   %-6s %-28s %8d %8d %10s %s" % (stage, v, n, d, _mins(s),
                                                 ("%.1f" % m if m is not None else "-") + (" / tl %.1f" % mt if mt is not None else "")))
    def complete(index_dir):
        try:
            return bool(json.loads((Path(index_dir) / "build_stats.json").read_text(encoding="utf-8")).get("complete"))
        except (OSError, ValueError):
            return False

    idx_left, n_idx_left = 0.0, 0
    for r in plan["index"]:
        if not complete(r["index_dir"]):
            n_idx_left += 1
            idx_left += r["est_s"]
    for r in plan.get("step_indexes") or []:          # built inside dev/test steps
        if not complete(r["index_dir"]):
            remaining_s[r["stage"]] += r["est_s"]
    total = l3["remaining_s"] + idx_left + sum(remaining_s.values())
    print("== remaining GPU estimate: l3 %s, index %s (%d of %d indexes not complete), dev %s, test %s -> total %s" % (
        _mins(l3["remaining_s"]), _mins(idx_left), n_idx_left, len(plan["index"]), _mins(remaining_s["dev"]),
        _hours(remaining_s["test"]), _hours(total)))
    if not plan.get("timeline_file_present") and (QDIR / "test_timeline.jsonl").is_file():
        print("NOTE: test_timeline.jsonl appeared after PLAN.json was written: re-run `bash ragbench/tools/core_plan.sh plan`")
    return 0


# --------------------------------------------------------------------------
# cost
# --------------------------------------------------------------------------

def _telemetry_busy_s(path, threshold=30):
    rows = []
    try:
        with open(path, encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                try:
                    rows.append((dt.datetime.fromisoformat(r["ts"]), float(r["util_pct"])))
                except (KeyError, ValueError):
                    pass
    except OSError:
        return 0.0, 0.0
    busy = total = 0.0
    for (t0, u), (t1, _) in zip(rows, rows[1:]):
        gap = min((t1 - t0).total_seconds(), 60.0)
        total += gap
        busy += gap if u >= threshold else 0.0
    return busy, total


def cmd_cost(args):
    L = ["# CORE PLAN cost summary", "", "Generated %s by `python -m ragbench.tools.core_plan cost`." % dt.datetime.now().isoformat(
        timespec="seconds"), ""]
    # steps.tsv: wall time of every ragbench step
    steps = collections.defaultdict(lambda: [0, 0, 0])
    tsv = LOGS / "steps.tsv"
    for line in (tsv.read_text(encoding="utf-8").splitlines() if tsv.is_file() else []):
        p = line.split("\t")
        if len(p) >= 3:
            st = p[0].split("_", 1)[0]
            steps[st][0] += 1
            steps[st][1] += p[1] != "0"
            steps[st][2] += int(p[2].rstrip("s") or 0)
    # journal: stage run_start/run_stop
    jr = collections.defaultdict(float)
    for e in _read_jsonl(WORK / "journal" / "events.jsonl"):
        if e.get("kind") == "run_stop" and str(e.get("name", "")).startswith("core_"):
            jr[e["name"][5:]] += float(e.get("duration_s") or 0)
    # telemetry: GPU busy time (utilisation >= 30%) per stage
    tele = collections.defaultdict(lambda: [0.0, 0.0])
    for f in sorted((WORK / "telemetry").glob("gpu_core_*.csv")):
        m = re.match(r"gpu_core_(\w+?)_\d{8}_\d{6}\.csv", f.name)
        if m:
            b, t = _telemetry_busy_s(f)
            tele[m.group(1)][0] += b
            tele[m.group(1)][1] += t
    known = list(STAGES) + ["agg", "sanity", "cost", "all"]
    present = set(steps) | set(jr) | set(tele)
    stage_names = [s for s in known if s in present] + sorted(present - set(known))
    L += ["## Per stage", "", "| Stage | Steps (failed) | Step wall | Stage wall (journal) | GPU busy (telemetry >= 30%) |",
          "|---|---|---|---|---|"]
    for s in stage_names:
        v = steps.get(s, [0, 0, 0])
        L.append("| %s | %d (%d) | %s | %s | %s of %s sampled |" % (s, v[0], v[1], _hours(v[2]), _hours(jr.get(s, 0.0)),
                                                                 _hours(tele[s][0]) if s in tele else "–",
                                                                 _hours(tele[s][1]) if s in tele else "–"))
    # ledgers: model time per run directory / variant / phase
    L += ["", "## Model time from the ledgers", "", "| Run dir | Variant | Phase | Chat calls | Replayed | Chat model time | Embed calls | "
          "Embed time | Prompt tok | Completion tok |", "|---|---|---|---|---|---|---|---|---|---|"]
    tot = collections.Counter()
    for d in sorted((WORK / "runs").glob("core*")):
        if not d.is_dir():
            continue
        for f in sorted(d.glob("ledger_*.jsonl")):
            agg = collections.defaultdict(collections.Counter)
            for l in _read_jsonl(f):
                ph = l.get("phase") or "?"
                a = agg[ph]
                if l.get("kind") == "chat":
                    a["chat"] += 1
                    if l.get("cache_hit"):
                        a["replayed"] += 1
                    else:
                        a["chat_ms"] += l.get("wall_ms") or 0
                    a["ptok"] += l.get("prompt_tokens") or 0
                    a["ctok"] += l.get("completion_tokens") or 0
                elif l.get("kind") == "embed":
                    a["embed"] += 1
                    a["embed_ms"] += l.get("wall_ms") or 0
            for ph, a in sorted(agg.items()):
                L.append("| %s | %s | %s | %d | %d | %s | %d | %s | %d | %d |" % (
                    d.name, f.stem[len("ledger_"):], ph, a["chat"], a["replayed"], _hours(a["chat_ms"] / 1000.0), a["embed"],
                    _mins(a["embed_ms"] / 1000.0), a["ptok"], a["ctok"]))
                tot[d.name] += (a["chat_ms"] + a["embed_ms"]) / 1000.0
    L += ["", "Model hours per run directory (chat + embed wall time of non-replayed calls): " +
          (", ".join("%s %s" % (k, _hours(v)) for k, v in tot.items()) or "none") +
          "; total %s." % _hours(sum(tot.values())), "",
          "Stage wall includes CPU work and idle waits; GPU busy is sampled every 30 s by gpu_monitor. Note that "
          "`work/runs/core` holds the stage-l3 index ledger (phase index).", ""]
    out = CORE / "COST.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))
    print("-> %s" % out)
    return 0


def main(argv=None):
    _stdout_utf8()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("busy")
    c = sub.add_parser("check-timeline")
    c.add_argument("--path", default=None)
    p = sub.add_parser("plan")
    p.add_argument("--dry-dir", default=str(DRY_DIR))
    p.add_argument("--out", default=str(CORE / "PLAN.md"))
    sub.add_parser("status")
    sub.add_parser("cost")
    args = ap.parse_args(argv)
    return {"busy": cmd_busy, "check-timeline": cmd_check_timeline, "plan": cmd_plan, "status": cmd_status,
            "cost": cmd_cost}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
