"""Aggregate RunRecords into results.csv / results.md / comparisons.csv.

    .venv/Scripts/python.exe -m ragbench.eval.aggregate --runs runs/pilot --questions ragbench/data/questions/pilot.jsonl \
        --judge none                      # or: --judge openrouter --judge-max-calls 50 --judge-max-cost 0.02

Inputs: run directories written by ragbench.run (runs_<variant>.jsonl, index_<variant>_<set>.json) or
RunRecord JSONL files. Outputs (default <first run dir>/results/):
  scored.jsonl      one line per (question, approach, setting): metrics + judge verdict
  results.csv       per split x approach x corpus x setting (plus setting "ALL", pooled over settings). The split
                    (dev | test | other) is the record's stamped `split`, or for older records the split of the
                    question file holding the question; dev and test are never pooled or paired.
                    ts_<name> columns: mean of approach-reported trace.stats counters (OURS plan_repair, widen, ...;
                    B4 narrowed_items, repair_calls, ...)
  comparisons.csv   paired cluster bootstrap (resampling questions) 95% CIs: every OURS variant minus
                    every baseline, and every OURS ablation minus full OURS
  results.md        the tables above, with provenance and metric definitions
  timeline.csv      timeline (series) questions only: tl_* metrics per split x approach x corpus x setting
                    (see eval/timeline.py), with calls, tokens and latency; written when such records exist
  cost_by_qtype.csv per split x approach x corpus x question type (lookup, timeline, ...): n, accuracy, chat/embed
                    calls, prompt/completion tokens, latency p50/p95 and no-load p50 (cost per question type)
  judge_summary.json

Scoring (--scoring):
  strict    (default) match_correct is the strict matcher of eval/metrics.py for every question.
  keyfacts  a question that carries meta.key_facts (locate, multi-section, cross-year) uses its
            kf_correct instead; every other question is unchanged. The strict number stays in the
            `acc_match_strict` column and row, so both are always reported.
  kf_recall (share of a question's key facts present in the answer) and acc_kf are columns either
  way. Write a keyfacts aggregation to its own --out directory to leave results/ untouched.

Accuracy counts run errors as incorrect; cost and latency statistics exclude errored records.
Judge accuracy is over judged records only (n_judged is reported; records without a verdict because
of the budget stay unjudged, never counted as wrong).
"""
import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from ..config import load_config
from ..contracts import load_questions
from ..llm import Ledger
from ..util import atomic_write_json, atomic_write_text, now_iso, read_json, read_jsonl
from .judge import make_judge
from .metrics import MatchConfig, score_record
from . import timeline as TL

SETTING_ORDER = ["fa-fa", "en-en", "fa-en", "en-fa"]
BASE_ORDER = ["B0", "B1", "B2", "B3", "B4", "B5", "OURS"]
BASE_NAMES = {"B0": "no retrieval", "B1": "gold evidence", "B2": "BM25", "B3": "vector (bge-m3)",
              "B4": "tree navigation", "B5": "grep agent", "OURS": "enriched index + planned grep"}
COMPARE_METRICS = ["acc_judge", "acc_match", "acc_kf", "kf_recall", "ev_hit", "ev_recall", "llm_calls",
                   "prompt_tokens", "latency_ms", "tl_coverage", "tl_series_correct"]


# --------------------------------------------------------------------------
# paths and loading
# --------------------------------------------------------------------------

def resolve_path(p, cfg=None):
    """Absolute paths as given; relative paths starting with 'runs' go under paths.work_root
    (so 'runs/pilot' == work/runs/pilot); other relative paths resolve against project_root."""
    cfg = cfg or load_config()
    pth = Path(p)
    if pth.is_absolute():
        return pth
    if pth.parts and pth.parts[0] == "runs":
        return cfg.path("work_root") / pth
    return cfg.project_root / pth


def run_files(paths, cfg=None):
    files = []
    for p in paths:
        pth = resolve_path(p, cfg)
        if pth.is_file():
            files.append(pth)
        elif pth.is_dir():
            # records moved to a `_superseded*` folder (an earlier version of a method, kept for the audit trail) are
            # not results of the run directory; before this they were read too, and only the sort order ('_' before
            # 'r') made the current file's record win the de-duplication below
            files += sorted(f for f in pth.rglob("runs_*.jsonl")
                            if not any(part.startswith("_superseded") for part in f.relative_to(pth).parts))
    return list(dict.fromkeys(files))


def load_run_records(paths, cfg=None):
    """RunRecord dicts, deduplicated on (corpus, question_id, approach, setting): the last
    successful record wins; an errored record is kept only if no successful one exists."""
    best = {}
    n_raw = 0
    for f in run_files(paths, cfg):
        for r in read_jsonl(f):
            if not isinstance(r, dict) or "question_id" not in r or "approach" not in r:
                continue
            n_raw += 1
            k = (r.get("corpus"), r["question_id"], r["approach"], r["setting"])   # question ids are unique across splits
            old = best.get(k)
            if old is None or not r.get("error") or old.get("error"):
                best[k] = r
    return list(best.values()), n_raw


def cached_index_cost(index_dir):
    """Original model cost of EVERY cached item an index uses, from the reports the approaches
    write for exactly this purpose, or None when the index has no such report.

    build_stats.json totals only count calls ledgered during this index's own build attempts. An
    index assembled from a warm cache (OURS ablations reuse the shared enrichment cache; B4 re-uses
    summaries; B3 re-uses embeddings; smoke tests warm all three) would otherwise report ~0 index
    calls. Known reports: B4 summary_stats.json:original_cost, OURS index.json:enrichment.
    cost_all_cached_calls, B3 vector_meta.json:embed_*_all_texts."""
    if not index_dir:
        return None
    d = Path(index_dir)
    if not d.is_dir():
        return None
    out = {"llm_calls": 0.0, "prompt_tokens": 0.0, "completion_tokens": 0.0, "llm_wall_ms": 0.0,
           "embed_tokens": 0.0, "embed_wall_ms": 0.0, "sources": []}
    s = read_json(d / "summary_stats.json")
    if isinstance(s, dict) and isinstance(s.get("original_cost"), dict):
        c = s["original_cost"]
        out.update(llm_calls=c.get("calls") or 0, prompt_tokens=c.get("prompt_tokens") or 0,
                   completion_tokens=c.get("completion_tokens") or 0, llm_wall_ms=c.get("wall_ms") or 0)
        out["sources"].append("summary_stats.json:original_cost")
    s = read_json(d / "index.json")
    c = ((s or {}).get("enrichment") or {}).get("cost_all_cached_calls") if isinstance(s, dict) else None
    if isinstance(c, dict):
        out.update(llm_calls=c.get("calls") or 0, prompt_tokens=c.get("prompt_tokens") or 0,
                   completion_tokens=c.get("completion_tokens") or 0, llm_wall_ms=c.get("wall_ms") or 0)
        out["sources"].append("index.json:enrichment.cost_all_cached_calls")
    s = read_json(d / "vector_meta.json")
    if isinstance(s, dict) and "embed_tokens_all_texts" in s:
        out.update(embed_tokens=s.get("embed_tokens_all_texts") or 0, embed_wall_ms=s.get("embed_wall_ms_all_texts") or 0)
        out["sources"].append("vector_meta.json:embed_*_all_texts")
    return out if out["sources"] else None


def load_index_stats(paths, cfg=None):
    """{(variant, corpus_set): build_stats} from index_<variant>_<set>.json files next to the runs."""
    out = {}
    for p in paths:
        pth = resolve_path(p, cfg)
        cands = [pth.parent] if pth.is_file() else [pth]
        for d in cands:
            if not d.is_dir():
                continue
            for f in sorted(d.rglob("index_*.json")):
                st = read_json(f)
                if isinstance(st, dict) and st.get("approach") and st.get("corpus_set"):
                    if "cached_cost" not in st:      # copies written before run.py recorded it
                        st["cached_cost"] = cached_index_cost(st.get("index_dir"))
                    out[(st["approach"], st["corpus_set"])] = st
    return out


# --------------------------------------------------------------------------
# scoring
# --------------------------------------------------------------------------

def _setting_idx(s):
    return SETTING_ORDER.index(s) if s in SETTING_ORDER else len(SETTING_ORDER)


SCORING_MODES = ["strict", "keyfacts"]


def apply_scoring(row, scoring):
    """The headline `match_correct` / `acc_match` of one scored row under the chosen scoring.
    `acc_match_strict` always keeps the strict matcher's value, so both can be reported."""
    row["acc_match_strict"] = row["match_correct"]
    if scoring == "keyfacts" and row.get("kf_correct") is not None:
        row["match_correct"] = row["kf_correct"]
    row["acc_match"] = row["match_correct"]
    row["acc_kf"] = row.get("kf_correct")
    return row


def score_records(records, questions, judge=None, match_cfg=None, abstain_token="NOT_FOUND", log=print, cfg=None,
                  unit_resolver=None, timeline_scoring="auto", scoring="strict"):
    qmap = {q.id: q for q in questions}
    scored, missing = [], set()
    if unit_resolver is None and any(TL.is_timeline(q) for q in questions):
        unit_resolver = TL.CorpusUnitResolver(cfg)     # loads a corpus set only when a structured citation needs it
    # judge in (question, setting) order, all approaches of a key together: when the judge budget
    # runs out, the verdicts cover complete paired groups instead of every record of the approaches
    # whose run files happen to sort first (B0, B1, ...) and none of OURS
    records = sorted(records, key=lambda r: (str(r.get("corpus")), str(r["question_id"]), _setting_idx(r["setting"]),
                                             _variant_sort_key(r["approach"], r.get("approach_base"))))
    for i, r in enumerate(records):
        q = qmap.get(r["question_id"])
        if q is None:
            missing.add(r["question_id"])
            continue
        row = score_record(r, q, match_cfg or MatchConfig(), abstain_token, unit_resolver=unit_resolver,
                           timeline_scoring=timeline_scoring)
        v = judge.verdict(q, r) if judge is not None else None
        if v:
            row.update(judge_correct=_f(v.get("correct")), judge_partial=_f(v.get("partially_correct")),
                       judge_reason=v.get("reason"), judge_source=v.get("source"), judge_label=v.get("judge"),
                       judge_weaker=v.get("weaker"))
            s = v.get("series")
            if isinstance(s, dict) and q.answer_type == "series":
                nc = row.get("tl_comparable") is False
                row.update(tl_judge_coverage=s.get("coverage"),
                           tl_judge_flag=_f(s.get("flags_definition_change")) if nc else None,
                           tl_judge_false_trend=_f(s.get("false_trend")) if nc else None)
        else:
            row.update(judge_correct=None, judge_partial=None, judge_reason=None, judge_source=None,
                       judge_label=None, judge_weaker=None)
        row["acc_judge"] = row["judge_correct"]
        row["acc_judge_or_partial"] = (None if row["judge_correct"] is None
                                       else float(bool(row["judge_correct"] or row["judge_partial"])))
        apply_scoring(row, scoring)
        scored.append(row)
        if judge is not None and judge.provider != "none" and (i + 1) % 25 == 0:
            log("  judged %d/%d  %s" % (i + 1, len(records), judge.summary()))
    if missing:
        log("WARNING: %d question ids in the runs are not in the question file(s): %s" % (
            len(missing), sorted(missing)[:10]))
    return scored


def _f(x):
    return None if x is None else float(bool(x))


# --------------------------------------------------------------------------
# aggregation
# --------------------------------------------------------------------------

def _mean(vals):
    v = [float(x) for x in vals if x is not None and not (isinstance(x, float) and math.isnan(x))]
    return (sum(v) / len(v)) if v else None


def _pct(vals, q):
    v = [float(x) for x in vals if x is not None]
    return float(np.percentile(np.array(v), q)) if v else None


def _variant_sort_key(approach, base):
    b = BASE_ORDER.index(base) if base in BASE_ORDER else len(BASE_ORDER)
    return (b, approach != base, approach)


FULL_KEYS = ("llm_calls", "prompt_tokens", "completion_tokens", "embed_tokens", "model_ms")


def index_totals(index_stats, variant, sets):
    """index_* : calls ledgered during this index's build attempts (build_stats totals).
    index_full_* : the same, raised to the original cost of every cached item the index uses
    (cached_index_cost), so an index assembled from a warm cache reports its real model cost."""
    keys = ("llm_calls", "embed_calls", "prompt_tokens", "completion_tokens", "embed_tokens", "build_ms")
    out = {"index_" + k: None for k in keys}
    out.update({"index_full_" + k: None for k in FULL_KEYS})
    out.update(index_size_bytes=None, index_complete=None, index_sets=None, index_cost_source=None)
    found = [index_stats[(variant, s)] for s in sets if (variant, s) in index_stats]
    if not found:
        return out
    for k in keys:
        out["index_" + k] = round(sum((st.get("totals") or {}).get(k) or 0 for st in found), 1)
    full = {k: 0.0 for k in FULL_KEYS}
    sources = set()
    for st in found:
        t = st.get("totals") or {}
        led = {"llm_calls": t.get("llm_calls") or 0, "prompt_tokens": t.get("prompt_tokens") or 0,
               "completion_tokens": t.get("completion_tokens") or 0, "embed_tokens": t.get("embed_tokens") or 0,
               "model_ms": t.get("llm_wall_ms") or 0}
        cc = st.get("cached_cost")
        if cc:
            rep = {"llm_calls": cc.get("llm_calls") or 0, "prompt_tokens": cc.get("prompt_tokens") or 0,
                   "completion_tokens": cc.get("completion_tokens") or 0, "embed_tokens": cc.get("embed_tokens") or 0,
                   "model_ms": (cc.get("llm_wall_ms") or 0) + (cc.get("embed_wall_ms") or 0)}
            for k in FULL_KEYS:
                full[k] += max(led[k], rep[k])
            sources.update(cc.get("sources") or [])
        else:
            for k in FULL_KEYS:
                full[k] += led[k]
            sources.add("build_stats totals")
    for k in FULL_KEYS:
        out["index_full_" + k] = round(full[k], 1)
    out["index_cost_source"] = "; ".join(sorted(sources))
    out["index_size_bytes"] = sum(st.get("size_bytes") or 0 for st in found)
    out["index_complete"] = all(st.get("complete") for st in found)
    out["index_sets"] = ",".join(st["corpus_set"] for st in found)
    return out


def aggregate(scored, index_stats=None):
    index_stats = index_stats or {}
    groups = defaultdict(list)
    for r in scored:
        sp = r.get("split") or "unknown"
        groups[(sp, r["approach"], r["corpus"], r["setting"])].append(r)
        groups[(sp, r["approach"], r["corpus"], "ALL")].append(r)
    rows = []
    for (split, approach, corpus, setting), rs in groups.items():
        base = rs[0].get("approach_base")
        ok = [r for r in rs if not r["errored"]]
        answerable = [r for r in rs if r["answer_type"] != "unanswerable"]
        judged = [r for r in rs if r.get("acc_judge") is not None]
        row = {
            "split": split,
            "approach": approach, "approach_base": base, "flags": json.dumps(rs[0].get("flags") or {}, sort_keys=True),
            "corpus": corpus, "setting": setting, "n": len(rs), "n_questions": len({r["question_id"] for r in rs}),
            "n_errors": sum(1 for r in rs if r["errored"]), "n_answerable": len(answerable),
            "n_unanswerable": len(rs) - len(answerable),
            "acc_judge": _mean(r["acc_judge"] for r in rs), "n_judged": len(judged),
            "acc_judge_or_partial": _mean(r["acc_judge_or_partial"] for r in rs),
            "judge_label": (judged[0].get("judge_label") if judged else None),
            "acc_match": _mean(r["acc_match"] for r in rs),
            "acc_match_strict": _mean(r.get("acc_match_strict") for r in rs),
            "acc_kf": _mean(r.get("acc_kf") for r in rs), "kf_recall": _mean(r.get("kf_recall") for r in rs),
            "n_keyfact": sum(1 for r in rs if r.get("kf_total")),
            "numeric_match": _mean(r.get("numeric_match") for r in rs), "n_numeric": sum(1 for r in rs if r["answer_type"] == "numeric"),
            "string_match": _mean(r.get("string_match") for r in rs),
            "token_f1": _mean(r.get("token_f1") for r in rs),
            "abstain_correct": _mean(r.get("abstain_correct") for r in rs),
            "false_abstain": _mean(r.get("false_abstain") for r in rs),
            "abstained": _mean(r["abstained"] for r in rs),
            "ev_hit": _mean(r.get("ev_hit") for r in rs), "ev_recall": _mean(r.get("ev_recall") for r in rs),
            "ret_hit": _mean(r.get("ret_hit") for r in rs), "ret_recall": _mean(r.get("ret_recall") for r in rs),
            "llm_calls_mean": _mean(r["llm_calls"] for r in ok), "embed_calls_mean": _mean(r["embed_calls"] for r in ok),
            "prompt_tokens_mean": _mean(r["prompt_tokens"] for r in ok),
            "completion_tokens_mean": _mean(r["completion_tokens"] for r in ok),
            "embed_tokens_mean": _mean(r["embed_tokens"] for r in ok),
            "evidence_chars_mean": _mean(r["evidence_chars"] for r in ok),
            "llm_calls_max": max([r["llm_calls"] or 0 for r in ok], default=None),
            "latency_p50_ms": _pct([r["latency_ms"] for r in ok], 50), "latency_p95_ms": _pct([r["latency_ms"] for r in ok], 95),
            "latency_noload_p50_ms": _pct([r["latency_noload_ms"] for r in ok], 50),
            "latency_noload_p95_ms": _pct([r["latency_noload_ms"] for r in ok], 95),
            "failed_calls": sum(r.get("failed_calls") or 0 for r in rs),
            "config_hashes": ",".join(sorted({r.get("config_hash") or "" for r in rs})),
            "ollama_versions": ",".join(sorted({r.get("ollama_version") or "" for r in rs} - {""})),
        }
        # approach-reported counters: mean per successful question (a 0/1 counter's mean is its rate)
        for k in sorted({k for r in ok for k in r if k.startswith("ts_")}):
            row[k] = _mean(r.get(k) for r in ok)
        # timeline (series) questions: their own metrics, over the group's series records only
        series = [r for r in rs if r["answer_type"] == "series"]
        if series:
            row["n_timeline"] = len(series)
            for k in TL.TIMELINE_METRICS + TL.TIMELINE_JUDGE_METRICS:
                row[k] = _mean(r.get(k) for r in series)
        sets = sorted({r["corpus_set"] for r in rs if r.get("corpus_set")})
        row.update(index_totals(index_stats, approach, sets))
        rows.append(row)
    rows.sort(key=lambda r: (_split_idx(r["split"]), r["corpus"],
                             (["ALL"] + SETTING_ORDER).index(r["setting"]) if r["setting"] in (["ALL"] + SETTING_ORDER) else 9,
                             _variant_sort_key(r["approach"], r["approach_base"])))
    return rows


def qtype_cost_rows(scored):
    """Cost and accuracy per split x approach x corpus x question type (lookup, locate, multi-section, cross-year,
    unanswerable, timeline, ...), so the paper can report the price of each kind of question."""
    groups = defaultdict(list)
    for r in scored:
        groups[(r.get("split") or "unknown", r["approach"], r["corpus"], r.get("qtype") or "?")].append(r)
    rows = []
    for (split, approach, corpus, qtype), rs in groups.items():
        ok = [r for r in rs if not r["errored"]]
        rows.append({
            "split": split, "approach": approach, "approach_base": rs[0].get("approach_base"), "corpus": corpus,
            "qtype": qtype, "answer_types": ",".join(sorted({r["answer_type"] for r in rs})), "n": len(rs),
            "n_questions": len({r["question_id"] for r in rs}), "n_errors": len(rs) - len(ok),
            "acc_match": _mean(r["acc_match"] for r in rs), "acc_judge": _mean(r.get("acc_judge") for r in rs),
            "acc_match_strict": _mean(r.get("acc_match_strict") for r in rs),
            "acc_kf": _mean(r.get("acc_kf") for r in rs), "kf_recall": _mean(r.get("kf_recall") for r in rs),
            "n_keyfact": sum(1 for r in rs if r.get("kf_total")),
            "n_judged": sum(1 for r in rs if r.get("acc_judge") is not None),
            "llm_calls_mean": _mean(r["llm_calls"] for r in ok), "llm_calls_max": max([r["llm_calls"] or 0 for r in ok], default=None),
            "embed_calls_mean": _mean(r["embed_calls"] for r in ok),
            "prompt_tokens_mean": _mean(r["prompt_tokens"] for r in ok),
            "completion_tokens_mean": _mean(r["completion_tokens"] for r in ok),
            "evidence_chars_mean": _mean(r["evidence_chars"] for r in ok),
            "latency_p50_ms": _pct([r["latency_ms"] for r in ok], 50), "latency_p95_ms": _pct([r["latency_ms"] for r in ok], 95),
            "latency_noload_p50_ms": _pct([r["latency_noload_ms"] for r in ok], 50),
            "latency_mean_ms": _mean(r["latency_ms"] for r in ok),
        })
    rows.sort(key=lambda r: (_split_idx(r["split"]), r["corpus"], r["qtype"], _variant_sort_key(r["approach"], r["approach_base"])))
    return rows


SPLIT_ORDER = ["dev", "test", "other", "unknown"]


def _split_idx(s):
    return SPLIT_ORDER.index(s) if s in SPLIT_ORDER else len(SPLIT_ORDER)


# --------------------------------------------------------------------------
# paired bootstrap
# --------------------------------------------------------------------------

def paired_bootstrap(pairs, n_boot=10000, seed=0, alpha=0.05):
    """pairs: [(cluster_id, a, b)]. Resamples clusters (questions) with replacement and returns
    {n_pairs, n_clusters, mean_a, mean_b, diff, ci_low, ci_high, p_boot} for mean(a) - mean(b)."""
    if not pairs:
        return None
    by = defaultdict(lambda: [0.0, 0.0, 0])
    for cid, a, b in pairs:
        s = by[cid]
        s[0] += a
        s[1] += b
        s[2] += 1
    cl = list(by.values())
    sa = np.array([c[0] for c in cl])
    sb = np.array([c[1] for c in cl])
    cnt = np.array([c[2] for c in cl], dtype=float)
    mean_a, mean_b = sa.sum() / cnt.sum(), sb.sum() / cnt.sum()
    diff = mean_a - mean_b
    rng = np.random.default_rng(seed)
    nC = len(cl)
    if nC < 2:
        return {"n_pairs": len(pairs), "n_clusters": nC, "mean_a": mean_a, "mean_b": mean_b, "diff": diff,
                "ci_low": None, "ci_high": None, "p_boot": None}
    diffs = np.empty(n_boot)
    step = max(1, 2_000_000 // nC)
    for s in range(0, n_boot, step):
        e = min(n_boot, s + step)
        idx = rng.integers(0, nC, size=(e - s, nC))
        n = cnt[idx].sum(1)
        diffs[s:e] = (sa[idx].sum(1) - sb[idx].sum(1)) / n
    lo, hi = np.percentile(diffs, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    p = 2.0 * min((diffs <= 0).mean(), (diffs >= 0).mean())
    return {"n_pairs": len(pairs), "n_clusters": nC, "mean_a": float(mean_a), "mean_b": float(mean_b),
            "diff": float(diff), "ci_low": float(lo), "ci_high": float(hi), "p_boot": float(min(1.0, p))}


def comparison_pairs(scored):
    """[(variant_a, variant_b)]: each OURS variant vs each baseline variant, and each OURS
    ablation vs full OURS."""
    variants = {}
    for r in scored:
        variants[r["approach"]] = r.get("approach_base")
    ours = sorted([v for v, b in variants.items() if b == "OURS"], key=lambda v: (v != "OURS", v))
    base = sorted([v for v, b in variants.items() if b != "OURS"], key=lambda v: _variant_sort_key(v, variants[v]))
    out = [(a, b) for a in ours for b in base]
    if "OURS" in variants:
        out += [(a, "OURS") for a in ours if a != "OURS"]
    # AGENTIC2 (2026-09-16): each AGENTIC2 variant vs every other variant; its ablations vs full AGENTIC2.
    # The AGENTIC3 family is left out here and owns that direction below, so a v3-vs-v2 pair is listed once.
    ag2 = sorted([v for v, b in variants.items() if b == "AGENTIC2"], key=lambda v: (v != "AGENTIC2", v))
    others = [v for v in base if variants[v] not in ("AGENTIC2", "AGENTIC3")]
    out += [(a, b) for a in ag2 for b in others] + [(a, b) for a in ag2 for b in ours]
    if "AGENTIC2" in variants:
        out += [(a, "AGENTIC2") for a in ag2 if a != "AGENTIC2"]
    # AGENTIC3 (2026-09-20): same pattern - each AGENTIC3 variant vs every other variant (the AGENTIC2 family
    # included), and each v3 ablation against full AGENTIC3, which is what makes the v3 ablations readable.
    ag3 = sorted([v for v, b in variants.items() if b == "AGENTIC3"], key=lambda v: (v != "AGENTIC3", v))
    others3 = [v for v in base if variants[v] != "AGENTIC3"]
    out += [(a, b) for a in ag3 for b in others3] + [(a, b) for a in ag3 for b in ours]
    if "AGENTIC3" in variants:
        out += [(a, "AGENTIC3") for a in ag3 if a != "AGENTIC3"]
    return out


def compare(scored, metrics=COMPARE_METRICS, n_boot=10000, seed=0):
    """Paired comparisons within one split and corpus (dev and test records are never paired or pooled)."""
    idx = defaultdict(dict)   # (split, corpus, approach) -> {(qid, setting): row}
    for r in scored:
        idx[(r.get("split") or "unknown", r["corpus"], r["approach"])][(r["question_id"], r["setting"])] = r
    groups = sorted({(r.get("split") or "unknown", r["corpus"]) for r in scored}, key=lambda g: (_split_idx(g[0]), g[1]))
    rows = []
    for split, corpus in groups:
        for a, b in comparison_pairs([r for r in scored if r["corpus"] == corpus and (r.get("split") or "unknown") == split]):
            ra, rb = idx.get((split, corpus, a), {}), idx.get((split, corpus, b), {})
            common = sorted(set(ra) & set(rb))
            if not common:
                continue
            settings = sorted({k[1] for k in common}, key=lambda s: SETTING_ORDER.index(s) if s in SETTING_ORDER else 9)
            for setting in ["ALL"] + settings:
                keys = [k for k in common if setting == "ALL" or k[1] == setting]
                for m in metrics:
                    pairs = []
                    for k in keys:
                        xa, xb = ra[k].get(m), rb[k].get(m)
                        if xa is None or xb is None:
                            continue
                        if m in ("llm_calls", "prompt_tokens", "latency_ms") and (ra[k]["errored"] or rb[k]["errored"]):
                            continue
                        pairs.append((k[0], float(xa), float(xb)))
                    res = paired_bootstrap(pairs, n_boot=n_boot, seed=seed)
                    if res is None:
                        continue
                    sig = (res["ci_low"] is not None and (res["ci_low"] > 0 or res["ci_high"] < 0))
                    rows.append({"split": split, "corpus": corpus, "setting": setting, "a": a, "b": b, "metric": m, **res,
                                 "significant": sig, "n_boot": n_boot, "seed": seed})
    return rows


# --------------------------------------------------------------------------
# outputs
# --------------------------------------------------------------------------

def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        atomic_write_text(path, "")
        return path
    cols = list(dict.fromkeys(k for r in rows for k in r))
    import io
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=cols, lineterminator="\n")
    w.writeheader()
    for r in rows:
        w.writerow({k: _csv_val(r.get(k)) for k in cols})
    # utf-8-sig so Excel opens Persian flags/labels correctly
    path.write_text(buf.getvalue(), encoding="utf-8-sig", newline="")
    return path


def _csv_val(v):
    if isinstance(v, float):
        return "%.6g" % v
    if isinstance(v, (dict, list)):
        return json.dumps(v, ensure_ascii=False)
    return "" if v is None else v


def _p(x, n=None):
    if x is None:
        return "–"
    s = "%.1f" % (100.0 * x)
    return s if n is None else "%s (%d)" % (s, n)


def _num(x, fmt="%.1f"):
    return "–" if x is None else fmt % x


def _secs(ms):
    return "–" if ms is None else ("%.1f" % (ms / 1000.0))


def _size(b):
    if b is None:
        return "–"
    for unit in ("B", "KB", "MB", "GB"):
        if b < 1024 or unit == "GB":
            return ("%d %s" % (b, unit)) if unit == "B" else ("%.1f %s" % (b, unit))
        b /= 1024.0


def _md_table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(c).replace("|", "\\|") for c in r) + " |")
    return "\n".join(out)


def render_md(rows, comps, meta):
    L = ["# ragbench results", "",
         "Generated %s. Questions: %s. Runs: %s." % (meta["generated"], ", ".join(meta["questions"]), ", ".join(meta["runs"])),
         "",
         "Judge: **%s**.%s" % (meta["judge_label"], (" Local fallback judge: weaker than a hosted judge%s; treat judge accuracy as indicative only." % (
             ", and it is the same model that produced the answers (self-grading bias)" if "same-model" in meta["judge_label"] else ""))
             if meta.get("judge_weaker") else ""),
         "Primary accuracy column: **%s**." % ("judge" if meta["primary"] == "acc_judge" else "match (no judge verdicts)"),
         "Scoring: **%s**.%s" % (meta.get("scoring", "strict"),
                                 " Questions carrying meta.key_facts are scored by their key facts; every other "
                                 "question keeps the strict matcher (column `Acc strict %`)."
                                 if meta.get("scoring") == "keyfacts" else ""),
         "Models: %s. Config hashes: %s." % (meta.get("models") or "–", meta.get("config_hashes") or "–"),
         ""]
    primary = meta["primary"]
    all_rows, all_comps = rows, comps
    groups = sorted({(r.get("split") or "unknown", r["corpus"]) for r in all_rows}, key=lambda g: (_split_idx(g[0]), g[1]))
    for split, corpus in groups:
        rows = [r for r in all_rows if (r.get("split") or "unknown") == split]
        comps = [c for c in all_comps if (c.get("split") or "unknown") == split]
        L += ["## Corpus %s, %s split" % (corpus, split.upper() if split == "test" else split), "",
              "### All settings pooled", ""]
        hdr = ["Approach", "n", "Acc judge % (n judged)", "Acc match %", "Acc strict %", "Key-fact acc % (n)",
               "Key-fact recall %", "Abstain %", "Ev. hit %", "Ev. recall %", "Calls",
               "Prompt tok", "Compl. tok", "Latency p50/p95 s", "No-load p50/p95 s", "Index chat calls†",
               "Index tokens in/out/embed†", "Index model time s†", "Index build s", "Index size", "Errors"]
        body = []
        for r in [x for x in rows if x["corpus"] == corpus and x["setting"] == "ALL"]:
            body.append([r["approach"], r["n"], _p(r["acc_judge"], r["n_judged"]), _p(r["acc_match"]),
                         _p(r.get("acc_match_strict")), _p(r.get("acc_kf"), r.get("n_keyfact")),
                         _p(r.get("kf_recall")), _p(r["abstained"]),
                         _p(r["ev_hit"]),
                         _p(r["ev_recall"]), _num(r["llm_calls_mean"], "%.2f"), _num(r["prompt_tokens_mean"], "%.0f"),
                         _num(r["completion_tokens_mean"], "%.0f"),
                         "%s / %s" % (_secs(r["latency_p50_ms"]), _secs(r["latency_p95_ms"])),
                         "%s / %s" % (_secs(r["latency_noload_p50_ms"]), _secs(r["latency_noload_p95_ms"])),
                         _num(r["index_full_llm_calls"], "%.0f"),
                         "–" if r["index_full_prompt_tokens"] is None else "%.0f / %.0f / %.0f" % (
                             r["index_full_prompt_tokens"], r["index_full_completion_tokens"], r["index_full_embed_tokens"]),
                         _secs(r["index_full_model_ms"]), _secs(r["index_build_ms"]), _size(r["index_size_bytes"]),
                         r["n_errors"]])
        L += [_md_table(hdr, body), ""]
        L += TL.render_timeline_md(meta.get("timeline_rows") or [], split, corpus)
        qrows = [x for x in (meta.get("qtype_rows") or []) if x["split"] == split and x["corpus"] == corpus]
        if len({x["qtype"] for x in qrows}) > 1:
            L += ["### Cost by question type (mean per record; errored records excluded from cost)", ""]
            hdr = ["Question type", "Approach", "n", "Acc match %", "Acc strict %", "Key-fact acc %", "Key-fact recall %",
                   "Calls", "Prompt tok", "Compl. tok", "Latency p50/p95 s", "No-load p50 s"]
            body = [[x["qtype"], x["approach"], x["n"], _p(x["acc_match"]), _p(x.get("acc_match_strict")),
                     _p(x.get("acc_kf")), _p(x.get("kf_recall")), _num(x["llm_calls_mean"], "%.2f"),
                     _num(x["prompt_tokens_mean"], "%.0f"), _num(x["completion_tokens_mean"], "%.0f"),
                     "%s / %s" % (_secs(x["latency_p50_ms"]), _secs(x["latency_p95_ms"])), _secs(x["latency_noload_p50_ms"])]
                    for x in qrows]
            L += [_md_table(hdr, body), ""]
        pooled = [x for x in rows if x["corpus"] == corpus and x["setting"] == "ALL"]
        ts_keys = sorted({k for x in pooled for k in x if k.startswith("ts_") and x.get(k) is not None})
        if ts_keys:
            L += ["### Approach counters (mean per question; a 0/1 counter's mean is its rate)", ""]
            hdr = ["Approach"] + [k[3:] for k in ts_keys]
            body = [[x["approach"]] + ["–" if x.get(k) is None else "%.2f" % x[k] for k in ts_keys]
                    for x in pooled if any(x.get(k) is not None for k in ts_keys)]
            L += [_md_table(hdr, body), ""]
        settings = [s for s in SETTING_ORDER if any(x["corpus"] == corpus and x["setting"] == s for x in rows)]
        if settings:
            L += ["### By setting (question language - document language)", ""]
            variants = [x["approach"] for x in rows if x["corpus"] == corpus and x["setting"] == "ALL"]
            cell = {(x["approach"], x["setting"]): x for x in rows if x["corpus"] == corpus}
            hdr = ["Approach"] + ["%s acc" % s for s in settings] + ["%s ev.hit" % s for s in settings]
            body = []
            for v in variants:
                line = [v]
                for s in settings:
                    c = cell.get((v, s))
                    line.append("–" if c is None else _p(c[primary], c["n_judged"] if primary == "acc_judge" else c["n"]))
                for s in settings:
                    c = cell.get((v, s))
                    line.append("–" if c is None else _p(c["ev_hit"]))
                body.append(line)
            L += [_md_table(hdr, body), ""]
        # markdown keeps the primary accuracy (pooled + per setting) and pooled evidence hit; the csv has everything
        cc = [c for c in comps if c["corpus"] == corpus and
              (c["metric"] == primary or (c["metric"] == "ev_hit" and c["setting"] == "ALL"))]
        if cc:
            L += ["### Paired bootstrap: A minus B (95%% CI, %d resamples of questions)" % cc[0]["n_boot"], ""]
            hdr = ["A", "B", "Setting", "Metric", "A %", "B %", "Diff (pp)", "95% CI (pp)", "Pairs (questions)", "p"]
            body = []
            for c in sorted(cc, key=lambda c: (c["a"] != "OURS", c["a"], c["b"], (["ALL"] + SETTING_ORDER).index(c["setting"]) if c["setting"] in ["ALL"] + SETTING_ORDER else 9, c["metric"])):
                ci = "–" if c["ci_low"] is None else "[%+.1f, %+.1f]%s" % (100 * c["ci_low"], 100 * c["ci_high"], " *" if c["significant"] else "")
                body.append([c["a"], c["b"], c["setting"], c["metric"], _p(c["mean_a"]), _p(c["mean_b"]),
                             "%+.1f" % (100 * c["diff"]), ci, "%d (%d)" % (c["n_pairs"], c["n_clusters"]),
                             "–" if c["p_boot"] is None else "%.3f" % c["p_boot"]])
            L += [_md_table(hdr, body), "", "`*` the CI excludes 0. Full metric set (calls, tokens, latency) in comparisons.csv.", ""]
    L += ["## Definitions", "",
          "- **Acc judge**: share of judged records the judge marked correct (partially correct counts as wrong; `acc_judge_or_partial` in results.csv). Rule verdicts without a model call: run error or empty answer = wrong; an exact `NOT_FOUND` = correct only for unanswerable questions.",
          "- **Acc match**: the headline under the chosen scoring (`%s`). **Acc strict**: always the strict matcher — numeric match (numeric), normalised exact/containment match (short, span), abstention (unanswerable). Numeric tolerance: %s." % (meta.get("scoring", "strict"), json.dumps(meta["match_cfg"])),
          "- **Key-fact acc / recall**: over the questions that carry `meta.key_facts` (locate, multi-section, cross-year). A key fact is a number or a short string a correct answer must contain; numbers match on the readings of `approaches/agentic/numbers.py`, strings on folded containment either way. Key-fact accuracy is 1 only when every key fact of the question is present. It rewards containment, so a verbose wrong answer can pass (see work/analysis/SCORING.md).",
          "- **Ev. hit**: any gold unit of the document language is inside the evidence given to the answer call (after the %s-character budget). **Ev. recall**: fraction of gold units inside it. B0 has no evidence (–)." % meta.get("budget", "?"),
          "- **Calls / tokens**: local chat calls and Ollama prompt_eval_count / eval_count per question (query phase only; index-time calls are in the Index columns).",
          "- **Index † columns**: model cost of building the index over the sets used (both document languages when pooled): the calls ledgered in its build attempts, raised to the original cost of every cached item it re-used (enrichment, node summaries, embeddings), so an index assembled from a warm or shared cache (e.g. an OURS ablation re-using the full OURS enrichment) is not reported as free. Index build s is the wall time of the ledgered build attempts only; results.csv has both the ledgered (`index_*`) and cache-inclusive (`index_full_*`) values and their source.",
          "- **Latency**: wall time of the whole answer() per question; no-load subtracts Ollama model load time and time spent waiting for the shared GPU lock behind other processes (`lock_wait_ms`). Errored records are excluded from cost and latency, and count as wrong for accuracy.",
          "- **Bootstrap**: paired on (question, setting); clusters = questions resampled with replacement; percentile interval.",
          ""]
    if meta.get("warnings"):
        L += ["## Warnings", ""] + ["- " + w for w in meta["warnings"]] + [""]
    return "\n".join(L)


def fairness_warnings(records, cfg):
    """Checks reviewers will ask about, from the records alone: evidence block within the shared
    budget, query-time chat calls within each approach's cap, one answer call for B0-B3."""
    caps = {"B0": 1, "B1": 1, "B2": 1, "B3": 1, "B4": cfg.caps["tree_nav_max_calls"],
            "B5": cfg.caps["grep_agent_max_steps"], "OURS": cfg.caps["ours_max_calls"],
            "AGENTIC": cfg.caps.get("agentic_max_calls", 8), "AGENTIC2": cfg.caps.get("agentic2_max_calls", 4),
            "AGENTIC3": cfg.caps.get("agentic3_max_calls", 4),
            "AGENTIC4": cfg.caps.get("agentic4_max_calls", 4), "LEX": cfg.caps.get("lex_max_calls", 3),
            "LEX2": cfg.caps.get("lex_max_calls", 3)}
    budget = cfg.evidence["max_chars"]
    over_budget = defaultdict(int)
    over_calls = defaultdict(int)
    cap_of = {}
    for r in records:
        if (r.get("evidence_chars") or 0) > budget:
            over_budget[r["approach"]] += 1
        cap = caps.get(r.get("approach_base"))
        if cap is not None and (r.get("llm_calls") or 0) > cap:
            over_calls[r["approach"]] += 1
            cap_of[r["approach"]] = cap
    out = ["%s: %d records with an evidence block over the %d-char budget" % (v, n, budget) for v, n in sorted(over_budget.items())]
    out += ["%s: %d records with more query-time chat calls than its cap (%d)" % (v, n, cap_of[v])
            for v, n in sorted(over_calls.items())]
    return out


def coverage_warnings(records):
    """Pooled rows are only comparable when every variant answered the same (question, setting) keys."""
    per = defaultdict(lambda: defaultdict(set))
    for r in records:
        per[(r.get("split") or "unknown", r.get("corpus"))][r["approach"]].add((r["question_id"], r["setting"]))
    out = []
    for (split, corpus), by in sorted(per.items(), key=lambda kv: (str(kv[0][0]), str(kv[0][1]))):
        if len(by) < 2:
            continue
        union = set().union(*by.values())
        short = {v: len(union - keys) for v, keys in by.items() if keys != union}
        if short:
            out.append("corpus %s (%s split): variants cover different (question, setting) keys (missing of %d: %s); pooled "
                       "rows are not directly comparable - rerun the missing keys or use --common-only" % (
                           corpus, split, len(union), ", ".join("%s %d" % (v, n) for v, n in sorted(short.items()))))
    return out


def question_file_split(path):
    """dev for pilot*/dev* question files, test for test*, other otherwise (same rule as ragbench.run)."""
    name = Path(str(path)).name.lower()
    if name.startswith("pilot") or name.startswith("dev"):
        return "dev"
    if name.startswith("test"):
        return "test"
    return "other"


def assign_splits(records, split_of_question, log=print):
    """Records without a stamped split (older runs) get the split of the question file that holds their question;
    a stamped split that disagrees with the question file is kept and reported."""
    warnings, n_filled, bad = [], 0, []
    for r in records:
        qs = split_of_question.get(r["question_id"])
        if not r.get("split"):
            if qs:
                r["split"] = qs
                n_filled += 1
        elif qs and r["split"] != qs:
            bad.append((r["approach"], r["question_id"], r["setting"], r["split"], qs))
    if n_filled:
        warnings.append("%d records had no split stamped (runs made before splits); taken from their question file" % n_filled)
    if bad:
        warnings.append("%d records carry a split that differs from their question file's: %s" % (len(bad), bad[:5]))
    return warnings


def aggregate_runs(runs, questions_paths, out_dir=None, judge_kind="none", judge_model=None, judge_max_calls=None,
                   judge_max_cost=None, match_cfg=None, n_boot=10000, seed=0, common_only=False, corpus=None,
                   approaches=None, cfg=None, log=print, variants=None, split=None, timeline_scoring="auto",
                   scoring="strict"):
    cfg = cfg or load_config()
    records, n_raw = load_run_records(runs, cfg)
    warnings = []
    questions, split_of_question = [], {}
    for qp in questions_paths:
        qfile = resolve_path(qp, cfg)
        qsplit = question_file_split(qfile)
        for q in load_questions(qfile):
            if split_of_question.get(q.id, qsplit) != qsplit:
                warnings.append("question %s appears in files of two splits" % q.id)
            split_of_question[q.id] = qsplit
            questions.append(q)
    warnings += assign_splits(records, split_of_question, log)
    if split:
        records = [r for r in records if (r.get("split") or "unknown") == split]
    if corpus:
        records = [r for r in records if r.get("corpus") == corpus]
    if approaches:
        records = [r for r in records if r["approach"] in approaches or r.get("approach_base") in approaches]
    if variants:
        records = [r for r in records if r["approach"] in variants]
    if common_only and records:
        per_c = defaultdict(lambda: defaultdict(set))
        for r in records:
            per_c[(r.get("split"), r["corpus"])][r["approach"]].add((r["question_id"], r["setting"]))
        keep = {c: set.intersection(*v.values()) for c, v in per_c.items()}
        before = len(records)
        records = [r for r in records if (r["question_id"], r["setting"]) in keep[(r.get("split"), r["corpus"])]]
        warnings.append("--common-only kept %d of %d records (pairs present for every approach)" % (len(records), before))
    out = resolve_path(out_dir, cfg) if out_dir else (resolve_path(runs[0], cfg) / "results" if resolve_path(runs[0], cfg).is_dir()
                                                      else resolve_path(runs[0], cfg).parent / "results")
    out.mkdir(parents=True, exist_ok=True)
    ledger = Ledger(path=str(out / "ledger_judge.jsonl"))
    judge = make_judge(judge_kind, cfg, ledger, model=judge_model, max_calls=judge_max_calls, max_cost_usd=judge_max_cost)
    log("records: %d (raw lines %d) from %s; questions: %d; judge: %s" % (len(records), n_raw, runs, len(questions), judge.label))
    mcfg = match_cfg or MatchConfig()
    scored = score_records(records, questions, judge, mcfg, cfg.evidence.get("abstain_token", "NOT_FOUND"), log=log, cfg=cfg,
                           timeline_scoring=timeline_scoring, scoring=scoring)
    n_kf = len({r["question_id"] for r in scored if r.get("kf_total")})
    log("key facts: %d of %d questions carry them (%d scored records); scoring=%s" % (
        n_kf, len({r["question_id"] for r in scored}), sum(1 for r in scored if r.get("kf_total")), scoring))
    index_stats = load_index_stats(runs, cfg)
    rows = aggregate(scored, index_stats)
    comps = compare(scored, n_boot=n_boot, seed=seed)
    timeline_rows = TL.timeline_rows(scored, SETTING_ORDER)
    qtype_rows = qtype_cost_rows(scored)
    # provenance warnings
    by_var = defaultdict(set)
    for r in records:
        by_var[r["approach"]].add(r.get("config_hash"))
    for v, hs in sorted(by_var.items()):
        if len(hs) > 1:
            warnings.append("%s mixes config hashes %s" % (v, sorted(h or "" for h in hs)))
    versions = sorted({(r.get("meta") or {}).get("ollama_version") for r in records} - {None, ""})
    if len(versions) > 1:
        warnings.append("records were answered under several Ollama versions %s" % versions)
    splits = sorted({r.get("split") or "unknown" for r in records}, key=_split_idx)
    if len(splits) > 1:
        warnings.append("runs hold several splits %s; every table and comparison is per split, never pooled" % splits)
    warnings += fairness_warnings(records, cfg)
    if not common_only:
        warnings += coverage_warnings(records)
    n_err = sum(1 for r in records if r.get("error"))
    if n_err:
        warnings.append("%d records have errors (counted as wrong)" % n_err)
    if judge.provider != "none":
        js = judge.summary()
        if js.get("budget"):
            warnings.append("judge budget stopped %d verdicts: %s" % (js["budget"], js.get("exhausted")))
        if js.get("error") or js.get("unparsable"):
            warnings.append("judge errors: %d, unparsable replies: %d" % (js.get("error", 0), js.get("unparsable", 0)))
    any_judged = any(r.get("acc_judge") is not None for r in scored)
    models = sorted({json.dumps(r.get("models"), sort_keys=True) for r in records if r.get("models")})
    meta = {"generated": now_iso(), "questions": [str(q) for q in questions_paths], "runs": [str(r) for r in runs],
            "judge_label": judge.label, "judge_weaker": judge.weaker, "primary": "acc_judge" if any_judged else "acc_match",
            "models": "; ".join(models), "config_hashes": ", ".join(sorted({h or "" for hs in by_var.values() for h in hs})),
            "match_cfg": mcfg.to_dict(), "budget": cfg.evidence.get("max_chars"), "warnings": warnings,
            "splits": splits, "ollama_versions": versions, "timeline_scoring": timeline_scoring, "scoring": scoring,
            "n_questions_with_key_facts": n_kf}
    atomic_write_text(out / "scored.jsonl", "".join(json.dumps(r, ensure_ascii=False, default=str) + "\n" for r in scored))
    write_csv(out / "results.csv", rows)
    write_csv(out / "comparisons.csv", comps)
    write_csv(out / "cost_by_qtype.csv", qtype_rows)
    if timeline_rows:
        write_csv(out / "timeline.csv", timeline_rows)
    atomic_write_text(out / "results.md", render_md(rows, comps, dict(meta, timeline_rows=timeline_rows, qtype_rows=qtype_rows)))
    atomic_write_json(out / "judge_summary.json", dict(judge.summary(), meta=meta), indent=1)
    for w in warnings:
        log("WARNING: " + w)
    log("wrote %s (results.csv %d rows, comparisons.csv %d rows, scored.jsonl %d rows%s)" % (
        out, len(rows), len(comps), len(scored), (", timeline.csv %d rows" % len(timeline_rows)) if timeline_rows else ""))
    return {"out": out, "rows": rows, "comparisons": comps, "scored": scored, "judge": judge.summary(), "meta": meta,
            "timeline_rows": timeline_rows, "qtype_rows": qtype_rows}


def main(argv=None):
    ap = argparse.ArgumentParser(description="Aggregate ragbench runs into results.csv / results.md")
    ap.add_argument("--runs", action="append", required=True, help="run dir or runs_*.jsonl (repeatable); 'runs/x' = work/runs/x")
    ap.add_argument("--questions", action="append", required=True, help="question JSONL (repeatable)")
    ap.add_argument("--out", default=None, help="output dir (default <first run dir>/results)")
    ap.add_argument("--judge", default="none", choices=["none", "openrouter", "ollama"],
                    help="none (match metrics only) | openrouter (paid, budgeted) | ollama (local, WEAKER fallback)")
    ap.add_argument("--judge-model", default=None)
    ap.add_argument("--judge-max-calls", type=int, default=None, help="per-invocation cap on judge model calls")
    ap.add_argument("--judge-max-cost", type=float, default=None, help="per-invocation cap on estimated judge cost (USD)")
    ap.add_argument("--rel-tol", type=float, default=MatchConfig.rel_tol)
    ap.add_argument("--no-rounding", action="store_true")
    ap.add_argument("--bootstrap", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--common-only", action="store_true", help="keep only (question, setting) pairs run by every approach")
    ap.add_argument("--corpus", default=None, choices=["A", "B", "C", "D", "K", "L"])
    ap.add_argument("--approaches", default=None, help="comma list of variants or bases to include (a base includes its ablations)")
    ap.add_argument("--variants", default=None, help="comma list of exact variant names to include (e.g. B0,B1,OURS without OURS+...)")
    ap.add_argument("--split", default=None, choices=["dev", "test", "other"],
                    help="keep only records of this split (tables are always per split)")
    ap.add_argument("--scoring", default="strict", choices=SCORING_MODES,
                    help="strict = the strict matcher is the headline for every question (default, unchanged); "
                         "keyfacts = questions carrying meta.key_facts use kf_correct instead, the rest are unchanged. "
                         "The strict number is always kept in acc_match_strict. Use a separate --out for a keyfacts run.")
    ap.add_argument("--timeline-scoring", default="auto", choices=["auto", "text", "structured"],
                    help="timeline (series) answers: auto/structured = a record's structured series when present, else "
                         "its answer text (the original rule); text = the answer text of EVERY approach (structured "
                         "output ignored; the fair headline comparison), with the structured scores as secondary tls_* "
                         "columns")
    args = ap.parse_args(argv)
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass
    if args.judge == "openrouter" and args.judge_max_calls is None:
        ap.error("--judge openrouter needs an explicit --judge-max-calls (credit is limited)")
    mcfg = MatchConfig(rel_tol=args.rel_tol, allow_rounding=not args.no_rounding)
    aggregate_runs(args.runs, args.questions, args.out, args.judge, args.judge_model, args.judge_max_calls,
                   args.judge_max_cost, mcfg, args.bootstrap, args.seed, args.common_only, args.corpus,
                   args.approaches.split(",") if args.approaches else None,
                   variants=args.variants.split(",") if args.variants else None, split=args.split,
                   timeline_scoring=args.timeline_scoring, scoring=args.scoring)
    return 0


if __name__ == "__main__":
    sys.exit(main())
