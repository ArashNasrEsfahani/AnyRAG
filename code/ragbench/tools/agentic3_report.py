"""AGENTIC3 reports, all of them read-only over files that already exist: no chat model call, no GPU.

    python -m ragbench.tools.agentic3_report cost      --runs runs/core_dev --runs runs/core_dev_pilot_timeline
    python -m ragbench.tools.agentic3_report ablations --runs runs/core_dev --runs runs/core_dev_pilot_timeline
    python -m ragbench.tools.agentic3_report ceiling   --questions dev_core.jsonl --questions pilot_timeline.jsonl
    python -m ragbench.tools.agentic3_report warm      --runs runs/v3warm --order work/runs/v3warm/order.tsv

Each command writes one Markdown file under `work/runs/core/` and prints it; `--out` moves it. Every number on
`agentic3_progress.html` comes from one of these four files.

  cost       AGENTIC3_COST.md   calls, tokens, the query clock and the wall clock (p50 / p95 / max), the routes
                                taken, records over the cap, and preprocessing: ingestion milliseconds per
                                document and per 100k characters from the knowledge base's `ingest_log`, plus the
                                lazy first-touch embedding these runs paid.
  ablations  V3_ABLATIONS.md    one row per AGENTIC3 variant: dev lookups (strict and key-fact), timeline
                                coverage / fully correct / false trends, calls, p50 latency, and the model seconds
                                the variant's dev records cost. Then the route mix with accuracy, and the FAST
                                gate's refusal reasons over every record of every variant.
  ceiling    DB_CEILING_V3.md   the bound of the database path against the REAL `facts` table, the way
                                `work/analysis/DB_CEILING.md` measured it on v2's cache: what share of the dev gold
                                values sits in a fact row at all, and what share sits in a row a label regex can
                                name (question words, and the terms the v3 planner actually produced).
  warm       WARM_CURVE.md      the index-by-use experiment: a database copy that starts with zero verified facts,
                                a seeded question order, and calls / latency / accuracy against the number of
                                questions already asked.

DATA FORMATS. Run records are `contracts.RunRecord` dicts in `work/runs/<dir>/runs_<variant>.jsonl`. The fields
read here are `trace.timing` (`query_clock_ms`, `lazy_index_ms`, `wall_with_embedding_ms`, `totals.llm_wall_ms`),
`trace.stats` (`route_*`, `calls`, `verify_calls`, `warm_rows`, `written_back`, `over_time`),
`trace.stages.route.detail.fast.reason` (why the FAST gate refused) and `trace.stages.lazy_index` (one entry per
document embedded on first touch, as AGENTIC2 writes it). Scores come from the aggregation `eval.aggregate`
already wrote next to the records: `<dir>/results/scored.jsonl` (strict match, one row per record),
`<dir>/results_keyfacts/scored.jsonl` (the key-fact scorer) and `<dir>/results/timeline.csv` (series metrics).

LATENCY. The latency reported here is `trace.timing.wall_with_embedding_ms` (query clock + lazy embedding), never
`latency_ms`: a record replayed from the per-question call cache spends 0.2 s replaying and would print as 0.2 s.
Both numbers are built the same way - program time plus each call's RECORDED wall time - so a replayed record
reports the latency of the run that produced it. `trace.timing.totals.session_ms` is the real elapsed time and is
not comparable across a replay.

MODEL SECONDS. `trace.timing.totals.llm_wall_ms` is the wall time of the question's chat calls as the model
reported it, replayed or not. Summed over a variant's records it is what that ablation costs the GPU when it is
run cold, which is the number the paper wants; the stage's measured duration is in `work/journal/events.jsonl`.
"""
import argparse
import csv
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

from ..approaches.agentic.numbers import find_numbers
from ..approaches.agentic.planner import question_terms
from ..approaches.agentic3.defaults import DEFAULTS
from ..config import load_config
from ..contracts import load_questions
from ..eval.aggregate import resolve_path
from ..kb import ingest as I
from ..kb import query as Q
from ..kb.schema import connect
from ..normalize import normalize
from ..util import read_jsonl, safe_name

SCALES = (1e-12, 1e-9, 1e-6, 1e-3, 1.0, 1e3, 1e6, 1e9, 1e12)   # DB_CEILING's scale-aware value rule
REL_TOL = 5e-4
ROUTES = ("FAST", "STANDARD", "ESCALATED", "TIMELINE")


def pct(vals, q):
    v = sorted(float(x) for x in vals if x is not None)
    if not v:
        return None
    pos = (len(v) - 1) * q / 100.0
    lo = int(math.floor(pos))
    hi = min(lo + 1, len(v) - 1)
    return v[lo] + (v[hi] - v[lo]) * (pos - lo)


def s(ms):
    return "-" if ms is None else "%.1f" % (ms / 1000.0)


def mean(vals):
    vals = [v for v in vals if v is not None]
    return sum(vals) / len(vals) if vals else None


def fmt(x, nd=2):
    return "-" if x is None else ("%." + str(nd) + "f") % x


def timing(rec):
    return ((rec.get("trace") or {}).get("timing") or {})


def stats(rec):
    return ((rec.get("trace") or {}).get("stats") or {})


def stages(rec):
    return ((rec.get("trace") or {}).get("stages") or {})


def latency_ms(rec):
    """The latency the paper quotes: query clock + lazy embedding, from the recorded call wall times."""
    t = timing(rec)
    if t.get("wall_with_embedding_ms") is not None:
        return t["wall_with_embedding_ms"]
    return (t.get("query_clock_ms") or 0.0) + (t.get("lazy_index_ms") or 0.0)


def route_of(rec):
    st = stats(rec)
    for r in ROUTES:
        if st.get("route_%s" % r.lower()):
            return r
    return (rec.get("structured") or {}).get("route")


def is_timeline(rec):
    return (rec.get("meta") or {}).get("answer_format") == "timeline"


def variants(dirs, base="AGENTIC3"):
    """-> {variant: [record]} for every AGENTIC3 variant found in the given run directories, errors dropped."""
    out = defaultdict(list)
    for d in dirs:
        for p in sorted(Path(d).glob("runs_%s*.jsonl" % safe_name(base))):
            for rec in read_jsonl(p):
                if rec.get("error"):
                    continue
                rec["_dir"] = Path(d).name
                out[rec["approach"]].append(rec)
    return out


def scored_rows(d, sub="results"):
    """-> {(dir, approach, question_id, setting): row} from <dir>/<sub>/scored.jsonl.

    The directory is part of the key because two run directories hold the same question under different document
    restrictions - `dev_core_timeline` and `pilot_timeline` share C-TL-divar-02 and C-TL-cafebazaar-01, and they
    do not always score the same - so a record must be scored by its OWN directory.
    """
    name = Path(d).name
    out = {}
    for row in read_jsonl(Path(d) / sub / "scored.jsonl"):
        out[(name, row["approach"], row["question_id"], row["setting"])] = row
    return out


def csv_rows(path):
    p = Path(path)
    if not p.is_file():
        return []
    with open(p, encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def num(row, key):
    v = (row or {}).get(key)
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------------------------------------------
# cost
# ---------------------------------------------------------------------------------------------------------------

def cost_blocks(recs):
    blocks = defaultdict(list)
    for r in recs:
        key = ("TL" if is_timeline(r) else r["corpus"], r["setting"])
        blocks[key].append(r)
        blocks[("ALL", "timeline" if is_timeline(r) else "single-fact")].append(r)
        blocks[("ALL", "all")].append(r)
    return blocks


def cost_table(blocks, cap_s):
    L = ["| Block | n | Calls mean (max) | Verify calls | Prompt tok | Completion tok | "
         "Latency p50 / p95 / max s | Query clock p50 / p95 / max s | Over %d s | Lazy-index s |" % cap_s,
         "|---|---|---|---|---|---|---|---|---|---|"]
    for k in sorted(blocks, key=lambda k: (k[0] == "ALL", k[0], k[1])):
        rs = blocks[k]
        lat = [latency_ms(r) for r in rs]
        clock = [timing(r).get("query_clock_ms") for r in rs]
        lazy = [timing(r).get("lazy_index_ms") or 0.0 for r in rs]
        L.append("| %s %s | %d | %.2f (%d) | %.2f | %d | %d | %s / %s / %s | %s / %s / %s | %d | %.1f |" % (
            k[0], k[1], len(rs), mean([stats(r).get("calls") or r["llm_calls"] for r in rs]),
            max(stats(r).get("calls") or r["llm_calls"] for r in rs),
            mean([stats(r).get("verify_calls", 0) for r in rs]),
            mean([r["prompt_tokens"] for r in rs]), mean([r["completion_tokens"] for r in rs]),
            s(pct(lat, 50)), s(pct(lat, 95)), s(max(lat)), s(pct(clock, 50)), s(pct(clock, 95)),
            s(max(c for c in clock if c is not None)),
            sum(1 for c in clock if c and c > cap_s * 1000), sum(lazy) / 1000.0))
    return L


def route_cost_table(recs):
    by = defaultdict(list)
    for r in recs:
        by[route_of(r) or "?"].append(r)
    L = ["| Route | n | Share | Calls mean | Verify calls | Vectors run | p50 latency s | p95 latency s | "
         "Candidates mean |", "|---|---|---|---|---|---|---|---|---|"]
    for name in ROUTES + ("?",):
        rs = by.get(name) or []
        if not rs and name == "?":
            continue
        if not rs:
            L.append("| %s | 0 | 0.0%% | - | - | - | - | - | - |" % name)
            continue
        lat = [latency_ms(r) for r in rs]
        L.append("| %s | %d | %.1f%% | %.2f | %.2f | %d | %s | %s | %.1f |" % (
            name, len(rs), 100.0 * len(rs) / len(recs), mean([stats(r).get("calls") or r["llm_calls"] for r in rs]),
            mean([stats(r).get("verify_calls", 0) for r in rs]),
            sum(1 for r in rs if (stages(r).get("vectors") or {}).get("ran") or stats(r).get("vector_hits")),
            s(pct(lat, 50)), s(pct(lat, 95)), mean([stats(r).get("candidates", 0) for r in rs])))
    return L


def ingest_preprocessing(cfg):
    """Stage timings from `ingest_log`, per document and per 100k characters of report text."""
    con = connect(I.kb_path(cfg), read_only=True)
    st = I.status(con, cfg)
    rows = list(con.execute(
        "SELECT l.stage AS stage, l.ms AS ms, d.n_chars AS n_chars FROM ingest_log l "
        "JOIN documents d ON d.doc_id = l.doc_id WHERE l.ok = 1 AND d.n_chars > 0 AND l.id IN "
        "(SELECT max(id) FROM ingest_log WHERE ok = 1 GROUP BY doc_id, stage)"))
    per100k = defaultdict(list)
    for r in rows:
        per100k[r["stage"]].append(100000.0 * r["ms"] / r["n_chars"])
    con.close()
    return st, per100k


def lazy_preprocessing(recs):
    """The vector path's embedding cost, summed over the records that ran it (`trace.stages.vectors`).

    AGENTIC3 records it per question, not per document: `embed` is `retrieve.embed_pending` filling `passages.vec`
    for documents that have none, `query` is the question's own two vectors. Both are stage `lazy-index` in the
    ledger and both are outside the 60 s query clock.
    """
    out = Counter()
    for r in recs:
        v = stages(r).get("vectors") or {}
        e, qv = v.get("embed") or {}, v.get("query") or {}
        if not v:
            continue
        out["records"] += 1
        for k in ("passages", "from_cache", "embedded", "calls", "docs"):
            out[k] += e.get(k) or 0
        out["embed_ms"] += e.get("embed_ms") or 0.0
        out["query_calls"] += qv.get("embed_calls") or 0
        out["query_ms"] += qv.get("embed_ms") or 0.0
    return out


def embed_stage_costs(cfg):
    """The `embed` stage per document, split by what it really paid: `detail` is {embedded, cached} and a document
    whose passages were already in the shared embedding cache costs milliseconds, not seconds."""
    con = connect(I.kb_path(cfg), read_only=True)
    new, cached = [], []
    for r in con.execute(
            "SELECT l.ms AS ms, l.rows AS rows, l.detail AS detail, d.n_chars AS n_chars FROM ingest_log l "
            "JOIN documents d ON d.doc_id = l.doc_id WHERE l.stage = 'embed' AND l.ok = 1 AND l.rows > 0 "
            "AND d.n_chars > 0"):
        row = dict(r)
        try:
            row["embedded"] = json.loads(row["detail"] or "{}").get("embedded") or 0
        except ValueError:
            row["embedded"] = 0
        (new if row["embedded"] else cached).append(row)
    con.close()
    return new, cached


def cmd_cost(args, cfg):
    dirs = [resolve_path(r, cfg) for r in args.runs]
    recs = [r for v, rs in variants(dirs, args.variant).items() if v == args.variant for r in rs]
    if not recs:
        raise SystemExit("no %s records in %s" % (args.variant, ", ".join(args.runs)))
    cap_s = float(cfg.caps.get("agentic3_max_seconds") or DEFAULTS["max_seconds_default"])
    cap_c = int(cfg.caps.get("agentic3_max_calls") or DEFAULTS["max_calls_default"])
    L = ["# %s cost report" % args.variant, "",
         "Runs: %s. Records without errors: %d. Caps: %d calls, %d s of query clock." % (
             ", ".join(args.runs), len(recs), cap_c, cap_s), "",
         "Latency = query clock + the lazy first-touch embedding, from each call's recorded wall time "
         "(`trace.timing.wall_with_embedding_ms`); the cap applies to the clock alone.", ""]
    L += cost_table(cost_blocks(recs), cap_s)
    over_calls = [r for r in recs if (stats(r).get("calls") or r["llm_calls"]) > cap_c]
    over_time = [r for r in recs if stats(r).get("over_time")]
    L += ["", "Over the caps: %d of %d records used more than %d calls, %d ran past %d s of query clock."
          % (len(over_calls), len(recs), cap_c, len(over_time), cap_s),
          "", "## Routes taken", ""]
    L += route_cost_table(recs)
    L += ["", "## Preprocessing 1: ingestion (no model call)", ""]
    st, per100k = ingest_preprocessing(cfg)
    L += ["Knowledge base `%s`, %.1f MB, %d documents, %d facts, %d passages, %d verified facts." % (
        st["path"], st["size_bytes"] / 1e6, st["counts"]["documents"], st["counts"]["facts"],
        st["counts"]["passages"], st["counts"]["verified_facts"]), "",
        "| Stage | documents | median ms | mean ms | max ms | total s | median ms / 100k chars |",
        "|---|---|---|---|---|---|---|"]
    for name, q in st["stages"].items():
        p = sorted(per100k.get(name) or [])
        L.append("| %s | %d | %.1f | %.1f | %.1f | %.1f | %s |" % (
            name, q["n"], q["median"], q["mean"], q["max"], q["total"] / 1000.0,
            fmt(p[len(p) // 2], 1) if p else "-"))
    e = lazy_preprocessing(recs)
    L += ["", "## Preprocessing 2: vectors, paid at query time (the only GPU work outside the query's own calls)",
          "", "The escalated and timeline routes are the only ones that touch vectors: %d of %d records ran the "
          "vector pass. A passage is embedded once, into `passages.vec` and into the shared embedding cache that "
          "B3 and AGENTIC2 also fill, so most passages of these runs cost nothing." % (e["records"], len(recs)), "",
          "| | value |", "|---|---|",
          "| passages still needing a vector when the route asked | %d |" % e["passages"],
          "| ... served by the shared embedding cache | %d |" % e["from_cache"],
          "| ... embedded during these runs | %d |" % e["embedded"],
          "| documents embedded on first touch | %d |" % e["docs"],
          "| passage embed calls / wall s | %d / %.1f |" % (e["calls"], e["embed_ms"] / 1000.0),
          "| question-vector embed calls / wall s | %d / %.1f |" % (e["query_calls"], e["query_ms"] / 1000.0)]
    new, cached = embed_stage_costs(cfg)
    for name, rows in (("computed its vectors", new), ("found them in the shared cache", cached)):
        if not rows:
            continue
        per_doc = sorted(r["ms"] / 1000.0 for r in rows)
        chars = sum(r["n_chars"] for r in rows)
        L += ["", "A document that %s (`ingest_log.embed`, %d documents, %d passages): median %.2f s, mean %.2f s, "
              "max %.2f s per document; %.1f s per 100k characters." % (
                  name, len(rows), sum(r["rows"] for r in rows), per_doc[len(per_doc) // 2],
                  sum(per_doc) / len(per_doc), per_doc[-1], 100000.0 * sum(per_doc) / max(1, chars))]
    return "\n".join(L) + "\n"


# ---------------------------------------------------------------------------------------------------------------
# ablations
# ---------------------------------------------------------------------------------------------------------------

def variant_scores(dirs, variant):
    """Accuracy over the corpus A dev records under both scorers: strict match (`results/scored.jsonl`) and the
    key-fact scorer (`results_keyfacts/scored.jsonl`), which report the same field under different rules."""
    hit = {"results": [], "results_keyfacts": []}
    for d in dirs:
        for sub in hit:
            for key, row in scored_rows(d, sub).items():
                if key[1] == variant and row.get("corpus") == "A" and row.get("acc_match") is not None:
                    hit[sub].append(float(row["acc_match"]))
    return mean(hit["results"]), mean(hit["results_keyfacts"]), len(hit["results"])


def timeline_row(dirs, variant):
    """The variant's ALL row of the largest timeline set found, i.e. `pilot_timeline` (8) over `dev_core_timeline`."""
    rows = [row for d in dirs for row in csv_rows(Path(d) / "results" / "timeline.csv")
            if row.get("approach") == variant and row.get("setting") == "ALL"]
    return max(rows, key=lambda r: int(r.get("n") or 0)) if rows else None


def cmd_ablations(args, cfg):
    dirs = [resolve_path(r, cfg) for r in args.runs]
    by_variant = variants(dirs, args.variant)
    base = args.variant
    if base not in by_variant:
        raise SystemExit("no %s records in %s" % (base, ", ".join(args.runs)))
    order = [base] + sorted(v for v in by_variant if v != base)
    L = ["# AGENTIC3 dev ablations", "",
         "Dev files only: `dev_core.jsonl` (corpus A, 5 questions x 4 settings), `dev_core_timeline.jsonl` (2) and "
         "`pilot_timeline.jsonl` (8 timeline questions, corpus C fa-fa). Lookup columns are the 20 corpus A "
         "records; timeline columns are the 8 `pilot_timeline` questions. Latency is the query clock plus the lazy "
         "embedding, p50 over every record of the variant.", "",
         "| Variant | n | dev A strict | dev A key-fact | TL coverage | TL fully correct | TL false trends | "
         "calls | p50 s | model s |", "|---|---|---|---|---|---|---|---|---|---|"]
    rows = {}
    for v in order:
        recs = by_variant[v]
        strict, kf, n_a = variant_scores(dirs, v)
        tl = timeline_row(dirs, v)
        model_s = sum((timing(r).get("totals") or {}).get("llm_wall_ms") or 0.0 for r in recs) / 1000.0
        rows[v] = {"n": len(recs), "strict": strict, "kf": kf, "n_a": n_a,
                   "coverage": num(tl, "tl_coverage"), "correct": num(tl, "tl_series_correct"),
                   "false": num(tl, "tl_false_trend"),
                   "calls": mean([stats(r).get("calls") or r["llm_calls"] for r in recs]),
                   "p50": pct([latency_ms(r) for r in recs], 50), "model_s": model_s}
        d = rows[v]
        L.append("| %s | %d | %s | %s | %s | %s | %s | %.2f | %s | %.0f |" % (
            v, d["n"], fmt(d["strict"]), fmt(d["kf"]), fmt(d["coverage"], 3), fmt(d["correct"], 3),
            fmt(d["false"], 2), d["calls"], s(d["p50"]), d["model_s"]))
    b = rows[base]
    L += ["", "Deltas against %s (dev A strict %s, timeline coverage %s):" % (
        base, fmt(b["strict"]), fmt(b["coverage"], 3)), ""]
    for v in order[1:]:
        d = rows[v]
        L.append("- `%s`: dev A %+.2f, timeline coverage %+.3f, model time %+.0f s" % (
            v.split("+", 1)[-1], (d["strict"] or 0) - (b["strict"] or 0),
            (d["coverage"] or 0) - (b["coverage"] or 0), d["model_s"] - b["model_s"]))
    L += ["", "## Route mix of %s, with accuracy" % base, ""]
    L += route_accuracy_table(dirs, by_variant[base], base)
    L += ["", "## The FAST gate: every refusal, over every record of every variant", ""]
    reasons = Counter()
    fired = 0
    for v, recs in by_variant.items():
        for r in recs:
            if route_of(r) == "FAST":
                fired += 1
            f = ((stages(r).get("route") or {}).get("detail") or {}).get("fast") or {}
            if f.get("reason"):
                reasons[f["reason"]] += 1
    total = sum(len(rs) for rs in by_variant.values())
    L += ["FAST fired %d times in %d records (%d variants). The gate was reached %d times; it refused for:"
          % (fired, total, len(by_variant), sum(reasons.values())), ""]
    for name, n in reasons.most_common():
        L.append("- `%s` %d" % (name, n))
    return "\n".join(L) + "\n"


def route_accuracy_table(dirs, recs, variant):
    scored, kf = {}, {}
    for d in dirs:
        scored.update(scored_rows(d, "results"))
        kf.update(scored_rows(d, "results_keyfacts"))
    by = defaultdict(list)
    for r in recs:
        by[route_of(r) or "?"].append(r)
    L = ["| Route | n | Share | Strict correct | Key-fact correct | p50 latency s | Calls |",
         "|---|---|---|---|---|---|---|"]
    for name in ROUTES:
        rs = by.get(name) or []
        if not rs:
            L.append("| %s | 0 | 0.0%% | - | - | - | - |" % name)
            continue
        keys = [(r["_dir"], variant, r["question_id"], r["setting"]) for r in rs]
        sc = [num(scored.get(k), "acc_match") for k in keys]
        kc = [num(kf.get(k), "acc_match") for k in keys]
        sc = [x for x in sc if x is not None]
        kc = [x for x in kc if x is not None]
        L.append("| %s | %d | %.1f%% | %s | %s | %s | %.2f |" % (
            name, len(rs), 100.0 * len(rs) / len(recs),
            "%d/%d (%.2f)" % (sum(sc), len(sc), mean(sc)) if sc else "-",
            "%d/%d (%.2f)" % (sum(kc), len(kc), mean(kc)) if kc else "-",
            s(pct([latency_ms(r) for r in rs], 50)),
            mean([stats(r).get("calls") or r["llm_calls"] for r in rs])))
    return L


# ---------------------------------------------------------------------------------------------------------------
# the database-path ceiling
# ---------------------------------------------------------------------------------------------------------------

def gold_values(q, d_lang):
    """Every number a correct answer must carry. -> [{tag, value, docs, units}], one per wanted value.

    A lookup contributes its `answer_value`, a multi-section question the numeric entries of `meta.key_facts`, and
    a timeline question one entry per period, each restricted to the report OF that period (the gold rule of
    `eval/timeline.py`).
    """
    out = []
    scope = q.scope(d_lang) or []
    gold = q.gold(d_lang) or []
    if q.answer_value is not None:
        out.append({"kind": "answer", "tag": "answer", "value": float(q.answer_value), "docs": scope,
                    "units": gold})
    for kf in (q.meta or {}).get("key_facts") or []:
        if kf.get("kind") == "number" and kf.get("value") is not None:
            out.append({"kind": "key_fact", "tag": "key fact", "value": float(kf["value"]), "docs": scope,
                        "units": gold})
    for item in (q.meta or {}).get("answer_series") or []:
        v = item.get("value_base") if item.get("value_base") is not None else item.get("value")
        if v is None:
            continue
        out.append({"kind": "period", "tag": "period %s" % item.get("period"), "value": float(v),
                    "docs": [item["doc"]] if item.get("doc") else scope,
                    "units": (item.get("gold_units") or {}).get(d_lang) or []})
    return out


def readings_of(row):
    """Every number the fact row can be read as: the typed value, the extractor's readings, the glitch readings."""
    out = set()
    if row["value_num"] is not None:
        out.add(float(row["value_num"]))
    if row["value_hi"] is not None:
        out.add(float(row["value_hi"]))
    for tok in find_numbers(row["value_text"] or ""):
        out.update(float(x) for x in tok.readings)
    try:
        out.update(float(x) for x in json.loads(row["alt_readings"] or "[]"))
    except (TypeError, ValueError):
        pass
    return out


def carries(row, gold):
    """DB_CEILING's scale-aware rule: a reading counts when it is gold x 10^k within 5e-4 relative."""
    for r in readings_of(row):
        for k in SCALES:
            t = gold * k
            if t == 0.0:
                if r == 0.0:
                    return True
                continue
            if abs(r - t) <= REL_TOL * abs(t):
                return True
    return False


def word_terms(text, top=None):
    """Content words of the question as a `Terms` matcher, split by script the way a plan's terms are."""
    words = question_terms(text) if top == 6 else _all_words(text)
    fa = [w for w in words if any("؀" <= ch <= "ۿ" for ch in w)]
    en = [w for w in words if w not in fa]
    return Q.Terms({"terms": {"fa": fa, "en": en}})


def _all_words(text):
    out = []
    for w in re.split(r"[\s‌،,.?!؟:;()«»\"']+", normalize(text)):
        if len(w) >= 3 and not re.fullmatch(r"[\d.,%/]+", w) and w not in out:
            out.append(w)
    return out


def plan_terms(record):
    plan = (stages(record).get("plan") or {})
    t = plan.get("terms") or {}
    return Q.Terms({"terms": {"fa": list(t.get("fa") or []), "en": list(t.get("en") or [])}})


def fact_rows(con, doc_ids):
    if not doc_ids:
        return []
    sql = ("SELECT fact_id, doc_id, unit_id, label_fold, value_text, value_num, value_hi, alt_readings "
           "FROM facts WHERE doc_id IN (%s)" % ", ".join("?" * len(doc_ids)))
    return [dict(r) for r in con.execute(sql, list(doc_ids))]


def unit_texts(con, doc_ids):
    """Every unit of the documents, so a value missing from `facts` can be looked for in the raw text (the third
    measure of `work/analysis/DB_CEILING.md`: extraction loss, as opposed to a value the corpus never carried)."""
    if not doc_ids:
        return []
    sql = "SELECT doc_id, unit_id, text FROM units WHERE doc_id IN (%s)" % ", ".join("?" * len(doc_ids))
    return [dict(r) for r in con.execute(sql, list(doc_ids))]


def in_text(units, docs, gold):
    """-> the first unit whose raw text carries the value, or None. `facts` is built from these units, so a value
    found here and not there is extraction loss rather than a value the corpus never carried."""
    for u in units:
        if u["doc_id"] not in docs:
            continue
        for tok in find_numbers(u["text"] or ""):
            for r in tok.readings:
                for k in SCALES:
                    t = gold * k
                    if t and abs(float(r) - t) <= REL_TOL * abs(t):
                        return u
    return None


def cmd_ceiling(args, cfg):
    con = connect(I.kb_path(cfg), read_only=True)
    plans = {}
    for d in [resolve_path(r, cfg) for r in args.runs or []]:
        for rec in read_jsonl(Path(d) / ("runs_%s.jsonl" % safe_name(args.variant))):
            if not rec.get("error"):
                plans[(rec["question_id"], rec["setting"])] = rec
    langs = ("fa", "en")
    per_group, misses, cache, seen = defaultdict(lambda: Counter()), [], {}, set()
    for path in args.questions:
        for q in load_questions(resolve_path(path, cfg)):
            if q.id in seen:
                continue
            seen.add(q.id)
            for d_lang in langs:
                # only a language the question HAS an edition in: corpus C is Persian only, and asking a Persian
                # label to match English question words would measure nothing.
                if not q.gold(d_lang):
                    continue
                wanted = gold_values(q, d_lang)
                if not wanted:
                    continue
                docs = sorted({d for w in wanted for d in w["docs"]})
                if not all(docs):
                    continue
                key = tuple(docs)
                if key not in cache:
                    cache[key] = (fact_rows(con, list(docs)), unit_texts(con, list(docs)))
                rows, units = cache[key]
                if not rows:
                    continue
                qt6 = word_terms(q.text(d_lang), top=6)
                qta = word_terms(q.text(d_lang))
                pt = next((plan_terms(plans[(q.id, st)]) for st in ("%s-%s" % (l, d_lang) for l in langs)
                           if (q.id, st) in plans), None)
                for w in wanted:
                    hits = [r for r in rows if r["doc_id"] in set(w["docs"]) and carries(r, w["value"])]
                    at_unit = [r for r in hits if r["unit_id"] in set(w["units"])]
                    named6 = [r for r in hits if qt6 and qt6.concepts_in(r["label_fold"] or "")]
                    nameda = [r for r in hits if qta and qta.concepts_in(r["label_fold"] or "")]
                    namedp = [r for r in hits if pt and pt.concepts_in(r["label_fold"] or "")]
                    for k in ((q.corpus, d_lang), ("ALL", "ALL"), ("ALL", d_lang), ("ALL", w["kind"])):
                        c = per_group[k]
                        c["n"] += 1
                        c["present"] += bool(hits)
                        c["at_unit"] += bool(at_unit)
                        c["named6"] += bool(named6)
                        c["named_any"] += bool(nameda)
                        c["named_plan"] += bool(namedp) if pt else 0
                        c["has_plan"] += 1 if pt else 0
                    if not hits or not named6:
                        misses.append({"q": q.id, "lang": d_lang, "tag": w["tag"], "value": w["value"],
                                       "present": bool(hits), "at_unit": bool(at_unit),
                                       "named_any": bool(nameda), "named_plan": bool(namedp) if pt else None,
                                       "in_text": None if hits else in_text(units, set(w["docs"]), w["value"])})
    con.close()
    return ceiling_md(args, per_group, misses)


def ceiling_md(args, per_group, misses):
    L = ["# The database path's ceiling on dev, measured against the real facts table", "",
         "Method: `work/analysis/DB_CEILING.md` section 1, re-run against `work/kb/kb.sqlite` instead of v2's "
         "in-memory fact cache. One row per GOLD VALUE (a lookup's answer, a multi-section question's key facts, "
         "and one per period of a timeline question, each restricted to the report of that period) x document "
         "language. A fact row carries the value when one of its readings - the typed `value_num`, the extractor's "
         "readings of `value_text`, the stored `alt_readings` - equals the gold value x 10^k for k in "
         "{0, +-3, +-6, +-9, +-12} within 5e-4 relative.", "",
         "Question files: %s." % ", ".join(args.questions), "",
         "| Group | values | in a fact row | at the gold unit | label matches the question's top 6 words | "
         "... any question word | ... the v3 plan's own terms |", "|---|---|---|---|---|---|---|"]
    def share(c, k, d="n"):
        return "-" if not c[d] else "%.0f%% (%d/%d)" % (100.0 * c[k] / c[d], c[k], c[d])
    for key in sorted(per_group, key=lambda k: (k[0] == "ALL", k)):
        c = per_group[key]
        L.append("| %s %s | %d | %s | %s | %s | %s | %s |" % (
            key[0], key[1], c["n"], share(c, "present"), share(c, "at_unit"), share(c, "named6"),
            share(c, "named_any"), share(c, "named_plan", "has_plan")))
    L += ["", "## Every gold value the question's top 6 words could not name", ""]
    if not misses:
        L.append("None: every gold value sits in a fact row whose label matches the question's top 6 content words.")
    else:
        L += ["| question | doc lang | value | in a fact row | at the gold unit | any question word | plan terms | "
              "in the raw unit text |", "|---|---|---|---|---|---|---|---|"]
        for m in misses:
            L.append("| %s (%s) | %s | %s | %s | %s | %s | %s | %s |" % (
                m["q"], m["tag"], m["lang"], m["value"], "yes" if m["present"] else "NO",
                "yes" if m["at_unit"] else "no", "yes" if m["named_any"] else "no",
                "-" if m["named_plan"] is None else ("yes" if m["named_plan"] else "no"),
                "-" if m["present"] else ((m["in_text"] or {}).get("unit_id") or "NOT IN THE CORPUS")))
    lost = [m for m in misses if not m["present"] and m["in_text"]]
    if lost:
        bare = [m for m in lost if len((m["in_text"]["text"] or "").split()) <= 3]
        L += ["", "%d of the %d values are in the raw unit text and not in `facts`: extraction loss, which the "
              "lookup measurement on v2's cache put at 0 of 53. %d of them sit in a unit whose whole text is the "
              "number and at most two other words, i.e. a number the extraction detached from its label (spec "
              "section 11 risk 1), which the kpi rule did not recover:" % (len(lost), len(misses), len(bare)), ""]
        for m in lost:
            L.append("- `%s` = `%s` (%s, %s)" % (m["in_text"]["unit_id"], (m["in_text"]["text"] or "")[:60],
                                                 m["q"], m["tag"]))
    return "\n".join(L) + "\n"


# ---------------------------------------------------------------------------------------------------------------
# the warm curve
# ---------------------------------------------------------------------------------------------------------------

def warm_order(path):
    """The seeded order as it was run: [(position, question_id, setting)] from the run's order.tsv."""
    out = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        parts = line.rstrip("\r").split("\t")
        if len(parts) == 5:
            out.append((int(parts[0]), parts[3], parts[4]))
    return out


def cmd_warm(args, cfg):
    d = resolve_path(args.runs, cfg)
    recs = {}
    for p in sorted(Path(d).glob("runs_%s*.jsonl" % safe_name(args.variant))):
        for rec in read_jsonl(p):
            if not rec.get("error"):
                recs[(rec["question_id"], rec["setting"])] = rec
    order = warm_order(args.order or (Path(d) / "order.tsv"))
    warm_score = {k[2:]: v for k, v in scored_rows(d).items()}
    # the cold reference is matched on the question FILE as well as the id: `dev_core_timeline` and
    # `pilot_timeline` hold the same two C questions under different document restrictions.
    files = {(r["question_id"], r["setting"]): (r.get("meta") or {}).get("questions_file") for r in recs.values()}
    # FIXED 2026-09-21 (truth pass). This used to write `cold[k] = rec` in directory order, so when two cold
    # directories both held a (question, setting) pair the LAST one silently won and the report named no
    # directory. `C-TL-divar-02` fa-fa and `C-TL-cafebazaar-01` fa-fa are exactly that case, and
    # `C-TL-cafebazaar-01` disagrees with itself (0.0 under the 6-document restriction, 1.0 under the
    # 16-document one), so which copy is used is worth a whole question of 28. The first directory named on the
    # command line now wins, every duplicate is recorded, and the report says which copy it used.
    cold, cold_score, cold_dir, dupes = {}, {}, {}, []
    for cd in [resolve_path(x, cfg) for x in args.cold or []]:
        sc = scored_rows(cd)
        for rec in read_jsonl(Path(cd) / ("runs_%s.jsonl" % safe_name(args.variant))):
            k = (rec["question_id"], rec["setting"])
            if rec.get("error") or (rec.get("meta") or {}).get("questions_file") != files.get(k):
                continue
            if k in cold:
                dupes.append((k, cold_dir[k], Path(cd).name,
                              num(cold_score.get(k), "acc_match"),
                              num(sc.get((Path(cd).name, args.variant) + k), "acc_match")))
                continue
            cold[k] = rec
            cold_score[k] = sc.get((Path(cd).name, args.variant) + k)
            cold_dir[k] = Path(cd).name
    L = ["# Warm mode: does the database get better as it is used?", "",
         "Spec section 8. A COPY of the knowledge base with `verified_facts` emptied, write-back on, the `warm` "
         "flag on, and the 28 dev questions asked one per process in a seeded order (`order.tsv`, seed 42). "
         "Dev only, on its own database file, so nothing reaches the cold results. The cold column is the same "
         "question from the v3dev run, which is a different run of the same code: the planner's run-to-run noise "
         "on this set is +-0.10 on accuracy, so read the CALLS and the warm hits, not the accuracy.", "",
         "| # | question | setting | route | warm rows | calls | latency s | written back | verified rows after |",
         "|---|---|---|---|---|---|---|---|---|"]
    total_written = 0
    curve = []
    for pos, qid, st in order:
        r = recs.get((qid, st))
        if r is None:
            L.append("| %d | %s | %s | (not run) | | | | | |" % (pos, qid, st))
            continue
        w = stats(r)
        total_written += w.get("written_back", 0)
        curve.append({"pos": pos, "warm_rows": w.get("warm_rows", 0),
                      "calls": w.get("calls") or r["llm_calls"], "latency": latency_ms(r),
                      "route": route_of(r), "written": w.get("written_back", 0),
                      "acc": num(warm_score.get((qid, st)), "acc_match")})
        L.append("| %d | %s | %s | %s | %d | %d | %s | %d | %d |" % (
            pos, qid, st, route_of(r), w.get("warm_rows", 0), w.get("calls") or r["llm_calls"],
            s(latency_ms(r)), w.get("written_back", 0), total_written))
    L += ["", "## The curve, in quarters of the seeded order", "",
          "| questions already asked | n | warm rows mean | questions with a warm hit | calls mean | "
          "p50 latency s | correct |", "|---|---|---|---|---|---|---|"]
    q = max(1, len(curve) // 4)
    for i in range(0, len(curve), q):
        part = curve[i:i + q]
        if not part:
            continue
        acc = [p["acc"] for p in part if p["acc"] is not None]
        L.append("| %d-%d | %d | %.1f | %d | %.2f | %s | %s |" % (
            part[0]["pos"] - 1, part[-1]["pos"] - 1, len(part), mean([p["warm_rows"] for p in part]),
            sum(1 for p in part if p["warm_rows"]), mean([p["calls"] for p in part]),
            s(pct([p["latency"] for p in part], 50)),
            "%d/%d" % (sum(acc), len(acc)) if acc else "-"))
    hits = sum(1 for p in curve if p["warm_rows"])
    L += ["", "Warm hits: %d of %d questions read a verified fact back. Rows written: %d." % (
        hits, len(curve), total_written)]
    if cold:
        cw = [cold[(qid, st)] for _p, qid, st in order if (qid, st) in cold]
        ca = [num(cold_score.get((qid, st)), "acc_match") for _p, qid, st in order]
        wa = [p["acc"] for p in curve if p["acc"] is not None]
        ca = [x for x in ca if x is not None]
        L += ["", "| | n | calls mean | p50 latency s | FAST | correct |", "|---|---|---|---|---|---|",
              "| warm | %d | %.2f | %s | %d | %s |" % (
                  len(curve), mean([p["calls"] for p in curve]), s(pct([p["latency"] for p in curve], 50)),
                  sum(1 for p in curve if p["route"] == "FAST"),
                  "%d/%d (%.2f)" % (sum(wa), len(wa), mean(wa)) if wa else "-"),
              "| cold (v3dev, same questions) | %d | %.2f | %s | %d | %s |" % (
                  len(cw), mean([stats(r).get("calls") or r["llm_calls"] for r in cw]),
                  s(pct([latency_ms(r) for r in cw], 50)),
                  sum(1 for r in cw if route_of(r) == "FAST"),
                  "%d/%d (%.2f)" % (sum(ca), len(ca), mean(ca)) if ca else "-"), "",
              "The warm arm asked one question per process, so every one of its latencies carries the ~0.8 s the "
              "first database query of a process costs against a 220 MB file; the cold reference answered many "
              "questions in one process and does not. The warm latencies are therefore an over-estimate.", ""]
        used = sorted(set(cold_dir.values()))
        L += ["Cold copies used, by directory: %s."
              % ", ".join("%s (%d records)" % (d, sum(1 for v in cold_dir.values() if v == d)) for d in used), ""]
        if dupes:
            L += ["**%d (question, setting) pairs are held by more than one cold directory.** The FIRST directory "
                  "named on the command line is the one used; the others are listed here rather than dropped "
                  "silently, because a duplicated pair can disagree with itself." % len(dupes), "",
                  "| question | setting | used (kept) | also in | kept score | dropped score |",
                  "|---|---|---|---|---|---|"]
            for (qid, st2), kept, other, sa, sb in dupes:
                L.append("| %s | %s | %s | %s | %s | %s |" % (qid, st2, kept, other, sa, sb))
            L.append("")
        L += warm_vs_cold(order, recs, warm_score, cold, cold_score)
    return "\n".join(L) + "\n"


PLAN_IGNORE = ("model",)        # the raw decode transcript, not a field of the plan


def plan_diff(a, b):
    """Which fields of two recorded plans differ. [] means the two plans are identical.

    FIXED 2026-09-21 (truth pass). This used to compare `plan["terms"]` alone, which is one field of the plan
    object, and it is why the first warm write-up said "all four warm losses had an identical plan". On the 28
    warm records a terms-only comparison calls 17 of them the same plan; comparing every plan field calls 10 of
    them the same, and the seven it changes its mind about differ on `facets`, `unit`, `entity` or
    `entity_terms` - fields the router and the retriever both read. A difference is attributable to warm mode
    only when the WHOLE plan is unchanged, so the whole plan is what is compared. `plan["model"]` is skipped: it
    is the decode transcript the plan was parsed from, not a field of the plan.
    """
    pa, pb = (a or {}), (b or {})
    keys = sorted((set(pa) | set(pb)) - set(PLAN_IGNORE))
    return [k for k in keys
            if json.dumps(pa.get(k), sort_keys=True, ensure_ascii=False)
            != json.dumps(pb.get(k), sort_keys=True, ensure_ascii=False)]


def warm_vs_cold(order, recs, warm_score, cold, cold_score):
    """Every question the two arms answered differently, with the route and the plan check that says why.

    A warm row changes the ROUTE (`warm_verified_fact` is a FAST entry), so a difference is only attributable to
    warm mode when the plan is unchanged; the plan column says whether it is, and names the fields that moved when
    it is not. The comparison is over the whole plan object, not over `plan["terms"]` - see `plan_diff`.
    """
    L = ["## What warm mode changed, question by question", "",
         "The `same plan` column compares the WHOLE recorded plan object. A row that says anything but `yes` is "
         "warm mode plus a different plan, and is not attributable to warm mode alone.", "",
         "| question | setting | warm route | cold route | warm rows | warm | cold | same plan |",
         "|---|---|---|---|---|---|---|---|"]
    n = same_plan = 0
    for _pos, qid, st in order:
        r, c = recs.get((qid, st)), cold.get((qid, st))
        a, b = num(warm_score.get((qid, st)), "acc_match"), num(cold_score.get((qid, st)), "acc_match")
        if r is None or c is None or a is None or b is None or a == b:
            continue
        n += 1
        diff = plan_diff(stages(r).get("plan"), stages(c).get("plan"))
        same_plan += not diff
        L.append("| %s | %s | %s | %s | %d | %s | %s | %s |" % (
            qid, st, route_of(r), route_of(c), stats(r).get("warm_rows", 0), "correct" if a else "wrong",
            "correct" if b else "wrong", "yes" if not diff else "no (%s)" % ", ".join(diff)))
    if not n:
        L.append("| - | - | - | - | - | - | - | - |")
    else:
        L += ["", "%d of the %d discordant records have a byte-identical plan; the other %d changed the plan as "
                  "well as the route, so warm mode is not the only difference in them."
              % (same_plan, n, n - same_plan)]
    return L


# ---------------------------------------------------------------------------------------------------------------

DEFAULT_OUT = {"cost": "work/runs/core/AGENTIC3_COST.md", "ablations": "work/runs/core/V3_ABLATIONS.md",
               "ceiling": "work/runs/core/DB_CEILING_V3.md", "warm": "work/runs/core/WARM_CURVE.md"}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("command", choices=sorted(DEFAULT_OUT))
    ap.add_argument("--runs", action="append", default=[])
    ap.add_argument("--cold", action="append", default=[], help="warm: run directories of the cold reference")
    ap.add_argument("--questions", action="append", default=[], help="ceiling: question files")
    ap.add_argument("--order", default=None, help="warm: the seeded order file (default <runs>/order.tsv)")
    ap.add_argument("--variant", default="AGENTIC3")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass
    cfg = load_config()
    if args.command == "warm":
        args.runs = args.runs[0] if args.runs else "runs/v3warm"
    text = {"cost": cmd_cost, "ablations": cmd_ablations, "ceiling": cmd_ceiling, "warm": cmd_warm}[
        args.command](args, cfg)
    print(text)
    out = Path(resolve_path(args.out or DEFAULT_OUT[args.command], cfg))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    print("-> %s" % out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
