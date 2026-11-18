"""DEV analysis tables over an aggregated run directory (no model calls, no judge).

    .venv/Scripts/python.exe -m ragbench.tools.dev_analysis --runs runs/dev_full --questions ragbench/data/questions/pilot.jsonl \
        [--results runs/dev_full/results] [--out runs/dev_full/DEV_TABLES.md] [--fix1 runs/dev_fix1]

Reads runs_*.jsonl, results/scored.jsonl, results/comparisons.csv, results/sanity.json and index_*.json, and writes
markdown tables:
  1. index-time cost per approach and set
  2. headline per variant x corpus (all settings pooled) and per variant x corpus x setting:
     match accuracy, evidence hit, abstention, calls, prompt/completion tokens, latency p50/p95
  3. paired bootstrap (harness comparisons.csv): OURS vs every baseline, every OURS ablation vs OURS; plus pooled
     monolingual (fa-fa+en-en) vs cross-lingual (fa-en+en-fa) and all-corpora groups (aggregate.paired_bootstrap)
  4. wrong-number diagnosis of numeric answers (a distractor number copied from the evidence), per variant, with the
     evidence item kind the number came from; OURS vs OURS+facts_bare plan identity
  5. B4 abstentions on answerable questions with gold in the evidence
  6. sanity checks that did not PASS
"""
import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from ..contracts import load_questions
from ..eval.aggregate import SETTING_ORDER, paired_bootstrap, resolve_path
from ..normalize import normalize
from ..util import read_json, read_jsonl
from ..approaches.ours.indexer import _NUM, number_variants

ORDER = ["B0", "B1", "B2", "B3", "B4", "B5", "OURS", "OURS+facts_bare", "OURS+no_enrichment", "OURS+no_facts"]
CROSS = ("fa-en", "en-fa")
MONO = ("fa-fa", "en-en")
_CITE = re.compile(r"\[\d+\]")


def vorder(v):
    return (ORDER.index(v) if v in ORDER else len(ORDER), v)


def pct(x):
    return "–" if x is None else "%.1f" % (100.0 * x)


def mean(xs):
    xs = [x for x in xs if x is not None]
    return (sum(xs) / len(xs)) if xs else None


def md_table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def num_set(text):
    s = set()
    t = _CITE.sub(" ", normalize(text or ""))
    for tok in _NUM.findall(t):
        s |= {v for v in number_variants(tok)}
    return s


def num_tokens(text):
    return _NUM.findall(_CITE.sub(" ", normalize(text or "")))


def canon(tok):
    try:
        return "%.10g" % float(tok.replace(",", ""))
    except ValueError:
        return tok


# --------------------------------------------------------------------------

def summary_row(rs):
    ok = [r for r in rs if not r.get("errored")]
    lat = [r["latency_ms"] / 1000.0 for r in ok if r.get("latency_ms") is not None]
    p50, p95 = (np.percentile(lat, 50), np.percentile(lat, 95)) if lat else (None, None)
    answerable = [r for r in rs if r.get("answer_type") != "unanswerable"]
    return [len(rs), pct(mean(r.get("acc_match") for r in rs)), pct(mean(r.get("ev_hit") for r in rs)),
            pct(mean(r.get("abstained") for r in rs)), pct(mean(r.get("abstained") for r in answerable)),
            "%.2f" % mean(r.get("llm_calls") for r in ok) if ok else "–",
            "%.0f" % mean(r.get("prompt_tokens") for r in ok) if ok else "–",
            "%.0f" % mean(r.get("completion_tokens") for r in ok) if ok else "–",
            ("%.1f / %.1f" % (p50, p95)) if lat else "–", sum(1 for r in rs if r.get("errored"))]


SUMMARY_HEADER = ["Variant", "n", "Acc match %", "Ev. hit %", "Abstain %", "Abstain on answerable %", "Calls",
                  "Prompt tok", "Compl. tok", "Latency p50 / p95 s", "Errors"]


def headline_tables(scored):
    out = []
    by = defaultdict(list)
    for r in scored:
        by[(r["corpus"], r["approach"])].append(r)
        by[("ALL", r["approach"])].append(r)
    corpora = sorted({c for c, _ in by if c != "ALL"}) + ["ALL"]
    for c in corpora:
        rows = [[v] + summary_row(by[(c, v)]) for v in sorted({v for cc, v in by if cc == c}, key=vorder)]
        out.append("### Corpus %s, all settings pooled\n\n%s" % (c if c != "ALL" else "A+B+C", md_table(SUMMARY_HEADER, rows)))
    return "\n\n".join(out)


def setting_tables(scored):
    out = []
    by = defaultdict(list)
    for r in scored:
        by[(r["corpus"], r["setting"], r["approach"])].append(r)
    for c in sorted({r["corpus"] for r in scored}):
        rows = []
        for s in SETTING_ORDER:
            for v in sorted({k[2] for k in by if k[0] == c and k[1] == s}, key=vorder):
                rows.append([s, v] + summary_row(by[(c, s, v)]))
        out.append("### Corpus %s by setting\n\n%s" % (c, md_table(["Setting"] + SUMMARY_HEADER, rows)))
    return "\n\n".join(out)


def compact_setting_table(scored, metric="acc_match"):
    """variant rows x (corpus, setting) columns of one metric."""
    cols = [(c, s) for c in sorted({r["corpus"] for r in scored}) for s in SETTING_ORDER
            if any(r["corpus"] == c and r["setting"] == s for r in scored)]
    by = defaultdict(list)
    for r in scored:
        by[(r["approach"], r["corpus"], r["setting"])].append(r)
        by[(r["approach"], r["corpus"], "mono")].append(r) if r["setting"] in MONO else None
        by[(r["approach"], r["corpus"], "cross")].append(r) if r["setting"] in CROSS else None
    rows = []
    for v in sorted({r["approach"] for r in scored}, key=vorder):
        row = [v]
        for c, s in cols:
            row.append(pct(mean(x.get(metric) for x in by[(v, c, s)])))
        rows.append(row)
    return md_table(["Variant"] + ["%s %s" % cs for cs in cols], rows)


# --------------------------------------------------------------------------

def comparison_tables(comp_csv, metrics=("acc_match", "ev_hit")):
    rows = list(csv.DictReader(open(comp_csv, encoding="utf-8-sig")))
    out = []

    def fmt(r):
        lo, hi = r.get("ci_low"), r.get("ci_high")
        ci = "[%+.1f, %+.1f]" % (100 * float(lo), 100 * float(hi)) if lo not in (None, "") else "–"
        return [r["corpus"], r["setting"], r["a"], r["b"], r["metric"], pct(float(r["mean_a"])), pct(float(r["mean_b"])),
                "%+.1f" % (100 * float(r["diff"])), ci, "%s (%s)" % (r["n_pairs"], r["n_clusters"]),
                "%.3f" % float(r["p_boot"]) if r.get("p_boot") not in (None, "") else "–",
                "yes" if r.get("significant") == "True" else ""]

    head = ["Corpus", "Setting", "A", "B", "Metric", "A %", "B %", "Diff pp", "95% CI pp", "Pairs (questions)", "p", "CI excl. 0"]
    for title, pred in (("OURS vs baselines", lambda r: r["a"] == "OURS" and not r["b"].startswith("OURS")),
                        ("OURS ablations minus OURS", lambda r: r["b"] == "OURS" and r["a"].startswith("OURS+"))):
        sel = [r for r in rows if pred(r) and r["metric"] in metrics]
        sel.sort(key=lambda r: (r["corpus"], (["ALL"] + SETTING_ORDER).index(r["setting"]), vorder(r["a"]), vorder(r["b"]),
                                r["metric"]))
        out.append("### %s (harness comparisons.csv)\n\n%s" % (title, md_table(head, [fmt(r) for r in sel])))
    return "\n\n".join(out)


def pooled_comparisons(scored, pairs_wanted, metrics=("acc_match", "ev_hit"), n_boot=10000):
    """Bootstrap over question clusters for pooled setting groups the harness does not form."""
    idx = defaultdict(dict)
    for r in scored:
        idx[r["approach"]][(r["corpus"], r["question_id"], r["setting"])] = r
    groups = [("A+B", "mono", lambda k: k[0] in ("A", "B") and k[2] in MONO),
              ("A+B", "cross", lambda k: k[0] in ("A", "B") and k[2] in CROSS),
              ("A", "mono", lambda k: k[0] == "A" and k[2] in MONO), ("A", "cross", lambda k: k[0] == "A" and k[2] in CROSS),
              ("B", "mono", lambda k: k[0] == "B" and k[2] in MONO), ("B", "cross", lambda k: k[0] == "B" and k[2] in CROSS),
              ("A+B+C", "all", lambda k: True)]
    rows = []
    for a, b in pairs_wanted:
        if a not in idx or b not in idx:
            continue
        common = sorted(set(idx[a]) & set(idx[b]))
        for g, gname, pred in groups:
            keys = [k for k in common if pred(k)]
            for m in metrics:
                pairs = [(k[1], float(idx[a][k][m]), float(idx[b][k][m])) for k in keys
                         if idx[a][k].get(m) is not None and idx[b][k].get(m) is not None]
                res = paired_bootstrap(pairs, n_boot=n_boot, seed=0)
                if not res:
                    continue
                ci = "[%+.1f, %+.1f]" % (100 * res["ci_low"], 100 * res["ci_high"]) if res["ci_low"] is not None else "–"
                sig = res["ci_low"] is not None and (res["ci_low"] > 0 or res["ci_high"] < 0)
                rows.append([g, gname, a, b, m, pct(res["mean_a"]), pct(res["mean_b"]), "%+.1f" % (100 * res["diff"]), ci,
                             "%d (%d)" % (res["n_pairs"], res["n_clusters"]), "%.3f" % res["p_boot"] if res["p_boot"] is not None else "–",
                             "yes" if sig else ""])
    head = ["Corpora", "Settings", "A", "B", "Metric", "A %", "B %", "Diff pp", "95% CI pp", "Pairs (questions)", "p", "CI excl. 0"]
    return md_table(head, rows)


# --------------------------------------------------------------------------

def evidence_items_with_text(rec):
    """The included evidence items with their text: retrieved_units matched in order to evidence_items (a cut item keeps
    its full text here, so the check is slightly generous)."""
    ret = list(rec.get("retrieved_units") or [])
    out, j = [], 0
    for ei in rec.get("evidence_items") or []:
        while j < len(ret) and not (ret[j].get("doc_id") == ei.get("doc_id") and ret[j].get("section_id") == ei.get("section_id")
                                    and ret[j].get("kind") == ei.get("kind")):
            j += 1
        if j < len(ret):
            out.append(ret[j])
            j += 1
    return out


def wrong_number_analysis(records, scored, questions):
    """Numeric answerable questions answered (not abstained) and wrong by match: classify the answer's numbers."""
    sc = {(r["approach"], r["question_id"], r["setting"]): r for r in scored}
    qmap = {q.id: q for q in questions}
    counts = defaultdict(lambda: defaultdict(int))
    cases = defaultdict(list)
    for rec in records:
        v = rec["approach"]
        s = sc.get((v, rec["question_id"], rec["setting"]))
        q = qmap.get(rec["question_id"])
        if not s or not q or q.answer_type != "numeric":
            continue
        counts[v]["numeric"] += 1
        if s.get("errored") or s.get("abstained") or s.get("acc_match"):
            counts[v]["correct" if s.get("acc_match") else ("abstained" if s.get("abstained") else "errored")] += 1
            continue
        counts[v]["wrong"] += 1
        gold = num_set(q.answer_fa) | num_set(q.answer_en) | (num_set(str(q.answer_value)) if q.answer_value is not None else set())
        qnums = num_set(q.question_fa) | num_set(q.question_en)
        toks = [t for t in num_tokens(rec.get("answer_text")) if not (number_variants(t) & (gold | qnums))]
        if not num_tokens(rec.get("answer_text")):
            counts[v]["wrong_no_number"] += 1
            continue
        if not toks:
            counts[v]["wrong_gold_number_only"] += 1     # the gold number is there, but match failed (unit/extra parts)
            continue
        items = evidence_items_with_text(rec)
        src = []
        for t in toks:
            vs = number_variants(t)
            kinds = [(i + 1, it["kind"]) for i, it in enumerate(items) if vs & num_set(it.get("text"))]
            if kinds:
                src.append((t, kinds))
        ev_hit = bool(s.get("ev_hit"))
        if src:
            key = "wrong_distractor_gold_shown" if ev_hit else "wrong_distractor_gold_missing"
            counts[v][key] += 1
            first_kind = src[0][1][0][1]
            counts[v]["%s_from_%s" % (key, "fact" if first_kind == "fact" else "text")] += 1
            cases[v].append({"q": rec["question_id"], "setting": rec["setting"], "ev_hit": ev_hit,
                             "answer": " ".join((rec.get("answer_text") or "").split())[:160],
                             "gold": q.answer(rec["q_lang"])[:80], "numbers": src[:3]})
        else:
            counts[v]["wrong_number_not_in_evidence"] += 1
    return counts, cases


def wrong_number_tables(counts, cases, focus=("OURS", "OURS+facts_bare", "OURS+no_enrichment", "OURS+no_facts")):
    head = ["Variant", "Numeric records", "Correct", "Abstained", "Wrong", "Distractor number, gold shown",
            "… first seen in a fact item", "Distractor number, gold not shown", "Number not in evidence",
            "Gold number but match failed", "No number"]
    rows = []
    for v in sorted(counts, key=vorder):
        c = counts[v]
        rows.append([v, c["numeric"], c["correct"], c["abstained"], c["wrong"], c["wrong_distractor_gold_shown"],
                     c["wrong_distractor_gold_shown_from_fact"], c["wrong_distractor_gold_missing"],
                     c["wrong_number_not_in_evidence"], c["wrong_gold_number_only"], c["wrong_no_number"]])
    out = [md_table(head, rows)]
    for v in focus:
        if not cases.get(v):
            continue
        crow = [[x["q"], x["setting"], "yes" if x["ev_hit"] else "no", x["answer"].replace("|", "/"), x["gold"].replace("|", "/"),
                 "; ".join("%s in %s" % (t, ", ".join("[%d] %s" % k for k in ks[:3])) for t, ks in x["numbers"])]
                for x in sorted(cases[v], key=lambda x: (x["q"], SETTING_ORDER.index(x["setting"])))]
        out.append("#### %s: wrong answers with a number from the evidence\n\n%s" % (
            v, md_table(["Question", "Setting", "Gold shown", "Answer", "Gold", "Wrong number in evidence item(s)"], crow)))
    return "\n\n".join(out)


def plan_identity(records, a="OURS", b="OURS+facts_bare"):
    pa = {(r["question_id"], r["setting"]): (r.get("trace") or {}).get("used_plan") for r in records if r["approach"] == a}
    pb = {(r["question_id"], r["setting"]): (r.get("trace") or {}).get("used_plan") for r in records if r["approach"] == b}
    common = set(pa) & set(pb)
    same = sum(1 for k in common if pa[k] == pb[k])
    return same, len(common)


# --------------------------------------------------------------------------

def b4_abstentions(records, scored, questions):
    sc = {(r["approach"], r["question_id"], r["setting"]): r for r in scored}
    qmap = {q.id: q for q in questions}
    rows, per = [], defaultdict(lambda: [0, 0, 0])
    for rec in records:
        if rec["approach"] != "B4":
            continue
        s = sc.get(("B4", rec["question_id"], rec["setting"]))
        q = qmap.get(rec["question_id"])
        if not s or not q or q.answer_type == "unanswerable":
            continue
        key = (rec["corpus"], rec["setting"])
        per[key][0] += 1
        if not s.get("abstained"):
            continue
        per[key][1] += 1
        if not s.get("ev_hit"):
            continue
        per[key][2] += 1
        gold = set(q.gold(rec["d_lang"]))
        items = rec.get("evidence_items") or []
        pos = [(i + 1, it["kind"], it.get("chars"), bool(it.get("cut")), [u.split("::", 1)[1] for u in it["unit_ids"] if u in gold])
               for i, it in enumerate(items) if gold & set(it.get("unit_ids") or [])]
        first = pos[0] if pos else None
        tr = rec.get("trace") or {}
        rows.append([rec["question_id"], rec["setting"], q.type, len(items), rec.get("evidence_chars"),
                     "[%d] %s %s chars%s" % (first[0], first[1], first[2], " cut" if first[3] else "") if first else "–",
                     ", ".join(first[4][:3]) if first else "–", len(pos),
                     (tr.get("stats") or {}).get("narrowed_items", "–")])
    summ = [[c, s, n, ab, g] for (c, s), (n, ab, g) in sorted(per.items(), key=lambda kv: (kv[0][0], SETTING_ORDER.index(kv[0][1])))]
    return (md_table(["Corpus", "Setting", "Answerable records", "Abstained", "Abstained with gold in evidence"], summ) + "\n\n" +
            md_table(["Question", "Setting", "Type", "Items", "Evidence chars", "First gold item", "Gold unit(s) in it",
                      "Items with gold", "Narrowed items"], sorted(rows, key=lambda r: (r[0], SETTING_ORDER.index(r[1])))))


def index_cost_table(run_dir):
    rows = []
    for f in sorted(Path(run_dir).glob("index_*.json")):
        st = read_json(f) or {}
        tot = st.get("totals") or {}
        cc = st.get("cached_cost") or {}
        cs = st.get("corpus_stats") or {}
        secs = cs.get("sections") or 0
        build_s = (tot.get("build_ms") or 0) / 1000.0
        calls = tot.get("llm_calls") or 0
        rows.append([st.get("approach"), st.get("corpus_set"), cs.get("docs"), secs, calls, tot.get("embed_calls") or 0,
                     "%d / %d" % (tot.get("prompt_tokens") or 0, tot.get("completion_tokens") or 0),
                     "%.2f" % ((tot.get("llm_wall_ms") or 0) / 3.6e6), "%.2f" % (build_s / 3600.0),
                     "%.1f" % (build_s / secs) if secs else "–", "%.1f" % ((tot.get("llm_wall_ms") or 0) / 1000.0 / calls) if calls else "–",
                     int(cc.get("llm_calls") or 0), "%.2f" % ((cc.get("llm_wall_ms") or 0) / 3.6e6), bool(st.get("complete"))])
    head = ["Approach", "Set", "Docs", "Sections", "Chat calls (this build)", "Embed calls", "Tokens in / out",
            "Model h", "Build h", "Build s / section", "Model s / call", "Calls incl. cache", "Model h incl. cache", "Complete"]
    return md_table(head, rows)


def sanity_table(results_dir):
    js = read_json(Path(results_dir) / "sanity.json") or {}
    checks = js.get("checks") or []
    bad = [[c["check"], c["status"], str(c.get("detail"))[:300].replace("|", "/")] for c in checks if c.get("status") != "PASS"]
    return "%d checks, %d PASS, %d not PASS\n\n%s" % (len(checks), sum(1 for c in checks if c.get("status") == "PASS"), len(bad),
                                                     md_table(["Check", "Status", "Detail"], bad) if bad else "(all PASS)")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--questions", required=True)
    ap.add_argument("--results", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--fix1", default=None, help="an older run dir (e.g. runs/dev_fix1) for the wrong-number count before the fix")
    a = ap.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    run_dir = resolve_path(a.runs)
    res_dir = resolve_path(a.results) if a.results else run_dir / "results"
    questions = load_questions(resolve_path(a.questions))
    scored = [r for r in read_jsonl(res_dir / "scored.jsonl") if (r.get("split") or "dev") == "dev"]
    records = [r for f in sorted(run_dir.glob("runs_*.jsonl")) for r in read_jsonl(f) if (r.get("split") or "dev") == "dev"]
    parts = ["# DEV tables (generated by ragbench.tools.dev_analysis)", "",
             "Runs: `%s`. Questions: `%s`. Match accuracy only (no judge)." % (run_dir, a.questions), ""]
    parts += ["## 1. Index-time cost", "", index_cost_table(run_dir), ""]
    parts += ["## 2. Headline", "", headline_tables(scored), ""]
    parts += ["### Match accuracy by corpus and setting", "", compact_setting_table(scored, "acc_match"), ""]
    parts += ["### Evidence hit by corpus and setting", "", compact_setting_table(scored, "ev_hit"), ""]
    parts += ["### Abstention by corpus and setting", "", compact_setting_table(scored, "abstained"), ""]
    parts += ["## 2b. Per variant x corpus x setting", "", setting_tables(scored), ""]
    comp = res_dir / "comparisons.csv"
    if comp.is_file():
        parts += ["## 3. Paired bootstrap", "", comparison_tables(comp), ""]
    variants = sorted({r["approach"] for r in scored}, key=vorder)
    wanted = [("OURS", v) for v in variants if v != "OURS" and not v.startswith("OURS")] + \
             [(v, "OURS") for v in variants if v.startswith("OURS+")]
    parts += ["### Pooled setting groups (bootstrap over questions, 10000 resamples)", "", pooled_comparisons(scored, wanted), ""]
    counts, cases = wrong_number_analysis(records, scored, questions)
    same, n = plan_identity(records)
    parts += ["## 4. Wrong numbers in numeric answers", "",
              "OURS and OURS+facts_bare used the identical final plan on %d of %d (question, setting) pairs." % (same, n), "",
              wrong_number_tables(counts, cases), ""]
    if a.fix1:
        fdir = resolve_path(a.fix1)
        fsc = [r for r in read_jsonl(fdir / "results" / "scored.jsonl")]
        frec = [r for f in sorted(fdir.glob("runs_OURS*.jsonl")) for r in read_jsonl(f)]
        fc, fcases = wrong_number_analysis(frec, fsc, questions)
        parts += ["### Same diagnosis on %s (before the fix, 5 corpus-A questions)" % a.fix1, "",
                  wrong_number_tables(fc, fcases, focus=("OURS",)), ""]
    parts += ["## 5. B4 abstentions on answerable questions", "", b4_abstentions(records, scored, questions), ""]
    parts += ["## 6. Sanity", "", sanity_table(res_dir), ""]
    text = "\n".join(parts)
    out = resolve_path(a.out) if a.out else run_dir / "DEV_TABLES.md"
    out.write_text(text, encoding="utf-8")
    print("wrote %s (%d chars)" % (out, len(text)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
