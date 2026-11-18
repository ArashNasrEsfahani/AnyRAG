"""Sanity checks over a run directory and its aggregated results (no model calls).

    .venv/Scripts/python.exe -m ragbench.eval.sanity --runs runs/smoke [--results runs/smoke/results]
        [--questions a.jsonl --questions b.jsonl | --questions a.jsonl,b.jsonl]   (default: pilot.jsonl)

Checks (each prints PASS/FAIL/WARN; exit 1 on any FAIL):
  * query-time chat calls match each approach's design: B0-B3 exactly 1; B4 <= caps.tree_nav_max_calls;
    B5 <= caps.grep_agent_max_steps; OURS <= caps.ours_max_calls; AGENTIC <= caps.agentic_max_calls (default 8);
    AGENTIC2 <= caps.agentic2_max_calls (default 4);
    B3 exactly 1 query embed call
  * records whose question is in none of the --questions files (WARN: doc-scope and B1 checks skip them)
  * every record's call/token counts equal the sum of its ledger lines (phase=query, same question/setting)
  * evidence block within evidence.max_chars; evidence doc ids are in the setting's DOCUMENT language
    and inside the question's doc scope (cross-lingual settings really read the other-language documents)
  * latency and token fields populated (>0) for every successful record
  * evidence recall / hit values in [0, 1]; B0 has no evidence
  * B1 (gold evidence) accuracy >= B0 (no retrieval) accuracy per corpus
  * only B1 records can contain gold unit ids without having retrieved them (B1 evidence == gold)
Writes <results>/sanity.json.
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

from ..config import load_config
from ..contracts import load_questions
from ..util import atomic_write_json, read_jsonl
from .aggregate import load_run_records, resolve_path


def main(argv=None):
    ap = argparse.ArgumentParser(description="sanity checks for a ragbench run directory")
    ap.add_argument("--runs", required=True)
    ap.add_argument("--results", default=None, help="aggregate output dir (default <runs>/results)")
    ap.add_argument("--questions", action="append", default=None,
                    help="question JSONL (repeatable or comma list; default ragbench/data/questions/pilot.jsonl)")
    args = ap.parse_args(argv)
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass
    cfg = load_config()
    run_dir = resolve_path(args.runs, cfg)
    res_dir = resolve_path(args.results, cfg) if args.results else run_dir / "results"
    qfiles = [x.strip() for v in (args.questions or ["ragbench/data/questions/pilot.jsonl"]) for x in str(v).split(",")
              if x.strip()]
    qmap = {q.id: q for f in qfiles for q in load_questions(resolve_path(f, cfg))}
    records, _ = load_run_records([str(run_dir)], cfg)
    out = {"checks": [], "n_records": len(records)}
    n_fail = 0

    def check(name, ok, detail="", warn=False):
        nonlocal n_fail
        status = "PASS" if ok else ("WARN" if warn else "FAIL")
        n_fail += 0 if (ok or warn) else 1
        out["checks"].append({"check": name, "status": status, "detail": detail})
        print("%s %s%s" % (status, name, (" - " + str(detail)) if detail else ""))

    # provenance: split and Ollama version stamped by ragbench.run ------------------------------------
    no_split = [(r["approach"], r["question_id"], r["setting"]) for r in records if not r.get("split")]
    check("records.split_stamped", not no_split, "%d records without split (runs made before splits): %s" % (
        len(no_split), no_split[:3]), warn=True)
    splits = sorted({r.get("split") for r in records if r.get("split")})
    check("records.single_split", len(splits) <= 1, "splits %s in one run directory" % splits, warn=True)
    versions = sorted({(r.get("meta") or {}).get("ollama_version") for r in records} - {None, ""})
    check("records.single_ollama_version", len(versions) <= 1, "versions %s" % versions)
    no_version = sum(1 for r in records if not (r.get("meta") or {}).get("ollama_version"))
    check("records.ollama_version_stamped", no_version == 0, "%d records without meta.ollama_version" % no_version, warn=True)

    caps = {"B0": (1, 1), "B1": (1, 1), "B2": (1, 1), "B3": (1, 1), "B4": (1, cfg.caps["tree_nav_max_calls"]),
            "B5": (1, cfg.caps["grep_agent_max_steps"]), "OURS": (1, cfg.caps["ours_max_calls"]),
            "AGENTIC": (1, cfg.caps.get("agentic_max_calls", 8)),
            "AGENTIC2": (1, cfg.caps.get("agentic2_max_calls", 4)),
            "AGENTIC3": (1, cfg.caps.get("agentic3_max_calls", 4)),
            "AGENTIC4": (1, cfg.caps.get("agentic4_max_calls", 4)),
            "LEX": (1, cfg.caps.get("lex_max_calls", 3)), "LEX2": (1, cfg.caps.get("lex_max_calls", 3))}
    unknown_q = sorted({r["question_id"] for r in records} - set(qmap))
    check("records.questions_known", not unknown_q, "%d question id(s) in no --questions file %s: %s" % (
        len(unknown_q), qfiles, unknown_q[:5]), warn=True)
    ok_recs = [r for r in records if not r.get("error")]
    check("records.no_errors", len(ok_recs) == len(records),
          "%d records, %d errored: %s" % (len(records), len(records) - len(ok_recs),
                                         [(r["approach"], r["question_id"], r["setting"], (r["error"] or "")[:80])
                                          for r in records if r.get("error")][:5]))

    # call caps ------------------------------------------------------------------------------
    by_var = defaultdict(list)
    for r in ok_recs:
        by_var[r["approach"]].append(r)
    for v, rs in sorted(by_var.items()):
        lo, hi = caps.get(rs[0]["approach_base"], (0, 10 ** 9))
        calls = [r["llm_calls"] for r in rs]
        bad = [(r["question_id"], r["setting"], r["llm_calls"]) for r in rs if not lo <= r["llm_calls"] <= hi]
        check("calls.%s within [%d, %d]" % (v, lo, hi), not bad,
              "min %d max %d mean %.2f; violations %s" % (min(calls), max(calls), sum(calls) / len(calls), bad[:5]))
        if rs[0]["approach_base"] == "B3":
            be = [(r["question_id"], r["setting"], r["embed_calls"]) for r in rs if r["embed_calls"] != 1]
            check("calls.%s one query embed" % v, not be, be[:5])

    # ledger consistency ------------------------------------------------------------------------
    ledgers = defaultdict(lambda: {"chat": 0, "embed": 0, "prompt_tokens": 0, "completion_tokens": 0})
    for f in sorted(run_dir.glob("ledger_*.jsonl")):
        for l in read_jsonl(f):
            if l.get("phase") != "query":
                continue
            k = (l.get("run_id"), l.get("approach"), l.get("question_id"), l.get("setting"))
            e = ledgers[k]
            e[l.get("kind")] = e.get(l.get("kind"), 0) + 1
            if l.get("kind") == "chat":
                e["prompt_tokens"] += l.get("prompt_tokens") or 0
                e["completion_tokens"] += l.get("completion_tokens") or 0
    mism = []
    for r in ok_recs:
        e = ledgers.get((r["run_id"], r["approach"], r["question_id"], r["setting"]))
        if e is None:
            mism.append((r["approach"], r["question_id"], r["setting"], "no ledger lines"))
            continue
        if (e["chat"], e["embed"], e["prompt_tokens"], e["completion_tokens"]) != (
                r["llm_calls"], r["embed_calls"], r["prompt_tokens"], r["completion_tokens"]):
            mism.append((r["approach"], r["question_id"], r["setting"], dict(e), r["llm_calls"], r["prompt_tokens"]))
    check("ledger.records_match_ledger_lines", not mism, "%d mismatches %s" % (len(mism), mism[:3]))

    # evidence: budget, language, scope ---------------------------------------------------------------
    budget = cfg.evidence["max_chars"]
    over = [(r["approach"], r["question_id"], r["setting"], r["evidence_chars"]) for r in ok_recs if r["evidence_chars"] > budget]
    check("evidence.within_budget_%d" % budget, not over, over[:5])
    wrong_lang, out_scope, set_lang = [], [], []
    n_cross = n_cross_ev = 0
    for r in ok_recs:
        q = qmap.get(r["question_id"])
        if not r["corpus_set"].endswith("_" + r["d_lang"]) and not (r["corpus"] == "C" and r["d_lang"] == "fa"):
            set_lang.append((r["approach"], r["question_id"], r["setting"], r["corpus_set"]))
        scope = q.scope(r["d_lang"]) if q else "corpus"
        docs = {it["doc_id"] for it in r["evidence_items"]} | {u.split("::")[0] for u in r["evidence_units"]}
        if r["q_lang"] != r["d_lang"]:
            n_cross += 1
            n_cross_ev += bool(docs)
        for d in docs:
            if not d.endswith("." + r["d_lang"]):
                wrong_lang.append((r["approach"], r["question_id"], r["setting"], d))
            if scope != "corpus" and d not in scope:
                out_scope.append((r["approach"], r["question_id"], r["setting"], d))
    check("evidence.corpus_set_is_document_language", not set_lang, set_lang[:5])
    check("evidence.docs_in_document_language", not wrong_lang,
          "%d cross-lingual records, %d of them with evidence; violations %s" % (n_cross, n_cross_ev, wrong_lang[:5]))
    check("evidence.docs_inside_doc_scope", not out_scope, out_scope[:5])
    b0_ev = [(r["question_id"], r["setting"]) for r in ok_recs if r["approach_base"] == "B0" and r["evidence_units"]]
    check("evidence.B0_has_none", not b0_ev, b0_ev[:5])
    b1_bad = []
    for r in ok_recs:
        if r["approach_base"] != "B1":
            continue
        q = qmap.get(r["question_id"])
        if q and set(r["evidence_units"]) - set(q.gold(r["d_lang"])):
            b1_bad.append((r["question_id"], r["setting"]))
    check("evidence.B1_evidence_is_gold_only", not b1_bad, b1_bad[:5])

    # populated fields -----------------------------------------------------------------------------------
    empty = [(r["approach"], r["question_id"], r["setting"]) for r in ok_recs
             if not (r["latency_ms"] > 0 and r["prompt_tokens"] > 0 and r["completion_tokens"] > 0 and r["llm_wall_ms"] > 0)]
    check("fields.latency_tokens_populated", not empty, empty[:5])

    # scored metrics ---------------------------------------------------------------------------------------
    scored_path = res_dir / "scored.jsonl"
    if scored_path.is_file():
        scored = read_jsonl(scored_path)
        bad = [(s["approach"], s["question_id"], s["setting"], k, s.get(k)) for s in scored
               for k in ("ev_hit", "ev_recall", "ret_hit", "ret_recall") if s.get(k) is not None and not 0 <= s[k] <= 1]
        check("metrics.recall_in_0_1", not bad, bad[:5])
        acc = defaultdict(lambda: defaultdict(list))
        for s in scored:            # match accuracy: available for every record (judge verdicts may be partial)
            key = s["corpus"] if not s.get("split") else "%s/%s" % (s["corpus"], s["split"])
            acc[key][s["approach"]].append(s["acc_match"])
        for corpus, by in sorted(acc.items()):
            if "B0" in by and "B1" in by:
                a0 = sum(by["B0"]) / len(by["B0"])
                a1 = sum(by["B1"]) / len(by["B1"])
                check("accuracy.B1_ge_B0 corpus %s (match)" % corpus, a1 >= a0, "B1 %.2f vs B0 %.2f" % (a1, a0))
        out["acc_match"] = {c: {v: round(sum(x) / len(x), 3) for v, x in by.items()} for c, by in acc.items()}
    else:
        check("metrics.scored_present", False, "no %s (run ragbench.eval.aggregate first)" % scored_path, warn=True)
    out["n_failed"] = n_fail
    res_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(res_dir / "sanity.json", out, indent=1)
    print("\n%d checks, %d failed -> %s" % (len(out["checks"]), n_fail, res_dir / "sanity.json"))
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
