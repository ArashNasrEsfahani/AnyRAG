"""AGENTIC tiny live check (real local model; sequential; a handful of calls per question).

    .venv/Scripts/python.exe -m ragbench.approaches.agentic.live_check
    .venv/Scripts/python.exe -m ragbench.approaches.agentic.live_check --only snapp --synth

(a) snapp       "How many drivers does Snapp have and how has that changed over the years?" (fa-fa) over corpus
                C (load_set("C"), every document; doc scope "corpus", so the planner must choose the Snapp
                reports from the metadata listing)
(b) [removed in the public release: a second check on a corpus-A development question]
--synth also answers (a) with the synth_answer prompt: the plan/verify calls are replayed from the shared call
cache, so it costs one more call.

Writes work/runs/agentic_smoke/: runs_<variant>.jsonl (RunRecords), ledger_AGENTIC.jsonl, agentic_calls.sqlite
(the call cache: a rerun replays finished calls), summary_<id>.json (per-stage table). Model calls queue behind
any other GPU job through the shared lock.
"""
import argparse
import json
import sys
import time

from ...config import load_config
from ...contracts import Question, RunLog, load_questions
from ...corpus.registry import load_set
from ...llm import Ledger, OllamaClient
from ...util import atomic_write_json
from .approach import AgenticPipeline

SNAPP_Q = Question(
    id="AGENTIC-snapp-drivers", corpus="C",
    question_fa="اسنپ چند راننده دارد و تعداد رانندگان آن در طول سال‌ها چگونه تغییر کرده است؟",
    question_en="How many drivers does Snapp have and how has that changed over the years?",
    answer_fa="", answer_en="", answer_type="short", gold_units={"fa": []}, doc_scope="corpus", type="cross-year",
    meta={"author": "agentic-live-check"})


def stage_table(rec):
    tm = (rec.trace or {}).get("timing") or {}
    lines = ["stage      wall_s  calls  ptok   ctok   llm_s   rg_ms  cand/kept/sel     acc  model_rej  guard_rej  unparsed"]
    for s in tm.get("stages") or []:
        cand = ""
        if s["stage"] in ("search", "gap_fill") and ("hit_lines" in s):
            cand = "%s/%s/%s" % (s.get("windows", ""), s.get("kept", ""), s.get("selected", ""))
        elif s["stage"] == "verify":
            cand = "verified %s" % s.get("verified", 0)
        lines.append("%-10s %6.1f  %5d  %5d  %5d  %6.1f  %6.1f  %-16s  %4s  %9s  %9s  %8s" % (
            s["stage"], s["wall_ms"] / 1000.0, s.get("llm_calls", 0), s.get("prompt_tokens", 0),
            s.get("completion_tokens", 0), (s.get("llm_wall_ms") or 0) / 1000.0, s.get("rg_ms", 0.0) or 0.0, cand,
            s.get("accepted", ""), s.get("model_rejected", ""), s.get("guard_rejected", ""), s.get("unparsed", "")))
    t = tm.get("totals") or {}
    lines.append("total      %6.1f  %5d  %5d  %5d  %6.1f  %6.1f  (ledger reconciles: %s; latency %.1fs; lock wait %.1fs)" % (
        t.get("wall_ms", 0) / 1000.0, t.get("llm_calls", 0), t.get("prompt_tokens", 0), t.get("completion_tokens", 0),
        (t.get("llm_wall_ms") or 0) / 1000.0, t.get("rg_ms", 0) or 0, tm.get("reconciles"), rec.latency_ms / 1000.0,
        (rec.lock_wait_ms or 0) / 1000.0))
    return "\n".join(lines)


def summarize(rec):
    tr = rec.trace or {}
    st = tr.get("structured") or {}
    out = {"question_id": rec.question_id, "variant": rec.approach, "setting": rec.setting, "error": rec.error,
           "answer": rec.answer_text, "llm_calls": rec.llm_calls, "prompt_tokens": rec.prompt_tokens,
           "completion_tokens": rec.completion_tokens, "latency_ms": rec.latency_ms, "lock_wait_ms": rec.lock_wait_ms,
           "evidence_chars": rec.evidence_chars, "stats": tr.get("stats"), "guard_reasons": tr.get("guard_reasons"),
           "plan": {k: (tr.get("stages") or {}).get("plan", {}).get(k) for k in
                    ("entity", "metric", "kind", "unit", "terms", "period", "companies", "fixes")},
           "doc_selection": ((tr.get("stages") or {}).get("plan") or {}).get("doc_selection"),
           "timing": tr.get("timing"), "table": ((tr.get("stages") or {}).get("answer") or {}).get("table"),
           "series": st.get("series"), "comparable": st.get("comparable"), "definition_classes": st.get("definition_classes"),
           "periods_missing": st.get("periods_missing"), "conflicts": st.get("conflicts"),
           "restatements": st.get("restatements"), "notes": st.get("notes")}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="snapp")
    ap.add_argument("--synth", action="store_true")
    ap.add_argument("--out", default="agentic_smoke")
    args = ap.parse_args(argv)
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass
    cfg = load_config()
    out = cfg.path("runs_root") / args.out
    out.mkdir(parents=True, exist_ok=True)
    ledger = Ledger(path=str(out / "ledger_AGENTIC.jsonl"))
    client = OllamaClient(cfg, ledger)
    print("ollama:", {k: client.health().get(k) for k in ("reachable", "version", "chat_present", "chat_digest_match")})
    cache = str(out / "agentic_calls.sqlite")
    which = [w.strip() for w in args.only.split(",") if w.strip()]
    todo = []
    if "snapp" in which:
        todo.append(("snapp", SNAPP_Q, "fa-fa", "C", {}))
        if args.synth:
            todo.append(("snapp", SNAPP_Q, "fa-fa", "C", {"synth_answer": True}))
    approaches = {}
    t_all = time.perf_counter()
    results = []
    for name, q, setting, set_name, flags in todo:
        key = json.dumps(flags, sort_keys=True)
        if key not in approaches:
            approaches[key] = AgenticPipeline(cfg=cfg, client=client, flags=flags, options={"call_cache_path": cache})
        a = approaches[key]
        if set_name not in a._indexes:
            t0 = time.perf_counter()
            cs = load_set(set_name, cfg)
            print("index %s over %s ..." % (a.variant, cs), flush=True)
            h = a.build_index(cs, ledger)
            print("  ready in %.1fs (%s files, store %.0f ms, warm %.0f ms)" % (
                time.perf_counter() - t0, h.payload.get("n_files"), h.payload.get("store_ms") or 0,
                h.payload.get("warm_ms") or 0), flush=True)
        print("\n=== %s %s %s" % (a.variant, q.id, setting), flush=True)
        rec = a.answer(q, setting)
        rec.meta = {"live_check": True, "questions_file": "pilot.jsonl" if q.id.startswith("A-") else None}
        RunLog(out / ("runs_%s.jsonl" % a.variant.replace("+", "_"))).append(rec)
        s = summarize(rec)
        atomic_write_json(out / ("summary_%s_%s.json" % (q.id, a.variant.replace("+", "_"))), s, indent=1)
        print(stage_table(rec))
        if rec.error:
            print("ERROR:", rec.error)
            print((rec.trace or {}).get("traceback", "")[-1500:])
        print("plan:", json.dumps(s["plan"], ensure_ascii=False))
        print("docs:", (s["doc_selection"] or {}).get("chosen"))
        print("guard:", s["guard_reasons"])
        if s["table"]:
            print(s["table"])
        print("answer:", rec.answer_text)
        results.append(s)
    print("\nlive check wall %.1fs" % (time.perf_counter() - t_all))
    return 0 if all(not r["error"] for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
