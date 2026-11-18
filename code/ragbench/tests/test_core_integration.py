"""Offline tests for the CORE PLAN integration (2026-09-16): registry and flags (OURS index levels, AGENTIC), the shared
timeline answer prompt and its budget, RunRecord.structured, the per-corpus --final guard, the dry-run index plan, the
index cost copy, and the AGENTIC call cap in sanity/aggregate. No model calls.

    .venv/Scripts/python.exe -m ragbench.tests.test_core_integration
"""
import argparse
import io
import json
import re
import shutil
import sys
import tempfile
from contextlib import redirect_stdout
from pathlib import Path

from ..approaches import AGENTIC_FLAGS, APPROACHES, OURS_FLAGS, get_approach
from ..config import load_config
from ..contracts import (ABSTAIN_RULE_EVIDENCE, ABSTAIN_RULE_NONE, ANSWER_PROMPT_VERSIONS, ANSWER_SYSTEM,
                         ANSWER_TIMELINE_SYSTEM, AnswerResult, Approach, IndexHandle, LANG_NAMES, Query, Question,
                         RECORD_HASH_SECTIONS, RunRecord, SOURCE_RULE_EVIDENCE, SOURCE_RULE_NONE, answer_format,
                         answer_num_predict, answer_prompt_version, build_answer_messages, load_questions)
from ..llm import Ledger
from .. import run as RUN

RESULTS = []
PKG = Path(__file__).resolve().parents[1]


def check(name, cond, detail=""):
    RESULTS.append((name, bool(cond)))
    print(("PASS " if cond else "FAIL ") + name + (" - " + str(detail) if detail != "" else ""), flush=True)
    return cond


def raises(fn, exc=SystemExit):
    try:
        fn()
    except exc:
        return True
    except Exception:  # noqa: BLE001
        return False
    return False


# --------------------------------------------------------------------------
# registry and flags
# --------------------------------------------------------------------------

def test_registry_and_flags():
    from ..approaches.agentic.defaults import FLAGS as AG_FLAGS
    check("registry.agentic", APPROACHES.get("AGENTIC") == "ragbench.approaches.agentic:AgenticPipeline")
    check("registry.ours_index_level_flag", "index_level" in OURS_FLAGS)
    check("registry.agentic_flags_match_package", tuple(AGENTIC_FLAGS) == tuple(AG_FLAGS), (AGENTIC_FLAGS, AG_FLAGS))
    check("flags.index_level_L1", RUN.parse_flags(["index_level=L1"], "OURS") == {"index_level": "L1"})
    check("flags.index_level_lowercase", RUN.parse_flags(["index_level=l0"], "OURS") == {"index_level": "L0"})
    check("flags.index_level_L3_is_plain_ours", RUN.parse_flags(["index_level=L3"], "OURS") == {})
    check("flags.index_level_invalid", raises(lambda: RUN.parse_flags(["index_level=L9"], "OURS")))
    check("flags.index_level_combines", RUN.parse_flags(["monolingual_patterns,index_level=L1"], "OURS") ==
          {"monolingual_patterns": True, "index_level": "L1"})
    check("flags.agentic", RUN.parse_flags(["no_guard", "synth_answer"], "AGENTIC") == {"no_guard": True, "synth_answer": True})
    check("flags.agentic_rejects_ours_flag", raises(lambda: RUN.parse_flags(["index_level=L1"], "AGENTIC")))
    check("flags.b3_rejects_ablation", raises(lambda: RUN.parse_flags(["no_guard"], "B3")))
    check("flags.b5_agent_answer", RUN.parse_flags(["agent_answer"], "B5") == {"agent_answer": True})
    cfg = load_config()
    names = {lv: get_approach("OURS", cfg=cfg, flags=RUN.parse_flags(["index_level=%s" % lv], "OURS")).variant
             for lv in ("L0", "L1", "L3")}
    check("variant.names", names == {"L0": "OURS+index_level=L0", "L1": "OURS+index_level=L1", "L3": "OURS"}, names)
    check("variant.agentic", get_approach("AGENTIC", cfg=cfg).variant == "AGENTIC")
    from ..approaches import AGENTIC2_FLAGS
    from ..approaches.agentic2.defaults import FLAGS as AG2_FLAGS
    check("registry.agentic2", APPROACHES.get("AGENTIC2") == "ragbench.approaches.agentic2:TreeFactVerify")
    check("registry.agentic2_flags_match_package", tuple(AGENTIC2_FLAGS) == tuple(AG2_FLAGS), (AGENTIC2_FLAGS, AG2_FLAGS))
    check("flags.agentic2", RUN.parse_flags(["no_verify", "candidates=32"], "AGENTIC2") == {"no_verify": True, "candidates": 32})
    check("flags.agentic2_rejects_bad_value", raises(lambda: RUN.parse_flags(["candidates=many"], "AGENTIC2")))
    check("flags.agentic2_rejects_v1_flag", raises(lambda: RUN.parse_flags(["no_guard"], "AGENTIC2")))
    check("variant.agentic2", get_approach("AGENTIC2", cfg=cfg).variant == "AGENTIC2" and get_approach(
        "AGENTIC2", cfg=cfg, flags={"candidates": 32}).variant == "AGENTIC2+candidates=32")
    check("caps.agentic2", cfg.caps.get("agentic2_max_calls") == 4 and float(cfg.caps.get("agentic2_max_seconds")) == 60.0,
          cfg.caps)
    args = RUN.build_parser().parse_args(["--approach", "ours", "--index-level", "L1", "--questions", "x.jsonl",
                                          "--corpus", "A", "--out", "runs/x"])
    check("cli.index_level_option", args.index_level == "L1")
    check("cli.index_level_choices", raises(lambda: RUN.build_parser().parse_args(
        ["--approach", "ours", "--index-level", "L7", "--questions", "x", "--corpus", "A", "--out", "y"])))


# --------------------------------------------------------------------------
# the shared answer prompt per answer format
# --------------------------------------------------------------------------

class _StubClient:
    def __init__(self):
        self.ledger = Ledger(path=None)
        self.calls = []

    def chat(self, messages, purpose=None, num_predict=None, **kw):
        self.calls.append({"messages": messages, "purpose": purpose, "num_predict": num_predict})
        return type("R", (), {"text": " 1398: 5 people "})()


class _StubApproach(Approach):
    name = "STUB"

    def _build_index(self, corpus_set, index_dir):
        return {}

    def _answer(self, query, index):
        return AnswerResult(answer_text=self.generate_answer(query, None), structured={"series": [{"period": "1398"}]})


def _q(qtype="lookup", answer_type="short"):
    return Question(id="t-%s" % qtype, corpus="C", question_fa="چند نفر؟", question_en="How many people?", answer_fa="",
                    answer_en="", answer_type=answer_type, gold_units={"fa": ["C.x.fa::p1_b1"]},
                    doc_scope=["C.x_annual-report_1400.fa"], type=qtype)


def test_answer_formats():
    cfg = load_config()
    abstain = cfg.evidence["abstain_token"]
    ev = type("EV", (), {"text": "[1] X\nbody"})()
    qd = Query("q1", "How many?", "en", "en", "en-en", "A", "A_en", ["A.x.en"])
    old_system = ("You answer questions about company, market and sector reports.\n" + SOURCE_RULE_EVIDENCE + "\n"
                  + ABSTAIN_RULE_EVIDENCE.format(abstain=abstain) + "\n" + "Answer in English. Reply with the answer only, in one "
                  "or two short sentences. Copy numbers exactly as they are written, with their unit and period. Do not "
                  "explain your reasoning.")
    msgs = build_answer_messages(qd, ev, cfg)
    check("answer.default_prompt_unchanged", msgs[0]["content"] == old_system and qd.answer_format == "default"
          and answer_prompt_version(qd) == "answer-v2", msgs[0]["content"][-160:])
    check("answer.default_budget_unchanged", answer_num_predict(qd, cfg) == int(cfg.generation["num_predict_answer"]) == 256)
    qt = Query("q2", "چند نفر از 1397 تا 1400؟", "fa", "fa", "fa-fa", "C", "C", ["C.x.fa"], answer_format="timeline")
    mt = build_answer_messages(qt, ev, cfg)
    check("answer.timeline_prompt", mt[0]["content"].startswith("You answer questions about company") and
          "one line per period" in mt[0]["content"] and "<period>: %s" % abstain in mt[0]["content"]
          and LANG_NAMES["fa"] in mt[0]["content"] and SOURCE_RULE_EVIDENCE in mt[0]["content"]
          and mt[1] == msgs[1] | {"content": "Evidence:\n[1] X\nbody\n\nQuestion: چند نفر از 1397 تا 1400؟"},
          mt[0]["content"])
    mt0 = build_answer_messages(qt, None, cfg)
    check("answer.timeline_prompt_no_evidence_B0", SOURCE_RULE_NONE in mt0[0]["content"]
          and ABSTAIN_RULE_NONE.format(abstain=abstain) in mt0[0]["content"] and mt0[1]["content"].startswith("Question:"))
    check("answer.timeline_budget", answer_num_predict(qt, cfg) == int(cfg.get("answer.timeline_num_predict")) > 256
          and answer_prompt_version(qt) == "answer-timeline-v1", answer_num_predict(qt, cfg))
    check("answer.versions_declared_in_final", (cfg.get("final") or {}).get("answer_prompts") == ANSWER_PROMPT_VERSIONS,
          ((cfg.get("final") or {}).get("answer_prompts"), ANSWER_PROMPT_VERSIONS))
    no_answer_cfg = cfg.copy_with({"answer": None})
    check("answer.missing_timeline_budget_is_an_error", raises(lambda: answer_num_predict(qt, no_answer_cfg), KeyError)
          and answer_num_predict(qd, no_answer_cfg) == 256)
    check("answer.unknown_format_is_an_error", raises(lambda: answer_num_predict("haiku", cfg), ValueError))
    check("answer.format_of_question", answer_format(_q("timeline", "series")) == "timeline" and answer_format(_q()) == "default"
          and answer_format(_q("cross-year")) == "default")
    check("answer.timeline_template_fields", set(re.findall(r"{(\w+)}", ANSWER_TIMELINE_SYSTEM)) ==
          {"source_rule", "abstain_rule", "answer_lang", "abstain"} and set(re.findall(r"{(\w+)}", ANSWER_SYSTEM)) ==
          {"source_rule", "abstain_rule", "answer_lang"})
    check("record.hash_sections", RECORD_HASH_SECTIONS == ("models", "generation", "evidence", "chunking", "caps", "answer"))
    # Approach.answer: make_query, generate_answer budget, meta stamps, structured copy
    for qtype, atype, fmt in (("lookup", "short", "default"), ("timeline", "series", "timeline")):
        client = _StubClient()
        ap = _StubApproach(cfg=cfg, client=client)
        ap._indexes["C"] = IndexHandle("STUB", "C", Path("."), "fp", None, {}, {})
        rec = ap.answer(_q(qtype, atype), "fa-fa")
        call = client.calls[-1] if client.calls else {}
        check("approach.%s_question" % fmt, rec.error is None and call.get("num_predict") == answer_num_predict(fmt, cfg)
              and call.get("purpose") == "answer"
              and (("one line per period" in call["messages"][0]["content"]) == (fmt == "timeline"))
              and rec.meta == {"answer_format": fmt, "answer_prompt": ANSWER_PROMPT_VERSIONS[fmt],
                               "answer_num_predict": answer_num_predict(fmt, cfg)}
              and rec.structured == {"series": [{"period": "1398"}]} and "structured" in rec.to_dict(), (rec.error, rec.meta))
    check("record.structured_defaults", AnswerResult("x").structured is None and
          RunRecord.__dataclass_fields__["structured"].default is None)


def test_answer_call_sources():
    """Every approach makes its final answer through the shared helpers: no approach module reads the answer budget
    from config directly, every module that builds answer messages also takes answer_num_predict, and no approach
    reads Query.answer_format (it selects the answer prompt only)."""
    bad_budget, bad_pair, reads_fmt = [], [], []
    for f in sorted((PKG / "approaches").rglob("*.py")):
        if "test" in f.stem or f.stem in ("smoke_test", "live_check"):
            continue
        src = f.read_text(encoding="utf-8")
        rel = f.relative_to(PKG).as_posix()
        if re.search(r"""\[\s*["']num_predict_answer["']\s*\]|get\(\s*["']num_predict_answer""", src):
            bad_budget.append(rel)
        if re.search(r"=\s*build_answer_messages\(", src) and "answer_num_predict(" not in src:
            bad_pair.append(rel)
        if "answer_format" in src:
            reads_fmt.append(rel)
    check("sources.no_direct_answer_budget", not bad_budget, bad_budget)
    check("sources.answer_messages_with_shared_budget", not bad_pair, bad_pair)
    check("sources.approaches_do_not_read_answer_format", not reads_fmt, reads_fmt)
    gen = [f.relative_to(PKG).as_posix() for f in sorted((PKG / "approaches").rglob("*.py"))
           if "test" not in f.stem and "generate_answer(" in f.read_text(encoding="utf-8")]
    check("sources.generate_answer_users", {"approaches/baselines.py", "approaches/tree_nav.py", "approaches/grep_agent.py",
                                            "approaches/ours/approach.py"} <= set(gen), gen)


# --------------------------------------------------------------------------
# --final guard
# --------------------------------------------------------------------------

def _args(extra):
    return RUN.build_parser().parse_args(extra)


def _final(qfile, corpus, settings, variant, cfg=None, extra=()):
    cfg = cfg or load_config()
    a = _args(["--final", "--docs-from-questions", "--questions", "ragbench/data/questions/%s" % qfile, "--corpus", corpus,
               "--settings", settings, "--out", "runs/x"] + list(extra))
    from ..contracts import Setting
    return RUN.final_problems(a, cfg, variant, [Setting.parse(s) for s in settings.split(",")])


def test_final_guard():
    cfg = load_config()
    fin = cfg.get("final") or {}
    core = ["B0", "B1", "B2", "B3", "B4", "B5", "OURS+index_level=L0", "OURS+index_level=L1", "OURS", "AGENTIC",
            "OURS+monolingual_patterns", "AGENTIC2",     # AGENTIC2 (v2) declared 2026-09-16 for stage v2test
            "AGENTIC3",                                 # AGENTIC3 (v3) declared 2026-09-20 for stage v3test
            "AGENTIC4",                                  # AGENTIC4 (v4) declared 2026-09-21 for stage v4test
            "LEX2", "LEX2+llm_regex", "LEX2+free_keywords"]  # declared 2026-09-23 for the fresh holdout (PREREG.md)
    check("final.variants_exactly_core", sorted(fin.get("variants") or []) == sorted(core), fin.get("variants"))
    check("final.agentic2_ablation_refused", any("final.variants" in x for x in
                                                   _final("test_core.jsonl", "A", "fa-fa", "AGENTIC2+no_verify")))
    check("final.settings_per_corpus", fin.get("settings") == {"A": ["fa-fa", "en-en", "fa-en", "en-fa"], "B": ["fa-fa", "en-en"],
                                                               "C": ["fa-fa"], "D": ["en-en"]}, fin.get("settings"))
    ok = {v: _final("test_core.jsonl", "A", "fa-fa,en-en,fa-en,en-fa", v) for v in core}
    check("final.core_A_all_settings_all_variants", not any(ok.values()), {k: v for k, v in ok.items() if v})
    check("final.core_B_monolingual", not _final("test_core.jsonl", "B", "fa-fa,en-en", "B4"))
    p = _final("test_core.jsonl", "B", "fa-en", "B4")
    check("final.core_B_crosslingual_refused", any("final.settings.B" in x for x in p), p)
    p = _final("test_core.jsonl", "C", "en-fa", "OURS")
    check("final.core_C_en_fa_refused", any("final.settings.C" in x for x in p), p)
    check("final.single_setting_subset_ok", not _final("test_core.jsonl", "A", "fa-en", "AGENTIC"))
    p = _final("test_core.jsonl", "A", "fa-fa", "OURS+no_facts")
    check("final.undeclared_variant_refused", any("final.variants" in x for x in p), p)
    p = _final("test_core.jsonl", "A", "fa-fa", "OURS+index_level=L2")
    check("final.L2_refused", any("final.variants" in x for x in p), p)
    p = _final("test.jsonl", "A", "fa-fa", "B3")
    check("final.undeclared_question_file_refused", any("final.question_files" in x for x in p), p)
    p = _final("test_timeline.jsonl", "C", "fa-fa", "OURS")
    p2 = _final("test_timeline.jsonl", "C", "fa-fa", "OURS+monolingual_patterns")
    check("final.timeline_excludes_L3", any("exclude_variants" in x for x in p) and any("exclude_variants" in x for x in p2), (p, p2))
    ok = {v: _final("test_timeline.jsonl", "C", "fa-fa", v) for v in core if v not in ("OURS", "OURS+monolingual_patterns")}
    check("final.timeline_other_variants_ok", not any(ok.values()), {k: v for k, v in ok.items() if v})
    p = _final("test_core.jsonl", "A", "fa-fa", "B3", extra=["--limit", "3"])
    check("final.subset_still_refused", any("--limit" in x for x in p), p)
    p = _final("test_core.jsonl", "A", "fa-fa", "B3", cfg=cfg.copy_with({"final": {"answer_prompts": {"default": "answer-v2",
                                                                                                    "timeline": "answer-timeline-v0"}}}))
    check("final.answer_prompt_version_mismatch_refused", any("final.answer_prompts" in x for x in p), p)
    p = _final("test_core.jsonl", "B", "en-en", "B3", cfg=cfg.copy_with({"final": {"settings": ["fa-fa"]}}))
    check("final.settings_list_form", any("final.settings" in x for x in p), p)
    p = _final("test_core.jsonl", "B", "fa-en", "B3", cfg=cfg.copy_with({"final": {"settings": None}}))
    check("final.settings_absent_falls_back_to_global", not [x for x in p if "settings" in x], p)
    p = _final("test_core.jsonl", "A", "fa-fa", "B3", cfg=cfg.copy_with({"final": {"settings": {"A": None}}}))
    check("final.corpus_without_entry_refused", any("final.settings.A" in x for x in p), p)


# --------------------------------------------------------------------------
# dry-run plan, index cost copy
# --------------------------------------------------------------------------

def _dry(argv):
    buf = []
    rc = RUN.main(argv + ["--dry-run"], log=buf.append)
    js = [json.loads(l[len("DRYRUN-JSON "):]) for l in buf if l.startswith("DRYRUN-JSON ")]
    return rc, (js[0] if js else None), buf


def test_dry_run(tmp):
    out = str(tmp / "dry")
    rc, js, buf = _dry(["--final", "--docs-from-questions", "--approach", "b2", "--questions",
                        "ragbench/data/questions/test_core.jsonl", "--corpus", "B", "--settings", "fa-fa,en-en", "--out", out])
    check("dry.test_core_B", rc == 0 and js and js["planned"] == 20 and js["variant"] == "B2" and js["split"] == "test"
          and js["by_setting"] == {"fa-fa": {"planned": 10, "todo": 10}, "en-en": {"planned": 10, "todo": 10}}
          and sorted(s["set"] for s in js["sets"]) == ["B_en", "B_fa"]
          and all(s["doc_ids"] == ["B.digikala-1401.%s" % s["set"][-2:]] for s in js["sets"]), (rc, js and js["planned"], buf[-3:]))
    rc, js, buf = _dry(["--final", "--docs-from-questions", "--approach", "b2", "--questions",
                        "ragbench/data/questions/test_core.jsonl", "--corpus", "B", "--settings", "fa-en", "--out", out])
    check("dry.final_guard_applies_to_dry_run", rc == 2 and js is None, buf[-2:])
    rc, js, buf = _dry(["--approach", "ours", "--index-level", "L1", "--docs-from-questions", "--questions",
                        "ragbench/data/questions/dev_core.jsonl", "--corpus", "A", "--settings", "fa-fa,en-en", "--out", out])
    check("dry.index_level_shorthand", rc == 0 and js and js["variant"] == "OURS+index_level=L1" and js["planned"] == 10
          and all("OURS_index_level_L1" in s["index_dir"] for s in js["sets"]), (rc, js and js["variant"]))
    tl = load_config().path("questions_root") / "pilot_timeline.jsonl"
    rc, js, buf = _dry(["--approach", "agentic", "--docs-from-questions", "--questions", str(tl), "--corpus", "C",
                        "--settings", "fa-fa", "--out", out])
    check("dry.timeline_answer_format_counts", rc == 0 and js and js["by_format"] == {"timeline": {"planned": 8, "todo": 8}},
          js and js["by_format"])
    d = tmp / "idx"
    d.mkdir()
    check("index_cost.missing", RUN._index_cost(d) is None)
    (d / "index.json").write_text(json.dumps({"index_cost": {"llm_calls": 3}}), encoding="utf-8")
    check("index_cost.read", RUN._index_cost(d) == {"llm_calls": 3})
    (d / "index.json").write_text("[1, 2]", encoding="utf-8")
    check("index_cost.non_dict", RUN._index_cost(d) is None)


# --------------------------------------------------------------------------
# sanity / aggregate call caps
# --------------------------------------------------------------------------

def test_agentic_caps(tmp):
    from ..eval import aggregate as AGG
    from ..eval import sanity as SAN
    cfg = load_config()
    run_dir = tmp / "sanity_run"
    run_dir.mkdir()
    base = {"approach": "AGENTIC", "approach_base": "AGENTIC", "flags": {}, "setting": "fa-fa", "q_lang": "fa", "d_lang": "fa",
            "corpus": "C", "corpus_set": "C", "answer_text": "x", "abstained": False, "retrieved_units": [], "evidence_units": [],
            "evidence_items": [], "evidence_chars": 10, "embed_calls": 0, "prompt_tokens": 10, "completion_tokens": 5,
            "embed_tokens": 0, "llm_wall_ms": 5.0, "load_ms": 0.0, "latency_ms": 9.0, "failed_calls": 0, "error": None,
            "trace": None, "models": {}, "config_hash": "h", "index_fingerprint": "fp", "ts": "t", "run_id": "r1",
            "split": "dev", "meta": {"ollama_version": "0.32.15"}}
    q = load_questions(cfg.path("questions_root") / "pilot_timeline.jsonl")[:2]
    recs = [dict(base, question_id=q[0].id, llm_calls=8), dict(base, question_id=q[1].id, llm_calls=9)]
    with open(run_dir / "runs_AGENTIC.jsonl", "w", encoding="utf-8") as fh:
        for r in recs:
            fh.write(json.dumps(r) + "\n")
    with open(run_dir / "ledger_AGENTIC.jsonl", "w", encoding="utf-8") as fh:
        for r in recs:
            for i in range(r["llm_calls"]):
                fh.write(json.dumps({"phase": "query", "kind": "chat", "run_id": "r1", "approach": "AGENTIC",
                                     "question_id": r["question_id"], "setting": "fa-fa",
                                     "prompt_tokens": 10 if i == 0 else 0, "completion_tokens": 5 if i == 0 else 0}) + "\n")
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = SAN.main(["--runs", str(run_dir), "--results", str(tmp / "sanity_res"), "--questions",
                       "ragbench/data/questions/pilot.jsonl,ragbench/data/questions/pilot_timeline.jsonl"])
    txt = buf.getvalue()
    check("sanity.agentic_cap", rc == 1 and "FAIL calls.AGENTIC within [1, 8]" in txt and "PASS records.questions_known" in txt,
          [l for l in txt.splitlines() if "AGENTIC" in l or "questions_known" in l])
    w = AGG.fairness_warnings(recs, cfg)
    check("aggregate.agentic_cap_warning", any("AGENTIC" in x and "cap (8)" in x for x in w), w)


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass
    tmp = Path(tempfile.mkdtemp(prefix="ragbench_core_integration_"))
    try:
        test_registry_and_flags()
        test_answer_formats()
        test_answer_call_sources()
        test_final_guard()
        test_dry_run(tmp)
        test_agentic_caps(tmp)
    finally:
        failed = [n for n, ok in RESULTS if not ok]
        print("\n%d checks, %d failed%s" % (len(RESULTS), len(failed), (": " + ", ".join(failed)) if failed else ""))
        shutil.rmtree(tmp, ignore_errors=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
