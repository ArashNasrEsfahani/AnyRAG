"""Offline tests of the datasets-v2 test-freeze fixes (FREEZE_ASTRA_R1): unknown-usage token bounds (tools/token_bounds),
the --test-protocol gate (run.test_protocol_problems) and the reader's attempt policy record.

    .venv/Scripts/python.exe -m ragbench.tests.test_freeze

No network, no model.
"""
import argparse
import hashlib
import json
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[2]))

from ragbench import run as RUN  # noqa: E402
from ragbench.tools import token_bounds as TB  # noqa: E402

RESULTS = {}


def check(name, cond, detail=""):
    RESULTS[name] = {"ok": bool(cond), "detail": detail}
    print(("PASS " if cond else "FAIL ") + name + (" - " + str(detail)[:300] if detail and not cond else ""), flush=True)
    return cond


def call(purpose, ok=True, pc=1000, pt=250, ct=50, known=True, setting="fa-fa", max_tokens=None, phase="query"):
    return {"kind": "chat", "phase": phase, "purpose": purpose, "model": "m", "ok": ok, "usage_known": known,
            "prompt_chars": pc, "prompt_tokens": pt if ok else 0, "completion_tokens": ct if ok else 0,
            "setting": setting, "max_tokens": max_tokens, "question_id": "q1"}


def test_token_bounds():
    led = [call("sel", pc=1000, pt=300, max_tokens=600), call("sel", pc=1000, pt=200, setting="en-en", max_tokens=600),
           call("answer", pc=2000, pt=400, max_tokens=3256), call("warmup", pc=10, pt=900, phase="smoke")]
    cal = TB.calibrate(led, margin=1.10)
    check("tb.calibration_is_max_ratio_x_margin", abs(cal["ratios"]["m|sel|fa"] - 0.33) < 1e-9
          and abs(cal["ratios"]["m|sel|*"] - 0.33) < 1e-9 and abs(cal["ratios"]["m|sel|en"] - 0.22) < 1e-9, cal)
    check("tb.smoke_calls_excluded", "m|warmup|*" not in cal["ratios"], cal["ratios"])
    unk = call("sel", ok=False, known=False, pc=1000, max_tokens=600)
    b = TB.call_bounds(unk, cal)
    check("tb.unknown_upper_is_prompt_bound_plus_cap", b["lower"] == 0 and b["upper"] == 330 + 600 and not b["known"], b)
    old = dict(unk, max_tokens=None)                       # an old record without max_tokens: the purpose cap
    check("tb.cap_fallback", TB.call_bounds(old, cal)["upper"] == 930)
    two = TB.question_bounds([call("answer"), unk, dict(unk), call("warmup", phase="smoke")], cal)
    check("tb.summed_per_call_not_per_question", two == {"lower": 300, "upper": 300 + 2 * 930, "calls": 3,
                                                         "unknown": 2}, two)
    g = TB.gate({"lower": 200, "upper": 260}, {"lower": 100, "upper": 100}, 2.5)
    check("tb.gate_unresolved_when_bounds_straddle", g["status"] == "unresolved", g)
    check("tb.gate_passed", TB.gate({"lower": 200, "upper": 240}, {"lower": 100, "upper": 100}, 2.5)["status"]
          == "passed")
    check("tb.gate_failed", TB.gate({"lower": 260, "upper": 300}, {"lower": 100, "upper": 100}, 2.5)["status"]
          == "failed")
    try:
        TB.call_bounds(dict(unk, model="other"), cal)
        check("tb.unknown_model_refused", False)
    except KeyError:
        check("tb.unknown_model_refused", True)


def args_(**kw):
    base = dict(final=False, reader="openrouter:z-ai/glm-5.3-flash", pool=None, allow_mixed=False, limit=None,
                docs=None, docs_from_questions=None)
    base.update(kw)
    return argparse.Namespace(**base)


def test_protocol_gate(tmp):
    qf = tmp / "test_v2run_RPT-fa.jsonl"
    qf.write_text('{"id": "x"}\n', encoding="utf-8")
    pool = tmp / "RPT-fa.json"
    pool.write_text('{"pool_id": "RPT-fa"}', encoding="utf-8")
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()       # noqa: E731
    pins = {"z-ai/glm-5.3-flash": {"provider": "together", "params": ["reasoning"]}}
    man = {"question_files": {qf.name: {"sha256": sha(qf)}}, "pools": {"RPT-fa": {"sha256": sha(pool)}},
           "arms": [{"variant": "HYBRID", "env": {}}, {"variant": "LEX5B", "env": {"LEX8_IDX_DIR": "I"}}],
           "reader": "openrouter:z-ai/glm-5.3-flash", "reader_pins": pins, "reader_max_attempts": 7,
           "selector_pin_env": None, "code_commit": "abc"}
    mp = tmp / "MANIFEST.json"
    mp.write_text(json.dumps(man), encoding="utf-8")
    env = {"OPENROUTER_PROVIDER_PINS": json.dumps(pins), "READER_MAX_ATTEMPTS": "7"}
    ok = RUN.test_protocol_problems(args_(pool=str(pool)), mp, qf, "HYBRID", environ=env, head="abc", dirty="")
    check("gate.matching_run_allowed", ok == [], ok)
    cases = {
        "variant": (args_(pool=str(pool)), "HR_9", env, "abc", ""),
        "reader": (args_(pool=str(pool), reader="openrouter:other"), "HYBRID", env, "abc", ""),
        "attempts": (args_(pool=str(pool)), "HYBRID", dict(env, READER_MAX_ATTEMPTS="3"), "abc", ""),
        "pins": (args_(pool=str(pool)), "HYBRID", dict(env, OPENROUTER_PROVIDER_PINS="{}"), "abc", ""),
        "selector_pin": (args_(pool=str(pool)), "HYBRID", dict(env, LEX5B_SELECTOR_PIN='{"provider": "x"}'), "abc", ""),
        "arm_env": (args_(pool=str(pool)), "LEX5B", env, "abc", ""),
        "commit": (args_(pool=str(pool)), "HYBRID", env, "def", ""),
        "dirty": (args_(pool=str(pool)), "HYBRID", env, "abc", " M ragbench/run.py"),
        "final": (args_(pool=str(pool), final=True), "HYBRID", env, "abc", ""),
        "limit": (args_(pool=str(pool), limit=3), "HYBRID", env, "abc", ""),
        "tuning_env": (args_(pool=str(pool)), "HYBRID", dict(env, LEX_OPTIONS="{}"), "abc", ""),
    }
    for name, (a, variant, e, head, dirty) in cases.items():
        p = RUN.test_protocol_problems(a, mp, qf, variant, environ=e, head=head, dirty=dirty)
        check("gate.refuses_%s" % name, len(p) >= 1, p)
    qf.write_text('{"id": "y"}\n', encoding="utf-8")
    p = RUN.test_protocol_problems(args_(pool=str(pool)), mp, qf, "HYBRID", environ=env, head="abc", dirty="")
    check("gate.refuses_changed_question_file", any("sha256" in x for x in p), p)
    ok2 = RUN.test_protocol_problems(args_(pool=str(pool)), mp, qf, "LEX5B", environ=dict(env, LEX8_IDX_DIR="I"),
                                     head="abc", dirty="")
    check("gate.arm_env_declared_ok", [x for x in ok2 if "sha256" not in x] == [], ok2)


def test_reader_attempts():
    from ragbench.config import load_config
    from ragbench.llm import Ledger
    from ragbench.llm_openrouter import OpenRouterReader
    cfg = load_config()
    d = OpenRouterReader(cfg, Ledger(path=None), "x/y")
    one = OpenRouterReader(cfg, Ledger(path=None), "x/y", attempts=1)
    seven = OpenRouterReader(cfg, Ledger(path=None), "x/y", attempts=7)
    check("reader.default_attempts_7", d.or_retries + 1 == 7)
    check("reader.explicit_attempts", one.or_retries + 1 == 1 and seven.or_retries + 1 == 7)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    tmp = Path(tempfile.mkdtemp(prefix="ragbench_freeze_"))
    for name, fn in (("token_bounds", test_token_bounds), ("gate", lambda: test_protocol_gate(tmp)),
                     ("reader", test_reader_attempts)):
        try:
            fn()
        except Exception as e:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            check("%s.raised" % name, False, "%s: %s" % (type(e).__name__, e))
    failed = [k for k, v in RESULTS.items() if not v["ok"]]
    print("\n%d checks, %d failed%s" % (len(RESULTS), len(failed), (": " + ", ".join(failed)) if failed else ""))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
