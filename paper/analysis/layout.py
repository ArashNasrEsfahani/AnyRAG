"""Path resolution for the paper's analysis scripts (no absolute paths).

Two layouts are recognised, relative to a repository root:

  workspace   <root>/work/datasets_v2/test_run      (test records, verdicts, report_test.py)
              <root>/work/datasets_v2/test_freeze   (MANIFEST.json, token_calibration.json, questions/)
              code: <root>/../RAG-Paper-upgrades     (the ragbench package)
  release     <root>/results/test                   (test records, verdicts, symmetric human merge)
              <root>/results/protocol               (MANIFEST.json, token_calibration.json, questions/)
              <root>/code/test_run/report_test.py,  code: <root>/code

The root defaults to two levels above this file (paper/analysis -> repository root), so the scripts run unchanged
from either checkout. Override with --root, or with the environment variable ANYRAG_ROOT; override the ragbench
location with --code or ANYRAG_CODE_DIR.
"""
import argparse
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def parse_args(argv=None, description=None):
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("--root", default=os.environ.get("ANYRAG_ROOT"),
                    help="repository root (default: two levels above paper/analysis)")
    ap.add_argument("--code", default=os.environ.get("ANYRAG_CODE_DIR"),
                    help="directory that contains the ragbench package (default: by layout)")
    return ap.parse_args(argv)


def resolve(root=None, code=None):
    """-> dict(layout, root, test, freeze, questions, report_dir, code)."""
    root = Path(root).resolve() if root else HERE.parent.parent
    if (root / "work" / "datasets_v2" / "test_run").is_dir():
        ds = root / "work" / "datasets_v2"
        P = dict(layout="workspace", test=ds / "test_run", freeze=ds / "test_freeze", report_dir=ds / "test_run",
                 code=root.parent / "RAG-Paper-upgrades")
    elif (root / "results" / "test").is_dir():
        P = dict(layout="release", test=root / "results" / "test", freeze=root / "results" / "protocol",
                 report_dir=root / "code" / "test_run", code=root / "code")
    else:
        sys.exit("layout not recognised under %s (expected work/datasets_v2/test_run or results/test); "
                 "pass --root" % root)
    if code:
        P["code"] = Path(code).resolve()
    P["root"] = root
    P["questions"] = P["freeze"] / "questions" / "test_v2judge_all.jsonl"
    return P


def import_report_test(P):
    """Import report_test.py with its data paths pointed at this layout (env overrides, results-neutral)."""
    os.environ["ANYRAG_TEST_DIR"] = str(P["test"])
    os.environ["ANYRAG_FREEZE_DIR"] = str(P["freeze"])
    os.environ["ANYRAG_CODE_DIR"] = str(P["code"])
    for d in (P["code"], P["report_dir"]):
        if str(d) not in sys.path:
            sys.path.insert(0, str(d))
    import report_test  # noqa: E402
    if Path(report_test.HERE).resolve() != Path(P["test"]).resolve():
        sys.exit("report_test.py at %s does not read %s; use the release copy (code/test_run/report_test.py), "
                 "which honours ANYRAG_TEST_DIR" % (report_test.__file__, P["test"]))
    return report_test
