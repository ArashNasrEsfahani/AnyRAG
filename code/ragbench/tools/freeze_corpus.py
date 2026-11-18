"""Freeze the benchmark corpus so that editing a live KareOne report can never move it.

    .venv/Scripts/python.exe -m ragbench.tools.freeze_corpus freeze     # once; refuses to overwrite a freeze
    .venv/Scripts/python.exe -m ragbench.tools.freeze_corpus verify     # any time; exit 1 on drift

WHY. Corpus A was read straight from the working report folders under D:/KareOne, which are live projects. On
2026-09-21 one of those reports was rewritten while the study was running: some of its units changed, including
gold units of dev questions, and corpus A's index fingerprint moved under the dev runs. Corpora B and C were never exposed: they are PDFs and their extraction output
inside this project.

WHAT. `freeze` copies exactly the files the corpus-A adapter reads (corpus/kareone.py: report.meta.json,
chapters/*.json, and build/99_assemble.py for the chapter ORDER and the fingerprint) into corpus/frozen/kareone/,
keeping each report's relative path, with shutil.copy2 so size AND mtime are preserved. kareone.fingerprint hashes
name, size and mtime_ns, so the frozen copy fingerprints identically to the source at the moment of freezing: no
cached Document is rebuilt and every existing record's fingerprint stays valid. config.yaml `paths.kareone_root` then
points at the frozen copy; `paths` is outside the records' config hash (RECORD_HASH_SECTIONS), so no hash moves.

The manifest corpus/frozen/FROZEN.json records the time, the live root it was copied from, and sha256 + size + mtime
of every frozen corpus-A file AND of every corpus-B/C extraction file the loaders read (document.json and
tables/*.json under corpus/extracted), so `verify` detects a change to any of the three corpora.

WHICH VERSION. There is no version control under D:/KareOne, so no earlier state of a report can be recovered: the
freeze takes the reports as they are. The core-test reports last
changed before the core test ran on 2026-09-16, so the test's corpus is exactly what was frozen; the rewritten report
is frozen as rewritten, and the dev questions that read it are re-checked against the frozen text.

PUBLIC RELEASE. Corpus A (an early development corpus of internal reports) is not part of the release; its report
list in config.yaml is empty and benchmark/corpus_manifest/FROZEN_reports.json keeps only the B/C extraction hashes.
"""
import argparse
import hashlib
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

from ..config import load_config

FROZEN_DIR = "corpus/frozen"
KAREONE_SUBDIR = "kareone"
MANIFEST = "FROZEN.json"


def _sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _entry(p, rel):
    st = p.stat()
    return {"path": rel, "sha256": _sha256(p), "size": st.st_size, "mtime_ns": st.st_mtime_ns}


def report_files(report_dir):
    """The files corpus/kareone.py reads for one report, in a stable order."""
    report_dir = Path(report_dir)
    files = [report_dir / "report.meta.json", report_dir / "build" / "99_assemble.py"]
    files += sorted(report_dir.glob("chapters/*.json"))
    return [p for p in files if p.is_file()]


def extraction_files(cfg):
    root = cfg.path("extracted_root")
    return sorted([p for p in root.glob("*/*/document.json")] + [p for p in root.glob("*/*/tables/*.json")])


def freeze(cfg, live_root, force=False, notes=None):
    project = cfg.project_root
    out = project / FROZEN_DIR
    dest_root = out / KAREONE_SUBDIR
    if (out / MANIFEST).exists() and not force:
        raise SystemExit("already frozen (%s); `verify` checks it, --force re-freezes" % (out / MANIFEST))
    live_root = Path(live_root)
    a_files = []
    for r in cfg.corpora["A"]["reports"]:
        src_dir = live_root / r["path"]
        files = report_files(src_dir)
        if not files:
            raise SystemExit("report %s has no source files under %s" % (r["key"], src_dir))
        for p in files:
            rel = p.relative_to(live_root).as_posix()
            dst = dest_root / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, dst)
            e = _entry(dst, rel)
            e["report"] = r["key"]
            if e["sha256"] != _sha256(p) or dst.stat().st_mtime_ns != p.stat().st_mtime_ns:
                raise SystemExit("copy of %s is not identical (content or mtime); nothing is trusted" % p)
            a_files.append(e)
    bc = [_entry(p, p.relative_to(project).as_posix()) for p in extraction_files(cfg)]
    manifest = {"frozen_at": datetime.now().isoformat(timespec="seconds"), "live_root": live_root.as_posix(),
                "kareone_root": dest_root.as_posix(), "corpus_A": a_files, "corpus_BC_extraction": bc,
                "counts": {"A_reports": len(cfg.corpora["A"]["reports"]), "A_files": len(a_files),
                           "BC_files": len(bc)},
                "notes": notes or []}
    (out / MANIFEST).write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8", newline="\n")
    return manifest


def verify(cfg):
    """-> (ok, problems). Every manifest file must exist with the same sha256; kareone_root must be the frozen copy."""
    project = cfg.project_root
    man_path = project / FROZEN_DIR / MANIFEST
    if not man_path.exists():
        return False, ["not frozen: %s is missing" % man_path]
    man = json.loads(man_path.read_text(encoding="utf-8"))
    problems = []
    if Path(cfg.paths["kareone_root"]).resolve() != Path(man["kareone_root"]).resolve():
        problems.append("config paths.kareone_root is %s, not the frozen copy %s" % (
            cfg.paths["kareone_root"], man["kareone_root"]))
    root = Path(man["kareone_root"])
    for e in man["corpus_A"]:
        p = root / e["path"]
        if not p.is_file():
            problems.append("A: missing %s" % e["path"])
        elif _sha256(p) != e["sha256"]:
            problems.append("A: changed %s" % e["path"])
    for e in man["corpus_BC_extraction"]:
        p = project / e["path"]
        if not p.is_file():
            problems.append("B/C: missing %s" % e["path"])
        elif _sha256(p) != e["sha256"]:
            problems.append("B/C: changed %s" % e["path"])
    return not problems, problems


def main(argv=None):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    ap = argparse.ArgumentParser(prog="python -m ragbench.tools.freeze_corpus", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("freeze")
    f.add_argument("--live-root", default="D:/KareOne", help="where the working KareOne reports live")
    f.add_argument("--force", action="store_true")
    f.add_argument("--note", action="append")
    sub.add_parser("verify")
    args = ap.parse_args(argv)
    cfg = load_config()
    if args.cmd == "freeze":
        m = freeze(cfg, args.live_root, force=args.force, notes=args.note)
        print("frozen %d corpus-A files from %d reports into %s; manifest also covers %d B/C extraction files" % (
            m["counts"]["A_files"], m["counts"]["A_reports"], m["kareone_root"], m["counts"]["BC_files"]))
        return 0
    ok, problems = verify(cfg)
    if ok:
        man = json.loads((cfg.project_root / FROZEN_DIR / MANIFEST).read_text(encoding="utf-8"))
        print("frozen corpus verified: %d files match FROZEN.json, kareone_root is the frozen copy, 0 failures"
              % (len(man["corpus_A"]) + len(man["corpus_BC_extraction"])))
        return 0
    print("%d problem(s):" % len(problems))
    for p in problems[:40]:
        print("  -", p)
    return 1


if __name__ == "__main__":
    sys.exit(main())
