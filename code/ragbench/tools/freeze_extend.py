"""Extend the corpus freeze with NEW material only, never re-copying a report already frozen.

    .venv/Scripts/python.exe -m ragbench.tools.freeze_extend

Adds (1) every corpus-A report in config.yaml that is not yet frozen (2026-09-23: the one fully
unseen report of the fresh holdout) - its files copied with copy2 so they fingerprint like the live
source; (2) every corpus-B/C/D extraction file not yet in the manifest (the corpus-D English reports). A full
`freeze_corpus freeze --force` would re-copy every live report and could pick up edits made since the first freeze
(one report was rewritten once already), so it is not used. Idempotent; appends a note with the time and what was added.
"""
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

from ..config import load_config
from .freeze_corpus import FROZEN_DIR, KAREONE_SUBDIR, MANIFEST, _entry, extraction_files, report_files


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    cfg = load_config()
    out = cfg.project_root / FROZEN_DIR
    man_path = out / MANIFEST
    man = json.loads(man_path.read_text(encoding="utf-8"))
    live_root = Path(man["live_root"])
    dest_root = out / KAREONE_SUBDIR
    have_a = {e["path"] for e in man["corpus_A"]}
    added_a, added_x = [], []
    for r in cfg.corpora["A"]["reports"]:
        rel_dir = Path(r["path"]).as_posix()
        if any(p.startswith(rel_dir + "/") for p in have_a):
            continue
        files = report_files(live_root / r["path"])
        if not files:
            raise SystemExit("report %s has no source files under %s" % (r["key"], live_root / r["path"]))
        for p in files:
            rel = p.relative_to(live_root).as_posix()
            dst = dest_root / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, dst)
            e = _entry(dst, rel)
            e["report"] = r["key"]
            man["corpus_A"].append(e)
            added_a.append(rel)
    have_x = {e["path"] for e in man["corpus_BC_extraction"]}
    project = cfg.project_root              # the manifest's extraction paths are project-relative (freeze_corpus)
    for p in extraction_files(cfg):
        rel = p.relative_to(project).as_posix()
        if rel not in have_x:
            man["corpus_BC_extraction"].append(_entry(p, rel))
            added_x.append(rel)
    if not added_a and not added_x:
        print("nothing new to freeze")
        return 0
    man["counts"]["A_files"] = len(man["corpus_A"])
    man["counts"]["A_reports"] = len({e.get("report") or e["path"].split("/chapters")[0] for e in man["corpus_A"]})
    man["counts"]["BC_files"] = len(man["corpus_BC_extraction"])
    note = "%s: freeze_extend added %d corpus-A files (%s) and %d extraction files" % (
        datetime.now().isoformat(timespec="seconds"), len(added_a),
        sorted({a.split("/chapters")[0].split("/report.meta")[0].split("/build")[0] for a in added_a}), len(added_x))
    notes = man.get("notes")
    man["notes"] = (notes if isinstance(notes, list) else [notes] if notes else []) + [note]
    tmp = man_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(man, ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.replace(man_path)
    print(note)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
