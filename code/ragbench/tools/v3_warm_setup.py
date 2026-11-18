"""Set up (or report on) the warm experiment's own knowledge base and config. No model call.

Used by `tools/v3_warm.sh`. Kept in Python rather than in the shell script on purpose: the setup needs a regex
with escapes, and a Bash heredoc corrupts backslashes silently on this machine.

    python -m ragbench.tools.v3_warm_setup reset  --kb work/kb/kb_warm2.sqlite --config <path>
    python -m ragbench.tools.v3_warm_setup config --kb work/kb/kb_warm2.sqlite --config <path>
    python -m ragbench.tools.v3_warm_setup count  --kb work/kb/kb_warm2.sqlite
"""
import argparse
import re
import shutil
import sqlite3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE_KB = ROOT / "work" / "kb" / "kb.sqlite"


def reset(kb):
    """A fresh copy of the shared knowledge base with an empty `verified_facts` table."""
    for suffix in ("", "-wal", "-shm"):
        p = Path(str(kb) + suffix)
        if p.exists():
            p.unlink()
    kb.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(SOURCE_KB, kb)
    con = sqlite3.connect(kb)
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("DELETE FROM verified_facts")
    con.commit()
    n = con.execute("SELECT count(*) FROM verified_facts").fetchone()[0]
    con.close()
    print("fresh copy %s (%.1f MB), verified_facts rows: %d" % (kb, kb.stat().st_size / 1e6, n))


def write_config(kb, out):
    """The project config with `kb.path` pointed at the copy and nothing else changed."""
    src = (ROOT / "ragbench" / "config.yaml").read_text(encoding="utf-8")
    text, n = re.subn(r"(?m)^(  path: ).*kb\.sqlite.*$", "  path: " + str(kb).replace("\\", "/"), src, count=1)
    if n != 1 or "kb_warm" not in text:
        raise SystemExit("could not point kb.path at %s (matched %d times)" % (kb, n))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    print("wrote %s with kb.path = %s" % (out, kb))


def count(kb):
    con = sqlite3.connect(kb)
    rows = con.execute("SELECT count(*) FROM verified_facts").fetchone()[0]
    qs = con.execute("SELECT count(DISTINCT question_hash) FROM verified_facts").fetchone()[0]
    con.close()
    print("verified_facts: %d rows from %d distinct questions" % (rows, qs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["reset", "config", "count"])
    ap.add_argument("--kb", default="work/kb/kb_warm2.sqlite")
    ap.add_argument("--config", default="work/runs/v3warm2/config_warm2.yaml")
    args = ap.parse_args()
    kb = Path(args.kb) if Path(args.kb).is_absolute() else ROOT / args.kb
    cfg = Path(args.config) if Path(args.config).is_absolute() else ROOT / args.config
    if args.action == "reset":
        reset(kb)
        write_config(kb, cfg)
    elif args.action == "config":
        write_config(kb, cfg)
    else:
        count(kb)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
