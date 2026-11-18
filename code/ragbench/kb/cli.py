"""`python -m ragbench.kb <command>`: the knowledge base from the command line (AGENTIC3_SPEC.md section 4.1).

    add <file-or-folder> [--company snapp --year 1403 --lang fa --corpus U --pair-of <doc_id>] [--re-extract]
    watch [--inbox corpus/inbox] [--interval 2] [--no-embed]
    sync [--sets A_fa,A_en,B_fa,B_en,C] [--force]
    embed [--doc <doc_id> ...] [--questions <files>] [--all] [--limit N] [--dry-run]
    remove <doc_id> [--yes]
    status [--md work/runs/core/INGEST_COST.md] [--json]
    sql "SELECT ..." [--limit 50] [--json]

Every command but `embed` runs without a model. `embed` is the only GPU user: it embeds the passages that have no
vector, reusing the shared embedding cache that B3 and AGENTIC2 fill, and with no `--doc` / `--all` it embeds only
the documents the core dev and test question files reference (the `--docs-from-questions` rule of `run.py`).
`sql` opens a read-only connection, so it is safe to run while an ingest is going on.

Every command is resumable: state lives in the database, each stage commits on its own, and a document that is
already loaded with the same content and the current component versions runs zero stages.
"""
import argparse
import json
import sys
import time
from pathlib import Path

from ..config import load_config
from . import ingest as I

CORE_QUESTION_FILES = ("test_core.jsonl", "test_timeline.jsonl", "dev_core.jsonl", "dev_core_timeline.jsonl",
                       "pilot_timeline.jsonl")
DEFAULT_COST_MD = "work/runs/core/INGEST_COST.md"


def _utf8():
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8")
        except Exception:  # noqa: BLE001
            pass


def _fmt_stages(stages):
    return " ".join("%s %.0fms" % (k, v) for k, v in stages.items())


def cmd_add(args, cfg, con):
    res = I.add_file(con, args.path, cfg, company=args.company, year=args.year, lang=args.lang,
                     corpus=args.corpus, key=args.key, pair_of=args.pair_of, re_extract=args.re_extract,
                     force=args.force)
    for r in res.get("documents", [res]):
        print("%s  %s  %s" % (r.get("doc_id"), r.get("action"), _fmt_stages(r.get("stages") or {})))
        if r.get("action") == "rejected":
            print("  reasons: " + "; ".join(r.get("reasons") or []))
        if r.get("rows"):
            print("  rows: " + ", ".join("%s %s" % kv for kv in sorted(r["rows"].items())))
    _graph_current(con)
    print("total %.2f s" % ((res.get("ms") or 0.0) / 1000.0))
    return 0


def _graph_current(con):
    """The fact graph is a whole-corpus pass (stems are validated against the corpus), so it is rebuilt ONCE after a
    command that ingested something, not once per document."""
    from . import graph as G
    G.ensure_current(con, log=print)


def cmd_sync(args, cfg, con):
    names = [s.strip() for s in args.sets.split(",") if s.strip()]
    t0 = time.perf_counter()
    counts = {}

    def log(res):
        counts[res["action"]] = counts.get(res["action"], 0) + 1
        if res["action"] != "skipped" or args.verbose:
            print("%-40s %-8s %s" % (res["doc_id"], res["action"], _fmt_stages(res.get("stages") or {})),
                  flush=True)

    I.sync_sets(con, names, cfg, force=args.force, log=log)
    _graph_current(con)
    ran = sum(n for a, n in counts.items() if a != "skipped")
    print("sync %s: %s in %.1f s (%d document(s) executed stages)" %
          (",".join(names), ", ".join("%s %d" % kv for kv in sorted(counts.items())),
           time.perf_counter() - t0, ran))
    return 0


def _embed_targets(args, cfg, con):
    if args.doc:
        return list(args.doc)
    if args.all:
        return [r["doc_id"] for r in con.execute(
            "SELECT DISTINCT doc_id FROM passages WHERE vec IS NULL ORDER BY doc_id")]
    files = [Path(f) if Path(f).is_absolute() else cfg.path("questions_root") / f
             for f in (args.questions.split(",") if args.questions else CORE_QUESTION_FILES)]
    return I.question_doc_ids(con, [f for f in files if f.is_file()], cfg)


def cmd_embed(args, cfg, con):
    targets = _embed_targets(args, cfg, con)
    pending = I.pending_passages(con, targets)
    docs = sorted({p["doc_id"] for p in pending})
    if args.limit:
        docs = docs[: args.limit]
        pending = [p for p in pending if p["doc_id"] in set(docs)]
    print("%d document(s) in scope, %d passage(s) without a vector, %d character(s)" %
          (len(targets), len(pending), sum(len(p["text"]) for p in pending)))
    if args.dry_run or not pending:
        return 0
    stats = I.embed_documents(con, docs, cfg,
                              log=lambda r: print("  %-40s %4d passages, %4d new, %.0f ms" %
                                                  (r["doc_id"], r["passages"], r["embedded"], r["ms"]), flush=True))
    print(json.dumps(stats, ensure_ascii=False))
    return 0


def cmd_remove(args, cfg, con):
    res = I.remove_document(con, args.doc_id)
    if not res["before"]["documents"]:
        print("%s is not in the knowledge base" % args.doc_id)
        return 1
    print("%s removed: %s" % (args.doc_id, ", ".join("%s %d" % kv for kv in sorted(res["before"].items()))))
    left = {k: v for k, v in res["after"].items() if v}
    print("rows left for this document: %s" % (left or "none"))
    return 0


def cmd_status(args, cfg, con):
    st = I.status(con, cfg)
    if args.json:
        print(json.dumps(st, ensure_ascii=False, indent=1))
    else:
        print("%s  %.1f MB  schema %s" % (st["path"], st["size_bytes"] / 1e6, st["versions"]["schema"]))
        print("rows:   " + ", ".join("%s %d" % kv for kv in st["counts"].items()))
        print("status: " + (", ".join("%s %d" % kv for kv in sorted(st["by_status"].items())) or "no documents") +
              ("; %d stale" % st["stale"] if st["stale"] else ""))
        print("vectors: %d of %d passages" % (st["passages_with_vec"], st["passages"]))
        print()
        print("%-8s %6s %9s %9s %9s %12s" % ("corpus", "docs", "sections", "units", "facts", "chars"))
        for r in st["by_corpus"]:
            print("%-8s %6d %9d %9d %9d %12d" % (r["corpus"], r["docs"], r["sections"] or 0, r["units"] or 0,
                                                 r["facts"] or 0, r["chars"] or 0))
        print()
        print("%-10s %6s %10s %10s %10s %12s" % ("stage", "docs", "median ms", "mean ms", "max ms", "rows"))
        for name, q in st["stages"].items():
            print("%-10s %6d %10s %10s %10s %12d" % (name, q["n"], q["median"], q["mean"], q["max"], q["rows"]))
    if args.md is not None:
        path = Path(args.md or DEFAULT_COST_MD)
        path = path if path.is_absolute() else cfg.project_root / path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(I.cost_markdown(con, cfg), encoding="utf-8", newline="\n")
        print("\nwrote %s" % path)
    return 0


def cmd_sql(args, cfg, con):
    import sqlite3
    rows = []
    try:
        cur = con.execute(args.query)
    except sqlite3.Error as e:
        print("sql: %s" % e)                      # a write on the read-only connection lands here
        return 2
    for i, r in enumerate(cur):
        if i >= args.limit:
            print("... (limit %d)" % args.limit)
            break
        rows.append(r)
    if args.json:
        print(json.dumps(rows, ensure_ascii=False, default=str))
    elif rows:
        cols = list(rows[0])
        print(" | ".join(cols))
        for r in rows:
            print(" | ".join("" if r[c] is None else str(r[c])[:120] for c in cols))
    print("%d row(s)" % len(rows))
    return 0


def cmd_watch(args, cfg, con):
    from .watch import Watcher
    w = Watcher(cfg, inbox=args.inbox, interval_s=args.interval, embed=not args.no_embed, log=print)
    return w.run(once=args.once)


def cmd_labels(args, cfg, con):
    from . import labels as L
    if args.action == "build":
        print(json.dumps(L.build(con, doc_ids=args.doc, rebuild=args.rebuild, log=print), ensure_ascii=False))
    elif args.action == "status":
        print(json.dumps(L.status(con), ensure_ascii=False, indent=1))
    else:
        if not args.term:
            print("labels match needs a term")
            return 2
        rows = L.match(con, args.term, kinds=tuple(args.kind or ("metric", "entity", "section")), limit=100000)
        print("%d rows (%d word hits)" % (len(rows), sum(1 for r in rows if r["hit"] == "word")))
        for r in rows[:args.limit]:
            print("  %-7s %-5s %-34s %s" % (r["kind"], r["hit"], r["doc_id"][:34], r["text"][:60]))
    return 0


def build_parser():
    ap = argparse.ArgumentParser(prog="python -m ragbench.kb", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", help="knowledge base file (default: kb.path, work/kb/kb.sqlite)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("add", help="ingest one file or KareOne report folder")
    a.add_argument("path")
    a.add_argument("--company")
    a.add_argument("--year", type=int)
    a.add_argument("--lang", choices=("fa", "en"))
    a.add_argument("--corpus", default=I.USER_CORPUS, help="corpus letter (default U, never a benchmark set)")
    a.add_argument("--key", help="document key; default: the file stem without the language suffix")
    a.add_argument("--pair-of", help="doc id of the other-language edition")
    a.add_argument("--re-extract", action="store_true", help="run the extractor again even if its output is there")
    a.add_argument("--force", action="store_true", help="rebuild every stage even if the content is unchanged")

    w = sub.add_parser("watch", help="poll the inbox and ingest whatever is dropped in")
    w.add_argument("--inbox")
    w.add_argument("--interval", type=float, default=2.0)
    w.add_argument("--no-embed", action="store_true", help="do not embed in the background")
    w.add_argument("--once", action="store_true", help="one pass over the inbox, then stop (tests)")

    s = sub.add_parser("sync", help="load the benchmark corpora from the cached Documents")
    s.add_argument("--sets", default=",".join(I.BENCH_SETS))
    s.add_argument("--force", action="store_true")
    s.add_argument("--verbose", action="store_true", help="also print the documents that run no stage")

    e = sub.add_parser("embed", help="fill missing passage vectors (the only GPU work)")
    e.add_argument("--doc", action="append", help="repeatable")
    e.add_argument("--questions", help="comma-separated question files; default: the core dev and test files")
    e.add_argument("--all", action="store_true")
    e.add_argument("--limit", type=int, help="at most this many documents")
    e.add_argument("--dry-run", action="store_true")

    r = sub.add_parser("remove", help="delete a document and every row it owns")
    r.add_argument("doc_id")

    st = sub.add_parser("status", help="documents, rows per table, stage timings, database size")
    st.add_argument("--json", action="store_true")
    st.add_argument("--md", nargs="?", const="", help="also write INGEST_COST.md (default %s)" % DEFAULT_COST_MD)

    q = sub.add_parser("sql", help="read-only query")
    q.add_argument("query")
    q.add_argument("--limit", type=int, default=50)
    q.add_argument("--json", action="store_true")

    lb = sub.add_parser("labels", help="the label layer: short metric / entity names and definition tags")
    lb.add_argument("action", choices=("build", "status", "match"))
    lb.add_argument("term", nargs="?", help="for match: the term, any spelling")
    lb.add_argument("--rebuild", action="store_true", help="for build: every document, not only the missing ones")
    lb.add_argument("--doc", action="append", help="for build: repeatable")
    lb.add_argument("--kind", action="append", choices=("metric", "entity", "section"), help="for match")
    lb.add_argument("--limit", type=int, default=20)
    return ap


def main(argv=None):
    _utf8()
    args = build_parser().parse_args(argv)
    cfg = load_config()
    path = args.db or I.kb_path(cfg)
    read_only = args.cmd == "sql" or (args.cmd == "labels" and args.action != "build")
    if read_only and not Path(path).is_file():
        print("no knowledge base at %s; run `python -m ragbench.kb sync` first" % path)
        return 2
    con = I.open_kb(cfg, path=path, read_only=read_only)
    try:
        return {"add": cmd_add, "watch": cmd_watch, "sync": cmd_sync, "embed": cmd_embed, "remove": cmd_remove,
                "status": cmd_status, "sql": cmd_sql, "labels": cmd_labels}[args.cmd](args, cfg, con)
    except KeyboardInterrupt:
        print("\ninterrupted; every finished stage is committed, re-run the same command to continue")
        return 130
    finally:
        con.close()


if __name__ == "__main__":
    sys.exit(main())
