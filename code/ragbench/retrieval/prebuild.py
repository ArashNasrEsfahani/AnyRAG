"""Prebuild the shared hybrid index of one or more corpus sets, with EVERY chunk vector computed ahead of time.

    .venv/Scripts/python.exe -m ragbench.retrieval.prebuild --set C
    .venv/Scripts/python.exe -m ragbench.retrieval.prebuild --set D --set A_fa --batch-size 16 --commit-every 4
    .venv/Scripts/python.exe -m ragbench.retrieval.prebuild --set C --status        # what is there, no model call
    .venv/Scripts/python.exe -m ragbench.retrieval.prebuild --set C --no-embed      # chunks + BM25 + cached vectors only

Resumable: vectors go to B3's shared cache (<index_root>/_embed_cache/embeddings.sqlite, B3's keys) in one sqlite
transaction every --commit-every batches; a stopped run loses at most those batches and the next run skips every
cached text. Progress lines go to stderr; a summary JSON to <index dir>/prebuild_log.jsonl. The process lowers its
own priority (below normal; --priority idle for the lowest) and numpy threads default to 2 (gentle local jobs).
"""
import argparse
import json
import os
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")


def lower_priority(level="below_normal"):
    """Best effort; returns what was done."""
    try:
        if os.name == "nt":
            import ctypes
            cls = {"below_normal": 0x00004000, "idle": 0x00000040}[level]
            k32 = ctypes.windll.kernel32
            k32.GetCurrentProcess.restype = ctypes.c_void_p
            k32.SetPriorityClass.argtypes = (ctypes.c_void_p, ctypes.c_uint32)
            ok = k32.SetPriorityClass(k32.GetCurrentProcess(), cls)
            return "%s (%s)" % (level, "ok" if ok else "failed")
        os.nice(10 if level == "below_normal" else 19)
        return level
    except Exception as e:  # noqa: BLE001
        return "unchanged (%s)" % type(e).__name__


def prebuild(corpus_set, cfg, client, embed=True, batch_size=None, commit_every=1, progress=True, max_batches=None):
    """Build (or complete) the hybrid index of one CorpusSet. -> status dict. Raises what the client raises, after
    committing the finished batches."""
    from .hybrid import HybridIndex, index_dir_for
    from ..util import append_jsonl, now_iso
    t0 = time.perf_counter()
    rec = {"set": corpus_set.name, "started": now_iso(), "embed": bool(embed), "index_dir": str(index_dir_for(corpus_set, cfg))}
    prog = (lambda d: print(d["msg"], file=sys.stderr, flush=True)) if progress is True else (progress or None)
    try:
        hx = HybridIndex.build(corpus_set, cfg, client=client if embed else None, embed=embed, progress=prog,
                               batch_size=batch_size, commit_every=commit_every, max_batches=max_batches)
        rec.update(hx.status(), ok=True, embed_stats=(hx.meta and read_embed_stats(hx)))
        return rec
    except BaseException as e:
        rec.update(ok=False, error="%s: %s" % (type(e).__name__, str(e)[:300]))
        raise
    finally:
        rec["wall_s"] = round(time.perf_counter() - t0, 1)
        try:
            append_jsonl(index_dir_for(corpus_set, cfg) / "prebuild_log.jsonl", rec)
        except Exception:  # noqa: BLE001
            pass


def read_embed_stats(hx):
    from ..util import read_json
    return (read_json(hx.index_dir / "vector_meta.json") or {}).get("embed_stats")


def status(corpus_set, cfg):
    from .hybrid import HybridIndex, index_dir_for
    try:
        hx = HybridIndex.load(corpus_set, cfg, refresh_vectors=False)
        return hx.status()
    except (FileNotFoundError, RuntimeError) as e:
        return {"corpus_set": corpus_set.name, "index_dir": str(index_dir_for(corpus_set, cfg)), "built": False,
                "why": str(e)[:200]}


def build_parser():
    ap = argparse.ArgumentParser(prog="python -m ragbench.retrieval.prebuild", description=__doc__.split("\n\n")[0])
    ap.add_argument("--set", action="append", required=True, help="corpus set (repeatable or comma list): C, D, A_fa ...")
    ap.add_argument("--no-embed", action="store_true", help="chunks + BM25 + vectors already cached; no model call")
    ap.add_argument("--status", action="store_true", help="print the index state and exit (no build, no model call)")
    ap.add_argument("--batch-size", type=int, default=None, help="texts per embed call (default embedding.batch_size)")
    ap.add_argument("--commit-every", type=int, default=4, help="batches per cache transaction (default 4)")
    ap.add_argument("--priority", choices=("below_normal", "idle"), default="below_normal")
    ap.add_argument("--max-batches", type=int, default=None, help="stop after this many embed batches (a slice; resumable)")
    return ap


def main(argv=None, cfg=None, client=None, sets=None):
    """sets: optional {name: CorpusSet} (tests pass synthetic sets); client: optional embed client."""
    args = build_parser().parse_args(argv)
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass
    from ..config import load_config
    cfg = cfg or load_config()
    names = [x.strip() for v in args.set for x in v.split(",") if x.strip()]
    print("[prebuild] priority: %s; OMP_NUM_THREADS=%s" % (lower_priority(args.priority), os.environ.get("OMP_NUM_THREADS")),
          file=sys.stderr, flush=True)

    def get_set(name):
        if sets and name in sets:
            return sets[name]
        from ..corpus import load_set
        return load_set(name, cfg)

    out = []
    for name in names:
        cs = get_set(name)
        if args.status:
            out.append(status(cs, cfg))
            continue
        cl = None
        if not args.no_embed:
            if client is not None:
                cl = client
            else:
                from ..llm import Ledger, OllamaClient
                cl = OllamaClient(cfg, Ledger(path=None))
        with_ledger = getattr(cl, "ledger", None)
        if with_ledger is not None and hasattr(with_ledger, "scope"):
            with with_ledger.scope(phase="index", approach="HYBRID-prebuild", corpus_set=cs.name):
                out.append(prebuild(cs, cfg, cl, embed=not args.no_embed, batch_size=args.batch_size,
                                    commit_every=args.commit_every, max_batches=args.max_batches))
        else:
            out.append(prebuild(cs, cfg, cl, embed=not args.no_embed, batch_size=args.batch_size,
                                commit_every=args.commit_every, max_batches=args.max_batches))
    for r in out:
        print(json.dumps(r, ensure_ascii=False, default=str))
    return out


if __name__ == "__main__":
    main()
