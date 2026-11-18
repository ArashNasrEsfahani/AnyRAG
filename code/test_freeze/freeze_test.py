"""FREEZE fix (e) inputs: the test adapter conversion, pool / index manifests with hashes, the pre-test duplicate audit,
and MANIFEST.json (the file `ragbench.run --test-protocol` checks). No model call; no test answer.

    python freeze_test.py inputs      convert + verify the 96 test questions, pool + index hashes, raw-record audit
    python freeze_test.py manifest    MANIFEST.json for the committed code (refuses a dirty ragbench/)
    python freeze_test.py audit       the raw-record duplicate audit alone (run again right before the first answer)

Prints counts only (no test question text).
"""
import datetime as dt
import glob
import hashlib
import json
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
DS = HERE.parent
WORK = DS.parent
WT = r"D:\KareOne\RAG-Paper-upgrades"
sys.path.insert(0, WT)
from ragbench.config import load_config  # noqa: E402
from ragbench.tools import verify_v2run as V  # noqa: E402

QDIR = HERE / "questions"
EXPERIMENT_ID = "datasets-v2-test"
POOLS = ("RPT-fa", "RPT-en", "BOOK-fa", "BOOK-en", "LAW-fa", "LAW-en")
POOL_CORPUS = {"RPT-fa": ("C", "fa-fa"), "RPT-en": ("D", "en-en"), "BOOK-fa": ("K", "fa-fa"),
               "BOOK-en": ("K", "en-en"), "LAW-fa": ("L", "fa-fa"), "LAW-en": ("L", "en-en")}
SRC = ("rpt_test.jsonl", "book_test.jsonl", "law_test.jsonl")
L8 = {"LEX8_IDX_DIR": str(DS / "lex8" / "idx"), "LEX8_SYNTH_DIR": str(DS / "lex8" / "synth")}
READER = "openrouter:z-ai/glm-5.3-flash"
READER_MAX_ATTEMPTS = 7          # = the dev default (6 retries); frozen explicitly, logged on every ledger record
ARMS = [  # (arm, variant written by ragbench.run, run args, env) - all eight chosen BEFORE the test (PROTOCOL.md)
    ("HYBRID_6000", "HYBRID", ["--approach", "hybrid"], {}),
    ("HR_6000", "HYBRID+rerank", ["--approach", "hybrid", "--ablation", "rerank"], {}),
    ("LEX5B_6000", "LEX5B", ["--approach", "lex5b"], L8),
    ("HYBRID_15500", "HYBRID15500", ["--approach-class", "ragbench.approaches.lex10.control:Hybrid15500"], {}),
    ("HR_15500", "HR15500+rerank", ["--approach-class", "ragbench.approaches.budget_matrix:HR15500"], {}),
    ("LEX5B_15500", "LEX5B_15500", ["--approach-class", "ragbench.approaches.budget_matrix:Lex5b_15500"], L8),
    # 2026-10-01 re-freeze (PROTOCOL.md amendment 1): the zero-shot SetR literature baseline at both budgets
    ("SETR_6000", "SETR", ["--approach", "setr"], {}),
    ("SETR_15500", "SETR_15500", ["--approach-class", "ragbench.approaches.budget_matrix:Setr15500"], {})]
INDEX_DIRS = [WORK / "indexes" / "_hybrid", DS / "lex8" / "idx", DS / "lex8" / "synth"] + [
    WORK / "indexes" / v for v in ("HYBRID", "HYBRID15500", "HR15500", "LEX5B", "LEX5B_15500", "SETR", "SETR_15500")]


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def dir_manifest(d):
    files = sorted(p for p in Path(d).rglob("*") if p.is_file())
    rows = {str(p.relative_to(d)).replace("\\", "/"): {"bytes": p.stat().st_size, "sha256": sha256(p)} for p in files}
    return {"files": len(rows), "bytes": sum(r["bytes"] for r in rows.values()),
            "tree_sha256": hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest(), "rows": rows}


def write_json(p, obj):
    Path(p).write_text(json.dumps(obj, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")   # no BOM


def duplicate_audit():
    """Raw-record duplicate audit: any V2-test- id in any run / ledger / verdict / failure file under work/."""
    hits = Counter()
    pats = ("runs_*.jsonl", "ledger_*.jsonl", "verdicts_*.jsonl", "judge_failures_*.jsonl",
            "needs_adjudication_*.jsonl")
    for pat in pats:
        for f in glob.glob(str(WORK / "**" / pat), recursive=True):
            try:
                with open(f, encoding="utf-8", errors="ignore") as fh_:
                    if "V2-test-" in fh_.read():
                        hits[str(Path(f).relative_to(WORK))] += 1
            except OSError:
                hits["unreadable:" + f] += 1
    return {"at": dt.datetime.now().isoformat(timespec="seconds"), "pattern": "V2-test-", "scanned": list(pats),
            "root": str(WORK), "files_with_test_ids": dict(hits), "clean": not hits}


def audit_only():
    a = duplicate_audit()
    write_json(HERE / "duplicate_audit_latest.json", a)
    print(json.dumps({"at": a["at"], "clean": a["clean"], "files_with_test_ids": len(a["files_with_test_ids"])}))
    return 0 if a["clean"] else 1


def inputs():
    frozen_path = DS / "FROZEN_questions.json"
    FZ = V.load_frozen(frozen_path)
    # 1. the six frozen question files are unchanged
    fh = {}
    for k, v in FZ["files"].items():
        now = sha256(DS / k)
        fh[k] = {"sha256": now, "frozen": v["sha256"], "ok": now == v["sha256"], "n": v["n_questions"]}
    assert all(x["ok"] for x in fh.values()), fh
    # 2. convert the three test files (verify_v2run.convert_record, experiment datasets-v2-test)
    QDIR.mkdir(exist_ok=True)
    qs = []
    for name in SRC:
        p = DS / "questions" / name
        s = sha256(p)
        for line in p.read_text(encoding="utf-8").splitlines():
            if line.strip():
                qs.append(V.convert_record(json.loads(line), name, s, EXPERIMENT_ID))
    errs, flags, summ = V.verify(qs, FZ)
    assert not errs, {k: len(v) for k, v in errs.items()}
    out = {"test_v2run_all.jsonl": qs}
    by = {}
    for q in qs:
        by.setdefault(q["pool_id"], []).append(q)
    for pid in POOLS:
        e2, _f2, _s2 = V.verify(by[pid], FZ)
        assert not e2, pid
        out["test_v2run_%s.jsonl" % pid] = by[pid]
    # evaluator-only judge file (as devrun/make_judge_questions.py)
    judge = []
    for q in qs:
        q = json.loads(json.dumps(q))
        r = q["meta"]["v2"]
        ev = {e["id"]: e["doc_id"] for e in r["evidence"]}
        sets = []
        for b in r.get("evidence_bundles") or []:
            ds_ = sorted({ev[e] for e in b if e in ev})
            if ds_ and ds_ not in sets:
                sets.append(ds_)
        q["accepted_evidence_sets"] = sets
        q["source_block"] = r.get("source_block")
        judge.append(q)
    out["test_v2judge_all.jsonl"] = judge
    files = {}
    for name, rows in out.items():
        p = QDIR / name
        p.write_text("".join(json.dumps(q, ensure_ascii=False) + "\n" for q in rows), encoding="utf-8")
        files[name] = {"sha256": sha256(p), "n": len(rows),
                       "answerable": sum(1 for q in rows if q["answer_type"] != "unanswerable")}
    # 3. pools: the frozen manifests, their hashes, their verification, every test evidence doc inside its pool
    from ragbench.corpus.registry import load_set
    from ragbench.pools import load_pool, pool_root
    cfg = load_config()
    pools = {}
    for pid in POOLS:
        pp = pool_root(cfg) / ("%s.json" % pid)
        pool = load_pool(pp)
        sets = {n: load_set(n, cfg) for n in (pool.manifest.get("frozen_sets") or {})}
        probs = pool.verify(sets)
        docs = set(pool.doc_ids())
        outside = sum(1 for q in by[pid] for e in q["meta"]["v2"]["evidence"] if e["doc_id"] not in docs)
        pools[pid] = {"path": str(pp), "sha256": sha256(pp), "manifest_sha1": pool.sha1, "docs": len(docs),
                      "corpus": POOL_CORPUS[pid][0], "setting": POOL_CORPUS[pid][1], "verify": probs or "OK",
                      "test_evidence_outside_pool": outside, "test_questions": len(by[pid])}
        assert not probs and not outside, pid
    # 4. dev/test disjointness (documents of evidence, source blocks, information needs)
    def fields(split):
        rows = [json.loads(l) for n in ("rpt", "book", "law")
                for l in (DS / "questions" / ("%s_%s.jsonl" % (n, split))).read_text(encoding="utf-8").splitlines()
                if l.strip()]
        return ({e["doc_id"] for r in rows for e in r["evidence"]}, {r.get("source_block") for r in rows} - {None},
                {r.get("information_need_id") for r in rows} - {None}, {r.get("source_family") for r in rows} - {None})
    d, t = fields("dev"), fields("test")
    disjoint = {k: len(a & b) for k, a, b in zip(("evidence_docs", "source_blocks", "information_needs",
                                                  "source_families"), d, t)}
    # 5. raw-record duplicate audit
    audit = duplicate_audit()
    hits = audit["files_with_test_ids"]
    # 6. indexes (hash of every file) - the indexes the six arms load
    idx = {}
    for dpath in INDEX_DIRS:
        if Path(dpath).exists():
            m = dir_manifest(dpath)
            idx[str(dpath)] = {k: m[k] for k in ("files", "bytes", "tree_sha256")}
            (HERE / "index_manifests").mkdir(exist_ok=True)
            write_json(HERE / "index_manifests" / ("%s.json" % Path(dpath).name), m)
        else:
            idx[str(dpath)] = "MISSING"
    res = {"at": dt.datetime.now().isoformat(timespec="seconds"), "frozen_questions": {"path": str(frozen_path),
           "sha256": sha256(frozen_path), "files": fh}, "converted": files, "verify_summary": summ,
           "verify_flags": {k: v for k, v in flags.items()}, "experiment_id": EXPERIMENT_ID, "pools": pools,
           "dev_test_overlap": disjoint, "duplicate_audit": audit, "indexes": idx}
    write_json(HERE / "inputs.json", res)
    print(json.dumps({"questions": files["test_v2run_all.jsonl"], "per_pool": {p: len(by[p]) for p in POOLS},
                      "verify_errors": 0, "flags": len(flags), "pools_ok": len(pools), "dev_test_overlap": disjoint,
                      "audit_clean": audit["clean"], "audit_hits": len(hits),
                      "indexes": {Path(k).name: (v if isinstance(v, str) else v["files"]) for k, v in idx.items()}}))


def manifest():
    head = subprocess.run(["git", "-C", WT, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "-C", WT, "status", "--porcelain", "--", "ragbench"], capture_output=True,
                           text=True).stdout.strip()
    if dirty:
        raise SystemExit("REFUSED: ragbench/ has uncommitted changes:\n" + dirty)
    branch = subprocess.run(["git", "-C", WT, "rev-parse", "--abbrev-ref", "HEAD"], capture_output=True,
                            text=True).stdout.strip()
    inp = json.loads((HERE / "inputs.json").read_text(encoding="utf-8"))
    sys.path.insert(0, str(DS / "lex7" / "v71" / "eval"))
    rp = json.loads((HERE / "reader_pins.json").read_text(encoding="utf-8"))   # dev pins + the 4 new readers
    pins = rp["pins"]
    jp = json.loads((HERE / "judge_pin.json").read_text(encoding="utf-8"))
    cal = HERE / "token_calibration.json"
    man = {
        "schema": "v2-test-protocol-manifest-2", "frozen_at": dt.datetime.now().isoformat(timespec="seconds"),
        "protocol": str(HERE / "PROTOCOL.md"), "code_worktree": WT, "code_branch": branch, "code_commit": head,
        "experiment": "v2-test-eval", "question_experiment_id": EXPERIMENT_ID,
        "question_files": {n: {"sha256": v["sha256"], "n": v["n"], "answerable": v["answerable"]}
                           for n, v in inp["converted"].items() if n.startswith("test_v2run_")},
        "judge_question_file": {"name": "test_v2judge_all.jsonl", **inp["converted"]["test_v2judge_all.jsonl"]},
        "frozen_questions": inp["frozen_questions"]["sha256"],
        "pools": {p: {"sha256": v["sha256"], "manifest_sha1": v["manifest_sha1"], "docs": v["docs"],
                      "corpus": v["corpus"], "setting": v["setting"]} for p, v in inp["pools"].items()},
        "indexes": inp["indexes"],
        "arms": [{"arm": a, "variant": v, "args": args, "env": env} for a, v, args, env in ARMS],
        "reader": READER, "reader_pins": pins, "reader_max_attempts": READER_MAX_ATTEMPTS,
        "readers": rp["readers"], "primary_reader": rp["primary"],
        "reader_pins_file": {"path": str(HERE / "reader_pins.json"), "sha256": sha256(HERE / "reader_pins.json")},
        "replay": {"tool": "ragbench.tools.reader_replay (reader-replay-v1)", "pack_env": "RAGBENCH_PACK_DIR",
                   "pack_schema": "answer-pack-v1", "attempts": READER_MAX_ATTEMPTS,
                   "rule": "retrieval, reranking and selector calls run once per question x arm (primary reader GLM); "
                           "the identical answer-call messages are replayed to the other readers"},
        "setr_selector": {"model": "z-ai/glm-5.3-flash", "pin": "via reader_pins", "attempts": 1, "deadline_s": 8.0,
                          "max_tokens_total": 2000, "reasoning_effort": "low", "top": 20, "zero_shot": True},
        "selector_pin_env": None,
        "selector": {"model": "z-ai/glm-5.3-flash", "pin": "via reader_pins", "attempts": 1, "deadline_s": 3.0,
                     "max_tokens_total": 600, "reasoning_effort": "low"},
        "judge": {"model": jp["judge"], "provider_pin": jp["provider_pin"], "routing": jp["routing"],
                  "max_judge_attempts": 3, "on_malformed": "adjudicate", "http_attempts": 3,
                  "rubric": "judge-multi v2 (strict)"},
        "token_calibration": {"path": str(cal), "sha256": sha256(cal)},
        "duplicate_audit_at_freeze": inp["duplicate_audit"]["clean"],
    }
    write_json(HERE / "MANIFEST.json", man)
    print(json.dumps({"code_commit": head, "branch": branch, "arms": [a[0] for a in ARMS],
                      "manifest_sha256": sha256(HERE / "MANIFEST.json")}))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    raise SystemExit({"inputs": inputs, "manifest": manifest, "audit": audit_only}[sys.argv[1]]())
