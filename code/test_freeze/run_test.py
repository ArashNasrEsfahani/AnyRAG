"""The ONE-SHOT datasets-v2 test runner (PROTOCOL.md, re-frozen 2026-10-01: amendment 1 = 8 arms x 6 readers).
The b31cd4c version is kept as run_test_b31cd4c.py. Every step is resumable and never redoes a finished cell.

    python run_test.py plan                          cell order + a --dry-run gate check of all 8 arms (no model call)
    python run_test.py run --confirm-one-shot        PRIMARY cells: 96 questions x 8 arms, reader GLM (interleaved,
                                                     rotated); retrieval / rerank / selector run here once; the answer
                                                     PACK of every cell is saved (RAGBENCH_PACK_DIR)
    python run_test.py replay --confirm-one-shot [--follow]
                                                     the 5 other readers answer the SAME saved packs (reader_replay);
                                                     --follow keeps polling while the primary run is still going
    python run_test.py judge --confirm-one-shot      judge every record (6 readers, one judge dir per reader, pinned)
    python run_test.py status                        counts + spend (also written to test_run/STATUS.json)

Rules (PROTOCOL.md sections 3 and 9): one `ragbench.run --test-protocol MANIFEST.json --ids <q>` process per primary
cell, one at a time; per question all 8 arms back to back, arm order rotated by the question's index (k = i mod 8);
a cell with an answer is never run again (primary: a non-error record in its runs file; replay: a non-error record
with the replay identity rid; judge: a verdict / needs_adjudication row with the same key); a cell whose record
ERRORED (no answer: infrastructure) may be re-run ONCE (logged in reruns.jsonl), a second error is a system failure
(scored 0, reported); a selector deadline / failure is a completed method execution, never a rerun reason.
Budget: stops before a cell when the shared reader file reaches START + STOP_ABOVE_START_USD, or when the OpenRouter
account has less than MIN_ACCOUNT_USD left (checked once per question). Resume = run the same command again.
Outputs: work/datasets_v2/test_run/ (runs/<pool>/<arm>/, packs/<variant>/, replay/<reader>/<pool>/<arm>/,
judge/<reader>/, answers.log, replay.log, reruns.jsonl, PROGRESS.md).
"""
import datetime as dt
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
DS = HERE.parent
WORK = DS.parent
TR = DS / "test_run"
MAN_PATH = HERE / "MANIFEST.json"
PY = r"D:\KareOne\RAG Paper\.venv\Scripts\python.exe"
FALLBACK_ENV = r"D:\KareOne\Holding Dashboard\app\.env"
HF_ENV = {"HF_HOME": r"D:\KareOne\RAG Paper\work\models\hf", "HF_HUB_OFFLINE": "1", "HF_HUB_DISABLE_SYMLINKS_WARNING": "1"}
BUDGET_FILE = WORK / "openrouter" / "reader_budget.json"
STOP_ABOVE_START_USD = 5.00        # answers + selectors + replays (estimate 3.4); the judge is guarded separately
MIN_ACCOUNT_USD = 1.50             # never drain the shared account below this
BELOW_NORMAL = 0x00004000
PACKS = TR / "packs"
REPLAY_SHARDS = {"deepseek/deepseek-v4-flash": 3}   # concurrency per reader (shards by arm); default REPLAY_DEFAULT_SHARDS
REPLAY_DEFAULT_SHARDS = 2
RESUME_CMD = ('cd "D:\\KareOne\\RAG Paper\\work\\datasets_v2\\test_freeze"; '
              '& "D:\\KareOne\\RAG Paper\\.venv\\Scripts\\python.exe" run_test.py run --confirm-one-shot')


def man():
    return json.loads(MAN_PATH.read_text(encoding="utf-8"))


def qids():
    rows = [json.loads(l) for l in (HERE / "questions" / "test_v2run_all.jsonl").read_text(encoding="utf-8")
            .splitlines() if l.strip()]
    return [r["id"] for r in rows]


def pool_of(q):
    return "-".join(q.split("-")[2:4])


def slug(m):
    return m.replace("/", "__").replace(":", "_")


def total():
    for _ in range(20):
        try:
            return float(json.loads(BUDGET_FILE.read_text(encoding="utf-8-sig"))["cost_usd"])
        except (OSError, ValueError):
            time.sleep(0.2)
    raise RuntimeError("budget file unreadable")


def account_remaining():
    """OpenRouter account credits left (shared account); None when the check fails. Never prints the key."""
    import requests
    try:
        key = next(l.partition("=")[2].strip().strip('"').strip("'") for l in open(FALLBACK_ENV, encoding="utf-8")
                   if l.partition("=")[0].strip() == "OPENROUTER_API_KEY")
        d = requests.get("https://openrouter.ai/api/v1/credits", headers={"Authorization": "Bearer " + key},
                         timeout=20).json()["data"]
        return float(d["total_credits"]) - float(d["total_usage"])
    except Exception:  # noqa: BLE001
        return None


def records(out, qid=None):
    rs = []
    for f in Path(out).glob("runs_*.jsonl"):
        with open(f, encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                r = json.loads(line)
                if qid is None or r.get("question_id") == qid:
                    rs.append(r)
    return rs


def say(m, log="answers.log"):
    line = "%s %s" % (dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), m)
    print(line, flush=True)
    TR.mkdir(parents=True, exist_ok=True)
    with open(TR / log, "a", encoding="utf-8") as f:
        f.write(line + "\n")
        f.flush()


def progress(m):
    with open(TR / "PROGRESS.md", "a", encoding="utf-8", newline="\n") as f:
        f.write("- %s %s\n" % (time.strftime("%Y-%m-%d %H:%M"), m))
        f.flush()


def cell_env(m, arm_env):
    env = dict(os.environ, OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2", MKL_NUM_THREADS="2",
               PYTHONIOENCODING="utf-8", OPENROUTER_PROVIDER_PINS=json.dumps(m["reader_pins"]),
               READER_MAX_ATTEMPTS=str(m["reader_max_attempts"]), READER_FALLBACK_ENV=FALLBACK_ENV,
               RAGBENCH_PACK_DIR=str(PACKS), **HF_ENV)
    env.pop("LEX5B_SELECTOR_PIN", None)
    for k in ("LEX8_IDX_DIR", "LEX8_SYNTH_DIR"):
        env.pop(k, None)
    env.update(arm_env)
    return env


def cell_cmd(m, arm, qid, out, dry=False):
    pool = pool_of(qid)
    p = m["pools"][pool]
    return [PY, "-m", "ragbench.run"] + arm["args"] + [
        "--questions", str(HERE / "questions" / ("test_v2run_%s.jsonl" % pool)), "--corpus", p["corpus"],
        "--settings", p["setting"], "--pool", pool, "--reader", m["reader"], "--experiment", m["experiment"],
        "--ids", qid, "--out", str(out), "--test-protocol", str(MAN_PATH)] + (["--dry-run"] if dry else [])


def schedule(m):
    arms = m["arms"]
    out = []
    for i, q in enumerate(qids()):
        k = i % len(arms)
        out.append((q, arms[k:] + arms[:k]))
    return out


def primary_state(arm, qid):
    """-> ('ok', record) | ('failed', None) (two errors) | ('error1', record) | ('todo', None)."""
    rs = records(TR / "runs" / pool_of(qid) / arm["arm"], qid)
    ok = [r for r in rs if not r.get("error")]
    if ok:
        return "ok", ok[-1]
    n_err = len(rs)
    return ("failed", None) if n_err >= 2 else ("error1", rs[-1]) if n_err == 1 else ("todo", None)


HANG_S = 420            # a primary cell normally takes 5-90 s; past this with no record it is an infrastructure hang
HANG_RETRIES = 2
HANG_RC = -999


def run_cell_watchdog(m, arm, qid, out):
    """Run one primary cell; kill its process tree if it has written no record after HANG_S seconds (operational
    watchdog, 2026-10-01: two LEX5B_15500 cells hung in index loading under GPU memory pressure). -> rc or HANG_RC."""
    with open(out / "run.log", "a", encoding="utf-8") as fh:
        fh.write("\n=== %s %s\n" % (dt.datetime.now().isoformat(timespec="seconds"), qid))
        fh.flush()
        p = subprocess.Popen(cell_cmd(m, arm, qid, out), cwd=m["code_worktree"], env=cell_env(m, arm["env"]),
                             stdout=fh, stderr=subprocess.STDOUT,
                             creationflags=BELOW_NORMAL if os.name == "nt" else 0)
        t0 = time.time()
        while p.poll() is None:
            time.sleep(2)
            if time.time() - t0 > HANG_S and primary_state(arm, qid)[0] != "ok":
                subprocess.run(["taskkill", "/PID", str(p.pid), "/T", "/F"], capture_output=True)
                p.wait()
                fh.write("\n[watchdog] killed after %d s with no record\n" % HANG_S)
                return HANG_RC
        return p.returncode


def plan():
    m = man()
    sch = schedule(m)
    print(json.dumps({"questions": len(sch), "primary_cells": sum(len(a) for _q, a in sch),
                      "readers": m["readers"], "replay_cells": sum(len(a) for _q, a in sch) * (len(m["readers"]) - 1),
                      "first_rotations": [[a["arm"] for a in arms] for _q, arms in sch[:8]]}))
    q0, arms = sch[0]
    bad = 0
    for arm in arms:
        r = subprocess.run(cell_cmd(m, arm, q0, HERE / "_plan_dry", dry=True), cwd=m["code_worktree"],
                           env=cell_env(m, arm["env"]), capture_output=True, text=True, encoding="utf-8",
                           errors="ignore")
        ok = "TEST PROTOCOL: matches" in (r.stdout + r.stderr) and r.returncode == 0
        exists = [l for l in (r.stdout + r.stderr).splitlines() if "complete=" in l]
        bad += not ok
        print(arm["arm"], "gate OK" if ok else "GATE REFUSED rc=%d" % r.returncode,
              [l for l in (r.stdout + r.stderr).splitlines() if "TEST PROTOCOL" in l][:3], [e[-60:] for e in exists])
    return 1 if bad else 0


def counts(m=None):
    m = m or man()
    sch = schedule(m)
    n_prim = sum(len(a) for _q, a in sch)
    prim = sum(1 for q, arms in sch for a in arms if primary_state(a, q)[0] == "ok")
    rep = {}
    for rd in m["readers"]:
        if rd == m["primary_reader"]:
            continue
        n = 0
        for d in (TR / "replay" / slug(rd)).glob("*/*"):
            n += sum(1 for r in records(d) if not r.get("error"))
        rep[rd] = n
    return {"primary_ok": prim, "primary_total": n_prim, "replay_ok": rep, "replay_total_each": n_prim,
            "reader_file_usd": round(total(), 4)}


def run(confirm):
    if not confirm:
        print("REFUSED: the one-shot test needs --confirm-one-shot")
        return 2
    m = man()
    TR.mkdir(parents=True, exist_ok=True)
    sf = TR / "budget_start.json"
    if not sf.exists():
        # first start: the raw-record audit must be clean (no V2-test- id anywhere under work/)
        sys.path.insert(0, str(HERE))
        import freeze_test as FT
        a = FT.duplicate_audit()
        (TR / "duplicate_audit_at_start.json").write_text(json.dumps(a, ensure_ascii=False, indent=1) + "\n",
                                                           encoding="utf-8")
        if not a["clean"]:
            print("REFUSED: test ids already in raw records:", list(a["files_with_test_ids"])[:5])
            return 2
        sf.write_text(json.dumps({"start_usd": total(), "account_remaining_usd": account_remaining(),
                                  "at": dt.datetime.now().isoformat(timespec="seconds"),
                                  "manifest_code_commit": m["code_commit"]}), encoding="utf-8")
    stop = json.loads(sf.read_text(encoding="utf-8"))["start_usd"] + STOP_ABOVE_START_USD
    sch = schedule(m)
    for qi, (qid, arms) in enumerate(sch):
        states = [primary_state(a, qid)[0] for a in arms]
        if all(s in ("ok", "failed") for s in states):
            continue
        acct = account_remaining()
        if acct is not None and acct < MIN_ACCOUNT_USD:
            say("ACCOUNT STOP: OpenRouter credits left %.2f < %.2f" % (acct, MIN_ACCOUNT_USD))
            return 1
        for arm in arms:
            st, rec = primary_state(arm, qid)
            if st in ("ok", "failed"):
                continue                                         # answered (never again) / system failure
            if total() >= stop:
                say("BUDGET STOP file total %.4f (stop %.4f)" % (total(), stop))
                return 1
            if st == "error1":
                with open(TR / "reruns.jsonl", "a", encoding="utf-8") as fh:
                    fh.write(json.dumps({"at": dt.datetime.now().isoformat(timespec="seconds"), "question_id": qid,
                                         "arm": arm["arm"], "reader": m["reader"],
                                         "error": str(rec.get("error"))[:300]}) + "\n")
            out = TR / "runs" / pool_of(qid) / arm["arm"]
            out.mkdir(parents=True, exist_ok=True)
            for hang_try in range(1, HANG_RETRIES + 2):
                rc = run_cell_watchdog(m, arm, qid, out)
                if rc != HANG_RC:
                    break
                with open(TR / "reruns.jsonl", "a", encoding="utf-8") as fh:
                    fh.write(json.dumps({"at": dt.datetime.now().isoformat(timespec="seconds"), "question_id": qid,
                                         "arm": arm["arm"], "reader": m["reader"], "kind": "infrastructure: cell "
                                         "process hung > %d s with no record (killed; no answer existed)" % HANG_S,
                                         "try": hang_try}) + "\n")
                say("%s %s HUNG > %d s: killed (try %d)" % (qid, arm["arm"], HANG_S, hang_try))
            st2, rec2 = primary_state(arm, qid)
            say("%s %s rc=%d %s %s | file total %.4f" % (qid, arm["arm"], rc, st2,
                                                         ((rec2 or {}).get("answer_text") or "")[:60].replace("\n", " "),
                                                         total()))
            if rc == 2:
                say("STOP: the run refused (gate / resume rule) - see %s" % (out / "run.log"))
                return 2
        c = counts(m)
        progress("primary q %d/%d %s: cells %d/%d; replays %s; reader file $%.4f (start $%.4f). Resume: %s" % (
            qi + 1, len(sch), qid, c["primary_ok"], c["primary_total"],
            {k.split("/")[1][:12]: v for k, v in c["replay_ok"].items()}, c["reader_file_usd"],
            stop - STOP_ABOVE_START_USD, RESUME_CMD))
        time.sleep(2)                                           # pacing between questions, outside every timed cell
    say("END primary")
    progress("PRIMARY RUN COMPLETE: %s" % json.dumps(counts(m)))
    return 0


# ---------------------------------------------------------------------------------------------------------------
# replay: the other readers answer the identical saved packs
# ---------------------------------------------------------------------------------------------------------------
_STOP = threading.Event()


def _replay_worker(m, reader, shard, n_shards, stop_usd, follow, stats):
    sys.path.insert(0, m["code_worktree"])
    from ragbench.config import load_config
    from ragbench.llm import BudgetExceeded, Ledger
    from ragbench.tools import reader_replay as RR
    from ragbench.util import safe_name
    cfg = load_config()
    ledger = Ledger(path=None)
    client = RR.make_reader(cfg, ledger, "openrouter:" + reader, int(m["reader_max_attempts"]))
    arm_names = [a["arm"] for a in m["arms"]]
    mine = {a for i, a in enumerate(arm_names) if i % n_shards == shard}
    while not _STOP.is_set():
        pending = 0
        for qid, arms in schedule(m):
            for arm in arms:
                if arm["arm"] not in mine or _STOP.is_set():
                    continue
                st, prim = primary_state(arm, qid)
                if st == "failed":
                    continue
                if st != "ok":
                    pending += 1
                    continue
                pk = PACKS / safe_name(prim["approach"]) / ("%s__%s.json" % (qid, prim["setting"]))
                if not pk.exists():
                    stats["missing_pack"] = stats.get("missing_pack", 0) + 1
                    continue
                out = TR / "replay" / slug(reader) / pool_of(qid) / arm["arm"]
                rid = RR.replay_identity(prim, "openrouter/" + reader)["rid"]
                ok, errs = RR.done_rids(out, prim["approach"])
                if rid in ok or errs.get(rid, 0) >= 2:
                    continue
                if total() >= stop_usd:
                    say("REPLAY BUDGET STOP %s file total %.4f" % (reader, total()), "replay.log")
                    _STOP.set()
                    return
                if errs.get(rid, 0) == 1:
                    with open(TR / "reruns.jsonl", "a", encoding="utf-8") as fh:
                        fh.write(json.dumps({"at": dt.datetime.now().isoformat(timespec="seconds"),
                                             "question_id": qid, "arm": arm["arm"], "reader": reader,
                                             "kind": "replay"}) + "\n")
                out.mkdir(parents=True, exist_ok=True)
                ledger.path = str(out / ("ledger_%s.jsonl" % safe_name(prim["approach"])))
                try:
                    rec = RR.replay_one(cfg, pk, prim, "openrouter:" + reader, out, client=client, ledger=ledger)
                except BudgetExceeded as e:
                    say("REPLAY BUDGET GUARD %s: %s" % (reader, e), "replay.log")
                    _STOP.set()
                    return
                stats["done"] = stats.get("done", 0) + 1
                say("%s %s %s %s %.1fs %s" % (reader.split("/")[1][:14], qid, arm["arm"],
                                              "ERROR " + rec["error"][:80] if rec.get("error") else "ok",
                                              rec["latency_ms"] / 1000.0,
                                              (rec.get("answer_text") or "")[:40].replace("\n", " ")), "replay.log")
        if not follow or pending == 0:
            return
        time.sleep(30)


def replay(confirm, follow):
    if not confirm:
        print("REFUSED: the replay needs --confirm-one-shot")
        return 2
    m = man()
    os.environ["OPENROUTER_PROVIDER_PINS"] = json.dumps(m["reader_pins"])
    os.environ["READER_FALLBACK_ENV"] = FALLBACK_ENV
    sf = TR / "budget_start.json"
    if not sf.exists():
        print("REFUSED: start the primary run first")
        return 2
    stop = json.loads(sf.read_text(encoding="utf-8"))["start_usd"] + STOP_ABOVE_START_USD
    threads, stats = [], {}
    for rd in m["readers"]:
        if rd == m["primary_reader"]:
            continue
        n = REPLAY_SHARDS.get(rd, REPLAY_DEFAULT_SHARDS)
        for s in range(n):
            st = stats.setdefault("%s#%d" % (rd, s), {})
            t = threading.Thread(target=_replay_worker, args=(m, rd, s, n, stop, follow, st), daemon=True,
                                 name="%s#%d" % (rd, s))
            t.start()
            threads.append(t)
    last = time.time()
    while any(t.is_alive() for t in threads):
        time.sleep(5)
        if time.time() - last > 600:
            last = time.time()
            c = counts(m)
            progress("replay: %s; reader file $%.4f" % ({k.split("/")[1][:12]: v for k, v in c["replay_ok"].items()},
                                                        c["reader_file_usd"]))
    c = counts(m)
    progress("replay pass ended: %s" % json.dumps(c))
    say("replay pass ended: %s %s" % (json.dumps(c), stats), "replay.log")
    return 1 if _STOP.is_set() else 0


# ---------------------------------------------------------------------------------------------------------------
# judge: one judge_multi process per reader (own out dir, own verdict file), run in parallel
# ---------------------------------------------------------------------------------------------------------------
def judge(confirm, only=None):
    if not confirm:
        print("REFUSED: judging the test needs --confirm-one-shot")
        return 2
    m = man()
    j = m["judge"]
    env = dict(os.environ, PYTHONIOENCODING="utf-8", OMP_NUM_THREADS="2", READER_FALLBACK_ENV=FALLBACK_ENV)
    env.pop("OPENROUTER_PROVIDER_PINS", None)
    procs = []
    for rd in m["readers"]:
        if only and rd not in only:
            continue
        base = TR / "runs" if rd == m["primary_reader"] else TR / "replay" / slug(rd)
        dirs = [d for d in sorted(base.glob("*/*")) if d.is_dir()]
        if not dirs:
            continue
        out = TR / "judge" / slug(rd)
        out.mkdir(parents=True, exist_ok=True)
        args = [PY, "-m", "ragbench.tools.judge_multi", "--judge", j["model"], "--judge-provider", j["provider_pin"],
                "--require-pin", "--max-judge-attempts", str(j["max_judge_attempts"]), "--on-malformed",
                j["on_malformed"], "--http-attempts", str(j["http_attempts"]),
                "--questions", str(HERE / "questions" / m["judge_question_file"]["name"]), "--out", str(out)]
        for d in dirs:
            args += ["--runs", str(d)]
        fh = open(TR / ("judge_%s.log" % slug(rd)), "a", encoding="utf-8")
        procs.append((rd, subprocess.Popen(args, cwd=m["code_worktree"], env=env, stdout=fh, stderr=subprocess.STDOUT,
                                           creationflags=BELOW_NORMAL if os.name == "nt" else 0), fh))
    rcs = {}
    for rd, p, fh in procs:
        rcs[rd] = p.wait()
        fh.close()
    say("judge rc=%s" % rcs)
    progress("judge pass ended rc=%s" % rcs)
    return max(rcs.values()) if rcs else 0


def judge_follow(confirm, every_s=300):
    """Operational speed-up (2026-10-01, PROTOCOL.md section 9 note): judge finished cells while answering continues.
    Each pass is an ordinary resumable judge pass (judge_multi skips every key already judged / pending); the loop ends
    after a final pass once every primary and replay cell is answered or a system failure."""
    if not confirm:
        print("REFUSED: judging the test needs --confirm-one-shot")
        return 2
    m = man()
    while True:
        c = counts(m)
        sch = schedule(m)
        prim_done = all(primary_state(a, q)[0] in ("ok", "failed") for q, arms in sch for a in arms)
        rep_done = all(v >= c["primary_ok"] for v in c["replay_ok"].values())
        rc = judge(True)
        if prim_done and rep_done:
            progress("judge-follow: final pass done rc=%s" % rc)
            return rc
        time.sleep(every_s)


def status():
    c = counts()
    c["account_remaining_usd"] = account_remaining()
    (TR / "STATUS.json").write_text(json.dumps(c, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(c))
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    c = sys.argv[1]
    conf = "--confirm-one-shot" in sys.argv[2:]
    if c == "plan":
        raise SystemExit(plan())
    if c == "run":
        raise SystemExit(run(conf))
    if c == "replay":
        raise SystemExit(replay(conf, "--follow" in sys.argv[2:]))
    if c == "judge" and "--follow" in sys.argv[2:]:
        raise SystemExit(judge_follow(conf))
    if c == "judge":
        only = [a for a in sys.argv[2:] if not a.startswith("--")]
        raise SystemExit(judge(conf, only or None))
    if c == "status":
        raise SystemExit(status())
    raise SystemExit(2)
