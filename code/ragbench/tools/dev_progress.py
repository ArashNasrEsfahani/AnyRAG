"""Progress of dev_full: ledger calls per variant/phase, recent tokens/s, steps finished. Appends to logs/progress.log."""
import json, sys, time, datetime
from pathlib import Path
sys.stdout.reconfigure(encoding="utf-8")
OUT = Path("D:/KareOne/RAG Paper/work/runs/dev_full")
lines = []
now = datetime.datetime.now()
lines.append("---- %s" % now.strftime("%F %T"))
for f in sorted(OUT.glob("ledger_*.jsonl")):
    recs = []
    with open(f, encoding="utf-8") as fh:
        for l in fh:
            try:
                recs.append(json.loads(l))
            except Exception:
                pass
    for phase in ("index", "query"):
        rs = [r for r in recs if r.get("phase") == phase]
        if not rs:
            continue
        chat = [r for r in rs if r.get("kind") == "chat"]
        emb = [r for r in rs if r.get("kind") == "embed"]
        last = chat[-50:]
        ev_ms = sum(r.get("eval_ms") or 0 for r in last)
        ctok = sum(r.get("completion_tokens") or 0 for r in last)
        wall = sum(r.get("wall_ms") or 0 for r in last)
        tps = ctok / (ev_ms / 1000.0) if ev_ms else 0
        sets = {}
        for r in chat:
            sets[r.get("corpus_set")] = sets.get(r.get("corpus_set"), 0) + 1
        lines.append("%-40s %-5s chat=%d embed=%d fails=%d | last50: %.1f tok/s, %.1f s/call | per set %s | last %s" % (
            f.stem.replace("ledger_", ""), phase, len(chat), len(emb), sum(1 for r in rs if not r.get("ok")), tps,
            (wall / len(last) / 1000.0) if last else 0, sets, rs[-1].get("ts")))
steps = OUT / "logs" / "steps.tsv"
if steps.is_file():
    st = steps.read_text(encoding="utf-8").strip().split("\n")
    lines.append("steps done: %d; last: %s" % (len(st), st[-1] if st else ""))
for lf in sorted((OUT / "logs").glob("index_ours*.log")):
    t = [x for x in lf.read_text(encoding="utf-8", errors="replace").split("\n") if "[OURS index]" in x]
    if t:
        lines.append("%s: %s" % (lf.name, t[-1]))
msg = "\n".join(lines)
print(msg)
with open(OUT / "logs" / "progress.log", "a", encoding="utf-8") as fh:
    fh.write(msg + "\n")
