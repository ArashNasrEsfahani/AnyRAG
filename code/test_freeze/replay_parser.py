"""FREEZE fix (c): replay EVERY saved LEX5/LEX5B selector reply through the old parser (commit 02d0e37,
bundles_old_02d0e37.py, byte copy of `git show 02d0e37:ragbench/approaches/lex5/bundles.py`) and the new one
(ragbench/approaches/lex5/bundles.py v1.1: duplicate alternatives compared as unordered sets). Offline, no model call.

Sources: every runs_*.jsonl under work/datasets_v2 (lex5b, budget_matrix, dev60 are required; every other directory that
holds selector traces is replayed too) - the trace's selector block carries raw (<= 2,000 chars), finish and shown (n).
Also the selector-shaped replies of the two provider probes (lex5_pilot/providers.json, dev60/providers_probe.json;
n = 24 shown, as in those probes).
A record whose selector call returned no text (deadline / call error) has nothing to parse and is counted separately.
-> parser_replay.json (counts per directory, every difference listed) ; exit 1 if any difference.
"""
import glob
import importlib.util
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
DS = HERE.parent
sys.path.insert(0, r"D:\KareOne\RAG-Paper-upgrades")
from ragbench.approaches.lex5 import bundles as NEW  # noqa: E402

spec = importlib.util.spec_from_file_location("bundles_old", HERE / "bundles_old_02d0e37.py")
OLD = importlib.util.module_from_spec(spec)
spec.loader.exec_module(OLD)


def selectors(obj):
    """Yield every selector block under a trace (keys shown, status, model; raw is absent when no text came back)."""
    if isinstance(obj, dict):
        if "shown" in obj and "status" in obj and "model" in obj:
            yield obj
        for v in obj.values():
            yield from selectors(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from selectors(v)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    per = defaultdict(Counter)
    diffs, truncated_raw = [], 0
    for f in sorted(glob.glob(str(DS / "**" / "runs_*.jsonl"), recursive=True)):
        top = Path(f).relative_to(DS).parts[0]
        for ln, line in enumerate(open(f, encoding="utf-8"), 1):
            r = json.loads(line)
            for s in selectors(r.get("trace") or {}):
                raw, n, fin = s.get("raw"), s.get("shown"), s.get("finish")
                if raw is None:
                    per[top]["no_text"] += 1
                    continue
                if len(raw) >= 2000:
                    truncated_raw += 1
                a, b = OLD.parse(raw, n, finish_reason=fin), NEW.parse(raw, n, finish_reason=fin)
                per[top]["replies"] += 1
                per[top]["status_" + str(b["status"])] += 1
                if a != b:
                    per[top]["differ"] += 1
                    stg = next((v for v in (r.get("trace") or {}).values()
                                if isinstance(v, dict) and "shown_chunks" in v), {})
                    sh, fb = stg.get("shown_chunks"), stg.get("fallback_chunks")
                    diffs.append({"file": str(Path(f).relative_to(DS)), "line": ln, "question_id": r.get("question_id"),
                                  "raw": raw, "old": {k: a[k] for k in ("status", "reason")},
                                  "new": {k: b[k] for k in ("status", "reason")},
                                  "under_new_parser": "fallback pack (selector reply rejected)",
                                  "saved_shown_chunks": sh, "fallback_chunks": fb,
                                  "fallback_same_chunk_set": (sh is not None and fb is not None and set(sh) == set(fb)),
                                  "fallback_identical_order": sh == fb if sh is not None else None,
                                  "saved_evidence_chars": stg.get("evidence_chars")})
                else:
                    per[top]["identical"] += 1
                # the saved trace's own status must equal the old parser (sanity of the replay itself)
                if s.get("status") is not None and s.get("status") != a["status"]:
                    per[top]["saved_status_mismatch_old"] += 1
    for name, rel in (("probe:lex5_pilot", "lex5_pilot/providers.json"), ("probe:dev60", "dev60/providers_probe.json")):
        p = DS / rel
        if not p.exists():
            continue
        d = json.loads(p.read_text(encoding="utf-8"))
        for c in d.get("calls") or []:
            raw = c.get("text")
            if raw is None:
                per[name]["no_text"] += 1
                continue
            a, b = OLD.parse(raw, 24, finish_reason=c.get("finish")), NEW.parse(raw, 24, finish_reason=c.get("finish"))
            per[name]["replies"] += 1
            if a != b:
                per[name]["differ"] += 1
                diffs.append({"file": rel, "probe_call": c.get("pin"), "old": a["status"], "new": b["status"]})
            else:
                per[name]["identical"] += 1
    tot = Counter()
    for c in per.values():
        tot.update(c)
    out = {"old_parser": "02d0e37 lex5-bundles-v1", "new_parser": NEW.BUNDLES_VERSION,
           "required_dirs_present": {d: per[d]["replies"] for d in ("lex5b", "budget_matrix", "dev60")},
           "per_source": {k: dict(v) for k, v in sorted(per.items())}, "total": dict(tot),
           "raw_at_2000_char_cap": truncated_raw, "differences": diffs,
           "verdict": "RESULTS-NEUTRAL: every saved reply parses identically" if not diffs else
                      "%d DIFFERENCE(S) listed" % len(diffs)}
    (HERE / "parser_replay.json").write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({k: out[k] for k in ("required_dirs_present", "total", "raw_at_2000_char_cap", "verdict")}))
    return 1 if diffs else 0


if __name__ == "__main__":
    raise SystemExit(main())
