"""Fit LEX's window score from data instead of hand-setting it, and report it honestly (see approaches/lex/learn.py).

    dump    one LEX pass per dev record, writing every candidate window's feature vector and whether it is gold.
            Scoring and evidence are IDENTICAL to plain LEX (the model is None, the hand-set score is kept), so with
            the shared call cache every model call replays and the dump costs no GPU time.
    folds   k folds over QUESTION ids (never over windows: the windows of one question are dependent rows, and the
            fa and en record of one question must not be split). Each fold's model is fitted on the other folds,
            with its penalty chosen by an inner cross-validation on those folds only.
    report  the retrieval-level read, out of fold and with no model call: does the learned score put more gold
            windows in the top k than the hand-set score? Run this BEFORE spending GPU time on end-to-end arms.

    .venv/Scripts/python.exe -m ragbench.tools.lex_learn dump   --out work/runs/lex_learn
    .venv/Scripts/python.exe -m ragbench.tools.lex_learn folds  --out work/runs/lex_learn --k 5
    .venv/Scripts/python.exe -m ragbench.tools.lex_learn report --out work/runs/lex_learn

The end-to-end arm is then five runs of `LEX+learned`, one per fold, each with LEX_OPTIONS learned_model = that
fold's weights and --ids that fold's questions (core_plan.sh stage `lexlearn`).
"""
import argparse
import json
import os
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from ..approaches.lex import learn as LN
from ..approaches.lex.approach import LexAgent, _Session
from ..llm import ChatResult

ROOT = Path(__file__).resolve().parents[2]
PY = str(ROOT / ".venv" / "Scripts" / "python.exe")
PQ = "ragbench/data/questions/pilot.jsonl"
PT = "ragbench/data/questions/pilot_timeline.jsonl"
SHARED = "work/runs/lex_dev5/lex_calls_shared.sqlite"


class LexDump(LexAgent):
    """LEX, unchanged, plus a write of every candidate window's features and gold label."""
    name = "LEXDUMP"
    needs_gold = True
    collect_features = True

    def score_fn(self):
        return LN.scorer(None, keep=True)          # hand-set score kept; features collected

    def _answer(self, query, index):
        """The pipeline up to the windows, then a stub for call 3: the dump needs the candidate windows, not an
        answer, and an answer call over plain LEX's evidence is not in the shared cache (that arm ran with the cache
        off). Calls 1 and 2 are unchanged, so they replay."""
        s = _Session(self, query, index)
        real = s.chat

        def chat(purpose, messages, **kw):
            if purpose == "answer":
                return ChatResult(text="", prompt_tokens=0, completion_tokens=0, wall_ms=0.0, done_reason="dump",
                                  model=self.cfg.models["chat"], call_id="dump")
            return real(purpose, messages, **kw)

        s.chat = chat
        return s.run()

    def observe(self, q, wins, terms, kw, docs):
        path = self.opts.get("learn_dump")
        if not path:
            return
        gold = set(q.gold_unit_ids or [])
        rec = {"question_id": q.question_id, "setting": q.setting, "corpus": q.corpus, "d_lang": q.d_lang,
               "n_gold": len(gold), "docs": list(docs),
               "windows": [{"doc_id": w["doc_id"], "unit_ids": w["unit_ids"], "chars": len(w["text"] or ""),
                            "hand": w.get("hand_score"), "gold": bool(gold & set(w["unit_ids"])),
                            "feats": w.get("feats") or {}} for w in wins]}
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------------------------------------------
# dump
# ---------------------------------------------------------------------------------------------------------------

def cmd_dump(a):
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    dump = out / "windows.jsonl"
    if dump.exists() and not a.append:
        dump.unlink()
    opts = {"call_cache_path": str(ROOT / SHARED), "learn_dump": str(dump)}
    env = dict(os.environ, LEX_OPTIONS=json.dumps(opts))
    # the timeline file restricts corpus C to other documents, so its records go to their own directory (the runner
    # refuses to mix index fingerprints in one --out)
    blocks = [(PQ, "A", "fa-fa,en-en", ""), (PQ, "B", "fa-fa,en-en", ""), (PQ, "C", "fa-fa", ""),
              (PT, "C", "fa-fa", "_tl")]
    if a.block is not None:
        blocks = [blocks[a.block]]
    for qf, corpus, settings, suffix in blocks:
        cmd = [PY, "-m", "ragbench.run", "--approach-class", "ragbench.tools.lex_learn:LexDump",
               "--docs-from-questions", "--questions", qf, "--corpus", corpus, "--settings", settings,
               "--out", "runs/" + out.name + "_runs" + suffix, "--no-warmup", "--no-health"]
        print("::", " ".join(cmd[3:]), flush=True)
        r = subprocess.run(cmd, cwd=ROOT, env=env)
        if r.returncode:
            return r.returncode
    print("wrote", dump)
    return 0


def load(path):
    """-> (Dataset keyed by question id, [record]) from the dump."""
    data, recs = LN.Dataset(), []
    for line in Path(path).open(encoding="utf-8"):
        r = json.loads(line)
        recs.append(r)
        for i, w in enumerate(r["windows"]):
            w["row"] = len(data.rows)
            w["rec"] = len(recs) - 1
            data.add(r["question_id"], w["feats"], 1.0 if w["gold"] else 0.0)
    return data, recs


# ---------------------------------------------------------------------------------------------------------------
# folds
# ---------------------------------------------------------------------------------------------------------------

def fold_of(keys, k, seed=0):
    order = list(keys)
    np.random.RandomState(seed).shuffle(order)
    return {q: i % k for i, q in enumerate(order)}


def cmd_folds(a):
    out = Path(a.out)
    data, _recs = load(out / "windows.jsonl")
    keys = data.keys()
    assign = fold_of(keys, a.k, a.seed)
    (out / "folds.json").write_text(json.dumps({"k": a.k, "seed": a.seed, "assign": assign}, indent=1),
                                    encoding="utf-8")
    summary = []
    for i in range(a.k):
        tr = [q for q in keys if assign[q] != i]
        te = [q for q in keys if assign[q] == i]
        sub = data.subset(tr)
        lam, table = LN.choose_lam(sub, folds=a.inner, n_lam=a.n_lam, min_df=a.min_df, k=a.rank_k,
                                   seed=a.seed, rule=a.rule, phrases=not a.no_phrases)
        m = LN.fit(sub, lam, min_df=a.min_df, phrases=not a.no_phrases)
        m.meta.update(fold=i, train_questions=len(tr), test_questions=len(te), lam_path=table, rank_k=a.rank_k)
        m.save(out / ("model_fold%d.json" % i))
        (out / ("fold_ids_%d.txt" % i)).write_text(",".join(te), encoding="utf-8")   # core_plan.sh: --ids
        summary.append({"fold": i, "lam": lam, "p": m.meta["p"], "nonzero": m.meta["nonzero"],
                        "train_q": len(tr), "test_q": len(te), "rows": m.meta["n_rows"],
                        "positives": m.meta["positives"]})
        print("fold %d  lam %.5f  p %d  nonzero %d  train %d q  test %d q" %
              (i, lam, m.meta["p"], m.meta["nonzero"], len(tr), len(te)), flush=True)
    # one model on everything, for the record (NOT reportable on dev: it has seen every dev question)
    lam, _t = LN.choose_lam(data, folds=a.inner, n_lam=a.n_lam, min_df=a.min_df, k=a.rank_k, seed=a.seed,
                            rule=a.rule, phrases=not a.no_phrases)
    LN.fit(data, lam, min_df=a.min_df, phrases=not a.no_phrases).save(out / "model_all.json")
    (out / "folds_summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    return 0


# ---------------------------------------------------------------------------------------------------------------
# report (retrieval level, out of fold, no model call)
# ---------------------------------------------------------------------------------------------------------------

def _recall_at_k(rows, score, k):
    """Share of a question's gold windows that reach the top k by `score`."""
    gold = sum(1 for r in rows if r["gold"])
    if not gold:
        return None
    top = sorted(rows, key=lambda r: -score(r))[:k]
    return sum(1 for r in top if r["gold"]) / gold


def _chars_recall(rows, score, budget=6000):
    """The honest packing read: take windows best-first until the character budget is spent (what `select` does,
    without the overlap and reservation rules), and report the share of gold windows that got in."""
    gold = sum(1 for r in rows if r["gold"])
    if not gold:
        return None
    used, hit = 0, 0
    for r in sorted(rows, key=lambda r: -score(r)):
        c = r["chars"] + 80
        if used + c > budget:
            continue
        used += c
        hit += 1 if r["gold"] else 0
    return hit / gold


def cmd_report(a):
    out = Path(a.out)
    data, recs = load(out / "windows.jsonl")
    assign = json.loads((out / "folds.json").read_text(encoding="utf-8"))["assign"]
    models = {i: LN.Model.load(out / ("model_fold%d.json" % i)) for i in sorted(set(assign.values()))}
    hand = lambda r: r["hand"] or 0.0                                             # noqa: E731
    rows = {}
    for rec in recs:
        f = assign.get(rec["question_id"])
        if f is None:
            continue
        m = models[f]
        learned = lambda r: m.score(r["feats"])                                   # noqa: E731
        ws = rec["windows"]
        rows[(rec["question_id"], rec["setting"])] = {
            "n_windows": len(ws), "n_gold": sum(1 for w in ws if w["gold"]),
            "hand_k": _recall_at_k(ws, hand, a.rank_k), "learned_k": _recall_at_k(ws, learned, a.rank_k),
            "hand_budget": _chars_recall(ws, hand), "learned_budget": _chars_recall(ws, learned)}
    def mean(key):
        v = [r[key] for r in rows.values() if r[key] is not None]
        return sum(v) / len(v) if v else float("nan")
    n = sum(1 for r in rows.values() if r["hand_k"] is not None)
    better = sum(1 for r in rows.values() if r["learned_k"] is not None and r["learned_k"] > r["hand_k"])
    worse = sum(1 for r in rows.values() if r["learned_k"] is not None and r["learned_k"] < r["hand_k"])
    lines = ["# LEX learned window score - retrieval level, out of fold (no model call)", "",
             "records with gold among the candidate windows: %d of %d" % (n, len(rows)),
             "mean gold recall @ top %d      hand %.3f   learned %.3f" % (a.rank_k, mean("hand_k"), mean("learned_k")),
             "mean gold recall @ 6000 chars  hand %.3f   learned %.3f" % (mean("hand_budget"), mean("learned_budget")),
             "records where the learned score ranks gold better: %d, worse: %d, equal: %d" %
             (better, worse, n - better - worse), ""]
    summary = json.loads((out / "folds_summary.json").read_text(encoding="utf-8"))
    lines += ["| fold | lambda | p (features) | nonzero | train q | test q | training windows | gold windows |",
              "|---|---|---|---|---|---|---|---|"]
    for s in summary:
        lines.append("| %d | %.5f | %d | %d | %d | %d | %d | %d |" % (s["fold"], s["lam"], s["p"], s["nonzero"],
                                                                      s["train_q"], s["test_q"], s["rows"],
                                                                      s["positives"]))
    text = "\n".join(lines)
    (out / "RETRIEVAL.md").write_text(text + "\n", encoding="utf-8")
    (out / "per_record.json").write_text(json.dumps({"%s|%s" % k: v for k, v in rows.items()}, indent=1),
                                         encoding="utf-8")
    print(text)
    return 0


def main(argv=None):
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("dump", "folds", "report"):
        p = sub.add_parser(name)
        p.add_argument("--out", default="work/runs/lex_learn")
        if name == "dump":
            p.add_argument("--append", action="store_true")
            p.add_argument("--block", type=int, default=None, help="only this block (0-3)")
        if name in ("folds", "report"):
            p.add_argument("--rank-k", type=int, default=8, dest="rank_k")
        if name == "folds":
            p.add_argument("--k", type=int, default=5)
            p.add_argument("--inner", type=int, default=4)
            p.add_argument("--n-lam", type=int, default=10, dest="n_lam")
            p.add_argument("--min-df", type=int, default=2, dest="min_df")
            p.add_argument("--seed", type=int, default=0)
            p.add_argument("--rule", default="best", choices=["best", "1se"])
            p.add_argument("--no-phrases", action="store_true", dest="no_phrases",
                           help="generic features only: the low-dimensional model")
    a = ap.parse_args(argv)
    return {"dump": cmd_dump, "folds": cmd_folds, "report": cmd_report}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
