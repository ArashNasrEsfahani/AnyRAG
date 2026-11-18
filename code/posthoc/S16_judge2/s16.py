"""S16: judge 2 (mistralai/mistral-large-2512, pinned mistral/zdr, no fallbacks, require_parameters, temperature 0 /
top_p 1 / seed 42) on post-hoc variants that only judge 1 has scored. Same client and call path as S8 step 4
(judge2_posthoc.py, copied here; edits: stream tag S16, cap $0.90, 12 transport attempts by default) and S1
(j2lib.py, copied here; edit: reads S1_evaluator/judge2 shards). Verdicts through the frozen judge_multi.judge_records
(3 judge attempts, on_malformed adjudicate).

Records = exactly those judge 1 scored: every judge-1 verdict / needs_adjudication row of the variant gives the rid,
identity, setting and answer_sha1; the answer text is the stream's saved answer with that sha1 (asserted), or for
identity-run streams (S9, S11) the run record itself. Cells a stream reused from a frozen arm (source frozen:<arm>)
take S1's judge-2 verdict of that frozen answer.

    python s16.py --variants A,B [--limit-q 3] [--retry-failures]
Out: judge2/<VARIANT>/verdicts_*.jsonl, needs_adjudication_*.jsonl, cache.sqlite; ledger_j2.jsonl; money_checks.jsonl.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import judge2_posthoc as J2P  # noqa: E402  (copied S8 client; sets sys.path for report_test / ragbench)
import report_test as RT  # noqa: E402
from ragbench.llm import BudgetExceeded  # noqa: E402
from ragbench.tools import judge_multi as JM  # noqa: E402
from ragbench.util import KVCache, sha1_text  # noqa: E402

PH = J2P.PH
J1SLUG = "openai__gpt-oss-120b"
J1PIN = "deepinfra/bf16"
OUT = HERE / "judge2"
STALL_S = 20 * 60

# variant -> (judge-1 dir, method in the judge-1 rows, answer source)
VARIANTS = {
    "HYBRID_AGENT": (PH / "S9_gpu/judge/HYBRID_AGENT/test", "HYBRID_AGENT", ("runs", PH / "S9_gpu/runs/HYBRID_AGENT/test")),
    "DENSE_6000": (PH / "S10_retrieval/judge/DENSE_6000", "DENSE_6000", ("file", PH / "S10_retrieval/answers.jsonl")),
    "BM25_6000": (PH / "S10_retrieval/judge/BM25_6000", "BM25_6000", ("file", PH / "S10_retrieval/answers.jsonl")),
    "HYDE_6000": (PH / "S11_rewrite/judge/HYDE_6000/test", "HYDE_6000", ("runs", PH / "S11_rewrite/runs/HYDE_6000/test")),
    "MQ_6000": (PH / "S11_rewrite/judge/MQ_6000/test", "MQ_6000", ("runs", PH / "S11_rewrite/runs/MQ_6000/test")),
    "HYBRID_C500": (PH / "S12_chunks/judge", "HYBRID_C500_6000", ("file", PH / "S12_chunks/answers.jsonl")),
    "HR_C500": (PH / "S12_chunks/judge", "HR_C500_6000", ("file", PH / "S12_chunks/answers.jsonl")),
    "HYBRID_C2000": (PH / "S12_chunks/judge", "HYBRID_C2000_6000", ("file", PH / "S12_chunks/answers.jsonl")),
    "HR_C2000": (PH / "S12_chunks/judge", "HR_C2000_6000", ("file", PH / "S12_chunks/answers.jsonl")),
    "E5_DENSE": (PH / "S13_embed/judge/test", "E5_DENSE", ("file", PH / "S13_embed/answers_test.jsonl")),
    "E5_HYBRID": (PH / "S13_embed/judge/test", "E5_HYBRID", ("file", PH / "S13_embed/answers_test.jsonl")),
    "HR_6000_TRANS": (PH / "S15_crossling/judge", "HR_6000_TRANS", ("file", PH / "S15_crossling/answers.jsonl")),
}
ORDER = list(VARIANTS)


def jl(p):
    p = Path(p)
    return [json.loads(l) for l in open(p, encoding="utf-8-sig") if l.strip()] if p.is_file() else []


def j1_rows(variant):
    """{rid: first judge-1 row (verdict before needs_adjudication; first valid verdict is final)} for this method."""
    d, method, _ = VARIANTS[variant]
    out = {}
    for kind in ("verdicts", "needs_adjudication"):
        for v in jl(d / ("%s_%s.jsonl" % (kind, J1SLUG))):
            if v.get("method") == method and v.get("judge_provider_pin") == J1PIN:
                out.setdefault(v["rid"], v)
    return out


def frozen_cells(variant):
    """{qid: frozen arm} for cells the stream reused from a frozen arm (no own judge-1 row)."""
    _, method, (kind, src) = VARIANTS[variant]
    out = {}
    if kind == "file":
        for r in jl(src):
            if r.get("arm") == method and str(r.get("source") or "").startswith("frozen:"):
                out[r["qid"]] = r["source"].split(":", 1)[1]
    return out


def records(variant):
    """-> ({rid: record}, {qid: ('rec', rid, answer_sha1) | ('frozen', arm)})."""
    _, method, (kind, src) = VARIANTS[variant]
    rows = j1_rows(variant)
    recs, cell = {}, {}
    if kind == "runs":
        rr, _ = JM.latest_identity_records([src])
        for rid, v in rows.items():
            r = rr[rid]
            assert JM.answer_sha1(r) == v["answer_sha1"], (variant, rid)
            assert r["setting"] == v["setting"] and r["question_id"] == v["question_id"]
            recs[rid] = r
    else:
        texts = {}
        for r in jl(src):
            if r.get("arm") == method and not r.get("error"):
                texts.setdefault(sha1_text(r.get("answer_text") or "")[:16], r["answer_text"])
        for rid, v in rows.items():
            ident = v["identity"]
            assert ident["rid"] == rid
            recs[rid] = {"question_id": v["question_id"], "approach": method, "setting": v["setting"],
                         "answer_text": texts[v["answer_sha1"]], "error": None, "trace": {},
                         "meta": {"identity": ident}}
            assert JM.answer_sha1(recs[rid]) == v["answer_sha1"]
    for rid, r in recs.items():
        assert r["question_id"] not in cell, (variant, r["question_id"])
        cell[r["question_id"]] = ("rec", rid, JM.answer_sha1(r))
    for q, arm in frozen_cells(variant).items():
        assert q not in cell
        cell[q] = ("frozen", arm)
    return recs, cell


class Stalled(BudgetExceeded):
    pass


class Client(J2P.Client):
    """S8/S1 client plus the brief's stop rule: no successful call for more than 20 minutes -> stop."""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.last_ok = time.time()

    def chat(self, messages, **kw):
        if time.time() - self.last_ok > STALL_S:
            raise Stalled("provider refused for more than 20 minutes")
        try:
            res = super().chat(messages, **kw)
        except RuntimeError:
            if time.time() - self.last_ok > STALL_S:
                raise Stalled("provider refused for more than 20 minutes")
            raise
        self.last_ok = time.time()
        return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", default=",".join(ORDER))
    ap.add_argument("--limit-q", type=int, default=None)
    ap.add_argument("--retry-failures", action="store_true")
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    qs = JM.load_question_dicts(J2P.QFILE)
    client = Client(J2P.JUDGE2, J2P.PIN2, J2P.CAP_USD)
    qset = None
    if a.limit_q:
        seen = {}
        for q in RT.ANS:
            seen.setdefault(RT.QS[q]["domain"], q)
        qset = set(list(seen.values())[:a.limit_q])
    for var in a.variants.split(","):
        recs, cell = records(var)
        if qset:
            recs = {k: v for k, v in recs.items() if v["question_id"] in qset}
        out = OUT / var
        print(var, "records", len(recs), "frozen cells", sum(c[0] == "frozen" for c in cell.values()),
              "spent $%.4f" % J2P.stream_spent(), flush=True)
        try:
            items = sorted(recs.items())
            for i in range(0, len(items), 15):        # <= 45 judge calls per chunk; floor check every 50 calls
                JM.judge_records(dict(items[i:i + 15]), qs, client, J2P.JUDGE2, out, cache=KVCache(out / "cache.sqlite"),
                                 provider_pin=J2P.PIN2, max_attempts=3, on_malformed="adjudicate",
                                 retry_failures=a.retry_failures, log=lambda s: print(var, s, flush=True))
        except Stalled as e:
            print(var, "STALL STOP", e, flush=True)
            return 4
        except BudgetExceeded as e:
            print(var, "BUDGET STOP", e, flush=True)
            return 3
    print("done; spent $%.4f" % J2P.stream_spent(), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
