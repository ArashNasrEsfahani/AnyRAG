"""Offline tests for the multi-reader test plumbing (2026-10-01): the answer pack written by run.save_answer_pack, the
replay of that identical pack to another reader (tools/reader_replay.py) and the SETR 15,500 cell. No model call."""
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

from ragbench.config import load_config
from ragbench.llm import ChatResult, Ledger
from ragbench import run as R
from ragbench.tools import reader_replay as RR

MSGS = [{"role": "system", "content": "You answer questions ... Answer in Persian."},
        {"role": "user", "content": "Evidence:\n[1] متن نمونه 12\n\nQuestion: چند؟"}]


def _primary(tmp, h):
    ident = R.record_identity("v2-test-eval", "V2-test-X-fa-T1-001", "RPT-fa", "HYBRID", {}, "variants-v2",
                              SimpleNamespace(models={"chat": "openrouter/z-ai/glm-5.3-flash",
                                                      "chat_digest": "openrouter"}), 0, "fa-fa")
    ident.update(code_sig="c0de", question_sig="q5ig", pool_sha1="p001")
    return {"question_id": "V2-test-X-fa-T1-001", "approach": "HYBRID", "setting": "fa-fa", "q_lang": "fa",
            "corpus_set": "C_fa", "answer_text": "۱۲", "abstained": False, "retrieved_units": [{"u": 1}],
            "evidence_units": ["u1"], "evidence_items": [{"doc_id": "d"}], "evidence_chars": 42, "llm_calls": 1,
            "prompt_tokens": 100, "completion_tokens": 9, "latency_ms": 1500.0, "error": None,
            "trace": {"hybrid": {}}, "models": {"chat": "openrouter/z-ai/glm-5.3-flash", "embed": "bge-m3"},
            "meta": {"identity": ident, "answer_messages_sha256": h}}


class _Client:
    provider = ["deepinfra/fp8"]

    def __init__(self, text="12 units", fail=False):
        self.text, self.fail, self.seen = text, fail, None

    def chat(self, messages, purpose=None, num_predict=None, **kw):
        self.seen = (messages, purpose, num_predict)
        if self.fail:
            raise RuntimeError("all attempts failed")
        r = ChatResult(text=self.text, prompt_tokens=0, completion_tokens=0, wall_ms=5.0, done_reason="stop")
        r.provider = "DeepInfra"
        return r


def test_save_pack_and_sha():
    with tempfile.TemporaryDirectory() as d:
        ap = SimpleNamespace(_answer_call={"messages": MSGS, "num_predict": 256, "purpose": "answer"})
        rec = SimpleNamespace(error=None, meta={"identity": {"rid": "r1"}}, approach="HYBRID+rerank",
                              question_id="Q1", setting="fa-fa", q_lang="fa", evidence_chars=42)
        p = R.save_answer_pack(ap, rec, d)
        pk = json.loads(Path(p).read_text(encoding="utf-8"))
        assert pk["messages"] == MSGS and pk["num_predict"] == 256 and pk["rid"] == "r1"
        assert pk["messages_sha256"] == rec.meta["answer_messages_sha256"] == R.answer_messages_sha256(MSGS)
        assert Path(p).parent.name == "HYBRID_rerank" or "rerank" in Path(p).parent.name
        # no pack dir: only the sha is stamped; an errored record or no answer call: nothing
        rec2 = SimpleNamespace(error=None, meta={}, approach="A", question_id="Q", setting="s", q_lang="fa",
                               evidence_chars=0)
        assert R.save_answer_pack(ap, rec2, None) is None and "answer_messages_sha256" in rec2.meta
        rec3 = SimpleNamespace(error="x", meta={}, approach="A")
        assert R.save_answer_pack(ap, rec3, d) is None and rec3.meta == {}
        assert R.save_answer_pack(SimpleNamespace(_answer_call=None), rec2, d) is None


def test_replay_record():
    cfg = load_config()
    h = R.answer_messages_sha256(MSGS)
    with tempfile.TemporaryDirectory() as d:
        prim = _primary(d, h)
        pp = Path(d) / "pack.json"
        pp.write_text(json.dumps({"question_id": prim["question_id"], "variant": "HYBRID", "messages": MSGS,
                                  "num_predict": 256, "purpose": "answer", "messages_sha256": h}), encoding="utf-8")
        c = _Client("12 units")
        rec = RR.replay_one(cfg, pp, prim, "openrouter:google/gemma-3-27b-it", Path(d) / "out", client=c,
                            ledger=Ledger(path=None))
        assert c.seen == (MSGS, "answer", 256)                       # identical messages and allowance
        assert rec["answer_text"] == "12 units" and rec["error"] is None
        assert rec["evidence_units"] == prim["evidence_units"] and rec["evidence_chars"] == 42
        ri, pi = rec["meta"]["identity"], prim["meta"]["identity"]
        assert ri["rid"] != pi["rid"] and ri["reader"]["chat"] == "openrouter/google/gemma-3-27b-it"
        assert {k: ri[k] for k in ("experiment_id", "question_id", "pool_id", "method", "options", "setting",
                                   "code_sig", "question_sig", "pool_sha1")} == \
               {k: pi[k] for k in ("experiment_id", "question_id", "pool_id", "method", "options", "setting",
                                   "code_sig", "question_sig", "pool_sha1")}
        # the replay rid is computed exactly as run.record_identity computes a primary rid for that reader
        same = R.record_identity(pi["experiment_id"], pi["question_id"], pi["pool_id"], pi["method"], {},
                                 "variants-v2", SimpleNamespace(models={"chat": "openrouter/google/gemma-3-27b-it",
                                                                        "chat_digest": "openrouter"}), 0, "fa-fa")
        assert same["rid"] == ri["rid"]
        assert rec["models"]["chat"] == "openrouter/google/gemma-3-27b-it"
        assert rec["trace"]["replay"]["of_rid"] == pi["rid"] and "calls" not in rec["trace"]
        assert prim["answer_text"] == "۱۲"                           # the primary record is not modified
        ok, errs = RR.done_rids(Path(d) / "out", "HYBRID")
        assert ok == {ri["rid"]} and not errs
        # an infrastructure failure is a record with error, not an exception
        rec2 = RR.replay_one(cfg, pp, prim, "openrouter:google/gemma-3-27b-it", Path(d) / "out",
                             client=_Client(fail=True), ledger=Ledger(path=None))
        assert rec2["error"] and rec2["answer_text"] == ""
        ok, errs = RR.done_rids(Path(d) / "out", "HYBRID")
        assert errs == {ri["rid"]: 1}
        # a pack that does not match the primary is refused
        bad = dict(prim, meta=dict(prim["meta"], answer_messages_sha256="0" * 64))
        try:
            RR.replay_one(cfg, pp, bad, "openrouter:x/y", Path(d) / "out", client=_Client(), ledger=Ledger(path=None))
            raise AssertionError("mismatched pack accepted")
        except ValueError:
            pass


def test_setr_15500_cell():
    from ragbench.approaches import budget_matrix as BM
    from ragbench.approaches.setr.approach import SetR
    assert issubclass(BM.Setr15500, SetR) and BM.Setr15500.budget_chars == 15500
    assert BM.Setr15500.name == "SETR_15500"
    a = BM.Setr15500.__new__(BM.Setr15500)
    a.cfg = load_config()
    seen = {}
    import ragbench.approaches.budget_matrix as mod
    orig = mod.pack_evidence
    mod.pack_evidence = lambda items, corpus, budget=None, cfg=None: seen.setdefault("budget", budget)
    try:
        a.pack([], SimpleNamespace(corpus=None))
    finally:
        mod.pack_evidence = orig
    assert seen["budget"] == 15500


if __name__ == "__main__":
    import sys
    n = 0
    for k, f in list(globals().items()):
        if k.startswith("test_") and callable(f):
            f()
            n += 1
            print("ok", k)
    print("test_reader_replay: %d/%d ok" % (n, n))
    sys.exit(0)
