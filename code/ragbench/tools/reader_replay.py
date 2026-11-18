"""Reader replay (2026-10-01, multi-reader true test; work/datasets_v2/test_freeze/PROTOCOL.md amendment 1).

The primary run (`ragbench.run`, reader GLM) does retrieval, reranking and any selector call ONCE per question x arm and
saves the exact input of the shared answer call (the PACK: system + user messages with the rendered evidence, and
num_predict; run.save_answer_pack). This tool sends that identical pack to another reader and writes an ordinary
RunRecord, so the pack is byte-identical across readers and only the model that reads it differs.

    rec = replay_one(cfg, pack_path, primary_record, "openrouter:<model id>", out_dir, attempts=7)

Rules (the same answer contract and settings as the primary reader):
  - the messages and num_predict are the pack's, unchanged; the call goes through OpenRouterReader (same sampling
    parameters filtered by the pinned endpoint's supported list, same reasoning effort, same max_tokens rule), pinned
    through env OPENROUTER_PROVIDER_PINS; READER attempts = the frozen reader attempt policy;
  - the pack must match the primary record (meta.answer_messages_sha256) - otherwise refused;
  - the record copies the primary record's retrieval / evidence fields, replaces the answer, the call counters and
    latency (reader call only; the shared pre-answer part is the primary's), sets models.chat to the replay reader and
    a NEW identity (record_identity with that reader; code / question / pool signatures copied), and records
    meta.replay = {of_rid, of_reader, pack_sha256, provider, done_reason, ...};
  - a call that fails after every HTTP attempt gives a record with `error` (an infrastructure failure, like a primary
    cell's); BudgetExceeded propagates (stop, resume later).
"""
import copy
import json
import time
from pathlib import Path

from ..contracts import RunLog
from ..llm import BudgetExceeded, Ledger
from ..util import now_iso, read_json, safe_name

REPLAY_VERSION = "reader-replay-v1"


def _messages_sha256(messages):
    from ..run import answer_messages_sha256
    return answer_messages_sha256(messages)


def make_reader(cfg, ledger, reader, attempts):
    from ..llm_openrouter import OpenRouterReader, model_info
    if not reader.startswith("openrouter:"):
        raise ValueError("reader must be openrouter:<model id>")
    rid = reader.split(":", 1)[1]
    info = model_info(rid, cfg.project_root / "work" / "openrouter" / "catalog.json")
    return OpenRouterReader(cfg, ledger, rid, supported=info.get("params"), attempts=attempts)


def replay_identity(primary, reader_chat):
    """The replay record's identity: the primary's identity with reader.chat replaced; rid recomputed exactly as
    run.record_identity does; code_sig / question_sig / pool_sha1 copied."""
    from ..util import sha1_json
    old = dict(((primary.get("meta") or {}).get("identity")) or {})
    if not old.get("rid"):
        raise ValueError("primary record has no meta.identity")
    ident = {k: old[k] for k in ("experiment_id", "question_id", "pool_id", "method", "options")}
    ident["reader"] = {"chat": reader_chat, "chat_digest": "openrouter"}
    ident["replicate"] = old.get("replicate", 0)
    ident["setting"] = old["setting"]
    ident["rid"] = sha1_json(ident)[:20]
    for k in ("code_sig", "question_sig", "pool_sha1"):
        if k in old:
            ident[k] = old[k]
    return ident


def replay_one(cfg, pack_path, primary, reader, out_dir, attempts=7, client=None, ledger=None):
    """-> the written RunRecord dict. client/ledger: injectable (tests)."""
    pack = read_json(pack_path)
    if pack is None:
        raise FileNotFoundError(pack_path)
    h = _messages_sha256(pack["messages"])
    if h != pack.get("messages_sha256") or h != (primary.get("meta") or {}).get("answer_messages_sha256"):
        raise ValueError("pack %s does not match the primary record (messages sha256)" % pack_path)
    if pack["question_id"] != primary["question_id"] or pack["variant"] != primary["approach"]:
        raise ValueError("pack %s belongs to another question or arm" % pack_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = safe_name(primary["approach"])
    if ledger is None:
        ledger = Ledger(path=str(out_dir / ("ledger_%s.jsonl" % tag)))
    if client is None:
        client = make_reader(cfg, ledger, reader, attempts)
    model_id = reader.split(":", 1)[1]
    chat_label = "openrouter/" + model_id
    abstain = cfg.evidence["abstain_token"]
    err, res = None, None
    with ledger.scope(phase="query", approach=primary["approach"], question_id=primary["question_id"],
                      setting=primary["setting"], corpus_set=primary.get("corpus_set")) as sc:
        t0 = time.perf_counter()
        try:
            res = client.chat(pack["messages"], purpose=pack.get("purpose") or "answer",
                              num_predict=pack["num_predict"])
        except BudgetExceeded:
            raise
        except Exception as e:  # noqa: BLE001  every HTTP attempt failed: an infrastructure failure
            err = "%s: %s" % (type(e).__name__, str(e)[:500])
        latency = (time.perf_counter() - t0) * 1000.0
    summ = sc.summary()
    answer = (res.text.strip() if res is not None else "") or ""
    rec = copy.deepcopy(primary)
    ident = replay_identity(primary, chat_label)
    rec.update(answer_text=answer, abstained=answer.strip() == abstain, llm_calls=summ["llm_calls"], embed_calls=0,
               prompt_tokens=summ["prompt_tokens"], completion_tokens=summ["completion_tokens"], embed_tokens=0,
               llm_wall_ms=summ["llm_wall_ms"], load_ms=0.0, latency_ms=round(latency, 1),
               failed_calls=summ["failed_calls"], error=err, ts=now_iso(), run_id=ledger.run_id, lock_wait_ms=0.0,
               models={"chat": chat_label, "embed": (primary.get("models") or {}).get("embed")})
    rec["trace"] = {"replay": {
        "version": REPLAY_VERSION, "of_rid": ((primary.get("meta") or {}).get("identity") or {}).get("rid"),
        "of_reader": (primary.get("models") or {}).get("chat"), "pack": str(pack_path), "pack_sha256": h,
        "provider": getattr(res, "provider", None) if res is not None else None,
        "provider_pin": getattr(client, "provider", None),
        "done_reason": res.done_reason if res is not None else None,
        "reasoning_tokens": getattr(res, "reasoning_tokens", None) if res is not None else None,
        "cost_usd": res.cost_usd if res is not None else None,
        "primary_latency_ms": primary.get("latency_ms"), "primary_trace_keys": sorted((primary.get("trace") or {}))}}
    meta = dict(rec.get("meta") or {})
    meta["identity"] = ident
    meta["replay"] = {"version": REPLAY_VERSION, "of_rid": rec["trace"]["replay"]["of_rid"], "pack_sha256": h,
                      "reader": chat_label}
    meta["answer_messages_sha256"] = h
    rec["meta"] = meta
    with open(out_dir / ("runs_%s.jsonl" % tag), "a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return rec


def done_rids(out_dir, approach):
    """{rid: n_error_records, ...} and the set of rids with an answer, from <out_dir>/runs_<approach>.jsonl."""
    p = Path(out_dir) / ("runs_%s.jsonl" % safe_name(approach))
    ok, errs = set(), {}
    if p.is_file():
        for r in RunLog(p).records():
            rid = ((r.get("meta") or {}).get("identity") or {}).get("rid")
            if r.get("error"):
                errs[rid] = errs.get(rid, 0) + 1
            else:
                ok.add(rid)
    return ok, errs
