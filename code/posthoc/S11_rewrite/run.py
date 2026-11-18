"""S11 step 3 (CPU + API, no GPU): retrieval from the cached vectors, the frozen packer at 6,000, the frozen answer call.

Per question:
  REPRO  HYBRID_6000 re-implemented: hx.search(question, scope, k=40, depth=40, query_vector=re-encoded question) ->
         offer_pack(6000) -> build_answer_messages; compared byte-for-byte (json of the messages, and sha256) with the
         saved test pack packs/HYBRID/<qid>__<setting>.json. Also checks that this file's per-channel lists + rrf_fuse
         give the same 40 candidates in the same order as hx.search (the MQ code path).
  HYDE   BM25 channel on the original question, dense channel on the HyDE passage's vector (hx.search with
         query_vector = passage vector) -> same offer/pack/answer.
  MQ     BM25 and dense lists (depth 40 each) for the original question and each of the (up to) 3 rewrites, all 8 lists
         fused by the frozen rrf_fuse (k=60), top 40 -> same offer/pack/answer.
    python run.py dev | python run.py test --confirm-test
-> runs/<ARM>/<split>/runs_<ARM>.jsonl (records in the S9 format, judged by judge_multi), packs/<ARM>/<split>.jsonl,
   repro_<split>.jsonl."""
import argparse
import datetime as dt
import json
import sys
import time

import numpy as np

import s11lib as L
from ragbench.contracts import answer_num_predict, build_answer_messages

ARMS = {"HYDE": "HYDE_6000", "MQ": "MQ_6000"}
EXPERIMENT = "posthoc-S11-rewrite"
VERSION = "s11-rewrite-v1"


def rid_of(arm, q):
    ident = {"experiment_id": EXPERIMENT, "question_id": q.question_id, "pool_id": q.pool_id, "method": arm,
             "options": {"version": VERSION, "budget": L.BUDGET},
             "reader": {"chat": "openrouter/" + L.MODEL, "chat_digest": "openrouter"}, "replicate": 0,
             "setting": q.setting}
    ident["rid"] = L.hashlib.sha1(json.dumps(ident, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:20]
    return ident


def done(path):
    return {r["question_id"] for r in L.jl(path) if not r.get("error")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("split", choices=("dev", "test"))
    ap.add_argument("--confirm-test", action="store_true")
    ap.add_argument("--no-answer", action="store_true", help="build and check packs only")
    a = ap.parse_args()
    if a.split == "test" and not a.confirm_test:
        raise SystemExit("REFUSED: test needs --confirm-test")
    z = np.load(L.HERE / ("vectors_%s.npz" % a.split))
    VEC = {str(k): v for k, v in zip(z["keys"], z["vecs"])}
    rws = {(r["qid"], r["method"]): r for r in L.jl(L.HERE / ("rewrites_%s.jsonl" % a.split)) if r.get("ok")}
    qids = sorted({q for q, _ in rws})
    Q = L.queries(a.split, set(qids))
    paths = {}
    for m, arm in ARMS.items():
        d = L.HERE / "runs" / arm / a.split
        d.mkdir(parents=True, exist_ok=True)
        paths[m] = d / ("runs_%s.jsonl" % arm)
        (L.HERE / "packs" / arm).mkdir(parents=True, exist_ok=True)
    rp = L.HERE / ("repro_%s.jsonl" % a.split)
    repro_done = {r["qid"] for r in L.jl(rp)}
    fin = {m: done(paths[m]) for m in ARMS}
    for n, q in enumerate(Q):
        qid = q.question_id
        hx = L.hx_of(q.corpus_set)
        corpus = L.corpus_set(q.corpus_set)
        scope = list(q.doc_scope)
        mask, _ = hx.scope_mask(scope)
        qv = VEC["%s|ORIG|0" % qid]
        # ---- REPRO
        if qid not in repro_done:
            hits = hx.search(q.text, doc_ids=scope, k=L.K, depth=L.DEPTH, query_vector=qv)
            items = [h.to_item(rank=r) for r, h in enumerate(hits)]
            ev, n_off = L.offer_pack(items, corpus)
            msgs = build_answer_messages(q, ev, L.cfg)
            mine = {"bm25": L.bm25_list(hx, q.text, mask), "dense": L.dense_list(hx, qv, mask)}
            fitems, _top = L.fused_items(hx, mine)
            same_order = [it.meta["chunk"] for it in fitems] == [h.chunk for h in hits]
            row = {"qid": qid, "split": a.split, "search_eq_channel_code": same_order, "n_offered": n_off,
                   "sha": L.msgs_sha(msgs)}
            if a.split == "test":
                sp = L.saved_pack(q, "HYBRID")
                row["saved_sha_equal"] = row["sha"] == sp["messages_sha256"]
                row["bytes_equal"] = (json.dumps(msgs, ensure_ascii=False, sort_keys=True)
                                      == json.dumps(sp["messages"], ensure_ascii=False, sort_keys=True))
                row["evidence_chars_saved"] = sp.get("evidence_chars")
                row["evidence_chars_repro"] = ev.chars
            with open(rp, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(row) + "\n")
            print(qid, "REPRO", {k: v for k, v in row.items() if isinstance(v, bool)}, flush=True)
        # ---- HYDE / MQ
        for m, arm in ARMS.items():
            if qid in fin[m] or (qid, m) not in rws:
                continue
            t0 = time.perf_counter()
            rw = rws[(qid, m)]["rewrites"]
            if m == "HYDE":
                hits = hx.search(q.text, doc_ids=scope, k=L.K, depth=L.DEPTH, query_vector=VEC["%s|HYDE|0" % qid])
                items = [h.to_item(rank=r) for r, h in enumerate(hits)]
                top = [[h.chunk, round(h.rrf, 6), h.ranks.get("bm25"), h.ranks.get("dense")] for h in hits]
                chans = ["bm25(question)", "dense(hyde passage)"]
            else:
                rk = {"bm25:q": L.bm25_list(hx, q.text, mask), "dense:q": L.dense_list(hx, qv, mask)}
                for j, t in enumerate(rw):
                    rk["bm25:r%d" % j] = L.bm25_list(hx, t, mask)
                    rk["dense:r%d" % j] = L.dense_list(hx, VEC["%s|MQ|%d" % (qid, j)], mask)
                items, top = L.fused_items(hx, rk)
                chans = list(rk)
            ev, n_off = L.offer_pack(items, corpus)
            msgs = build_answer_messages(q, ev, L.cfg)
            npred = answer_num_predict(q, L.cfg)
            ret_ms = (time.perf_counter() - t0) * 1000.0
            with open(L.HERE / "packs" / arm / ("%s.jsonl" % a.split), "a", encoding="utf-8") as fh:
                fh.write(json.dumps({"question_id": qid, "arm": arm, "setting": q.setting, "messages": msgs,
                                     "messages_sha256": L.msgs_sha(msgs), "evidence_chars": ev.chars,
                                     "num_predict": npred}, ensure_ascii=False) + "\n")
            if a.no_answer:
                continue
            if n % 10 == 0 and m == "HYDE":
                rem = L.key_remaining()
                if rem is not None and rem < L.FLOOR_USD:
                    print("GLOBAL FLOOR", rem)
                    return 1
            err, ans = None, None
            t1 = time.perf_counter()
            try:
                ans = L.glm(msgs, purpose="answer", num_predict=npred,
                            ctx={"arm": arm, "question_id": qid, "split": a.split})
            except L.Stop as e:
                print("STOP", e)
                return 1
            except Exception as e:  # noqa: BLE001
                err = "%s: %s" % (type(e).__name__, str(e)[:300])
            answer = (ans["text"] if ans else "") or ""
            rec = {"question_id": qid, "approach": arm, "approach_base": arm, "flags": {}, "setting": q.setting,
                   "q_lang": q.q_lang, "d_lang": q.d_lang, "corpus": q.corpus, "corpus_set": q.corpus_set,
                   "pool_id": q.pool_id, "answer_text": answer,
                   "abstained": answer.strip() == L.cfg.evidence["abstain_token"],
                   "evidence_items": [{"doc_id": it.doc_id, "section_id": it.section_id, "chars": len(it.text),
                                       "unit_ids": it.unit_ids, "cut": "cut_from_chars" in (it.meta or {})}
                                      for it in ev.items],
                   "evidence_units": ev.unit_ids, "evidence_chars": ev.chars,
                   "latency_ms": round(ret_ms + (time.perf_counter() - t1) * 1000.0, 1), "error": err,
                   "trace": {"rewrites": rw, "channels": chans, "top": top, "offered": n_off,
                             "evidence_items": ev.n_included, "last_item_cut": ev.last_item_cut,
                             "retrieval_ms": round(ret_ms, 1)},
                   "answer_done_reason": ans["finish"] if ans else None,
                   "models": {"chat": "openrouter/" + L.MODEL, "embed": "bge-m3:latest"},
                   "meta": {"identity": rid_of(arm, q), "answer_format": q.answer_format, "answer_num_predict": npred,
                            "answer_messages_sha256": L.msgs_sha(msgs), "version": VERSION},
                   "answer_messages": msgs, "ts": dt.datetime.now().isoformat(timespec="seconds")}
            with open(paths[m], "a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            print(qid, arm, "ev=%d" % ev.chars, err or "", "|", answer[:70].replace("\n", " "),
                  "| $%.4f" % L.spent(), flush=True)
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
