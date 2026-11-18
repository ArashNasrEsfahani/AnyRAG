"""S6 arms (post-hoc, GLM-5.3-flash reader pinned as in the test; 6,000 characters).

    python run_arms.py --arm CLOSEDBOOK|LISTWISE|SETR_NOFILL --split dev|test [--ids q1,q2] [--limit N] [--workers 4]

CLOSEDBOOK   the frozen answer prompt's no-evidence variant (contracts.build_answer_messages(query, None): the same
             per-format system prompt with SOURCE_RULE_NONE "No documents are provided; answer from your own knowledge."
             and ABSTAIN_RULE_NONE "If you do not know the answer, reply with exactly NOT_FOUND ...", user message
             "Question: <q>") - the codebase's own B0 closed-book prompt; same num_predict.
LISTWISE     RankGPT (Sun et al., EMNLP 2023) listwise reranking by GLM, ONE call, over HYBRID's saved top-20 fused
             chunks (each passage: newlines collapsed, cut to PASSAGE_CAP=700 chars + " …"); RankGPT's multi-turn
             prompt verbatim; parse = RankGPT receive_permutation (ids in reply order, deduplicated, in range; the
             missing ids appended in HYBRID order); candidates 21-40 follow in HYBRID order; then HYBRID's frozen
             _offer + final_pack (fill, same clipping) + the shared answer call. Ranker call: pinned GLM endpoint,
             reasoning effort low, max_tokens 2000 incl. reasoning, 3 HTTP attempts, no deadline; a failed call or a
             reply with no id -> HYBRID order (fallback, logged).
SETR_NOFILL  SETR_6000's saved selected set (trace.setr.selector.selected, re-parse verified by verify_repack.py) in
             selection order, given to the frozen packer ALONE (no _offer fill from the HYBRID list; the packer's own
             6,000-char cut applies). Selector fallback (6 cells): no set exists -> the SETR fallback pack (HYBRID
             order with fill), i.e. SETR's own pack. A NOFILL pack byte-identical to the saved SETR pack is still
             answered afresh (no answer reuse).
Records: <split>/<ARM>/runs_<ARM>.jsonl (append-only; a cell with a non-error record is skipped); packs beside them.
"""
import argparse
import copy
import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import s6lib as L  # noqa: E402
import s6call as C  # noqa: E402
from ragbench.llm import BudgetExceeded  # noqa: E402
from ragbench.util import now_iso, sha1_json  # noqa: E402

PASSAGE_CAP = 700
TOP = 20
RG_SYSTEM = "You are RankGPT, an intelligent assistant that can rank passages based on their relevancy to the query."
RG_PREFIX = ("I will provide you with {num} passages, each indicated by number identifier []. \n"
             "Rank the passages based on their relevance to query: {query}.")
RG_POST = ("Search Query: {query}. \nRank the {num} passages above based on their relevance to the search query. "
           "The passages should be listed in descending order using identifiers. The most relevant passages should be "
           "listed first. The output format should be [] > [], e.g., [1] > [2]. Only response the ranking results, "
           "do not say any word or explain.")


def rg_passage(text):
    t = re.sub(r"\s*\n+\s*", " ", (text or "").strip())
    if len(t) > PASSAGE_CAP:
        return t[:PASSAGE_CAP].rstrip() + " …", True
    return t, False


def rankgpt_messages(query, texts):
    msgs = [{"role": "system", "content": RG_SYSTEM},
            {"role": "user", "content": RG_PREFIX.format(num=len(texts), query=query)},
            {"role": "assistant", "content": "Okay, please provide the passages."}]
    cut = 0
    for i, t in enumerate(texts):
        body, c = rg_passage(t)
        cut += c
        msgs.append({"role": "user", "content": "[%d] %s" % (i + 1, body)})
        msgs.append({"role": "assistant", "content": "Received passage [%d]." % (i + 1)})
    msgs.append({"role": "user", "content": RG_POST.format(query=query, num=len(texts))})
    return msgs, cut


def receive_permutation(text, n):
    """RankGPT: digits of the reply in order (non-digits -> space), 1-based, deduplicated, in range."""
    ids = []
    for tok in re.sub(r"[^0-9]", " ", text or "").split():
        k = int(tok)
        if 1 <= k <= n and k not in ids:
            ids.append(k)
    return ids


def base_records(split, arm):
    if split == "dev":
        return L.records(arm, base=(L.DS / "setr" / "runs") if arm == "SETR_6000" else L.DEV60)
    return L.records(arm)


def out_dir(split, arm):
    d = L.OUT / split / arm
    d.mkdir(parents=True, exist_ok=True)
    return d


def done_qids(d, arm):
    p = d / ("runs_%s.jsonl" % arm)
    return {r["question_id"] for r in L.jl(p) if not r.get("error")}


def make_record(arm, base, answer, err, msgs, ev, ordered, summ, s6trace, latency_ms):
    rec = copy.deepcopy(base)
    old = rec["meta"]["identity"]
    ident = {k: old[k] for k in ("experiment_id", "question_id", "pool_id")}
    ident.update(method=arm, options={"flags": {}, "variants_version": old["options"]["variants_version"],
                                      "posthoc": "S6_baselines"},
                 reader=old["reader"], replicate=0, setting=old["setting"])
    ident["rid"] = sha1_json(ident)[:20]
    abstain = L.CFG.evidence["abstain_token"]
    rec.update(approach=arm, approach_base=arm, answer_text=answer, abstained=answer.strip() == abstain,
               retrieved_units=[it.to_dict() for it in ordered] if ordered is not None else [],
               evidence_units=ev.unit_ids if ev else [],
               evidence_items=[{"doc_id": it.doc_id, "section_id": it.section_id, "kind": it.kind, "chars": len(it.text),
                                "unit_ids": it.unit_ids, "cut": "cut_from_chars" in (it.meta or {})}
                               for it in (ev.items if ev else [])],
               evidence_chars=ev.chars if ev else 0, llm_calls=summ["llm_calls"], embed_calls=0,
               prompt_tokens=summ["prompt_tokens"], completion_tokens=summ["completion_tokens"], embed_tokens=0,
               llm_wall_ms=summ["llm_wall_ms"], load_ms=0.0, latency_ms=round(latency_ms, 1),
               failed_calls=summ["failed_calls"], error=err, ts=now_iso(), run_id=C.LEDGER.run_id, lock_wait_ms=0.0,
               trace={"s6": s6trace, "last_item_cut": ev.last_item_cut if ev else False,
                      "evidence_items": ev.n_included if ev else 0})
    rec["meta"] = dict(rec["meta"], identity=ident, answer_messages_sha256=L.sha(msgs),
                       s6={"arm": arm, "of_rid": old["rid"], "of_method": old["method"]})
    return rec


def run_cell(arm, qid, base, setr_pack_sha, d, reader, ranker):
    q = L.query_of(qid)
    st = {"arm": arm}
    ordered, ev = None, None
    with C.LEDGER.scope(phase="query", approach=arm, question_id=qid, setting=q.setting,
                        corpus_set=q.corpus_set) as sc:
        t0 = time.perf_counter()
        if arm == "CLOSEDBOOK":
            msgs, npred = L.closed_book_messages(qid)
        elif arm == "LISTWISE":
            items = L.items_of(base)                         # HYBRID's fused order (40)
            top = min(TOP, len(items))
            rmsgs, cut = rankgpt_messages(q.text, [it.text for it in items[:top]])
            sel = {"shown": top, "passages_cut": cut, "prompt_chars": sum(len(m["content"]) for m in rmsgs)}
            t = time.perf_counter()
            try:
                res = ranker.chat(rmsgs, purpose="listwise_rank", num_predict=2000)
                perm = receive_permutation(res.text, top)
                sel.update(raw=res.text[:2000], prompt_tokens=res.prompt_tokens, completion_tokens=res.completion_tokens,
                           reasoning_tokens=getattr(res, "reasoning_tokens", None), cost_usd=res.cost_usd,
                           provider=getattr(res, "provider", None), finish=res.done_reason, parsed=list(perm))
                sel["status"] = "ok" if perm else "no_ids"
            except BudgetExceeded:
                raise
            except Exception as e:  # noqa: BLE001
                perm, sel["status"], sel["error"] = [], "call_error", str(e)[:300]
            sel["t_rank_ms"] = round((time.perf_counter() - t) * 1000.0, 1)
            sel["fallback"] = not perm
            sel["n_parsed"] = len(perm)
            full = perm + [i for i in range(1, top + 1) if i not in perm]
            ordered = [items[i - 1] for i in full] + items[top:]
            sel["order"] = [int(it.meta["chunk"]) for it in ordered[:top]]
            st["ranker"] = sel
            msgs, npred, ev, offered = L.pack_messages(ordered, qid, L.approach("HYBRID"))
            st["offered"] = [int(it.meta["chunk"]) for it in offered]
        elif arm == "SETR_NOFILL":
            s = base["trace"]["setr"]
            items = L.items_of(base)
            by = {int(it.meta["chunk"]): it for it in items}
            hyb = [by[c] for c in s["order40"]]
            selr = s["selector"]
            if selr["fallback"]:
                ordered = hyb
                msgs, npred, ev, offered = L.pack_messages(hyb, qid, L.approach("SETR"))
                st["setr_fallback"] = True
            else:
                ordered = [hyb[i - 1] for i in selr["selected"]]
                msgs, npred, ev, offered = L.pack_messages(ordered, qid, L.approach("SETR"), no_fill=True)
                st["setr_fallback"] = False
            st["selected"] = selr.get("selected")
            st["n_selected"] = len(selr.get("selected") or [])
            st["offered"] = [int(it.meta["chunk"]) for it in offered]
            st["identical_to_setr_pack"] = L.sha(msgs) == setr_pack_sha
        else:
            raise ValueError(arm)
        err, answer = None, ""
        try:
            r = reader.chat(msgs, purpose="answer", num_predict=npred)
            answer = r.text.strip()
            st["answer_done_reason"] = r.done_reason
            st["answer_provider"] = getattr(r, "provider", None)
        except BudgetExceeded:
            raise
        except Exception as e:  # noqa: BLE001
            err = "%s: %s" % (type(e).__name__, str(e)[:500])
        latency = (time.perf_counter() - t0) * 1000.0
    summ = sc.summary()
    rec = make_record(arm, base, answer, err, msgs, ev, ordered, summ, st, latency)
    with open(d / ("packs_%s.jsonl" % arm), "a", encoding="utf-8") as f:
        f.write(json.dumps({"question_id": qid, "messages": msgs, "num_predict": npred,
                            "messages_sha256": L.sha(msgs)}, ensure_ascii=False) + "\n")
    with open(d / ("runs_%s.jsonl" % arm), "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=("CLOSEDBOOK", "LISTWISE", "SETR_NOFILL"))
    ap.add_argument("--split", required=True, choices=("dev", "test"))
    ap.add_argument("--ids", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    src = "SETR_6000" if a.arm == "SETR_NOFILL" else "HYBRID_6000"
    bases = base_records(a.split, src)
    qids = a.ids.split(",") if a.ids else (L.QIDS if a.split == "test" else sorted(bases))
    d = out_dir(a.split, a.arm)
    todo = [q for q in qids if q not in done_qids(d, a.arm)]
    if a.limit is not None:
        todo = todo[: a.limit]
    print("%s %s: %d to run" % (a.arm, a.split, len(todo)), flush=True)
    reader, ranker = C.glm_reader(), C.glm_ranker()
    batch = 20 if a.arm == "LISTWISE" else 40          # <= 50 calls between floor checks
    try:
        for i in range(0, len(todo), batch):
            rem = C.floor_check()
            print("floor ok: limit_remaining=%s, S6 spent=$%.4f" % (rem, C.spent()), flush=True)
            chunk = todo[i:i + batch]

            def one(qid):
                sp = None
                if a.arm == "SETR_NOFILL" and a.split == "test":
                    sp = json.load(open(L.pack_file("SETR", qid), encoding="utf-8"))["messages_sha256"]
                return run_cell(a.arm, qid, bases[qid], sp, d, reader, ranker)
            with ThreadPoolExecutor(a.workers) as ex:
                for rec in ex.map(one, chunk):
                    print(rec["question_id"], "ERR " + rec["error"][:80] if rec["error"] else
                          "ok %d chars, %d ptok" % (len(rec["answer_text"]), rec["prompt_tokens"]), flush=True)
    except BudgetExceeded as e:
        print("BUDGET STOP:", e)
        return 3
    print("S6 spent so far: $%.4f" % C.spent())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
