"""SETR re-packing from the saved HYBRID order (order40) and a selector reply, with the frozen SetR code."""
import sys

sys.path.insert(0, r"D:\KareOne\RAG Paper\work\posthoc_2026-11\S2b_variance")
import lib as L  # noqa: E402

_APP = {}


def _approach(budget, pool):
    from ragbench.approaches.budget_matrix import Setr15500
    from ragbench.approaches.setr.approach import SetR
    k = (budget, id(pool))
    with L._LOCK:
        if k not in _APP:
            cls = SetR if budget == 6000 else Setr15500
            a = cls(cfg=L.CFG, client=L._NoClient(), flags={})
            a.variants_version = "variants-v2"
            a.pool = pool
            _APP[k] = a
        return _APP[k]


def setr_repack(rec, budget, text, call_failed=False):
    from ragbench.approaches.setr import approach as SA
    from ragbench.contracts import answer_num_predict, build_answer_messages
    from ragbench.retrieval.hybrid import Hit
    st = rec["trace"]["setr"]
    C = L._ctx(rec["corpus_set"], rec["meta"]["identity"]["pool_id"])
    hx = C["hx"]
    app = _approach(budget, C["pool"])
    query = app.make_query(L.question_obj(rec["question_id"]), rec["setting"])
    hits = []
    for ci in st["order40"]:
        c = hx.chunks[ci]
        hits.append(Hit(doc_id=c["doc_id"], section_id=c["section_id"], unit_ids=list(c["unit_ids"]), text=c["text"],
                        span=list(c["span"]), ranks={}, scores={}, rrf=0.0, chunk=ci,
                        section_ids=list(c.get("section_ids") or []),
                        partial_unit_ids=list(c.get("partial_unit_ids") or []), mode=c.get("mode")))
    items = [h.to_item(rank=r) for r, h in enumerate(hits)]
    top = min(int(SA.SETR_DEFAULTS["top"]), len(items))
    msgs_sel, _cut = SA.build_prompt(query.text, [it.text for it in items[:top]])
    ids, status = ([], "call_failed") if (call_failed or text is None) else SA.parse_selection(text, top)
    fallback = not ids
    ordered = list(items) if fallback else SA.reorder(items, ids, top)
    idx = L._Idx(C["cs"], hx)
    offered, ev = app.pack_ordered(ordered, idx, query)
    msgs = build_answer_messages(query, ev, L.CFG)
    return {"messages": msgs, "num_predict": answer_num_predict(query, L.CFG), "sha": L.msha(msgs),
            "selector_messages": msgs_sel, "selected": ids, "fallback": fallback, "status": status,
            "offered": [int(it.meta["chunk"]) for it in offered], "evidence_chars": ev.chars}
