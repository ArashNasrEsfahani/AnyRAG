"""Step 0 (no model calls): re-pack the saved HYBRID_6000 and SETR_6000 selections with the frozen code and compare the
messages sha256 with the saved packs (packs/HYBRID, packs/SETR). Writes verify_repack.jsonl + prints a summary.
SETR re-pack: HYBRID order = trace.setr.order40 (items looked up by chunk id in the saved retrieved_units); selected ids
= trace.setr.selector.selected unless fallback; ordered = setr.reorder(hybrid, selected, top=20); then _offer +
final_pack (= SetR.pack_ordered) + build_answer_messages."""
import json
import sys

sys.path.insert(0, r"D:\KareOne\RAG Paper\work\posthoc_2026-11\S6_baselines")
import s6lib as L  # noqa: E402
from ragbench.approaches.setr.approach import reorder, SETR_DEFAULTS, parse_selection  # noqa: E402

out = open(L.OUT / "verify_repack.jsonl", "w", encoding="utf-8")
summary = {}
for arm, variant in (("HYBRID_6000", "HYBRID"), ("SETR_6000", "SETR")):
    recs = L.records(arm)
    ok = bad = 0
    for qid in L.QIDS:
        r = recs[qid]
        pk = json.load(open(L.pack_file(variant, qid), encoding="utf-8"))
        items = L.items_of(r)
        row = {"arm": arm, "qid": qid}
        if arm == "SETR_6000":
            st = r["trace"]["setr"]
            by = {int(it.meta["chunk"]): it for it in items}
            hyb = [by[c] for c in st["order40"]]
            sel = st["selector"]
            # re-parse the saved raw reply with the frozen parser (raw is the first 4000 chars of the reply)
            reparsed = parse_selection(sel.get("raw"), sel["shown"])[0] if sel.get("raw") is not None else None
            row["reparse_matches"] = (reparsed == sel.get("selected")) if not sel["fallback"] else None
            ordered = hyb if sel["fallback"] else reorder(hyb, sel["selected"], SETR_DEFAULTS["top"])
            row["order_matches_saved"] = [int(i.meta["chunk"]) for i in ordered] == [int(i.meta["chunk"]) for i in items]
            row["fallback"] = sel["fallback"]
            app = L.approach("SETR")
        else:
            ordered = items
            app = L.approach("HYBRID")
        msgs, npred, ev, offered = L.pack_messages(ordered, qid, app)
        h = L.sha(msgs)
        row.update(sha_repack=h, sha_pack=pk["messages_sha256"], sha_record=r["meta"].get("answer_messages_sha256"),
                   match=(h == pk["messages_sha256"] == r["meta"].get("answer_messages_sha256")),
                   num_predict_match=(npred == pk["num_predict"]))
        ok += row["match"]
        bad += not row["match"]
        out.write(json.dumps(row) + "\n")
    summary[arm] = {"match": ok, "mismatch": bad}
print(json.dumps(summary))
