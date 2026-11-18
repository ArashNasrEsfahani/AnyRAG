"""Step 0: the rebuilt packs must reproduce the frozen packs byte-for-byte (messages sha256), all 6 frozen non-SETR
arms x 96 questions; plus the LEX5B fallback recomputation (B.hybrid_intact over f2_top40) must equal fallback_chunks.
Writes verify.json and verify_rows.jsonl."""
import json
import sys
from collections import Counter

import packs as PK

rows, summ = [], {}
for arm in ["HYBRID_6000", "HR_6000", "LEX5B_6000", "HYBRID_15500", "HR_15500", "LEX5B_15500"]:
    budget = 6000 if arm.endswith("_6000") else 15500
    recs = PK.frozen_records(arm)
    c = Counter()
    for qid in PK.qids_all():
        r = recs.get(qid)
        if r is None or r.get("error"):
            c["no_record"] += 1
            continue
        cs = PK.corpus(r["corpus_set"])
        row = {"arm": arm, "qid": qid}
        if arm.startswith("LEX5B"):
            st = r["trace"]["lex5b"]
            ev = PK.pack_lex_chunks(r["corpus_set"], st["shown_chunks"], budget)
            fb, cost = PK.fusion_fallback(r["corpus_set"], st["f2_top40"], budget)
            fb_ids = [p["chunk"] for p in fb]
            row["fallback_match"] = fb_ids == st["fallback_chunks"]
            row["selector_fallback"] = bool(st["selector"].get("fallback"))
            row["same_as_fallback"] = st["shown_chunks"] == st["fallback_chunks"]
            c["fallback_match"] += row["fallback_match"]
            # the fallback pack rendered: never clipped, exact cost
            evf = PK.pack_lex_chunks(r["corpus_set"], fb_ids, budget)
            row["fallback_chars"], row["fallback_cost"], row["fallback_cut"] = evf.chars, cost(fb), evf.last_item_cut
            c["fallback_exact_cost"] += evf.chars == cost(fb) and not evf.last_item_cut
        elif arm == "HYBRID_15500":
            ev = PK.pack_hybrid_direct(PK.items_from_record(r), cs, budget)
        else:
            ev = PK.pack_offer(PK.items_from_record(r), cs, budget)
        msgs, npred = PK.messages_for(qid, r["setting"], ev)
        h = PK.sha256_msgs(msgs)
        fp = PK.frozen_pack(arm, qid, r["setting"])
        row.update(sha_record=h == r["meta"].get("answer_messages_sha256"),
                   sha_packfile=(fp is not None and h == fp["messages_sha256"] and msgs == fp["messages"]),
                   npred_match=(fp is not None and npred == fp["num_predict"]), chars=ev.chars,
                   chars_record=r["evidence_chars"])
        c["sha_record"] += row["sha_record"]
        c["sha_packfile"] += row["sha_packfile"]
        c["npred_match"] += row["npred_match"]
        c["n"] += 1
        rows.append(row)
    summ[arm] = dict(c)
    print(arm, dict(c), flush=True)

with open(PK.HERE / "verify_rows.jsonl", "w", encoding="utf-8") as f:
    for r in rows:
        f.write(json.dumps(r) + "\n")
(PK.HERE / "verify.json").write_text(json.dumps(summ, indent=1), encoding="utf-8")
bad = [r for r in rows if not (r["sha_record"] and r["sha_packfile"])]
print("mismatches:", len(bad), bad[:5])
sys.exit(1 if bad else 0)
