"""Gate for SETR selector repeats: re-pack the saved SETR replies; compare answer-messages sha256 with saved packs."""
import json, sys
sys.path.insert(0, r"D:\KareOne\RAG Paper\work\posthoc_2026-11\S2b_variance")
import lib as L, setr_lib as S
out = {}
for arm, budget in (("SETR_6000", 6000),):
    rows = []
    for q in L.QIDS:
        rec = L.primary(arm, q)
        sel = rec["trace"]["setr"]["selector"]
        r = S.setr_repack(rec, budget, sel.get("raw"), call_failed=sel.get("raw") is None)
        rows.append({"q": q, "match": r["sha"] == rec["meta"]["answer_messages_sha256"],
                     "prompt_match": r["selector_messages"][1]["content"] == (rec["trace"].get("setr_prompt") or r["selector_messages"][1]["content"]),
                     "sel_match": r["selected"] == (sel.get("selected") or []), "offered_match": r["offered"] == rec["trace"]["setr"]["offered"],
                     "fallback": sel.get("fallback"), "reason": sel.get("reason"), "raw_len": len(sel.get("raw") or "")})
    out[arm] = {"n": len(rows), "sha_match": sum(x["match"] for x in rows), "sel_match": sum(x["sel_match"] for x in rows),
                "offered_match": sum(x["offered_match"] for x in rows), "fallback": sum(bool(x["fallback"]) for x in rows),
                "raw_at_cap_4000": sum(x["raw_len"] >= 4000 for x in rows),
                "mismatch": [x for x in rows if not x["match"]]}
    print(arm, {k: v for k, v in out[arm].items() if k != "mismatch"}, [x["q"] for x in out[arm]["mismatch"]][:10])
(L.HERE / "verify_setr.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
