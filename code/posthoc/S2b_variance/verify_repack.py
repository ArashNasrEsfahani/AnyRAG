"""Gate: re-pack every saved LEX5B selector reply (6000 and 15500) with the frozen packer and compare the answer
messages sha256 with the saved pack. No model call. Writes verify_repack.json."""
import json
import sys

sys.path.insert(0, r"D:\KareOne\RAG Paper\work\posthoc_2026-11\S2b_variance")
import lib as L  # noqa: E402

out = {}
for arm, budget in (("LEX5B_6000", 6000), ("LEX5B_15500", 15500)):
    rows = []
    for q in L.QIDS:
        rec = L.primary(arm, q)
        pk = json.loads(L.pack_path(arm, rec).read_text(encoding="utf-8"))
        sel = rec["trace"]["lex5b"]["selector"]
        failed = sel.get("status") == "fail" and sel.get("raw") is None
        r = L.lex5b_repack(rec, budget, sel.get("raw"), sel.get("finish"), call_failed=failed)
        rows.append({"q": q, "match": r["sha"] == pk["messages_sha256"] == rec["meta"]["answer_messages_sha256"],
                     "shown_match": r["shown_chunks"] == rec["trace"]["lex5b"]["shown_chunks"],
                     "fallback_saved": sel.get("fallback"), "fallback_repack": r["fallback"],
                     "saved_status": sel.get("status"), "saved_reason": sel.get("reason"),
                     "raw_len": len(sel.get("raw") or "")})
    n = sum(x["match"] for x in rows)
    out[arm] = {"n": len(rows), "sha_match": n, "shown_match": sum(x["shown_match"] for x in rows),
                "fallback_saved": sum(bool(x["fallback_saved"]) for x in rows),
                "saved_reasons": {}, "mismatches": [x for x in rows if not x["match"]]}
    for x in rows:
        if x["fallback_saved"]:
            k = str(x["saved_reason"])
            out[arm]["saved_reasons"][k] = out[arm]["saved_reasons"].get(k, 0) + 1
    print(arm, out[arm]["n"], out[arm]["sha_match"], out[arm]["shown_match"], out[arm]["fallback_saved"],
          out[arm]["saved_reasons"])
(L.HERE / "verify_repack.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
