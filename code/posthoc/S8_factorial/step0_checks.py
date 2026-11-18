"""S8 step 0 (no model calls): is the LEX5B selector prompt budget-independent, and does the frozen code re-pack /
re-render it byte-for-byte?  -> step0_checks.json
 (a) saved LEX5B_6000 prompt == saved LEX5B_15500 prompt (trace.prompt), and f2_top40 equal, per question;
 (b) the literal view rebuilt from f2_top40 (s8lib.view_prompt) == the saved 6000 prompt;
 (c) re-pack of the saved replies at their own budget == saved pack sha (6000 via Lex5b, 15500 via the generalised
     class, which must equal Lex5b_15500);
 (d) the fallback re-packed at 3000/10000/24000 == S2a's FUSION_<B> pack sha (so fallback cells may reuse S2a)."""
import json
import sys
sys.path.insert(0, r"D:\KareOne\RAG Paper\work\posthoc_2026-11\S8_factorial")
import s8lib as L  # noqa: E402

S2A = {}
for r in L.jl(L.WORK / "posthoc_2026-11" / "S2a_packs" / "packs_index.jsonl"):
    S2A[(r["arm"], r["qid"])] = r["sha"]
out = {"a_prompt_same": 0, "a_f2_same": 0, "b_view_rebuild": 0, "c_6000": 0, "c_15500": 0,
       "d_fallback_eq_fusion": {}, "n": 0, "mismatch": []}
for q in L.QIDS:
    r6, r15 = L.primary("LEX5B_6000", q), L.primary("LEX5B_15500", q)
    out["n"] += 1
    out["a_prompt_same"] += r6["trace"]["prompt"] == r15["trace"]["prompt"]
    out["a_f2_same"] += r6["trace"]["lex5b"]["f2_top40"] == r15["trace"]["lex5b"]["f2_top40"]
    p, meta, cap, est = L.view_prompt(r6, r6["trace"]["lex5b"]["f2_top40"])
    ok_b = p == r6["trace"]["prompt"]
    out["b_view_rebuild"] += ok_b
    for rec, b, key in ((r6, 6000, "c_6000"), (r15, 15500, "c_15500")):
        s = rec["trace"]["lex5b"]["selector"]
        failed = s.get("status") == "fail" and s.get("raw") is None
        rp = L.lex5b_repack(rec, b, s.get("raw"), s.get("finish"), call_failed=failed)
        ok = rp["sha"] == rec["meta"]["answer_messages_sha256"]
        out[key] += ok
        if not ok:
            out["mismatch"].append([q, key])
    if not ok_b:
        out["mismatch"].append([q, "b"])
    for b in (3000, 6000, 10000, 15500, 24000):
        rp = L.lex5b_repack(r6, b, None, None, call_failed=True)
        d = out["d_fallback_eq_fusion"].setdefault(str(b), 0)
        out["d_fallback_eq_fusion"][str(b)] = d + (rp["sha"] == S2A.get(("FUSION_%d" % b, q)))
print(json.dumps(out, indent=1))
(L.HERE / "step0_checks.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
