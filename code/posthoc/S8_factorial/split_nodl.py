"""Split LEX5B_NODL_6000 vs LEX5B_6000 (judge 1) into the 19 saved fallbacks and the 77 saved successful selections.
-> split_nodl.json (no model calls)."""
import json
import sys
sys.path.insert(0, r"D:\KareOne\RAG Paper\work\posthoc_2026-11\S8_factorial")
import analyze as A  # noqa: E402  (module import builds the cell functions; main() is not run)

rows, _ = A.s8_assign("nodl")
out = {}
for part in ("saved_fallback", "saved_success"):
    qs = [q for q in A.ANS if (rows[(q, 6000)]["saved_status"] == "fail") == (part == "saved_fallback")]
    lo, up, _u = A.NS["diff_vectors"](A.J1["LEX5B_NODL_6000"], A.J1["LEX5B_6000"], qs)
    b = w = 0
    for q in qs:
        al, ah = A.J1["LEX5B_NODL_6000"](q)
        bl, bh = A.J1["LEX5B_6000"](q)
        if al == ah and bl == bh:
            b += al > bl
            w += al < bl
    out[part] = {"n_answerable": len(qs), "pack_changed": sum(not rows[(q, 6000)]["pack_eq_lex5b6000"] for q in qs),
                 "net": [sum(lo), sum(up)], "better_worse": [b, w],
                 "nodl": A.tot(A.J1["LEX5B_NODL_6000"], qs), "saved": A.tot(A.J1["LEX5B_6000"], qs)}
allq = [q for q in A.RT.QIDS]
out["all96_pack_changed"] = sum(not rows[(q, 6000)]["pack_eq_lex5b6000"] for q in allq)
out["fallback19_pack_changed"] = sum(not rows[(q, 6000)]["pack_eq_lex5b6000"] for q in allq
                                     if rows[(q, 6000)]["saved_status"] == "fail")
print(json.dumps(out, indent=1))
(A.HERE / "split_nodl.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
