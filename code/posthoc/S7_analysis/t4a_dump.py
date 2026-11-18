"""Task 4a, step 1: collect every judge-1 / judge-2 disagreement on the primary reader (8 frozen cells, resolved under
both judges) with both judges' units, claims and reasons, and a structural cause of the stricter judge's 0.
-> j12_disagreements.json, j12_disagreements_dump.md (for the manual reading in t4a_classify.py)"""
import json
import sys
from collections import Counter

import s7lib as L

sys.path.insert(0, str(L.PH / "S1_evaluator"))
import j2lib as J2  # noqa: E402

J2._CACHE["rv"] = (L.REC, L.VER)
rows2, _ = J2._j2_rows()


def j1_verdict(a, q):
    r = L.REC[L.P][a][q]
    return L.VER[L.P].get((r["meta"]["identity"]["rid"], L.JM.answer_sha1(r)))


def structural(v, req):
    """Why this verdict is strict 0: flags from its units/claims."""
    if v.get("source") != "judge":
        return ["rule:" + str(v.get("source"))]
    u = v.get("units") or {}
    fl = []
    for rid in req:
        s = u.get(rid)
        if s != "correct":
            fl.append("req_%s" % s)
    for c in v.get("claims") or []:
        if c["status"] in ("wrong_value", "wrong_binding", "contradiction", "unsupported"):
            fl.append("claim_" + c["status"])
    return sorted(set(fl)) or ["none"]


out = []
for a in L.FROZEN_ARMS:
    for q in L.QIDS:
        r = L.REC[L.P][a][q]
        j1 = L.score(a, q)
        j2 = J2.judge2(a, q)
        if j1[0] != j1[1] or j2[0] != j2[1] or j1[0] == j2[0]:
            continue
        v1 = j1_verdict(a, q)["verdict"]
        v2 = rows2[(r["meta"]["identity"]["rid"], L.JM.answer_sha1(r))]
        req = [x["id"] for x in L.QV2[q].get("requirements") or [] if x.get("required")]
        strict_v, lax_v = (v1, v2) if j1[0] == 0 else (v2, v1)
        out.append(dict(arm=a, q=q, tier=L.tier(q), lang=L.lang(q), stricter="J1" if j1[0] == 0 else "J2",
                        question=L.QS[q]["gloss"] or L.QS[q]["question"], reference=L.QS[q]["reference_gloss"] or L.QS[q]["reference"],
                        answer=r.get("answer_text"), structural=structural(strict_v, req),
                        lax_units=lax_v.get("units"), strict_units=strict_v.get("units"),
                        strict_claims=[c for c in strict_v.get("claims") or [] if c["status"] != "correct"],
                        strict_reason=strict_v.get("reason"), lax_reason=lax_v.get("reason"),
                        requirements={x["id"]: x["proposition"] for x in L.QV2[q].get("requirements") or []}))
L.wjson(L.HERE / "j12_disagreements.json", out)
print(len(out), Counter(o["stricter"] for o in out))
md = []
for i, o in enumerate(out):
    md += ["## %d. %s %s (%s %s) stricter=%s structural=%s" % (i, o["arm"], o["q"], o["tier"], o["lang"], o["stricter"], o["structural"]),
           "Q: " + str(o["question"]), "REF: " + str(o["reference"]), "ANS: " + str(o["answer"]),
           "REQ: " + json.dumps(o["requirements"], ensure_ascii=False),
           "strict units: %s | lax units: %s" % (o["strict_units"], o["lax_units"]),
           "strict claims (non-correct): " + json.dumps(o["strict_claims"], ensure_ascii=False),
           "strict reason: " + str(o["strict_reason"]), "lax reason: " + str(o["lax_reason"]), ""]
(L.HERE / "j12_disagreements_dump.md").write_text("\n".join(md), encoding="utf-8")
