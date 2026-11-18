"""Synthetic tests for human_estimate.py (no model calls, no real human data).

    python test_human_estimate.py      -> writes synthetic/ inputs + outputs, prints PASS/FAIL lines
1. Parsers: packet A as the page Export JSON, packet B as a review-game answercheck JSONL (with a "half correct" note,
   a "partial" verdict and an unsure cell).
2. Identity: when the human equals the judge on the sample, PPI = naive and its SE = 0.
3. Coverage: Monte Carlo over 4,000 synthetic finite populations of 60 questions (judge stricter than the human with
   prob 0.25 on true-correct cells, laxer with prob 0.03), sampling 24 without replacement; the 95% PPI interval should
   cover the true total about 95% of the time (FPC included). Same for the RG estimator where defined.
4. Kappa: hand-checked values.
"""
import json
import random
from pathlib import Path

import human_estimate as HE
import j2lib as L

HERE = Path(__file__).resolve().parent
SYN = HERE / "synthetic"
SYN.mkdir(exist_ok=True)
ok = True


def check(name, cond, info=""):
    global ok
    ok &= bool(cond)
    print(("PASS " if cond else "FAIL ") + name + ("  " + info if info else ""))


keyA = json.loads((HERE / "packets/A_concordant24/key_A_concordant24.json").read_text(encoding="utf-8"))
keyB = json.loads((HERE / "packets/B_second_reviewer/key_B_second_reviewer.json").read_text(encoding="utf-8"))
rnd = random.Random(5)
FLIP = {"HYBRID_6000": 0.10, "HR_6000": 0.10, "LEX5B_6000": 0.10, "SETR_6000": 0.35}

# ---- 1. synthetic packet A (page export JSON): human = judge 1 with judge-strict flips, SETR flipped more often
res = []
for k in keyA["items_key"]:
    a = k["arms"][0]
    m = L.judge1(a, k["qid"])[0]
    h = 1 if (m == 1 and rnd.random() > 0.02) or (m == 0 and rnd.random() < FLIP[a]) else 0
    res.append({"item": k["item"], "verdict": "correct" if h else "wrong", "note": ""})
res[0]["note"] = "Half correct."
res[1]["verdict"] = "unsure"
(SYN / "A_export.json").write_text(json.dumps({"set": keyA["set"], "seconds": 1, "results": res}), encoding="utf-8")
# ---- synthetic packet B (review-game answercheck JSONL): reviewer 2 = reviewer 1 with 10% disagreement
H1, R = L.human_sym()
lines = []
for i, k in enumerate(keyB["items_key"]):
    h = H1[(k["arms"][0], k["qid"])]
    if rnd.random() < 0.10:
        h = 1 - h
    v = "correct" if h else ("partial" if i % 2 else "wrong")
    lines.append({"ts": "x", "qid": k["qid"], "answer_id": k["answer_id"], "points": {}, "unsupported_claim": False,
                  "verdict": v, "note": ""})
lines[2]["verdict"] = None
(SYN / "B_review.answercheck.jsonl").write_text("\n".join(json.dumps(x) for x in lines) + "\n", encoding="utf-8")

HA, unA = HE.read_review(SYN / "A_export.json", keyA)
check("A parser: cells", len(HA) + unA == 96, "usable %d unusable %d" % (len(HA), unA))
first = keyA["items_key"][0]
check("A parser: half-correct note -> 0", all(HA[(a, first["qid"])] == 0 for a in first["arms"]))
HB, unB = HE.read_review(SYN / "B_review.answercheck.jsonl", keyB)
check("B parser: cells", len(HB) + unB == 120, "usable %d unusable %d" % (len(HB), unB))
out = HE.run(SYN / "A_export.json", SYN / "B_review.answercheck.jsonl", 1, SYN / "out")
check("run(): per-pipeline rows", set(out["per_pipeline"]) == set(L.ARMS4))
print((SYN / "out" / "human_estimate_judge1.md").read_text(encoding="utf-8"))

# ---- 2. identity
Mc = [rnd.randint(0, 1) for _ in range(60)]
idx = rnd.sample(range(60), 24)
e, se = HE.ppi_total(Mc, [Mc[i] for i in idx], [Mc[i] for i in idx], 60)
check("identity: PPI = naive, se = 0", abs(e - sum(Mc)) < 1e-9 and se == 0)

# ---- 3. coverage
cov = cov_rg = n_rg = 0
T = 4000
for t in range(T):
    r = random.Random(1000 + t)
    H = [1 if r.random() < 0.8 else 0 for _ in range(60)]
    M = [(0 if r.random() < 0.25 else 1) if h else (1 if r.random() < 0.03 else 0) for h in H]
    s = r.sample(range(60), 24)
    e, se = HE.ppi_total(M, [M[i] for i in s], [H[i] for i in s], 60)
    cov += (e - 1.96 * se <= sum(H) <= e + 1.96 * se)
    g = HE.rg_total(M, [M[i] for i in s], [H[i] for i in s], 60)
    if g is not None:
        n_rg += 1
        cov_rg += (g["estimate"] - 1.96 * g["se"] <= sum(H) <= g["estimate"] + 1.96 * g["se"])
check("PPI 95% coverage in [0.90, 0.98]", 0.90 <= cov / T <= 0.98, "coverage %.3f" % (cov / T))
print("info RG coverage %.3f over %d defined runs (RG is not FPC-exact; reported for information)" % (cov_rg / max(1, n_rg), n_rg))

# ---- 4. kappa
check("kappa identical = 1", abs(L.kappa([(1, 1), (0, 0), (1, 1)]) - 1) < 1e-12)
# po = 0.75, px = 0.5, py = 0.75 -> pe = 0.5 -> kappa 0.5
check("kappa hand value 0.5", abs(L.kappa([(1, 1), (1, 1), (0, 1), (0, 0)]) - 0.5) < 1e-12)
check("kappa undefined -> None", L.kappa([(1, 1), (1, 1)]) is None)
print("ALL PASS" if ok else "SOME FAILED")
