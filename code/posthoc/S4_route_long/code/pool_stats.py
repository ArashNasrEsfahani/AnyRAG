"""Pool sizes (chars, estimated reader tokens) and gold-document sizes per test question. No model calls."""
import json
import statistics as st
from collections import defaultdict
import common as C

# tokens per character of GLM answer calls in the frozen test (primary reader ledgers), by language
rat = defaultdict(list)
for f in (C.TR / "runs").rglob("ledger_*.jsonl"):
    for c in C.jl(f):
        if c.get("purpose") == "answer" and c.get("usage_known") and c.get("prompt_chars"):
            lang = c["setting"].split("-")[0]
            rat[lang].append(c["prompt_tokens"] / c["prompt_chars"])
tpc = {l: st.median(v) for l, v in rat.items()}
out = {"tokens_per_char_glm_answer_median": tpc, "n_calls": {l: len(v) for l, v in rat.items()},
       "tokens_per_char_glm_answer_p90": {l: sorted(v)[int(0.9 * len(v))] for l, v in rat.items()},
       "fa_over_en": tpc["fa"] / tpc["en"], "pools": {}, "questions": {}}
Q = C.test_questions()
for pool in C.POOLS:
    docs = C.pool_docs(pool)
    lang = pool.split("-")[1]
    by = {d.id: d for d in docs}
    chars = [len(d.text) for d in docs]
    out["pools"][pool] = {"docs": len(docs), "chars": sum(chars), "est_tokens": round(sum(chars) * tpc[lang]),
                          "doc_chars_min": min(chars), "doc_chars_median": st.median(chars), "doc_chars_max": max(chars),
                          "doc_est_tokens_max": round(max(chars) * tpc[lang])}
    for qid, q in Q.items():
        if q["pool_id"] != pool:
            continue
        units = [u for v in q["gold_units"].values() for u in v]
        gdocs = sorted(q["meta"]["v2"]["doc_scope"]["oracle_docs"])
        miss = [u for u in units if not by[u.split("::")[0]].has_unit(u)]
        gc = sum(len(by[d].text) for d in gdocs)
        out["questions"][qid] = {"pool": pool, "tier": q["meta"]["tier"], "lang": lang, "gold_docs": gdocs,
                                 "gold_units": units, "units_missing": miss, "gold_chars": gc,
                                 "gold_est_tokens": round(gc * tpc[lang])}
json.dump(out, open(C.S4 / "pool_stats.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(json.dumps({k: out[k] for k in ("tokens_per_char_glm_answer_median", "n_calls", "fa_over_en")}, indent=1))
for p, v in out["pools"].items():
    print(p, v)
qs = out["questions"]
print("units missing", sum(len(v["units_missing"]) for v in qs.values()))
for cap in (32000, 64000, 100000, 128000, 200000, 1000000):
    tot = sum(min(v["gold_est_tokens"], cap) for v in qs.values() if v["gold_docs"])
    over = sum(v["gold_est_tokens"] > cap for v in qs.values() if v["gold_docs"])
    print("cap", cap, "gold tokens total", tot, "USD@0.15/M", round(tot * 0.15e-6, 3), "questions over cap", over)
