"""Query-time LLM tokens (frozen per-call bound rule, report_test.token_cell) and call counts of every GLM primary
cell, cached to S4/frozen_cell_tokens.json. No model calls."""
import json
import common as C
import stats_common as S

RT = S.RT
REC, VER = S.load()
out = {}
for a in RT.ARMS:
    out[a] = {}
    for q in S.QIDS:
        lo, hi, unk = RT.token_cell(REC, S.P, a, q)
        calls = RT.primary_calls(q, a)
        r = REC[S.P][a].get(q) or {}
        out[a][q] = {"tok_lo": lo, "tok_hi": hi, "unknown_calls": unk, "calls": len(calls),
                     "latency_ms": r.get("latency_ms")}
json.dump(out, open(C.S4 / "frozen_cell_tokens.json", "w", encoding="utf-8"), indent=0)
for a in RT.ARMS:
    print(a, sum(v["tok_lo"] for v in out[a].values()), sum(v["tok_hi"] for v in out[a].values()),
          sum(v["calls"] for v in out[a].values()))
