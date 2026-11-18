"""S7 analysis: shared loaders (no API calls, no GPU).

- questions (frozen judge question file, meta.v2: evidence, bundles, subtype, integration_operation)
- per-arm machine verdicts (lo, hi) for frozen arms (report_test.load/score) and the post-hoc arms of S2a, S3, S6
  (same rules as those streams' own analysis scripts, re-implemented read-only)
- per-arm reader evidence text (exact user message of the answer call)
- gold-evidence coverage of a pack (title-restricted verbatim matching)
- the revision_stats helpers, extracted with ast and executed unchanged (as S2a/report.py does)
"""
import ast
import itertools
import json
import os
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
ROOT = Path(r"D:\KareOne\RAG Paper")
WORK = ROOT / "work"
TR = WORK / "datasets_v2" / "test_run"
FZ = WORK / "datasets_v2" / "test_freeze"
PH = WORK / "posthoc_2026-11"
HERE = PH / "S7_analysis"
FROZEN = r"D:\KareOne\RAG-Paper-upgrades"
os.environ["ANYRAG_TEST_DIR"] = str(TR)
os.environ["ANYRAG_FREEZE_DIR"] = str(FZ)
os.environ["ANYRAG_CODE_DIR"] = FROZEN
for p in (FROZEN, str(TR)):
    if p not in sys.path:
        sys.path.insert(0, p)
import report_test as RT  # noqa: E402
from ragbench.tools import judge_multi as JM  # noqa: E402
from ragbench.util import sha1_json  # noqa: E402
from ragbench.normalize import normalize  # noqa: E402

P = RT.PRIMARY
QS, ANS, AB, QIDS = RT.QS, RT.ANS, RT.AB, RT.QIDS
REC, VER = RT.load()


def jl(p):
    p = Path(p)
    return [json.loads(l) for l in open(p, encoding="utf-8-sig") if l.strip()] if p.is_file() else []


QV2 = {}
QRUN = {}
for d in jl(FZ / "questions" / "test_v2judge_all.jsonl"):
    QV2[d["id"]] = d["meta"]["v2"]
    QRUN[d["id"]] = d


def setting(q):
    return QRUN[q]["meta"]["setting"]


def lang(q):
    return QS[q]["lang"]


def tier(q):
    return QS[q]["tier"]


def pool(q):
    return RT.pool_of(q)


# ------------------------------------------------------------------------------------------------ verdicts
FROZEN_ARMS = list(RT.ARMS)
FROZEN_PACKDIR = {"HYBRID_6000": "HYBRID", "HR_6000": "HYBRID_rerank-0c0612fe", "LEX5B_6000": "LEX5B",
                  "SETR_6000": "SETR", "HYBRID_15500": "HYBRID15500", "HR_15500": "HR15500_rerank-dcb91474",
                  "LEX5B_15500": "LEX5B_15500", "SETR_15500": "SETR_15500"}

# S2a
S2A = PH / "S2a_packs"
S2A_ANS = {}
for r in jl(S2A / "answers.jsonl"):
    if not r.get("error"):
        S2A_ANS[(r["arm"], r["qid"])] = r
S2A_ARMS = sorted({a for a, _ in S2A_ANS})
S2A_NV = {}
for d in [S2A / "judge"] + sorted(p for p in (S2A / "judge").iterdir() if p.is_dir()):
    for k, v in RT.final_verdicts_by_answer(d).items():
        S2A_NV.setdefault(k, v)


def s2a_rid(arm, qid, st):
    ident = {"experiment_id": "posthoc-2026-11-S2a", "question_id": qid, "pool_id": pool(qid), "method": arm,
             "options": {"flags": {}, "variants_version": "variants-v2"},
             "reader": {"chat": "openrouter/z-ai/glm-5.3-flash", "chat_digest": "openrouter"}, "replicate": 0,
             "setting": st}
    return sha1_json(ident)[:20]


def _s2a_score(arm, q):
    r = S2A_ANS.get((arm, q))
    if r is None:
        return None
    if r["source"].startswith("frozen:"):
        lo, hi, _ = RT.score(REC, VER, P, r["source"].split(":", 1)[1], q)
        return lo, hi
    v = S2A_NV.get((s2a_rid(arm, q, r["setting"]), JM.answer_sha1({"answer_text": r["answer_text"]})))
    if v is None or v["status"] == "needs_adjudication":
        return 0, 1
    return v["strict"], v["strict"]


# S6 and S3
def _own(arm_dir_runs, judge_dir):
    rec = {}
    for r in jl(arm_dir_runs):
        old = rec.get(r["question_id"])
        if old is None or old.get("error") or not r.get("error"):
            rec[r["question_id"]] = r
    return rec, RT.final_verdicts_by_answer(judge_dir)


OWN = {}
for a in ("CLOSEDBOOK", "LISTWISE", "SETR_NOFILL"):
    OWN[a] = _own(PH / "S6_baselines" / "test" / a / ("runs_%s.jsonl" % a), PH / "S6_baselines" / "test" / ("judge_%s" % a))
for a in ("GREP_AGENT", "ITER_BM25"):
    OWN[a] = _own(PH / "S3_agentic" / "runs" / a / "test" / ("runs_%s.jsonl" % a), PH / "S3_agentic" / "judge" / a / "test")


def _own_score(arm, q):
    rec, ver = OWN[arm]
    r = rec.get(q)
    if r is None or r.get("error"):
        return 0, 0
    v = ver.get((r["meta"]["identity"]["rid"], JM.answer_sha1(r)))
    if v is None or v["status"] == "needs_adjudication":
        return 0, 1
    return v["strict"], v["strict"]


def score(arm, q):
    if arm in FROZEN_ARMS:
        lo, hi, _ = RT.score(REC, VER, P, arm, q)
        return lo, hi
    if arm in OWN:
        return _own_score(arm, q)
    s = _s2a_score(arm, q)
    if s is None:
        raise KeyError((arm, q))
    return s


def S(arm):
    return lambda q: score(arm, q)


def answer_text(arm, q):
    if arm in FROZEN_ARMS:
        r = REC[P][arm].get(q)
        return r and r.get("answer_text")
    if arm in OWN:
        r = OWN[arm][0].get(q)
        return r and r.get("answer_text")
    r = S2A_ANS.get((arm, q))
    return r and r.get("answer_text")


# ------------------------------------------------------------------------------------------------ packs
_PACK_CACHE = {}


def _load_jsonl_packs(path, key="messages"):
    out = {}
    for r in jl(path):
        out[r["question_id"]] = r[key]
    return out


def messages(arm, q):
    """Exact reader messages of the answer call, or None."""
    if arm in FROZEN_PACKDIR:
        f = TR / "packs" / FROZEN_PACKDIR[arm] / ("%s__%s.json" % (q, setting(q)))
        return json.loads(f.read_text(encoding="utf-8"))["messages"] if f.is_file() else None
    if arm in ("LISTWISE", "SETR_NOFILL", "CLOSEDBOOK"):
        if arm not in _PACK_CACHE:
            _PACK_CACHE[arm] = _load_jsonl_packs(PH / "S6_baselines" / "test" / arm / ("packs_%s.jsonl" % arm))
        return _PACK_CACHE[arm].get(q)
    if arm in ("GREP_AGENT", "ITER_BM25"):
        r = OWN[arm][0].get(q)
        return r and r.get("answer_messages")
    f = S2A / "packs" / arm / ("%s.json" % q)
    return json.loads(f.read_text(encoding="utf-8"))["messages"] if f.is_file() else None


def evidence_block(msgs):
    u = msgs[1]["content"]
    if not u.startswith("Evidence:\n"):
        return ""
    i = u.rfind("\n\nQuestion: ")
    return u[len("Evidence:\n"):i]


_HDR = re.compile(r"^\[(\d+)\] (.*)$")


def parse_items(ev):
    """-> [(header_without_number, body)] in pack order (headers numbered 1..n consecutively)."""
    items, cur, want = [], None, 1
    for line in ev.split("\n"):
        m = _HDR.match(line)
        if m and int(m.group(1)) == want:
            if cur is not None:
                items.append(cur)
            cur = [m.group(2), []]
            want += 1
            continue
        if cur is not None:
            cur[1].append(line)
    if cur is not None:
        items.append(cur)
    return [(h, "\n".join(b)) for h, b in items]


def _n(s):
    return normalize(s or "")


MIN_SEG = 12


def coverage(quote, hay):
    """Fraction of the normalised quote's characters covered by verbatim segments (>= MIN_SEG chars, or the whole
    quote if shorter) found in hay (greedy longest-prefix matching)."""
    qn = _n(quote)
    if not qn:
        return 0.0
    if qn in hay:
        return 1.0
    if len(qn) <= MIN_SEG:
        return 0.0
    i = cov = 0
    n = len(qn)
    while i < n:
        lo, hi = 0, n - i                  # largest L with qn[i:i+L] in hay
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if qn[i:i + mid] in hay:
                lo = mid
            else:
                hi = mid - 1
        if lo >= MIN_SEG:
            cov += lo
            i += lo
        else:
            i += 1
    return cov / n


PRESENT = 0.9      # an evidence item reached the reader if >= 90% of its quote is verbatim in a same-document item


def gold_reach(q, msgs):
    """-> dict(cls in {complete, partial, none}, best_frac, items={eid: coverage}, bundles=[frac])."""
    v2 = QV2[q]
    ev = evidence_block(msgs) if msgs else ""
    items = parse_items(ev)
    by_title = defaultdict(list)
    for h, b in items:
        by_title[h].append(_n(b))
    cov = {}
    for e in v2["evidence"]:
        t = _n(e.get("doc_title") or "")
        bodies = [b for h, bl in by_title.items() if _n(h) == t or _n(h).startswith(t + " > ") for b in bl]
        cov[e["id"]] = coverage(e["quote"], "\x00".join(bodies)) if bodies else 0.0
    fr = []
    for bun in v2["evidence_bundles"]:
        fr.append(sum(cov[x] >= PRESENT for x in bun) / len(bun))
    best = max(fr) if fr else 0.0
    cls = "complete" if best == 1.0 else ("partial" if best > 0 else "none")
    return {"cls": cls, "best_frac": best, "items": cov, "bundles": fr, "n_items": len(items)}


# ------------------------------------------------------------------------------------------------ statistics
SRC = ROOT / "paper" / "analysis" / "revision_stats.py"
_tree = ast.parse(SRC.read_text(encoding="utf-8"))
_want = {"find", "sign_p", "diff_vectors", "sign_range", "boot_mean", "compare"}
_mod = ast.Module(body=[n for n in _tree.body if isinstance(n, ast.FunctionDef) and n.name in _want], type_ignores=[])
NS = {"RT": RT, "itertools": itertools, "random": random, "defaultdict": defaultdict,
      "N_BOOT": 10000, "SEED": 0, "ANS": ANS, "parent": {}}
exec(compile(_mod, str(SRC), "exec"), NS)
_constituents = {q: set(QV2[q]["source_family"].split("+")) for q in ANS}
for q in ANS:
    c = sorted(_constituents[q])
    for o in c[1:]:
        NS["parent"][NS["find"](o)] = NS["find"](c[0])
NS["comp"] = {q: NS["find"](sorted(_constituents[q])[0]) for q in ANS}
NS["block"] = {q: QV2[q]["source_block"] for q in ANS}
compare = NS["compare"]
sign_p = NS["sign_p"]


def total(arm, qs=ANS):
    lo = sum(score(arm, q)[0] for q in qs)
    hi = sum(score(arm, q)[1] for q in qs)
    return lo, hi


def fmt(lo, hi=None):
    if hi is None:
        lo, hi = lo
    return "%d" % lo if lo == hi else "%d..%d" % (lo, hi)


def fmt_net(c):
    lo, hi = c["net_bounds"]
    s = lambda x: "%+d" % x
    return s(lo) if lo == hi else "%s..%s" % (s(lo), s(hi))


def fmt_p(c):
    a, b = c["sign_p_range"]
    return "%.3f" % a if abs(a - b) < 5e-4 else "%.3f-%.3f" % (a, b)


def fmt_ci(c, k="ci_pp_question"):
    a, b = c[k]
    return "[%.1f, %.1f]" % (a, b)


def wjson(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=1), encoding="utf-8")
