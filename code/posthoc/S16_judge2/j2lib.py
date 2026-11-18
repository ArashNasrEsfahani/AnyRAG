"""Shared loaders for S1: judge-1 (frozen, pinned gpt-oss-120b) and judge-2 (this stream) cell verdicts, the symmetric
human merge, and small statistics helpers (Cohen's kappa, exact sign test, components). No model calls.
Cell value convention: (lo, hi) with lo == hi in {0, 1} for a resolved cell; (0, 1) for an unresolved one."""
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
TR = REPO / "work" / "datasets_v2" / "test_run"
sys.path.insert(0, str(TR))
import report_test as RT  # noqa: E402
from ragbench.tools import judge_multi as JM  # noqa: E402

ARMS4 = ["HYBRID_6000", "HR_6000", "LEX5B_6000", "SETR_6000"]
P = RT.PRIMARY
J2_MODEL, J2_PIN = "mistralai/mistral-large-2512", "mistral/zdr"
_CACHE = {}


def rec_ver():
    if "rv" not in _CACHE:
        _CACHE["rv"] = RT.load()
    return _CACHE["rv"]


def judge1(a, q):
    REC, VER = rec_ver()
    lo, hi, _ = RT.score(REC, VER, P, a, q)
    return lo, hi


def _j2_rows():
    if "j2" in _CACHE:
        return _CACHE["j2"]
    slug = JM._slug(J2_MODEL)
    rows, pend = {}, {}
    vs, ns = [], []
    for d in sorted((HERE.parent / "S1_evaluator" / "judge2").glob("shard_*")):          # S16 copy: read S1's verdicts
        vs += JM.read_jsonl(d / ("verdicts_%s.jsonl" % slug))
        ns += JM.read_jsonl(d / ("needs_adjudication_%s.jsonl" % slug))
    for v in sorted(vs, key=lambda v: v["at"]):            # FIRST valid verdict of a key is final (as for judge 1)
        if v.get("judge") == J2_MODEL and v.get("judge_provider_pin") == J2_PIN:
            rows.setdefault((v["rid"], v["answer_sha1"]), v)
    for n in sorted(ns, key=lambda v: v["at"]):
        pend.setdefault((n["rid"], n["answer_sha1"]), n)
    _CACHE["j2"] = (rows, pend)
    return rows, pend


def judge2(a, q):
    """-> (lo, hi, status) with the test report's rules: missing record / run error = 0; unjudged or
    needs_adjudication = bounds (0, 1)."""
    REC, _ = rec_ver()
    r = REC[P][a].get(q)
    if r is None or r.get("error"):
        return 0, 0, "system_failure"
    k = (r["meta"]["identity"]["rid"], JM.answer_sha1(r))
    rows, pend = _j2_rows()
    if k in rows:
        return rows[k]["strict"], rows[k]["strict"], "judged"
    if k in pend:
        return 0, 1, "needs_adjudication"
    return 0, 1, "unjudged"


def human_sym():
    """{(arm, q): 0/1} for the 120 human-reviewed cells (symmetric merge, half-correct rule applied)."""
    sym = json.loads((TR / "test_results_human_sym.json").read_text(encoding="utf-8"))
    out = {}
    for a, d in sym["per_question_glm_6000_sym"].items():
        for q, (lo, hi, st) in d.items():
            if st.startswith("human_"):
                assert lo == hi
                out[(a, q)] = lo
    return out, sym["selected_questions"]


def human_sym_cell(a, q):
    sym = _CACHE.setdefault("sym", json.loads((TR / "test_results_human_sym.json").read_text(encoding="utf-8")))
    lo, hi, _ = sym["per_question_glm_6000_sym"][a][q]
    return lo, hi


def kappa(pairs):
    """Cohen's kappa for binary pairs [(x, y)]; None when undefined (no variation)."""
    n = len(pairs)
    if not n:
        return None
    po = sum(1 for x, y in pairs if x == y) / n
    px, py = sum(x for x, _ in pairs) / n, sum(y for _, y in pairs) / n
    pe = px * py + (1 - px) * (1 - py)
    return None if pe >= 1 else (po - pe) / (1 - pe)


def sign_p(b, w):
    return RT.sign_p(b, w)


def components(qs):
    """Connected components of shared source families (same construction as paper/analysis/revision_stats.py)."""
    QROWS = {}
    for line in open(REPO / "work" / "datasets_v2" / "test_freeze" / "questions" / "test_v2judge_all.jsonl",
                     encoding="utf-8-sig"):
        if line.strip():
            d = json.loads(line)
            QROWS[d["id"]] = d["meta"]["v2"]
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    cons = {q: sorted(set(QROWS[q]["source_family"].split("+"))) for q in qs}
    for q in qs:
        for o in cons[q][1:]:
            parent[find(o)] = find(cons[q][0])
    return {q: find(cons[q][0]) for q in qs}


def wilson(k, n, z=1.96):
    if n == 0:
        return None
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h
