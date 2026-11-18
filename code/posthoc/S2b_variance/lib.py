"""S2b (variance) shared helpers: frozen ragbench imports (read-only), money guard, ledger, LEX5B re-packing,
reader calls, judging. Outputs only under work/posthoc_2026-11/S2b_variance/."""
import copy
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from types import MappingProxyType

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
sys.dont_write_bytecode = True
WT = r"D:\KareOne\RAG-Paper-upgrades"
if WT not in sys.path:
    sys.path.insert(0, WT)

ROOT = Path(r"D:\KareOne\RAG Paper")
WORK = ROOT / "work"
HERE = WORK / "posthoc_2026-11" / "S2b_variance"
FZ = WORK / "datasets_v2" / "test_freeze"
TR = WORK / "datasets_v2" / "test_run"
MAN = json.loads((FZ / "MANIFEST.json").read_text(encoding="utf-8"))
CAP_USD = 1.50
FLOOR_USD = 2.00
LEDGER_PATH = HERE / "ledger.jsonl"
SPEND_PATH = HERE / "spend.json"


def _key():
    for line in open(ROOT / ".env", encoding="utf-8"):
        k, _, v = line.partition("=")
        if k.strip() == "OPENROUTER_API_KEY":
            return v.strip().strip('"').strip("'")
    raise RuntimeError("no key")


os.environ["OPENROUTER_API_KEY"] = _key()
os.environ.pop("READER_FALLBACK_ENV", None)
os.environ["OPENROUTER_PROVIDER_PINS"] = json.dumps(MAN["reader_pins"])

from ragbench import llm_openrouter as LO  # noqa: E402
from ragbench.config import load_config  # noqa: E402
from ragbench.llm import BudgetExceeded, Ledger  # noqa: E402
from ragbench.util import FileLock, sha1_json  # noqa: E402


class StreamBudget:
    """Replaces the shared reader_budget.json (outside this stream's folder) by this stream's own file and cap."""

    def __init__(self, cfg=None):
        self.path = SPEND_PATH
        self.lock = FileLock(str(self.path) + ".lock")

    def state(self):
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {"cost_usd": 0.0, "calls": 0, "by_model": {}}

    def reserve(self, model):
        with self.lock:
            st = self.state()
            if st["cost_usd"] >= CAP_USD:
                raise BudgetExceeded("S2b cap reached: $%.4f of $%.2f" % (st["cost_usd"], CAP_USD))

    def settle(self, model, cost, ptok, ctok):
        with self.lock:
            st = self.state()
            st["cost_usd"] = round(st["cost_usd"] + float(cost or 0.0), 8)
            st["calls"] += 1
            m = st["by_model"].setdefault(model, {"calls": 0, "cost_usd": 0.0, "prompt_tokens": 0,
                                                  "completion_tokens": 0})
            m["calls"] += 1
            m["cost_usd"] = round(m["cost_usd"] + float(cost or 0.0), 8)
            m["prompt_tokens"] += int(ptok or 0)
            m["completion_tokens"] += int(ctok or 0)
            self.path.write_text(json.dumps(st, indent=1), encoding="utf-8")


LO.ReaderBudget = StreamBudget


def spent():
    return StreamBudget().state()["cost_usd"]


def account_remaining():
    import requests
    d = requests.get("https://openrouter.ai/api/v1/key", headers={"Authorization": "Bearer " + _key()},
                     timeout=20).json()["data"]
    return d.get("limit_remaining")


def money_check(tag=""):
    """Before every batch of <= 50 calls. -> True when we may continue."""
    rem = account_remaining()
    ok = (rem is None or float(rem) >= FLOOR_USD) and spent() < CAP_USD
    with open(HERE / "money_checks.jsonl", "a", encoding="utf-8") as fh:
        fh.write(json.dumps({"at": time.strftime("%Y-%m-%dT%H:%M:%S"), "tag": tag, "limit_remaining": rem,
                             "stream_spent": spent(), "ok": ok}) + "\n")
    return ok


CFG = load_config()
LEDGER = Ledger(path=str(LEDGER_PATH))


def jl(p):
    p = Path(p)
    return [json.loads(l) for l in open(p, encoding="utf-8-sig") if l.strip()] if p.is_file() else []


def append(p, row):
    with open(p, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def pool_of(q):
    return "-".join(q.split("-")[2:4])


def msha(messages):
    return hashlib.sha256(json.dumps(messages, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


ARMS = [a["arm"] for a in MAN["arms"]]
VARIANT = {a["arm"]: a["variant"] for a in MAN["arms"]}
QIDS = [json.loads(l)["id"] for l in open(FZ / "questions" / "test_v2run_all.jsonl", encoding="utf-8") if l.strip()]


def primary(arm, q):
    """the saved primary record (latest non-error) of arm x q."""
    best = None
    for f in sorted((TR / "runs" / pool_of(q) / arm).glob("runs_*.jsonl")):
        for r in jl(f):
            if r["question_id"] == q and (best is None or best.get("error") or not r.get("error")):
                best = r
    return best


def pack_path(arm, rec):
    from ragbench.util import safe_name
    return TR / "packs" / safe_name(rec["approach"]) / ("%s__%s.json" % (rec["question_id"], rec["setting"]))


# ------------------------------------------------------------------ reader
_READER = {}


def reader():
    import threading
    k = threading.get_ident()
    if k not in _READER:
        from ragbench.tools import reader_replay as RR
        _READER[k] = RR.make_reader(CFG, LEDGER, "openrouter:" + MAN["primary_reader"],
                                    int(MAN["reader_max_attempts"]))
    return _READER[k]


def new_identity(prim, replicate, method_suffix=None):
    old = dict(prim["meta"]["identity"])
    ident = {k: old[k] for k in ("experiment_id", "question_id", "pool_id", "method", "options", "reader")}
    if method_suffix:
        ident["method"] = old["method"] + "|" + method_suffix
    ident["replicate"] = replicate
    ident["setting"] = old["setting"]
    ident["rid"] = sha1_json(ident)[:20]
    for k in ("code_sig", "question_sig", "pool_sha1"):
        if k in old:
            ident[k] = old[k]
    return ident


def answer_messages(prim, messages, num_predict, replicate, out_file, tag, extra=None, method_suffix=None):
    """Send messages to the pinned GLM reader; write a record (copy of the primary with a new identity / answer)."""
    client = reader()
    abstain = CFG.evidence["abstain_token"]
    err, res = None, None
    with LEDGER.scope(phase="query", approach=prim["approach"] + "|" + tag, question_id=prim["question_id"],
                      setting=prim["setting"]) as sc:
        t0 = time.perf_counter()
        try:
            res = client.chat(messages, purpose="answer", num_predict=num_predict)
        except BudgetExceeded:
            raise
        except Exception as e:  # noqa: BLE001
            err = "%s: %s" % (type(e).__name__, str(e)[:500])
        lat = (time.perf_counter() - t0) * 1000.0
    s = sc.summary()
    ans = (res.text.strip() if res is not None else "") or ""
    rec = copy.deepcopy(prim)
    rec.update(answer_text=ans, abstained=ans.strip() == abstain, error=err, latency_ms=round(lat, 1),
               answer_prompt_tokens=s["prompt_tokens"], answer_completion_tokens=s["completion_tokens"])
    rec.pop("retrieved_units", None)
    ident = new_identity(prim, replicate, method_suffix)
    rec["meta"] = dict(prim["meta"], identity=ident, answer_messages_sha256=msha(messages))
    rec["trace"] = {"s2b": dict(tag=tag, replicate=replicate, provider=getattr(res, "provider", None) if res else None,
                                done_reason=res.done_reason if res else None,
                                cost_usd=res.cost_usd if res else None, **(extra or {}))}
    append(out_file, rec)
    return rec


# ------------------------------------------------------------------ judge
_JUDGE = {}


def judge_client():
    import threading
    k = threading.get_ident()
    if k not in _JUDGE:
        j = MAN["judge"]
        info = LO.model_info(j["model"], CFG.project_root / "work" / "openrouter" / "catalog.json")
        _JUDGE[k] = LO.OpenRouterReader(CFG, LEDGER, j["model"], supported=info.get("params"),
                                        provider=j["provider_pin"], attempts=j["http_attempts"])
    return _JUDGE[k]


def records_by_rid(out_dir):
    """{rid: record} from out_dir/runs.jsonl; the latest line of a rid wins, but an error never replaces an answer."""
    out = {}
    for r in jl(Path(out_dir) / "runs.jsonl"):
        rid = r["meta"]["identity"]["rid"]
        if rid not in out or out[rid].get("error") or not r.get("error"):
            out[rid] = r
    return out


def judge_dir(out_dir, max_model_calls=None):
    """Judge every record in out_dir/runs_*.jsonl with the pinned judge (frozen judge_multi.judge_records)."""
    from ragbench.tools import judge_multi as JM
    from ragbench.util import KVCache
    j = MAN["judge"]
    recs = records_by_rid(out_dir)
    qs = JM.load_question_dicts(FZ / "questions" / MAN["judge_question_file"]["name"])
    jd = Path(out_dir) / "judge"
    return JM.judge_records(recs, qs, judge_client(), j["model"], jd, cache=KVCache(jd / "judge_multi_cache.sqlite"),
                            ledger=LEDGER, provider_pin=j["provider_pin"], max_attempts=j["max_judge_attempts"],
                            on_malformed=j["on_malformed"], max_model_calls=max_model_calls)


def verdicts(out_dir):
    """{(question_id, replicate): (lo, hi, status)} from out_dir's records + judge verdicts (report_test policy)."""
    from ragbench.tools import judge_multi as JM
    recs = records_by_rid(out_dir)
    jd = Path(out_dir) / "judge"
    slug = JM._slug(MAN["judge"]["model"])
    V = {}
    for v in jl(jd / ("verdicts_%s.jsonl" % slug)):
        V.setdefault((v["rid"], v["answer_sha1"]), ("judged", v["strict"]))
    for n in jl(jd / ("needs_adjudication_%s.jsonl" % slug)):
        V.setdefault((n["rid"], n["answer_sha1"]), ("needs_adjudication", None))
    out = {}
    for rid, r in recs.items():
        key = (r["question_id"], r["meta"]["identity"]["replicate"], r["meta"]["identity"]["method"])
        if r.get("error"):
            out[key] = (0, 0, "system_failure")
            continue
        v = V.get((rid, JM.answer_sha1(r)))
        if v is None:
            out[key] = (0, 1, "unjudged")
        elif v[0] == "needs_adjudication":
            out[key] = (0, 1, "needs_adjudication")
        else:
            out[key] = (v[1], v[1], "judged")
    return out


# ------------------------------------------------------------------ LEX5B re-packing (frozen packer)
_CTX = {}
import threading as _th  # noqa: E402
_LOCK = _th.RLock()


def _ctx(set_name, pool_id):
    with _LOCK:
        return _ctx_locked(set_name, pool_id)


def _ctx_locked(set_name, pool_id):
    k = (set_name, pool_id)
    if k not in _CTX:
        from ragbench.approaches.lex4 import core as K
        from ragbench.corpus.registry import load_set
        from ragbench.pools import load_pool, resolve_pool_path
        from ragbench.retrieval import hybrid as H
        cs = load_set(set_name, CFG)
        hx = H.HybridIndex.load(cs, CFG, client=None, refresh_vectors=False)
        _CTX[k] = {"cs": cs, "hx": hx, "P": K.Packager(hx, cs),
                   "pool": load_pool(resolve_pool_path(pool_id, CFG))}
    return _CTX[k]


class _NoClient:
    ledger = LEDGER


_APP = {}


def _approach(budget, pool):
    from ragbench.approaches.budget_matrix import Lex5b_15500
    from ragbench.approaches.lex5b.approach import Lex5b
    k = (budget, id(pool))
    with _LOCK:
        if k not in _APP:
            cls = Lex5b if budget == 6000 else Lex5b_15500
            _APP[k] = cls(cfg=CFG, client=_NoClient(), flags={})
            _APP[k].variants_version = "variants-v2"
            _APP[k].pool = pool
        return _APP[k]


_QOBJ = {}


def question_obj(qid):
    with _LOCK:
        return _question_obj(qid)


def _question_obj(qid):
    if not _QOBJ:
        from ragbench.contracts import load_questions
        for f in FZ.glob("questions/test_v2run_*-*.jsonl"):
            for q in load_questions(f):
                _QOBJ[q.id] = q
    return _QOBJ[qid]


class _Idx:
    def __init__(self, cs, hx):
        self.corpus, self.payload = cs, hx


def lex5b_repack(rec, budget, text, finish, call_failed=False, fail_reason="deadline"):
    """Steps 2, 5, 6, 7 (without the answer call) of Lex5b._answer from the saved F2 top 40 and a selector reply.
    text None or call_failed -> the fallback. -> dict(messages, num_predict, shown, fallback, status, reason, ...)."""
    from ragbench.approaches.lex5 import bundles as B
    from ragbench.approaches.lex5b.approach import LEX5B_DEFAULTS
    from ragbench.contracts import answer_num_predict, build_answer_messages
    st = rec["trace"]["lex5b"]
    C = _ctx(rec["corpus_set"], rec["meta"]["identity"]["pool_id"])
    P = C["P"]
    app = _approach(budget, C["pool"])
    query = app.make_query(question_obj(rec["question_id"]), rec["setting"])
    order = [int(c) for c in st["f2_top40"]]
    pk = {c: MappingProxyType(dict(P.package(c))) for c in order}
    f2_pks = [pk[c] for c in order]
    rank = {c: i + 1 for i, c in enumerate(order)}
    cost = B.Coster(P.cost)
    fallback = B.hybrid_intact(f2_pks, cost, budget)
    present = f2_pks[:LEX5B_DEFAULTS["top"]]
    shown, status, reason, parts = fallback, "fail", fail_reason if call_failed else None, None
    if text is not None and not call_failed:
        pr = B.parse(text, len(present), finish_reason=finish)
        status, reason, parts = pr["status"], pr["reason"], [[list(b) for b in alts] for alts in pr["parts"]]
        if pr["ok"]:
            parts_pk = [[tuple(present[i - 1] for i in b) for b in alts] for alts in pr["parts"]]
            shown, _info = B.pack_exact(parts_pk, rank, cost, budget)
            shown, _finfo = B.fill(shown, parts_pk, f2_pks, cost, budget, bundles=True, rrf=True)
    items = [P.item(p) for p in shown]
    ev = app.pack(items, _Idx(C["cs"], C["hx"]))
    msgs = build_answer_messages(query, ev, CFG)
    return {"messages": msgs, "num_predict": answer_num_predict(query, CFG), "sha": msha(msgs),
            "shown_chunks": [p["chunk"] for p in shown], "fallback_chunks": [p["chunk"] for p in fallback],
            "fallback": status == "fail", "status": status, "reason": reason, "parts": parts,
            "evidence_chars": ev.chars}


# ------------------------------------------------------------------ selector calls
_SEL = {}


def selector(model, deadline_s=3.0, max_tokens_total=600, provider=None):
    with _LOCK:
        LO._bg_loop()
        return _selector_locked(model, deadline_s, max_tokens_total, provider)


def _selector_locked(model, deadline_s, max_tokens_total, provider):
    k = (model, deadline_s, max_tokens_total, json.dumps(provider))
    if k not in _SEL:
        info = LO.model_info(model, CFG.project_root / "work" / "openrouter" / "catalog.json")
        c = LO.OpenRouterReader(CFG, LEDGER, model, supported=info.get("params"), provider=provider, attempts=1,
                                deadline_s=deadline_s, max_tokens_total=max_tokens_total, reasoning_effort="low")
        try:
            c.warm_deadline()
        except Exception:  # noqa: BLE001
            pass
        _SEL[k] = c
    return _SEL[k]


def call_selector(client, prompt_messages, purpose, num_predict):
    """-> dict(text, finish, failed, reason, usage...) mirroring Lex5b._answer step 4."""
    out = {"model": client.model_id}
    t = time.perf_counter()
    try:
        res = client.chat(prompt_messages, purpose=purpose, num_predict=num_predict)
        out.update(text=res.text, finish=res.done_reason, failed=False, prompt_tokens=res.prompt_tokens,
                   completion_tokens=res.completion_tokens, reasoning_tokens=getattr(res, "reasoning_tokens", None),
                   cost_usd=res.cost_usd, provider=getattr(res, "provider", None))
    except BudgetExceeded:
        raise
    except Exception as e:  # noqa: BLE001
        out.update(text=None, finish=None, failed=True,
                   reason="deadline" if "deadline" in str(e).lower() else "call_error", error=str(e)[:300])
    out["t_select_ms"] = round((time.perf_counter() - t) * 1000.0, 1)
    return out
