"""S11 (post-hoc, Nov 2026): query rewriting on top of the frozen HYBRID pipeline at 6,000 characters.

Shared pieces: env rules, question loading (S9's make_query, copied), the pinned GLM call (S9's glm(), copied, with this
stream's ledger and cap), the frozen HybridIndex, the HYBRID offer+pack at 6,000 (S9 align.py offer_pack, which
reproduced the saved HYBRID_6000 packs 96/96), and the two rewrite prompts.
"""
import os

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_k] = "2"
import ctypes  # noqa: E402
import datetime as dt  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import re  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

try:
    ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x00004000)
except Exception:  # noqa: BLE001
    pass

CODE = r"D:\KareOne\RAG-Paper-upgrades"
if CODE not in sys.path:
    sys.path.insert(0, CODE)

import numpy as np  # noqa: E402
import requests  # noqa: E402

from ragbench.config import load_config  # noqa: E402
from ragbench.contracts import (Query, Setting, answer_format, answer_num_predict,  # noqa: E402
                                build_answer_messages, load_questions, pack_evidence)
from ragbench.corpus.registry import load_set, set_name_for  # noqa: E402
from ragbench.normalize import normalize  # noqa: E402
from ragbench.pools import load_pool, resolve_pool_path  # noqa: E402
from ragbench.retrieval import hybrid as H  # noqa: E402

HERE = Path(__file__).resolve().parent
WORK = HERE.parent.parent
DS = WORK / "datasets_v2"
TR = DS / "test_run"
LEDGER = HERE / "ledger.jsonl"
STREAM = "S11_rewrite"
MODEL = "z-ai/glm-5.3-flash"
PINS = json.loads((DS / "test_freeze" / "reader_pins.json").read_text(encoding="utf-8"))["pins"]
CATALOG = json.loads((WORK / "openrouter" / "catalog.json").read_text(encoding="utf-8"))
QFILES = {"dev": DS / "devrun" / "questions", "test": DS / "test_freeze" / "questions"}
POOLS = {"RPT-fa": ("C", "fa-fa"), "RPT-en": ("D", "en-en"), "BOOK-fa": ("K", "fa-fa"), "BOOK-en": ("K", "en-en"),
         "LAW-fa": ("L", "fa-fa"), "LAW-en": ("L", "en-en")}
STOP_USD = 0.30
FLOOR_USD = 2.00
BUDGET = 6000
K = 40           # HYBRID_DEFAULTS candidates
DEPTH = 40       # HYBRID_DEFAULTS depth
cfg = load_config()


# ------------------------------------------------------------------------------------------------ money + GLM
def _key():
    for line in open(r"D:\KareOne\RAG Paper\.env", encoding="utf-8"):
        k, _, v = line.partition("=")
        if k.strip() == "OPENROUTER_API_KEY":
            return v.strip().strip('"').strip("'")
    raise SystemExit("no OPENROUTER_API_KEY")


def key_remaining():
    d = requests.get("https://openrouter.ai/api/v1/key", headers={"Authorization": "Bearer " + _key()},
                     timeout=20).json()["data"]
    return float(d["limit_remaining"]) if d.get("limit_remaining") is not None else None


def spent():
    tot = 0.0
    if LEDGER.is_file():
        for line in open(LEDGER, encoding="utf-8"):
            if line.strip():
                tot += float(json.loads(line).get("cost_usd") or 0.0)
    return tot + judge_spent()


def judge_spent():
    tot = 0.0
    for f in ((HERE / "judge").rglob("judge_ledger_*.jsonl") if (HERE / "judge").is_dir() else []):
        for line in open(f, encoding="utf-8"):
            if line.strip():
                tot += float(json.loads(line).get("cost_usd") or 0.0)
    return tot


class Stop(Exception):
    pass


SUPPORTED = set(next(r for r in CATALOG if r["id"] == MODEL)["params"]) & set(PINS[MODEL]["params"])


def glm(messages, *, purpose, ctx, num_predict=None, max_tokens=None):
    """One pinned GLM call (S9 hybrid_agent.glm, copied): params filtered by the pinned endpoint's list, reasoning
    effort low (excluded), provider pin, no fallbacks, require_parameters. Logged to ledger.jsonl."""
    if spent() >= STOP_USD:
        raise Stop("stream cap $%.2f reached" % STOP_USD)
    g = cfg.generation
    p = {"model": MODEL, "messages": messages, "usage": {"include": True},
         "max_tokens": int(max_tokens or (int(num_predict) + 3000))}
    for k in ("temperature", "top_p", "seed"):
        if k in SUPPORTED:
            p[k] = g[k]
    p["reasoning"] = {"effort": "low", "exclude": True}
    p["provider"] = {"order": [PINS[MODEL]["provider"]], "allow_fallbacks": False, "require_parameters": True}
    headers = {"Authorization": "Bearer " + _key(), "Content-Type": "application/json", "X-Title": "ragbench-posthoc"}
    t0 = time.perf_counter()
    data, last, attempt = None, None, 0
    for attempt in range(1, 8):
        try:
            r = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=p,
                              timeout=(15, 240))
            if r.status_code == 429 or r.status_code >= 500:
                last = "HTTP %d %s" % (r.status_code, r.text[:200])
            elif r.status_code >= 400:
                last = "HTTP %d %s" % (r.status_code, r.text[:300])
                break
            else:
                data = r.json()
                if data.get("error") or not data.get("choices"):
                    last, data = "error body %s" % str(data.get("error"))[:200], None
                else:
                    break
        except (requests.RequestException, ValueError) as e:
            last = "%s: %s" % (type(e).__name__, str(e)[:200])
        time.sleep(min(60.0, 5.0 * 2 ** (attempt - 1)))
    wall = (time.perf_counter() - t0) * 1000.0
    row = dict(ts=dt.datetime.now().isoformat(timespec="seconds"), stream=STREAM, model=MODEL, purpose=purpose,
               attempts=attempt, wall_ms=round(wall, 1), max_tokens=p["max_tokens"],
               prompt_chars=sum(len(m.get("content") or "") for m in messages), **ctx)
    if data is None:
        row.update(ok=False, error=last, usage_known=False, prompt_tokens=0, completion_tokens=0, cost_usd=0.0)
        _log(row)
        raise RuntimeError("GLM call failed: %s" % last)
    ch = data["choices"][0]
    msg = ch.get("message") or {}
    u = data.get("usage") or {}
    content = msg.get("content") or ""
    if "</think>" in content:
        content = content.rsplit("</think>", 1)[1]
    row.update(ok=True, provider=data.get("provider"), usage_known=bool(u),
               prompt_tokens=int(u.get("prompt_tokens") or 0), completion_tokens=int(u.get("completion_tokens") or 0),
               reasoning_tokens=(u.get("completion_tokens_details") or {}).get("reasoning_tokens"),
               cost_usd=float(u.get("cost") or 0.0), finish=ch.get("finish_reason"), request_id=data.get("id"))
    _log(row)
    return dict(text=content.strip(), finish=ch.get("finish_reason"), row=row)


def _log(row):
    with open(LEDGER, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def jl(p):
    p = Path(p)
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()] if p.is_file() else []


# ------------------------------------------------------------------------------------------------ questions
_SETS, _HX = {}, {}


def corpus_set(name):
    if name not in _SETS:
        _SETS[name] = load_set(name, cfg)
    return _SETS[name]


def hx_of(name):
    if name not in _HX:
        _HX[name] = H.HybridIndex.load(corpus_set(name), cfg, client=None, refresh_vectors=False)
    return _HX[name]


def load_split(split):
    out = []
    for pid in POOLS:
        f = QFILES[split] / ("%s_v2run_%s.jsonl" % (split, pid))
        for q in load_questions(f):
            out.append((q, pid))
    return out


def make_query(q, pid):
    _corpus, setting = POOLS[pid]
    s = Setting.parse(setting)
    sn = set_name_for(q.corpus, s.d_lang)
    pool = load_pool(resolve_pool_path(pid, cfg))
    scope = pool.mask(q.scope(s.d_lang), sn)
    return Query(question_id=q.id, text=q.text(s.q_lang), q_lang=s.q_lang, d_lang=s.d_lang, setting=s.name,
                 corpus=q.corpus, corpus_set=sn, doc_scope=scope, answer_format=answer_format(q), pool_id=pid)


def queries(split, ids=None):
    out = []
    for q, pid in load_split(split):
        if ids and q.id not in ids:
            continue
        out.append(make_query(q, pid))
    return out


# ------------------------------------------------------------------------------------------------ rewrite prompts
LANG = {"fa": "Persian", "en": "English"}
GENRE = {"RPT": "a company annual report", "BOOK": "a book", "LAW": "a law or regulation"}
GENRE_PL = {"RPT": "company annual reports", "BOOK": "books", "LAW": "laws and regulations"}


def genre_of(query):
    return query.pool_id.split("-")[0]


HYDE_PROMPT = (
    "Write a short passage (about 60 to 120 words) that answers the question below, written the way the answer would "
    "appear in {genre}. Write it in {lang}, the language of the question. If you do not know the facts, write a "
    "plausible passage anyway with the kind of names, numbers and terms such a document would contain. Do not refuse, "
    "do not add notes or a title. Output only the passage.\n\nQuestion: {q}")

MQ_PROMPT = (
    "A search system looks for passages in {genre_pl} to answer the question below. Write 3 different search queries "
    "for it, in {lang} (the language of the question). Each query is either a paraphrase of the whole question with "
    "different wording, or a sub-question that targets one part of the information the question needs (one subject, "
    "period, document or value). Output exactly 3 lines, one query per line, with no numbering and nothing else."
    "\n\nQuestion: {q}")

REWRITE_MAX_TOKENS = 2000


def rewrite_messages(method, query):
    g = genre_of(query)
    if method == "HYDE":
        c = HYDE_PROMPT.format(genre=GENRE[g], lang=LANG[query.q_lang], q=query.text.strip())
    else:
        c = MQ_PROMPT.format(genre_pl=GENRE_PL[g], lang=LANG[query.q_lang], q=query.text.strip())
    return [{"role": "user", "content": c}]


_NUM = re.compile(r"^\s*(?:[-*\u2022]|\(?[0-9\u06f0-\u06f9]+[.)\-:\u066b]|[Qq]uery\s*[0-9]+\s*[:.])\s*")


def parse_mq(text, n=3):
    lines = []
    for ln in (text or "").splitlines():
        s = _NUM.sub("", ln).strip().strip('"').strip("\u201c\u201d\u00ab\u00bb").strip()
        if s:
            lines.append(s)
    return lines[:n]


# ------------------------------------------------------------------------------------------------ retrieval
def l2(v):
    v = np.asarray(v, np.float32)
    n = float(np.linalg.norm(v))
    return v / (n if n > 0 else 1.0)


def bm25_list(hx, text, mask, depth=DEPTH):
    """The frozen search()'s BM25 channel (same scores, floor, tie rule)."""
    s, _ = hx.bm25_scores(text)
    valid = mask & (s > 0.0)
    order = np.lexsort((np.arange(hx.n), -s))
    return [int(i) for i in order[valid[order]][:depth]]


def dense_list(hx, qvec, mask, depth=DEPTH):
    """The frozen search()'s dense channel (query re-normalised, E @ q, chunks with a vector, tie rule)."""
    q = l2(qvec)
    s = (hx.E @ q).astype(np.float32)
    valid = mask & hx.has_vec
    order = np.lexsort((np.arange(hx.n), -s))
    return [int(i) for i in order[valid[order]][:depth]]


def fused_items(hx, rankings, k=K):
    """rankings {channel: [chunk]} -> RetrievedItems of the top k by the frozen rrf_fuse (k=60)."""
    fused = H.rrf_fuse(rankings, k=H.RRF_K)
    items, top = [], []
    for r, (i, rrf, rk) in enumerate(fused[:k]):
        h = hx.hit_for_chunk(i)
        h.rrf = rrf
        h.ranks = dict(rk)
        items.append(h.to_item(rank=r))
        top.append([int(i), round(float(rrf), 6)] + [rk[c] for c in rankings])
    return items, top


def offer_pack(items, corpus, budget=BUDGET):
    """HybridRetrieval._offer (B2/B3 fill rule) + final_pack (pack_evidence), as S9 align.offer_pack."""
    from ragbench.approaches.baselines import DEFAULTS as BLD
    k = int(cfg.chunking["top_k"])
    offered = list(items[:k])
    cap = int(BLD["max_candidates"])
    nxt = len(offered)
    pk = lambda its: pack_evidence(its, corpus, budget=budget, cfg=cfg)  # noqa: E731
    if BLD["fill_budget"]:
        while nxt < min(len(items), cap):
            p1 = pk(offered)
            if p1.last_item_cut or p1.n_included < len(offered):
                break
            offered.append(items[nxt])
            nxt += 1
            p2 = pk(offered)
            if p2.n_included == p1.n_included:
                offered.pop()
                break
    return pk(offered), len(offered)


def msgs_sha(msgs):
    return hashlib.sha256(json.dumps(msgs, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def saved_pack(query, packdir="HYBRID"):
    p = TR / "packs" / packdir / ("%s__%s.json" % (query.question_id, query.setting))
    return json.loads(p.read_text(encoding="utf-8"))
