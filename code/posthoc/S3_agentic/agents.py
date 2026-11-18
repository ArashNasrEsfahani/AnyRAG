"""S3 (post-hoc, Nov 2026): two training-free agent arms with GLM-5.3-flash as agent AND answerer.

    GREP_AGENT  ReAct loop, tools over the raw text of the question's pool only:
                grep(pattern[, doc_id]) / read(doc_id, start, length) / list_docs() / finish(evidence=[spans])
    ITER_BM25   iterative agent, tool search(query) = BM25 (the frozen B2/HYBRID chunk index, bm25 channel only, CPU)
                over the frozen 1,000-char chunks of the pool, top 8 chunks per call / finish(chunk_ids=[...])

Both: at most MAX_ROUNDS (5) tool rounds (one agent turn = one round, at most MAX_CALLS_PER_ROUND tool calls in it),
then a forced finish turn. The evidence the agent picks is rendered by the frozen ragbench.contracts.pack_evidence at
EVIDENCE_BUDGET = 15,500 characters and answered by the frozen shared answer call (build_answer_messages +
answer_num_predict: the test's answer prompt and answer-format routing), with GLM pinned exactly as reader_pins.json.

    python agents.py --arm GREP_AGENT --split dev --ids A,B,C      (dev smoke / cost estimate)
    python agents.py --arm ITER_BM25 --split test --confirm-test   (one run on all 96 test questions)

Writes runs/<arm>/<split>/runs_<arm>.jsonl (identity records the frozen judge_multi reads) and ledger.jsonl (every
call). Append-only; a finished (question, arm) cell is skipped on rerun. No GPU, no Ollama, no dense vectors.
"""
import argparse
import datetime as dt
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
CODE = r"D:\KareOne\RAG-Paper-upgrades"
sys.path.insert(0, CODE)

import requests  # noqa: E402
import regex as rx  # noqa: E402

from ragbench.config import load_config  # noqa: E402
from ragbench.contracts import (Query, RetrievedItem, Setting, answer_format, answer_num_predict,  # noqa: E402
                                build_answer_messages, load_questions, pack_evidence)
from ragbench.corpus.registry import load_set, set_name_for  # noqa: E402
from ragbench.normalize import ZWNJ, normalize  # noqa: E402
from ragbench.pools import load_pool, resolve_pool_path  # noqa: E402

HERE = Path(__file__).resolve().parent
WORK = HERE.parent.parent
DS = WORK / "datasets_v2"
LEDGER = HERE / "ledger.jsonl"
STREAM = "S3_agentic"
MODEL = "z-ai/glm-5.3-flash"
PINS = json.loads((DS / "test_freeze" / "reader_pins.json").read_text(encoding="utf-8"))["pins"]
CATALOG = json.loads((WORK / "openrouter" / "catalog.json").read_text(encoding="utf-8"))
QFILES = {"dev": DS / "devrun" / "questions", "test": DS / "test_freeze" / "questions"}
POOLS = {"RPT-fa": ("C", "fa-fa"), "RPT-en": ("D", "en-en"), "BOOK-fa": ("K", "fa-fa"), "BOOK-en": ("K", "en-en"),
         "LAW-fa": ("L", "fa-fa"), "LAW-en": ("L", "en-en")}
EXPERIMENT = "posthoc-S3-agentic"
VERSION = "s3-agents-v1"

MAX_ROUNDS = 5
MAX_CALLS_PER_ROUND = 3
EVIDENCE_BUDGET = 15500
AGENT_MAX_TOKENS = 3000          # one agent turn, reasoning included (reasoning effort low, as the reader)
GREP_MAX_HITS = 20
GREP_SNIPPET = 200
GREP_OUT_CHARS = 3500
READ_MAX = 2500
READ_DEFAULT = 1500
SPAN_MAX = 4000                  # one finish evidence span
BM25_K = 8
KEEP_FULL_ROUNDS = 2             # tool results of older rounds are shortened (context/token control)
OLD_OBS_CHARS = 400
STOP_USD = 2.00                  # this stream's cap
FLOOR_USD = 2.00                 # global floor on the shared key


# ------------------------------------------------------------------------------------------------ money + calls
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
    return tot


class Stop(Exception):
    pass


def _supported():
    cat = next(r for r in CATALOG if r["id"] == MODEL)["params"]
    return set(cat) & set(PINS[MODEL]["params"])


SUPPORTED = _supported()


def glm(messages, *, cfg, purpose, ctx, num_predict=None, max_tokens=None, tools=None, tool_choice=None):
    """One pinned GLM call. Same parameter rule as the frozen OpenRouterReader._payload: sampling parameters filtered
    by the pinned endpoint's list (temperature 0, top_p 1; no seed on Together), reasoning effort low (excluded),
    max_tokens = num_predict + 3000 for the answer call; provider order [together], no fallbacks, require_parameters.
    -> dict(text, tool_calls, message, finish, usage...). Every call is logged to ledger.jsonl."""
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
    if tools:
        p["tools"] = tools
        if tool_choice:
            p["tool_choice"] = tool_choice
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
    row.update(ok=True, provider=data.get("provider"), usage_known=bool(u), prompt_tokens=int(u.get("prompt_tokens") or 0),
               completion_tokens=int(u.get("completion_tokens") or 0),
               reasoning_tokens=(u.get("completion_tokens_details") or {}).get("reasoning_tokens"),
               cached_tokens=(u.get("prompt_tokens_details") or {}).get("cached_tokens"),
               cost_usd=float(u.get("cost") or 0.0), finish=ch.get("finish_reason"),
               n_tool_calls=len(msg.get("tool_calls") or []), request_id=data.get("id"))
    _log(row)
    return dict(text=content.strip(), tool_calls=msg.get("tool_calls") or [], finish=ch.get("finish_reason"),
                row=row)


def _log(row):
    with open(LEDGER, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")


# ------------------------------------------------------------------------------------------------ Persian matching
_MAP = {0x064A: "\u06cc", 0x0649: "\u06cc", 0x0643: "\u06a9", 0x0626: "\u06cc",           # yeh / kaf variants
        0x0622: "\u0627", 0x0623: "\u0627", 0x0625: "\u0627", 0x0671: "\u0627",            # alef variants
        0x0629: "\u0647", 0x06C0: "\u0647", 0x0624: "\u0648",                              # teh marbuta, heh-yeh, waw-hamza
        0x066B: ".", 0x066C: ",", 0x066A: "%"}
for _i in range(10):
    _MAP[0x06F0 + _i] = str(_i)
    _MAP[0x0660 + _i] = str(_i)
_DROP = set("\u200c\u200d\u200b\u200e\u200f\u202a\u202b\u202c\u202d\u202e\u2066\u2067\u2068\u2069\u2060\ufeff"
            "\u061c\u00ad\u0640") | {chr(c) for c in range(0x064B, 0x0660)} | {"\u0670"}


def fold_char(ch):
    """One character -> its matching form ('' = dropped). The 1:1 / drop part of ragbench.normalize.normalize + fold
    (Arabic yeh/kaf -> Persian, alef/heh/waw/yeh-hamza variants, Persian/Arabic digits -> ASCII, ZWNJ and other
    joiners, tatweel and harakat removed), lower-cased; applied to text AND pattern so offsets can be mapped back."""
    if ch in _DROP:
        return ""
    c = _MAP.get(ord(ch), ch)
    return c.lower()


def fold_with_map(text):
    """-> (folded text, list orig offset per folded char). Newlines kept (grep is line-based)."""
    out, idx = [], []
    for i, ch in enumerate(text):
        f = fold_char(ch)
        for c in f:
            out.append(c)
            idx.append(i)
    return "".join(out), idx


def fold_pattern(p):
    """The same folding on a regex: each run of non-ASCII literal text goes through normalize() (NFKC, as
    grep_agent.normalize_pattern does) and then fold_char; ASCII (all regex syntax) is kept, lower-cased letters
    only. A literal ZWNJ in the pattern is dropped, as in the text, so 'می‌شود' and 'میشود' both match."""
    out, run = [], []

    def flush():
        if run:
            s = normalize("".join(run))
            out.append("".join(fold_char(c) for c in s))
            run.clear()

    for ch in p:
        if ord(ch) < 128 or ch.isspace():
            flush()
            out.append(ch)
        elif ch == ZWNJ:
            flush()
        else:
            run.append(ch)
    flush()
    return "".join(out)


# ------------------------------------------------------------------------------------------------ question loading
_SETS, _HX = {}, {}


def corpus_set(name, cfg):
    if name not in _SETS:
        _SETS[name] = load_set(name, cfg)
    return _SETS[name]


def load_split(split, cfg):
    """-> [(question, pool_id)] in file order (dev: the devrun adapter files; test: the frozen test adapter files)."""
    out = []
    for pid in POOLS:
        f = QFILES[split] / ("%s_v2run_%s.jsonl" % (split, pid))
        for q in load_questions(f):
            out.append((q, pid))
    return out


def make_query(q, pid, cfg):
    corpus, setting = POOLS[pid]
    s = Setting.parse(setting)
    sn = set_name_for(q.corpus, s.d_lang)
    pool = load_pool(resolve_pool_path(pid, cfg))
    scope = pool.mask(q.scope(s.d_lang), sn)
    return Query(question_id=q.id, text=q.text(s.q_lang), q_lang=s.q_lang, d_lang=s.d_lang, setting=s.name,
                 corpus=q.corpus, corpus_set=sn, doc_scope=scope, answer_format=answer_format(q), pool_id=pid)


def span_item(doc, a, b):
    """A verbatim RetrievedItem for doc.text[a:b] (section = that of the first unit it touches)."""
    us = doc.units_overlapping(a, b)
    return RetrievedItem(text=doc.text[a:b], doc_id=doc.id, section_id=(us[0].section_id if us else None),
                         unit_ids=[u.id for u in us], kind="window", span=[a, b])


def merge_spans(spans):
    """[(doc_id, a, b)] in order -> same order of first appearance, overlapping/adjacent spans of a doc merged."""
    out = []
    for d, a, b in spans:
        for i, (d2, a2, b2) in enumerate(out):
            if d2 == d and a <= b2 and b >= a2:
                out[i] = (d, min(a, a2), max(b, b2))
                break
        else:
            out.append((d, a, b))
    # a second pass, merges created by the first can chain
    changed = True
    while changed:
        changed = False
        for i in range(len(out)):
            for j in range(i + 1, len(out)):
                if out[i][0] == out[j][0] and out[j][1] <= out[i][2] and out[j][2] >= out[i][1]:
                    out[i] = (out[i][0], min(out[i][1], out[j][1]), max(out[i][2], out[j][2]))
                    del out[j]
                    changed = True
                    break
            if changed:
                break
    return out


# ------------------------------------------------------------------------------------------------ prompts + tools
COMMON_RULES = (
    "You have at most {rounds} tool rounds; in one round you may make up to {per} tool calls. After that you must call "
    "finish. Do not answer the question yourself: call finish with the evidence, and a separate reader will answer "
    "from exactly that evidence (at most {budget} characters in total; anything beyond is cut). Choose evidence that "
    "contains every value, date, name or statement the question needs, for every subject, period or document it "
    "mentions, with enough context to show what each number refers to (subject, period, unit, table heading). "
    "If you find nothing relevant, call finish with the best candidates you saw.")

GREP_SYSTEM = (
    "You find evidence in a fixed collection of documents (company reports, books or laws) to answer one question. "
    "The documents are in {dlang}. Tools:\n"
    "- grep(pattern, doc_id?): a case-insensitive regular expression (Python syntax) searched line by line in every "
    "document (or one doc_id). Returns matching lines as 'D<k> @<offset>: ...snippet...' plus the number of matches per "
    "document. Persian text is normalised on both sides: Arabic and Persian forms of ی/ک, alef variants, Persian, "
    "Arabic and Latin digits are the same, and the half-space (ZWNJ) is ignored, so write words plainly. Use short "
    "patterns (one or two words, or alternatives a|b); long phrases rarely match.\n"
    "- read(doc_id, start, length): the raw text doc_id[start:start+length] (length at most {readmax}). Use it to see "
    "the context around a grep hit (start a few hundred characters before the offset).\n"
    "- list_docs(): the documents with their titles and sizes (already shown below).\n"
    "- finish(evidence): ends the search. evidence = a list of spans {{\"doc_id\": \"D3\", \"start\": 1200, "
    "\"length\": 1500}} (each at most {spanmax} characters), most important first. Make spans generous (about "
    "1,000-3,000 characters around each fact, starting before its table or paragraph heading); the budget is large.\n"
    + COMMON_RULES)

BM25_SYSTEM = (
    "You find evidence in a fixed collection of documents (company reports, books or laws) to answer one question. "
    "The documents are in {dlang}. Tools:\n"
    "- search(query): keyword (BM25) search over passages of about 1,000 characters; returns the top {k} passages as "
    "'[C<number>] <document> > <section>' and their text. Matching is by words, so write queries in the documents' "
    "language with the distinctive words (names, terms, years) you expect in the passage; vary the words between "
    "searches instead of repeating a query.\n"
    "- finish(chunk_ids): ends the search. chunk_ids = the passage numbers to give the reader, most important first, "
    "e.g. [\"C120\", \"C77\"]. Include every passage that may be needed (up to about 15); the budget is large.\n"
    + COMMON_RULES)

USER_FIRST = "Question: {question}\n\n{extra}Make your first tool call."

LANG = {"fa": "Persian", "en": "English"}


def fn(name, desc, props, req):
    return {"type": "function", "function": {"name": name, "description": desc, "parameters": {
        "type": "object", "properties": props, "required": req}}}


GREP_TOOLS = [
    fn("grep", "Regex search, line by line, in all documents of the collection or in one.",
       {"pattern": {"type": "string"}, "doc_id": {"type": "string", "description": "optional, e.g. D3"}}, ["pattern"]),
    fn("read", "Read raw text of a document.", {"doc_id": {"type": "string"}, "start": {"type": "integer"},
                                               "length": {"type": "integer"}}, ["doc_id", "start"]),
    fn("list_docs", "List the documents.", {}, []),
    fn("finish", "End the search and hand the evidence spans to the reader.",
       {"evidence": {"type": "array", "items": {"type": "object", "properties": {
           "doc_id": {"type": "string"}, "start": {"type": "integer"}, "length": {"type": "integer"}},
           "required": ["doc_id", "start", "length"]}}}, ["evidence"]),
]
BM25_TOOLS = [
    fn("search", "BM25 keyword search over the passages; top %d." % BM25_K, {"query": {"type": "string"}}, ["query"]),
    fn("finish", "End the search and hand the chosen passages to the reader.",
       {"chunk_ids": {"type": "array", "items": {"type": "string"}}}, ["chunk_ids"]),
]


def _args(tc):
    try:
        a = json.loads((tc.get("function") or {}).get("arguments") or "{}")
        return a if isinstance(a, dict) else {}
    except ValueError:
        return {}


def _int(v, default=None):
    try:
        return int(float(str(v).strip()))
    except (TypeError, ValueError):
        return default


# ------------------------------------------------------------------------------------------------ the environments
class GrepEnv:
    def __init__(self, query, cfg):
        cs = corpus_set(query.corpus_set, cfg)
        self.docs = [cs.doc(d) for d in query.doc_scope if cs.has_doc(d)]
        self.handle = {"D%d" % (i + 1): d for i, d in enumerate(self.docs)}
        self.rev = {d.id: h for h, d in self.handle.items()}
        key = (query.corpus_set, tuple(query.doc_scope))
        if key not in _HX:
            folded = {}
            for d in self.docs:
                ft, idx = fold_with_map(d.text)
                folded[d.id] = (ft, idx)
            _HX[key] = folded
        self.folded = _HX[key]
        self.read_spans = []        # (doc_id, a, b) in read order
        self.grep_hits = []         # (doc_id, offset) first hits, in order

    def doc_list(self):
        return "\n".join("%s: %s (%d chars)" % (h, d.title, len(d.text)) for h, d in self.handle.items())

    def _doc(self, h):
        h = str(h or "").strip()
        if h in self.handle:
            return self.handle[h]
        if re.fullmatch(r"\d+", h) and "D" + h in self.handle:
            return self.handle["D" + h]
        raise ValueError("unknown doc_id %r; use D1..D%d" % (h, len(self.docs)))

    def call(self, name, a):
        if name == "list_docs":
            return self.doc_list()
        if name == "grep":
            return self.grep(a.get("pattern") or a.get("query") or "", a.get("doc_id"))
        if name == "read":
            return self.read(a.get("doc_id"), _int(a.get("start"), 0), _int(a.get("length"), READ_DEFAULT))
        raise ValueError("unknown tool %s" % name)

    def grep(self, pattern, doc_id=None):
        pattern = str(pattern).replace("\n", " ").strip()
        if not pattern or len(pattern) > 200:
            raise ValueError("the pattern must be 1-200 characters")
        fp = fold_pattern(pattern)
        try:
            r = rx.compile(fp, rx.IGNORECASE | rx.MULTILINE)
        except Exception as e:  # noqa: BLE001
            raise ValueError("regex error: %s" % str(e)[:120])
        if r.search("", timeout=0.5) or r.search("e", timeout=0.5) or r.search("ا", timeout=0.5):
            raise ValueError("the pattern matches empty text or a single letter; search for a word")
        docs = [self._doc(doc_id)] if doc_id else self.docs
        hits, per_doc, total = [], {}, 0
        for d in docs:
            ft, idx = self.folded[d.id]
            try:
                ms = list(r.finditer(ft, timeout=5.0))
            except TimeoutError:
                raise ValueError("the pattern is too slow; simplify it")
            if not ms:
                continue
            per_doc[self.rev[d.id]] = len(ms)
            total += len(ms)
            last_line = None
            for m in ms:
                o = idx[m.start()] if m.start() < len(idx) else len(d.text) - 1
                ln = d.text.rfind("\n", 0, o)
                if ln == last_line:                          # one hit per line
                    continue
                last_line = ln
                hits.append((d.id, o))
        if not hits:
            return "no matches for %r (normalised: %r)" % (pattern, fp)
        self.grep_hits += hits[:GREP_MAX_HITS]
        lines = []
        for did, o in hits[:GREP_MAX_HITS]:
            t = self.handle[self.rev[did]].text
            a = max(0, o - GREP_SNIPPET // 3)
            b = min(len(t), a + GREP_SNIPPET)
            snip = t[a:b].replace("\n", " | ")
            lines.append("%s @%d: %s%s%s" % (self.rev[did], o, "…" if a else "", snip, "…" if b < len(t) else ""))
        head = "%d matches in %d documents (%s)%s:\n" % (
            total, len(per_doc), ", ".join("%s:%d" % kv for kv in sorted(per_doc.items(), key=lambda kv: -kv[1])[:15]),
            "; first %d lines shown" % GREP_MAX_HITS if len(hits) > GREP_MAX_HITS else "")
        out = head + "\n".join(lines)
        return out[:GREP_OUT_CHARS]

    def read(self, doc_id, start, length):
        d = self._doc(doc_id)
        length = max(50, min(int(length or READ_DEFAULT), READ_MAX))
        a = max(0, min(int(start or 0), len(d.text)))
        b = min(len(d.text), a + length)
        self.read_spans.append((d.id, a, b))
        return "%s [%d-%d of %d]:\n%s" % (self.rev[d.id], a, b, len(d.text), d.text[a:b])

    def evidence(self, fin_args):
        """finish spans -> items; fallback = every read span, else windows around the first grep hits."""
        spans, src = [], "finish"
        for e in (fin_args or {}).get("evidence") or []:
            if not isinstance(e, dict):
                continue
            try:
                d = self._doc(e.get("doc_id"))
            except ValueError:
                continue
            a = max(0, _int(e.get("start"), 0))
            ln = max(1, min(_int(e.get("length"), READ_DEFAULT), SPAN_MAX))
            b = min(len(d.text), a + ln)
            if b > a:
                spans.append((d.id, a, b))
        if not spans:
            src = "fallback_read"
            spans = list(self.read_spans)
        if not spans:
            src = "fallback_grep_windows"
            spans = [(did, max(0, o - 300), o + 700) for did, o in self.grep_hits]
        spans = merge_spans(spans)
        cs = {d.id: d for d in self.docs}
        items = [span_item(cs[did], a, min(b, len(cs[did].text))) for did, a, b in spans]
        return items, src, [[self.rev[d], a, b] for d, a, b in spans]


class Bm25Env:
    def __init__(self, query, cfg):
        from ragbench.retrieval.hybrid import HybridIndex
        cs = corpus_set(query.corpus_set, cfg)
        if query.corpus_set not in _HX:
            _HX[query.corpus_set] = HybridIndex.load(cs, cfg, client=None, refresh_vectors=False)
        self.hx = _HX[query.corpus_set]
        self.cs = cs
        self.scope = list(query.doc_scope)
        self.seen = []              # chunk numbers in first-seen order

    def head(self, i):
        c = self.hx.chunks[i]
        doc = self.cs.doc(c["doc_id"])
        parts = [doc.title] + ([t for t in doc.section_path(c["section_id"]) if t] if c.get("section_id") else [])
        return "[C%d] %s" % (i, " > ".join(parts))

    def call(self, name, a):
        if name != "search":
            raise ValueError("unknown tool %s" % name)
        qtext = str(a.get("query") or a.get("pattern") or "").strip()
        if not qtext:
            raise ValueError("empty query")
        hits = self.hx.search(qtext, doc_ids=self.scope, k=BM25_K, channels=("bm25",))
        if not hits:
            return "no passages match %r" % qtext
        out = []
        for h in hits:
            i = h.chunk
            if i in self.seen:
                out.append(self.head(i) + "\n(already shown above)")
            else:
                self.seen.append(i)
                out.append(self.head(i) + "\n" + h.text.strip())
        return "\n\n".join(out)

    def evidence(self, fin_args):
        ids, src = [], "finish"
        for c in (fin_args or {}).get("chunk_ids") or []:
            n = _int(str(c).strip().lstrip("[Cc").rstrip("]"))
            if n is not None and 0 <= n < self.hx.n and self.hx.chunks[n]["doc_id"] in set(self.scope) and n not in ids:
                ids.append(n)
        if not ids:
            src, ids = "fallback_seen", list(self.seen)
        items = [self.hx.hit_for_chunk(i).to_item(rank=r) for r, i in enumerate(ids)]
        return items, src, ["C%d" % i for i in ids]


# ------------------------------------------------------------------------------------------------ the loop
def shorten_old(messages, n_round_starts):
    """Tool results older than the last KEEP_FULL_ROUNDS rounds are cut to OLD_OBS_CHARS."""
    if len(n_round_starts) <= KEEP_FULL_ROUNDS:
        return messages
    cutoff = n_round_starts[-KEEP_FULL_ROUNDS]
    out = []
    for i, m in enumerate(messages):
        if i < cutoff and m.get("role") == "tool" and len(m.get("content") or "") > OLD_OBS_CHARS:
            m = dict(m, content=m["content"][:OLD_OBS_CHARS] + " … [older result shortened]")
        out.append(m)
    return out


def run_agent(arm, q, pid, cfg, run_tag):
    query = make_query(q, pid, cfg)
    t0 = time.perf_counter()
    ctx = {"arm": arm, "question_id": q.id, "run_tag": run_tag}
    if arm == "GREP_AGENT":
        env = GrepEnv(query, cfg)
        system = GREP_SYSTEM.format(dlang=LANG[query.d_lang], readmax=READ_MAX, spanmax=SPAN_MAX, rounds=MAX_ROUNDS,
                                    per=MAX_CALLS_PER_ROUND, budget=EVIDENCE_BUDGET)
        extra = "Documents (list_docs):\n%s\n\n" % env.doc_list()
        tools = GREP_TOOLS
    else:
        env = Bm25Env(query, cfg)
        system = BM25_SYSTEM.format(dlang=LANG[query.d_lang], k=BM25_K, rounds=MAX_ROUNDS, per=MAX_CALLS_PER_ROUND,
                                    budget=EVIDENCE_BUDGET)
        extra = ""
        tools = BM25_TOOLS
    msgs = [{"role": "system", "content": system},
            {"role": "user", "content": USER_FIRST.format(question=query.text.strip(), extra=extra)}]
    steps, round_starts, fin, n_tool_calls, n_invalid = [], [], None, {}, 0
    rounds = 0
    for turn in range(MAX_ROUNDS + 2):          # rounds 1..5, then a forced finish (and one retry of it)
        forced = rounds >= MAX_ROUNDS
        if forced:
            msgs.append({"role": "user", "content": "No tool rounds are left. Call finish now with the evidence."})
        res = glm(shorten_old(msgs, round_starts), cfg=cfg, purpose="agent", max_tokens=AGENT_MAX_TOKENS,
                  tools=tools, tool_choice=({"type": "function", "function": {"name": "finish"}} if forced else "auto"),
                  ctx=dict(ctx, turn=turn, forced=forced))
        tcs = res["tool_calls"]
        step = {"turn": turn, "forced": forced, "finish_reason": res["finish"], "text": res["text"][:500],
                "calls": []}
        if not tcs:
            # no native tool call: try a JSON action in the content (the B5 protocol), else bounce once
            n_invalid += 1
            step["invalid"] = True
            steps.append(step)
            msgs.append({"role": "assistant", "content": res["text"] or ""})
            if n_invalid >= 3:
                break
            msgs.append({"role": "user", "content": "Reply with a tool call (grep/read/search or finish)."})
            if forced:
                break
            continue
        msgs.append({"role": "assistant", "content": res["text"] or "", "tool_calls": tcs})
        round_starts.append(len(msgs) - 1)
        n_used = 0
        for tc in tcs:
            name = (tc.get("function") or {}).get("name")
            a = _args(tc)
            n_tool_calls[name] = n_tool_calls.get(name, 0) + 1
            if name == "finish":
                fin = a
                out = "ok"
            elif forced:
                out = "error: no tool rounds left; call finish"
            elif n_used >= MAX_CALLS_PER_ROUND:
                out = "error: at most %d tool calls per round; this call was skipped" % MAX_CALLS_PER_ROUND
            else:
                n_used += 1
                try:
                    out = env.call(name, a)
                except Exception as e:  # noqa: BLE001
                    out = "error: %s" % str(e)[:200]
            step["calls"].append({"name": name, "args": a, "out_chars": len(out), "out": out[:300]})
            msgs.append({"role": "tool", "tool_call_id": tc.get("id") or "call", "content": out})
        steps.append(step)
        if not forced:
            rounds += 1
        if fin is not None:
            break
        if forced:
            break
    items, ev_src, ev_ids = env.evidence(fin)
    ev = pack_evidence(items, corpus_set(query.corpus_set, cfg), budget=EVIDENCE_BUDGET, cfg=cfg)
    msgs_ans = build_answer_messages(query, ev, cfg)
    npred = answer_num_predict(query, cfg)
    ans = glm(msgs_ans, cfg=cfg, purpose="answer", num_predict=npred, ctx=dict(ctx, turn="answer"))
    latency = (time.perf_counter() - t0) * 1000.0
    return query, ev, ans, msgs_ans, npred, dict(steps=steps, rounds=rounds, tool_calls=n_tool_calls,
                                                 evidence_source=ev_src, evidence_ids=ev_ids, finish_args=fin,
                                                 invalid_turns=n_invalid), latency


def rid_of(arm, q, pid, setting):
    ident = {"experiment_id": EXPERIMENT, "question_id": q.id, "pool_id": pid, "method": arm,
             "options": {"version": VERSION, "max_rounds": MAX_ROUNDS, "budget": EVIDENCE_BUDGET},
             "reader": {"chat": "openrouter/" + MODEL, "chat_digest": "openrouter"}, "replicate": 0,
             "setting": setting}
    ident["rid"] = hashlib.sha1(json.dumps(ident, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:20]
    return ident


def done_cells(path):
    ok = set()
    if path.is_file():
        for line in open(path, encoding="utf-8"):
            if line.strip():
                r = json.loads(line)
                if not r.get("error"):
                    ok.add(r["question_id"])
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=("GREP_AGENT", "ITER_BM25"))
    ap.add_argument("--split", required=True, choices=("dev", "test"))
    ap.add_argument("--ids", default=None)
    ap.add_argument("--tag", default="", help="dev only: a sub-directory for a prompt iteration")
    ap.add_argument("--confirm-test", action="store_true")
    a = ap.parse_args()
    if a.split == "test" and not a.confirm_test:
        raise SystemExit("REFUSED: the test run needs --confirm-test (run once)")
    cfg = load_config()
    out = HERE / "runs" / a.arm / (a.split + (("_" + a.tag) if a.tag else ""))
    out.mkdir(parents=True, exist_ok=True)
    rpath = out / ("runs_%s.jsonl" % a.arm)
    done = done_cells(rpath)
    plan = load_split(a.split, cfg)
    if a.ids:
        want = [x.strip() for x in a.ids.split(",") if x.strip()]
        plan = [(q, p) for q, p in plan if q.id in want]
    if a.split == "dev" and len(plan) > 10:
        raise SystemExit("dev: at most 10 questions")
    for k, (q, pid) in enumerate(plan):
        if q.id in done:
            continue
        if k % 5 == 0:
            rem = key_remaining()
            if rem is not None and rem < FLOOR_USD:
                print("GLOBAL FLOOR: key has $%.2f left" % rem)
                return 1
        if spent() >= STOP_USD:
            print("STREAM CAP reached: $%.4f" % spent())
            return 1
        err, rec = None, None
        try:
            query, ev, ans, msgs_ans, npred, trace, latency = run_agent(a.arm, q, pid, cfg, out.name)
        except Stop as e:
            print("STOP:", e)
            return 1
        except Exception as e:  # noqa: BLE001
            import traceback
            err = "%s: %s" % (type(e).__name__, str(e)[:400])
            print(q.id, "ERROR", err, traceback.format_exc()[-800:])
            query = make_query(q, pid, cfg)
            ev, ans, msgs_ans, npred, trace, latency = None, None, None, None, {}, 0.0
        ident = rid_of(a.arm, q, pid, query.setting)
        answer = (ans["text"] if ans else "") or ""
        rec = {"question_id": q.id, "approach": a.arm, "approach_base": a.arm, "flags": {}, "setting": query.setting,
               "q_lang": query.q_lang, "d_lang": query.d_lang, "corpus": query.corpus, "corpus_set": query.corpus_set,
               "pool_id": pid, "answer_text": answer, "abstained": answer.strip() == cfg.evidence["abstain_token"],
               "evidence_items": [{"doc_id": it.doc_id, "section_id": it.section_id, "chars": len(it.text),
                                   "unit_ids": it.unit_ids, "cut": "cut_from_chars" in (it.meta or {})}
                                  for it in (ev.items if ev else [])],
               "evidence_units": ev.unit_ids if ev else [], "evidence_chars": ev.chars if ev else 0,
               "latency_ms": round(latency, 1), "error": err, "trace": trace,
               "answer_done_reason": ans["finish"] if ans else None,
               "models": {"chat": "openrouter/" + MODEL, "embed": None},
               "meta": {"identity": ident, "answer_format": query.answer_format, "answer_num_predict": npred,
                        "answer_messages_sha256": hashlib.sha256(json.dumps(msgs_ans, ensure_ascii=False,
                                                                            sort_keys=True).encode()).hexdigest()
                        if msgs_ans else None, "version": VERSION},
               "answer_messages": msgs_ans, "ts": dt.datetime.now().isoformat(timespec="seconds")}
        with open(rpath, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print("%s %s rounds=%s tools=%s ev=%s/%d %.1fs | %s | spent $%.4f" % (
            q.id, a.arm, trace.get("rounds"), trace.get("tool_calls"), trace.get("evidence_source"),
            rec["evidence_chars"], latency / 1000.0, answer[:70].replace("\n", " "), spent()), flush=True)
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
