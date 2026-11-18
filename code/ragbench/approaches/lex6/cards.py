"""LEX6 offline hierarchical source cards (LEX6_DESIGN_R2 "Card generation"; Astra R1 step 1).

Built from the documents ALONE (no question, no gold annotation is ever read here). Cards guide retrieval only; they
never enter reader evidence.

    python -m ragbench.approaches.lex6.cards plan  [POOL ...]   # no model calls: blocks, segments, exact input tokens, $
    python -m ragbench.approaches.lex6.cards run   [POOL ...]   # LLM cards (resumable), pool JSONL + PROGRESS.md
    python -m ragbench.approaches.lex6.cards index [POOL ...]   # bge-m3 card vectors, card BM25, doc centroids

Per document
  1. SEGMENTS. The section tree is walked in document order. A section whose whole subtree fits in one block's text
     budget (BLOCK_TEXT_TOKENS) is ONE segment; a larger section contributes its own text (title + own units, split
     by units into pieces that fit; a single oversized unit is split by characters) and then recurses into its
     children. Units outside every section become their own segments. Every unit of the document lands in exactly
     one segment (checked: coverage == all units).
  2. BLOCKS. Consecutive segments are packed greedily into blocks; every block call is <= 4,096 input tokens
     INCLUDING the instructions (measured with the gpt-oss tokenizer; a block that would exceed it is split).
  3. BLOCK CALL (gpt-oss-20b, reasoning low, pinned provider): a <= 60-word block summary + a <= 25-word line for
     every marked segment (+ identity metadata with a verbatim evidence snippet, first block only).
  4. SECTION CARDS (<= 64 tokens each, keyed by section path, with the unit interval they cover):
       subtree   - a fitting section: its line from the block call (read from the source text itself)
       group     - consecutive small sibling subtrees (< GROUP_SMALL tokens each, same parent, same block) share one
                   line: the block call sees them as one marked segment
       combined  - an oversized section: one combine call over its children's cards and its own-text lines
                   (bottom-up; inputs over the call budget are reduced in bounded steps first)
  5. DOCUMENT: block summaries -> (bounded reduce if needed) -> one final call: kind, short title, summary, full card
     text, compact card text. Cards = deterministic header (display id + verified metadata) + model text; token
     lengths measured with the gpt-oss tokenizer; over-length -> one regeneration with a lower word target, then a
     token-boundary cut marked "truncated".
  Metadata: a field is kept only when the model's evidence snippet occurs verbatim (after fold()) in the block text;
  otherwise "unknown" with status "unsupported" (never inferred from frequently mentioned years).
  Failures are explicit: a failed block leaves its unit interval in `uncovered`; failed cards carry status "failed".
"""
import argparse
import hashlib
import json
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from ...normalize import fold, normalize

PROMPT_VERSION = "lex6-cards-v1"
MODEL = "openai/gpt-oss-20b"
BLOCK_MAX_INPUT = 4096          # tokens per call, instructions included
BLOCK_TEXT_TOKENS = 3600        # source text per block (markers and instructions fill the rest; checked per call)
REDUCE_INPUT = 3200             # tokens of summaries per reduce/combine call
GROUP_SMALL = 250               # subtree segments below this (tokens) may share one line with their siblings
GROUP_MAX = 900                 # a group's total tokens
FULL_MAX, COMPACT_MAX, SECTION_MAX = 128, 64, 64
BLOCK_SUMMARY_WORDS = 60
CONCURRENCY = 4
PRICE_IN, PRICE_OUT = 0.07e-6, 0.15e-6   # amazon-bedrock gpt-oss-20b (OpenRouter endpoints API, 2026-09-26)
TASK_CAP_USD = 1.50
HARD_STOP_USD = 7.00

ROOT = Path(r"D:\KareOne\RAG Paper")
OUT = ROOT / "work" / "datasets_v2" / "lex6" / "cards"
TOKENIZER = ROOT / "work" / "models" / "tok" / "gpt-oss-20b.tokenizer.json"
PROVIDERS = ROOT / "work" / "datasets_v2" / "lex5_pilot" / "providers.json"
BUDGET = ROOT / "work" / "openrouter" / "reader_budget.json"
POOLS = ["RPT-fa", "RPT-en", "BOOK-fa", "BOOK-en", "LAW-fa", "LAW-en"]
SET_OF_POOL = {"BOOK-en": "BOOK_en", "BOOK-fa": "BOOK_fa", "LAW-en": "LAW_en", "LAW-fa": "LAW_fa",
               "RPT-en": "D_en", "RPT-fa": "C"}
DID = {"RPT-fa": "RF", "RPT-en": "RE", "BOOK-fa": "BF", "BOOK-en": "BE", "LAW-fa": "LF", "LAW-en": "LE"}
LANG_NAME = {"fa": "Persian", "en": "English"}
META_FIELDS = ["title", "kind", "author_or_issuer", "publication_date_or_version", "period_covered"]
UNKNOWN = "unknown"

# ------------------------------------------------------------------------------------------------------------------
# tokenizer
# ------------------------------------------------------------------------------------------------------------------
_TOK = {}


def tok():
    if "t" not in _TOK:
        import tokenizers
        _TOK["t"] = tokenizers.Tokenizer.from_file(str(TOKENIZER))
    return _TOK["t"]


def ntok(text):
    return len(tok().encode(text or "", add_special_tokens=False).ids)


def ntok_many(texts):
    return [len(e.ids) for e in tok().encode_batch(list(texts), add_special_tokens=False)] if texts else []


def cut_tokens(text, n):
    """-> text cut at a token boundary so that text + '…' is <= n tokens."""
    ids = tok().encode(text, add_special_tokens=False).ids
    if len(ids) <= n:
        return text
    out = tok().decode(ids[:max(1, n - 1)]).rstrip()
    while out and ntok(out + "…") > n:
        out = out[:-1].rstrip()
    return out + "…"


def cut_words(text, n):
    w = (text or "").split()
    return text if len(w) <= n else " ".join(w[:n]) + " …"


# ------------------------------------------------------------------------------------------------------------------
# segmentation (no model calls)
# ------------------------------------------------------------------------------------------------------------------

def _path(doc, sid):
    return " › ".join(t for t in doc.section_path(sid) if t) or "(untitled)"


def segments(doc):
    """-> list of segments in document order. A segment: {kind: subtree|own|loose, sids, path, units (orders),
    span [a,b), tokens, parent}. Every unit is in exactly one segment."""
    units_tok = dict(zip([u.id for u in doc.units], ntok_many([u.text_raw or u.text for u in doc.units])))
    sec = {s.id: s for s in doc.sections}
    sub_tok = {}
    for s in reversed(doc.sections):        # pre-order reversed: children before parents
        sub_tok[s.id] = ntok(s.title_raw or s.title or "") + sum(units_tok[u] for u in s.unit_ids) + \
            sum(sub_tok[c] for c in s.children)
    out = []

    def unit_pieces(s, uids, kind):
        """own units of s -> pieces <= BLOCK_TEXT_TOKENS (an oversized unit is split by characters)."""
        cur, cur_t = [], ntok(s.title_raw or s.title or "") if s is not None else 0
        def flush():
            if cur:
                us = [doc.unit(u) for u in cur]
                out.append({"kind": kind, "sids": [s.id] if s else [], "path": _path(doc, s.id) if s else "(no section)",
                            "units": [us[0].order, us[-1].order], "span": [us[0].span[0], us[-1].span[1]],
                            "tokens": cur_t, "parent": s.parent if s else None})
        for u in uids:
            t = units_tok[u]
            if t > BLOCK_TEXT_TOKENS:
                flush(); cur.clear(); cur_t = 0
                uu = doc.unit(u)
                a, b = uu.span
                n = -(-t // int(BLOCK_TEXT_TOKENS * 0.9))
                step = -(-(b - a) // n)
                for i in range(n):
                    pa, pb = a + i * step, min(b, a + (i + 1) * step)
                    out.append({"kind": kind, "sids": [s.id] if s else [], "path": (_path(doc, s.id) if s else
                                "(no section)"), "units": [uu.order, uu.order], "span": [pa, pb],
                                "tokens": ntok(doc.text[pa:pb]), "parent": s.parent if s else None,
                                "unit_part": "%d/%d" % (i + 1, n)})
                continue
            if cur and cur_t + t > int(BLOCK_TEXT_TOKENS * 0.9):     # estimate margin
                flush(); cur.clear(); cur_t = 0
            cur.append(u)
            cur_t += t
        flush()

    def walk(s):
        if sub_tok[s.id] <= BLOCK_TEXT_TOKENS:
            # the unit-sum estimate can be a little low (joins, table layout): measure the real text once
            sub_tok[s.id] = max(sub_tok[s.id], ntok(doc.text[s.subtree_span[0]:s.subtree_span[1]]))
        if sub_tok[s.id] <= BLOCK_TEXT_TOKENS:
            us = doc.section_units(s.id, recursive=True)
            if not us:
                return
            out.append({"kind": "subtree", "sids": [s.id], "path": _path(doc, s.id),
                        "units": [min(u.order for u in us), max(u.order for u in us)],
                        "span": list(s.subtree_span), "tokens": sub_tok[s.id], "parent": s.parent})
            return
        unit_pieces(s, s.unit_ids, "own")
        for c in s.children:
            walk(sec[c])

    for r in doc.roots():
        walk(r)
    covered = {u for s in doc.sections for u in s.unit_ids}
    loose = [u.id for u in doc.units if u.id not in covered]
    if loose:
        unit_pieces(None, loose, "loose")
    out.sort(key=lambda g: (g["units"][0], g["span"][0]))
    return out, sub_tok


def blocks_of(doc, segs):
    """Greedy packing of consecutive segments -> blocks; small sibling subtrees are grouped into one marked item."""
    blocks, cur, cur_t = [], [], 0
    for g in segs:
        cost = g["tokens"] + ntok(g["path"]) + 10          # + its marker line
        if cur and cur_t + cost > BLOCK_TEXT_TOKENS:
            blocks.append(cur); cur, cur_t = [], 0
        cur.append(g)
        cur_t += cost
    if cur:
        blocks.append(cur)
    out = []
    for bl in blocks:
        items = []                       # marked items: list of segments sharing one line
        for g in bl:
            last = items[-1] if items else None
            if (last and g["kind"] == "subtree" and all(x["kind"] == "subtree" for x in last)
                    and last[0]["parent"] == g["parent"] and g["tokens"] < GROUP_SMALL
                    and all(x["tokens"] < GROUP_SMALL for x in last)
                    and sum(x["tokens"] for x in last) + g["tokens"] <= GROUP_MAX):
                last.append(g)
            else:
                items.append([g])
        out.append(items)
    return out


def fit_blocks(doc, title, blocks):
    """split any block whose full prompt would exceed BLOCK_MAX_INPUT (measured) in two, until all fit."""
    out, todo = [], list(blocks)
    while todo:
        b = todo.pop(0)
        if len(b) > 1 and prompt_tokens(block_prompt(doc, title, 0, 2, b, True)) > BLOCK_MAX_INPUT:
            h = len(b) // 2
            todo[:0] = [b[:h], b[h:]]
        else:
            out.append(b)
    return out


def item_key(doc, item):
    if len(item) == 1:
        return item[0]["path"] + ("  (part %s)" % item[0]["unit_part"] if item[0].get("unit_part") else "")
    par = _path(doc, item[0]["parent"]) + " › " if item[0]["parent"] else ""
    return par + doc.section(item[0]["sids"][0]).title + " … " + doc.section(item[-1]["sids"][0]).title


# ------------------------------------------------------------------------------------------------------------------
# prompts (domain-neutral)
# ------------------------------------------------------------------------------------------------------------------
SYS = ("You index documents for a search system. You read part of a document and describe what it contains, "
       "factually, using only the text shown. You reply with one JSON object and nothing else.")

RULES = """Rules:
- Write every description in {lang}, the document's own language.
- Copy names, numbers, dates, units and calendar labels exactly as the text writes them (keep the document's own calendar; never convert).
- Describe what the text contains (subjects, named entities, quantities, periods discussed, kinds of data); add no outside knowledge and no evaluation."""

META_RULES = """- "metadata": the document's identity, ONLY as the text states it explicitly (cover, title page, heading, dated line, signature). Fields: title, kind (e.g. what sort of document it is), author_or_issuer, publication_date_or_version (when THIS document was issued/published/versioned), period_covered (the period the document reports on, only if stated as such). Each field is {{"value": "...", "evidence": "<the exact words of the text that state it, at most 12 words, copied character for character>"}} or the string "unknown". The publication date is NOT a year the text merely mentions or discusses; never infer it from frequently mentioned years. When unsure, "unknown"."""


def block_prompt(doc, title, bi, nb, items, first):
    lang = LANG_NAME[doc.lang]
    parts = []
    for k, item in enumerate(items, 1):
        head = "=== [S%d] %s ===" % (k, item_key(doc, item))
        body = "\n".join(doc.text[g["span"][0]:g["span"][1]].strip() for g in item)
        parts.append(head + "\n" + body)
    ids = ", ".join("S%d" % k for k in range(1, len(items) + 1))
    shape = '{"block_summary": "<at most %d words: what this block covers>", "segments": {"S1": "<at most 25 words ' \
            'describing S1>", ...}%s}' % (BLOCK_SUMMARY_WORDS, ', "metadata": {...}' if first else "")
    user = ("Document: %s\nBlock %d of %d. The block is divided into marked segments (%s); each marker gives the "
            "segment's section path.\n\n<<<\n%s\n>>>\n\nReturn exactly this JSON shape:\n%s\n"
            "- \"segments\" has one entry for EVERY segment id: %s.\n%s%s") % (
        title, bi + 1, nb, ids, "\n\n".join(parts), shape, ids, RULES.format(lang=lang),
        ("\n" + META_RULES.format()) if first else "")
    return [{"role": "system", "content": SYS}, {"role": "user", "content": user}]


def combine_prompt(doc, title, what, lines, words, key="text"):
    lang = LANG_NAME[doc.lang]
    user = ("Document: %s\n%s\n\n<<<\n%s\n>>>\n\nReturn exactly this JSON shape:\n{\"%s\": \"<at most %d words>\"}\n%s") % (
        title, what, "\n".join(lines), key, words, RULES.format(lang=lang))
    return [{"role": "system", "content": SYS}, {"role": "user", "content": user}]


def final_prompt(doc, title, meta, summaries, heads, fw, cw):
    lang = LANG_NAME[doc.lang]
    mtxt = "\n".join("- %s: %s" % (f, (meta.get(f) or {}).get("value", UNKNOWN)) for f in META_FIELDS)
    user = ("Document (catalogue title): %s\nIdentity metadata already verified from the text (do not change it):\n%s\n"
            "Top-level sections: %s\n\nDescriptions of the document's consecutive parts, covering the whole document:\n"
            "<<<\n%s\n>>>\n\nReturn exactly this JSON shape:\n"
            "{\"kind\": \"<what sort of document this is, a few words; 'unknown' if unclear>\", "
            "\"short_title\": \"<the document's title shortened to at most 8 words, in its own words>\", "
            "\"summary\": \"<at most 80 words: the document as a whole>\", "
            "\"full_card\": \"<at most %d words: its main subjects, named entities, periods discussed and kinds of data, "
            "so a reader can decide whether to search it>\", "
            "\"compact_card\": \"<at most %d words: the same, only the most distinctive content>\"}\n%s\n"
            "- Do not restate the title or the metadata inside full_card or compact_card; no publication year there.") % (
        title, mtxt, " | ".join(heads) or "(none)", "\n".join(summaries), fw, cw, RULES.format(lang=lang))
    return [{"role": "system", "content": SYS}, {"role": "user", "content": user}]


def prompt_tokens(msgs):
    return sum(ntok(m["content"]) for m in msgs) + 12 * len(msgs)      # + chat template overhead (approx.)


def parse_json(text):
    t = (text or "").strip()
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t)
    a = t.find("{")
    if a < 0:
        raise ValueError("no JSON object")
    b = t.rfind("}")
    if b > a:
        try:
            return json.loads(t[a:b + 1])
        except ValueError:
            pass
    # gpt-oss sometimes stops before the outer closing brace(s): complete the object deterministically (logged by
    # the caller as a repaired reply)
    for tail in ("}", "}}", "\"}", "\"}}", "\"}}}", "}}}"):
        try:
            obj = json.loads(t[a:] + tail)
            obj["_repaired"] = tail
            return obj
        except ValueError:
            continue
    raise ValueError("invalid JSON object")


# ------------------------------------------------------------------------------------------------------------------
# LLM calls: cache (resumable), budget guard, <= 4 in flight
# ------------------------------------------------------------------------------------------------------------------
class Stop(Exception):
    pass


class Caller:
    def __init__(self, pool, dry=False):
        self.pool = pool
        self.dry = dry
        self.sem = threading.Semaphore(CONCURRENCY)
        self.lock = threading.Lock()
        self.local = threading.local()
        self.cache_path = OUT / "cache" / ("%s.calls.jsonl" % pool)
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache = {}
        if self.cache_path.exists():
            for line in open(self.cache_path, encoding="utf-8"):
                if line.strip():
                    r = json.loads(line)
                    self.cache[r["key"]] = r
        self.start_budget = budget_now()
        st = read_state()
        self.task_start = st.get("task_start_budget", self.start_budget)
        self.new = {"calls": 0, "ptok": 0, "ctok": 0, "cost": 0.0, "fail": 0}

    def reader(self):
        if not hasattr(self.local, "r"):
            from ...config import load_config
            from ...llm import Ledger
            from ...llm_openrouter import OpenRouterReader
            ch = json.load(open(PROVIDERS, encoding="utf-8"))["choice"][MODEL]
            self.local.r = OpenRouterReader(load_config(), Ledger(path=None, keep_in_memory=False), MODEL,
                                            provider={"provider": ch["provider"], "params": ch["params"]},
                                            attempts=3, reasoning_effort="low")
        return self.local.r

    def call(self, msgs, purpose, doc_id, num_predict):
        key = hashlib.sha1(json.dumps([PROMPT_VERSION, MODEL, msgs], ensure_ascii=False).encode("utf-8")).hexdigest()
        if key in self.cache and self.cache[key].get("ok"):
            return self.cache[key]
        ptoks = prompt_tokens(msgs)
        if ptoks > BLOCK_MAX_INPUT + 64:
            raise ValueError("prompt over budget: %d tokens" % ptoks)
        with self.sem:
            spent = budget_now()
            if spent >= HARD_STOP_USD:
                raise Stop("reader budget %.4f reached the hard stop %.2f" % (spent, HARD_STOP_USD))
            if spent - self.task_start >= TASK_CAP_USD:
                raise Stop("task spend %.4f reached the task cap %.2f" % (spent - self.task_start, TASK_CAP_USD))
            t0 = time.perf_counter()
            rec = {"key": key, "purpose": purpose, "doc": doc_id, "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
                   "est_prompt_tokens": ptoks}
            try:
                res = self.reader().chat(msgs, purpose="lex6-card-" + purpose, phase="index", approach="LEX6-cards",
                                         num_predict=num_predict)
                rec.update(ok=True, text=res.text, ptok=res.prompt_tokens, ctok=res.completion_tokens,
                           reasoning_tokens=getattr(res, "reasoning_tokens", None), cost=float(res.cost_usd or 0.0),
                           finish=res.done_reason, provider=getattr(res, "provider", None))
            except Exception as e:  # noqa: BLE001
                from ...llm import BudgetExceeded
                if isinstance(e, BudgetExceeded):
                    raise Stop(str(e))
                rec.update(ok=False, error=("%s: %s" % (type(e).__name__, e))[:300], ptok=0, ctok=0, cost=0.0)
            rec["wall_ms"] = round((time.perf_counter() - t0) * 1000.0, 1)
        with self.lock:
            with open(self.cache_path, "a", encoding="utf-8", newline="\n") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            self.cache[key] = rec
            self.new["calls"] += 1
            self.new["ptok"] += rec["ptok"]
            self.new["ctok"] += rec["ctok"]
            self.new["cost"] += rec["cost"]
            self.new["fail"] += 0 if rec["ok"] else 1
        return rec

    def json_call(self, msgs, purpose, doc_id, num_predict, check=None):
        """-> (obj or None, [call records]). One re-ask when the reply is not valid JSON / fails `check`."""
        recs = []
        for attempt in range(2):
            m = msgs if attempt == 0 else msgs + [{"role": "user", "content":
                                                   "Your previous reply was not the required JSON object. Reply "
                                                   "again with ONLY the JSON object, complete."}]
            r = self.call(m, purpose, doc_id, num_predict)
            recs.append(r)
            if not r.get("ok"):
                continue
            try:
                obj = parse_json(r["text"])
                if check:
                    check(obj)
                return obj, recs
            except (ValueError, KeyError, TypeError, AttributeError):
                continue
        return None, recs


def read_budget():
    try:
        return json.loads(BUDGET.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def budget_now():
    return float(read_budget().get("cost_usd", 0.0))


def read_state():
    p = OUT / "state.json"
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def write_json(p, obj, indent=1):
    tmp = str(p) + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:          # no BOM
        json.dump(obj, f, ensure_ascii=False, indent=indent)
    os.replace(tmp, p)


# ------------------------------------------------------------------------------------------------------------------
# per document
# ------------------------------------------------------------------------------------------------------------------

def verify_meta(meta, block_text):
    """keep a field only when its evidence snippet occurs in the block (after fold(); marker brackets ignored)."""
    out, ft = {}, fold(block_text)
    fsq = re.sub(r"\s+", " ", ft)
    for f in META_FIELDS:
        v = (meta or {}).get(f, UNKNOWN)
        if not isinstance(v, dict) or not str(v.get("value", "")).strip() or \
                str(v.get("value", "")).strip().lower() == UNKNOWN:
            out[f] = {"value": UNKNOWN, "status": "unknown"}
            continue
        ev = str(v.get("evidence") or "").strip()
        fev = re.sub(r"\s+", " ", fold(re.sub(r"^=+ \[S\d+\] |\s*=+$", "", ev))).strip(" .…\"'«»")
        if fev and len(fev) >= 3 and fev in fsq:
            out[f] = {"value": str(v["value"]).strip(), "evidence": ev, "status": "supported"}
        else:
            out[f] = {"value": UNKNOWN, "status": "unsupported", "model_value": str(v.get("value"))[:200],
                      "model_evidence": ev[:200]}
    return out


def reduce_lines(C, doc, title, lines, what, words, purpose, recs):
    """bounded reduce: consecutive groups of lines (<= REDUCE_INPUT tokens) -> one line each, until one call fits."""
    level = 0
    while sum(ntok_many(lines)) > REDUCE_INPUT:
        level += 1
        groups, cur, ct = [], [], 0
        for ln, t in zip(lines, ntok_many(lines)):
            if t > REDUCE_INPUT:
                ln, t = cut_tokens(ln, REDUCE_INPUT // 2), REDUCE_INPUT // 2
            if cur and ct + t > REDUCE_INPUT:
                groups.append(cur); cur, ct = [], 0
            cur.append(ln); ct += t
        if cur:
            groups.append(cur)
        new = []
        for gi, g in enumerate(groups):
            obj, rr = C.json_call(combine_prompt(doc, title, "Descriptions of consecutive parts of %s, in order. "
                                                 "Describe them together." % what, g, 60),
                                  purpose + "-reduce", doc.id, 400, check=lambda o: o["text"].strip())
            recs += rr
            new.append("[%d] %s" % (gi + 1, obj["text"].strip()) if obj else
                       "[%d] (reduce failed) %s" % (gi + 1, cut_tokens(" ".join(g), 200)))
        if len(new) >= len(lines):          # no progress (cannot happen with >1 line per group) - stop
            break
        lines = new
    return lines, level


def card_doc(C, pool, idx, doc):
    title = (doc.title_raw or doc.title or doc.id).strip()
    segs, sub_tok = segments(doc)
    blocks = fit_blocks(doc, title, blocks_of(doc, segs))
    recs, failures, uncovered = [], [], []
    nb = len(blocks)
    # ---- 3. block calls (parallel, <= CONCURRENCY in flight overall)
    def do_block(bi):
        items = blocks[bi]
        msgs = block_prompt(doc, title, bi, nb, items, bi == 0)
        if prompt_tokens(msgs) > BLOCK_MAX_INPUT:
            raise ValueError("block over the input budget")
        need = ["S%d" % k for k in range(1, len(items) + 1)]
        def check(o):
            assert isinstance(o["block_summary"], str) and o["block_summary"].strip()
            segs_o = o["segments"]
            missing = [k for k in need if not str(segs_o.get(k, "")).strip()]
            if len(missing) > max(0, len(need) // 4):
                raise ValueError("segments missing: %s" % missing[:5])
        obj, rr = C.json_call(msgs, "block", doc.id, 350 + 60 * len(items) + (250 if bi == 0 else 0), check=check)
        return bi, obj, rr, prompt_tokens(msgs), msgs
    with ThreadPoolExecutor(CONCURRENCY) as ex:
        res = list(ex.map(do_block, range(nb)))
    blocks_out, item_line, meta = [], {}, None
    for bi, obj, rr, pt, msgs in res:
        recs += rr
        items = blocks[bi]
        a = min(g["units"][0] for it in items for g in it)
        b = max(g["units"][1] for it in items for g in it)
        span = [items[0][0]["span"][0], items[-1][-1]["span"][1]]
        bo = {"i": bi, "units": [a, b], "span": span, "input_tokens": pt, "n_items": len(items)}
        if obj is None:
            bo.update(status="failed", summary=None)
            uncovered.append([a, b])
            failures.append({"what": "block", "block": bi, "units": [a, b],
                             "error": (rr[-1].get("error") if rr else None) or "invalid reply"})
        else:
            s = str(obj["block_summary"]).strip()
            bo.update(status="ok" if len(s.split()) <= BLOCK_SUMMARY_WORDS else "truncated",
                      summary=cut_words(s, BLOCK_SUMMARY_WORDS))
            if obj.get("_repaired"):
                bo["json_repaired"] = obj["_repaired"]
            so = obj.get("segments") or {}
            for k, it in enumerate(items, 1):
                ln = str(so.get("S%d" % k, "")).strip()
                item_line[(bi, k)] = ln or None
            if bi == 0:
                # what the model saw: the block text WITH its section-heading markers (headings are source
                # text too; a root title may sit only in the heading), plus the catalogue title
                btxt = msgs[1]["content"].split("<<<", 1)[1].rsplit(">>>", 1)[0] + " \n " + title
                meta = verify_meta(obj.get("metadata") if isinstance(obj.get("metadata"), dict) else {}, btxt)
        blocks_out.append(bo)
    if meta is None:
        meta = {f: {"value": UNKNOWN, "status": "unknown"} for f in META_FIELDS}
    # ---- 4. section cards
    cards = []                    # section card records
    by_sid_card = {}              # section id -> card text (for parents' combine)
    own_lines = {}                # oversized section id -> [own-text lines]
    for bi, items in enumerate(blocks):
        for k, it in enumerate(items, 1):
            ln = item_line.get((bi, k))
            if it[0]["kind"] in ("own", "loose"):
                if it[0]["sids"]:
                    own_lines.setdefault(it[0]["sids"][0], []).append(ln or "(description failed)")
                if it[0]["kind"] == "loose":
                    cards.append(sec_card(doc, it, ln, "loose", failures))
                continue
            kind = "subtree" if len(it) == 1 else "group"
            c = sec_card(doc, it, ln, kind, failures)
            cards.append(c)
            for g in it:
                by_sid_card[g["sids"][0]] = (c["key"], c["card"]) if len(it) == 1 else None
            if len(it) > 1:
                by_sid_card[it[0]["sids"][0]] = (c["key"], c["card"])        # the group's card, listed once
    oversized = [s for s in doc.sections if sub_tok[s.id] > BLOCK_TEXT_TOKENS]
    for s in reversed(oversized):                 # children before parents
        lines = ["(own text) " + x for x in own_lines.get(s.id, [])]
        for c in s.children:
            v = by_sid_card.get(c)
            if v and v[1]:
                lines.append("%s: %s" % (doc.section(c).title or "(untitled)", v[1]))
        us = doc.section_units(s.id, recursive=True)
        if not lines or not us:
            continue
        lines, lvl = reduce_lines(C, doc, title, lines, "the section '%s'" % _path(doc, s.id), 25, "section", recs)
        obj, rr = C.json_call(combine_prompt(doc, title, "Section: %s\nDescriptions of its parts and subsections, "
                                             "in order. Write ONE description of the whole section." % _path(doc, s.id),
                                             lines, 25, key="card"),
                              "section", doc.id, 300, check=lambda o: o["card"].strip())
        recs += rr
        c = {"key": _path(doc, s.id), "section_ids": [s.id], "kind": "combined",
             "units": [min(u.order for u in us), max(u.order for u in us)], "span": list(s.subtree_span),
             "source_tokens": sub_tok[s.id], "reduce_levels": lvl}
        if obj is None:
            c.update(card=None, tokens=0, status="failed")
            failures.append({"what": "section", "key": c["key"]})
        else:
            fit_card(c, str(obj["card"]).strip(), SECTION_MAX)
        cards.append(c)
        by_sid_card[s.id] = (c["key"], c["card"])
    cards.sort(key=lambda c: (c["units"][0], -c["units"][1]))
    # ---- 5. document summary + cards
    summ = ["[%d] %s" % (b["i"] + 1, b["summary"]) for b in blocks_out if b["summary"]]
    if not summ:
        failures.append({"what": "document", "error": "no block summary"})
    summ, lvl = reduce_lines(C, doc, title, summ, "the document", 60, "doc", recs)
    heads = [cut_tokens(r.title, 20) for r in doc.roots() if r.title][:25]
    if len(doc.roots()) == 1:
        heads = [cut_tokens(doc.section(c).title, 20) for c in doc.roots()[0].children if doc.section(c).title][:25]
    did = "%s%02d" % (DID[pool], idx + 1)
    fw, cw = (45, 14) if doc.lang == "fa" else (60, 18)
    out = {"doc_id": doc.id, "display_id": did, "pool": pool, "lang": doc.lang, "catalogue_title": title,
           "fingerprint": (doc.meta or {}).get("fingerprint"), "n_units": len(doc.units),
           "n_sections": len(doc.sections), "source_tokens": sum(g["tokens"] for g in segs), "n_blocks": nb,
           "metadata": meta, "doc_reduce_levels": lvl}
    final = None
    for attempt in range(2):
        obj, rr = C.json_call(final_prompt(doc, title, meta, summ, heads, fw, cw), "doc", doc.id, 700,
                              check=lambda o: o["full_card"].strip() and o["compact_card"].strip())
        recs += rr
        if obj is None:
            break
        final = obj
        full, compact = compose(did, meta, obj, title)
        if ntok(full) <= FULL_MAX and ntok(compact) <= COMPACT_MAX:
            break
        fw, cw = int(fw * 0.7), int(cw * 0.7)           # regenerate once with a lower word target
    if final is None:
        failures.append({"what": "document", "error": "final call failed"})
        out.update(kind={"value": meta["kind"]["value"], "source": "metadata"}, short_title=title, summary=None,
                   full_card=None, compact_card=None, full_tokens=0, compact_tokens=0, card_status="failed")
    else:
        kind = meta["kind"]["value"]
        ksrc = "metadata"
        if kind == UNKNOWN and str(final.get("kind", "")).strip() and str(final["kind"]).strip().lower() != UNKNOWN:
            kind, ksrc = str(final["kind"]).strip(), "content (model, not stated in the text)"
        full, compact = compose(did, meta, final, title)
        st = "ok"
        if ntok(full) > FULL_MAX:
            full, st = cut_tokens(full, FULL_MAX), "truncated"
        if ntok(compact) > COMPACT_MAX:
            compact, st = cut_tokens(compact, COMPACT_MAX), "truncated"
        out.update(kind={"value": kind, "source": ksrc}, short_title=str(final.get("short_title") or title).strip(),
                   summary=cut_words(str(final.get("summary") or "").strip(), 80), full_card=full,
                   compact_card=compact, full_tokens=ntok(full), compact_tokens=ntok(compact), card_status=st,
                   card_word_targets=[fw, cw])
    out.update(blocks=blocks_out, sections=cards, uncovered_units=uncovered, failures=failures,
               coverage={"units_total": len(doc.units),
                         "units_in_segments": sum(g["units"][1] - g["units"][0] + 1 for g in segs if
                                                  not g.get("unit_part") or g["unit_part"].startswith("1/")),
                         "units_uncovered": sum(b - a + 1 for a, b in uncovered)},
               usage={"calls": len(recs), "failed_calls": sum(1 for r in recs if not r.get("ok")),
                      "prompt_tokens": sum(r.get("ptok", 0) for r in recs),
                      "completion_tokens": sum(r.get("ctok", 0) for r in recs),
                      "cost_usd": round(sum(r.get("cost", 0.0) for r in recs), 6)},
               provenance={"prompt_version": PROMPT_VERSION, "model": MODEL, "provider": sorted(
                   {r.get("provider") for r in recs if r.get("provider")}), "reasoning": "low",
                   "tokenizer": "gpt-oss-20b tokenizer.json (o200k_harmony, HF openai/gpt-oss-20b)",
                   "code_sha1": hashlib.sha1(Path(__file__).read_bytes()).hexdigest()[:12],
                   "made": time.strftime("%Y-%m-%dT%H:%M:%S"), "questions_or_gold_read": False})
    return out


def sec_card(doc, it, ln, kind, failures):
    c = {"key": item_key(doc, it), "section_ids": [s for g in it for s in g["sids"]], "kind": kind,
         "units": [it[0]["units"][0], it[-1]["units"][1]], "span": [it[0]["span"][0], it[-1]["span"][1]],
         "source_tokens": sum(g["tokens"] for g in it)}
    if not ln:
        c.update(card=None, tokens=0, status="failed")
        failures.append({"what": "section", "key": c["key"]})
    else:
        fit_card(c, ln, SECTION_MAX)
    return c


def fit_card(c, text, mx):
    n = ntok(text)
    if n > mx:
        c.update(card=cut_tokens(text, mx), status="truncated")
    else:
        c.update(card=text, status="ok")
    c["tokens"] = ntok(c["card"])


def compose(did, meta, obj, title):
    def mv(f):
        v = meta[f]["value"]
        return None if v == UNKNOWN else v
    short = str(obj.get("short_title") or title).strip()
    kind = mv("kind")
    hdr_full = " | ".join(x for x in ["[%s] %s" % (did, mv("title") or title), kind, mv("author_or_issuer"),
                                      mv("publication_date_or_version"), mv("period_covered")] if x)
    full = hdr_full + " — " + str(obj.get("full_card", "")).strip()
    hdr_c = "[%s] %s" % (did, cut_tokens(short, 24))
    if mv("publication_date_or_version"):
        hdr_c += " (%s)" % cut_tokens(mv("publication_date_or_version"), 10)
    compact = hdr_c + " — " + str(obj.get("compact_card", "")).strip()
    if ntok(hdr_full) > FULL_MAX - 20:
        full = cut_tokens(hdr_full, FULL_MAX - 20) + " — " + str(obj.get("full_card", "")).strip()
    return full, compact


# ------------------------------------------------------------------------------------------------------------------
# pools
# ------------------------------------------------------------------------------------------------------------------
_CFG = {}


def cfg():
    if "c" not in _CFG:
        from ...config import load_config
        _CFG["c"] = load_config()
    return _CFG["c"]


def pool_docs(pool):
    m = json.load(open(ROOT / "work" / "multidoc" / "pools" / ("%s.json" % pool), encoding="utf-8"))
    return [d["doc_id"] for d in m["docs"] if d["set"] == SET_OF_POOL[pool]]


def corpus(pool):
    from ...corpus import load_set
    key = "cs:" + pool
    if key not in _CFG:
        _CFG[key] = load_set(SET_OF_POOL[pool], cfg())
    return _CFG[key]


def progress(line):
    with open(OUT / "PROGRESS.md", "a", encoding="utf-8", newline="\n") as f:
        f.write("- %s %s\n" % (time.strftime("%Y-%m-%d %H:%M"), line.rstrip()))


def done_docs(pool):
    p = OUT / ("%s.jsonl" % pool)
    if not p.exists():
        return {}
    out = {}
    for line in open(p, encoding="utf-8"):
        if line.strip():
            r = json.loads(line)
            out[r["doc_id"]] = r
    return out


def cmd_plan(pools):
    rows, tot_in, tot_calls = [], 0, 0
    for pool in pools:
        cs = corpus(pool)
        p_in = p_blocks = p_items = p_over = p_calls = 0
        for d in pool_docs(pool):
            doc = cs.doc(d)
            segs, sub_tok = segments(doc)
            title = (doc.title_raw or doc.title or doc.id).strip()
            bl = fit_blocks(doc, title, blocks_of(doc, segs))
            n_units = sum(g["units"][1] - g["units"][0] + 1 for g in segs
                          if not g.get("unit_part") or g["unit_part"].startswith("1/"))
            assert n_units == len(doc.units), (d, n_units, len(doc.units))
            title = (doc.title_raw or doc.title or doc.id).strip()
            for bi, items in enumerate(bl):
                pt = prompt_tokens(block_prompt(doc, title, bi, len(bl), items, bi == 0))
                assert pt <= BLOCK_MAX_INPUT, (d, bi, pt)
                p_in += pt
            over = sum(1 for s in doc.sections if sub_tok[s.id] > BLOCK_TEXT_TOKENS)
            p_blocks += len(bl); p_items += sum(len(b) for b in bl); p_over += over
            p_calls += len(bl) + over + 1
        # combine/doc calls: ~ (over + docs) calls of ~1,000 input tokens; output ~ 80 tok/item + 150/block + reasoning
        est_in = p_in + (p_over + len(pool_docs(pool))) * 1200
        est_out = p_items * 70 + p_blocks * 350 + (p_over + len(pool_docs(pool))) * 450
        cost = est_in * PRICE_IN + est_out * PRICE_OUT
        rows.append((pool, len(pool_docs(pool)), p_blocks, p_items, p_over, p_calls, est_in, est_out, cost))
        tot_in += est_in; tot_calls += p_calls
        print("%-8s docs %3d blocks %5d items %6d oversized-sections %5d calls~%5d in~%9d out~%8d $%.3f" % rows[-1],
              flush=True)
    tc = sum(r[-1] for r in rows)
    print("TOTAL calls~%d input~%d tokens  est $%.3f" % (tot_calls, tot_in, tc))
    return rows, tc


def cmd_run(pools):
    OUT.mkdir(parents=True, exist_ok=True)
    st = read_state()
    if "task_start_budget" not in st:
        st["task_start_budget"] = budget_now()
        write_json(OUT / "state.json", st)
    for pool in pools:
        C = Caller(pool)
        have = done_docs(pool)
        ids = pool_docs(pool)
        todo = [(i, d) for i, d in enumerate(ids) if d not in have]
        if not todo:
            print(pool, "already done", flush=True)
            continue
        progress("%s START %d/%d docs to do; reader budget %.4f (task start %.4f)" % (
            pool, len(todo), len(ids), budget_now(), C.task_start))
        cs = corpus(pool)
        t0 = time.perf_counter()
        lock = threading.Lock()
        stopped = []

        def one(item):
            i, d = item
            if stopped:
                return
            try:
                r = card_doc(C, pool, i, cs.doc(d))
            except Stop as e:
                stopped.append(str(e))
                return
            with lock:
                with open(OUT / ("%s.jsonl" % pool), "a", encoding="utf-8", newline="\n") as f:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            print("%s %s blocks %d sections %d full %s compact %s fails %d $%.4f" % (
                pool, r["display_id"], r["n_blocks"], len(r["sections"]), r["full_tokens"], r["compact_tokens"],
                len(r["failures"]), r["usage"]["cost_usd"]), flush=True)

        with ThreadPoolExecutor(2) as ex:          # 2 documents at a time; their calls share the 4-slot semaphore
            list(ex.map(one, todo))
        wall = time.perf_counter() - t0
        have = done_docs(pool)
        st = read_state()
        ps = st.setdefault("pools", {}).setdefault(pool, {"wall_s": 0.0})
        ps["wall_s"] = round(ps["wall_s"] + wall, 1)
        ps.update(docs_done=len(have), docs_total=len(ids),
                  prompt_tokens=sum(r["usage"]["prompt_tokens"] for r in have.values()),
                  completion_tokens=sum(r["usage"]["completion_tokens"] for r in have.values()),
                  cost_usd=round(sum(r["usage"]["cost_usd"] for r in have.values()), 6),
                  calls=sum(r["usage"]["calls"] for r in have.values()),
                  failed_calls=sum(r["usage"]["failed_calls"] for r in have.values()))
        write_json(OUT / "state.json", st)
        progress("%s %s: %d/%d docs; this run %d new calls (%d failed), %d+%d tokens, $%.4f, %.0f s; pool total "
                 "$%.4f; reader budget %.4f%s" % (
                     pool, "DONE" if len(have) == len(ids) else "PARTIAL", len(have), len(ids), C.new["calls"],
                     C.new["fail"], C.new["ptok"], C.new["ctok"], C.new["cost"], wall, ps["cost_usd"], budget_now(),
                     ("; STOPPED: " + stopped[0]) if stopped else ""))
        if stopped:
            print("STOP:", stopped[0], flush=True)
            return False
    return True


# ------------------------------------------------------------------------------------------------------------------
# index: bge-m3 vectors (the chunk index's embedder), card BM25, doc centroids
# ------------------------------------------------------------------------------------------------------------------

def section_text(r, c):
    tail = " › ".join(c["key"].split(" › ")[-2:])
    return "%s | %s: %s" % (r["short_title"] or r["catalogue_title"], tail, c["card"])


def cmd_index(pools):
    import bm25s
    from ...approaches.baselines import DEFAULTS as BLD, tokenize
    from ...llm import Ledger, OllamaClient
    from ...retrieval.hybrid import HybridIndex
    from .docrank import centroids
    client = OllamaClient(cfg(), Ledger(path=None, keep_in_memory=False))
    idx_dir = OUT / "index"
    idx_dir.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for pool in pools:
        have = done_docs(pool)
        ids = [d for d in pool_docs(pool) if d in have]
        t0 = time.perf_counter()
        docs = [have[d] for d in ids]
        full = [(r["doc_id"], r["full_card"]) for r in docs if r.get("full_card")]
        comp = [(r["doc_id"], r["compact_card"]) for r in docs if r.get("compact_card")]
        secs = [("%s#%d" % (r["doc_id"], k), section_text(r, c), r["doc_id"], c["key"], c["units"])
                for r in docs for k, c in enumerate(r["sections"]) if c.get("card")]

        def emb(texts):
            out = []
            for i in range(0, len(texts), 16):
                er = client.embed([normalize(t) for t in texts[i:i + 16]], purpose="doc-embed", phase="index",
                                  approach="LEX6-cards")
                out += [np.asarray(v, dtype=np.float32) for v in er.vectors]
            V = np.stack(out) if out else np.zeros((0, 1024), np.float32)
            return V / np.maximum(np.linalg.norm(V, axis=1, keepdims=True), 1e-12)
        VF, VC, VS = emb([t for _, t in full]), emb([t for _, t in comp]), emb([s[1] for s in secs])
        hx = HybridIndex.load(corpus(pool), cfg())
        cents = centroids(hx, pool_docs(pool))
        cent_ids = [d for d in pool_docs(pool) if d in cents]
        VZ = np.stack([cents[d] for d in cent_ids]).astype(np.float32)
        np.savez(idx_dir / ("%s.npz" % pool), full=VF, compact=VC, sections=VS, centroid=VZ)
        keys = {"full": [d for d, _ in full], "compact": [d for d, _ in comp],
                "sections": [{"id": s[0], "doc_id": s[2], "key": s[3], "units": s[4]} for s in secs],
                "centroid": cent_ids}
        write_json(idx_dir / ("%s.keys.json" % pool), keys)
        p = BLD["bm25"]
        for name, texts in (("full", [t for _, t in full]), ("sections", [s[1] for s in secs])):
            bm = bm25s.BM25(k1=p["k1"], b=p["b"], method=p["method"])
            bm.index([tokenize(normalize(t), BLD["tokenizer"]) for t in texts], show_progress=False)
            bm.save(str(idx_dir / ("%s.bm25_%s" % (pool, name))))
        wall = time.perf_counter() - t0
        manifest[pool] = {"full": len(full), "compact": len(comp), "sections": len(secs), "centroids": len(cent_ids),
                          "dim": int(VF.shape[1]) if len(VF) else None, "embed_model": cfg().models["embed"],
                          "wall_s": round(wall, 1), "note": "vectors L2-normalised; texts normalize()d before "
                          "embedding; BM25 = bm25s with B2's tokenizer/params over normalize()d card text; section "
                          "text = short title | last two path titles: card"}
        progress("%s INDEX: %d full, %d compact, %d section vectors, %d centroids, %.0f s" % (
            pool, len(full), len(comp), len(secs), len(cent_ids), wall))
        print(pool, manifest[pool], flush=True)
    old = {}
    try:
        old = json.loads((idx_dir / "MANIFEST.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        pass
    old.update(manifest)
    write_json(idx_dir / "MANIFEST.json", old)


# ------------------------------------------------------------------------------------------------------------------
# report
# ------------------------------------------------------------------------------------------------------------------

def _dist(xs):
    if not xs:
        return "-"
    a = np.asarray(xs)
    return "n %d, p10 %d, p50 %d, p90 %d, max %d, mean %.1f" % (len(a), np.percentile(a, 10), np.median(a),
                                                                 np.percentile(a, 90), a.max(), a.mean())


def cmd_report(pools):
    from collections import Counter
    st = read_state()
    L = ["# LEX6 offline cards", "",
         "Built by `ragbench/approaches/lex6/cards.py` (%s) from the documents alone: no question and no gold "
         "annotation was read. Model %s, reasoning low, pinned provider (lex5_pilot providers.json), <= %d calls in "
         "flight. Token lengths measured with the gpt-oss-20b tokenizer (HF `openai/gpt-oss-20b` tokenizer.json, "
         "o200k_harmony). Cards guide retrieval only; they never enter reader evidence." % (
             PROMPT_VERSION, MODEL, CONCURRENCY), "",
         "Method: section-aware blocks (<= %d input tokens per call incl. instructions; <= %d source tokens) cover "
         "every unit of every document (full coverage, no sampling); block call = <= %d-word block summary + "
         "<= 25-word line per marked segment (+ identity metadata from block 1, kept only when its evidence snippet "
         "occurs in the block text or its section headings). Section cards (<= %d tokens): fitting sections from "
         "their block line; small sibling sections grouped; oversized sections by a combine call over children "
         "(bounded reduce). Document: block summaries -> bounded reduce -> final call. Full card (<= %d) and compact "
         "card (<= %d) = header [display id + verified metadata] + model text; over-length -> one regeneration at "
         "0.7x the word target, then a token-boundary cut (status `truncated`)." % (
             BLOCK_MAX_INPUT, BLOCK_TEXT_TOKENS, BLOCK_SUMMARY_WORDS, SECTION_MAX, FULL_MAX, COMPACT_MAX), ""]
    L += ["## Counts, cost, time", "",
          "| pool | docs | blocks | section cards | calls | failed calls | input tok | output tok | $ | wall s |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    allr = {}
    tot = [0] * 8
    for pool in pools:
        have = done_docs(pool)
        allr[pool] = [have[d] for d in pool_docs(pool) if d in have]
        rs = allr[pool]
        cp = OUT / "cache" / ("%s.calls.jsonl" % pool)
        cache = [json.loads(x) for x in open(cp, encoding="utf-8") if x.strip()] if cp.exists() else []
        ps = (st.get("pools") or {}).get(pool, {})
        row = [len(rs), sum(r["n_blocks"] for r in rs), sum(len(r["sections"]) for r in rs), len(cache),
               sum(1 for c in cache if not c.get("ok")), sum(c.get("ptok", 0) for c in cache),
               sum(c.get("ctok", 0) for c in cache), sum(c.get("cost", 0.0) for c in cache)]
        tot = [a + b for a, b in zip(tot, row)]
        L.append("| %s | %d/%d | %d | %d | %d | %d | %d | %d | %.4f | %s |" % (
            pool, row[0], len(pool_docs(pool)), row[1], row[2], row[3], row[4], row[5], row[6], row[7],
            ps.get("wall_s", "-")))
    L.append("| **all** | %d | %d | %d | %d | %d | %d | %d | **%.4f** | %.0f |" % (
        tot[0], tot[1], tot[2], tot[3], tot[4], tot[5], tot[6], tot[7],
        sum((v or {}).get("wall_s", 0) for v in (st.get("pools") or {}).values())))
    L += ["", "Calls/tokens/$ = every call made (cache files, incl. superseded calls and JSON re-asks); per-document "
          "usage is in each JSONL row. Reader budget at task start %.4f, now %.4f. Wall = card-generation wall "
          "clock (2 docs x <= 4 calls in flight); index build times are in index/MANIFEST.json." % (
              st.get("task_start_budget", 0), budget_now()), ""]
    L += ["## Token lengths (gpt-oss tokenizer)", "", "| lang | full card | compact card | section cards |",
          "|---|---|---|---|"]
    for lg in ("fa", "en"):
        rs = [r for p in pools for r in allr[p] if r["lang"] == lg]
        L.append("| %s | %s | %s | %s |" % (lg, _dist([r["full_tokens"] for r in rs if r.get("full_card")]),
                                           _dist([r["compact_tokens"] for r in rs if r.get("compact_card")]),
                                           _dist([c["tokens"] for r in rs for c in r["sections"] if c.get("card")])))
    L += ["", "## Status and known failures", ""]
    for lg in ("fa", "en"):
        rs = [r for p in pools for r in allr[p] if r["lang"] == lg]
        L.append("- %s: doc card status %s; section card status %s; section kinds %s; metadata field status %s; "
                 "kind taken from content (not stated in the text) for %d docs" % (
                     lg, dict(Counter(r["card_status"] for r in rs)),
                     dict(Counter(c["status"] for r in rs for c in r["sections"])),
                     dict(Counter(c["kind"] for r in rs for c in r["sections"])),
                     dict(Counter(v["status"] for r in rs for v in r["metadata"].values())),
                     sum(1 for r in rs if r["kind"]["source"] != "metadata")))
    fails = [(r["display_id"], f) for p in pools for r in allr[p] for f in r["failures"]]
    unc = [(r["display_id"], r["uncovered_units"]) for p in pools for r in allr[p] if r["uncovered_units"]]
    L.append("- explicit failures: %d%s" % (len(fails), "".join(
        "\n  - %s %s" % (d, json.dumps(f, ensure_ascii=False)[:200]) for d, f in fails[:30])))
    L.append("- uncovered unit intervals (failed blocks): %s" % (unc or "none"))
    L.append("- publication date/version known for %d of %d docs (only when stated with verbatim evidence in block 1)"
             % (sum(1 for p in pools for r in allr[p]
                    if r["metadata"]["publication_date_or_version"]["value"] != UNKNOWN),
                sum(len(allr[p]) for p in pools)))
    rep = sum(1 for p in pools for r in allr[p] for b in r["blocks"] if b.get("json_repaired"))
    L.append("- %d block replies were missing their closing brace(s) and were completed deterministically "
             "(`json_repaired` on the block); 9 documents were re-run from the cache after this parser fix "
             "(earlier rows kept in cache/*.before-json-repair.jsonl)" % rep)
    L.append("- known quality issues (spot audit, not exhaustive): the model sometimes gives the period year as the "
             "publication year when the cover states only one year (e.g. pub = period = 1401), or a non-date as "
             "'version' (RE15: the report title); a section line may restate its heading ('Table of contents ...'); "
             "compact cards sometimes repeat the title instead of adding content; failed section cards are lines "
             "the block reply omitted (<= 25% of a block's segments tolerated), their text stays searchable via "
             "the chunk index")
    L += ["", "## Examples (5 per language: compact + full + 2 section cards)", ""]
    for lg in ("fa", "en"):
        lp = [p for p in pools if allr[p] and allr[p][0]["lang"] == lg]
        pick, k = [], 0                   # round-robin over the language's pools, spread inside each pool
        while len(pick) < 5 and k < 5:
            for p in lp:
                rs = allr[p]
                j = (len(rs) * (2 * k + 1)) // 6
                if len(pick) < 5 and j < len(rs) and rs[j] not in pick:
                    pick.append(rs[j])
            k += 1
        for r in pick:
            L += ["### %s `%s` (%s)" % (r["display_id"], r["doc_id"], r["pool"]), "",
                  "- compact (%d tok): %s" % (r["compact_tokens"], r["compact_card"]),
                  "- full (%d tok): %s" % (r["full_tokens"], r["full_card"])]
            secs = [c for c in r["sections"] if c.get("card") and c["kind"] != "combined"] or r["sections"]
            for c in (secs[:1] + secs[len(secs) // 2:len(secs) // 2 + 1])[:2]:
                L.append("- section `%s` units %s (%s, %d tok): %s" % (c["key"][:120], c["units"], c["kind"],
                                                                      c["tokens"], c["card"]))
            L.append("")
    L += ["## Files", "", "- `<pool>.jsonl`: one row per document (metadata with evidence/status, blocks with unit "
          "intervals, section cards keyed by section path with unit intervals, full/compact cards, failures, "
          "uncovered, usage, provenance)",
          "- `cache/<pool>.calls.jsonl`: every model call (resume cache)",
          "- `index/<pool>.npz` (full, compact, sections, centroid: L2-normalised bge-m3 vectors; centroid = mean of "
          "the doc's chunk vectors from the prebuilt HYBRID index, for S5) + `index/<pool>.keys.json` + "
          "`index/<pool>.bm25_full`, `.bm25_sections` (bm25s, B2 tokenizer over normalize()d text), `MANIFEST.json`",
          "- `state.json`, `PROGRESS.md`"]
    with open(OUT / "CARDS.md", "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L[:30]))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["plan", "run", "index", "report"])
    ap.add_argument("pools", nargs="*")
    a = ap.parse_args(argv)
    pools = a.pools or POOLS
    try:
        from ...retrieval.prebuild import lower_priority
        lower_priority("below_normal")
    except Exception:  # noqa: BLE001
        pass
    if a.cmd == "plan":
        cmd_plan(pools)
    elif a.cmd == "run":
        ok = cmd_run(pools)
        sys.exit(0 if ok else 3)
    elif a.cmd == "index":
        cmd_index(pools)
    else:
        cmd_report(pools)


if __name__ == "__main__":
    main()
