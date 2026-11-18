"""LEX6 ROUTER extension (Astra R2 section 5 step 6): one optional, strictly bounded document-routing call.

Trigger (declared before any call; an intervention policy, not a confidence claim):
    (a) the top-3 documents by V (chunk prior) and by card-only fusion differ as sets, OR
    (b) one document holds > 50% of B0's rendered characters while the shortlist has other documents,
    AND (c) a prospective intervention exists: some shortlisted document's best whole-question local package is not
        represented in B0.
Call: openai/gpt-oss-20b on the pinned endpoint (lex5_pilot/providers.json), 1 attempt, no repair, no hidden retry
(no structured-output contract: the empty-reply retry of OpenRouterReader.chat never fires), 650 ms TOTAL deadline
with cancellation, 160 completion tokens INCLUDING reasoning, reasoning effort low, prompt <= 800 tokens (content
measured with the gpt-oss tokenizer <= PROMPT_CONTENT_MAX; the rest is the chat template; the provider's count is
recorded). Shortlist cards are dropped from the end of the shortlist order until the prompt fits; fewer than 2 cards
-> no call ('no_fit').
Reply grammar (strict, no repair):
    P1: 3-7 | D2 D5        word range of the question part (numbered words, 1-based, inclusive) | 1-2 document ids
    P2: 9-12 | D1
    END
    or: NONE / END         (valid, status all_none -> deterministic path; NONE never forces reader abstention)
Rejections: empty, truncated (finish_reason length), no END / text after END, bad line, part index out of order,
range outside the question or reversed, unknown document id, repeated id in a part, > 2 ids in a part, > MAX_PARTS
parts, > 8 assignments, > 4 distinct documents. Any failure (deadline, HTTP, rejection) -> the deterministic
nominations; an integrity failure of the pipeline -> B0 (approach.py).
"""
import random
import re
import time
from pathlib import Path

from ..lex5 import bundles as B
from . import core as K6

ROUTER_VERSION = "lex6-router-v1"
MODEL = "openai/gpt-oss-20b"
R = {"deadline_s": 0.65, "max_tokens_total": 160, "prompt_max": 800, "content_max": 700, "attempts": 1,
     "reasoning_effort": "low", "max_parts": 4, "max_ids_part": 2, "max_assign": 8, "max_docs": 4,
     "dominance": 0.5, "permute_seed": 0}
TOKENIZER = Path(r"D:\KareOne\RAG Paper\work\models\tok\gpt-oss-20b.tokenizer.json")

PROMPT = """Which documents answer each part of the question? The documents:
{cards}

Question, with numbered words:
{words}

Split the question into the parts it asks for (at most 4; a simple question is one part). For each part give the range of its word numbers and the 1 or 2 documents above most likely to contain the answer to that part, best first. Use at most 4 different documents in total.
Reply only with lines in this form, then the line END:
P1: 3-7 | D2 D5
P2: 9-12 | D1
END
If no listed document fits, reply with the line NONE and then END."""

_DIGITS = str.maketrans("۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩", "01234567890123456789")
_INVIS = dict.fromkeys(map(ord, "‌‍‎‏؜‪‫‬‭‮﻿"), None)
_LINE = re.compile(r"^P(\d+)\s*:\s*(\d+)\s*[-–]\s*(\d+)\s*\|\s*(.+)$", re.I)
_ID = re.compile(r"^D(\d+)$", re.I)
_NONE = re.compile(r"^(?:none|هیچ)$", re.I)

_TOK = {}


def ntok(text):
    if "t" not in _TOK:
        import tokenizers
        _TOK["t"] = tokenizers.Tokenizer.from_file(str(TOKENIZER))
    return len(_TOK["t"].encode(text or "", add_special_tokens=False).ids)


def words(q):
    """-> [(start, end, word)] whitespace words of the question."""
    return [(m.start(), m.end(), m.group(0)) for m in re.finditer(r"\S+", q or "")]


def build_prompt(question, shortlist, compact, permute=False, content_max=None):
    """-> (prompt, shown doc ids in prompt order, tokens) or (None, [], tokens) when fewer than 2 cards fit."""
    content_max = content_max or R["content_max"]
    ws = words(question)
    wtxt = " ".join("%d:%s" % (i, w) for i, (_a, _b, w) in enumerate(ws, 1))
    docs = list(shortlist)
    if permute:
        random.Random(R["permute_seed"]).shuffle(docs)
    n = len(docs)
    while n >= 2:
        shown = docs[:n] if not permute else [d for d in docs if d in set(shortlist[:n])]
        cards = "\n".join("D%d %s" % (i, " ".join((compact(d) or "").split())) for i, d in enumerate(shown, 1))
        p = PROMPT.format(cards=cards, words=wtxt)
        t = ntok(p)
        if t <= content_max:
            return p, shown, t
        n -= 1
    return None, [], None


def parse(text, n_words, n_docs, finish_reason=None):
    """-> dict(ok, status ok|all_none|fail, reason, parts [{"words": [a, b], "ids": [1-based doc numbers]}])."""
    out = {"ok": False, "status": "fail", "reason": None, "parts": []}
    if finish_reason == "length":
        out["reason"] = "truncated"
        return out
    lines = [ln.translate(_INVIS).translate(_DIGITS).strip() for ln in (text or "").splitlines()]
    lines = [ln for ln in lines if ln]
    if lines and lines[0].startswith("```") and lines[-1] == "```":
        lines = lines[1:-1]
    if not lines:
        out["reason"] = "empty"
        return out
    up = [ln.upper() for ln in lines]
    if "END" not in up:
        out["reason"] = "no_end"
        return out
    if up.index("END") != len(lines) - 1:
        out["reason"] = "text_after_end"
        return out
    body = lines[:-1]
    if len(body) == 1 and _NONE.match(body[0]):
        out.update(ok=True, status="all_none")
        return out
    if not body:
        out["reason"] = "no_parts"
        return out
    if len(body) > R["max_parts"]:
        out["reason"] = "overflow_parts"
        return out
    parts, docs, assign = [], set(), 0
    for k, ln in enumerate(body, 1):
        m = _LINE.match(ln)
        if not m:
            out["reason"] = "bad_line"
            return out
        if int(m.group(1)) != k:
            out["reason"] = "part_index"
            return out
        a, b = int(m.group(2)), int(m.group(3))
        if not (1 <= a <= b <= n_words):
            out["reason"] = "bad_range"
            return out
        toks = [t for t in re.split(r"[\s,،]+", m.group(4).strip()) if t]
        ids = []
        for t in toks:
            mm = _ID.match(t)
            if not mm:
                out["reason"] = "bad_ids"
                return out
            i = int(mm.group(1))
            if not 1 <= i <= n_docs:
                out["reason"] = "id_out_of_range"
                return out
            if i in ids:
                out["reason"] = "duplicate_id_in_part"
                return out
            ids.append(i)
        if not ids:
            out["reason"] = "bad_ids"
            return out
        if len(ids) > R["max_ids_part"]:
            out["reason"] = "overflow_ids"
            return out
        assign += len(ids)
        docs |= set(ids)
        parts.append({"words": [a, b], "ids": ids})
    if assign > R["max_assign"]:
        out["reason"] = "overflow_assignments"
        return out
    if len(docs) > R["max_docs"]:
        out["reason"] = "overflow_docs"
        return out
    out.update(ok=True, status="ok", parts=parts)
    return out


def trigger(info):
    """info: plan() router input. -> dict(triggered, disagree, dominance, share, prospective, prospective_docs)."""
    Rk, SL, B0, ctx, package, hx, cost = (info[k] for k in ("R", "shortlist", "B0", "ctx", "package", "hx", "cost"))
    q = info["question"]
    card = Rk.get("card")
    disagree = bool(card) and set(Rk["V"][:3]) != set(card[:3])
    by = {}
    for n, p in enumerate(B0, 1):
        by[p["doc_id"]] = by.get(p["doc_id"], 0) + len(p["text"])
    tot = sum(by.values()) or 1
    share = max(by.values()) / tot if by else 0.0
    top_doc = max(by, key=by.get) if by else None
    dominance = share > R["dominance"] and any(d != top_doc for d in SL)
    pros = []
    for d in SL:
        alts = K6.alternatives(K6.local_order(ctx, d, q), package, n=1)
        if alts and not B.represented(alts[0][0], B0):
            pros.append(d)
    return {"triggered": bool((disagree or dominance) and pros), "disagree": disagree, "dominance": dominance,
            "share": round(share, 3), "prospective": bool(pros), "prospective_docs": pros}


def client(cfg, ledger):
    """The pinned router client (OPENROUTER_PROVIDER_PINS or lex5_pilot/providers.json)."""
    import json
    import os
    from ...llm_openrouter import OpenRouterReader, model_info
    pin = None
    if not (json.loads(os.environ.get("OPENROUTER_PROVIDER_PINS") or "{}") or {}).get(MODEL):
        pf = Path(cfg.project_root) / "work" / "datasets_v2" / "lex5_pilot" / "providers.json"
        if pf.exists():
            ch = json.loads(pf.read_text(encoding="utf-8"))["choice"].get(MODEL)
            if ch:
                pin = {"provider": ch["provider"], "params": ch["params"]}
    info = model_info(MODEL, Path(cfg.project_root) / "work" / "openrouter" / "catalog.json")
    return OpenRouterReader(cfg, ledger, MODEL, supported=info.get("params"), provider=pin, attempts=R["attempts"],
                            deadline_s=R["deadline_s"], max_tokens_total=R["max_tokens_total"],
                            reasoning_effort=R["reasoning_effort"])


def make_router(chat_client, compact, permute=False, force=False, ledger_ctx=None):
    """-> callable(plan router info) -> result dict. chat_client: an OpenRouterReader (or a fake with .chat);
    compact(doc) -> compact card text. force: call even when the trigger does not fire (diagnostics only)."""
    def route(info):
        t0 = time.perf_counter()
        out = {"version": ROUTER_VERSION, "use": False, "status": None, "reason": None}
        tg = trigger(info)
        out["trigger"] = tg
        out["t_trigger_ms"] = round((time.perf_counter() - t0) * 1000.0, 1)
        if not tg["triggered"] and not force:
            out.update(status="not_triggered")
            return out
        q = info["question"]
        p, shown, t = build_prompt(q, info["shortlist"], compact, permute=permute)
        out.update(prompt_tokens_est=t, shown_docs=shown, permuted=bool(permute))
        if p is None:
            out.update(status="skipped", reason="no_fit")
            return out
        out["prompt"] = p
        from ...llm import BudgetExceeded
        t1 = time.perf_counter()
        text, finish = None, None
        try:
            res = chat_client.chat([{"role": "user", "content": p}], purpose="lex6_route",
                                   num_predict=R["max_tokens_total"], **(ledger_ctx or {}))
            text, finish = res.text, res.done_reason
            out.update(prompt_tokens=res.prompt_tokens, completion_tokens=res.completion_tokens,
                       reasoning_tokens=getattr(res, "reasoning_tokens", None), cost_usd=res.cost_usd,
                       provider=getattr(res, "provider", None), finish=finish, usage_known=True, raw=(text or "")[:600])
        except BudgetExceeded:
            raise
        except Exception as e:  # noqa: BLE001  deadline / HTTP -> deterministic path, no repair
            out.update(status="fail", reason="deadline" if "deadline" in str(e).lower() else "call_error",
                       error=str(e)[:200], usage_known=False)
        out["t_call_ms"] = round((time.perf_counter() - t1) * 1000.0, 1)
        if text is not None:
            pr = parse(text, len(words(q)), len(shown), finish_reason=finish)
            out.update(status=pr["status"], reason=pr["reason"])
            if pr["ok"] and pr["status"] == "ok":
                ws = words(q)
                parts = []
                for prt in pr["parts"]:
                    a, b = prt["words"]
                    parts.append({"text": q[ws[a - 1][0]:ws[b - 1][1]], "words": [a, b],
                                  "docs": [shown[i - 1] for i in prt["ids"]], "source": "router"})
                out.update(use=True, parts=parts)
        out["t_router_ms"] = round((time.perf_counter() - t0) * 1000.0, 1)
        return out
    return route
