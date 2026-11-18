"""LEX7 order fixer: one small, strictly bounded LLM call that picks which of the top-12 candidates go first.

The gap left by the bi-encoder / sparse fusion is the ORDER of the first ~6 items (lex_research/OFFLINE_TESTS.md),
not recall. This call sees compact views only (section path + a <= 45-token query-hit snippet per candidate) and
returns the ids to put first. Everything else keeps the base order.

Call: openai/gpt-oss-20b, pinned endpoint (lex5_pilot/providers.json), reasoning effort low, 1 attempt, no repair,
deadline 1.5 s TOTAL, <= 120 completion tokens INCLUDING reasoning, prompt <= 900 tokens (content <= 820 by the
gpt-oss tokenizer). Over the content cap: snippet cap -5 tokens (down to 25), then drop candidates from the end (>= 6).
Reply grammar (strict):
    FIRST: 3 7 1       1..6 distinct ids in 1..n, best first
    END
    or NONE / END      (valid; base order kept)
Any failure (deadline, HTTP, finish=length, grammar) -> base order unchanged. Domain-neutral prompt, fa+en.
"""
import re
import time

from ..lex5 import select as S5
from ..lex6.router import ntok

ORDERFIX_VERSION = "lex7-orderfix-v1"
MODEL = "openai/gpt-oss-20b"
R = {"n": 12, "deadline_s": 1.5, "max_tokens_total": 120, "prompt_max": 900, "content_max": 820, "attempts": 1,
     "reasoning_effort": "low", "max_first": 6, "min_keep": 6, "snip_tokens": 45, "snip_min": 25, "path_tokens": 12}
# LEX7.1 (2026-09-26): v2 = v1 with a 200-token completion cap incl. reasoning (v1 truncated 12/54 calls at 120);
# prompt caps, deadline, model, pin, grammar and fallback unchanged
ORDERFIX_V2 = "lex7-orderfix-v2"
R2 = dict(R, max_tokens_total=200)

PROMPT = """Question: {question}

Candidate passages (section, then an excerpt):
{views}

Which passages most directly contain the information the question asks for? List up to 6 passage numbers, best first. If none is relevant, reply NONE.
Reply only with these two lines:
FIRST: <numbers separated by spaces>
END"""

_DIGITS = str.maketrans("۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩", "01234567890123456789")
_INVIS = dict.fromkeys(map(ord, "‌‍‎‏؜‪‫‬‭‮﻿"), None)
_FIRST = re.compile(r"^FIRST\s*:\s*([\d\s,]+)$", re.I)
_NONE = re.compile(r"^(?:FIRST\s*:\s*)?(?:none|هیچ)$", re.I)
_TOK = {}


def _cut(text, n):
    """text cut to at most n gpt-oss tokens (by decoding the first n ids)."""
    if "t" not in _TOK:
        import tokenizers
        from ..lex6.router import TOKENIZER
        _TOK["t"] = tokenizers.Tokenizer.from_file(str(TOKENIZER))
    enc = _TOK["t"].encode(text or "", add_special_tokens=False)
    if len(enc.ids) <= n:
        return text
    return _TOK["t"].decode(enc.ids[:n]).rstrip() + " …"


def views(question, cands, snip_tokens, path_tokens=None):
    """cands: [{"path": breadcrumb 'root > ... > leaf', "text": chunk text}] -> ["[n] root > leaf\\nsnippet"]."""
    path_tokens = path_tokens or R["path_tokens"]
    qwords = {w for _a, _b, w in S5._words(question) if len(w) > 1}
    df = {}
    for c in cands:
        for w in {w for _a, _b, w in S5._words(c["text"])} & qwords:
            df[w] = df.get(w, 0) + 1
    import math
    n = len(cands) or 1
    weights = {w: math.log(1.0 + n / df[w]) for w in df}
    out = []
    for i, c in enumerate(cands, 1):
        segs = [s.strip() for s in (c.get("path") or "").split(">") if s.strip()]
        path = " > ".join(segs if len(segs) <= 2 else [segs[0], segs[-1]])     # document root + leaf section
        path = _cut(" ".join(path.split()), path_tokens) if path else "-"
        sn = S5.snippet(c["text"], qwords, weights, width=260)
        out.append("[%d] %s\n%s" % (i, path, _cut(" ".join(sn.split()), snip_tokens)))
    return out


def build_prompt(question, cands):
    """-> (prompt, n shown, content tokens, snippet cap) or (None, 0, tokens, cap)."""
    n, cap = len(cands), R["snip_tokens"]
    while True:
        p = PROMPT.format(question=question.strip(), views="\n".join(views(question, cands[:n], cap)))
        t = ntok(p)
        if t <= R["content_max"]:
            return p, n, t, cap
        if cap - 5 >= R["snip_min"]:
            cap -= 5
        elif n - 1 >= R["min_keep"]:
            n -= 1
        else:
            return None, 0, t, cap


def parse(text, n, finish_reason=None):
    """-> dict(ok, status ok|none|fail, reason, ids [1-based])."""
    out = {"ok": False, "status": "fail", "reason": None, "ids": []}
    if finish_reason == "length":
        out["reason"] = "truncated"
        return out
    lines = [ln.translate(_INVIS).translate(_DIGITS).strip() for ln in (text or "").splitlines()]
    lines = [ln for ln in lines if ln]
    if not lines:
        out["reason"] = "empty"
        return out
    if len(lines) != 2 or lines[1].upper() != "END":
        out["reason"] = "no_end" if "END" not in [x.upper() for x in lines] else "bad_lines"
        return out
    if _NONE.match(lines[0]):
        out.update(ok=True, status="none")
        return out
    m = _FIRST.match(lines[0])
    if not m:
        out["reason"] = "bad_line"
        return out
    ids = [int(x) for x in re.split(r"[\s,]+", m.group(1).strip()) if x]
    if not ids or len(ids) > R["max_first"]:
        out["reason"] = "count"
        return out
    if len(set(ids)) != len(ids):
        out["reason"] = "duplicate"
        return out
    if any(not 1 <= i <= n for i in ids):
        out["reason"] = "out_of_range"
        return out
    out.update(ok=True, status="ok", ids=ids)
    return out


def reorder(order, shown, ids):
    first = [shown[i - 1] for i in ids]
    return first + [c for c in order if c not in set(first)]


def client(cfg, ledger, caps=None):
    """The pinned gpt-oss-20b client with the order fixer's caps (caps: R (v1, default) or R2 (v2))."""
    caps = caps or R
    import json
    import os
    from pathlib import Path
    from ...llm_openrouter import OpenRouterReader, model_info
    pin = None
    if not (json.loads(os.environ.get("OPENROUTER_PROVIDER_PINS") or "{}") or {}).get(MODEL):
        pf = Path(cfg.project_root) / "work" / "datasets_v2" / "lex5_pilot" / "providers.json"
        if pf.exists():
            ch = json.loads(pf.read_text(encoding="utf-8"))["choice"].get(MODEL)
            if ch:
                pin = {"provider": ch["provider"], "params": ch["params"]}
    info = model_info(MODEL, Path(cfg.project_root) / "work" / "openrouter" / "catalog.json")
    return OpenRouterReader(cfg, ledger, MODEL, supported=info.get("params"), provider=pin, attempts=caps["attempts"],
                            deadline_s=caps["deadline_s"], max_tokens_total=caps["max_tokens_total"],
                            reasoning_effort=caps["reasoning_effort"])


def fix(chat_client, question, order, cand_view, ledger_ctx=None, caps=None):
    """order: base order (candidate ids). cand_view(c) -> {"path", "text"}. -> (new order, info).
    caps: R (v1, default) or R2 (v2: 200-token completion cap); must match the client's caps."""
    caps = caps or R
    t0 = time.perf_counter()
    shown = list(order[:R["n"]])
    cands = [cand_view(c) for c in shown]
    p, n, t, cap = build_prompt(question, cands)
    info = {"version": ORDERFIX_V2 if caps is R2 else ORDERFIX_VERSION, "use": False, "status": None, "reason": None, "prompt_tokens_est": t,
            "n_shown": n, "snip_tokens": cap}
    if p is None:
        info.update(status="skipped", reason="no_fit")
        return list(order), info
    shown = shown[:n]
    info["prompt"] = p
    from ...llm import BudgetExceeded
    text, finish = None, None
    t1 = time.perf_counter()
    try:
        res = chat_client.chat([{"role": "user", "content": p}], purpose="lex7_orderfix",
                               num_predict=caps["max_tokens_total"], **(ledger_ctx or {}))
        text, finish = res.text, res.done_reason
        info.update(prompt_tokens=res.prompt_tokens, completion_tokens=res.completion_tokens,
                    reasoning_tokens=getattr(res, "reasoning_tokens", None), cost_usd=res.cost_usd,
                    provider=getattr(res, "provider", None), finish=finish, raw=(text or "")[:300])
    except BudgetExceeded:
        raise
    except Exception as e:  # noqa: BLE001  deadline / HTTP -> base order
        info.update(status="fail", reason="deadline" if "deadline" in str(e).lower() else "call_error",
                    error=str(e)[:200])
    info["t_call_ms"] = round((time.perf_counter() - t1) * 1000.0, 1)
    new = list(order)
    if text is not None:
        pr = parse(text, n, finish_reason=finish)
        info.update(status=pr["status"], reason=pr["reason"], ids=pr["ids"])
        if pr["ok"] and pr["status"] == "ok":
            new = reorder(order, shown, pr["ids"])
            info["use"] = True
            info["first"] = [shown[i - 1] for i in pr["ids"]]
    info["t_total_ms"] = round((time.perf_counter() - t0) * 1000.0, 1)
    return new, info
