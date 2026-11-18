"""The only way any approach talks to a model.

* OllamaClient.chat / .embed  -> local Ollama HTTP API, deterministic options from
  config, one global lock (threads AND processes) so the GPU sees one call at a time,
  timeouts and retries.
* OpenRouterClient.chat       -> the LLM judge, behind a persisted call/cost budget.
* Ledger                      -> one JSONL line per call:
  {phase: index|query|eval|smoke, approach, purpose, prompt_tokens, completion_tokens,
   wall_ms, ...}. Scopes collect the calls made inside a block, so run records and
  index build stats are filled from the ledger, never from approach self-reports.

    client = OllamaClient(cfg)                       # ledger defaults to work/ledger/<date>_<pid>.jsonl
    with client.ledger.scope(phase="query", approach="B3", question_id="q1") as sc:
        r = client.chat(messages, purpose="answer")
    sc.calls, sc.prompt_tokens, sc.completion_tokens, sc.wall_ms
"""
import contextvars
import json
import os
import re
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

import requests

from .config import load_config
from .util import FileLock, append_jsonl, now_iso, read_json, atomic_write_json

PHASES = ("index", "query", "eval", "smoke")

# --------------------------------------------------------------------------
# ledger
# --------------------------------------------------------------------------

_SCOPES = contextvars.ContextVar("ragbench_ledger_scopes", default=())


class LedgerScope:
    """Totals for the calls recorded while the scope is active (nested scopes all receive them)."""

    def __init__(self, ledger, ctx):
        self.ledger = ledger
        self.ctx = dict(ctx)
        self.records = []

    def add(self, rec):
        self.records.append(rec)

    def _sum(self, key, kinds):
        return sum((r.get(key) or 0) for r in self.records if r.get("kind") in kinds)

    @property
    def calls(self):            # chat calls to the local model (the "LLM calls" metric)
        return sum(1 for r in self.records if r.get("kind") == "chat")

    @property
    def embed_calls(self):
        return sum(1 for r in self.records if r.get("kind") == "embed")

    @property
    def judge_calls(self):
        return sum(1 for r in self.records if r.get("kind") == "judge")

    @property
    def prompt_tokens(self):
        return self._sum("prompt_tokens", ("chat",))

    @property
    def completion_tokens(self):
        return self._sum("completion_tokens", ("chat",))

    @property
    def embed_tokens(self):
        return self._sum("prompt_tokens", ("embed",))

    @property
    def wall_ms(self):
        return self._sum("wall_ms", ("chat", "embed"))

    @property
    def load_ms(self):
        return self._sum("load_ms", ("chat", "embed"))

    @property
    def failed_calls(self):
        return sum(1 for r in self.records if not r.get("ok"))

    @property
    def wait_ms(self):          # time spent waiting for the GPU lock (another thread/process was calling)
        return self._sum("wait_ms", ("chat", "embed"))

    def summary(self):
        return {"llm_calls": self.calls, "embed_calls": self.embed_calls, "judge_calls": self.judge_calls,
                "prompt_tokens": self.prompt_tokens, "completion_tokens": self.completion_tokens,
                "embed_tokens": self.embed_tokens, "llm_wall_ms": round(self.wall_ms, 1),
                "load_ms": round(self.load_ms, 1), "lock_wait_ms": round(self.wait_ms, 1),
                "failed_calls": self.failed_calls}


class Ledger:
    def __init__(self, path=None, run_id=None, keep_in_memory=True):
        self.path = path
        self.run_id = run_id or time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
        self.keep = keep_in_memory
        self.records = []
        self._lock = threading.Lock()

    @contextmanager
    def scope(self, **ctx):
        """Context for every call inside the block (phase, approach, question_id, setting,
        corpus_set, ...). Inner scopes inherit and override outer keys."""
        merged = dict(self.context())
        merged.update({k: v for k, v in ctx.items() if v is not None})
        sc = LedgerScope(self, merged)
        token = _SCOPES.set(_SCOPES.get() + (sc,))
        try:
            yield sc
        finally:
            _SCOPES.reset(token)

    def context(self):
        out = {}
        for sc in _SCOPES.get():
            if sc.ledger is self:
                out.update(sc.ctx)
        return out

    def record(self, rec):
        rec = dict(rec)
        rec.setdefault("run_id", self.run_id)
        with self._lock:
            if self.keep:
                self.records.append(rec)
            if self.path:
                append_jsonl(self.path, rec)
        for sc in _SCOPES.get():
            if sc.ledger is self:
                sc.add(rec)
        return rec

    def totals(self, **filters):
        sc = LedgerScope(self, filters)
        for r in self.records:
            if all(r.get(k) == v for k, v in filters.items()):
                sc.add(r)
        return sc.summary()


def default_ledger(cfg=None):
    cfg = cfg or load_config()
    d = cfg.ensure_dir("ledger_root")
    return Ledger(path=str(d / ("%s_%d.jsonl" % (time.strftime("%Y%m%d"), os.getpid()))))


# --------------------------------------------------------------------------
# results
# --------------------------------------------------------------------------

@dataclass
class ChatResult:
    text: str
    prompt_tokens: int
    completion_tokens: int
    wall_ms: float
    load_ms: float = 0.0
    done_reason: str = ""
    thinking: str = ""
    tool_calls: list = field(default_factory=list)
    model: str = ""
    call_id: str = ""
    cost_usd: float = None

    def json(self, default=None):
        """First JSON object/array found in the text (tolerates ``` fences and chatter)."""
        return extract_json(self.text, default)


@dataclass
class EmbedResult:
    vectors: list          # list[list[float]], one per input, input order
    prompt_tokens: int
    wall_ms: float
    calls: int
    model: str = ""


class ModelError(RuntimeError):
    pass


class BudgetExceeded(RuntimeError):
    pass


_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.S)


def extract_json(text, default=None):
    if not text:
        return default
    cands = [m.group(1) for m in _FENCE.finditer(text)] + [text]
    dec = json.JSONDecoder()
    for c in cands:
        c = c.strip()
        for i, ch in enumerate(c):
            if ch in "{[":
                try:
                    obj, _end = dec.raw_decode(c[i:])
                    return obj
                except json.JSONDecodeError:
                    continue
    return default


# --------------------------------------------------------------------------
# Ollama
# --------------------------------------------------------------------------

_THREAD_LOCK = threading.RLock()
_FILE_LOCKS = {}


class OllamaClient:
    def __init__(self, cfg=None, ledger=None):
        self.cfg = cfg or load_config()
        self.ledger = ledger or default_ledger(self.cfg)
        o = self.cfg.ollama
        self.host = o["host"].rstrip("/")
        self.timeout = (o["connect_timeout_s"], o["read_timeout_s"])
        self.retries = int(o["retries"])
        self.backoff = float(o["retry_backoff_s"])
        self.session = requests.Session()
        self._flock = None
        if o.get("cross_process_lock", True):
            lp = str(self.cfg.ensure_dir("locks_root") / "ollama.lock")
            self._flock = _FILE_LOCKS.setdefault(lp, FileLock(lp))

    # -- ledger plumbing ----------------------------------------------------
    @contextmanager
    def use_ledger(self, ledger):
        old = self.ledger
        self.ledger = ledger
        try:
            yield ledger
        finally:
            self.ledger = old

    def _ctx(self, phase, approach, purpose, extra):
        ctx = self.ledger.context()
        ctx.update({k: v for k, v in extra.items() if v is not None})
        if phase is not None:
            ctx["phase"] = phase
        if approach is not None:
            ctx["approach"] = approach
        if not purpose:
            raise ValueError("every model call needs a purpose")
        ctx["purpose"] = purpose
        if ctx.get("phase") not in PHASES:
            raise ValueError("every model call needs phase in %s (got %r); open a ledger scope or pass phase=" % (PHASES, ctx.get("phase")))
        ctx.setdefault("approach", "-")
        return ctx

    @contextmanager
    def _gpu_lock(self):
        t0 = time.perf_counter()
        with _THREAD_LOCK:
            if self._flock is not None:
                self._flock.acquire()
            try:
                yield (time.perf_counter() - t0) * 1000.0
            finally:
                if self._flock is not None:
                    self._flock.release()

    def options(self, num_predict=None, **overrides):
        g = self.cfg.generation
        opts = {"temperature": g["temperature"], "seed": g["seed"], "top_p": g["top_p"], "top_k": g["top_k"],
                "num_ctx": g["num_ctx"], "num_predict": num_predict or g["num_predict_default"]}
        opts.update(overrides)
        return opts

    def _post(self, endpoint, payload):
        """POST with the GPU lock held per attempt; retries on connection errors, timeouts, 5xx."""
        last = None
        wait_total = 0.0
        for attempt in range(1, self.retries + 2):
            with self._gpu_lock() as wait_ms:
                wait_total += wait_ms
                t0 = time.perf_counter()
                try:
                    resp = self.session.post(self.host + endpoint, json=payload, timeout=self.timeout)
                    wall = (time.perf_counter() - t0) * 1000.0
                    if resp.status_code >= 500:
                        last = ModelError("HTTP %d: %s" % (resp.status_code, resp.text[:300]))
                    elif resp.status_code >= 400:
                        raise ModelError("HTTP %d: %s" % (resp.status_code, resp.text[:300]))
                    else:
                        return resp.json(), wall, wait_total, attempt
                except (requests.ConnectionError, requests.Timeout) as e:
                    last = ModelError("%s: %s" % (type(e).__name__, str(e)[:300]))
            if attempt <= self.retries:
                time.sleep(self.backoff * attempt)
        raise last

    def _record(self, kind, model, ctx, **fields):
        rec = {"ts": now_iso(), "call_id": uuid.uuid4().hex[:12], "kind": kind, "model": model}
        rec.update(ctx)
        rec.update(fields)
        return self.ledger.record(rec)

    # -- chat ---------------------------------------------------------------
    def chat(self, messages, *, purpose, phase=None, approach=None, num_predict=None, format=None,
             tools=None, options=None, think=None, model=None, **ctx):
        """messages: [{"role": "system"|"user"|"assistant"|"tool", "content": str}, ...].
        format: None, "json" or a JSON schema dict (Ollama structured outputs).
        Returns ChatResult; raises ModelError after retries (the failed call is still in the ledger)."""
        ctx = self._ctx(phase, approach, purpose, ctx)
        model = model or self.cfg.models["chat"]
        payload = {"model": model, "messages": messages, "stream": False,
                   "options": self.options(num_predict=num_predict, **(options or {})),
                   "keep_alive": self.cfg.ollama["keep_alive"],
                   "think": self.cfg.ollama["think"] if think is None else think}
        if format is not None:
            payload["format"] = format
        if tools:
            payload["tools"] = tools
        prompt_chars = sum(len(m.get("content") or "") for m in messages)
        try:
            data, wall, wait, attempts = self._post("/api/chat", payload)
        except Exception as e:  # noqa: BLE001
            self._record("chat", model, ctx, ok=False, error=str(e)[:500], prompt_tokens=0, completion_tokens=0,
                         wall_ms=0.0, prompt_chars=prompt_chars)
            raise
        msg = data.get("message") or {}
        res = ChatResult(text=msg.get("content") or "", prompt_tokens=int(data.get("prompt_eval_count") or 0),
                         completion_tokens=int(data.get("eval_count") or 0), wall_ms=wall,
                         load_ms=(data.get("load_duration") or 0) / 1e6, done_reason=data.get("done_reason") or "",
                         thinking=msg.get("thinking") or "", tool_calls=msg.get("tool_calls") or [], model=model)
        rec = self._record("chat", model, ctx, ok=True, prompt_tokens=res.prompt_tokens,
                           completion_tokens=res.completion_tokens, wall_ms=round(wall, 1), wait_ms=round(wait, 1),
                           load_ms=round(res.load_ms, 1),
                           prompt_eval_ms=round((data.get("prompt_eval_duration") or 0) / 1e6, 1),
                           eval_ms=round((data.get("eval_duration") or 0) / 1e6, 1),
                           attempts=attempts, done_reason=res.done_reason, prompt_chars=prompt_chars,
                           completion_chars=len(res.text), thinking_chars=len(res.thinking))
        res.call_id = rec["call_id"]
        return res

    # -- embeddings ---------------------------------------------------------
    def embed(self, texts, *, purpose, phase=None, approach=None, model=None, batch_size=None, **ctx):
        """Embeds texts in batches (one ledger record per HTTP call). Returns EmbedResult."""
        if isinstance(texts, str):
            texts = [texts]
        ctx = self._ctx(phase, approach, purpose, ctx)
        model = model or self.cfg.models["embed"]
        e = self.cfg.embedding
        bs = int(batch_size or e["batch_size"])
        vectors, tokens, wall_total, calls = [], 0, 0.0, 0
        for i in range(0, len(texts), bs):
            batch = texts[i:i + bs]
            payload = {"model": model, "input": batch, "truncate": bool(e["truncate"]),
                       "options": {"num_ctx": e["num_ctx"]}, "keep_alive": self.cfg.ollama["keep_alive"]}
            try:
                data, wall, wait, attempts = self._post("/api/embed", payload)
            except Exception as ex:  # noqa: BLE001
                self._record("embed", model, ctx, ok=False, error=str(ex)[:500], prompt_tokens=0,
                             completion_tokens=0, wall_ms=0.0, n_inputs=len(batch))
                raise
            embs = data.get("embeddings") or []
            if len(embs) != len(batch):
                raise ModelError("embed returned %d vectors for %d inputs" % (len(embs), len(batch)))
            ptok = int(data.get("prompt_eval_count") or 0)
            self._record("embed", model, ctx, ok=True, prompt_tokens=ptok, completion_tokens=0,
                         wall_ms=round(wall, 1), wait_ms=round(wait, 1),
                         load_ms=round((data.get("load_duration") or 0) / 1e6, 1), attempts=attempts,
                         n_inputs=len(batch), prompt_chars=sum(len(t) for t in batch))
            vectors.extend(embs)
            tokens += ptok
            wall_total += wall
            calls += 1
        return EmbedResult(vectors=vectors, prompt_tokens=tokens, wall_ms=wall_total, calls=calls, model=model)

    # -- health -------------------------------------------------------------
    def health(self):
        out = {"host": self.host, "reachable": False}
        try:
            v = self.session.get(self.host + "/api/version", timeout=self.timeout).json()
            tags = self.session.get(self.host + "/api/tags", timeout=self.timeout).json()
        except Exception as e:  # noqa: BLE001
            out["error"] = str(e)[:300]
            return out
        models = {m["name"]: m.get("digest") for m in tags.get("models", [])}
        m = self.cfg.models
        out.update({"reachable": True, "version": v.get("version"), "models": models,
                    "chat_present": m["chat"] in models, "embed_present": m["embed"] in models,
                    "chat_digest_match": models.get(m["chat"]) == m.get("chat_digest"),
                    "embed_digest_match": models.get(m["embed"]) == m.get("embed_digest")})
        return out


# --------------------------------------------------------------------------
# OpenRouter judge
# --------------------------------------------------------------------------

class OpenRouterClient:
    """Judge client. Every call first reserves one unit of the persisted budget
    (judge.budget_file, shared by all processes); it refuses once max_calls_total or
    max_cost_usd is reached. The API key is read on demand and never logged."""

    def __init__(self, cfg=None, ledger=None):
        self.cfg = cfg or load_config()
        self.ledger = ledger or default_ledger(self.cfg)
        j = self.cfg.judge
        self.base = j["base_url"].rstrip("/")
        p = Path(j["budget_file"])
        self.budget_path = p if p.is_absolute() else self.cfg.project_root / p
        self._lock = FileLock(str(self.budget_path) + ".lock")

    def _headers(self):
        key = self.cfg.openrouter_key()
        if not key:
            raise ModelError("OPENROUTER_API_KEY not found in environment or .env")
        return {"Authorization": "Bearer " + key, "Content-Type": "application/json",
                "X-Title": "ragbench"}

    def budget_state(self):
        j = self.cfg.judge
        st = read_json(self.budget_path, {"calls": 0, "cost_usd": 0.0, "history": []})
        return {"calls": st["calls"], "cost_usd": round(st["cost_usd"], 6),
                "max_calls_total": j["max_calls_total"], "max_cost_usd": j["max_cost_usd"]}

    def _reserve(self, purpose):
        j = self.cfg.judge
        with self._lock:
            st = read_json(self.budget_path, {"calls": 0, "cost_usd": 0.0, "history": []})
            if st["calls"] >= j["max_calls_total"] or st["cost_usd"] >= j["max_cost_usd"]:
                raise BudgetExceeded("judge budget exhausted: %d calls, $%.4f" % (st["calls"], st["cost_usd"]))
            st["calls"] += 1
            st["history"].append({"ts": now_iso(), "purpose": purpose, "cost_usd": None})
            atomic_write_json(self.budget_path, st, indent=1)
            return len(st["history"]) - 1

    def _settle(self, idx, cost):
        with self._lock:
            st = read_json(self.budget_path)
            st["cost_usd"] += float(cost or 0.0)
            if 0 <= idx < len(st["history"]):
                st["history"][idx]["cost_usd"] = cost
            atomic_write_json(self.budget_path, st, indent=1)

    def key_info(self):
        """Remaining credit on the key (GET /key; costs nothing). Returns only numbers."""
        r = requests.get(self.base + "/key", headers=self._headers(), timeout=self.cfg.judge["timeout_s"])
        r.raise_for_status()
        d = r.json().get("data", {})
        return {k: d.get(k) for k in ("limit", "usage", "limit_remaining", "is_free_tier")}

    def chat(self, messages, *, purpose, model=None, max_tokens=None, phase="eval", approach="judge", **ctx):
        j = self.cfg.judge
        model = model or j["model"]
        idx = self._reserve(purpose)
        payload = {"model": model, "messages": messages, "temperature": j["temperature"],
                   "max_tokens": int(max_tokens or j["max_tokens"]), "usage": {"include": True}}
        rec_ctx = self.ledger.context()
        rec_ctx.update({k: v for k, v in ctx.items() if v is not None})
        rec_ctx.update({"phase": phase, "approach": approach, "purpose": purpose})
        t0 = time.perf_counter()
        try:
            r = requests.post(self.base + "/chat/completions", headers=self._headers(), json=payload,
                              timeout=j["timeout_s"])
            wall = (time.perf_counter() - t0) * 1000.0
            if r.status_code >= 400:
                raise ModelError("OpenRouter HTTP %d: %s" % (r.status_code, r.text[:300]))
            data = r.json()
        except Exception as e:  # noqa: BLE001
            self._settle(idx, 0.0)
            self.ledger.record(dict(rec_ctx, ts=now_iso(), call_id=uuid.uuid4().hex[:12], kind="judge", model=model,
                                    ok=False, error=str(e)[:300], prompt_tokens=0, completion_tokens=0, wall_ms=0.0))
            raise
        usage = data.get("usage") or {}
        cost = usage.get("cost")
        self._settle(idx, cost)
        text = ((data.get("choices") or [{}])[0].get("message") or {}).get("content") or ""
        rec = self.ledger.record(dict(rec_ctx, ts=now_iso(), call_id=uuid.uuid4().hex[:12], kind="judge", model=model,
                                      ok=True, prompt_tokens=int(usage.get("prompt_tokens") or 0),
                                      completion_tokens=int(usage.get("completion_tokens") or 0),
                                      wall_ms=round(wall, 1), cost_usd=cost))
        return ChatResult(text=text, prompt_tokens=rec["prompt_tokens"], completion_tokens=rec["completion_tokens"],
                          wall_ms=wall, model=model, call_id=rec["call_id"], cost_usd=cost)
