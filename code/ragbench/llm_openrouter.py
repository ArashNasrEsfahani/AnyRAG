"""A READER swap: every approach's chat calls go to an OpenRouter model instead of the local gemma, with nothing else
changed - same prompts, same 6000-char evidence, same call caps, same temperature and seed. Embeddings (B3, the
agentic vector layers) stay on the local bge-m3, so retrieval is identical across readers and only the model that
reads and plans differs.

    python -m ragbench.run ... --reader openrouter:deepseek/deepseek-v4-flash

MONEY AND QUOTA ARE ENFORCED HERE, before every call, across processes (work/openrouter/reader_budget.json):
  - paid models: refuse once the spend reaches READER_MAX_USD (default 5.0) - and the key's own hard limit refuses
    first if it is lower (HTTP 402 -> BudgetExceeded, never retried);
  - free models (":free"): refuse once today's count reaches READER_FREE_PER_DAY (default 950 of OpenRouter's 1000
    per account per day, headroom for probes).
BudgetExceeded stops the run (the runner's consecutive-error guard); a later run resumes where it stopped.
The key is read on demand from .env and never logged.
"""
import datetime as dt
import os
import re
import time
import uuid
from pathlib import Path

import requests

from .llm import BudgetExceeded, ChatResult, ModelError, OllamaClient
from .util import FileLock, atomic_write_json, now_iso, read_json

BASE = "https://openrouter.ai/api/v1"


def _today():
    return dt.datetime.utcnow().strftime("%Y-%m-%d")         # OpenRouter's free quota resets at UTC midnight


def _retry(fn, tries=40):
    """Windows refuses to open a file while another process is replacing it (PermissionError): wait and retry."""
    for i in range(tries):
        try:
            return fn()
        except PermissionError:
            if i == tries - 1:
                raise
            time.sleep(0.05 * (i + 1))


def cap_usd(root):
    """The spending cap: work/openrouter/budget_cap.json {"max_usd": ...} when present (raised by the user without a
    restart: every call reads it), else READER_MAX_USD, else 5.0."""
    f = Path(root) / "work" / "openrouter" / "budget_cap.json"
    try:
        return float(_retry(lambda: read_json(f, {})).get("max_usd"))
    except (TypeError, ValueError, OSError):
        return float(os.environ.get("READER_MAX_USD", "5.0"))


class ReaderBudget:
    def __init__(self, cfg):
        self.root = cfg.project_root
        self.path = cfg.project_root / "work" / "openrouter" / "reader_budget.json"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = FileLock(str(self.path) + ".lock")
        self.free_per_day = int(os.environ.get("READER_FREE_PER_DAY", "950"))

    @property
    def max_usd(self):
        return cap_usd(self.root)

    def state(self):
        return _retry(lambda: read_json(self.path, {"cost_usd": 0.0, "paid_calls": 0, "free_calls": {}, "by_model": {}}))

    def _write(self, st):
        _retry(lambda: atomic_write_json(self.path, st, indent=1))

    def reserve(self, model):
        free = model.endswith(":free")
        with self.lock:
            st = self.state()
            if free:
                n = st["free_calls"].get(_today(), 0)
                if n >= self.free_per_day:
                    raise BudgetExceeded("free-model quota for %s reached (%d of %d)" % (_today(), n, self.free_per_day))
                st["free_calls"][_today()] = n + 1
            else:
                if st["cost_usd"] >= self.max_usd:
                    raise BudgetExceeded("reader budget reached: $%.4f of $%.2f" % (st["cost_usd"], self.max_usd))
                st["paid_calls"] += 1
            self._write(st)

    def settle(self, model, cost, ptok, ctok):
        with self.lock:
            st = self.state()
            st["cost_usd"] = round(st["cost_usd"] + float(cost or 0.0), 8)
            m = st["by_model"].setdefault(model, {"calls": 0, "cost_usd": 0.0, "prompt_tokens": 0, "completion_tokens": 0})
            m["calls"] += 1
            m["cost_usd"] = round(m["cost_usd"] + float(cost or 0.0), 8)
            m["prompt_tokens"] += int(ptok or 0)
            m["completion_tokens"] += int(ctok or 0)
            self._write(st)


class OpenRouterReader(OllamaClient):
    """OllamaClient with chat() sent to OpenRouter. `model_id` is the OpenRouter id ("qwen/qwen3.8-27b:free")."""

    def __init__(self, cfg, ledger, model_id, supported=None, retries=6, provider=None, attempts=None,
                 deadline_s=None, max_tokens_total=None, reasoning_effort="low"):
        """Call options (2026-09-26, LEX5 spec 4; all default to the old behaviour):
        provider          a provider tag or list (OpenRouter provider routing): {"order": [...], "allow_fallbacks":
                          false, "require_parameters": true} - a pinned endpoint, no silent fallback. When None, the
                          env OPENROUTER_PROVIDER_PINS (JSON {model id: tag}) is consulted.
        attempts          total HTTP attempts (1 = no retry); default retries+1. Every ledger record carries
                          attempts (used) and max_attempts (the policy in force). Retried: HTTP 429/5xx, network
                          errors, an error body or no choices; never retried: other 4xx, a deadline, a spent key
                          (one switch to the fallback key)
        deadline_s        a TOTAL wall deadline for the whole call, enforced by cancelling the in-flight request
                          (asyncio task over httpx on a background loop: the caller is released at the deadline and
                          the connection is closed) -> DeadlineExceeded; usage of a cancelled call is unknown and
                          is recorded as such (usage_known false), never as zero
        max_tokens_total  the completion cap INCLUDING reasoning (no separate reasoning allowance)
        reasoning_effort  'low' (default) | 'medium' | 'high' | None (no reasoning field)"""
        super().__init__(cfg, ledger)
        self.model_id = model_id
        self.supported = set(supported or [])
        self.budget = ReaderBudget(cfg)
        self.or_retries = retries if attempts is None else max(0, int(attempts) - 1)
        if provider is None:
            # env pins: {model id: tag} or {model id: {"provider": tag, "params": [the pinned endpoint's supported
            # parameters]}} - with params, only those sampling parameters are sent (require_parameters is on)
            try:
                import json as _json
                provider = (_json.loads(os.environ.get("OPENROUTER_PROVIDER_PINS") or "{}") or {}).get(model_id)
            except ValueError:
                provider = None
        if isinstance(provider, dict):
            if provider.get("params") is not None:
                ep = set(provider["params"])
                self.supported = (self.supported & ep) if self.supported else ep
            provider = provider.get("provider")
        self.provider = [provider] if isinstance(provider, str) else (list(provider) if provider else None)
        self.deadline_s = float(deadline_s) if deadline_s else None
        self.max_tokens_total = int(max_tokens_total) if max_tokens_total else None
        self.reasoning_effort = reasoning_effort
        self.http = requests.Session()
        self._aclient = None

    def _key(self):
        """This project's key first; after it answers 402 (its own hard limit), the fallback key named by
        READER_FALLBACK_ENV (a .env path holding OPENROUTER_API_KEY). The dollar cap above spans both keys."""
        if not getattr(self, "_use_fallback", False) and os.environ.get("READER_FALLBACK_ENV") \
                and (self.cfg.project_root / "work" / "openrouter" / "project_key_spent.json").exists():
            self._use_fallback = True                   # an earlier process found the project key spent
        if getattr(self, "_use_fallback", False):
            return _read_env_key(os.environ["READER_FALLBACK_ENV"])
        key = self.cfg.openrouter_key()
        if not key:
            raise ModelError("OPENROUTER_API_KEY not found in environment or .env")
        return key

    def _payload(self, messages, num_predict, format):
        """Same sampling as the local reader, but only the parameters the model accepts (a provider answers 400 to
        an unsupported one - lfm rejects top_p). No response_format: forced JSON mode behaves differently from one
        provider to the next (gpt-oss returned "{ }"), while every JSON prompt already asks for JSON and every parser
        tolerates chatter - so all remote readers get the prompt alone."""
        g = self.cfg.generation
        sup = self.supported
        want = int(num_predict or g["num_predict_default"])
        reasoning = bool(sup & {"reasoning", "include_reasoning"})
        p = {"model": self.model_id, "messages": messages, "usage": {"include": True},
             # a reasoning model spends tokens thinking before it answers; the ANSWER keeps the shared length
             # budget, the thinking gets its own allowance (kept low where the model allows it)
             "max_tokens": want + (3000 if reasoning else 64)}
        if self.max_tokens_total:                   # one cap for reasoning + answer (the LEX5 selector)
            p["max_tokens"] = self.max_tokens_total
        for k in ("temperature", "top_p", "seed"):
            if not sup or k in sup:
                p[k] = g[k]
        if reasoning and self.reasoning_effort:
            p["reasoning"] = {"effort": self.reasoning_effort, "exclude": True}
        if self.provider:                           # a pinned endpoint: no silent provider fallback
            p["provider"] = {"order": list(self.provider), "allow_fallbacks": False, "require_parameters": True}
        # v2: the structured-output contract VALIDATED for this reader (or_contract.py), else prompt only
        self._contract = "prompt"
        if format is not None and not getattr(self, "_force_prompt", False):
            c = self.contract()
            if c == "json_schema" and isinstance(format, dict):
                p["response_format"] = {"type": "json_schema",
                                        "json_schema": {"name": "out", "strict": False, "schema": format}}
                self._contract = "json_schema"
            elif c in ("json_schema", "json_object"):
                p["response_format"] = {"type": "json_object"}
                self._contract = "json_object"
        return p

    def contract(self):
        """The structured-output contract chosen for this reader by ragbench/tools/or_contract.py (a live check with
        the benchmark's own schemas), from work/openrouter/contracts.json; 'prompt' when there is none."""
        if not hasattr(self, "_contracts"):
            self._contracts = read_json(self.cfg.project_root / "work" / "openrouter" / "contracts.json", {}) or {}
        return (self._contracts.get(self.model_id) or {}).get("contract", "prompt")

    def chat(self, messages, *, purpose, phase=None, approach=None, num_predict=None, format=None,
             tools=None, options=None, think=None, model=None, **ctx):
        res = self._chat_once(messages, purpose=purpose, phase=phase, approach=approach, num_predict=num_predict,
                              format=format, **ctx)
        # v2 rule (documented, recorded): a structured-output call that came back EMPTY (gpt-oss did, 2 of 6 calls in
        # the smoke test) is retried once under the prompt-only contract - the prompt itself asks for JSON
        if format is not None and not res.text and getattr(self, "_contract", "prompt") != "prompt":
            self._force_prompt = True
            try:
                res = self._chat_once(messages, purpose=purpose, phase=phase, approach=approach,
                                      num_predict=num_predict, format=format, retry_of_empty=True, **ctx)
            finally:
                self._force_prompt = False
        return res

    def _chat_once(self, messages, *, purpose, phase=None, approach=None, num_predict=None, format=None, **ctx):
        ctx = self._ctx(phase, approach, purpose, ctx)
        label = "openrouter/" + self.model_id
        prompt_chars = sum(len(m.get("content") or "") for m in messages)
        payload = self._payload(messages, num_predict, format)
        self.budget.reserve(self.model_id)
        headers = {"Authorization": "Bearer " + self._key(), "Content-Type": "application/json",
                   "X-Title": "ragbench-reader"}
        last, t_all, attempt = None, time.perf_counter(), 0
        data = None
        for attempt in range(1, self.or_retries + 2):
            t0 = time.perf_counter()
            try:
                if self.deadline_s:
                    remaining = self.deadline_s - (time.perf_counter() - t_all)
                    if remaining <= 0:
                        raise DeadlineExceeded("deadline %.1fs reached before attempt %d" % (self.deadline_s, attempt))
                    r = self._post_deadline(BASE + "/chat/completions", headers, payload, remaining)
                else:
                    r = self.http.post(BASE + "/chat/completions", headers=headers, json=payload, timeout=(15, 180))
                wall = (time.perf_counter() - t0) * 1000.0
                # a spent key: 402 (credits) or 403 "Key limit exceeded" (the key's own cap - what OpenRouter
                # actually answered on 2026-09-23; the switch first waited for 402 only and five lanes failed)
                spent = r.status_code == 402 or (r.status_code == 403 and "limit exceeded" in r.text.lower())
                if spent:
                    if os.environ.get("READER_FALLBACK_ENV") and not getattr(self, "_use_fallback", False):
                        self._use_fallback = True                   # the project key is spent: switch, retry now
                        _mark_spent(self.cfg.project_root)
                        headers["Authorization"] = "Bearer " + self._key()
                        last = ModelError("OpenRouter %d on the project key; switched to the fallback key"
                                          % r.status_code)
                        continue
                    raise BudgetExceeded("OpenRouter %d (key limit or credits): %s" % (r.status_code, r.text[:200]))
                if r.status_code == 429 or r.status_code >= 500:
                    last = ModelError("OpenRouter HTTP %d: %s" % (r.status_code, r.text[:200]))
                elif r.status_code >= 400:
                    last = ModelError("OpenRouter HTTP %d: %s" % (r.status_code, r.text[:300]))
                    bad = re.search(r"`(\w+)` parameter is not supported", r.text.replace('\\"', '"'))
                    if bad and bad.group(1) in payload:          # the catalogue said yes, the provider says no
                        payload.pop(bad.group(1))
                        self.supported.discard(bad.group(1))
                        continue
                    break                                          # a bad request does not get better with retries
                else:
                    data = r.json()
                    if data.get("error"):
                        last = ModelError("OpenRouter error: %s" % str(data["error"])[:300])
                        data = None
                    elif not data.get("choices"):
                        last = ModelError("OpenRouter returned no choices")
                        data = None
                    else:
                        break
            except DeadlineExceeded as e:                              # never retried: the deadline is total
                last = e
                break
            except (requests.RequestException, ValueError) as e:     # dropped streams, timeouts, bad JSON bodies
                last = ModelError("%s: %s" % (type(e).__name__, str(e)[:200]))
            except Exception as e:  # noqa: BLE001  (httpx errors on the deadline path)
                if type(e).__module__.startswith("httpx"):
                    last = ModelError("%s: %s" % (type(e).__name__, str(e)[:200]))
                else:
                    raise
            if attempt <= self.or_retries:
                time.sleep(min(60.0, 5.0 * 2 ** (attempt - 1)))          # 5, 10, 20, 40, 60, 60 s
        if data is None:
            # the provider may have done (and billed) work for a failed or cancelled call: usage is UNKNOWN, not 0
            self._record("chat", label, ctx, ok=False, error=str(last)[:500], prompt_tokens=0, completion_tokens=0,
                         wall_ms=0.0, wall_total_ms=round((time.perf_counter() - t_all) * 1000.0, 1),
                         prompt_chars=prompt_chars, attempts=attempt, max_attempts=self.or_retries + 1,
                         usage_known=False, max_tokens=payload.get("max_tokens"),
                         deadline_exceeded=isinstance(last, DeadlineExceeded), provider_pin=self.provider,
                         call_ts=now_iso())
            self.budget.settle(self.model_id, 0.0, 0, 0)
            raise last or ModelError("OpenRouter call failed")
        ch = data["choices"][0]
        msg = ch.get("message") or {}
        u = data.get("usage") or {}
        cost = u.get("cost") or 0.0
        content = msg.get("content") or ""
        # v2, a documented rule (never silent): a reply that carries its deliberation in the content, closed by
        # </think>, keeps only what follows the last </think>; the call record says it happened. Anything else is
        # kept exactly as returned - an unfinished deliberation stays the answer and is graded as such, and a
        # finish_reason of "length" is recorded (the judge's rule counts a truncated answer as incorrect).
        stripped = False
        if "</think>" in content:
            content = content.rsplit("</think>", 1)[1]
            stripped = True
        res = ChatResult(text=content.strip(), prompt_tokens=int(u.get("prompt_tokens") or 0),
                         completion_tokens=int(u.get("completion_tokens") or 0), wall_ms=wall,
                         done_reason=ch.get("finish_reason") or "", thinking=msg.get("reasoning") or "",
                         model=label, cost_usd=cost)
        self.budget.settle(self.model_id, cost, res.prompt_tokens, res.completion_tokens)
        rdet = u.get("completion_tokens_details") or {}
        res.reasoning_tokens = rdet.get("reasoning_tokens")
        res.provider = data.get("provider")
        rec = self._record("chat", label, ctx, ok=True, prompt_tokens=res.prompt_tokens,
                           reasoning_tokens=rdet.get("reasoning_tokens"), usage_known=bool(u),
                           cached_tokens=(u.get("prompt_tokens_details") or {}).get("cached_tokens"),
                           provider_pin=self.provider, max_tokens=payload.get("max_tokens"),
                           completion_tokens=res.completion_tokens, wall_ms=round(wall, 1),
                           wall_total_ms=round((time.perf_counter() - t_all) * 1000.0, 1), wait_ms=0.0, load_ms=0.0,
                           attempts=attempt, max_attempts=self.or_retries + 1, done_reason=res.done_reason,
                           prompt_chars=prompt_chars,
                           completion_chars=len(res.text), thinking_chars=len(res.thinking), cost_usd=cost,
                           provider=data.get("provider"), call_ts=now_iso(), request_id=data.get("id"),
                           contract=getattr(self, "_contract", "prompt"), thinking_stripped=stripped,
                           content_empty=not res.text)
        res.call_id = rec["call_id"] or uuid.uuid4().hex[:12]
        return res


class DeadlineExceeded(ModelError):
    """The total call deadline passed; the in-flight request was cancelled."""


class _Resp:
    def __init__(self, status_code, text):
        self.status_code, self.text = status_code, text

    def json(self):
        import json as _json
        return _json.loads(self.text)


_LOOP = {}


def _bg_loop():
    """One asyncio loop per process on a daemon thread: deadline calls run there as tasks, so cancelling the task
    really aborts the HTTP request (and frees the caller at once) while keep-alive connections are reused."""
    import asyncio
    import threading
    if "loop" not in _LOOP:
        loop = asyncio.new_event_loop()
        threading.Thread(target=loop.run_forever, daemon=True, name="openrouter-deadline").start()
        _LOOP["loop"] = loop
    return _LOOP["loop"]


async def _apost(reader, url, headers, payload, timeout_s):
    import httpx
    if reader._aclient is None:
        reader._aclient = httpx.AsyncClient(timeout=httpx.Timeout(60.0, connect=15.0))
    r = await reader._aclient.post(url, headers=headers, json=payload, timeout=timeout_s + 1.0)
    return _Resp(r.status_code, r.text)


async def _aget(reader, url, timeout_s):
    import httpx
    if reader._aclient is None:
        reader._aclient = httpx.AsyncClient(timeout=httpx.Timeout(60.0, connect=15.0))
    r = await reader._aclient.get(url, timeout=timeout_s)
    return r.status_code


def _post_deadline(self, url, headers, payload, remaining_s):
    import asyncio
    import concurrent.futures
    fut = asyncio.run_coroutine_threadsafe(_apost(self, url, headers, payload, remaining_s), _bg_loop())
    try:
        return fut.result(timeout=remaining_s)
    except concurrent.futures.TimeoutError:
        fut.cancel()                               # cancels the task in the loop: httpx aborts the request
        raise DeadlineExceeded("deadline %.1fs exceeded; request cancelled" % self.deadline_s)


def _warm_deadline(self):
    """Untimed: open the keep-alive connection of the deadline path (a free GET, no model call)."""
    import asyncio
    fut = asyncio.run_coroutine_threadsafe(_aget(self, BASE + "/models/%s/endpoints" % self.model_id, 20.0),
                                           _bg_loop())
    return fut.result(timeout=25.0)


OpenRouterReader._post_deadline = _post_deadline
OpenRouterReader.warm_deadline = _warm_deadline


def _mark_spent(root):
    """Remember across processes that the project key is spent, so every later step starts on the fallback key."""
    try:
        atomic_write_json(Path(root) / "work" / "openrouter" / "project_key_spent.json", {"at": now_iso()})
    except OSError:
        pass


def _read_env_key(path):
    for line in open(path, encoding="utf-8", errors="ignore"):
        k, _, v = line.partition("=")
        if k.strip() == "OPENROUTER_API_KEY":
            return v.strip().strip('"').strip("'")
    raise ModelError("no OPENROUTER_API_KEY in the fallback env file")


def model_info(model_id, catalog_path):
    """-> the catalogue row of a model (supported parameters, prices), or {} when unknown."""
    rows = read_json(catalog_path, []) or []
    return next((r for r in rows if r["id"] == model_id), {})
