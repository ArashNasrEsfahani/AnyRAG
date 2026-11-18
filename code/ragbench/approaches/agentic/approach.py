"""AGENTIC: an explicit, budgeted agentic pipeline (plan -> search -> verify -> gap fill -> reconcile + answer).

Unlike B5 (a free ReAct loop in which the model decides every next grep/read and when to stop) the control
flow here is fixed code: the model is called only at named decision points, each with a strict JSON
contract, and every model output is checked by code before it is used. Unlike OURS (one planning call, then
ranked sections go straight to the answer call) the retrieved text is VERIFIED item by item and reconciled
into a table before the answer call.

Calls per question (at most caps.agentic_max_calls, default 8; every call goes through self.client and the
shared ledger; the answer call is always reserved):
    plan          1   (+1 plan_repair only when the reply is not usable JSON and the budget allows)
    verify        ceil(candidates / verify_batch_size), stopping early to keep the answer (+ gap) call
    verify_gap    <= gap_max_batches (1), only for periods/documents with no verified value
    answer        1   shared prompt over pack_evidence (answer_mode "shared", the default) or the synth prompt
                      (flag synth_answer); same model, generation options, num_predict_answer and 6000-char budget

Timing (for the paper): trace["timing"]["stages"] has one record per stage (plan, search, verify, gap_fill,
reconcile, answer) with wall_ms, program_ms, the stage's ledger-scope numbers (llm_calls, prompt/completion
tokens, llm_wall_ms, load_ms, lock_wait_ms), cache hits, ripgrep ms and candidate/guard counts; verify and
gap_fill list their batches with the same numbers per batch. Every model call carries ledger field
stage=<stage>. trace["timing"]["check"] compares the stage sums with the question's whole ledger scope.

Resumability: run.py appends one RunRecord per question and skips finished keys. Inside a question every
model call is cached in a KVCache (one sqlite file next to the run's ledger, committed on every put) keyed by
the full request (model + digest, purpose, prompt version, messages, schema, options). A restarted question
replays the finished calls (ledger records with cache_hit=true and the original token counts and wall time)
and repeats at most the call that was in flight. Model/server errors propagate (the shared client retries
first); the question's record then carries the error and run.py retries it on the next invocation.

Flags (ablations): no_guard, no_gap_fill, line_windows (the hit unit only, no neighbouring units),
no_doc_selection (every document in scope), synth_answer (own timeline answer prompt).
"""
import re
import time
import uuid
from contextlib import contextmanager
from dataclasses import fields as dc_fields
from pathlib import Path

from ...contracts import AnswerResult, Approach, LANG_NAMES, RetrievedItem, answer_num_predict, build_answer_messages
from ...llm import BudgetExceeded, ChatResult
from ...normalize import fold, normalize
from ...store import STORE_FORMAT, build_plain_store, grep
from ...util import KVCache, now_iso, read_json, safe_name, sha1_json
from . import planner as P
from . import reconcile as R
from . import search as S
from . import verify as V
from .defaults import (ANSWER_MODES, DEFAULTS, FLAGS, PLAN_PROMPT_VERSION, SYNTH_PROMPT_VERSION,
                       VERIFY_PROMPT_VERSION)
from .docmeta import doc_meta, select_docs
from .numbers import find_numbers

CACHE_VERSION = 1
SCOPE_KEYS = ("llm_calls", "prompt_tokens", "completion_tokens", "llm_wall_ms", "load_ms", "lock_wait_ms", "failed_calls")


def parse_flags(flags):
    if flags is None:
        return {}
    if isinstance(flags, str):
        flags = {p.strip(): True for p in flags.replace("+", ",").split(",") if p.strip()}
    out = {}
    for k, v in dict(flags).items():
        if k not in FLAGS:
            raise ValueError("unknown AGENTIC flag %r; known: %s" % (k, ", ".join(FLAGS)))
        v = v if isinstance(v, bool) else str(v).lower() in ("1", "true", "yes", "on")
        if v:
            out[k] = True
    return out


class CallBudget:
    """Hard cap on model calls per question. can(n, reserve) asks whether n calls fit while `reserve` calls
    stay available for later stages; spend() raises BudgetExceeded past the cap."""

    def __init__(self, max_calls):
        self.max = int(max_calls)
        self.used = 0
        self.log = []

    def left(self):
        return self.max - self.used

    def can(self, n=1, reserve=0):
        return self.used + n + reserve <= self.max

    def spend(self, purpose):
        if self.used >= self.max:
            raise BudgetExceeded("AGENTIC call cap %d reached before %s" % (self.max, purpose))
        self.used += 1
        self.log.append(purpose)


class CallCache:
    """Durable per-call cache (see module docstring). path None = off."""

    def __init__(self, path):
        self.path = str(path) if path else None
        self.kv = KVCache(path) if path else None

    def key(self, payload):
        return sha1_json(dict(payload, cache_version=CACHE_VERSION))

    def get(self, key):
        return self.kv.get(key) if self.kv is not None else None

    def put(self, key, value):
        if self.kv is not None:
            self.kv.put(key, value)


SYNTH_SYSTEM = (
    "You answer questions about company, market and sector reports.\n"
    "Use only the numbered evidence passages in the user message. Passage [1] is a table of values that a program "
    "checked against the source passages it cites; the notes under the table are part of the evidence.\n"
    "If the answer is not there, reply with exactly {abstain} and nothing else.\n"
    "Answer in {answer_lang}. For a question about change over time give each period's value with its unit and "
    "definition, citing the source passage like [2]. If a note says the definitions differ, say that the values "
    "are not comparable and do not describe a trend. Copy numbers exactly as they are written. Be brief and do not "
    "explain your reasoning."
)


class AgenticPipeline(Approach):
    name = "AGENTIC"
    needs_gold = False

    def __init__(self, cfg=None, client=None, flags=None, options=None):
        super().__init__(cfg=cfg, client=client, flags=parse_flags(flags))
        self.opts = dict(DEFAULTS, **(options or {}))
        f = self.flags
        if f.get("no_guard"):
            self.opts["guard"] = False
        if f.get("no_gap_fill"):
            self.opts["gap_fill"] = False
        if f.get("synth_answer"):
            self.opts["answer_mode"] = "synth"
        if self.opts["answer_mode"] not in ANSWER_MODES:
            raise ValueError("answer_mode must be one of %s" % (ANSWER_MODES,))
        self.line_windows = bool(f.get("line_windows"))
        self.doc_selection = not f.get("no_doc_selection")
        caps = self.cfg.caps or {}
        self.max_calls = int(caps.get("agentic_max_calls") or self.opts["max_calls_default"])
        self._cache = None
        self._cache_for = None

    def config_signature(self):
        return {"store": "plain", "store_format": STORE_FORMAT, "normalized": True}

    def _build_index(self, corpus_set, index_dir):
        t0 = time.perf_counter()
        root = build_plain_store(corpus_set, root=Path(index_dir) / "store", cfg=self.cfg)
        store_ms = round((time.perf_counter() - t0) * 1000.0, 1)
        man = read_json(Path(root) / "manifest.json", {}) or {}
        t1 = time.perf_counter()
        try:     # one no-hit pass so a cold store does not time out the first search of a question
            grep(root, "@@AGENTIC-WARM-UP-NO-SUCH-LINE@@", fixed_strings=True, timeout_s=600, cfg=self.cfg)
        except Exception:  # noqa: BLE001
            pass
        return {"store_root": str(root), "n_files": man.get("n_files", 0),
                "corpus_fingerprint": man.get("corpus_fingerprint"), "store_ms": store_ms,
                "warm_ms": round((time.perf_counter() - t1) * 1000.0, 1)}

    def _load_index(self, corpus_set, index_dir):
        return self._build_index(corpus_set, index_dir)

    def call_cache(self):
        """KVCache next to the run's ledger file (run.py: work/runs/<name>/agentic_calls_<variant>.sqlite);
        an explicit options["call_cache_path"] wins; call_cache "off" or no ledger file and no path = no cache."""
        path = self.opts.get("call_cache_path")
        if not path and self.opts.get("call_cache", "run") == "run" and getattr(self.client.ledger, "path", None):
            path = Path(self.client.ledger.path).parent / ("agentic_calls_%s.sqlite" % safe_name(self.variant))
        if self.opts.get("call_cache") == "off":
            path = None
        if self._cache is None or str(path) != str(self._cache_for):
            self._cache = CallCache(path)
            self._cache_for = path
        return self._cache

    def _answer(self, query, index):
        return _Session(self, query, index).run()


# --------------------------------------------------------------------------
# one question
# --------------------------------------------------------------------------

def _ms(t0):
    return round((time.perf_counter() - t0) * 1000.0, 1)


class _Session:
    def __init__(self, ap, query, index):
        self.a = ap
        self.q = query
        self.index = index
        self.cfg = ap.cfg
        self.o = ap.opts
        self.corpus = index.corpus
        self.root = index.payload["store_root"]
        self.budget = CallBudget(ap.max_calls)
        self.cache = ap.call_cache()
        self.calls = []
        self.timing = []
        self.stats = {"calls": 0, "cache_hits": 0, "plan_repair": 0, "plan_program": 0, "plan_parse_repaired": 0,
                      "docs_scope": 0, "docs_targeted": 0, "hit_lines": 0, "windows": 0, "candidates": 0,
                      "candidates_gap": 0, "verify_calls": 0, "candidates_verified": 0,
                      "candidates_unverified_budget": 0, "accepted": 0, "model_rejected": 0, "model_omitted": 0,
                      "guard_rejected": 0, "reattributed": 0,
                      "unparsed": 0, "gap_fill": 0, "gap_accepted": 0, "rows": 0, "periods_found": 0,
                      "periods_missing": 0, "comparable": -1, "conflicts": 0, "unverified_fallback": 0,
                      "rg_ms": 0.0, "program_ms": 0.0}
        self.guard_reasons = {}

    # -- timing ------------------------------------------------------------------
    @contextmanager
    def stage(self, name):
        """A stage record: wall and program time, the stage's own ledger scope (every call in it carries
        stage=<name>), cache hits; the body adds its counters to the yielded dict."""
        t0 = time.perf_counter()
        n0 = len(self.calls)
        rec = {"stage": name}
        with self.a.client.ledger.scope(stage=name) as sc:
            yield rec
        summ = sc.summary()
        calls = self.calls[n0:]
        rec.update(wall_ms=_ms(t0), **{k: summ.get(k, 0) for k in SCOPE_KEYS})
        rec["cache_hits"] = sum(1 for c in calls if c["cache_hit"])
        rec["call_elapsed_ms"] = round(sum(c["elapsed_ms"] for c in calls), 1)
        rec["program_ms"] = round(rec["wall_ms"] - rec["call_elapsed_ms"], 1)
        self.timing.append(rec)

    # -- model calls ---------------------------------------------------------
    def chat(self, purpose, messages, *, fmt=None, num_predict=None, version=""):
        client = self.a.client
        if not self.budget.can(1):
            raise BudgetExceeded("AGENTIC call cap %d reached before %s" % (self.budget.max, purpose))
        payload = {"model": self.cfg.models["chat"], "digest": self.cfg.models.get("chat_digest"), "purpose": purpose,
                   "version": version, "messages": messages, "format": fmt, "num_predict": num_predict,
                   "options": client.options(num_predict=num_predict), "think": self.cfg.ollama.get("think")}
        # the request the model really gets after variants.py injection (card, list rule, allowance); None = unchanged,
        # so the keys of every run without an injecting option stay what they were
        eff_sig = getattr(client, "effective_request_sig", None)
        eff = eff_sig(messages, purpose, num_predict) if callable(eff_sig) else None
        if eff:
            payload["effective_request"] = eff
        key = self.cache.key(payload)
        hit = self.cache.get(key)
        self.budget.spend(purpose)
        t0 = time.perf_counter()
        if hit is not None:
            ctx = dict(client.ledger.context())
            ctx.setdefault("phase", "query")
            ctx.setdefault("approach", self.a.variant)
            rec = dict(ctx, ts=now_iso(), call_id=uuid.uuid4().hex[:12], kind="chat", model=self.cfg.models["chat"],
                       purpose=purpose, ok=True, prompt_tokens=int(hit.get("prompt_tokens") or 0),
                       completion_tokens=int(hit.get("completion_tokens") or 0), wall_ms=float(hit.get("wall_ms") or 0.0),
                       wait_ms=0.0, load_ms=0.0, attempts=0, done_reason=hit.get("done_reason") or "",
                       prompt_chars=sum(len(m.get("content") or "") for m in messages),
                       completion_chars=len(hit.get("text") or ""), cache_hit=True,
                       cached_call_id=hit.get("call_id"))
            client.ledger.record(rec)
            r = ChatResult(text=hit.get("text") or "", prompt_tokens=rec["prompt_tokens"],
                           completion_tokens=rec["completion_tokens"], wall_ms=rec["wall_ms"],
                           done_reason=rec["done_reason"], model=rec["model"], call_id=rec["call_id"])
            self.stats["cache_hits"] += 1
        else:
            r = client.chat(messages, purpose=purpose, format=fmt, num_predict=num_predict)
            self.cache.put(key, {"text": r.text, "prompt_tokens": r.prompt_tokens, "completion_tokens": r.completion_tokens,
                                 "wall_ms": round(r.wall_ms, 1), "done_reason": r.done_reason, "call_id": r.call_id,
                                 "ts": now_iso(), "purpose": purpose, "version": version})
        self.stats["calls"] = self.budget.used
        self.calls.append({"n": self.budget.used, "purpose": purpose, "stage": client.ledger.context().get("stage"),
                           "call_id": r.call_id, "prompt_tokens": r.prompt_tokens,
                           "completion_tokens": r.completion_tokens, "wall_ms": round(r.wall_ms, 1),
                           "load_ms": round(r.load_ms or 0.0, 1), "done_reason": r.done_reason,
                           "cache_hit": hit is not None, "elapsed_ms": _ms(t0)})
        return r

    # -- stage 1 -----------------------------------------------------------------
    def plan(self, metas):
        choose = len(metas) > 1 and self.a.doc_selection
        company_keys = sorted({m["company"] for m in metas}) if choose else None
        msgs = P.plan_messages(self.q.text, metas, choose, self.o)
        schema = P.plan_schema(company_keys)
        r = self.chat("plan", msgs, fmt=schema, num_predict=int(self.o["plan_num_predict"]), version=PLAN_PROMPT_VERSION)
        obj, how = P.parse_plan_text(r.text)
        problem = P.plan_problem(obj)
        rep = {"parse": how, "problem": problem, "model_output": (r.text or "")[:1500], "done_reason": r.done_reason}
        if how in ("extract", "suffix_repair"):
            self.stats["plan_parse_repaired"] = 1
        # the repair call needs room for it, one verify batch and the answer
        if problem and self.o["plan_repair_call"] and self.budget.can(1, reserve=2):
            msgs2 = msgs + [{"role": "assistant", "content": (r.text or "")[:1500]},
                            {"role": "user", "content": P.REPAIR_USER.format(problem=problem)}]
            r2 = self.chat("plan_repair", msgs2, fmt=schema, num_predict=int(self.o["plan_num_predict"]),
                           version=PLAN_PROMPT_VERSION)
            obj2, how2 = P.parse_plan_text(r2.text)
            problem2 = P.plan_problem(obj2)
            rep["repair"] = {"parse": how2, "problem": problem2, "model_output": (r2.text or "")[:1500]}
            self.stats["plan_repair"] = 1
            if not problem2:
                obj, problem = obj2, None
        if problem:
            plan = P.program_plan(self.q.text, self.q.q_lang, metas, choose, self.o)
            self.stats["plan_program"] = 1
        else:
            plan = P.validate_plan(obj, question=self.q.text, q_lang=self.q.q_lang, metas=metas,
                                   choose_companies=choose, opts=self.o)
        plan["model"] = rep
        return plan

    # -- stage 3 -----------------------------------------------------------------
    def verify_batch(self, batch, plan, purpose, rows, meta_by, batch_no):
        t0 = time.perf_counter()
        msgs = V.verify_messages(self.q.text, plan, batch, self.corpus)
        n = len(batch)
        num_predict = int(self.o["verify_num_predict_base"]) + int(self.o["verify_num_predict_per_item"]) * n
        prompt_ms = _ms(t0)
        r = self.chat(purpose, msgs, fmt=V.verify_schema(n), num_predict=num_predict, version=VERIFY_PROMPT_VERSION)
        call = self.calls[-1]
        self.stats["verify_calls"] += 1
        t1 = time.perf_counter()
        items, how = V.parse_items(r.text, n)
        out = {"batch": batch_no, "purpose": purpose, "candidates": [c.cid for c in batch], "n": n,
               "num_predict": num_predict, "call": call, "parse": how, "items": []}
        counts = {"accepted": 0, "model_rejected": 0, "model_omitted": 0, "guard_rejected": 0, "unparsed": 0,
                  "reattributed": 0}
        batch_values = set()
        accepted_cids = set()
        reasons = {}
        guard_ms = 0.0
        unit_kind = plan.get("unit") or "count"
        term_rx = self.metric_term_rx(plan) if (self.o["quote_term_guard"] and unit_kind != "text") else None
        order = {"accepted": 0, "reattributed": 1, "guard": 1, "model_rejected": 2, "model_omitted": 3, "unparsed": 4}
        for i, c in enumerate(batch, 1):
            its = items.get(i) or []
            self.stats["candidates_verified"] += 1
            if not its:
                # a parsed reply that skips a snippet says nothing relevant about it (the 4B verifier often lists
                # only the snippets it accepts or discusses); an unparseable reply is "unparsed"
                v = "model_omitted" if how else "unparsed"
                c.verdict = v
                counts[v] += 1
                out["items"].append({"cid": c.cid, "doc_id": c.doc_id, "verdict": v})
                continue
            verdicts = []
            for it in its:
                rec = {"cid": c.cid, "doc_id": c.doc_id}
                rel = it.get("relevant")
                rel = rel if isinstance(rel, bool) else str(rel).strip().lower() in ("true", "yes", "1")
                rec.update({k: str(it.get(k) or "")[:300] for k in ("value", "unit", "period", "definition", "quote")})
                rec["relevant"] = rel
                if not rel:
                    rec["verdict"] = "model_rejected"
                    verdicts.append(rec["verdict"])
                    out["items"].append(rec)
                    continue
                tg = time.perf_counter()
                if self.o["guard"]:
                    ok, reason, ginfo = V.guard(it, c.text, unit_kind=unit_kind,
                                                min_quote_letters=int(self.o["min_quote_letters"]), term_rx=term_rx)
                else:
                    toks = [t for t in find_numbers(normalize(str(it.get("value") or ""))) if not t.is_date]
                    ok, reason = True, "guard_off"
                    ginfo = {"value_num": toks[0].main if toks else None, "value_token": str(it.get("value") or ""),
                             "reading": "unchecked"}
                src = c
                if (not ok and reason == "quote_not_in_snippet" and self.o["guard"] and self.o["reattribute"]):
                    # the verifier filed the value under a neighbouring snippet key (seen live: s11 held s12's
                    # values, s13 held s14's); accept it only for the snippet whose text passes the whole guard
                    for other in sorted((b for b in batch if b is not c), key=lambda b: abs(batch.index(b) - (i - 1))):
                        ok2, reason2, ginfo2 = V.guard(it, other.text, unit_kind=unit_kind,
                                                       min_quote_letters=int(self.o["min_quote_letters"]), term_rx=term_rx)
                        if ok2:
                            ok, reason, ginfo, src = True, "ok_reattributed", ginfo2, other
                            rec["reattributed_to"] = other.cid
                            counts["reattributed"] += 1
                            break
                guard_ms += (time.perf_counter() - tg) * 1000.0
                rec["guard"] = reason
                if not ok:
                    rec["verdict"] = "guard:" + reason
                    counts["guard_rejected"] += 1
                    reasons[reason] = reasons.get(reason, 0) + 1
                    self.guard_reasons[reason] = self.guard_reasons.get(reason, 0) + 1
                    verdicts.append(rec["verdict"])
                    out["items"].append(rec)
                    continue
                vkey = (src.cid, ginfo.get("value_num"), V.canon(str(it.get("quote") or "")))
                if vkey in batch_values:          # the same value twice (also under a shifted key)
                    rec["verdict"] = "duplicate"
                    out["items"].append(rec)
                    continue
                batch_values.add(vkey)
                uid = V.value_unit_id(src, self.corpus, ginfo.get("value_token"), str(it.get("quote") or ""))
                row = R.make_row(it, src, ginfo, meta_by.get(src.doc_id, {}), uid, verified=self.o["guard"])
                rows.append(row)
                rec.update(verdict="accepted", value_num=ginfo.get("value_num"), reading=ginfo.get("reading"),
                           row_period=row["period"], def_class=row["def_class"], role=row["role"])
                counts["accepted"] += 1
                accepted_cids.add(src.cid)
                verdicts.append("accepted" if src is c else "reattributed")
                out["items"].append(rec)
            if verdicts and all(v == "model_rejected" for v in verdicts):
                counts["model_rejected"] += 1
            c.verdict = min(verdicts, key=lambda v: order.get(v.split(":")[0], 9)) if verdicts else "duplicate"
        for c in batch:
            if c.cid in accepted_cids:
                c.verdict = "accepted"
        for k, v in counts.items():
            self.stats[k] += v
        out.update(counts, guard_reasons=reasons, prompt_build_ms=prompt_ms,
                   parse_guard_ms=_ms(t1), guard_ms=round(guard_ms, 2), wall_ms=_ms(t0),
                   llm_calls=1, prompt_tokens=call["prompt_tokens"], completion_tokens=call["completion_tokens"],
                   llm_wall_ms=call["wall_ms"], cache_hit=call["cache_hit"], done_reason=call["done_reason"])
        return out

    def metric_term_rx(self, plan):
        """Regexes of the metric's terms in both languages (morphology-aware), for the quote term guard."""
        if getattr(self, "_term_rx", None) is None:
            rx = []
            for lang in ("fa", "en"):
                for t in (plan.get("terms") or {}).get(lang) or []:
                    for p in P.term_patterns(t, lang):
                        rx.append(re.compile(p, re.I))
            self._term_rx = rx
        return self._term_rx

    def known_stems(self, plan):
        """Folded stems that explain a definition word: the metric terms (morphology stems), entity names,
        measure / modifier / generic words, the metric phrase itself."""
        stems = set()
        words = []
        for lang in ("fa", "en"):
            for t in (plan.get("terms") or {}).get(lang) or []:
                words += [(lang, w) for w in re.split(r"[\s‌]+", normalize(t)) if w]
        words += [("fa" if P._lang_of(w) == "fa" else "en", w) for e in (plan.get("entity_terms") or []) + [plan.get("metric") or ""]
                  for w in re.split(r"[\s‌]+", normalize(e)) if w]
        for lang, w in words:
            forms = P.fa_word_variants(w) if lang == "fa" else [w.lower().rstrip("s")]
            stems.update(fold(x) for x in forms + [w] if len(x) >= 3)
        stems.update(fold(w) for w in P.WEAK_WORDS | P.MODIFIER_WORDS | P.GENERIC_TERM_WORDS)
        return stems

    def verify_all(self, cands, plan, purpose, rows, meta_by, reserve, max_batches=None):
        bs = max(1, int(self.o["verify_batch_size"]))
        batches = [cands[i:i + bs] for i in range(0, len(cands), bs)]
        logs, skipped = [], []
        for k, b in enumerate(batches):
            if (max_batches is not None and k >= max_batches) or not self.budget.can(1, reserve=reserve):
                skipped += b
                continue
            logs.append(self.verify_batch(b, plan, purpose, rows, meta_by, k + 1))
        for c in skipped:
            c.verdict = "not_verified_budget"
        self.stats["candidates_unverified_budget"] += len(skipped)
        return logs, skipped

    # -- stage 5 -----------------------------------------------------------------
    def evidence(self, rec, verified):
        """[table] + the windows of the table's rows, fitted to the evidence budget; nothing verified: the top
        unverified windows (unverified_fallback)."""
        by_cid = {c.cid: c for c in verified}
        budget = int(self.cfg.evidence["max_chars"])
        if not rec["series"]:
            if not self.o["unverified_fallback"]:
                return [], "none"
            self.stats["unverified_fallback"] = 1
            pool = sorted([c for c in verified], key=lambda c: -c.score)[: int(self.o["unverified_fallback_items"])]
            return [c.to_item(rank=i, meta={"verified": False}) for i, c in enumerate(pool, 1)], "unverified_candidates"
        order = []
        for r in rec["series"][: int(self.o["table_rows_max"])]:
            if r["cid"] in by_cid and r["cid"] not in order:
                order.append(r["cid"])
        windows = [by_cid[cid].to_item(meta={"verified": True}) for cid in order]
        # fit the windows under the budget left by a table with placeholder numbers, then number them
        table_doc = rec["series"][0]["doc_id"]
        probe = R.table_text(rec, self.q.q_lang, {cid: 10 for cid in order})
        fit = self.a.pack([RetrievedItem(text=probe, doc_id=table_doc, kind="fact")] + windows, self.index, budget=budget)
        included = [it.meta.get("cid") for it in fit.items[1:]]
        nums = {cid: i + 2 for i, cid in enumerate(included)}
        text = R.table_text(rec, self.q.q_lang, nums)
        units = []
        for r in rec["series"]:
            if r.get("unit_id") and r["unit_id"] not in units:
                units.append(r["unit_id"])
        table = RetrievedItem(text=text, doc_id=table_doc, section_id=None, unit_ids=units, kind="fact",
                              rank=0, meta={"table": True, "rows": len(rec["series"])})
        items = [table] + [w for w in windows if w.meta.get("cid") in nums]
        for i, it in enumerate(items):
            it.rank = i + 1
        return items, "verified_table"

    def answer_call(self, evidence):
        abstain = self.cfg.evidence["abstain_token"]
        if self.o["answer_mode"] == "shared":
            msgs = build_answer_messages(self.q, evidence, self.cfg)
            version = "shared"
        else:
            system = SYNTH_SYSTEM.format(abstain=abstain, answer_lang=LANG_NAMES[self.q.q_lang])
            user = "Evidence:\n%s\n\nQuestion: %s" % (evidence.text if evidence.text else "(no passages found)",
                                                       self.q.text.strip())
            msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
            version = SYNTH_PROMPT_VERSION
        # the shared answer budget of this question's format (contracts.answer_num_predict: 256, or the timeline budget)
        r = self.chat("answer", msgs, num_predict=answer_num_predict(self.q, self.cfg), version=version)
        return r.text.strip()

    # -- the pipeline -----------------------------------------------------------------
    def run(self):
        t_all = time.perf_counter()
        o = self.o
        trace = {"variant": self.a.variant, "max_calls": self.budget.max,
                 "prompts": {"plan": PLAN_PROMPT_VERSION, "verify": VERIFY_PROMPT_VERSION,
                             "answer": "shared" if o["answer_mode"] == "shared" else SYNTH_PROMPT_VERSION},
                 "params": {k: o[k] for k in sorted(o) if not k.startswith("call_cache")},
                 "call_cache": self.cache.path, "stages": {}}
        scope_docs = self.a.docs_in_scope(self.q, self.index)
        metas = [doc_meta(d) for d in scope_docs]
        meta_by = {m["doc_id"]: m for m in metas}
        self.stats["docs_scope"] = len(metas)
        with self.a.client.ledger.scope() as whole:
            # 1. plan (+ document selection from metadata)
            with self.stage("plan") as st:
                plan = self.plan(metas)
                period = plan["period"] if any(p is not None for p in plan["period"]) else None
                chosen, sel = select_docs(metas, companies_chosen=plan["companies"], period=period, kind=plan["kind"],
                                          question=self.q.text, max_docs=int(o["max_target_docs"]),
                                          selection=self.a.doc_selection)
                self.stats["docs_targeted"] = len(chosen)
                st.update(docs_scope=len(metas), docs_targeted=len(chosen), kind=plan["kind"],
                          repair=self.stats["plan_repair"], program_plan=self.stats["plan_program"])
            trace["stages"]["plan"] = {k: plan[k] for k in ("entity", "metric", "kind", "unit", "terms", "entity_terms",
                                                             "period", "companies", "dropped_terms", "fixes", "model")}
            trace["stages"]["plan"]["doc_selection"] = sel

            # 2. search (no model)
            with self.stage("search") as st:
                cands, sinfo = S.search(self.root, self.corpus, plan, chosen, d_lang=self.q.d_lang, question=self.q.text,
                                        opts=o, round_no=1, line_only=self.a.line_windows, cfg=self.cfg)
                for i, c in enumerate(cands, 1):
                    c.cid = "r1-%d" % i
                    c.verdict = None
                st.update(rg_ms=sinfo.get("rg_ms", 0.0), rg_invocations=sinfo.get("rg_invocations", 0),
                          windows_ms=sinfo.get("windows_ms", 0.0), patterns=len(sinfo.get("patterns") or []),
                          hit_lines=sinfo.get("hit_lines", 0), windows=sinfo.get("windows", 0),
                          windows_with_number=sinfo.get("windows", 0) - sinfo.get("windows_without_number", 0),
                          kept=sinfo.get("kept", 0), selected=len(cands))
            self.stats.update(candidates=len(cands), hit_lines=sinfo.get("hit_lines", 0), windows=sinfo.get("windows", 0))
            self.stats["rg_ms"] += sinfo.get("rg_ms", 0.0)
            sinfo["candidates"] = [c.trace(int(o["trace_text_chars"])) for c in cands[: int(o["trace_candidates_max"])]]
            trace["stages"]["search"] = sinfo

            # 3. verify (keep the answer call, and one gap call when the budget has room for at least one verify
            # batch + one gap batch + the answer)
            rows = []
            with self.stage("verify") as st:
                reserve = 1 + (1 if (o["gap_fill"] and self.budget.left() >= 3) else 0)
                logs, skipped = self.verify_all(cands, plan, "verify", rows, meta_by, reserve=reserve)
                st.update(batches=len(logs), candidates=len(cands), verified=sum(b["n"] for b in logs),
                          skipped_for_budget=len(skipped), accepted=sum(b["accepted"] for b in logs),
                          model_rejected=sum(b["model_rejected"] for b in logs),
                          model_omitted=sum(b["model_omitted"] for b in logs),
                          reattributed=sum(b["reattributed"] for b in logs),
                          guard_rejected=sum(b["guard_rejected"] for b in logs),
                          unparsed=sum(b["unparsed"] for b in logs))
            trace["stages"]["verify"] = {"batches": logs, "skipped_for_budget": [c.cid for c in skipped]}
            verified = [c for c in cands if getattr(c, "verdict", None) not in (None, "not_verified_budget")]

            # 4. gap fill (<= 1 round): target documents without a verified value
            with self.stage("gap_fill") as st:
                if plan["kind"] == "single":
                    missing = [m["doc_id"] for m in chosen] if not rows else []
                else:
                    have = {r["doc_id"] for r in rows}
                    missing = [m["doc_id"] for m in chosen if m["doc_id"] not in have]
                gap = {"missing_docs": missing, "ran": False}
                st.update(missing_docs=len(missing), ran=0)
                if o["gap_fill"] and missing and self.budget.can(1, reserve=1):
                    seen_units = set()
                    for c in verified + skipped:
                        seen_units.update(c.number_unit_ids)
                        seen_units.update(c.hit_unit_ids)
                    gcands, ginfo = S.search(self.root, self.corpus, plan, chosen, d_lang=self.q.d_lang,
                                             question=self.q.text, opts=o, round_no=2, per_doc_cap=int(o["gap_per_doc"]),
                                             global_cap=int(o["gap_max_candidates"]), exclude_units=seen_units,
                                             only_docs=set(missing), line_only=self.a.line_windows, cfg=self.cfg)
                    for i, c in enumerate(gcands, 1):
                        c.cid = "r2-%d" % i
                        c.verdict = None
                    n_before = len(rows)
                    glogs, gskipped = self.verify_all(gcands, plan, "verify_gap", rows, meta_by, reserve=1,
                                                      max_batches=int(o["gap_max_batches"]))
                    ginfo["candidates"] = [c.trace(int(o["trace_text_chars"])) for c in gcands]
                    gap.update(ran=True, search=ginfo, batches=glogs, skipped_for_budget=[c.cid for c in gskipped],
                               accepted=len(rows) - n_before)
                    self.stats.update(gap_fill=1, candidates_gap=len(gcands), gap_accepted=len(rows) - n_before)
                    self.stats["rg_ms"] += ginfo.get("rg_ms", 0.0)
                    st.update(ran=1, rg_ms=ginfo.get("rg_ms", 0.0), windows_ms=ginfo.get("windows_ms", 0.0),
                              search_ms=ginfo.get("ms", 0.0), hit_lines=ginfo.get("hit_lines", 0),
                              windows=ginfo.get("windows", 0), kept=ginfo.get("kept", 0), selected=len(gcands),
                              excluded_seen=ginfo.get("excluded_seen", 0), batches=len(glogs),
                              verified=sum(b["n"] for b in glogs), skipped_for_budget=len(gskipped),
                              accepted=sum(b["accepted"] for b in glogs),
                              model_rejected=sum(b["model_rejected"] for b in glogs),
                              model_omitted=sum(b["model_omitted"] for b in glogs),
                              reattributed=sum(b["reattributed"] for b in glogs),
                              guard_rejected=sum(b["guard_rejected"] for b in glogs),
                              unparsed=sum(b["unparsed"] for b in glogs))
                    verified += [c for c in gcands if getattr(c, "verdict", None) not in (None, "not_verified_budget")]
                    cands = cands + gcands
            trace["stages"]["gap_fill"] = gap

            # 5a. reconcile (code) and the evidence pack
            with self.stage("reconcile") as st:
                target_periods = []
                if plan["kind"] != "single":
                    p0, p1 = plan["period"]
                    for m in chosen:
                        if m["years"]:
                            for y in range(m["years"][0], m["years"][1] + 1):
                                if (p0 is None or y >= p0) and (p1 is None or y <= p1) and y not in target_periods:
                                    target_periods.append(y)
                    target_periods.sort()
                rec = R.reconcile(rows, kind=plan["kind"], target_periods=target_periods, opts=o, q_lang=self.q.q_lang,
                                  known_stems=self.known_stems(plan))
                items, ev_from = self.evidence(rec, verified)
                evidence = self.a.pack(items, self.index)
                self.stats.update(rows=len(rec["series"]), periods_found=len(rec["periods_found"]),
                                  periods_missing=len(rec["periods_missing"]), conflicts=len(rec["conflicts"]),
                                  comparable=(-1 if rec["comparable"] is None else int(bool(rec["comparable"]))))
                st.update(rows_accepted=len(rows), series_rows=len(rec["series"]), restatements=len(rec["restatements"]),
                          conflicts=len(rec["conflicts"]), periods_found=len(rec["periods_found"]),
                          periods_missing=len(rec["periods_missing"]), evidence_from=ev_from,
                          evidence_items=evidence.n_included, evidence_chars=evidence.chars)

            # 5b. the answer call
            with self.stage("answer") as st:
                if not self.budget.can(1):
                    raise BudgetExceeded("AGENTIC: no call left for the answer")
                answer = self.answer_call(evidence)
                st.update(answer_mode=o["answer_mode"], answer_chars=len(answer))
            whole_summary = whole.summary()
        total_ms = _ms(t_all)
        structured = {
            "kind": plan["kind"], "entity": plan["entity"], "metric": plan["metric"], "unit_kind": plan["unit"],
            "series": [_series_row(r) for r in rec["series"]],
            "comparable": rec["comparable"], "main_class": rec["main_class"], "definition_classes": rec["definition_classes"],
            "trend": rec["trend"], "notes": R.table_notes(rec, self.q.q_lang) if rec["series"] else [],
            "periods_found": rec["periods_found"], "periods_missing": rec["periods_missing"],
            "target_periods": [str(p) for p in target_periods], "conflicts": rec["conflicts"],
            "restatements": [_series_row(r) for r in rec["restatements"]],
            "narrower": [_series_row(r) for r in rec["narrower"]],
            "projections": [_series_row(r) for r in rec["projections"]],
            "rows_all": [_series_row(r) for r in rows], "evidence_from": ev_from,
        }
        trace["stages"]["answer"] = {"evidence_from": ev_from, "table": items[0].text if ev_from == "verified_table" else None,
                                     "n_offered": evidence.n_offered, "n_included": evidence.n_included,
                                     "chars": evidence.chars, "answer_mode": o["answer_mode"]}
        trace["structured"] = structured
        trace["guard_reasons"] = self.guard_reasons
        trace["calls"] = self.calls
        trace["calls_by_purpose"] = {p: self.budget.log.count(p) for p in dict.fromkeys(self.budget.log)}
        trace["timing"] = self._timing(whole_summary, total_ms)
        trace["ledger_calls"] = whole_summary["llm_calls"]
        self.stats["program_ms"] = trace["timing"]["totals"]["program_ms"]
        trace["stats"] = self.stats
        order = {c.cid: i for i, c in enumerate(cands)}
        verified.sort(key=lambda c: (c.verdict != "accepted", order.get(c.cid, 0)))
        retrieved = [c.to_item(rank=i, meta={"verdict": c.verdict}) for i, c in enumerate(verified, 1)]
        res = AnswerResult(answer_text=answer, evidence=evidence, retrieved=retrieved, trace=trace)
        # RunRecord.structured is added by the integration patch; until then it lives in trace["structured"]
        if "structured" in {f.name for f in dc_fields(AnswerResult)}:
            res.structured = structured
        else:
            object.__setattr__(res, "structured", structured)
        return res

    def _timing(self, whole, total_ms):
        stages = self.timing
        totals = {k: round(sum((s.get(k) or 0) for s in stages), 1) for k in
                  ("wall_ms", "program_ms", "call_elapsed_ms", "cache_hits") + SCOPE_KEYS}
        totals["rg_ms"] = round(sum((s.get("rg_ms") or 0) for s in stages), 1)
        totals["session_ms"] = total_ms
        totals["outside_stages_ms"] = round(total_ms - totals["wall_ms"], 1)
        check = {k: {"stages": totals[k], "ledger": whole.get(k, 0)} for k in SCOPE_KEYS}
        ok = all(abs((v["stages"] or 0) - (v["ledger"] or 0)) <= (1.0 if k.endswith("_ms") else 0) for k, v in check.items())
        return {"stages": stages, "totals": totals, "ledger": {k: whole.get(k, 0) for k in SCOPE_KEYS},
                "check": check, "reconciles": ok}


SERIES_FIELDS = ("period", "value", "unit", "definition", "quote", "unit_id", "doc_id", "verified")


def _series_row(r):
    out = {k: r.get(k) for k in SERIES_FIELDS}
    for k in ("value_num", "value_raw", "value_reading", "def_class", "def_flags", "role", "period_how",
              "period_inferred", "doc_years", "section_id", "cid", "round", "model_value", "model_period", "qualifiers"):
        out[k] = r.get(k)
    return out
