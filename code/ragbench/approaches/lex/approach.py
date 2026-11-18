"""LEX: layer-1 report cards, LLM keywords, the reports' own phrases, code-compiled regex over the whole reports,
structure-aware windows, the shared answer call. At most 3 model calls and 60 s per question. See defaults.py.

    index      the shared knowledge base (AGENTIC3's ingestion: units, sections, facts - no model) and its report
               cards (kb/graph.py doc_cards over the label layer - no model)
    call 1     keywords, names, years, answer kind                                   (prompts.KW_*)
    route      the cards, when the scope is not given or is larger than route_min_docs   (search.route)
    pass 1     the keywords compiled and run over the chosen reports; the best windows feed the phrase list
    call 2     pick the reports' own phrases (or, llm_regex, write the regexes)     (prompts.PICK_* / REGEX_*)
    pass 2     picked phrases + added terms (weight 1) + keywords (weight w_keyword) compiled and run again
    windows    structure-aware (or fixed) around every hit, scored on the text they SHOW, then chosen by the
               characters they cost: one per report first when the question needs several, then by score, no
               repeats, so the shared packer never cuts one
    call 3     the shared answer prompt over the shared 6000-char evidence budget

Upgrades (plus.py; flags on LEX, all three on in LEX2): `bm25` (a BM25 safety net fused by reciprocal rank, its own
windows capped), `report_vocab` (every report its own share of call 2's phrase list, with the reports' years shown),
`year_pass` (a year with no supporting window is searched again with the other years' labels). `no_labels` ("lite"):
cards and phrase list from section titles only - no fact or label layer needed.

Every call goes through a durable call cache (the agentic pipeline's), so a stopped run resumes and an ablation that
asks the same question with the same prompt replays the call instead of making it.
"""
import json
import os
import time
import uuid
from pathlib import Path

from ...contracts import AnswerResult, Approach, RetrievedItem, answer_num_predict, build_answer_messages, item_header
from ...kb import graph as G
from ...kb import ingest as I
from ...kb import labels as L
from ...kb import schema as KB
from ...llm import ChatResult
from ...util import atomic_write_json, now_iso, read_json, safe_name
from ..agentic.approach import CallBudget, CallCache
from ..agentic3 import retrieve as RT
from . import learn as LN
from . import plus as X
from . import prompts as P
from . import search as S
from .defaults import APPROACH_VERSION, DEFAULTS, FLAGS, PROMPT_VERSION, UPGRADES, VALUE_FLAGS

OPTIONS_ENV = "LEX_OPTIONS"          # DEV tuning only (a shared call cache path, ...); a --final run refuses it


def parse_flags(flags):
    if flags is None:
        return {}
    if isinstance(flags, str):
        d = {}
        for p in flags.replace("+", ",").split(","):
            p = p.strip()
            if p:
                k, _, v = p.partition("=")
                d[k.strip()] = v.strip() if v else True
        flags = d
    out = {}
    for k, v in dict(flags).items():
        if k not in FLAGS:
            raise ValueError("unknown LEX flag %r; known: %s" % (k, ", ".join(FLAGS)))
        if k in VALUE_FLAGS:
            n = int(v)
            if n < 50:
                raise ValueError("LEX flag %s must be >= 50 characters" % k)
            out[k] = n
        elif v if isinstance(v, bool) else str(v).lower() in ("1", "true", "yes", "on"):
            out[k] = True
    if out.get("llm_regex") and out.get("free_keywords"):
        raise ValueError("llm_regex and free_keywords are exclusive (both replace call 2)")
    return out


def _ms(t0):
    return round((time.perf_counter() - t0) * 1000.0, 1)


class LexAgent(Approach):
    name = "LEX"
    needs_gold = False
    ON = ()                              # upgrades always on (LEX2: all of them)
    collect_features = False             # tools/lex_learn.py's dump subclass: keep every window's feature vector

    def __init__(self, cfg=None, client=None, flags=None, options=None):
        super().__init__(cfg=cfg, client=client, flags=parse_flags(flags))
        env = os.environ.get(OPTIONS_ENV)
        self.env_options = json.loads(env) if env else None
        self.opts = dict(DEFAULTS, **(self.env_options or {}), **(options or {}))
        if self.flags.get("window"):
            self.opts["window_chars"] = int(self.flags["window"])
        self.up = {k: (k in self.ON) or bool(self.flags.get(k)) for k in UPGRADES}
        self.opts["report_vocab"] = self.up["report_vocab"]
        self.opts["no_labels"] = bool(self.flags.get("no_labels"))
        caps = self.cfg.caps or {}
        self.max_calls = int(caps.get("lex_max_calls") or self.opts["max_calls_default"])
        self.max_seconds = float(caps.get("lex_max_seconds") or self.opts["max_seconds_default"])
        self._con = None
        self._cache = None
        self._cache_for = None
        # the learned window score (learn.py). The dump subclass keeps the hand-set score and only collects features,
        # so `learned` without a model file is an error for a scored run but not for the dump.
        self.model = None
        if self.flags.get("learned"):
            path = self.opts.get("learned_model")
            if not path and not self.collect_features:
                raise ValueError("flag `learned` needs options[\"learned_model\"] (LEX_OPTIONS)")
            self.model = LN.Model.load(path) if path else None

    # -- index ----------------------------------------------------------------------------------------------------
    def kb(self):
        if self._con is None:
            self._con = RT.open_kb(self.cfg)
        return self._con

    def config_signature(self):
        return {"kb": KB.component_versions(self.cfg), "kb_path": str(I.kb_path(self.cfg)), "approach": APPROACH_VERSION,
                "labels": L.LABELS_VERSION, "cards": "graph._cards/%s" % G.GRAPH_VERSION}

    def _build_index(self, corpus_set, index_dir):
        """The shared knowledge base holds every report of the set (AGENTIC3's ingestion, no model) and the report
        cards are current (only the cards: plus.refresh_cards, not the whole graph). `build_ms` is what THIS build
        paid - milliseconds when an earlier run left the knowledge base filled - so `cost_from_scratch` records what
        the set's reports cost when nothing is cached (the ingestion log), the number to report for the method."""
        t0 = time.perf_counter()
        con = self.kb()
        stats = RT.ingest_corpus(con, corpus_set, self.cfg)
        t1 = time.perf_counter()
        cards = X.refresh_cards(con)
        payload = {"kb_path": str(I.kb_path(self.cfg)), "n_docs": len(corpus_set.docs), "doc_ids": corpus_set.doc_ids(),
                   "versions": KB.component_versions(self.cfg), "ingest": stats, "cards": cards,
                   "ingest_ms": round((t1 - t0) * 1000.0, 1), "cards_ms": _ms(t1), "build_ms": _ms(t0),
                   "cost_from_scratch": X.cost_from_scratch(con, corpus_set.doc_ids())}
        atomic_write_json(index_dir / "lex_index.json", payload, indent=1)
        return payload

    def _load_index(self, corpus_set, index_dir):
        payload = read_json(index_dir / "lex_index.json") or {}
        if not payload or RT.missing_documents(self.kb(), corpus_set.doc_ids()):
            return self._build_index(corpus_set, index_dir)
        X.refresh_cards(self.kb())
        return payload

    def call_cache(self):
        path = self.opts.get("call_cache_path")
        if not path and getattr(self.client.ledger, "path", None):
            path = Path(self.client.ledger.path).parent / ("lex_calls_%s.sqlite" % safe_name(self.variant))
        if self.opts.get("call_cache") == "off":
            path = None
        if self._cache is None or str(path) != str(self._cache_for):
            self._cache = CallCache(path)
            self._cache_for = path
        return self._cache

    def score_fn(self):
        """-> the window scorer `search.windows` should use, or None for the hand-set weights (every other arm)."""
        if not self.flags.get("learned"):
            return None
        return LN.scorer(self.model, keep=self.collect_features)

    def observe(self, q, wins, terms, kw, docs):
        """Behaviour-neutral hook: the candidate windows of the second pass, before fusion and selection. The dump
        subclass writes them out; LEX itself does nothing."""

    def _answer(self, query, index):
        return _Session(self, query, index).run()


class _Session:
    def __init__(self, ap, query, index):
        self.a, self.q, self.index, self.cfg, self.o = ap, query, index, ap.cfg, ap.opts
        self.con = ap.kb()
        self.budget = CallBudget(ap.max_calls)
        self.cache = ap.call_cache()
        self.calls, self.timing = [], {}
        self.stats = {"calls": 0, "cache_hits": 0}

    # a model call through the durable cache (the agentic pipeline's scheme: same payload -> same stored reply)
    def chat(self, purpose, messages, *, fmt=None, num_predict=None):
        client = self.a.client
        self.budget.spend(purpose)
        payload = {"model": self.cfg.models["chat"], "digest": self.cfg.models.get("chat_digest"), "purpose": purpose,
                   "version": PROMPT_VERSION if purpose != "answer" else "shared", "messages": messages, "format": fmt,
                   "num_predict": num_predict, "options": client.options(num_predict=num_predict),
                   "think": self.cfg.ollama.get("think")}
        # the request the model really gets after variants.py injection (card, list rule, allowance); None = unchanged,
        # so the keys of every run without an injecting option stay what they were
        eff_sig = getattr(client, "effective_request_sig", None)
        eff = eff_sig(messages, purpose, num_predict) if callable(eff_sig) else None
        if eff:
            payload["effective_request"] = eff
        key = self.cache.key(payload)
        hit = self.cache.get(key)
        t0 = time.perf_counter()
        if hit is not None:
            ctx = dict(client.ledger.context())
            ctx.setdefault("phase", "query")
            ctx.setdefault("approach", self.a.variant)
            rec = dict(ctx, ts=now_iso(), call_id=uuid.uuid4().hex[:12], kind="chat", model=self.cfg.models["chat"],
                       purpose=purpose, ok=True, prompt_tokens=int(hit.get("prompt_tokens") or 0),
                       completion_tokens=int(hit.get("completion_tokens") or 0),
                       wall_ms=float(hit.get("wall_ms") or 0.0), wait_ms=0.0, load_ms=0.0, attempts=0,
                       done_reason=hit.get("done_reason") or "",
                       prompt_chars=sum(len(m.get("content") or "") for m in messages),
                       completion_chars=len(hit.get("text") or ""), cache_hit=True, cached_call_id=hit.get("call_id"))
            client.ledger.record(rec)
            r = ChatResult(text=hit.get("text") or "", prompt_tokens=rec["prompt_tokens"],
                           completion_tokens=rec["completion_tokens"], wall_ms=rec["wall_ms"],
                           done_reason=rec["done_reason"], model=rec["model"], call_id=rec["call_id"])
            self.stats["cache_hits"] += 1
        else:
            r = client.chat(messages, purpose=purpose, format=fmt, num_predict=num_predict)
            self.cache.put(key, {"text": r.text, "prompt_tokens": r.prompt_tokens, "completion_tokens": r.completion_tokens,
                                 "wall_ms": round(r.wall_ms, 1), "done_reason": r.done_reason, "call_id": r.call_id,
                                 "ts": now_iso(), "purpose": purpose})
        self.stats["calls"] = self.budget.used
        self.calls.append({"n": self.budget.used, "purpose": purpose, "call_id": r.call_id,
                           "prompt_tokens": r.prompt_tokens, "completion_tokens": r.completion_tokens,
                           "wall_ms": round(r.wall_ms, 1), "done_reason": r.done_reason, "cache_hit": hit is not None,
                           "elapsed_ms": _ms(t0)})
        return r

    def stage(self, name, t0):
        self.timing[name] = _ms(t0)

    def item(self, w, rank, reports):
        """A window -> RetrievedItem. A table window records WHICH rows it shows: its unit id stands for the whole table
        (the evidence metric counts unit ids), so the rows are the provenance that says whether the answer row is in."""
        meta = {"hit": w["hit"], "terms": w["terms"], "breadcrumb": reports.breadcrumb(w["doc_id"], w["section_id"])}
        if w.get("rows") is not None:
            meta.update(table_rows=w["rows"], table_rows_total=w["rows_total"])
        return RetrievedItem(text=w["text"], doc_id=w["doc_id"], section_id=w["section_id"],
                             unit_ids=list(w["unit_ids"]), kind="window" if w["kind"] == "window" else "unit",
                             score=w["score"], rank=rank, span=w["span"], meta=meta)

    def cost(self, w):
        """The characters a window takes in the shared evidence block: its header ("[n] title > path", n <= 99) +
        newline + text + the separator, exactly as contracts.pack_evidence renders it."""
        it = RetrievedItem(text=w["text"], doc_id=w["doc_id"], section_id=w["section_id"])
        return len(item_header(99, it, self.index.corpus)) + 1 + len((w["text"] or "").strip()) + 2

    def scope(self):
        """-> (doc ids to search, route trace). The question's reports, unless `open_corpus` withholds them; the
        cards route when the candidate list is longer than `route_min_docs` (always, with open_corpus)."""
        given = [d.id for d in self.a.docs_in_scope(self.q, self.index)]
        pool = list(self.index.corpus.doc_ids()) if self.a.flags.get("open_corpus") else given
        info = {"mode": "open_corpus" if self.a.flags.get("open_corpus") else "given", "given": given,
                "pool": len(pool), "routed": pool, "scores": []}
        return pool, info

    def run(self):
        o, q, fl = self.o, self.q, self.a.flags
        t_all = time.perf_counter()
        trace = {"variant": self.a.variant, "versions": {"approach": APPROACH_VERSION, "prompts": PROMPT_VERSION},
                 "params": dict(o), "flags": dict(fl), "stages": {}}

        # 1. keywords
        t0 = time.perf_counter()
        r = self.chat("keywords", P.kw_messages(q.text, q.d_lang, o), fmt=P.KW_SCHEMA, num_predict=int(o["kw_num_predict"]))
        kw, how = P.parse_keywords(r.text, o)
        self.stage("keywords", t0)
        trace["stages"]["keywords"] = dict(kw, parse=how, model_output=(r.text or "")[:800])

        # 2. layer 1
        t0 = time.perf_counter()
        pool, rinfo = self.scope()
        docs = pool
        if len(pool) > int(o["route_min_docs"]) or fl.get("open_corpus"):
            docs, scored = S.route(self.con, pool, kw, q.d_lang, o, titles_only=bool(fl.get("title_routing")),
                                   no_labels=bool(fl.get("no_labels")))
            rinfo.update(routed=docs, scores=scored[:12], titles_only=bool(fl.get("title_routing")))
        rinfo["recall"] = (len(set(rinfo["given"]) & set(docs)) / len(rinfo["given"])) if rinfo["given"] else None
        self.stage("route", t0)
        trace["stages"]["route"] = rinfo

        reports = S.Reports(self.con)
        doc_texts = None
        if fl.get("fixed_window"):
            doc_texts = {d: self.index.corpus.doc(d).text for d in docs if self.index.corpus.has_doc(d)}
        mode = "fixed" if fl.get("fixed_window") else "structure"

        # 3. first pass with the keywords (the phrase list's feedback, and the whole search for free_keywords)
        t0 = time.perf_counter()
        kterms = S.compile_terms(kw["keywords"], q.d_lang, "keyword", 1.0, o)
        hits1, by_id = S.search(reports, docs, kterms)
        wins1, _wms = S.windows(reports, hits1, by_id, kterms, kw, o, mode=mode, doc_texts=doc_texts)
        self.stage("pass1", t0)
        trace["stages"]["pass1"] = {"terms": [t.trace() for t in kterms], "hit_units": len(hits1), "windows": len(wins1)}

        # 4. call 2: the reports' own phrases (or the model's regexes)
        terms, c2 = kterms, {"ran": False}
        if not fl.get("free_keywords"):
            t0 = time.perf_counter()
            vocab = S.vocabulary(self.con, reports, docs, kw, q.d_lang, wins1, o)
            if self.a.up["report_vocab"] and len(docs) > 1:      # each phrase shows the years of its reports
                yrs = {d: r.get("year_from") for d, r in S.doc_rows(self.con, docs).items()}
                for v in vocab:
                    ys = sorted({yrs.get(d) for d in v.get("docs") or [] if yrs.get(d)})
                    v["where"] = ", ".join(str(y) for y in ys) if 0 < len(ys) < len(set(yrs.values())) else ""
            c2 = {"ran": True, "vocab": [v["text"] for v in vocab], "vocab_feedback": sum(v["feedback"] for v in vocab)}
            if fl.get("llm_regex"):
                r = self.chat("regex", P.regex_messages(q.text, vocab, kw["keywords"], q.d_lang, o), fmt=P.REGEX_SCHEMA,
                              num_predict=int(o["regex_num_predict"]))
                got, how = P.parse_regex(r.text, o)
                rterms, bad = S.compile_regexes(got["regex"], weight=float(o["w_picked"]))
                c2.update(parse=how, shape=got["shape"], regex=got["regex"], n_usable=len(rterms),
                          regex_invalid=[{"pattern": p, "error": e} for p, e in bad], done_reason=r.done_reason,
                          model_output=(r.text or "")[:800])
                # MATCHED to the phrase-pick branch below (second opinion, 2026-09-23): the model's regexes take the
                # picked phrases' place and the call-1 keywords stay beside them at w_keyword, exactly as LEX2 keeps
                # them; the keywords alone only when nothing usable came back (the same trigger as "no phrase
                # picked"). Before, valid-but-useless regexes lost the keyword channel LEX2 kept.
                if rterms and o["regex_keep_keywords"]:
                    seen = {t.text for t in rterms}
                    kw2 = S.compile_terms([k for k in kw["keywords"] if k not in seen], q.d_lang, "keyword",
                                          float(o["w_keyword"]), o)
                    terms = rterms + kw2
                elif rterms:
                    terms = rterms
                elif o["llm_regex_fallback"]:
                    terms = kterms
                    c2["regex_fallback"] = True
                else:
                    terms = []
            elif vocab:
                r = self.chat("pick", P.pick_messages(q.text, vocab, q.d_lang, o), fmt=P.PICK_SCHEMA,
                              num_predict=int(o["pick_num_predict"]))
                got, how = P.parse_pick(r.text, len(vocab), o)
                picked = [vocab[i]["text"] for i in got["pick"]]
                pterms = S.compile_terms(picked + got["add"], q.d_lang, "picked", float(o["w_picked"]), o)
                seen = {t.text for t in pterms}
                kw2 = S.compile_terms([k for k in kw["keywords"] if k not in seen], q.d_lang, "keyword",
                                      float(o["w_keyword"]), o)
                terms = pterms + kw2 if pterms else kterms
                c2.update(parse=how, picked=picked, added=got["add"], n_usable=len(pterms), done_reason=r.done_reason,
                          model_output=(r.text or "")[:800])
            else:
                c2.update(ran=False, skipped="no phrase shares a word with the keywords")
            self.stage("call2", t0)
        trace["stages"]["call2"] = c2

        # 5. the search that feeds the answer
        t0 = time.perf_counter()
        sfn = self.a.score_fn()
        if terms is kterms and sfn is None:
            hits, wins = hits1, wins1
        else:
            hits, by_id = S.search(reports, docs, terms)
            wins, _wms = S.windows(reports, hits, by_id, terms, kw, o, mode=mode, doc_texts=doc_texts, score_fn=sfn)
        self.a.observe(q, wins, terms, kw, docs)
        # one window per report first only when the question needs several: the benchmark gave it several reports (a
        # series, a comparison), or - reports routed by the cards - it asks about several years or names
        routed = bool(rinfo.get("scores"))
        multi_q = len(S.question_years(kw["years"])) >= 2 or len(kw["names"]) >= 2
        reserve = len(docs) > 1 and (not routed or multi_q)
        prior = {}
        if routed:                    # the card rank as a modest prior, not a multiplier on the window score
            prior = {d: float(o["w_card_prior"]) * (len(docs) - i) / len(docs) for i, d in enumerate(docs)}
        caps, binfo = None, None
        if self.a.up["bm25"]:
            # BM25 on the ORIGINAL question (it keeps what call 1 dropped) when it is in the reports' language, plus
            # the keywords and names (which are); fused by reciprocal rank, BM25-only windows capped
            qtexts = ([q.text] if q.q_lang == q.d_lang else []) + kw["keywords"] + kw["names"]
            bm, binfo = X.bm25_windows(reports, docs, qtexts, q.d_lang, o)
            wins = X.fuse(wins, bm, o)
            caps = {"bm25": int(o["bm25_max_chars"])}
            prior = {d: p * float(o["rrf_prior_scale"]) for d, p in prior.items()}   # onto the fused scale
            binfo["fused_both"] = sum(1 for w in wins if w.get("source") == "both")
        budget = self.cfg.evidence["max_chars"]
        chosen, sinfo = S.select(wins, docs, o, reserve=reserve, cost=self.cost, budget=budget, prior=prior,
                                 source_caps=caps)
        yinfo = None
        if self.a.up["year_pass"]:
            chosen, yinfo = X.year_pass(self.con, reports, docs, chosen, terms, kw, q.d_lang, o, self.cost, budget)
        self.stage("search", t0)
        learned = {"model": self.o.get("learned_model"), "meta": (self.a.model.meta if self.a.model else None)} \
            if fl.get("learned") else None
        trace["stages"]["search"] = {"terms": [t.trace() for t in terms], "hit_units": len(hits), "windows": len(wins),
                                     "reserve": reserve, "select": sinfo, "bm25": binfo, "year_pass": yinfo,
                                     "learned": learned,
                                     "selected": [{k: w.get(k) for k in ("doc_id", "hit", "score", "kind", "terms",
                                                                         "source")}
                                                  | {"chars": len(w["text"])} for w in chosen]}

        # 6. pack and answer (the shared prompt and budget)
        t0 = time.perf_counter()
        items = [self.item(w, i, reports) for i, w in enumerate(chosen, 1)]
        evidence = self.a.final_pack(items, self.index, q)
        self.stage("pack", t0)
        t0 = time.perf_counter()
        msgs = build_answer_messages(q, evidence, self.cfg)
        r = self.chat("answer", msgs, num_predict=answer_num_predict(q, self.cfg))
        self.stage("answer", t0)
        # failure bookkeeping (second opinion): an EMPTY answer and a TRUNCATED one (finish = length) are different
        # failures and both stay in the denominator; search terms that hit nothing, by where they came from
        zero = {}
        for t in terms:
            if t.df == 0:
                zero[t.source] = zero.get(t.source, 0) + 1
        trace.update(calls=self.calls, timing=dict(self.timing, total_ms=_ms(t_all)), stats=self.stats,
                     evidence={"n_offered": evidence.n_offered, "n_included": evidence.n_included,
                               "chars": evidence.chars},
                     failures={"answer_empty": not (r.text or "").strip(), "answer_truncated": r.done_reason == "length",
                               "call2_truncated": c2.get("done_reason") == "length",
                               "call2_parse": c2.get("parse"), "call2_shape": c2.get("shape"),
                               "call2_usable": c2.get("n_usable"), "regex_fallback": bool(c2.get("regex_fallback")),
                               "zero_hit_terms": zero, "n_terms": len(terms)})
        return AnswerResult(answer_text=r.text.strip(), evidence=evidence, retrieved=items, trace=trace)


class Lex2Agent(LexAgent):
    """LEX2: LEX with every upgrade on - the BM25 safety net, per-report phrase lists, the missing-year pass. Accepts
    LEX's flags (open_corpus, window=<n>, ...) on top."""
    name = "LEX2"
    ON = UPGRADES
