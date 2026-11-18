"""LEX3: hybrid-grounded lexical planning (2026-09-24; Astra's items 3, 5, 6 and 8 of UPGRADE_PLAN.md). A separately
named method: LEX and LEX2 are untouched (this class reuses LexAgent only for its knowledge-base handle and call cache).

    index     LEX's knowledge base (units, sections, report cards; no model) + the shared HybridIndex (B2/B3 chunks,
              BM25, prebuilt vectors; document vectors are never computed at query time)
    question  (code) BM25 + dense over the whole allowed pool from the ORIGINAL question, top 40 per channel - these
              candidates survive whatever the planner does
    call 1    interpret: entities, explicit periods, operation, <= 3 facets              (prompts.INTERPRET_*)
    route     (code) > route_min_docs reports: cards + capped best-chunk evidence, shortlist <= 8, explicit
              entity/period matches kept; question candidates outside the shortlist stay in (the escape channel)
    feedback  (code) a phrase list built from the retrieved evidence, IDs + sources, spread over facets and reports
    call 2    pick phrase IDs per facet (+ bounded added terms, marked not observed)     (prompts.PICK_*)
    lexical   (code) picked phrases -> LEX's Persian-aware regexes over the shortlist; lower-weight morphology;
              issuer-only name demotion
    fuse      (code) equal-weight RRF over regions (one contribution per channel per region) -> build_blocks with one
              block reserved per requirement -> one bounded follow-up retrieval per missing requirement
    call 3    the shared answer prompt (the question's answer format) over the shared evidence budget

Fallback is a mode, not an accident (trace.degraded_mode): an empty or failed interpretation answers from the question
channels alone (no call 2); no usable phrase list skips call 2 and keeps the facet terms beside the question channels;
a failed or empty call 2 keeps the facet terms and the question channels. Fallback is never triggered by how few units
matched.
"""
import json
import os
import time
import uuid
from pathlib import Path

from ...contracts import AnswerResult, Approach, answer_num_predict, build_answer_messages
from ...evidence import build_blocks
from ...kb import graph as G
from ...kb import ingest as I
from ...kb import labels as L
from ...kb import schema as KB
from ...llm import BudgetExceeded, ChatResult
from ...normalize import fold
from ...retrieval import hybrid as H
from ...util import atomic_write_json, now_iso, read_json
from ..agentic.approach import CallBudget
from ..agentic3 import retrieve as RT
from ..lex import plus as X
from ..lex import search as S
from ..lex.approach import LexAgent
from . import core as C
from . import prompts as P
from .defaults import APPROACH_VERSION, DEFAULTS, FLAGS, PROMPT_VERSION

OPTIONS_ENV = "LEX3_OPTIONS"         # DEV tuning only; a --final run refuses it (run.TUNING_ENV)
EMBED_AT_BUILD = True                # like HYBRID: missing chunk vectors are embedded at INDEX time (prebuild first)

# call purposes: "plan" and "pick" are variants.PLANNING_PURPOSES, so the doc_card option reaches both planning calls
PURPOSE_INTERPRET, PURPOSE_PICK, PURPOSE_ANSWER = "plan", "pick", "answer"


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
            raise ValueError("unknown LEX3 flag %r; known: %s" % (k, ", ".join(FLAGS)))
        if v if isinstance(v, bool) else str(v).lower() in ("1", "true", "yes", "on"):
            out[k] = True
    if out.get("no_lexical") and out.get("no_pick"):
        raise ValueError("no_lexical already removes call 2; drop no_pick")
    return out


def _ms(t0):
    return round((time.perf_counter() - t0) * 1000.0, 1)


def _hit_cand(h):
    return {"doc_id": h.doc_id, "span": list(h.span), "unit_ids": list(h.unit_ids), "section_id": h.section_id,
            "chunk": h.chunk}


def _channel(hits, ch, keep=None):
    rows = [h for h in hits if h.ranks.get(ch) is not None and (keep is None or h.doc_id in keep)]
    rows.sort(key=lambda h: (h.ranks[ch], h.chunk))
    return [_hit_cand(h) for h in rows]


class Lex3Agent(LexAgent):
    name = "LEX3"
    needs_gold = False

    def __init__(self, cfg=None, client=None, flags=None, options=None):
        Approach.__init__(self, cfg=cfg, client=client, flags=parse_flags(flags))
        env = os.environ.get(OPTIONS_ENV)
        self.env_options = json.loads(env) if env else None
        self.opts = dict(DEFAULTS, **(self.env_options or {}), **(options or {}))
        caps = self.cfg.caps or {}
        self.max_calls = int(caps.get("lex_max_calls") or self.opts["max_calls_default"])
        self.max_seconds = float(caps.get("lex_max_seconds") or self.opts["max_seconds_default"])
        self._con = None
        self._cache = None
        self._cache_for = None
        self.model = None
        self.up = {}
        self._prep = {}

    # -- index ----------------------------------------------------------------------------------------------------
    def config_signature(self):
        return {"kb": KB.component_versions(self.cfg), "kb_path": str(I.kb_path(self.cfg)), "approach": APPROACH_VERSION,
                "labels": L.LABELS_VERSION, "cards": "graph._cards/%s" % G.GRAPH_VERSION,
                "hybrid": H.signature(self.cfg)}

    def _hybrid(self, corpus_set, build):
        if not build:
            try:
                return H.HybridIndex.load(corpus_set, self.cfg, client=self.client)
            except (FileNotFoundError, RuntimeError):
                pass
        hx = H.HybridIndex.open(corpus_set, self.cfg, client=self.client, embed=EMBED_AT_BUILD)
        if EMBED_AT_BUILD and not hx.dense_complete:
            hx = H.HybridIndex.build(corpus_set, self.cfg, client=self.client, embed=True)
        return hx

    def _build_index(self, corpus_set, index_dir):
        lex = LexAgent._build_index(self, corpus_set, index_dir)       # KB ingest + cards (lex_index.json)
        hx = self._hybrid(corpus_set, build=True)
        atomic_write_json(Path(index_dir) / "hybrid_ref.json", hx.status(), indent=1)
        return {"lex": lex, "hybrid": hx}

    def _load_index(self, corpus_set, index_dir):
        lex = read_json(Path(index_dir) / "lex_index.json") or {}
        if not lex or RT.missing_documents(self.kb(), corpus_set.doc_ids()):
            lex = LexAgent._build_index(self, corpus_set, index_dir)
        else:
            X.refresh_cards(self.kb())
        return {"lex": lex, "hybrid": self._hybrid(corpus_set, build=False)}

    # -- the question channels, computed once per query (the variants card hook may ask first) --------------------
    def prepare(self, query, index):
        """-> the question-only stage: pool, question hits (BM25 + dense over the pool, from the ORIGINAL question) and
        the question-only routing. Memoised per query, so the doc_card hook and the session share one search."""
        key = (query.question_id, query.text, query.setting, json.dumps(query.doc_scope, sort_keys=True, default=str))
        if key in self._prep:
            return self._prep[key]
        t0 = time.perf_counter()
        o = self.opts
        hx = index.payload["hybrid"]
        given = [d.id for d in self.docs_in_scope(query, index)]
        pool = list(index.corpus.doc_ids()) if self.flags.get("open_corpus") else given
        hits = hx.search(query.text, doc_ids=pool, k=int(o["hybrid_k"]), depth=int(o["hybrid_depth"]),
                         client=self.client)
        routing = len(pool) > int(o["route_min_docs"])
        years = P.years_of(query.text)
        if routing:
            kw = {"keywords": [query.text] if query.q_lang == query.d_lang else [], "names": [], "years": years}
            short, rinfo = C.route(self.kb(), pool, hits, kw, query.d_lang, o)
        else:
            short, rinfo = list(pool), {}
        prep = {"given": given, "pool": pool, "hits": hits, "routing": routing, "shortlist_q": short,
                "route_q": rinfo, "ms": _ms(t0)}
        self._prep = {key: prep}                  # one query at a time
        return prep

    def variant_routed_ids(self, query, index):
        """variants-v2 card hook: the question-only shortlist (before call 1), best first."""
        try:
            return list(self.prepare(query, index)["shortlist_q"])
        except Exception:  # noqa: BLE001 - the hook must never break a query; the session will raise it properly
            return None

    def _answer(self, query, index):
        return _Session3(self, query, index).run()


class _Session3:
    def __init__(self, ap, query, index):
        self.a, self.q, self.index, self.cfg, self.o = ap, query, index, ap.cfg, ap.opts
        self.con = ap.kb()
        self.hx = index.payload["hybrid"]
        self.budget = CallBudget(ap.max_calls)
        self.cache = ap.call_cache()
        self.calls, self.timing = [], {}
        self.stats = {"calls": 0, "cache_hits": 0}

    # a model call through the durable cache (LEX's scheme, LEX3's prompt version, the effective request in the key)
    def chat(self, purpose, messages, *, fmt=None, num_predict=None):
        client = self.a.client
        self.budget.spend(purpose)
        payload = {"model": self.cfg.models["chat"], "digest": self.cfg.models.get("chat_digest"), "purpose": purpose,
                   "version": PROMPT_VERSION if purpose != PURPOSE_ANSWER else "shared", "messages": messages,
                   "format": fmt, "num_predict": num_predict, "options": client.options(num_predict=num_predict),
                   "think": self.cfg.ollama.get("think")}
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

    # -- lexical search -----------------------------------------------------------------------------------------------
    def compile(self, specs):
        """specs: [(text, source, weight, facet, observed, phrase_id)] -> ([Term], provenance by term text)."""
        terms, prov, seen = [], {}, set()
        for text, source, weight, facet, observed, pid in specs:
            f = fold(text)
            if not f or f in seen:
                if f in seen and prov.get(f):
                    prov[f]["facets"] = sorted(set(prov[f]["facets"]) | ({facet} if facet else set()))
                continue
            got = S.compile_terms([text], self.q.d_lang, source, weight, self.o)
            if not got:
                continue
            seen.add(f)
            terms += got
            prov[f] = {"text": text, "source": source, "weight": weight, "facets": [facet] if facet else [],
                       "observed": observed, "phrase_id": pid}
        if not self.a.flags.get("no_morph"):
            terms += C.morph_terms(list(terms), self.q.d_lang, self.o)
        return terms, prov

    def lexical(self, reports, docs, terms, kwin, entities, heads):
        hits, by_id = S.search(reports, docs, terms)
        C.align_morph_idf(terms)
        wins, _wms = S.windows(reports, hits, by_id, terms, kwin, self.o)
        wins = C.name_bonus(wins, entities, heads, self.o)
        return hits, wins

    # -- the question ---------------------------------------------------------------------------------------------
    def run(self):
        o, q, fl, a = self.o, self.q, self.a.flags, self.a
        t_all = time.perf_counter()
        trace = {"variant": a.variant, "versions": {"approach": APPROACH_VERSION, "prompts": PROMPT_VERSION},
                 "params": dict(o), "flags": dict(fl), "stages": {}, "degraded_mode": "none"}
        st = trace["stages"]
        degraded, why = "none", None

        # 0. the question channels (independent of planning; may have run already for the card hook)
        t0 = time.perf_counter()
        prep = a.prepare(q, self.index)
        pool, given, qhits = prep["pool"], prep["given"], prep["hits"]
        self.stage("question", t0)
        st["question"] = {"diagnostics": qhits.diagnostics, "pool": len(pool), "hits": len(qhits),
                          "top": [[h.doc_id, h.chunk, round(h.rrf, 5), h.ranks.get("bm25"), h.ranks.get("dense")]
                                  for h in qhits[:12]], "ms": prep["ms"]}

        # 1. call 1: interpret
        t0 = time.perf_counter()
        interp, problems, how, out = None, [], None, ""
        try:
            r = self.chat(PURPOSE_INTERPRET, P.interpret_messages(q.text, q.d_lang, o), fmt=P.INTERPRET_SCHEMA,
                          num_predict=int(o["interp_num_predict"]))
            out = r.text or ""
            interp, problems, how = P.validate_interpretation(out, o)
            if interp["empty"]:
                degraded, why = "empty_plan", "the interpretation names no facet and no entity"
        except BudgetExceeded:
            raise
        except Exception as e:  # noqa: BLE001 - a failed planner call must not remove the question channels
            interp, _p, _h = P.validate_interpretation("", o)
            degraded, why, how = "empty_plan", "interpret call failed: %s: %s" % (type(e).__name__, str(e)[:160]), "error"
        facets, entities = interp["facets"], interp["entities"]
        periods = C.expand_periods(interp["periods"])
        self.stage("interpret", t0)
        st["interpret"] = {"interp": interp, "periods_expanded": periods, "problems": problems[:20], "parse": how,
                           "model_output": out[:800]}

        # 2. routing: the question-only shortlist, refined by the interpretation (forced explicit matches)
        t0 = time.perf_counter()
        shortlist, rinfo = list(prep["shortlist_q"]), {"routed": prep["routing"], "stage_q": prep["route_q"]}
        if prep["routing"] and degraded == "none":
            kw = {"keywords": [t for f in facets for t in f["terms"]] + ([q.text] if q.q_lang == q.d_lang else []),
                  "names": entities, "years": periods}
            forced = C.explicit_matches(self.con, pool, entities, periods)
            shortlist, rinfo["stage_interp"] = C.route(self.con, pool, qhits, kw, q.d_lang, o, forced=forced)
        rinfo.update(mode="open_corpus" if fl.get("open_corpus") else "given", pool=len(pool), shortlist=shortlist,
                     shortlist_q=prep["shortlist_q"])
        if fl.get("open_corpus") and given:          # reported against the benchmark's list, never used to decide
            rinfo["recall_vs_given"] = round(len(set(given) & set(shortlist)) / len(set(given)), 4)
        self.stage("route", t0)
        st["route"] = rinfo
        heads = C.issuer_heads(self.con, list(dict.fromkeys(pool)))

        # 3. the feedback vocabulary (from the question's own retrieved evidence)
        vocab, pick = [], None
        lex_on = not fl.get("no_lexical")
        if degraded == "none" and lex_on:
            t0 = time.perf_counter()
            vhits = [h for h in qhits if h.doc_id in set(shortlist)] + [h for h in qhits if h.doc_id not in set(shortlist)]
            vocab, vinfo = C.feedback_vocabulary(vhits, self.index.corpus, facets, q.text, q.d_lang, o, con=self.con,
                                                 years=periods)
            self.stage("vocab", t0)
            st["vocab"] = dict(vinfo, phrases=[{k: v[k] for k in ("id", "text", "kind", "facets", "source", "n_sources")}
                                               for v in vocab])
            if not vocab:
                degraded, why = "no_shortlist", "no usable phrase in the retrieved evidence"

        # 4. call 2: phrase IDs per facet
        c2 = {"ran": False}
        if degraded == "none" and lex_on and not fl.get("no_pick"):
            t0 = time.perf_counter()
            if not self.budget.can(1, reserve=1):
                c2["skipped"] = "call cap (the answer call is reserved)"
                degraded, why = "pick_failed", c2["skipped"]
            else:
                try:
                    r = self.chat(PURPOSE_PICK, P.pick_messages(q.text, facets, vocab, q.d_lang, o), fmt=P.PICK_SCHEMA,
                                  num_predict=int(o["pick_num_predict"]))
                    pick, pproblems, phow = P.validate_pick(r.text, len(facets), vocab, o)
                    c2.update(ran=True, parse=phow, shape=pick["shape"], problems=pproblems[:20],
                              by_facet=pick["by_facet"], done_reason=r.done_reason, model_output=(r.text or "")[:800])
                    if phow == "failed" or pick["shape"] == "none":
                        degraded, why = "pick_failed", "call 2 reply unusable (%s)" % phow
                        pick = None
                    elif not any(s["phrases"] or s["add"] for s in pick["by_facet"].values()):
                        degraded, why = "pick_empty", "call 2 picked no phrase and added no term"
                        pick = None
                except Exception as e:  # noqa: BLE001 - keep keyword + question retrieval
                    c2.update(ran=True, error="%s: %s" % (type(e).__name__, str(e)[:160]))
                    degraded, why = "pick_failed", "call 2 failed: %s" % type(e).__name__
            self.stage("pick", t0)
        elif fl.get("no_pick"):
            c2["skipped"] = "no_pick"
        st["pick"] = c2

        # 5. lexical searches over the shortlist: picked phrases (observed), added terms (not observed), facet terms
        t0 = time.perf_counter()
        specs = []
        byid = {v["id"]: v for v in vocab}
        facet_terms = [list(f["terms"]) for f in facets]
        if pick is not None:
            for fi, slot in sorted(pick["by_facet"].items()):
                for pid in slot["phrases"]:
                    specs.append((byid[pid]["text"], "picked", float(o["w_picked"]), fi + 1, True, pid))
                    facet_terms[fi].append(byid[pid]["text"])
                for t in slot["add"]:
                    specs.append((t, "added", float(o["w_added"]), fi + 1, False, None))
                    facet_terms[fi].append(t)
            kw_weight = float(o["w_keyword"])
        else:
            kw_weight = 1.0                  # the facet terms ARE the lexical channel (no_pick, or a fallback)
        if degraded != "empty_plan" and lex_on:
            for i, f in enumerate(facets):
                for t in f["terms"]:
                    specs.append((t, "keyword", kw_weight, i + 1, False, None))
        terms, prov = self.compile(specs) if specs else ([], {})
        lwins, lhits = [], {}
        if terms:
            reports = S.Reports(self.con)
            kwin = {"years": periods or P.years_of(q.text), "names": [], "answer": interp["answer"]}
            lhits, lwins = self.lexical(reports, shortlist, terms, kwin, entities, heads)
        self.stage("lexical", t0)
        st["lexical"] = {"terms": [dict(t.trace(), provenance=prov.get(fold(t.text))) for t in terms],
                         "hit_units": len(lhits), "windows": len(lwins),
                         "top": [{k: w.get(k) for k in ("doc_id", "hit", "score", "terms", "name_bonus")}
                                 for w in lwins[:10]]}

        # 6. fusion: question BM25 / dense (whole pool: the escape channel) + grounded lexical, over regions
        t0 = time.perf_counter()
        keep = None if not fl.get("no_escape") else set(shortlist)
        channels = {"bm25": _channel(qhits, "bm25", keep), "dense": _channel(qhits, "dense", keep)}
        if q.q_lang != q.d_lang and degraded != "empty_plan":
            # cross-language: the question's own BM25 cannot match; the interpreted terms (in the reports' language)
            # get a BM25 channel of their own, recorded as such
            qt = " ".join([t for f in facets for t in f["terms"]] + entities + periods)
            if qt.strip():
                th = self.hx.search(qt, doc_ids=pool, k=int(o["hybrid_depth"]), channels=("bm25",))
                channels["bm25_terms"] = _channel(th, "bm25", keep)
        if lwins:
            channels["lex"] = [{"doc_id": w["doc_id"], "span": w.get("span"), "unit_ids": list(w["unit_ids"]),
                                "section_id": w["section_id"], "hit": w["hit"]} for w in lwins[: int(o["lex_windows"])]]
        channels = {k: v for k, v in channels.items() if v}
        regions, finfo = C.fuse_regions(channels, self.index.corpus, o)
        cands = C.block_candidates(regions, int(o["fused_candidates"]))
        short_set = set(shortlist)
        finfo["escape_regions"] = sum(1 for r in regions if r["doc_id"] not in short_set)
        finfo["top"] = [[r["doc_id"], r["rid"], round(r["rrf"], 5), r["ranks"]] for r in regions[:12]]
        self.stage("fuse", t0)
        st["fuse"] = finfo

        # 7. requirements (from the question's interpretation) and evidence blocks with reservation
        t0 = time.perf_counter()
        if degraded == "empty_plan":
            qyears = C.expand_periods(P.years_of(q.text))
            reqs, rqinfo = C.requirements([], [], qyears, [], o)
            rqinfo["from"] = "question years (empty plan)"
        else:
            reqs, rqinfo = C.requirements(facets, entities, periods, facet_terms, o)
            rqinfo["from"] = "interpretation"
        budget = int(self.cfg.evidence["max_chars"])
        items, brep = build_blocks(cands, self.index.corpus, budget=budget, requirements=reqs, question=q.text,
                                   cfg=self.cfg)
        self.stage("blocks", t0)

        # bounded follow-up: one retrieval per missing requirement (its own wording + its source phrases), over the
        # whole pool (so a wrong shortlist is recoverable), plus a lexical pass over reports that explicitly match it
        t0 = time.perf_counter()
        fu = []
        if reqs and not fl.get("no_followup"):
            status = brep.get("requirements") or {}
            todo = [r for r in reqs if (status.get(r["id"]) or {}).get("status") == "missing"]
            todo += [r for r in reqs if (status.get(r["id"]) or {}).get("status") == "partial"]
            extra = []
            have = {(c["doc_id"], tuple(c["span"]) if c.get("span") else tuple(c["unit_ids"])) for c in cands}
            for r in todo[: int(o["followup_max"])]:
                words = list((r.get("terms") or [])[:4]) + ([r["entity"]] if r.get("entity") else []) + \
                    ([r["period"]] if r.get("period") else [])
                text = " ".join(words).strip() or q.text
                fh = self.hx.search(text, doc_ids=pool, k=int(o["followup_k"]), depth=int(o["hybrid_depth"]),
                                    client=a.client)
                new = [dict(_hit_cand(h), ranks=dict(h.ranks), rrf=h.rrf, channel="followup") for h in fh]
                tdocs = C.explicit_matches(self.con, pool, [r["entity"]] if r.get("entity") else entities,
                                           [r["period"]] if r.get("period") else [])
                tdocs = [d for d in tdocs if d not in short_set][:3]
                if (r.get("terms") or []) and (tdocs or shortlist):
                    fterms, _fp = self.compile([(t, "followup", 1.0, None, False, None) for t in r["terms"][:6]])
                    if fterms:
                        _h, fw = self.lexical(S.Reports(self.con), list(dict.fromkeys(tdocs + shortlist)), fterms,
                                              {"years": [r["period"]] if r.get("period") else periods, "names": [],
                                               "answer": interp["answer"]}, entities, heads)
                        new += [{"doc_id": w["doc_id"], "span": w.get("span"), "unit_ids": list(w["unit_ids"]),
                                 "section_id": w["section_id"], "channel": "followup_lex"}
                                for w in fw[: int(o["followup_k"])]]
                added = 0
                for c in new:
                    k = (c["doc_id"], tuple(c["span"]) if c.get("span") else tuple(c["unit_ids"]))
                    if k not in have:
                        have.add(k)
                        extra.append(c)
                        added += 1
                fu.append({"requirement": r["id"], "query": text[:200], "new_candidates": added,
                           "targeted_docs": tdocs, "docs": sorted({c["doc_id"] for c in new})})
            if extra:
                items, brep = build_blocks(cands + extra, self.index.corpus, budget=budget, requirements=reqs,
                                           question=q.text, cfg=self.cfg)
        self.stage("followup", t0)
        for it in items:                              # the variants-v2 section expansion must preserve these
            res = (it.meta or {}).get("reserved_for") or []
            if res:
                it.meta["reserved"] = True
                it.meta["requirement"] = ",".join(res)
        st["requirements"] = {"list": reqs, "info": rqinfo, "status": brep.get("requirements"), "followup": fu}
        st["blocks"] = brep

        # 8. pack and answer (the shared prompt of the question's answer format, the shared budget)
        t0 = time.perf_counter()
        evidence = a.final_pack(items, self.index, q)
        self.stage("pack", t0)
        t0 = time.perf_counter()
        msgs = build_answer_messages(q, evidence, self.cfg)
        r = self.chat(PURPOSE_ANSWER, msgs, num_predict=answer_num_predict(q, self.cfg))
        self.stage("answer", t0)
        ev_docs = list(dict.fromkeys(it.doc_id for it in evidence.items))
        trace.update(degraded_mode=degraded, degraded_reason=why, calls=self.calls,
                     timing=dict(self.timing, total_ms=_ms(t_all)), stats=self.stats,
                     evidence={"n_offered": evidence.n_offered, "n_included": evidence.n_included,
                               "chars": evidence.chars, "docs": ev_docs,
                               "escape_docs": [d for d in ev_docs if d not in short_set],
                               "given_docs_in_evidence": (round(len(set(given) & set(ev_docs)) / len(set(given)), 4)
                                                          if fl.get("open_corpus") and given else None)},
                     failures={"answer_empty": not (r.text or "").strip(), "answer_truncated": r.done_reason == "length",
                               "interpret_problems": len(problems), "pick_problems": len(c2.get("problems") or []),
                               "zero_hit_terms": sum(1 for t in terms if t.df == 0), "n_terms": len(terms)})
        return AnswerResult(answer_text=(r.text or "").strip(), evidence=evidence, retrieved=items, trace=trace)
