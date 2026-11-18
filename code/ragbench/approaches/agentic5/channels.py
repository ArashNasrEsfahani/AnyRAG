"""Retrieval channels of AGENTIC5, behind one adapter.

THE SHARED FOUNDATION (built in parallel, 2026-09-24). When `ragbench.retrieval.hybrid` is importable AGENTIC5 uses

    HybridIndex.search(query_text, doc_ids=None, k=40, channels=("bm25", "dense"), query_vector=None) -> [Hit]
    Hit: doc_id, section_id, unit_ids, text, span, ranks {"bm25", "dense"}, scores, rrf

for the pooled question retrieval (routing and the escape channel) and for each requirement's own query. The index
is built at INDEX time (`RequirementAgent._build_index` -> `HybridIndex.open(corpus_set, cfg, embed=False)`: chunks
and BM25, vectors from the shared cache only; the prebuild CLI does the embedding) and only LOADED at query time
(`HybridIndex.load(corpus_set, cfg, client=...)`; the client may embed the QUESTION, never a chunk). The search's
own diagnostics (missing vectors in scope, a skipped dense channel) are copied into `channels.dense`. What happened
is recorded in the trace (`channels.hybrid`): `loaded`, `absent` (no foundation), `disabled` (flag no_hybrid),
`not_prebuilt: ...`, or `factory` (`opts["hybrid_factory"]`, tests and wiring).

THE FALLBACK (no foundation, or flag no_hybrid) is AGENTIC3's own path over the knowledge base: typed facts and units
(`kb/query.py::retrieve`) and the stored passage vectors (`kb/query.py::vector_search`). Two differences from
AGENTIC3, both for cold and isolated experiments:

  * NO QUERY-TIME DOCUMENT EMBEDDING. AGENTIC3's escalated path calls `retrieve.embed_pending`, which embeds every
    passage of a targeted document that has no vector yet, off the query clock (measured: ~111 s for a median
    English report). AGENTIC5 never calls it. Documents without prebuilt vectors are skipped by the dense channel
    and listed in `dense.missing_docs`; when no document in scope has vectors the channel is skipped entirely and
    the question is not even embedded (`dense.skipped = "no_prebuilt_vectors"`).
  * READ-ONLY. The session's connection is opened with `mode=ro` + `query_only` unless write-back was asked for.
"""
import time

from ...kb import query as Q
from ..agentic import planner as P
from ..agentic import search as S
from ..agentic3 import retrieve as RT

REQ_KEY = "req"


def _ms(t0):
    return round((time.perf_counter() - t0) * 1000.0, 1)


def open_hybrid(ap, index, opts):
    """-> (hybrid object | None, diagnostic string). Cached per corpus set on the approach.

    The query side only LOADS a prebuilt index (`HybridIndex.load`, which also re-reads the shared vector cache):
    building one means chunking and BM25 over the whole set, which belongs to index time (`_build_index` does it with
    embed=False, the prebuild CLI does the embeddings). `opts["hybrid_factory"]` (a callable (cfg, index) -> object
    with `.search`) overrides, for tests and wiring."""
    if not opts.get("hybrid", True):
        return None, "disabled"
    cache = ap.__dict__.setdefault("_hybrid_cache", {})
    key = getattr(index, "corpus_set", None) or id(index)
    if key in cache:
        return cache[key]
    factory = opts.get("hybrid_factory")
    if factory is not None:
        try:
            h = factory(ap.cfg, index)
            cache[key] = (h, "factory") if h is not None and hasattr(h, "search") else (None, "factory_returned_none")
        except Exception as e:  # noqa: BLE001
            cache[key] = (None, "factory_failed: %s" % str(e)[:200])
        return cache[key]
    try:
        from ...retrieval.hybrid import HybridIndex
    except ImportError:
        cache[key] = (None, "absent")
        return cache[key]
    try:
        cache[key] = (HybridIndex.load(index.corpus, ap.cfg, client=ap.client), "loaded")
    except Exception as e:  # noqa: BLE001  (absent or stale index: the KB path, recorded)
        cache[key] = (None, "not_prebuilt: %s" % str(e)[:200])
    return cache[key]


def prebuild_hybrid(ap, corpus_set):
    """Index time: the hybrid index over the corpus set (chunks + BM25; vectors from the shared cache only, never
    embedded here). -> status dict or the reason it was not built."""
    if not ap.opts.get("hybrid", True):
        return {"skipped": "disabled"}
    try:
        from ...retrieval.hybrid import HybridIndex
    except ImportError:
        return {"skipped": "absent"}
    try:
        return HybridIndex.open(corpus_set, ap.cfg, client=None, embed=False).status()
    except Exception as e:  # noqa: BLE001
        return {"skipped": "build_failed: %s" % str(e)[:200]}


def hit_candidate(hit, source, round_no=1):
    c = S.Candidate(hit.doc_id, getattr(hit, "section_id", None), list(getattr(hit, "unit_ids", None) or []),
                    hit.text, list(hit.span) if getattr(hit, "span", None) else None, "passage",
                    list(getattr(hit, "unit_ids", None) or []), round_no=round_no)
    ranks = dict(getattr(hit, "ranks", None) or {})
    c.score = float(getattr(hit, "rrf", 0.0) or 0.0)
    c.features = {"src": [source], "channels": sorted(k for k, v in ranks.items() if v is not None),
                  "ranks": ranks, "rrf": c.score, "facet": []}
    return c


class Channels:
    """One question's retrieval. `diag` goes to the trace as-is."""

    def __init__(self, sess):
        self.s = sess
        self.o = sess.o
        self.con = sess.con
        self.reader = sess.reader
        self.hybrid, how = open_hybrid(sess.a, sess.index, sess.o) if not sess.a.flags.get("no_hybrid") \
            else (None, "disabled")
        self.diag = {"hybrid": how, "dense": {}, "calls": []}
        self._qvec = None
        self._vec_docs = None

    # -- dense: prebuilt vectors only --------------------------------------------------------------------------
    def docs_with_vectors(self, doc_ids):
        if self._vec_docs is None:
            self._vec_docs = {}
        todo = [d for d in doc_ids if d not in self._vec_docs]
        for i in range(0, len(todo), 400):
            part = todo[i:i + 400]
            have = {r["doc_id"] for r in self.con.execute(
                "SELECT DISTINCT doc_id FROM passages WHERE vec IS NOT NULL AND doc_id IN (%s)"
                % ",".join("?" * len(part)), part).fetchall()}
            for d in part:
                self._vec_docs[d] = d in have
        return [d for d in doc_ids if self._vec_docs.get(d)]

    def query_vector(self, plan):
        """The QUESTION's vectors (one embed call, never a document); None when dense is off or has nothing to read."""
        if self._qvec is not None:
            return self._qvec
        if self.o.get("dense") == "off" or self.s.a.flags.get("no_dense") or not self.o.get("dense_query_embed"):
            self.diag["dense"]["skipped"] = "off"
            return None
        try:
            qvec, tvec, qinfo = RT.query_vectors(self.s.a.lazy, self.s.a.client, self.s.q.text, plan,
                                                 self.s.q.d_lang, self.o)
        except Exception as e:  # noqa: BLE001  (no embedding server: the dense channel is skipped, recorded)
            self.diag["dense"]["skipped"] = "query_embed_failed: %s" % str(e)[:160]
            self._qvec = False
            return None
        self.s.stats["query_embed_calls"] += qinfo.get("embed_calls", 0)
        self._qvec = (qvec, tvec)
        return self._qvec

    def dense(self, plan, doc_ids, *, top_k=None):
        d = self.diag["dense"]
        if self.o.get("dense") == "off" or self.s.a.flags.get("no_dense"):
            d["skipped"] = "off"
            return [], {"skipped": "off"}
        have = self.docs_with_vectors(doc_ids)
        missing = [x for x in doc_ids if x not in set(have)]
        d["missing_docs"] = sorted(set(d.get("missing_docs") or []) | set(missing))[:50]
        d["n_missing_docs"] = len(set(d.get("missing_docs") or []) | set(missing))
        d["query_time_document_embedding"] = False
        if not have:
            d.setdefault("skipped", "no_prebuilt_vectors")
            return [], {"skipped": "no_prebuilt_vectors", "missing": len(missing)}
        qv = self.query_vector(plan)
        if not qv:
            return [], {"skipped": d.get("skipped")}
        opts = dict(self.o)
        if top_k is not None:
            opts["vec_top_k_per_doc"] = opts["vec_top_k_single"] = int(top_k)
        cands, info = RT.vector_candidates(self.con, have, qv[0], qv[1], plan=plan, d_lang=self.s.q.d_lang,
                                           opts=opts, reader=self.reader, tree=True)
        for c in cands:
            c.features.setdefault("channels", ["dense"])
        return cands, dict(info, docs=len(have), missing=len(missing))

    # -- lexical: the knowledge base -----------------------------------------------------------------------------
    def lexical(self, plan, doc_ids, cap):
        if not doc_ids:
            return [], {"docs": 0}
        trigram = len(doc_ids) > 1 or bool(self.o["trigram_single_doc"])
        cands, info = Q.retrieve(self.con, plan, doc_ids, d_lang=self.s.q.d_lang, opts=self.o, trigram=trigram,
                                 warm=False, tree=True, reader=self.reader, cap=int(cap))
        for c in cands:
            c.features.setdefault("channels", ["lexical"])
        return cands, {k: info.get(k) for k in ("facts_rows", "units_rows", "sections_rows", "pre_cap", "selected",
                                                  "ms")}

    # -- hybrid: the shared foundation -------------------------------------------------------------------------
    def hybrid_search(self, text, doc_ids, k, plan=None, source="hybrid"):
        if self.hybrid is None:
            return [], {"skipped": self.diag["hybrid"]}
        chans = tuple(self.o.get("hybrid_channels") or ("bm25", "dense"))
        if self.s.a.flags.get("no_dense") or self.o.get("dense") == "off":
            chans = tuple(c for c in chans if c != "dense")
        t0 = time.perf_counter()
        scope = list(doc_ids) if doc_ids is not None else None
        client = self.s.a.client if "dense" in chans else None
        try:
            hits = self.hybrid.search(text, doc_ids=scope, k=int(k), channels=chans, client=client)
        except TypeError:           # a foundation without the client keyword
            hits = self.hybrid.search(text, doc_ids=scope, k=int(k), channels=chans)
        except Exception as e:  # noqa: BLE001
            if "dense" not in chans:
                self.diag.setdefault("hybrid_errors", []).append(str(e)[:160])
                return [], {"error": str(e)[:160]}
            try:
                hits = self.hybrid.search(text, doc_ids=scope, k=int(k), channels=("bm25",))
                self.diag.setdefault("hybrid_dense_failed", str(e)[:160])
            except Exception as e2:  # noqa: BLE001
                self.diag.setdefault("hybrid_errors", []).append(str(e2)[:160])
                return [], {"error": str(e2)[:160]}
        diag = dict(getattr(hits, "diagnostics", None) or {})
        dense = diag.get("dense") or {}
        if dense:
            d = self.diag["dense"]
            d["hybrid_missing_vectors_in_scope"] = max(int(d.get("hybrid_missing_vectors_in_scope") or 0),
                                                       int(dense.get("missing_vectors_in_scope") or 0))
            if dense.get("skipped"):
                d["hybrid_skipped"] = dense.get("skipped")
            d["query_time_document_embedding"] = False
        out = [hit_candidate(h, source) for h in hits or []]
        self.diag["calls"].append({"text": text[:120], "docs": None if doc_ids is None else len(doc_ids), "k": k,
                                   "hits": len(out), "ms": _ms(t0), "channels": list(chans),
                                   "warnings": list(diag.get("warnings") or [])[:3]})
        return out, {"hits": len(out), "ms": _ms(t0)}

    # -- the pooled question retrieval (routing + escape) ------------------------------------------------------
    def pooled(self, plan, scope_ids):
        """The question's own retrieval over the WHOLE allowed pool, independent of the plan's success. -> (ranked
        candidates, info). Each candidate keeps its per-channel rank (`features.pool_rank`)."""
        k = int(self.o["route_hits_k"])
        t0 = time.perf_counter()
        info = {}
        if self.hybrid is not None:
            cands, info["hybrid"] = self.hybrid_search(self.s.q.text, scope_ids, k, plan=plan, source="pool")
            for i, c in enumerate(cands, 1):
                c.features["pool_rank"] = {"hybrid": i}
            info["ms"] = _ms(t0)
            return cands, info
        qplan = question_plan(plan, self.s.q.text)
        lex, info["lexical"] = self.lexical(qplan, scope_ids, k)
        for i, c in enumerate(lex, 1):
            c.features["pool_rank"] = {"lexical": i}
            c.features["src"] = sorted(set(c.features.get("src") or []) | {"pool"})
        dense, info["dense"] = self.dense(plan, scope_ids, top_k=3)
        dense = sorted(dense, key=lambda c: -c.score)[:k]
        for i, c in enumerate(dense, 1):
            c.features["pool_rank"] = {"dense": i}
            c.features["src"] = sorted(set(c.features.get("src") or []) | {"pool"})
        info["ms"] = _ms(t0)
        return lex + dense, info

    # -- one requirement ---------------------------------------------------------------------------------------
    def for_requirement(self, req, rplan, doc_ids, cap, *, round_no=1, text=None):
        """The requirement's own retrieval over `doc_ids`: lexical (facts + units) with the requirement's wording,
        and the hybrid search (or stored vectors) with its text. -> (candidates tagged with the requirement, info)"""
        info = {}
        out = []
        lex, info["lexical"] = self.lexical(rplan, doc_ids, cap)
        out += lex
        rtext = text or requirement_text(req)
        if self.hybrid is not None:
            hy, info["hybrid"] = self.hybrid_search(rtext, doc_ids, cap, plan=rplan, source="hybrid")
            out += hy
        else:
            dn, info["dense"] = self.dense(rplan, doc_ids)
            out += dn
        for c in out:
            c.round = round_no
            c.features[REQ_KEY] = sorted(set(c.features.get(REQ_KEY) or []) | {req["id"]})
        info["candidates"] = len(out)
        return out, info


def question_plan(plan, question):
    """The plan's terms plus the question's own content words: the pooled lexical channel must not depend on the
    planner having produced usable terms (a weak planner's empty keyword list is the common failure)."""
    terms = {k: list(v) for k, v in (plan.get("terms") or {}).items()}
    for w in P.question_terms(question, plan.get("entity_terms") or []):
        lang = P._lang_of(w) or "en"
        terms.setdefault(lang, [])
        if w not in terms[lang]:
            terms[lang].append(w)
    return dict(plan, terms=terms, facets=[])


def requirement_text(req):
    terms = req.get("terms") or {}
    parts = [req.get("entity") or "", req.get("metric") or "", req.get("period_key") or ""]
    parts += list(terms.get("en") or [])[:2] + list(terms.get("fa") or [])[:2]
    return " ".join(p for p in parts if p).strip()
