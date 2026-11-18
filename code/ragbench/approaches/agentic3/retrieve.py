"""Stage 2 and stage 4 for one question: the knowledge base, read through `kb/query.py`.

Nothing here decides the path (that is `route.py`) and nothing here calls a chat model. The one model call this
module can make is an EMBEDDING call, and only on the escalated path: `embed_pending` fills `passages.vec` for a
document that has never been embedded (spec 4.2 stage 8, "the first question that escalates"). It is logged as
stage "lazy-index" with phase "query" - v2's arrangement, kept so `eval/sanity.py` still finds every embed call
of a record among that record's own ledger lines - and it is excluded from the 60 s query clock and reported
separately, because it is preprocessing that happens to be paid late.

    docs, dinfo   = target_documents(con, scope_ids, plan, opts)
    cands, info   = retrieve_all(con, plan, doc_ids, d_lang=..., opts=..., facets=facets, reader=reader)
    ...                                          # route.decide reads `cands` and `info`
    vcands, vinfo = vector_candidates(con, doc_ids, qvec, terms_vec, ...)   # escalated only
    merged, minfo = merge(cands + vcands, doc_ids, facets, opts, cap=...)

FACETS (spec 6 stage 1 and 4). A facet is one group of the plan's terms. The base pass searches with every term;
each facet then searches with its own terms alone, so a question that needs several sections cannot have all its
candidates taken by the strongest facet. The passes share one `DocReader`, so the units of a document are read
once. In the merge every facet keeps at least `facet_min_slots` slots before the ordinary round robin over
documents fills the rest. With no facets the selection is byte-identical to `kb/query.py::select_candidates`.

SCALES. A fact row's score and a cosine similarity are not the same quantity. A vector hit is priced in the fact
score's own units (`vec_score_scale * similarity`) so that one `rank_candidates` call can dedupe across both
paths - a passage covering the same window as a fact row merges into it and keeps both `src` tags - but pricing
is not enough to make the second path exist: the fact score carries constant terms a cosine has no counterpart
for, so a full fact pool outranks every passage and takes the whole cap. The vector path therefore keeps
`vec_min_slots` reserved slots in `merge`, the same way a facet does. The database stays first (the reservation
is a minority of the cap); vectors stop being silently absent.

FAIRNESS. Every query here is restricted to the documents the question's corpus set holds, which is what
`Approach.docs_in_scope` returns. The knowledge base is shared and larger than any one benchmark set, so a query
that forgot the restriction would read documents the other approaches cannot see. `gap_doc_ids` applies the same
restriction to `reconcile_sql.gap_queries`, whose own SQL searches every visible document by design.
"""
import time

from ...kb import ingest as I
from ...kb import query as Q
from ...kb.schema import connect, ensure_schema
from ...normalize import normalize
from ...util import sha1_text
from ..agentic2 import tree as T
from .reconcile_sql import GAP_FACTS_SQL

FACET_KEY = "facet"


def _ms(t0):
    return round((time.perf_counter() - t0) * 1000.0, 1)


# ---------------------------------------------------------------------------------------------------------------
# connection and documents
# ---------------------------------------------------------------------------------------------------------------

def open_kb(cfg, path=None, read_only=False):
    """The shared knowledge base, schema ensured (idempotent). Write access is needed for write-back and for the
    lazy vectors of the escalated path."""
    con = connect(path or I.kb_path(cfg), read_only=read_only)
    if not read_only:
        ensure_schema(con, cfg)
    return con


def target_documents(con, scope_ids, plan, opts, *, question="", selection=True):
    """Stage 1's document choice, over the documents of THIS corpus set only. -> (metas, info)."""
    period = plan["period"] if any(p is not None for p in (plan.get("period") or (None, None))) else None
    return Q.select_documents(con, doc_scope=list(scope_ids), companies=plan.get("companies"), period=period,
                              kind=plan.get("kind") or "single", question=question,
                              max_docs=int(opts["max_target_docs"]), selection=selection)


def missing_documents(con, scope_ids):
    """Doc ids of the corpus set that the knowledge base cannot answer from (absent, or not `ready`)."""
    have = {m["doc_id"] for m in Q.document_metas(con, doc_ids=list(scope_ids))}
    return [d for d in scope_ids if d not in have]


# ---------------------------------------------------------------------------------------------------------------
# facets
# ---------------------------------------------------------------------------------------------------------------

def facet_plans(plan, opts):
    """-> [(name, plan for that facet)] from `plan["facets"]`, each facet's terms alone."""
    out = []
    for i, f in enumerate(plan.get("facets") or [], 1):
        terms = f.get("terms") or {}
        if not (terms.get("fa") or terms.get("en")):
            continue
        out.append(("f%d" % i, dict(plan, terms=terms, facets=[])))
        if len(out) >= int(opts["max_facets"]):
            break
    return out


def tag_facet(cands, name):
    for c in cands:
        c.features[FACET_KEY] = sorted(set(c.features.get(FACET_KEY) or []) | {name}) if name else []
    return cands


def facets_of(cand):
    return set((cand.features or {}).get(FACET_KEY) or [])


# ---------------------------------------------------------------------------------------------------------------
# stage 2 + 4: the database pass (base plan, then one pass per facet)
# ---------------------------------------------------------------------------------------------------------------

def retrieve_all(con, plan, doc_ids, *, d_lang="fa", opts, trigram=True, warm=False, tree=True, units=None,
                 reader=None, facets=(), cap=None):
    """Every database path for one question, base terms then each facet. -> (candidates, info).

    The candidates are NOT capped here: `merge` decides the final selection, because the facet reservation needs
    the whole pool. `info["passes"]` holds one `kb/query.py::retrieve` info per pass.
    """
    t0 = time.perf_counter()
    reader = reader or Q.DocReader(con)
    passes = []
    cands, base_info = Q.retrieve(con, plan, doc_ids, d_lang=d_lang, opts=opts, trigram=trigram, warm=warm,
                                  tree=tree, units=units, reader=reader,
                                  cap=int(cap if cap is not None else opts["max_candidates_multi"]))
    tag_facet(cands, None)
    passes.append(dict(base_info, pass_name="base"))
    n_base = len(cands)
    for name, fplan in facets:
        fc, finfo = Q.retrieve(con, fplan, doc_ids, d_lang=d_lang, opts=opts, trigram=trigram, warm=False,
                               tree=tree, units=units, reader=reader, cap=int(opts["facet_cap"]))
        tag_facet(fc, name)
        cands += fc
        passes.append(dict(finfo, pass_name=name))
    info = dict(base_info)
    info.update(passes=passes, facets=[n for n, _p in facets], base_candidates=n_base,
                facet_candidates=len(cands) - n_base, pool=len(cands), ms=_ms(t0))
    return cands, info


def vector_candidates(con, doc_ids, qvec, terms_vec, *, plan, d_lang="fa", opts, reader, tree=True):
    """The second path: stored passage vectors -> candidates priced in the fact score's units. -> (cands, info)."""
    t0 = time.perf_counter()
    unit_kind = plan.get("unit") or "count"
    ctx = T.search_context(plan, d_lang)
    hits, info = Q.vector_search(con, doc_ids, qvec, terms_vec, opts=opts)
    scale = float(opts["vec_score_scale"])
    out = []
    for row, sim in hits:
        for c in Q.expand_passage(reader, row, ctx, opts, unit_kind, tree=tree):
            c.score = round(scale * float(sim), 4)
            c.features = {"src": ["vector"], "vec": round(float(sim), 4), "pid": row["passage_id"],
                          FACET_KEY: []}
            out.append(c)
    info = {k: v for k, v in info.items() if k != "unit_sim"}
    info.update(hits=len(hits), candidates=len(out), scale=scale, ms=_ms(t0))
    return out, info


# ---------------------------------------------------------------------------------------------------------------
# merge: dedupe, rank, facet slots, cap
# ---------------------------------------------------------------------------------------------------------------

def absorb_facets(ranked, pool):
    """A facet's candidate is usually the SAME window the base pass already found, and `rank_candidates` keeps one
    of them. The surviving candidate inherits the facet tags of every candidate that sits inside it, so the facet
    reservation asks "which kept candidates carry this facet's vocabulary" rather than "which objects survived"."""
    for k in ranked:
        tags = set(facets_of(k))
        for c in pool:
            if c is not k and c.doc_id == k.doc_id and c.section_id == k.section_id \
                    and set(c.unit_ids) <= set(k.unit_ids):
                tags |= facets_of(c)
        k.features[FACET_KEY] = sorted(tags)
    return ranked


def is_vector(cand):
    return "vector" in ((cand.features or {}).get("src") or [])


def merge(cands, doc_order, facets, opts, cap):
    """-> (selected, info). Every facet keeps `facet_min_slots` slots and the vector path keeps `vec_min_slots`,
    then the round robin over documents fills the rest; `cid`s are assigned in the final order, which is also the
    packing order of the evidence.

    The vector reservation is what makes the second path exist. A cosine and a fact score are not the same
    quantity, and the fact score is the larger one: even after `vec_score_scale` a passage at bge-m3's real
    similarity is worth less than the weakest fact row that matched one generic term. Measured on dev in round 1,
    over the 11 questions whose route asked for vectors, the best passage of the question scored below the WORST
    fact row the merge still selected in 9 of them - so a full fact pool does not rank passages low, it excludes
    them. Reserving slots keeps the database first (the reservation is a minority of the cap) without pretending
    the two numbers can be compared.
    """
    t0 = time.perf_counter()
    ranked = absorb_facets(Q.rank_candidates(cands), cands)
    names = [n for n, _p in facets]
    slots = int(opts["facet_min_slots"]) if names else 0
    picked, taken = [], set()
    for _round in range(slots):
        for name in names:
            if len(picked) >= cap:
                break
            c = next((x for x in ranked if id(x) not in taken and name in facets_of(x)), None)
            if c is not None:
                picked.append(c)
                taken.add(id(c))
    kept_vec = sum(1 for x in picked if is_vector(x))
    for c in ranked:
        if len(picked) >= cap or kept_vec >= int(opts["vec_min_slots"]):
            break
        if id(c) not in taken and is_vector(c):
            picked.append(c)
            taken.add(id(c))
            kept_vec += 1
    rest = [c for c in ranked if id(c) not in taken]
    tail = Q.select_candidates(rest, list(doc_order), max(0, cap - len(picked)))
    out = picked + tail
    for i, c in enumerate(out, 1):
        c.cid = "c%d" % i
        c.verdict = None
    info = {"pre_cap": len(ranked), "pool": len(cands), "selected": len(out), "cap": cap,
            "facet_slots": {n: sum(1 for c in out if n in facets_of(c)) for n in names},
            "vec_slots": sum(1 for c in out if is_vector(c)), "ms": _ms(t0)}
    return out, info


# ---------------------------------------------------------------------------------------------------------------
# gap fill (stage 6): the same document restriction as every other query
# ---------------------------------------------------------------------------------------------------------------

def gap_doc_ids(gap, scope_ids):
    """`reconcile_sql.gap_queries` searches every visible document; a benchmark question may only see its own
    corpus set. -> (doc ids inside the scope, how many were dropped)."""
    keep = [d for d in gap["doc_ids"] if d in set(scope_ids)]
    return keep, len(gap["doc_ids"]) - len(keep)


def scope_gap(gap, scope_ids):
    """The gap entry restricted to `scope_ids`, with its SQL and parameters rebuilt. -> (gap, dropped)."""
    keep, dropped = gap_doc_ids(gap, scope_ids)
    params = {k: v for k, v in gap["params"].items() if not k.startswith("d")}
    params.update({"d%d" % k: d for k, d in enumerate(keep)})
    out = dict(gap, doc_ids=keep, docs=[d for d in gap["docs"] if d["doc_id"] in set(keep)], params=params,
               sql=GAP_FACTS_SQL % (", ".join(":d%d" % k for k in range(len(keep))) or "NULL"),
               runnable=bool(keep and gap["pattern"]), dropped_out_of_scope=dropped)
    return out, dropped


def plan_gap_pattern(plan, d_lang, opts):
    """A gap regex from the PLAN's wording, for the case `reconcile_sql.gap_queries` cannot serve: it builds its
    pattern from the labels of the rows already verified, and a question where the verifier accepted nothing has
    no such wording. Same builder (`label_pattern`), so the pattern is folded exactly the same way."""
    from .reconcile_sql import label_pattern
    terms = plan.get("terms") or {}
    phrases = [plan.get("metric") or ""] + list(terms.get(d_lang) or []) + \
        list(terms.get("en" if d_lang == "fa" else "fa") or [])
    pats = []
    for p in phrases[: int(opts["max_terms_per_lang"])]:
        pat = label_pattern(p)
        if pat and pat not in pats:
            pats.append(pat)
    return "|".join(pats)


def gap_candidates(reader, rows, plan, d_lang, opts, *, tree=True, round_no=2):
    """Fact rows returned by a gap query -> candidates the verifier can read."""
    unit_kind = plan.get("unit") or "count"
    ctx = T.search_context(plan, d_lang)
    out = []
    for r in rows:
        c = Q.expand_unit(reader, r["doc_id"], r["unit_id"], r["value_text"], ctx, opts, unit_kind, tree=tree,
                          round_no=round_no)
        if c is None:
            continue
        c.score = float(r.get("score") or 0.0)
        c.features = {"src": ["gap"], FACET_KEY: [],
                      "fact": {"fact_id": r["fact_id"], "label": r["label"], "value": r["value_text"],
                               "unit": r.get("unit_norm"), "unit_kind": r.get("unit_kind"), "year": r.get("year"),
                               "def_class": r.get("def_class"), "source_kind": None}}
        out.append(c)
    return out


# ---------------------------------------------------------------------------------------------------------------
# vectors: the query vectors, and the lazy first touch of a document
# ---------------------------------------------------------------------------------------------------------------

def terms_text(plan, d_lang):
    """The second query vector's text: entity, metric and every term, the document's language first (v2's rule)."""
    terms = plan.get("terms") or {}
    other = "en" if d_lang == "fa" else "fa"
    parts = [plan.get("entity") or "", plan.get("metric") or ""] + list(terms.get(d_lang) or []) + \
        list(terms.get(other) or [])
    return " ; ".join(p for p in parts if p).strip()


def query_vectors(lazy, client, question, plan, d_lang, opts):
    """-> (qvec, terms_vec | None, info). One embed call for both texts, as v2 makes it."""
    texts = [normalize(question)]
    tt = terms_text(plan, d_lang) if opts["vec_terms_query"] else ""
    if tt:
        texts.append(tt)
    vecs, info = lazy.query_vectors(texts, client, purpose="query-embed")
    return vecs[0], (vecs[1] if len(vecs) > 1 else None), dict(info, terms_text=tt)


def embed_pending(con, doc_ids, cfg, client, lazy, *, purpose="lazy-index-embed", log=None):
    """Fill `passages.vec` for documents that have none yet. -> stats.

    The vectors go through the SAME shared embedding cache B3 and v2 fill (`lazy.text_key`), so a passage another
    approach has already embedded costs nothing, and they are committed per document, so a run killed in the
    middle of a corpus keeps every document it finished. `documents.status` becomes `vector_ready` when a document
    has no passage left without a vector; that is the only place this module changes a document's status.
    """
    import numpy as np
    t0 = time.perf_counter()
    rows = I.pending_passages(con, doc_ids)
    opts = I.passage_opts(cfg)
    model = cfg.models["embed"]
    bs = int(cfg.embedding["batch_size"])
    kv = lazy.kv()
    stats = {"passages": len(rows), "from_cache": 0, "embedded": 0, "calls": 0, "tokens": 0, "embed_ms": 0.0,
             "mismatched_sha1": 0, "docs": 0}
    by_doc = {}
    for r in rows:
        by_doc.setdefault(r["doc_id"], []).append(r)
    for doc_id, items in by_doc.items():
        with I.stage(con, doc_id, "embed") as st:
            texts = []
            for r in items:
                et = I.embed_text_of(r["title"], r["path"], r["text"], opts["embed_with_path"])
                if sha1_text(et) != r["embed_sha1"]:
                    stats["mismatched_sha1"] += 1
                texts.append(et)
            todo = [(i, t) for i, t in enumerate(texts) if kv.get_array("v:" + lazy.text_key(t)) is None]
            stats["from_cache"] += len(texts) - len(todo)
            for j in range(0, len(todo), bs):
                batch = todo[j:j + bs]
                er = client.embed([t for _, t in batch], purpose=purpose, batch_size=bs)
                for (_i, t), v in zip(batch, er.vectors):
                    key = lazy.text_key(t)
                    kv.put_array("v:" + key, np.asarray(v, dtype=np.float32))
                    kv.put("c:" + key, {"tokens": er.prompt_tokens / len(batch), "wall_ms": er.wall_ms / len(batch)})
                stats["calls"] += er.calls
                stats["tokens"] += er.prompt_tokens
                stats["embed_ms"] += er.wall_ms
                stats["embedded"] += len(batch)
            now = time.strftime("%Y-%m-%dT%H:%M:%S")
            for r, t in zip(items, texts):
                arr = kv.get_array("v:" + lazy.text_key(t))
                if arr is None:
                    continue
                con.execute("UPDATE passages SET vec = ?, dim = ?, model = ?, embedded_at = ? WHERE passage_id = ?",
                            (arr.astype(np.float32).tobytes(), int(arr.shape[0]), model, now, r["passage_id"]))
            con.execute("UPDATE documents SET status = 'vector_ready' WHERE doc_id = ? AND status = 'ready' "
                        "AND NOT EXISTS (SELECT 1 FROM passages WHERE doc_id = ? AND vec IS NULL)", (doc_id, doc_id))
            con.commit()
            st.rows = len(items)
            st.detail = {"embedded": len(todo), "cached": len(items) - len(todo)}
        stats["docs"] += 1
        if log:
            log({"doc_id": doc_id, "passages": len(items), "embedded": len(todo), "ms": st.ms})
    stats["embed_ms"] = round(stats["embed_ms"], 1)
    stats["wall_ms"] = _ms(t0)
    return stats


# ---------------------------------------------------------------------------------------------------------------
# ingestion, for _build_index
# ---------------------------------------------------------------------------------------------------------------

def ingest_corpus(con, corpus_set, cfg, *, force=False, log=None):
    """Stage 4-7 of the ingestion pipeline for every document of a corpus set. -> stats.

    This is what `_build_index` does, and it sees the corpus and never a question (fairness rule 2). It is
    `ingest.sync_sets` per Document rather than per set name, because a run may restrict the corpus
    (`--docs-from-questions`) and only the documents actually in the set may be loaded.
    """
    from ...corpus import extracted
    t0 = time.perf_counter()
    stats = {"documents": 0, "loaded": 0, "rebuilt": 0, "skipped": 0, "stages": {}, "rows": {}, "chars": 0,
             "per_doc": []}
    for doc in corpus_set.docs:
        st = extracted.read_status(doc.source) or {}
        res = I.ingest_document(con, doc, cfg, source_path=doc.source, source_sha256=st.get("sha256"),
                                quality="suspect" if st.get("verdict") == "suspect" else "ok", force=force)
        stats["documents"] += 1
        stats[res["action"]] = stats.get(res["action"], 0) + 1
        for k, v in (res.get("stages") or {}).items():
            stats["stages"][k] = round(stats["stages"].get(k, 0.0) + v, 1)
        for k, v in (res.get("rows") or {}).items():
            stats["rows"][k] = stats["rows"].get(k, 0) + v
        stats["chars"] += res.get("chars") or 0
        stats["per_doc"].append({"doc_id": res["doc_id"], "action": res["action"], "ms": res["ms"],
                                 "stages": res.get("stages") or {}})
        if log:
            log(res)
    stats["ingest_ms"] = _ms(t0)
    return stats
