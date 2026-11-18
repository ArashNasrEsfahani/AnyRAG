"""Stage 2 of AGENTIC5: recoverable routing over a pool of reports (97 in the multi-document design).

AGENTIC3 chose documents from the planner's company keys and, when that failed, fell back to the LATEST 12 documents
of the scope (`docmeta.select_docs`) - recency, not relevance. LEX keeps three routed reports. Neither recovers from
a wrong shortlist. Here (reviewer, "Both methods need routing that can recover from a wrong shortlist"):

  score      a document's score is its CARD score (the requirement vocabulary in its title / frequent terms /
             headings, `variants.document_card`) plus a CAPPED contribution of its best 1-2 chunk hits in the
             question's own pooled retrieval (reciprocal rank, k=60) - never the sum of all its chunk votes, which
             favours long reports - plus a bonus for an explicitly requested entity and for a requested period
             inside its years.
  shortlist  up to `shortlist_max` (8) documents. Explicit matches are RETAINED first: for every requirement, the
             best document of its entity whose years hold its period (or of its entity at all). The rest of the
             slots go to the best scores. With nothing to go on the shortlist is empty rather than recent.
  escape     the pooled candidates of documents OUTSIDE the shortlist stay available (`escape_max` of them), so a
             wrong shortlist still leaves the verifier something from the right report.
  expansion  a requirement that no shortlisted document can serve (after batch 1: a MISSING requirement) adds the
             best documents of its own entity / period from the pool (`expand_max_docs`), not a score-gap rule.
"""
import re

from ...normalize import fold, normalize
from .defaults import ROUTE_VERSION

RRF_K = 60.0


def _words(text):
    return {w for w in re.split(r"[\s‌\-_.,:;()«»\"'|/]+", fold(normalize(text or ""))) if len(w) >= 3}


def card_score(card, reqs):
    """Share of the requirements' metric term concepts found in the card (0..1), best over requirements."""
    cw = _words(card)
    cf = fold(normalize(card or ""))
    best = 0.0
    for r in reqs:
        terms = [t for lang in ("fa", "en") for t in (r.get("terms") or {}).get(lang) or []]
        if not terms:
            continue
        hit = sum(1 for t in terms if fold(normalize(t)) in cf or (_words(t) and _words(t) <= cw))
        best = max(best, hit / float(len(terms)))
    return best


def chunk_contribution(pool, best_n):
    """doc_id -> sum over its best `best_n` pooled hits of 1/(k+rank) * k (so one rank-1 hit is ~1). Capped by
    construction: a document with 200 hits counts its best two only."""
    per_doc = {}
    for c in pool:
        for _ch, rank in ((c.features or {}).get("pool_rank") or {}).items():
            per_doc.setdefault(c.doc_id, []).append(RRF_K / (RRF_K + float(rank)))
    return {d: round(sum(sorted(v, reverse=True)[:best_n]), 4) for d, v in per_doc.items()}


def entity_match(meta, req):
    return bool(meta.get("company")) and meta["company"] in (req.get("entity_keys") or [])


def period_match(meta, req):
    p = req.get("period")
    ys = meta.get("years")
    if not p or not ys:
        return None
    return ys[0] <= p["year"] <= ys[1] or ys[0] <= p["year"] + 1 <= ys[1]   # own report, or the next (restates it)


def score_documents(metas, cards, pool, matrix, opts):
    reqs = matrix["requirements"]
    chunk = chunk_contribution(pool, int(opts["route_best_chunks"]))
    rows = []
    for m in metas:
        d = m["doc_id"]
        cs = card_score(cards.get(d, m.get("title") or ""), reqs)
        ents = [r["id"] for r in reqs if entity_match(m, r)]
        pers = [r["id"] for r in reqs if period_match(m, r)]
        s = (float(opts["route_card_weight"]) * cs + float(opts["route_chunk_weight"]) * chunk.get(d, 0.0)
             + (float(opts["route_entity_bonus"]) if ents else 0.0)
             + (float(opts["route_period_bonus"]) if (ents and set(ents) & set(pers)) else 0.0))
        rows.append({"doc_id": d, "score": round(s, 4), "card": round(cs, 3), "chunks": chunk.get(d, 0.0),
                     "entity_reqs": ents, "period_reqs": pers})
    rows.sort(key=lambda r: (-r["score"], r["doc_id"]))
    return rows


def shortlist(metas, cards, pool, matrix, opts, cap=None):
    """-> (doc ids, info)."""
    cap = int(cap or opts["shortlist_max"])
    if len(metas) <= 1:
        ids = [m["doc_id"] for m in metas]
        return ids, {"version": ROUTE_VERSION, "rule": "single_doc_scope", "shortlist": ids, "retained": ids}
    scored = score_documents(metas, cards, pool, matrix, opts)
    by = {r["doc_id"]: r for r in scored}
    meta_by = {m["doc_id"]: m for m in metas}
    retained = []
    # 1. explicit matches: per requirement, the best document of its entity holding its period, else of its entity
    for r in matrix["requirements"]:
        own = [x for x in scored if r["id"] in x["entity_reqs"]]
        if not own:
            continue
        exact = [x for x in own if r["id"] in x["period_reqs"]]
        # no report of that period: the entity's best report is still retained (a later one may restate it)
        pick = exact[:1] or own[:1]
        for x in pick:
            if x["doc_id"] not in retained:
                retained.append(x["doc_id"])
    retained = retained[:cap]
    out = list(retained)
    for x in scored:
        if len(out) >= cap:
            break
        if x["doc_id"] in out:
            continue
        if x["score"] <= 0.0:
            continue            # no evidence at all: never filled by recency
        out.append(x["doc_id"])
    info = {"version": ROUTE_VERSION, "rule": "scored", "shortlist": out, "retained": retained,
            "scores": scored[: max(cap * 2, 16)]}
    return out, info


def escape_candidates(pool, shortlist_ids, opts):
    """The pooled candidates of documents outside the shortlist, best pooled rank first."""
    inside = set(shortlist_ids)

    def best_rank(c):
        return min(((c.features or {}).get("pool_rank") or {"x": 10 ** 6}).values())
    out = sorted((c for c in pool if c.doc_id not in inside), key=best_rank)
    seen, keep = set(), []
    for c in out:
        k = (c.doc_id, tuple(c.unit_ids))
        if k in seen:
            continue
        seen.add(k)
        c.features["src"] = sorted(set(c.features.get("src") or []) | {"escape"})
        keep.append(c)
        if len(keep) >= int(opts["escape_max"]):
            break
    return keep


def expansion_docs(req, metas, current, scored_rows, opts):
    """Documents of a requirement's own entity / period that are not yet targeted, best score first."""
    meta_by = {m["doc_id"]: m for m in metas}
    rank = {r["doc_id"]: i for i, r in enumerate(scored_rows or [])}
    cands = []
    for m in metas:
        if m["doc_id"] in current:
            continue
        e = entity_match(m, req) if req.get("entity_keys") else None
        p = period_match(m, req)
        if e is False:
            continue
        if e is None and not p:
            continue
        cands.append((0 if (e and p) else 1 if e else 2, rank.get(m["doc_id"], 10 ** 6), m["doc_id"]))
    cands.sort()
    return [d for _a, _b, d in cands[: int(opts["expand_max_docs"])] if d in meta_by]


def serving_docs(req, metas, shortlist_ids):
    """Shortlisted documents that could hold the requirement (its entity, or no entity named; its period or the
    next year's report, or no period named)."""
    meta_by = {m["doc_id"]: m for m in metas}
    out = []
    for d in shortlist_ids:
        m = meta_by.get(d)
        if m is None:
            continue
        if req.get("entity_keys") and not entity_match(m, req):
            continue
        if req.get("period") and period_match(m, req) is False:
            continue
        out.append(d)
    return out
