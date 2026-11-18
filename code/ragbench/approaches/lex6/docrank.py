"""LEX6 document-ranking signals (no model calls; card vectors are supplied by the caller).

Each signal ranks the documents of one pool for one question; every ranking is a full, deterministic order of the
pool (ties: pool order), so recall@k and ranks are defined for every document.

    S1 chunk-vote     sum over a document's chunks of their RRF score in the question hybrid candidate set
                      (BM25 top Q ∪ dense top Q over the pool, RRF k=60 - exactly HYBRID's fusion)
    S2 chunk-max      the best single chunk RRF of the document
    S3 card BM25      bm25s (B2's tokenizer and parameters) over the pool's document cards (variants.document_card);
                      idf of the pool's cards
    S4 card dense     cosine(question vector, card vector)
    S5 centroid       cosine(question vector, mean of the document's L2-normalised chunk vectors)
    S6 fused          RRF (k=60) of the S1, S3, S4 document rankings
"""
import numpy as np

from ...retrieval import hybrid as H
from ..baselines import DEFAULTS as BLD, tokenize
from ..lex4.core import C as K4, rank_lists

DOCRANK_VERSION = "lex6-docrank-v1"
RRF_K = 60


def order_by(scores, pool):
    """scores {doc: float} (missing = -inf) -> pool docs best first; ties keep pool order."""
    pos = {d: i for i, d in enumerate(pool)}
    return sorted(pool, key=lambda d: (-scores.get(d, float("-inf")), pos[d]))


def rrf_docs(orders, pool, k=RRF_K):
    """orders: list of full doc orders -> fused doc order (ties: pool order)."""
    s = {d: 0.0 for d in pool}
    for o in orders:
        for r, d in enumerate(o, 1):
            s[d] += 1.0 / (k + r)
    return order_by(s, pool)


def chunk_votes(hx, qtext, qvec, pool, depth=None):
    """-> (S1 scores, S2 scores, fused chunk list [(chunk, rrf)])."""
    depth = depth or K4["q_depth"]
    mask, _ = hx.scope_mask(pool)
    bm, dn = rank_lists(hx, qtext, qvec, mask, depth)
    fused = [(c, s) for c, s, _r in H.rrf_fuse({"bm25": bm, "dense": dn}, k=RRF_K)]
    s1, s2 = {}, {}
    for c, s in fused:
        d = hx.chunks[c]["doc_id"]
        s1[d] = s1.get(d, 0.0) + s
        s2[d] = max(s2.get(d, 0.0), s)
    return s1, s2, fused


def centroids(hx, pool):
    """-> {doc: unit centroid vector} of the doc's chunk vectors (docs without vectors left out)."""
    out = {}
    for d in pool:
        di = hx.doc_ids.index(d)
        m = (hx.doc_pos == di) & hx.has_vec
        if m.any():
            v = hx.E[m].mean(axis=0)
            out[d] = v / (float(np.linalg.norm(v)) or 1.0)
    return out


class CardIndex:
    """BM25 + dense over one pool's document cards."""

    def __init__(self, pool, cards, vectors=None):
        import bm25s
        self.pool = list(pool)
        self.cards = cards
        toks = [tokenize(cards[d], BLD["tokenizer"]) for d in self.pool]
        p = BLD["bm25"]
        self.bm = bm25s.BM25(k1=p["k1"], b=p["b"], method=p["method"])
        self.bm.index(toks, show_progress=False)
        self.V = None
        if vectors is not None:
            V = np.stack([np.asarray(vectors[d], dtype=np.float32) for d in self.pool])
            self.V = V / np.linalg.norm(V, axis=1, keepdims=True)

    def bm25(self, qtext):
        toks = [t for t in H.query_tokens(qtext) if t in self.bm.vocab_dict]
        if not toks:
            return {d: 0.0 for d in self.pool}
        s = self.bm.get_scores(toks)
        return {d: float(x) for d, x in zip(self.pool, s)}

    def dense(self, qvec):
        q = np.asarray(qvec, dtype=np.float32)
        q = q / (float(np.linalg.norm(q)) or 1.0)
        return {d: float(x) for d, x in zip(self.pool, self.V @ q)}


def centroid_scores(cents, qvec, pool):
    q = np.asarray(qvec, dtype=np.float32)
    q = q / (float(np.linalg.norm(q)) or 1.0)
    return {d: float(cents[d] @ q) for d in pool if d in cents}


def all_orders(hx, card_index, cents, qtext, qvec, pool):
    """-> ({signal: full doc order}, fused chunk list)."""
    s1, s2, fused = chunk_votes(hx, qtext, qvec, pool)
    o = {"S1 chunk-vote": order_by(s1, pool), "S2 chunk-max": order_by(s2, pool),
         "S3 card-BM25": order_by(card_index.bm25(qtext), pool),
         "S4 card-dense": order_by(card_index.dense(qvec), pool),
         "S5 centroid": order_by(centroid_scores(cents, qvec, pool), pool)}
    o["S6 fused(S1,S3,S4)"] = rrf_docs([o["S1 chunk-vote"], o["S3 card-BM25"], o["S4 card-dense"]], pool)
    return o, fused


def restricted_order(hx, qtext, qvec, docs, depth=None):
    """HYBRID's fused chunk order with the scope restricted to `docs` (BM25 top Q ∪ dense top Q inside them)."""
    depth = depth or K4["q_depth"]
    mask, _ = hx.scope_mask(docs)
    bm, dn = rank_lists(hx, qtext, qvec, mask, depth)
    return [c for c, _s, _r in H.rrf_fuse({"bm25": bm, "dense": dn}, k=RRF_K)]


def reserve_first(hx, order, docs, escape=None):
    """Per-document reservation: the best chunk of each doc in `docs` (in the order's own ranking), then the escape
    chunks (e.g. the global top-2), then the rest of the order. Duplicates dropped."""
    lead = []
    for d in docs:
        c = next((c for c in order if hx.chunks[c]["doc_id"] == d), None)
        if c is not None:
            lead.append(c)
    lead.sort(key=order.index)
    return list(dict.fromkeys((escape or []) + lead + list(order)))
