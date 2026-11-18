"""LEX11-M presentation order (Astra R1, work/analysis/LEX11_ASTRA_R1.log, steps 3-4).

The ordinary chunk packer (contracts.pack_evidence) runs on the F2 top 40 first; its included items - displayed text
(the clipped last item included), source interval and provenance - are FROZEN. Only those items are reordered:

    H(c) = 1[r_bm25(c) <= 40] / (60 + r_bm25(c)) + 1[r_dense(c) <= 40] / (60 + r_dense(c))

with r_bm25 / r_dense HYBRID's own channel ranks (HybridIndex.search: BM25 + the HYBRID dense embedding; the lex8 HF
bge-m3 dense list does not reproduce HYBRID's order). Missing ranks contribute 0, so sparse-only items keep their
membership and sort last. Ties: best BM25/dense rank, F2 rank, chunk id. The frozen blocks are re-rendered with fresh
citation numbers; nothing is refilled, re-expanded or truncated again. Because the header numbers 1..n are the same
multiset in any order, the rendered length is identical to the F2 pack's.
"""
from ...contracts import EvidencePack, item_header

ORDER_VERSION = "lex11-order-v1"
RRF_K = 60
DEPTH = 40
INF = 10 ** 9


def h_score(r_bm25, r_dense, k=RRF_K, depth=DEPTH):
    s = 0.0
    for r in (r_bm25, r_dense):
        if r is not None and 1 <= r <= depth:
            s += 1.0 / (k + r)
    return s


def channel_ranks(bm25_list, dense_list, depth=DEPTH):
    """{chunk: (r_bm25 | None, r_dense | None)} from two best-first lists (1-based, first `depth` only)."""
    out = {}
    for j, lst in enumerate((bm25_list, dense_list)):
        for r, c in enumerate(list(lst)[:depth], 1):
            cur = list(out.get(int(c), (None, None)))
            if cur[j] is None:
                cur[j] = r
            out[int(c)] = tuple(cur)
    return out


def sort_key(chunk, ranks, f2_rank):
    rb, rd = ranks.get(int(chunk), (None, None))
    best = min([r for r in (rb, rd) if r is not None and r <= DEPTH], default=INF)
    return (-h_score(rb, rd), best, f2_rank.get(int(chunk), INF), int(chunk))


def reorder(items, ranks, f2_rank):
    """items: the frozen pack items (RetrievedItem, meta['chunk']) -> (new list, keys)."""
    keyed = [(sort_key(it.meta["chunk"], ranks, f2_rank), it) for it in items]
    keyed.sort(key=lambda t: t[0])
    return [it for _k, it in keyed], [k for k, _it in keyed]


def render(items, corpus, like):
    """Re-render frozen items in the given order exactly as pack_evidence renders whole items. `like` = the frozen
    EvidencePack (budget, n_offered, last_item_cut are carried over)."""
    parts = []
    for n, it in enumerate(items, 1):
        sep = "\n\n" if parts else ""
        parts.append(sep + item_header(n, it, corpus) + "\n" + (it.text or "").strip())
    text = "".join(parts)
    return EvidencePack(items=list(items), text=text, chars=len(text), budget=like.budget, n_offered=like.n_offered,
                        n_included=len(items), last_item_cut=like.last_item_cut)


def displayed(ev):
    """(chunk, doc_id, span, displayed body) per item - the invariance fingerprint."""
    return [(int(it.meta["chunk"]), it.doc_id, tuple(it.span) if it.span else None, (it.text or "").strip())
            for it in ev.items]


def check_invariance(frozen, new):
    """Membership and displayed content identical; rendered length identical and within budget. Raises on failure."""
    a, b = displayed(frozen), displayed(new)
    if sorted(a) != sorted(b):
        raise AssertionError("LEX11: membership / displayed content changed by the reorder")
    if new.chars != frozen.chars or new.chars > frozen.budget:
        raise AssertionError("LEX11: rendered chars %d vs frozen %d (budget %d)" % (new.chars, frozen.chars, frozen.budget))
    return True


def lex11_pack(frozen, ranks, f2_order, corpus):
    """frozen F2 pack -> (reordered pack, info). f2_order: the F2 top-40 chunk order."""
    f2_rank = {int(c): i + 1 for i, c in enumerate(f2_order)}
    items, keys = reorder(frozen.items, ranks, f2_rank)
    ev = render(items, corpus, frozen)
    check_invariance(frozen, ev)
    info = {"pre": [int(it.meta["chunk"]) for it in frozen.items], "post": [int(it.meta["chunk"]) for it in items],
            "H": [round(-k[0], 6) for k in keys],
            "ranks": {str(int(it.meta["chunk"])): list(ranks.get(int(it.meta["chunk"]), (None, None))) for it in items},
            "sparse_only": [int(it.meta["chunk"]) for it in items if h_score(*ranks.get(int(it.meta["chunk"]), (None, None))) == 0.0],
            "cut_chunk": next((int(it.meta["chunk"]) for it in items if (it.meta or {}).get("cut_from_chars")), None)}
    return ev, info
