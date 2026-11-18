"""LEX6 core: document-aware local retrieval with bounded evidence replacement (no model calls here).

Spec: work/analysis/LEX6_DESIGN_FINAL.md = Astra R2 section 5, steps 1-8 (LEX6_ASTRA_R2.log). Every function works on
the shared HybridIndex (hx), the lex4 Packager (P) and immutable packages; the router call and the answer call live in
router.py / approach.py. The offline evaluator and the online approach run the SAME plan() below.

    1. question hybrid: BM25 top 40 U dense top 40 over the pool, RRF k=60 (HybridIndex.search, existing chunk ties);
       H(c) = the fused RRF score. B0 = the exact HYBRID-intact pack (lex5 bundles.hybrid_intact, 6,000 chars).
    2. document ranking, missing evidence gets NO rank credit (Astra R2 section 4):
           V(d) = max_{c in d} H(c) over the fused candidates        ranked only when V(d) > 0
           C(d) = q . mu_d, mu_d = normalize(mean of d's unit chunk vectors)   ranked only when mu_d exists
           F0(d) = 1[V>0]/(60+r_V) + 1[mu]/(60+r_C)                   (+ 0.5/(60+r_cardBM25) for card BM25 > 0
                                                                        + 0.5/(60+r_cardDense) for a card vector: CARDS)
       every document-level ranking breaks ties by document id; a document with F = 0 is not ranked at all.
       shortlist (<= 8, dedup): top 2 by V, top 2 by C, (CARDS/ROUTER: top 2 by card-only RRF), rest by F.
    3. nominations: top 3 by F (whole question as the part); a valid router result replaces them (<= 4 docs,
       <= 8 part/doc assignments). LOCAL retrieval over ALL chunks of the nominated document:
           L(c) = 1[BM25(q [+ q_i], c) > 0]/(60+r_B) + 1[c has a vector]/(60+r_D)    (dense reuses the question vector)
       ties by chunk index; 2 distinct intact package alternatives per slot (neither represents the other).
    4. a slot is represented when B0 shows the full provenance of one of its alternatives (bundles.represented).
    5. packing: anchor = the shortest prefix of B0 whose exact rendered cost reaches 3,000 (all of B0 if shorter);
       enumerate alt1 / alt2 / omit per unresolved slot (<= 3^8); shown = canonical(anchor + chosen); reservation
       additions (rendered, headers and separators included) <= 3,000; total <= 6,000; then fill from the global
       RRF order. Lexicographic: (1) most slots represented (by anchor + reservations), (2) lowest summed alternative
       rank, (3) fewest B0 source chars lost, (4) highest summed global H of the final pack, (5) lowest rendered
       cost, (6) stable package ids. No unresolved slot represented -> B0 EXACTLY (the no-change branch).
Arms (flags): rank=s2|s5 (one document channel), absent_only (slots only for documents absent from B0),
global_only (alternatives from the global candidates of the document, no fresh local search), sections (<= 2
section cards per nominated document, one best intact package each as an optional 1-alternative slot), cards
(card channels in F + card nominations in the shortlist), router (router.py), no_anchor (packing ablation).
"""
import itertools

import numpy as np

from ...retrieval import hybrid as H
from ..lex5 import bundles as B

LEX6_CORE_VERSION = "lex6-core-v1"
C = {"q_depth": 40, "rrf_k": 60, "budget": 6000, "anchor": 3000, "add_cap": 3000, "shortlist": 8, "nominate": 3,
     "top_v": 2, "top_c": 2, "top_card": 2, "alts": 2, "max_slots": 8, "card_w": 0.5, "sections_per_doc": 2,
     "router_docs": 4, "router_assign": 8}
RK = C["rrf_k"]


# ------------------------------------------------------------------------------------------------------------------
# step 2: document ranking (zero credit for missing evidence, document-id ties)
# ------------------------------------------------------------------------------------------------------------------

def rank_of(scores, keep=lambda d, s: True):
    """{doc: score} -> docs best first (ties: doc id), only those `keep` accepts."""
    return sorted([d for d, s in scores.items() if keep(d, s)], key=lambda d: (-scores[d], d))


def chunk_prior(hx, fused):
    """fused [(chunk, H)] -> V {doc: max H} (only documents with a fused candidate)."""
    V = {}
    for c, h in fused:
        d = hx.chunks[c]["doc_id"]
        if h > V.get(d, 0.0):
            V[d] = float(h)
    return V


def centroid_scores(cents, qvec, docs):
    q = np.asarray(qvec, dtype=np.float32)
    q = q / (float(np.linalg.norm(q)) or 1.0)
    return {d: float(cents[d] @ q) for d in docs if d in cents}


def rrf_scores(orders, weights=None):
    s = {}
    for i, o in enumerate(orders):
        w = 1.0 if weights is None else weights[i]
        for r, d in enumerate(o, 1):
            s[d] = s.get(d, 0.0) + w / (RK + r)
    return s


def rank_docs(V, cent, card_bm=None, card_dn=None, channels="s2s5", card_w=None):
    """-> dict of orders + scores. V/cent/card_*: {doc: score}; card_bm counts only when > 0; card_dn only for docs
    with a card vector (present in the dict). channels: s2s5 (default) | s2 | s5."""
    card_w = C["card_w"] if card_w is None else card_w
    oV = rank_of(V, lambda d, s: s > 0)
    oC = rank_of(cent)
    orders, weights = [], []
    if channels in ("s2s5", "s2"):
        orders.append(oV)
        weights.append(1.0)
    if channels in ("s2s5", "s5"):
        orders.append(oC)
        weights.append(1.0)
    out = {"V": oV, "C": oC, "V_scores": V, "C_scores": cent}
    if card_bm is not None or card_dn is not None:
        oB = rank_of(card_bm or {}, lambda d, s: s > 0)
        oD = rank_of(card_dn or {})
        orders += [oB, oD]
        weights += [card_w, card_w]
        card_only = rrf_scores([oB, oD])
        out.update(cardB=oB, cardD=oD, cardB_scores=card_bm or {}, cardD_scores=card_dn or {},
                   card=rank_of(card_only), card_scores=card_only)
    F = rrf_scores(orders, weights)
    out["F"] = rank_of(F, lambda d, s: s > 0)
    out["F_scores"] = F
    return out


def shortlist(R, k=None, cards=False):
    k = k or C["shortlist"]
    out = []

    def add(ds, n):
        m = 0
        for d in ds:
            if m >= n or len(out) >= k:
                break
            if d not in out:
                out.append(d)
            m += 1
    add(R["V"], C["top_v"])
    add(R["C"], C["top_c"])
    if cards and "card" in R:
        add(R["card"], C["top_card"])
    add(R["F"], k)
    return out[:k]


# ------------------------------------------------------------------------------------------------------------------
# step 3: local retrieval inside one document
# ------------------------------------------------------------------------------------------------------------------

class QCtx:
    """Per-question caches: BM25 scores per query text, dense scores of the question vector, doc chunk lists."""

    def __init__(self, hx, qvec):
        self.hx = hx
        self._bm = {}
        self._docs = {}
        self.dense = None
        if qvec is not None and hx.E.shape[1]:
            q = np.asarray(qvec, dtype=np.float32)
            q = q / (float(np.linalg.norm(q)) or 1.0)
            self.dense = (hx.E @ q).astype(np.float32)

    def bm25(self, text):
        if text not in self._bm:
            self._bm[text] = self.hx.bm25_scores(text)[0]
        return self._bm[text]

    def doc_chunks(self, d):
        if d not in self._docs:
            mask, _ = self.hx.scope_mask([d])
            self._docs[d] = [int(i) for i in np.nonzero(mask)[0]]
        return self._docs[d]


def local_order(ctx, doc, text):
    """-> [(chunk, L, bm25 rank|None, dense rank|None)] over ALL chunks of `doc`, best first (ties: chunk index)."""
    idx = ctx.doc_chunks(doc)
    s = ctx.bm25(text)
    bm = sorted([i for i in idx if s[i] > 0.0], key=lambda i: (-float(s[i]), i))
    dn = []
    if ctx.dense is not None:
        dn = sorted([i for i in idx if ctx.hx.has_vec[i]], key=lambda i: (-float(ctx.dense[i]), i))
    fused = H.rrf_fuse({"bm25": bm, "dense": dn}, k=RK)
    out = [(c, sc, rk.get("bm25"), rk.get("dense")) for c, sc, rk in fused]
    out.sort(key=lambda t: (-t[1], t[0]))
    return out


def alternatives(order, package, n=None, add_cap=None, exclude=()):
    """First n distinct intact packages of an ordered chunk list: a package represented by an earlier alternative (or
    representing one) is skipped; a package whose own rendered cost as a first item exceeds add_cap is skipped.
    order: [(chunk, ...)] or [chunk]; package(chunk) -> frozen package. -> [(package, local_rank)]."""
    n = n or C["alts"]
    add_cap = add_cap or C["add_cap"]
    out = []
    for r, t in enumerate(order, 1):
        c = t[0] if isinstance(t, (tuple, list)) else t
        if c in exclude:
            continue
        p = package(c)
        if p.get("_cost1", 0) > add_cap:
            continue
        if any(B.represented_by(p, q) or B.represented_by(q, p) for q, _ in out):
            continue
        out.append((p, r))
        if len(out) >= n:
            break
    return out


# ------------------------------------------------------------------------------------------------------------------
# steps 4-5: representation and bounded packing
# ------------------------------------------------------------------------------------------------------------------

def anchor_of(B0, cost, target=None):
    target = C["anchor"] if target is None else target
    if target <= 0:
        return []
    for i in range(1, len(B0) + 1):
        if cost(B0[:i]) >= target:
            return list(B0[:i])
    return list(B0)


def slot_represented(slot, shown):
    """-> index of the best alternative shown (0-based) or None."""
    for ai, a in enumerate(slot["alts"]):
        if B.represented(a, shown):
            return ai
    return None


def pack(B0, slots, rrf_order, Hs, cost, budget=None, add_cap=None, anchor_target=None, max_slots=None,
         max_eval=None):
    """B0: the HYBRID-intact pack; slots: [{"key": .., "alts": [pkg, ...]}]; rrf_order: global packages in RRF order;
    Hs: {chunk: H}. -> (shown, info). info records the anchor, the unresolved slots, the choice, the reservations,
    the displaced B0 packages and the reason when B0 is returned unchanged."""
    budget = C["budget"] if budget is None else budget
    add_cap = C["add_cap"] if add_cap is None else add_cap
    max_slots = C["max_slots"] if max_slots is None else max_slots
    anchor = anchor_of(B0, cost, anchor_target)
    info = {"anchor": [p["chunk"] for p in anchor], "anchor_cost": cost(anchor), "b0": [p["chunk"] for p in B0],
            "slots_total": len(slots), "resolved_in_b0": [], "unresolved": [], "evaluated": 0, "changed": False}
    un = []
    for i, s in enumerate(slots):
        if not s["alts"]:
            continue
        if slot_represented(s, B0) is not None:
            info["resolved_in_b0"].append(i)
        else:
            un.append(i)
    un = un[:max_slots]
    info["unresolved"] = un
    if not un:
        info["reason"] = "no_unresolved_slot"
        return list(B0), info
    a_cost = info["anchor_cost"]
    parts_pk = [[(a,) for a in slots[i]["alts"]] for i in un]
    groups = {}
    seen = set()
    for combo in itertools.product(*[range(-1, len(slots[i]["alts"])) for i in un]):
        res = []
        for i, c in zip(un, combo):
            if c >= 0:
                p = slots[i]["alts"][c]
                if p["chunk"] not in {q["chunk"] for q in res}:
                    res.append(p)
        key0 = tuple(p["chunk"] for p in res)
        if key0 in seen:
            continue
        seen.add(key0)
        shown = B.canonical(list(anchor) + res)
        info["evaluated"] += 1
        if max_eval and info["evaluated"] > max_eval:
            break
        c_ = cost(shown)
        if c_ - a_cost > add_cap or c_ > budget:
            continue
        cov, rk = B._covered(parts_pk, shown)
        groups.setdefault((-cov, rk), []).append((shown, combo, res))
    best_k = min(groups)
    if -best_k[0] == 0:
        info["reason"] = "no_slot_fits"
        return list(B0), info
    best = None
    for shown, combo, res in groups[best_k]:
        full, _ = B.fill(list(shown), [], rrf_order, cost, budget, bundles=False, rrf=True)
        lost = sum(len(p["text"]) for p in B0 if not B.represented(p, full))
        hsum = sum(Hs.get(p["chunk"], 0.0) for p in full)
        key = (lost, -round(hsum, 12), cost(full), tuple(sorted(p["chunk"] for p in full)))
        if best is None or key < best[0]:
            best = (key, full, combo, res, shown)
    key, full, combo, res, shown = best
    info.update(changed=[p["chunk"] for p in full] != info["b0"], covered=-best_k[0], alt_rank_sum=best_k[1],
                choice=[{"slot": i, "alt": c} for i, c in zip(un, combo)],
                reservations=[p["chunk"] for p in res], b0_chars_lost=key[0], h_sum=-key[1], cost=key[2],
                additions_cost=cost(shown) - a_cost,
                displaced=[p["chunk"] for p in B0 if not B.represented(p, full)])
    info["reason"] = "replaced" if info["changed"] else "same_as_b0"
    return full, info


# ------------------------------------------------------------------------------------------------------------------
# the whole deterministic plan (shared by the approach and the offline evaluator)
# ------------------------------------------------------------------------------------------------------------------

def plan(hx, P, freeze, cost, docs, qtext, qvec, fused, cents, opts=None, cards=None, router=None, ctx=None):
    """One question. fused: [(chunk, H)] global candidates (<= 80) in RRF order; docs: the pool/scope document ids;
    cents: {doc: unit centroid}; opts: flags (rank, absent_only, global_only, sections, cards, router, no_anchor);
    cards: a cardsrc.CardView for the scope (needed by cards / sections / router); router: callable(shortlist info)
    -> router result dict (router.py) or None. -> dict(shown, b0, trace)."""
    opts = dict(opts or {})
    ctx = ctx or QCtx(hx, qvec)
    pk = {}

    def package(c):
        if c not in pk:
            p = dict(P.package(c))
            p["_cost1"] = cost.fn(freeze(p), 1) if hasattr(cost, "fn") else 0
            pk[c] = freeze(p)
        return pk[c]
    Hs = {c: float(h) for c, h in fused}
    rrf_order = [package(c) for c, _ in fused]
    B0 = B.hybrid_intact(rrf_order, cost, C["budget"])
    tr = {"version": LEX6_CORE_VERSION, "opts": {k: v for k, v in opts.items() if v},
          "b0": [p["chunk"] for p in B0], "b0_cost": cost(B0)}
    # step 2 ------------------------------------------------------------------------------------------------------
    V = chunk_prior(hx, fused)
    V = {d: v for d, v in V.items() if d in set(docs)}
    cent = centroid_scores(cents, qvec, docs) if qvec is not None else {}
    use_cards = bool(opts.get("cards") or opts.get("router"))
    cbm = cdn = None
    if use_cards and cards is not None:
        cbm, cdn = cards.doc_scores(qtext, qvec)
    R = rank_docs(V, cent, cbm if opts.get("cards") else None, cdn if opts.get("cards") else None,
                  channels=opts.get("rank") or "s2s5")
    if opts.get("router") and not opts.get("cards") and cards is not None:      # card-only order for the shortlist
        Rc = rank_docs(V, cent, cbm, cdn)
        R.update({k: Rc[k] for k in ("cardB", "cardD", "card", "card_scores", "cardB_scores", "cardD_scores")})
    SL = shortlist(R, cards=use_cards and "card" in R)
    nominated = R["F"][:C["nominate"]]
    tr["docrank"] = {"V": [[d, round(R["V_scores"][d], 6)] for d in R["V"][:12]],
                     "C": [[d, round(R["C_scores"][d], 5)] for d in R["C"][:12]],
                     "F": [[d, round(R["F_scores"][d], 6)] for d in R["F"][:12]],
                     "n_ranked": {"V": len(R["V"]), "C": len(R["C"]), "F": len(R["F"])}}
    if "card" in R:
        tr["docrank"].update(cardB=[[d, round(R["cardB_scores"][d], 4)] for d in R["cardB"][:12]],
                             cardD=[[d, round(R["cardD_scores"][d], 5)] for d in R["cardD"][:12]],
                             card=[d for d in R["card"][:12]])
    tr["shortlist"] = SL
    tr["nominated"] = list(nominated)
    # step 3 slots -------------------------------------------------------------------------------------------------
    parts = [{"text": None, "docs": list(nominated), "source": "deterministic"}]
    if router is not None:
        rres = router({"question": qtext, "shortlist": SL, "R": R, "B0": B0, "ctx": ctx, "package": package,
                       "hx": hx, "cost": cost})
        tr["router"] = {k: v for k, v in (rres or {}).items() if not k.startswith("_")}
        if rres and rres.get("use"):
            parts = rres["parts"]
            tr["nominated"] = sorted({d for p in parts for d in p["docs"]}, key=lambda d: SL.index(d)
                                     if d in SL else 99)
    b0_docs = {p["doc_id"] for p in B0}
    slots = []
    for pi, part in enumerate(parts):
        qi = qtext if not part["text"] else (qtext + " " + part["text"])
        for d in part["docs"]:
            if opts.get("absent_only") and d in b0_docs:
                continue
            if opts.get("global_only"):
                order = [c for c, _ in fused if hx.chunks[c]["doc_id"] == d]
                alts = alternatives(order, package)
                cand = [{"chunk": c, "rank": r} for r, c in enumerate(order[:5], 1)]
            else:
                lo = local_order(ctx, d, qi)
                alts = alternatives(lo, package)
                cand = [{"chunk": c, "rank": r, "L": round(s, 6), "bm25_rank": rb, "dense_rank": rd}
                        for r, (c, s, rb, rd) in enumerate(lo[:5], 1)]
            slots.append({"key": "P%d:%s" % (pi + 1, d), "part": pi + 1, "doc": d, "kind": "doc",
                          "alts": [a for a, _ in alts], "alt_ranks": [r for _, r in alts], "top": cand})
    if opts.get("sections") and cards is not None:
        used = {a["chunk"] for s in slots for a in s["alts"]}
        for d in tr["nominated"]:
            lo = None
            for sec in cards.sections(d, qvec, C["sections_per_doc"]):
                if lo is None:
                    lo = local_order(ctx, d, qtext)
                a, b = sec["span"]
                inside = [t for t in lo if a <= hx.chunks[t[0]]["span"][0] < b]
                alts = alternatives(inside, package, n=1, exclude=used)
                if alts:
                    used.add(alts[0][0]["chunk"])
                    slots.append({"key": "S:%s#%s" % (d, sec["key"]), "part": 0, "doc": d, "kind": "section",
                                  "section": sec["key"], "section_score": round(sec["score"], 5),
                                  "alts": [alts[0][0]], "alt_ranks": [alts[0][1]], "top": []})
    tr["slots"] = [{"key": s["key"], "kind": s["kind"], "doc": s["doc"], "part": s["part"],
                    "section": s.get("section"),
                    "alts": [{"chunk": a["chunk"], "local_rank": r, "chars": len(a["text"]), "cost1": a["_cost1"],
                              "provenance": [list(x) for x in a["provenance"]],
                              "in_b0": B.represented(a, B0)} for a, r in zip(s["alts"], s["alt_ranks"])],
                    "top": s["top"]} for s in slots]
    shown, info = pack(B0, slots, rrf_order, Hs, cost,
                       anchor_target=0 if opts.get("no_anchor") else None)
    tr["pack"] = info
    tr["final"] = [p["chunk"] for p in shown]
    why = {}
    anc = set(info["anchor"])
    res = set(info.get("reservations") or [])
    for p in shown:
        c = p["chunk"]
        why[c] = "anchor" if c in anc else ("reservation" if c in res else ("fill-b0" if c in tr["b0"]
                                                                             else "fill-rrf"))
    tr["final_why"] = why
    tr["provenance"] = {str(p["chunk"]): [list(x) for x in p["provenance"]] for p in shown}
    return {"shown": shown, "b0": B0, "trace": tr, "slots": slots, "R": R, "packages": pk, "rrf_order": rrf_order,
            "ctx": ctx}
