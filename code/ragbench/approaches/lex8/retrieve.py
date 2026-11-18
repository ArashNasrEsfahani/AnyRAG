"""LEX8 query time (no LLM, no cross-encoder): two granularities, score-aware fusion, unit utility, candidates.

    chunks: BM25 (shared HybridIndex) + bge-m3 dense + bge-m3 sparse, top DEPTH each (scope), union Uc
    units:  BM25 (unit field) + bge-m3 dense + bge-m3 sparse, top DEPTH each (scope), union Uu
    R_chunk, R_unit = MaxMean over the real scores, min-max per channel over each union (clipped for non-members)
    u(a) = 0.8 R_unit(a) + 0.2 max_parents R_chunk(p)            [optional synthetic field: u_exp = 0.8 u + 0.2 e]
    candidates: top N_COMPACT compact candidates (one unit, or an adjacent pair, <= 900 body chars) built from Uu
                (+ synthetic-linked units), by highest contained u (ties: lower cost, key); + top N_CHUNKS chunks of
                Uc by R_chunk (ties: best channel rank, id)
"""
import time

import numpy as np

from . import fusion as FU
from . import packer as PK
from .units import MAX

RETRIEVE_VERSION = "lex8-retrieve-v1"
C = {"depth": 40, "n_compact": 80, "n_chunks": 40}


def _lists(arrays, valid, depth):
    return {ch: FU.top_ids(arr, valid[ch], depth) for ch, arr in arrays.items()}


def chunk_channels(six, qtext, qm, mask):
    hx, X = six.hx, six.chunk_m3
    s_bm, _ = hx.bm25_scores(qtext)
    s_bm = np.asarray(s_bm, np.float32)
    qd = np.asarray(qm["dense"], np.float32)
    s_dn = X.dense @ qd
    s_sp = X.sparse_scores(qm["sparse"])
    arrays = {"bm25": s_bm, "dense": s_dn, "sparse": s_sp}
    valid = {"bm25": mask & (s_bm > 0), "dense": mask.copy(), "sparse": mask & (s_sp > 0)}
    return arrays, valid


def unit_channels(six, qtext, qm, umask):
    X = six.unit_m3
    s_bm = six.unit_bm25_scores(qtext)
    qd = np.asarray(qm["dense"], np.float32)
    s_dn = X.dense @ qd
    s_sp = X.sparse_scores(qm["sparse"])
    arrays = {"bm25": s_bm, "dense": s_dn, "sparse": s_sp}
    valid = {"bm25": umask & (s_bm > 0), "dense": umask.copy(), "sparse": umask & (s_sp > 0)}
    return arrays, valid


def fuse_all(arrays, lists, n):
    """MaxMean R for ALL n rows (vectorised; normaliser from the union of the lists) + the union order."""
    order, sc, N = FU.fuse(arrays, lists, "maxmean")
    R = FU.maxmean_vec(N.zvec(range(n)))
    return order, R, N


def run(six, qtext, qm, doc_ids=None, synth=None, depth=None, n_compact=None, n_chunks=None):
    """-> dict with channel lists, union orders, u (array over all units), candidates (compact + chunk), timings.
    synth: optional callable(qtext, umask) -> {unit id: e(a)} (the synthetic-question field)."""
    depth, n_compact, n_chunks = depth or C["depth"], n_compact or C["n_compact"], n_chunks or C["n_chunks"]
    st = {}
    t = time.perf_counter()
    hx = six.hx
    mask, _ = hx.scope_mask(doc_ids)
    umask = six.unit_mask(doc_ids)
    ca, cv = chunk_channels(six, qtext, qm, mask)
    cl = _lists(ca, cv, depth)
    c_order, Rc, Nc = fuse_all(ca, cl, hx.n)
    st["t_chunk_ms"] = (time.perf_counter() - t) * 1000.0
    t = time.perf_counter()
    ua, uv = unit_channels(six, qtext, qm, umask)
    ul = _lists(ua, uv, depth)
    u_order, Ru, Nu = fuse_all(ua, ul, six.nu)
    par = six.upar
    pr = np.where(par >= 0, Rc[np.clip(par, 0, None)], 0.0).max(axis=1) if par.shape[1] else np.zeros(six.nu)
    u = FU.W_UNIT * Ru + FU.W_PARENT * pr
    e = {}
    if synth is not None:
        e = synth(qtext, umask)
        if e:
            ev = np.zeros(six.nu)
            for i, x in e.items():
                ev[i] = x
            u = FU.W_BASE * u + FU.W_SYNTH * ev
    st["t_unit_ms"] = (time.perf_counter() - t) * 1000.0
    t = time.perf_counter()
    seeds = list(dict.fromkeys(list(u_order) + [i for i in e if umask[i]]))
    cands = compact_candidates(six, seeds, u, n_compact)
    rank = {}
    for ch, v in cl.items():
        for r, x in enumerate(v, 1):
            rank[x] = min(rank.get(x, 10 ** 9), r)
    corder = sorted(c_order, key=lambda x: (-Rc[x], rank.get(x, 10 ** 9), x))[:n_chunks]
    cands += [chunk_candidate(six, ci, u) for ci in corder]
    st["t_cands_ms"] = (time.perf_counter() - t) * 1000.0
    return {"chunk_lists": cl, "unit_lists": ul, "chunk_union": c_order, "unit_union": u_order, "Rc": Rc, "Ru": Ru,
            "u": u, "e": e, "cands": cands, "chunk_order": corder, "timing": st,
            "chunk_scores": {ch: {int(x): float(arr[x]) for x in c_order} for ch, arr in ca.items()},
            "unit_scores": {ch: {int(x): float(arr[x]) for x in u_order} for ch, arr in ua.items()}}


def _cand(six, key, i, j, u):
    U = six.units
    a, b = U[i]["a"], U[j]["b"]
    deps = [list(x) for x in U[i]["deps"]] + ([list(x) for x in U[j]["deps"]] if j != i else [])
    members = [i] if i == j else [i, j]
    return {"key": key, "doc_id": U[i]["doc_id"], "section_id": U[i]["section_id"], "a": a, "b": b, "deps": deps,
            "units": members, "score": float(max(u[m] for m in members)), "body_chars": b - a}


def compact_candidates(six, seeds, u, n):
    seen, out = set(), []
    for a in seeds:
        if six.ub[a] - six.ua[a] <= MAX:
            k = ("u", int(a))
            if k not in seen:
                seen.add(k)
                out.append(_cand(six, k, a, a, u))
        for i, j in six.pairs.get(a, []):
            k = ("p", int(i), int(j))
            if k not in seen:
                seen.add(k)
                out.append(_cand(six, k, i, j, u))
    for c in out:
        c["cost1"] = c["body_chars"] + sum(b - a for a, b in c["deps"])       # tie-break proxy: displayed chars
    out.sort(key=lambda c: (-c["score"], c["cost1"], c["key"]))
    return out[:n]


def chunk_candidate(six, ci, u):
    ch = six.hx.chunks[ci]
    a, b = int(ch["span"][0]), int(ch["span"][1])
    D = six.docs[ch["doc_id"]]
    members = D.units_in(*PK.trim(D.text, a, b))
    return {"key": ("c", int(ci)), "doc_id": ch["doc_id"], "section_id": ch["section_id"], "a": a, "b": b, "deps": [],
            "units": members, "score": float(max((u[m] for m in members), default=0.0)), "body_chars": b - a}


def attach_docs(six):
    """six.docs: {doc_id: packer.Doc} (unit offsets + header paths), built once per set."""
    six.docs = {d: PK.Doc(six.cs.doc(d), ids, six.ua, six.ub, six.cs) for d, ids in six.by_doc.items()}
    for c in six.hx.chunks:
        if c["doc_id"] not in six.docs:
            six.docs[c["doc_id"]] = PK.Doc(six.cs.doc(c["doc_id"]), [], six.ua, six.ub, six.cs)
    return six
