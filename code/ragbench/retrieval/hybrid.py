"""HybridIndex: the public chunk retrieval service shared by HYBRID, LEX3 and AGENTIC5 (2026-09-24).

One index per CorpusSet holds exactly B2/B3's chunks (baselines.build_chunks with the same config and chunker
options), B2's BM25 (same tokenizer, same bm25s parameters, same "<breadcrumb>\\n<chunk>" index text) and B3's
L2-normalised bge-m3 vectors, read from B3's shared embedding cache under the SAME keys (model, digest, truncate,
num_ctx, text), so vectors B3 already computed are cache hits and nothing is embedded twice.

    hx = HybridIndex.open(corpus_set, cfg, client=client)          # load, or build (embed=False: cache only)
    hits = hx.search("question text", doc_ids=[...], k=40)         # list[Hit], best first
    hits.diagnostics                                               # scope, per-channel counts, missing vectors

Ranking is separated from text construction: a Hit is one chunk (its identity is the chunk number, never the order
in which channels delivered it) with its verbatim text and span; evidence text is built later (ragbench.evidence).

Fusion: equal-weight reciprocal-rank fusion, k = 60. Each channel ranks the chunks in scope once, so a chunk makes at
most ONE contribution per channel: rrf = sum over channels of 1 / (60 + rank). Ties: best single-channel rank, then
chunk number (deterministic, independent of channel order).

Scope: `doc_ids` masks the SAME full index; BM25 statistics (idf, average length) are those of the whole set.

Vectors: document vectors are never computed at query time. Chunks without a vector are left out of the dense channel
and reported in hits.diagnostics["dense"] (missing_vectors_in_scope, missing_docs, complete=False). Only the query
vector may be embedded, through the client given, and only when no query_vector is passed.
"""
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..approaches import baselines as BL
from ..contracts import RetrievedItem
from ..normalize import normalize
from ..util import KVCache, atomic_write_json, read_json, read_jsonl, sha1_json

HYBRID_VERSION = 1
RRF_K = 60
CHANNELS = ("bm25", "dense")
SUBDIR = "_hybrid"                      # <index_root>/_hybrid/<set>-<fp8>/


# --------------------------------------------------------------------------------------------------------------
# identity
# --------------------------------------------------------------------------------------------------------------

def signature(cfg):
    """Everything that changes the chunks, the BM25 index or the vectors (not batch sizes)."""
    ch = cfg.chunking
    e = cfg.embedding
    return {"v": HYBRID_VERSION, "chunk_chars": int(ch["chunk_chars"]), "overlap_chars": int(ch["overlap_chars"]),
            "chunker": BL.DEFAULTS["chunker"], "index_with_path": BL.DEFAULTS["index_with_path"],
            "tokenizer": BL.DEFAULTS["tokenizer"], "bm25": BL.DEFAULTS["bm25"],
            "embed_model": cfg.models["embed"], "embed_digest": cfg.models.get("embed_digest"),
            "truncate": bool(e["truncate"]), "num_ctx": e["num_ctx"]}


def fingerprint(corpus_set, cfg):
    return sha1_json({"corpus": corpus_set.fingerprint(), "sig": signature(cfg)})


def index_dir_for(corpus_set, cfg):
    return cfg.path("index_root") / SUBDIR / ("%s-%s" % (corpus_set.name, fingerprint(corpus_set, cfg)[:8]))


def embed_cache_path(cfg):
    """B3's shared cache (baselines.DEFAULTS["shared_embed_cache"] is True)."""
    return cfg.path("index_root") / "_embed_cache" / "embeddings.sqlite"


def embed_text_key(cfg, text):
    """Exactly VectorRetrieval._text_key: a B3 vector of the same text is a cache hit."""
    e = cfg.embedding
    return sha1_json({"m": cfg.models["embed"], "d": cfg.models.get("embed_digest"),
                      "t": bool(e["truncate"]), "c": e["num_ctx"], "x": text})


def index_text(chunk):
    return BL._ChunkRetrieval.index_text(chunk)


def query_tokens(text):
    """B2's query tokens: tokenize(normalize(text)), first occurrence of each."""
    seen, out = set(), []
    for t in BL.tokenize(normalize(text or ""), BL.DEFAULTS["tokenizer"]):
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


# --------------------------------------------------------------------------------------------------------------
# fusion
# --------------------------------------------------------------------------------------------------------------

def rrf_fuse(rankings, k=RRF_K, depth=None):
    """rankings: {channel: [candidate id, best first]} -> [(id, rrf, {channel: rank|None})], best first.

    One contribution per channel per candidate: a candidate listed twice by one channel counts at its best rank only.
    Ties: best single-channel rank, then the id itself (ids must be orderable), so the result does not depend on the
    order of the channels. Ranks are 1-based."""
    chans = list(rankings)
    ranks = {}
    for ch in chans:
        seen = set()
        lst = rankings[ch] if depth is None else list(rankings[ch])[:depth]
        r = 0
        for cid in lst:
            if cid in seen:
                continue
            seen.add(cid)
            r += 1
            ranks.setdefault(cid, {c: None for c in chans})[ch] = r
    out = []
    for cid, rk in ranks.items():
        s = sum(1.0 / (k + r) for r in rk.values() if r is not None)
        out.append((cid, s, rk))
    out.sort(key=lambda t: (-t[1], min(r for r in t[2].values() if r is not None), t[0]))
    return out


# --------------------------------------------------------------------------------------------------------------
# hits
# --------------------------------------------------------------------------------------------------------------

@dataclass
class Hit:
    doc_id: str
    section_id: str
    unit_ids: list
    text: str                       # the chunk, verbatim: Document.text[span[0]:span[1]]
    span: list                      # [start, end) in Document.text
    ranks: dict                     # {"bm25": rank|None, "dense": rank|None}, 1-based, within the scope
    scores: dict                    # {"bm25": score|None, "dense": cosine|None}
    rrf: float
    chunk: int = None               # chunk number in the index (the candidate's identity)
    section_ids: list = field(default_factory=list)
    partial_unit_ids: list = field(default_factory=list)
    mode: str = None                # packed | window (the chunker's)

    def to_item(self, rank=None):
        """RetrievedItem for pack_evidence (kind chunk, score = rrf); ranks/scores kept in meta."""
        return RetrievedItem(text=self.text, doc_id=self.doc_id, section_id=self.section_id,
                             unit_ids=list(self.unit_ids), kind="chunk", score=round(float(self.rrf), 6), rank=rank,
                             span=list(self.span), meta={"chunk": self.chunk, "mode": self.mode,
                                                         "partial_unit_ids": list(self.partial_unit_ids),
                                                         "ranks": dict(self.ranks), "scores": dict(self.scores),
                                                         "rrf": round(float(self.rrf), 6)})

    def to_dict(self):
        return {"doc_id": self.doc_id, "section_id": self.section_id, "unit_ids": list(self.unit_ids),
                "span": list(self.span), "ranks": dict(self.ranks), "scores": dict(self.scores),
                "rrf": round(float(self.rrf), 6), "chunk": self.chunk, "mode": self.mode, "chars": len(self.text)}


class HitList(list):
    """A list of Hits with the search's diagnostics attached (hits.diagnostics)."""

    def __init__(self, hits=(), diagnostics=None):
        super().__init__(hits)
        self.diagnostics = diagnostics or {}


# --------------------------------------------------------------------------------------------------------------
# the index
# --------------------------------------------------------------------------------------------------------------

def _log(progress, msg, **kw):
    if progress is None:
        return
    if callable(progress):
        progress(dict(kw, msg=msg))
    else:
        print(msg, file=sys.stderr, flush=True)


def _chunks(corpus_set, cfg, index_dir):
    """baselines._ChunkRetrieval._chunks, verbatim in effect: chunks.jsonl reused when it matches the corpus."""
    path = Path(index_dir) / "chunks.jsonl"
    meta_path = Path(index_dir) / "chunks_meta.json"
    meta = read_json(meta_path)
    if path.is_file() and meta and meta.get("corpus_fingerprint") == corpus_set.fingerprint():
        chunks = read_jsonl(path)
        if len(chunks) == meta.get("n_chunks") and BL.chunks_fingerprint(chunks) == meta.get("chunks_fingerprint"):
            return chunks, meta
    t0 = time.perf_counter()
    chunks = BL.build_chunks(corpus_set, cfg, BL.DEFAULTS["chunker"])
    BL._write_jsonl_atomic(path, chunks)
    lens = np.array([len(c["text"]) for c in chunks]) if chunks else np.zeros(1)
    meta = {"corpus_set": corpus_set.name, "corpus_fingerprint": corpus_set.fingerprint(), "n_chunks": len(chunks),
            "chunks_fingerprint": BL.chunks_fingerprint(chunks), "chunking": cfg.chunking,
            "chunker": BL.DEFAULTS["chunker"], "chars_mean": round(float(lens.mean()), 1),
            "chars_p50": int(np.median(lens)), "chars_max": int(lens.max()),
            "n_window": sum(1 for c in chunks if c["mode"] == "window"),
            "n_packed": sum(1 for c in chunks if c["mode"] == "packed"),
            "chunk_ms": round((time.perf_counter() - t0) * 1000.0, 1)}
    atomic_write_json(meta_path, meta, indent=1)
    return chunks, meta


def _bm25(chunks, cmeta, index_dir):
    import bm25s
    index_dir = Path(index_dir)
    meta = read_json(index_dir / "bm25_meta.json")
    if meta and (index_dir / "bm25").is_dir() and meta.get("chunks_fingerprint") == cmeta["chunks_fingerprint"]:
        return bm25s.BM25.load(str(index_dir / "bm25"), show_progress=False), meta
    t0 = time.perf_counter()
    tokens = [BL.tokenize(index_text(c), BL.DEFAULTS["tokenizer"]) for c in chunks]
    p = BL.DEFAULTS["bm25"]
    retr = bm25s.BM25(k1=p["k1"], b=p["b"], method=p["method"])
    retr.index(tokens, show_progress=False)
    retr.save(str(index_dir / "bm25"), show_progress=False)
    meta = {"n_chunks": len(chunks), "chunks_fingerprint": cmeta["chunks_fingerprint"], "vocab": len(retr.vocab_dict),
            "tokens_mean": round(float(np.mean([len(t) for t in tokens])) if tokens else 0.0, 1),
            "empty_chunks": sum(1 for t in tokens if not t), "bm25": p, "tokenizer": BL.DEFAULTS["tokenizer"],
            "index_ms": round((time.perf_counter() - t0) * 1000.0, 1)}
    atomic_write_json(index_dir / "bm25_meta.json", meta, indent=1)
    return retr, meta


def embed_missing(cfg, texts, client, *, batch_size=None, commit_every=1, progress=None, label="", max_batches=None,
                  cache=None):
    """Embed every text whose vector is not yet in the shared cache, in batches, committing to the cache every
    `commit_every` batches (one sqlite transaction). Resumable: an interrupted run loses at most the uncommitted
    batches. -> stats dict. Raises whatever the client raises (after committing what was done)."""
    own = cache is None
    cache = cache or KVCache(embed_cache_path(cfg))
    stats = {"unique_texts": 0, "todo": 0, "embedded_now": 0, "batches": 0, "commits": 0, "tokens_now": 0,
             "wall_ms_now": 0.0}
    try:
        uniq = {}
        for t in texts:
            uniq.setdefault(embed_text_key(cfg, t), t)
        stats["unique_texts"] = len(uniq)
        missing = cache.missing(["v:" + k for k in uniq])
        todo = [(k[2:], uniq[k[2:]]) for k in missing]
        stats["todo"] = len(todo)
        bs = int(batch_size or cfg.embedding["batch_size"])
        commit_every = max(1, int(commit_every))
        pend_a, pend_o = [], []
        t0 = time.perf_counter()
        _log(progress, "[prebuild %s] %d unique texts, %d cached, %d to embed" % (
            label, len(uniq), len(uniq) - len(todo), len(todo)), phase="start", todo=len(todo), unique=len(uniq))

        def flush():
            if pend_a:
                cache.put_many(arrays=pend_a, objs=pend_o)
                stats["commits"] += 1
                pend_a.clear()
                pend_o.clear()

        try:
            for j in range(0, len(todo), bs):
                if max_batches is not None and stats["batches"] >= max_batches:
                    break
                batch = todo[j:j + bs]
                er = client.embed([t for _, t in batch], purpose="index-embed", batch_size=bs)
                if len(er.vectors) != len(batch):
                    raise RuntimeError("embed returned %d vectors for %d texts" % (len(er.vectors), len(batch)))
                for (k, _), v in zip(batch, er.vectors):
                    pend_a.append(("v:" + k, np.asarray(v, dtype=np.float32)))
                    pend_o.append(("c:" + k, {"tokens": er.prompt_tokens / len(batch), "wall_ms": er.wall_ms / len(batch)}))
                stats["batches"] += 1
                stats["embedded_now"] += len(batch)
                stats["tokens_now"] += int(er.prompt_tokens or 0)
                stats["wall_ms_now"] += float(er.wall_ms or 0.0)
                if stats["batches"] % commit_every == 0:
                    flush()
                done = stats["embedded_now"]
                el = time.perf_counter() - t0
                rate = done / el if el > 0 else 0.0
                eta = (len(todo) - done) / rate if rate > 0 else float("nan")
                _log(progress, "[prebuild %s] embedded %d/%d (%.1f/s, eta %.0fs)" % (label, done, len(todo), rate, eta),
                     phase="batch", done=done, todo=len(todo), rate=rate, eta_s=eta)
        finally:
            flush()
        stats["wall_ms_now"] = round(stats["wall_ms_now"], 1)
        return stats
    finally:
        if own:
            cache.close()


class HybridIndex:
    """See the module docstring. Attributes: chunks (B2/B3 chunk dicts), doc_ids, doc_pos (doc position per chunk),
    bm25 (bm25s.BM25), E (n x d float32, L2-normalised, zero rows where missing), has_vec (bool per chunk), meta."""

    def __init__(self, corpus_set, cfg, index_dir, chunks, bm25, E, has_vec, meta, client=None):
        self.corpus_set = corpus_set
        self.cfg = cfg
        self.index_dir = Path(index_dir)
        self.chunks = chunks
        self.bm25 = bm25
        self.E = E
        self.has_vec = has_vec
        self.meta = meta
        self.client = client
        doc_ids = corpus_set.doc_ids()
        pos = {d: i for i, d in enumerate(doc_ids)}
        self.doc_ids = doc_ids
        self.doc_pos = np.array([pos[c["doc_id"]] for c in chunks], dtype=np.int32)
        self.last_diagnostics = None

    # -- construction ---------------------------------------------------------------------------------------------
    @classmethod
    def build(cls, corpus_set, cfg, client=None, embed=True, index_dir=None, progress=None, batch_size=None,
              commit_every=1, max_batches=None):
        """Chunks + BM25 (always) + vectors: with embed=True and a client, every chunk text missing from the shared
        cache is embedded first (index time, resumable); otherwise vectors come from the cache only and missing ones
        are recorded. Persists under <index_root>/_hybrid/<set>-<fp8>/."""
        index_dir = Path(index_dir or index_dir_for(corpus_set, cfg))
        index_dir.mkdir(parents=True, exist_ok=True)
        t0 = time.perf_counter()
        chunks, cmeta = _chunks(corpus_set, cfg, index_dir)
        bm, bmeta = _bm25(chunks, cmeta, index_dir)
        texts = [index_text(c) for c in chunks]
        estats = None
        if embed and client is not None and chunks:
            estats = embed_missing(cfg, texts, client, batch_size=batch_size, commit_every=commit_every,
                                   progress=progress, label=corpus_set.name, max_batches=max_batches)
        E, has_vec, vmeta = cls._assemble_vectors(cfg, chunks, texts, index_dir, cmeta)
        if estats:
            vmeta["embed_stats"] = estats
            atomic_write_json(index_dir / "vector_meta.json", vmeta, indent=1)
        meta = cls._write_meta(corpus_set, cfg, index_dir, cmeta, bmeta, vmeta, round((time.perf_counter() - t0) * 1000.0, 1))
        _log(progress, "[prebuild %s] index %s: %d chunks, %d with vectors, %d missing" % (
            corpus_set.name, index_dir, len(chunks), int(has_vec.sum()), int((~has_vec).sum())), phase="done")
        return cls(corpus_set, cfg, index_dir, chunks, bm, E, has_vec, meta, client=client)

    @classmethod
    def load(cls, corpus_set, cfg, client=None, refresh_vectors=True, index_dir=None):
        """Load a built index; raises FileNotFoundError / RuntimeError when absent or stale. With refresh_vectors,
        chunks without a vector are looked up in the shared cache again (a prebuild may have filled them)."""
        import bm25s
        index_dir = Path(index_dir or index_dir_for(corpus_set, cfg))
        meta = read_json(index_dir / "hybrid_meta.json")
        if not meta:
            raise FileNotFoundError("no hybrid index at %s (build it: python -m ragbench.retrieval.prebuild --set %s)"
                                    % (index_dir, corpus_set.name))
        if meta.get("fingerprint") != fingerprint(corpus_set, cfg):
            raise RuntimeError("hybrid index at %s is stale (fingerprint mismatch)" % index_dir)
        chunks = read_jsonl(index_dir / "chunks.jsonl")
        if len(chunks) != meta.get("n_chunks") or BL.chunks_fingerprint(chunks) != meta.get("chunks_fingerprint"):
            raise RuntimeError("hybrid index at %s: chunks.jsonl does not match its meta" % index_dir)
        bm = bm25s.BM25.load(str(index_dir / "bm25"), show_progress=False)
        E = np.load(index_dir / "embeddings.npy")
        has_vec = np.load(index_dir / "has_vec.npy")
        if E.shape[0] != len(chunks) or has_vec.shape[0] != len(chunks):
            raise RuntimeError("hybrid index at %s: %d vectors for %d chunks" % (index_dir, E.shape[0], len(chunks)))
        if refresh_vectors and not bool(has_vec.all()):
            cmeta = read_json(index_dir / "chunks_meta.json")
            n_before = int(has_vec.sum())
            E2, has2, vmeta = cls._assemble_vectors(cfg, chunks, [index_text(c) for c in chunks], index_dir, cmeta)
            if int(has2.sum()) > n_before:
                E, has_vec = E2, has2
                meta = cls._write_meta(corpus_set, cfg, index_dir, cmeta, read_json(index_dir / "bm25_meta.json"),
                                       vmeta, meta.get("build_ms"))
        return cls(corpus_set, cfg, index_dir, chunks, bm, E, has_vec, meta, client=client)

    @classmethod
    def open(cls, corpus_set, cfg, client=None, embed=False, **kw):
        """load(), or build() when there is no usable index (embed=False: never calls the model)."""
        try:
            return cls.load(corpus_set, cfg, client=client, index_dir=kw.get("index_dir"))
        except (FileNotFoundError, RuntimeError):
            return cls.build(corpus_set, cfg, client=client, embed=embed, **kw)

    @staticmethod
    def _assemble_vectors(cfg, chunks, texts, index_dir, cmeta):
        cache = KVCache(embed_cache_path(cfg))
        try:
            got, dim = {}, None
            for t in texts:
                k = embed_text_key(cfg, t)
                if k in got:
                    continue
                v = cache.get_array("v:" + k)
                got[k] = v
                if v is not None and dim is None:
                    dim = int(v.shape[0])
        finally:
            cache.close()
        n = len(chunks)
        dim = dim or 0
        E = np.zeros((n, dim), dtype=np.float32)
        has_vec = np.zeros(n, dtype=bool)
        for i, t in enumerate(texts):
            v = got[embed_text_key(cfg, t)]
            if v is None:
                continue
            if v.shape[0] != dim:
                raise RuntimeError("embedding cache holds vectors of different sizes (%d vs %d)" % (v.shape[0], dim))
            E[i] = v
            has_vec[i] = True
        norms = np.linalg.norm(E, axis=1, keepdims=True)       # exactly B3's normalisation (bit-identical vectors)
        E = E / np.where(norms > 0, norms, 1.0)
        index_dir = Path(index_dir)
        for name, arr in (("embeddings.npy", E), ("has_vec.npy", has_vec)):
            tmp = index_dir / (".%s.tmp-%d.npy" % (name[:-4], os.getpid()))
            np.save(tmp, arr)
            os.replace(tmp, index_dir / name)
        missing_docs = {}
        for c, ok in zip(chunks, has_vec):
            if not ok:
                missing_docs[c["doc_id"]] = missing_docs.get(c["doc_id"], 0) + 1
        vmeta = {"n_chunks": n, "chunks_fingerprint": cmeta["chunks_fingerprint"], "dim": dim,
                 "unique_texts": len(got), "with_vector": int(has_vec.sum()), "missing": int(n - has_vec.sum()),
                 "missing_docs": missing_docs, "model": cfg.models["embed"],
                 "embed_digest": cfg.models.get("embed_digest"), "cache": str(embed_cache_path(cfg))}
        atomic_write_json(index_dir / "vector_meta.json", vmeta, indent=1)
        return E, has_vec, vmeta

    @staticmethod
    def _write_meta(corpus_set, cfg, index_dir, cmeta, bmeta, vmeta, build_ms):
        meta = {"version": HYBRID_VERSION, "fingerprint": fingerprint(corpus_set, cfg), "signature": signature(cfg),
                "corpus_set": corpus_set.name, "corpus_fingerprint": corpus_set.fingerprint(),
                "n_chunks": cmeta["n_chunks"], "chunks_fingerprint": cmeta["chunks_fingerprint"],
                "bm25_vocab": (bmeta or {}).get("vocab"), "dim": vmeta["dim"], "with_vector": vmeta["with_vector"],
                "missing_vectors": vmeta["missing"], "dense_complete": vmeta["missing"] == 0,
                "build_ms": build_ms, "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
        atomic_write_json(Path(index_dir) / "hybrid_meta.json", meta, indent=1)
        return meta

    # -- state ------------------------------------------------------------------------------------------------
    @property
    def n(self):
        return len(self.chunks)

    @property
    def dense_complete(self):
        return bool(self.has_vec.all())

    def scope_mask(self, doc_ids=None):
        """-> (bool mask over chunks, info). doc_ids None or "corpus" = every chunk."""
        if doc_ids is None or doc_ids == "corpus":
            return np.ones(self.n, dtype=bool), {"scope": "corpus", "scope_docs": len(self.doc_ids)}
        want = set(doc_ids)
        pos = [i for i, d in enumerate(self.doc_ids) if d in want]
        unknown = sorted(want - set(self.doc_ids))
        return np.isin(self.doc_pos, np.array(pos, dtype=np.int32)), {"scope": "docs", "scope_docs": len(pos),
                                                                       "scope_docs_unknown": unknown}

    # -- query ------------------------------------------------------------------------------------------------
    def bm25_scores(self, query_text):
        toks = query_tokens(query_text)
        known = [t for t in toks if t in self.bm25.vocab_dict]
        if not known or not self.n:
            return np.zeros(self.n, dtype=np.float32), {"query_tokens": toks, "query_tokens_in_vocab": len(known)}
        return (np.asarray(self.bm25.get_scores(known), dtype=np.float32),
                {"query_tokens": toks, "query_tokens_in_vocab": len(known)})

    def query_vector(self, query_text, client=None):
        """Embed the QUESTION only (never documents); None when no client is available."""
        client = client or self.client
        if client is None:
            return None, {"query_embedded": False}
        er = client.embed([normalize(query_text or "")], purpose="query-embed")
        return np.asarray(er.vectors[0], dtype=np.float32), {"query_embedded": True, "query_embed_tokens": er.prompt_tokens}

    def search(self, query_text, doc_ids=None, k=40, channels=CHANNELS, query_vector=None, depth=None, client=None):
        """-> HitList (list of Hit, best first, at most k) with .diagnostics. Each channel ranks the chunks in scope
        and contributes its top `depth` (default k) to equal-weight RRF (k=60)."""
        t0 = time.perf_counter()
        channels = tuple(channels)
        bad = [c for c in channels if c not in CHANNELS]
        if bad:
            raise ValueError("unknown channel(s) %s; choose from %s" % (bad, CHANNELS))
        depth = int(depth or k)
        mask, sinfo = self.scope_mask(doc_ids)
        diag = {"channels": list(channels), "k": int(k), "depth": depth, "rrf_k": RRF_K, "chunks_total": self.n,
                "chunks_in_scope": int(mask.sum()), "warnings": []}
        diag.update(sinfo)
        order_idx = np.arange(self.n)
        rankings, raw = {}, {}
        if "bm25" in channels:
            s, info = self.bm25_scores(query_text)
            valid = mask & (s > 0.0)                         # B2's floor (min_score 0.0)
            order = np.lexsort((order_idx, -s))
            order = order[valid[order]][:depth]
            rankings["bm25"] = [int(i) for i in order]
            raw["bm25"] = s
            diag["bm25"] = dict(info, scored=int(valid.sum()), returned=len(order))
        if "dense" in channels:
            dd = {}
            missing_scope = mask & ~self.has_vec
            n_missing = int(missing_scope.sum())
            dd["missing_vectors_in_scope"] = n_missing
            dd["complete"] = n_missing == 0
            if n_missing:
                md = {}
                for i in np.nonzero(missing_scope)[0]:
                    d = self.chunks[int(i)]["doc_id"]
                    md[d] = md.get(d, 0) + 1
                dd["missing_docs"] = dict(sorted(md.items(), key=lambda kv: (-kv[1], kv[0]))[:20])
                diag["warnings"].append("dense: %d chunk(s) in scope have no vector (not embedded at query time; run "
                                        "python -m ragbench.retrieval.prebuild --set %s)" % (n_missing, self.corpus_set.name))
            q = None
            if self.E.shape[1] == 0 or not bool((mask & self.has_vec).any()):
                dd["skipped"] = "no document vectors in scope"
                diag["warnings"].append("dense: skipped (no chunk in scope has a vector)")
            elif query_vector is not None:
                q = np.asarray(query_vector, dtype=np.float32)
                dd["query_vector"] = "given"
            else:
                q, qi = self.query_vector(query_text, client)
                dd.update(qi)
                dd["query_vector"] = "embedded" if q is not None else None
                if q is None:
                    dd["skipped"] = "no query vector and no client"
                    diag["warnings"].append("dense: skipped (no query vector and no client)")
            if q is not None:
                if q.shape[0] != self.E.shape[1]:
                    raise ValueError("query vector has %d dims, the index %d" % (q.shape[0], self.E.shape[1]))
                nq = float(np.linalg.norm(q))
                q = q / (nq if nq > 0 else 1.0)
                s = (self.E @ q).astype(np.float32)
                valid = mask & self.has_vec
                order = np.lexsort((order_idx, -s))
                order = order[valid[order]][:depth]
                rankings["dense"] = [int(i) for i in order]
                raw["dense"] = s
                dd["scored"] = int(valid.sum())
                dd["returned"] = len(order)
            diag["dense"] = dd
        fused = rrf_fuse(rankings, k=RRF_K)
        hits = []
        for i, rrf, rk in fused[:int(k)]:
            c = self.chunks[i]
            ranks = {ch: rk.get(ch) for ch in CHANNELS}
            scores = {ch: (round(float(raw[ch][i]), 6) if ch in raw and rk.get(ch) is not None else None) for ch in CHANNELS}
            hits.append(Hit(doc_id=c["doc_id"], section_id=c["section_id"], unit_ids=list(c["unit_ids"]), text=c["text"],
                            span=list(c["span"]), ranks=ranks, scores=scores, rrf=rrf, chunk=int(i),
                            section_ids=list(c.get("section_ids") or []), partial_unit_ids=list(c["partial_unit_ids"]),
                            mode=c.get("mode")))
        diag["fused"] = len(fused)
        diag["returned"] = len(hits)
        diag["both_channels"] = sum(1 for h in hits if all(h.ranks.get(ch) for ch in rankings))
        diag["ms"] = round((time.perf_counter() - t0) * 1000.0, 1)
        self.last_diagnostics = diag
        return HitList(hits, diag)

    def hit_for_chunk(self, i):
        c = self.chunks[int(i)]
        return Hit(doc_id=c["doc_id"], section_id=c["section_id"], unit_ids=list(c["unit_ids"]), text=c["text"],
                   span=list(c["span"]), ranks={ch: None for ch in CHANNELS}, scores={ch: None for ch in CHANNELS},
                   rrf=0.0, chunk=int(i), section_ids=list(c.get("section_ids") or []),
                   partial_unit_ids=list(c["partial_unit_ids"]), mode=c.get("mode"))

    def status(self):
        return {"index_dir": str(self.index_dir), "corpus_set": self.corpus_set.name, "n_chunks": self.n,
                "with_vector": int(self.has_vec.sum()), "missing_vectors": int((~self.has_vec).sum()),
                "dense_complete": self.dense_complete, "fingerprint": self.meta.get("fingerprint")}
