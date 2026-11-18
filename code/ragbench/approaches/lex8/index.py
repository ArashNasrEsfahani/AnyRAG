"""LEX8 index-time artifacts for one corpus set (question-independent; built once, offline).

    <root>/<set>/units.jsonl      compact source units (units.py), with parents
    <root>/<set>/bm25/            bm25s over units.index_text (the B2 tokenizer / k1 / b / method)
    <root>/<set>/m3.npz           bge-m3 dense (f16) + sparse lexical weights (CSR) of units.index_text
    <root>/<set>/manifest.json    identity manifests (identity.py) of the unit m3 artifact
    <root>/chunks/<set>.npz       chunk-side bge-m3 dense + sparse (from lex_research/m3, NOT re-encoded)
    <root>/chunks/<set>.json      its manifest (texts = hybrid.index_text of the shared chunks; token counts checked)
Every load re-derives the texts / rows and refuses on any identity mismatch.
"""
import json
import time
from pathlib import Path

import numpy as np

from ...approaches import baselines as BL
from ...retrieval import hybrid as H
from ...retrieval import m3multi as M3
from . import identity as ID
from . import units as U

INDEX_VERSION = "lex8-index-v1"
UNIT_PRE = {"index_text": "lex8.units.index_text", "units": U.UNITS_VERSION, "target": U.TARGET, "max": U.MAX,
            "query": "normalize"}
CHUNK_PRE = {"index_text": "hybrid.index_text (breadcrumb + chunk)", "query": "normalize"}
VOCAB = 250002


def _csr(dicts):
    return M3.SparseMatrix(dicts, VOCAB).M


def _save_bm25(texts, path):
    import bm25s
    p = BL.DEFAULTS["bm25"]
    tokens = [BL.tokenize(t, BL.DEFAULTS["tokenizer"]) for t in texts]
    retr = bm25s.BM25(k1=p["k1"], b=p["b"], method=p["method"])
    retr.index(tokens, show_progress=False)
    retr.save(str(path), show_progress=False)
    return retr


def build_units(corpus_set, hx, root, encoder, batch_size=32, progress=print):
    """units + BM25 + bge-m3 (GPU when free) for one set; resumable per stage."""
    d = Path(root) / corpus_set.name
    d.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    uf = d / "units.jsonl"
    if uf.exists():
        units = [json.loads(l) for l in open(uf, encoding="utf-8")]
    else:
        units = U.build_set_units(corpus_set, hx.chunks)
        with open(str(uf) + ".tmp", "w", encoding="utf-8", newline="\n") as f:
            for u in units:
                f.write(json.dumps(u, ensure_ascii=False) + "\n")
        Path(str(uf) + ".tmp").replace(uf)
    texts = [U.index_text(corpus_set.doc(u["doc_id"]), u) for u in units]
    rows = [(u["doc_id"], u["a"], u["b"]) for u in units]
    t_units = time.perf_counter() - t0
    if not (d / "bm25" / "params.index.json").exists():
        t = time.perf_counter()
        _save_bm25(texts, d / "bm25")
        progress("bm25 %s: %d units in %.1f s" % (corpus_set.name, len(units), time.perf_counter() - t))
    if not (d / "m3.npz").exists():
        encoder.load()
        order = np.argsort([len(x) for x in texts], kind="stable")       # length-sorted batches (less padding)
        dense = np.zeros((len(texts), 1024), np.float16)
        sparse = [None] * len(texts)
        ntok = np.zeros(len(texts), np.int32)
        t = time.perf_counter()
        for s in range(0, len(order), batch_size):
            js = order[s:s + batch_size]
            res = encoder.encode([texts[j] for j in js], max_length=M3.M3_DEFAULTS["max_length"],
                                 batch_size=batch_size, colbert=False)
            for j, r in zip(js, res):
                dense[j] = r["dense"].astype(np.float16)
                sparse[j] = r["sparse"]
                ntok[j] = r["tokens"]
        secs = time.perf_counter() - t
        S = _csr(sparse)
        np.savez(d / "m3.npz", dense=dense, data=S.data, indices=S.indices, indptr=S.indptr, shape=np.asarray(S.shape),
                 tokens=ntok)
        enc = ID.encoder_info(encoder.model_name, getattr(encoder, "revision", None), M3.M3_DEFAULTS["max_length"],
                              encoder.fp16, M3.M3_VERSION)
        man = ID.manifest("lex8-units-m3", texts, rows, enc, UNIT_PRE)
        man.update({"device": encoder.device, "encode_s": round(secs, 1), "tokens_mean": float(ntok.mean()),
                    "at_cap": int((ntok >= M3.M3_DEFAULTS["max_length"]).sum()), "index_version": INDEX_VERSION})
        (d / "manifest.json").write_text(json.dumps(man, indent=1), encoding="utf-8")
        progress("m3 units %s: %d units encoded in %.0f s on %s (mean %.0f tokens, at cap %d)" % (
            corpus_set.name, len(units), secs, encoder.device, ntok.mean(), man["at_cap"]))
    return {"units": len(units), "t_units_s": round(t_units, 2)}


def build_chunk_m3(corpus_set, hx, pkl_path, root, revision, progress=print):
    """chunk-side m3 artifact from the lex_research pkl (no re-encode) + a manifest. The pkl stores per-chunk token
    counts: the recomputed bge-m3 token count of hybrid.index_text(chunk) must match every row (text check)."""
    import pickle
    from transformers import AutoTokenizer
    out = Path(root) / "chunks"
    out.mkdir(parents=True, exist_ok=True)
    m = pickle.load(open(pkl_path, "rb"))
    if list(m["chunks"]) != list(range(hx.n)):
        raise ID.IdentityError("pkl rows are not the shared chunks 0..n-1")
    texts = [H.index_text(c) for c in hx.chunks]
    rows = [(c["doc_id"], c["span"][0], c["span"][1]) for c in hx.chunks]
    enc = M3.M3Encoder()
    snaps = Path(M3.default_hf_home()) / "hub" / "models--BAAI--bge-m3" / "snapshots"
    tok = AutoTokenizer.from_pretrained(str(snaps / revision))
    n = [len(x) for x in tok(texts, truncation=True, max_length=M3.M3_DEFAULTS["max_length"])["input_ids"]]
    bad = [i for i, (a, b) in enumerate(zip(n, m["tokens"])) if int(a) != int(b)]
    if bad:
        raise ID.IdentityError("%s: %d chunks whose token count differs from the pkl (first %s)" % (
            corpus_set.name, len(bad), bad[:5]))
    S = _csr(m["sparse"])
    np.savez(out / ("%s.npz" % corpus_set.name), dense=m["dense_hf"].astype(np.float16), data=S.data,
             indices=S.indices, indptr=S.indptr, shape=np.asarray(S.shape))
    man = ID.manifest("lex8-chunks-m3", texts, rows,
                      ID.encoder_info(enc.model_name, revision, M3.M3_DEFAULTS["max_length"], True, M3.M3_VERSION),
                      CHUNK_PRE)
    man.update({"source": str(pkl_path), "token_count_check": "all %d rows match" % len(texts),
                "revision_note": "inferred: the only complete local snapshot (M3Encoder never downloads when one "
                                 "exists)", "device": m.get("device")})
    (out / ("%s.json" % corpus_set.name)).write_text(json.dumps(man, indent=1), encoding="utf-8")
    progress("chunk m3 %s: %d rows, token counts match, manifest written" % (corpus_set.name, len(texts)))


class _M3:
    def __init__(self, npz):
        from scipy.sparse import csr_matrix
        z = np.load(npz)
        self.dense = z["dense"].astype(np.float32)
        self.S = csr_matrix((z["data"], z["indices"], z["indptr"]), shape=tuple(int(x) for x in z["shape"]))

    def sparse_scores(self, q):
        qv = np.zeros(self.S.shape[1], np.float32)
        for t, w in q.items():
            qv[int(t)] = w
        return np.asarray(self.S @ qv).ravel()


class SetIndex:
    """Both granularities of one set, identity-checked."""

    def __init__(self, corpus_set, hx, root, encoder_revision=None):
        import bm25s
        root = Path(root)
        self.cs, self.hx = corpus_set, hx
        cm = json.loads((root / "chunks" / ("%s.json" % corpus_set.name)).read_text(encoding="utf-8"))
        ID.check(cm, [H.index_text(c) for c in hx.chunks], [(c["doc_id"], c["span"][0], c["span"][1]) for c in hx.chunks],
                 preprocess=CHUNK_PRE)
        self.chunk_m3 = _M3(root / "chunks" / ("%s.npz" % corpus_set.name))
        d = root / corpus_set.name
        self.units = [json.loads(l) for l in open(d / "units.jsonl", encoding="utf-8")]
        um = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
        texts = [U.index_text(corpus_set.doc(u["doc_id"]), u) for u in self.units]
        ID.check(um, texts, [(u["doc_id"], u["a"], u["b"]) for u in self.units], preprocess=UNIT_PRE)
        if encoder_revision is not None:
            for m in (cm, um):
                if m["encoder"]["revision"] != encoder_revision:
                    raise ID.IdentityError("encoder revision %s != artifact %s" % (encoder_revision,
                                                                                   m["encoder"]["revision"]))
        self.manifests = {"chunks": cm, "units": um}
        self.unit_m3 = _M3(d / "m3.npz")
        self.unit_bm25 = bm25s.BM25.load(str(d / "bm25"), show_progress=False)
        self.nu = len(self.units)
        self.ua = np.asarray([u["a"] for u in self.units], np.int64)
        self.ub = np.asarray([u["b"] for u in self.units], np.int64)
        docs = sorted({u["doc_id"] for u in self.units})
        self.doc_ix = {dd: i for i, dd in enumerate(docs)}
        self.udoc = np.asarray([self.doc_ix[u["doc_id"]] for u in self.units], np.int64)
        mp = max(len(u["parents"]) for u in self.units) if self.units else 1
        self.upar = np.full((self.nu, mp), -1, np.int64)
        for i, u in enumerate(self.units):
            self.upar[i, :len(u["parents"])] = u["parents"]
        self.by_doc = {}
        for i, u in enumerate(self.units):
            self.by_doc.setdefault(u["doc_id"], []).append(i)
        self.pairs = {}
        for i, j in U.pairs(self.units):
            self.pairs.setdefault(i, []).append((i, j))
            self.pairs.setdefault(j, []).append((i, j))

    def unit_bm25_scores(self, query_text):
        toks = [w for w in H.query_tokens(query_text) if w in self.unit_bm25.vocab_dict]
        if not toks:
            return np.zeros(self.nu, np.float32)
        return np.asarray(self.unit_bm25.get_scores(toks), np.float32)

    def unit_mask(self, doc_ids=None):
        if doc_ids is None:
            return np.ones(self.nu, bool)
        keep = np.zeros(len(self.doc_ix), bool)
        for d in doc_ids:
            if d in self.doc_ix:
                keep[self.doc_ix[d]] = True
        return keep[self.udoc]
