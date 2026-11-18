"""LEX9 query-time features (no text-generating model, no cross-encoder): four channels over the shared chunks, the
union of their top-40 lists, and a fixed feature vector per (question, candidate chunk).

Channels (scope mask; REAL scores for every union member):
    bm25    the shared HybridIndex B2 BM25 (breadcrumb + chunk)
    dense   bge-m3 dense (lex8 chunk m3 artifact)
    sparse  bge-m3 lexical weights (same artifact)
    synth   the LEX8 synthetic-question BM25 field lifted to chunks: chunk score = max BM25 score of its own
            synthetic questions (exclude_q removes one question = the training query itself; leakage control)
Features: FEATURES below (the order of the columns). Everything is computed from index-time artifacts + the question.
"""
import json
import re
from pathlib import Path

import numpy as np

from ...approaches import baselines as BL
from ...normalize import normalize
from ...retrieval import hybrid as H

FEATURES_VERSION = "lex9-features-v1"
CH = ("bm25", "dense", "sparse", "synth")
DEPTH = 40
RRF_K = 60
PER_CH = ("raw", "z", "rel", "logrank", "top")
FEATURES = ["%s_%s" % (c, f) for c in CH for f in PER_CH] + [
    "n_hit", "rrf", "maxmean3", "maxmean4", "combsum4",
    "doc_max", "doc_best", "doc_share", "doc_cent", "doc_cent_logrank",
    "log_chars", "pos", "is_table", "adj_max",
    "cov_fold", "cov_exact", "cov_crumb", "q_nums", "num_match",
    "lang_fa", "q_len"]

_DIG = str.maketrans("۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩", "01234567890123456789")
_NUM = re.compile(r"\d+(?:[.,٫٬]\d+)*")
_WORD = re.compile(r"\w+")


def numbers(text):
    t = (text or "").translate(_DIG)
    out = set()
    for m in _NUM.finditer(t):
        s = re.sub(r"[,٬]", "", m.group(0)).replace("٫", ".")
        s = s.rstrip(".")
        if s:
            out.add(s.lstrip("0") or "0")
    return out


def raw_words(text):
    return set(_WORD.findall((text or "").lower()))


def fold_tokens(text):
    return set(BL.tokenize(text or "", BL.DEFAULTS["tokenizer"]))


def top_ids(scores, valid, k):
    o = np.lexsort((np.arange(len(scores)), -np.asarray(scores)))
    return [int(i) for i in o[np.asarray(valid)[o]][:k]]


class SynthChunkField:
    """BM25 over a set's accepted synthetic questions; per-chunk max."""

    def __init__(self, d, n_chunks):
        import bm25s
        d = Path(d)
        self.retr = bm25s.BM25.load(str(d / "bm25q"), show_progress=False)
        self.qchunk = np.asarray([json.loads(l)["chunk"] for l in open(d / "questions.jsonl", encoding="utf-8")],
                                 np.int64)
        self.n = n_chunks

    def chunk_scores(self, qtext, exclude_q=None):
        out = np.zeros(self.n, np.float32)
        toks = [w for w in H.query_tokens(qtext) if w in self.retr.vocab_dict]
        if not toks:
            return out
        s = np.asarray(self.retr.get_scores(toks), np.float32)
        if exclude_q is not None:
            s[int(exclude_q)] = 0.0
        np.maximum.at(out, self.qchunk, s)
        return out


class SetFeatures:
    """Per-set static data (built once): chunk token sets, numbers, positions, doc centroids."""

    def __init__(self, hx, chunk_m3, synth_dir, language):
        self.hx, self.m3 = hx, chunk_m3
        self.n = hx.n
        self.lang_fa = 1.0 if language == "fa" else 0.0
        ch = hx.chunks
        self.tok = [fold_tokens(c["text"]) for c in ch]
        self.raw = [raw_words(c["text"]) for c in ch]
        self.crumb = [fold_tokens(c.get("breadcrumb") or "") for c in ch]
        self.nums = [numbers(c["text"]) for c in ch]
        self.chars = np.asarray([len((c["text"] or "").strip()) for c in ch], np.float64)
        self.table = np.asarray([1.0 if " | " in (c["text"] or "") else 0.0 for c in ch])
        docs = sorted({c["doc_id"] for c in ch})
        self.doc_ix = {d: i for i, d in enumerate(docs)}
        self.cdoc = np.asarray([self.doc_ix[c["doc_id"]] for c in ch], np.int64)
        by = {}
        for i, c in enumerate(ch):
            by.setdefault(c["doc_id"], []).append(i)
        order = {}
        for d, ids in by.items():
            ids = sorted(ids, key=lambda i: (ch[i]["span"][0], ch[i]["span"][1], i))
            for r, i in enumerate(ids):
                order[i] = (r, len(ids))
        self.pos = np.asarray([order[i][0] / max(1, order[i][1] - 1) for i in range(self.n)])
        self.prev = np.full(self.n, -1, np.int64)
        self.next = np.full(self.n, -1, np.int64)
        for d, ids in by.items():
            ids = sorted(ids, key=lambda i: (ch[i]["span"][0], ch[i]["span"][1], i))
            for a, b in zip(ids, ids[1:]):
                self.next[a], self.prev[b] = b, a
        dn = chunk_m3.dense / np.maximum(np.linalg.norm(chunk_m3.dense, axis=1, keepdims=True), 1e-9)
        cent = np.zeros((len(docs), dn.shape[1]), np.float64)
        np.add.at(cent, self.cdoc, dn)
        self.cent = (cent / np.maximum(np.linalg.norm(cent, axis=1, keepdims=True), 1e-9)).astype(np.float32)
        self.synth = SynthChunkField(synth_dir, self.n) if synth_dir and (Path(synth_dir) / "bm25q").exists() else None

    # ------------------------------------------------------------------------------------------------------------
    def channels(self, qtext, qm, mask, exclude_q=None):
        s_bm, _ = self.hx.bm25_scores(qtext)
        s_bm = np.asarray(s_bm, np.float32)
        s_dn = self.m3.dense @ np.asarray(qm["dense"], np.float32)
        s_sp = self.m3.sparse_scores(qm["sparse"])
        s_sy = self.synth.chunk_scores(qtext, exclude_q) if self.synth else np.zeros(self.n, np.float32)
        arrays = {"bm25": s_bm, "dense": s_dn, "sparse": s_sp, "synth": s_sy}
        valid = {"bm25": mask & (s_bm > 0), "dense": mask.copy(), "sparse": mask & (s_sp > 0),
                 "synth": mask & (s_sy > 0)}
        return arrays, valid

    def features(self, qtext, qm, mask, exclude_q=None):
        """-> (cand ids [n], X [n, len(FEATURES)], lists {ch: top-40}, f2 [n] (3-channel MaxMean))."""
        arrays, valid = self.channels(qtext, qm, mask, exclude_q)
        lists = {c: top_ids(arrays[c], valid[c], DEPTH) for c in CH}
        U = np.asarray(sorted(set().union(*[set(v) for v in lists.values()])), np.int64)
        m = len(U)
        X = np.zeros((m, len(FEATURES)), np.float64)
        Z = np.zeros((m, len(CH)))
        col = 0
        rrf = np.zeros(m)
        nhit = np.zeros(m)
        pos_in = {c: {x: r for r, x in enumerate(lists[c], 1)} for c in CH}
        for j, c in enumerate(CH):
            arr = np.asarray(arrays[c], np.float64)
            s = arr[U]
            lo, hi = float(s.min()) if m else 0.0, float(s.max()) if m else 0.0
            z = np.clip((s - lo) / (hi - lo), 0, 1) if hi > lo else np.zeros(m)
            vs = np.sort(arr[valid[c]])[::-1]
            mx = float(vs[0]) if len(vs) else 0.0
            rank = np.searchsorted(-vs, -s, side="left") + 1.0
            rank = np.where(valid[c][U], rank, len(vs) + 1.0)
            top = np.asarray([1.0 if int(x) in pos_in[c] else 0.0 for x in U])
            rrf += np.asarray([1.0 / (RRF_K + pos_in[c][int(x)]) if int(x) in pos_in[c] else 0.0 for x in U])
            nhit += top
            X[:, col:col + 5] = np.stack([s, z, s / mx if mx > 0 else np.zeros(m), np.log1p(rank), top], 1)
            Z[:, j] = z
            col += 5
        mm3 = 0.5 * Z[:, :3].max(1) + 0.5 * Z[:, :3].mean(1)
        mm4 = 0.5 * Z.max(1) + 0.5 * Z.mean(1)
        X[:, col:col + 5] = np.stack([nhit, rrf, mm3, mm4, Z.sum(1)], 1)
        col += 5
        # document prior
        cd = self.cdoc[U]
        dmax = {}
        dcnt = {}
        for d, v in zip(cd, mm4):
            dmax[d] = max(dmax.get(d, 0.0), v)
            dcnt[d] = dcnt.get(d, 0) + 1
        qd = np.asarray(qm["dense"], np.float32)
        scope_docs = np.unique(self.cdoc[mask])
        cs = self.cent[scope_docs] @ qd
        csd = dict(zip(scope_docs.tolist(), cs.tolist()))
        srt = np.sort(cs)[::-1]
        dm = np.asarray([dmax[d] for d in cd])
        cent = np.asarray([csd.get(int(d), 0.0) for d in cd])
        crank = np.searchsorted(-srt, -cent, side="left") + 1.0
        X[:, col:col + 5] = np.stack([dm, (mm4 >= dm - 1e-12).astype(float),
                                      np.asarray([dcnt[d] / m for d in cd]), cent, np.log1p(crank)], 1)
        col += 5
        # chunk
        mmU = dict(zip(U.tolist(), mm4.tolist()))
        adj = np.asarray([max(mmU.get(int(self.prev[x]), 0.0), mmU.get(int(self.next[x]), 0.0)) for x in U])
        X[:, col:col + 4] = np.stack([np.log1p(self.chars[U]), self.pos[U], self.table[U], adj], 1)
        col += 4
        # coverage, numbers, language
        qf = fold_tokens(normalize(qtext))
        qr = raw_words(qtext)
        qn = numbers(qtext)
        for i, x in enumerate(U):
            x = int(x)
            X[i, col] = len(qf & self.tok[x]) / len(qf) if qf else 0.0
            X[i, col + 1] = len(qr & self.raw[x]) / len(qr) if qr else 0.0
            X[i, col + 2] = len(qf & self.crumb[x]) / len(qf) if qf else 0.0
            X[i, col + 3] = float(len(qn))
            X[i, col + 4] = len(qn & self.nums[x]) / len(qn) if qn else 0.0
        col += 5
        X[:, col] = self.lang_fa
        X[:, col + 1] = float(len(qf))
        assert col + 2 == len(FEATURES)
        return U, X, lists, mm3
