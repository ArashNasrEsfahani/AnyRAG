"""Baselines B0-B3 (owned by the baselines agent).

    B0 NoRetrieval      one call, no evidence: the shared prompt's "answer from your own
                        knowledge, else NOT_FOUND" variant (build_answer_messages(query, None)).
    B1 GoldEvidence     one call over the question's gold units in the DOCUMENT language of the
                        setting (Query.gold_unit_ids; the only class with needs_gold = True).
    B2 BM25Retrieval    BM25 (bm25s, Lucene variant) over section-aware chunks, one call.
    B3 VectorRetrieval  bge-m3 (through OllamaClient.embed) over the same chunks, cosine, one call.

Fairness notes
  * Chunk size, overlap and top_k come only from config.yaml `chunking`; the evidence budget and
    the answer prompt come only from contracts (self.pack / self.generate_answer).
  * B2 and B3 build byte-identical chunk lists (same function, same parameters); every chunk is a
    verbatim slice of Document.text with its span, and lists every unit it overlaps (unit_ids) plus
    the ones it only partly contains (meta.partial_unit_ids) so evidence recall can be strict or lenient.
  * Ranked chunks are offered to pack_evidence best-first: at least top_k, and more while every offered
    chunk still fits whole and budget remains (DEFAULTS["fill_budget"]), so B2/B3 are not handicapped by
    short chunks. Nothing but the question text, doc_scope and the corpus drives retrieval.
  * Index-time work never sees questions. Query embeddings are not cached (latency stays honest).

Chunking (chunk_document)
  Section blocks (title + own units, pre-order) are packed whole into chunks of at most
  chunking.chunk_chars while they fit (so a packed chunk always starts and ends on section boundaries; its
  section_id is the deepest common ancestor, None across top-level sections); a block longer than chunk_chars is
  cut into windows of at most chunk_chars with chunking.overlap_chars of overlap, breaking preferably at
  a unit boundary (blank line), then a line (table row), then a sentence end, then a space. A block
  shorter than min_standalone_chars (typically a bare heading) that precedes an oversized block is
  glued to that block's first window instead of becoming a stub chunk.

Tokenisation (tokenize)
  normalize() (the extractor rule) -> harakat removed (before splitting, so a diacritic never cuts a word in two)
  -> ragbench.normalize.word_tokens(folded=False) (lower-case, numbers
  keep separators, ZWNJ-joined Persian words kept whole) -> per token: fold() (alef/heh/waw/yeh variants,
  harakat) keeping the ZWNJ join; the ZWNJ/apostrophe parts are added as extra tokens (so "شرکت‌ها" also
  matches "شرکت" and "iran's" matches "iran"); numbers with thousands separators or % also emit the plain
  digits; Latin-letter tokens get Lucene's EnglishMinimalStemmer (plural -s only; Persian is not stemmed);
  light fa+en stopword list; letter tokens shorter than 2 characters dropped.
"""
import functools
import json
import os
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..contracts import Approach, AnswerResult, RetrievedItem
from ..normalize import _RE_HARAKAT, ZWNJ, fold, normalize, word_tokens
from ..util import KVCache, atomic_write_json, atomic_write_text, read_json, read_jsonl, sha1_json

# Non-fairness-critical tunables of this module. Fairness-critical values (chunk_chars, overlap_chars,
# top_k, evidence budget, models, generation options) live only in config.yaml.
DEFAULTS = {
    "tokenizer": {
        # v2: harakat (fatha/kasra/damma/shadda/tanwin/hamza-above...) are stripped BEFORE word splitting. In v1
        # word_tokens treated them as separators, so "مُدیریت" became "م"+"دیریت" and never matched "مدیریت"
        # (C has ~1.1k mid-word harakat, A_fa 85). "may" left the English stopwords (month name in dates).
        "version": 2,
        "strip_harakat": True,
        "zwnj_subtokens": True,      # add the parts of ZWNJ/apostrophe-joined words as extra tokens
        "stopwords": True,
        "min_letter_token_chars": 2,
        "number_variants": True,     # "264,926" also emits "264926"; "87%" also emits "87"
        "en_s_stem": True,           # Latin-letter tokens: Lucene EnglishMinimalStemmer (plural s); Persian is not stemmed
    },
    "chunker": {
        "version": 1,
        "min_standalone_chars": 200,  # a smaller block before an oversized one is glued to its first window
        "min_window_frac": 0.5,       # a window break is searched in [start + frac*chunk_chars, start + chunk_chars]
        # whole blocks of different top-level sections may share a chunk. B/C heading trees are flat (C: 2,655 of
        # 7,873 sections are roots), and forbidding it turned 9% of C chunks into <200-char stubs (722 -> 40 when allowed)
        "merge_across_roots": True,
    },
    "index_with_path": True,          # index/embed "<doc title> > <section path>\n<chunk text>" (text given to the model is the chunk only)
    "bm25": {"k1": 1.5, "b": 0.75, "method": "lucene"},
    "fill_budget": True,              # offer more than top_k chunks while they all still fit whole in the budget
    "max_candidates": 40,             # hard cap on chunks offered to pack_evidence
    "shared_embed_cache": True,       # B3: reuse vectors across index fingerprints (per text + model digest)
}

# --------------------------------------------------------------------------
# tokenisation (B2)
# --------------------------------------------------------------------------

_STOP_EN = set("""
a an the of and or in on at to for from by with as is are was were be been being it its this that these those
which what who whom whose when where why how many much did does do done there their they them than then also into
over under per about has have had not no but if so such can could would should will might each any all
""".split())

_STOP_FA_RAW = """
و در به از که این آن را با برای است هست بود شد شده شود می نمی ها های تا یا هم ای اي چه چند چقدر چگونه چطور کدام
کجا آیا بر نیز اگر هر همه ولی اما پس یعنی دیگر باید کرد کرده کند کنند کردن دارد دارند داشت داشته بوده باشد
خود وی او ما شما آنها آن‌ها این‌ها ایشان تر ترین مورد سال‌های
"""


@functools.lru_cache(maxsize=1)
def _stopwords():
    out = set(_STOP_EN)
    for w in _STOP_FA_RAW.split():
        out.add(ZWNJ.join(fold(w).split()))
    return frozenset(out)


_ASCII_WORD = re.compile(r"^[a-z]+$")


def s_stem(w):
    """Harman's S-stemmer, as Lucene's EnglishMinimalStemmer (plural -s/-es/-ies only)."""
    n = len(w)
    if n < 3 or w[-1] != "s":
        return w
    c = w[-2]
    if c in "us":
        return w
    if c == "e":
        if n > 3 and w[-3] == "i" and w[-4] not in "ae":
            return w[:-3] + "y"
        if w[-3] in "iaoe":
            return w
    return w[:-1]


@functools.lru_cache(maxsize=400_000)
def _expand_token(tok, zwnj_subtokens, stopwords, min_letter, number_variants, en_s_stem):
    """tok: one token of word_tokens(folded=False) (normalised, lower-case, may contain ZWNJ or ')."""
    stop = _stopwords() if stopwords else frozenset()
    if tok[:1].isdigit():
        out = [tok]
        if number_variants:
            plain = tok.rstrip("%").replace(",", "")
            if plain and plain != tok:
                out.append(plain)
        return tuple(out)
    zparts = fold(tok).split()                      # fold: variants + harakat removed, ZWNJ -> space
    joined = ZWNJ.join(zparts)
    if joined in stop:
        joined = ""

    def stem(w):
        return s_stem(w) if en_s_stem and _ASCII_WORD.match(w) else w

    out = []
    if joined and len(joined) >= min_letter:
        out.append(stem(joined))
    if zwnj_subtokens:
        parts = [p for zp in zparts for p in zp.split("'") if p]
        if len(parts) > 1:
            for p in parts:
                if len(p) >= min_letter and p not in stop:
                    p = stem(p)
                    if p != joined and p not in out:
                        out.append(p)
    return tuple(out)


def tokenize(text, opts=None):
    """BM25 tokens for index text and queries alike (see module docstring)."""
    o = dict(DEFAULTS["tokenizer"], **(opts or {}))
    t = normalize(text or "")
    if o.get("strip_harakat", True):
        t = _RE_HARAKAT.sub("", t)     # the shared fold() rule's harakat set, applied before splitting
    out = []
    for tok in word_tokens(t, folded=False):
        out.extend(_expand_token(tok, bool(o["zwnj_subtokens"]), bool(o["stopwords"]),
                                 int(o["min_letter_token_chars"]), bool(o["number_variants"]), bool(o["en_s_stem"])))
    return out


# --------------------------------------------------------------------------
# chunking (B2 and B3)
# --------------------------------------------------------------------------

_SENT_END = re.compile(r"[.!?؟؛;](?=\s)")


def _best_break(text, lo, hi):
    """End offset (exclusive) of a window whose end lies in [lo, hi]."""
    i = text.rfind("\n\n", lo, hi)
    if i >= lo:
        return i
    i = text.rfind("\n", lo, hi)
    if i >= lo:
        return i
    last = None
    for m in _SENT_END.finditer(text, lo, hi):
        last = m.end()
    if last is not None:
        return last
    i = text.rfind(" ", lo, hi)
    if i >= lo:
        return i
    return hi


def _next_start(text, start, end, overlap, limit):
    if overlap > 0:
        p = max(end - overlap, start + 1)
        while p < end and not (text[p - 1].isspace() and not text[p].isspace()):
            p += 1
    else:
        p = end
    while p < limit and text[p].isspace():
        p += 1
    return p


def _windows(text, a, b, size, overlap, min_frac):
    out = []
    start = a
    while start < b:
        if b - start <= size:
            out.append((start, b))
            break
        lo = start + max(1, int(size * min_frac))
        hi = start + size
        end = _best_break(text, lo, hi)
        while end > start and text[end - 1].isspace():
            end -= 1
        if end <= start:
            end = hi
        out.append((start, end))
        nxt = _next_start(text, start, end, overlap, b)
        if nxt <= start:          # cannot happen with overlap < size*min_frac; guard against loops anyway
            nxt = end
        start = nxt
    return out


def _lca(doc, section_ids):
    paths = []
    for sid in section_ids:
        p = []
        s = doc.section(sid)
        while s is not None:
            p.append(s.id)
            s = doc.section(s.parent) if s.parent else None
        paths.append(p[::-1])
    common = None
    for level in zip(*paths):
        if all(x == level[0] for x in level):
            common = level[0]
        else:
            break
    return common


def chunk_document(doc, chunk_chars, overlap_chars, opts=None):
    """Chunks of one Document: list of dicts {doc_id, section_id, section_ids, span, text, unit_ids,
    partial_unit_ids, mode: packed|window}. Deterministic; every chunk is doc.text[span]."""
    o = dict(DEFAULTS["chunker"], **(opts or {}))
    size, overlap = int(chunk_chars), int(overlap_chars)
    if overlap >= size * float(o["min_window_frac"]):
        raise ValueError("overlap_chars must be smaller than min_window_frac * chunk_chars")
    text = doc.text
    root_of = {}
    for s in doc.sections:                      # pre-order: a parent precedes its children
        root_of[s.id] = root_of[s.parent] if s.parent else s.id
    blocks = [(s.span[0], s.span[1], s.id) for s in doc.sections
              if s.span[1] > s.span[0] and text[s.span[0]:s.span[1]].strip()]
    span_of = {sid: (a, b) for a, b, sid in blocks}
    merge_roots = bool(o["merge_across_roots"])
    small = int(o["min_standalone_chars"])

    runs, pend = [], None                        # run: [start, end, [section ids]]
    for a, b, sid in blocks:
        if pend is not None:
            same = merge_roots or root_of[pend[2][0]] == root_of[sid]
            if same and b - pend[0] <= size:
                pend[1] = b
                pend[2].append(sid)
                continue
            if same and b - a > size and pend[1] - pend[0] < small:
                runs.append([pend[0], b, pend[2] + [sid]])
                pend = None
                continue
            runs.append(pend)
            pend = None
        if b - a > size:
            runs.append([a, b, [sid]])
        else:
            pend = [a, b, [sid]]
    if pend is not None:
        runs.append(pend)

    chunks = []
    for a, b, sids in runs:
        if b - a <= size:
            spans, mode = [(a, b)], "packed"
        else:
            spans, mode = _windows(text, a, b, size, overlap, float(o["min_window_frac"])), "window"
        for wa, wb in spans:
            secs = [sid for sid in sids if span_of[sid][0] < wb and span_of[sid][1] > wa]
            units = doc.units_overlapping(wa, wb)
            chunks.append({
                "doc_id": doc.id, "section_id": _lca(doc, secs) if secs else None, "section_ids": secs,
                "span": [wa, wb], "text": text[wa:wb], "unit_ids": [u.id for u in units],
                "partial_unit_ids": [u.id for u in units if u.span[0] < wa or u.span[1] > wb], "mode": mode})
    return chunks


def breadcrumb(doc, section_id):
    parts = [doc.title] + (doc.section_path(section_id) if section_id else [])
    return " > ".join(p for p in parts if p)


def build_chunks(corpus_set, cfg, opts=None):
    ch = cfg.chunking
    out = []
    for doc in corpus_set.docs:
        for c in chunk_document(doc, ch["chunk_chars"], ch["overlap_chars"], opts):
            c["i"] = len(out)
            c["breadcrumb"] = breadcrumb(doc, c["section_id"])
            out.append(c)
    return out


def chunks_fingerprint(chunks):
    return sha1_json([[c["doc_id"], c["span"], c["section_id"]] for c in chunks])


def _write_jsonl_atomic(path, rows):
    atomic_write_text(path, "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))


# --------------------------------------------------------------------------
# B0 / B1
# --------------------------------------------------------------------------

class NoRetrieval(Approach):
    """B0: the model answers from memory (shared prompt, no-evidence variant)."""
    name = "B0"

    def _build_index(self, corpus_set, index_dir):
        return {"index": None}

    def _load_index(self, corpus_set, index_dir):
        return {"index": None}

    def _answer(self, query, index):
        answer = self.generate_answer(query, None)
        return AnswerResult(answer_text=answer, evidence=None, retrieved=[], trace={"evidence": None})


class GoldEvidence(Approach):
    """B1: the gold evidence units in the setting's document language, in the author's order."""
    name = "B1"
    needs_gold = True

    def _build_index(self, corpus_set, index_dir):
        return {"index": None}

    def _load_index(self, corpus_set, index_dir):
        return {"index": None}

    def _answer(self, query, index):
        items, missing = [], []
        for uid in query.gold_unit_ids or []:
            try:
                u = index.corpus.unit(uid)
            except KeyError:
                missing.append(uid)
                continue
            items.append(RetrievedItem(text=u.text, doc_id=u.doc_id, section_id=u.section_id, unit_ids=[u.id],
                                       kind="unit", rank=len(items), span=list(u.span)))
        ev = self.pack(items, index)
        answer = self.generate_answer(query, ev)
        return AnswerResult(answer_text=answer, evidence=ev, retrieved=items,
                            trace={"gold_units": len(query.gold_unit_ids or []), "missing_gold_units": missing,
                                   "evidence_items": ev.n_included, "last_item_cut": ev.last_item_cut})


# --------------------------------------------------------------------------
# chunk retrieval base (B2 / B3)
# --------------------------------------------------------------------------

@dataclass
class ChunkIndex:
    chunks: list                    # chunk dicts (chunks.jsonl)
    doc_ids: list                   # doc id per doc position
    doc_pos: np.ndarray             # doc position per chunk
    extra: object = None            # bm25s.BM25 (B2) | np.ndarray n x d, L2-normalised (B3)
    meta: dict = field(default_factory=dict)


class _ChunkRetrieval(Approach):
    name = "CHUNK"
    # doc_card / section_expand only change the query-time client (ragbench/variants.py): B2, B3 and HYBRID
    # with them reuse the plain index instead of building (B3: re-embedding) an identical one
    index_neutral_flags = ("doc_card", "section_expand")
    min_score = None                # chunks at or below this score are not retrieved (None: no floor)
    verbose = False

    def config_signature(self):
        return {"chunking": self.cfg.chunking, "chunker": DEFAULTS["chunker"],
                "index_with_path": DEFAULTS["index_with_path"]}

    # -- index ------------------------------------------------------------------
    def _chunks(self, corpus_set, index_dir):
        """chunks.jsonl is rebuilt deterministically if missing; reused otherwise."""
        path = Path(index_dir) / "chunks.jsonl"
        meta_path = Path(index_dir) / "chunks_meta.json"
        meta = read_json(meta_path)
        if path.is_file() and meta and meta.get("corpus_fingerprint") == corpus_set.fingerprint():
            chunks = read_jsonl(path)
            if len(chunks) == meta.get("n_chunks") and chunks_fingerprint(chunks) == meta.get("chunks_fingerprint"):
                return chunks, meta
        t0 = time.perf_counter()
        chunks = build_chunks(corpus_set, self.cfg, DEFAULTS["chunker"])
        _write_jsonl_atomic(path, chunks)
        lens = np.array([len(c["text"]) for c in chunks]) if chunks else np.zeros(1)
        meta = {"corpus_set": corpus_set.name, "corpus_fingerprint": corpus_set.fingerprint(), "n_chunks": len(chunks),
                "chunks_fingerprint": chunks_fingerprint(chunks), "chunking": self.cfg.chunking,
                "chunker": DEFAULTS["chunker"], "chars_mean": round(float(lens.mean()), 1),
                "chars_p50": int(np.median(lens)), "chars_max": int(lens.max()),
                "n_window": sum(1 for c in chunks if c["mode"] == "window"),
                "n_packed": sum(1 for c in chunks if c["mode"] == "packed"),
                "chunk_ms": round((time.perf_counter() - t0) * 1000.0, 1)}
        atomic_write_json(meta_path, meta, indent=1)
        return chunks, meta

    @staticmethod
    def index_text(chunk):
        if DEFAULTS["index_with_path"] and chunk.get("breadcrumb"):
            return chunk["breadcrumb"] + "\n" + chunk["text"]
        return chunk["text"]

    @staticmethod
    def _doc_positions(corpus_set, chunks):
        doc_ids = corpus_set.doc_ids()
        pos = {d: i for i, d in enumerate(doc_ids)}
        return doc_ids, np.array([pos[c["doc_id"]] for c in chunks], dtype=np.int32)

    # -- query ------------------------------------------------------------------
    def _scores(self, query, index):
        """-> (np.ndarray of one score per chunk, trace dict)"""
        raise NotImplementedError

    def _scope_mask(self, query, index):
        ci = index.payload
        if query.doc_scope == "corpus" or query.doc_scope is None:
            return np.ones(len(ci.chunks), dtype=bool), None
        scope = set(query.doc_scope)
        wanted = [i for i, d in enumerate(ci.doc_ids) if d in scope]
        return np.isin(ci.doc_pos, np.array(wanted, dtype=np.int32)), len(wanted)

    def _item(self, index, i, score, rank):
        c = index.payload.chunks[i]
        return RetrievedItem(text=c["text"], doc_id=c["doc_id"], section_id=c["section_id"], unit_ids=list(c["unit_ids"]),
                             kind="chunk", score=round(float(score), 6), rank=rank, span=list(c["span"]),
                             meta={"chunk": int(i), "mode": c["mode"], "partial_unit_ids": list(c["partial_unit_ids"])})

    def _offer(self, items, index):
        """At least top_k items; with fill_budget, keep adding while every offered item fits whole."""
        k = int(self.cfg.chunking["top_k"])
        offered = list(items[:k])
        if not DEFAULTS["fill_budget"]:
            return offered
        cap = int(DEFAULTS["max_candidates"])
        nxt = len(offered)
        while nxt < min(len(items), cap):
            pk = self.pack(offered, index)
            if pk.last_item_cut or pk.n_included < len(offered):
                break
            offered.append(items[nxt])
            nxt += 1
            pk2 = self.pack(offered, index)
            if pk2.n_included == pk.n_included:      # the new item would be dropped entirely
                offered.pop()
                break
        return offered

    def _answer(self, query, index):
        t0 = time.perf_counter()
        scores, trace = self._scores(query, index)
        mask, n_scope_docs = self._scope_mask(query, index)
        ci = index.payload
        n = len(ci.chunks)
        valid = mask.copy()
        if self.min_score is not None:
            valid &= scores > self.min_score
        order = np.lexsort((np.arange(n), -scores))
        order = order[valid[order]]
        cap = max(int(DEFAULTS["max_candidates"]), int(self.cfg.chunking["top_k"]))
        items = [self._item(index, int(i), scores[i], r) for r, i in enumerate(order[:cap])]
        offered = self._offer(items, index)
        ev = self.final_pack(offered, index, query)
        retrieval_ms = (time.perf_counter() - t0) * 1000.0
        answer = self.generate_answer(query, ev)
        trace = dict(trace or {})
        trace.update({"chunks_total": n, "chunks_in_scope": int(mask.sum()), "scope_docs_found": n_scope_docs,
                      "chunks_scored": int(valid.sum()), "offered": len(offered), "evidence_items": ev.n_included,
                      "last_item_cut": ev.last_item_cut, "retrieval_ms": round(retrieval_ms, 1),
                      # query-time tunables are not part of the index fingerprint or config_hash: record them per question
                      "params": {"top_k": int(self.cfg.chunking["top_k"]), "fill_budget": bool(DEFAULTS["fill_budget"]),
                                 "max_candidates": int(DEFAULTS["max_candidates"]), "min_score": self.min_score},
                      "top": [[it.meta["chunk"], it.score] for it in offered]})
        return AnswerResult(answer_text=answer, evidence=ev, retrieved=offered, trace=trace)


# --------------------------------------------------------------------------
# B2 BM25
# --------------------------------------------------------------------------

class BM25Retrieval(_ChunkRetrieval):
    """B2: BM25 (bm25s, Lucene variant) over the shared chunks."""
    name = "B2"
    min_score = 0.0

    def config_signature(self):
        sig = super().config_signature()
        sig.update({"tokenizer": DEFAULTS["tokenizer"], "bm25": DEFAULTS["bm25"]})
        return sig

    def _build_index(self, corpus_set, index_dir):
        import bm25s
        index_dir = Path(index_dir)
        chunks, cmeta = self._chunks(corpus_set, index_dir)
        t0 = time.perf_counter()
        tokens = [tokenize(self.index_text(c), DEFAULTS["tokenizer"]) for c in chunks]
        p = DEFAULTS["bm25"]
        retr = bm25s.BM25(k1=p["k1"], b=p["b"], method=p["method"])
        retr.index(tokens, show_progress=False)
        retr.save(str(index_dir / "bm25"), show_progress=False)
        meta = {"n_chunks": len(chunks), "chunks_fingerprint": cmeta["chunks_fingerprint"],
                "vocab": len(retr.vocab_dict), "tokens_mean": round(float(np.mean([len(t) for t in tokens])) if tokens else 0.0, 1),
                "empty_chunks": sum(1 for t in tokens if not t), "bm25": p, "tokenizer": DEFAULTS["tokenizer"],
                "index_ms": round((time.perf_counter() - t0) * 1000.0, 1)}
        atomic_write_json(index_dir / "bm25_meta.json", meta, indent=1)
        doc_ids, doc_pos = self._doc_positions(corpus_set, chunks)
        return ChunkIndex(chunks, doc_ids, doc_pos, retr, dict(cmeta, **meta))

    def _load_index(self, corpus_set, index_dir):
        import bm25s
        index_dir = Path(index_dir)
        meta = read_json(index_dir / "bm25_meta.json")
        chunks, cmeta = self._chunks(corpus_set, index_dir)
        if not meta or not (index_dir / "bm25").is_dir() or meta.get("chunks_fingerprint") != cmeta["chunks_fingerprint"]:
            raise RuntimeError("B2 index at %s is incomplete or stale; delete build_stats.json there to rebuild" % index_dir)
        retr = bm25s.BM25.load(str(index_dir / "bm25"), show_progress=False)
        doc_ids, doc_pos = self._doc_positions(corpus_set, chunks)
        return ChunkIndex(chunks, doc_ids, doc_pos, retr, dict(cmeta, **meta))

    def query_tokens(self, text):
        seen, out = set(), []
        for t in tokenize(normalize(text), DEFAULTS["tokenizer"]):
            if t not in seen:
                seen.add(t)
                out.append(t)
        return out

    def _scores(self, query, index):
        retr = index.payload.extra
        toks = self.query_tokens(query.text)
        known = [t for t in toks if t in retr.vocab_dict]
        n = len(index.payload.chunks)
        if not known:
            scores = np.zeros(n, dtype=np.float32)
        else:
            scores = np.asarray(retr.get_scores(known), dtype=np.float32)
        return scores, {"query_tokens": toks, "query_tokens_in_vocab": len(known)}


# --------------------------------------------------------------------------
# B3 vector
# --------------------------------------------------------------------------

class VectorRetrieval(_ChunkRetrieval):
    """B3: bge-m3 embeddings (OllamaClient.embed) over the shared chunks; cosine similarity."""
    name = "B3"
    min_score = None

    def config_signature(self):
        sig = super().config_signature()
        sig.update({"embedding": self.cfg.embedding, "embed_model": self.cfg.models["embed"],
                    "embed_digest": self.cfg.models.get("embed_digest")})
        return sig

    def _cache_path(self, index_dir):
        if DEFAULTS["shared_embed_cache"]:
            return self.cfg.path("index_root") / "_embed_cache" / "embeddings.sqlite"
        return Path(index_dir) / "embed_cache.sqlite"

    def _text_key(self, text):
        e = self.cfg.embedding
        return sha1_json({"m": self.cfg.models["embed"], "d": self.cfg.models.get("embed_digest"),
                          "t": bool(e["truncate"]), "c": e["num_ctx"], "x": text})

    def _build_index(self, corpus_set, index_dir):
        index_dir = Path(index_dir)
        chunks, cmeta = self._chunks(corpus_set, index_dir)
        texts = [self.index_text(c) for c in chunks]
        keys = [self._text_key(t) for t in texts]
        cache = KVCache(self._cache_path(index_dir))
        try:
            todo, seen = [], set()
            for k, t in zip(keys, texts):
                if k not in seen and cache.get_array("v:" + k) is None:
                    todo.append((k, t))
                seen.add(k)
            n_unique = len(seen)
            bs = int(self.cfg.embedding["batch_size"])
            fresh_tokens, fresh_ms, t0 = 0, 0.0, time.perf_counter()
            for j in range(0, len(todo), bs):
                batch = todo[j:j + bs]
                er = self.client.embed([t for _, t in batch], purpose="index-embed", batch_size=bs)
                for (k, _), v in zip(batch, er.vectors):
                    cache.put_array("v:" + k, np.asarray(v, dtype=np.float32))
                    cache.put("c:" + k, {"tokens": er.prompt_tokens / len(batch), "wall_ms": er.wall_ms / len(batch)})
                fresh_tokens += er.prompt_tokens
                fresh_ms += er.wall_ms
                if self.verbose and (j // bs) % 25 == 0:
                    print("[B3 %s] embedded %d/%d new texts (%.0fs)" % (corpus_set.name, min(j + bs, len(todo)), len(todo),
                                                                     time.perf_counter() - t0), file=sys.stderr, flush=True)
            vecs, cost_tokens, cost_ms = [], 0.0, 0.0
            for k in keys:
                v = cache.get_array("v:" + k)
                if v is None:
                    raise RuntimeError("embedding cache lost a vector during the build (%s)" % k[:12])
                vecs.append(v)
            for k in seen:
                c = cache.get("c:" + k) or {}
                cost_tokens += c.get("tokens") or 0.0
                cost_ms += c.get("wall_ms") or 0.0
        finally:
            cache.close()
        dim = int(vecs[0].shape[0]) if vecs else 0
        E = np.vstack(vecs).astype(np.float32) if vecs else np.zeros((0, dim), dtype=np.float32)
        norms = np.linalg.norm(E, axis=1, keepdims=True)
        E = E / np.where(norms > 0, norms, 1.0)
        tmp = index_dir / (".embeddings.tmp-%d.npy" % os.getpid())
        np.save(tmp, E)
        os.replace(tmp, index_dir / "embeddings.npy")
        meta = {"n_chunks": len(chunks), "chunks_fingerprint": cmeta["chunks_fingerprint"], "dim": dim,
                "unique_texts": n_unique, "embedded_now": len(todo), "from_cache": n_unique - len(todo),
                "embed_tokens_now": fresh_tokens, "embed_wall_ms_now": round(fresh_ms, 1),
                "embed_tokens_all_texts": round(cost_tokens, 1), "embed_wall_ms_all_texts": round(cost_ms, 1),
                "cache": str(self._cache_path(index_dir)), "model": self.cfg.models["embed"],
                "note": "build_stats totals count only calls made in this build; *_all_texts apportions the "
                        "original batch cost of every vector, cached or not, for index-cost reporting"}
        atomic_write_json(index_dir / "vector_meta.json", meta, indent=1)
        doc_ids, doc_pos = self._doc_positions(corpus_set, chunks)
        return ChunkIndex(chunks, doc_ids, doc_pos, E, dict(cmeta, **meta))

    def _load_index(self, corpus_set, index_dir):
        index_dir = Path(index_dir)
        meta = read_json(index_dir / "vector_meta.json")
        chunks, cmeta = self._chunks(corpus_set, index_dir)
        ep = index_dir / "embeddings.npy"
        if not meta or not ep.is_file() or meta.get("chunks_fingerprint") != cmeta["chunks_fingerprint"]:
            raise RuntimeError("B3 index at %s is incomplete or stale; delete build_stats.json there to rebuild" % index_dir)
        E = np.load(ep)
        if E.shape[0] != len(chunks):
            raise RuntimeError("B3 index at %s: %d vectors for %d chunks" % (index_dir, E.shape[0], len(chunks)))
        doc_ids, doc_pos = self._doc_positions(corpus_set, chunks)
        return ChunkIndex(chunks, doc_ids, doc_pos, E, dict(cmeta, **meta))

    def _scores(self, query, index):
        E = index.payload.extra
        er = self.client.embed([normalize(query.text)], purpose="query-embed")
        q = np.asarray(er.vectors[0], dtype=np.float32)
        nq = float(np.linalg.norm(q))
        q = q / (nq if nq > 0 else 1.0)
        scores = E @ q if len(E) else np.zeros(0, dtype=np.float32)
        return scores.astype(np.float32), {"query_embed_tokens": er.prompt_tokens}
