"""Lazy bge-m3 vectors of the tree leaves (passages) of only the documents a question targets.

No index build step: the first question that targets a document embeds its passages (ledger purpose
"lazy-index-embed", trace stage "lazy-index"); every vector is committed at once to the SHARED embedding cache that B3
uses (work/indexes/_embed_cache/embeddings.sqlite, key = text + embed model + digest + truncate + num_ctx, B3's own
key function), together with its apportioned embedding cost ("c:" entries: tokens and wall ms), so

  * a stopped run loses at most one embed batch and a restart re-embeds nothing that was committed;
  * the preprocessing cost of a document can be reported cache-inclusively (cached_cost_ms), independent of which
    question or run touched it first;
  * the per-process matrix cache makes every later question on the same document free.

The ledger phase stays "query" (ragbench.llm allows index|query|eval|smoke and eval.sanity sums a record's embed calls
over its phase=query ledger lines); the lazy-index work is separated by the ledger field stage="lazy-index" and by
trace.timing, and reported as preprocessing.
"""
import time

import numpy as np

from ...util import KVCache, sha1_json
from ..baselines import VectorRetrieval
from .defaults import TREE_VERSION
from .tree import build_passages


def _ms(t0):
    return round((time.perf_counter() - t0) * 1000.0, 1)


class LazyVectors:
    def __init__(self, approach):
        self.a = approach
        self.cfg = approach.cfg
        self._kv = None
        self._docs = {}      # (doc id, text hash, passage opts) -> (passages, matrix L2-normalised)

    # the shared embedding cache and B3's key function (same model digest, truncate and num_ctx)
    def cache_path(self):
        return VectorRetrieval._cache_path(self.a, None)

    def text_key(self, text):
        return VectorRetrieval._text_key(self.a, text)

    def kv(self):
        if self._kv is None:
            self._kv = KVCache(self.cache_path())
        return self._kv

    def doc_key(self, doc, opts):
        return sha1_json({"doc": doc.id, "text": sha1_json(doc.text), "tree": TREE_VERSION,
                          "p": {k: opts[k] for k in ("passage_chars", "passage_max_chars", "passage_overlap_chars",
                                                     "embed_with_path")}})

    def ensure(self, doc, opts, client, purpose="lazy-index-embed"):
        """-> (passages, matrix, info). Embeds only passages whose vector is not cached."""
        key = self.doc_key(doc, opts)
        if key in self._docs:
            ps, m = self._docs[key]
            return ps, m, {"doc_id": doc.id, "passages": len(ps), "embedded_now": 0, "embed_calls": 0,
                           "embed_tokens_now": 0, "embed_ms_now": 0.0, "memory_hit": 1, "load_ms": 0.0,
                           "passages_ms": 0.0, "cached_cost_ms": None}
        t0 = time.perf_counter()
        ps = build_passages(doc, opts)
        passages_ms = _ms(t0)
        kv = self.kv()
        t1 = time.perf_counter()
        keys = [self.text_key(p.embed_text) for p in ps]
        vecs = {}
        todo, seen = [], set()
        for k, p in zip(keys, ps):
            if k in seen:
                continue
            seen.add(k)
            v = kv.get_array("v:" + k)
            if v is None:
                todo.append((k, p.embed_text))
            else:
                vecs[k] = v
        load_ms = _ms(t1)
        bs = int(self.cfg.embedding["batch_size"])
        calls, tokens, wall = 0, 0, 0.0
        t2 = time.perf_counter()
        for j in range(0, len(todo), bs):
            batch = todo[j:j + bs]
            er = client.embed([t for _, t in batch], purpose=purpose, batch_size=bs)
            for (k, _), v in zip(batch, er.vectors):
                arr = np.asarray(v, dtype=np.float32)
                kv.put_array("v:" + k, arr)
                kv.put("c:" + k, {"tokens": er.prompt_tokens / len(batch), "wall_ms": er.wall_ms / len(batch)})
                vecs[k] = arr
            calls += er.calls
            tokens += er.prompt_tokens
            wall += er.wall_ms
        embed_elapsed = _ms(t2)
        cost_ms = 0.0
        for k in seen:
            c = kv.get("c:" + k) or {}
            cost_ms += float(c.get("wall_ms") or 0.0)
        if ps:
            m = np.vstack([vecs[k] for k in keys]).astype(np.float32)
            n = np.linalg.norm(m, axis=1, keepdims=True)
            m = m / np.where(n > 0, n, 1.0)
        else:
            m = np.zeros((0, 1), dtype=np.float32)
        self._docs[key] = (ps, m)
        return ps, m, {"doc_id": doc.id, "passages": len(ps), "unique_texts": len(seen), "embedded_now": len(todo),
                       "embed_calls": calls, "embed_tokens_now": tokens, "embed_ms_now": round(wall, 1),
                       "embed_elapsed_ms": embed_elapsed, "memory_hit": 0, "load_ms": load_ms,
                       "passages_ms": passages_ms, "cached_cost_ms": round(cost_ms, 1),
                       "chars": sum(len(p.embed_text) for p in ps)}

    def query_vectors(self, texts, client, purpose="query-embed"):
        er = client.embed(list(texts), purpose=purpose)
        out = []
        for v in er.vectors:
            q = np.asarray(v, dtype=np.float32)
            nq = float(np.linalg.norm(q))
            out.append(q / (nq if nq > 0 else 1.0))
        return out, {"embed_calls": er.calls, "embed_tokens": er.prompt_tokens, "embed_ms": round(er.wall_ms, 1)}

    def close(self):
        if self._kv is not None:
            self._kv.close()
            self._kv = None
