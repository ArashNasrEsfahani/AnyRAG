"""The concept layer: bge-m3 vectors of the SHORT METRIC NAMES, so a question's word reaches the report's word for the
same thing. Lazy, cached forever, additive, and the one place the fact graph uses a model.

WHY. The word-level graph (graph.py) cannot follow a report that renames a metric. Divar's reports call their
headcount «جمعی از دانشجویان و فارغ‌التحصیلان» in 1397, «تعداد کل پرسنل» in 1398 and «تعداد افراد شاغل» in 1400,
and the plan asked for «کارکنان / نیروی انسانی» - not one shared word. A probe before this module was written
(2026-09-21, one embed call, 0.2 s) put the three gold names FIRST against the plan's terms - 0.747, 0.700, 0.607 -
above every distractor the word-level graph had ranked higher («ناظر انسانی», "human moderator": 0.516).

LAZY, LIKE THE PASSAGE VECTORS. Nothing is embedded at ingestion: `ensure` embeds the distinct metric names of the
documents a question targets the first time they are touched, and commits every vector to the SHARED embedding cache
B3 and AGENTIC2/3 use (same file, same key function, same "c:" cost entries), so a name is embedded once for good and
its cost is reported as preprocessing. The session runs it inside a "lazy-index" stage, which the 60 s query clock
excludes. Ingestion stays model-free.

WHAT IS EMBEDDED. `fact_labels.metric_name` (the readable name, not the folded key) for facts whose label passed the
quality gate, one text per distinct folded name per document. The name is what carries meaning; the value and the
year are typed columns and do not belong in the vector.

    nv = NameVectors(lazy)                                  # lazy = the approach's LazyVectors (cache + key)
    mats, info = nv.ensure(con, doc_ids, client)            # {doc_id: (names, matrix)}
    hits = nv.nearest(mats, qvecs, per_doc=6, min_sim=0.6)  # [(doc_id, metric_fold, sim)]
"""
import time

import numpy as np

CONCEPTS_VERSION = "kb-concepts-v1"


def _ms(t0):
    return round((time.perf_counter() - t0) * 1000.0, 1)


def doc_names(con, doc_id):
    """-> [(metric_fold, metric_name)] distinct names of a document's clean facts, in a stable order."""
    return [(r["metric_fold"], r["metric_name"]) for r in con.execute(
        """SELECT metric_fold, min(metric_name) AS metric_name FROM fact_labels
           WHERE doc_id = ? AND label_quality = 'ok' AND metric_fold IS NOT NULL
           GROUP BY metric_fold ORDER BY metric_fold""", (doc_id,))]


class NameVectors:
    def __init__(self, lazy):
        self.lazy = lazy
        self._mem = {}          # doc_id -> (names, matrix)

    def ensure(self, con, doc_ids, client, purpose="lazy-name-embed"):
        """Embed the names of `doc_ids` that are not cached yet. -> ({doc_id: (names, matrix)}, info)."""
        t0 = time.perf_counter()
        kv = self.lazy.kv()
        out, todo, per_doc = {}, [], {}
        for d in doc_ids:
            if d in self._mem:
                out[d] = self._mem[d]
                continue
            names = doc_names(con, d)
            keys = [self.lazy.text_key(n) for _f, n in names]
            per_doc[d] = (names, keys)
            for k, (_f, n) in zip(keys, names):
                if kv.get_array("v:" + k) is None:
                    todo.append((k, n))
        todo = list(dict(todo).items())
        bs = int(self.lazy.cfg.embedding["batch_size"])
        calls, tokens, wall = 0, 0, 0.0
        for j in range(0, len(todo), bs):
            batch = todo[j:j + bs]
            er = client.embed([t for _k, t in batch], purpose=purpose, batch_size=bs)
            for (k, _t), v in zip(batch, er.vectors):
                kv.put_array("v:" + k, np.asarray(v, dtype=np.float32))
                kv.put("c:" + k, {"tokens": er.prompt_tokens / len(batch), "wall_ms": er.wall_ms / len(batch)})
            calls += er.calls
            tokens += er.prompt_tokens
            wall += er.wall_ms
        for d, (names, keys) in per_doc.items():
            if names:
                m = np.vstack([kv.get_array("v:" + k) for k in keys]).astype(np.float32)
                n = np.linalg.norm(m, axis=1, keepdims=True)
                m = m / np.where(n > 0, n, 1.0)
            else:
                m = np.zeros((0, 1), dtype=np.float32)
            self._mem[d] = out[d] = (names, m)
        return out, {"docs": len(doc_ids), "names": sum(len(out[d][0]) for d in out), "embedded_now": len(todo),
                     "embed_calls": calls, "embed_tokens": tokens, "embed_ms": round(wall, 1), "ms": _ms(t0)}

    @staticmethod
    def nearest(mats, qvecs, per_doc=6, min_sim=0.6):
        """-> [(doc_id, metric_fold, sim)]: per document the names closest to ANY of the query vectors (max over
        them, as the probe did), at least `min_sim`, at most `per_doc`, best first overall."""
        if not qvecs:
            return []
        q = np.vstack([np.asarray(v, dtype=np.float32) for v in qvecs])
        out = []
        for d, (names, m) in mats.items():
            if not names or m.shape[0] == 0:
                continue
            sims = (m @ q.T).max(axis=1)
            order = np.argsort(-sims)[:per_doc]
            out += [(d, names[i][0], float(sims[i])) for i in order if sims[i] >= min_sim]
        out.sort(key=lambda x: -x[2])
        return out

    @staticmethod
    def closest_in(mats, doc_id, vec, min_sim):
        """-> (metric_fold, sim) of the name in `doc_id` closest to one name vector, or None below `min_sim`."""
        names, m = mats.get(doc_id, ([], None))
        if not names or m is None or m.shape[0] == 0:
            return None
        sims = m @ np.asarray(vec, dtype=np.float32)
        i = int(np.argmax(sims))
        return (names[i][0], float(sims[i])) if sims[i] >= min_sim else None

    @staticmethod
    def vector_of(mats, doc_id, metric_fold):
        names, m = mats.get(doc_id, ([], None))
        for i, (f, _n) in enumerate(names):
            if f == metric_fold:
                return m[i]
        return None
