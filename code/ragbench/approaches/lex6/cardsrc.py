"""LEX6 card source: read-only access to the generated hierarchical cards (lex6/cards.py output) for the CARDS,
SECTIONS and ROUTER arms. Cards guide retrieval only; they never enter reader evidence.

Files (work/datasets_v2/lex6/cards/): <POOL>.jsonl (one record per document: doc_id, full_card, compact_card,
short_title, sections [{key, card, span, status, ...}]), index/<POOL>.npz (full, compact, sections, centroid; unit
vectors, bge-m3 over normalize()d text) and index/<POOL>.keys.json (row ids of each matrix).

CardView(scope docs):
    doc_scores(q, qvec) -> ({doc: full-card BM25 (bm25s, B2 tokenizer, idf of the scope's cards)},
                            {doc: cos(q, full-card vector)})       documents without a card are absent (no credit)
    sections(doc, qvec, n) -> the n best section cards of doc by cos(q, section vector): [{key, span, score}]
    compact(doc) -> the compact card text (the router's view)
"""
import json
from pathlib import Path

import numpy as np

from ...normalize import normalize

CARDS_DIR = Path(r"D:\KareOne\RAG Paper\work\datasets_v2\lex6\cards")


class CardStore:
    """All pools' cards, loaded once per process."""

    def __init__(self, root=None):
        self.root = Path(root or CARDS_DIR)
        self.rec, self.full_v, self.sec = {}, {}, {}
        for f in sorted(self.root.glob("*.jsonl")):
            pool = f.stem
            with open(f, encoding="utf-8") as fh:
                for line in fh:
                    if line.strip():
                        r = json.loads(line)
                        self.rec[r["doc_id"]] = r
            kp, vp = self.root / "index" / ("%s.keys.json" % pool), self.root / "index" / ("%s.npz" % pool)
            if kp.exists() and vp.exists():
                keys = json.loads(kp.read_text(encoding="utf-8"))
                npz = np.load(vp)
                for d, v in zip(keys.get("full") or [], npz["full"]):
                    self.full_v[d] = np.asarray(v, dtype=np.float32)
                for s, v in zip(keys.get("sections") or [], npz["sections"]):
                    k = int(s["id"].rsplit("#", 1)[1])             # index into the record's sections
                    sc = (self.rec.get(s["doc_id"]) or {}).get("sections") or []
                    span = sc[k].get("span") if k < len(sc) else None
                    if span:
                        self.sec.setdefault(s["doc_id"], []).append(("%d:%s" % (k, s["key"]), list(span),
                                                                     np.asarray(v, dtype=np.float32)))
        self._views = {}

    def view(self, docs):
        k = tuple(sorted(docs))
        if k not in self._views:
            self._views[k] = CardView(self, list(docs))
        return self._views[k]

    def ready(self, docs):
        return all(d in self.rec for d in docs)


class CardView:
    def __init__(self, store, docs):
        import bm25s
        from ..baselines import DEFAULTS as BLD, tokenize
        self.store = store
        self.docs = [d for d in docs if d in store.rec and (store.rec[d].get("full_card") or "")]
        self.bm = None
        if self.docs:
            p = BLD["bm25"]
            self.bm = bm25s.BM25(k1=p["k1"], b=p["b"], method=p["method"])
            self.bm.index([tokenize(normalize(store.rec[d]["full_card"]), BLD["tokenizer"]) for d in self.docs],
                          show_progress=False)
        self.vdocs = [d for d in docs if d in store.full_v]
        self.V = np.stack([store.full_v[d] for d in self.vdocs]) if self.vdocs else None

    def doc_scores(self, qtext, qvec):
        from ...retrieval.hybrid import query_tokens
        bm = {}
        if self.bm is not None:
            toks = [t for t in query_tokens(qtext) if t in self.bm.vocab_dict]
            if toks:
                s = self.bm.get_scores(toks)
                bm = {d: float(x) for d, x in zip(self.docs, s)}
        dn = {}
        if self.V is not None and qvec is not None:
            q = np.asarray(qvec, dtype=np.float32)
            q = q / (float(np.linalg.norm(q)) or 1.0)
            dn = {d: float(x) for d, x in zip(self.vdocs, self.V @ q)}
        return bm, dn

    def sections(self, doc, qvec, n):
        rows = self.store.sec.get(doc, [])
        if not rows or qvec is None:
            return []
        q = np.asarray(qvec, dtype=np.float32)
        q = q / (float(np.linalg.norm(q)) or 1.0)
        sc = sorted(((float(v @ q), k, sp) for k, sp, v in rows), key=lambda t: (-t[0], t[1]))
        return [{"key": k, "span": sp, "score": s} for s, k, sp in sc[:n]]

    def compact(self, doc):
        r = self.store.rec.get(doc) or {}
        return r.get("compact_card") or r.get("full_card") or ""
