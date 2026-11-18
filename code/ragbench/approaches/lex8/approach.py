"""LEX8 (2026-09-26, Astra R1 converged design): compact source units + chunks, score-aware fusion, budgeted coverage
packing. No cross-encoder, no query-time LLM call (0 extra tokens); the shared answer call is unchanged.

    LEX8            question encoded once by bge-m3 (dense + sparse; GPU when free); BM25 + dense + sparse top 40 at
                    two granularities (the shared chunks and the compact source units, lex8/units.py); MaxMean fusion
                    over REAL channel scores; u(a) = 0.8 R_unit + 0.2 R_parent; 80 compact + 40 chunk candidates;
                    coverage packer (lex8/packer.py) -> <= 6,000 rendered chars -> the shared answer call.
    flags: ranked   ablation C1: the same candidates, ordinary ranked packing
           synth    S1: + the synthetic-question BM25 field (index time), u_exp = 0.8 u + 0.2 e
Index-time artifacts (identity-checked on load: texts / rows hashes, encoder revision, preprocessing):
<LEX8_IDX_DIR or work/datasets_v2/lex8/idx>/<set>/ and .../chunks/<set>.*; synthetic field
<LEX8_SYNTH_DIR or work/datasets_v2/lex8/synth>/<set>/. The encoder load and warm-up happen at index load, outside the
timed query. trace["lex8"]: channel lists, unions, u of candidate units, packer runs, chosen keys, stage ms.
"""
import os
import time
from pathlib import Path

from ...normalize import normalize
from ..hybrid import HybridRetrieval
from . import index as IX
from . import packer as PK
from . import retrieve as RV

LEX8_VERSION = "lex8-v1"
LEX8_DEFAULTS = {"query_max_length": 128, "budget": PK.BUDGET, **RV.C}


def _root(cfg, env, sub):
    return Path(os.environ.get(env) or (Path(cfg.project_root) / "work" / "datasets_v2" / "lex8" / sub))


class Lex8(HybridRetrieval):
    name = "LEX8"
    index_neutral_flags = ("ranked", "synth")
    _enc = None

    def config_signature(self):
        from ...retrieval import hybrid as H
        return {"hybrid": H.signature(self.cfg), "lex8": LEX8_DEFAULTS, "version": LEX8_VERSION,
                "packer": PK.PACKER_VERSION, "retrieve": RV.RETRIEVE_VERSION, "index": IX.INDEX_VERSION}

    def _prep(self, corpus_set, hx):
        from ...retrieval import m3multi as M3
        if Lex8._enc is None:
            t = time.perf_counter()
            Lex8._enc = M3.M3Encoder().load()
            Lex8._enc.encode(["warm up"], max_length=16, colbert=False)
            self._load_ms = (time.perf_counter() - t) * 1000.0
        six = IX.SetIndex(corpus_set, hx, _root(self.cfg, "LEX8_IDX_DIR", "idx"),
                          encoder_revision=getattr(Lex8._enc, "revision", None))
        RV.attach_docs(six)
        six.synth = None
        if (self.flags or {}).get("synth"):
            from .synthfield import SynthField
            d = _root(self.cfg, "LEX8_SYNTH_DIR", "synth") / corpus_set.name
            six.synth = SynthField(d) if SynthField.exists(d) else None
        # warm the query path once (bm25s / scipy / numpy first-call costs stay outside the timed query)
        RV.run(six, "warm up", {"dense": [0.0] * 1024, "sparse": {}}, doc_ids=None)
        self._six = six
        return six

    def _build_index(self, corpus_set, index_dir):
        hx = super()._build_index(corpus_set, index_dir)
        self._prep(corpus_set, hx)
        return hx

    def _load_index(self, corpus_set, index_dir):
        hx = super()._load_index(corpus_set, index_dir)
        self._prep(corpus_set, hx)
        return hx

    def _answer(self, query, index):
        T0 = time.perf_counter()
        six = self._six
        st = {"version": LEX8_VERSION, "device": getattr(Lex8._enc, "device", None),
              "load_ms": round(getattr(self, "_load_ms", 0.0), 1)}
        doc_ids = None if query.doc_scope in (None, "corpus") else list(query.doc_scope)
        t = time.perf_counter()
        q = Lex8._enc.encode([normalize(query.text)], max_length=LEX8_DEFAULTS["query_max_length"], colbert=False)[0]
        st["t_encode_ms"] = round((time.perf_counter() - t) * 1000.0, 1)
        synth = six.synth.e if six.synth is not None else None
        R = RV.run(six, query.text, q, doc_ids=doc_ids, synth=synth)
        st.update({k: round(v, 1) for k, v in R["timing"].items()})
        t = time.perf_counter()
        if (self.flags or {}).get("ranked"):
            ranked = sorted(R["cands"], key=lambda c: (-c["score"], c["body_chars"], c["key"]))
            ev = PK.ranked_pack(six.docs, ranked, index.corpus, self.cfg, budget=LEX8_DEFAULTS["budget"])
            items, info, chosen = list(ev.items), {"packer": "ranked"}, None
        else:
            stt, info = PK.coverage_pack(six.docs, R["u"], R["cands"], budget=LEX8_DEFAULTS["budget"])
            items = PK.to_items(stt.items(), R["u"])
            ev = PK.final_pack(items, index.corpus, self.cfg, budget=LEX8_DEFAULTS["budget"], expect_chars=stt.cost())
            chosen = [list(k) for k in stt.keys]
        st["t_pack_ms"] = round((time.perf_counter() - t) * 1000.0, 1)
        st["t_retrieval_ms"] = round((time.perf_counter() - T0) * 1000.0, 1)
        t = time.perf_counter()
        answer = self.generate_answer(query, ev)
        st["t_answer_ms"] = round((time.perf_counter() - t) * 1000.0, 1)
        cand_units = sorted({m for c in R["cands"] for m in c["units"] if c["key"][0] != "c"})
        st.update({"chunk_lists": R["chunk_lists"], "unit_lists": R["unit_lists"],
                   "u": {str(a): round(float(R["u"][a]), 4) for a in cand_units},
                   "cands": [[list(c["key"]), round(c["score"], 4)] for c in R["cands"]], "packer": info,
                   "chosen": chosen, "synth_hits": len(R["e"]), "evidence_items": ev.n_included,
                   "evidence_chars": ev.chars})
        trace = {"lex8": st, "retrieval_ms": st["t_retrieval_ms"], "chunks_total": six.hx.n, "units_total": six.nu,
                 "evidence_items": ev.n_included, "last_item_cut": ev.last_item_cut}
        return self._result(answer, ev, items, trace)
