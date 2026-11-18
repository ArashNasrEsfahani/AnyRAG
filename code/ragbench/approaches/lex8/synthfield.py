"""LEX8 synthetic-question field (index time: grounded questions per chunk, generated once by gpt-oss-20b; see
work/datasets_v2/lex8/synth8.py). A SEPARATE BM25 field over the accepted questions; each question links to the source
units under its verbatim quote. Generated text is never evidence and never enters the chunk / unit representations.

    e(a) = max over the top-40 question hits linking unit a of the min-max-normalised (over those hits) BM25 score
"""
import json
from pathlib import Path

import numpy as np

from ...retrieval import hybrid as H

SYNTH_VERSION = "lex8-synth-v1"
TOP = 40


class SynthField:
    def __init__(self, d):
        import bm25s
        d = Path(d)
        self.retr = bm25s.BM25.load(str(d / "bm25q"), show_progress=False)
        self.qs = [json.loads(l) for l in open(d / "questions.jsonl", encoding="utf-8")]
        self.n = len(self.qs)

    @staticmethod
    def exists(d):
        return (Path(d) / "bm25q" / "params.index.json").exists()

    def hits(self, qtext):
        toks = [w for w in H.query_tokens(qtext) if w in self.retr.vocab_dict]
        if not toks:
            return [], None
        s = np.asarray(self.retr.get_scores(toks), np.float32)
        o = np.lexsort((np.arange(len(s)), -s))
        return [int(i) for i in o[:TOP] if s[i] > 0], s

    def e(self, qtext, umask):
        top, s = self.hits(qtext)
        if not top:
            return {}
        lo, hi = float(s[top].min()), float(s[top].max())
        out = {}
        for i in top:
            z = (float(s[i]) - lo) / (hi - lo) if hi > lo else 0.0
            for a in self.qs[i]["units"]:
                if umask[a] and z > out.get(a, -1.0):
                    out[a] = z
        return out
