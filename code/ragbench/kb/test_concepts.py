"""Offline tests for the concept layer (no model: a fake embedder whose synonyms share a direction).

    .venv/Scripts/python.exe -m ragbench.kb.test_concepts

  names     only clean metric names are embedded, one text per distinct name per document
  lazy      a first `ensure` embeds, a second embeds nothing; a NEW NameVectors over the same cache embeds nothing
            either (the vectors are committed to the shared cache, not held in memory)
  nearest   the plan's words reach the report's words for the same thing across a renaming (staff -> personnel,
            people employed), not the unrelated names; the per-document cap and the similarity floor hold
  walk      a seed name's closest name in ANOTHER report crosses the renaming between years
  clock     AGENTIC4's query clock takes a NESTED lazy-index out of its parent stage (v1-v3 do not)
"""
import json
import shutil
import sys
from types import SimpleNamespace

import numpy as np

from ..config import load_config
from ..util import KVCache, sha1_json
from . import concepts as C
from . import labels as L
from . import schema as S

FAILS = []


def check(cond, label):
    print(("  ok   " if cond else "  FAIL ") + label)
    if not cond:
        FAILS.append(label)


# a concept per synonym group: every word of a group points the same way
GROUPS = {"staff": ["کارکنان", "پرسنل", "شاغل", "افراد", "staff", "personnel"],
          "ads": ["آگهی", "ads"], "paper": ["کاغذ", "paper"], "users": ["کاربران", "کاربر", "users"]}
AXIS = {w: i for i, (_g, ws) in enumerate(GROUPS.items()) for w in ws}


class FakeClient:
    def __init__(self):
        self.calls = 0
        self.texts = 0

    def embed(self, texts, purpose=None, batch_size=None, **_kw):
        self.calls += 1
        self.texts += len(texts)
        vecs = []
        for t in texts:
            v = np.full(len(GROUPS) + 1, 0.05)
            for w in L._f(t).split():
                if w in AXIS:
                    v[AXIS[w]] += 1.0
            vecs.append(v.tolist())
        return SimpleNamespace(vectors=vecs, prompt_tokens=10 * len(texts), wall_ms=1.0, calls=1)


class FakeLazy:
    def __init__(self, path, cfg):
        self.cfg = cfg
        self._kv = KVCache(path)

    def kv(self):
        return self._kv

    def text_key(self, text):
        return sha1_json({"t": text, "m": "fake"})


def _db(cfg):
    root = cfg.path("work_root") / "kb" / "_test_concepts"
    shutil.rmtree(root, ignore_errors=True)
    con = S.connect(root / "kb.sqlite")
    S.ensure_schema(con, cfg)
    L.ensure_labels_schema(con)
    rows = [("D.y1397.fa", 1, "جمعی از کارکنان", "ok"), ("D.y1398.fa", 2, "تعداد کل پرسنل", "ok"),
            ("D.y1400.fa", 3, "تعداد افراد شاغل", "ok"), ("D.y1400.fa", 4, "بررسی آگهی", "ok"),
            ("D.y1400.fa", 5, "کاهش مصرف کاغذ", "ok"), ("D.y1400.fa", 6, "صفحه", "junk"),
            ("D.y1400.fa", 7, "تعداد افراد شاغل", "ok")]
    for d in sorted({r[0] for r in rows}):
        con.execute("INSERT INTO documents (doc_id, corpus, key, lang, status, versions) VALUES (?,?,?,?,?,?)",
                    (d, "U", d, "fa", "ready", json.dumps(S.component_versions())))
    for d, fid, name, q in rows:
        con.execute("INSERT INTO facts (fact_id, doc_id, label, label_fold) VALUES (?,?,?,?)", (fid, d, name, L._f(name)))
        con.execute("INSERT INTO fact_labels (fact_id, doc_id, metric_name, metric_fold, label_quality) "
                    "VALUES (?,?,?,?,?)", (fid, d, name if q == "ok" else None, L._f(name) if q == "ok" else None, q))
    con.commit()
    return con, root


def concept_tests(cfg):
    con, root = _db(cfg)
    print("\n[names]")
    names = C.doc_names(con, "D.y1400.fa")
    check([n for _f, n in names] == ["بررسی آگهی", "تعداد افراد شاغل", "کاهش مصرف کاغذ"],
          "one text per DISTINCT clean name, junk left out: %s" % [n for _f, n in names])

    print("\n[lazy]")
    client = FakeClient()
    nv = C.NameVectors(FakeLazy(root / "cache.sqlite", cfg))
    docs = ["D.y1397.fa", "D.y1398.fa", "D.y1400.fa"]
    mats, info = nv.ensure(con, docs, client)
    check(info["embedded_now"] == 5 and client.texts == 5, "the first touch embeds the 5 distinct names (%s)" % info)
    _m, info2 = nv.ensure(con, docs, client)
    check(info2["embedded_now"] == 0 and client.texts == 5, "a second touch embeds nothing")
    nv2 = C.NameVectors(FakeLazy(root / "cache.sqlite", cfg))
    mats2, info3 = nv2.ensure(con, docs, client)
    check(info3["embedded_now"] == 0 and client.texts == 5, "a NEW instance over the same cache embeds nothing")

    print("\n[nearest]")
    q = [np.asarray(v) / np.linalg.norm(v) for v in client.embed(["کارکنان"]).vectors]
    hits = C.NameVectors.nearest(mats, q, per_doc=6, min_sim=0.6)
    got = {(d, f) for d, f, _s in hits}
    want = {("D.y1397.fa", L._f("جمعی از کارکنان")), ("D.y1398.fa", L._f("تعداد کل پرسنل")),
            ("D.y1400.fa", L._f("تعداد افراد شاغل"))}
    check(want <= got and not any("کاغذ" in f or "آگهی" in f for _d, f in got),
          "«کارکنان» reaches personnel and people-employed in every year, not ads or paper: %s" % sorted(got))
    check(len(C.NameVectors.nearest(mats, q, per_doc=1, min_sim=0.0)) == 3, "at most per_doc names per document")
    # «جمعی از کارکنان» points exactly the query's way (cosine 1.0); «تعداد افراد شاغل» holds two staff words (0.99+)
    tight = C.NameVectors.nearest(mats, q, per_doc=6, min_sim=0.9999)
    check({d for d, _f, _s in tight} == {"D.y1397.fa", "D.y1398.fa"} and all(s >= 0.9999 for _d, _f, s in tight),
          "the similarity floor holds: at 0.9999 only the same-direction names pass (%s)"
          % [(d, round(s, 5)) for d, _f, s in tight])

    print("\n[walk]")
    v = C.NameVectors.vector_of(mats, "D.y1398.fa", L._f("تعداد کل پرسنل"))
    hop = C.NameVectors.closest_in(mats, "D.y1400.fa", v, 0.6)
    check(hop and hop[0] == L._f("تعداد افراد شاغل"),
          "1398's «پرسنل» hops to 1400's «افراد شاغل», across the renaming: %s" % (hop,))
    con.close()


def clock_tests():
    print("\n[clock]")
    from ..approaches.agentic4.approach import _Session4
    s = _Session4.__new__(_Session4)
    s.calls = [{"stage": "retrieve", "wall_ms": 100.0}]
    s.timing = [{"stage": "lazy-index", "wall_ms": 900.0, "parent": "retrieve", "call_elapsed_ms": 0.0},
                {"stage": "retrieve", "wall_ms": 1000.0, "call_elapsed_ms": 100.0, "parent": None},
                {"stage": "answer", "wall_ms": 500.0, "call_elapsed_ms": 400.0, "parent": None}]
    s.calls.append({"stage": "answer", "wall_ms": 400.0})
    # retrieve: 1000 wall - 100 in calls - 900 nested embedding + 100 recorded call = 100; answer: 500 - 400 + 400
    check(abs(s.clock_s() - 0.6) < 1e-9, "a nested lazy-index is taken out of its parent: clock %.3f s (want 0.600)"
          % s.clock_s())


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    cfg = load_config()
    concept_tests(cfg)
    clock_tests()
    print("\n%d failure(s)" % len(FAILS))
    for f in FAILS:
        print("  -", f)
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
