"""Offline tests of LEX8 (approaches/lex8: units.py, fusion.py, packer.py, identity.py).

    .venv/Scripts/python.exe -m ragbench.tests.test_lex8

No model, no network: the synthetic corpus of test_hybrid plus hand-built documents.
"""
import sys
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[2]))

import numpy as np  # noqa: E402

from ragbench.approaches.lex8 import fusion as FU  # noqa: E402
from ragbench.approaches.lex8 import identity as ID  # noqa: E402
from ragbench.approaches.lex8 import packer as PK  # noqa: E402
from ragbench.approaches.lex8 import units as U  # noqa: E402
from ragbench.config import load_config  # noqa: E402
from ragbench.contracts import pack_evidence  # noqa: E402
from ragbench.corpus.model import DocBuilder  # noqa: E402
from ragbench.corpus.registry import CorpusSet  # noqa: E402
from ragbench.tests.test_hybrid import filler, make_doc  # noqa: E402

RESULTS = {}


def check(name, cond, detail=""):
    RESULTS[name] = {"ok": bool(cond), "detail": detail}
    print(("PASS " if cond else "FAIL ") + name + (" - " + str(detail)[:300] if detail and not cond else ""), flush=True)
    return cond


def short_doc():
    b = DocBuilder("T.short.en", "T", "en", "Short lines report", source="synthetic")
    b.add_section("s1", "Lines")
    for i in range(12):
        b.add_unit("s1", "s1#p%d" % i, "paragraph", "Line %d holds figure %d." % (i, 100 + i))
    b.add_section("s2", "Other")
    b.add_unit("s2", "s2#p0", "paragraph", "Second section text.")
    b.add_section("s3", "Long")
    b.add_unit("s3", "s3#p0", "paragraph", "Intro sentence here. " + ("x" * 950) + ". Tail sentence one. Tail two.")
    return b.build()


# --------------------------------------------------------------------------------------------------------------
# units
# --------------------------------------------------------------------------------------------------------------

def test_units():
    doc = make_doc("alpha", "Alpha Corp", 2021)
    us = U.doc_units(doc)
    check("units.verbatim_nonempty", all(doc.text[u["a"]:u["b"]].strip() == doc.text[u["a"]:u["b"]] and u["b"] > u["a"]
                                        for u in us))
    secs = {s.id: s.span for s in doc.sections}
    check("units.within_section", all(secs[u["section_id"]][0] <= u["a"] and u["b"] <= secs[u["section_id"]][1]
                                      for u in us))
    prose = [u for u in us if u["kind"] == "prose"]
    check("units.prose_max", all(u["b"] - u["a"] <= U.MAX for u in prose), max(u["b"] - u["a"] for u in prose))
    ops = [u for u in prose if u["section_id"].endswith("::ops")]
    long_p = doc.unit("T.alpha.en::ops#p0")
    check("units.long_paragraph_split", len(ops) >= 3 and ops[0]["a"] == long_p.span[0], [(u["a"], u["b"]) for u in ops])
    check("units.split_at_sentence_ends", all(doc.text[u["b"] - 1] in ".!?" for u in ops),
          [doc.text[u["b"] - 3:u["b"]] for u in ops])
    rows = [u for u in us if u["kind"] == "row"]
    check("units.table_rows", len(rows) == 5, [doc.text[u["a"]:u["b"]] for u in rows])
    cap = doc.unit("T.alpha.en::fin#cap")
    first = rows[0]
    dep_txt = [doc.text[a:b] for a, b in first["deps"]]
    check("units.row_deps_caption_header", "Table 3: Revenue by segment and year" in dep_txt and "Segment | 2021 | 2022"
          in dep_txt, dep_txt)
    check("units.row_body_is_row", doc.text[first["a"]:first["b"]] == "Steel | 100 | 120")
    it = U.index_text(doc, first)
    check("units.index_text", it.startswith("Alpha Corp annual report 2021 > Financial results\n") and
          it.endswith("\nSteel | 100 | 120") and "Segment | 2021 | 2022" in it, it)
    check("units.caption_not_inside_row_deps", all(not (u["a"] <= a and b <= u["b"]) for u in rows for a, b in u["deps"]))
    _ = cap
    sd = short_doc()
    su = U.doc_units(sd)
    s1 = [u for u in su if u["section_id"].endswith("::s1")]
    check("units.short_lines_grouped", len(s1) == 1 and s1[0]["b"] - s1[0]["a"] < U.MAX, [(u["a"], u["b"]) for u in s1])
    s3 = [sd.text[u["a"]:u["b"]] for u in su if u["section_id"].endswith("::s3")]
    check("units.oversized_segment_dropped", s3 == ["Intro sentence here.", "Tail sentence one. Tail two."], s3)
    check("units.no_cross_section_pair", all(us[i]["section_id"] == us[j]["section_id"] and
                                             us[j]["b"] - us[i]["a"] <= U.MAX for i, j in U.pairs(us)))
    chunks = [{"doc_id": doc.id, "span": [0, 700]}, {"doc_id": doc.id, "span": [600, 2000]},
              {"doc_id": doc.id, "span": [1900, len(doc.text)]}]
    U.attach_parents(us, chunks)
    ok = True
    for u in us:
        cont = [ci for ci, c in enumerate(chunks) if c["span"][0] <= u["a"] and u["b"] <= c["span"][1]]
        over = [ci for ci, c in enumerate(chunks) if c["span"][0] < u["b"] and u["a"] < c["span"][1]]
        ok &= u["parents"] == (cont if cont else over)
    check("units.parents_contain_else_overlap", ok)


# --------------------------------------------------------------------------------------------------------------
# fusion
# --------------------------------------------------------------------------------------------------------------

def test_fusion():
    arrays = {"bm25": np.array([10.0, 0.0, 5.0, 2.0, 99.0]), "dense": np.array([0.5, 0.9, 0.7, 0.1, 0.0]),
              "sparse": np.array([0.0, 0.2, 0.2, 0.2, 0.0])}
    N = FU.Normalizer(arrays, [0, 1, 2])
    check("fusion.minmax", N.z("bm25", 0) == 1.0 and N.z("bm25", 1) == 0.0 and abs(N.z("bm25", 2) - 0.5) < 1e-12)
    check("fusion.clip_non_members", N.z("bm25", 4) == 1.0 and N.z("dense", 3) == 0.0)
    N1 = FU.Normalizer(arrays, [1, 2])
    check("fusion.constant_channel_zero", N1.z("sparse", 1) == 0.0 and N1.z("sparse", 0) == 0.0)
    Z = N.zvec([0, 1, 2, 3, 4])
    check("fusion.zvec_matches", np.allclose(Z, np.array([N.zs(i) for i in range(5)])))
    check("fusion.maxmean", abs(FU.maxmean([1.0, 0.0, 0.0]) - (0.5 + 0.5 / 3)) < 1e-12 and
          abs(FU.maxmean([0.3, 0.3, 0.3]) - 0.3) < 1e-12)
    check("fusion.single_strong_beats_mediocre_consensus", FU.maxmean([1, 0, 0]) > FU.maxmean([0.4, 0.4, 0.4]))
    check("fusion.combsum", FU.combsum([0.2, 0.3, 0.5]) == 1.0)
    # real scores for every union member: item 3 is in no list of 'dense' but keeps its real dense score
    lists = {"bm25": [0, 2], "dense": [1, 2], "sparse": [3]}
    order, sc, N2 = FU.fuse(arrays, lists, "maxmean")
    zs3 = N2.zs(3)
    check("fusion.real_scores_not_zero", zs3[0] > 0.0, zs3)
    check("fusion.order_by_score", order == sorted(order, key=lambda x: (-sc[x], x)) or
          all(sc[order[i]] >= sc[order[i + 1]] for i in range(len(order) - 1)), (order, sc))
    check("fusion.unit_utility", abs(FU.unit_utility(0.5, [1, 2], lambda p: {1: 0.2, 2: 0.9}[p]) - (0.4 + 0.18)) < 1e-12
          and abs(FU.unit_utility(0.5, [], lambda p: 1.0) - 0.4) < 1e-12)
    check("fusion.expanded", abs(FU.expanded_utility(0.5, 1.0) - 0.6) < 1e-12)


# --------------------------------------------------------------------------------------------------------------
# packer
# --------------------------------------------------------------------------------------------------------------

def fake_six(docs):
    units = []
    for d in docs:
        units += U.doc_units(d)
    for i, u in enumerate(units):
        u["i"] = i
    ua = np.array([u["a"] for u in units])
    ub = np.array([u["b"] for u in units])
    cs = CorpusSet("T_en", docs)
    by = {}
    for i, u in enumerate(units):
        by.setdefault(u["doc_id"], []).append(i)
    pdocs = {d.id: PK.Doc(d, by.get(d.id, []), ua, ub, cs) for d in docs}
    return SimpleNamespace(units=units, ua=ua, ub=ub, cs=cs, docs=pdocs)


def ucand(six, i, j=None):
    j = i if j is None else j
    x, y = six.units[i], six.units[j]
    return {"key": ("u", i) if i == j else ("p", i, j), "doc_id": x["doc_id"], "section_id": x["section_id"],
            "a": x["a"], "b": y["b"], "deps": list(x["deps"]) + (list(y["deps"]) if j != i else []),
            "units": [i] if i == j else [i, j], "score": 0.0}


def test_packer():
    cfg = load_config()
    alpha, beta = make_doc("alpha", "Alpha Corp", 2021), make_doc("beta", "Beta Ltd", 2023)
    six = fake_six([alpha, beta])
    D = six.docs[alpha.id]
    ua_ids = [i for i, u in enumerate(six.units) if u["doc_id"] == alpha.id]
    a0, a1, a2 = ua_ids[0], ua_ids[1], ua_ids[2]
    # merge: two touching units -> one item; a disjoint one stays separate
    p0, p1 = next((i, j) for i, j in U.pairs(six.units) if six.units[i]["doc_id"] == alpha.id)
    items = PK.render_doc(D, [ucand(six, p0), ucand(six, p1)])
    check("packer.merge_touching", len(items) == 1 and set(items[0]["units"]) >= {p0, p1}, [(i["A"], i["B"]) for i in items])
    items = PK.render_doc(D, [ucand(six, a0), ucand(six, a1)])
    check("packer.no_merge_across_title", len(items) == 2, [(i["A"], i["B"]) for i in items])
    far = ua_ids[-1]
    items = PK.render_doc(D, [ucand(six, a0), ucand(six, far)])
    check("packer.disjoint_separate", len(items) == 2)
    rows = [i for i in ua_ids if six.units[i]["kind"] == "row"]
    it = PK.render_doc(D, [ucand(six, rows[1])])[0]
    check("packer.row_deps_prefixed", it["text"].startswith("Table 3: Revenue by segment and year\nSegment | 2021 | 2022\n")
          and it["text"].endswith("Mining | 50 | 55"), it["text"])
    it2 = PK.render_doc(D, [ucand(six, rows[0], rows[1]), ucand(six, rows[1])])
    check("packer.overlap_once", len(it2) == 1 and it2[0]["text"].count("Mining | 50 | 55") == 1)
    # coverage pack: exact cost, never over budget, never cut, each unit counted once
    u = np.linspace(1.0, 0.1, len(six.units))
    cands = [ucand(six, i) for i in range(len(six.units)) if six.ub[i] - six.ua[i] <= U.MAX]
    cands += [ucand(six, i, i + 1) for i, j in U.pairs(six.units)][:10]
    for budget in (600, 1500, 6000):
        st, info = PK.coverage_pack(six.docs, u, cands, budget=budget)
        items = PK.to_items(st.items(), u)
        ev = PK.final_pack(items, six.cs, cfg, budget=budget, expect_chars=st.cost())
        check("packer.exact_cost_%d" % budget, ev.chars == st.cost() and ev.chars <= budget and not ev.last_item_cut,
              (ev.chars, st.cost(), info))
        disp = [x for it in st.items() for x in it["units"]]
        check("packer.units_once_%d" % budget, len(disp) == len(set(disp)) and
              abs(st.F - sum(float(u[x]) ** 2 for x in set(disp))) < 1e-9)
        check("packer.best_of_three_%d" % budget, st.F >= max(v["F"] for k, v in info.items() if k != "chosen") - 1e-6,
              info)
    # a big candidate that does not fit is skipped, the rest still packs
    big = {"key": ("c", 0), "doc_id": alpha.id, "section_id": None, "a": 0, "b": len(alpha.text), "deps": [],
           "units": ua_ids, "score": 1.0}
    st, info = PK.coverage_pack(six.docs, u, [big] + cands[:6], budget=800)
    check("packer.skip_not_fitting", ("c", 0) not in st.keys and st.keys and st.cost() <= 800, (st.keys, info))
    # greedy ratio prefers cheap high-utility units
    uu = np.zeros(len(six.units))
    uu[a0] = 1.0
    st, _ = PK.coverage_pack(six.docs, uu, [ucand(six, a0), big], budget=6000)
    check("packer.best_F", a0 in [x for it in st.items() for x in it["units"]])
    # ranked pack: skips candidates without new units; pack_evidence semantics
    ev = PK.ranked_pack(six.docs, [ucand(six, a0), ucand(six, a0), ucand(six, a1)], six.cs, cfg, budget=6000)
    check("packer.ranked_skip_dup", ev.n_offered == 2 and ev.n_included == 2, (ev.n_offered, ev.n_included))
    ev = PK.ranked_pack(six.docs, [big, ucand(six, far)], six.cs, cfg, budget=900)
    check("packer.ranked_cut_first_misfit", ev.n_included == 1 and ev.last_item_cut, (ev.n_included, ev.last_item_cut))
    check("packer.header_total", PK.header_total(0) == 0 and PK.header_total(1) == 4 and
          PK.header_total(10) == 9 * 4 + 5 + 18)
    ref = pack_evidence(PK.to_items(PK.render_doc(D, [ucand(six, a0), ucand(six, far)]), u), six.cs, budget=10 ** 6, cfg=cfg)
    fixed = sum(i["fixed"] for i in PK.render_doc(D, [ucand(six, a0), ucand(six, far)]))
    check("packer.cost_formula", ref.chars == fixed + PK.header_total(2), (ref.chars, fixed))
    _ = a2


# --------------------------------------------------------------------------------------------------------------
# identity
# --------------------------------------------------------------------------------------------------------------

def test_identity():
    texts, rows = ["a", "b", "c"], [("d1", 0, 1), ("d1", 1, 2), ("d2", 0, 1)]
    enc = ID.encoder_info("BAAI/bge-m3", "rev1", 512, True, "m3multi-v1")
    pre = {"index_text": "x", "units": "v1"}
    m = ID.manifest("t", texts, rows, enc, pre)
    ok = True
    try:
        ID.check(m, texts, rows, enc, pre)
    except ID.IdentityError:
        ok = False
    check("identity.ok", ok)

    def fails(t, r, e=enc, p=pre):
        try:
            ID.check(m, t, r, e, p)
        except ID.IdentityError as ex:
            return str(ex)
        return None
    check("identity.changed_text", fails(["a", "B", "c"], rows) and "texts_sha1" in fails(["a", "B", "c"], rows))
    check("identity.reordered_rows_same_doc", "rows_sha1" in (fails(texts, [rows[1], rows[0], rows[2]]) or ""))
    check("identity.count", "n " in (fails(texts[:2], rows[:2]) or ""))
    check("identity.revision", "revision" in (fails(texts, rows, dict(enc, revision="rev2")) or ""))
    check("identity.max_length", "max_length" in (fails(texts, rows, dict(enc, max_length=640)) or ""))
    check("identity.preprocess", "preprocess" in (fails(texts, rows, enc, {"index_text": "y"}) or ""))
    check("identity.boundary_safe", ID.sha_texts(["ab", "c"]) != ID.sha_texts(["a", "bc"]))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    test_units()
    test_fusion()
    test_packer()
    test_identity()
    bad = [k for k, v in RESULTS.items() if not v["ok"]]
    print("\n%d checks, %d failed%s" % (len(RESULTS), len(bad), (": " + ", ".join(bad)) if bad else ""))
    raise SystemExit(1 if bad else 0)
