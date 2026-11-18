"""Offline tests of the upgrade foundation: ragbench.retrieval.hybrid, ragbench.retrieval.prebuild,
ragbench.evidence and the HYBRID approach.

    .venv/Scripts/python.exe -m ragbench.tests.test_hybrid

No model, no network: synthetic two-report corpus in a temp dir, fake embedders (hashed bag of tokens), fake chat.
Nothing is written outside the temp dir (index_root and the embedding cache are redirected there).
"""
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[2]))

import numpy as np  # noqa: E402

from ragbench.approaches import APPROACHES, get_approach  # noqa: E402
from ragbench.approaches import baselines as BL  # noqa: E402
from ragbench.config import load_config  # noqa: E402
from ragbench.contracts import Query, Question, RetrievedItem, pack_evidence  # noqa: E402
from ragbench.corpus.model import DocBuilder  # noqa: E402
from ragbench.corpus.registry import CorpusSet  # noqa: E402
from ragbench.evidence import block_overlap, build_blocks, row_overlap, table_structure, _DocView  # noqa: E402
from ragbench.llm import EmbedResult, Ledger, ModelError  # noqa: E402
from ragbench.retrieval import hybrid as H  # noqa: E402
from ragbench.retrieval import prebuild as P  # noqa: E402
from ragbench.tests.test_baselines import FakeOllama, fake_vector  # noqa: E402
from ragbench.util import KVCache  # noqa: E402

RESULTS = {}


def check(name, cond, detail=""):
    RESULTS[name] = {"ok": bool(cond), "detail": detail}
    print(("PASS " if cond else "FAIL ") + name + (" - " + str(detail)[:300] if detail and not cond else ""), flush=True)
    return cond


# --------------------------------------------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------------------------------------------

def filler(topic, n, start=0):
    return " ".join("Sentence %d on %s covers routine group matters in some detail for the reader." % (i, topic)
                    for i in range(start, start + n))


def make_doc(key, company, year, table_rows=None, n_sections=3):
    b = DocBuilder("T.%s.en" % key, "T", "en", "%s annual report %d" % (company, year), source="synthetic")
    b.add_section("intro", "Introduction")
    b.add_unit("intro", "intro#p0", "paragraph", filler("the introduction", 5))
    b.add_section("ops", "Operations review")
    long_para = filler("operations", 12) + " Steel output reached 4.2 million tonnes at the %s mills. " % company + filler("operations", 12, 12)
    b.add_unit("ops", "ops#p0", "paragraph", long_para)
    b.add_unit("ops", "ops#p1", "paragraph", "Plant maintenance was completed on schedule. " + filler("maintenance", 2))
    for s in range(1, n_sections + 1):
        b.add_section("sec%d" % s, "Topic %d" % s)
        b.add_unit("sec%d" % s, "sec%d#p0" % s, "paragraph", ("Topic%d keyword appears here. " % s) + filler("topic %d" % s, 8))
    b.add_section("fin", "Financial results")
    b.add_unit("fin", "fin#cap", "caption", "Table 3: Revenue by segment and year")
    rows = table_rows or [("Steel", 100, 120), ("Mining", 50, 55), ("Logistics", 20, 22)]
    b.add_unit("fin", "t1#table.0", "table", "\n".join(["Segment | 2021 | 2022"] + ["%s | %s | %s" % r for r in rows]))
    b.add_unit("fin", "t1#table.1", "table", "Segment | 2021 | 2022\nRevenue total | 170 | 197\nOther income | 3 | 4")
    b.add_unit("fin", "fin#p0", "paragraph", "Revenue of %s rose in %d. " % (company, year) + filler("results", 3))
    b.add_section("risk", "Risks")
    b.add_unit("risk", "risk#p0", "paragraph", filler("risks", 7))
    return b.build()


def make_set():
    alpha = make_doc("alpha", "Alpha Corp", 2021, n_sections=6)
    beta = make_doc("beta", "Beta Ltd", 2023,
                    table_rows=[("Row%02d label text" % i, 1000 + i, 2000 + i) for i in range(40)])
    return CorpusSet("T_en", [alpha, beta])


class FakeEmbed:
    """Minimal embed client: records calls, can fail after N calls."""

    def __init__(self, fail_after=None, dim=64):
        self.calls, self.texts, self.purposes, self.fail_after, self.dim = 0, 0, [], fail_after, dim

    def embed(self, texts, *, purpose, batch_size=None, **kw):
        if self.fail_after is not None and self.calls >= self.fail_after:
            raise ModelError("injected embed failure")
        self.calls += 1
        self.texts += len(texts)
        self.purposes.append(purpose)
        return EmbedResult(vectors=[fake_vector(t, dim=self.dim) for t in texts], prompt_tokens=10 * len(texts), wall_ms=1.0, calls=1)


def tcfg(root):
    return load_config().copy_with({"paths": {"index_root": str(root)}, "ollama": {"cross_process_lock": False}})


def span_of(doc, uid, a=0, b=None):
    u = doc.unit(uid)
    return [u.span[0] + a, u.span[0] + (b if b is not None else len(u.text))]


# --------------------------------------------------------------------------------------------------------------
# fusion
# --------------------------------------------------------------------------------------------------------------

def test_fusion():
    r = H.rrf_fuse({"bm25": [3, 1, 2], "dense": [2, 3, 5]})
    got = {cid: s for cid, s, _ in r}
    check("rrf.math", abs(got[3] - (1 / 61 + 1 / 62)) < 1e-12 and abs(got[2] - (1 / 63 + 1 / 61)) < 1e-12
          and abs(got[1] - 1 / 62) < 1e-12 and abs(got[5] - 1 / 63) < 1e-12, r)
    check("rrf.order", [c for c, _, _ in r] == [3, 2, 1, 5], r)
    check("rrf.ranks_recorded", r[0][2] == {"bm25": 1, "dense": 2} and r[3][2] == {"bm25": None, "dense": 3}, r)
    r2 = H.rrf_fuse({"dense": [2, 3, 5], "bm25": [3, 1, 2]})
    check("rrf.channel_order_independent", [(c, round(s, 12)) for c, s, _ in r] == [(c, round(s, 12)) for c, s, _ in r2])
    dup = H.rrf_fuse({"bm25": [7, 7, 8, 7]})
    check("rrf.one_contribution_per_channel", dict((c, s) for c, s, _ in dup) == {7: 1 / 61, 8: 1 / 62}, dup)
    tie = H.rrf_fuse({"bm25": [1, 2], "dense": [2, 1]})
    check("rrf.tie_break_deterministic", [c for c, _, _ in tie] == [1, 2], tie)
    check("rrf.depth", [c for c, _, _ in H.rrf_fuse({"bm25": [1, 2, 3]}, depth=2)] == [1, 2])


# --------------------------------------------------------------------------------------------------------------
# the index
# --------------------------------------------------------------------------------------------------------------

def mkquery(text, scope="corpus"):
    return Query(question_id="q", text=text, q_lang="en", d_lang="en", setting="en-en", corpus="T", corpus_set="T_en",
                 doc_scope=scope)


def test_index(tmp):
    cs = make_set()
    cfg = tcfg(tmp / "idx1")
    fo = FakeOllama(cfg, Ledger(path=None))
    b2 = get_approach("B2", cfg=cfg, client=fo)
    b3 = get_approach("B3", cfg=cfg, client=fo)
    h2 = b2.build_index(cs)
    h3 = b3.build_index(cs)
    n_b3_embeds = fo.embed_inputs
    hx = H.HybridIndex.build(cs, cfg, client=None, embed=False)
    check("index.same_chunks_as_b2_b3", [c["span"] for c in hx.chunks] == [c["span"] for c in h2.payload.chunks]
          == [c["span"] for c in h3.payload.chunks] and [c["text"] for c in hx.chunks] == [c["text"] for c in h2.payload.chunks])
    check("index.same_embed_keys_as_b3", all(H.embed_text_key(cfg, H.index_text(c)) == b3._text_key(b3.index_text(c))
                                             for c in hx.chunks))
    check("index.b3_vectors_are_cache_hits", hx.dense_complete and fo.embed_inputs == n_b3_embeds,
          (hx.status(), fo.embed_inputs, n_b3_embeds))
    check("index.vectors_equal_b3", np.allclose(hx.E, h3.payload.extra, atol=1e-6))
    check("index.persisted_under_index_root", str(hx.index_dir).startswith(str(tmp / "idx1" / "_hybrid"))
          and (hx.index_dir / "hybrid_meta.json").is_file())
    hl = H.HybridIndex.load(cs, cfg)
    check("index.load_roundtrip", hl.n == hx.n and np.array_equal(hl.has_vec, hx.has_vec) and hl.meta["fingerprint"] == hx.meta["fingerprint"])

    q = "steel output tonnes"
    s2, _ = b2._scores(mkquery(q), h2)
    order = np.lexsort((np.arange(len(s2)), -s2))
    b2_order = [int(i) for i in order if s2[i] > 0][:40]
    hits = hx.search(q, channels=("bm25",), k=40)
    check("search.bm25_only_equals_b2", [h.chunk for h in hits] == b2_order, ([h.chunk for h in hits][:8], b2_order[:8]))
    qv = np.asarray(fake_vector(H.normalize(q)), dtype=np.float32)
    E = h3.payload.extra
    s3 = E @ (qv / np.linalg.norm(qv))
    b3_order = [int(i) for i in np.lexsort((np.arange(len(s3)), -s3))][:40]
    hd = hx.search(q, channels=("dense",), k=40, query_vector=qv)
    check("search.dense_only_equals_b3", [h.chunk for h in hd] == b3_order[:len(hd)] and len(hd) == min(40, hx.n))
    hb = hx.search(q, k=10, query_vector=qv)
    exp = H.rrf_fuse({"bm25": b2_order, "dense": b3_order})[:10]
    check("search.hybrid_is_rrf_of_b2_b3", [h.chunk for h in hb] == [c for c, _, _ in exp]
          and all(abs(h.rrf - s) < 1e-12 for h, (_, s, _) in zip(hb, exp)))
    check("search.hit_fields", all(h.text == cs.doc(h.doc_id).text[h.span[0]:h.span[1]] and set(h.ranks) == {"bm25", "dense"}
                                   and set(h.scores) == {"bm25", "dense"} for h in hb))
    check("search.one_contribution_per_channel", all(abs(h.rrf - sum(1 / (60 + r) for r in h.ranks.values() if r)) < 1e-12 for h in hb))
    check("search.diagnostics", hb.diagnostics["dense"]["complete"] and hb.diagnostics["dense"]["query_vector"] == "given"
          and not hb.diagnostics["warnings"], hb.diagnostics)

    # scope mask over the same full index: BM25 statistics unchanged
    hs = hx.search(q, doc_ids=["T.beta.en"], k=40, query_vector=qv)
    full = {h.chunk: h.scores["bm25"] for h in hx.search(q, channels=("bm25",), k=hx.n)}
    check("scope.only_scoped_docs", hs and all(h.doc_id == "T.beta.en" for h in hs), [h.doc_id for h in hs][:5])
    check("scope.bm25_statistics_fixed", all(h.scores["bm25"] is None or abs(h.scores["bm25"] - full[h.chunk]) < 1e-6 for h in hs))
    check("scope.ranks_within_scope", sorted(h.ranks["bm25"] for h in hs if h.ranks["bm25"])[:1] == [1])
    hu = hx.search(q, doc_ids=["T.nope.en"], k=5, query_vector=qv)
    check("scope.unknown_doc_reported", len(hu) == 0 and hu.diagnostics["scope_docs_unknown"] == ["T.nope.en"])
    try:
        hx.search(q, channels=("bm25", "colbert"))
        check("search.unknown_channel_rejected", False)
    except ValueError:
        check("search.unknown_channel_rejected", True)

    # query embedding through a client: only the question is embedded
    fe = FakeEmbed(dim=1024)
    fe_hits = hx.search(q, k=5, client=fe)
    check("search.query_only_embedded", fe.calls == 1 and fe.purposes == ["query-embed"] and fe.texts == 1
          and fe_hits.diagnostics["dense"]["query_vector"] == "embedded", fe.purposes)
    nov = hx.search(q, k=5)
    check("search.no_client_no_vector_dense_skipped", nov.diagnostics["dense"].get("skipped") and nov
          and all(h.ranks["dense"] is None for h in nov) and nov.diagnostics["warnings"], nov.diagnostics)
    return cs, cfg, hx


def test_missing_vectors(tmp):
    cs = make_set()
    cfg = tcfg(tmp / "idx2")
    fe = FakeEmbed()
    hx = H.HybridIndex.build(cs, cfg, client=fe, embed=False)
    check("missing.no_embedding_when_embed_false", fe.calls == 0 and not hx.dense_complete and int(hx.has_vec.sum()) == 0)
    hits = hx.search("steel output", k=10, client=fe)
    d = hits.diagnostics["dense"]
    check("missing.all_missing_reported", d["missing_vectors_in_scope"] == hx.n and d["complete"] is False
          and d.get("skipped") and hits.diagnostics["warnings"] and fe.calls == 0, d)
    check("missing.bm25_still_answers", len(hits) > 0 and all(h.ranks["bm25"] for h in hits))
    # a partial prebuild (one batch) -> some vectors; search never embeds documents
    P.prebuild(cs, cfg, fe, batch_size=4, max_batches=1, progress=False)
    hx2 = H.HybridIndex.load(cs, cfg)
    n_have = int(hx2.has_vec.sum())
    check("missing.partial_after_one_batch", 0 < n_have < hx2.n, (n_have, hx2.n))
    calls0 = fe.calls
    qv = np.asarray(fake_vector("steel output", dim=64), dtype=np.float32)
    h2 = hx2.search("steel output", k=hx2.n, query_vector=qv, client=fe)
    d2 = h2.diagnostics["dense"]
    check("missing.partial_reported", d2["missing_vectors_in_scope"] == hx2.n - n_have and d2["missing_docs"]
          and not d2["complete"] and any("prebuild" in w for w in h2.diagnostics["warnings"]), d2)
    check("missing.dense_only_ranks_vectors", all(hx2.has_vec[h.chunk] for h in h2 if h.ranks["dense"]))
    check("missing.no_document_embedding_at_query_time", fe.calls == calls0)
    return cs, cfg


def test_prebuild(tmp):
    cs = make_set()
    cfg = tcfg(tmp / "idx3")
    lines = []
    fe = FakeEmbed(fail_after=2)
    try:
        P.prebuild(cs, cfg, fe, batch_size=4, commit_every=1, progress=lambda d: lines.append(d["msg"]))
        check("prebuild.injected_failure_raises", False)
    except ModelError:
        check("prebuild.injected_failure_raises", True)
    cache = KVCache(H.embed_cache_path(cfg))
    n_cached = sum(1 for k in cache.keys() if k.startswith("v:"))
    cache.close()
    check("prebuild.committed_batches_survive", n_cached == 8, n_cached)
    check("prebuild.progress_lines", any("embedded 8/" in x for x in lines) and any("to embed" in x for x in lines), lines[:4])
    texts = {H.embed_text_key(cfg, H.index_text(c)) for c in H._chunks(cs, cfg, H.index_dir_for(cs, cfg))[0]}
    fe2 = FakeEmbed()
    out = P.main(["--set", "T_en", "--batch-size", "4", "--commit-every", "3"], cfg=cfg, client=fe2, sets={"T_en": cs})
    check("prebuild.resume_embeds_only_the_rest", fe2.texts == len(texts) - 8 and out[0]["dense_complete"], (fe2.texts, len(texts)))
    fe3 = FakeEmbed()
    P.main(["--set", "T_en"], cfg=cfg, client=fe3, sets={"T_en": cs})
    check("prebuild.rerun_is_noop", fe3.calls == 0)
    st = P.main(["--set", "T_en", "--status"], cfg=cfg, sets={"T_en": cs})
    check("prebuild.status", st[0]["dense_complete"] and st[0]["missing_vectors"] == 0, st)
    check("prebuild.log_written", (H.index_dir_for(cs, cfg) / "prebuild_log.jsonl").is_file())
    # a failure mid-run with commit_every > 1 still commits the finished batches (finally-flush)
    cfg4 = tcfg(tmp / "idx4")
    fe4 = FakeEmbed(fail_after=2)
    try:
        P.prebuild(cs, cfg4, fe4, batch_size=4, commit_every=5, progress=False)
    except ModelError:
        pass
    cache = KVCache(H.embed_cache_path(cfg4))
    n4 = sum(1 for k in cache.keys() if k.startswith("v:"))
    cache.close()
    check("prebuild.flush_on_failure", n4 == 8, n4)
    check("prebuild.low_priority_callable", isinstance(P.lower_priority("below_normal"), str))


# --------------------------------------------------------------------------------------------------------------
# evidence blocks
# --------------------------------------------------------------------------------------------------------------

def test_evidence(cs, cfg):
    alpha, beta = cs.doc("T.alpha.en"), cs.doc("T.beta.en")
    ops = alpha.unit("T.alpha.en::ops#p0")
    i = ops.text.index("Steel output")
    j = ops.text.index(".", i) + 1
    cand = {"doc_id": alpha.id, "span": [ops.span[0] + i, ops.span[0] + j], "ranks": {"bm25": 1, "dense": 3}, "rrf": 0.03}
    items, rep = build_blocks([cand], cs, budget=6000, question="steel output", cfg=cfg)
    b = items[0]
    check("prose.length_700_1000", len(items) == 1 and 700 <= len(b.text) <= 1000, len(b.text))
    check("prose.sentence_bounded", b.text[0].isupper() and b.text.rstrip()[-1] == "." and "Steel output reached" in b.text, b.text[:60])
    check("prose.verbatim_span", alpha.text[b.span[0]:b.span[1]] == b.text)
    check("prose.report_fields", rep["blocks"][0]["ranks"] == {"bm25": 1, "dense": 3} and rep["blocks"][0]["spans"] == [b.span]
          and rep["blocks"][0]["section_id"] == ops.section_id, rep["blocks"][0])

    # a chunk across two sections -> one block per section, none crossing a heading
    intro = alpha.unit("T.alpha.en::intro#p0")
    cross = {"doc_id": alpha.id, "span": [intro.span[1] - 50, ops.span[0] + 60]}
    items, rep = build_blocks([cross], cs, budget=6000, question="introduction operations", cfg=cfg)
    secs = [it.section_id for it in items]
    ok = len(items) == 2 and len(set(secs)) == 2
    for it in items:
        s = alpha.section(it.section_id)
        first = alpha.unit(s.unit_ids[0]).span[0]
        ok = ok and first <= it.span[0] and it.span[1] <= s.span[1]
    check("prose.no_cross_section", ok, [(it.section_id, it.span) for it in items])
    check("prose.title_not_duplicated", all("Operations review" not in it.text and "Introduction\n" not in it.text for it in items))

    # overlapping support in one section merges before selection
    c1 = {"doc_id": alpha.id, "span": [ops.span[0] + 100, ops.span[0] + 400], "rrf": 0.03}
    c2 = {"doc_id": alpha.id, "span": [ops.span[0] + 300, ops.span[0] + 600], "rrf": 0.02}
    items, rep = build_blocks([c1, c2], cs, budget=6000, question="operations", cfg=cfg)
    check("merge.overlapping", len(items) == 1 and items[0].meta["candidates"] == [0, 1] and rep["merged_supports"] >= 1, rep["blocks"])
    # directly adjacent units of one section merge
    p1 = alpha.unit("T.alpha.en::ops#p1")
    c3 = {"doc_id": alpha.id, "span": [ops.span[1] - 80, ops.span[1]]}
    c4 = {"doc_id": alpha.id, "span": [p1.span[0], p1.span[0] + 40]}
    items, rep = build_blocks([c3, c4], cs, budget=6000, question="maintenance", cfg=cfg)
    check("merge.adjacent_units", len(items) == 1 and ops.id in items[0].unit_ids and p1.id in items[0].unit_ids, rep["blocks"])
    # different sections never merge
    s1, s2 = alpha.unit("T.alpha.en::sec1#p0"), alpha.unit("T.alpha.en::sec2#p0")
    items, _ = build_blocks([{"doc_id": alpha.id, "span": list(s1.span)}, {"doc_id": alpha.id, "span": list(s2.span)}], cs,
                            budget=6000, cfg=cfg)
    check("merge.not_across_headings", len(items) == 2 and items[0].section_id != items[1].section_id)

    # tables: caption + heading + relevant rows only
    t0 = alpha.unit("T.alpha.en::t1#table.0")
    items, rep = build_blocks([{"doc_id": alpha.id, "span": list(t0.span), "ranks": {"bm25": 2, "dense": None}}], cs,
                              budget=6000, question="Mining revenue in 2022", cfg=cfg)
    tb = items[0]
    check("table.caption_heading_row", tb.meta["block"] == "table" and tb.text.split("\n")[0] == "Table 3: Revenue by segment and year"
          and "Segment | 2021 | 2022" in tb.text and "Mining | 50 | 55" in tb.text and "Steel | 100" not in tb.text, tb.text)
    check("table.rows_reported", rep["blocks"][0]["rows"] == [[t0.id, 1]] and tb.meta["caption_unit"] == "T.alpha.en::fin#cap"
          and "T.alpha.en::fin#cap" in tb.unit_ids, (rep["blocks"][0]["rows"], tb.unit_ids))
    check("table.non_contiguous_has_no_span", tb.span is None and len(tb.meta["spans"]) == 3)
    t1 = alpha.unit("T.alpha.en::t1#table.1")
    items, rep = build_blocks([{"doc_id": alpha.id, "span": list(t0.span)}, {"doc_id": alpha.id, "span": list(t1.span)}], cs,
                              budget=6000, cfg=cfg)
    tb = items[0]
    check("table.units_of_one_table_merge_by_row", len(items) == 1 and tb.text.count("Segment | 2021 | 2022") == 1
          and "Steel | 100 | 120" in tb.text and "Revenue total | 170 | 197" in tb.text, tb.text)
    bt = beta.unit("T.beta.en::t1#table.0")
    items, rep = build_blocks([{"doc_id": beta.id, "span": list(bt.span)}], cs, budget=6000, cfg=cfg)
    tabs = [it for it in items if it.meta["block"] == "table"]
    rows = [tuple(r) for it in tabs for r in it.meta["rows"]]
    check("table.max_1200_and_split_by_rows", len(tabs) >= 2 and all(len(it.text) <= 1200 for it in tabs)
          and len(rows) == len(set(rows)) and all(it.text.startswith("Table 3") for it in tabs), [len(it.text) for it in tabs])
    check("table.row_overlap_not_table_id", row_overlap(tabs[0].meta["rows"], tabs[1].meta["rows"]) == 0.0
          and block_overlap(tabs[0], tabs[1]) == 0.0 and block_overlap(tabs[0], tabs[0]) == 1.0)
    ts = table_structure(alpha, _DocView(alpha), t0)
    check("table.structure", ts["header"][0] == "Segment | 2021 | 2022" and len(ts["rows"][t0.id]) == 3
          and ts["caption"][0][2] == "T.alpha.en::fin#cap")

    # requirement reservation, entity grouping, caps
    secs = [alpha.unit("T.alpha.en::sec%d#p0" % s) for s in range(1, 7)]
    many = [{"doc_id": alpha.id, "span": list(u.span), "rrf": 0.03 - 0.001 * n} for n, u in enumerate(secs)]
    many.append({"doc_id": alpha.id, "span": list(alpha.unit("T.alpha.en::fin#p0").span), "rrf": 0.02})
    many.append({"doc_id": beta.id, "span": list(beta.unit("T.beta.en::fin#p0").span), "rrf": 0.01})
    items, rep = build_blocks(many, cs, budget=6000, question="revenue", cfg=cfg)
    check("select.ordinary_cap_6", len(items) == 6 and all(it.doc_id == alpha.id for it in items)
          and any(d["doc_id"] == beta.id and d["reason"] == "cap" for d in rep["dropped"]), (len(items), rep["n_dropped"]))
    reqs = [{"id": "a", "entity": "Alpha", "terms": ["revenue"], "period": "2021"},
            {"id": "b", "entity": "Beta", "terms": ["revenue"], "period": "2023"}]
    items, rep = build_blocks(many, cs, budget=6000, requirements=reqs, question="revenue", cfg=cfg)
    docs = [it.doc_id for it in items]
    check("select.requirement_reserved", beta.id in docs and rep["requirements"]["a"]["status"] == "covered"
          and rep["requirements"]["b"]["status"] == "covered", rep["requirements"])
    check("order.comparison_grouped_by_entity", rep["mode"] == "entity" and docs == sorted(docs, key=lambda d: d != alpha.id), docs)
    check("select.report_requirement_ids", any("b" in it.meta["reserved_for"] for it in items if it.doc_id == beta.id))
    reqs8 = [{"id": "s%d" % s, "terms": ["Topic%d keyword" % s]} for s in range(1, 7)] + reqs
    items, rep = build_blocks(many, cs, budget=12000, requirements=reqs8, cfg=cfg)
    check("select.up_to_8_with_requirements", len(items) == 8 and all(v["status"] == "covered" for v in rep["requirements"].values()),
          (len(items), rep["requirements"]))

    # series by period
    ser = [{"doc_id": beta.id, "span": list(beta.unit("T.beta.en::fin#p0").span)},
           {"doc_id": alpha.id, "span": list(alpha.unit("T.alpha.en::fin#p0").span)}]
    items, rep = build_blocks(ser, cs, budget=6000, question="Revenue in 2021 and 2023?", cfg=cfg)
    check("order.series_by_period", rep["mode"] == "period" and [it.doc_id for it in items] == [alpha.id, beta.id],
          [it.doc_id for it in items])
    items, rep = build_blocks(ser, cs, budget=6000, question="revenue", cfg=cfg)
    check("order.single_fact_by_relevance", rep["mode"] == "relevance" and [it.doc_id for it in items] == [beta.id, alpha.id])

    # exact budget: every selected block fits whole under the shared renderer
    for bud in (6000, 2500, 1500, 900):
        items, rep = build_blocks(many, cs, budget=bud, question="revenue topic", cfg=cfg)
        pk = pack_evidence(items, cs, budget=bud, cfg=cfg)
        check("budget.exact_%d" % bud, pk.n_included == len(items) and not pk.last_item_cut and pk.chars <= bud
              and pk.chars == rep["rendered_chars"] and len(items) >= 1, (pk.chars, len(items), rep["n_dropped"]))
    items, rep = build_blocks(many, cs, budget=1500, question="revenue topic", cfg=cfg)
    check("budget.dropped_listed", rep["n_dropped"]["budget"] + rep["n_dropped"]["cap"] > 0 and rep["dropped"][0]["reason"])
    check("budget.trim_sentence_bounded", all(not it.meta["trimmed"] or it.text.rstrip()[-1] == "." for it in items))

    # candidates without spans (unit ids), RetrievedItems, unknown docs
    ri = RetrievedItem(text="x", doc_id=alpha.id, unit_ids=["T.alpha.en::risk#p0"], kind="window", meta={"ranks": {"bm25": 4}})
    items, rep = build_blocks([ri, {"doc_id": "T.none.en", "span": [0, 5]}], cs, budget=6000, cfg=cfg)
    check("input.retrieved_items_and_unit_ids", len(items) == 1 and items[0].section_id == "T.alpha.en::risk"
          and rep["blocks"][0]["ranks"] == {"bm25": 4} and rep["unmapped"][0]["doc_id"] == "T.none.en", rep)


# --------------------------------------------------------------------------------------------------------------
# the HYBRID approach
# --------------------------------------------------------------------------------------------------------------

def test_approach(tmp):
    from ragbench.run import ALIASES, parse_flags
    check("approach.registered", APPROACHES.get("HYBRID") == "ragbench.approaches.hybrid:HybridRetrieval"
          and ALIASES.get("hybrid") == "HYBRID" and parse_flags(["blocks"], "HYBRID") == {"blocks": True})
    try:
        parse_flags(["no_facts"], "HYBRID")
        check("approach.unknown_flag_rejected", False)
    except SystemExit:
        check("approach.unknown_flag_rejected", True)
    cs = make_set()
    cfg = tcfg(tmp / "idx5")
    q = Question(id="t-steel", corpus="T", question_fa="", question_en="How many tonnes of steel did Alpha Corp produce?",
                 answer_fa="", answer_en="4.2 million tonnes", answer_type="numeric",
                 gold_units={"en": ["T.alpha.en::ops#p0"]}, doc_scope=["T.alpha.en"], type="lookup", answer_value=4.2e6)
    recs = {}
    for flags in (None, {"blocks": True}):
        fo = FakeOllama(cfg, Ledger(path=None))
        ap = get_approach("HYBRID", cfg=cfg, client=fo, flags=flags)
        ap.build_index(cs)
        n_idx_embeds = fo.embed_inputs
        rec = ap.answer(q, "en-en")
        recs[ap.variant] = rec
        tr = rec.trace or {}
        check("approach.%s.no_error" % ap.variant, rec.error is None, rec.error)
        check("approach.%s.query_only_embedded" % ap.variant, fo.embed_inputs - n_idx_embeds == 1 and rec.embed_calls == 1,
              (fo.embed_inputs, n_idx_embeds, rec.embed_calls))
        check("approach.%s.evidence_in_scope_and_budget" % ap.variant, 0 < rec.evidence_chars <= cfg.evidence["max_chars"]
              and all(it["doc_id"] == "T.alpha.en" for it in rec.evidence_items), rec.evidence_chars)
        check("approach.%s.trace" % ap.variant, tr.get("hybrid", {}).get("dense", {}).get("complete") and tr.get("dense_complete")
              and rec.llm_calls == 1 and tr["params"]["rrf_k"] == 60, list(tr))
        check("approach.%s.gold_in_evidence" % ap.variant, "T.alpha.en::ops#p0" in rec.evidence_units)
    check("approach.variant_names", set(recs) == {"HYBRID", "HYBRID+blocks"})
    rb = recs["HYBRID+blocks"]
    check("approach.blocks_report", rb.trace.get("blocks", {}).get("n_blocks") == len(rb.evidence_items)
          and all(it["kind"] == "block" and not it["cut"] for it in rb.evidence_items), rb.evidence_items)
    check("approach.same_shared_index", len({str(H.index_dir_for(cs, cfg))}) == 1
          and (H.index_dir_for(cs, cfg) / "hybrid_meta.json").is_file())


def main():
    with tempfile.TemporaryDirectory(prefix="ragbench_hybrid_", ignore_cleanup_errors=True) as d:
        tmp = Path(d)
        test_fusion()
        cs, cfg, _hx = test_index(tmp)
        test_missing_vectors(tmp)
        test_prebuild(tmp)
        test_evidence(cs, cfg)
        test_approach(tmp)
    bad = [k for k, v in RESULTS.items() if not v["ok"]]
    print("\n%d checks, %d failed%s" % (len(RESULTS), len(bad), (": " + ", ".join(bad)) if bad else ""))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
