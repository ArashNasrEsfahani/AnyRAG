"""Baselines tests (B0 NoRetrieval, B1 GoldEvidence, B2 BM25Retrieval, B3 VectorRetrieval).

    .venv/Scripts/python.exe -m ragbench.tests.test_baselines               # offline, no GPU
    .venv/Scripts/python.exe -m ragbench.tests.test_baselines --ollama      # + mini run on the real model
    .venv/Scripts/python.exe -m ragbench.tests.test_baselines --build-real  # + real B2 indexes for A_fa/A_en (CPU)

Offline: tokenizer; chunk invariants over every A_fa/A_en document and a synthetic edge-case document;
B0-B3 end to end through a fake Ollama (OllamaClient with _post replaced, so the real ledger, scopes,
build_stats and RunRecord code run) over a one-report slice; B3 resume after an injected embed failure.
Test indexes live in work/test_baselines/ (wiped for the offline part, kept for the mini run).

--ollama: 3 hand-made questions on one corpus-A report, fa-fa and en-en, B0-B3 (24 answer calls) plus
fa-en/en-fa for one question with B2 and B3 (4 calls); B3 embeds only that report (~750 chunks).
Runs are resumable (work/runs/test_baselines_mini.jsonl). No judge calls. Model-heavy: run alone.
"""
import argparse
import hashlib
import math
import re
import shutil
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[2]))

import numpy as np  # noqa: E402

from ragbench.config import load_config  # noqa: E402
from ragbench.contracts import Query, Question, RunLog  # noqa: E402
from ragbench.corpus import load_set  # noqa: E402
from ragbench.corpus.model import DocBuilder  # noqa: E402
from ragbench.corpus.registry import CorpusSet  # noqa: E402
from ragbench.llm import Ledger, ModelError, OllamaClient  # noqa: E402
from ragbench.approaches import get_approach  # noqa: E402
from ragbench.approaches import baselines as BL  # noqa: E402
from ragbench.util import read_json  # noqa: E402

RESULTS = {}


def check(name, cond, detail=""):
    RESULTS[name] = {"ok": bool(cond), "detail": detail}
    print(("PASS " if cond else "FAIL ") + name + (" - " + str(detail) if detail else ""), flush=True)
    return cond


# --------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------

# The original fixtures were three questions on one report of the development corpus A (not part of the public
# release). Neutral placeholders keep the structure; the corpus-A checks need that corpus to run.
SAMPLE_GOLD = ["front/front#lede", "front/front/kpis#kpi.0", "sec1/sec1#keyInsight", "sec1/sec1/table1#table.0"]


def sample_questions():
    scope = ["A.rpt-a.en"]
    return [
        Question(id="bl-sample-01", corpus="A",
                 question_fa="<پرسش نمونهٔ ۱>", question_en="<placeholder question 1>",
                 answer_fa="264,926 تن", answer_en="264,926 t", answer_type="numeric",
                 gold_units={l: ["A.rpt-a.%s::%s" % (l, u) for u in SAMPLE_GOLD] for l in ("fa", "en")},
                 doc_scope=scope, type="lookup", answer_value=264926, answer_unit="t"),
        Question(id="bl-sample-02", corpus="A",
                 question_fa="<پرسش نمونهٔ ۲>", question_en="<placeholder question 2>",
                 answer_fa="$372M", answer_en="$372M", answer_type="numeric",
                 gold_units={l: ["A.rpt-a.%s::sec2/sec2#lede" % l, "A.rpt-a.%s::sec2/sec2/table2#table.0" % l] for l in ("fa", "en")},
                 doc_scope=scope, type="lookup", answer_value=372e6, answer_unit="USD"),
        Question(id="bl-sample-03", corpus="A",
                 question_fa="<پرسش نمونهٔ ۳>", question_en="<placeholder question 3>",
                 answer_fa="160,000 تن", answer_en="160,000 t", answer_type="numeric",
                 gold_units={l: ["A.rpt-a.%s::sec3/sec3/table3#table.0" % l] for l in ("fa", "en")},
                 doc_scope=scope, type="lookup", answer_value=160000, answer_unit="t"),
    ]


def fake_vector(text, dim=1024):
    """Hashed bag of BM25 tokens: cosine ~ lexical overlap (plumbing tests only)."""
    v = np.zeros(dim, dtype=np.float32)
    for t in BL.tokenize(text):
        h = int(hashlib.sha1(t.encode("utf-8")).hexdigest()[:8], 16)
        v[h % dim] += 1.0 if (h >> 12) & 1 else -1.0
    n = float(np.linalg.norm(v))
    return (v / n if n else v).tolist()


class FakeOllama(OllamaClient):
    """Real OllamaClient (ledger, scopes, records) with the HTTP layer replaced."""

    def __init__(self, cfg, ledger, fail_embed_after=None):
        super().__init__(cfg, ledger)
        self.fail_embed_after = fail_embed_after
        self.embed_http = 0
        self.embed_inputs = 0
        self.chat_payloads = []

    def _post(self, endpoint, payload):
        if endpoint == "/api/embed":
            if self.fail_embed_after is not None and self.embed_http >= self.fail_embed_after:
                raise ModelError("injected embed failure")
            self.embed_http += 1
            self.embed_inputs += len(payload["input"])
            return ({"embeddings": [fake_vector(t) for t in payload["input"]],
                     "prompt_eval_count": sum(max(1, len(t) // 4) for t in payload["input"])}, 2.0, 0.0, 1)
        if endpoint == "/api/chat":
            self.chat_payloads.append(payload)
            chars = sum(len(m["content"]) for m in payload["messages"])
            return ({"message": {"content": "FAKE"}, "prompt_eval_count": chars // 4, "eval_count": 2,
                     "done_reason": "stop"}, 3.0, 0.0, 1)
        raise ModelError("unexpected endpoint %s" % endpoint)


def test_cfg(root):
    return load_config().copy_with({"paths": {"index_root": root}})


def sample_slice(lang):
    return load_set("A_%s" % lang).subset(["A.rpt-a.%s" % lang])


def gold_rank(rec, gold):
    for it in rec.retrieved_units:
        if set(gold) & set(it["unit_ids"]):
            return it["rank"]
    return None


# --------------------------------------------------------------------------
# offline tests
# --------------------------------------------------------------------------

def test_tokenizer():
    t = BL.tokenize
    check("tok.normalises_letters_and_digits", "کتاب" in t("كتاب ۱۲۳") and "123" in t("كتاب ۱۲۳"), t("كتاب ۱۲۳"))
    toks = t("شرکت‌ها و سرمایه‌گذاری")
    check("tok.zwnj_word_kept_plus_parts", "شرکت‌ها" in toks and "شرکت" in toks and "سرمایه‌گذاری" in toks
          and "ها" not in toks and "و" not in toks, toks)
    check("tok.stopwords_fa_en", t("what is the of و در از که چند") == [], t("what is the of و در از که چند"))
    check("tok.number_variants", t("264,926 t and 87%") == ["264,926", "264926", "87%", "87"], t("264,926 t and 87%"))
    check("tok.fold_alef_madda", t("آبزیان") == t("ابزیان"), (t("آبزیان"), t("ابزیان")))
    check("tok.harakat_do_not_split_words", t("مُدیریت اهمیّت برنامهٔ معمولاً") == t("مدیریت اهمیت برنامه معمولا") == ["مدیریت", "اهمیت", "برنامه", "معمولا"],
          (t("مُدیریت اهمیّت برنامهٔ معمولاً"), t("مدیریت اهمیت برنامه معمولا")))
    check("tok.month_may_kept", t("in May 2024") == ["may", "2024"], t("in May 2024"))
    check("tok.english_possessive", t("Iran's Türkiye’s") == ["iran's", "iran", "türkiye"], t("Iran's Türkiye’s"))
    check("tok.english_s_stemmer", t("farms tonnes companies glass status ماهی‌ها") == ["farm", "tonne", "company", "glass", "status", "ماهی‌ها", "ماهی"],
          t("farms tonnes companies glass status ماهی‌ها"))


def _chunk_invariants(doc, chunks, size, overlap):
    problems = []
    text = doc.text
    cov = np.zeros(len(text), dtype=bool)
    root_of = {}
    for s in doc.sections:
        root_of[s.id] = root_of[s.parent] if s.parent else s.id
    starts = {s.span[0] for s in doc.sections}
    ends = {s.span[1] for s in doc.sections}
    prev = None
    for c in chunks:
        a, b = c["span"]
        if text[a:b] != c["text"]:
            problems.append("span/text mismatch %s" % c["span"])
        if not (0 < b - a <= size) or not c["text"].strip():
            problems.append("bad length %d at %s" % (b - a, c["span"]))
        if c["unit_ids"] != [u.id for u in doc.units_overlapping(a, b)]:
            problems.append("unit_ids mismatch at %s" % c["span"])
        if not set(c["partial_unit_ids"]) <= set(c["unit_ids"]):
            problems.append("partial not subset at %s" % c["span"])
        if not c["section_ids"]:
            problems.append("no sections at %s" % c["span"])
        if not BL.DEFAULTS["chunker"]["merge_across_roots"] and len({root_of[s] for s in c["section_ids"]}) != 1:
            problems.append("crosses top-level sections at %s" % c["span"])
        if c["mode"] == "window":   # a window run = small glued blocks (< min_standalone_chars) + one oversized block, last
            glued = sum(doc.section(s).span[1] - doc.section(s).span[0] for s in c["section_ids"][:-1])
            last = doc.section(c["section_ids"][-1])
            if glued >= BL.DEFAULTS["chunker"]["min_standalone_chars"] or last.span[1] - last.span[0] <= size:
                problems.append("window run is not glue+oversized block at %s" % c["span"])
        if c["mode"] == "packed" and not (a in starts and b in ends):
            problems.append("packed chunk not on section boundaries at %s" % c["span"])
        if prev is not None and a < prev[1] and prev[1] - a > overlap:
            problems.append("overlap %d > %d at %s" % (prev[1] - a, overlap, c["span"]))
        cov[a:b] = True
        prev = (a, b)
    for u in doc.units:
        seg = cov[u.span[0]:u.span[1]]
        if not seg.all():
            missing = [i for i, ok in enumerate(seg) if not ok and not text[u.span[0] + i].isspace()]
            if missing:
                problems.append("unit %s has %d uncovered chars" % (u.id, len(missing)))
    return problems


def test_chunker(cfg):
    size, overlap = cfg.chunking["chunk_chars"], cfg.chunking["overlap_chars"]
    total, problems, stats = 0, [], {}
    for name in ("A_fa", "A_en"):
        cs = load_set(name)
        t0 = time.perf_counter()
        chunks = BL.build_chunks(cs, cfg)
        ms = (time.perf_counter() - t0) * 1000
        again = BL.build_chunks(cs, cfg)
        if BL.chunks_fingerprint(chunks) != BL.chunks_fingerprint(again):
            problems.append("%s: chunking not deterministic" % name)
        by_doc = {}
        for c in chunks:
            by_doc.setdefault(c["doc_id"], []).append(c)
        for d in cs.docs:
            problems += ["%s: %s" % (d.id, p) for p in _chunk_invariants(d, by_doc.get(d.id, []), size, overlap)]
        L = np.array([len(c["text"]) for c in chunks])
        stats[name] = "%d chunks in %.0f ms, chars mean %.0f p50 %d max %d, windows %d, <200 chars %d" % (
            len(chunks), ms, L.mean(), np.median(L), L.max(), sum(c["mode"] == "window" for c in chunks), int((L < 200).sum()))
        total += len(chunks)
    check("chunk.invariants_A_fa_A_en", not problems, "%d chunks; %s; problems: %s" % (total, stats, problems[:5]))

    # synthetic edge cases: a bare heading before an oversized section, a long line without breaks, a long table
    b = DocBuilder("A.synthetic.en", "A", "en", "Synthetic", source="test")
    b.add_section("h", "A short heading")
    b.add_section("h/big", "Big section", parent="h")
    b.add_unit("h/big", "h/big#p0", "paragraph", " ".join("word%d" % i for i in range(420)))
    b.add_unit("h/big", "h/big#p1", "paragraph", "One sentence here. " * 90)
    b.add_unit("h/big", "h/big#table.0", "table", "\n".join("row %d | %d | value" % (i, i * 7) for i in range(120)))
    b.add_section("tail", "Tail")
    b.add_unit("tail", "tail#p0", "paragraph", "A small closing paragraph.")
    doc = b.build()
    ch = BL.chunk_document(doc, size, overlap)
    probs = _chunk_invariants(doc, ch, size, overlap)
    glued = ch[0]["span"][0] == 0 and ch[0]["text"].startswith("A short heading\n\nBig section") and ch[0]["mode"] == "window"
    table_id = "A.synthetic.en::h/big#table.0"
    table_breaks = all(c["text"].endswith("value") for c in ch if c["unit_ids"] and c["unit_ids"][-1] == table_id)
    check("chunk.synthetic_edge_cases", not probs and glued and table_breaks and ch[-1]["section_ids"] == ["A.synthetic.en::tail"],
          "%d chunks %s; glued heading=%s; table windows end on row boundaries=%s; problems=%s" % (
              len(ch), [c["span"] for c in ch], glued, table_breaks, probs[:3]))


def test_b0_b1(cfg):
    ledger = Ledger()
    client = FakeOllama(cfg, ledger)
    q1, q2, _ = sample_questions()
    b0 = get_approach("B0", cfg, client)
    b0.build_index(sample_slice("en"))
    r = b0.answer(q1, "en-en")
    msgs = client.chat_payloads[-1]["messages"]
    check("B0.no_evidence_one_call", r.error is None and r.llm_calls == 1 and r.evidence_chars == 0 and not r.evidence_units
          and "own knowledge" in msgs[0]["content"] and "Evidence:" not in msgs[1]["content"]
          and q1.question_en in msgs[1]["content"], "calls=%d ev=%d err=%s" % (r.llm_calls, r.evidence_chars, r.error))
    check("B0.no_gold_in_query", b0.make_query(q1, "en-en").gold_unit_ids is None)

    b1 = get_approach("B1", cfg, client)
    b1.build_index(sample_slice("en"))
    b1.build_index(sample_slice("fa"))
    r = b1.answer(q2, "en-en")
    check("B1.gold_units_doc_language_en", r.error is None and r.evidence_units == q2.gold("en") and r.llm_calls == 1
          and "Evidence:" in client.chat_payloads[-1]["messages"][1]["content"],
          "evidence_units=%s chars=%d" % (r.evidence_units, r.evidence_chars))
    r = b1.answer(q2, "en-fa")
    user = client.chat_payloads[-1]["messages"][1]["content"]
    check("B1.cross_lingual_uses_fa_gold_en_question", r.evidence_units == q2.gold("fa") and q2.question_en in user
          and bool(user), "setting=%s units=%s" % (r.setting, r.evidence_units))
    bad = Question.from_dict(dict(q1.to_dict(), id="bl-missing", gold_units={"en": ["A.rpt-a.en::nope#x", "A.rpt-a.en::front/front#lede"]}))
    r = b1.answer(bad, "en-en")
    check("B1.missing_gold_reported", r.trace["missing_gold_units"] == ["A.rpt-a.en::nope#x"] and r.evidence_units == ["A.rpt-a.en::front/front#lede"],
          r.trace)
    un = Question.from_dict(dict(q1.to_dict(), id="bl-unans", answer_type="unanswerable", type="unanswerable", gold_units={}))
    r = b1.answer(un, "en-en")
    check("B1.unanswerable_empty_evidence", r.evidence_chars == 0 and "(no passages found)" in client.chat_payloads[-1]["messages"][1]["content"],
          "chars=%d" % r.evidence_chars)


def test_b2(cfg):
    ledger = Ledger()
    client = FakeOllama(cfg, ledger)
    qs = sample_questions()
    b2 = get_approach("B2", cfg, client)
    t0 = time.perf_counter()
    h_en = b2.build_index(sample_slice("en"))
    h_fa = b2.build_index(sample_slice("fa"))
    st = h_en.stats
    check("B2.index_build_stats", st["complete"] and st["size_bytes"] > 0 and st["totals"]["llm_calls"] == 0
          and st["totals"]["embed_calls"] == 0,
          "sample en: %d chunks, vocab %d, %d bytes; both built in %.1fs" % (h_en.payload.meta["n_chunks"], h_en.payload.meta["vocab"],
                                                                           st["size_bytes"], time.perf_counter() - t0))
    budget = cfg.evidence["max_chars"]
    ranks = {}
    ok = True
    for q in qs:
        for s in ("en-en", "fa-fa"):
            r = b2.answer(q, s)
            gold = q.gold(s[-2:])
            rk = gold_rank(r, gold)
            ranks["%s/%s" % (q.id, s)] = (rk, bool(set(gold) & set(r.evidence_units)), r.evidence_chars)
            ok = ok and r.error is None and r.llm_calls == 1 and r.embed_calls == 0 and r.evidence_chars <= budget \
                and all(it["kind"] == "chunk" for it in r.retrieved_units)
    in_ev = sum(1 for v in ranks.values() if v[1])
    check("B2.sample_questions_monolingual", ok and in_ev >= 5, "gold (rank, in evidence, evidence chars): %s" % ranks)
    r = b2.answer(qs[0], "en-en")
    check("B2.budget_filled", r.evidence_chars >= budget - cfg.chunking["chunk_chars"] - 200 and len(r.retrieved_units) >= cfg.chunking["top_k"],
          "evidence %d chars, offered %d, included %d, cut=%s, query tokens %s" % (
              r.evidence_chars, len(r.retrieved_units), len(r.evidence_items), r.trace["last_item_cut"], r.trace["query_tokens"]))
    r = b2.answer(qs[0], "fa-en")
    check("B2.cross_lingual_runs", r.error is None, "fa question over en doc: tokens in vocab %d, gold rank %s" % (
        r.trace["query_tokens_in_vocab"], gold_rank(r, qs[0].gold("en"))))

    # reload from disk in a fresh instance: same ranking
    b2b = get_approach("B2", cfg, FakeOllama(cfg, Ledger()))
    b2b.build_index(sample_slice("en"))
    r2 = b2b.answer(qs[0], "en-en")
    r1 = b2.answer(qs[0], "en-en")
    top1 = [(i["meta"]["chunk"], i["score"]) for i in r1.retrieved_units]
    top2 = [(i["meta"]["chunk"], i["score"]) for i in r2.retrieved_units]
    check("B2.reload_same_ranking", top1 == top2 and b2b._indexes["A_en"].stats["complete"], "top chunks %s" % top2[:8])

    # stopword-only query -> nothing retrieved, still one call with the empty-evidence prompt
    idx = b2._indexes["A_en"]
    qq = Query("bl-stop", "what is the of", "en", "en", "en-en", "A", "A_en", ["A.rpt-a.en"])
    with client.ledger.scope(phase="query", approach="B2"):
        res = b2._answer(qq, idx)
    check("B2.no_match_empty_evidence", not res.retrieved and res.evidence.chars == 0
          and "(no passages found)" in client.chat_payloads[-1]["messages"][1]["content"], res.trace)

    # doc_scope over the full A_en set
    full = get_approach("B2", cfg, FakeOllama(cfg, Ledger()))
    t0 = time.perf_counter()
    hf = full.build_index(load_set("A_en"))
    build_s = time.perf_counter() - t0
    q_scoped = Query("bl-scope", qs[0].question_en, "en", "en", "en-en", "A", "A_en", ["A.rpt-a.en"])
    q_corpus = Query("bl-corpus", "tonnes of imports by country", "en", "en", "en-en", "A", "A_en", "corpus")
    with full.client.ledger.scope(phase="query", approach="B2"):
        rs = full._answer(q_scoped, hf)
        rc = full._answer(q_corpus, hf)
    check("B2.doc_scope_filters", rs.retrieved and all(i.doc_id == "A.rpt-a.en" for i in rs.retrieved)
          and len({i.doc_id for i in rc.retrieved}) >= 1 and rc.trace["chunks_in_scope"] == hf.payload.meta["n_chunks"],
          "A_en index %d chunks built in %.1fs; scoped docs %s; corpus-wide docs %s" % (
              hf.payload.meta["n_chunks"], build_s, sorted({i.doc_id for i in rs.retrieved}), sorted({i.doc_id for i in rc.retrieved})))
    return h_en


def test_b3(cfg, b2_handle):
    qs = sample_questions()
    sl = sample_slice("en")
    # 1) injected failure after 3 embed batches
    ledger = Ledger()
    failing = FakeOllama(cfg, ledger, fail_embed_after=3)
    b3 = get_approach("B3", cfg, failing)
    try:
        b3.build_index(sl)
        crashed = False
    except ModelError:
        crashed = True
    st = read_json(b3.index_dir(sl) / "build_stats.json")
    check("B3.failed_build_recorded", crashed and st and st["complete"] is False and st["attempts"][-1]["ok"] is False
          and st["attempts"][-1]["embed_calls"] == 4 and st["attempts"][-1]["failed_calls"] == 1, "attempts=%s" % [(a.get("ok"), a.get("embed_calls"), a.get("failed_calls")) for a in st["attempts"]])
    # 2) resume: the 48 cached vectors are not re-embedded
    ledger = Ledger()
    client = FakeOllama(cfg, ledger)
    b3 = get_approach("B3", cfg, client)
    h = b3.build_index(sl)
    m = h.payload.meta
    bs = cfg.embedding["batch_size"]
    phases = {r["phase"] for r in ledger.records if r["kind"] == "embed"}
    check("B3.resume_uses_cache", m["from_cache"] == 3 * bs and m["embedded_now"] == m["unique_texts"] - 3 * bs
          and client.embed_inputs == m["embedded_now"] and h.stats["complete"] and phases == {"index"},
          "chunks=%d unique=%d from_cache=%d embedded_now=%d http=%d totals=%s" % (
              m["n_chunks"], m["unique_texts"], m["from_cache"], m["embedded_now"], client.embed_http, h.stats["totals"]))
    check("B3.same_chunks_as_B2", m["chunks_fingerprint"] == b2_handle.payload.meta["chunks_fingerprint"],
          "%s vs %s" % (m["chunks_fingerprint"][:10], b2_handle.payload.meta["chunks_fingerprint"][:10]))
    # 3) fresh instance loads from disk with no embed calls; one query embed per question
    ledger = Ledger()
    client = FakeOllama(cfg, ledger)
    b3b = get_approach("B3", cfg, client)
    hb = b3b.build_index(sl)
    loaded_calls = client.embed_http
    r = b3b.answer(qs[0], "en-en")
    qrec = [x for x in ledger.records if x["kind"] == "embed"]
    check("B3.load_and_query", loaded_calls == 0 and hb.payload.extra.shape == (m["n_chunks"], 1024) and r.error is None
          and r.embed_calls == 1 and r.llm_calls == 1 and qrec and qrec[-1]["phase"] == "query" and qrec[-1]["purpose"] == "query-embed"
          and bool(set(qs[0].gold("en")) & set(r.evidence_units)),
          "gold rank %s, evidence %d chars, err=%s" % (gold_rank(r, qs[0].gold("en")), r.evidence_chars, r.error))
    norms = np.linalg.norm(hb.payload.extra, axis=1)
    check("B3.vectors_normalised", bool(np.allclose(norms[norms > 0], 1.0, atol=1e-4)), "min %.4f max %.4f" % (norms.min(), norms.max()))


def test_index_neutral_flags(cfg):
    """doc_card / section_expand are query-time options: B2, B3 and HYBRID keep the plain index (fingerprint and
    index_dir) while the run variant name still carries them; other flags and other approaches are unchanged."""
    sl = sample_slice("en")
    combos = [{"doc_card": True}, {"section_expand": True}, {"doc_card": True, "section_expand": True}]
    bad, names = [], []
    for name in ("B2", "B3", "HYBRID"):
        plain = get_approach(name, cfg, FakeOllama(cfg, Ledger()))
        for fl in combos:
            ap = get_approach(name, cfg, FakeOllama(cfg, Ledger()), flags=fl)
            names.append(ap.variant)
            if ap.index_dir(sl) != plain.index_dir(sl) or ap.index_fingerprint(sl) != plain.index_fingerprint(sl) \
                    or ap.variant == plain.variant or ap.index_variant != name:
                bad.append((name, fl, str(ap.index_dir(sl))))
    check("flags.index_neutral_same_index", not bad, "variants %s; bad %s" % (names, bad))
    hb = get_approach("HYBRID", cfg, FakeOllama(cfg, Ledger()), flags={"blocks": True, "section_expand": True})
    hp = get_approach("HYBRID", cfg, FakeOllama(cfg, Ledger()))
    l2 = get_approach("LEX2", cfg, FakeOllama(cfg, Ledger()), flags={"doc_card": True})
    l2p = get_approach("LEX2", cfg, FakeOllama(cfg, Ledger()))
    check("flags.other_flags_still_index_keyed", hb.index_variant == "HYBRID+blocks" and hb.index_dir(sl) != hp.index_dir(sl)
          and l2.index_dir(sl) != l2p.index_dir(sl),
          "%s | %s" % (hb.index_dir(sl).name, l2.index_dir(sl).parent.name))
    # B3+section_expand over the slice test_b3 built: loads the plain index, no embedding at build time
    client = FakeOllama(cfg, Ledger())
    b3x = get_approach("B3", cfg, client, flags={"section_expand": True})
    h = b3x.build_index(sl)
    check("flags.B3_section_expand_reuses_plain_index", client.embed_http == 0 and h.stats["complete"]
          and h.path == get_approach("B3", cfg, FakeOllama(cfg, Ledger())).index_dir(sl),
          "embed http %d, path %s" % (client.embed_http, h.path))


# --------------------------------------------------------------------------
# optional: real indexes and the mini run
# --------------------------------------------------------------------------

def test_chunker_extracted(cfg):
    """Chunk invariants over whatever B/C extractions exist now (noisy headings, huge sections)."""
    size, overlap = cfg.chunking["chunk_chars"], cfg.chunking["overlap_chars"]
    for name in ("B", "C"):
        cs = load_set(name)
        if not len(cs):
            check("chunk.invariants_%s" % name, True, "no documents yet")
            continue
        chunks = BL.build_chunks(cs, cfg)
        by_doc = {}
        for c in chunks:
            by_doc.setdefault(c["doc_id"], []).append(c)
        problems = []
        for d in cs.docs:
            problems += ["%s: %s" % (d.id, p) for p in _chunk_invariants(d, by_doc.get(d.id, []), size, overlap)]
        L = np.array([len(c["text"]) for c in chunks])
        check("chunk.invariants_%s" % name, not problems, "%s -> %d chunks, chars mean %.0f p50 %d, <200 chars %d, windows %d; problems %s" % (
            cs, len(chunks), L.mean(), np.median(L), int((L < 200).sum()), sum(c["mode"] == "window" for c in chunks), problems[:3]))


def build_real_bm25():
    cfg = load_config()
    ap = get_approach("B2", cfg, FakeOllama(cfg, Ledger()))
    for name in ("A_fa", "A_en"):
        t0 = time.perf_counter()
        h = ap.build_index(load_set(name))
        check("real.B2.%s" % name, h.stats["complete"], "%s: %d chunks, vocab %d, %d bytes, %.1fs (build_ms total %s)" % (
            h.path, h.payload.meta["n_chunks"], h.payload.meta["vocab"], h.stats["size_bytes"], time.perf_counter() - t0,
            h.stats["totals"]["build_ms"]))


def _digits(s):
    return re.sub(r"[^0-9.]", "", (s or "").translate(str.maketrans("۰۱۲۳۴۵۶۷۸۹", "0123456789")).replace(",", ""))


def local_match(q, answer):
    """Cheap local check only (the real metric is the evaluation agent's judge + numeric match)."""
    gold = {"bl-sample-01": "264926", "bl-sample-02": "372", "bl-sample-03": "160000"}[q.id]
    return gold in _digits(answer.replace(" ", "")) or gold in re.sub(r"[,\s]", "", answer)


def mini_run(dry=False):
    if dry:     # the same code path with the fake client, to catch bugs before spending GPU time
        cfg = test_cfg("work/test_baselines/dry_indexes")
        shutil.rmtree(cfg.path("index_root"), ignore_errors=True)
        client = FakeOllama(cfg, Ledger())
        log_path = Path(cfg.path("index_root")) / "dry_runs.jsonl"
    else:
        cfg = test_cfg("work/test_baselines/mini_indexes")
        ledger = Ledger(path=str(cfg.ensure_dir("ledger_root") / "test_baselines_mini.jsonl"))
        client = OllamaClient(cfg, ledger)
        h = client.health()
        if not check("mini.ollama_health", h.get("reachable") and h.get("chat_present") and h.get("embed_present"),
                     {k: h.get(k) for k in ("version", "chat_digest_match", "embed_digest_match")}):
            return
        log_path = cfg.ensure_dir("runs_root") / "test_baselines_mini.jsonl"
    log = RunLog(log_path)
    done = log.completed()
    qs = sample_questions()
    aps = {n: get_approach(n, cfg, client) for n in ("B0", "B1", "B2", "B3")}
    aps["B3"].verbose = True
    for lang in ("en", "fa"):
        for n in ("B0", "B1", "B2", "B3"):
            t0 = time.perf_counter()
            hd = aps[n].build_index(sample_slice(lang))
            if n == "B3":
                mm = hd.payload.meta
                print("B3 index %s: %d chunks, embedded now %d, cached %d, %.0fs, totals %s" % (
                    hd.corpus_set, mm["n_chunks"], mm["embedded_now"], mm["from_cache"], time.perf_counter() - t0, hd.stats["totals"]), flush=True)
    jobs = [(q, s, n) for q in qs for s in ("en-en", "fa-fa") for n in ("B0", "B1", "B2", "B3")]
    jobs += [(qs[0], s, n) for s in ("fa-en", "en-fa") for n in ("B2", "B3")]
    rows = []
    for q, s, n in jobs:
        key = (q.id, aps[n].variant, s)
        if key in done:
            rec = next(r for r in log.records() if (r["question_id"], r["approach"], r["setting"]) == key and not r.get("error"))
        else:
            rec = aps[n].answer(q, s).to_dict()
            log.append(rec)
        gold = q.gold(s[-2:])
        rows.append(rec)
        print("%-3s %-5s %-22s match=%-5s recall=%-5s calls=%d emb=%d ptok=%-5d ctok=%-3d lat=%6.0fms load=%5.0f ev=%4d | %s" % (
            n, s, q.id, local_match(q, rec["answer_text"]), bool(set(gold) & set(rec["evidence_units"])) if n != "B0" else "-",
            rec["llm_calls"], rec["embed_calls"], rec["prompt_tokens"], rec["completion_tokens"], rec["latency_ms"], rec["load_ms"],
            rec["evidence_chars"], rec["answer_text"].replace("\n", " ")[:90]), flush=True)
    errs = [r for r in rows if r.get("error")]
    check("%s.all_records_ok" % ("dry" if dry else "mini"), not errs and all(r["llm_calls"] == 1 for r in rows),
          "%d records, errors: %s" % (len(rows), [e["error"] for e in errs][:3]))
    for n in ("B0", "B1", "B2", "B3"):
        mine = [r for r in rows if r["approach"] == n]
        qmap = {q.id: q for q in qs}
        acc = sum(local_match(qmap[r["question_id"]], r["answer_text"]) for r in mine)
        rec_ = sum(bool(set(qmap[r["question_id"]].gold(r["d_lang"])) & set(r["evidence_units"])) for r in mine)
        lat = sorted(r["latency_ms"] for r in mine)
        print("SUMMARY %s: local numeric match %d/%d, gold in evidence %d/%d, prompt tokens mean %.0f, latency p50 %.0f ms" % (
            n, acc, len(mine), rec_, len(mine), np.mean([r["prompt_tokens"] for r in mine]), lat[len(lat) // 2]), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ollama", action="store_true", help="mini run on the real model (model-heavy)")
    ap.add_argument("--build-real", action="store_true", help="build real B2 indexes for A_fa and A_en (CPU only)")
    ap.add_argument("--extracted", action="store_true", help="chunk invariants over the current B/C extractions (CPU)")
    ap.add_argument("--dry-mini", action="store_true", help="the mini run with the fake client (no GPU)")
    ap.add_argument("--skip-offline", action="store_true")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    if args.extracted:
        test_chunker_extracted(load_config())
    if not args.skip_offline:
        cfg = test_cfg("work/test_baselines/indexes")
        shutil.rmtree(cfg.path("index_root"), ignore_errors=True)
        test_tokenizer()
        test_chunker(cfg)
        test_b0_b1(cfg)
        h = test_b2(cfg)
        test_b3(cfg, h)
        test_index_neutral_flags(cfg)
    if args.build_real:
        build_real_bm25()
    if args.dry_mini:
        mini_run(dry=True)
    if args.ollama:
        mini_run()
    fails = [k for k, v in RESULTS.items() if not v["ok"]]
    print("\n%d checks, %d failed%s" % (len(RESULTS), len(fails), (": " + ", ".join(fails)) if fails else ""))
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
