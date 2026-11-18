"""Offline tests for LEX (no model: a scripted client; the live knowledge base is read, never written).

    .venv/Scripts/python.exe -m ragbench.approaches.lex.test_lex

  flags     the ablation flags, their exclusivity, variant names, the window value
  compile   terms -> regexes over folded text: ZWNJ / space / no space, plurals, yeh / kaf variants
  regex     a model-written regex keeps its escapes through folding; a bad or empty-matching one is reported
  route     layer 1 over the live cards: Divar's headcount question finds Divar's reports, every year of the series
  vocab     the reports' own name for the measure («افراد شاغل») reaches call 2's list from «نیروی کار»
  windows   a structure window stays inside its section and its budget; a table window keeps the header line; a fixed
            window is a verbatim slice of the report
  learned   the lasso, the window features, and the fitted scorer wired into windows()
  session   a whole question with a scripted model: 3 calls (2 with free_keywords), the evidence within the shared
            budget, the trace; llm_regex with no valid regex falls back and says so
"""
import json
import sys
from types import SimpleNamespace

from ...config import load_config
from ...contracts import Query
from ...corpus.registry import load_set
from ...kb import ingest as I
from ...kb.schema import connect
from ...llm import ChatResult
from ...normalize import fold
from . import search as S
from .approach import LexAgent, _Session, parse_flags
from .defaults import DEFAULTS

FAILS = []
DIVAR = ["C.divar_annual-report_%d.fa" % y for y in (1397, 1398, 1399, 1400)]


def check(cond, label):
    print(("  ok   " if cond else "  FAIL ") + label)
    if not cond:
        FAILS.append(label)


def flag_tests():
    print("\n[flags]")
    check(parse_flags("llm_regex,window=300") == {"llm_regex": True, "window": 300}, "parse a comma list with a value")
    for bad in ("llm_regex,free_keywords", "window=20", "no_such_flag"):
        try:
            parse_flags(bad)
            check(False, "refused: %s" % bad)
        except ValueError:
            check(True, "refused: %s" % bad)
    a = LexAgent(flags={"fixed_window": True, "window": 300})
    check(a.variant == "LEX+fixed_window,window=300" and a.opts["window_chars"] == 300 and a.max_calls == 3,
          "variant name, window override, the 3-call cap: %s" % a.variant)


def compile_tests():
    print("\n[compile]")
    o = dict(DEFAULTS)
    t = S.compile_terms(["نرم‌افزار"], "fa", "k", 1.0, o)[0]
    check(all(t.search(fold(x)) >= 0 for x in ("نرم‌افزار", "نرم افزار", "نرمافزار")), "ZWNJ, space and no space")
    t = S.compile_terms(["کاربران"], "fa", "k", 1.0, o)[0]
    check(t.search(fold("تعداد کاربر فعال")) >= 0, "a plural keyword finds the singular")
    t = S.compile_terms(["کارکنان"], "fa", "k", 1.0, o)[0]
    check(t.search(fold("كاركنان شركت")) >= 0, "Arabic kaf / yeh in the text")
    t = S.compile_terms(["active users"], "en", "k", 1.0, o)[0]
    check(t.search(fold("Active User base")) >= 0, "English: case and plural")
    weak = S.compile_terms(["تعداد", "میزان رشد"], "fa", "k", 1.0, o)
    check(all(x.weak for x in weak), "a term of measure words only is weak")
    # second opinion, 2026-09-22: word boundaries, measure-aware AND, phrase above relaxed
    t = S.compile_terms(["کار"], "fa", "k", 1.0, o)[0]
    check(all(t.search(fold(x)) < 0 for x in ("کاربران فعال", "کسب‌وکار کوچک")) and
          all(t.search(fold(x)) >= 0 for x in ("بازار کار ایران", "کارها")),
          "a short word starts AND ends a word: «کار» is not inside «کاربر» or «کسب‌وکار»")
    t = S.compile_terms(["سهم واردات"], "fa", "k", 1.0, o)[0]
    check(t.search(fold("ارزش واردات کشور")) < 0, "the relaxed match keeps measure words: «سهم واردات» needs «سهم»")
    off, rel = t.match(fold("سهم ایران از واردات"))
    check(off >= 0 and rel is True, "... and all its words in one text is a RELAXED match")
    off, rel = t.match(fold("سهم واردات از چین"))
    check(off >= 0 and rel is False, "the phrase itself is not relaxed")
    t = S.compile_terms(["تعداد کارکنان"], "fa", "k", 1.0, o)[0]
    check(t.search(fold("کارکنان شرکت: 350 نفر")) >= 0, "a count word is not required («تعداد کارکنان» finds «کارکنان»)")
    check(S.compile_terms(["نرخ رشد"], "fa", "k", 1.0, o)[0].words == [],
          "a term of measure words only gets no relaxed match (it would match any growth rate)")
    check(S.term_weight(t, dict(o), True) < S.term_weight(t, dict(o), False) or t.idf == 0,
          "a relaxed match weighs less than the phrase")


def regex_tests():
    print("\n[regex]")
    check(S.fold_regex(r"\D+\d{2}") == r"\D+\d{2}", "escapes survive folding (\\D is not lower-cased): %s"
          % S.fold_regex(r"\D+\d{2}"))
    terms, bad = S.compile_regexes([r"كاربر(ان)?\s+فعال", r"(", r".*", r"x?"])
    check(len(terms) == 1 and terms[0].search(fold("کاربران فعال")) >= 0, "a Persian regex is folded and matches")
    check(len(bad) == 3 and any("compile" in e for _p, e in bad) and any("empty" in e for _p, e in bad),
          "a broken regex and empty-matching ones are reported, not used: %s" % bad)


def route_tests(con):
    print("\n[route]")
    o = dict(DEFAULTS)
    pool = [r["doc_id"] for r in con.execute("SELECT doc_id FROM documents WHERE corpus = 'C' AND lang = 'fa' "
                                             "AND status IN ('ready', 'vector_ready')")]
    kw = {"keywords": ["کارکنان", "نیروی کار", "تعداد پرسنل"], "names": ["دیوار"], "years": ["1397", "1398", "1399", "1400"],
          "answer": "number"}
    keep, scored = S.route(con, pool, kw, "fa", o)
    check(set(DIVAR) <= set(keep), "Divar's headcount series routes to all four Divar annual reports (of %d): %s"
          % (len(pool), keep))
    keep_t, _s = S.route(con, pool, kw, "fa", o, titles_only=True)
    check(keep_t[0].startswith("C.divar"), "titles only still finds Divar first: %s" % keep_t[:3])


def vocab_tests(con):
    print("\n[vocab]")
    o = dict(DEFAULTS)
    rep = S.Reports(con)
    kw = {"keywords": ["نیروی کار", "کارکنان", "شاغل"], "names": [], "years": [], "answer": "number"}
    terms = S.compile_terms(kw["keywords"], "fa", "keyword", 1.0, o)
    hits, by_id = S.search(rep, DIVAR, terms)
    wins, _ms = S.windows(rep, hits, by_id, terms, kw, o)
    voc = S.vocabulary(con, rep, DIVAR, kw, "fa", wins, o)
    texts = [fold(v["text"]) for v in voc]
    check(any("افراد شاغل" in t for t in texts), "the reports' own «افراد شاغل» is in the list (%d phrases)" % len(voc))
    check(len(voc) <= o["vocab_max"], "at most vocab_max phrases")
    return rep, terms, hits, by_id, kw


def window_tests(con, rep, terms, hits, by_id, kw, corpus):
    print("\n[windows]")
    o = dict(DEFAULTS)
    wins, _ms = S.windows(rep, hits, by_id, terms, kw, o)
    ok = True
    for w in wins[:30]:
        u = by_id[w["hit"]]
        if u.kind == "table":
            continue
        same_sec = all(by_id[x].section_id == u.section_id for x in w["unit_ids"] if x in by_id)
        within = len(w["text"]) <= o["window_chars"] or len(w["unit_ids"]) == 1
        ok = ok and same_sec and within
    check(ok, "structure windows stay in their section and under window_chars (%d windows)" % len(wins))
    doc = corpus.doc(DIVAR[3])
    w0 = next(w for w in wins if w["span"] and w["doc_id"] == DIVAR[3])
    check(doc.text[w0["span"][0]:w0["span"][1]] == w0["text"], "a structure window is a verbatim slice of the report")
    fixed, _ms = S.windows(rep, hits, by_id, terms, kw, o, mode="fixed", doc_texts={d: corpus.doc(d).text for d in DIVAR})
    check(all(len(w["text"]) <= o["window_chars"] and corpus.doc(w["doc_id"]).text[w["span"][0]:w["span"][1]] == w["text"]
              for w in fixed), "fixed windows: at most window_chars, verbatim")
    # a table window keeps its header line, and the MATCHING row even when it sits late in a long table
    row = con.execute("SELECT unit_id, doc_id FROM units WHERE kind = 'table' AND length(text) > 2000 LIMIT 1").fetchone()
    rep.load(row["doc_id"])
    u = next(x for x in rep.units[row["doc_id"]] if x.id == row["unit_id"])
    lines = u.text.split("\n")
    late = len(lines) - 2
    lang = "fa" if row["doc_id"].endswith(".fa") else "en"
    label = next((c.strip() for c in lines[late].split("|") if sum(ch.isalpha() for ch in c) >= 4), lines[late].split("|")[0])
    t = S.compile_terms([label], lang, "k", 1.0, o)
    w = S.structure_window(rep, u, t, o)
    check(w["text"].split("\n")[0] == lines[0] and len(w["text"]) < len(u.text),
          "a long table's window: its header line + the matching rows (%d of %d chars)" % (len(w["text"]), len(u.text)))
    check(any(fold(label) in fold(x) for x in w["text"].split("\n")[1:]) and w["rows"][0] == 0,
          "the matching row near the END of the table is kept, and the rows kept are recorded (%s of %d)"
          % (w["rows"][:6], w["rows_total"]))
    # credit is taken from the emitted text: a term that matches only an omitted row is not credited
    other = next((c.strip() for ln in lines[1:4] for c in ln.split("|") if sum(ch.isalpha() for ch in c) >= 4
                  and fold(c.strip()) not in fold(w["text"])), None)
    if other:
        t2 = S.compile_terms([label, other], lang, "k", 1.0, o)
        for x in t2:
            x.idf = 1.0
        wf = "\n".join(fold(x) for x in S.structure_window(rep, u, t2[:1], o)["text"].split("\n"))
        got, _sc = S.credit(t2, wf, o, table=True)
        check(0 in got and 1 not in got, "a term matching only a row the window left out is not credited")

    def win(d, hit, score, span, text="x" * 300, units=None):
        return {"doc_id": d, "hit": hit, "score": score, "span": span, "unit_ids": units or [hit], "text": text,
                "kind": "window"}
    ws = [win("a", "a1", 9, [0, 300]), win("a", "a2", 8, [100, 400]), win("a", "a3", 7, [1000, 1300]),
          win("b", "b1", 2, [0, 300]), win("c", "c1", 1, [0, 300])]
    cost = lambda w: len(w["text"]) + 100                     # noqa: E731
    sel, info = S.select(ws, ["a", "b", "c"], o, reserve=True, cost=cost, budget=6000)
    check([w["hit"] for w in sel][:3] == ["a1", "b1", "c1"] and "a2" not in [w["hit"] for w in sel],
          "reserve: one window per report first; a window mostly shown by a chosen one is skipped (%s)"
          % [w["hit"] for w in sel])
    sel, info = S.select(ws, ["a", "b", "c"], o, reserve=False, cost=cost, budget=6000)
    check([w["hit"] for w in sel] == ["a1", "a3", "b1", "c1"], "no reserve: by score (%s)" % [w["hit"] for w in sel])
    sel, info = S.select(ws, ["a", "b", "c"], o, reserve=False, cost=cost, budget=900)
    check(sum(cost(w) for w in sel) <= 900 and info["skipped_budget"] > 0,
          "chosen by the characters they cost: nothing for the packer to cut (%d chars)" % sum(cost(w) for w in sel))
    sel, _i = S.select(ws, ["a", "b", "c"], o, reserve=False, cost=cost, budget=6000, prior={"b": 20.0})
    check(sel[0]["hit"] == "b1", "a routed report's card prior moves its windows up")


class FakeClient:
    """Scripted replies by purpose; records every call."""

    def __init__(self, replies):
        self.replies, self.calls = replies, []
        self.ledger = SimpleNamespace(context=lambda: {}, record=lambda rec: None, path=None)

    def options(self, num_predict=None, **_kw):
        return {"num_predict": num_predict}

    def chat(self, messages, *, purpose, format=None, num_predict=None, **_kw):
        self.calls.append(purpose)
        text = self.replies[purpose] if isinstance(self.replies[purpose], str) else json.dumps(self.replies[purpose],
                                                                                               ensure_ascii=False)
        return ChatResult(text=text, prompt_tokens=10, completion_tokens=5, wall_ms=1.0, done_reason="stop",
                          model="fake", call_id="x%d" % len(self.calls))


def session_tests(cfg, corpus):
    print("\n[session]")
    q = Query(question_id="t", text="بنا بر گزارش‌های سالانهٔ دیوار از 1397 تا 1400، دیوار در هر سال چند نفر نیروی کار "
              "داشته است؟", q_lang="fa", d_lang="fa", setting="fa-fa", corpus="C", corpus_set=corpus.name,
              doc_scope=list(DIVAR), answer_format="timeline")
    replies = {"keywords": {"keywords": ["نیروی کار", "کارکنان", "شاغل"], "names": ["دیوار"],
                            "years": ["1397", "1400"], "answer": "number"},
               "pick": {"pick": [1, 2, 99], "add": ["پرسنل"]}, "regex": {"regex": ["(", ".*"]},
               "answer": "1397: 350 نفر"}
    index = SimpleNamespace(corpus=corpus, payload={})

    def run(flags):
        client = FakeClient(replies)
        ap = LexAgent(cfg=cfg, client=client, flags=flags, options={"call_cache": "off"})
        res = _Session(ap, q, index).run()
        return client, res

    client, res = run({})
    tr = res.trace
    check(client.calls == ["keywords", "pick", "answer"], "3 calls: keywords, pick, answer (%s)" % client.calls)
    check(res.evidence.chars <= cfg.evidence["max_chars"] and res.evidence.n_included > 0,
          "the evidence fits the shared budget (%d chars, %d items)" % (res.evidence.chars, res.evidence.n_included))
    check(res.answer_text == "1397: 350 نفر", "the answer is the answer call's text")
    check(len(tr["stages"]["call2"]["picked"]) == 2 and tr["stages"]["call2"]["added"] == ["پرسنل"],
          "an out-of-range pick is dropped, the added term kept")
    check({w["doc_id"] for w in tr["stages"]["search"]["selected"]} == set(DIVAR),
          "every report of the series has a window")
    check(tr["stages"]["route"]["routed"] == DIVAR and tr["stages"]["route"]["recall"] == 1.0,
          "a given scope of 4 reports is not re-routed")
    client, res = run({"free_keywords": True})
    check(client.calls == ["keywords", "answer"], "free_keywords: 2 calls (%s)" % client.calls)
    client, res = run({"llm_regex": True})
    c2 = res.trace["stages"]["call2"]
    check(client.calls == ["keywords", "regex", "answer"] and c2.get("regex_fallback") and len(c2["regex_invalid"]) == 2,
          "llm_regex with no usable regex: falls back to the keywords and records why")
    client, res = run({"open_corpus": True})
    rt = res.trace["stages"]["route"]
    check(rt["mode"] == "open_corpus" and rt["pool"] > 4 and rt["recall"] == 1.0,
          "open_corpus: routed among %d reports, all 4 of the question's found" % rt["pool"])


def upgrade_tests(cfg, con, corpus):
    """LEX2's upgrades (plus.py) and the lite card."""
    from . import plus as X
    from .approach import Lex2Agent
    print("\n[upgrades]")
    o = dict(DEFAULTS)
    rep = S.Reports(con)
    bm, info = X.bm25_windows(rep, DIVAR, ["تعداد کارکنان دیوار", "نیروی کار"], "fa", o)
    check(bm and bm[0]["score"] > 0 and {w["doc_id"] for w in bm} <= set(DIVAR) and len(bm) <= o["bm25_top_k"],
          "BM25 over structure chunks of the chosen reports (%d chunks, top %.2f)" % (info["chunks"], bm[0]["score"]))
    lex = [dict(bm[0], score=5.0, source=None), {"doc_id": "Z", "hit": "z", "score": 4.0, "span": [0, 10],
                                                  "unit_ids": ["z"], "text": "z" * 10, "kind": "window"}]
    fused = X.fuse(lex, bm[:3], o)
    check(fused[0]["source"] == "both" and sum(1 for w in fused if w["source"] == "bm25") == 2,
          "fusion: a BM25 chunk a LEX window already shows adds its rank to it; the others enter as bm25")
    sel, sinfo = S.select(fused + [dict(w, hit=w["hit"] + "x", span=[w["span"][0] + 5000, w["span"][1] + 5000]
                                        if w["span"] else None) for w in bm[3:10]], DIVAR, o,
                          cost=lambda w: len(w["text"]) + 100, budget=6000, source_caps={"bm25": 1200})
    check(sinfo["by_source"].get("bm25", 0) <= 1200, "BM25-only windows stay within bm25_max_chars (%s)"
          % sinfo["by_source"])
    kw = {"keywords": ["نیروی کار", "کارکنان", "شاغل"], "names": [], "years": [], "answer": "number"}
    terms = S.compile_terms(kw["keywords"], "fa", "keyword", 1.0, o)
    hits, by_id = S.search(rep, DIVAR, terms)
    wins, _ms = S.windows(rep, hits, by_id, terms, kw, o)
    glob = S.vocabulary(con, rep, DIVAR, kw, "fa", wins, dict(o, vocab_max=12))
    per = S.vocabulary(con, rep, DIVAR, kw, "fa", wins, dict(o, vocab_max=12, report_vocab=True))
    cover = lambda voc: sum(1 for d in DIVAR if any(d in v["docs"] for v in voc))          # noqa: E731
    check(cover(per) >= cover(glob) and cover(per) == sum(1 for d in DIVAR if any(d in v["docs"] for v in S.vocabulary(
        con, rep, DIVAR, kw, "fa", wins, dict(o, vocab_max=400)))), "report_vocab: every report that has a phrase gets "
          "its own share of a short list (%d vs %d of 4 reports)" % (cover(per), cover(glob)))
    lite = S.vocabulary(con, rep, DIVAR, kw, "fa", wins, dict(o, no_labels=True, vocab_max=400))
    titles = {fold(r["title"]) for r in con.execute("SELECT title FROM sections WHERE doc_id IN (%s) AND title IS NOT NULL"
                                                    % ",".join("?" * 4), DIVAR)}
    check(lite and all(fold(v["text"]) in titles for v in lite), "no_labels: the phrase list holds section titles only")
    toks = S.card_tokens(con, DIVAR[:1], no_labels=True)[DIVAR[0]]
    full = S.card_tokens(con, DIVAR[:1])[DIVAR[0]]
    check(0 < len(toks) < len(full), "no_labels: a card without metric names (%d vs %d tokens)" % (len(toks), len(full)))
    c = X.cost_from_scratch(con, DIVAR)
    check(c["lex_ms"] > 0 and "facts" in c["stages_ms"], "cost from scratch, from the ingestion log: %.0f ms for LEX's "
          "stages (%s)" % (c["lex_ms"], sorted(c["stages_ms"])))
    # the missing-year pass: a supported 1398 window, nothing for 1400
    picked = S.compile_terms(["تعدادکل پرسنل"], "fa", "picked", 1.0, o)
    h2, b2 = S.search(rep, [DIVAR[1]], picked)
    w2, _ms = S.windows(rep, h2, b2, picked, kw, o)
    sup = [w for w in w2 if X.supports(w, {"تعدادکل پرسنل"})]
    if sup:
        kwy = dict(kw, years=["1397", "1400"])
        out, yinfo = X.year_pass(con, rep, DIVAR, sup[:1], picked, kwy, "fa", o, lambda w: len(w["text"]) + 100, 6000)
        check(yinfo["ran"] and DIVAR[3] in yinfo["missing"] and DIVAR[1] not in yinfo["missing"],
              "year_pass: the years with no supporting window are found (%d missing, %d filled, labels %s)"
              % (len(yinfo["missing"]), len(yinfo["filled"]), yinfo["labels"][:3]))
    else:
        check(False, "year_pass: a supporting 1398 window exists to start from")
    # LEX2, a whole question with the scripted model
    q = Query(question_id="t", text="بنا بر گزارش‌های سالانهٔ دیوار از 1397 تا 1400، دیوار در هر سال چند نفر نیروی کار "
              "داشته است؟", q_lang="fa", d_lang="fa", setting="fa-fa", corpus="C", corpus_set=corpus.name,
              doc_scope=list(DIVAR), answer_format="timeline")
    replies = {"keywords": {"keywords": ["نیروی کار", "کارکنان", "شاغل"], "names": ["دیوار"], "years": ["1397", "1400"],
                            "answer": "number"}, "pick": {"pick": [1, 2, 3], "add": []}, "answer": "ok"}
    client = FakeClient(replies)
    ap = Lex2Agent(cfg=cfg, client=client, options={"call_cache": "off"})
    res = _Session(ap, q, SimpleNamespace(corpus=corpus, payload={})).run()
    st = res.trace["stages"]["search"]
    check(client.calls == ["keywords", "pick", "answer"] and st["bm25"] and st["year_pass"] and st["year_pass"]["ran"]
          and res.evidence.chars <= cfg.evidence["max_chars"] and st["select"]["by_source"].get("bm25", 0) <= 1200,
          "LEX2: 3 calls, the BM25 net and the year pass ran, the evidence fits (%d chars; by source %s)"
          % (res.evidence.chars, st["select"]["by_source"]))
    msgs = [m for m in client.calls]
    check(any("(" in v for v in res.trace["stages"]["call2"]["vocab"]) is False, "the phrase list itself stays plain "
          "text (the years are added only in the prompt)")



def learn_tests(rep, terms0, hits0, by_id0, kw0):
    """The learned window score (learn.py): the lasso itself, the features, and the scorer wired into windows()."""
    import numpy as np

    from . import learn as LN
    print("\n[learned score]")

    # 1. the lasso recovers a sparse signal and zeroes the rest
    rng = np.random.RandomState(0)
    X = rng.binomial(1, 0.3, size=(300, 40)).astype(float)
    beta = np.zeros(40)
    beta[3], beta[17] = 2.0, -1.5
    y = X @ beta + 0.05 * rng.randn(300)
    b0, b = LN.lasso(X, y, 0.02)
    big = sorted(range(40), key=lambda j: -abs(b[j]))[:2]
    check(set(big) == {3, 17}, "lasso: the two true columns are the two largest coefficients (%.2f, %.2f)"
          % (b[3], b[17]))
    check(sum(1 for v in b if v != 0) < 40, "lasso: the penalty zeroes coefficients (%d of 40 nonzero)"
          % sum(1 for v in b if v != 0))
    check(LN.lam_max(X, y) > 0 and all(v == 0 for v in LN.lasso(X, y, LN.lam_max(X, y) * 1.01)[1]),
          "lasso: the path top zeroes every coefficient")

    # 2. features: one column per matched phrase, plus the generic block
    o = dict(DEFAULTS)
    terms = S.compile_terms(["سود خالص", "تعداد"], "fa", "picked", 1.0, o)
    for t in terms:
        t.idf = 2.0
    w = {"text": "سود خالص شرکت در سال 1399 برابر 120 میلیارد ریال بود", "kind": "window", "table": False,
         "hand_score": 1.25}
    got = {0: False}
    f = LN.features(w, got, terms, {"answer": "number"}, ["1399"], [])
    check(f.get("ph:سود خالص") == 1.0, "features: the matched phrase gets its own column")
    check(f.get("idf_phrase") == 2.0 and f.get("n_phrase") == 1.0, "features: the phrase match carries its idf")
    check(f.get("has_digit") == 1.0 and f.get("number_wanted") == 1.0 and f.get("year_hit") == 1.0,
          "features: number, number-wanted and year-hit fire")
    check("ph:تعداد" not in f, "features: a term that did not match has no column")

    # 3. a model scores by dot product and ignores a phrase it never saw
    m = LN.Model(0.5, {"ph:سود خالص": 2.0, "has_digit": 1.0})
    check(abs(m.score(f) - 3.5) < 1e-9, "model: bias + the weights of the features present (%.2f)" % m.score(f))
    check(abs(m.score({"ph:چیز تازه": 1.0}) - 0.5) < 1e-9, "model: an unseen phrase contributes nothing")

    # 4. the scorer changes the order of real windows, and keeps the hand score next to it
    kterms, hits, by_id, kw = terms0, hits0, by_id0, kw0
    plain, _ms = S.windows(rep, hits, by_id, kterms, kw, o)
    flat = LN.Model(0.0, {"is_table": 5.0})            # a model that wants tables first
    scored, _ms = S.windows(rep, hits, by_id, kterms, kw, o, score_fn=LN.scorer(flat))
    check(len(plain) == len(scored) > 0, "windows: the candidate set is the same, only the score changes (%d)"
          % len(scored))
    check(all("hand_score" in w for w in scored), "windows: the hand-set score is kept beside the learned one")
    check([w["score"] for w in scored] != [w["score"] for w in plain], "windows: the learned score is the one used")
    keep, _ms = S.windows(rep, hits, by_id, kterms, kw, o, score_fn=LN.scorer(None, keep=True))
    check(all("feats" in w for w in keep) and [w["score"] for w in keep] == [w["score"] for w in plain],
          "windows: the dump keeps the hand-set score and collects a feature vector per window")

    # 5. dataset, fit and folds: questions never straddle a fold
    d = LN.Dataset()
    for q in range(12):
        for i in range(6):
            gold = (i == 0)
            d.add("q%d" % q, {"ph:a" if gold else "ph:b": 1.0, "has_digit": 1.0}, 1.0 if gold else 0.0)
    mm = LN.fit(d, 0.01)
    check(mm.w.get("ph:a", 0) > mm.w.get("ph:b", 0), "fit: the phrase that marks gold gets the larger weight")
    check(LN.rank_score(d, mm, k=1) == 1.0, "rank score: the gold window is ranked first for every question")
    from ...tools.lex_learn import fold_of
    a = fold_of(d.keys(), 5)
    check(sorted(a) == sorted(d.keys()) and max(a.values()) == 4, "folds: every question gets exactly one of k folds")


def v6_tests():
    """lex-v6 (second opinion 2): the regex parser's shapes, and the matched keyword retention."""
    from . import prompts as P
    print("\n[lex-v6 parser + matched arm]")
    o = dict(DEFAULTS)
    got, how = P.parse_regex('["revenue", "sales"]', o)
    check(got["regex"] == ["revenue", "sales"] and got["shape"] == "array", "a bare JSON array is two patterns (%s)" % got)
    got, how = P.parse_regex('{"regex": "revenue"}', o)
    check(got["regex"] == ["revenue"] and got["shape"] == "object_string",
          "a single string is ONE pattern, not seven letters (%s)" % got)
    got, how = P.parse_regex('{"regex": ["a+b", 3, null, "c"]}', o)
    check(got["regex"] == ["a+b", "c"] and got["shape"] == "object", "non-string items are dropped (%s)" % got)
    got, how = P.parse_regex('{"patterns": ["x"]}', o)
    check(got["regex"] == ["x"] and got["shape"] == "alias", "an alias key is accepted and labelled (%s)" % got)
    got, how = P.parse_regex('{"answer": 42}', o)
    check(got["regex"] == [] and got["shape"] == "wrong_type", "JSON without patterns is wrong_type, not a success")
    got, how = P.parse_regex("sure! here: [\"a\", \"b\"] hope it helps", o)
    check(got["regex"] == ["a", "b"] and how == "extract", "an array inside chatter is extracted (%s, %s)" % (got, how))
    check(o["regex_keep_keywords"] is True, "the matched arm keeps the call-1 keywords by default")


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    cfg = load_config()
    con = connect(I.kb_path(cfg), read_only=True)
    flag_tests()
    compile_tests()
    regex_tests()
    route_tests(con)
    rep, terms, hits, by_id, kw = vocab_tests(con)
    corpus = load_set("C", cfg)
    window_tests(con, rep, terms, hits, by_id, kw, corpus)
    session_tests(cfg, corpus)
    upgrade_tests(cfg, con, corpus)
    learn_tests(rep, terms, hits, by_id, kw)
    v6_tests()
    print("\n%d failure(s)" % len(FAILS))
    for f in FAILS:
        print("  -", f)
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
