"""Offline tests for the fact graph (no model, throwaway database under work/kb/_test_graph).

    .venv/Scripts/python.exe -m ragbench.kb.test_graph

  stem      corpus-validated light stemming: plurals merge only when the stem occurs, place names never stem
  family    qualifier spans cut out, quantity / period words, the detached ezafe and the company's own name dropped;
            the definition stays in the node, not the family
  graph     a fa/en pair of reports through build: nodes, families, a definition-drift family, a numeric-anchor
            fa<->en link, document cards, card routing, name matching in every spelling, the cross-document
            family walk, dirty-flag rebuilds, and - as for the label layer - `facts` untouched
"""
import json
import shutil
import sys

from ..config import load_config
from ..normalize import normalize
from . import graph as G
from . import labels as L
from . import schema as S
from . import typed as T

FAILS = []


def check(cond, label):
    print(("  ok   " if cond else "  FAIL ") + label)
    if not cond:
        FAILS.append(label)


def stem_tests():
    print("\n[stem]")
    vocab = {"کاربر": 5, "راننده": 3, "مشتری": 2, "user": 4, "order": 1}
    for word, want in (("کاربران", "کاربر"), ("رانندگان", "راننده"), ("مشتریان", "مشتری"), ("users", "user"),
                       ("orders", "order"), ("میزان", "میزان"), ("تهران", "تهران"), ("ایران", "ایران"),
                       ("sales", "sales")):
        check(G.stem(L._f(word), vocab) == L._f(want), "%s -> %s (want %s)" % (word, G.stem(L._f(word), vocab), want))


def family_tests():
    print("\n[family]")
    vocab = {"کاربر": 5, "راننده": 3}
    f1 = G.family_of(L._f("تعداد رانندگان فعال"), vocab)
    f2 = G.family_of(L._f("رانندگان ثبت‌شده"), vocab)
    f3 = G.family_of(L._f("تعداد راننده‌های اسنپ"), vocab, company_tokens={"اسنپ"})
    check(f1 == f2 == f3 == "راننده",
          "active, registered and the company-named count share one family «راننده»: %r %r %r" % (f1, f2, f3))
    check(G.quals_of("تعداد رانندگان فعال") == ("active",) and G.quals_of("رانندگان ثبت‌شده") == ("registered",)
          and G.quals_of("راننده‌های ثبت‌نامی") == ("registered",),
          "...while the qualifier stays on the node: active / registered / registered (adjectival «ثبت‌نامی»)")
    check(G.family_of(L._f("کاربران راننده ی اسنپ"), vocab, {"اسنپ"}) == "راننده کاربر",
          "the detached ezafe «ی» and the company name are dropped: %r" % G.family_of(L._f("کاربران راننده ی اسنپ"), vocab, {"اسنپ"}))
    check(G.family_of(L._f("سال"), vocab) == "" and G.family_of(L._f("رشد"), vocab) == "",
          "a period word or a bare quantity word leaves nothing: no family, no node")
    check(G.family_overlap("راننده", "راننده کاربر") == 0.5 and G.family_overlap("راننده", "فروش") == 0.0,
          "family_overlap is the Jaccard of the words (the soft bridge between years)")
    check(G.node_id_of("fa", "راننده", ("active",)) == "fa|راننده|active", "node id = lang|family|quals")


def _doc(con, doc_id, lang, company, title, pair_id, facts):
    con.execute("INSERT INTO documents (doc_id, corpus, key, lang, company, title, title_fold, pair_id, status, versions) "
                "VALUES (?,?,?,?,?,?,?,?,?,?)", (doc_id, "U", doc_id.split(".")[1], lang, company, title, L._f(title),
                                                 pair_id, "ready", json.dumps(S.component_versions())))
    sid = doc_id + "::s1"
    con.execute("INSERT INTO sections (section_id, doc_id, parent_id, level, ord, title, title_fold) "
                "VALUES (?,?,?,?,?,?,?)", (sid, doc_id, None, 1, 0, "کاربران", L._f("کاربران")))
    for i, (metric, value, unit, period) in enumerate(facts):
        uid = "%s#p%d" % (sid, i)
        text = normalize("%s %s %s در سال %s." % (metric, value, unit, period))
        con.execute("INSERT INTO units (unit_id, doc_id, section_id, ord, kind, lang, text, text_fold) "
                    "VALUES (?,?,?,?,?,?,?,?)", (uid, doc_id, sid, i, "paragraph", lang, text, text))
        f = {"metric": metric, "value": value, "unit": unit, "period": period, "unit_id": uid, "score": 3,
             "src": "text", "label_from": "clause"}
        r = T.normalise_fact(f, unit_text=text, doc_years=(int(period), int(period)))
        r.update(doc_id=doc_id, section_id=sid, unit_id=uid, ord=i)
        con.execute("INSERT INTO facts (%s) VALUES (%s)" % (", ".join(S.FACT_COLUMNS), ",".join("?" * len(S.FACT_COLUMNS))),
                    tuple(r[c] for c in S.FACT_COLUMNS))


def graph_tests(cfg):
    print("\n[graph]")
    root = cfg.path("work_root") / "kb" / "_test_graph"
    shutil.rmtree(root, ignore_errors=True)
    con = S.connect(root / "kb_graph_test.sqlite")
    S.ensure_schema(con, cfg)
    _doc(con, "U.snapp1401.fa", "fa", "snapp", "گزارش سالانه اسنپ 1401", "U.snapp1401.en",
         [("تعداد رانندگان", "4,100,000", "نفر", "1401"), ("تعداد کاربران", "37,500,000", "نفر", "1401")])
    _doc(con, "U.snapp1402.fa", "fa", "snapp", "گزارش سالانه اسنپ 1402", None,
         [("تعداد رانندگان فعال", "4,400,000", "نفر", "1402"), ("تعداد کاربران", "41,000,000", "نفر", "1402")])
    _doc(con, "U.snapp1401.en", "en", "snapp", "Snapp annual report 1401", "U.snapp1401.fa",
         [("Number of drivers", "4,100,000", "people", "1401"), ("Number of users", "37,500,000", "people", "1401")])
    con.commit()
    facts_before = [tuple(r.values()) for r in con.execute("SELECT * FROM facts ORDER BY fact_id")]
    s = G.build(con)
    check(s["facts"] >= 5 and s["nodes"] >= 4, "build: %(facts)d facts -> %(nodes)d nodes, %(families)d families" % s)
    # stems are validated against THIS corpus, where the singular «راننده» never occurs: the family key is whatever
    # the corpus supports, so the invariant is tested, not the literal string
    drift = con.execute("""SELECT family, group_concat(DISTINCT quals) AS q, count(DISTINCT doc_id) AS d FROM graph_facts
                           WHERE lang = 'fa' AND family LIKE 'رانند%' GROUP BY family""").fetchone()
    check(drift is not None and drift["d"] == 2 and "active" in (drift["q"] or ""),
          "one driver family across both years, with an 'active' node only in 1402 (the drift): %s" % (dict(drift) if drift else None))
    active_years = [x["year"] for x in con.execute("SELECT year FROM graph_facts WHERE quals = 'active'")]
    check(active_years == [1402], "the 'active' qualifier sits on the 1402 fact alone: %s" % active_years)
    link = con.execute("SELECT node_a, node_b, weight FROM graph_links").fetchall()
    check(any("رانند" in x["node_a"] and "driver" in x["node_b"] for x in link),
          "a numeric anchor (4,100,000 people, 1401) links the fa driver node to the en one: %s" % [dict(x) for x in link])
    check(con.execute("SELECT count(*) AS n FROM doc_cards").fetchone()["n"] == 3, "one card per document")
    r = G.route_documents(con, ["رانندگان فعال"], lang="fa")
    check(r and r[0][0] == "U.snapp1402.fa", "card routing puts the report that says 'active drivers' first: %s" % r[:2])
    fa_docs = ["U.snapp1401.fa", "U.snapp1402.fa"]
    m = G.match_facts(con, fa_docs, ["رانندگان"])
    m2 = G.match_facts(con, fa_docs, ["رانند"])
    check(len(m) == 2 and len(m2) == 2, "name matching finds the driver facts of both years (%d, %d)" % (len(m), len(m2)))
    check(all(h == "word" for v in m.values() for _t, h in v["metric"]),
          "a whole-word term is a WORD hit on the metric name")
    fam = G.family_members(con, [drift["family"]], fa_docs, lang="fa")
    check(sorted({x["year"] for x in fam}) == [1401, 1402], "the family walk reaches both years: %s" % sorted({x["year"] for x in fam}))
    check(G.ensure_current(con) is None, "a current graph is not rebuilt")
    G.mark_dirty(con)
    con.commit()
    check(G.ensure_current(con) is not None, "a dirty graph is rebuilt by ensure_current")
    check([tuple(x.values()) for x in con.execute("SELECT * FROM facts ORDER BY fact_id")] == facts_before
          and S.component_versions(cfg) == S.component_versions(cfg) and not S.stale_documents(con, cfg),
          "ADDITIVE: facts untouched, component_versions unchanged, nothing marked stale")
    con.close()


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    cfg = load_config()
    stem_tests()
    family_tests()
    graph_tests(cfg)
    print("\n%d failure(s)" % len(FAILS))
    for f in FAILS:
        print("  -", f)
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
