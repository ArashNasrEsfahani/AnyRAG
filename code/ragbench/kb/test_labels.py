"""Offline tests for the label layer (no model, throwaway database under work/kb/_test_labels).

    .venv/Scripts/python.exe -m ragbench.kb.test_labels

  solid     the three spellings of a Persian compound (ZWNJ, space, solid) share one match key
  tags      DEF_TAGS on the boundaries a timeline turns on: gross vs net, net vs internet, active vs inactive,
            capacity vs production, registered, consolidated; the ',a,b,' field format
  quality   every gate code, and the participles that must NOT count as verbs («ثبت‌شده», «داده‌ها», "reported")
  split     text cleaning (trailing unit words off, leading ones kept), a table cell split by offset into entity +
            metric, a year-header table (the row is the measure), the fallback when the offset is missing
  build     one document through build_doc: rows per fact, the trigram surface, `match` identical across
            spellings, idempotence, the version in meta, a no-op second build, cascade on delete, and - the point of
            the design - `facts` and `schema.component_versions` untouched, so AGENTIC3's index fingerprint holds
"""
import json
import shutil
import sys

from ..config import load_config
from ..normalize import normalize
from ..approaches.ours.lite import extract_facts
from . import labels as L
from . import schema as S
from . import typed as T

FAILS = []


def check(cond, label):
    print(("  ok   " if cond else "  FAIL ") + label)
    if not cond:
        FAILS.append(label)


def solid_tests():
    print("\n[solid]")
    keys = {L.term_forms(t)[1] for t in ("نرم‌افزار", "نرم افزار", "نرمافزار")}
    check(len(keys) == 1, "ZWNJ, space and solid spellings of «نرم‌افزار» share one solid key (%s)" % keys)
    f, s = L.term_forms("کسب‌وکار")
    check(f == "کسب وکار" and s == "کسبوکار", "term_forms gives (folded, solid): %r, %r" % (f, s))
    check(L.term_forms("کسب و کار")[0] != f and L.term_forms("کسب و کار")[1] == s,
          "«کسب‌وکار» and «کسب و کار» fold DIFFERENTLY (the ZWNJ sits before «و») but share the solid key")


def tag_tests():
    print("\n[tags]")
    cases = [("سود ناخالص", {"gross", "profit"}, {"net"}),
             ("سود خالص", {"net", "profit"}, {"gross"}),
             ("net margin", {"net"}, set()),
             ("internet users", {"users"}, {"net"}),
             ("کاربران غیرفعال", {"inactive", "users"}, {"active"}),
             ("کاربران غیر فعال", {"inactive"}, {"active"}),
             ("کاربران فعال", {"active", "users"}, {"inactive"}),
             ("رانندگان ثبت‌شده", {"registered"}, set()),
             ("ظرفیت اسمی", {"capacity", "nominal"}, {"production"}),
             ("تولید واقعی", {"production", "actual"}, {"capacity"}),
             ("درآمد تلفیقی", {"consolidated", "revenue"}, {"standalone"}),
             ("Consolidated net sales per year", {"consolidated", "net", "sales", "annual"}, set())]
    for text, want, never in cases:
        got = set(L.def_tags_of(text))
        check(want <= got and not (never & got), "%s -> %s" % (text, sorted(got)))
    check(L.tags_field(["net", "gross"]) == ",net,gross," and L.tags_field([]) == ",",
          "def_tags is ',a,b,' (so LIKE '%,net,%' is exact) and ',' when empty")


def quality_tests():
    print("\n[quality]")

    def q(label):
        mw, _ew, _how = L.split_fact(label, "text")
        return L.quality_of(mw, [w for _s, _e, w in L._words(normalize(label))])
    cases = [("www barsamtech com", "url"), ("صفحه", "junk"), ("Slide", "junk"), ("تهران", "entity"),
             ("خراسان رضوی", "entity"), ("سیستان و بلوچستان", "entity"), ("مرد", "entity"), ("آقایان", "entity"),
             ("رشد", "generic"), ("تعداد", "generic"), ("همکاران دیجی‌کالا اهل کجای ایران هستند", "prose"),
             ("چند وقت یکبار خرید می‌کنند", "prose"),
             ("تعداد کاربران فعال ماهانه در شبکه های اجتماعی کشور", "too_long"),
             ("سهم از واردات", "ok"), ("تعداد رانندگان فعال", "ok")]
    for label, want in cases:
        check(q(label) == want, "%s -> %s (want %s)" % (label, q(label), want))
    for label in ("رانندگان ثبت‌شده", "حجم داده‌ها", "reported revenue", "سود به دست آمده"):
        check(q(label) == "ok", "a participle is not a verb: %s -> %s" % (label, q(label)))
    for label in ("کاربران ثبت‌شد", "خرید نمی‌کنند", "رانندگان ثبت‌شده‌اند"):
        check(q(label) == "prose", "a joined FINITE verb is still a verb: %s -> %s" % (label, q(label)))


def split_tests():
    print("\n[split]")
    mw, ew, how = L.split_fact("درآمد میلیارد تومان", "text")
    check(mw == ["درآمد"] and how == "cleaned", "trailing unit and scale words go: %s" % mw)
    mw, _ew, _how = L.split_fact("ساعت کاری", "text")
    check(mw == ["ساعت", "کاری"], "a leading unit word stays («ساعت کاری» is a metric): %s" % mw)
    mw, _ew, _how = L.split_fact("از تعداد رانندگان", "text")
    check(mw == ["تعداد", "رانندگان"], "edge stopwords go: %s" % mw)

    table = "کشور | سهم از واردات (%)\nایران | 31.4\nچین | 12.4"
    facts, _st = extract_facts([("d::t#1", table)], "", None)
    f = next(x for x in facts if x["value"] == "31.4")
    row = T.normalise_fact(f, unit_text=table)
    mw, ew, how = L.split_fact(row["label"], "table", table, row["char_start"])
    check(how == "table" and " ".join(mw) == "سهم از واردات" and " ".join(ew) == "ایران",
          "a table cell splits by offset: %r = entity %r + metric %r" % (row["label"], " ".join(ew), " ".join(mw)))

    table2 = "شاخص | 1400 | 1401\nدرآمد | 500 | 620"
    facts2, _st = extract_facts([("d::t#2", table2)], "", None)
    f2 = next(x for x in facts2 if x["value"] == "620")
    row2 = T.normalise_fact(f2, unit_text=table2)
    mw, ew, how = L.split_fact(row2["label"], "table", table2, row2["char_start"])
    check(how == "table_row" and mw == ["درآمد"] and not ew,
          "under a year header the row is the measure: %s %s" % (how, mw))
    mw, ew, how = L.split_fact("ایران سهم از واردات", "table", "", None)
    check(how == "table_fallback" and not ew, "no offset -> cleaned like text, no entity guessed (%s)" % how)


def _seed(con, doc_id="U.labels.fa"):
    con.execute("INSERT INTO documents (doc_id, corpus, key, lang, title, status, versions) VALUES (?,?,?,?,?,?,?)",
                (doc_id, "U", "labels", "fa", "گزارش نمونه", "ready", json.dumps(S.component_versions())))
    con.execute("INSERT INTO sections (section_id, doc_id, parent_id, level, ord, title, title_fold) "
                "VALUES (?,?,?,?,?,?,?)", (doc_id + "::s1", doc_id, None, 1, 0, "نرم‌افزار و کسب‌وکار",
                                           "نرم افزار و کسب و کار"))
    table = "کشور | سهم از واردات (%)\nایران | 31.4\nچین | 12.4"
    text = normalize("درآمد نرم‌افزار در سال 1401 برابر 500 میلیارد تومان بود.")
    units = [(doc_id + "::s1#t1", "table", table), (doc_id + "::s1#p1", "paragraph", text)]
    for i, (uid, kind, t) in enumerate(units):
        con.execute("INSERT INTO units (unit_id, doc_id, section_id, ord, kind, lang, text, text_fold) "
                    "VALUES (?,?,?,?,?,?,?,?)", (uid, doc_id, doc_id + "::s1", i, kind, "fa", t, t))
    facts, _st = extract_facts([(u, t) for u, _k, t in units], "نرم‌افزار", None)
    for n, f in enumerate(facts):
        uid = f["unit_id"]
        r = T.normalise_fact(f, unit_text=dict((u, t) for u, _k, t in units)[uid], doc_years=(1401, 1401))
        r.update(doc_id=doc_id, section_id=doc_id + "::s1", unit_id=uid, ord=n)
        con.execute("INSERT INTO facts (%s) VALUES (%s)" % (", ".join(S.FACT_COLUMNS), ",".join("?" * len(S.FACT_COLUMNS))),
                    tuple(r[c] for c in S.FACT_COLUMNS))
    con.commit()
    return doc_id, len(facts)


def build_tests(cfg):
    print("\n[build]")
    root = cfg.path("work_root") / "kb" / "_test_labels"
    shutil.rmtree(root, ignore_errors=True)
    con = S.connect(root / "kb_labels_test.sqlite")
    S.ensure_schema(con, cfg)
    doc_id, n_facts = _seed(con)
    facts_before = [tuple(r.values()) for r in con.execute("SELECT * FROM facts ORDER BY fact_id")]
    versions_before = S.component_versions(cfg)

    r = L.build_doc(con, doc_id)
    n_rows = con.execute("SELECT count(*) AS n FROM fact_labels WHERE doc_id = ?", (doc_id,)).fetchone()["n"]
    check(n_facts >= 3 and n_rows == n_facts, "one fact_labels row per fact (%d facts, %d rows)" % (n_facts, n_rows))
    t = con.execute("SELECT metric_name, entity_name, split FROM fact_labels WHERE entity_name IS NOT NULL").fetchone()
    check(t is not None and t["metric_name"] == "سهم از واردات" and t["entity_name"] in ("ایران", "چین")
          and t["split"] == "table", "the table facts carry metric + entity: %s" % (dict(t) if t else None))
    kinds = {x["kind"] for x in con.execute("SELECT DISTINCT kind FROM labels_tri WHERE doc_id = ?", (doc_id,))}
    check({"metric", "entity", "section"} <= kinds, "labels_tri holds metric, entity and section rows (%s)" % kinds)

    hits = [L.match(con, s, kinds=("section",)) for s in ("نرم‌افزار", "نرم افزار", "نرمافزار")]
    check(all(len(h) == 1 for h in hits) and len({h[0]["ref_id"] for h in hits}) == 1,
          "the section title is found by all three spellings of «نرم‌افزار» (%s)" % [len(h) for h in hits])
    check(all(h[0]["hit"] == "word" for h in hits),
          "every spelling of the same word is a WORD hit, the solid one included (%s)" % [h[0]["hit"] for h in hits])
    sub = L.match(con, "زاروکس", kinds=("section",))
    check(sub and sub[0]["hit"] == "solid",
          "a match that starts inside a word («زاروکس» inside «افزار و کسب») is only a 'solid' hit")

    L.build_doc(con, doc_id)
    check(con.execute("SELECT count(*) AS n FROM fact_labels").fetchone()["n"] == n_rows
          and con.execute("SELECT count(*) AS n FROM labels_tri WHERE doc_id = ?", (doc_id,)).fetchone()["n"]
          == r["tri"], "build_doc is idempotent (no duplicate rows)")

    out = L.build(con)
    check(S.get_meta(con, "labels.version") == L.LABELS_VERSION, "build records labels.version in meta")
    check(L.build(con)["documents"] == 0, "a second build with nothing missing is a no-op (%s)" % out)
    S.set_meta(con, "labels.version", "kb-labels-v0")
    con.commit()
    check(L.build(con)["documents"] == 1, "a moved LABELS_VERSION rebuilds every document")

    facts_after = [tuple(x.values()) for x in con.execute("SELECT * FROM facts ORDER BY fact_id")]
    check(facts_before == facts_after, "ADDITIVE: every facts row is byte-identical after building the layer")
    check(S.component_versions(cfg) == versions_before and "labels" not in S.component_versions(cfg),
          "ADDITIVE: schema.component_versions is unchanged, so AGENTIC3's index fingerprint is unchanged")
    check(S.stale_documents(con, cfg) == [], "ADDITIVE: no document is marked stale by the layer")

    con.execute("DELETE FROM documents WHERE doc_id = ?", (doc_id,))
    con.commit()
    left = {tbl: con.execute("SELECT count(*) AS n FROM %s WHERE doc_id = ?" % tbl, (doc_id,)).fetchone()["n"]
            for tbl in ("fact_labels", "labels_tri")}
    check(all(v == 0 for v in left.values()), "deleting the document removes its fact_labels and labels_tri rows (%s)"
          % left)
    con.close()


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    cfg = load_config()
    solid_tests()
    tag_tests()
    quality_tests()
    split_tests()
    build_tests(cfg)
    print("\n%d failure(s)" % len(FAILS))
    for f in FAILS:
        print("  -", f)
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
