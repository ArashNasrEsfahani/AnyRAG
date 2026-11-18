"""Offline tests for the knowledge base (no model, throwaway database under work/kb/_test).

    .venv/Scripts/python.exe -m ragbench.kb.test_kb

  numbers      every format of the docstring of approaches/agentic/numbers.py in BOTH languages: grouped thousands,
               decimals, percent before and after, the Persian decimal slash, scale words fa and en, attached
               English scales, currency symbols, ranges, and the right-to-left extraction glitch (group-reversed
               thousands); plus the two repairs the L0 regex cannot do alone, the Persian-comma thousands
               "2،788،550" and dot-grouped thousands "4.318.701"
  units        unit_norm / unit_kind / currency / value_cmp over the eleven kinds, in both languages
  periods      year, range, date, half, quarter, month, short year, calendar (jalali / gregorian), year_source
               own vs doc
  def_class    reconcile.DEF_CLASSES on real Persian labels
  L0 -> typed  extract_facts over Persian and English paragraphs and a table, normalised into facts rows; the
               ingestion contract that repair_separators runs BEFORE extract_facts
  schema       creation, a second ensure_schema as a no-op, REGEXP (including a pattern that does not compile),
               read-only connections, per-document version staleness, foreign-key cascade delete, and a trigram
               substring match on Persian text
"""
import json
import shutil
import sqlite3
import sys

from ..config import load_config
from ..normalize import normalize, normalize_lines
from ..approaches.ours.lite import extract_facts
from . import schema as S
from . import typed as T

FAILS = []


def check(cond, label):
    print(("  ok   " if cond else "  FAIL ") + label)
    if not cond:
        FAILS.append(label)


def fact(metric, value, unit="", period="", **kw):
    f = {"metric": metric, "value": value, "unit": unit, "period": period, "unit_id": "d::u#1", "score": 3,
         "src": "text", "label_from": "clause"}
    f.update(kw)
    return f


def near(a, b):
    return a is not None and abs(float(a) - float(b)) <= 1e-6 * max(1.0, abs(float(b)))


# ---------------------------------------------------------------------------------------------------------------
# numbers: every format of numbers.py, both languages
# ---------------------------------------------------------------------------------------------------------------

# (value, unit, expected value_num, expected value_hi, expected is_percent, label)
NUMBER_CASES = [
    ("6,152,509", "نفر", 6152509.0, None, 0, "fa grouped thousands 6,152,509"),
    ("1,088", "", 1088.0, None, 0, "en grouped thousands 1,088"),
    ("97.4", "", 97.4, None, 0, "decimal 97.4"),
    ("%97.4", "", 97.4, None, 1, "percent BEFORE the number %97.4 (right-to-left extraction)"),
    ("97.4%", "", 97.4, None, 1, "percent AFTER the number 97.4%"),
    ("4/9", "", 4.9, None, 0, "Persian decimal slash 4/9"),
    ("47/8", "درصد", 47.8, None, 1, "Persian decimal slash + درصد 47/8"),
    ("2میلیون", "", 2e6, None, 0, "fa attached scale 2میلیون"),
    ("4.4 میلیون", "تومان", 4.4e6, None, 0, "fa scale word 4.4 میلیون تومان"),
    ("4.4", "میلیون تومان", 4.4e6, None, 0, "fa scale word in the unit field"),
    ("38.9B", "$", 38.9e9, None, 0, "en attached scale $38.9B"),
    ("38.9", "$B", 38.9e9, None, 0, "en attached scale in the unit field"),
    ("4.4M", "", 4.4e6, None, 0, "en attached scale 4.4M"),
    ("38.9", "billion", 38.9e9, None, 0, "en scale word 38.9 billion"),
    ("700", "هزار تن", 700e3, None, 0, "fa scale word 700 هزار تن"),
    ("700", "thousand tonnes", 700e3, None, 0, "en scale word 700 thousand tonnes"),
    ("150 تا 550", "هزار تن", 150e3, 550e3, 0, "fa range 150 تا 550"),
    ("22 to 40", "%", 22.0, 40.0, 1, "en range 22 to 40"),
    ("979„379„4", "", 4379979.0, None, 0, "glitch token 979„379„4 (groups reversed)"),
    ("567„ 150", "", 150567.0, None, 0, "glitch token 567„ 150 (reversed reading is the main one)"),
    ("000„+18", "", 18000.0, None, 0, "glitch token 000„+18"),
    ("2،788،550", "نفر", 2788550.0, None, 0, "Persian-comma thousands 2،788،550"),
    ("4.318.701", "ریال", 4318701.0, None, 0, "dot-grouped thousands 4.318.701"),
    ("4.318", "", 4.318, None, 0, "ONE dot stays a decimal point 4.318"),
    ("12.5", "%", 12.5, None, 1, "en percent unit 12.5%"),
    ("2", "همت", 2e12, None, 0, "همت = هزار میلیارد تومان"),
]


def number_tests():
    print("\n[numbers]")
    for value, unit, num, hi, pct, label in NUMBER_CASES:
        r = T.normalise_fact(fact("برچسب", value, unit), unit_text="")
        ok = near(r["value_num"], num) and (r["value_hi"] is None if hi is None else near(r["value_hi"], hi)) \
            and r["is_percent"] == pct
        check(ok, "%s -> %s (got %s, hi=%s, pct=%s)" % (label, num, r["value_num"], r["value_hi"], r["is_percent"]))

    r = T.normalise_fact(fact("x", "567„ 150"), unit_text="")
    check(json.loads(r["alt_readings"]) == [567150.0], "alt_readings keeps the as-written glitch reading 567150")
    r = T.normalise_fact(fact("x", "4.4M"), unit_text="")
    check(json.loads(r["alt_readings"]) == [4.4], "alt_readings keeps the unscaled reading of 4.4M")
    r = T.normalise_fact(fact("x", "4.4 میلیون", "میلیون تومان"), unit_text="")
    check(near(r["value_num"], 4.4e6), "a scale word in BOTH value and unit is applied once")

    check(T.repair_separators("2،788،550 نفر") == "2,788,550 نفر", "repair_separators: Arabic-comma thousands")
    check(T.repair_separators("4.318.701") == "4,318,701", "repair_separators: dot-grouped thousands")
    check(T.repair_separators("4.318") == "4.318", "repair_separators leaves a single decimal point alone")


# ---------------------------------------------------------------------------------------------------------------
# units
# ---------------------------------------------------------------------------------------------------------------

UNIT_CASES = [
    ("نفر", "person", "people", None), ("people", "person", "people", None),
    ("میلیون تومان", "toman", "money", "toman"), ("میلیارد ریال", "rial", "money", "rial"),
    ("$B", "usd", "money", "usd"), ("million dollars", "usd", "money", "usd"),
    ("درصد", "percent", "percent", None), ("%", "percent", "percent", None),
    ("percentage points", "pp", "percent", None),
    ("هزار تن", "tonne", "mass", None), ("kt", "tonne", "mass", None), ("کیلوگرم", "kg", "mass", None),
    ("مگاوات ساعت", "kwh", "energy", None), ("kwh", "kwh", "energy", None), ("مگاوات", "mw", "energy", None),
    ("کیلومتر", "metre", "length", None), ("هکتار", "hectare", "area", None),
    ("ساعت", "hour", "time", None), ("سال", "year", "time", None),
    ("برابر", "times", "ratio", None), ("x", "times", "ratio", None),
    ("امتیاز", "point", "count", None), ("", "", "count", None), ("billion", "", "count", None),
    ("لیتر", "litre", "other", None), ("قلم", "", "other", None),
]


def unit_tests():
    print("\n[units]")
    for text, norm, kind, currency in UNIT_CASES:
        u = T.parse_unit(text)
        check(u["unit_norm"] == norm and u["unit_kind"] == kind and u["currency"] == currency,
              "unit %-18r -> %-9s %-8s %s (got %r %s %s)" % (text, norm or "''", kind, currency,
                                                             u["unit_norm"], u["unit_kind"], u["currency"]))
    check(T.parse_unit("kwh/t")["unit_norm"] == "kwh/tonne" and T.parse_unit("kwh/t")["unit_kind"] == "energy",
          "a compound unit keeps its parts and takes the kind of the first one (kwh/t -> energy)")
    r = T.normalise_fact(fact("درآمد", "12", "هزار میلیارد تومان"), unit_text="")
    check(near(r["value_num"], 12e12) and near(r["value_cmp"], 120e12) and r["unit_norm"] == "toman",
          "value_cmp: toman -> rial equivalent (x10)")
    r = T.normalise_fact(fact("درآمد", "120", "هزار میلیارد ریال"), unit_text="")
    check(near(r["value_cmp"], 120e12), "value_cmp: rial stays as it is")
    r = T.normalise_fact(fact("revenue", "38.9", "$B"), unit_text="")
    check(r["value_cmp"] is None, "value_cmp is NULL for a currency that is neither toman nor rial")
    r = T.normalise_fact(fact("سهم", "%47.8"), unit_text="")
    check(r["unit_norm"] == "percent" and r["unit_kind"] == "percent",
          "a percent sign on the value alone still types the fact as percent")


# ---------------------------------------------------------------------------------------------------------------
# periods and definition classes
# ---------------------------------------------------------------------------------------------------------------

def period_tests():
    print("\n[periods]")
    cases = [
        ("1401", "", None, 1401, None, "year", "jalali", "own"),
        ("2022", "", None, 2022, None, "year", "gregorian", "own"),
        ("1398-1402", "", None, 1398, 1402, "range", "jalali", "own"),
        ("1400/02/15", "", None, 1400, None, "date", "jalali", "own"),
        ("نیمه اول 1401", "", None, 1401, None, "half", "jalali", "own"),
        ("سه‌ماهه دوم 1400", "", None, 1400, None, "quarter", "jalali", "own"),
        ("Q3 2021", "", None, 2021, None, "quarter", "gregorian", "own"),
        ("مهر 1402", "", None, 1402, None, "month", "jalali", "own"),
        ("سال 98", "", None, 1398, None, "year", "jalali", "own"),
        ("", "در سال 1402 رشد کرد", (1401, 1403), 1402, None, "year", "jalali", "own"),
        ("", "بدون سال", (1399, 1399), 1399, None, "year", "jalali", "doc"),
        ("", "", None, None, None, None, None, None),
    ]
    for period, quote, years, y, y2, kind, cal, src in cases:
        p = T.parse_period_fields(period, quote, years)
        check(p["year"] == y and p["year_to"] == y2 and p["period_kind"] == kind and p["calendar"] == cal
              and p["year_source"] == src,
              "period %-18r quote=%-22r doc=%-12s -> %s" % (period, quote, years, (y, y2, kind, cal, src)))
    r = T.normalise_fact(fact("تعداد", "12", "نفر", ""), unit_text="در سال 1400 تعداد 12 نفر بود",
                         doc_years=(1400, 1400))
    check(r["year"] == 1400 and r["calendar"] == "jalali",
          "a year found in the fact's own quote dates the fact")


DEF_CASES = [("تعداد رانندگان فعال", "active"), ("کاربران ثبت‌نام‌شده", "registered"), ("مجموع سفرها", "total"),
             ("کارکنان تمام‌وقت", "fulltime"), ("کارکنان پاره‌وقت", "parttime"), ("رانندگان جدید", "new"),
             ("سفرهای روزانه", "daily"), ("درآمد ماهانه", "monthly"), ("حساب‌های کاربری", "accounts"),
             ("درآمد عملیاتی", "unspecified"), ("تعداد زنان", "subgroup")]


def defclass_tests():
    print("\n[def_class]")
    for label, cls in DEF_CASES:
        got = T.def_class_of(label, "")
        check(got == cls, "def_class %-24r -> %-12s (got %s)" % (label, cls, got))
    check(T.def_class_of("تعداد رانندگان", "راننده فعال در پایان سال") == "active",
          "def_class also reads the quote, not only the label")
    check(T.def_class_of("رانندگان زن فعال", "") == "active",
          "a subgroup flag does not replace the main class")


# ---------------------------------------------------------------------------------------------------------------
# L0 -> typed
# ---------------------------------------------------------------------------------------------------------------

FA_TEXT = "تعداد رانندگان فعال در پایان سال 1401 برابر 2،788،550 نفر بود."
EN_TEXT = "Active drivers reached 4.4 million in 2022."
TABLE = "سال | درآمد عملیاتی (میلیارد تومان)\n1401 | 1,200\n1402 | 1,850"


def l0_tests():
    print("\n[L0 -> typed]")
    units = [("d::u#1", T.repair_separators(normalize(FA_TEXT))), ("d::u#2", normalize(EN_TEXT)),
             ("d::u#3", normalize_lines(TABLE))]
    facts, _st = extract_facts(units[:2], "عملکرد")
    rows = [T.normalise_fact(f, unit_text=dict(units)[f["unit_id"]], doc_years=(1401, 1401)) for f in facts]
    fa = [r for r in rows if near(r["value_num"] or 0, 2788550.0)]
    check(len(fa) == 1 and fa[0]["unit_kind"] == "people" and fa[0]["year"] == 1401
          and fa[0]["year_source"] == "own" and fa[0]["def_class"] == "active",
          "Persian paragraph: 2,788,550 person, year 1401 (own), def_class active")
    check(fa and fa[0]["char_start"] is not None
          and dict(units)["d::u#1"][fa[0]["char_start"]:fa[0]["char_end"]] == fa[0]["value_text"],
          "char offsets point at the value inside the unit text")
    check(fa and "رانندگان" in fa[0]["label_fold"] and fa[0]["quote"],
          "label_fold and quote are filled")
    en = [r for r in rows if near(r["value_num"] or 0, 4.4e6)]
    check(len(en) == 1 and en[0]["year"] == 2022 and en[0]["calendar"] == "gregorian",
          "English paragraph: 4.4 million, year 2022, gregorian")

    raw, _s2 = extract_facts([("d::u#1", normalize(FA_TEXT))], "عملکرد")
    vals = {f["value"] for f in raw}
    check("2,788,550" not in vals and any(v in vals for v in ("2", "788", "550")),
          "WITHOUT the repair the L0 regex splits 2،788،550 into separate numbers (so ingestion must repair first)")

    tfacts, _st3 = extract_facts([("d::u#3", normalize_lines(TABLE))], "مالی")
    trows = [T.normalise_fact(f, unit_text=normalize_lines(TABLE), doc_years=(1402, 1402)) for f in tfacts]
    money = [r for r in trows if r["unit_kind"] == "money"]
    check(len(money) >= 2 and all(r["currency"] == "toman" and r["source_kind"] == "table" for r in money)
          and any(near(r["value_num"], 1.2e12) for r in money) and any(near(r["value_num"], 1.85e12) for r in money),
          "table row: the header unit (میلیارد تومان) scales both cells, source_kind table")
    check(any(r["year"] == 1401 and r["year_source"] == "own" for r in money),
          "table column header dates its cell (year_source own)")
    check(T.source_kind_of({"src": "text", "label_from": "prev_unit"}) == "kpi",
          "a detached number labelled from another unit is source_kind kpi")


# ---------------------------------------------------------------------------------------------------------------
# schema
# ---------------------------------------------------------------------------------------------------------------

def _seed(con, doc_id="U.demo.fa"):
    con.execute("INSERT INTO documents (doc_id, corpus, key, lang, title, status, versions) VALUES (?,?,?,?,?,?,?)",
                (doc_id, "U", "demo", "fa", "گزارش نمونه", "ready", json.dumps(S.component_versions())))
    con.execute("INSERT INTO sections (section_id, doc_id, parent_id, level, ord, title, title_fold) "
                "VALUES (?,?,?,?,?,?,?)", (doc_id + "::s1", doc_id, None, 1, 0, "رانندگان", "رانندگان"))
    con.execute("INSERT INTO units (unit_id, doc_id, section_id, ord, kind, lang, text, text_fold) "
                "VALUES (?,?,?,?,?,?,?,?)",
                (doc_id + "::s1#p1", doc_id, doc_id + "::s1", 0, "paragraph", "fa", FA_TEXT, normalize(FA_TEXT)))
    row = T.normalise_fact(fact("تعداد رانندگان فعال", "2,788,550", "نفر", "1401"),
                           unit_text=normalize(FA_TEXT), doc_years=(1401, 1401))
    row.update(doc_id=doc_id, section_id=doc_id + "::s1", unit_id=doc_id + "::s1#p1", ord=0)
    con.execute("INSERT INTO facts (%s) VALUES (%s)" % (", ".join(S.FACT_COLUMNS), ", ".join("?" * len(S.FACT_COLUMNS))),
                tuple(row[c] for c in S.FACT_COLUMNS))
    fid = con.execute("SELECT fact_id FROM facts WHERE doc_id = ?", (doc_id,)).fetchone()["fact_id"]
    con.execute("INSERT INTO passages (passage_id, doc_id, section_id, ord, unit_ids, text) VALUES (?,?,?,?,?,?)",
                (doc_id + "::p0", doc_id, doc_id + "::s1", 0, json.dumps([doc_id + "::s1#p1"]), FA_TEXT))
    con.executemany("INSERT INTO tri (text_fold, kind, ref_id, doc_id) VALUES (?,?,?,?)",
                    [(row["label_fold"], "fact_label", str(fid), doc_id),
                     (normalize(FA_TEXT), "unit", doc_id + "::s1#p1", doc_id),
                     ("رانندگان", "section_title", doc_id + "::s1", doc_id)])
    con.commit()
    return doc_id


def _counts(con, doc_id):
    return {t: con.execute("SELECT count(*) AS n FROM %s WHERE doc_id = ?" % t, (doc_id,)).fetchone()["n"]
            for t in ("sections", "units", "facts", "passages", "tri")}


def schema_tests(cfg):
    print("\n[schema]")
    root = cfg.path("work_root") / "kb" / "_test"
    shutil.rmtree(root, ignore_errors=True)
    path = root / "kb_test.sqlite"
    con = S.connect(path)
    versions = S.ensure_schema(con, cfg)
    check(path.exists() and con.execute("PRAGMA journal_mode").fetchone()["journal_mode"] == "wal",
          "the file is created in WAL mode")
    tables = {r["name"] for r in con.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
    check({"meta", "documents", "sections", "units", "facts", "passages", "tri", "verified_facts",
           "ingest_log"} <= tables, "every table of spec section 5 exists")
    idx = {r["name"] for r in con.execute("SELECT name FROM sqlite_master WHERE type = 'index'")}
    check({"ix_facts_doc_year", "ix_facts_kind_year", "ix_facts_unit", "ix_sections_doc", "ix_sections_parent",
           "ix_units_section", "ix_docs_company", "ix_vf_lookup"} <= idx, "every index of spec section 5 exists")
    check(versions["schema"] == S.SCHEMA_VERSION and S.get_meta(con, "version.factx")
          and S.get_meta(con, "version.typed") == T.TYPED_VERSION and S.get_meta(con, "version.tree")
          and S.get_meta(con, "version.embed") == cfg.get("models.embed"),
          "meta records the schema, factx, typed, tree and embed versions")

    doc_id = _seed(con)
    before = (sorted((r["type"], r["name"], r["sql"]) for r in con.execute("SELECT type, name, sql FROM sqlite_master")),
              sorted((r["key"], r["value"]) for r in con.execute("SELECT key, value FROM meta")))
    S.ensure_schema(con, cfg)
    after = (sorted((r["type"], r["name"], r["sql"]) for r in con.execute("SELECT type, name, sql FROM sqlite_master")),
             sorted((r["key"], r["value"]) for r in con.execute("SELECT key, value FROM meta")))
    check(before == after and _counts(con, doc_id)["facts"] == 1,
          "a second ensure_schema is a no-op (same objects, same meta, rows untouched)")

    hits = con.execute("SELECT count(*) AS n FROM facts WHERE label_fold REGEXP ?", ("رانند|driver",)).fetchone()
    check(hits["n"] == 1, "REGEXP matches a Persian label")
    bad = con.execute("SELECT count(*) AS n FROM facts WHERE label_fold REGEXP ?", ("([",)).fetchone()
    check(bad["n"] == 0, "a pattern that does not compile matches nothing and raises nothing")
    tri = con.execute("SELECT count(*) AS n FROM tri WHERE tri MATCH ?", ("رانند",)).fetchone()
    check(tri["n"] >= 2, "trigram prefilter finds a Persian SUBSTRING inside the indexed text")
    joined = con.execute("SELECT f.value_num AS v FROM tri JOIN facts f ON f.fact_id = CAST(tri.ref_id AS INTEGER) "
                         "WHERE tri.kind = 'fact_label' AND tri MATCH ? AND f.label_fold REGEXP ?",
                         ("رانند", "رانند")).fetchall()
    check(len(joined) == 1 and near(joined[0]["v"], 2788550.0),
          "trigram prefilter + REGEXP together return the typed value")

    check(S.stale_stages(json.dumps(S.component_versions(cfg)), cfg) == (),
          "a document built with the current versions is not stale")
    v = dict(S.component_versions(cfg), typed="kb-typed-v0")
    check(S.stale_stages(v, cfg) == ("facts",), "a typed-version change rebuilds the facts stage only")
    v = dict(S.component_versions(cfg), tree="agentic2-tree-v0")
    check(S.stale_stages(v, cfg) == ("passages", "embed"), "a tree-version change rebuilds passages and embeddings")
    v = dict(S.component_versions(cfg), embed="other-model")
    check(S.stale_stages(v, cfg) == ("embed",), "an embedding-model change rebuilds embeddings only")
    check(S.stale_stages(None, cfg) == S.STAGES and S.stale_stages("not json", cfg) == S.STAGES,
          "an unreadable versions field marks every stage stale")
    con.execute("UPDATE documents SET versions = ? WHERE doc_id = ?",
                (json.dumps(dict(S.component_versions(cfg), typed="kb-typed-v0")), doc_id))
    con.commit()
    check(S.stale_documents(con, cfg) == [(doc_id, ("facts",))], "stale_documents reports the stage per document")

    ro = S.connect(path, read_only=True)
    check(ro.execute("SELECT count(*) AS n FROM facts").fetchone()["n"] == 1,
          "a read-only connection opens the same file (path with a space)")
    try:
        ro.execute("DELETE FROM facts")
        wrote = True
    except sqlite3.OperationalError:
        wrote = False
    check(not wrote, "a read-only connection refuses to write")
    ro.close()

    counts = _counts(con, doc_id)
    check(all(counts[t] >= 1 for t in counts), "the seeded document has sections, units, facts, passages and trigrams")
    con.execute("DELETE FROM documents WHERE doc_id = ?", (doc_id,))
    con.commit()
    gone = _counts(con, doc_id)
    check(all(n == 0 for n in gone.values()),
          "deleting the document row cascades to sections, units, facts, passages and trigram rows (%s)" % gone)
    con.close()


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    cfg = load_config()
    number_tests()
    unit_tests()
    period_tests()
    defclass_tests()
    l0_tests()
    schema_tests(cfg)
    print("\n%d failure(s)" % len(FAILS))
    for f in FAILS:
        print("  -", f)
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
