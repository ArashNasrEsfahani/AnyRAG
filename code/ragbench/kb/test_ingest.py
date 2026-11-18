"""Offline tests for ingestion (no model, throwaway database and corpus under work/kb/_test_ingest).

    .venv/Scripts/python.exe -m ragbench.kb.test_ingest

  separators   the ingestion contract of typed.py: `repair_separators` runs BEFORE `extract_facts`, so
               "2،788،550" becomes one fact of 2788550 instead of three facts of 2, 788 and 550; the repair is
               one character for one character, which is why facts.char_start / char_end still index into
               units.text as it is stored
  tables       a table unit goes through `normalize_lines`, so `lite.is_table_text` still sees a table and the
               row / header facts survive
  load         sections carry parent_id and path, units carry table_id, n_numbers and the fold columns, and the
               trigram rows point at real sections, units and facts
  idempotent   the same content runs zero stages the second time
  replace      the same doc id with new content replaces every row in one transaction, and a verified_facts row
               written against the old content stops matching
  versions     a bumped component version rebuilds only the stage it belongs to, never the extraction
  visibility   a document is 'ready' only after its last stage; it is 'loading' before that
  verdicts     an extraction verdict of fail is rejected with its reasons and loads nothing; suspect is loaded
               and marked
  passages     the embed text rebuilt from title + section path + text is exactly the one agentic2/tree.py embeds
  remove       every row of one document goes, every row of the other stays
"""
import json
import shutil
import sys
import time

from ..config import load_config
from ..corpus.model import DocBuilder
from ..normalize import fold
from ..approaches.agentic2.tree import build_passages
from ..approaches.ours.lite import extract_facts, is_table_text
from ..util import sha1_text
from . import ingest as I
from . import schema as S
from . import typed as T

FAILS = []
ROOT = "work/kb/_test_ingest"


def check(cond, label):
    print(("  ok   " if cond else "  FAIL ") + label)
    if not cond:
        FAILS.append(label)


def test_cfg(cfg):
    """A config whose corpus paths point into the throwaway tree, so nothing real is touched."""
    return cfg.copy_with({"paths": {"manifest": ROOT + "/natural/manifest.csv",
                                    "extracted_root": ROOT + "/extracted",
                                    "corpus_cache": ROOT + "/corpus"}})


def make_doc(doc_id, units, title="گزارش آزمایشی 1402"):
    """One section per unit list entry group: sections "s0" (paragraphs) and "s1" (a table), as a real report has."""
    b = DocBuilder(doc_id, doc_id.split(".", 1)[0], doc_id.rsplit(".", 1)[1], title, source="test")
    b.add_section("s0", "درآمد و سفر")
    for i, t in enumerate(units):
        b.add_unit("s0", "s0#p.%d" % i, "paragraph", t)
    b.add_section("s1", "جدول", parent="s0")
    b.add_unit("s1", "s1#table.0", "table",
               "سال | تعداد سفر | درآمد\n1401 | 2،788،550 | 4.318.701\n1402 | 3,100,000 | 5,001,222")
    return b.build()


# ---------------------------------------------------------------------------------------------------------------
# the separator contract
# ---------------------------------------------------------------------------------------------------------------

SEP_TEXT = "تعداد سفرهای انجام‌شده در سال 1402 برابر 2،788،550 سفر و درآمد 4.318.701 ریال بود."


def separator_tests(cfg, con):
    print("\nseparators (repair_separators BEFORE extract_facts)")
    raw, _ = extract_facts([("d::u", SEP_TEXT)])
    split = [f["value"] for f in raw]
    check(any(v in ("788", "550") for v in split),
          "without the repair the L0 regex splits 2،788،550 into pieces (%s)" % split[:4])
    repaired = T.repair_separators(SEP_TEXT)
    check(len(repaired) == len(SEP_TEXT), "the repair is one character for one character, so offsets do not move")
    fixed, _ = extract_facts([("d::u", repaired)])
    vals = {T.normalise_fact(f, unit_text=repaired)["value_num"] for f in fixed}
    check(2788550.0 in vals, "with the repair the fact carries 2,788,550 -> 2788550.0")
    check(4318701.0 in vals, "dot-grouped thousands 4.318.701 -> 4318701.0")

    doc = make_doc("U.septest_1402.fa", [SEP_TEXT])
    I.ingest_document(con, doc, cfg, source_path="test", source_sha256="0")
    rows = list(con.execute("SELECT value_text, value_num, char_start, char_end, unit_id FROM facts "
                            "WHERE doc_id = ? AND value_num = 2788550.0", (doc.id,)))
    check(bool(rows), "ingestion produces the 2788550 fact end to end")
    if rows:
        r = rows[0]
        text = con.execute("SELECT text FROM units WHERE unit_id = ?", (r["unit_id"],)).fetchone()["text"]
        check(r["char_start"] is not None and T.repair_separators(text)[r["char_start"]:r["char_end"]] ==
              r["value_text"], "char_start / char_end index into units.text as stored (%s..%s)"
              % (r["char_start"], r["char_end"]))
    n_split = con.execute("SELECT count(*) AS n FROM facts WHERE doc_id = ? AND value_text IN ('788', '550')",
                          (doc.id,)).fetchone()["n"]
    check(n_split == 0, "no fact of the broken pieces 788 / 550 is stored")


def table_tests(cfg, con):
    print("\ntables (normalize_lines, not normalize)")
    from ..normalize import normalize, normalize_lines
    table = "سال | تعداد سفر | درآمد\n1401 | 2،788،550 | 4.318.701\n1402 | 3,100,000 | 5,001,222"
    check(is_table_text(normalize_lines(table)), "normalize_lines keeps the newlines, so is_table_text sees a table")
    check(not is_table_text(normalize(table)), "normalize collapses them and is_table_text stops seeing one")
    rows = list(con.execute("SELECT f.label, f.value_num, f.year, f.source_kind FROM facts f "
                            "JOIN units u ON u.unit_id = f.unit_id WHERE u.kind = 'table' AND f.doc_id = ?",
                            ("U.septest_1402.fa",)))
    check(any(r["source_kind"] == "table" for r in rows), "the table unit produced table facts (%d)" % len(rows))
    check(any(r["value_num"] == 2788550.0 and r["year"] == 1401 for r in rows),
          "a table cell keeps its row year: 1401 -> 2788550")


# ---------------------------------------------------------------------------------------------------------------
# load, idempotency, replace, versions, visibility
# ---------------------------------------------------------------------------------------------------------------

def load_tests(cfg, con):
    print("\nload")
    doc = make_doc("U.loadtest_1402.fa", ["درآمد شرکت در سال 1402 برابر 12.5 میلیارد تومان بود.",
                                          "تعداد رانندگان فعال 61,131 نفر است."])
    res = I.ingest_document(con, doc, cfg, source_path="test", source_sha256="abc")
    check(res["action"] == "loaded", "a new document is loaded")
    check(set(res["stages"]) == {"load", "facts", "labels", "passages", "ready"},
          "all four stages ran, plus the label layer derived from the facts: %s" % res["stages"])
    n_labels = con.execute("SELECT count(*) AS n FROM fact_labels WHERE doc_id = ?", (doc.id,)).fetchone()["n"]
    n_facts = con.execute("SELECT count(*) AS n FROM facts WHERE doc_id = ?", (doc.id,)).fetchone()["n"]
    check(n_facts and n_labels == n_facts,
          "a newly ingested document has its label layer in the same pass (%d facts, %d label rows)" % (n_facts, n_labels))
    d = con.execute("SELECT * FROM documents WHERE doc_id = ?", (doc.id,)).fetchone()
    check(d["status"] == "ready", "status is ready")
    check(d["company"] == "loadtest" and d["year_from"] == 1402,
          "company and years come from the id, as docmeta derives them (%s, %s)" % (d["company"], d["year_from"]))
    check(d["content_sha1"] == sha1_text(doc.text), "content_sha1 is the sha1 of Document.text")
    check(d["title_fold"] == fold(d["title"]), "title_fold is the folded title")
    check(json.loads(d["versions"]) == S.component_versions(cfg), "the document carries the current versions")
    sec = con.execute("SELECT * FROM sections WHERE section_id = ?", (doc.id + "::s1",)).fetchone()
    check(sec["parent_id"] == doc.id + "::s0", "sections carry parent_id")
    check(sec["path"] == "درآمد و سفر > جدول" and sec["path_fold"] == fold(sec["path"]),
          "sections carry the title path and its fold (%s)" % sec["path"])
    u = con.execute("SELECT * FROM units WHERE unit_id = ?", (doc.id + "::s1#table.0",)).fetchone()
    check(u["table_id"] == "s1", "units carry the table id read off their own local id (%s)" % u["table_id"])
    check(u["n_numbers"] >= 6, "units carry a number count (%s)" % u["n_numbers"])
    check(u["text_fold"] == fold(u["text"]), "units carry the folded text")
    check(con.execute("SELECT count(*) AS n FROM tri WHERE doc_id = ? AND kind = 'unit'",
                      (doc.id,)).fetchone()["n"] == len(doc.units), "one trigram row per unit")
    orphans = con.execute("SELECT count(*) AS n FROM tri WHERE doc_id = ? AND kind = 'fact_label' "
                          "AND CAST(ref_id AS INTEGER) NOT IN (SELECT fact_id FROM facts)", (doc.id,)).fetchone()
    check(orphans["n"] == 0, "every fact_label trigram row points at a real fact_id")
    hit = con.execute("SELECT ref_id FROM tri WHERE tri MATCH ? AND kind = 'unit' AND doc_id = ?",
                      ('"رانند"', doc.id)).fetchall()
    check(bool(hit), "the trigram index finds a Persian substring inside a unit")
    return doc


def idempotency_tests(cfg, con, doc):
    print("\nidempotency and replacement")
    before = I.document_counts(con, doc.id)
    res = I.ingest_document(con, doc, cfg, source_path="test", source_sha256="abc")
    check(res["action"] == "skipped" and not res["stages"], "the same content runs zero stages")
    check(I.document_counts(con, doc.id) == before, "and changes no row")

    con.execute("INSERT INTO verified_facts (doc_id, doc_content_sha1, unit_id, value_text, verified_at) "
                "VALUES (?, ?, ?, '61,131', datetime('now'))",
                (doc.id, sha1_text(doc.text), doc.id + "::s0#p.1"))
    con.commit()
    matched = con.execute("SELECT count(*) AS n FROM verified_facts v JOIN documents d ON d.doc_id = v.doc_id "
                          "AND d.content_sha1 = v.doc_content_sha1").fetchone()["n"]
    check(matched == 1, "a verified fact matches its own document content")

    doc2 = make_doc(doc.id, ["درآمد شرکت در سال 1402 برابر 19.9 میلیارد تومان بود.",
                             "تعداد رانندگان فعال 70,000 نفر است.", "یک بند تازه با 5 عدد."])
    res = I.ingest_document(con, doc2, cfg, source_path="test", source_sha256="def")
    check(res["action"] == "loaded", "new content under the same doc id is loaded again")
    check(con.execute("SELECT count(*) AS n FROM documents WHERE doc_id = ?", (doc.id,)).fetchone()["n"] == 1,
          "still exactly one documents row")
    check(con.execute("SELECT count(*) AS n FROM units WHERE doc_id = ? AND text LIKE '%61,131%'",
                      (doc.id,)).fetchone()["n"] == 0, "the old units are gone")
    check(con.execute("SELECT count(*) AS n FROM facts WHERE doc_id = ? AND value_text = '70,000'",
                      (doc.id,)).fetchone()["n"] >= 1, "the new facts are there")
    matched = con.execute("SELECT count(*) AS n FROM verified_facts v JOIN documents d ON d.doc_id = v.doc_id "
                          "AND d.content_sha1 = v.doc_content_sha1").fetchone()["n"]
    check(matched == 0, "the verified fact written against the old content stops matching")
    return doc2


def version_tests(cfg, con, doc):
    print("\nversions: only the affected stage is rebuilt")
    cur = S.component_versions(cfg)
    con.execute("UPDATE documents SET versions = ? WHERE doc_id = ?",
                (json.dumps(dict(cur, factx="older-factx"), sort_keys=True), doc.id))
    con.commit()
    check(S.stale_stages(dict(cur, factx="older-factx"), current=cur) == ("facts",),
          "a factx bump is stale in `facts` only")
    mark = con.execute("SELECT coalesce(max(id), 0) AS m FROM ingest_log").fetchone()["m"]
    res = I.ingest_document(con, doc, cfg, source_path="test", source_sha256="def")
    ran = [r["stage"] for r in con.execute("SELECT stage FROM ingest_log WHERE id > ? AND doc_id = ?",
                                           (mark, doc.id))]
    check(res["action"] == "rebuilt", "the document is rebuilt, not reloaded")
    check(ran == ["facts", "labels", "ready"],
          "only facts (with the label layer derived from them) and ready ran: %s" % ran)
    check(json.loads(con.execute("SELECT versions FROM documents WHERE doc_id = ?",
                                 (doc.id,)).fetchone()["versions"]) == cur, "the versions are current again")

    con.execute("UPDATE documents SET versions = ? WHERE doc_id = ?",
                (json.dumps(dict(cur, tree="older-tree"), sort_keys=True), doc.id))
    con.commit()
    mark = con.execute("SELECT coalesce(max(id), 0) AS m FROM ingest_log").fetchone()["m"]
    I.ingest_document(con, doc, cfg, source_path="test", source_sha256="def")
    ran = [r["stage"] for r in con.execute("SELECT stage FROM ingest_log WHERE id > ? AND doc_id = ?",
                                           (mark, doc.id))]
    check(ran == ["passages", "ready"], "a tree bump rebuilds passages and embed only: %s" % ran)


def visibility_tests(cfg, con):
    print("\nvisibility")
    doc = make_doc("U.vistest_1402.fa", ["یک بند با عدد 12,345 تومان."])
    versions = S.component_versions(cfg)
    I.load_document(con, doc, cfg, source_path="t", source_sha256="0", quality="ok", versions=versions,
                    content_sha1=sha1_text(doc.text))
    st = con.execute("SELECT status FROM documents WHERE doc_id = ?", (doc.id,)).fetchone()["status"]
    check(st == "loading", "between load and ready the document is 'loading', so queries cannot see it")
    visible = con.execute("SELECT count(*) AS n FROM documents WHERE doc_id = ? AND status IN "
                          "('ready', 'vector_ready')", (doc.id,)).fetchone()["n"]
    check(visible == 0, "the visibility filter excludes it")
    I.ingest_document(con, doc, cfg, source_path="t", source_sha256="0")
    st = con.execute("SELECT status FROM documents WHERE doc_id = ?", (doc.id,)).fetchone()["status"]
    check(st == "ready", "after the ready stage it is visible")
    check(con.execute("SELECT count(*) AS n FROM passages WHERE doc_id = ? AND vec IS NULL",
                      (doc.id,)).fetchone()["n"] > 0, "its passages have no vector yet (embedding is stage 8)")


# ---------------------------------------------------------------------------------------------------------------
# the full add path, with a planted extraction
# ---------------------------------------------------------------------------------------------------------------

def _plant(cfg, company, stem, verdict, blocks, reasons=()):
    """A complete extraction output and its source file and manifest row, so `add` runs without a real PDF."""
    natural = cfg.path("manifest").parent / company
    natural.mkdir(parents=True, exist_ok=True)
    src = natural / ("%s.pdf" % stem)
    src.write_bytes(("%%PDF-1.4 %s" % stem).encode())
    man = cfg.path("manifest")
    if not man.is_file():
        man.write_text('"lane","company","title_fa","title_en","year_persian","language","format","local_path",'
                       '"sha256","notes"\n', encoding="utf-8-sig", newline="")
    with open(man, "a", encoding="utf-8-sig", newline="") as fh:
        fh.write('"test","%s","%s","","1402","fa","pdf","%s/%s.pdf","",""\n' % (company, stem, company, stem))
    out = cfg.path("extracted_root") / company / stem
    (out / "tables").mkdir(parents=True, exist_ok=True)
    doc = {"page_count": 1, "pages_extracted": [1, 1], "metadata": {"title": stem}, "outline": [],
           "manifest": {"company": company, "title_fa": stem, "year_persian": "1402"},
           "pages": [{"page": 1, "blocks": blocks, "tables": [], "images": []}]}
    (out / "document.json").write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
    (out / "quality.json").write_text(json.dumps({"document": {"verdict": verdict, "reasons": list(reasons)}},
                                                 ensure_ascii=False), encoding="utf-8")
    (out / "extract_status.json").write_text(
        json.dumps({"complete": True, "sha256": "0" * 64, "extractor_version": "1.1.0", "pages_requested": None,
                    "page_count": 1, "verdict": verdict}), encoding="utf-8")
    return src


def add_tests(cfg, con):
    print("\nadd: register -> extract -> model -> load -> facts -> passages -> ready")
    blocks = [{"id": "p1_b1", "type": "paragraph", "text_raw": "درآمد کل 1,234,567 ریال در سال 1402.",
               "text_norm": "درآمد کل 1,234,567 ریال در سال 1402."},
              {"id": "p1_b2", "type": "paragraph", "text_raw": "تعداد کاربران فعال 2،788،550 نفر.",
               "text_norm": "تعداد کاربران فعال 2،788،550 نفر."}]
    src = _plant(cfg, "acme", "acme_annual-report_1402_fa", "ok", blocks)
    t0 = time.perf_counter()
    res = I.add_file(con, src, cfg)
    ms = (time.perf_counter() - t0) * 1000.0
    check(res["doc_id"] == "U.acme_annual-report_1402.fa",
          "the doc id comes from the file name pattern: %s" % res["doc_id"])
    check(res["action"] == "loaded" and res["register"]["company"] == "acme", "the file is ingested")
    check(set(res["stages"]) >= {"register", "extract", "model", "load", "facts", "passages", "ready"},
          "all eight stages up to ready are timed: %s" % sorted(res["stages"]))
    check(not res["extract"]["ran"], "an extraction that is already complete on disk is reused")
    check(con.execute("SELECT count(*) AS n FROM facts WHERE doc_id = ? AND value_num = 2788550.0",
                      (res["doc_id"],)).fetchone()["n"] >= 1, "the Persian-comma thousands survive the add path")
    check(ms < 5000, "the whole add takes under 5 s without extraction (%.0f ms)" % ms)
    again = I.add_file(con, src, cfg)
    check(again["action"] == "skipped", "adding the same file again runs zero stages")

    src = _plant(cfg, "acme", "acme_broken-report_1402_fa", "fail", blocks, reasons=["no text layer", "image-only"])
    res = I.add_file(con, src, cfg)
    check(res["action"] == "rejected" and "no text layer" in res["reasons"],
          "verdict fail is rejected with its reasons")
    row = con.execute("SELECT status, quality FROM documents WHERE doc_id = ?", (res["doc_id"],)).fetchone()
    check(row["status"] == "rejected" and row["quality"] == "fail", "it is recorded as rejected, not ready")
    check(con.execute("SELECT count(*) AS n FROM units WHERE doc_id = ?", (res["doc_id"],)).fetchone()["n"] == 0,
          "and nothing of it is loaded")

    src = _plant(cfg, "acme", "acme_odd-report_1402_fa", "suspect", blocks)
    res = I.add_file(con, src, cfg)
    row = con.execute("SELECT status, quality FROM documents WHERE doc_id = ?", (res["doc_id"],)).fetchone()
    check(row["status"] == "ready" and row["quality"] == "suspect", "verdict suspect is loaded and marked")
    check((cfg.path("manifest").parent / I.USER_MANIFEST).is_file() is False,
          "a file already in the manifest needs no user manifest row")
    return res["doc_id"]


def passage_tests(cfg, con):
    print("\npassages and the embed text")
    doc = make_doc("U.embedtest_1402.fa", ["درآمد شرکت در سال 1402 برابر 12.5 میلیارد تومان بود." * 6,
                                           "تعداد رانندگان فعال 61,131 نفر است."])
    I.ingest_document(con, doc, cfg, source_path="t", source_sha256="0")
    built = build_passages(doc, I.passage_opts(cfg))
    rows = list(con.execute("SELECT p.*, d.title, s.path FROM passages p JOIN documents d ON d.doc_id = p.doc_id "
                            "LEFT JOIN sections s ON s.section_id = p.section_id WHERE p.doc_id = ? ORDER BY p.ord",
                            (doc.id,)))
    check(len(rows) == len(built), "one passages row per tree leaf (%d)" % len(rows))
    same_text = all(r["text"] == p.text for r, p in zip(rows, built))
    check(same_text, "the stored text is the passage text verbatim")
    rebuilt = [I.embed_text_of(r["title"], r["path"], r["text"]) for r in rows]
    check(all(t == p.embed_text for t, p in zip(rebuilt, built)),
          "the embed text rebuilt from title + section path + text is agentic2/tree.py's own")
    check(all(sha1_text(t) == r["embed_sha1"] for t, r in zip(rebuilt, rows)), "and matches passages.embed_sha1")
    check(all(json.loads(r["unit_ids"]) == p.unit_ids for r, p in zip(rows, built)), "unit_ids round-trip as JSON")
    check(all(r["vec"] is None for r in rows), "no vector is computed at ingestion time")


def remove_tests(cfg, con, keep_id):
    print("\nremove")
    doc_id = "U.embedtest_1402.fa"
    keep_before = I.document_counts(con, keep_id)
    res = I.remove_document(con, doc_id)
    check(all(v > 0 for v in res["before"].values()), "the document had rows in every table: %s" % res["before"])
    check(all(v == 0 for v in res["after"].values()), "every one of them is gone: %s" % res["after"])
    check(I.document_counts(con, keep_id) == keep_before, "the other document is untouched")
    check(con.execute("SELECT count(*) AS n FROM ingest_log WHERE doc_id = ?", (doc_id,)).fetchone()["n"] > 0,
          "its ingest_log rows are kept on purpose (they are the measurement record)")


def status_tests(cfg, con):
    print("\nstatus and the cost report")
    st = I.status(con, cfg)
    check(st["counts"]["documents"] > 0 and st["counts"]["facts"] > 0, "status counts every table")
    check(set(st["stages"]) >= {"load", "facts", "passages", "ready"}, "status reports per-stage timings")
    md = I.cost_markdown(con, cfg)
    check("Time from a new file on disk to the first answerable state" in md, "the cost report has the headline")
    check("| stage |" in md, "and the per-stage table")


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    base = load_config()
    root = base.project_root / ROOT
    shutil.rmtree(root, ignore_errors=True)
    root.mkdir(parents=True, exist_ok=True)
    (root / "natural").mkdir(parents=True, exist_ok=True)
    cfg = test_cfg(base)
    con = I.open_kb(cfg, path=root / "kb_test.sqlite")
    try:
        separator_tests(cfg, con)
        table_tests(cfg, con)
        doc = load_tests(cfg, con)
        doc2 = idempotency_tests(cfg, con, doc)
        version_tests(cfg, con, doc2)
        visibility_tests(cfg, con)
        keep_id = add_tests(cfg, con)
        passage_tests(cfg, con)
        remove_tests(cfg, con, keep_id)
        status_tests(cfg, con)
    finally:
        con.close()
    print("\n%d failure(s)" % len(FAILS))
    for f in FAILS:
        print("  -", f)
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
