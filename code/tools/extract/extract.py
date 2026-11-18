"""Corpus extractor: company report PDFs (and one HTML page) -> document.json,
document.md, tables/, images/, quality.json.

  python extract.py --manifest manifest.csv --out corpus/extracted --files <local_path> [...]
                    [--force] [--pages START-END]

Idempotent: a file is skipped when its output folder holds extract_status.json with the
same source sha256, extractor version and page range. Every output folder is written
to a temporary sibling and renamed into place only when complete.
"""
import argparse
import csv
import hashlib
import json
import os
import shutil
import sys
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import fa_text as F  # noqa: E402

EXTRACTOR_VERSION = "1.1.0"
IN_SCOPE_EXCLUDE = ["ocr", "image-only", "no text", "scanned", "garbled", "mojibake"]


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def norm_key(p):
    return p.replace("\\", "/").strip().lower()


def read_manifest(path):
    with open(path, encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def in_scope(row):
    notes = (row.get("notes") or "").lower()
    return bool((row.get("local_path") or "").strip()) and not any(k in notes for k in IN_SCOPE_EXCLUDE)


def write_json(path, obj):
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=1)


def write_table_files(tdir, payloads):
    os.makedirs(tdir, exist_ok=True)
    for stem, p in payloads.items():
        with open(os.path.join(tdir, stem + ".csv"), "w", encoding="utf-8-sig", newline="") as fh:
            w = csv.writer(fh)
            for row in p["cells"]:
                w.writerow([(c["text_norm"] if c else "") for c in row])
        write_json(os.path.join(tdir, stem + ".json"), p)


# --------------------------------------------------------------------------
# PDF
# --------------------------------------------------------------------------

def run_pdf(src, row, lang, page_range, tmp):
    import pdf_extract
    import assemble
    import fitz
    t0 = time.time()
    fitz.TOOLS.reset_mupdf_warnings()
    doc = pdf_extract.extract_pdf(src, page_range, lang)
    mupdf_msgs = [m for m in (fitz.TOOLS.mupdf_warnings() or "").splitlines() if m.strip()]
    json_pages, md, table_payloads, q_pages, body = assemble.build_outputs(doc, row, None, doc["n_pages"], lang)
    os.makedirs(tmp, exist_ok=True)
    if doc["image_files"]:
        idir = os.path.join(tmp, "images")
        os.makedirs(idir, exist_ok=True)
        for name, data in doc["image_files"].items():
            with open(os.path.join(idir, name), "wb") as fh:
                fh.write(data)
    if table_payloads:
        write_table_files(os.path.join(tmp, "tables"), table_payloads)
    extra = []
    if doc["lamalef_fonts"]:
        extra.append("lam-alef restored on wide alef glyphs for fonts: %s (lexicon-gated: %s)" % (", ".join(doc["lamalef_fonts"]), json.dumps({k: v for k, v in doc["lamalef_scores"].items() if k in doc["lamalef_fonts"]}, ensure_ascii=False)))
    if doc.get("narrow_fonts"):
        extra.append("spurious narrow letter glyphs dropped for fonts: %s (oracle-gated against MuPDF plain-text words: %s)"
                     % (", ".join(doc["narrow_fonts"]), json.dumps({k: v for k, v in doc["narrow_scores"].items() if k in doc["narrow_fonts"]}, ensure_ascii=False)))
    if doc.get("garbage_fonts"):
        extra.append("fonts dropped as symbol garbage (no Unicode text, no replacement text; e.g. map labels drawn in Type3 fonts): %s"
                     % json.dumps(doc["garbage_fonts"], ensure_ascii=False))
    if doc.get("actualtext_fonts"):
        extra.append("fonts whose visible glyphs carry no usable code and whose text comes only from ActualText phrases "
                     "(Canva-style export; the glyphs themselves were dropped): %s" % json.dumps(doc["actualtext_fonts"], ensure_ascii=False))
    if doc.get("kashida_gids"):
        extra.append("glyphs mapped to a letter but used as justification kashida (narrow, repeated), read as tatweel: %s"
                     % json.dumps(doc["kashida_gids"], ensure_ascii=False))
    if doc.get("mark_gids"):
        extra.append("harakat dropped that only came from a letter+mark ToUnicode entry of these glyphs: %s"
                     % json.dumps(doc["mark_gids"], ensure_ascii=False))
    gs = doc.get("gap_spaces") or {}
    if gs.get("applied"):
        extra.append("word spaces rebuilt from glyph gaps: the document writes no space glyphs (%.2f%% of %d sampled "
                     "glyphs are spaces), so %d spaces were inserted where two left-to-right glyphs are more than %.2f "
                     "font sizes apart (letters of one word sit at 0.00; Arabic script never split)"
                     % (100.0 * gs["space_share"][0], gs["space_share"][1], gs["inserted"],
                        pdf_extract.GAP_SPACE_EM))
    if doc.get("bracket_coding"):
        extra.append("bracket glyph coding (S: code is the drawn shape, mirrored in right-to-left runs; L: code is "
                     "the logical character): %s" % json.dumps(doc["bracket_coding"], ensure_ascii=False))
    if doc.get("gid_recovery_report"):
        extra.append("glyphs without ToUnicode (U+FFFD), per font, and recovery from sibling subsets of the same font family: %s"
                     % json.dumps(doc["gid_recovery_report"], ensure_ascii=False))
    dq = assemble.doc_quality(q_pages, row, lang)
    if mupdf_msgs:
        uniq = list(dict.fromkeys(mupdf_msgs))
        extra.append("MuPDF messages (%d, %d distinct): %s" % (len(mupdf_msgs), len(uniq), " | ".join(uniq[:10])))
    if extra:
        dq["notes"] = extra
    outline = [{"level": lvl, "title": title, "title_norm": F.normalize(title), "page": pg} for lvl, title, pg in doc["toc"]]
    document = {
        "page_count": doc["n_pages"],
        "pages_extracted": doc["range"],
        "metadata": doc["metadata"],
        "outline": outline,
        "body_font_size": body,
        "document_direction": "ltr" if lang == "en" else "rtl",
        "pages": json_pages,
    }
    return document, md, q_pages, dq, time.time() - t0


# --------------------------------------------------------------------------
# HTML
# --------------------------------------------------------------------------

def run_html(src, row, lang, tmp):
    import html_extract
    import assemble
    t0 = time.time()
    h = html_extract.extract_html(src)
    blocks = []
    md = ["<!-- page 1 -->"]
    table_payloads = {}
    texts = []
    nt = ni = nb = 0
    for kind, p in h["items"]:
        if kind == "table":
            nt += 1
            stem = "p001_t%d" % nt
            rows = [[{"text_raw": c["text"], "text_norm": F.normalize(c["text"]), "bbox": None} for c in r] for r in p["rows"]]
            ncols = max(len(r) for r in rows)
            rows = [r + [None] * (ncols - len(r)) for r in rows]
            header = 0 if all(c.get("th") for c in p["rows"][0]) else None
            table_payloads[stem] = {"page": 1, "id": stem, "bbox": None, "n_rows": len(rows), "n_cols": ncols,
                                    "direction": "rtl" if lang != "en" else "ltr", "header_row_guess": header,
                                    "merged_or_missing_cells": 0, "caption": None,
                                    "method": "html <table> (DOM order is logical order)", "cells": rows}
            blocks.append({"id": stem, "type": "table_ref", "table": "tables/%s.csv" % stem, "bbox": None,
                           "n_rows": len(rows), "n_cols": ncols})
            md.append(assemble.table_markdown({"rows": rows}))
            texts.extend(c["text_norm"] for r in rows for c in r if c)
            continue
        if kind == "image":
            ni += 1
            blocks.append({"id": "p1_img%d" % ni, "type": "image_ref", "image": None, "src": p["src"], "alt": p["alt"], "bbox": None})
            md.append("![%s](%s)" % ((p["alt"] or "image").replace("]", ")").replace("[", "("), p["src"] or ""))
            continue
        nb += 1
        norm = F.normalize(p["text"])
        b = {"id": "p1_b%d" % nb, "type": kind}
        if kind == "heading":
            b["level"] = p["level"]
        rr = sum(1 for ch in norm if F.char_class(ch) == "R")
        ll = sum(1 for ch in norm if F.char_class(ch) == "L")
        b.update({"text_raw": p["text"], "text_norm": norm, "bbox": None, "font_size": None,
                  "bold": kind == "heading", "direction": "rtl" if rr >= ll else "ltr", "html_tag": p.get("tag")})
        blocks.append(b)
        texts.append(norm)
        if kind == "heading":
            md.append("#" * p["level"] + " " + norm)
        elif kind == "list_item":
            md.append("- " + norm)
        elif kind == "caption":
            md.append("*" + norm + "*")
        else:
            md.append(norm)
    os.makedirs(tmp, exist_ok=True)
    if table_payloads:
        write_table_files(os.path.join(tmp, "tables"), table_payloads)
    content = " ".join(texts)
    f, r, fb, rb = F.order_hits(content)
    chars = assemble.nonspace(content)
    rl = len(assemble._RE_R.findall(content))
    lt = len(assemble._RE_L.findall(content))
    raw_all = " ".join(p["text"] for k, p in h["items"] if k not in ("table", "image"))
    pf = sum(1 for ch in raw_all if F.is_presentation_form(ch))
    note = ("HTML source: one logical page; content taken from %s (site menus, sidebar and reader comments "
            "excluded; in-article table of contents skipped x%d); images referenced by URL, not downloaded"
            % (h["content_root"], h["skipped"]["toc"]))
    qp = {"page": 1, "chars": chars, "chars_with_headers_footers": chars, "header_footer_chars": 0,
          "plain_get_text_chars": None, "persian_letter_ratio": round(rl / max(rl + lt, 1), 3),
          "presentation_form_share_raw": round(pf / max(assemble.nonspace(raw_all), 1), 3),
          "fwd_word_hits": f, "rev_word_hits": r, "fwd_bigram_hits": fb, "rev_bigram_hits": rb,
          "baseline_plain_get_text": {"fwd_word_hits": f, "rev_word_hits": r, "fwd_bigram_hits": fb, "rev_bigram_hits": rb},
          "reversal_fix_applied": False, "visual_order_tokens_reversed": 0, "lines_reordered_vs_stream": 0,
          "block_word_order_fixes": 0, "lam_alef_restored": 0, "invisible_text_chars": 0, "garbage_chars_dropped": 0,
          "latin_in_persian_word": len(assemble._RE_MIXED.findall(content)),
          "empty": chars == 0, "near_empty": chars < 30, "n_tables": nt, "n_images_saved_refs": 0,
          "has_vector_chart": False, "verdict": "ok" if chars > 0 and r <= max(2, 0.1 * f) else "suspect",
          "reasons": [note], "_text": content}
    dq = assemble.doc_quality([qp], row, lang, [])
    dq["notes"] = [note]
    document = {
        "page_count": 1, "pages_extracted": [1, 1], "metadata": {"title": h["title"], "format": "html", "html_chars": h["html_chars"]},
        "outline": [{"level": b["level"], "title": b["text_raw"], "title_norm": b["text_norm"], "page": 1} for b in blocks if b.get("type") == "heading"],
        "body_font_size": None, "document_direction": "rtl" if lang != "en" else "ltr",
        "pages": [{"page": 1, "width": None, "height": None, "rotation": 0, "direction": "rtl" if lang != "en" else "ltr",
                   "blocks": blocks,
                   "tables": [{"id": k, "csv": "tables/%s.csv" % k, "json": "tables/%s.json" % k, "n_rows": v["n_rows"], "n_cols": v["n_cols"], "direction": v["direction"]} for k, v in table_payloads.items()],
                   "images": [{"id": b["id"], "file": None, "src": b["src"], "alt": b["alt"]} for b in blocks if b["type"] == "image_ref"],
                   "small_images_skipped": 0,
                   "links": [], "drawings": {"count": None, "has_vector_chart": False}}],
    }
    return document, "\n\n".join(md) + "\n", [qp], dq, time.time() - t0


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------

def extraction_options():
    """Non-default extraction options (empty for the default behaviour, so every earlier extraction still matches):
    {"gap_spaces": True} when EXTRACT_GAP_SPACES=1 (word spaces rebuilt from glyph gaps, pdf_extract.py)."""
    return {"gap_spaces": True} if os.environ.get("EXTRACT_GAP_SPACES") == "1" else {}


def parse_pages(s):
    if not s:
        return None
    if "-" in s:
        a, b = s.split("-", 1)
        return (int(a), int(b))
    return (int(s), int(s))


def process(row, args, natural_root):
    local = row["local_path"].strip()
    rel = local.replace("\\", "/")
    src = os.path.join(natural_root, rel)
    company_folder = rel.split("/")[0]
    stem = os.path.splitext(os.path.basename(rel))[0]
    out_dir = os.path.join(args.out, company_folder, stem)
    status_path = os.path.join(out_dir, "extract_status.json")
    page_range = parse_pages(args.pages)
    if not os.path.isfile(src):
        return {"path": local, "status": "error", "error": "source missing"}
    sha = sha256_file(src)
    if not args.force and os.path.isfile(status_path):
        try:
            with open(status_path, encoding="utf-8") as fh:
                st = json.load(fh)
            if st.get("sha256") == sha and st.get("extractor_version") == EXTRACTOR_VERSION \
                    and st.get("pages_requested") == (list(page_range) if page_range else None) and st.get("complete") \
                    and (st.get("options") or {}) == extraction_options():
                return {"path": local, "status": "skipped", "verdict": st.get("verdict")}
        except Exception:  # noqa: BLE001
            pass
    fmt = (row.get("format") or os.path.splitext(rel)[1].lstrip(".")).lower()
    lang = (row.get("language") or "fa").strip().lower()
    parent = os.path.join(args.out, company_folder)
    os.makedirs(parent, exist_ok=True)
    tmp = os.path.join(parent, ".%s.tmp-%d" % (stem, os.getpid()))
    # leftovers of an interrupted run never look complete (no extract_status.json in
    # the final folder); clear them before starting, but only when they are over an
    # hour old, so a concurrent run of the same file is not deleted mid-write
    for name in os.listdir(parent):
        p = os.path.join(parent, name)
        if name.startswith(".%s.tmp-" % stem) and p != tmp:
            try:
                if time.time() - os.path.getmtime(p) > 3600:
                    shutil.rmtree(p, ignore_errors=True)
            except OSError:
                pass
    if fmt == "html":
        document, md, q_pages, dq, secs = run_html(src, row, lang, tmp)
    else:
        document, md, q_pages, dq, secs = run_pdf(src, row, lang, page_range, tmp)
    manifest_sha = (row.get("sha256") or "").strip().lower()
    if manifest_sha and manifest_sha != sha:
        dq["reasons"].append("source sha256 differs from manifest sha256")
    head = {
        "schema_version": 1,
        "extractor": {"name": "rag-paper-corpus-extract", "version": EXTRACTOR_VERSION,
                      "python": sys.version.split()[0], "pymupdf": _pymupdf_version()},
        "source": {"local_path": local, "format": fmt, "sha256": sha, "size_bytes": os.path.getsize(src),
                   "manifest_sha256_match": (manifest_sha == sha) if manifest_sha else None},
        "manifest": {k: row.get(k) for k in row.keys()},
    }
    document = dict(head, **document)
    quality = dict(head, document=dq, pages=q_pages)
    quality.pop("manifest", None)
    quality["manifest"] = {k: row.get(k) for k in ("company", "title_fa", "title_en", "year_persian", "language", "format", "pages")}
    write_json(os.path.join(tmp, "document.json"), document)
    with open(os.path.join(tmp, "document.md"), "w", encoding="utf-8", newline="\n") as fh:
        fh.write(md if md.endswith("\n") else md + "\n")
    write_json(os.path.join(tmp, "quality.json"), quality)
    status = {"complete": True, "sha256": sha, "extractor_version": EXTRACTOR_VERSION, **({"options": extraction_options()} if extraction_options() else {}),
              "pages_requested": list(page_range) if page_range else None,
              "page_count": document["page_count"], "verdict": dq["verdict"], "seconds": round(secs, 1),
              "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
    write_json(os.path.join(tmp, "extract_status.json"), status)
    if os.path.exists(out_dir):
        for attempt in range(5):
            try:
                shutil.rmtree(out_dir)
                break
            except OSError:
                time.sleep(0.5)
    for attempt in range(10):
        try:
            os.replace(tmp, out_dir)
            break
        except PermissionError:
            # antivirus / indexer holding a freshly written file (WinError 32)
            if attempt == 9:
                raise
            time.sleep(1.0)
    return {"path": local, "status": "done", "verdict": dq["verdict"], "pages": document["page_count"],
            "seconds": round(secs, 1), "reasons": dq["reasons"][:3]}


def _pymupdf_version():
    try:
        import fitz
        return fitz.VersionBind
    except Exception:  # noqa: BLE001
        return None


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--files", nargs="+", required=True, help="local_path values from the manifest")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--pages", help="START-END (1-based, inclusive)")
    args = ap.parse_args(argv)
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    try:
        import fitz
        # MuPDF prints parser complaints to stdout, which would break the one-JSON-line-
        # per-file output; they are collected per file into quality.json instead
        fitz.TOOLS.mupdf_display_errors(False)
        fitz.TOOLS.mupdf_display_warnings(False)
    except Exception:  # noqa: BLE001
        pass
    natural_root = os.path.dirname(os.path.abspath(args.manifest))
    rows = read_manifest(args.manifest)
    by_path = {norm_key(r["local_path"]): r for r in rows if (r.get("local_path") or "").strip()}
    rc = 0
    for f in args.files:
        row = by_path.get(norm_key(f))
        if row is None:
            print(json.dumps({"path": f, "status": "error", "error": "not in manifest"}, ensure_ascii=False))
            rc = 2
            continue
        if not in_scope(row):
            print(json.dumps({"path": f, "status": "error", "error": "out of scope (notes flag OCR/no text)"}, ensure_ascii=False))
            rc = 2
            continue
        try:
            res = process(row, args, natural_root)
        except Exception as e:  # noqa: BLE001
            res = {"path": f, "status": "error", "error": "%s: %s" % (type(e).__name__, e), "trace": traceback.format_exc()[-1500:]}
            rc = 1
        print(json.dumps(res, ensure_ascii=False), flush=True)
    return rc


if __name__ == "__main__":
    sys.exit(main())
