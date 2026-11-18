"""Corpora B and C adapter: PDF/HTML extraction outputs.

Input: corpus/extracted/<company>/<stem>/{document.json, extract_status.json, tables/<id>.json}
written by tools/extract/extract.py (schema_version 1, extractor 1.0.0):
  document.json  {schema_version, extractor, source{local_path, format, sha256}, manifest{...row},
                  page_count, pages_extracted[a,b], metadata, outline[{level,title,title_norm,page}],
                  body_font_size, document_direction,
                  pages[{page, blocks[...], tables[{id,csv,json,n_rows,n_cols}], images[...]}]}
  blocks         {id "p<page>_b<n>", type heading|paragraph|list_item|caption|header_footer|table_ref|image_ref,
                  level (headings: 1..3 from font size), text_raw, text_norm, stat_group? "p<page>_g<n>",
                  table "tables/<id>.csv" (table_ref)}
  tables/<id>.json {cells[[{text_raw,text_norm,bbox}|null]], header_row_guess, caption}
extract_status.json {complete, sha256, extractor_version, pages_requested, verdict ok|suspect|fail}

Sections: PDF outline when it has >= 3 entries (boundary = matching heading/line on
the outline page, else the page start); otherwise the heading blocks (nested by level)
when there are >= 3; otherwise one section per page. Content before the first
boundary goes to a "front" section titled with the document title. Empty leaf
sections are pruned (heading detection on infographic pages is noisy upstream).
Units: paragraph / list_item -> paragraph / list; caption -> caption; a stat-tile
group (consecutive blocks sharing stat_group) -> one fact unit; table -> table units
of at most corpora.table_rows_per_unit rows (header row repeated); header_footer and
image_ref are skipped. Unit local id = extractor block id (tables "<table id>#table.<k>",
groups "<group id>"), so ids are stable for a given extractor version.
"""
import hashlib
import json
from pathlib import Path

from ..normalize import fold
from .model import DocBuilder

ADAPTER = "extracted-2"


def out_dir_for(extracted_root, local_path):
    rel = local_path.replace("\\", "/").strip()
    company = rel.split("/")[0]
    stem = Path(rel).stem
    return Path(extracted_root) / company / stem


def read_status(doc_dir):
    p = Path(doc_dir) / "extract_status.json"
    if not p.is_file():
        return None
    try:
        with open(p, encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:  # noqa: BLE001
        return None


def is_usable(doc_dir, allow_partial=False, exclude_verdicts=("fail",)):
    st = read_status(doc_dir)
    if not st or not st.get("complete") or not (Path(doc_dir) / "document.json").is_file():
        return False, "no complete extraction"
    if st.get("pages_requested") and not allow_partial:
        return False, "partial extraction (pages %s)" % st.get("pages_requested")
    if st.get("verdict") in exclude_verdicts:
        return False, "verdict %s" % st.get("verdict")
    return True, st.get("verdict")


def fingerprint(doc_dir):
    doc_dir = Path(doc_dir)
    st = read_status(doc_dir) or {}
    h = hashlib.sha1(ADAPTER.encode())
    h.update(json.dumps([st.get("sha256"), st.get("extractor_version"), st.get("pages_requested"),
                         st.get("finished_at")]).encode())
    p = doc_dir / "document.json"
    if p.is_file():
        s = p.stat()
        h.update(("%d|%d" % (s.st_size, s.st_mtime_ns)).encode())
    return h.hexdigest()


def _table_rows(doc_dir, table_id):
    p = Path(doc_dir) / "tables" / ("%s.json" % table_id)
    if not p.is_file():
        return None, None
    with open(p, encoding="utf-8") as fh:
        t = json.load(fh)
    raw, norm = [], []
    for r in t.get("cells") or []:
        rr = [(c.get("text_raw") or c.get("text_norm") or "") if c else "" for c in r]
        nn = [(c.get("text_norm") if c.get("text_norm") is not None else c.get("text_raw") or "") if c else "" for c in r]
        if any(x.strip() for x in rr) or any(x.strip() for x in nn):
            raw.append([x.replace("\n", " ") for x in rr])
            norm.append([x.replace("\n", " ") for x in nn])
    return (raw, norm), t


def _flatten(doc, doc_dir, rows_per_unit):
    """-> list of items {page, kind, local, text_raw, block_ids, heading_level?, meta}; headings kept as kind 'heading'."""
    items = []
    for pg in doc.get("pages", []):
        pno = pg.get("page")
        blocks = pg.get("blocks") or []
        i = 0
        while i < len(blocks):
            b = blocks[i]
            t = b.get("type")
            if t in ("header_footer", "image_ref"):
                i += 1
                continue
            if b.get("stat_group"):
                gid = b["stat_group"]
                grp = []
                while i < len(blocks) and blocks[i].get("stat_group") == gid:
                    grp.append(blocks[i])
                    i += 1
                keep = [g for g in grp if (g.get("text_raw") or g.get("text_norm") or "").strip()]
                items.append({"page": pno, "kind": "fact", "local": gid,
                              "text_raw": " ".join((g.get("text_raw") or "").strip() for g in keep),
                              "text_norm": " ".join((g.get("text_norm") or "").strip() for g in keep),
                              "meta": {"page": pno, "blocks": [g["id"] for g in grp]}})
                continue
            if t == "table_ref":
                (raw, norm), tj = _table_rows(doc_dir, b["id"]) if (Path(doc_dir) / "tables" / ("%s.json" % b["id"])).is_file() else ((None, None), None)
                if raw:
                    head_r = head_n = None
                    hg = (tj or {}).get("header_row_guess")
                    if hg == 0 and len(raw) > 1:
                        head_r, raw = raw[0], raw[1:]
                        head_n, norm = norm[0], norm[1:]
                    cap = (tj or {}).get("caption") or b.get("caption")
                    for k in range(0, len(raw), rows_per_unit):
                        lr, ln = [], []
                        if cap and k == 0:
                            lr.append(cap)
                            ln.append(cap)
                        if head_r:
                            lr.append(" | ".join(head_r))
                            ln.append(" | ".join(head_n))
                        lr.extend(" | ".join(r) for r in raw[k:k + rows_per_unit])
                        ln.extend(" | ".join(r) for r in norm[k:k + rows_per_unit])
                        items.append({"page": pno, "kind": "table", "local": "%s#table.%d" % (b["id"], k // rows_per_unit),
                                      "text_raw": "\n".join(lr), "text_norm": "\n".join(ln),
                                      "meta": {"page": pno, "table": b["id"],
                                               "rows": [k, k + len(raw[k:k + rows_per_unit])]}})
                i += 1
                continue
            text = b.get("text_raw") or ""
            tnorm = b.get("text_norm")
            if t == "heading":
                items.append({"page": pno, "kind": "heading", "local": b["id"], "text_raw": text, "text_norm": tnorm,
                              "level": int(b.get("level") or 3), "meta": {"page": pno}})
            else:
                kind = {"list_item": "list", "caption": "caption"}.get(t, "paragraph")
                items.append({"page": pno, "kind": kind, "local": b["id"], "text_raw": text, "text_norm": tnorm,
                              "meta": {"page": pno, "block_type": t}})
            i += 1
    return items


def build_extracted(doc_dir, doc_id, corpus, lang, title=None, pair_id=None, alignment=None, rows_per_unit=20):
    doc_dir = Path(doc_dir)
    with open(doc_dir / "document.json", encoding="utf-8") as fh:
        doc = json.load(fh)
    man = doc.get("manifest") or {}
    title = title or man.get("title_%s" % lang) or (doc.get("metadata") or {}).get("title") or doc_dir.name
    st = read_status(doc_dir) or {}
    meta = {"adapter": ADAPTER, "fingerprint": fingerprint(doc_dir), "local_path": (doc.get("source") or {}).get("local_path"),
            "company": man.get("company"), "year_persian": man.get("year_persian"),
            "year_gregorian": man.get("year_gregorian"), "page_count": doc.get("page_count"),
            "pages_extracted": doc.get("pages_extracted"), "verdict": st.get("verdict")}
    b = DocBuilder(doc_id, corpus, lang, title, source=str(doc_dir).replace("\\", "/"), pair_id=pair_id,
                   alignment=alignment, meta=meta)
    items = _flatten(doc, doc_dir, rows_per_unit)
    headings = [it for it in items if it["kind"] == "heading"]
    outline = [o for o in (doc.get("outline") or []) if (o.get("title") or "").strip()]

    def add_content(sec, it):
        b.add_unit(sec, it["local"], it["kind"], it["text_raw"], meta=it["meta"], text_norm=it.get("text_norm"))

    if len(outline) >= 3:
        mode = "outline"
        # boundary index in items for each outline entry
        bounds = []
        used = set()
        for oi, o in enumerate(outline):
            tf = fold(o.get("title_norm") or o.get("title"))
            cand = None
            first_on_page = None
            for idx, it in enumerate(items):
                if it["page"] != o.get("page") or idx in used:
                    continue
                if first_on_page is None:
                    first_on_page = idx
                f = fold(it["text_raw"])
                if f and tf and (f == tf or (len(f) <= 2 * len(tf) + 10 and (f.startswith(tf) or tf.startswith(f)))):
                    cand = idx
                    break
            if cand is None:
                cand = first_on_page
            if cand is None:  # page without content: next item after that page
                cand = next((idx for idx, it in enumerate(items) if it["page"] > (o.get("page") or 0)), len(items))
            used.add(cand)
            bounds.append(cand)
        stack = []
        b.add_section("front", title)
        cur = "front"
        starts = {}
        for oi, idx in enumerate(bounds):
            starts.setdefault(idx, []).append(oi)
        for idx, it in enumerate(items + [None]):
            for oi in starts.get(idx, []):
                o = outline[oi]
                lvl = int(o.get("level") or 1)
                while stack and stack[-1][0] >= lvl:
                    stack.pop()
                local = "o%03d" % oi
                b.add_section(local, o.get("title") or "", parent=stack[-1][1] if stack else None,
                              meta={"page": o.get("page"), "outline_level": lvl})
                stack.append((lvl, local))
                cur = local
            if it is None:
                break
            if it["kind"] == "heading" and fold(it["text_raw"]) == fold(b._sections[cur]["title_raw"]):
                continue  # the heading that the outline entry already names
            add_content(cur, it)
    elif len(headings) >= 3:
        mode = "headings"
        b.add_section("front", title)
        cur = "front"
        stack = []
        n = 0
        for it in items:
            if it["kind"] == "heading":
                lvl = it["level"]
                while stack and stack[-1][0] >= lvl:
                    stack.pop()
                n += 1
                local = "h%03d" % n
                b.add_section(local, it["text_raw"], parent=stack[-1][1] if stack else None,
                              meta={"page": it["page"], "heading_level": lvl, "block": it["local"]})
                stack.append((lvl, local))
                cur = local
            else:
                add_content(cur, it)
    else:
        mode = "pages"
        seen = set()
        for it in items:
            local = "page%03d" % it["page"]
            if local not in seen:
                b.add_section(local, ("صفحه %d" if lang == "fa" else "Page %d") % it["page"], meta={"page": it["page"]})
                seen.add(local)
            add_content(local, it)
    dropped = b.prune_empty()
    d = b.build()
    d.meta.update({"structure": mode, "outline_entries": len(outline), "heading_blocks": len(headings),
                   "empty_sections_pruned": dropped})
    return d
