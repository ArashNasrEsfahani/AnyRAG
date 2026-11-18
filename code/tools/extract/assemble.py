"""Document-level assembly: running headers/footers, block types, heading levels,
reading order with tables and images, Markdown, and the quality report."""
import math
import re
from collections import Counter, defaultdict

import fa_text as F
from pdf_extract import (CAPTION_RE, NUM_LIST_RE, is_bullet_line, join_lines, r2, rect_area,
                         x_overlap, xy_order)

MONTHS = ["فروردین", "اردیبهشت", "خرداد", "تیر", "مرداد", "شهریور", "مهر", "آبان", "آذر", "دی", "بهمن", "اسفند"]
COMPANY_FA = {
    "dataak": ["دیتاک"], "irancell": ["ایرانسل"], "mci": ["همراه اول"], "techrasa": ["تک‌رسا", "تکراسا"],
    "ilia": ["ایلیا"], "derakcloud": ["دراک"], "irvc": ["سرمایه‌گذاری خطرپذیر"], "digikala": ["دیجی‌کالا"],
    "divar": ["دیوار"], "cafe bazaar": ["بازار"], "myket": ["مایکت"], "sheypoor": ["شیپور"],
    "yektanet": ["یکتانت"], "tapsell": ["تپسل"], "najva": ["نجوا"], "metrix": ["متریکس"],
    "dezhino": ["دژینو"], "trendix": ["ترندیکس"], "triboon": ["تریبون"], "deema": ["دیما"],
    "max holding": ["مکس"], "okala": ["اکالا"], "khanoumi": ["خانومی"], "karnameh": ["کارنامه"],
    "melkradar": ["ملک‌رادار"], "achareh": ["آچاره"], "snapp": ["اسنپ"], "alibaba": ["علی‌بابا"],
    "aparat": ["آپارات"], "digipay": ["دیجی‌پی"], "jabama": ["جاباما"], "shaparak": ["شاپرک"],
    "filimo": ["فیلیمو"], "snapptrip": ["اسنپ‌تریپ"], "flytoday": ["فلای‌تودی"], "payman": ["پیمان"],
    "nobitex": ["نوبیتکس"], "tetherland": ["تترلند"], "bitestan": ["بیتستان"], "meligold": ["ملی‌گلد"],
    "tipax": ["تیپاکس"], "podro": ["پادرو"], "taline": ["طلاین"],
}
_RE_R = re.compile("[\u0600-\u06ff]")
_RE_L = re.compile("[A-Za-z\u00c0-\u024f]")
_RE_MIXED = re.compile("[\u0620-\u064a\u067e-\u06d3][\u0100-\u024f]|[\u0100-\u024f][\u0620-\u064a\u067e-\u06d3]")
_RE_NUMERIC = re.compile(r"^[\s\d.,%+\-()/:×x*]+$")
_RE_HARAKAT = re.compile("[\u064b-\u0652]")
_GLUE_WORDS = set("\u062f\u0631 \u0627\u0632 \u06a9\u0647 \u0628\u0627 \u0631\u0627 \u0628\u0631\u0627\u06cc \u062a\u0627 \u0627\u06cc\u0646 \u0627\u0633\u062a".split())
_RE_ASCII_JUNK = re.compile(r"[{}|`^~\\\[\]]")
_RE_FA_WORD = re.compile("[\u0621-\u063a\u0641-\u064a\u067e\u0686\u0698\u06a9\u06af\u06cc\u06c0\u0622-\u0626]+")


def loose(s):
    return re.sub("[\\s\u200c]", "", s)


def nonspace(s):
    return len(re.sub(r"\s", "", s or ""))


# --------------------------------------------------------------------------
# blocks -> typed blocks
# --------------------------------------------------------------------------

def finish_blocks(pages):
    for pg in pages:
        for blk in pg["blocks"]:
            lines = blk["lines"]
            raw = join_lines([ln["text_raw"] for ln in lines])
            norm = F.normalize(join_lines([ln["text_raw"] for ln in lines], norm=True))
            w = [max(ln["nchars"], 1) for ln in lines]
            tot = sum(w)
            blk["text_raw"] = raw
            blk["text_norm"] = norm
            blk["size"] = round(sorted((ln["size"] for ln in lines), key=lambda s: s)[len(lines) // 2], 2)
            blk["font_size"] = round(sum(ln["size"] * wi for ln, wi in zip(lines, w)) / tot, 2)
            blk["bold"] = sum(ln["bold_share"] * wi for ln, wi in zip(lines, w)) / tot > 0.6
            rl = sum(wi for ln, wi in zip(lines, w) if ln["rtl"])
            blk["rtl"] = rl >= tot / 2.0
            blk["bbox"] = r2(blk["bbox"])
            blk["type"] = "paragraph"


def detect_headers_footers(pages):
    n = len(pages)
    if n >= 10:
        thr = max(3, math.ceil(0.2 * n))
    else:
        thr = max(2, math.ceil(0.4 * n))
    sizes = Counter()
    for pg in pages:
        for blk in pg["blocks"]:
            for ln in blk["lines"]:
                sizes[round(ln["size"] * 2) / 2.0] += ln["nchars"]
    body = sizes.most_common(1)[0][0] if sizes else 10.0
    keys = defaultdict(set)
    bands = defaultdict(set)
    cands = []
    for pg in pages:
        W, H = pg["width"], pg["height"]
        for blk in pg["blocks"]:
            t = blk["text_norm"]
            if not t or len(t) > 400:
                continue
            # running heads are set small; a repeated slide title or chart legend in
            # display type is content
            if blk.get("font_size", body) > 1.3 * body:
                continue
            x0, y0, x1, y1 = blk["bbox"]
            zone = None
            if y1 <= 0.18 * H:
                zone = "top"
            elif y0 >= 0.82 * H:
                zone = "bottom"
            elif x1 <= 0.10 * W:
                zone = "left"
            elif x0 >= 0.90 * W:
                zone = "right"
            if zone is None:
                continue
            cy = (y0 + y1) / 2.0 / H
            cx = (x0 + x1) / 2.0 / W
            masked = re.sub(r"\d+", "#", t)
            key = (masked, zone, round(cy * 25), round(cx * 6) if zone in ("top", "bottom") else 0)
            keys[key].add(pg["pno"])
            # the same running head whose x position alternates on facing pages
            band = (masked, zone, round(cy * 25))
            bands[band].add(pg["pno"])
            cands.append((pg, blk, key, zone, band))
    for pg, blk, key, zone, band in cands:
        t = blk["text_norm"]
        if CAPTION_RE.match(t) and not (blk["bbox"][3] <= 0.06 * pg["height"] or blk["bbox"][1] >= 0.94 * pg["height"]):
            # a repeated "Source: ..." line under charts is content, not a running footer
            continue
        H, W = pg["height"], pg["width"]
        x0, y0, x1, y1 = blk["bbox"]
        rep = len(keys[key]) >= thr or len(bands[band]) >= thr
        # chapter running heads ("CSR", a chapter name, "22 فصل دوم", or a long
        # "report title | chapter" line in the top margin) repeat on a chapter's
        # pages only: text in the page margin, identical on >= 4 pages
        margin = (y1 <= 0.12 * H or y0 >= 0.88 * H) if zone in ("top", "bottom") else True
        tight = (y1 <= 0.07 * H or y0 >= 0.93 * H) if zone in ("top", "bottom") else True
        letters = len(_RE_R.findall(t)) + len(_RE_L.findall(t))
        if not rep and len(bands[band]) >= 4 and margin and letters >= 2 \
                and (len(t.split()) <= 6 or (tight and len(t) <= 160)):
            rep = True
        is_num = False
        m = re.fullmatch(r"(?:page|صفحه)?\s*(\d{1,4})(?:\s*(?:/|of|از)\s*\d{1,4})?", t, re.I)
        if m and (y1 <= 0.10 * H or y0 >= 0.90 * H or x1 <= 0.07 * W or x0 >= 0.93 * W):
            v = int(m.group(1))
            if abs(v - (pg["pno"] + 1)) <= 20:
                is_num = True
        if rep and not _RE_NUMERIC.match(t.replace("#", "")) or is_num or (rep and m):
            blk["type"] = "header_footer"
            blk["hf_zone"] = zone
            blk["hf_reason"] = "page number" if is_num else "repeated on %d pages" % max(len(keys[key]), len(bands[band]))


def body_size(pages):
    c = Counter()
    for pg in pages:
        for blk in pg["blocks"]:
            if blk["type"] == "header_footer":
                continue
            for ln in blk["lines"]:
                c[round(ln["size"] * 2) / 2.0] += ln["nchars"]
    if not c:
        return 10.0
    return c.most_common(1)[0][0]


def classify_blocks(pages, body):
    for pg in pages:
        # a page with running prose (>= 15 words at about body size) is where a short
        # bold line can be a heading; on infographic pages bold labels stay paragraphs
        prose_page = any(blk["type"] != "header_footer" and len(blk["text_norm"].split()) >= 15
                         and abs(blk["font_size"] - body) <= 0.2 * body for blk in pg["blocks"])
        for blk in pg["blocks"]:
            if blk["type"] == "header_footer":
                continue
            t = blk["text_norm"]
            if not t:
                blk["type"] = "paragraph"
                continue
            letters = len(_RE_R.findall(t)) + len(_RE_L.findall(t))
            digits = sum(ch.isdigit() for ch in t)
            words = len(t.split())
            size = blk["font_size"]
            first_raw = blk["lines"][0]["text_raw"]
            if CAPTION_RE.match(t) and words <= 40:
                blk["type"] = "caption"
                continue
            if is_bullet_line(first_raw):
                blk["type"] = "list_item"
                continue
            heading = False
            ends_like_sentence = bool(re.search(r"[.:؛،,]$", t))
            if letters >= 3 and digits * 2 <= letters and len(blk["lines"]) <= 3 and not ends_like_sentence:
                if size >= 1.25 * body and words <= 14 and (size >= 1.6 * body or blk["bold"] or prose_page):
                    heading = True
                elif blk["bold"] and size >= 1.08 * body and words <= 12 and prose_page:
                    heading = True
                elif blk["bold"] and size >= 0.95 * body and words <= 8 and prose_page and len(blk["lines"]) == 1:
                    heading = True
            if heading:
                blk["type"] = "heading"
                blk["level"] = 1 if size >= 1.9 * body else (2 if size >= 1.3 * body else 3)
                continue
            if NUM_LIST_RE.match(t) and size < 1.2 * body:
                blk["type"] = "list_item"
                continue
            blk["type"] = "paragraph"


# --------------------------------------------------------------------------
# page assembly
# --------------------------------------------------------------------------

def md_escape_cell(s):
    return (s or "").replace("|", "\\|").replace("\n", " ").strip()


def table_markdown(tb):
    rows = [[md_escape_cell(c["text_norm"]) if c else "" for c in r] for r in tb["rows"]]
    if not rows:
        return ""
    ncols = max(len(r) for r in rows)
    rows = [r + [""] * (ncols - len(r)) for r in rows]
    out = ["| " + " | ".join(rows[0]) + " |", "|" + "---|" * ncols]
    for r in rows[1:]:
        out.append("| " + " | ".join(r) + " |")
    return "\n".join(out)


def nearest_caption(bbox, blocks, used, max_dist=80.0):
    best = None
    bestd = None
    for blk in blocks:
        if blk["type"] != "caption" or id(blk) in used:
            continue
        if x_overlap(bbox, blk["bbox"]) < 0.0 - 20:
            continue
        d = min(abs(blk["bbox"][1] - bbox[3]), abs(bbox[1] - blk["bbox"][3]))
        if y_inside(blk["bbox"], bbox):
            d = 0.0
        if d <= max_dist and (bestd is None or d < bestd):
            best, bestd = blk, d
    return best


def y_inside(a, b):
    return a[1] >= b[1] and a[3] <= b[3]


def _digit_share(t):
    ns = re.sub(r"\s", "", t)
    return sum(ch.isdigit() for ch in ns) / max(len(ns), 1)


def _is_figure(t):
    return bool(re.search(r"\d", t)) and (len(t.split()) <= 3 or _digit_share(t) >= 0.3)


def stat_groups(content):
    """Infographic stat tiles: a short figure ("308 هزار", "41.5") set directly above a
    short label ("فروشنده"). XY-cut would read a row of figures and then a row of
    labels; grouping keeps each figure next to its label in the reading order."""
    cands = [b for b in content if b["text_norm"] and len(b["text_norm"].split()) <= 14
             and b["type"] not in ("caption", "list_item") and not b.get("rotated")]
    cands.sort(key=lambda b: b["bbox"][1])
    below = {}
    above_taken = set()
    for i, b in enumerate(cands):
        best = None
        for a in cands[:i]:
            if id(a) in below or a is b:
                continue
            if len(a["text_norm"].split()) > 5:
                continue
            gap = b["bbox"][1] - a["bbox"][3]
            smin = min(a["font_size"], b["font_size"])
            smax = max(a["font_size"], b["font_size"])
            if gap < -0.5 * smin or gap > min(max(2.2 * smin, 0.8 * smax), 4.0 * smin, 40.0):
                continue
            if len(b["text_norm"].split()) > 8 and not _is_figure(a["text_norm"]):
                continue
            if _is_figure(a["text_norm"]) and _is_figure(b["text_norm"]):
                continue
            wa = a["bbox"][2] - a["bbox"][0]
            wb = b["bbox"][2] - b["bbox"][0]
            ca = (a["bbox"][0] + a["bbox"][2]) / 2.0
            cb = (b["bbox"][0] + b["bbox"][2]) / 2.0
            aligned = abs(ca - cb) <= 0.35 * max(wa, wb) or abs(a["bbox"][2] - b["bbox"][2]) <= 0.1 * max(wa, wb) \
                or abs(a["bbox"][0] - b["bbox"][0]) <= 0.1 * max(wa, wb)
            if not aligned or x_overlap(a["bbox"], b["bbox"]) <= 0:
                continue
            if best is None or gap < best[0]:
                best = (gap, a)
        if best is not None and id(b) not in above_taken:
            below[id(best[1])] = b
            above_taken.add(id(b))
    heads = [b for b in cands if id(b) in below and id(b) not in above_taken]
    groups = []
    for h in heads:
        chain = [h]
        while id(chain[-1]) in below:
            chain.append(below[id(chain[-1])])
        figs = [_is_figure(b["text_norm"]) for b in chain]
        if len(chain) >= 2 and any(figs) and not all(figs):
            groups.append(chain)
    return groups


def assemble_page(pg, doc_rtl):
    pnum = pg["pno"] + 1
    W, H = pg["width"], pg["height"]
    page_area = W * H
    blocks = pg["blocks"]
    content = [b for b in blocks if b["type"] != "header_footer"]
    hf = [b for b in blocks if b["type"] == "header_footer"]
    items = []
    grouped = set()
    for gi, members in enumerate(stat_groups(content), 1):
        gid = "p%d_g%d" % (pnum, gi)
        for b in members:
            grouped.add(id(b))
            b["stat_group"] = gid
            if b["type"] == "heading":
                b["type"] = "paragraph"
                b.pop("level", None)
        items.append({"kind": "group", "bbox": [min(b["bbox"][0] for b in members), min(b["bbox"][1] for b in members),
                                                max(b["bbox"][2] for b in members), max(b["bbox"][3] for b in members)],
                      "size": min(b["font_size"] for b in members), "ref": members})
    for b in content:
        if id(b) not in grouped:
            items.append({"kind": "block", "bbox": b["bbox"], "size": b["font_size"], "ref": b})
    tables_out = []
    for i, tb in enumerate(pg["tables"], 1):
        stem = "p%03d_t%d" % (pnum, i)
        tb["id"] = stem
        items.append({"kind": "table", "bbox": tb["bbox"], "size": 10.0, "ref": tb})
        tables_out.append(tb)
    # images: background-size images go first; images overlapping text follow that block
    after_block = defaultdict(list)
    lead_images = []
    img_counter = 0
    for im in pg["images"]:
        img_counter += 1
        im["id"] = "p%03d_img%d" % (pnum, img_counter)
        if im["area_share"] >= 0.5:
            lead_images.append(im)
            continue
        best, best_ov = None, 0.0
        for b in content:
            ov = max(0.0, x_overlap(im["bbox"], b["bbox"])) * max(0.0, min(im["bbox"][3], b["bbox"][3]) - max(im["bbox"][1], b["bbox"][1]))
            if ov > best_ov:
                best, best_ov = b, ov
        if best is not None and best_ov > 0.2 * max(rect_area(best["bbox"]), 1.0):
            after_block[id(best)].append(im)
        else:
            items.append({"kind": "image", "bbox": im["bbox"], "size": 10.0, "ref": im})
    ordered = xy_order(items, pg["rtl"]) if items else []
    # captions
    used = set()
    for it in ordered:
        if it["kind"] in ("image", "table"):
            cap = nearest_caption(it["bbox"], content, used)
            if cap is not None:
                used.add(id(cap))
                it["ref"]["caption"] = cap["text_norm"]
                it["ref"]["caption_block"] = cap.get("id")
                cap["caption_of"] = it["ref"].get("id") or it["ref"].get("file")
    for lst in after_block.values():
        for im in lst:
            cap = nearest_caption(im["bbox"], content, used)
            if cap is not None:
                used.add(id(cap))
                im["caption"] = cap["text_norm"]
                cap["caption_of"] = im.get("file")
    # final sequence
    seq = []
    top_hf = [b for b in hf if b.get("hf_zone") in ("top", "left", "right")]
    bot_hf = [b for b in hf if b.get("hf_zone") == "bottom"]
    for b in sorted(top_hf, key=lambda b: (b["bbox"][1], -b["bbox"][2])):
        seq.append(("block", b))
    for im in lead_images:
        seq.append(("image", im))
    for it in ordered:
        seq.append((it["kind"], it["ref"]))
        if it["kind"] == "block":
            for im in after_block.get(id(it["ref"]), []):
                seq.append(("image", im))
        elif it["kind"] == "group":
            for b in it["ref"]:
                for im in after_block.get(id(b), []):
                    seq.append(("image", im))
    for b in sorted(bot_hf, key=lambda b: (b["bbox"][1], -b["bbox"][2])):
        seq.append(("block", b))
    return seq, tables_out


def build_outputs(doc, manifest_row, source_info, page_count, lang):
    pages = doc["pages"]
    finish_blocks(pages)
    detect_headers_footers(pages)
    body = body_size(pages)
    classify_blocks(pages, body)
    doc_rtl = lang != "en"
    json_pages = []
    md = []
    table_payloads = {}
    q_pages = []
    for pg in pages:
        pnum = pg["pno"] + 1
        bi = 0
        for blk in pg["blocks"]:
            bi += 1
            blk["id"] = "p%d_b%d" % (pnum, bi)
        seq, tables = assemble_page(pg, doc_rtl)
        jblocks = []
        md.append("<!-- page %d -->" % pnum)
        n = 0
        numeric_short = 0
        content_chars_norm = []
        hf_chars = 0
        for kind, ref in seq:
            n += 1
            if kind == "group":
                texts = []
                for blk in ref:
                    jb = {"id": blk["id"], "type": blk["type"], "text_raw": blk["text_raw"], "text_norm": blk["text_norm"],
                          "bbox": blk["bbox"], "font_size": blk["font_size"], "bold": blk["bold"],
                          "direction": "rtl" if blk["rtl"] else "ltr", "n_lines": len(blk["lines"]),
                          "stat_group": blk["stat_group"]}
                    jblocks.append(jb)
                    if blk["text_norm"]:
                        texts.append(blk["text_norm"])
                        content_chars_norm.append(blk["text_norm"])
                        if _RE_NUMERIC.match(blk["text_norm"]) and len(blk["text_norm"]) <= 12:
                            numeric_short += 1
                if texts:
                    md.append(" ".join(texts))
                continue
            if kind == "block":
                blk = ref
                jb = {"id": blk["id"], "type": blk["type"]}
                if blk["type"] == "heading":
                    jb["level"] = blk["level"]
                jb.update({"text_raw": blk["text_raw"], "text_norm": blk["text_norm"], "bbox": blk["bbox"],
                           "font_size": blk["font_size"], "bold": blk["bold"],
                           "direction": "rtl" if blk["rtl"] else "ltr", "n_lines": len(blk["lines"])})
                if blk.get("rotated"):
                    jb["rotated"] = True
                if blk.get("order_fix"):
                    jb["order_fix"] = blk["order_fix"]
                if blk.get("hf_reason"):
                    jb["hf_reason"] = blk["hf_reason"]
                if blk.get("caption_of"):
                    jb["caption_of"] = blk["caption_of"]
                jblocks.append(jb)
                t = blk["text_norm"]
                if blk["type"] == "header_footer":
                    hf_chars += nonspace(t)
                    continue
                if not t:
                    continue
                content_chars_norm.append(t)
                if _RE_NUMERIC.match(t) and len(t) <= 12:
                    numeric_short += 1
                if blk["type"] == "heading":
                    md.append("#" * blk["level"] + " " + t)
                elif blk["type"] == "list_item":
                    tt = t
                    if tt and (tt[0] in F.BULLETS or tt[0] in "-–—*"):
                        tt = tt[1:].strip()
                    md.append("- " + tt)
                elif blk["type"] == "caption":
                    if blk.get("caption_of") and str(blk["caption_of"]).startswith("images/"):
                        continue
                    md.append("*" + t + "*")
                else:
                    md.append(t)
            elif kind == "table":
                tb = ref
                jb = {"id": tb["id"], "type": "table_ref", "table": "tables/%s.csv" % tb["id"], "bbox": tb["bbox"],
                      "n_rows": tb["n_rows"], "n_cols": tb["n_cols"], "direction": tb["direction"]}
                if tb.get("caption"):
                    jb["caption"] = tb["caption"]
                jblocks.append(jb)
                md.append(table_markdown(tb))
                for r in tb["rows"]:
                    for c in r:
                        if c and c["text_norm"]:
                            content_chars_norm.append(c["text_norm"])
            elif kind == "image":
                im = ref
                jb = {"id": im["id"], "type": "image_ref", "image": im["file"], "bbox": im["bbox"]}
                if im.get("caption"):
                    jb["caption"] = im["caption"]
                jblocks.append(jb)
                if im["file"]:
                    cap = im.get("caption") or "image"
                    md.append("![%s](%s)" % (cap.replace("]", ")").replace("[", "("), im["file"]))
        # tables payloads
        jtables = []
        for tb in tables:
            payload = {"page": pnum, "id": tb["id"], "bbox": tb["bbox"], "n_rows": tb["n_rows"], "n_cols": tb["n_cols"],
                       "direction": tb["direction"], "header_row_guess": tb["header_row_guess"],
                       "merged_or_missing_cells": tb["merged_or_missing_cells"], "grid_cols_find_tables": tb.get("grid_cols_find_tables"),
                       "rows_from_text_baselines": tb.get("rows_from_text_baselines", False),
                       "caption": tb.get("caption"),
                       "method": tb["method"], "cells": tb["rows"]}
            table_payloads[tb["id"]] = payload
            jtables.append({"id": tb["id"], "csv": "tables/%s.csv" % tb["id"], "json": "tables/%s.json" % tb["id"],
                            "bbox": tb["bbox"], "n_rows": tb["n_rows"], "n_cols": tb["n_cols"], "direction": tb["direction"],
                            "header_row_guess": tb["header_row_guess"], "caption": tb.get("caption")})
        dr = dict(pg["drawings"])
        cnt = dr.get("count") or 0
        dr["has_vector_chart"] = bool(cnt >= 10 and (dr.get("filled_shapes", 0) >= 6 or dr.get("stroked_lines", 0) >= 12 or dr.get("curved_paths", 0) >= 6) and numeric_short >= 3)
        jimages = []
        for im in pg["images"]:
            jimages.append({k: im.get(k) for k in ("id", "file", "xref", "width_px", "height_px", "bbox", "area_share", "caption", "error") if im.get(k) is not None})
        json_pages.append({"page": pnum, "width": round(pg["width"], 2), "height": round(pg["height"], 2),
                           "rotation": pg["rotation"], "direction": "rtl" if pg["rtl"] else "ltr",
                           "blocks": jblocks, "tables": jtables, "images": jimages,
                           "small_images_skipped": pg["small_images_skipped"], "links": pg["links"], "drawings": dr})
        # ---- quality
        content_text = " ".join(content_chars_norm)
        all_raw = " ".join(ln["text_raw"] for blk in pg["blocks"] for ln in blk["lines"])
        raw_ns = nonspace(all_raw) + sum(nonspace(c["text_raw"]) for tb in tables for r in tb["rows"] for c in r if c)
        presform = sum(ln["presform"] for blk in pg["blocks"] for ln in blk["lines"])
        r_letters = len(_RE_R.findall(content_text))
        l_letters = len(_RE_L.findall(content_text))
        f, r, fb, rb = F.order_hits(content_text)
        plain_norm = F.normalize(pg["plain_text"])
        bf, br, bfb, brb = F.order_hits(plain_norm)
        stats = pg["stats"]
        chars = nonspace(content_text)
        plain_chars = nonspace(plain_norm)
        garbage = stats.get("garbage_chars", 0)
        mixed = len(_RE_MIXED.findall(content_text))
        fa_words = max(len(F.fa_tokens(content_text)), 1)
        reasons = []
        verdict = "ok"
        has_graphics = bool(pg["images"]) or (pg["drawings"].get("count") or 0) > 0
        if chars == 0 and hf_chars == 0 and pg.get("blank"):
            reasons.append("blank page (nothing drawn)")
        elif chars == 0 and hf_chars == 0:
            if has_graphics:
                verdict = "suspect"
                reasons.append("no text layer on a page with graphics (outlined text or image-only page)")
        elif chars == 0:
            reasons.append("only running header/footer text on this page")
        elif chars < 30:
            reasons.append("near-empty page")
        if r >= 3 and r > f:
            verdict = "fail"
            reasons.append("reversed function words outnumber forward (%d vs %d)" % (r, f))
        elif r >= 2 and r >= 0.3 * max(f, 1):
            verdict = "suspect" if verdict == "ok" else verdict
            reasons.append("many reversed function words (%d vs %d forward)" % (r, f))
        if rb >= 2 and rb > fb:
            verdict = "suspect" if verdict == "ok" else verdict
            reasons.append("swapped common bigrams (%d vs %d)" % (rb, fb))
        if garbage and garbage >= 0.05 * (garbage + chars):
            verdict = "suspect" if verdict == "ok" else verdict
            reasons.append("%d unmapped-glyph garbage chars dropped" % garbage)
        sym = len(_RE_ASCII_JUNK.findall(content_text))
        if doc_rtl and sym >= 20 and sym >= 0.03 * max(chars, 1):
            verdict = "suspect" if verdict == "ok" else verdict
            reasons.append("%d ASCII symbol characters in Persian text (undecoded font garbage)" % sym)
        unm = stats.get("unmapped_glyphs", 0)
        if unm >= 5 and unm >= 0.02 * max(chars, 1):
            verdict = "suspect" if verdict == "ok" else verdict
            reasons.append("%d glyphs without a Unicode mapping were dropped (words missing letters)" % unm)
        if mixed >= 5 and mixed >= 0.03 * fa_words:
            verdict = "suspect" if verdict == "ok" else verdict
            reasons.append("Latin-extended letters inside Persian words (%d): broken font ToUnicode" % mixed)
        extracted = chars + hf_chars
        replaced = stats.get("glyphs_replaced_by_actualtext", 0) + stats.get("garbage_font_chars_dropped", 0)
        if plain_chars >= 200 and not stats.get("glyphs_replaced_by_actualtext") and extracted < 0.85 * max(plain_chars - garbage - replaced, 1):
            verdict = "suspect" if verdict == "ok" else verdict
            reasons.append("extracted chars %d < 85%% of plain get_text chars %d" % (extracted, plain_chars))
        fix_applied = bool(stats.get("visual_tokens", 0) or stats.get("fallback_fixes", 0) or stats.get("lamalef_restored", 0))
        q_pages.append({
            "page": pnum, "chars": chars, "chars_with_headers_footers": chars + hf_chars, "header_footer_chars": hf_chars,
            "plain_get_text_chars": plain_chars,
            "persian_letter_ratio": round(r_letters / max(r_letters + l_letters, 1), 3),
            "presentation_form_share_raw": round(presform / max(raw_ns, 1), 3),
            "fwd_word_hits": f, "rev_word_hits": r, "fwd_bigram_hits": fb, "rev_bigram_hits": rb,
            "baseline_plain_get_text": {"fwd_word_hits": bf, "rev_word_hits": br, "fwd_bigram_hits": bfb, "rev_bigram_hits": brb},
            "reversal_fix_applied": fix_applied,
            "visual_order_tokens_reversed": stats.get("visual_tokens", 0),
            "lines_reordered_vs_stream": stats.get("lines_reordered", 0),
            "block_word_order_fixes": stats.get("fallback_fixes", 0),
            "lam_alef_restored": stats.get("lamalef_restored", 0), "invisible_text_chars": stats.get("invisible_chars", 0),
            "garbage_chars_dropped": garbage, "latin_in_persian_word": mixed,
            "unmapped_glyphs_dropped": stats.get("unmapped_glyphs", 0),
            "glyphs_recovered_from_sibling_font": stats.get("glyphs_recovered_from_sibling_font", 0),
            "spurious_narrow_glyphs_dropped": stats.get("spurious_narrow_glyphs_dropped", 0),
            "duplicate_glyphs_dropped": stats.get("duplicate_glyphs_dropped", 0),
            "garbage_font_chars_dropped": stats.get("garbage_font_chars_dropped", 0),
            "glyphs_replaced_by_actualtext": stats.get("glyphs_replaced_by_actualtext", 0),
            "empty": chars + hf_chars == 0, "blank": bool(pg.get("blank")) and chars + hf_chars == 0,
            "near_empty": chars < 30,
            "zwnj": content_text.count("\u200c"), "plain_zwnj": plain_norm.count("\u200c"),
            "harakat": len(_RE_HARAKAT.findall(content_text)), "plain_harakat": len(_RE_HARAKAT.findall(plain_norm)),
            "ascii_symbol_chars": sym, "brackets_mirrored": stats.get("brackets_mirrored", 0),
            "stray_marks_attached": stats.get("stray_marks_attached", 0), "stray_marks_dropped": stats.get("stray_marks_dropped", 0),
            "lam_alef_reverted_in_words": stats.get("lamalef_reverted", 0),
            "table_glyphs_left_in_text_flow": stats.get("table_glyphs_left_in_text_flow", 0),
            "actualtext_overlaps_trimmed": stats.get("actualtext_overlaps_trimmed", 0),
            "n_tables": len(tables), "n_images_saved_refs": sum(1 for im in pg["images"] if im.get("file")),
            "has_vector_chart": dr["has_vector_chart"],
            "verdict": verdict, "reasons": reasons,
            "_text": content_text,
        })
        md.append("")
    return json_pages, "\n\n".join(x for x in md if x is not None).replace("\n\n\n", "\n\n"), table_payloads, q_pages, body


def _lexicon_words():
    try:
        from pdf_extract import _lexicon
        return _lexicon()
    except Exception:  # noqa: BLE001
        return {}


def doc_quality(q_pages, manifest_row, lang, extra_reasons=None):
    n = len(q_pages)
    text = " ".join(p["_text"] for p in q_pages)
    for p in q_pages:
        p.pop("_text", None)
    f = sum(p["fwd_word_hits"] for p in q_pages)
    r = sum(p["rev_word_hits"] for p in q_pages)
    fb = sum(p["fwd_bigram_hits"] for p in q_pages)
    rb = sum(p["rev_bigram_hits"] for p in q_pages)
    bf = sum(p["baseline_plain_get_text"]["fwd_word_hits"] for p in q_pages)
    br = sum(p["baseline_plain_get_text"]["rev_word_hits"] for p in q_pages)
    bfb = sum(p["baseline_plain_get_text"]["fwd_bigram_hits"] for p in q_pages)
    brb = sum(p["baseline_plain_get_text"]["rev_bigram_hits"] for p in q_pages)
    blank = sum(1 for p in q_pages if p.get("blank"))
    empty = sum(1 for p in q_pages if p["empty"]) - blank
    fails = [p["page"] for p in q_pages if p["verdict"] == "fail"]
    susp = [p["page"] for p in q_pages if p["verdict"] == "suspect"]
    chars = sum(p["chars"] for p in q_pages)
    reasons = list(extra_reasons or [])
    verdict = "ok"
    checks = {}
    ltext = loose(text)
    if lang != "en":
        title = F.normalize(manifest_row.get("title_fa", "") or "")
        if title:
            core = re.sub(r"[()\[\]:،—\-–]", " ", title)
            words = [w for w in core.split() if len(w) > 1]
            found_words = sum(1 for w in words if loose(w) in ltext)
            checks["title_fa_verbatim_found"] = loose(title) in ltext
            checks["title_fa_words_found"] = "%d/%d" % (found_words, len(words))
        comp = COMPANY_FA.get((manifest_row.get("company") or "").strip().lower(), [])
        if comp:
            checks["company_fa_name_found"] = any(loose(F.normalize(c)) in ltext for c in comp)
            checks["company_fa_name_count"] = sum(ltext.count(loose(F.normalize(c))) for c in comp)
        toks = set(F.fa_tokens(text))
        checks["month_names_found"] = [m for m in MONTHS if m in toks and m not in ("تیر", "مهر", "دی")] + \
            [m for m in ("تیر", "مهر", "دی") if m in toks]
        checks["phrase_gozaresh_salane"] = {"forward": text.count("گزارش سالانه"), "swapped": text.count("سالانه گزارش")}
    pages_with_text = n - empty - blank
    n = n - blank
    if n and empty >= 0.5 * n:
        verdict = "fail"
        reasons.append("no text on %d of %d pages (text outlined or image-only; needs OCR)" % (empty, n))
    elif n and empty >= 0.2 * n and empty >= 3:
        verdict = "suspect"
        reasons.append("no text on %d of %d pages" % (empty, n))
    if lang != "en":
        if r >= 5 and r > 0.5 * f:
            verdict = "fail"
            reasons.append("reversed function words %d vs forward %d" % (r, f))
        elif r >= 5 and r > 0.1 * f:
            verdict = "suspect" if verdict == "ok" else verdict
            reasons.append("reversed function words %d vs forward %d" % (r, f))
        if chars > 2000 and f < 10:
            verdict = "suspect" if verdict == "ok" else verdict
            reasons.append("very few Persian function words in %d chars" % chars)
    if pages_with_text and len(fails) > 0.2 * pages_with_text:
        verdict = "fail"
        reasons.append("%d pages failed" % len(fails))
    elif pages_with_text and (len(susp) + len(fails)) > 0.25 * pages_with_text:
        verdict = "suspect" if verdict == "ok" else verdict
        reasons.append("%d suspect/failed pages" % (len(susp) + len(fails)))
    garbage = sum(p["garbage_chars_dropped"] for p in q_pages)
    if garbage > 0.05 * max(chars, 1):
        verdict = "suspect" if verdict == "ok" else verdict
        reasons.append("%d garbage chars from fonts without usable ToUnicode were dropped" % garbage)
    unm = sum(p.get("unmapped_glyphs_dropped", 0) for p in q_pages)
    if unm >= 20 and unm > 0.5 * max(chars, 1):
        verdict = "fail"
        reasons.append("%d glyphs without a Unicode mapping dropped (%.1f%% of chars): most text is unreadable without OCR"
                       % (unm, 100.0 * unm / max(chars, 1)))
    elif unm > 0.01 * max(chars, 1) and unm >= 20:
        verdict = "suspect" if verdict == "ok" else verdict
        reasons.append("%d glyphs without a Unicode mapping dropped (%.1f%% of chars)" % (unm, 100.0 * unm / max(chars, 1)))
    mixed = sum(p["latin_in_persian_word"] for p in q_pages)
    if mixed > 50:
        verdict = "suspect" if verdict == "ok" else verdict
        reasons.append("%d Latin-extended letters inside Persian words (broken font mapping; words garbled)" % mixed)
    unm_pages = sum(1 for p in q_pages if p.get("unmapped_glyphs_dropped", 0) > 0)
    if unm >= 50 and unm > 0.002 * max(chars, 1) and unm_pages >= 5 and verdict == "ok":
        verdict = "suspect"
        reasons.append("%d glyphs without a Unicode mapping dropped across the document (words missing letters)" % unm)
    zw = sum(p.get("zwnj", 0) for p in q_pages)
    pzw = sum(p.get("plain_zwnj", 0) for p in q_pages)
    hk = sum(p.get("harakat", 0) for p in q_pages)
    phk = sum(p.get("plain_harakat", 0) for p in q_pages)
    sym = sum(p.get("ascii_symbol_chars", 0) for p in q_pages)
    checks["zwnj_extracted_vs_plain"] = [zw, pzw]
    checks["harakat_extracted_vs_plain"] = [hk, phk]
    if lang != "en":
        if bf >= 100 and f < 0.7 * bf:
            verdict = "suspect" if verdict == "ok" else verdict
            reasons.append("Persian function words %d vs %d in MuPDF plain text (words glued or lost)" % (f, bf))
        # free-standing function words joined to neighbours by ZWNJ («در‌سال‌های»): real
        # ZWNJ use never does that, a space glyph read (or set) as ZWNJ does
        glued = sum(1 for tok in re.findall("[؀-ۿ‌]+", text) if "‌" in tok
                    for part in tok.split("‌") if part in _GLUE_WORDS)
        checks["zwnj_glued_function_words"] = glued
        if glued >= 30 and glued >= 0.03 * max(f, 1):
            verdict = "suspect" if verdict == "ok" else verdict
            reasons.append("%d function words glued to neighbours by ZWNJ (word spaces encoded as ZWNJ; %d ZWNJ vs %d "
                           "in MuPDF plain text)" % (glued, zw, pzw))
        if hk >= 100 and hk > 3 * phk + 50:
            verdict = "suspect" if verdict == "ok" else verdict
            reasons.append("harakat %d vs %d in MuPDF plain text (spurious diacritics)" % (hk, phk))
        if sym >= 200 and sym > 0.01 * max(chars, 1):
            verdict = "suspect" if verdict == "ok" else verdict
            reasons.append("%d ASCII symbol characters in Persian text (undecoded font garbage)" % sym)
        words = [w for w in _RE_FA_WORD.findall(text.replace("\u200c", "")) if len(w) >= 3]
        if words:
            lex = _lexicon_words()
            cov = sum(1 for w in words if w in lex) / len(words)
            checks["lexicon_coverage"] = round(cov, 3)
            if len(words) >= 500 and cov < 0.70:
                verdict = "suspect" if verdict == "ok" else verdict
                reasons.append("only %.0f%% of Persian words are known words (garbled text: wrong ToUnicode, "
                               "misplaced spaces or glued words)" % (100 * cov))
    return {
        "verdict": verdict, "reasons": reasons, "pages": len(q_pages), "chars": chars, "empty_pages": empty,
        "blank_pages": blank,
        "fail_pages": fails, "suspect_pages": susp,
        "fwd_word_hits": f, "rev_word_hits": r, "fwd_bigram_hits": fb, "rev_bigram_hits": rb,
        "baseline_plain_get_text": {"fwd_word_hits": bf, "rev_word_hits": br, "fwd_bigram_hits": bfb, "rev_bigram_hits": brb},
        "known_string_checks": checks,
        "reversal_fixes": {
            "visual_order_tokens_reversed": sum(p["visual_order_tokens_reversed"] for p in q_pages),
            "lines_reordered_vs_stream": sum(p["lines_reordered_vs_stream"] for p in q_pages),
            "block_word_order_fixes": sum(p["block_word_order_fixes"] for p in q_pages),
            "lam_alef_restored": sum(p["lam_alef_restored"] for p in q_pages),
        },
    }
