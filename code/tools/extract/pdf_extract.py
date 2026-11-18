"""PDF -> structured document for the corpus extractor (PyMuPDF based)."""
import contextlib
import hashlib
import io
import math
import os
import re
import statistics
import unicodedata
from collections import Counter, defaultdict

import fitz

import fa_text as F

BOLD_RE = re.compile(r"bold|black|heavy|extrabold|semibold|demibold|ultra", re.I)
CAPTION_RE = re.compile(r"^\s*(شکل|نمودار|جدول|تصویر|منبع|مأخذ|ماخذ|اینفوگرافیک|Figure|Fig\.|Table|Chart|Graph|Source|Exhibit)\b", re.I)
NUM_LIST_RE = re.compile(r"^\s*([0-9]{1,2}|[a-zA-Z]|[٠-٩۰-۹]{1,2}|[آ-ی])\s*[\.\)\-–]\s+\S")
TEXT_FLAGS = fitz.TEXT_PRESERVE_WHITESPACE | fitz.TEXT_PRESERVE_LIGATURES | fitz.TEXT_MEDIABOX_CLIP

ALEF_LEXICON = set(F.normalize(w) for w in (
    "با تا ما ها های هایی سال سالانه بازار کاربران کاربر خدمات تلاش اما تنها جایگاه چالش "
    "اعتماد آسان سازمان زمان میزان مشتریان رانندگان فعالان همکاران کارکنان مدیران تعداد "
    "معنادار تامین بالا حالا طلا آنها شما برنامه ارائه مالی دارای کالا کالاها فروشندگان "
    "تهران ایران میان پایان جهان انسانی استان شهرها ماه ماهانه قرار بیشترین شهرستان "
    "کسب‌وکارها سفر سفرها ساختار تاثیر تأثیر یکپارچه انتظارات فراهم داشته است برای این از"
).split())


# --------------------------------------------------------------------------
# small geometry helpers
# --------------------------------------------------------------------------

def rect_area(b):
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def union(boxes):
    boxes = list(boxes)
    return [min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes), max(b[3] for b in boxes)]


def inside(cx, cy, b, pad=0.0):
    return b[0] - pad <= cx <= b[2] + pad and b[1] - pad <= cy <= b[3] + pad


def x_overlap(a, b):
    return min(a[2], b[2]) - max(a[0], b[0])


def y_overlap(a, b):
    return min(a[3], b[3]) - max(a[1], b[1])


def r2(b):
    return [round(float(v), 2) for v in b]


# --------------------------------------------------------------------------
# character collection
# --------------------------------------------------------------------------

JOINER_CHARS = set("‌‍​‎‏­⁠﻿")

# per-document decisions made by the pre-pass over all pages (build_gid_recovery);
# reset by extract_pdf for every document
_DOC = {"bracket_coding": {}, "mark_gids": set(), "kashida_gids": set(), "vocab": set()}


def bracket_coding(c, cls):
    """'S' / 'L' / None for a bracket glyph enclosing content of class cls ('L' Latin,
    'R' Arabic script or digits) -- see fa_text.fix_brackets. Decided per font+glyph
    id, else per font+bracket family, else per document+family, always within the
    content class: Word exports code the same glyph logically around Persian text and
    by shape around Latin text."""
    bc = _DOC["bracket_coding"]
    if not bc:
        return None
    fam = F.BRACKET_FAMILY.get(c["c"])
    for key in ((c.get("font"), c.get("gid"), c["c"], cls), (c.get("font"), fam, cls), ("*", fam, cls)):
        v = bc.get(key)
        if v:
            return v
    return None


def _bracket_votes(baselines, votes):
    """Which side of its content a bracket glyph hugs tells its drawn shape: a glyph
    with content touching on its right and a space (or nothing) on its left is drawn
    as an opening-shaped '(' / '«'. Comparing that with the glyph's code says whether
    the font codes brackets by shape ('S') or by logical meaning ('L')."""
    for raw in baselines.values():
        raw.sort(key=lambda it: (it[0] + it[1]) / 2.0)
        items = []
        for it in raw:
            # a glyph painted twice (fake bold) would be its own neighbour
            if items and items[-1][2] == it[2] and abs(items[-1][0] - it[0]) < 0.3 and abs(items[-1][1] - it[1]) < 0.3:
                continue
            items.append(it)
        n = len(items)
        for i, (x0, x1, ch, font, gid, size) in enumerate(items):
            if ch not in F.MIRROR:
                continue
            lt = items[i - 1] if i > 0 else None
            rt = items[i + 1] if i + 1 < n else None

            def content(it):
                return it is not None and not it[2].isspace() and it[2] not in F.MIRROR \
                    and F.char_class(it[2]) in ("R", "L", "D")

            def sep(it, d):
                return it is None or it[2].isspace() or d > 0.25 * size
            d_l = (x0 - lt[1]) if lt is not None else 1e9
            d_r = (rt[0] - x1) if rt is not None else 1e9
            side = None
            if content(rt) and d_r < 0.15 * size and sep(lt, d_l):
                side, inner = "left", rt[2]
            elif content(lt) and d_l < 0.15 * size and sep(rt, d_r):
                side, inner = "right", lt[2]
            if side is None:
                continue
            cls = "L" if F.char_class(inner) == "L" else "R"
            k = "S" if ((ch in F.OPENERS) == (side == "left")) else "L"
            fam = F.BRACKET_FAMILY[ch]
            votes[(font, gid, ch, cls)][k] += 1
            votes[(font, fam, cls)][k] += 1
            votes[("*", fam, cls)][k] += 1


def _decide_brackets(votes):
    out = {}
    for key, c in votes.items():
        tot = sum(c.values())
        best, bn = c.most_common(1)[0]
        need = 2 if key[0] != "*" and not isinstance(key[1], str) else (3 if key[0] != "*" else 5)
        if tot >= need and bn >= 0.75 * tot:
            out[key] = best
    return out


def actualtext_joiners(page):
    """MuPDF structured-text chars keyed by glyph position. Used for what texttrace
    gets wrong: InDesign writes ZWNJ as a space glyph wrapped in
    /Span <</ActualText (U+200C)>> (texttrace reports the space, rawdict the ZWNJ);
    some YekanBakh/Irancell subsets map the space glyph to U+200C in ToUnicode
    (texttrace reports ZWNJ, rawdict a space); some glyphs texttrace reports as U+FFFD
    are mapped in rawdict; and InDesign small caps carry per-letter ActualText."""
    out = {}
    try:
        raw = page.get_text("rawdict", flags=TEXT_FLAGS)
    except Exception:  # noqa: BLE001
        return out
    for b in raw["blocks"]:
        if b.get("type", 0) != 0:
            continue
        for l in b["lines"]:
            for s in l["spans"]:
                for c in s["chars"]:
                    x0, _, x1, _ = c["bbox"]
                    key = (round(x0 * 2), round(x1 * 2), round(c["origin"][1]))
                    out[key] = out.get(key, "") + c["c"]
    return out


def build_gid_recovery(doc, page_numbers):
    """Glyphs whose font has no ToUnicode entry come out as U+FFFD. Subsets of one font
    family embedded in the same PDF (Irancell-Light, Irancell-Light-SC700, -Bold ...)
    usually share glyph ids, and another subset may map the same gid. A donor font is
    accepted only if, on the gids both fonts map, they agree (after NFKC) >= 90% over
    >= 15 gids; the recovered char is the donors' majority vote."""
    maps = defaultdict(lambda: defaultdict(Counter))
    unmapped = defaultdict(Counter)
    patterns = []
    vocab_words = []
    font_stats = defaultdict(Counter)
    bvotes = defaultdict(Counter)
    mark_stats = defaultdict(Counter)      # (font, gid) -> total / with_mark
    kash_stats = defaultdict(Counter)      # (font, gid) -> n / self_adjacent / narrow
    for pno in page_numbers:
        try:
            trace = doc[pno].get_texttrace()
        except Exception:  # noqa: BLE001
            continue
        baselines = defaultdict(list)
        for span in trace:
            f = span["font"] or ""
            size = float(span["size"]) or 1.0
            dx, dy = span["dir"]
            horizontal = abs(dy) < 0.05 and dx > 0 and not span.get("wmode")
            entries = []
            for (u, gid, origin, bbox) in span["chars"]:
                ch = chr(u) if 0 <= u < 0x110000 else "\ufffd"
                if gid == -1:
                    if entries:
                        entries[-1][1] += ch
                    continue
                entries.append([gid, ch, bbox])
                if horizontal:
                    baselines[round(origin[1] / 2.0)].append((bbox[0], bbox[2], ch, f, gid, size))
            fstat = font_stats[f]
            prev_e = None
            for gid, s, bbox in entries:
                for ch in s[:1]:
                    if ch.isspace():
                        continue
                    cp = ord(ch)
                    if 0x0600 <= cp <= 0x06FF or 0xFB50 <= cp <= 0xFDFF or 0xFE70 <= cp <= 0xFEFE:
                        fstat["ar"] += 1
                    elif ch.isascii() and ch.isalpha():
                        fstat["lat"] += 1
                    elif ch.isdigit():
                        fstat["dig"] += 1
                    elif ch in ".,:%()-/+":
                        fstat["punct"] += 1
                    elif ch == "\ufffd" or unicodedata.category(ch) in ("Cc", "Co"):
                        fstat["bad"] += 1
                    elif F.char_class(ch) == "U" or unicodedata.category(ch) == "Cn":
                        # variation selectors, combining half marks, CJK punctuation
                        fstat["odd"] += 1
                    else:
                        fstat["sym"] += 1
                if len(s) >= 4 and any("\u0600" <= ch <= "\u06ff" for ch in s):
                    fstat["actualtext"] += 1
                key = (f, gid)
                if s and F.char_class(s[0]) == "R":
                    mark_stats[key]["total"] += 1
                    if len(s) == 2 and 0x064B <= ord(s[1]) <= 0x0652:
                        mark_stats[key]["with_mark"] += 1
                if len(s) == 1 and s in "\u0627\u062f\u0630\u0631\u0632\u0698\u0648":
                    w = (bbox[2] - bbox[0]) / size
                    kash_stats[key]["n"] += 1
                    if w < 0.2:
                        kash_stats[key]["narrow"] += 1
                    if prev_e is not None and prev_e[0] == gid and bbox[0] >= prev_e[2][2] - 0.1 * size:
                        kash_stats[key]["self_adjacent"] += 1
                prev_e = (gid, s, bbox)
            for gid, s, _ in entries:
                if s == "\ufffd":
                    unmapped[f][gid] += 1
                elif s and len(s) <= 3 and not any(ch.isspace() for ch in s) \
                        and not any(unicodedata.category(ch) in ("Cc", "Co") for ch in s):
                    # single glyph mappings only (a letter, a ligature): a glyph carrying an
                    # ActualText phrase must not become the "recovered" text of that glyph id
                    # in another font
                    maps[f][gid][unicodedata.normalize("NFKC", s)] += 1
            # word patterns for the self-supervised stage: runs of entries between spaces
            word = []
            for gid, s, _ in entries + [[None, " ", None]]:
                if s.isspace() or (s and F.char_class(s) in ("N", "D", "S") and s != "\ufffd"):
                    if word and any(isinstance(w, tuple) for w in word):
                        patterns.append(word)
                    elif word:
                        vocab_words.append("".join(word))
                    word = []
                elif s == "\ufffd":
                    word.append((f, gid))
                else:
                    word.append(s)
        _bracket_votes(baselines, bvotes)
    recovery = {}
    report = {}
    for f, gids in unmapped.items():
        if not maps.get(f):
            continue
        mine = {g: c.most_common(1)[0][0] for g, c in maps[f].items()}
        donors = []
        for g2, m2 in maps.items():
            if g2 == f:
                continue
            shared = [g for g in mine if g in m2]
            if len(shared) < 15:
                continue
            agree = sum(1 for g in shared if m2[g].most_common(1)[0][0] == mine[g])
            if agree >= 0.9 * len(shared):
                donors.append(g2)
        rec = {}
        for gid in gids:
            votes = Counter(maps[d][gid].most_common(1)[0][0] for d in donors if gid in maps[d])
            if votes:
                best, n = votes.most_common(1)[0]
                if n >= 0.7 * sum(votes.values()):
                    rec[gid] = best
        if rec:
            recovery[f] = rec
        report[f] = {"unmapped_glyph_ids": len(gids), "unmapped_occurrences": sum(gids.values()),
                     "donor_fonts": donors, "recovered_glyph_ids": len(rec),
                     "recovered_occurrences": sum(gids[g] for g in rec)}
    garbage = {}
    for f, c in font_stats.items():
        tot = sum(v for k, v in c.items() if k != "actualtext")
        if tot >= 8 and c["ar"] < 0.05 * tot and c["lat"] < 0.5 * tot and (c["bad"] + c["sym"] + c["odd"]) >= 0.3 * tot \
                and c["actualtext"] == 0:
            garbage[f] = dict(c)
        elif tot >= 20 and c["odd"] >= 0.1 * tot and (tot - c["ar"]) >= 0.4 * tot and c["actualtext"] == 0:
            # a cmap that maps glyphs to variation selectors / CJK punctuation mixed with
            # stray Arabic letters (Podro 1400 Kalameh headings): nothing readable
            garbage[f] = dict(c, rule="odd_codepoints_digits_kept")
    report["_garbage_fonts"] = garbage
    # Canva-style exports: one ActualText phrase per paragraph rides on a single glyph,
    # while the paragraph's visible glyphs (on this and other lines) carry no usable
    # code (U+FFFD or ASCII junk). In such a font only the phrases are text.
    atf = {}
    for f, c in font_stats.items():
        tot = sum(v for k, v in c.items() if k != "actualtext")
        rest = tot - c["actualtext"]
        if c["actualtext"] >= 1 and rest >= 8 and (c["ar"] - c["actualtext"]) < 0.2 * rest:
            atf[f] = dict(c)
    report["_actualtext_fonts"] = atf
    # a glyph whose ToUnicode is letter+harakat on (nearly) every use: a mapping bug
    # (Jabama 1402 Alibaba-Regular: every dal carries a fatha no one painted)
    report["_mark_gids"] = {k for k, c in mark_stats.items() if c["with_mark"] >= 20 and c["with_mark"] >= 0.9 * c["total"]}
    # a narrow glyph mapped to a non-joining letter that keeps sitting next to itself
    # is a justification kashida with a wrong ToUnicode (Ilia 1399: «خوووودمات»)
    report["_kashida_gids"] = {k for k, c in kash_stats.items()
                               if c["n"] >= 20 and c["narrow"] >= 0.9 * c["n"] and c["self_adjacent"] >= 0.25 * c["n"]}
    report["_bracket_coding"] = _decide_brackets(bvotes)
    report["_vocab"] = set(F.normalize(w).replace("‌", "") for w in vocab_words)
    vocab_rec = _vocab_recovery(patterns, vocab_words, unmapped, recovery)
    for f, rec in vocab_rec.items():
        recovery.setdefault(f, {}).update(rec)
        r = report.setdefault(f, {"unmapped_glyph_ids": len(unmapped[f]), "unmapped_occurrences": sum(unmapped[f].values()),
                                  "donor_fonts": [], "recovered_glyph_ids": 0, "recovered_occurrences": 0})
        r["recovered_by_document_vocabulary"] = {str(g): v for g, v in rec.items()}
        r["recovered_glyph_ids"] += len(rec)
        r["recovered_occurrences"] += sum(unmapped[f][g] for g in rec)
    return recovery, report


_PERSIAN_LETTERS = [chr(c) for c in range(0x0621, 0x063B)] + [chr(c) for c in range(0x0641, 0x0649)] + \
    ["\u067e", "\u0686", "\u0698", "\u06a9", "\u06af", "\u06cc", "\u0622", "\u0624", "\u0626", "\u06c0"]


_LEXICON = None


def _lexicon():
    """Static Persian word list (fa_lexicon.txt, built by build_lexicon.py from clean
    Unicode Persian text). Used only to confirm glyph-recovery guesses."""
    global _LEXICON
    if _LEXICON is None:
        _LEXICON = {}
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fa_lexicon.txt")
        if os.path.exists(path):
            with open(path, encoding="utf-8") as fh:
                for line in fh:
                    if line.startswith("#"):
                        continue
                    parts = line.rstrip("\n").split("\t")
                    w = parts[0].strip()
                    if 2 <= len(w) <= 14:
                        try:
                            _LEXICON[w] = int(parts[1]) if len(parts) > 1 else 1
                        except ValueError:
                            _LEXICON[w] = 1
    return _LEXICON


def _vocab_recovery(patterns, vocab_words, unmapped, already, rounds=3):
    """Unmapped glyph with no donor font: find the Persian letter that, in place of
    that glyph, turns its words into words occurring elsewhere in the same document
    (properly mapped glyphs) or in the static lexicon. Words with one or two unknown
    glyphs are matched against a masked index (both letter orders, since the stream
    may be visual).

    Two votes per glyph: unweighted (each matching word shares one vote among its
    candidate letters) and frequency-weighted (candidate words weighted by document
    plus lexicon frequency, so 'کاربران' outvotes 'کارگران' for 'کا??ران'). A letter
    is accepted when either vote gives >= 3 votes, 2.5x the runner-up and >= 30% of
    that glyph's words (the weighted vote also needs >= 2 distinct words); if both
    votes pass with different letters the glyph stays unmapped. Accepted glyphs are
    fed back and the words re-scored, up to `rounds` passes, so a glyph that only
    ever occurs next to another unmapped glyph becomes a one-unknown mask."""
    if not patterns:
        return {}
    lex = _lexicon()
    dc = Counter()
    for w in vocab_words:
        n = F.normalize(w).replace("\u200c", "")
        if 2 <= len(n) <= 14:
            dc[n] += 1
    freq = Counter()
    for w, n in dc.items():
        freq[w] += n
        freq[w[::-1]] += n
    for w, n in lex.items():
        freq[w] += n
        freq[w[::-1]] += n
    index1 = defaultdict(set)
    index2 = defaultdict(set)
    for w in freq:
        n = len(w)
        for i in range(n):
            index1[w[:i] + "?" + w[i + 1:]].add(w)
        if n <= 12:
            for i in range(n):
                for j in range(i + 1, n):
                    index2[w[:i] + "?" + w[i + 1:j] + "?" + w[j + 1:]].add(w)
    found = defaultdict(dict)
    known = {f: dict(m) for f, m in already.items()}
    for _ in range(rounds):
        wvotes = defaultdict(Counter)
        uvotes = defaultdict(Counter)
        support = defaultdict(lambda: defaultdict(set))
        masks_all = defaultdict(set)
        masks_for = defaultdict(lambda: defaultdict(set))
        totals = Counter()
        for pat in patterns:
            elems = []
            for e in pat:
                if isinstance(e, tuple):
                    if e[1] in known.get(e[0], {}):
                        elems.extend(F.normalize(known[e[0]][e[1]]))
                    else:
                        elems.append(e)
                else:
                    elems.extend(F.normalize(e).replace("\u200c", ""))
            pos = [i for i, e in enumerate(elems) if isinstance(e, tuple)]
            if not (1 <= len(pos) <= 2) or len(elems) < 2:
                continue
            mask = "".join("?" if isinstance(e, tuple) else e for e in elems)
            for k in set(elems[i] for i in pos):
                totals[k] += 1
                masks_all[k].add(mask)
            cands = (index1 if len(pos) == 1 else index2).get(mask, ())
            if not cands:
                continue
            opts = []
            for w in cands:
                assign = {}
                ok = True
                for i in pos:
                    k = elems[i]
                    if k in assign and assign[k] != w[i]:
                        ok = False
                        break
                    assign[k] = w[i]
                if ok and all(F.char_class(L) == "R" for L in assign.values()):
                    opts.append((w, assign))
            if not opts:
                continue
            per_key = defaultdict(set)
            for w, assign in opts:
                for k, L in assign.items():
                    per_key[k].add(L)
            for k, letters in per_key.items():
                for L in letters:
                    uvotes[k][L] += 1.0 / len(letters)
                    masks_for[k][L].add(mask)
            tot = sum(freq[w] for w, _ in opts)
            for w, assign in opts:
                for k, L in assign.items():
                    wvotes[k][L] += freq[w] / tot
                    support[k][L].add(w)
        new = 0
        for k in set(uvotes) | set(wvotes):
            f, gid = k
            if gid in known.get(f, {}):
                continue
            res = []
            for votes, need_words in ((uvotes[k], False), (wvotes[k], True)):
                ranked = votes.most_common(2)
                if not ranked:
                    res.append(None)
                    continue
                best, bn = ranked[0]
                second = ranked[1][1] if len(ranked) > 1 else 0.0
                # coverage: 30% of the glyph's occurrences, or (when one frequent word is
                # missing from every vocabulary) 40% of its distinct word patterns, >= 4
                nm = len(masks_for[k][best])
                covered = bn >= 0.3 * totals[k] or (nm >= 4 and nm >= 0.4 * len(masks_all[k]))
                ok = bn >= 3 and bn >= 2.5 * second and covered
                if need_words:
                    ok = ok and len(support[k][best]) >= 2
                res.append(best if ok else None)
            u, w = res
            if u and w and u != w:
                continue
            best = u or w
            if best:
                found[f][gid] = best
                known.setdefault(f, {})[gid] = best
                new += 1
        if not new:
            break
    return found


def collect_lines(page, gid_recovery=None):
    """Glyph entries in raw content-stream order from page.get_texttrace().

    Why not rawdict: MuPDF re-orders visual-order Arabic runs itself and, while doing
    so, splits multi-codepoint ligature glyphs (lam-alef, sa, ka ...) into chars that
    land in the wrong order. texttrace keeps the stream order and marks the extra
    codepoints of such a glyph with gid -1, so each glyph stays one entry ("\u0644\u0627") and
    the geometric ordering in fa_text can put entries, not codepoints, in order."""
    rect = page.rect
    try:
        trace = page.get_texttrace()
    except Exception:  # noqa: BLE001
        return []
    overrides = actualtext_joiners(page)
    mark_gids = _DOC["mark_gids"]
    kashida_gids = _DOC["kashida_gids"]
    lines = []
    seq = 0
    cur = None
    for span_no, span in enumerate(trace):
        dx, dy = span["dir"]
        horizontal = abs(dy) < 0.05 and dx > 0 and not span.get("wmode")
        size = float(span["size"]) or 1.0
        font = span["font"] or ""
        bold = bool(span["flags"] & 16) or bool(BOLD_RE.search(font))
        invisible = span.get("type") == 3 or (span.get("opacity") is not None and span["opacity"] == 0)
        prev = None
        clipped = False
        rec_map = gid_recovery.get(font) if gid_recovery else None
        for (u, gid, origin, bbox) in span["chars"]:
            ch = chr(u) if 0 <= u < 0x110000 else "\ufffd"
            if gid == -1:
                if clipped:
                    continue
                if prev is not None and (font, prev["gid"]) in mark_gids and 0x064B <= u <= 0x0652:
                    # harakat that only come from a broken letter+mark ToUnicode entry
                    continue
                if prev is not None:
                    prev["c"] += ch
                elif cur is not None and cur["chars"]:
                    cur["chars"][-1]["c"] += ch
                continue
            x0, y0, x1, y1 = bbox
            if x1 < x0:
                x0, x1 = x1, x0
            cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
            if not (rect.x0 - 1 <= cx <= rect.x1 + 1 and rect.y0 - 1 <= cy <= rect.y1 + 1):
                prev = None
                clipped = True
                continue
            clipped = False
            recovered = False
            if ch == "\ufffd" and rec_map and gid in rec_map:
                ch = rec_map[gid]
                recovered = True
            if (font, gid) in kashida_gids:
                ch = "\u0640"
            if overrides and (ch.isspace() or ch == "\ufffd" or ch == "\u200c" or (ch.isascii() and ch.isalpha())):
                ov = overrides.get((round(x0 * 2), round(x1 * 2), round(origin[1])))
                if not ov:
                    pass
                elif ch == "\u200c":
                    if not ov.strip(" "):
                        # the font maps its space glyph to ZWNJ; MuPDF's text device
                        # (and the page itself) has a word space here
                        ch = " "
                elif ch.isspace():
                    if all(o in JOINER_CHARS for o in ov):
                        ch = ov
                elif ch == "\ufffd":
                    # MuPDF's structured text sometimes knows a glyph texttrace cannot map
                    # (encoding Differences, ActualText). Glyph-id-as-codepoint fallbacks
                    # (Latin Extended letters) and controls are not accepted.
                    clean = "".join(o for o in ov if o not in JOINER_CHARS and o != "\ufffd" and not o.isspace()
                                    and unicodedata.category(o) not in ("Cc", "Co", "Cn")
                                    and not (0x0100 <= ord(o) <= 0x024F))
                    if clean:
                        ch = clean
                        recovered = True
                elif len(ov) == 1 and ov != ch and ov.lower() == ch.lower():
                    # InDesign small caps: uppercase glyphs with per-letter ActualText
                    ch = ov
            e = {"c": ch, "x0": x0, "y0": y0, "x1": x1, "y1": y1, "ox": origin[0], "oy": origin[1],
                 "size": size, "font": font, "gid": gid, "span": span_no, "bold": bold, "seq": seq,
                 "invisible": invisible, "recovered": recovered}
            seq += 1
            prev = e
            # same raw line: same direction, same baseline, not a far jump
            if cur is not None and cur["horizontal"] == horizontal and horizontal:
                tol = 0.35 * min(size, cur["size"])
                far = e["x0"] > cur["x1"] + 3.0 * size or e["x1"] < cur["x0"] - 3.0 * size
                # a combining mark sits above or below the baseline; it stays with the
                # line whose glyphs it is drawn on
                is_mark = x1 - x0 <= 0.05 * size and F.char_class(ch) == "M"
                if (abs(e["oy"] - cur["oy"]) <= tol or (is_mark and abs(e["oy"] - cur["oy"]) <= 1.2 * size)) and not far:
                    cur["chars"].append(e)
                    cur["x0"] = min(cur["x0"], x0)
                    cur["x1"] = max(cur["x1"], x1)
                    continue
            elif cur is not None and not horizontal and not cur["horizontal"] and cur["dir"] == span["dir"]:
                if math.hypot(e["ox"] - cur["chars"][-1]["ox"], e["oy"] - cur["chars"][-1]["oy"]) < 3.0 * size:
                    cur["chars"].append(e)
                    continue
            cur = {"chars": [e], "horizontal": horizontal, "dir": span["dir"], "size": size,
                   "oy": e["oy"], "x0": x0, "x1": x1}
            lines.append(cur)
    # the same glyph painted twice at (almost) the same spot: fake bold, a fill then a
    # stroke pass, or an offset drop shadow. Keep the first paint only. Two paints of
    # one glyph code overlap by more than half the glyph's width and sit within a
    # quarter em vertically; distinct glyphs of the same letter never do.
    seen = defaultdict(list)
    dropped = 0
    for ln in lines:
        keep = []
        for c in ln["chars"]:
            q = max(0.12 * c["size"], 0.6)
            w = c["x1"] - c["x0"]
            tx = max(q if w <= 0.01 else min(q, 0.9 * w), min(0.5 * w, 0.6 * c["size"]))
            ty = max(q, 0.25 * c["size"])
            bucket = int(c["oy"] // 4)
            near = False
            for b in (bucket - 1, bucket, bucket + 1):
                for other in seen.get((c["c"], b), ()):
                    if abs(other["x0"] - c["x0"]) <= tx and abs(other["oy"] - c["oy"]) <= ty \
                            and abs((other["x1"] - other["x0"]) - w) <= q:
                        near = True
                        break
                if near:
                    break
            if near:
                dropped += 1
                continue
            seen[(c["c"], bucket)].append(c)
            keep.append(c)
        ln["chars"] = keep
    lines = [ln for ln in lines if any(not c["c"].isspace() for c in ln["chars"])]
    if _DOC.get("gap_spaces"):
        for ln in lines:
            if ln["horizontal"]:
                _DOC["gap_spaces_inserted"] = _DOC.get("gap_spaces_inserted", 0) + insert_gap_spaces(ln)
    for ln in lines:
        cs = ln["chars"]
        ln["bbox"] = [min(c["x0"] for c in cs), min(c["y0"] for c in cs), max(c["x1"] for c in cs), max(c["y1"] for c in cs)]
        ln["duplicate_glyphs_dropped"] = 0
    if lines:
        lines[0]["duplicate_glyphs_dropped"] = dropped
    return lines


# OPT-IN (2026-09-23, after review): off unless EXTRACT_GAP_SPACES=1, so the extractor behaves exactly as when the study
# corpus was frozen; extract.py records the option in extract_status.json and only reuses an extraction made with the
# same options. The demo (demo/server.py) turns it on.
GAP_SPACES_ENABLED = os.environ.get("EXTRACT_GAP_SPACES") == "1"
GAP_SPACE_EM = 0.12          # a gap wider than this (in font sizes) between two adjacent glyphs is a word space
GAP_SPACE_DOC_SHARE = 0.01   # ... applied only when under 1% of the document's glyphs are space glyphs


def _is_arabic(ch):
    return any("؀" <= c <= "ۿ" or "ﭐ" <= c <= "﻿" or "ݐ" <= c <= "ݿ" for c in ch)


def document_space_share(doc, pnos, max_pages=40):
    """-> (space glyphs / all glyphs, glyphs counted) over up to max_pages pages spread through the document."""
    step = max(1, len(pnos) // max_pages)
    tot = sp = 0
    for pno in pnos[::step][:max_pages]:
        try:
            for s in doc[pno].get_texttrace():
                for c in s["chars"]:
                    if c[1] == -1:
                        continue
                    tot += 1
                    sp += c[0] in (32, 0xA0)
        except Exception:  # noqa: BLE001
            continue
    return (sp / tot if tot else 1.0), tot


def insert_gap_spaces(ln):
    """Documents that write NO space glyphs (words are separated by positioning alone - a PDF viewer infers the
    spaces from the gaps, as MuPDF's rawdict does) would otherwise extract as glued words. Measured on such a file
    (2026-09-23): gaps between letters of a word sit at 0.00 +- 0.05 font sizes, gaps between words at 0.15-0.40.
    A synthetic space goes between two stream-adjacent, left-to-right glyphs of a line that has no space of its own
    when the gap exceeds GAP_SPACE_EM. Arabic-script glyphs are never split (their stream order is not visual).
    Only called when the whole document shows the defect (document_space_share). Returns spaces inserted."""
    cs = ln["chars"]
    if any(c["c"].isspace() for c in cs) or len(cs) < 2:
        return 0
    out, n = [cs[0]], 0
    for a, b in zip(cs, cs[1:]):
        size = min(a["size"], b["size"]) or 1.0
        gap = b["x0"] - a["x1"]
        if gap > GAP_SPACE_EM * size and gap < 3.0 * size and not _is_arabic(a["c"]) and not _is_arabic(b["c"]):
            sp = dict(a, c=" ", x0=a["x1"], x1=b["x0"], gid=-2, seq=a["seq"] + 0.5, recovered=False,
                      synthetic_space=True)
            out.append(sp)
            n += 1
        out.append(b)
    ln["chars"] = out
    return n


def mark_garbage(chars):
    """Mark runs of non-Arabic glyph entries that contain control-character garbage
    (fonts without a usable ToUnicode). Returns number of entries marked."""
    n = 0
    run = []

    def flush(run):
        k = 0
        if not run:
            return 0
        bad = sum(1 for c in run if F.char_class(c["c"]) == "U" and c["c"][:1] < "")
        nonspace = sum(1 for c in run if not c["c"].isspace())
        if bad and nonspace and bad >= 0.05 * nonspace:
            for c in run:
                c["garbage"] = True
                k += 1
        return k

    for c in chars:
        cls = F.char_class(c["c"])
        if cls in ("R", "M", "Z") and "؀" <= c["c"][:1] <= "﻾":
            n += flush(run)
            run = []
        else:
            run.append(c)
    n += flush(run)
    return n


LAMALEF_LEXICON = set(F.normalize(w) for w in (
    "تلاش تلاش‌ها تلاشمان اطلاعات اطلاع اطلاع‌رسانی خلاق خلاقیت کلان علاوه طلا طلای بالا بالای کالا کالاها "
    "کالای حالا دلار دلاری میلادی خلاصه انقلاب علاقه علاقه‌مند علاقه‌مندان آنلاین لازم لایه ملاک پلاک اعلام "
    "سلامت سلامتی اسلامی استقلال ملاحظه اصلاح اصلاحات کلاس بلاگ صلاحیت ملاقات احتمالا معمولا فعلا اولا کاملا "
    "نسبتا بلافاصله گلایه ولایت پلاستیک کلاهبرداری اخلاق اخلاقی کلاب پلاس بلاک بلاکچین لاین آفلاین ویلا "
    "تلاشی تلاشی کلاسیک اسلاید لاتاری طلایی بالاترین بالاتر والا اعلامیه ملاک‌ها غلات صلاح"
).split())
_ALEF_FORMS = ("ا", "ﺍ", "ﺎ")


def lamalef_candidates(chars, ratio):
    return [c for c in chars if c["c"] in _ALEF_FORMS and (c["x1"] - c["x0"]) >= ratio * c["size"]]


_NARROW_OK = set("اآأإـءﺍﺎﺁﺂﺃﺄﺇﺈﺀ")


def narrow_spurious_candidates(chars):
    """Indices of glyphs mapped to an Arabic letter but drawn far narrower than any
    letter (< 0.16 em) right next to a real letter glyph: some fonts draw one letter
    with two glyphs and map both, giving 'پپیشرفته' or 'ککند'."""
    out = []
    n = len(chars)
    for i, c in enumerate(chars):
        s = c["c"]
        if len(s) != 1 or s in _NARROW_OK or F.char_class(s) != "R":
            continue
        size = c["size"] or 1.0
        if (c["x1"] - c["x0"]) >= 0.16 * size or (c["x1"] - c["x0"]) <= 0.0:
            continue
        # the letter's real (wide, mapped) glyph sits immediately to the right of the
        # narrow piece; when that wide glyph is unmapped (U+FFFD) the narrow piece is
        # the only carrier of the letter and is kept
        for j in (i - 1, i + 1):
            if 0 <= j < n:
                o = chars[j]
                if "�" not in o["c"] and F.char_class(o["c"]) == "R" and (o["x1"] - o["x0"]) >= 0.25 * size and \
                        abs(o["x0"] - c["x1"]) < 0.1 * size:
                    out.append(i)
                    break
    return out


def detect_spurious_narrow_fonts(doc, page_numbers, gid_recovery=None):
    """Per font, drop spurious narrow letter glyphs only if doing so turns clearly more
    words into words that MuPDF's own plain-text extraction of the same page also
    contains (an independent oracle; letter order inside a word is checked both ways)."""
    sample = list(page_numbers)
    if len(sample) > 40:
        step = len(sample) / 40.0
        sample = [sample[int(i * step)] for i in range(40)]
    score = defaultdict(lambda: [0, 0, 0])
    for pno in sample:
        page = doc[pno]
        lines = collect_lines(page, gid_recovery)
        if not any(narrow_spurious_candidates(ln["chars"]) for ln in lines if ln["horizontal"]):
            continue
        plain = set()
        for t in F.fa_tokens(F.normalize(page.get_text())):
            plain.add(t)
            plain.add(t[::-1])
        for ln in lines:
            if not ln["horizontal"]:
                continue
            idx = narrow_spurious_candidates(ln["chars"])
            if not idx:
                continue
            fonts = set(ln["chars"][i]["font"] for i in idx)
            texts = []
            for drop in (False, True):
                cs = [dict(c) for k, c in enumerate(ln["chars"]) if not (drop and k in set(idx))]
                objs = build_line_objects({"chars": cs}, True, None, defaultdict(int))
                texts.append(F.normalize(" ".join(o["text_raw"] for o in objs)))
            ta = F.fa_tokens(texts[0])
            tb = F.fa_tokens(texts[1])
            wa = sum(1 for w in ta if w in plain)
            wb = sum(1 for w in tb if w in plain)
            for f in fonts:
                score[f][0] += wa
                score[f][1] += wb
                score[f][2] += sum(1 for i in idx if ln["chars"][i]["font"] == f)
    enabled = set()
    for font, (wa, wb, n) in score.items():
        if wb - wa >= 5 and (wb - wa) >= 0.5 * n:
            enabled.add(font)
    # weights/subsets of the same family with too few samples of their own follow the
    # family decision, provided their own evidence does not point the other way
    fams = set(font_family(f) for f in enabled)
    for font, (wa, wb, n) in score.items():
        if font not in enabled and font_family(font) in fams and wb > wa:
            enabled.add(font)
    return enabled, {k: {"plain_word_hits_keep": v[0], "plain_word_hits_drop": v[1], "narrow_glyphs": v[2]} for k, v in score.items()}


_WEIGHT_RE = re.compile(r"[-_ ]?(SC\d+|UltraLight|ExtraLight|Light|Thin|Regular|Book|Medium|SemiBold|DemiBold|ExtraBold|UltraBold|Bold|Black|Heavy|Italic|FaNum|Num)$", re.I)


def font_family(name):
    base = re.sub(r"^[A-Z]{6}\+", "", name or "")
    prev = None
    while prev != base:
        prev = base
        base = _WEIGHT_RE.sub("", base)
    return base.lower()


def detect_lamalef_fonts(doc, page_numbers, gid_recovery=None):
    """A lam-alef ligature glyph whose ToUnicode says only "ا" loses its lam (e.g.
    'تاش' for 'تلاش'). Such glyphs are about twice as wide as a plain alef. Per font,
    restore "لا" on wide alefs only if that turns clearly more words into known words."""
    sample = list(page_numbers)
    if len(sample) > 40:
        step = len(sample) / 40.0
        sample = [sample[int(i * step)] for i in range(40)]
    widths = defaultdict(list)
    per_page = []
    for pno in sample:
        lines = collect_lines(doc[pno], gid_recovery)
        per_page.append(lines)
        for ln in lines:
            for c in ln["chars"]:
                if c["c"] in _ALEF_FORMS and c["size"] > 0:
                    widths[c["font"]].append((c["x1"] - c["x0"]) / c["size"])
    ratio = {}
    for font, ws in widths.items():
        if len(ws) >= 20:
            med = statistics.median(ws)
            ratio[font] = max(0.45, 1.7 * med)
    score = defaultdict(lambda: [0, 0, 0])
    for lines in per_page:
        for ln in lines:
            if not ln["horizontal"]:
                continue
            cands = [c for c in ln["chars"] if c["font"] in ratio and c["c"] in _ALEF_FORMS
                     and (c["x1"] - c["x0"]) >= ratio[c["font"]] * c["size"]]
            if not cands:
                continue
            fonts = set(c["font"] for c in cands)
            texts = []
            for fix in (False, True):
                cs = [dict(c) for c in ln["chars"]]
                if fix:
                    for c in cs:
                        if c["font"] in ratio and c["c"] in _ALEF_FORMS and (c["x1"] - c["x0"]) >= ratio[c["font"]] * c["size"]:
                            c["c"] = "لا"
                objs = build_line_objects({"chars": cs}, True, None, defaultdict(int))
                texts.append(F.normalize(" ".join(o["text_raw"] for o in objs)))
            wa = sum(1 for w in F.fa_tokens(texts[0]) if w in LAMALEF_LEXICON)
            wb = sum(1 for w in F.fa_tokens(texts[1]) if w in LAMALEF_LEXICON)
            for f in fonts:
                score[f][0] += wa
                score[f][1] += wb
                score[f][2] += len([c for c in cands if c["font"] == f])
    enabled = {}
    for font, (wa, wb, n) in score.items():
        if wb - wa >= 3 and wb >= 2 * max(wa, 1):
            enabled[font] = ratio[font]
    return enabled, {k: {"hits_without": v[0], "hits_with": v[1], "wide_alefs": v[2], "ratio": round(ratio.get(k, 0), 3)} for k, v in score.items()}


# --------------------------------------------------------------------------
# line assembly
# --------------------------------------------------------------------------

def merge_baseline_lines(lines):
    """Merge PyMuPDF lines sharing a baseline and touching horizontally."""
    items = []
    for ln in lines:
        cs = ln["chars"]
        size = statistics.median([c["size"] for c in cs])
        oy = statistics.median([c["oy"] for c in cs])
        items.append({"chars": list(cs), "size": size, "oy": oy,
                      "x0": min(c["x0"] for c in cs), "x1": max(c["x1"] for c in cs),
                      "y0": min(c["y0"] for c in cs), "y1": max(c["y1"] for c in cs)})
    items.sort(key=lambda it: (round(it["oy"]), it["x0"]))
    merged = True
    while merged:
        merged = False
        out = []
        items.sort(key=lambda it: (it["oy"], it["x0"]))
        used = [False] * len(items)
        for i, a in enumerate(items):
            if used[i]:
                continue
            for j in range(i + 1, len(items)):
                if used[j]:
                    continue
                b = items[j]
                tol = 0.3 * min(a["size"], b["size"])
                if b["oy"] - a["oy"] > tol:
                    break
                if abs(a["oy"] - b["oy"]) <= tol:
                    gap = max(b["x0"] - a["x1"], a["x0"] - b["x1"])
                    short = len(b["chars"]) <= 2 and (len(a["chars"]) <= 2 or len(a["chars"]) <= 8 and all(F.char_class(c["c"]) in ("D", "S") for c in a["chars"]))
                    if gap < 1.0 * max(a["size"], b["size"]) or (short and gap < 2.5 * max(a["size"], b["size"])):
                        a["chars"].extend(b["chars"])
                        a["x0"] = min(a["x0"], b["x0"])
                        a["x1"] = max(a["x1"], b["x1"])
                        a["y0"] = min(a["y0"], b["y0"])
                        a["y1"] = max(a["y1"], b["y1"])
                        a["size"] = max(a["size"], b["size"]) if len(b["chars"]) > len(a["chars"]) else a["size"]
                        used[j] = True
                        merged = True
            out.append(a)
        items = out
    for it in items:
        it["chars"].sort(key=lambda c: c["seq"])
    return items


def _blobify(chars):
    """Runs of >=3 consecutive stream chars sharing one x position carry no geometry
    (ActualText replacements, broken Type3 fonts). Give them synthetic increasing
    x positions in stream order and mark them, so they stay together in stream order."""
    n = len(chars)
    i = 0
    blobs = 0
    while i < n:
        j = i
        while j + 1 < n and abs(chars[j + 1]["x0"] - chars[i]["x0"]) < 0.3 and abs(chars[j + 1]["x1"] - chars[i]["x1"]) < 0.3:
            j += 1
        # stacked tatweels (kashida stretching) and marks legitimately share one x
        letters = sum(1 for k in range(i, j + 1) if chars[k]["c"] not in ("ـ", " ")
                      and F.char_class(chars[k]["c"]) in ("R", "L", "D"))
        if j - i + 1 >= 3 and letters >= 3:
            blobs += 1
            base = chars[i]["x0"]
            for k in range(i, j + 1):
                c = chars[k]
                c["blob"] = blobs
                c["x0"] = base + (k - i) * 0.001
                c["x1"] = c["x0"]
            i = j + 1
        else:
            # one glyph carrying a whole phrase (replacement text): no inner geometry
            s = chars[i]["c"]
            if len(s) >= 4 and (any(ch.isspace() for ch in s) or len(s) >= 6):
                blobs += 1
                chars[i]["blob"] = blobs
            i += 1
    return blobs


def _attach_stray_marks(toks, stats):
    """A token made only of combining marks (a mark whose stream neighbours are not
    letters, or one painted on a raised baseline) belongs to the letter it is drawn
    on: attach it to the Arabic token whose extent contains it, else drop it."""
    stray = [t for t in toks if t.kind == "N" and t.chars and all(F.char_class(c["c"]) == "M" for c in t.chars)]
    if not stray:
        return toks
    rtoks = [t for t in toks if t.kind == "R"]
    for st in stray:
        cx = (st.x0 + st.x1) / 2.0
        home = None
        for t in rtoks:
            if t.x0 <= cx <= t.x1:
                home = t
                break
        if home is not None:
            home.chars.extend(st.chars)
            F._finish_token(home)
            stats["stray_marks_attached"] += len(st.chars)
        else:
            stats["stray_marks_dropped"] += len(st.chars)
    ids = set(id(t) for t in stray)
    return [t for t in toks if id(t) not in ids]


def _check_lamalef_words(toks, stats):
    """The lam-alef repair is decided per font; per word it can be wrong where a wide
    alef is just a wide alef ('تعدلاد'). Undo it in a word when the word without
    the lam is a lexicon word and the word with it is not a known word."""
    lex = _lexicon()
    vocab = _DOC["vocab"]
    for t in toks:
        if t.kind != "R":
            continue
        restored = [c for c in t.chars if c.get("lamalef")]
        if not restored:
            continue
        w_with = F.normalize(t.text).replace("\u200c", "")
        for c in restored:
            c["c"] = "\u0627"
        F._finish_token(t)
        w_without = F.normalize(t.text).replace("\u200c", "")
        if w_without in lex and not (w_with in lex or w_with in vocab or w_with in LAMALEF_LEXICON):
            for c in restored:
                c["lamalef"] = False
            stats["lamalef_reverted"] += len(restored)
        else:
            for c in restored:
                c["c"] = "\u0644\u0627"
            F._finish_token(t)


def build_line_objects(item, default_rtl, lamalef_fonts, stats, split_factor=3.0):
    chars = [c for c in item["chars"] if not c.get("garbage")]
    if not chars:
        return []
    if lamalef_fonts:
        for c in chars:
            r = lamalef_fonts.get(c["font"])
            if r and c["c"] in _ALEF_FORMS and (c["x1"] - c["x0"]) >= r * c["size"]:
                c["c"] = "\u0644\u0627"
                c["lamalef"] = True
                stats["lamalef_restored"] += 1
    _blobify(chars)
    # a line with Arabic words but (almost) no space glyphs: words are separate text
    # objects, and a visible gap between runs is the only word boundary (Dezhino 1403)
    n_space = sum(1 for c in chars if c["c"].isspace())
    n_r = sum(1 for c in chars if F.char_class(c["c"]) == "R")
    spaceless = n_r >= 8 and n_space <= 0.08 * n_r
    for c in chars:
        c["spaceless"] = spaceless
    toks = F.tokenize(chars)
    toks = [t for t in toks if t.kind != "U"]
    toks = _attach_stray_marks(toks, stats)
    if any(c.get("lamalef") for c in chars):
        _check_lamalef_words(toks, stats)
    if not toks:
        return []
    vis = F.merge_letterspaced_year(F.visual_tokens(toks, False))
    groups = F.split_visual_on_gaps(vis, split_factor)
    out = []
    for g in groups:
        gchars = [c for t in g if t.kind != "S" or not t.synthetic for c in t.chars]
        if not gchars:
            continue
        rtl = F.line_direction(gchars, default_rtl)
        blob_toks = [t for t in g if any(c.get("blob") for c in t.chars)]
        if blob_toks and len(blob_toks) >= 0.5 * len([t for t in g if t.kind != "S"]):
            # geometry-less text: keep stream order; the block-level word test decides
            logical = sorted([t for t in g if not t.synthetic], key=lambda t: t.chars[0]["seq"])
        else:
            logical = F.logical_from_visual(g, rtl)
            stats["brackets_mirrored"] += F.fix_brackets(logical, bracket_coding)
        text = F.tokens_text(logical)
        if not text:
            continue
        real = [t for t in logical if t.kind != "S"]
        seqs = [t.chars[0]["seq"] for t in real]
        reordered = any(b < a for a, b in zip(seqs, seqs[1:]))
        nvis = sum(1 for t in real if t.visual)
        sizes = [c["size"] for c in gchars if not c["c"].isspace()] or [c["size"] for c in gchars]
        vis_chars = [c for c in gchars if not c["c"].isspace()]
        bold_share = (sum(1 for c in vis_chars if c["bold"]) / len(vis_chars)) if vis_chars else 0.0
        fonts = Counter(c["font"] for c in vis_chars)
        ys0 = [c["y0"] for c in gchars]
        ys1 = [c["y1"] for c in gchars]
        real_x = [c for c in gchars if not c.get("blob")] or gchars
        bbox = [min(c["x0"] for c in real_x), min(ys0), max(max(c["x1"] for c in real_x), min(c["x0"] for c in real_x)), max(ys1)]
        out.append({
            "text_raw": text, "tokens": logical, "rtl": rtl, "bbox": bbox,
            "size": statistics.median(sizes), "bold_share": bold_share,
            "font": fonts.most_common(1)[0][0] if fonts else "",
            "oy": statistics.median([c["oy"] for c in gchars]),
            "reordered": reordered, "visual_tokens": nvis, "blob": bool(blob_toks),
            "presform": sum(1 for c in gchars for ch in c["c"] if F.is_presentation_form(ch)),
            "nchars": len(vis_chars),
        })
        if reordered:
            stats["lines_reordered"] += 1
        stats["visual_tokens"] += nvis
    return out


def text_from_chars(chars, default_rtl):
    """Text of an arbitrary set of chars (a table cell): group by baseline, order each."""
    if not chars:
        return ""
    chars = sorted(chars, key=lambda c: c["seq"])
    rows = []
    for c in chars:
        placed = False
        for r in rows:
            if abs(r["oy"] - c["oy"]) <= 0.45 * max(r["size"], c["size"]):
                r["chars"].append(c)
                placed = True
                break
        if not placed:
            rows.append({"oy": c["oy"], "size": c["size"], "chars": [c]})
    rows.sort(key=lambda r: r["oy"])
    parts = []
    dummy = defaultdict(int)
    for r in rows:
        item = {"chars": r["chars"]}
        for lo in build_line_objects(item, default_rtl, None, dummy):
            parts.append(lo["text_raw"])
    return " ".join(p for p in parts if p)


# --------------------------------------------------------------------------
# block assembly
# --------------------------------------------------------------------------

def first_char(text):
    t = text.strip()
    return t[0] if t else ""


def is_bullet_line(text):
    fc = first_char(text)
    if fc in F.BULLETS or ("" <= fc <= ""):
        return True
    if fc in "-–—*" and len(text.strip()) > 2 and text.strip()[1] == " ":
        return True
    return False


def group_blocks(lines):
    lines = sorted(lines, key=lambda l: (l["bbox"][1], -l["bbox"][2] if l["rtl"] else l["bbox"][0]))
    blocks = []
    for ln in lines:
        best = None
        best_gap = None
        for blk in blocks:
            last = blk["lines"][-1]
            size = max(ln["size"], last["size"])
            # baseline-to-baseline distance: Persian is set with generous leading
            # (1.6-2x the font size), so a bbox gap test would split every line
            delta = ln["oy"] - last["oy"]
            gap = delta
            if delta < 0.5 * min(ln["size"], last["size"]):
                continue
            limit = (2.15 if (ln["rtl"] and last["rtl"]) else 1.75) * size
            if delta > limit:
                continue
            if len(blk["lines"]) >= 2:
                lead = statistics.median([b["oy"] - a["oy"] for a, b in zip(blk["lines"], blk["lines"][1:])])
                if delta > 1.3 * lead + 0.1 * size:
                    continue
            if abs(ln["size"] - last["size"]) > 0.2 * size:
                continue
            ov = x_overlap(ln["bbox"], blk["bbox"])
            if ov < 0.25 * min(ln["bbox"][2] - ln["bbox"][0], blk["bbox"][2] - blk["bbox"][0]):
                continue
            if (ln["bold_share"] > 0.9) != (last["bold_share"] > 0.9) and min(len(ln["text_raw"]), len(last["text_raw"])) > 0:
                if (ln["bold_share"] > 0.9 and ln["nchars"] < 80) or (last["bold_share"] > 0.9 and last["nchars"] < 80):
                    continue
            if is_bullet_line(ln["text_raw"]):
                continue
            if best is None or gap < best_gap:
                best, best_gap = blk, gap
        if best is None:
            blocks.append({"lines": [ln], "bbox": list(ln["bbox"])})
        else:
            best["lines"].append(ln)
            best["bbox"] = union([best["bbox"], ln["bbox"]])
    return blocks


def join_lines(texts, norm=False):
    out = ""
    for t in texts:
        if not out:
            out = t
            continue
        if norm and re.search(r"[A-Za-z]-$", out) and re.match(r"[a-z]", t):
            out = out[:-1] + t
        else:
            out = out + " " + t
    return out


# --------------------------------------------------------------------------
# block-level order fix (common-word test)
# --------------------------------------------------------------------------

def rotated_line_objects(ro, page_rtl, stats):
    """Rotated text (chart axes, vertical labels): project glyphs onto the writing
    direction and run the same geometric ordering as for horizontal lines."""
    dx, dy = ro["dir"]
    norm = math.hypot(dx, dy) or 1.0
    dx, dy = dx / norm, dy / norm
    cs = []
    for c in ro["chars"]:
        us = [x * dx + y * dy for x in (c["x0"], c["x1"]) for y in (c["y0"], c["y1"])]
        e = dict(c)
        e["x0"], e["x1"] = min(us), max(us)
        # a glyph box projected this way is wider than its advance; shrink toward centre
        mid = (e["x0"] + e["x1"]) / 2.0
        half = max(0.0, (e["x1"] - e["x0"]) / 2.0 - 0.5 * abs(dx * dy) * c["size"])
        e["x0"], e["x1"] = mid - half, mid + half
        e["oy"] = -c["ox"] * dy + c["oy"] * dx
        cs.append(e)
    objs = build_line_objects({"chars": cs}, page_rtl, None, stats)
    out = []
    for lo in objs:
        lo["bbox"] = list(ro["bbox"])
        lo["rotated"] = True
        lo["oy"] = ro["bbox"][3]
        out.append(lo)
    return out


def block_order_fix(blk, stats):
    """Common-word fallback. Only for lines whose glyphs carry no usable geometry
    (blob lines): where glyph positions exist, the geometric order already decided,
    and a word test on a few words is weaker evidence than the glyph positions."""
    rtl_lines = [ln for ln in blk["lines"] if ln["rtl"]]
    if not rtl_lines or not any(ln.get("blob") for ln in blk["lines"]):
        return None
    cur = F.normalize(" ".join(ln["text_raw"] for ln in blk["lines"]))
    f, r, fb, rb = F.order_hits(cur)
    if not ((r >= 2 and r > f) or (rb >= 1 and rb > fb)):
        return None
    cands = {}
    variants = {
        "reverse_chars": lambda seq: F.reverse_rtl_chars(seq),
        "reverse_words": lambda seq: F.reverse_word_order(seq),
        "reverse_both": lambda seq: F.reverse_word_order(F.reverse_rtl_chars(seq)),
    }
    base = F.order_score(cur)
    best_name, best_score, best_texts = None, base, None
    for name, fn in variants.items():
        texts = []
        for ln in blk["lines"]:
            if ln["rtl"] and ln.get("blob"):
                texts.append(F.tokens_text(fn(ln["tokens"])))
            else:
                texts.append(ln["text_raw"])
        sc = F.order_score(F.normalize(" ".join(texts)))
        cands[name] = sc
        if sc >= best_score + 2 and (best_name is None or sc > best_score):
            best_name, best_score, best_texts = name, sc, texts
    if best_name:
        for ln, t in zip(blk["lines"], best_texts):
            ln["text_raw"] = t
        stats["fallback_fixes"] += 1
        return {"fix": best_name, "score_before": base, "score_after": best_score}
    return None


# --------------------------------------------------------------------------
# reading order: recursive XY cut
# --------------------------------------------------------------------------

def _split(items, axis, min_gap):
    lo, hi = (0, 2) if axis == "x" else (1, 3)
    s = sorted(items, key=lambda it: it["bbox"][lo])
    groups = [[s[0]]]
    end = s[0]["bbox"][hi]
    gaps = []
    for it in s[1:]:
        if it["bbox"][lo] > end + min_gap:
            gaps.append(it["bbox"][lo] - end)
            groups.append([it])
        else:
            groups[-1].append(it)
        end = max(end, it["bbox"][hi])
    return groups, gaps


def _rows_aligned(groups):
    """True when items across vertical groups sit on shared rows (table/key-value)."""
    if len(groups) < 2:
        return False
    tops = [sorted(it["bbox"][1] for it in g) for g in groups]
    aligned = total = 0
    for gi, g in enumerate(groups):
        for it in g:
            total += 1
            y = it["bbox"][1]
            tol = 0.6 * max(it.get("size", 10.0), 4.0)
            if any(any(abs(y - t) <= tol for t in tops[gj]) for gj in range(len(groups)) if gj != gi):
                aligned += 1
    return total > 0 and aligned >= 0.6 * total and min(len(g) for g in groups) >= 2


def xy_order(items, rtl, depth=0):
    if len(items) <= 1 or depth > 60:
        return list(items)
    hgroups, hgaps = _split(items, "y", 1.0)
    vgroups, vgaps = _split(items, "x", 4.0)
    use = None
    if len(vgroups) > 1 and len(hgroups) > 1:
        if _rows_aligned(vgroups):
            use = "h"
        elif max(vgaps) >= 9.0:
            use = "v"
        else:
            use = "h"
    elif len(vgroups) > 1:
        use = "v"
    elif len(hgroups) > 1:
        use = "h"
    if use == "h":
        out = []
        for g in hgroups:
            out.extend(xy_order(g, rtl, depth + 1))
        return out
    if use == "v":
        vgroups.sort(key=lambda g: min(it["bbox"][0] for it in g), reverse=rtl)
        out = []
        for g in vgroups:
            out.extend(xy_order(g, rtl, depth + 1))
        return out
    return sorted(items, key=lambda it: (round(it["bbox"][1] / 3.0), -it["bbox"][2] if rtl else it["bbox"][0]))


# --------------------------------------------------------------------------
# tables
# --------------------------------------------------------------------------

MAX_TABLE_PATHS = 20000


def find_page_tables(page, page_area, paths=None):
    out = []
    if paths is not None and len(paths) > MAX_TABLE_PATHS:
        # photo-like vector art (60k+ paths): find_tables costs seconds per page there
        # and does not find tables in it
        return out, "find_tables skipped: %d vector paths on the page" % len(paths)
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            tf = page.find_tables(paths=paths) if paths is not None else page.find_tables()
    except Exception as e:  # noqa: BLE001
        return out, "find_tables error: %s" % e
    for t in tf.tables:
        bbox = list(t.bbox)
        if t.row_count < 2 or t.col_count < 2:
            continue
        if rect_area(bbox) > 0.97 * page_area:
            continue
        out.append(t)
    return out, None


def _baseline_clusters(chars):
    cl = []
    for c in sorted((c for c in chars if not c["c"].isspace()), key=lambda c: c["oy"]):
        if cl and c["oy"] - cl[-1]["oy"] <= 0.45 * max(c["size"], cl[-1]["size"]):
            cl[-1]["chars"].append(c)
        else:
            cl.append({"oy": c["oy"], "size": c["size"], "chars": [c]})
    return cl


def _baseline_count(chars):
    return len(_baseline_clusters(chars))


def _rows_from_baselines(rows_cells, group_of, remap, ncols, colx0, tbbox, rtl, default_rtl):
    """find_tables returns one row for a block of rows that has no horizontal rules
    (common in Word tables with alternating shading). Keep its column partition and
    rebuild rows from text baselines; leading rows without any digit are merged into
    one header row (wrapped header text)."""
    nfine = len(group_of)
    used_seqs = set()
    bounds = []
    for j in range(nfine):
        x0 = colx0[j] if colx0[j] is not None else tbbox[0]
        x1 = tbbox[2]
        for k in range(j + 1, nfine):
            if colx0[k] is not None:
                x1 = colx0[k]
                break
        bounds.append((x0, x1))
    allchars = []
    seen_ids = set()
    for cells in rows_cells:
        for c in cells:
            for ch in c["_chars"]:
                # overlapping find_tables cells can hold the same glyph twice
                if id(ch) not in seen_ids:
                    seen_ids.add(id(ch))
                    allchars.append(ch)
    if not any(not ch["c"].isspace() for ch in allchars):
        return [], set()
    granges = {}
    for j, (x0, x1) in enumerate(bounds):
        g = group_of[j]
        if g not in remap:
            continue
        k = remap[g]
        lo, hi = granges.get(k, (x0, x1))
        granges[k] = (min(lo, x0), max(hi, x1))
    table_rows = []
    taken = set()
    for cl in _baseline_clusters(allchars):
        lo_y = cl["oy"] - 0.45 * cl["size"]
        hi_y = cl["oy"] + 0.45 * cl["size"]
        members = [ch for ch in cl["chars"] if id(ch) not in taken] + \
            [ch for ch in allchars if ch["c"].isspace() and id(ch) not in taken and lo_y <= ch["oy"] <= hi_y]
        taken.update(id(ch) for ch in members)
        # phrases: split a baseline where glyphs are far apart (cell padding)
        phrases = build_line_objects({"chars": [dict(ch) for ch in sorted(members, key=lambda c: c["seq"])]},
                                     default_rtl, None, defaultdict(int), split_factor=0.9)
        row = [None] * ncols
        for ph in phrases:
            px0, px1 = ph["bbox"][0], ph["bbox"][2]
            cx = (px0 + px1) / 2.0
            home = None
            for k, (x0, x1) in granges.items():
                if x0 - 0.5 <= cx < x1 + 0.5:
                    home = k
                    break
            if home is None:
                continue
            used_seqs.update(c["seq"] for t in ph["tokens"] for c in t.chars)
            covered = [k for k, (x0, x1) in granges.items() if px0 < (x0 + x1) / 2.0 < px1]
            span_lo = min(covered + [home])
            span_hi = max(covered + [home])
            pos = (ncols - 1 - span_hi) if rtl else span_lo
            cell = {"text_raw": ph["text_raw"], "text_norm": F.normalize(ph["text_raw"]), "bbox": r2(ph["bbox"]),
                    "colspan": span_hi - span_lo + 1, "_chars": []}
            if row[pos] is None:
                row[pos] = cell
            else:
                # RTL phrases in one cell come right-to-left; LTR left-to-right
                first, second = (row[pos], cell)
                if (rtl and cell["bbox"][0] > row[pos]["bbox"][0]) or (not rtl and cell["bbox"][0] < row[pos]["bbox"][0]):
                    first, second = cell, row[pos]
                row[pos] = {"text_raw": first["text_raw"] + " " + second["text_raw"],
                            "text_norm": (first["text_norm"] + " " + second["text_norm"]).strip(),
                            "bbox": [min(first["bbox"][0], second["bbox"][0]), min(first["bbox"][1], second["bbox"][1]),
                                     max(first["bbox"][2], second["bbox"][2]), max(first["bbox"][3], second["bbox"][3])],
                            "colspan": max(first["colspan"], second["colspan"]), "_chars": []}
        if any(c is not None for c in row):
            table_rows.append(row)
    # merge leading digit-free rows (a wrapped header) into one
    def numeric_row(r):
        vals = [c["text_norm"] for c in r if c is not None and c["text_norm"]]
        return bool(vals) and sum(1 for v in vals if re.fullmatch(r"[\s\d.,/%()\-+]+", v)) >= 0.5 * len(vals)
    head = 0
    while head < len(table_rows) and not numeric_row(table_rows[head]):
        head += 1
    if head >= 2 and head < len(table_rows):
        merged = [None] * ncols
        for r in table_rows[:head]:
            for k, c in enumerate(r):
                if c is None:
                    continue
                if merged[k] is None:
                    merged[k] = dict(c)
                else:
                    merged[k]["text_raw"] += " " + c["text_raw"]
                    merged[k]["text_norm"] = (merged[k]["text_norm"] + " " + c["text_norm"]).strip()
                    merged[k]["_chars"] = merged[k]["_chars"] + c["_chars"]
                    b0, b1 = merged[k]["bbox"], c["bbox"]
                    merged[k]["bbox"] = [min(b0[0], b1[0]), min(b0[1], b1[1]), max(b0[2], b1[2]), max(b0[3], b1[3])]
        table_rows = [merged] + table_rows[head:]
    return table_rows, used_seqs


def build_table(t, chars, default_rtl):
    """Cells from page.find_tables(), re-gridded.

    find_tables often returns a grid finer than the visible table (vertical rules in
    Word/InDesign tables rarely line up), so a value cell spans several grid columns
    while its header sits in another. Columns are merged into the coarsest partition
    in which no row has two different text cells on either side of a boundary; empty
    columns are dropped. Text is re-extracted from the glyphs inside each cell and
    ordered by geometry. RTL tables are written in logical order (first column =
    rightmost)."""
    grid = [list(r.cells) for r in t.rows]
    nfine = max((len(r) for r in grid), default=0)
    for r in grid:
        while len(r) < nfine:
            r.append(None)
    colx0 = []
    for j in range(nfine):
        xs = [r[j][0] for r in grid if r[j] is not None]
        colx0.append(statistics.median(xs) if xs else None)
    rows_cells = []
    missing = 0
    tb_box = list(t.bbox)
    assigned = set()
    for r in grid:
        cells = []
        cur = None
        for j, cb in enumerate(r):
            if cb is not None:
                # a glyph whose centre is a hair outside the table's outer rule still
                # belongs to the edge cell ('ن' of «نام» at x 521.3 vs rule at 520.7)
                ecb = [cb[0] - (1.5 if abs(cb[0] - tb_box[0]) < 1.0 else 0.0), cb[1],
                       cb[2] + (1.5 if abs(cb[2] - tb_box[2]) < 1.0 else 0.0), cb[3]]
                cs = [c for c in chars if id(c) not in assigned
                      and inside((c["x0"] + c["x1"]) / 2.0, (c["y0"] + c["y1"]) / 2.0, ecb)]
                assigned.update(id(c) for c in cs)
                raw = text_from_chars(cs, default_rtl)
                cur = {"start": j, "end": j, "bbox": cb, "text_raw": raw, "text_norm": F.normalize(raw), "_chars": cs}
                cells.append(cur)
            else:
                missing += 1
                if cur is not None and colx0[j] is not None and cur["bbox"][2] > colx0[j] + 1.0:
                    cur["end"] = j
                else:
                    cur = None
        rows_cells.append([c for c in cells if c["text_norm"]])
    # a cell boundary that cuts through a word means the "table" is a layout grid
    # drawn over running text (or a chart), not a table
    word_cuts = 0
    for cells in rows_cells:
        for a, b in zip(cells, cells[1:]):
            ca = [c for c in a["_chars"] if not c["c"].isspace()]
            cb_ = [c for c in b["_chars"] if not c["c"].isspace()]
            if not ca or not cb_ or b["bbox"][0] - a["bbox"][2] > 1.0:
                continue
            # per baseline: real cells keep padding between their texts; a layout grid
            # laid over flowing text leaves glyphs of one word touching across the rule
            cut = False
            for la in ca:
                if cut:
                    break
                size = max(la["size"], 1.0)
                if la["x1"] < a["bbox"][2] - 1.5 * size:
                    continue
                for fb in cb_:
                    if abs(la["oy"] - fb["oy"]) < 0.3 * size and fb["x0"] <= b["bbox"][0] + 1.5 * size \
                            and -0.3 * size < (fb["x0"] - la["x1"]) < 0.2 * size:
                        cut = True
                        break
            if cut:
                word_cuts += 1
    forbid = [False] * max(nfine - 1, 0)
    for cells in rows_cells:
        cover = {}
        for k, c in enumerate(cells):
            for j in range(c["start"], c["end"] + 1):
                cover[j] = k
        for j in range(nfine - 1):
            if j in cover and (j + 1) in cover and cover[j] != cover[j + 1]:
                forbid[j] = True
    group_of = []
    g = 0
    for j in range(nfine):
        if j > 0 and forbid[j - 1]:
            g += 1
        group_of.append(g)
    used = sorted(set(group_of[j] for cells in rows_cells for c in cells for j in range(c["start"], c["end"] + 1)))
    remap = {gid: k for k, gid in enumerate(used)}
    ncols = len(used)
    allc = [ch for cells in rows_cells for c in cells for ch in c["_chars"]]
    rr = sum(1 for c in allc if F.char_class(c["c"]) == "R")
    ll = sum(1 for c in allc if F.char_class(c["c"]) == "L")
    rtl = rr > ll or (rr == ll and default_rtl)
    rows = []
    rerowed = False
    under = any(sum(1 for c in cells if _baseline_count(c["_chars"]) >= 4) >= 2 for cells in rows_cells)
    used_seqs = set()
    if under and ncols >= 2:
        rows, used_seqs = _rows_from_baselines(rows_cells, group_of, remap, ncols, colx0, t.bbox, rtl, default_rtl)
        rerowed = bool(rows)
    for cells in ([] if rerowed else rows_cells):
        row = [None] * ncols
        for c in cells:
            gs = remap[group_of[c["start"]]]
            ge = remap[group_of[c["end"]]]
            pos = (ncols - 1 - ge) if rtl else gs
            cell = {"text_raw": c["text_raw"], "text_norm": c["text_norm"], "bbox": r2(c["bbox"]),
                    "colspan": ge - gs + 1, "_chars": c["_chars"]}
            used_seqs.update(ch["seq"] for ch in c["_chars"])
            if row[pos] is None:
                row[pos] = cell
            else:
                # two find_tables cells in one output cell: right-to-left order in RTL tables
                first, second = row[pos], cell
                if (rtl and cell["bbox"][0] > row[pos]["bbox"][0]) or (not rtl and cell["bbox"][0] < row[pos]["bbox"][0]):
                    first, second = cell, row[pos]
                row[pos] = {"text_raw": first["text_raw"] + " " + second["text_raw"],
                            "text_norm": (first["text_norm"] + " " + second["text_norm"]).strip(),
                            "bbox": [min(first["bbox"][0], second["bbox"][0]), min(first["bbox"][1], second["bbox"][1]),
                                     max(first["bbox"][2], second["bbox"][2]), max(first["bbox"][3], second["bbox"][3])],
                            "colspan": max(first["colspan"], second["colspan"]),
                            "_chars": first["_chars"] + second["_chars"]}
        rows.append(row)
    # drop rows with no text at all
    rows = [r for r in rows if any(c is not None for c in r)]
    cells_total = sum(len(r) for r in rows)
    nonempty = sum(1 for r in rows for c in r if c is not None and c["text_norm"])
    header_guess = None
    if rows and len(rows) >= 2:
        def numeric_share(row):
            vals = [c["text_norm"] for c in row if c is not None and c["text_norm"]]
            if not vals:
                return 0.0
            return sum(1 for v in vals if re.fullmatch(r"[\s\d.,%()\-+/]+", v)) / len(vals)
        first = numeric_share(rows[0])
        rest = [numeric_share(r) for r in rows[1:]]
        rest_avg = sum(rest) / len(rest) if rest else 0.0
        bold0 = [ch for cell in rows[0] if cell is not None for ch in cell["_chars"]]
        bold_share0 = (sum(1 for c in bold0 if c["bold"]) / len(bold0)) if bold0 else 0.0
        if first < 0.5 and (rest_avg > first + 0.2 or bold_share0 > 0.6):
            header_guess = 0
    for r in rows:
        for c in r:
            if c is not None:
                c.pop("_chars", None)
    avg_len = (sum(len(c["text_norm"]) for r in rows for c in r if c is not None) / max(nonempty, 1))
    lens = [len(c["text_norm"]) for r in rows for c in r if c is not None and c["text_norm"]]
    top_share = (max(lens) / sum(lens)) if lens else 0.0
    return {
        "bbox": r2(t.bbox), "n_rows": len(rows), "n_cols": ncols, "grid_cols_find_tables": nfine,
        "direction": "rtl" if rtl else "ltr", "header_row_guess": header_guess,
        "merged_or_missing_cells": missing, "cells_total": cells_total, "cells_nonempty": nonempty,
        "avg_cell_chars": round(avg_len, 1), "rows": rows, "word_cuts": word_cuts,
        "rows_from_text_baselines": rerowed, "top_cell_share": round(top_share, 3), "text_chars": sum(lens),
        "_used_seqs": used_seqs,
        "method": "pymupdf.find_tables(strategy=lines); grid columns merged to the coarsest consistent partition; cell text re-ordered by glyph geometry",
    }


def table_is_plausible(tb, page_area=None):
    # one cell holding most of the text: a chart or map cut into a grid
    if tb.get("text_chars", 0) >= 150 and tb.get("top_cell_share", 0) >= 0.6:
        return False
    # a frame around the whole page (running head, page number and a caption as "cells")
    if page_area and rect_area(tb["bbox"]) >= 0.7 * page_area and tb["n_rows"] <= 5:
        return False
    # chart value labels such as "9 (34.6%)" cut across cells by bar-chart gridlines
    cells = [c["text_norm"] for r in tb["rows"] for c in r if c is not None and c["text_norm"]]
    cut_labels = [c for c in cells if c.count("(") != c.count(")") and re.search(r"\(\d|\d%?\)", c)]
    if len(cut_labels) >= 2 and len(cut_labels) >= 0.2 * max(len(cells), 1):
        return False
    if tb["cells_nonempty"] < 4 or tb["n_rows"] < 2 or tb["n_cols"] < 2 or (tb["n_rows"] < 3 and tb["n_cols"] < 3):
        return False
    if tb["word_cuts"] >= 2 or (tb["word_cuts"] >= 1 and tb["n_rows"] <= 4):
        return False
    if tb["cells_nonempty"] < 0.3 * max(tb["cells_total"], 1):
        return False
    if tb["n_rows"] * tb["n_cols"] <= 6 and tb["avg_cell_chars"] > 150:
        return False
    if tb["avg_cell_chars"] > 400:
        return False
    return True


# --------------------------------------------------------------------------
# images and drawings
# --------------------------------------------------------------------------

def save_image(doc, xref, smask_hint=None):
    """Return (bytes, ext) for an embedded image, png unless a plain RGB/gray JPEG."""
    info = doc.extract_image(xref)
    if not info:
        return None, None
    ext = (info.get("ext") or "").lower()
    smask = info.get("smask") or 0
    cs_n = info.get("colorspace") or 0
    if ext in ("jpeg", "jpg") and not smask and cs_n in (1, 3):
        return info["image"], "jpg"
    pix = fitz.Pixmap(doc, xref)
    if smask:
        try:
            mask = fitz.Pixmap(doc, smask)
            pix = fitz.Pixmap(pix, mask)
        except Exception:  # noqa: BLE001
            pass
    if pix.colorspace is not None and pix.colorspace.n not in (1, 3):
        pix = fitz.Pixmap(fitz.csRGB, pix)
    elif pix.colorspace is None and not pix.alpha:
        pass
    try:
        data = pix.tobytes("png")
    except Exception:  # noqa: BLE001
        pix = fitz.Pixmap(fitz.csRGB, pix)
        data = pix.tobytes("png")
    return data, "png"


def drawing_stats(page, page_area, table_boxes, paths=None):
    if paths is None:
        try:
            paths = page.get_drawings()
        except Exception:  # noqa: BLE001
            return {"count": None, "has_vector_chart": False, "error": True}
    n = len(paths)
    fills = strokes = curves = 0
    for p in paths:
        r = p.get("rect")
        if r is None:
            continue
        area = abs(r.width * r.height)
        cx, cy = (r.x0 + r.x1) / 2.0, (r.y0 + r.y1) / 2.0
        if any(inside(cx, cy, tb) for tb in table_boxes):
            continue
        items = p.get("items", [])
        kinds = set(it[0] for it in items)
        if p.get("fill") is not None and 1.0 < area < 0.25 * page_area:
            fills += 1
        if p.get("color") is not None and kinds <= {"l"}:
            strokes += 1
        if "c" in kinds:
            curves += 1
    return {"count": n, "filled_shapes": fills, "stroked_lines": strokes, "curved_paths": curves}


# --------------------------------------------------------------------------
# the document
# --------------------------------------------------------------------------

_SQUASH = re.compile("[\\s\u0640\u200c]+")


def trim_overlapping_phrases(lines):
    """Canva ActualText phrases can hold their own line plus the beginning (or all) of
    the next line ("...رمزارزها \\nکاربران تترلند و بازار تت"), and the next phrase repeats
    that line. Where the end of one phrase is the start of another phrase on the page
    (>= 8 characters, ignoring spaces and tatweel), cut the overlap off the first."""
    phrases = [c for ln in lines for c in ln["chars"] if len(c["c"]) >= 6]
    if len(phrases) < 2:
        return 0
    keys = [_SQUASH.sub("", c["c"]) for c in phrases]
    n = 0
    for i, c in enumerate(phrases):
        ki = keys[i]
        cut = None
        for j, kj in enumerate(keys):
            if j == i or len(kj) < 8 or kj == ki:
                continue
            p = ki.find(kj[:8], 1)
            while p >= 0:
                if kj.startswith(ki[p:]) or ki[p:].startswith(kj):
                    if cut is None or p < cut:
                        cut = p
                    break
                p = ki.find(kj[:8], p + 1)
        if cut is None:
            continue
        raw = c["c"]
        idx = 0
        seen = 0
        while idx < len(raw) and seen < cut:
            if not _SQUASH.fullmatch(raw[idx]):
                seen += 1
            idx += 1
        c["c"] = raw[:idx].rstrip()
        n += 1
    return n


def extract_pdf(path, page_range, lang, images_dir_rel="images"):
    doc = fitz.open(path)
    n_pages = doc.page_count
    start, end = 1, n_pages
    if page_range:
        start, end = max(1, page_range[0]), min(n_pages, page_range[1])
    pnos = list(range(start - 1, end))
    default_rtl = lang != "en"
    for k in ("mark_gids", "kashida_gids", "vocab"):
        _DOC[k] = set()
    _DOC["bracket_coding"] = {}
    share, counted = document_space_share(doc, pnos)
    # English documents only: three Persian study reports also write under 1% space glyphs (their word breaks come
    # through the ZWNJ / ActualText paths above), and their extraction must not move
    _DOC["gap_spaces"] = GAP_SPACES_ENABLED and lang == "en" and counted >= 500 and share < GAP_SPACE_DOC_SHARE
    _DOC["gap_spaces_inserted"] = 0
    _DOC["space_share"] = (round(share, 4), counted)
    gid_recovery, gid_report = build_gid_recovery(doc, pnos)
    garbage_fonts = gid_report.pop("_garbage_fonts", {})
    actualtext_fonts = gid_report.pop("_actualtext_fonts", {})
    _DOC["mark_gids"] = gid_report.pop("_mark_gids", set())
    _DOC["kashida_gids"] = gid_report.pop("_kashida_gids", set())
    _DOC["bracket_coding"] = gid_report.pop("_bracket_coding", {})
    _DOC["vocab"] = gid_report.pop("_vocab", set())
    lamalef_fonts, lamalef_scores = detect_lamalef_fonts(doc, pnos, gid_recovery)
    narrow_fonts, narrow_scores = detect_spurious_narrow_fonts(doc, pnos, gid_recovery)

    _DOC["gap_spaces_inserted"] = 0      # count the final page pass only (font detection above also collects lines)
    pages = []
    image_store = {}      # digest -> filename
    xref_store = {}       # xref -> filename
    image_files = {}      # filename -> bytes
    table_files = {}      # stem -> table dict
    img_counter = 0
    for pno in pnos:
        page = doc[pno]
        W, H = page.rect.width, page.rect.height
        page_area = W * H
        stats = defaultdict(int)
        # ---- tables first so their chars leave the text flow
        try:
            paths = page.get_drawings()
        except Exception:  # noqa: BLE001
            paths = None
        raw_tables, terr = find_page_tables(page, page_area, paths)
        lines = collect_lines(page, gid_recovery)
        if actualtext_fonts:
            stats["actualtext_overlaps_trimmed"] += trim_overlapping_phrases(lines)
        for ln in lines:
            # fonts whose glyph codes are symbol garbage with no replacement text
            if garbage_fonts:
                for c in ln["chars"]:
                    if c["font"] in garbage_fonts and not c.get("garbage") and not (garbage_fonts[c["font"]].get("rule") and F.char_class(c["c"]) == "D"):
                        c["garbage"] = True
                        stats["garbage_font_chars_dropped"] += 1
            if actualtext_fonts:
                for c in ln["chars"]:
                    if c["font"] in actualtext_fonts and len(c["c"]) < 4 and not c["c"].isspace() \
                            and not c.get("garbage"):
                        c["garbage"] = True
                        c["covered"] = True
            # Canva-style exports: visible glyphs carry no mapping, a replacement
            # (ActualText) phrase on the same line carries the text
            covered = any(len(c["c"]) >= 4 and any("\u0600" <= ch <= "\u06ff" for ch in c["c"]) for c in ln["chars"])
            if covered:
                for c in ln["chars"]:
                    if "\ufffd" in c["c"] or F.char_class(c["c"]) == "U":
                        c["garbage"] = True
                        c["covered"] = True
            if narrow_fonts and ln["horizontal"]:
                idx = [i for i in narrow_spurious_candidates(ln["chars"]) if ln["chars"][i]["font"] in narrow_fonts]
                if idx:
                    drop = set(idx)
                    ln["chars"] = [c for k, c in enumerate(ln["chars"]) if k not in drop]
                    stats["spurious_narrow_glyphs_dropped"] += len(drop)
            stats["duplicate_glyphs_dropped"] += ln.get("duplicate_glyphs_dropped", 0)
            stats["garbage_chars"] += mark_garbage(ln["chars"])
            stats["glyphs_recovered_from_sibling_font"] += sum(1 for c in ln["chars"] if c.get("recovered"))
            stats["unmapped_glyphs"] += sum(1 for c in ln["chars"] if "\ufffd" in c["c"] and not c.get("covered"))
            stats["glyphs_replaced_by_actualtext"] += sum(1 for c in ln["chars"] if c.get("covered"))
            stats["invisible_chars"] += sum(1 for c in ln["chars"] if c["invisible"])
            if lamalef_fonts:
                for c in ln["chars"]:
                    r = lamalef_fonts.get(c["font"])
                    if r and c["c"] in _ALEF_FORMS and (c["x1"] - c["x0"]) >= r * c["size"]:
                        c["c"] = "\u0644\u0627"
                        c["lamalef"] = True
                        stats["lamalef_restored"] += 1
        all_chars = [c for ln in lines if ln["horizontal"] for c in ln["chars"] if not c.get("garbage")]
        tables = []
        claimed = set()
        for t in raw_tables:
            tb = build_table(t, all_chars, default_rtl)
            used = tb.pop("_used_seqs", set())
            if not table_is_plausible(tb, page_area):
                continue
            # only glyphs that ended up in a cell leave the text flow; text inside the
            # table box that no cell took (null merged cells, rows the baseline rebuild
            # could not place) stays in the page text instead of vanishing
            unplaced = 0
            tbx = tb["bbox"]
            for c in all_chars:
                if c["seq"] in used:
                    claimed.add(id(c))
                elif not c["c"].isspace() and inside((c["x0"] + c["x1"]) / 2.0, (c["y0"] + c["y1"]) / 2.0, tbx, 0.5):
                    unplaced += 1
            tb["glyphs_left_in_text_flow"] = unplaced
            stats["table_glyphs_left_in_text_flow"] += unplaced
            tables.append(tb)
        # ---- text lines
        horiz = []
        rotated = []
        for ln in lines:
            if ln["horizontal"]:
                cs = [c for c in ln["chars"] if id(c) not in claimed]
                if cs:
                    horiz.append({"chars": cs})
            else:
                cs = [c for c in ln["chars"] if not c.get("garbage")]
                if cs:
                    rotated.append({"chars": cs, "bbox": ln["bbox"], "dir": ln["dir"]})
        page_r = sum(1 for ln in lines for c in ln["chars"] if F.char_class(c["c"]) == "R")
        page_l = sum(1 for ln in lines for c in ln["chars"] if F.char_class(c["c"]) == "L")
        page_rtl = page_r > page_l or (page_r == page_l and default_rtl)
        merged = merge_baseline_lines(horiz)
        line_objs = []
        for it in merged:
            line_objs.extend(build_line_objects(it, page_rtl, None, stats))
        for ro in rotated:
            for lo in rotated_line_objects(ro, page_rtl, stats):
                line_objs.append(lo)
        blocks = []
        rot_lines = [lo for lo in line_objs if lo.get("rotated")]
        for blk in group_blocks([lo for lo in line_objs if not lo.get("rotated")]):
            blocks.append(blk)
        for lo in rot_lines:
            blocks.append({"lines": [lo], "bbox": list(lo["bbox"]), "rotated": True})
        for blk in blocks:
            fix = block_order_fix(blk, stats)
            if fix:
                blk["order_fix"] = fix
        # ---- images
        imgs = []
        try:
            infos = page.get_image_info(xrefs=True)
        except Exception:  # noqa: BLE001
            infos = []
        small = 0
        dict_imgs = None
        for inf in infos:
            w, h = inf.get("width", 0), inf.get("height", 0)
            xref = inf.get("xref", 0)
            bbox = list(inf.get("bbox", (0, 0, 0, 0)))
            if w < 150 or h < 150:
                small += 1
                continue
            vis_box = [max(0, bbox[0]), max(0, bbox[1]), min(W, bbox[2]), min(H, bbox[3])]
            if rect_area(vis_box) <= 1.0:
                small += 1
                continue
            fname = None
            err = None
            if xref and xref in xref_store:
                fname = xref_store[xref]
            elif xref:
                try:
                    data, ext = save_image(doc, xref)
                    if data:
                        dg = hashlib.sha1(data).hexdigest()
                        if dg in image_store:
                            fname = image_store[dg]
                        else:
                            img_counter += 1
                            n_on_page = sum(1 for k in image_files if k.startswith("p%03d_" % (pno + 1))) + 1
                            fname = "p%03d_i%d.%s" % (pno + 1, n_on_page, ext)
                            image_files[fname] = data
                            image_store[dg] = fname
                        xref_store[xref] = fname
                except Exception as e:  # noqa: BLE001
                    err = str(e)[:200]
            else:
                # no xref: an inline image, or one MuPDF cannot resolve to an object
                # (images inside form XObjects/patterns); the structured-text image
                # block at the same place carries the decoded image
                if dict_imgs is None:
                    try:
                        dict_imgs = [b for b in page.get_text("dict", flags=fitz.TEXT_PRESERVE_IMAGES)["blocks"]
                                     if b.get("type") == 1 and b.get("image")]
                    except Exception:  # noqa: BLE001
                        dict_imgs = []
                match = None
                for b in dict_imgs:
                    bb = b.get("bbox", (0, 0, 0, 0))
                    if b.get("width") == w and b.get("height") == h and all(abs(bb[k] - bbox[k]) <= 2.0 for k in range(4)):
                        match = b
                        break
                if match is None:
                    err = "inline image (no xref); no matching image block, not saved"
                else:
                    try:
                        data, ext = match["image"], (match.get("ext") or "").lower()
                        if ext == "jpeg":
                            ext = "jpg"
                        if ext not in ("jpg", "png"):
                            data, ext = fitz.Pixmap(data).tobytes("png"), "png"
                        dg = hashlib.sha1(data).hexdigest()
                        if dg in image_store:
                            fname = image_store[dg]
                        else:
                            n_on_page = sum(1 for k in image_files if k.startswith("p%03d_" % (pno + 1))) + 1
                            fname = "p%03d_i%d.%s" % (pno + 1, n_on_page, ext)
                            image_files[fname] = data
                            image_store[dg] = fname
                    except Exception as e:  # noqa: BLE001
                        err = "inline image: %s" % str(e)[:160]
            imgs.append({"file": (images_dir_rel + "/" + fname) if fname else None, "xref": xref,
                         "width_px": w, "height_px": h, "bbox": r2(bbox),
                         "area_share": round(rect_area(vis_box) / page_area, 4), "error": err})
        # ---- links, drawings
        links = []
        for lk in page.get_links():
            d = {"kind": lk.get("kind"), "bbox": r2(lk["from"]) if lk.get("from") is not None else None}
            if lk.get("uri"):
                d["uri"] = lk["uri"]
            if lk.get("page") is not None and lk.get("kind") in (fitz.LINK_GOTO, fitz.LINK_GOTOR):
                d["target_page"] = lk["page"] + 1 if lk["page"] >= 0 else None
            if lk.get("file"):
                d["file"] = lk["file"]
            links.append(d)
        dstats = drawing_stats(page, page_area, [tb["bbox"] for tb in tables], paths)
        blank = False
        if not any(not c["c"].isspace() for ln in lines for c in ln["chars"]):
            # a page with no text layer: is anything drawn on it at all?
            try:
                pix = page.get_pixmap(dpi=12, colorspace=fitz.csGRAY, alpha=False)
                s = pix.samples
                if len(s):
                    mode = Counter(s).most_common(1)[0][0]
                    blank = sum(1 for v in s if abs(v - mode) > 24) <= 0.001 * len(s)
            except Exception:  # noqa: BLE001
                blank = False
        pages.append({
            "pno": pno, "width": W, "height": H, "rotation": page.rotation, "rtl": page_rtl,
            "blocks": blocks, "tables": tables, "images": imgs, "small_images_skipped": small,
            "links": links, "drawings": dstats, "stats": dict(stats), "table_error": terr, "blank": blank,
            "plain_text": page.get_text(),
        })
    meta = dict(doc.metadata or {})
    toc = doc.get_toc(simple=True)
    doc.close()
    return {"n_pages": n_pages, "range": [start, end], "pages": pages, "metadata": meta, "toc": toc,
            "image_files": image_files, "lamalef_fonts": sorted(lamalef_fonts), "actualtext_fonts": actualtext_fonts, "gid_recovery_report": gid_report, "garbage_fonts": garbage_fonts, "narrow_fonts": sorted(narrow_fonts), "narrow_scores": narrow_scores,
            "lamalef_scores": lamalef_scores,
            "kashida_gids": sorted("%s gid %s" % k for k in _DOC["kashida_gids"]),
            "mark_gids": sorted("%s gid %s" % k for k in _DOC["mark_gids"]),
            "bracket_coding": {"|".join(str(x) for x in k): v for k, v in sorted(_DOC["bracket_coding"].items(), key=lambda kv: str(kv[0]))},
            "gap_spaces": {"applied": bool(_DOC.get("gap_spaces")), "inserted": _DOC.get("gap_spaces_inserted", 0),
                           "space_share": _DOC.get("space_share")}}
