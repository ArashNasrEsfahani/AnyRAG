"""Decisive spans and the span-retention metric (offline, no model calls).

A decisive span is the part of a gold evidence quote (question-v2 `evidence[].quote`, a verbatim slice of
Document.text at [start, end)) that carries the key value/phrase of the requirement(s) it supports:

    1. keys = the values and aliases of every requirement the evidence item supports (numbers also in their
       thousands-separated / decimal-slash forms; entity values also split at parentheses, ';' and ',');
    2. the quote is split into segments (sentence ends . ! ? ؟ ؛ ; followed by whitespace, and line breaks);
       the span is the minimal run of consecutive segments that holds every key that matches somewhere in the quote
       (method "key"); no key matches -> the whole quote (method "quote");
    3. a span shorter than MIN_SPAN folded chars grows to the whole quote ("+quote"); still short: the enclosing line
       of Document.text when it is long enough or is a table row holding the key ("+line"), else the whole unit
       ("+unit"), else neighbouring lines until MIN_SPAN (bare chart labels such as "40": "+window", weak).

Matching and containment use canon() = normalize.fold (NFKC, Persian/Arabic letters and digits unified, harakat
removed, ZWNJ -> space, whitespace collapsed, lower case) on BOTH sides.

Retention of a question for a list of shown evidence texts (each the text one item shows the reader):
    a span is present when canon(span) is contained in canon(one shown item)   (exact, per item, no stitching)
    bundle b (question-v2 evidence_bundles: alternative sufficient sets):
        span_frac(b) = present spans / spans of b
        req_frac(b)  = required requirements whose spans in b are all present / required requirements with spans in b
    the question's score is that of its best bundle (max span_frac, then req_frac).
"""
import re

from ...normalize import fold

SPANS_VERSION = "decisive-spans-v1"
MIN_SPAN = 20                       # folded chars

_SEG_END = re.compile(r"[.!?؟؛;](?=\s)|\n")


def canon(text):
    return fold(text or "")


# --------------------------------------------------------------------------------------------------------------
# keys
# --------------------------------------------------------------------------------------------------------------

def _num_forms(v):
    s = str(v).strip()
    try:
        x = float(s)
    except ValueError:
        return []
    out = set()
    if x == int(x):
        n = int(x)
        out |= {str(n), "{:,}".format(n)}
        if abs(n) >= 1000:
            out.add("{:,}".format(n).replace(",", " "))
            out.add("{:,}".format(n).replace(",", "."))
    else:
        t = ("%f" % x).rstrip("0").rstrip(".")
        out |= {t, t.replace(".", "/"), t.replace(".", ",")}
    out.add(s)
    return sorted(o for o in out if o and o != "-")


def keys_for(req):
    """-> [(kind, text)] key strings of one requirement (kind: num | text)."""
    keys = []
    v, vt = req.get("value"), (req.get("value_type") or "")
    if vt == "number" and v is not None:
        keys += [("num", f) for f in _num_forms(v)]
    elif vt == "boolean" or v is None:
        pass
    else:
        s = str(v).strip()
        if s:
            keys.append(("text", s))
            if vt in ("entity", "list"):
                for p in re.split(r"[();,]|\s->\s", s):
                    p = p.strip(" '\"[]")
                    if len(canon(p)) >= 3 and p != s:
                        keys.append(("text", p))
    for a in req.get("aliases") or []:
        a = str(a).strip()
        if not a:
            continue
        keys.append(("text", a))
        m = re.fullmatch(r"\s*(-?\d+(?:[.,/]\d+)?)\s*", a)
        if m:
            keys += [("num", f) for f in _num_forms(m.group(1).replace("/", "."))]
    seen, out = set(), []
    for k in keys:
        c = (k[0], canon(k[1]))
        if c[1] and c not in seen:
            seen.add(c)
            out.append((k[0], k[1]))
    return out


def _find_key(ctext, kind, key):
    """Position of key in canon text ctext (numbers with digit boundaries), or None."""
    ck = canon(key)
    if not ck:
        return None
    if kind == "num":
        m = re.search(r"(?<![\d.,/])" + re.escape(ck) + r"(?![\d]|[.,/]\d)", ctext)
        return (m.start(), m.end()) if m else None
    if re.fullmatch(r"[\w\s]+", ck) and len(ck) <= 3:            # short words: word boundaries
        m = re.search(r"(?<!\w)" + re.escape(ck) + r"(?!\w)", ctext)
        return (m.start(), m.end()) if m else None
    i = ctext.find(ck)
    return (i, i + len(ck)) if i >= 0 else None


# --------------------------------------------------------------------------------------------------------------
# segments
# --------------------------------------------------------------------------------------------------------------

def segments(text):
    """[(a, b)] segment offsets in text (sentence ends and line breaks), whitespace-stripped, non-empty."""
    out, a = [], 0
    for m in _SEG_END.finditer(text):
        b = m.end()
        out.append((a, b))
        a = b
    out.append((a, len(text)))
    res = []
    for a, b in out:
        while a < b and text[a].isspace():
            a += 1
        while b > a and text[b - 1].isspace():
            b -= 1
        if b > a:
            res.append((a, b))
    return res


def _line_around(text, a, b):
    lo = text.rfind("\n", 0, a) + 1
    hi = text.find("\n", b)
    hi = len(text) if hi < 0 else hi
    return lo, hi


# --------------------------------------------------------------------------------------------------------------
# decisive spans of one question
# --------------------------------------------------------------------------------------------------------------

def decisive_spans(q, corpus_lookup):
    """q: a question-v2 dict. corpus_lookup(doc_id) -> Document or None.
    -> record {id, pool_id, tier, language, domain, answerable, requirements, spans, bundles, unresolved}."""
    reqs = {r["id"]: r for r in q.get("requirements") or []}
    rec = {"id": q["id"], "pool_id": q.get("pool_id"), "tier": q.get("tier"), "language": q.get("language"),
           "domain": q.get("domain"), "answerable": bool(q.get("answerable")), "version": SPANS_VERSION,
           "requirements": [], "spans": [], "bundles": [], "unresolved": []}
    if not rec["answerable"]:
        return rec
    ev = {e["id"]: e for e in q.get("evidence") or []}
    for eid, e in ev.items():
        doc = corpus_lookup(e["doc_id"])
        quote = e.get("quote") or ""
        if doc is None:
            rec["unresolved"].append({"evidence": eid, "why": "document not in corpus"})
            continue
        s0, s1 = int(e.get("start", -1)), int(e.get("end", -1))
        if not (0 <= s0 < s1 <= len(doc.text)) or doc.text[s0:s1] != quote:
            i = doc.text.find(quote) if quote else -1
            if i < 0:
                rec["unresolved"].append({"evidence": eid, "why": "quote not found in Document.text"})
                continue
            s0, s1 = i, i + len(quote)
        sup = [r for r in e.get("supports") or [] if r in reqs]
        ckeys = []
        for rid in sup:
            for kind, k in keys_for(reqs[rid]):
                ckeys.append((rid, kind, k))
        segs = segments(quote)
        cseg = [canon(quote[a:b]) for a, b in segs]
        hit_segs, matched = set(), []
        for rid, kind, k in ckeys:
            for j, cs in enumerate(cseg):
                if _find_key(cs, kind, k):
                    hit_segs.add(j)
                    matched.append({"req": rid, "key": k, "segment": j})
                    break
        if hit_segs:
            lo, hi = min(hit_segs), max(hit_segs)
            a, b = segs[lo][0], segs[hi][1]
            method = "key"
        else:
            a, b = (segs[0][0], segs[-1][1]) if segs else (0, len(quote))
            method = "quote"
        a, b = s0 + a, s0 + b
        if len(canon(doc.text[a:b])) < MIN_SPAN:
            if len(canon(quote)) >= MIN_SPAN and (a, b) != (s0, s1):
                a, b, method = s0, s1, method + "+quote"
            else:
                la, lb = _line_around(doc.text, s0, s1)
                line_c = canon(doc.text[la:lb])
                try:
                    u = doc.unit(e["unit_id"])
                except KeyError:
                    u = None
                if len(line_c) >= MIN_SPAN or (matched and "|" in doc.text[la:lb] and len(line_c) >= len(canon(quote))):
                    a, b, method = la, lb, method + "+line"          # a table row that holds the key
                elif u is not None and len(canon(doc.text[u.span[0]:u.span[1]])) >= MIN_SPAN:
                    a, b, method = u.span[0], u.span[1], method + "+unit"
                else:                                                  # a bare label: neighbouring lines
                    a, b = la, lb
                    while len(canon(doc.text[a:b])) < MIN_SPAN and (a > 0 or b < len(doc.text)):
                        if a > 0:
                            a = doc.text.rfind("\n", 0, max(0, a - 1)) + 1
                        if b < len(doc.text):
                            nb = doc.text.find("\n", b + 1)
                            b = len(doc.text) if nb < 0 else nb
                    method = method + "+window"
        text = doc.text[a:b]
        rec["spans"].append({"sid": eid, "evidence": eid, "doc_id": e["doc_id"], "unit_id": e.get("unit_id"),
                             "start": a, "end": b, "text": text, "canon": canon(text), "chars": len(text),
                             "method": method, "keys_matched": matched, "supports": sup,
                             "role": e.get("role"), "quote_chars": len(quote)})
    have = {s["sid"] for s in rec["spans"]}
    for rid, r in reqs.items():
        rec["requirements"].append({"id": rid, "required": bool(r.get("required", True)), "kind": r.get("kind"),
                                    "value_type": r.get("value_type"),
                                    "spans": [s["sid"] for s in rec["spans"] if rid in s["supports"]]})
    for b in q.get("evidence_bundles") or [[e for e in ev]]:
        rec["bundles"].append([x for x in b if x in have])
        missing = [x for x in b if x not in have]
        if missing:
            rec["unresolved"].append({"bundle": b, "why": "bundle evidence without a span: %s" % missing})
    return rec


# --------------------------------------------------------------------------------------------------------------
# retention
# --------------------------------------------------------------------------------------------------------------

def retention(rec, shown_texts):
    """rec: decisive_spans() record; shown_texts: [text shown to the reader, one per evidence item].
    -> {span_frac, req_frac, full, present: [sid], best_bundle, req_retained: [rid], req_required: [rid],
        docs_needed, docs_present} or None for an unanswerable question / no spans."""
    if not rec["answerable"] or not rec["spans"]:
        return None
    # source identity (2026-09-26, Astra R4): an item given as (doc_id, text) - or an object with .doc_id / .text -
    # retains a span only when it comes from the span's own document; a bare string keeps the old text-only match
    # (identity unknown, reported as such)
    shown, identity = [], True
    for t in shown_texts:
        if isinstance(t, str):
            d, x, identity = None, t, False
        elif isinstance(t, (tuple, list)):
            d, x = t[0], t[1]
        else:
            d, x = getattr(t, "doc_id", None), getattr(t, "text", "")
        if x:
            shown.append((d, canon(x)))
    present = {s["sid"]: any(s["canon"] in x and (d is None or d == s["doc_id"]) for d, x in shown)
               for s in rec["spans"]}
    span_doc = {s["sid"]: s["doc_id"] for s in rec["spans"]}
    req_spans = {r["id"]: set(r["spans"]) for r in rec["requirements"] if r["required"]}
    best = None
    for bi, b in enumerate(rec["bundles"] or [[s["sid"] for s in rec["spans"]]]):
        if not b:
            continue
        bs = set(b)
        sf = sum(present[x] for x in b) / len(b)
        rel = {rid: sp & bs for rid, sp in req_spans.items() if sp & bs}
        kept = [rid for rid, sp in rel.items() if all(present[x] for x in sp)]
        rf = len(kept) / len(rel) if rel else sf
        docs = sorted({span_doc[x] for x in b})
        docs_present = sorted({span_doc[x] for x in b if present[x]})
        cand = (sf, rf, -bi)
        if best is None or cand > best[0]:
            best = (cand, {"span_frac": round(sf, 4), "req_frac": round(rf, 4), "full": sf == 1.0,
                           "best_bundle": bi, "req_retained": sorted(kept), "req_required": sorted(rel),
                           "docs_needed": docs, "docs_present": docs_present,
                           "spans_in_bundle": len(b), "spans_present": sum(present[x] for x in b)})
    if best is None:
        return None
    out = best[1]
    out["present"] = sorted(k for k, v in present.items() if v)
    out["source_identity"] = identity and bool(shown)
    return out
