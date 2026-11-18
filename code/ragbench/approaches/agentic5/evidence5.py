"""Stage 7 of AGENTIC5: the evidence table and its source snippets, built JOINTLY under the shared budget, with
citation numbers assigned ONCE, after the final packing decision.

AGENTIC (v1-v3) numbered the table's sources from a probe pack, then packed again; the synthetic table also claimed
the units of every reconciled row, even rows whose source window did not fit. Here:

  1. rows      one table row per kept claim of each requirement (company | period | metric/definition | value/unit |
               status | sources), a `missing` row for every requirement with no value (unknown, never zero), the
               alternatives (other definitions) after the primary, then the deterministic arithmetic results with
               their operand citations, then the reconciliation notes.
  2. sources   one verbatim snippet per distinct source candidate, centred on the quote(s) it supports; two claims
               from one table row share one snippet and one citation.
  3. fit       the table (rendered with the widest citation placeholder) and the snippets must fit TOGETHER: the
               snippets are shortened first (down to `snippet_min_chars`), then the lowest-priority sources are
               dropped (alternatives before primary rows, later requirements before earlier ones). A dropped
               source's alternative rows leave the table; a primary row whose source was dropped stays, marked
               `source not shown`.
  4. number    only now are citation numbers assigned: [1] is the table, the snippets follow in packing order. The
               table is rendered with those numbers, the pack is built by the shared `pack_evidence`, and it is
               checked to hold every item uncut - the numbers in the table are exactly the numbers the reader sees.
  5. units     the table item's unit ids are those of the rows whose source snippet is in the pack.

Anything left of the budget goes, whole items only, to unverified candidates of MISSING requirements (labelled
unverified by the reader-visible header of the table's note), never cited by the table.
"""
import re

from ...contracts import RetrievedItem, item_header, pack_evidence
from ...normalize import normalize
from .defaults import EVIDENCE_VERSION

TEXT = {
    "en": {
        "title": "Verified values by requirement (each value and its quote were checked against the cited source; a "
                 "missing value was not found and is unknown, not zero)",
        "header": "company | period | metric / definition | value | status | sources",
        "status": {"verified": "verified", "conflict": "sources disagree", "missing": "missing - not found",
                   "alternative": "other definition"},
        "any_period": "any",
        "no_source": "source not shown",
        "computed": "Computed by the program from the verified values above:",
        "unknown": "unknown",
        "restated": "{doc} restates {period} as {value} (kept: {kept}).",
        "disagree": "{req}: the sources give different values; none is preferred.",
        "open_series": "{req}: values found for {periods}.",
        "extra": "Unverified passages follow for the missing values (not checked by the program).",
    },
    "fa": {
        "title": "جدول مقادیر تأییدشده برای هر خواستهٔ پرسش (هر مقدار و نقل‌قولش با منبع ذکرشده مقایسه شده است؛ "
                 "مقدارِ پیدانشده نامعلوم است، نه صفر)",
        "header": "شرکت | دوره | شاخص / تعریف | مقدار | وضعیت | منبع",
        "status": {"verified": "تأییدشده", "conflict": "منابع اختلاف دارند", "missing": "پیدا نشد",
                   "alternative": "تعریف دیگر"},
        "any_period": "هر دوره",
        "no_source": "منبع نمایش داده نشده",
        "computed": "محاسبهٔ برنامه از مقادیر تأییدشدهٔ بالا:",
        "unknown": "نامعلوم",
        "restated": "{doc} مقدار دورهٔ {period} را {value} آورده است (مقدار نگه‌داشته: {kept}).",
        "disagree": "{req}: منابع مقادیر متفاوتی می‌دهند؛ هیچ‌کدام ترجیح داده نشده است.",
        "open_series": "{req}: مقدار برای این دوره‌ها پیدا شد: {periods}.",
        "extra": "متن‌های تأییدنشده برای مقادیرِ پیدانشده در ادامه آمده است (برنامه آن‌ها را بررسی نکرده است).",
    },
}

PLACEHOLDER = 99      # the widest citation number a 6000-char pack can hold (two digits)
_RID = re.compile(r"\bR\d+\b")


def _L(q_lang):
    return TEXT.get(q_lang, TEXT["en"])


def _clip(s, n):
    s = normalize(s or "").replace("\n", " ").replace("|", "/").strip()
    return s if len(s) <= n else s[: max(1, n - 1)].rstrip() + "…"


# ---------------------------------------------------------------------------------------------------------------
# snippets
# ---------------------------------------------------------------------------------------------------------------

def _locate(cand, claims):
    """(lo, hi) offsets in cand.text covering the claims' quotes (first line piece) or value tokens."""
    lo = hi = None
    for c in claims:
        k, n = -1, 0
        for piece in [p.strip() for p in normalize(c.get("quote") or "").split("\n") if p.strip()]:
            k = cand.text.find(piece)
            if k >= 0:
                n = len(piece)
                break
        if k < 0:
            tok = str(c.get("value_raw") or c.get("value") or "")
            k = cand.text.find(tok) if tok else -1
            n = len(tok)
        if k < 0:
            continue
        lo = k if lo is None else min(lo, k)
        hi = k + n if hi is None else max(hi, k + n)
    return (lo, hi) if lo is not None else (0, min(len(cand.text), 1))


def excerpt(cand, claims, max_chars):
    """-> (text, start offset in cand.text). A verbatim piece of the candidate around the claims' quotes, snapped to
    line ends when that costs little."""
    t = cand.text
    if len(t) <= max_chars:
        return t, 0
    lo, hi = _locate(cand, claims)
    need = hi - lo
    if need >= max_chars:
        return t[lo: lo + max_chars], lo
    pad = (max_chars - need) // 2
    a = max(0, lo - pad)
    b = min(len(t), a + max_chars)
    a = max(0, b - max_chars)
    nl = t.rfind("\n", a, lo)
    if nl >= 0 and lo - nl <= 120:
        a2 = nl + 1
        b = min(len(t), a2 + max_chars)
        a = a2
    nr = t.find("\n", hi, b)
    if nr >= 0 and b - nr <= 120 and nr > hi:
        b = nr
    return t[a:b], a


def snippet_item(cand, claims, corpus, max_chars):
    body, off = excerpt(cand, claims, max_chars)
    span = None
    if cand.span:
        try:
            doc = corpus.doc(cand.doc_id)
            if doc.text[cand.span[0]:cand.span[1]] == cand.text:
                span = [cand.span[0] + off, cand.span[0] + off + len(body)]
        except KeyError:
            span = None
    units = []
    if span is not None:
        doc = corpus.doc(cand.doc_id)
        want = set(cand.unit_ids)
        units = [u.id for u in doc.units_overlapping(span[0], span[1]) if u.id in want]
    if not units:
        units = [u for u in dict.fromkeys([c.get("unit_id") for c in claims if c.get("unit_id")])]
    return RetrievedItem(text=body, doc_id=cand.doc_id, section_id=cand.section_id, unit_ids=units, kind="window",
                         span=span, meta={"cid": cand.cid, "claims": [c.get("claim_id") for c in claims],
                                          "verified": True, "excerpt_of_chars": len(cand.text)})


# ---------------------------------------------------------------------------------------------------------------
# the table
# ---------------------------------------------------------------------------------------------------------------

def req_label(r):
    parts = [r.get("entity") or "", r.get("period_key") or "", r.get("metric") or ""]
    return "%s (%s)" % (r["id"], ", ".join(p for p in parts if p))


def display_value(c):
    """value + unit, without repeating a scale word the value token already carries ("100 million" + "million
    USD" -> "100 million USD")."""
    v = normalize(str(c.get("value") or "")).strip()
    u = normalize(str(c.get("unit") or "")).strip()
    tail = re.findall(r"[^\W\d_]+", v)
    if tail and u.lower().startswith(tail[-1].lower()):
        u = u[len(tail[-1]):].strip()
    return ("%s %s" % (v, u)).strip()


def row_line(row, L, nums, opts):
    r = row["req"]
    c = row.get("claim")
    company = _clip((c or {}).get("entity_name") or r.get("entity") or (c or {}).get("doc_company") or "-", 40)
    per = (c or {}).get("period_key") or r.get("period_key") or L["any_period"]
    if c and c.get("period_inferred"):
        per += "*"
    metric = r.get("metric") or ""
    if c and c.get("definition"):
        metric = "%s / %s" % (metric, c.get("definition"))
    metric = _clip(metric, int(opts.get("table_def_chars", 60)))
    value = display_value(c) if c else "-"
    status = L["status"].get(row["status"], row["status"])
    srcs = [nums[s] for s in row.get("sources") or [] if s in nums]
    if c and not srcs:
        src = L["no_source"]
    else:
        src = "".join("[%d]" % n for n in srcs) or "-"
    return "%s | %s | %s | %s | %s | %s" % (company, per, metric, _clip(value, 48), status, src)


def render(table_rows, computed, notes, labels, q_lang, nums, opts):
    L = _L(q_lang)
    lines = [L["title"], L["header"]]
    for row in table_rows:
        lines.append(row_line(row, L, nums, opts))
    if computed:
        lines.append(L["computed"])
        for res in computed:
            txt = res["text"] if res["status"] == "ok" else "%s (%s)" % (L["unknown"], res.get("reason") or "")
            txt = _RID.sub(lambda m: labels.get(m.group(0), m.group(0)), txt)
            srcs = []
            for k in res.get("sources") or []:
                if k in nums and nums[k] not in srcs:
                    srcs.append(nums[k])
            lines.append("- %s: %s %s" % (res["op"], txt, "".join("[%d]" % n for n in srcs)))
    return "\n".join(lines + list(notes))


# ---------------------------------------------------------------------------------------------------------------
# the joint build
# ---------------------------------------------------------------------------------------------------------------

def table_rows(matrix, recon):
    """-> (rows, source order). A row: {req, claim | None, status, sources: [cid], priority}."""
    rows, order = [], []
    for pri, r in enumerate(matrix["requirements"]):
        e = recon["requirements"].get(r["id"]) or {}
        kept = e.get("rows") or []
        if not kept:
            rows.append({"req": r, "claim": None, "status": "missing", "sources": [], "priority": (0, pri)})
        for c in kept:
            rows.append({"req": r, "claim": c, "status": c.get("req_status") or e.get("status") or "verified",
                         "sources": [c["cid"]], "priority": (0, pri)})
        for c in e.get("alternatives") or []:
            rows.append({"req": r, "claim": c, "status": "alternative", "sources": [c["cid"]], "priority": (1, pri)})
    for row in sorted(rows, key=lambda x: x["priority"]):
        for s in row["sources"]:
            if s not in order:
                order.append(s)
    return rows, order


def notes_for(matrix, recon, q_lang):
    L = _L(q_lang)
    notes = []
    labels = {r["id"]: req_label(r) for r in matrix["requirements"]}
    if any((row.get("period_inferred")) for e in recon["requirements"].values() for row in e.get("rows") or []):
        notes.append("* " + ("دوره از سال گزارش گرفته شده است" if q_lang == "fa" else "period taken from the report's year"))
    for rid, e in recon["requirements"].items():
        if e["status"] == "conflict":
            notes.append(L["disagree"].format(req=labels.get(rid, rid)))
        for c in e.get("conflicts") or []:
            notes.append(L["restated"].format(doc=c["restated_doc"], period=c.get("period") or "",
                                              value=c["restated"], kept=", ".join(c["kept"])))
        if e.get("periods_found") is not None and e["requirement"]["operation"] == "series":
            notes.append(L["open_series"].format(req=labels.get(rid, rid), periods=", ".join(e["periods_found"]) or "-"))
    return notes


def build(matrix, recon, computed, cands_by_cid, index, cfg, q_lang, opts, extra=(), state=None):
    """-> (EvidencePack, info). `extra`: unverified candidates for missing requirements, in priority order.
    `state` (optional dict) receives what `expand_after_fit` needs to re-render the table with new numbers."""
    corpus = index.corpus
    budget = int(cfg.evidence["max_chars"])
    rows, order = table_rows(matrix, recon)
    labels = {r["id"]: req_label(r) for r in matrix["requirements"]}
    claims_of = {}
    for row in rows:
        if row["claim"] is not None:
            for s in row["sources"]:
                claims_of.setdefault(s, []).append(row["claim"])
    by_claim = {row["claim"]["claim_id"]: row["claim"] for row in rows if row["claim"] is not None}
    for res in computed:
        res["sources"] = [by_claim[k]["cid"] for k in res.get("operands") or [] if k in by_claim]
    notes = notes_for(matrix, recon, q_lang)
    order = [s for s in order if s in cands_by_cid]
    info = {"version": EVIDENCE_VERSION, "budget": budget, "rows": len(rows), "sources_wanted": len(order),
            "dropped_sources": [], "snippet_chars": None, "table_rows_dropped": 0}

    def total_len(tbl_text, snips):
        n = len(item_header(1, RetrievedItem(text=tbl_text, doc_id=tbl_doc), corpus)) + 1 + len(tbl_text.strip())
        for it in snips:
            n += 2 + len(item_header(PLACEHOLDER, it, corpus)) + 1 + len(it.text.strip())
        return n

    first = next((cands_by_cid[s] for s in order), None)
    tbl_doc = first.doc_id if first is not None else (matrix.get("doc_hint") or "")
    kept_sources = list(order)
    live_rows = list(rows)
    snips = []
    chosen_size = None
    sizes = []
    s = int(opts.get("snippet_max_chars", 700))
    while s >= int(opts.get("snippet_min_chars", 200)):
        sizes.append(s)
        s = int(s * 0.75)
    sizes = sizes or [int(opts.get("snippet_min_chars", 200))]

    def placeholder_nums(srcs):
        return {k: PLACEHOLDER for k in srcs}

    while True:
        ph = placeholder_nums(kept_sources)
        tbl = render(live_rows, computed, notes, labels, q_lang, ph, opts)
        fitted = None
        for size in sizes:
            snips = [snippet_item(cands_by_cid[k], claims_of.get(k, []), corpus, size) for k in kept_sources]
            if total_len(tbl, snips) <= budget:
                fitted = size
                break
        if fitted is not None or not kept_sources:
            chosen_size = fitted
            if fitted is None:
                snips = []
            break
        # drop the lowest-priority source: its alternative rows leave the table
        drop = kept_sources.pop()
        info["dropped_sources"].append(drop)
        before = len(live_rows)
        live_rows = [r for r in live_rows if not (r["status"] == "alternative" and drop in r["sources"])]
        info["table_rows_dropped"] += before - len(live_rows)
    # a table that does not fit even alone: drop alternative rows, then the notes, then clip rows
    tbl = render(live_rows, computed, notes, labels, q_lang, placeholder_nums(kept_sources), opts)
    while total_len(tbl, snips) > budget and (any(r["status"] == "alternative" for r in live_rows) or notes):
        if any(r["status"] == "alternative" for r in live_rows):
            live_rows = [r for r in live_rows if r["status"] != "alternative"]
        else:
            notes = notes[:-1]
        tbl = render(live_rows, computed, notes, labels, q_lang, placeholder_nums(kept_sources), opts)
    info["snippet_chars"] = chosen_size
    # extra: unverified support for MISSING requirements in the budget that is left. `extra` is either a callable
    # room -> items (the shared `evidence.build_blocks` constructor, bounded by that room) or candidates added whole.
    extra_items = []
    if extra and recon.get("missing"):
        L = _L(q_lang)
        notes2 = notes + [L["extra"]]
        tbl2 = render(live_rows, computed, notes2, labels, q_lang, placeholder_nums(kept_sources), opts)
        room = budget - total_len(tbl2, snips) - 2
        if room > 0:
            if callable(extra):
                # build_blocks numbers its own items from [1]: allow for the wider final numbers
                cand_items = list(extra(max(0, room - 8)) or [])
            else:
                cand_items = [c.to_item(meta={"verified": False, "for_missing": True}) for c in extra]
            for it in cand_items:
                it.meta = dict(it.meta or {}, verified=False, for_missing=True)
                if total_len(tbl2, snips + extra_items + [it]) <= budget:
                    extra_items.append(it)
            if extra_items:
                notes = notes2
    # 4. numbers, once: [1] = the table, then the snippets in packing order, then the unverified extras
    nums = {k: i + 2 for i, k in enumerate(kept_sources)}
    text = render(live_rows, computed, notes, labels, q_lang, nums, opts)
    units = []
    for r in live_rows:
        c = r.get("claim")
        if c and any(s in nums for s in r["sources"]) and c.get("unit_id") and c["unit_id"] not in units:
            units.append(c["unit_id"])
    table = RetrievedItem(text=text, doc_id=tbl_doc, section_id=None, unit_ids=units, kind="fact",
                          meta={"table": True, "rows": len(live_rows), "version": EVIDENCE_VERSION})
    items = [table] + snips + extra_items
    for i, it in enumerate(items, 1):
        it.rank = i
    pack = pack_evidence(items, corpus, budget=budget, cfg=cfg)
    while (pack.n_included < len(items) or pack.last_item_cut) and extra_items:
        # only an unverified extra can be the one that does not fit (the rest was measured with wider numbers)
        extra_items.pop()
        if not extra_items and notes and notes[-1] == _L(q_lang)["extra"]:
            notes = notes[:-1]
            text = render(live_rows, computed, notes, labels, q_lang, nums, opts)
            table.text = text
        items = [table] + snips + extra_items
        pack = pack_evidence(items, corpus, budget=budget, cfg=cfg)
    info.update(items=len(items), included=pack.n_included, cut=pack.last_item_cut, chars=pack.chars,
                citations={k: n for k, n in nums.items()}, extra=len(extra_items), table_chars=len(text),
                exact=pack.n_included == len(items) and not pack.last_item_cut)
    if state is not None:
        state.update(live_rows=live_rows, computed=computed, notes=notes, labels=labels, q_lang=q_lang, opts=opts,
                     nums=dict(nums), items=list(items), table=table, snips=list(snips), budget=budget, pack=pack,
                     exact=info["exact"])
    return pack, info, text


# ---------------------------------------------------------------------------------------------------------------
# the section_expand option (ragbench/variants.py), at AGENTIC5's final boundary
# ---------------------------------------------------------------------------------------------------------------

def _body(it):
    return (it.text or "").strip()


def expand_after_fit(state, corpus, question, cfg):
    """section_expand for AGENTIC5 -> (EvidencePack | None, table text | None, decision).

    Runs once, AFTER the joint table + snippet fit (build's `state`). The shared section choice
    (variants.expand_sections) places the whole section after the table; items it contains are dropped as duplicates
    only when their text is shown verbatim inside it. Citation numbers are then assigned again, from the expanded
    order (a source whose snippet now lives inside the section cites the section), the table is re-rendered with them
    and packed. The expansion is kept only if that pack holds every item whole, every document, every unit and every
    citation of the unexpanded pack, and each cited number shows its source's verbatim snippet; otherwise it is
    skipped (None) and the decision says why."""
    from ...variants import VARIANTS_V2_VERSION, expand_sections
    items, before = state["items"], state["pack"]
    dec = {"version": VARIANTS_V2_VERSION, "boundary": "agentic5: after the joint table + snippet fit",
           "applied": False, "citations_before": dict(state["nums"])}

    def skip(reason, **kw):
        dec.update(kw, applied=False, skipped=True, reason=reason)
        return None, None, dec

    if not state.get("exact"):
        return skip("the unexpanded pack is not exact (an item was cut or dropped)")
    items2, d0 = expand_sections(items, corpus, question, state["budget"])
    d0.pop("version", None)
    dec.update(d0)
    dec["applied"] = False
    if not d0.get("applied"):
        return skip(d0.get("reason") or "no section chosen")
    secs = [it for it in items2 if (it.meta or {}).get("section_expand")]
    kept_ids = {id(it) for it in items2}
    hosts = {}
    for it in items:
        if id(it) in kept_ids:
            continue
        host = next((s for s in secs if _body(it) and _body(it) in s.text), None)
        if host is None:
            return skip("an item the section would replace is not shown verbatim inside it",
                        lost=[(it.meta or {}).get("cid") or it.doc_id])
        hosts[id(it)] = host
    table = state["table"]
    if not items2 or items2[0] is not table:
        return skip("the table would not stay first")
    pos = {id(it): i + 1 for i, it in enumerate(items2)}
    snip_by_cid = {(s.meta or {}).get("cid"): s for s in state["snips"]}
    nums = {}
    for cid in state["nums"]:
        s = snip_by_cid.get(cid)
        if s is None:
            return skip("a cited source has no snippet", lost=[cid])
        nums[cid] = pos[id(hosts.get(id(s), s))]
    text = render(state["live_rows"], state["computed"], state["notes"], state["labels"], state["q_lang"], nums,
                  state["opts"])
    table2 = RetrievedItem(text=text, doc_id=table.doc_id, section_id=None, unit_ids=list(table.unit_ids),
                           kind=table.kind, meta=dict(table.meta or {}, renumbered_after="section_expand"))
    items3 = [table2 if it is table else it for it in items2]
    after = pack_evidence(items3, corpus, budget=state["budget"], cfg=cfg)
    dec.update(citations_after=dict(nums), packed_chars_before=before.chars, packed_chars_after=after.chars,
               items_before=len(items), items_after=len(items3))
    if after.n_included < len(items3) or after.last_item_cut:
        return skip("the expanded pack does not hold every item whole (%d of %d, cut=%s)" % (
            after.n_included, len(items3), after.last_item_cut))
    lost_docs = [d for d in dict.fromkeys(it.doc_id for it in before.items)
                 if d not in {it.doc_id for it in after.items}]
    lost_units = [u for u in before.unit_ids if u not in set(after.unit_ids)]
    if lost_docs or lost_units:
        return skip("the expanded pack would lose %d document(s) and %d unit(s)" % (len(lost_docs), len(lost_units)),
                    lost_docs=lost_docs, lost_units=len(lost_units))
    for cid, n in nums.items():
        if _body(snip_by_cid[cid]) not in (after.items[n - 1].text or ""):
            return skip("citation [%d] would not show its source's snippet" % n, lost=[cid])
    for i, it in enumerate(after.items, 1):
        it.rank = i
    dec.update(applied=True, skipped=False, renumbered=nums != state["nums"],
               dropped_as_inside=len(hosts))
    return after, text, dec
