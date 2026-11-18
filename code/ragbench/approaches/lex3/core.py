"""LEX3's model-free machinery: routing with an escape channel, the hybrid-grounded phrase list, controlled lexical
refinements (morphology, issuer-only name demotion), order-independent region fusion and question-derived
requirements. No model call anywhere in this file.

Everything lexical reuses LEX (approaches/lex/search.py): the same folding, the same Persian-aware term compiler,
the same unit search and structure windows. What is new is where the phrases come from (the evidence the ORIGINAL
question retrieved, with IDs and sources), how the lexical windows meet the question's own BM25/dense chunks (RRF over
regions, one contribution per channel per region) and how evidence is reserved (one block per requirement).
"""
import re
from collections import defaultdict

from ...normalize import fold
from ...retrieval.hybrid import rrf_fuse
from ..lex import search as S

_CLAUSE = re.compile(r"[.;:!?\n()\[\]«»\"،؛؟,]+")
_CAPTION_HEAD = re.compile(r"^\s*(?:table|figure|chart|جدول|نمودار|شکل)\s*[\d۰-۹.\-]*\s*[:.\-–]?\s*", re.I)
_LETTER = re.compile(r"[^\W\d_]", re.UNICODE)


# ---------------------------------------------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------------------------------------------

def stems_of(texts, lang):
    """Folded content stems (LEX's query_stems without measure/document words, >= 3 letters)."""
    return [s for s in S.query_stems(texts, lang) if s not in S.NONCONTENT and len(s) >= 3]


def shares(stems, text):
    words = fold(text).split()
    return [s for s in stems if any(S.stem_hit(s, w) for w in words)]


def expand_periods(periods):
    """Explicit periods -> the years they bound (LEX's question_years: a range is filled in, up to 12 years)."""
    return [str(y) for y in sorted(S.question_years(periods or []))]


def issuer_heads(con, doc_ids):
    """-> {doc_id: folded "title | company"}: what names a report's own issuer."""
    out = {}
    for d, r in S.doc_rows(con, doc_ids).items():
        out[d] = "%s | %s" % (r.get("title_fold") or fold(r.get("title") or ""), fold(r.get("company") or ""))
    return out


def is_issuer(name, head):
    """A question name is the report's own issuer when the name is (part of) the report's title/company, or the
    company is part of the name. A subsidiary, plant or region named alongside is NOT the issuer (its name is not in
    the parent's title): «Alpha Steel Mills» in «Alpha Corp annual report» stays a discriminating name."""
    nm = fold(name)
    if not nm or not head:
        return False
    title, _, company = head.partition(" | ")
    return nm in title or nm == company or (len(company) >= 3 and company in nm)


# ---------------------------------------------------------------------------------------------------------------
# 7. routing
# ---------------------------------------------------------------------------------------------------------------

def chunk_contribution(hits, best=2):
    """-> {doc_id: sum of its best `best` chunk RRF scores}: a report's evidence from the question's own chunks, capped
    at its best one or two (summing every chunk vote favours long reports)."""
    per = defaultdict(list)
    for h in hits:
        per[h.doc_id].append(float(h.rrf))
    return {d: sum(sorted(v, reverse=True)[:best]) for d, v in per.items()}


def route(con, pool, hits, kw, lang, o, forced=()):
    """-> (shortlist, info). Card score (LEX's card BM25 + name + year bonuses, search.route) normalised to [0, 1],
    plus the capped chunk contribution normalised to [0, 1]; shortlist of up to `shortlist_max`, with `forced` reports
    (explicit entity/period matches) kept first, up to `forced_max`."""
    opts = dict(o, route_k=len(pool), route_max=len(pool))
    _keep, scored = S.route(con, pool, kw, lang, opts)
    card = dict(scored)
    cmax = max(card.values(), default=0.0) or 1.0
    chunk = chunk_contribution(hits, int(o["route_best_chunks"]))
    kmax = max(chunk.values(), default=0.0) or 1.0
    total = {d: float(o["w_route_card"]) * card.get(d, 0.0) / cmax + float(o["w_route_chunk"]) * chunk.get(d, 0.0) / kmax
             for d in pool}
    ranked = sorted(pool, key=lambda d: (-total[d], d))
    forced = [d for d in ranked if d in set(forced)][: int(o["forced_max"])]
    short = list(forced)
    for d in ranked:
        if len(short) >= max(int(o["shortlist_max"]), len(forced)):
            break
        if d not in short:
            short.append(d)
    info = {"ranked": [[d, round(total[d], 4), round(card.get(d, 0.0), 3), round(chunk.get(d, 0.0), 5)]
                       for d in ranked[:16]], "forced": forced}
    return short, info


def explicit_matches(con, pool, entities, periods):
    """Reports whose title/company names a requested entity AND (when periods are given) whose own year is one of
    them - the explicit requests routing must keep."""
    if not entities and not periods:
        return []
    rows = S.doc_rows(con, pool)
    heads = issuer_heads(con, pool)
    ys = {int(y) for y in periods if str(y).isdigit()}
    out = []
    for d in pool:
        r = rows.get(d) or {}
        ok_e = (not entities) or any(fold(e) and (fold(e) in heads.get(d, "") or is_issuer(e, heads.get(d, "")))
                                     for e in entities)
        y0, y1 = r.get("year_from"), r.get("year_to") or r.get("year_from")
        ok_p = (not ys) or (y0 is not None and any(y0 <= y <= y1 for y in ys))
        if ok_e and ok_p and (entities or ys):
            out.append(d)
    return out


# ---------------------------------------------------------------------------------------------------------------
# 3. the feedback vocabulary
# ---------------------------------------------------------------------------------------------------------------

def _clean_phrase(t, o):
    t = " ".join((t or "").split()).strip(" -–|:•*#")
    f = fold(t)
    if not f or len(_LETTER.findall(f)) < int(o["vocab_min_letters"]):
        return None
    n = len(f.split())
    if n < int(o["vocab_min_words"]) or n > int(o["vocab_max_words"]):
        return None
    if not S.content_words(t):                   # measure / period / document words only
        return None
    return t


def _windows_around(words, idx, width):
    lo = max(0, idx - (width - 1) // 2)
    hi = min(len(words), lo + width)
    return words[lo:hi]


def _clause_phrases(text, stems, o, width=4):
    """Short verbatim word runs of a clause around a word a stem matches (at most `width` words), and the clause
    itself when it is short enough."""
    out = []
    for cl in _CLAUSE.split(text or ""):
        words = cl.split()
        if not words:
            continue
        if len(words) <= int(o["vocab_max_words"]) and shares(stems, cl):
            out.append(" ".join(words))
            continue
        fw = [fold(w) for w in words]
        for i, w in enumerate(fw):
            if any(S.stem_hit(s, x) for s in stems for x in w.split()):
                run = _windows_around(words, i, width)
                while run and (fold(run[0]) in S.NONCONTENT or fold(run[0]).isdigit()):
                    run = run[1:]
                while run and (fold(run[-1]) in S.NONCONTENT or fold(run[-1]).isdigit()):
                    run = run[:-1]
                if run:
                    out.append(" ".join(run))
    return out


def _table_labels(u):
    lines = (u.text or "").split("\n")
    out = []
    if lines:
        out += [c.strip() for c in lines[0].split("|")]
    for ln in lines[1:13]:
        cells = [c.strip() for c in ln.split("|")]
        if cells:
            out.append(cells[0])
    return out


def feedback_vocabulary(hits, corpus, facets, question, lang, o, con=None, years=()):
    """The phrase list call 2 chooses from, every phrase an exact string of the evidence the ORIGINAL question
    retrieved: section headings, captions, table labels (heading cells, row labels), short clauses around a
    facet/question word, and (when the knowledge base has them) the metric names of the hit units' facts.

    Each phrase: {id, text, kind, facets (1-based facet numbers it shares a stem with), docs, source {doc_id,
    section_id, unit_id, chunk, hit_rank}, n_sources, score, where}. Spread over facets and reports: every facet its
    best `vocab_per_facet_min`, every report its best `vocab_per_doc_min`, then by score, `vocab_max` in all.
    -> (vocab, info)"""
    fstems = [stems_of(f["terms"] + [f["name"]], lang) for f in facets]
    qstems = stems_of([question], lang)
    allstems = list(dict.fromkeys([s for fs in fstems for s in fs] + qstems))
    cand = {}
    kind_bonus = {"heading": 0.3, "caption": 0.3, "label": 0.2, "metric": 0.2, "clause": 0.0}

    def add(text, kind, h, rank, unit_id=None):
        t = _clean_phrase(text, o)
        if t is None:
            return
        f = fold(t)
        if any(f == fold(y) for y in years):
            return
        fs = [i + 1 for i, st in enumerate(fstems) if shares(st, t)]
        qs = shares(qstems, t)
        c = cand.get(f)
        if c is None:
            c = cand[f] = {"text": t, "kind": kind, "facets": set(), "docs": set(), "n_sources": 0, "best": rank,
                           "source": {"doc_id": h.doc_id, "section_id": h.section_id, "unit_id": unit_id,
                                      "chunk": h.chunk, "hit_rank": rank + 1}, "qs": set()}
        c["facets"].update(fs)
        c["qs"].update(qs)
        c["docs"].add(h.doc_id)
        c["n_sources"] += 1
        if rank < c["best"]:
            c["best"] = rank

    n_hits = 0
    for rank, h in enumerate(hits[: int(o["vocab_hits"])]):
        n_hits += 1
        if not corpus.has_doc(h.doc_id):
            continue
        doc = corpus.doc(h.doc_id)
        for sid in dict.fromkeys([h.section_id] + list(getattr(h, "section_ids", None) or [])):
            try:
                for title in doc.section_path(sid):
                    add(title, "heading", h, rank)
            except KeyError:
                pass
        units = [doc.unit(u) for u in h.unit_ids if doc.has_unit(u)]
        for j, u in enumerate(units):
            if u.kind == "caption":
                add(_CAPTION_HEAD.sub("", u.text), "caption", h, rank, u.id)
            elif u.kind == "table":
                # the caption unit directly before the table in its section
                try:
                    sec_units = doc.section_units(u.section_id)
                    k = next(i for i, x in enumerate(sec_units) if x.id == u.id)
                    if k > 0 and sec_units[k - 1].kind == "caption":
                        add(_CAPTION_HEAD.sub("", sec_units[k - 1].text), "caption", h, rank, sec_units[k - 1].id)
                except (KeyError, StopIteration):
                    pass
                for lab in _table_labels(u):
                    add(lab, "label", h, rank, u.id)
            elif u.kind == "heading":
                add(u.text, "heading", h, rank, u.id)
            else:
                a, b = max(h.span[0], u.span[0]), min(h.span[1], u.span[1])
                seg = doc.text[a:b] if b > a else u.text
                for ph in _clause_phrases(seg, allstems, o):
                    add(ph, "clause", h, rank, u.id)
        if con is not None and h.unit_ids:
            try:
                for r in con.execute("SELECT DISTINCT fl.metric_name FROM fact_labels fl JOIN facts f ON f.fact_id = "
                                     "fl.fact_id WHERE f.unit_id IN (%s) AND fl.label_quality = 'ok' AND "
                                     "fl.metric_name IS NOT NULL" % ",".join("?" * len(h.unit_ids)), list(h.unit_ids)):
                    add(r["metric_name"], "metric", h, rank)
            except Exception:  # noqa: BLE001  (a knowledge base without the label layer)
                pass
    # clauses need a facet/question stem (they are cut around one); headings, captions, labels and metric names are
    # kept as feedback even without one - they are what the report calls the thing near the evidence
    keep = [c for c in cand.values() if c["kind"] != "clause" or c["facets"] or c["qs"]]
    for c in keep:
        c["score"] = round(1.0 / (1.0 + c["best"]) + 1.0 * len(c["facets"]) + 0.5 * min(len(c["qs"]), 2)
                           + kind_bonus[c["kind"]] + 0.1 * min(c["n_sources"] - 1, 3), 4)
    keep.sort(key=lambda c: (-c["score"], c["best"], len(c["text"]), c["text"]))
    cap = int(o["vocab_max"])
    chosen, seen = [], set()

    def take(c):
        if c["text"] not in seen and len(chosen) < cap:
            chosen.append(c)
            seen.add(c["text"])

    for i in range(1, len(facets) + 1):                  # every facet its own share
        for c in [c for c in keep if i in c["facets"]][: int(o["vocab_per_facet_min"])]:
            take(c)
    docs = list(dict.fromkeys(h.doc_id for h in hits[: int(o["vocab_hits"])]))
    for d in docs:                                        # every report its own share
        for c in [c for c in keep if d in c["docs"] and c["text"] not in seen][: int(o["vocab_per_doc_min"])]:
            take(c)
    for c in keep:
        take(c)
    chosen.sort(key=lambda c: (-c["score"], c["best"], len(c["text"]), c["text"]))
    titles = {}
    vocab = []
    for n, c in enumerate(chosen, 1):
        d = c["source"]["doc_id"]
        if d not in titles:
            titles[d] = (corpus.doc(d).title if corpus.has_doc(d) else d)[:48]
        vocab.append({"id": "P%d" % n, "text": c["text"], "kind": c["kind"], "facets": sorted(c["facets"]),
                      "docs": sorted(c["docs"]), "source": c["source"], "n_sources": c["n_sources"],
                      "score": c["score"], "where": "%s · %s" % (c["kind"], titles[d])})
    info = {"hits_mined": n_hits, "candidates": len(cand), "kept": len(keep), "shown": len(vocab),
            "by_kind": {k: sum(1 for v in vocab if v["kind"] == k) for k in kind_bonus},
            "by_facet": {i: sum(1 for v in vocab if i in v["facets"]) for i in range(1, len(facets) + 1)},
            "docs": len({d for v in vocab for d in v["docs"]})}
    return vocab, info


# ---------------------------------------------------------------------------------------------------------------
# 8. lexical refinements
# ---------------------------------------------------------------------------------------------------------------

# inflectional endings only (no derivational strip: "production" is not "product", «کاربری» stays LEX's own business)
_EN_SUFFIX = ("ies", "es", "ed", "ing", "s")
_EN_TAIL = r"(?:s|es|ed|ing|e|ies|y)?"
_FA_SUFFIX = ("هایی", "های", "ها", "ترین", "تر", "ات", "ان", "یی", "ای", "ی")
_FA_TAIL = r"(?:\s?(?:هایی|های|ها|ترین|تر|ات|ان|یی|ای|ی))?"
_BOUND_L = r"(?<![^\W\d_])"
_BOUND_R = r"(?![^\W\d_])"


def morph_stem(word, lang, min_stem):
    w = fold(word)
    for suf in (_FA_SUFFIX if lang == "fa" else _EN_SUFFIX):
        if w.endswith(suf) and len(w) - len(suf) >= min_stem:
            return w[: -len(suf)]          # "companies" -> "compan", allowed back as compan(ies|y)
    return w if len(w) >= min_stem else None


class MorphTerm(S.Term):
    """The lower-weight morphological twin of a literal term: every content word reduced to its inflectional stem
    (a closed list of endings) and allowed any of them. It matches only where its literal term does NOT (so a text
    is never credited twice), and weighs `w_morph` x the literal term's weight with the literal term's IDF."""

    def __init__(self, literal, patterns, w_morph):
        super().__init__(literal.text + "~m", "morph:" + literal.source, literal.weight * float(w_morph), patterns,
                         weak=literal.weak)
        self.literal = literal

    def match(self, folded):
        if self.literal.match(folded)[0] >= 0:
            return -1, None
        for p in self.patterns:
            m = p.search(folded)
            if m:
                return m.start(), False
        return -1, None

    def trace(self):
        d = super().trace()
        d["morph_of"] = self.literal.text
        return d


def morph_terms(terms, lang, o):
    """-> [MorphTerm] for the literal terms whose stemmed form differs from what the literal regexes already allow."""
    out = []
    min_stem = int(o["morph_min_stem"]) if lang == "en" else max(3, int(o["morph_min_stem"]) - 1)
    tail = _FA_TAIL if lang == "fa" else _EN_TAIL
    for t in terms:
        if isinstance(t, MorphTerm) or t.source == "llm_regex":
            continue
        words = fold(t.text).split()
        parts, changed = [], False
        for w in words:
            st = morph_stem(w, lang, min_stem) if (w not in S.NONCONTENT and not w.isdigit()) else None
            if st and st != w:
                changed = True
                parts.append(re.escape(st) + tail)
            elif st:
                parts.append(re.escape(st) + tail)
            else:
                parts.append(re.escape(w))
        if not changed and all(len(w) < min_stem or w in S.NONCONTENT for w in words):
            continue
        pat = _BOUND_L + r"\s+".join(parts) + _BOUND_R
        try:
            rx = re.compile(pat, re.I)
        except re.error:
            continue
        out.append(MorphTerm(t, [rx], o["w_morph"]))
    return out


def align_morph_idf(terms):
    """After search(): a morph term keeps its literal term's IDF, so a rarer morphological form never outweighs the
    literal term it came from."""
    for t in terms:
        if isinstance(t, MorphTerm):
            t.idf = t.literal.idf


def name_bonus(wins, names, heads, o):
    """Add the name bonus LEX's windows() gives, doc-aware: a question name inside its OWN issuer's report is
    boilerplate there (w_issuer_name); any other name - a subsidiary, a plant, a region, another company - keeps the
    full w_name. Windows re-sorted. -> the windows."""
    fn = [(n, fold(n)) for n in names if len(fold(n)) >= 2]
    for w in wins:
        f = fold(w["text"])
        best, why = 0.0, None
        for n, nm in fn:
            if nm in f:
                own = is_issuer(n, heads.get(w["doc_id"], ""))
                b = float(o["w_name"]) * (float(o["w_issuer_name"]) if own else 1.0)
                if b > best:
                    best, why = b, ("issuer" if own else "name")
        if best:
            w["score"] = round(w["score"] + best, 4)
            w["name_bonus"] = [why, round(best, 3)]
    wins.sort(key=lambda w: (-w["score"], w["doc_id"], w["hit"]))
    return wins


# ---------------------------------------------------------------------------------------------------------------
# 5. fusion over regions
# ---------------------------------------------------------------------------------------------------------------

def _extent(c, corpus):
    if c.get("span"):
        return [list(c["span"])]
    if not corpus.has_doc(c["doc_id"]):
        return []
    doc = corpus.doc(c["doc_id"])
    return [list(doc.unit(u).span) for u in c.get("unit_ids") or [] if doc.has_unit(u)]


def _ov(ea, eb):
    inter = sum(max(0, min(a[1], b[1]) - max(a[0], b[0])) for a in ea for b in eb)
    la = sum(max(1, a[1] - a[0]) for a in ea)
    lb = sum(max(1, b[1] - b[0]) for b in eb)
    return inter / max(1, min(la, lb))


def fuse_regions(channels, corpus, o):
    """channels: {name: [candidate dicts best first]}, each candidate {doc_id, span | unit_ids, ...}.

    Candidates of one report whose text overlaps by more than `region_overlap` of the smaller are one REGION (union-
    find); region ids follow (doc_id, start), so they do not depend on channel order. Each channel ranks the regions
    by its best candidate in them - ONE contribution per channel per region - and the regions are fused by equal-weight
    RRF (k = rrf_k). -> (regions best first: [{rid, rrf, ranks, doc_id, members: [candidates]}], info)"""
    cands = []
    for ch in sorted(channels):
        for r, c in enumerate(channels[ch], 1):
            c = dict(c, channel=ch, rank=r)
            c["_ext"] = _extent(c, corpus)
            if c["_ext"]:
                cands.append(c)
    parent = list(range(len(cands)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    by_doc = defaultdict(list)
    for i, c in enumerate(cands):
        by_doc[c["doc_id"]].append(i)
    thr = float(o["region_overlap"])
    for d, idx in by_doc.items():
        for x in range(len(idx)):
            for y in range(x + 1, len(idx)):
                i, j = idx[x], idx[y]
                if _ov(cands[i]["_ext"], cands[j]["_ext"]) > thr:
                    parent[find(i)] = find(j)
    groups = defaultdict(list)
    for i in range(len(cands)):
        groups[find(i)].append(i)
    keyed = sorted(groups.values(), key=lambda g: (cands[g[0]]["doc_id"],
                                                    min(e[0] for i in g for e in cands[i]["_ext"])))
    region_of = {}
    for rid, g in enumerate(keyed):
        for i in g:
            region_of[i] = rid
    rankings = {}
    for ch in sorted(channels):
        mine = sorted((cands[i]["rank"], region_of[i]) for i in range(len(cands)) if cands[i]["channel"] == ch)
        rankings[ch] = [rid for _r, rid in mine]
    fused = rrf_fuse(rankings, k=int(o["rrf_k"]))
    regions = []
    for rid, s, rk in fused:
        members = sorted((cands[i] for i in keyed[rid]), key=lambda c: (c["rank"], c["channel"]))
        for m in members:
            m.pop("_ext", None)
        regions.append({"rid": rid, "rrf": s, "ranks": rk, "doc_id": members[0]["doc_id"], "members": members})
    info = {"candidates": len(cands), "regions": len(regions),
            "by_channel": {ch: len(v) for ch, v in rankings.items()},
            "multi_channel": sum(1 for r in regions if sum(1 for v in r["ranks"].values() if v) > 1)}
    return regions, info


def block_candidates(regions, limit):
    """Regions -> build_blocks candidates, best region first; inside a region its members by their best rank. Every
    candidate carries its REGION's channel ranks and RRF (ranking decided; text is build_blocks' job)."""
    out = []
    for reg in regions[:limit]:
        for m in reg["members"]:
            out.append({"doc_id": m["doc_id"], "span": m.get("span"), "unit_ids": list(m.get("unit_ids") or []),
                        "ranks": dict(reg["ranks"]), "rrf": reg["rrf"], "channel": m["channel"],
                        "region": reg["rid"]})
    return out


# ---------------------------------------------------------------------------------------------------------------
# requirements (from the QUESTION's interpretation, never from gold)
# ---------------------------------------------------------------------------------------------------------------

def requirements(facets, entities, periods, facet_terms, o):
    """-> [{id, terms, entity?, period?}] for build_blocks: one per facet x entity x period. The entity dimension is
    split only when the question names more than one entity (a comparison), the period dimension only when it names
    more than one period (a series); the SAME facet vocabulary is used for every entity and period (nothing is removed
    from later requirements). Over `max_requirements`: facets are merged first, then the list is cut (recorded)."""
    cap = int(o["max_requirements"])
    ents = entities if len(entities) > 1 else [None]
    pers = periods if len(periods) > 1 else [None]
    fac = [(i + 1, facet_terms[i]) for i in range(len(facets))] or [(None, [])]

    def build(fac_list):
        out = []
        for fi, terms in fac_list:
            for e in ents:
                for p in pers:
                    r = {"id": "|".join(x for x in ("F%d" % fi if fi else "", ("e:" + e) if e else "",
                                                     ("p:" + p) if p else "") if x) or "q"}
                    if terms:
                        r["terms"] = list(terms)
                    if e:
                        r["entity"] = e
                    if p:
                        r["period"] = p
                    if len(r) > 1:
                        out.append(r)
        return out

    reqs = build(fac)
    info = {"merged_facets": False, "truncated": 0}
    if len(reqs) > cap and len(fac) > 1:
        merged = [(None, list(dict.fromkeys(t for _fi, ts in fac for t in ts)))]
        reqs = build(merged)
        info["merged_facets"] = True
    if len(reqs) > cap:
        info["truncated"] = len(reqs) - cap
        reqs = reqs[:cap]
    return reqs, info
