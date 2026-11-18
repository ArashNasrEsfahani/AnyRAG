"""AGENTIC4's graph retrieval: seed facts by NAME in the label layer, walk each seed's family to the other target
reports, and hand back candidates in the exact shape the v3 verifier, merge and pack already read. No model call.

    cands, info = graph_candidates(con, reader, plan, doc_ids, d_lang="fa", opts=o)

WHAT IS DIFFERENT FROM v3's DATABASE PATH. v3 regexes the RAW fact label (`facts.label_fold`): the extractor's
label words, with the Persian zero-width-joiner split and the junk hubs. The graph path searches the label layer's
short metric and entity NAMES through their solid key, so «نرم‌افزار», «نرم افزار» and «نرمافزار» are one term, and a
table cell is found by its column header («سهم از واردات») or its row entity («ایران») rather than by the glued
string. It is additive: v3's pool is still built and merged, because a quarter of all facts have no clean metric
name (label_quality != ok) and only the raw-label and vector paths reach those.

  1. card routing   only when the scope is BROAD (more target documents than `cards_route_min_docs`) and the plan
                    names no company: the document cards (graph.py) pick the documents to open. In the benchmark
                    every question carries its doc_scope, so this fires rarely there; it is the path a user asking
                    across the whole knowledge base takes.
  2. seeds          facts whose metric or entity name carries the plan's words, scored by WORD COVERAGE - the share
                    of the plan's distinct content words the name holds - not by term count: the planner emits the
                    phrase «قیمت محصول» AND its words «قیمت», «محصول», and counting terms would score one match
                    three times. Plus the typed kind and year scores v3 uses (kb/query.py), a table/KPI bonus, the
                    QUALIFIER test and the hub penalty below.
  3. family walk    a timeline or comparison plan (or any plan over several reports): every target report gets the
                    members of the seed families that it holds, matched SOFTLY - by the overlap of the families'
                    words, `family_min_overlap` - because each year's report phrases a metric its own way
                    («تعداد راننده‌های اسنپ» in 1398, «کاربران راننده» in 1400). This is the cross-document hop a
                    series needs, done in SQL over graph_facts.
  4. qualifiers     the plan's own qualifiers (graph.quals_of over its metric and terms: active, registered, gross,
                    net, ...) against each fact's node: a fact whose node carries them gains `w_g_qual_match`, one
                    whose node carries a DIFFERENT qualifier loses `w_g_qual_other`. A question about ACTIVE drivers
                    prefers the active node, and the other-definition facts are still returned - they are what the
                    reconciliation needs to see the definition change.
  5. hub penalty    a family spanning more than `graph_hub_docs` documents is not walked, and its seeds lose
                    `w_g_hub` per document above the cap.

Candidates carry features["src"] = ["graph"] and features["graph"] = {node, family, quals, cover, hits, walk}, so
the merge can reserve their slots and the trace can say why each one is there.
"""
import time
from collections import defaultdict

from ...kb import graph as G
from ...kb.concepts import NameVectors
from ...kb import labels as L
from ...kb import query as Q
from ..agentic2 import tree as T

GRAPH_RET_VERSION = "agentic4-graphret-v3"    # v3: new-windows merge, concept fallback, kind and restatement drops (v2: concepts)


def _ms(t0):
    return round((time.perf_counter() - t0) * 1000.0, 1)


def plan_terms(plan, d_lang):
    """The plan's terms in the documents' language (the other language when that one is empty), deduped."""
    t = plan.get("terms") or {}
    out = list(t.get(d_lang) or []) or list(t.get("fa" if d_lang == "en" else "en") or [])
    return list(dict.fromkeys(x for x in out if x and x.strip()))


def content_words(terms):
    """The plan's distinct content words, folded: the unit WORD COVERAGE is measured in."""
    ws = []
    for t in terms:
        for w in L._f(t).split():
            if w not in L._STOP and w not in L.GENERIC and len(w) > 1 and w not in ws:
                ws.append(w)
    return ws


def concept_texts(plan, d_lang, k=3):
    """The texts embedded to look for metric NAMES: the plan's metric and its first terms in the documents' language
    (the probe took the max over the terms, and so does `NameVectors.nearest`)."""
    out = [plan.get("metric") or ""] + plan_terms(plan, d_lang)[:k]
    return [t for t in dict.fromkeys(x.strip() for x in out) if t]


def concept_seeds(con, mats, qvecs, doc_ids, o):
    """-> {fact_id: sim}: clean facts whose NAME is among the names closest in meaning to the plan's terms."""
    hits = NameVectors.nearest({d: mats[d] for d in doc_ids if d in mats}, qvecs,
                               per_doc=int(o["concept_per_doc"]), min_sim=float(o["concept_min_sim"]))
    out = {}
    for d, mf, sim in hits:
        for r in con.execute("SELECT fact_id FROM fact_labels WHERE doc_id = ? AND metric_fold = ? "
                             "AND label_quality = 'ok'", (d, mf)):
            out[r["fact_id"]] = max(out.get(r["fact_id"], 0.0), sim)
    return out


def concept_walk(con, seeds, names_of, mats, doc_ids, o):
    """-> {fact_id: (seed name, sim)}: from each top seed's NAME, the closest name in every OTHER target report -
    the cross-year hop by meaning, for reports that rename the metric («پرسنل» 1398, «افراد شاغل» 1400)."""
    out = {}
    done = set()
    for fid, doc in seeds:
        mf = names_of.get(fid)
        if not mf or (doc, mf) in done:
            continue
        done.add((doc, mf))
        v = NameVectors.vector_of(mats, doc, mf)
        if v is None:
            continue
        for d in doc_ids:
            if d == doc:
                continue
            hit = NameVectors.closest_in(mats, d, v, float(o["concept_walk_min_sim"]))
            if hit is None:
                continue
            for r in con.execute("SELECT fact_id FROM fact_labels WHERE doc_id = ? AND metric_fold = ? "
                                 "AND label_quality = 'ok'", (d, hit[0])):
                if hit[1] > out.get(r["fact_id"], ("", 0.0))[1]:
                    out[r["fact_id"]] = (mf, hit[1])
        if len(done) >= int(o["graph_max_families"]):
            break
    return out


def plan_quals(plan, terms):
    return set(G.quals_of(" ".join([plan.get("metric") or ""] + list(terms))))


def _fact_rows(con, fact_ids):
    out = {}
    ids = list(fact_ids)
    for i in range(0, len(ids), 900):
        chunk = ids[i:i + 900]
        for r in con.execute(Q.FACT_SELECT + " WHERE f.fact_id IN (%s)" % ",".join("?" * len(chunk)), chunk):
            out[r["fact_id"]] = dict(r)
    return out


def _names(con, fact_ids):
    out = {}
    ids = list(fact_ids)
    for i in range(0, len(ids), 900):
        chunk = ids[i:i + 900]
        for r in con.execute("SELECT fact_id, doc_id, metric_fold, metric_solid, entity_fold FROM fact_labels "
                             "WHERE fact_id IN (%s)" % ",".join("?" * len(chunk)), chunk):
            out[r["fact_id"]] = dict(r)
    return out


def doc_years(con, doc_ids):
    """-> {doc_id: (year_from, year_to, calendar)}: the period each report is ABOUT."""
    ids = list(doc_ids)
    if not ids:
        return {}
    out = {}
    for r in con.execute("SELECT doc_id, year_from, year_to, calendar FROM documents WHERE doc_id IN (%s)"
                         % ",".join("?" * len(ids)), ids):
        if r["year_from"] is not None:
            out[r["doc_id"]] = (r["year_from"], r["year_to"] if r["year_to"] is not None else r["year_from"],
                                r["calendar"] or Q.calendar_of(r["year_from"]))
    return out


def restated(row, years, doc_ids):
    """True when a fact of one target report is about a single year ANOTHER target report covers: the later report
    restating the earlier year (the 1400 report's «1399: 18%» against the 1399 report's own 20%). For a series that
    year's own report is the authority, which is also the rule the gold answers follow. Same calendar only; a
    multi-year fact (1397-1400, cumulative) is not a restatement of one year."""
    y = row.get("year")
    own = years.get(row["doc_id"])
    if y is None or own is None or (row.get("year_to") not in (None, y)):
        return False
    if (row.get("calendar") or Q.calendar_of(y)) != own[2] or own[0] <= y <= own[1]:
        return False
    return any(d != row["doc_id"] and d in years and years[d][2] == own[2] and years[d][0] <= y <= years[d][1]
               for d in doc_ids)


def coverage(words, metric_fold, metric_solid, idf=None):
    """Share of the plan's content words the metric name holds - as a word, or inside its solid key (a compound
    written the other way) - each word weighted by its IDF over the metric names when `idf` is given. Unweighted,
    «تولید» ("production") counted as much as «خودرو» ("car"), and a generic production row outranked the
    question's own metric on dev."""
    if not words or not metric_fold:
        return 0.0
    toks = set(metric_fold.split())
    wt = {w: (idf.get(w, idf.get("__max__", 1.0)) if idf else 1.0) for w in words}
    hit = sum(wt[w] for w in words if w in toks or (len(w) >= 3 and w in (metric_solid or "")))
    return hit / (sum(wt.values()) or 1.0)


_IDF = {}


def word_idf(con):
    """IDF of every word over the metric names (log N / df), cached per connection file."""
    key = id(con)
    if key not in _IDF:
        import math
        from collections import Counter
        df, n = Counter(), 0
        for r in con.execute("SELECT metric_fold FROM fact_labels WHERE metric_fold IS NOT NULL"):
            n += 1
            df.update(set(r["metric_fold"].split()))
        idf = {w: math.log(1.0 + n / c) for w, c in df.items()}
        idf["__max__"] = math.log(1.0 + n)
        _IDF[key] = idf
    return _IDF[key]


_COMPANY = {}


def company_words(con, doc_ids):
    """The folded words naming the companies of these reports: their `company` key, and the name words graph.py
    finds in every title of a company's reports (Snapp -> «اسنپ»)."""
    key = id(con)
    if key not in _COMPANY:
        _COMPANY[key] = (G.company_tokens(con), {r["doc_id"]: r["company"] for r in con.execute(
            "SELECT doc_id, company FROM documents")})
    names, by_doc = _COMPANY[key]
    out = set()
    for d in doc_ids:
        c = by_doc.get(d)
        if c:
            out |= set(names.get(c) or ())
            out |= set(L._f(c).split())
    return out


def entity_match(ent_terms, entity_fold, entity_solid):
    """-> 1 when a plan entity is the fact's entity, -1 when the fact names a DIFFERENT entity, 0 otherwise."""
    if not ent_terms or not entity_fold:
        return 0
    for t in ent_terms:
        f, s = L.term_forms(t)
        if s and (L._word_hit(f, entity_fold) or s in (entity_solid or "")):
            return 1
    return -1


def score_row(row, name, grow, marks, words, pquals, plan, o, idf=None, concept=0.0):
    """-> (score, features) of one graph seed. `concept` is the cosine of its NAME to the plan's terms."""
    f = {}
    cov = coverage(words, name.get("metric_fold"), name.get("metric_solid"), idf if o["graph_idf"] else None)
    word_hits = sum(1 for _t, h in marks.get("metric", []) if h == "word")
    solid_hits = sum(1 for _t, h in marks.get("metric", []) if h == "solid")
    ent_hits = len(marks.get("entity", []))
    s = (float(o["w_g_cover"]) * cov + float(o["w_g_word"]) * min(word_hits, 3)
         + float(o["w_g_solid"]) * min(solid_hits, 2) + float(o["w_g_entity"]) * min(ent_hits, 2))
    if concept:
        s += float(o["w_g_concept"]) * concept
        f["concept"] = round(concept, 3)
    em = entity_match(plan.get("entity_terms") or [], name.get("entity_fold"), L.solid(name.get("entity_fold") or ""))
    if em > 0:
        s += float(o["w_g_entity_match"])
    elif em < 0:
        s += float(o["w_g_entity_other"])          # another row of the same table: Iran's line, not Saudi Arabia's
    f.update(cover=round(cov, 3), word_hits=word_hits, solid_hits=solid_hits, entity_hits=ent_hits, entity=em)
    ks, f["kind"] = Q.kind_score(row, plan.get("unit") or "count", o)
    ys, f["year"] = Q.year_score(row, plan.get("period"), o)
    s += ks + ys
    src = row.get("source_kind") or "text"
    s += float(o["w_source_table"]) if src == "table" else (float(o["w_source_kpi"]) if src == "kpi" else 0.0)
    quals = set((grow or {}).get("quals", "").split(",")) - {""}
    if pquals:
        if pquals <= quals:
            s += float(o["w_g_qual_match"])
            f["quals"] = "match"
        elif quals - pquals:
            s += float(o["w_g_qual_other"])
            f["quals"] = "other"
    fam_docs = int((grow or {}).get("family_docs") or 1)
    hub = int(o["graph_hub_docs"])
    if fam_docs > hub:
        s += float(o["w_g_hub"]) * (fam_docs - hub)
        f["hub"] = fam_docs
    return round(s, 4), f


def family_walk(con, seeds, target_docs, d_lang, o):
    """-> {fact_id: (seed family, overlap)} for facts of the target reports whose family overlaps a seed family.

    Soft, because each year phrases one metric its own way. One SQL LIKE per seed-family word finds the members
    that share at least that word; the overlap is then computed exactly and must reach `family_min_overlap`."""
    fams = []
    for fam, fam_docs in seeds:
        if fam and fam_docs <= int(o["graph_hub_docs"]) and fam not in fams:
            fams.append(fam)
        if len(fams) >= int(o["graph_max_families"]):
            break
    if not fams or not target_docs:
        return {}
    ph = ",".join("?" * len(target_docs))
    found = {}
    for fam in fams:
        for w in fam.split():
            for r in con.execute("SELECT fact_id, family FROM graph_facts WHERE lang = ? AND doc_id IN (%s) "
                                 "AND (' ' || family || ' ') LIKE ?" % ph, (d_lang, *target_docs, "% " + w + " %")):
                ov = G.family_overlap(fam, r["family"])
                if ov >= float(o["family_min_overlap"]) and ov > found.get(r["fact_id"], ("", 0.0))[1]:
                    found[r["fact_id"]] = (fam, ov)
    return found


def is_graph(cand):
    return "graph" in ((cand.features or {}).get("src") or [])


def graph_slots(opts, doc_ids, unit_kind):
    """The graph path's reserved slots for one plan: none when it expects text, else the single / multi value."""
    if unit_kind == "text":
        return int(opts["graph_text_slots"])
    return int(opts["graph_min_slots_multi" if len(doc_ids) > 1 else "graph_min_slots"])


def _overlaps(a, b):
    return a.doc_id == b.doc_id and bool(set(a.unit_ids or ()) & set(b.unit_ids or ()))


def merge_graph(pool, doc_ids, facets, opts, cap, per_period=True, slots=None):
    """v3's selection, plus at most `slots` windows v3 did NOT select, from the graph. -> (selected, info).

    WHY (dev GPU run, 2026-09-21, work/analysis/V4_DEV.md). The first merge ("legacy" below) ranked the graph's
    candidates together with v3's. A graph score is not a v3 score - with the concept term it reached 4-7 against
    v3's ~3.5 - and `rank_candidates` keeps the HIGHER-scored copy of a duplicate, so the graph re-ranked even the
    windows v3 found and filled 3-12 of the 24 multi-report slots against the 4 reserved. What it crowded in were
    near-miss measures (a count of rejected ads for the rejection RATE, ads for advertisers), and the 4B answerer
    used them. Here the two scores are never compared:

      1. v3's merge over v3's pool at the full cap: what v3 alone would keep
      2. the graph's candidates that overlap none of those windows are the NEW ones; up to `slots` of them are
         taken, one per report first (a series needs every year), then by graph score
      3. v3's merge over v3's pool again, at the cap minus what the graph took: v3's own order, unchanged
      4. a v3 window the graph ALSO found keeps v3's score and gets the graph's node and qualifiers attached
         (features["graph"]), which is what `annotate_rows` reads to give a verified row its definition

    So the graph can add evidence v3 missed, and cannot reorder, displace beyond its slots, or re-score v3's."""
    from ..agentic3 import retrieve as RT
    if not any(is_graph(c) for c in pool):
        out, info = RT.merge(pool, doc_ids, facets, opts, cap)   # no graph candidate: v3's merge, exactly
        info.update(graph_slots=0, graph_in_selection=0)
        return out, info
    if opts.get("graph_merge_mode", "new_windows") == "legacy":
        return _merge_legacy(pool, doc_ids, facets, opts, cap, per_period, slots)
    base = [c for c in pool if not is_graph(c)]
    g = Q.rank_candidates([c for c in pool if is_graph(c)])
    multi = len(doc_ids) > 1
    want = min(cap, int(slots if slots is not None else opts["graph_min_slots_multi" if multi else "graph_min_slots"]))
    full, _i = RT.merge(list(base), doc_ids, facets, opts, cap)
    new = [c for c in g if not any(_overlaps(c, s) for s in full)]
    picked, taken = [], set()
    if multi and per_period:
        for d in doc_ids:
            if len(picked) >= want:
                break
            c = next((x for x in new if x.doc_id == d and id(x) not in taken), None)
            if c is not None:
                picked.append(c)
                taken.add(id(c))
    for c in new:
        if len(picked) >= want:
            break
        if id(c) not in taken:
            picked.append(c)
            taken.add(id(c))
    sel, info = RT.merge(base, doc_ids, facets, opts, max(0, cap - len(picked)))
    confirmed = 0
    for s in sel:
        m = next((c for c in g if _overlaps(c, s)), None)
        if m is not None and "graph" not in (s.features or {}):
            s.features["graph"] = dict(m.features.get("graph") or {}, confirmed=True)
            confirmed += 1
    out = picked + sel
    for i, c in enumerate(out, 1):
        c.cid = "c%d" % i
        c.verdict = None
    info.update(graph_slots=len(picked), graph_in_selection=len(picked), graph_confirmed=confirmed,
                graph_new=len(new), graph_pool=len(g), selected=len(out), cap=cap, pool=len(pool))
    return out, info


def _merge_legacy(pool, doc_ids, facets, opts, cap, per_period=True, slots=None):
    """graphret-v2's merge, kept for the replay's comparison (graph_merge_mode="legacy"): graph slots reserved
    first, then v3's merge over everything else, the graph's candidates included and ranked by their own score."""
    from ..agentic3 import retrieve as RT
    ranked = Q.rank_candidates(list(pool))
    multi = len(doc_ids) > 1
    want = int(slots if slots is not None else opts["graph_min_slots_multi" if multi else "graph_min_slots"])
    g = [c for c in ranked if is_graph(c)]
    picked, taken = [], set()
    if multi and per_period:
        for d in doc_ids:
            if len(picked) >= want:
                break
            c = next((x for x in g if x.doc_id == d and id(x) not in taken), None)
            if c is not None:
                picked.append(c)
                taken.add(id(c))
    for c in g:
        if len(picked) >= min(want, cap):
            break
        if id(c) not in taken:
            picked.append(c)
            taken.add(id(c))
    rest = [c for c in ranked if id(c) not in taken]
    sel, info = RT.merge(rest, doc_ids, facets, opts, max(0, cap - len(picked)))
    out = picked + sel
    for i, c in enumerate(out, 1):
        c.cid = "c%d" % i
        c.verdict = None
    info.update(graph_slots=len(picked), graph_in_selection=sum(1 for c in out if is_graph(c)),
                selected=len(out), cap=cap, pre_cap=len(ranked), pool=len(pool))
    return out, info


def graph_candidates(con, reader, plan, doc_ids, *, d_lang="fa", opts, tree=True, family=True, cards=True,
                     concepts=None):
    """The graph path for one question. -> (candidates, info). Candidates are NOT capped: the merge decides."""
    o = opts
    t0 = time.perf_counter()
    terms = plan_terms(plan, d_lang)
    ent_terms = [t for t in (plan.get("entity_terms") or []) if t and t.strip()]
    words = content_words(terms)
    pq = plan_quals(plan, terms)
    info = {"version": GRAPH_RET_VERSION, "terms": terms, "words": words, "plan_quals": sorted(pq),
            "docs": list(doc_ids)}
    # the graph's hop is ACROSS reports: a one-report plan - a comparison of two sections included, which the dev
    # replay showed it hurting (multi-section 0.606 -> 0.556) - keeps v3's pool untouched
    single = len(doc_ids) <= 1
    if single and not o["graph_single_doc"]:
        info.update(skipped="one_report", seeds=0, walked=0, candidates=0, ms=_ms(t0))
        return [], info
    if not doc_ids or not (terms or ent_terms):
        info.update(seeds=0, walked=0, candidates=0, ms=_ms(t0))
        return [], info

    routed = list(doc_ids)
    if cards and len(doc_ids) > int(o["cards_route_min_docs"]) and not plan.get("companies"):
        ranked = G.route_documents(con, terms + ent_terms, doc_ids=doc_ids, lang=d_lang)
        keep = [d for d, s, _h in ranked if s > 0][:int(o["cards_route_k"])]
        if keep:
            routed = keep
        info["cards"] = {"routed": routed, "top": [(d, s) for d, s, _h in ranked[:8]]}

    idf = word_idf(con) if o["graph_idf"] else None
    # The company of the target reports is not an entity to look for - every fact in Divar's report is about Divar -
    # and a MEASURE must match for a fact to be a seed. A first version searched metric names with the entity terms
    # too, and for "how many people did Divar employ" the company name «دیوار» filled all sixty seed slots with
    # Divar's paper use, city count and app opens, while the headcount facts never got a seat.
    own = company_words(con, routed)
    terms = [t for t in terms if not set(L._f(t).split()) <= own]
    words = [w for w in words if w not in own]
    ent_terms = [t for t in ent_terms if not set(L._f(t).split()) & own]
    info.update(terms=terms, words=words, entity_terms=ent_terms, company_words=sorted(own))
    marks = G.match_facts(con, routed, terms)
    if ent_terms:
        for fid, m in G.match_facts(con, routed, ent_terms).items():
            if m["entity"]:
                marks.setdefault(fid, {"metric": [], "entity": []})["entity"] += m["entity"]
    csim = {}
    fallback = bool(o["concept_fallback_only"])
    if concepts is not None and o["graph_concepts"]:
        mats, qvecs = concepts
        csim = concept_seeds(con, mats, qvecs, routed, o)
        if fallback and csim:
            # MEANING ONLY WHERE WORDS FAIL. On dev a concept seed next to good lexical seeds was a near-miss
            # measure («آگهی», ads, for «آگهی‌گذار», advertisers); the case concepts exist for is a report that
            # names the metric in other words altogether («افراد شاغل» for «نیروی کار»). So a report whose lexical
            # seeds already cover `concept_fallback_cover` of the plan's words gets no concept seed.
            lex = [fid for fid, m in marks.items() if m["metric"]]
            ln = _names(con, lex)
            covered = {n["doc_id"] for n in ln.values()
                       if coverage(words, n.get("metric_fold"), n.get("metric_solid"),
                                   idf if o["graph_idf"] else None) >= float(o["concept_fallback_cover"])}
            cn = _names(con, csim)
            csim = {fid: s for fid, s in csim.items() if (cn.get(fid) or {}).get("doc_id") not in covered}
            info["concept_covered_docs"] = len(covered)
        for fid in csim:
            marks.setdefault(fid, {"metric": [], "entity": []})
        info["concept_seeds"] = len(csim)
    marks = {fid: m for fid, m in marks.items() if m["metric"] or fid in csim}   # the measure must match, in words or meaning
    grows = G.graph_rows(con, marks)
    names = _names(con, marks)
    rows = _fact_rows(con, marks)
    multi_docs = len(routed) > 1
    years = doc_years(con, routed) if (multi_docs and o["graph_drop_restated"]) else {}
    dropped = {"kind_conflict": 0, "restated": 0}

    def keep(row, f):
        """The graph adds only evidence that can be the answer: a unit kind that does not contradict the plan (a
        count for a percent question was the dev failure), and a year's value from that year's own report."""
        if o["graph_drop_kind_conflict"] and f.get("kind") == "conflict":
            dropped["kind_conflict"] += 1
            return False
        if years and restated(row, years, routed):
            dropped["restated"] += 1
            return False
        return True

    scored = []
    for fid, row in rows.items():
        # a concept score only for a seed that has no word match (with `concept_fallback_only`); before, a lexical
        # seed that was also near in meaning scored both, which is how the graph's scores outgrew v3's
        cs = csim.get(fid, 0.0) if not (fallback and marks[fid]["metric"]) else 0.0
        s, f = score_row(row, names.get(fid, {}), grows.get(fid), marks[fid], words, pq, plan, o, idf, concept=cs)
        if not keep(row, f):
            continue
        f["walk"] = False
        scored.append((s, fid, f))
    scored.sort(key=lambda x: (-x[0], x[1]))
    scored = scored[:int(o["graph_max_seeds"])]
    info["seeds"] = len(scored)
    info["matched"] = len(marks)

    multi = len(routed) > 1 or plan.get("kind") in ("timeline", "comparison")
    walked = {}
    if family and multi and scored:
        fam_rank = []
        for _s, fid, _f in scored:
            g = grows.get(fid)
            if g and g["family"] not in [x[0] for x in fam_rank]:
                fam_rank.append((g["family"], int(g.get("family_docs") or 1)))
        walked = family_walk(con, fam_rank, routed, d_lang, o)
        cwalk = {}
        if concepts is not None and o["graph_concepts"]:
            top = [(fid, rows[fid]["doc_id"]) for _s, fid, _f in scored[:int(o["graph_max_families"]) * 2]]
            mf_of = {fid: names.get(fid, {}).get("metric_fold") for _s, fid, _f in scored}
            fam_of_mf = {mf_of[fid]: (grows.get(fid) or {}).get("family") for fid, _d in top}
            cw = concept_walk(con, top, mf_of, concepts[0], routed, o)
            for fid, (seed_mf, sim) in cw.items():
                cwalk[fid] = sim
                if sim > walked.get(fid, ("", 0.0))[1]:
                    walked[fid] = (fam_of_mf.get(seed_mf) or seed_mf, sim)   # credited to the SEED's family
            info["concept_walked"] = len(cw)
        have = {fid for _s, fid, _f in scored}
        extra = [fid for fid in walked if fid not in have]
        if extra:
            best_by_fam = {}
            for s, fid, _f in scored:
                g = grows.get(fid)
                if g and g["family"] not in best_by_fam:
                    best_by_fam[g["family"]] = s
            erows = _fact_rows(con, extra)
            egrows = G.graph_rows(con, extra)
            enames = _names(con, extra)
            for fid in extra:
                row = erows.get(fid)
                if row is None:
                    continue
                fam, ov = walked[fid]
                # a walked fact is credited through its seed (w_g_walk x overlap-or-similarity x the seed's score),
                # not by a concept score of its own as well
                s, f = score_row(row, enames.get(fid, {}), egrows.get(fid), {}, words, pq, plan, o, idf,
                                 concept=0.0 if fallback else cwalk.get(fid, 0.0))
                if not keep(row, f):
                    continue
                s = round(s + float(o["w_g_walk"]) * ov * max(best_by_fam.get(fam, 0.0), 0.0), 4)
                f.update(walk=True, overlap=round(ov, 3), from_family=fam)
                scored.append((s, fid, f))
                rows[fid] = row
                grows[fid] = egrows.get(fid)
                names[fid] = enames.get(fid, {})
            scored.sort(key=lambda x: (-x[0], x[1]))
    info["walked"] = len(walked)
    info["dropped"] = dict(dropped)

    ctx = T.search_context(plan, d_lang)
    unit_kind = plan.get("unit") or "count"
    cands = []
    for s, fid, f in scored:
        row = rows[fid]
        c = Q.expand_fact(reader, row, ctx, o, unit_kind, tree=tree)
        if c is None:
            continue
        g = grows.get(fid) or {}
        c.score = s
        c.features = {"src": ["graph"], "db": f, "facet": [],
                      "graph": {"node": g.get("node_id"), "family": g.get("family"), "quals": g.get("quals"),
                                "family_docs": g.get("family_docs"), "walk": f.get("walk", False)},
                      "fact": {"fact_id": fid, "label": row["label"], "value": row["value_text"],
                               "unit": row["unit_text"], "unit_kind": row["unit_kind"], "year": row["year"],
                               "def_class": row["def_class"], "source_kind": row["source_kind"]}}
        cands.append(c)
    per_doc = defaultdict(int)
    for c in cands:
        per_doc[c.doc_id] += 1
    info.update(candidates=len(cands), per_doc=dict(per_doc), ms=_ms(t0))
    return cands, info
