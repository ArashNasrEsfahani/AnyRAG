"""The fact graph: metric nodes, families, cross-language links and document cards - built from the label layer with
no model, rebuilt in seconds, queried by regex and SQL.

WHAT IT IS. The label layer (labels.py) gives every fact a short metric name, an entity name and definition tags.
This module turns those into a graph an agent can walk:

  family     the measure OBJECT, qualifiers taken out and words light-stemmed: «تعداد رانندگان فعال»,
             «رانندگان ثبت‌شده» and «رانندگان» are all the family «راننده» (drivers).
  node       a family + its QUALIFIERS (the QUAL_TAGS of the fact's own label): drivers|active and
             drivers|registered are two nodes of ONE family. That is the definition-drift question made
             structural - a series whose years sit on different nodes of the same family is not comparable, and
             the graph says so before any model reads a number.
  link       fa <-> en between nodes of the two editions of one report, found by NUMERIC ANCHORS: a value, year and
             unit that occur exactly once on each side name the same fact, so the two labels name the same node
             (measured before the build: 2,597 pairs from the bilingual corpora, 265 with two or more anchors).
  card       one short text per document - title, company, years, section titles, its most frequent metric and
             entity names - the surface catalog routing scores (2 ms for all 129 documents on dev: the right
             document top-3 84%).
  edges      are rows, not a separate table: two facts sharing a family ARE a cross-document bridge, found through
             the family index; the section tree (sections.parent_id) is the containment graph; graph_facts.company
             and year are the entity and period axes.

NO MODEL, ADDITIVE, VERSIONED - the same discipline as the label layer. Nothing here changes `facts`, `tri`,
`fact_labels` or `schema.component_versions`, so v3 is untouched. GRAPH_VERSION lives in meta (`graph.version`);
ingestion marks the graph dirty (`graph.dirty`) and `ensure_current` rebuilds it when it is dirty or stale. A build
is a whole-corpus pass because stems are validated against the corpus vocabulary: a new document can change a stem.

LIGHT STEMMING, CORPUS-VALIDATED. fold() already splits «شبکه‌های» into «شبکه های», and «ها / های / هایی» are dropped.
For «ان / گان / یان / ات» and English «s / es / ies» the stem is taken ONLY if the stem itself occurs as a word in
the corpus's metric names: «کاربران» -> «کاربر» (it occurs), «رانندگان» -> «راننده», but «میزان» stays «میزان»
because «میز» never occurs. Place names are never stemmed («تهران», «ایران»). This is what merges کاربر/کاربران, the
two top "separate" nodes of the first label build.

QUALIFIERS. QUAL_TAGS are the definition tags that change WHAT a number counts (active, registered, gross, net,
consolidated, target, annual, share, growth, ...). Their keyword spans are cut out of the metric name before the
family is formed, using labels._TAG_RX - the same compiled, boundary-guarded patterns that set def_tags - so a
multi-word qualifier such as «ثبت شده» leaves the family whole. Measure-family tags (users, revenue, sales, capacity,
production, profit, cost) are NOT qualifiers: they are the object and stay in the family.

HUB PENALTY. graph_families.n_docs is the document frequency of a family; the retrieval that walks the graph skips
or down-weights families above a cap. After the label layer removed the junk hubs the largest family spans fifteen
documents, not fifty.
"""
import json
import math
import re
import time
from collections import Counter, defaultdict

from . import labels as L

GRAPH_VERSION = "kb-graph-v2"     # v2: company names read from letter words (Divar's «(دیوار)» title)

QUAL_TAGS = ("active", "inactive", "registered", "total", "cumulative", "new", "accounts", "fulltime", "parttime",
             "subgroup", "gross", "net", "operating", "consolidated", "standalone", "target", "actual", "nominal",
             "adjusted", "annual", "half", "quarterly", "monthly", "daily", "average", "median", "max", "min",
             "per_capita", "share", "ratio", "growth", "domestic", "export", "import")
ANCHOR_MAX_GLOBAL = 30          # a value occurring more often than this anywhere is not an anchor (round numbers)
CARD_MAX_METRICS = 150
CARD_MAX_ENTITIES = 50

DDL = """
CREATE TABLE IF NOT EXISTS graph_facts (
  fact_id INTEGER PRIMARY KEY REFERENCES facts(fact_id) ON DELETE CASCADE,
  doc_id TEXT NOT NULL, lang TEXT, company TEXT, node_id TEXT, family TEXT, quals TEXT, entity_key TEXT,
  year INTEGER);
CREATE INDEX IF NOT EXISTS ix_graph_facts_family ON graph_facts (family, doc_id);
CREATE INDEX IF NOT EXISTS ix_graph_facts_node ON graph_facts (node_id);
CREATE INDEX IF NOT EXISTS ix_graph_facts_doc ON graph_facts (doc_id);
CREATE TABLE IF NOT EXISTS graph_nodes (
  node_id TEXT PRIMARY KEY, lang TEXT, family TEXT, quals TEXT, name TEXT, n_docs INTEGER, n_facts INTEGER,
  n_companies INTEGER, year_min INTEGER, year_max INTEGER);
CREATE INDEX IF NOT EXISTS ix_graph_nodes_family ON graph_nodes (family);
CREATE TABLE IF NOT EXISTS graph_families (
  family TEXT, lang TEXT, n_docs INTEGER, n_nodes INTEGER, n_facts INTEGER, n_companies INTEGER,
  PRIMARY KEY (family, lang));
CREATE TABLE IF NOT EXISTS graph_links (
  node_a TEXT, node_b TEXT, kind TEXT, weight INTEGER, PRIMARY KEY (node_a, node_b, kind));
CREATE TABLE IF NOT EXISTS doc_cards (
  doc_id TEXT PRIMARY KEY REFERENCES documents(doc_id) ON DELETE CASCADE, lang TEXT, company TEXT,
  card_fold TEXT, card_solid TEXT, n_parts INTEGER);
"""


def ensure_graph_schema(con):
    L.ensure_labels_schema(con)
    con.executescript(DDL)


# ---------------------------------------------------------------------------------------------------------------
# families
# ---------------------------------------------------------------------------------------------------------------

# Dropped from a family: stop and quantity words, period words («سال» is when, not what), the plural markers fold()
# splits off, and the detached ezafe «ی» («راننده ی اسنپ»).
_DROP = L._STOP | L.GENERIC | L._PERIOD | {"ها", "های", "هایی", "ی", "ای"}
_TITLE_GENERIC = {L._f(w) for w in """گزارش عملکرد سالانه سالیانه گروه شرکت اقتصادی نیمه اول دوم ویژه نامه
report annual performance group company economic half edition review summary""".split()}


def stem(tok, vocab):
    """Light, corpus-validated stem of one folded token (see the module docstring)."""
    if len(tok) < 4 or tok in L.PLACE or not tok.isalpha():
        return tok
    cands = []
    if tok.endswith("گان"):
        cands.append(tok[:-3] + "ه")
    if tok.endswith("یان"):
        cands.append(tok[:-3] + "ی")
    if tok.endswith("ان"):
        cands.append(tok[:-2])
    if tok.endswith("ات"):
        cands += [tok[:-2] + "ه", tok[:-2]]
    if tok.isascii():
        if tok.endswith("ies"):
            cands.append(tok[:-3] + "y")
        if tok.endswith("es"):
            cands.append(tok[:-2])
        if tok.endswith("s") and not tok.endswith("ss"):
            cands.append(tok[:-1])
    for c in cands:
        if len(c) >= 3 and vocab.get(c, 0) > 0:
            return c
    return tok


def quals_of(label):
    """The qualifier tags of a fact's OWN label (not the window around it): what the number counts."""
    tags = set(L.def_tags_of(label))
    return tuple(t for t in QUAL_TAGS if t in tags)


def family_of(metric_fold, vocab, company_tokens=()):
    """The measure object: qualifier spans cut out, quantity / period / stop words and the company's own name
    dropped, light-stemmed, order-free. -> '' when nothing measurable is left (the fact then stays out of the graph
    and is reached by the raw-label and vector paths instead).

    The company's name is dropped because it is the ENTITY axis (graph_facts.company), not the measure: Snapp's
    1398 report writes «تعداد راننده‌های اسنپ» and its 1403 report «کاربران راننده‌ی اسنپ», and with «اسنپ» in the
    family the two years could never meet."""
    core = metric_fold or ""
    for t in QUAL_TAGS:
        core = L._TAG_RX[t].sub(" ", core)
    drop = _DROP | set(company_tokens)
    toks = [x for x in core.split() if x not in drop and len(x) > 1]
    return " ".join(sorted({stem(x, vocab) for x in toks}))


def company_tokens(con):
    """company -> the words every one of its report titles shares (its name), for companies with two or more
    reports; generic title words («گزارش», «سالانه», "annual") are never part of a name."""
    titles = defaultdict(list)
    for r in con.execute("SELECT company, title_fold FROM documents WHERE company IS NOT NULL"):
        # letter words only: one Divar title writes the name «(دیوار)», and a whitespace split kept the brackets on,
        # so the name was missing from one title and the intersection lost it (Divar had no name in graph-v1)
        titles[r["company"]].append(set(re.findall(r"[^\W\d_]+", r["title_fold"] or "")))
    out = {}
    for c, ts in titles.items():
        if len(ts) >= 2:
            common = set.intersection(*ts) - _TITLE_GENERIC - L._STOP - L._PERIOD
            out[c] = {x for x in common if not x.isdigit()}
    return out


def family_overlap(a, b):
    """Jaccard overlap of two families' words: the SOFT bridge between years that phrase one metric differently."""
    sa, sb = set((a or "").split()), set((b or "").split())
    return len(sa & sb) / len(sa | sb) if sa and sb else 0.0


def node_id_of(lang, family, quals):
    return "%s|%s|%s" % (lang, family, ",".join(quals))


# ---------------------------------------------------------------------------------------------------------------
# building
# ---------------------------------------------------------------------------------------------------------------

def _vocab(con):
    v = Counter()
    for r in con.execute("SELECT metric_fold FROM fact_labels WHERE metric_fold IS NOT NULL"):
        v.update(r["metric_fold"].split())
    return v


def _anchor_links(con):
    """fa <-> en node links from numeric anchors between the two editions of one report. -> [(a, b, weight)]"""
    freq = Counter(round(r["value_num"], 4) for r in con.execute(
        "SELECT value_num FROM facts WHERE value_num IS NOT NULL"))
    pairs = [(r["doc_id"], r["pair_id"]) for r in con.execute(
        "SELECT doc_id, pair_id FROM documents WHERE lang = 'fa' AND pair_id IS NOT NULL")]
    links = Counter()
    for fa_doc, en_doc in pairs:
        side = {}
        for d in (fa_doc, en_doc):
            idx = defaultdict(set)
            for r in con.execute("""SELECT g.node_id, f.value_num, f.year, f.unit_norm FROM graph_facts g
                                    JOIN facts f USING (fact_id) WHERE g.doc_id = ? AND f.value_num IS NOT NULL""",
                                 (d,)):
                v = round(r["value_num"], 4)
                if freq[v] <= ANCHOR_MAX_GLOBAL:
                    idx[(v, r["year"], r["unit_norm"])].add(r["node_id"])
            side[d] = idx
        for key, fa_nodes in side[fa_doc].items():
            en_nodes = side[en_doc].get(key)
            if en_nodes and len(fa_nodes) == 1 and len(en_nodes) == 1:
                links[(next(iter(fa_nodes)), next(iter(en_nodes)))] += 1
    return [(a, b, w) for (a, b), w in links.items()]


def _cards(con):
    out = []
    docs = con.execute("SELECT doc_id, lang, company, title_fold, year_from, year_to FROM documents "
                       "WHERE status IN ('ready', 'vector_ready')").fetchall()
    for d in docs:
        parts = [d["title_fold"] or "", d["company"] or ""]
        parts += [str(y) for y in {d["year_from"], d["year_to"]} if y]
        parts += [r["title_fold"] for r in con.execute(
            "SELECT DISTINCT title_fold FROM sections WHERE doc_id = ? AND title_fold IS NOT NULL", (d["doc_id"],))]
        parts += [r["metric_fold"] for r in con.execute(
            """SELECT metric_fold, count(*) AS c FROM fact_labels WHERE doc_id = ? AND metric_fold IS NOT NULL
               GROUP BY metric_fold ORDER BY c DESC, metric_fold LIMIT ?""", (d["doc_id"], CARD_MAX_METRICS))]
        parts += [r["entity_fold"] for r in con.execute(
            """SELECT entity_fold, count(*) AS c FROM fact_labels WHERE doc_id = ? AND entity_fold IS NOT NULL
               GROUP BY entity_fold ORDER BY c DESC, entity_fold LIMIT ?""", (d["doc_id"], CARD_MAX_ENTITIES))]
        parts = [p for p in dict.fromkeys(" ".join(p.split()) for p in parts) if p]
        out.append((d["doc_id"], d["lang"], d["company"], " | ".join(parts),
                    " | ".join(L.solid(p) for p in parts), len(parts)))
    return out


def build(con, log=None):
    """Rebuild the whole graph from the label layer. -> summary. One transaction; a stop leaves the old graph."""
    from .schema import set_meta
    t0 = time.perf_counter()
    ensure_graph_schema(con)
    L.build(con)                                            # the label layer must be current first
    vocab = _vocab(con)
    names = company_tokens(con)
    companies = {r["doc_id"]: (r["company"], r["lang"]) for r in con.execute(
        "SELECT doc_id, company, lang FROM documents")}
    rows = []
    for r in con.execute("""SELECT l.fact_id, l.doc_id, l.metric_fold, l.entity_fold, f.label, f.year
                            FROM fact_labels l JOIN facts f USING (fact_id) WHERE l.label_quality = 'ok'"""):
        company, lang = companies.get(r["doc_id"], (None, None))
        fam = family_of(r["metric_fold"], vocab, names.get(company, ()))
        if not fam:
            continue
        q = quals_of(r["label"])
        ent = " ".join(sorted({stem(x, vocab) for x in (r["entity_fold"] or "").split() if x not in L._STOP})) or None
        rows.append((r["fact_id"], r["doc_id"], lang, company, node_id_of(lang, fam, q), fam, ",".join(q), ent,
                     r["year"]))
    with con:
        for t in ("graph_facts", "graph_nodes", "graph_families", "graph_links", "doc_cards"):
            con.execute("DELETE FROM %s" % t)
        con.executemany("INSERT INTO graph_facts VALUES (?,?,?,?,?,?,?,?,?)", rows)
        con.execute("""INSERT INTO graph_nodes
            SELECT g.node_id, g.lang, g.family, g.quals,
                   (SELECT l.metric_name FROM graph_facts g2 JOIN fact_labels l USING (fact_id)
                    WHERE g2.node_id = g.node_id GROUP BY l.metric_name ORDER BY count(*) DESC, l.metric_name LIMIT 1),
                   count(DISTINCT g.doc_id), count(*), count(DISTINCT g.company), min(g.year), max(g.year)
            FROM graph_facts g GROUP BY g.node_id""")
        con.execute("""INSERT INTO graph_families
            SELECT family, lang, count(DISTINCT doc_id), count(DISTINCT node_id), count(*), count(DISTINCT company)
            FROM graph_facts GROUP BY family, lang""")
        links = _anchor_links(con)
        con.executemany("INSERT INTO graph_links VALUES (?,?,?,?)", [(a, b, "translation", w) for a, b, w in links])
        con.executemany("INSERT INTO doc_cards VALUES (?,?,?,?,?,?)", _cards(con))
        set_meta(con, "graph.version", GRAPH_VERSION)
        set_meta(con, "graph.dirty", "0")
    s = summary(con)
    s["seconds"] = round(time.perf_counter() - t0, 2)
    if log:
        log("graph: %(facts)d facts -> %(nodes)d nodes in %(families)d families, %(links)d fa-en links, "
            "%(cards)d cards, %(seconds).1f s" % s)
    return s


def mark_dirty(con):
    from .schema import set_meta
    set_meta(con, "graph.dirty", "1")


def ensure_current(con, log=None):
    """Rebuild when ingestion marked the graph dirty or GRAPH_VERSION moved. -> summary or None when current."""
    from .schema import get_meta
    ensure_graph_schema(con)
    if get_meta(con, "graph.version") == GRAPH_VERSION and get_meta(con, "graph.dirty") == "0":
        return None
    return build(con, log=log)


def summary(con):
    def one(q):
        return list(dict(con.execute(q).fetchone()).values())[0]
    fams = [r["n_docs"] for r in con.execute("SELECT n_docs FROM graph_families")]
    return {"version": GRAPH_VERSION, "facts": one("SELECT count(*) FROM graph_facts"),
            "nodes": one("SELECT count(*) FROM graph_nodes"), "families": len(fams),
            "families_in_2plus_docs": sum(1 for n in fams if n >= 2), "family_max_docs": max(fams or [0]),
            "multi_node_families": one("SELECT count(*) FROM graph_families WHERE n_nodes >= 2"),
            "links": one("SELECT count(*) FROM graph_links"),
            "links_2plus_anchors": one("SELECT count(*) FROM graph_links WHERE weight >= 2"),
            "cards": one("SELECT count(*) FROM doc_cards")}


# ---------------------------------------------------------------------------------------------------------------
# querying (no model)
# ---------------------------------------------------------------------------------------------------------------

def term_list(terms):
    """[(folded, solid)] for a list of plain query terms, empties and duplicates dropped."""
    out = []
    for t in terms or []:
        f, s = L.term_forms(t)
        if s and (f, s) not in out:
            out.append((f, s))
    return out


def route_documents(con, terms, doc_ids=None, lang=None, k=None):
    """Score document cards by the DISTINCT terms they carry, rarer terms weighing more (idf over the cards); a word
    hit counts 1, a solid-only hit 0.6. -> [(doc_id, score, [terms hit])] best first."""
    forms = term_list(terms)
    sql, args = "SELECT doc_id, card_fold, card_solid FROM doc_cards", []
    cond = []
    if doc_ids:
        cond.append("doc_id IN (%s)" % ",".join("?" * len(doc_ids)))
        args += list(doc_ids)
    if lang:
        cond.append("lang = ?")
        args.append(lang)
    if cond:
        sql += " WHERE " + " AND ".join(cond)
    cards = con.execute(sql, args).fetchall()
    if not cards or not forms:
        return [(c["doc_id"], 0.0, []) for c in cards][:k]
    df = Counter()
    hits = {}
    for c in cards:
        h = []
        for f, s in forms:
            if re.search(r"(?:^|[\s|])" + re.escape(f) + r"(?:[\s|]|$)", c["card_fold"] or ""):
                h.append((f, 1.0))
            elif s in (c["card_solid"] or ""):
                h.append((f, 0.6))
        hits[c["doc_id"]] = h
        df.update({f for f, _w in h})
    n = len(cards)
    out = [(d, round(sum(w * math.log(1 + n / df[f]) for f, w in h), 4), [f for f, _w in h]) for d, h in hits.items()]
    out.sort(key=lambda x: (-x[1], x[0]))
    return out[:k] if k else out


def match_facts(con, doc_ids, terms, limit=3000):
    """Facts whose metric or entity NAME contains a term (the label layer's solid key, so every spelling), within
    `doc_ids`. -> {fact_id: {"metric": [(folded term, 'word'|'solid')], "entity": [...]}}"""
    forms = term_list(terms)
    if not forms or not doc_ids:
        return {}
    marks = defaultdict(lambda: {"metric": [], "entity": []})
    ph = ",".join("?" * len(doc_ids))
    for f, s in forms:
        for r in con.execute("SELECT ref_id, kind FROM labels_tri WHERE text LIKE ? AND kind IN ('metric', 'entity') "
                             "AND doc_id IN (%s) LIMIT ?" % ph, ("%" + s + "%", *doc_ids, limit)):
            marks[int(r["ref_id"])][r["kind"]].append(f)
    if not marks:
        return {}
    ids = list(marks)
    folded = {}
    for i in range(0, len(ids), 900):
        chunk = ids[i:i + 900]
        for r in con.execute("SELECT fact_id, metric_fold, entity_fold FROM fact_labels WHERE fact_id IN (%s)" %
                             ",".join("?" * len(chunk)), chunk):
            folded[r["fact_id"]] = (r["metric_fold"], r["entity_fold"])
    out = {}
    for fid, m in marks.items():
        mf, ef = folded.get(fid, (None, None))
        out[fid] = {"metric": [(t, "word" if L._word_hit(t, mf) else "solid") for t in dict.fromkeys(m["metric"])],
                    "entity": [(t, "word" if L._word_hit(t, ef) else "solid") for t in dict.fromkeys(m["entity"])]}
    return out


def graph_rows(con, fact_ids):
    """graph_facts rows (+ the family's document frequency) for fact ids. -> {fact_id: row}"""
    out = {}
    ids = list(fact_ids)
    for i in range(0, len(ids), 900):
        chunk = ids[i:i + 900]
        for r in con.execute("""SELECT g.*, fm.n_docs AS family_docs, fm.n_nodes AS family_nodes
                                FROM graph_facts g LEFT JOIN graph_families fm ON fm.family = g.family AND fm.lang = g.lang
                                WHERE g.fact_id IN (%s)""" % ",".join("?" * len(chunk)), chunk):
            out[r["fact_id"]] = dict(r)
    return out


def family_members(con, families, doc_ids, lang=None):
    """Every fact of the given families inside `doc_ids`: the cross-document bridges. -> [graph_facts row]"""
    if not families or not doc_ids:
        return []
    sql = ("SELECT * FROM graph_facts WHERE family IN (%s) AND doc_id IN (%s)" %
           (",".join("?" * len(families)), ",".join("?" * len(doc_ids))))
    args = list(families) + list(doc_ids)
    if lang:
        sql += " AND lang = ?"
        args.append(lang)
    return [dict(r) for r in con.execute(sql, args)]


def linked_nodes(con, node_ids, min_weight=2):
    """fa <-> en partners of nodes through numeric-anchor links. -> {node_id: [(partner, weight)]}"""
    out = defaultdict(list)
    ids = list(node_ids)
    for i in range(0, len(ids), 400):
        chunk = ids[i:i + 400]
        ph = ",".join("?" * len(chunk))
        for r in con.execute("SELECT node_a, node_b, weight FROM graph_links WHERE weight >= ? AND "
                             "(node_a IN (%s) OR node_b IN (%s))" % (ph, ph), (min_weight, *chunk, *chunk)):
            if r["node_a"] in chunk:
                out[r["node_a"]].append((r["node_b"], r["weight"]))
            if r["node_b"] in chunk:
                out[r["node_b"]].append((r["node_a"], r["weight"]))
    return dict(out)


def main(argv=None):
    import argparse
    import sys
    from .ingest import open_kb
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    ap = argparse.ArgumentParser(prog="python -m ragbench.kb.graph", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("build")
    sub.add_parser("status")
    r = sub.add_parser("route", help="rank document cards for some terms")
    r.add_argument("terms", nargs="+")
    r.add_argument("--k", type=int, default=5)
    fm = sub.add_parser("family", help="the nodes of the families matching a term")
    fm.add_argument("term")
    args = ap.parse_args(argv)
    con = open_kb(read_only=args.cmd not in ("build",))
    if args.cmd == "build":
        print(json.dumps(build(con, log=print), ensure_ascii=False))
    elif args.cmd == "status":
        print(json.dumps(summary(con), ensure_ascii=False, indent=1))
    elif args.cmd == "route":
        for d, s, h in route_documents(con, args.terms, k=args.k):
            print("  %6.2f  %-44s %s" % (s, d, ", ".join(h)))
    else:
        f, s = L.term_forms(args.term)
        for r in con.execute("""SELECT family, node_id, name, n_docs, n_facts, year_min, year_max FROM graph_nodes
                                WHERE replace(family, ' ', '') LIKE ? ORDER BY family, n_facts DESC LIMIT 40""",
                             ("%" + s + "%",)):
            print("  %-28s %-40s docs %2d facts %4d years %s-%s | %s" % (
                r["family"][:28], r["node_id"].split("|", 2)[2] or "(no qualifier)", r["n_docs"], r["n_facts"],
                r["year_min"], r["year_max"], (r["name"] or "")[:40]))
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
