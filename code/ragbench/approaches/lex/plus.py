"""LEX's optional upgrades (LEX2 turns all three on), and the card-only index build. No model call in this file.

    bm25        a BM25 safety net over structure chunks of the chosen reports, on the ORIGINAL question (+ the call-1
                keywords, which carry the documents' language), fused with LEX's windows by reciprocal rank; windows
                only BM25 found are capped at `bm25_max_chars` of the evidence (second opinion, 2026-09-22: extra
                recall alone is not enough - V4_DEV.md - so the net must not crowd out what the regex found)
    year_pass   a question over several years: a year whose report has no chosen window that SUPPORTS the metric (a
                number, and a term the model picked matched as a phrase) is searched again, in that report only, with
                the short labels of the supported windows of the other years
    cards       refresh_cards rebuilds only the report cards, not the whole graph (families, links: 5.2 of the 5.3 s of
                a graph build are parts LEX never reads)
"""
import math
import re
import time
from collections import defaultdict

from ...kb import graph as G
from ...kb import labels as L
from ...normalize import fold
from ..agentic.planner import fa_word_variants
from . import search as S

_TOKEN = re.compile(r"[^\W\d_]+|\d+(?:[.,/]\d+)*", re.UNICODE)


def _ms(t0):
    return round((time.perf_counter() - t0) * 1000.0, 1)


# ---------------------------------------------------------------------------------------------------------------
# the card-only build
# ---------------------------------------------------------------------------------------------------------------

def refresh_cards(con):
    """Rebuild doc_cards (and nothing else of the graph) when a ready report has no card or the cards are older than
    the label layer. -> {"rebuilt": bool, "cards": n, "ms": ...}. The graph's own `graph.dirty` flag is left alone, so
    AGENTIC4 still rebuilds its families when it next runs."""
    t0 = time.perf_counter()
    G.ensure_graph_schema(con)
    ready = {r["doc_id"] for r in con.execute("SELECT doc_id FROM documents WHERE status IN ('ready', 'vector_ready')")}
    have = {r["doc_id"] for r in con.execute("SELECT doc_id FROM doc_cards")}
    meta = {r["key"]: r["value"] for r in con.execute("SELECT key, value FROM meta WHERE key LIKE 'lex.cards%'")}
    stamp = "%s|%d" % (L.LABELS_VERSION, len(ready))
    if ready <= have and meta.get("lex.cards_stamp") == stamp:
        return {"rebuilt": False, "cards": len(have), "ms": _ms(t0)}
    rows = G._cards(con)
    con.execute("DELETE FROM doc_cards")
    con.executemany("INSERT INTO doc_cards (doc_id, lang, company, card_fold, card_solid, n_parts) VALUES (?,?,?,?,?,?)",
                    rows)
    con.execute("INSERT OR REPLACE INTO meta (key, value) VALUES ('lex.cards_stamp', ?)", (stamp,))
    con.commit()
    return {"rebuilt": True, "cards": len(rows), "ms": _ms(t0)}


def cost_from_scratch(con, doc_ids):
    """The preprocessing a set's reports cost when nothing is cached, from the ingestion log (the latest run of each
    stage per report), split into what LEX needs and what it does not. The extraction (PDF -> text) is corpus
    preparation every method shares; it is reported, not charged to LEX."""
    ids = list(doc_ids)
    out = {"docs": len(ids), "stages_ms": {}, "lex_ms": 0.0, "shared_extract_ms": 0.0, "unused_ms": 0.0}
    if not ids:
        return out
    ph = ",".join("?" * len(ids))
    for r in con.execute("SELECT stage, SUM(ms) AS ms FROM (SELECT doc_id, stage, ms, MAX(rowid) FROM ingest_log "
                         "WHERE doc_id IN (%s) GROUP BY doc_id, stage) GROUP BY stage" % ph, ids):
        out["stages_ms"][r["stage"]] = round(r["ms"] or 0.0, 1)
    for st, ms in out["stages_ms"].items():
        if st in ("load", "facts", "labels", "register", "model", "ready"):
            out["lex_ms"] += ms
        elif st == "extract":
            out["shared_extract_ms"] += ms
        else:                                  # passages, embed: AGENTIC2/3's vector path, not LEX's
            out["unused_ms"] += ms
    if "labels" not in out["stages_ms"]:
        # the label layer was built in one batch for these reports (graph / labels CLI), so the ingestion log has no
        # per-report row: estimated from a whole-knowledge-base rebuild timed on 2026-09-22 (22.3 s, 63,388 facts)
        n_facts = con.execute("SELECT count(*) AS c FROM facts WHERE doc_id IN (%s)" % ph, ids).fetchone()["c"]
        est = round(n_facts * LABELS_MS_PER_FACT, 1)
        out["stages_ms"]["labels_estimated"] = est
        out["lex_ms"] += est
    out["lex_ms"] = round(out["lex_ms"], 1)
    return out


LABELS_MS_PER_FACT = 22300.0 / 63388.0


# ---------------------------------------------------------------------------------------------------------------
# BM25 over structure chunks
# ---------------------------------------------------------------------------------------------------------------

def tokens(text, lang):
    """Folded word tokens, stop words out, Persian words reduced to their shortest plural/-ان stem."""
    out = []
    for t in _TOKEN.findall(fold(text)):
        if len(t) < 2 or t in L._STOP:
            continue
        if lang == "fa" and not t[0].isdigit():
            t = min(fa_word_variants(t), key=len) if fa_word_variants(t) else t
        out.append(t)
    return out


def chunks(reports, doc_ids, o):
    """Structure chunks: consecutive units of one section up to `window_chars` (a table: its own chunk, cut to the
    header + its first rows at `window_chars` x `table_window_factor`). Each chunk has the window shape."""
    cap = int(o["window_chars"])
    tcap = int(cap * float(o["table_window_factor"]))
    out = []
    for d in doc_ids:
        reports.load(d)
        for (doc, _sid), us in ((k, v) for k, v in reports.by_section.items() if k[0] == d):
            cur, size = [], 0
            for u in us:
                if u.kind == "table":
                    if cur:
                        out.append(cur)
                        cur, size = [], 0
                    out.append([u])
                    continue
                if cur and size + len(u.text) + 2 > cap:
                    out.append(cur)
                    cur, size = [], 0
                cur.append(u)
                size += len(u.text) + 2
            if cur:
                out.append(cur)
    wins = []
    for us in out:
        u0 = us[0]
        if u0.kind == "table":
            text = u0.text if len(u0.text) <= tcap else "\n".join(u0.text[:tcap].split("\n")[:-1] or [u0.text[:tcap]])
            span = list(u0.span) if len(u0.text) <= tcap else None
        else:
            text = "\n\n".join(x.text for x in us)
            span = [us[0].span[0], us[-1].span[1]]
        wins.append({"doc_id": u0.doc_id, "section_id": u0.section_id, "hit": u0.id, "unit_ids": [x.id for x in us],
                     "text": text, "span": span, "kind": "window" if (u0.kind != "table" or span is None) else "unit",
                     "score": 0.0, "terms": [], "rows": None, "rows_total": None, "source": "bm25"})
    return wins


def bm25_windows(reports, doc_ids, query_texts, lang, o):
    """-> the top `bm25_top_k` structure chunks by BM25 of the query tokens (k1, b of the card routing)."""
    t0 = time.perf_counter()
    wins = chunks(reports, doc_ids, o)
    q = []
    for t in query_texts:
        for tok in tokens(t, lang):
            if tok not in q:
                q.append(tok)
    if not wins or not q:
        return [], {"chunks": len(wins), "query_tokens": q, "ms": _ms(t0)}
    toks = [tokens(w["text"], lang) for w in wins]
    n = len(wins)
    avg = sum(len(t) for t in toks) / n or 1.0
    df = defaultdict(int)
    for ts in toks:
        for tok in set(ts):
            df[tok] += 1
    k1, b = float(o["bm25_k1"]), float(o["bm25_b"])
    for w, ts in zip(wins, toks):
        tf = defaultdict(int)
        for tok in ts:
            tf[tok] += 1
        s = 0.0
        for tok in q:
            f = tf.get(tok, 0)
            if f:
                idf = math.log(1.0 + (n - df[tok] + 0.5) / (df[tok] + 0.5))
                s += idf * f * (k1 + 1.0) / (f + k1 * (1.0 - b + b * len(ts) / avg))
        w["score"] = round(s, 4)
    wins = [w for w in wins if w["score"] > 0]
    wins.sort(key=lambda w: (-w["score"], w["doc_id"], w["hit"]))
    return wins[: int(o["bm25_top_k"])], {"chunks": n, "query_tokens": q[:30], "ms": _ms(t0)}


def fuse(lex_wins, bm_wins, o):
    """Reciprocal-rank fusion of LEX's windows and BM25's chunks (k = `rrf_k`). A BM25 chunk that a LEX window already
    shows (`overlap_max`) adds its rank to that window instead of entering twice. -> windows with `score` = the fused
    score, `lex_score` kept, and `source` lex | bm25 | both."""
    k = float(o["rrf_k"])
    out = []
    for r, w in enumerate(lex_wins, 1):
        out.append(dict(w, lex_score=w["score"], score=1.0 / (k + r), source="lex"))
    for r, c in enumerate(bm_wins, 1):
        twin = next((w for w in out if w["source"] != "bm25" and (S.overlap(c, w) > float(o["overlap_max"])
                                                                   or S.overlap(w, c) > float(o["overlap_max"]))), None)
        if twin is not None:
            twin["score"] += 1.0 / (k + r)
            twin["source"] = "both"
        else:
            out.append(dict(c, lex_score=0.0, score=1.0 / (k + r), source="bm25"))
    out.sort(key=lambda w: (-w["score"], w["doc_id"], w["hit"]))
    return out


# ---------------------------------------------------------------------------------------------------------------
# the missing-year pass
# ---------------------------------------------------------------------------------------------------------------

def supports(w, picked):
    """A window SUPPORTS the metric when it shows a number and a term the model picked matched it as a phrase (a
    window with only a year or a relaxed match does not - second opinion: "contains a number" alone is inadequate)."""
    return bool(re.search(r"\d", w["text"])) and any(t in picked for t in w.get("terms") or [])


def year_pass(con, reports, docs, chosen, terms, kw, lang, o, cost, budget):
    """A multi-year question: for each target report of a year asked about with no supporting window, search that
    report again with the short labels of the supporting windows of the other years (their matched phrases and the
    metric names of the facts they show), and add its best window that shows a number - in the room left, else in
    place of the weakest window of a report that has more than one. -> (chosen, info)."""
    t0 = time.perf_counter()
    years = S.question_years(kw.get("years") or [])
    info = {"ran": False, "missing": [], "filled": [], "labels": []}
    if len(years) < 2 or len(docs) < 2:
        return chosen, info
    rows = S.doc_rows(con, docs)
    picked = {t.text for t in terms if t.source == "picked"} or {t.text for t in terms}
    by_doc = defaultdict(list)
    for w in chosen:
        by_doc[w["doc_id"]].append(w)
    target = [d for d in docs if (rows.get(d) or {}).get("year_from") in years]
    missing = [d for d in target if not any(supports(w, picked) for w in by_doc[d])]
    info.update(ran=True, missing=missing)
    if not missing:
        return chosen, info
    labels = []
    for w in chosen:
        if w["doc_id"] in missing or not supports(w, picked):
            continue
        labels += [t for t in w["terms"] if t in picked]
        if w["unit_ids"]:
            for r in con.execute("SELECT DISTINCT fl.metric_name FROM fact_labels fl JOIN facts f ON f.fact_id = "
                                 "fl.fact_id WHERE f.unit_id IN (%s) AND fl.label_quality = 'ok' AND fl.metric_name IS "
                                 "NOT NULL" % ",".join("?" * len(w["unit_ids"])), w["unit_ids"]):
                if len((r["metric_name"] or "").split()) <= int(o["year_pass_label_words"]):
                    labels.append(r["metric_name"])
    labels = list(dict.fromkeys(labels))[: int(o["year_pass_labels"])]
    info["labels"] = labels
    lterms = S.compile_terms(labels, lang, "year_pass", float(o["w_picked"]), o)
    if not lterms:
        return chosen, info
    used = sum(cost(w) for w in chosen)
    for d in missing:
        hits, by_id = S.search(reports, [d], lterms)
        wins, _ms2 = S.windows(reports, hits, by_id, lterms, kw, o)
        cand = next((w for w in wins if re.search(r"\d", w["text"])
                     and not any(S.overlap(w, k) > float(o["overlap_max"]) for k in chosen)), None)
        if cand is None:
            continue
        c = cost(cand)
        if used + c > budget:
            crowded = [w for w in chosen if len(by_doc[w["doc_id"]]) > 1]
            if not crowded:
                continue
            weakest = min(crowded, key=lambda w: w["score"])
            if used - cost(weakest) + c > budget:
                continue
            chosen = [w for w in chosen if w is not weakest]
            by_doc[weakest["doc_id"]].remove(weakest)
            used -= cost(weakest)
        cand = dict(cand, source="year_pass")
        chosen.append(cand)
        by_doc[d].append(cand)
        used += c
        info["filled"].append(d)
    info["ms"] = _ms(t0)
    return chosen, info
