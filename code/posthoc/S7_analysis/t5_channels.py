"""Task 5 (review item 17): per-language contribution of the retrieval channels to gold-evidence recall@40 and to the
delivered packs, from saved candidate lists + an exact CPU recomputation of the BM25 channel; and a BM25 normalisation
ablation (recall only, no reader). No GPU: dense and learned-sparse query encodings are NOT recomputed.
-> channels.json, channels_tables.md, fig_channels.dat"""
import json
import re
import unicodedata
from collections import Counter, defaultdict

import numpy as np

import s7lib as L
from ragbench.config import load_config
from ragbench.contracts import Setting, load_questions
from ragbench.corpus.registry import load_set, set_name_for
from ragbench.pools import load_pool, resolve_pool_path
from ragbench.retrieval import hybrid as H
from ragbench.approaches import baselines as BL

CFG = load_config()
POOLS = {"RPT-fa": ("C", "fa-fa"), "RPT-en": ("D", "en-en"), "BOOK-fa": ("K", "fa-fa"), "BOOK-en": ("K", "en-en"),
         "LAW-fa": ("L", "fa-fa"), "LAW-en": ("L", "en-en")}
_SETS, _HX = {}, {}


def hx_for(sn):
    if sn not in _HX:
        _SETS[sn] = load_set(sn, CFG)
        _HX[sn] = H.HybridIndex.load(_SETS[sn], CFG, client=None, refresh_vectors=False)
    return _HX[sn]


QUERY = {}
for pid, (corpus, st) in POOLS.items():
    for q in load_questions(L.FZ / "questions" / ("test_v2run_%s.jsonl" % pid)):
        s = Setting.parse(st)
        sn = set_name_for(q.corpus, s.d_lang)
        pool = load_pool(resolve_pool_path(pid, CFG))
        QUERY[q.id] = {"text": q.text(s.q_lang), "set": sn, "scope": list(pool.mask(q.scope(s.d_lang), sn))}


def gold_cov(hx, q, chunk_ids):
    """-> {eid: fraction of the gold span covered by the union of these chunks' spans (same doc)}"""
    out = {}
    for e in L.QV2[q]["evidence"]:
        a, b = e["start"], e["end"]
        cov = np.zeros(max(1, b - a), bool)
        for c in chunk_ids:
            ch = hx.chunks[c]
            if ch["doc_id"] != e["doc_id"] or not ch.get("span"):
                continue
            s0, s1 = ch["span"]
            lo, hi = max(a, s0), min(b, s1)
            if hi > lo:
                cov[lo - a:hi - a] = True
        out[e["id"]] = float(cov.mean())
    return out


def bundle_cls(q, cov, thr=0.9):
    fr = [sum(cov[x] >= thr for x in bun) / len(bun) for bun in L.QV2[q]["evidence_bundles"]]
    best = max(fr)
    return "complete" if best == 1.0 else ("partial" if best > 0 else "none")


def item_recall(q, cov, thr=0.9):
    ids = sorted({x for bun in L.QV2[q]["evidence_bundles"] for x in bun})
    return sum(cov[x] >= thr for x in ids), len(ids)


# ------------------------------------------------------------------ channel lists per question
ROWS = {}
chk = Counter()
for q in L.ANS:
    Q = QUERY[q]
    hx = hx_for(Q["set"])
    bm = [h.chunk for h in hx.search(Q["text"], doc_ids=Q["scope"], k=40, channels=("bm25",))]
    # saved fused lists with per-channel ranks (same query, same index): HYBRID_6000 (40), HR_6000 / HR_15500 (50)
    dense_rank, bm_rank_saved = {}, {}
    for arm in ("HYBRID_6000", "HR_6000", "HYBRID_15500", "HR_15500"):
        for u in L.REC[L.P][arm][q]["retrieved_units"]:
            m = u.get("meta") or {}
            rk = m.get("ranks") or {}
            if rk.get("dense"):
                dense_rank[m["chunk"]] = rk["dense"]
            if rk.get("bm25"):
                bm_rank_saved[m["chunk"]] = rk["bm25"]
        for t in L.REC[L.P][arm][q]["trace"].get("top") or []:
            if len(t) >= 4:
                if t[3]:
                    dense_rank[t[0]] = t[3]
                if t[2]:
                    bm_rank_saved[t[0]] = t[2]
    # check the CPU BM25 against the saved ranks
    bmr = {c: i + 1 for i, c in enumerate(bm)}
    for c, r in bm_rank_saved.items():
        chk["bm25_rank_match" if bmr.get(c) == r else "bm25_rank_mismatch"] += 1
    dk = sorted(dense_rank.items(), key=lambda kv: kv[1])
    known = {r for _, r in dk}
    t = 0
    while t + 1 in known:
        t += 1
    dense_known40 = [c for c, r in dk if r <= 40]
    hyb40 = [u["meta"]["chunk"] for u in L.REC[L.P]["HYBRID_6000"][q]["retrieved_units"]][:40]
    f2 = [int(c) for c in L.REC[L.P]["LEX5B_6000"][q]["trace"]["lex5b"]["f2_top40"]]
    ROWS[q] = dict(bm=bm, dense_known=dense_known40, dense_complete_depth=t, dense_unknown_slots=40 - len(dense_known40),
                   hyb40=hyb40, f2=f2, hx=Q["set"])

# ------------------------------------------------------------------ recall@40 per channel
res = {"bm25_cpu_check": dict(chk), "per_q": {}, "summary": {}}
for q, r in ROWS.items():
    hx = hx_for(r["hx"])
    d = {}
    for k in ("bm", "dense_known", "hyb40", "f2"):
        cov = gold_cov(hx, q, r[k])
        d[k] = {"cls": bundle_cls(q, cov), "items": item_recall(q, cov), "cov": cov}
    union = sorted(set(r["bm"]) | set(r["dense_known"]))
    cov = gold_cov(hx, q, union)
    d["bm_or_dense"] = {"cls": bundle_cls(q, cov), "items": item_recall(q, cov), "cov": cov}
    # unique contribution per gold item
    ids = sorted({x for bun in L.QV2[q]["evidence_bundles"] for x in bun})
    uniq = Counter()
    for x in ids:
        inb, ind, inf = d["bm"]["cov"][x] >= .9, d["dense_known"]["cov"][x] >= .9, d["f2"]["cov"][x] >= .9
        uniq["both" if inb and ind else "bm25_only" if inb else "dense_only" if ind else "neither_bm25_dense"] += 1
        if inf and not inb and not ind:
            uniq["f2_only(sparse_or_deep_dense)"] += 1
    d["unique"] = dict(uniq)
    d["dense_complete_depth"] = r["dense_complete_depth"]
    d["dense_unknown_slots"] = r["dense_unknown_slots"]
    res["per_q"][q] = {k: ({kk: vv for kk, vv in v.items() if kk != "cov"} if isinstance(v, dict) and "cov" in v else v)
                       for k, v in d.items()}

for lg in ("fa", "en"):
    qs = [q for q in L.ANS if L.lang(q) == lg]
    s = {"n": len(qs)}
    for k in ("bm", "dense_known", "bm_or_dense", "hyb40", "f2"):
        c = Counter(res["per_q"][q][k]["cls"] for q in qs)
        hit = sum(res["per_q"][q][k]["items"][0] for q in qs)
        tot = sum(res["per_q"][q][k]["items"][1] for q in qs)
        s[k] = {"complete": c["complete"], "partial": c["partial"], "none": c["none"], "items_hit": hit, "items": tot}
    u = Counter()
    for q in qs:
        u.update(res["per_q"][q]["unique"])
    s["unique_items"] = dict(u)
    s["dense_complete_depth_median"] = float(np.median([res["per_q"][q]["dense_complete_depth"] for q in qs]))
    s["dense_unknown_slots_total"] = sum(res["per_q"][q]["dense_unknown_slots"] for q in qs)
    res["summary"][lg] = s

# ------------------------------------------------------------------ delivered packs: which channel brought the gold
reach = {(x["arm"], x["q"]): x for x in json.loads((L.HERE / "reach.json").read_text(encoding="utf-8"))["per_cell"]}
deliv = {}
for arm in ("HYBRID_6000", "HR_6000", "LEX5B_6000", "HYBRID_15500", "HR_15500"):
    for lg in ("fa", "en"):
        c = Counter()
        for q in [q for q in L.ANS if L.lang(q) == lg]:
            r = ROWS[q]
            hx = hx_for(r["hx"])
            if arm.startswith("LEX5B"):
                shown = [int(x) for x in L.REC[L.P][arm][q]["trace"]["lex5b"]["shown_chunks"]]
            else:
                shown = [u["meta"]["chunk"] for u in L.REC[L.P][arm][q]["retrieved_units"]]
            items = reach[(arm, q)]["items"]
            bmset, dset = set(r["bm"]), set(r["dense_known"])
            for e in L.QV2[q]["evidence"]:
                if items.get(e["id"], 0) < L.PRESENT:
                    continue
                src = [ch for ch in shown if gold_cov(hx, q, [ch])[e["id"]] > 0]
                if not src:
                    c["unattributed"] += 1
                    continue
                inb = any(ch in bmset for ch in src)
                ind = any(ch in dset for ch in src)
                c["both" if inb and ind else "bm25_only" if inb else "dense_only" if ind else
                  ("f2_only(sparse_or_deep_dense)" if arm.startswith("LEX5B") else "unattributed")] += 1
        deliv["%s|%s" % (arm, lg)] = dict(c)
res["delivered_gold_items_by_channel"] = deliv


# ------------------------------------------------------------------ BM25 normalisation ablation (CPU)
_RE_WORD_RAW = re.compile(r"\d+(?:[.,/]\d+)*%?|[^\W\d_]+(?:[‌'][^\W\d_]+)*", re.UNICODE)


def tok_raw(text):
    """plain Unicode word tokenizer: lower-case only, no normalize()/fold(), no harakat strip, no subtokens/variants"""
    return [t for t in _RE_WORD_RAW.findall((text or "").lower())]


def build(sn, tok):
    import bm25s
    hx = hx_for(sn)
    toks = [tok(H.index_text(c)) for c in hx.chunks]
    p = BL.DEFAULTS["bm25"]
    m = bm25s.BM25(k1=p["k1"], b=p["b"], method=p["method"])
    m.index(toks, show_progress=False)
    return m


def search(m, hx, qtoks, scope, k=40):
    known = [t for t in dict.fromkeys(qtoks) if t in m.vocab_dict]
    if not known:
        return []
    s = np.asarray(m.get_scores(known), dtype=np.float32)
    mask, _ = hx.scope_mask(scope)
    valid = mask & (s > 0)
    order = np.lexsort((np.arange(hx.n), -s))
    return [int(i) for i in order[valid[order]][:k]]


VARIANTS = {"frozen": None, "raw_query": "raw_query", "raw_both": "raw_both"}
abl = {v: {"fa": Counter(), "en": Counter()} for v in VARIANTS}
sets = sorted({ROWS[q]["hx"] for q in L.ANS})
raw_models = {sn: build(sn, tok_raw) for sn in sets}
for q in L.ANS:
    Q = QUERY[q]
    hx = hx_for(Q["set"])
    lg = L.lang(q)
    lists = {"frozen": ROWS[q]["bm"]}
    # raw_query: frozen document index, the query tokenised without normalisation
    lists["raw_query"] = search(hx.bm25, hx, tok_raw(Q["text"]), Q["scope"])
    lists["raw_both"] = search(raw_models[Q["set"]], hx, tok_raw(Q["text"]), Q["scope"])
    for v, lst in lists.items():
        cov = gold_cov(hx, q, lst)
        abl[v][lg][bundle_cls(q, cov)] += 1
        h, t = item_recall(q, cov)
        abl[v][lg]["items_hit"] += h
        abl[v][lg]["items"] += t
res["bm25_normalisation_ablation"] = {v: {lg: dict(c) for lg, c in d.items()} for v, d in abl.items()}
res["bm25_normalisation_ablation_note"] = (
    "frozen = repo tokenizer (normalize() + harakat strip + fold() variants + ZWNJ subtokens + number variants + stopwords) on "
    "both sides. raw_query = frozen document index, question tokenised by a plain Unicode tokenizer (lower-case only). "
    "raw_both = plain Unicode tokenizer on both sides. The corpus text itself was normalised at extraction time "
    "(normalize() is a port of the extractor rule), so 'raw_both' removes only the tokenizer-level normalisation.")
# query-side diagnostics: how many Persian questions contain Persian digits / Arabic letter variants / ZWNJ
diag = Counter()
for q in L.ANS:
    t = QUERY[q]["text"]
    if L.lang(q) != "fa":
        continue
    diag["questions"] += 1
    diag["persian_digits"] += bool(re.search("[۰-۹٠-٩]", t))
    diag["arabic_yeh_kaf"] += bool(re.search("[يكى]", t))
    diag["zwnj"] += "‌" in t
    diag["changed_by_normalize"] += L.normalize(t) != re.sub(r"\s+", " ", t).strip()
res["persian_query_diagnostics"] = dict(diag)
L.wjson(L.HERE / "channels.json", res)

# ------------------------------------------------------------------ outputs
with open(L.HERE / "fig_channels.dat", "w", encoding="utf-8") as fh:
    fh.write("# S7 task 5: gold-bundle recall@40 per channel and language (45 answerable each); channels.json\n")
    fh.write("lang channel complete partial none items_hit items\n")
    for lg in ("fa", "en"):
        for k in ("bm", "dense_known", "bm_or_dense", "hyb40", "f2"):
            s = res["summary"][lg][k]
            fh.write("%s %s %d %d %d %d %d\n" % (lg, k, s["complete"], s["partial"], s["none"], s["items_hit"], s["items"]))
md = ["# S7 task 5: Persian retrieval by channel (generated by t5_channels.py -> channels.json)", "",
      "BM25 top-40 recomputed on CPU (frozen index, frozen tokenizer); check against the saved per-channel ranks: %s." % dict(chk),
      "Dense top-40 is known only through the saved fused lists (HYBRID_6000 top 40, HR top 50 at both budgets), so dense ",
      "recall below is a lower bound; learned-sparse lists were never saved (LEX5B saves only its fused F2 top 40) and need a GPU",
      "query encoding to recompute.", "",
      "## gold recall@40 (bundle complete / partial / none; gold items found)", "",
      "| channel | fa C/P/N | fa items | en C/P/N | en items |", "|---|---|---|---|---|"]
NAMES = {"bm": "BM25 top 40 (exact)", "dense_known": "bge-m3 dense top 40 (known part; lower bound)",
         "bm_or_dense": "BM25 ∪ dense (known)", "hyb40": "HYBRID RRF top 40", "f2": "Lex-RAG F2 top 40 (BM25+dense+sparse)"}
for k in ("bm", "dense_known", "bm_or_dense", "hyb40", "f2"):
    f, e = res["summary"]["fa"][k], res["summary"]["en"][k]
    md.append("| %s | %d/%d/%d | %d/%d | %d/%d/%d | %d/%d |" % (NAMES[k], f["complete"], f["partial"], f["none"], f["items_hit"],
                                                            f["items"], e["complete"], e["partial"], e["none"], e["items_hit"], e["items"]))
md += ["", "Unique contribution per gold item (within the two top-40 lists): fa %s; en %s." % (
    res["summary"]["fa"]["unique_items"], res["summary"]["en"]["unique_items"]),
    "Dense list known completely to median depth fa %.0f, en %.0f; unknown dense top-40 slots: fa %d, en %d (of 45x40)." % (
        res["summary"]["fa"]["dense_complete_depth_median"], res["summary"]["en"]["dense_complete_depth_median"],
        res["summary"]["fa"]["dense_unknown_slots_total"], res["summary"]["en"]["dense_unknown_slots_total"]), "",
    "## gold items that reached the reader, by the channel(s) of the chunk that carried them", "",
    "| arm | lang | " + " | ".join(["both", "bm25_only", "dense_only", "f2_only(sparse_or_deep_dense)", "unattributed"]) + " |",
    "|---|---|---|---|---|---|---|"]
for k, c in deliv.items():
    a, lg = k.split("|")
    md.append("| %s | %s | %s |" % (a, lg, " | ".join(str(c.get(x, 0)) for x in ["both", "bm25_only", "dense_only",
                                                                                 "f2_only(sparse_or_deep_dense)", "unattributed"])))
md += ["", "## BM25 normalisation ablation (recall@40 only, no reader)", "", res["bm25_normalisation_ablation_note"], "",
       "| variant | fa C/P/N | fa items | en C/P/N | en items |", "|---|---|---|---|---|"]
for v, d in res["bm25_normalisation_ablation"].items():
    f, e = d["fa"], d["en"]
    md.append("| %s | %d/%d/%d | %d/%d | %d/%d/%d | %d/%d |" % (v, f.get("complete", 0), f.get("partial", 0), f.get("none", 0),
                                                            f["items_hit"], f["items"], e.get("complete", 0), e.get("partial", 0),
                                                            e.get("none", 0), e["items_hit"], e["items"]))
md += ["", "Persian question diagnostics: %s" % dict(diag)]
(L.HERE / "channels_tables.md").write_text("\n".join(md) + "\n", encoding="utf-8")
print("\n".join(md))
