"""KW and DOC: two retrieval ideas (Arash, 2026-09-26), tested in work/datasets_v2/idea_test.

KW  keyword-count ranking. ONE small LLM call (default openai/gpt-oss-20b, pinned endpoint, reasoning low) writes ~10
    search keywords / short phrases a passage answering the question would contain (the question's language, plus the
    other language where natural; inflected variants; no question words). Each chunk of the pool is scored by the
    number of DISTINCT keywords it contains; a keyword counts when it matches
        exact    its folded token sequence occurs contiguously in the chunk (ragbench.normalize.fold: Persian/Arabic
                 letter variants, digits, harakat folded; the Persian half-space is a word break) - or its tokens
                 written solid (half-space dropped: 'X Y' == 'XY', as one token or two adjacent tokens joined)
        similar  (a) light morphology: every content token of the keyword has a chunk token with the same light stem
                 (fa/en plural and suffix stripping) or a shared prefix of >= max(5, shorter length - 2) chars;
                 numbers match exactly only.  (b) embedding-similar words: NOT implemented (not cheap).
    Ties are broken by the question's BM25 score (HYBRID's BM25 channel), then chunk order.
    KW          the KW ranking alone
    HYBRID+KW   equal-weight RRF (k=60) of BM25 top 40, dense top 40 and KW top 40
DOC document-first. Pool documents ranked by keyword coverage: sum over the keywords found anywhere in the document
    of idf = ln(1 + N_docs / df) (df = pool documents containing the keyword); ties: the document's best chunk BM25,
    then doc id. HYBRID+KW restricted to the top-k documents. Fallback to the whole pool when no chunk of the chosen
    documents contains at least min(ceil(n_keywords/2), 2) keywords (declared before the test run).

The keyword index (tokens, postings of tokens and solid bigrams, the vocabulary by 5-char prefix and by light stem)
is built once per HybridIndex over its chunk texts and kept in memory; keyword matches are cached per keyword.
Online use: HYBRID flags `kw` and `doc=<k>` (approaches/hybrid.py -> answer_kw below), same offer/pack/answer as
HYBRID. Keywords may be supplied by a JSON cache (env KWDOC_KEYWORDS: {sha1(normalize(question)): {keywords, ...}}),
so an offline replay and the answer run use the same keywords.
"""
import hashlib
import json
import math
import os
import re
import time

import numpy as np

from ..kb import labels as L
from ..normalize import normalize, word_tokens
from ..retrieval import hybrid as H

KWDOC_VERSION = "kwdoc-v1"
KW_DEFAULTS = {"model": "openai/gpt-oss-20b", "n_keywords": 10, "max_keywords": 16, "attempts": 1,
               "deadline_s": 6.0, "max_tokens_total": 1024, "reasoning_effort": "low", "depth": 40,
               "prefix_min": 5, "prefix_slack": 2, "rrf_k": 60}

PROMPT = """A search engine will look for passages that answer the question below by matching keywords in the text.

Question: {question}

Write about {n} search keywords or short phrases (1-3 words each) that are likely to appear word for word in a passage that ANSWERS the question - the specific names, terms, entities and numbers such a passage would contain, not the words used to ask (no question words such as what / how many / which, and no generic words).
- Write them in the language of the question. Where a passage could naturally use the other language (English or Persian), add that form too.
- Add inflected or alternative forms that a passage might use (plural and singular, other word forms, common synonyms).

Reply with one keyword per line and nothing else."""

_FA_SUFFIXES = ("هایی", "هایی", "های", "ها", "ترین", "تر", "ات", "ان", "یی", "ای", "ی")
_FA_AFFIX_TOKENS = {"ها", "های", "هایی", "ی", "ای", "تر", "ترین", "می", "نمی", "ام", "اش", "ات", "ان", "شان", "مان"}
_EN_SUFFIXES = (("ies", "y"), ("sses", "ss"), ("ing", ""), ("ed", ""), ("es", ""), ("ly", ""), ("s", ""))
_BULLET = re.compile(r"^\s*(?:[-*•●]+|\d+[.)\-:]|[۰-۹]+[.)\-:])\s*")


def qkey(question):
    return hashlib.sha1(normalize(question).encode("utf-8")).hexdigest()


def _is_fa(t):
    return any("؀" <= ch <= "ۿ" for ch in t)


def stem(t):
    """Light stem of a folded token: numbers and short tokens unchanged."""
    if any(ch.isdigit() for ch in t) or len(t) < 4:
        return t
    if _is_fa(t):
        for s in _FA_SUFFIXES:
            if t.endswith(s) and len(t) - len(s) >= 3:
                return t[:-len(s)]
        return t
    for s, r in _EN_SUFFIXES:
        if t.endswith(s) and len(t) - len(s) >= 3 and not (s == "s" and t.endswith("ss")):
            return t[:-len(s)] + r
    return t


def _cp(a, b):
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


def similar(a, b):
    """Light-morphology similarity of two folded tokens (a != b)."""
    if a == b:
        return True
    if any(ch.isdigit() for ch in a) or any(ch.isdigit() for ch in b):
        return False
    if stem(a) == stem(b):
        return True
    need = max(KW_DEFAULTS["prefix_min"], min(len(a), len(b)) - KW_DEFAULTS["prefix_slack"])
    return min(len(a), len(b)) >= KW_DEFAULTS["prefix_min"] and _cp(a, b) >= need


def content_tokens(toks):
    return [t for t in toks if t not in L._STOP and t not in _FA_AFFIX_TOKENS and (len(t) > 1 or t.isdigit())]


def parse_keywords(text, cap=None):
    """One keyword per line; bullets / numbering / quotes stripped; folded duplicates dropped."""
    cap = cap or KW_DEFAULTS["max_keywords"]
    out, seen = [], set()
    for ln in (text or "").splitlines():
        ln = _BULLET.sub("", ln).strip().strip("\"'`“”«»").strip()
        if not ln or len(ln) > 60 or ln.lower().startswith(("keyword", "here ")):
            continue
        toks = word_tokens(ln)
        if not toks:
            continue
        k = " ".join(toks)
        if k in seen:
            continue
        seen.add(k)
        out.append(ln)
        if len(out) >= cap:
            break
    return out


class KeywordIndex:
    """Inverted index over the chunk texts of one HybridIndex (built once; read-only afterwards)."""

    def __init__(self, hx):
        t0 = time.perf_counter()
        self.hx = hx
        self.n = hx.n
        self.toks = []
        post, post2 = {}, {}
        for i, c in enumerate(hx.chunks):
            tk = word_tokens(c["text"])
            self.toks.append(tk)
            for t in set(tk):
                post.setdefault(t, []).append(i)
            for t in set(a + b for a, b in zip(tk, tk[1:])):
                post2.setdefault(t, []).append(i)
        self.post = {t: np.asarray(v, dtype=np.int32) for t, v in post.items()}
        self.post2 = {t: np.asarray(v, dtype=np.int32) for t, v in post2.items()}
        self.by_prefix, self.by_stem = {}, {}
        for t in self.post:
            if any(ch.isdigit() for ch in t):
                continue
            if len(t) >= KW_DEFAULTS["prefix_min"]:
                self.by_prefix.setdefault(t[:KW_DEFAULTS["prefix_min"]], []).append(t)
            self.by_stem.setdefault(stem(t), []).append(t)
        self.doc_pos = np.asarray(hx.doc_pos)
        self._cache = {}
        self.build_ms = round((time.perf_counter() - t0) * 1000.0, 1)

    def _ids(self, t):
        a = self.post.get(t)
        b = self.post2.get(t)
        if a is None and b is None:
            return np.zeros(0, dtype=np.int32)
        if a is None:
            return b
        if b is None:
            return a
        return np.union1d(a, b)

    def similar_vocab(self, t):
        if any(ch.isdigit() for ch in t):
            return []
        cands = set(self.by_stem.get(stem(t), ()))
        if len(t) >= KW_DEFAULTS["prefix_min"]:
            cands.update(self.by_prefix.get(t[:KW_DEFAULTS["prefix_min"]], ()))
        return sorted(v for v in cands if v != t and similar(t, v))

    def match(self, keyword):
        """-> (exact chunk ids, exact-or-similar chunk ids), both sorted int32 arrays over the whole index."""
        toks = tuple(word_tokens(keyword))
        if toks in self._cache:
            return self._cache[toks]
        if not toks:
            r = (np.zeros(0, dtype=np.int32),) * 2
            self._cache[toks] = r
            return r
        solid = "".join(toks)
        if len(toks) == 1:
            exact = self._ids(toks[0])
        else:
            cand = None
            for t in toks:
                p = self.post.get(t)
                if p is None:
                    cand = np.zeros(0, dtype=np.int32)
                    break
                cand = p if cand is None else np.intersect1d(cand, p, assume_unique=True)
            hits = []
            m = len(toks)
            for c in (cand if cand is not None else []):
                tk = self.toks[int(c)]
                for j in range(len(tk) - m + 1):
                    if tk[j] == toks[0] and tuple(tk[j:j + m]) == toks:
                        hits.append(int(c))
                        break
            exact = np.union1d(np.asarray(hits, dtype=np.int32), self._ids(solid))
        ct = content_tokens(list(toks)) or [t for t in toks if t not in _FA_AFFIX_TOKENS]
        sim = None
        for t in ct:
            s = self._ids(t)
            for v in self.similar_vocab(t):
                s = np.union1d(s, self.post[v])
            sim = s if sim is None else np.intersect1d(sim, s)
            if not len(sim):
                break
        anym = np.union1d(exact, sim) if sim is not None else exact
        r = (exact.astype(np.int32), anym.astype(np.int32))
        self._cache[toks] = r
        return r


def match_text(keyword, text):
    """The same rules on one text (analysis helper): -> 'exact' | 'similar' | None."""
    toks = word_tokens(keyword)
    tk = word_tokens(text)
    if not toks or not tk:
        return None
    m = len(toks)
    solid = "".join(toks)
    grams = set(tk) | set(a + b for a, b in zip(tk, tk[1:]))
    if any(tuple(tk[j:j + m]) == tuple(toks) for j in range(len(tk) - m + 1)) or solid in grams:
        return "exact"
    ct = content_tokens(toks) or [t for t in toks if t not in _FA_AFFIX_TOKENS]
    if ct and all(t in grams or any(similar(t, v) for v in set(tk)) for t in ct):
        return "similar"
    return None


_KWI = {}


def keyword_index(hx):
    key = id(hx)
    if key not in _KWI:
        _KWI[key] = KeywordIndex(hx)
    return _KWI[key]


# --------------------------------------------------------------------------------------------------------------
# scoring / ranking
# --------------------------------------------------------------------------------------------------------------

def kw_counts(kwi, keywords, mask):
    """-> (counts over all chunks (0 outside mask), exact counts, per-keyword any-match arrays)."""
    cnt = np.zeros(kwi.n, dtype=np.int32)
    ex = np.zeros(kwi.n, dtype=np.int32)
    per = []
    for k in keywords:
        e, a = kwi.match(k)
        cnt[a] += 1
        ex[e] += 1
        per.append(a)
    cnt[~mask] = 0
    ex[~mask] = 0
    return cnt, ex, per


def kw_rank(cnt, bm25, mask, depth=None):
    idx = np.nonzero(mask & (cnt > 0))[0]
    o = np.lexsort((idx, -bm25[idx], -cnt[idx]))
    r = [int(i) for i in idx[o]]
    return r[:depth] if depth else r


def doc_rank(kwi, per, mask, pool_docs, bm25):
    """-> [(doc_id, score, n_keywords)] best first, over the pool documents."""
    hx = kwi.hx
    pos_of = {d: i for i, d in enumerate(hx.doc_ids)}
    pool_pos = [pos_of[d] for d in pool_docs if d in pos_of]
    N = len(pool_pos)
    dp = kwi.doc_pos
    docsets = []
    for a in per:
        a = a[mask[a]] if len(a) else a
        docsets.append(set(int(x) for x in np.unique(dp[a])) if len(a) else set())
    score = {p: 0.0 for p in pool_pos}
    nk = {p: 0 for p in pool_pos}
    for ds in docsets:
        if not ds:
            continue
        idf = math.log(1.0 + N / len(ds))
        for p in ds:
            if p in score:
                score[p] += idf
                nk[p] += 1
    best_bm = {p: 0.0 for p in pool_pos}
    s = np.where(mask, bm25, 0.0)
    for p in pool_pos:
        m = dp == p
        if m.any():
            best_bm[p] = float(s[m].max())
    order = sorted(pool_pos, key=lambda p: (-score[p], -best_bm[p], hx.doc_ids[p]))
    return [(hx.doc_ids[p], round(score[p], 4), nk[p]) for p in order]


def search(hx, kwi, qtext, qvec, keywords, pool_docs, doc_k=None, depth=None):
    """-> (orders dict {'kw', 'hybrid_kw'} of chunk ids best first, info). doc_k restricts HYBRID+KW to the top-k
    documents by keyword coverage (DOC), with the declared fallback."""
    from .lex4.core import rank_lists
    depth = depth or KW_DEFAULTS["depth"]
    info = {"version": KWDOC_VERSION, "n_keywords": len(keywords)}
    t0 = time.perf_counter()
    mask, _ = hx.scope_mask(pool_docs)
    bm25, _ = hx.bm25_scores(qtext)
    cnt, ex, per = kw_counts(kwi, keywords, mask)
    info["t_kw_ms"] = round((time.perf_counter() - t0) * 1000.0, 1)
    info["kw_hits"] = [int(mask[a].sum()) if len(a) else 0 for a in per]
    use = mask
    if doc_k:
        t1 = time.perf_counter()
        dr = doc_rank(kwi, per, mask, pool_docs, bm25)
        chosen = [d for d, _s, _n in dr[:int(doc_k)]]
        m2, _ = hx.scope_mask(chosen)
        m2 = m2 & mask
        thr = min(int(math.ceil(len(keywords) / 2.0)), 2) if keywords else 1
        best = int(cnt[m2].max()) if m2.any() else 0
        fb = best < thr
        use = mask if fb else m2
        info.update(doc_k=int(doc_k), docs=chosen, doc_scores=dr[:max(5, int(doc_k))], fallback=fb,
                    fallback_threshold=thr, best_count_in_docs=best,
                    t_doc_ms=round((time.perf_counter() - t1) * 1000.0, 1))
    t2 = time.perf_counter()
    kw_list = kw_rank(cnt, bm25, use)
    bm, dn = rank_lists(hx, qtext, qvec, use, depth)
    fused = H.rrf_fuse({"bm25": bm, "dense": dn, "kw": kw_list[:depth]}, k=KW_DEFAULTS["rrf_k"])
    orders = {"kw": kw_list, "hybrid_kw": [c for c, _s, _r in fused]}
    info["ranks"] = {"bm25": bm, "dense": dn, "kw": kw_list[:depth]}
    info["t_rank_ms"] = round((time.perf_counter() - t2) * 1000.0, 1)
    info["t_search_ms"] = round((time.perf_counter() - t0) * 1000.0, 1)
    info["_fused"] = fused
    info["_cnt"] = cnt
    return orders, info


# --------------------------------------------------------------------------------------------------------------
# the keyword call
# --------------------------------------------------------------------------------------------------------------

def keyword_client(cfg, ledger, model_id=None):
    from ..llm_openrouter import OpenRouterReader, model_info
    mid = model_id or KW_DEFAULTS["model"]
    info = model_info(mid, cfg.project_root / "work" / "openrouter" / "catalog.json")
    return OpenRouterReader(cfg, ledger, mid, supported=info.get("params"), attempts=KW_DEFAULTS["attempts"],
                            deadline_s=KW_DEFAULTS["deadline_s"], max_tokens_total=KW_DEFAULTS["max_tokens_total"],
                            reasoning_effort=KW_DEFAULTS["reasoning_effort"])


def generate_keywords(client, question):
    """-> dict(keywords, status, raw, usage..., t_ms). On failure the question's content words are the keywords
    (status 'fallback_question_words')."""
    msgs = [{"role": "user", "content": PROMPT.format(question=question.strip(), n=KW_DEFAULTS["n_keywords"])}]
    t = time.perf_counter()
    out = {"model": client.model_id, "provider_pin": client.provider}
    try:
        res = client.chat(msgs, purpose="kw_keywords", num_predict=KW_DEFAULTS["max_tokens_total"])
        kws = parse_keywords(res.text)
        out.update(raw=(res.text or "")[:2000], prompt_tokens=res.prompt_tokens, completion_tokens=res.completion_tokens,
                   reasoning_tokens=getattr(res, "reasoning_tokens", None), cost_usd=res.cost_usd,
                   provider=getattr(res, "provider", None), finish=res.done_reason, usage_known=True)
        out["status"] = "ok" if kws else "empty"
    except Exception as e:  # noqa: BLE001
        from ..llm import BudgetExceeded
        if isinstance(e, BudgetExceeded):
            raise
        kws = []
        out.update(status="fail", error=str(e)[:300], usage_known=False)
    if not kws:
        kws = list(dict.fromkeys(content_tokens(word_tokens(question))))[:KW_DEFAULTS["n_keywords"]]
        out["status"] = out["status"] + "->fallback_question_words"
    out["keywords"] = kws
    out["t_ms"] = round((time.perf_counter() - t) * 1000.0, 1)
    return out


def cached_keywords(question):
    p = os.environ.get("KWDOC_KEYWORDS")
    if not p or not os.path.exists(p):
        return None
    if not hasattr(cached_keywords, "_d") or cached_keywords._p != p:
        with open(p, encoding="utf-8") as f:
            cached_keywords._d = json.load(f)
        cached_keywords._p = p
    return cached_keywords._d.get(qkey(question))


# --------------------------------------------------------------------------------------------------------------
# online arm (HYBRID flags kw / doc=k)
# --------------------------------------------------------------------------------------------------------------

def answer_kw(ap, query, index):
    from ..contracts import AnswerResult
    T0 = time.perf_counter()
    hx = index.payload
    kwi = keyword_index(hx)                       # built at index load (outside the timed query) - see hybrid.py
    pool = None if query.doc_scope in (None, "corpus") else list(query.doc_scope)
    if pool is None:
        pool = list(hx.doc_ids)
    st = {"version": KWDOC_VERSION}
    kc = cached_keywords(query.text)
    if kc is not None:
        kw = dict(kc, source="cache")
    else:
        if getattr(ap, "_kwclient", None) is None:
            ap._kwclient = keyword_client(ap.cfg, ap.client.ledger)
        kw = generate_keywords(ap._kwclient, query.text)
        kw["source"] = "live"
    st["keywords"] = kw
    t = time.perf_counter()
    qvec, qinfo = hx.query_vector(query.text, ap.client)
    st["t_embed_ms"] = round((time.perf_counter() - t) * 1000.0, 1)
    doc_k = ap.flags.get("doc")
    doc_k = int(doc_k) if doc_k not in (None, True, "") else None
    orders, info = search(hx, kwi, query.text, qvec, kw["keywords"], pool, doc_k=doc_k)
    fused = info.pop("_fused")
    info.pop("_cnt")
    k = max(40, int(ap.cfg.chunking["top_k"]))
    hits = []
    for c, rrf, rk in fused[:k]:
        h = hx.hit_for_chunk(c)
        h.rrf = rrf
        h.ranks = {"bm25": rk.get("bm25"), "dense": rk.get("dense"), "kw": rk.get("kw")}
        hits.append(h)
    items = [h.to_item(rank=r) for r, h in enumerate(hits)]
    offered = ap._offer(items, index)
    ev = ap.final_pack(offered, index, query)
    st["t_retrieval_ms"] = round((time.perf_counter() - T0) * 1000.0, 1)
    t = time.perf_counter()
    answer = ap.generate_answer(query, ev)
    st["t_answer_ms"] = round((time.perf_counter() - t) * 1000.0, 1)
    info.pop("ranks", None)
    st["search"] = info
    trace = {"kwdoc": st, "retrieval_ms": st["t_retrieval_ms"], "evidence_items": ev.n_included,
             "last_item_cut": ev.last_item_cut, "offered": len(offered), "query_embed": qinfo,
             "top": [[h.chunk, round(h.rrf, 6), h.ranks.get("bm25"), h.ranks.get("dense"), h.ranks.get("kw")]
                     for h in hits]}
    return AnswerResult(answer_text=answer, evidence=ev, retrieved=items, trace=trace)
