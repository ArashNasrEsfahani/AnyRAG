"""LEX4 retrieval core (no model calls): candidate lists, intact packages, lexical concept groups with local binding,
and the two scoring/packing rules under test. Used by the offline replay; written to be reused by the approach.

Candidates (one question, one pool, supplied query vectors - one batched embed of question + requirements):
    q-bm25 top Q_DEPTH and q-dense top Q_DEPTH over the pool (exactly HYBRID's channel rankings)
    per requirement r_i: bm25 top R_DEPTH and dense top R_DEPTH over the pool, or over its source document when the
    requirement's source reference resolved to exactly one document (the q-lists stay unrestricted: escape channel)
    option pool_only: requirement lists ranked inside the question candidates only (no new candidates)

Intact package of a candidate chunk: the shared chunk, verbatim; a boundary cut mid-sentence is completed to the
next/previous sentence end (end first) with <= COMPLETE_ADD added chars in total and <= CEILING chars; the caption and
column-heading line of a table the package shows rows of are prefixed when they are outside it. No minimum length.

Local support units of a package: each sentence of a prose unit (+ its section heading path) and each table row
(+ the table's caption and heading line + section path). A requirement's lexical score over concept groups:
    groups: aspect (weight 2), subject (weight 1, when given), each constraint (weight 2); question-clause
    requirements: each content word a group (weight 2), each number a constraint (weight 2)
    group match in one support unit = best alternative: exact folded phrase (word boundaries) 1, controlled
    morphology (every content word of the alternative matches a word by stem prefix; numbers exact) 0.5, else 0
    L_i(package) = max over support units of sum(w * m) / sum(w)          (binding: co-occurrence in ONE unit)

Rule A (Astra R1): f(r) = 61/(60+r); Q = (f(q_bm25)+f(q_dense))/2; R_i = (f(r_i bm25)+f(r_i dense))/2;
    S_i = g * (0.25 Q + 0.5 R_i + 0.25 L_i), g = 0.25 on a known source mismatch;
    eligible: S_i >= 0.5, no source mismatch, and an aspect match or r_i dense rank <= 10;
    cover: repeatedly the fitting package maximising (#newly represented requirements / incremental rendered chars),
    ties: sum of S_i for those, Q, chunk; fill: maximise marginal gain per char of F = sum_i(s_i1 + 0.35 s_i2);
    redundant: contained, or > 60% overlap of the smaller span unless it adds >= 200 new chars or a new table row.
Rule B (Claude R2): per requirement plain RRF (k=60) over 5 lists: q-bm25, q-dense, r_i-bm25, r_i-dense,
    r_i-lexical (L_i > 0, by L_i desc, ties chunk order); eligible: in the fused top 3 AND (L_i > 0 or r_i dense top 3);
    cover: per requirement in order, the best eligible package unless a selected package is already eligible for it;
    fill: round-robin over the requirements' fused lists, then the q-fused list; redundant: contained or > 50% overlap.
Budget: BUDGET rendered chars exactly as contracts.pack_evidence renders (item headers, separators); no item is cut.
"""
import re
import time

import numpy as np

from ...contracts import RetrievedItem, item_header, pack_evidence
from ...evidence import _DocView, table_structure
from ...kb import labels as L
from ...normalize import fold, word_tokens
from ...retrieval import hybrid as H
from .spans import segments

CORE_VERSION = "lex4-core-v1.1"      # v1.1 (2026-09-26): rule-B package-mutation fix; package provenance
C = {"q_depth": 40, "r_depth": 20, "rrf_k": 60, "budget": 6000, "complete_add": 300, "ceiling": 1200,
     # rule B
     "b_top": 3, "b_dense_top": 3, "b_overlap": 0.5,
     # rule A (Astra R1)
     "a_wq": 0.25, "a_wr": 0.5, "a_wl": 0.25, "a_g": 0.25, "a_elig": 0.5, "a_dense_top": 10, "a_second": 0.35,
     "a_overlap": 0.6, "a_new_chars": 200,
     # lexical groups
     "w_aspect": 2.0, "w_subject": 1.0, "w_constraint": 2.0, "morph": 0.5}

_SENT_END_CH = ".!?؟؛;"


def f_rank(r):
    return 0.0 if r is None else 61.0 / (60.0 + r)


# --------------------------------------------------------------------------------------------------------------
# rank lists
# --------------------------------------------------------------------------------------------------------------

def rank_lists(hx, text, vec, mask, depth):
    """-> (bm25 list, dense list) of chunk indices, best first, exactly as HybridIndex.search ranks each channel."""
    order_idx = np.arange(hx.n)
    s, _ = hx.bm25_scores(text)
    valid = mask & (s > 0.0)
    o = np.lexsort((order_idx, -s))
    bm = [int(i) for i in o[valid[o]][:depth]]
    dn = []
    if vec is not None and hx.E.shape[1]:
        q = np.asarray(vec, dtype=np.float32)
        q = q / (float(np.linalg.norm(q)) or 1.0)
        d = (hx.E @ q).astype(np.float32)
        valid = mask & hx.has_vec
        o = np.lexsort((order_idx, -d))
        dn = [int(i) for i in o[valid[o]][:depth]]
    return bm, dn


# --------------------------------------------------------------------------------------------------------------
# packages
# --------------------------------------------------------------------------------------------------------------

def _clean_start(text, a):
    if a <= 0:
        return True
    j = a - 1
    while j >= 0 and text[j] in " \t":
        j -= 1
    return j < 0 or text[j] == "\n" or text[j] in _SENT_END_CH


def _clean_end(text, b):
    if b >= len(text):
        return True
    j = b - 1
    while j >= 0 and text[j].isspace():
        j -= 1
    return text[b] == "\n" or (j >= 0 and text[j] in _SENT_END_CH and (b >= len(text) or text[b].isspace()))


def _next_end(text, b, limit):
    for i in range(b, min(len(text), b + limit)):
        ch = text[i]
        if ch == "\n":
            return i
        if ch in _SENT_END_CH and (i + 1 >= len(text) or text[i + 1].isspace()):
            return i + 1
    return None


def _prev_start(text, a, limit):
    for i in range(a - 1, max(-1, a - 1 - limit), -1):
        if i < 0:
            return 0
        ch = text[i]
        if ch == "\n":
            return i + 1
        if ch in _SENT_END_CH and i + 1 < len(text) and text[i + 1].isspace():
            k = i + 1
            while k < a and text[k].isspace():
                k += 1
            return k
    return 0 if a - limit <= 0 else None


class Packager:
    """Builds packages and their local support units; per-document views are cached (read-only structures)."""

    def __init__(self, hx, corpus):
        self.hx, self.corpus = hx, corpus
        self._views = {}
        self._pk = {}

    def view(self, doc_id):
        if doc_id not in self._views:
            self._views[doc_id] = _DocView(self.corpus.doc(doc_id))
        return self._views[doc_id]

    def package(self, ci):
        if ci in self._pk:
            return self._pk[ci]
        c = self.hx.chunks[ci]
        doc = self.corpus.doc(c["doc_id"])
        text = doc.text
        a, b = int(c["span"][0]), int(c["span"][1])
        while a < b and text[a].isspace():
            a += 1
        while b > a and text[b - 1].isspace():
            b -= 1
        a0, b0 = a, b
        added = 0
        ceiling, allow = C["ceiling"], C["complete_add"]
        if not _clean_end(text, b):
            e = _next_end(text, b, allow)
            if e is not None and e - a <= ceiling:
                added += e - b
                b = e
        if not _clean_start(text, a) and added < allow:
            s = _prev_start(text, a, allow - added)
            if s is not None and b - s <= ceiling:
                added += a - s
                a = s
        while a < b and text[a].isspace():
            a += 1
        while b > a and text[b - 1].isspace():
            b -= 1
        body = text[a:b]
        units = doc.units_overlapping(a, b)
        view = self.view(doc.id)
        prefix, prefix_src, sus, seen_tables = [], [], [], set()
        for u in units:
            ua, ub = max(u.span[0], a), min(u.span[1], b)
            if ub <= ua:
                continue
            head = " > ".join(t for t in doc.section_path(u.section_id) if t) if u.section_id else ""
            if u.kind == "table":
                ts = table_structure(doc, view, u)
                th = " ".join([x[0] for x in ts["caption"]] + ([ts["header"][0]] if ts["header"] else []))
                if ts["key"] not in seen_tables:
                    seen_tables.add(ts["key"])
                    for x in ts["caption"] + ([ts["header"]] if ts["header"] else []):
                        if not (a <= x[1][0] and x[1][1] <= b) and x[0].strip():
                            prefix.append(x[0].strip())
                            prefix_src.append((int(x[1][0]), int(x[1][1])))
                for _j, ln, sp in ts["rows"].get(u.id, []):
                    if sp[1] > a and sp[0] < b and ln.strip():
                        seg = text[max(sp[0], a):min(sp[1], b)]
                        sus.append({"text": fold(seg), "head": fold(head + " " + th), "row": True,
                                    "span": [max(sp[0], a), min(sp[1], b)]})
            else:
                piece = text[ua:ub]
                for sa, sb in segments(piece):
                    sus.append({"text": fold(piece[sa:sb]), "head": fold(head), "row": False,
                                "span": [ua + sa, ua + sb]})
        keep = {}
        for t, sp in zip(prefix, prefix_src):
            keep.setdefault(t, sp)
        prefix = list(keep)
        shown = ("\n".join(prefix) + "\n" + body) if prefix else body
        # provenance (2026-09-26, LEX5 spec 2): every displayed source interval, table prefixes included
        prov = tuple([(doc.id, sp[0], sp[1], "prefix") for sp in keep.values()] + [(doc.id, a, b, "body")])
        pk = {"chunk": ci, "doc_id": doc.id, "section_id": c["section_id"], "span": [a, b], "chunk_span": [a0, b0],
              "added": added, "text": shown, "prefix_chars": len("\n".join(prefix)) if prefix else 0,
              "unit_ids": [u.id for u in units], "supports": sus, "chars": len(shown), "provenance": prov}
        self._pk[ci] = pk
        return pk

    def item(self, pk):
        return RetrievedItem(text=pk["text"], doc_id=pk["doc_id"], section_id=pk["section_id"],
                             unit_ids=list(pk["unit_ids"]), kind="package", score=None, rank=None,
                             span=list(pk["span"]) if not pk["prefix_chars"] else None,
                             meta={"chunk": pk["chunk"], "completed": pk["added"], "prefix": pk["prefix_chars"],
                                   # every displayed source interval (2026-09-26, LEX6 / Astra R2 section 4)
                                   "provenance": [list(x) for x in pk.get("provenance") or ()]})

    def cost(self, pk, n):
        """Rendered chars of pk as item n (1-based) including the separator."""
        return (2 if n > 1 else 0) + len(item_header(n, self.item(pk), self.corpus)) + 1 + len(pk["text"].strip())


# --------------------------------------------------------------------------------------------------------------
# lexical concept groups
# --------------------------------------------------------------------------------------------------------------

def _stem(t):
    if any(ch.isdigit() for ch in t) or len(t) < 5:
        return t
    return t[: max(4, len(t) - 2)]


class Alt:
    __slots__ = ("phrase", "rx", "stems")

    def __init__(self, text):
        self.phrase = fold(text)
        self.rx = re.compile(r"(?<!\w)" + re.escape(self.phrase) + r"(?!\w)") if self.phrase else None
        toks = [t for t in word_tokens(text) if t not in L._STOP]
        self.stems = [(_stem(t), any(ch.isdigit() for ch in t)) for t in toks]

    def match(self, s, words):
        if not self.phrase:
            return 0.0
        if self.phrase in s and self.rx.search(s):
            return 1.0
        if self.stems:
            for st, num in self.stems:
                if num:
                    if st not in words:
                        return 0.0
                elif not any(w.startswith(st) for w in words):
                    return 0.0
            return C["morph"]
        return 0.0


def concept_groups(req):
    """-> [(kind, weight, [Alt])]."""
    g = []
    if req.get("origin") == "question_clause":
        for w in req.get("clause_words") or []:
            g.append(("aspect", C["w_aspect"], [Alt(w)]))
    else:
        if req.get("aspect"):
            g.append(("aspect", C["w_aspect"], [Alt(x) for x in req["aspect"]]))
        if req.get("subject"):
            g.append(("subject", C["w_subject"], [Alt(x) for x in req["subject"]]))
    for grp in req.get("constraints") or []:
        g.append(("constraint", C["w_constraint"], [Alt(x) for x in grp]))
    return g


def lexical(pk, groups):
    """-> (L, aspect_match) of one package for one requirement's concept groups (binding in one support unit)."""
    if not groups:
        return 0.0, False
    tot = sum(w for _, w, _ in groups)
    best, asp = 0.0, False
    for su in pk["supports"]:
        s = su["text"] + " ‖ " + su["head"]
        words = None
        acc = 0.0
        for kind, w, alts in groups:
            m = 0.0
            for al in alts:
                if al.phrase and al.phrase in s:
                    m = max(m, al.match(s, ()))
                    if m == 1.0:
                        break
            if m < 1.0:
                if words is None:
                    words = set(word_tokens(su["text"] + " " + su["head"], folded=False))
                for al in alts:
                    m = max(m, al.match(s, words))
                    if m == 1.0:
                        break
            if m > 0 and kind == "aspect":
                asp = True
            acc += w * m
        best = max(best, acc / tot)
        if best >= 1.0:
            break
    return best, asp


# --------------------------------------------------------------------------------------------------------------
# redundancy
# --------------------------------------------------------------------------------------------------------------

def _overlap(p, q):
    if p["doc_id"] != q["doc_id"]:
        return 0, False
    a, b = p["span"]
    c, d = q["span"]
    ov = max(0, min(b, d) - max(a, c))
    contained = c <= a and b <= d
    return ov, contained


def redundant_b(pk, selected, thr):
    for q in selected:
        ov, cont = _overlap(pk, q)
        if cont:
            return True
        small = min(pk["span"][1] - pk["span"][0], q["span"][1] - q["span"][0]) or 1
        if ov / small > thr:
            return True
    return False


def redundant_a(pk, selected):
    covered = []
    for q in selected:
        ov, cont = _overlap(pk, q)
        if cont:
            return True
        if ov:
            covered.append(q["span"])
    if not covered:
        return False
    big = False
    for q in selected:
        ov, _ = _overlap(pk, q)
        small = min(pk["span"][1] - pk["span"][0], q["span"][1] - q["span"][0]) or 1
        if ov / small > C["a_overlap"]:
            big = True
    if not big:
        return False
    a, b = pk["span"]
    cov = np.zeros(b - a, dtype=bool)
    for c, d in covered:
        cov[max(c, a) - a:max(0, min(d, b) - a)] = True
    if int((~cov).sum()) >= C["a_new_chars"]:
        return False
    for su in pk["supports"]:
        if su["row"] and not cov[max(su["span"][0], a) - a:max(0, min(su["span"][1], b) - a)].all():
            return False
    return True


# --------------------------------------------------------------------------------------------------------------
# one question
# --------------------------------------------------------------------------------------------------------------

class Question:
    """Candidate lists and per-requirement signals of one question (shared by the rules)."""

    def __init__(self, hx, packager, pool_docs, qtext, qvec, reqs, rvecs, pool_only=False, lex=True):
        self.hx, self.P, self.reqs = hx, packager, reqs
        t0 = time.perf_counter()
        mask, _ = hx.scope_mask(pool_docs)
        self.q_bm, self.q_dn = rank_lists(hx, qtext, qvec, mask, C["q_depth"])
        self.q_pool = list(dict.fromkeys(self.q_bm + self.q_dn))
        qset = np.zeros(hx.n, dtype=bool)
        qset[self.q_pool] = True
        self.r_bm, self.r_dn = [], []
        for r, v in zip(reqs, rvecs):
            if r.get("source_doc"):
                m, _ = hx.scope_mask([r["source_doc"]])
                m = m & mask
            else:
                m = mask
            if pool_only:
                m = m & qset
            bm, dn = rank_lists(hx, r["text"], v, m, C["r_depth"])
            self.r_bm.append(bm)
            self.r_dn.append(dn)
        cands = list(self.q_pool)
        for bm, dn in zip(self.r_bm, self.r_dn):
            cands += bm + dn
        self.cands = list(dict.fromkeys(cands))
        self.t_lists = time.perf_counter() - t0
        t0 = time.perf_counter()
        self.pk = {c: packager.package(c) for c in self.cands}
        self.t_pack = time.perf_counter() - t0
        t0 = time.perf_counter()
        self.L, self.asp = [], []
        for r in reqs:
            groups = concept_groups(r) if lex else []
            li, ai = {}, {}
            for c in self.cands:
                l, a = lexical(self.pk[c], groups) if groups else (0.0, False)
                li[c], ai[c] = l, a
            self.L.append(li)
            self.asp.append(ai)
        self.t_lex = time.perf_counter() - t0
        self.q_rank_bm = {c: i + 1 for i, c in enumerate(self.q_bm)}
        self.q_rank_dn = {c: i + 1 for i, c in enumerate(self.q_dn)}
        self.r_rank_bm = [{c: i + 1 for i, c in enumerate(x)} for x in self.r_bm]
        self.r_rank_dn = [{c: i + 1 for i, c in enumerate(x)} for x in self.r_dn]

    def q_fused(self):
        return [cid for cid, _s, _r in H.rrf_fuse({"bm25": self.q_bm, "dense": self.q_dn}, k=C["rrf_k"])]

    def mismatch(self, i, c):
        src = self.reqs[i].get("source_doc")
        return bool(src) and self.pk[c]["doc_id"] != src


class Packer:
    """Budgeted selection with exact rendered cost."""

    def __init__(self, P, budget):
        self.P, self.budget = P, budget
        self.sel, self.used, self.ids = [], 0, set()

    def inc(self, pk):
        return self.P.cost(pk, len(self.sel) + 1)

    def fits(self, pk):
        return self.used + self.inc(pk) <= self.budget

    def add(self, pk):
        self.used += self.inc(pk)
        self.sel.append(pk)
        self.ids.add(pk["chunk"])


def run_rule_b(Qn, lexical_list=True, cover=True, plain=False):
    P = Qn.P
    lists, fused, franks, fscore = [], [], [], []
    for i in range(len(Qn.reqs)):
        rk = {"q_bm25": Qn.q_bm, "q_dense": Qn.q_dn, "r_bm25": Qn.r_bm[i], "r_dense": Qn.r_dn[i]}
        if lexical_list:
            lx = [c for c in Qn.cands if Qn.L[i][c] > 0]
            lx.sort(key=lambda c: (-Qn.L[i][c], c))
            rk["r_lex"] = lx
        fu = H.rrf_fuse(rk, k=C["rrf_k"])
        fused.append([c for c, _s, _r in fu])
        franks.append({c: j + 1 for j, (c, _s, _r) in enumerate(fu)})
        fscore.append({c: s for c, s, _r in fu})
        lists.append(rk)

    def eligible(i, c):
        if franks[i].get(c, 10 ** 9) > C["b_top"]:
            return False
        if not lexical_list:
            return True
        return Qn.L[i][c] > 0 or Qn.r_rank_dn[i].get(c, 10 ** 9) <= C["b_dense_top"]

    pk = Qn.pk
    B = Packer(P, C["budget"])
    covered = {}
    thr = C["b_overlap"]
    if plain:
        order = sorted(Qn.cands, key=lambda c: (-max(fscore[i].get(c, 0.0) for i in range(len(Qn.reqs))), c)) \
            if Qn.reqs else Qn.q_fused()
        for c in order:
            p = pk[c]
            if not redundant_b(p, B.sel, thr) and B.fits(p):
                B.add(p)
        return B, {"covered": {}}
    if cover:
        for i in range(len(Qn.reqs)):
            if any(eligible(i, c) for c in B.ids):     # fixed 2026-09-26 (was `for p["chunk"] in B.ids`, which
                #                                           wrote every selected id into the last package dict)
                covered[Qn.reqs[i]["id"]] = "by_selected"
                continue
            for c in fused[i]:
                if franks[i][c] > C["b_top"]:
                    break
                if not eligible(i, c):
                    continue
                p = pk[c]
                if redundant_b(p, B.sel, thr) or not B.fits(p):
                    continue
                B.add(p)
                covered[Qn.reqs[i]["id"]] = c
                break
    ptr = [0] * len(Qn.reqs)
    active = True
    while active:
        active = False
        for i in range(len(Qn.reqs)):
            while ptr[i] < len(fused[i]):
                c = fused[i][ptr[i]]
                ptr[i] += 1
                p = pk[c]
                if p["chunk"] in B.ids or redundant_b(p, B.sel, thr) or not B.fits(p):
                    continue
                B.add(p)
                active = True
                break
    for c in Qn.q_fused():
        p = pk[c]
        if p not in B.sel and not redundant_b(p, B.sel, thr) and B.fits(p):
            B.add(p)
    return B, {"covered": covered}


def run_rule_a(Qn):
    P, pk = Qn.P, Qn.pk
    n = len(Qn.reqs)
    Q = {c: 0.5 * f_rank(Qn.q_rank_bm.get(c)) + 0.5 * f_rank(Qn.q_rank_dn.get(c)) for c in Qn.cands}
    S, E = [], []
    for i in range(n):
        si, ei = {}, {}
        for c in Qn.cands:
            R = 0.5 * f_rank(Qn.r_rank_bm[i].get(c)) + 0.5 * f_rank(Qn.r_rank_dn[i].get(c))
            g = C["a_g"] if Qn.mismatch(i, c) else 1.0
            s = g * (C["a_wq"] * Q[c] + C["a_wr"] * R + C["a_wl"] * Qn.L[i][c])
            si[c] = s
            ei[c] = (s >= C["a_elig"] and not Qn.mismatch(i, c)
                     and (Qn.asp[i][c] or Qn.r_rank_dn[i].get(c, 10 ** 9) <= C["a_dense_top"]))
        S.append(si)
        E.append(ei)
    B = Packer(P, C["budget"])
    rep = set()
    covered = {}
    while True:
        best = None
        for c in Qn.cands:
            p = pk[c]
            if p["chunk"] in B.ids:
                continue
            new = [i for i in range(n) if i not in rep and E[i][c]]
            if not new or not B.fits(p) or redundant_a(p, B.sel):
                continue
            key = (len(new) / B.inc(p), sum(S[i][c] for i in new), Q[c], -c)
            if best is None or key > best[0]:
                best = (key, c, new)
        if best is None:
            break
        B.add(pk[best[1]])
        for i in best[2]:
            rep.add(i)
            covered[Qn.reqs[i]["id"]] = best[1]

    def F(sel):
        tot = 0.0
        for i in range(n):
            v = sorted((S[i][p["chunk"]] for p in sel if E[i][p["chunk"]]), reverse=True)
            if v:
                tot += v[0] + (C["a_second"] * v[1] if len(v) > 1 else 0.0)
        return tot

    while True:
        f0 = F(B.sel)
        best = None
        for c in Qn.cands:
            p = pk[c]
            if p["chunk"] in B.ids or not any(E[i][c] for i in range(n)) or not B.fits(p) or redundant_a(p, B.sel):
                continue
            gain = F(B.sel + [p]) - f0
            if gain <= 1e-12:
                continue
            key = (gain / B.inc(p), Q[c], -c)
            if best is None or key > best[0]:
                best = (key, c)
        if best is None:
            break
        B.add(pk[best[1]])
    return B, {"covered": covered, "eligible_counts": [sum(1 for c in Qn.cands if E[i][c]) for i in range(n)]}


def run_hybrid_intact(Qn):
    B = Packer(Qn.P, C["budget"])
    for c in Qn.q_fused():
        p = Qn.pk[c]
        if not redundant_b(p, B.sel, C["b_overlap"]) and B.fits(p):
            B.add(p)
    return B, {}


def render(P, packer, cfg):
    items = [P.item(p) for p in packer.sel]
    ev = pack_evidence(items, P.corpus, budget=C["budget"], cfg=cfg)
    return ev
