"""LEX8 evidence packers over compact source-unit candidates and full chunks (Astra R1, design step 4).

Candidate: {key, doc_id, section_id, a, b, deps [[a, b]], units [unit ids fully displayed], score}. Its displayed text is
Document.text[a:b] (+ dependency lines outside it, e.g. a table's caption / column heading).

Coverage packer (P-cov):
    A(S) = source units whose body lies fully inside the displayed text of S;  F(S) = sum_{a in A(S)} u(a)^2
    Displayed intervals of one document that overlap or touch (only whitespace between) are merged into ONE item;
    disjoint intervals stay separate items; dependency lines outside an item are prefixed to it.
    Cost C(S) = exact rendered chars of the whole pack as contracts.pack_evidence renders it ("[n] title > path"
    headers, "\\n\\n" separators, prefixes).
    Three deterministic packs: greedy max dF, greedy max dF/dC, best single fitting candidate; the result with the
    highest F wins (ties: lower cost, then candidate keys). A candidate that does not fit is skipped, never cut.
    Items are ordered by their best unit utility (ties: doc id, offset).
Ranked packer (ablation C1): candidates by score; one item each; a candidate adding no new unit is skipped;
    contracts.pack_evidence semantics (whole items while they fit, the first misfit cut, stop).
"""
import bisect
from types import SimpleNamespace

from ...contracts import RetrievedItem, item_header, pack_evidence

PACKER_VERSION = "lex8-packer-v1"
BUDGET = 6000


def header_total(n):
    """rendered chars of n headers' "[k] " parts plus the n-1 separators."""
    return sum(len(str(k)) + 3 for k in range(1, n + 1)) + 2 * max(0, n - 1)


class Doc:
    """per-document helpers: text, unit offsets (sorted), header path."""

    def __init__(self, doc, unit_ids, ua, ub, corpus):
        self.doc, self.text = doc, doc.text
        self.uids = sorted(unit_ids, key=lambda i: (ua[i], ub[i]))
        self.starts = [int(ua[i]) for i in self.uids]
        self.ub = ub
        self.corpus = corpus
        self._path = {}

    def path(self, section_id):
        if section_id not in self._path:
            h = item_header(1, SimpleNamespace(doc_id=self.doc.id, section_id=section_id), self.corpus)
            self._path[section_id] = h[len("[1] "):]
        return self._path[section_id]

    def units_in(self, A, B):
        i = bisect.bisect_left(self.starts, A)
        out = []
        while i < len(self.uids) and self.starts[i] < B:
            u = self.uids[i]
            if int(self.ub[u]) <= B:
                out.append(u)
            i += 1
        return out


def trim(text, a, b):
    while a < b and text[a].isspace():
        a += 1
    while b > a and text[b - 1].isspace():
        b -= 1
    return a, b


def render_doc(D, cands):
    """-> list of items {A, B, section_id, prefix [[a, b]], text, fixed (= len(path) + 1 + len(text)), units}."""
    iv = sorted(((c["a"], c["b"], c) for c in cands), key=lambda t: (t[0], t[1]))
    merged = []
    for a, b, c in iv:
        if merged:
            m = merged[-1]
            if a <= m["b"] or not D.text[m["b"]:a].strip():
                m["b"] = max(m["b"], b)
                m["deps"].extend(c["deps"])
                continue
        merged.append({"a": a, "b": b, "deps": list(c["deps"]), "section_id": c["section_id"]})
    items = []
    for m in merged:
        A, B = trim(D.text, m["a"], m["b"])
        pre, seen = [], set()
        for da, db in sorted(tuple(x) for x in m["deps"]):
            if (da >= A and db <= B) or (da, db) in seen or not D.text[da:db].strip():
                continue
            seen.add((da, db))
            pre.append([da, db])
        body = D.text[A:B]
        text = ("\n".join(D.text[da:db].strip() for da, db in pre) + "\n" + body) if pre else body
        text = text.strip()
        items.append({"doc_id": D.doc.id, "A": A, "B": B, "section_id": m["section_id"], "prefix": pre, "text": text,
                      "fixed": len(D.path(m["section_id"])) + 1 + len(text), "units": D.units_in(A, B)})
    return items


class State:
    def __init__(self, docs, u2):
        self.docs, self.u2 = docs, u2
        self.sel = {}            # doc -> [cands]
        self.res = {}            # doc -> (items, fixed, n, F)
        self.fixed = self.n = 0
        self.F = 0.0
        self.keys = []
        self.ver = {}            # doc -> version (bumped on every add; try_add results are cached per version)
        self._cache = {}

    def doc_eval(self, d, cands):
        items = render_doc(self.docs[d], cands)
        return items, sum(i["fixed"] for i in items), len(items), sum(self.u2[u] for i in items for u in i["units"])

    def try_add(self, c):
        d = c["doc_id"]
        _items, fx0, n0, F0 = self.res.get(d, ([], 0, 0, 0.0))
        ck = (id(c), self.ver.get(d, 0))
        r = self._cache.get(ck)
        if r is None:
            r = self._cache[ck] = self.doc_eval(d, self.sel.get(d, []) + [c])
        fixed = self.fixed - fx0 + r[1]
        n = self.n - n0 + r[2]
        F = self.F - F0 + r[3]
        return fixed + header_total(n), F, r

    def cost(self):
        return self.fixed + header_total(self.n)

    def add(self, c, r):
        d = c["doc_id"]
        _items, fx0, n0, F0 = self.res.get(d, ([], 0, 0, 0.0))
        self.sel.setdefault(d, []).append(c)
        self.res[d] = r
        self.ver[d] = self.ver.get(d, 0) + 1
        self.fixed += r[1] - fx0
        self.n += r[2] - n0
        self.F += r[3] - F0
        self.keys.append(c["key"])

    def items(self):
        return [it for r in self.res.values() for it in r[0]]


def _greedy(docs, u2, cands, budget, ratio):
    st = State(docs, u2)
    left = list(cands)
    while True:
        best = None
        c0 = st.cost()
        for k, c in enumerate(left):
            cost, F, r = st.try_add(c)
            dF, dC = F - st.F, cost - c0
            if cost > budget or dF <= 1e-12:
                continue
            key = ((dF / max(dC, 1)) if ratio else dF, -dC, -k)
            if best is None or key > best[0]:
                best = (key, k, r)
        if best is None:
            return st
        _key, k, r = best
        st.add(left[k], r)
        left.pop(k)


def coverage_pack(docs, u, cands, budget=BUDGET):
    """-> (winning State, info). u: {unit id: utility} (or an array)."""
    u2 = _Sq(u)
    runs = {"greedy_dF": _greedy(docs, u2, cands, budget, False), "greedy_ratio": _greedy(docs, u2, cands, budget, True)}
    best1 = None
    for c in cands:
        st = State(docs, u2)
        cost, F, r = st.try_add(c)
        if cost <= budget and (best1 is None or (F, -cost) > (best1[0], -best1[1])):
            best1 = (F, cost, c, r)
    st1 = State(docs, u2)
    if best1:
        st1.add(best1[2], best1[3])
    runs["best_single"] = st1
    name = max(runs, key=lambda k: (round(runs[k].F, 12), -runs[k].cost(), [str(x) for x in sorted(runs[k].keys, key=str)]))
    info = {k: {"F": round(v.F, 6), "cost": v.cost(), "n_cands": len(v.keys)} for k, v in runs.items()}
    info["chosen"] = name
    return runs[name], info


class _Sq:
    def __init__(self, u):
        self.u = u
        self.c = {}

    def __getitem__(self, i):
        if i not in self.c:
            self.c[i] = float(self.u[i]) ** 2
        return self.c[i]


def to_items(items, u):
    """ordered RetrievedItems (best unit utility first; ties doc id, offset) with exact source provenance."""
    def best(it):
        return max((float(u[x]) for x in it["units"]), default=0.0)
    items = sorted(items, key=lambda it: (-best(it), it["doc_id"], it["A"]))
    out = []
    for r, it in enumerate(items):
        prov = [[it["doc_id"], a, b, "prefix"] for a, b in it["prefix"]] + [[it["doc_id"], it["A"], it["B"], "body"]]
        out.append(RetrievedItem(text=it["text"], doc_id=it["doc_id"], section_id=it["section_id"], unit_ids=[],
                                 kind="lex8", score=round(best(it), 6), rank=r,
                                 span=None if it["prefix"] else [it["A"], it["B"]],
                                 meta={"provenance": prov, "units": list(it["units"]), "best_u": round(best(it), 6)}))
    return out


def final_pack(items, corpus, cfg=None, budget=BUDGET, expect_chars=None):
    ev = pack_evidence(items, corpus, budget=budget, cfg=cfg)
    if ev.n_included != len(items) or ev.last_item_cut:
        raise AssertionError("coverage pack does not fit as computed (%d of %d items, cut %s)" % (
            ev.n_included, len(items), ev.last_item_cut))
    if expect_chars is not None and ev.chars != expect_chars:
        raise AssertionError("rendered %d chars, computed %d" % (ev.chars, expect_chars))
    return ev


def ranked_pack(docs, cands, corpus, cfg=None, budget=BUDGET):
    """ablation C1: candidates by (score desc, cost, key); one item per candidate; skip if it adds no new unit;
    pack_evidence semantics (first misfit cut, stop)."""
    seen, items = set(), []
    for c in cands:
        new = set(c["units"]) - seen
        if not new:
            continue
        seen |= set(c["units"])
        D = docs[c["doc_id"]]
        it = render_doc(D, [c])[0]
        items.append(RetrievedItem(text=it["text"], doc_id=it["doc_id"], section_id=it["section_id"], unit_ids=[],
                                   kind="lex8", score=c["score"], rank=len(items),
                                   span=None if it["prefix"] else [it["A"], it["B"]],
                                   meta={"provenance": [[it["doc_id"], a, b, "prefix"] for a, b in it["prefix"]]
                                         + [[it["doc_id"], it["A"], it["B"], "body"]], "key": list(c["key"])}))
    return pack_evidence(items, corpus, budget=budget, cfg=cfg)
