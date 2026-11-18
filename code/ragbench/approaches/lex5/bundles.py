"""LEX5 (converged design, work/analysis/LEX5_DESIGN_FINAL.md): the selector's bundle grammar, its strict parser, and
the packers. No model calls, no index: everything here works on immutable packages and is unit-tested.

Selector reply grammar (spec 4):
    parts=N                       1 <= N <= MAX_PARTS
    P1: 3+9 | 12                  '+' joins jointly necessary passages (one support bundle); '|' separates
    P2: NONE                      alternative sufficient bundles (<= MAX_ALTS, best first); NONE = no passage supports it
    END
Strict validation (spec 5): anything else is rejected - no repair. Rejections: empty, no parts line, bad N, overflow
(N > MAX_PARTS or > MAX_ALTS alternatives), missing / duplicate / out-of-order part index, a part count that differs
from N, invalid or out-of-range ids, an id repeated inside one bundle, two alternatives of one part that are the same
passage SET in any order ('1+2 | 2+1'; v1.1), an empty bundle, prose lines, missing END, text
after END, a truncated reply (finish_reason 'length'). Tolerated (kept from the v2 parser, documented): Persian and
Arabic-Indic digits, the Persian word for none, invisible direction marks, a surrounding ``` fence, blank lines.
All-NONE is a VALID reply (status 'all_none'), distinct from a failure in telemetry.

Packing (spec 6), over packages that are immutable Mappings with keys chunk, doc_id, text, provenance:
    represented(p, S)  p is shown by S: some q in S whose provenance intervals contain every interval of p (exact
                       duplicates and fully contained packages; partial overlaps are kept)
    canonical(order)   drop an item represented by another shown item (the first of equals is kept)
    exact              enumerate one alternative or none per part (<= 3^8 choices); among feasible unions (exact
                       rendered cost <= budget) maximise covered parts, then minimise the summed alternative rank of
                       the covered parts (best satisfied alternative: preferred 0, second 1), then the rendered chars,
                       then the RRF ranks and ids of the shown packages. Render by part order, then bundle order.
    greedy (ablation)  parts in order, the first alternative that still fits whole
    flat (ablation)    the listed ids one by one (first id of each part first), no bundle completeness
    fill               remaining selector bundles (complete bundles only), then RRF packages; skip what does not fit
    fallback           HYBRID-intact: RRF packages in order under the same rules (no selector)
Coverage is selector-PREDICTED support, not verified entailment.
"""
import itertools
import re

BUNDLES_VERSION = "lex5-bundles-v1.1"   # v1.1 (2026-10-01): duplicate alternatives compared as unordered sets
MAX_PARTS = 8
MAX_ALTS = 2

_DIGITS = str.maketrans("۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩", "01234567890123456789")
_INVIS = dict.fromkeys(map(ord, "‌‍‎‏؜‪‫‬‭‮﻿"), None)
_PARTS = re.compile(r"^parts\s*=\s*(\d+)$", re.I)
_LINE = re.compile(r"^P(\d+)\s*:\s*(.*)$", re.I)
_NONE = re.compile(r"^(?:none|هیچ)$", re.I)
_IDS = re.compile(r"^\d+(?:\s*\+\s*\d+)*$")


def parse(text, n, finish_reason=None):
    """-> dict(ok, status, reason, n_parts, parts) where parts = [[bundle, ...] per part], bundle = tuple of 1-based
    candidate numbers; status in ok | all_none | fail."""
    out = {"ok": False, "status": "fail", "reason": None, "n_parts": None, "parts": []}
    if finish_reason == "length":
        out["reason"] = "truncated"
        return out
    lines = [ln.translate(_INVIS).translate(_DIGITS).strip() for ln in (text or "").splitlines()]
    lines = [ln for ln in lines if ln]
    if lines and lines[0].startswith("```") and lines[-1] == "```":      # a surrounding fence only
        lines = lines[1:-1]
    if not lines:
        out["reason"] = "empty"
        return out
    m = _PARTS.match(lines[0])
    if not m:
        out["reason"] = "no_parts_line"
        return out
    N = int(m.group(1))
    out["n_parts"] = N
    if N < 1:
        out["reason"] = "bad_part_count"
        return out
    if N > MAX_PARTS:
        out["reason"] = "overflow_parts"
        return out
    if "END" not in [ln.upper() for ln in lines]:
        out["reason"] = "no_end"
        return out
    end = [ln.upper() for ln in lines].index("END")
    if end != len(lines) - 1:
        out["reason"] = "text_after_end"
        return out
    body = lines[1:end]
    if len(body) != N:
        out["reason"] = "part_count_mismatch"
        return out
    parts = []
    for k, ln in enumerate(body, 1):
        m = _LINE.match(ln)
        if not m:
            out["reason"] = "bad_line"
            return out
        if int(m.group(1)) != k:
            out["reason"] = "part_index"
            return out
        rhs = m.group(2).strip()
        if _NONE.match(rhs):
            parts.append([])
            continue
        alts = [a.strip() for a in rhs.split("|")]
        if len(alts) > MAX_ALTS:
            out["reason"] = "overflow_alternatives"
            return out
        bundles = []
        for a in alts:
            if not a:
                out["reason"] = "empty_bundle"
                return out
            if not _IDS.match(a):
                out["reason"] = "bad_ids"
                return out
            ids = [int(x) for x in a.split("+")]
            if any(i < 1 or i > n for i in ids):
                out["reason"] = "id_out_of_range"
                return out
            if len(set(ids)) != len(ids):
                out["reason"] = "duplicate_id_in_bundle"
                return out
            bundles.append(tuple(ids))
        # alternatives must be DIFFERENT passage sets: compared as unordered sets ('1+2 | 2+1' is one set twice);
        # the presentation order inside each bundle is kept as written (FREEZE_ASTRA_R1 fix, v1.1)
        if len({frozenset(b) for b in bundles}) != len(bundles):
            out["reason"] = "duplicate_alternative"
            return out
        parts.append(bundles)
    out["parts"] = parts
    out["ok"] = True
    out["status"] = "all_none" if all(not p for p in parts) else "ok"
    return out


# ------------------------------------------------------------------------------------------------------------------
# representation / canonical order / exact rendered cost
# ------------------------------------------------------------------------------------------------------------------

def represented_by(p, q):
    """True when q displays every source interval p displays (same document)."""
    for d, a, b, _r in p["provenance"]:
        if not any(d == d2 and a2 <= a and b <= b2 for d2, a2, b2, _r2 in q["provenance"]):
            return False
    return True


def represented(p, shown):
    return any(q is p or q["chunk"] == p["chunk"] or represented_by(p, q) for q in shown)


def canonical(order):
    """Drop items represented by another item of the list (exact duplicates keep the first)."""
    out = []
    for i, p in enumerate(order):
        drop = False
        for j, q in enumerate(order):
            if i == j:
                continue
            if q["chunk"] == p["chunk"] or represented_by(p, q):
                mutual = represented_by(q, p)
                if not mutual or j < i:           # strictly contained -> drop; equals -> keep the first
                    drop = True
                    break
        if not drop:
            out.append(p)
    return out


class Coster:
    """Exact rendered cost of an ordered item list, as contracts.pack_evidence renders it (separator + '[n] header'
    + newline + stripped body); cost_fn(pk, n) is lex4 Packager.cost, cached per (chunk, n)."""

    def __init__(self, cost_fn):
        self.fn, self.cache = cost_fn, {}

    def __call__(self, order):
        tot = 0
        for n, p in enumerate(order, 1):
            k = (p["chunk"], n)
            if k not in self.cache:
                self.cache[k] = self.fn(p, n)
            tot += self.cache[k]
        return tot


# ------------------------------------------------------------------------------------------------------------------
# packers
# ------------------------------------------------------------------------------------------------------------------

def _covered(parts_pk, shown):
    """-> (covered part count, summed best-alternative rank over covered parts)."""
    cov, rank = 0, 0
    for alts in parts_pk:
        best = None
        for ai, b in enumerate(alts):
            if all(represented(p, shown) for p in b):
                best = ai
                break
        if best is not None:
            cov += 1
            rank += best
    return cov, rank


def pack_exact(parts_pk, rrf_rank, cost, budget):
    """parts_pk: [[bundle (tuple of packages), ...] per part]. -> (ordered packages, info)."""
    choices = [range(-1, len(alts)) for alts in parts_pk]
    best, seen, n_eval = None, set(), 0
    for combo in itertools.product(*choices):
        order = []
        for alts, c in zip(parts_pk, combo):
            if c >= 0:
                for p in alts[c]:
                    if p["chunk"] not in {q["chunk"] for q in order}:
                        order.append(p)
        key0 = tuple(p["chunk"] for p in order)
        if key0 in seen:
            continue
        seen.add(key0)
        shown = canonical(order)
        c_ = cost(shown)
        n_eval += 1
        if c_ > budget:
            continue
        cov, rk = _covered(parts_pk, shown)
        key = (-cov, rk, c_, tuple(sorted(rrf_rank.get(p["chunk"], 10 ** 6) for p in shown)),
               tuple(sorted(p["chunk"] for p in shown)))
        if best is None or key < best[0]:
            best = (key, shown, combo)
    if best is None:                               # only the empty union can be infeasible-free; nothing fits
        return [], {"covered": 0, "choice": None, "evaluated": n_eval}
    return best[1], {"covered": -best[0][0], "alt_rank_sum": best[0][1], "chars": best[0][2],
                     "choice": list(best[2]), "evaluated": n_eval}


def _try_add(shown, pkgs, cost, budget):
    """shown + the not-yet-represented packages of pkgs (all or nothing) -> new list or None when it does not fit."""
    new = [p for p in pkgs if not represented(p, shown)]
    if not new:
        return shown
    cand = canonical(shown + new)
    return cand if cost(cand) <= budget else None


def pack_greedy(parts_pk, cost, budget):
    shown, choice = [], []
    for alts in parts_pk:
        took = -1
        for ai, b in enumerate(alts):
            nxt = _try_add(shown, list(b), cost, budget)
            if nxt is not None:
                shown, took = nxt, ai
                break
        choice.append(took)
    cov, rk = _covered(parts_pk, shown)
    return shown, {"covered": cov, "alt_rank_sum": rk, "chars": cost(shown), "choice": choice}


def pack_flat(parts_pk, cost, budget):
    order = [alts[0][0] for alts in parts_pk if alts]
    order += [p for alts in parts_pk for b in alts for p in b]
    shown = []
    for p in order:
        nxt = _try_add(shown, [p], cost, budget)
        if nxt is not None:
            shown = nxt
    cov, rk = _covered(parts_pk, shown)
    return shown, {"covered": cov, "alt_rank_sum": rk, "chars": cost(shown)}


def fill(shown, parts_pk, rrf_order, cost, budget, bundles=True, rrf=True):
    """Remaining selector bundles (complete only) in part / alternative order, then RRF packages. -> (list, info)."""
    n_b, n_r = 0, 0
    if bundles:
        for alts in parts_pk:
            for b in alts:
                if all(represented(p, shown) for p in b):
                    continue
                nxt = _try_add(shown, list(b), cost, budget)
                if nxt is not None:
                    shown = nxt
                    n_b += 1
    if rrf:
        for p in rrf_order:
            if represented(p, shown):
                continue
            nxt = _try_add(shown, [p], cost, budget)
            if nxt is not None:
                shown = nxt
                n_r += 1
    return shown, {"fill_bundles": n_b, "fill_rrf": n_r}


def hybrid_intact(rrf_order, cost, budget):
    """The fallback (and the HYBRID-intact arm): RRF packages in order, whole, exact/contained dedupe."""
    return fill([], [], rrf_order, cost, budget, bundles=False, rrf=True)[0]
