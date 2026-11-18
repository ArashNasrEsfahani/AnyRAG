"""Stage 3 of AGENTIC5: candidate allocation across REQUIREMENTS first, then channels.

AGENTIC3's `retrieve.merge` reserves facet slots and vector slots BEFORE the round robin over documents, so on a
multi-part question those reservations could take the whole first verify batch (reviewer: "Current facet/vector
reservations occur before document round-robin and can monopolize the early verification batch"). Here the order
of precedence is:

  1. requirements   round robin over the requirements, in matrix order: every requirement that has a candidate gets
                    one before any requirement gets a second;
  2. channels       inside one requirement, its candidates are interleaved by channel (lexical facts/units, dense
                    or hybrid, gap), best first within each, so no single channel fills a requirement's share;
  3. escape         one slot, after the first round, for the pooled candidate from outside the shortlist (the
                    recovery channel), when there is one;
  4. dedupe         a window that serves two requirements is one candidate carrying both ids, counted for both.

The first batch holds at most `batch1_max` (12) candidates; the rest stays ordered the same way for batch 2.
"""
from ..agentic import search as S  # noqa: F401  (Candidate type, for readers)
from .channels import REQ_KEY


def channel_of(c):
    src = set((c.features or {}).get("src") or [])
    if "gap" in src:
        return "gap"
    if "hybrid" in src:
        return "hybrid"
    if "vector" in src:
        return "dense"
    if "facts" in src:
        return "facts"
    if "units" in src:
        return "units"
    return "other"


def _key(c):
    return (c.doc_id, c.section_id, tuple(sorted(c.unit_ids)), c.text[:60])


def dedupe(cands):
    """One object per window; requirement ids and sources are unioned. A candidate whose units lie inside another of
    the same section is absorbed into it (the wider window keeps both tags)."""
    kept = []
    for c in sorted(cands, key=lambda c: (-float(c.score or 0.0), -len(c.unit_ids))):
        dup = None
        for k in kept:
            if k.doc_id == c.doc_id and k.section_id == c.section_id and set(c.unit_ids) <= set(k.unit_ids) \
                    and (c.text in k.text or _key(k) == _key(c) or k.kind != "clip"):
                dup = k
                break
        if dup is not None:
            dup.features[REQ_KEY] = sorted(set(dup.features.get(REQ_KEY) or []) | set(c.features.get(REQ_KEY) or []))
            dup.features["src"] = sorted(set(dup.features.get("src") or []) | set(c.features.get("src") or []))
            continue
        kept.append(c)
    return kept


def per_requirement(cands, reqs):
    """rid -> that requirement's candidates, interleaved by channel (best first within a channel)."""
    out = {}
    for r in reqs:
        mine = [c for c in cands if r["id"] in (c.features.get(REQ_KEY) or [])]
        queues = {}
        for c in sorted(mine, key=lambda c: -float(c.score or 0.0)):
            queues.setdefault(channel_of(c), []).append(c)
        order = [ch for ch in ("facts", "hybrid", "dense", "units", "gap", "other") if ch in queues]
        inter = []
        while any(queues[ch] for ch in order):
            for ch in order:
                if queues[ch]:
                    inter.append(queues[ch].pop(0))
        out[r["id"]] = inter
    return out


def allocate(cands, reqs, cap, *, escape=(), exclude=(), prefix="c", start=1):
    """-> (batch, rest, info). `exclude`: candidates already verified (by identity). cids are assigned to the
    batch and then the rest, in order."""
    cands = dedupe([c for c in cands if id(c) not in {id(x) for x in exclude}])
    queues = per_requirement(cands, reqs)
    taken, batch = set(), []
    share = {r["id"]: 0 for r in reqs}
    escape = [c for c in escape if id(c) not in {id(x) for x in exclude}]
    esc_used = False
    rnd = 0
    while len(batch) < cap:
        progressed = False
        for r in reqs:
            if len(batch) >= cap:
                break
            q = queues.get(r["id"]) or []
            while q and id(q[0]) in taken:
                q.pop(0)
            if not q:
                continue
            c = q.pop(0)
            taken.add(id(c))
            batch.append(c)
            for rid in c.features.get(REQ_KEY) or []:
                if rid in share:
                    share[rid] += 1
            progressed = True
        rnd += 1
        if rnd == 1 and not esc_used and escape and len(batch) < cap:
            e = next((x for x in escape if id(x) not in taken), None)
            if e is not None:
                taken.add(id(e))
                batch.append(e)
                esc_used = True
                progressed = True
        if not progressed:
            break
    rest = []
    for r in reqs:
        for c in queues.get(r["id"]) or []:
            if id(c) not in taken:
                taken.add(id(c))
                rest.append(c)
    rest += [c for c in cands if id(c) not in taken]
    rest += [c for c in escape if id(c) not in taken]
    n = start
    for c in batch + rest:
        c.cid = "%s%d" % (prefix, n)
        c.verdict = None
        n += 1
    starved = [rid for rid, k in share.items() if k == 0]
    info = {"batch": len(batch), "rest": len(rest), "cap": cap, "share": share, "starved": starved,
            "escape_in_batch": esc_used,
            "channels_in_batch": {ch: sum(1 for c in batch if channel_of(c) == ch)
                                  for ch in sorted({channel_of(c) for c in batch})}}
    return batch, rest, info
