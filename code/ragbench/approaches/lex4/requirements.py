"""Domain-neutral requirements for LEX4: {id, subject, aspect, constraints, source_ref, source_doc, text, origin}.

    subject      [alternatives]            who/what the fact is about (may be empty)
    aspect       [alternatives]            what is asked about it (facet name + its search terms)
    constraints  [[alternatives], ...]     each an explicit condition (time, version, place ...); one group each
    source_ref   str | None                the plan's reference to a source: a single-entity subject; source_doc =
                                           the one pool document it resolves to (unique title match) or None
    text         the requirement's retrieval text (BM25 + dense)

Two producers, both deterministic (no model call here):

from_lex3_plan(interp, titles): LEX3's saved call-1 plan (entities, periods, operation, facets).
    - a year that belongs to a name stays in the name: a period whose digits occur in an entity or facet name/term is
      NOT a constraint;
    - one requirement per facet (aspect = facet name + terms; subject = the entities as alternatives; the remaining
      periods = one constraint group);
    - no entity x period cross product. Only for operation "compare": when >= 2 free periods (else >= 2 entities) are
      the compared members and facets x members <= MAX_REQS, each facet gets one requirement per member along that
      ONE axis (period axis: the member is the constraint; entity axis: the member is the subject);
    - at most MAX_REQS (in order); an empty plan -> None (the caller falls back to question clauses).

from_question(question): no planner. The question split into clauses at ? ؟ , ، ; ؛ : and at coordinating
    conjunctions (and / or / but / whereas / while / compared with; و / یا / اما / ولی / در حالی که / در مقایسه با);
    fragments with < 2 content words join the previous clause; <= MAX_REQS clauses. Concept groups of a clause:
    every content word is its own group (aspect weight), every number a constraint group.
"""
import re

from ...kb import labels as L
from ...normalize import fold, word_tokens

MAX_REQS = 8
_QWORDS = {fold(w) for w in """how what which who whom whose when where why whether does did do is are was were
can could would should many much roughly approximately about according report reports its their per
چه چند چقدر کدام کدامین چرا چگونه آیا کی کجا چطور طبق به نوشته گزارش گزارش‌های نوشتهٔ""".split()}
_YEAR = re.compile(r"(?<!\d)(\d{4})(?!\d)")


def content_words(text):
    out = []
    for t in word_tokens(text):
        if t in L._STOP or t in _QWORDS:
            continue
        if not any(ch.isdigit() for ch in t) and len(t) < 2:
            continue
        out.append(t)
    return out


def _digits(s):
    return set(_YEAR.findall(fold(s)))


def resolve_source(refs, titles):
    """refs: [str]; titles: {doc_id: title}. -> (doc_id | None, matches)."""
    hits = set()
    for r in refs or []:
        fr = fold(r)
        if len(fr) < 4:
            continue
        for d, t in titles.items():
            ft = fold(t)
            if ft and (fr in ft or ft in fr):
                hits.add(d)
    return (next(iter(hits)) if len(hits) == 1 else None), sorted(hits)


def _req(i, subject, aspect, constraints, titles, origin):
    subject = [s for s in dict.fromkeys(subject) if s and s.strip()]
    aspect = [a for a in dict.fromkeys(aspect) if a and a.strip()]
    constraints = [[c for c in dict.fromkeys(g) if c and c.strip()] for g in constraints]
    constraints = [g for g in constraints if g]
    # the source reference is the subject when it names ONE thing; a multi-entity subject is not a source reference
    # (one of several entities matching a title would wrongly confine the others)
    src, matches = resolve_source(subject, titles) if titles and len(subject) == 1 else (None, [])
    words = subject + aspect + [c for g in constraints for c in g]
    return {"id": "R%d" % i, "subject": subject, "aspect": aspect, "constraints": constraints,
            "source_ref": subject[0] if src else None,
            "source_doc": src, "source_matches": matches, "text": " ".join(words), "origin": origin}


def from_lex3_plan(interp, titles=None):
    if not interp or interp.get("empty"):
        return None, {"why": "empty plan"}
    facets = [f for f in interp.get("facets") or [] if f.get("name") or f.get("terms")]
    ents = [e for e in interp.get("entities") or [] if e and e.strip()]
    periods = [p for p in (interp.get("periods_raw") or interp.get("periods") or []) if p and str(p).strip()]
    if not facets:
        return None, {"why": "no facet"}
    name_years = set()
    for s in ents + [f.get("name") or "" for f in facets] + [t for f in facets for t in f.get("terms") or []]:
        name_years |= _digits(s)
    free = [p for p in periods if not (_digits(p) and _digits(p) <= name_years)]
    info = {"entities": ents, "periods": periods, "free_periods": free, "name_years": sorted(name_years),
            "operation": interp.get("operation"), "split": None}
    members, axis = None, None
    if interp.get("operation") == "compare":
        if len(free) >= 2:
            members, axis = free, "period"
        elif len(ents) >= 2:
            members, axis = ents, "entity"
        if members and len(facets) * len(members) > MAX_REQS:
            info["split_refused"] = "%d facets x %d members > %d" % (len(facets), len(members), MAX_REQS)
            members, axis = None, None
    reqs = []
    for f in facets:
        aspect = [f.get("name") or ""] + list(f.get("terms") or [])
        if members is None:
            reqs.append((ents, aspect, [free] if free else []))
        elif axis == "period":
            for m in members:
                reqs.append((ents, aspect, [[m]]))
        else:
            for m in members:
                reqs.append(([m], aspect, [free] if free else []))
    info["split"] = axis
    info["truncated"] = max(0, len(reqs) - MAX_REQS)
    return [_req(i + 1, s, a, c, titles, "lex3_plan") for i, (s, a, c) in enumerate(reqs[:MAX_REQS])], info


_SPLIT = re.compile(r"[?؟,،;؛:]|\s(?:and|or|but|whereas|while|compared with|compared to|versus|vs\.?)\s|"
                    r"\s(?:و|یا|اما|ولی|در حالی که|درحالی‌که|در مقایسه با|نسبت به)\s", re.IGNORECASE)


def from_question(question, titles=None):
    parts = [p.strip() for p in _SPLIT.split(question or "") if p and p.strip()]
    clauses = []
    for p in parts:
        if clauses and len(content_words(p)) < 2:
            clauses[-1] = clauses[-1] + " " + p
        else:
            clauses.append(p)
    clauses = [c for c in clauses if content_words(c)] or [question]
    info = {"clauses": len(clauses), "truncated": max(0, len(clauses) - MAX_REQS)}
    reqs = []
    for i, c in enumerate(clauses[:MAX_REQS]):
        words = content_words(c)
        nums = [w for w in words if any(ch.isdigit() for ch in w)]
        plain = [w for w in words if w not in nums]
        src, matches = resolve_source([c], titles) if titles else (None, [])
        # titles contained in the clause: resolve_source checks both directions; only the title-in-clause direction
        # is meaningful for a clause
        if titles:
            fc = fold(c)
            hits = sorted(d for d, t in titles.items() if fold(t) and len(fold(t)) >= 4 and fold(t) in fc)
            src = hits[0] if len(hits) == 1 else None
            matches = hits
        reqs.append({"id": "R%d" % (i + 1), "subject": [], "aspect": [], "clause_words": plain,
                     "constraints": [[n] for n in nums], "source_ref": c if src else None, "source_doc": src,
                     "source_matches": matches, "text": c, "origin": "question_clause"})
    return reqs, info
