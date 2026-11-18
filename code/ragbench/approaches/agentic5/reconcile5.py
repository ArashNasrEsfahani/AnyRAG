"""Stage 5 of AGENTIC5: entity- and metric-aware reconciliation of typed claims (no model, no SQL needed).

AGENTIC3's reconciler (`agentic3/reconcile_sql.py`) keeps, dedupes, flags conflicts and computes coverage by YEAR:
two companies that each report 100 in 2024 became one row. Here the unit of reconciliation is the IDENTITY

    (entity, metric, period key incl. its type, definition class, unit family)

and every rule runs inside one identity only:

  bind        a claim belongs to a requirement (`claims.bind`), which fixes entity, metric and period; a claim bound
              to none is kept in `unbound` (shown in the trace, never merged into a row).
  role        v1's rule per claim: authoritative (the report of that period), restatement (a later report),
              projection (an earlier report; never a row).
  keep        per identity: the authoritative claims, else those of the LATEST restating report.
  dedupe      equal values (`same_number` on the base-unit value) of ONE identity are one row; the same value in two
              identities is two rows. A shared source unit is not a duplicate: two values of one table row are two
              claims.
  lineage     a restatement is compared only with the kept rows of its own identity: a different value is a
              `restated` conflict note; the kept value stays.
  disagree    two different authoritative values in one identity are an UNRESOLVED disagreement: both stay, the
              requirement's status is `conflict`, arithmetic will not pick one.
  definitions several definition classes for one requirement are kept apart; the plainest (fewest unexplained
              qualifiers, not a subgroup) is the primary, the others are listed as alternatives.
  coverage    per REQUIREMENT, never per year: verified | conflict | missing, and `incompatible` claims (a quarter
              for a year, an unverified unit) recorded next to a missing requirement.
"""
from ..agentic import reconcile as R
from ..agentic.numbers import same_number
from .defaults import RECONCILE_VERSION

STATUS_ORDER = ("verified", "conflict", "missing")


def role_of(claim):
    years = claim.get("doc_years")
    if claim.get("period_type") not in (None, "annual") and years is None:
        return "authoritative"
    y = claim.get("period_year")
    if years is None:
        return "authoritative"
    if y is None:
        return "unknown_period"
    if years[0] <= y <= years[1]:
        return "authoritative"
    return "restatement" if y < years[0] else "projection"


def identity(claim):
    return (claim.get("entity_key") or "", claim.get("metric_id") or "", claim.get("period_key") or "",
            claim.get("def_class") or "unspecified", claim.get("unit_family") or "")


def _num(c):
    v = c.get("value_norm")
    return v if v is not None else c.get("value_num")


def _differs(a, b, tol):
    if a is None or b is None:
        return False
    return not same_number(a, b) and abs(a - b) > tol * max(abs(a), abs(b))


def reconcile_group(claims, tol):
    """One identity -> {"kept": [...], "restatements": [...], "conflicts": [...], "disagreement": bool}."""
    auth = [c for c in claims if c["role"] == "authoritative"]
    rest = [c for c in claims if c["role"] == "restatement"]
    kept_role = "authoritative"
    if auth:
        kept = auth
    elif rest:
        latest = max((c.get("doc_years") or [0, 0])[1] for c in rest)
        kept = [c for c in rest if (c.get("doc_years") or [0, 0])[1] == latest]
        rest = [c for c in rest if (c.get("doc_years") or [0, 0])[1] != latest]
        kept_role = "latest_restatement"
    else:
        kept = []
    uniq = []
    for c in sorted(kept, key=lambda c: -(c.get("score") or 0)):
        v = _num(c)
        if v is not None and any(_num(u) is not None and same_number(_num(u), v) for u in uniq):
            continue
        if v is None and any(u.get("value") == c.get("value") for u in uniq):
            continue
        uniq.append(dict(c, kept_role=kept_role))
    conflicts = []
    for r in rest:
        if uniq and all(_differs(_num(u), _num(r), tol) for u in uniq):
            conflicts.append({"kind": "restated", "period": r.get("period_key"), "kept": [u["value"] for u in uniq],
                              "kept_doc": uniq[0]["doc_id"], "restated": r["value"], "restated_doc": r["doc_id"]})
    values = [_num(u) for u in uniq if _num(u) is not None]
    disagreement = len(uniq) > 1 and any(_differs(a, b, tol) for i, a in enumerate(values) for b in values[i + 1:])
    if len(uniq) > 1 and not values:
        disagreement = True
    return {"kept": uniq, "restatements": rest, "conflicts": conflicts, "disagreement": disagreement}


def reconcile(claims, matrix, *, known_stems=None, opts=None):
    """claims (typed rows) + the requirement matrix -> the reconciliation.

    -> {"requirements": {rid: {...}}, "order": [rid...], "rows": [table rows], "missing": [...], "conflicting":
        [...], "unbound": [...], "projections": [...], "series_open": {...}, "version"}
    """
    opts = opts or {}
    tol = float(opts.get("restatement_rel_tol", 0.005))
    reqs = matrix["requirements"]
    for c in claims:
        c["role"] = role_of(c)
        if known_stems is not None:
            c["qualifiers"] = R.qualifiers(c.get("definition"), known_stems)
    out = {"version": RECONCILE_VERSION, "requirements": {}, "order": [r["id"] for r in reqs], "rows": [],
           "missing": [], "conflicting": [], "unbound": [], "projections": [], "incompatible": []}
    bound = {}
    for c in claims:
        if c.get("role") == "projection":
            out["projections"].append(c)
            continue
        if not c.get("req_id"):
            out["unbound"].append(c)
            continue
        bound.setdefault(c["req_id"], []).append(c)
    for r in reqs:
        rc = bound.get(r["id"], [])
        groups = {}
        for c in rc:
            if c["role"] == "unknown_period" and r.get("period"):
                continue
            key = identity(c) if r.get("period") else identity(c)
            groups.setdefault(key, []).append(c)
        recs = {k: reconcile_group(v, tol) for k, v in groups.items()}
        # the primary identity: not a subgroup, fewest unexplained qualifiers, most support, first seen
        def prim_key(k):
            g = recs[k]
            kept = g["kept"]
            quals = min((len(c.get("qualifiers") or []) for c in kept), default=9)
            return (k[3] == "subgroup", quals, -len(groups[k]), list(groups).index(k))
        live = [k for k in recs if recs[k]["kept"]]
        entry = {"id": r["id"], "requirement": r, "groups": [], "status": "missing", "primary": None,
                 "alternatives": [], "conflicts": [], "restatements": []}
        if r["operation"] == "series" and not r.get("period"):
            # an OPEN series: one identity per period key; covered when two or more periods have a value
            by_period = {}
            for k in live:
                by_period.setdefault(k[2], []).append(k)
            rows = []
            for pk in sorted(by_period, key=lambda p: (p == "", p)):
                ks = sorted(by_period[pk], key=prim_key)
                g = recs[ks[0]]
                rows += [dict(c, req_status="conflict" if g["disagreement"] else "verified") for c in g["kept"]]
                entry["conflicts"] += g["conflicts"]
                entry["alternatives"] += [c for k in ks[1:] for c in recs[k]["kept"]]
            entry["rows"] = rows
            entry["periods_found"] = sorted({c.get("period_key") for c in rows if c.get("period_key")})
            dis = any(c["req_status"] == "conflict" for c in rows)
            entry["status"] = "missing" if not rows else ("conflict" if dis else "verified")
            entry["open_series_covered"] = len(entry["periods_found"]) >= 2
        elif live:
            ks = sorted(live, key=prim_key)
            g = recs[ks[0]]
            entry["primary"] = ks[0]
            entry["rows"] = [dict(c, req_status="conflict" if g["disagreement"] else "verified") for c in g["kept"]]
            entry["status"] = "conflict" if g["disagreement"] else "verified"
            entry["conflicts"] = [c for k in ks for c in recs[k]["conflicts"]]
            entry["restatements"] = [c for k in ks for c in recs[k]["restatements"]]
            entry["alternatives"] = [dict(c, req_status="alternative") for k in ks[1:] for c in recs[k]["kept"]]
        else:
            entry["rows"] = []
        entry["groups"] = [{"identity": list(k), "claims": len(groups[k]), "kept": len(recs[k]["kept"]),
                            "disagreement": recs[k]["disagreement"]} for k in groups]
        out["requirements"][r["id"]] = entry
        if entry["status"] == "missing":
            out["missing"].append(r["id"])
        elif entry["status"] == "conflict":
            out["conflicting"].append(r["id"])
        out["rows"] += entry["rows"] + entry["alternatives"]
    return out


def near_misses(verify_logs, matrix):
    """Values the verifier tied to a requirement that code refused on type (a quarter for a year, another company):
    per requirement id, the reasons - for the table's `missing` notes and the trace."""
    out = {}
    for lg in verify_logs:
        for it in lg.get("items") or []:
            want = str(it.get("req") or "").upper()
            if want and it.get("verdict", "").startswith(("unbound", "guard")):
                out.setdefault(want, []).append(it.get("verdict"))
    return out


def summary(recon):
    return {rid: {"status": e["status"], "rows": len(e.get("rows") or []), "alternatives": len(e["alternatives"]),
                  "conflicts": len(e["conflicts"]), "groups": e["groups"]}
            for rid, e in recon["requirements"].items()}
