"""Stage 6 of AGENTIC5: deterministic arithmetic over VERIFIED, COMPATIBLE operands.

The requested operation of the requirement matrix decides what is computed:

  difference   two operands: the later period minus the earlier (one entity), else R1 minus R2 in question order
  ratio        the same two operands, first / second
  sum          every requirement's value
  comparison   ordering (largest first) of the requirements' values, and the pairwise difference when there are two
  series       the change from the first to the last period with a value (absolute and relative)

An operand is usable only when its requirement is `verified` with ONE kept value that has a base-unit number and a
verified (or empty) unit, and all operands share one unit family (kind and currency). Anything else leaves the
result `unknown` with the reason: a missing requirement never becomes zero, and absence of evidence is never
evidence of absence. Each result carries the claim ids of its operands; the evidence table cites their sources.
"""
from ..agentic.numbers import fmt_number


def _operand(entry):
    """-> (claim, None) or (None, reason)."""
    if entry is None:
        return None, "no_requirement"
    if entry["status"] == "missing":
        return None, "missing"
    if entry["status"] == "conflict":
        return None, "conflict"
    rows = [c for c in entry.get("rows") or [] if c.get("req_status") == "verified"]
    if len(rows) != 1:
        return None, "several_values" if rows else "no_value"
    c = rows[0]
    if c.get("value_norm") is None:
        return None, "not_a_number"
    if c.get("unit_status") == "unverified":
        return None, "unit_unverified"
    return c, None


def _unit_label(c):
    return c.get("unit") or c.get("unit_norm") or ""


def _fmt(x):
    if x is None:
        return "?"
    if abs(x) >= 1000 or float(x).is_integer():
        return fmt_number(round(x, 2) if not float(x).is_integer() else x)
    return ("%.4g" % x)


def _scale_back(c, v):
    """A base-unit number back into the operand's written scale ("million USD"), so the reader sees 12.5 million
    USD and not 12500000 when both operands were written in millions."""
    sc = float(c.get("scale") or 1.0) * (10.0 if c.get("currency") == "toman" else 1.0)
    return v / sc if sc else v


def _same_family(cs):
    fams = {c.get("unit_family") for c in cs}
    return len(fams) == 1


def _same_scale(cs):
    return len({(float(c.get("scale") or 1.0), c.get("currency")) for c in cs}) == 1


def _pair(matrix, recon):
    reqs = matrix["requirements"]
    if len(reqs) < 2:
        return None
    a, b = reqs[0], reqs[1]
    same_entity = a["entity_id"] == b["entity_id"] and a["metric_id"] == b["metric_id"]
    if same_entity and a.get("period") and b.get("period") and a["period"]["key"] > b["period"]["key"]:
        a, b = b, a
    # one entity over two periods: later minus earlier; otherwise the question's order
    return (b, a) if same_entity else (a, b)


def compute(matrix, recon, opts=None):
    """-> list of result dicts {op, status: ok|unknown, value, text, unit, operands: [claim cid/value ids],
    operand_reqs, reason}."""
    op = matrix["operation"]
    ents = recon["requirements"]
    out = []

    def result(kind, operands, reqs, value=None, text=None, unit="", reason=None):
        return {"op": kind, "status": "ok" if reason is None else "unknown", "value": value,
                "text": text if reason is None else "unknown (%s)" % reason, "unit": unit,
                "operands": [c.get("claim_id") for c in operands], "operand_reqs": [r["id"] for r in reqs],
                "reason": reason}

    def operands(reqs):
        cs, why = [], None
        for r in reqs:
            c, reason = _operand(ents.get(r["id"]))
            if c is None:
                why = why or "%s %s" % (r["id"], reason)
            else:
                cs.append(c)
        if why is None and not _same_family(cs):
            why = "incompatible units"
        return cs, why

    if op in ("difference", "ratio") or (op == "comparison" and len(matrix["requirements"]) == 2):
        pair = _pair(matrix, recon)
        if pair:
            first, second = pair
            cs, why = operands([first, second])
            kind = "ratio" if op == "ratio" else "difference"
            if why:
                out.append(result(kind, cs, [first, second], reason=why))
            else:
                a, b = cs
                if kind == "ratio":
                    if not b["value_norm"]:
                        out.append(result(kind, cs, [first, second], reason="division by zero"))
                    else:
                        v = a["value_norm"] / b["value_norm"]
                        out.append(result(kind, cs, [first, second], value=v, text="%s / %s = %s" % (
                            first["id"], second["id"], _fmt(v)), unit="x"))
                else:
                    v = a["value_norm"] - b["value_norm"]
                    shown = _scale_back(a, v) if _same_scale(cs) else v
                    unit = _unit_label(a) if _same_scale(cs) else (a.get("unit_norm") or "")
                    rel = (v / b["value_norm"] * 100.0) if b["value_norm"] else None
                    txt = "%s - %s = %s %s" % (first["id"], second["id"], _fmt(shown), unit)
                    if rel is not None:
                        txt += " (%s%%)" % _fmt(rel)
                    out.append(result(kind, cs, [first, second], value=v, text=txt.strip(), unit=unit))
    if op == "sum":
        reqs = matrix["requirements"]
        cs, why = operands(reqs)
        if why:
            out.append(result("sum", cs, reqs, reason=why))
        else:
            v = sum(c["value_norm"] for c in cs)
            shown = _scale_back(cs[0], v) if _same_scale(cs) else v
            unit = _unit_label(cs[0]) if _same_scale(cs) else (cs[0].get("unit_norm") or "")
            out.append(result("sum", cs, reqs, value=v, text="%s = %s %s" % (
                " + ".join(r["id"] for r in reqs), _fmt(shown), unit), unit=unit))
    if op == "comparison" and len(matrix["requirements"]) >= 2:
        reqs = matrix["requirements"]
        cs, why = operands(reqs)
        if why:
            out.append(result("order", cs, reqs, reason=why))
        else:
            ranked = sorted(zip(reqs, cs), key=lambda p: -p[1]["value_norm"])
            out.append(result("order", [c for _r, c in ranked], [r for r, _c in ranked],
                              text=" > ".join(r["id"] for r, _c in ranked)))
    if op == "series":
        rows = []
        for r in matrix["requirements"]:
            e = ents.get(r["id"])
            if e is None:
                continue
            if r.get("period"):
                c, _why = _operand(e)
                if c is not None:
                    rows.append((r["period"]["key"], r, c))
            else:
                for c in e.get("rows") or []:
                    if c.get("req_status") == "verified" and c.get("value_norm") is not None:
                        rows.append((c.get("period_key") or "", r, c))
        by_ent = {}
        for pk, r, c in rows:
            by_ent.setdefault((r["entity_id"], r["metric_id"]), []).append((pk, r, c))
        for key, pts in by_ent.items():
            pts = sorted([p for p in pts if p[0]], key=lambda p: p[0])
            reqs = [p[1] for p in pts]
            if len(pts) < 2:
                continue
            first, last = pts[0], pts[-1]
            cs = [first[2], last[2]]
            if not _same_family(cs):
                out.append(result("change", cs, [first[1], last[1]], reason="incompatible units"))
                continue
            v = last[2]["value_norm"] - first[2]["value_norm"]
            rel = (v / first[2]["value_norm"] * 100.0) if first[2]["value_norm"] else None
            shown = _scale_back(first[2], v) if _same_scale(cs) else v
            unit = _unit_label(first[2]) if _same_scale(cs) else (first[2].get("unit_norm") or "")
            txt = "%s -> %s: %s %s" % (first[0], last[0], _fmt(shown), unit)
            if rel is not None:
                txt += " (%s%%)" % _fmt(rel)
            out.append(result("change", cs, reqs[:1] + reqs[-1:], value=v, text=txt.strip(), unit=unit))
    return out
