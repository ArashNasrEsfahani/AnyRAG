"""Corpus A adapter: KareOne bilingual reports (chapters/*.json with {"fa","en"} leaves).

One pass over the JSON builds the fa and en Documents together, so both get exactly
the same sections and unit local ids (unit-aligned: swap_lang(unit_id) is the same
passage in the other language). A unit is kept when either language has text.

JSON shape (verified on all 10 reports, 124 chapter files):
  report.meta.json          {title{fa,en}, lede{fa,en}, meta{eyebrow, facts[{label,value}]}, provenance}
  chapters/<stem>.json      {sections[], figures{id: {label{fa,en}, value, unit, period?, src?, conf?}}, sources?}
  sections[]                {id, number, headline{fa,en}, lede?, summary?, keyInsight?, caveat?,
                             takeaways[{text{fa,en}}]?, part?: true (sub-section of the previous
                             non-part section), charts[]}
  charts[]                  {kind, id, title{fa,en}, subtitle?, note?, source?, data[], ...kind-specific}
  prose embeds {{figureId}} placeholders, resolved here against the report's merged
  figures with the report renderer's unit rules (shell/charts.js fmtVal), plus
  **bold**, *italic* and [label](url) markup, which is stripped.
Chapter order follows ORDER in build/99_assemble.py (then remaining files sorted).

Tree: "_report" (title, lede, facts) | chapter section (non-part) -> part sections ->
chart sections (only charts with a title; untitled charts' units stay in the parent).
Units: lede/summary/keyInsight/caveat/takeaways -> paragraph; KPI tiles -> fact;
text/insight items -> paragraph; chart data (ranked/explorer/bar/line/...) -> table
units of at most corpora.table_rows_per_unit rows with the header repeated;
subtitle/source -> caption; note -> paragraph; calculator/assessment/journey
definitions -> list.
"""
import ast
import hashlib
import json
import os
import re
from pathlib import Path

from .model import DocBuilder, make_doc_id

LANGS = ("fa", "en")
ADAPTER = "kareone-1"

# --------------------------------------------------------------------------
# figure formatting (port of the report shell's fmtVal; digits stay ASCII)
# --------------------------------------------------------------------------


def _grp(v):
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, int):
        return "{:,}".format(v)
    if isinstance(v, float):
        if v != v or v in (float("inf"), float("-inf")):
            return "—"
        if v.is_integer():
            return "{:,}".format(int(v))
        s = repr(v)
        if "e" in s or "E" in s:
            return s
        ip, fp = s.split(".")
        neg = ip.startswith("-")
        ip = ip.lstrip("-")
        return ("-" if neg else "") + "{:,}".format(int(ip)) + "." + fp
    return str(v)


_UNIT_WORDS = {
    # unit: (english template, persian template); {n} is the grouped number
    "$B": ("${n}B", "${n}B"), "$M": ("${n}M", "${n}M"), "$": ("${n}", "${n}"), "%": ("{n}%", "{n}%"),
    "x": ("{n}×", "{n}×"), "×": ("{n}×", "{n}×"),
    "h": ("{n}h", "{n} ساعت"), "t": ("{n} t", "{n} تن"),
    "trn_rial": ("{n} trn rials", "{n} هزار میلیارد ریال"), "bn_rial": ("{n} bn rials", "{n} میلیارد ریال"),
    "mrial": ("{n} mn rials", "{n} میلیون ریال"), "mrial_t": ("{n} mn rials/t", "{n} میلیون ریال بر تن"),
    "Mt": ("{n} Mt", "{n} میلیون تن"), "kt": ("{n} kt", "{n} هزار تن"),
    "$/t": ("${n}/t", "${n}"), "$/kg": ("${n}/kg", "{n} دلار بر کیلوگرم"), "NOK/kg": ("{n} NOK/kg", "{n} کرون بر کیلوگرم"),
    "µg/kg": ("{n} µg/kg", "{n} میکروگرم بر کیلوگرم"), "kg/h": ("{n} kg/h", "{n} کیلوگرم در ساعت"),
    "t/h": ("{n} t/h", "{n} تن در ساعت"), "kGy": ("{n} kGy", "{n} کیلوگری"), "MeV": ("{n} MeV", "{n} مگاالکترون‌ولت"),
    "m3/yr": ("{n} m³/yr", "{n} متر مکعب در سال"), "€": ("€{n}", "{n} یورو"), "€/t": ("€{n}/t", "{n} یورو بر تن"),
    "km": ("{n} km", "{n} کیلومتر"), "mn": ("{n} million", "{n} میلیون"), "bn": ("{n} billion", "{n} میلیارد"),
    "truck": ("{n} trucks", "{n} کامیون"), "/100k": ("{n} per 100,000", "{n} در هر صد هزار نفر"),
    "DALY/100k": ("{n} DALYs per 100,000", "{n} DALY در هر صد هزار نفر"),
    "notif": ("{n} notifications", "{n} اعلان"), "notif/kt": ("{n} per thousand tonnes", "{n} اعلان در هر هزار تن"),
    "notif/100kt": ("{n} per hundred thousand tonnes", "{n} اعلان در هر صد هزار تن"),
    "notif/€100M": ("{n} per hundred million euros imported", "{n} اعلان در هر صد میلیون یورو واردات"),
    "cert": ("{n} certificates", "{n} گواهی"), "consign": ("{n} consignments", "{n} محموله"),
    "yrs": ("{n} years", "{n} سال"), "€M": ("€{n}M", "{n} میلیون یورو"),
    "ng/kg": ("{n} ng per kg", "{n} نانوگرم بر کیلوگرم"), "mg/kg": ("{n} mg per kg", "{n} میلی‌گرم بر کیلوگرم"),
}
_PASS_UNITS = {"GJ/t", "MWh/t", "kWh/t", "tCO2/t", "kgCO2/t", "t/emp", "m3/t", "mcm/d"}


def fmt_value(v, unit, lang):
    """A figure value as the report prints it (unknown units print the bare number)."""
    if v is None or v == "":
        return "—"
    if isinstance(v, str):
        try:
            float(v.strip())
            num = v.strip()
        except ValueError:
            return v            # names and pre-formatted bands print verbatim
        v = float(num) if "." in num else int(num)
    unit = (unit or "").strip()
    if unit == "yr":
        return str(int(round(float(v))))
    if unit == "d":
        n = _grp(v)
        return ("%s روز" % n) if lang == "fa" else ("%s day" % n if float(v) == 1 else "%s days" % n)
    if unit == "pp":
        s = ("+" if float(v) > 0 else "") + _grp(v)
        return ("%s واحد درصد" % s) if lang == "fa" else ("%s%% pts" % s)
    if unit in _UNIT_WORDS:
        en, fa = _UNIT_WORDS[unit]
        return (fa if lang == "fa" else en).format(n=_grp(v))
    if unit in _PASS_UNITS:
        return "%s %s" % (_grp(v), unit)
    return _grp(v)


_RE_FIG = re.compile(r"\{\{(\w[\w.\-]*)\}\}")
_RE_BOLD = re.compile(r"\*\*(.+?)\*\*", re.S)
_RE_ITAL = re.compile(r"(?<![\w*])\*(?![\s*])(.+?)(?<![\s*])\*(?![\w*])")
_RE_LINK = re.compile(r"\[([^\[\]\n]*)\]\((?:https?://|mailto:|tel:|#)[^\s()<>]*\)")


class Resolver:
    def __init__(self, figures):
        self.figures = figures
        self.unresolved = set()

    def text(self, s, lang):
        if s is None:
            return ""
        s = str(s)

        def rep(m):
            f = self.figures.get(m.group(1))
            if f is None:
                self.unresolved.add(m.group(1))
                return "?"
            return fmt_value(f.get("value"), f.get("unit"), lang)

        s = _RE_FIG.sub(rep, s)
        s = _RE_LINK.sub(r"\1", s)
        s = _RE_BOLD.sub(r"\1", s)
        s = _RE_ITAL.sub(r"\1", s)
        return s

    def pair(self, v):
        """{fa,en} dict, plain string or number -> {"fa": str, "en": str}."""
        if isinstance(v, dict) and ("fa" in v or "en" in v):
            return {l: self.text(v.get(l), l) for l in LANGS}
        if v is None or isinstance(v, (dict, list, bool)):
            return {"fa": "", "en": ""}
        if isinstance(v, (int, float)):
            return {"fa": _grp(v), "en": _grp(v)}
        return {l: self.text(v, l) for l in LANGS}


def is_pair(v):
    return isinstance(v, dict) and set(v.keys()) <= {"fa", "en"} and ("fa" in v or "en" in v)


# --------------------------------------------------------------------------
# chapter order and fingerprint
# --------------------------------------------------------------------------

def chapter_order(report_dir):
    report_dir = Path(report_dir)
    files = sorted(report_dir.glob("chapters/*.json"))
    by_name = {p.stem: p for p in files}
    order = []
    asm = report_dir / "build" / "99_assemble.py"
    if asm.is_file():
        m = re.search(r"^ORDER\s*=\s*(\[.*?\])", asm.read_text(encoding="utf-8"), re.S | re.M)
        if m:
            try:
                order = [x for x in ast.literal_eval(re.sub(r"#.*", "", m.group(1)))]
            except Exception:  # noqa: BLE001
                order = []
    out = [by_name.pop(n) for n in order if n in by_name]
    out += [by_name[k] for k in sorted(by_name)]
    return out


def fingerprint(report_dir):
    report_dir = Path(report_dir)
    h = hashlib.sha1(ADAPTER.encode())
    for p in [report_dir / "report.meta.json", report_dir / "build" / "99_assemble.py"] + sorted(report_dir.glob("chapters/*.json")):
        if p.is_file():
            st = p.stat()
            h.update(("%s|%d|%d;" % (p.name, st.st_size, st.st_mtime_ns)).encode())
    return h.hexdigest()


def load_figures(report_dir):
    """Merged figure registry of a report (for QUESTION AUTHORING only, never for retrieval)."""
    figs = {}
    for p in chapter_order(report_dir):
        with open(p, encoding="utf-8") as fh:
            figs.update(json.load(fh).get("figures", {}))
    return figs


# --------------------------------------------------------------------------
# building
# --------------------------------------------------------------------------

class _Pair:
    """Adds the same section/unit to the fa and en builders."""

    def __init__(self, builders, resolver):
        self.b = builders
        self.r = resolver

    def section(self, local, title_pair, parent=None, meta=None):
        base, n = local, 2
        while self.b["fa"].has_section(local):
            local = "%s~%d" % (base, n)
            n += 1
        for l in LANGS:
            self.b[l].add_section(local, title_pair.get(l, ""), parent=parent, meta=meta)
        return local

    def unit(self, section_local, local, kind, text_pair, meta=None):
        fa, en = (text_pair.get("fa") or "").strip(), (text_pair.get("en") or "").strip()
        if not fa and not en:
            return None
        for l in LANGS:
            self.b[l].add_unit(section_local, local, kind, text_pair.get(l) or "", meta=meta, keep_empty=True)
        return local


def _join(parts, sep):
    return {l: sep.join(p[l] for p in parts if p[l].strip()) for l in LANGS}


def _table_units(P, sec_local, prefix, header, rows, rows_per_unit, meta):
    """header: list of pairs; rows: list of list of pairs."""
    if not rows:
        return
    head = {l: " | ".join(h[l] for h in header) for l in LANGS}
    for k in range(0, len(rows), rows_per_unit):
        chunk = rows[k:k + rows_per_unit]
        body = {l: "\n".join(" | ".join(c[l] for c in row) for row in chunk) for l in LANGS}
        text = {l: (head[l] + "\n" + body[l]) if head[l].strip(" |") else body[l] for l in LANGS}
        P.unit(sec_local, "%s#table.%d" % (prefix, k // rows_per_unit), "table", text,
               meta=dict(meta, rows=[k, k + len(chunk)]))


def _chart(P, R, sec_local, ch, idx, rows_per_unit, stem):
    kind = ch.get("kind", "")
    cid = str(ch.get("id") or "chart%d" % idx)
    title = R.pair(ch.get("title"))
    prefix = "%s/%s" % (sec_local, cid)
    meta = {"chapter": stem, "chart_id": cid, "chart_kind": kind}
    target = sec_local
    if title["fa"].strip() or title["en"].strip():
        target = P.section(prefix, title, parent=sec_local, meta=meta)
        prefix = target
    if is_pair(ch.get("subtitle")):
        P.unit(target, prefix + "#subtitle", "caption", R.pair(ch["subtitle"]), meta)

    items = ch.get("items") or []
    if kind == "kpis":
        for i, it in enumerate(items):
            lab, sub = R.pair(it.get("label")), R.pair(it.get("sub"))
            val = {l: fmt_value(it.get("value"), it.get("unit"), l) for l in LANGS}
            text = {l: (lab[l] + ": " if lab[l].strip() else "") + val[l] + ((" (" + sub[l] + ")") if sub[l].strip() else "")
                    for l in LANGS}
            P.unit(target, "%s#kpi.%d" % (prefix, i), "fact", text, meta)
    elif kind == "insights":
        for i, it in enumerate(items):
            text = _join([R.pair(it.get("headline")), R.pair(it.get("summary")), R.pair(it.get("action"))], ". ")
            P.unit(target, "%s#insight.%d" % (prefix, i), "paragraph", text, meta)
    else:
        for i, it in enumerate(items):
            if is_pair(it):
                P.unit(target, "%s#item.%d" % (prefix, i), "paragraph", R.pair(it), meta)

    # tables
    if ch.get("columns"):
        cols = ch["columns"]
        rows_src = ch.get("rows") or ch.get("data") or []
        header = [R.pair(c.get("label")) for c in cols]
        rows = []
        for r in rows_src:
            if not isinstance(r, dict):
                continue
            row = []
            for c in cols:
                v = r.get(c.get("key"))
                if c.get("num") and isinstance(v, (int, float)) and not isinstance(v, bool):
                    row.append({l: fmt_value(v, c.get("unit"), l) for l in LANGS})
                else:
                    row.append(R.pair(v))
            rows.append(row)
        _table_units(P, target, prefix, header, rows, rows_per_unit, meta)
    elif isinstance(ch.get("data"), list) and ch["data"] and isinstance(ch["data"][0], dict) and kind != "network":
        data = ch["data"]
        cat = ch.get("category") or {}
        ckey = cat.get("key")
        series = [s for s in (ch.get("series") or []) if isinstance(s, dict) and s.get("key")]
        chart_unit = ch.get("unit") or ""
        if not series:
            keys = []
            for r in data:
                for k, v in r.items():
                    if k != ckey and k not in keys and (isinstance(v, (int, float, str)) or is_pair(v)) and not isinstance(v, bool):
                        keys.append(k)
            series = [{"key": k, "label": {"fa": k, "en": k}} for k in keys]
        header = []
        if ckey:
            header.append(R.pair(cat.get("label")))
        for s in series:
            u = s.get("unit") or chart_unit
            lab = R.pair(s.get("label"))
            header.append({l: lab[l] + ((" (%s)" % u) if u and u not in ("count",) else "") for l in LANGS})
        rows = []
        for r in data:
            row = []
            if ckey:
                row.append(R.pair(r.get(ckey)))
            for s in series:
                row.append(R.pair(r.get(s["key"])))
            rows.append(row)
        _table_units(P, target, prefix, header, rows, rows_per_unit, meta)

    if kind == "network" and ch.get("nodes"):
        text = {l: "; ".join("%s (%s)" % (n.get("label", ""), n.get("group", "")) for n in ch["nodes"]) for l in LANGS}
        P.unit(target, prefix + "#nodes", "list", text, meta)
    for i, ax in enumerate(ch.get("axes") or []):
        parts = [R.pair(ax.get("label")), R.pair(ax.get("help"))]
        lv = [R.pair(x) for x in (ax.get("levels") or [])]
        levels = {l: " ".join("%d) %s" % (j + 1, x[l]) for j, x in enumerate(lv)) for l in LANGS}
        parts.append(levels)
        if is_pair(ax.get("fix")):
            parts.append(R.pair(ax["fix"]))
        P.unit(target, "%s#axis.%d" % (prefix, i), "list", _join(parts, ": "), meta)
    if ch.get("bench"):
        text = _join([R.pair(b.get("label")) for b in ch["bench"]], "; ")
        P.unit(target, prefix + "#bench", "list", text, meta)
    for i, it in enumerate(ch.get("inputs") or []):
        P.unit(target, "%s#input.%d" % (prefix, i), "list", _join([R.pair(it.get("label")), R.pair(it.get("help"))], ": "), meta)
    for i, it in enumerate(ch.get("outputs") or []):
        P.unit(target, "%s#output.%d" % (prefix, i), "list", _join([R.pair(it.get("label")), R.pair(it.get("sub"))], ": "), meta)
    for i, it in enumerate(ch.get("assumptions") or []):
        P.unit(target, "%s#assumption.%d" % (prefix, i), "paragraph", R.pair(it), meta)
    for i, st in enumerate(ch.get("steps") or []):
        parts = [R.pair(st.get("label")), R.pair(st.get("owner")), R.pair(st.get("pain")), R.pair(st.get("fix"))]
        caps = [R.pair(c.get("label")) for c in (st.get("caps") or [])]
        if caps:
            parts.append(_join(caps, ", "))
        P.unit(target, "%s#step.%d" % (prefix, i), "list", _join(parts, " — "), meta)
    for key in ("gateNote", "outHigh", "outLow"):
        if is_pair(ch.get(key)):
            P.unit(target, "%s#%s" % (prefix, key), "paragraph", R.pair(ch[key]), meta)
    if is_pair(ch.get("note")):
        P.unit(target, prefix + "#note", "paragraph", R.pair(ch["note"]), meta)
    if is_pair(ch.get("source")):
        P.unit(target, prefix + "#source", "caption", R.pair(ch["source"]), meta)


def build_report(report_key, report_dir, rows_per_unit=20):
    """-> {"fa": Document, "en": Document} for one KareOne report."""
    report_dir = Path(report_dir)
    with open(report_dir / "report.meta.json", encoding="utf-8") as fh:
        rmeta = json.load(fh)
    chapters = chapter_order(report_dir)
    payloads = []
    figures = {}
    for p in chapters:
        with open(p, encoding="utf-8") as fh:
            d = json.load(fh)
        payloads.append((p.stem, d))
        figures.update(d.get("figures", {}))
    R = Resolver(figures)
    title = R.pair(rmeta.get("title"))
    fp = fingerprint(report_dir)
    ids = {l: make_doc_id("A", report_key, l) for l in LANGS}
    builders = {}
    for l in LANGS:
        other = "en" if l == "fa" else "fa"
        builders[l] = DocBuilder(ids[l], "A", l, title[l], source=str(report_dir).replace("\\", "/"),
                                 pair_id=ids[other], alignment="unit",
                                 meta={"report_key": report_key, "fingerprint": fp, "adapter": ADAPTER,
                                       "chapters": [s for s, _ in payloads]})
    P = _Pair(builders, R)

    # report front matter
    front = P.section("_report", title, meta={"chapter": "_report"})
    P.unit(front, "_report#lede", "paragraph", R.pair(rmeta.get("lede")))
    for i, f in enumerate(((rmeta.get("meta") or {}).get("facts")) or []):
        P.unit(front, "_report#fact.%d" % i, "fact", _join([R.pair(f.get("label")), R.pair(f.get("value"))], ": "))

    for stem, d in payloads:
        chapter_local = None
        for s in d.get("sections", []):
            sid = str(s.get("id") or "section")
            smeta = {"chapter": stem, "section_json_id": sid, "number": s.get("number"), "part": bool(s.get("part"))}
            parent = chapter_local if (s.get("part") and chapter_local) else None
            local = P.section("%s/%s" % (stem, sid), R.pair(s.get("headline")), parent=parent, meta=smeta)
            if parent is None:
                chapter_local = local
            for key in ("lede", "summary", "keyInsight"):
                if is_pair(s.get(key)):
                    P.unit(local, "%s#%s" % (local, key), "paragraph", R.pair(s[key]), smeta)
            charts = s.get("charts") or []
            chart_ids = set()
            for i, ch in enumerate(charts):
                cid = str(ch.get("id") or "chart%d" % i)
                if cid in chart_ids:
                    ch = dict(ch, id="%s~%d" % (cid, i))
                chart_ids.add(cid)
                _chart(P, R, local, ch, i, rows_per_unit, stem)
            if is_pair(s.get("caveat")):
                P.unit(local, "%s#caveat" % local, "paragraph", R.pair(s["caveat"]), smeta)
            for i, t in enumerate(s.get("takeaways") or []):
                v = t.get("text") if isinstance(t, dict) and "text" in t else t
                P.unit(local, "%s#takeaway.%d" % (local, i), "paragraph", R.pair(v), smeta)

    for l in LANGS:
        builders[l].prune_empty()
    docs = {l: builders[l].build() for l in LANGS}
    for l in LANGS:
        docs[l].meta["unresolved_figures"] = sorted(R.unresolved)
        docs[l].meta["n_figures"] = len(figures)
    return docs


def report_dirs(cfg):
    root = Path(cfg.paths["kareone_root"])
    return [(r["key"], root / r["path"]) for r in cfg.corpora["A"]["reports"]]
