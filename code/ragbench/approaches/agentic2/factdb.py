"""The fact DB: an in-memory SQLite table of regex-extracted facts, built on the fly for the targeted documents only.

Rows come from the OURS L0 extractor (approaches/ours/lite.py extract_facts, FACTX_VERSION; ~0.1 s per report, no
model), filtered like L0 (every number of the value occurs in the section text). One row per fact:

    facts(doc_id, year, period, section_id, section_path, label, value, unit, unit_id, quote, score)

  year     the 4-digit year of the fact's own period (a year / range end / date in its clause or table column), else the
           document's year from its id/title (docmeta), else NULL (corpus A reports have no year)
  label    the extractor's label words (the number's clause, a detached label line or unit, a table row + header)
  quote    the unit text around the value (<= 240 chars)

Rows are cached per document CONTENT hash (sha1 of the text + extractor versions + options) in
work/indexes/AGENTIC2/_factdb/facts.sqlite, so a later question or run loads them instead of re-extracting.
query() keeps rows whose label (or unit) matches a plan term regex (both languages, morphology from the v1 planner) and
whose year lies in the question's period (rows without a year always pass), best first: distinct terms matched, then
extractor score.
"""
import re
import sqlite3
import time

from ...normalize import normalize
from ...util import KVCache, sha1_json
from ..agentic import planner as P
from ..ours.indexer import fact_value_ok, text_number_set
from ..ours.lite import FACTX_VERSION, L0_DEFAULTS, extract_facts
from .defaults import FACTDB_VERSION

_YEAR = re.compile(r"(?<![0-9])(1[34][0-9]{2}|19[0-9]{2}|20[0-9]{2})(?![0-9])")


def _ms(t0):
    return round((time.perf_counter() - t0) * 1000.0, 1)


def _year_of(period):
    ys = _YEAR.findall(normalize(period or ""))
    return int(ys[-1]) if ys else None


def doc_fact_rows(doc, doc_years=None):
    """-> [row dict] for one document (program only)."""
    rows = []
    for s in doc.sections:
        units = [(uid, doc.unit(uid).text) for uid in s.unit_ids if (doc.unit(uid).text or "").strip()]
        if not units:
            continue
        facts, _st = extract_facts(units, s.title, L0_DEFAULTS)
        nums = text_number_set(s.title + "\n" + "\n".join(t for _, t in units))
        path = " > ".join(t for t in doc.section_path(s.id) if t)
        for f in facts:
            ok, _why = fact_value_ok(f["value"], nums)
            if not ok:
                continue
            utext = doc.unit(f["unit_id"]).text if f.get("unit_id") else ""
            p = utext.find(f["value"])
            a = max(0, p - 120) if p >= 0 else 0
            quote = utext[a:a + 240]
            year = _year_of(f.get("period"))
            if year is None and doc_years:
                year = doc_years[1]
            rows.append({"doc_id": doc.id, "year": year, "period": f.get("period") or "", "section_id": s.id,
                         "section_path": path[:200], "label": f.get("metric") or "", "value": f["value"],
                         "unit": f.get("unit") or "", "unit_id": f.get("unit_id") or "", "quote": quote,
                         "score": float(f.get("score") or 0)})
    return rows


class FactDB:
    """Per-question in-memory table over the targeted documents; rows cached per document content hash."""

    COLUMNS = ("doc_id", "year", "period", "section_id", "section_path", "label", "value", "unit", "unit_id", "quote",
               "score")

    def __init__(self, cfg, cache_path=None):
        self.cfg = cfg
        self.cache_path = cache_path or (cfg.path("index_root") / "AGENTIC2" / "_factdb" / "facts.sqlite")
        self._kv = None
        self._mem = {}

    def kv(self):
        if self._kv is None:
            self._kv = KVCache(self.cache_path)
        return self._kv

    def doc_key(self, doc):
        return sha1_json({"text": sha1_json(doc.text), "doc": doc.id, "factx": FACTX_VERSION, "db": FACTDB_VERSION,
                          "l0": L0_DEFAULTS})

    def rows_for(self, doc, doc_years=None):
        """-> (rows, info) from memory, the sqlite cache, or a fresh extraction."""
        key = self.doc_key(doc) + ":%s" % (list(doc_years) if doc_years else None)
        if key in self._mem:
            return self._mem[key], {"doc_id": doc.id, "source": "memory", "ms": 0.0, "rows": len(self._mem[key])}
        t0 = time.perf_counter()
        hit = self.kv().get("f:" + key)
        if hit is not None:
            rows = hit["rows"]
            info = {"doc_id": doc.id, "source": "cache", "ms": _ms(t0), "rows": len(rows),
                    "extract_ms_original": hit.get("extract_ms")}
        else:
            rows = doc_fact_rows(doc, doc_years)
            ms = _ms(t0)
            self.kv().put("f:" + key, {"rows": rows, "extract_ms": ms})
            info = {"doc_id": doc.id, "source": "extracted", "ms": ms, "rows": len(rows), "extract_ms_original": ms}
        self._mem[key] = rows
        return rows, info

    @staticmethod
    def table(rows):
        con = sqlite3.connect(":memory:")

        def regexp(pattern, value):
            try:
                return 1 if re.search(pattern, value or "", re.I) else 0
            except re.error:
                return 0
        con.create_function("REGEXP", 2, regexp)
        con.execute("CREATE TABLE facts (doc_id TEXT, year INTEGER, period TEXT, section_id TEXT, section_path TEXT, "
                    "label TEXT, value TEXT, unit TEXT, unit_id TEXT, quote TEXT, score REAL)")
        con.executemany("INSERT INTO facts VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                        [tuple(r.get(c) for c in FactDB.COLUMNS) for r in rows])
        return con

    @staticmethod
    def term_regexes(plan, d_lang):
        terms = plan.get("terms") or {}
        langs = [d_lang] if terms.get(d_lang) else [l for l in ("fa", "en") if terms.get(l)]
        out = []
        for lang in langs:
            for t in terms.get(lang) or []:
                pats = P.term_patterns(t, lang)
                if pats:
                    out.append((t, "|".join("(?:%s)" % p for p in pats)))
        return out

    def query(self, con, plan, d_lang, doc_ids, period=None, max_rows=24, per_doc=6):
        """-> ([row dict + matched_terms], info)"""
        t0 = time.perf_counter()
        terms = self.term_regexes(plan, d_lang)
        info = {"terms": [t for t, _ in terms], "period": list(period) if period else None}
        if not terms or not doc_ids:
            info.update(matched=0, kept=0, ms=_ms(t0))
            return [], info
        any_rx = "|".join("(?:%s)" % rx for _t, rx in terms)
        sql = ("SELECT %s FROM facts WHERE doc_id IN (%s) AND (label REGEXP ? OR unit REGEXP ?)" %
               (", ".join(self.COLUMNS), ",".join("?" * len(doc_ids))))
        args = list(doc_ids) + [any_rx, any_rx]
        p0, p1 = period if period else (None, None)
        if p0 is not None or p1 is not None:
            sql += " AND (year IS NULL OR (year >= ? AND year <= ?))"
            args += [p0 if p0 is not None else -10 ** 9, p1 if p1 is not None else 10 ** 9]
        rows = []
        for tup in con.execute(sql, args):
            r = dict(zip(self.COLUMNS, tup))
            text = (r["label"] or "") + " " + (r["unit"] or "")
            r["matched_terms"] = [t for t, rx in terms if re.search(rx, text, re.I)]
            rows.append(r)
        rows.sort(key=lambda r: (-len(r["matched_terms"]), -r["score"], r["doc_id"], r["unit_id"]))
        kept, per = [], {}
        for r in rows:
            if per.get(r["doc_id"], 0) >= per_doc:
                continue
            per[r["doc_id"]] = per.get(r["doc_id"], 0) + 1
            kept.append(r)
            if len(kept) >= max_rows:
                break
        info.update(matched=len(rows), kept=len(kept), ms=_ms(t0))
        return kept, info

    def close(self):
        if self._kv is not None:
            self._kv.close()
            self._kv = None
