"""Document -> Section tree -> Units.

IDs
  doc id      "<corpus>.<key>.<lang>"             e.g. "A.rpt-a.fa", "B.digikala-1400.en", "C.dataak_annual-report_1401.fa"
  section id  "<doc id>::<section local id>"      e.g. "A.rpt-a.fa::market/market-trade"
  unit id     "<doc id>::<unit local id>"         e.g. "A.rpt-a.fa::market/market-trade#lede"
The local part of a unit id never contains the language, so for unit-aligned corpora
(A) `swap_lang("A.rpt-a.fa::market#lede", "en") == "A.rpt-a.en::market#lede"` is the
same passage in English. Corpus B is only document-aligned: swap_lang gives the
paired DOCUMENT id, but unit ids do not correspond.

Text and spans
  Document.text is the whole document, linearised in section pre-order. Each section
  contributes "<title>\\n<unit>\\n\\n<unit>..." and blocks are separated by "\\n\\n".
  Unit.span / Section.span are [start, end) offsets into Document.text, so
  Document.text[u.span[0]:u.span[1]] == u.text. Section.span covers the title and the
  section's OWN units; Section.subtree_span also covers every descendant.
  All text is normalised with ragbench.normalize.normalize_lines (titles with normalize);
  the raw strings are kept in text_raw / title_raw for the no_normalization ablation.

Fairness: approaches may use ids, titles, levels, tree structure, unit kinds and
text. Document.meta / Unit.meta / Section.meta are provenance for question authoring
and debugging and must not drive retrieval.
"""
import bisect
import json
from dataclasses import dataclass, field, asdict
from pathlib import Path

from ..normalize import normalize, normalize_lines, detect_lang

UNIT_KINDS = ("paragraph", "list", "caption", "fact", "table", "heading")


def make_doc_id(corpus, key, lang):
    return "%s.%s.%s" % (corpus, key, lang)


def split_id(full_id):
    """'A.rpt-a.fa::market#lede' -> ('A.rpt-a.fa', 'market#lede'); a bare doc id -> (doc id, '')."""
    if "::" in full_id:
        d, l = full_id.split("::", 1)
        return d, l
    return full_id, ""


def doc_lang(doc_id):
    return doc_id.rsplit(".", 1)[-1]


def swap_lang(any_id, lang):
    """Same id in the other language edition (doc, section or unit id)."""
    d, l = split_id(any_id)
    base = d.rsplit(".", 1)[0]
    nd = "%s.%s" % (base, lang)
    return nd + ("::" + l if l else "")


@dataclass
class Unit:
    id: str
    doc_id: str
    section_id: str
    kind: str            # paragraph | list | caption | fact | table | heading
    lang: str            # detected script of this unit's text ("fa" / "en")
    text: str            # normalised; may contain newlines (tables)
    text_raw: str
    span: list           # [start, end) in Document.text
    order: int           # position in document order
    meta: dict = field(default_factory=dict)

    @property
    def local_id(self):
        return split_id(self.id)[1]


@dataclass
class Section:
    id: str
    doc_id: str
    title: str
    title_raw: str
    level: int           # depth in the tree, roots are 1
    parent: str          # section id or None
    children: list       # section ids, document order
    unit_ids: list       # the section's OWN units, document order
    span: list           # [start, end) title + own units
    subtree_span: list   # [start, end) including all descendants
    order: int           # pre-order index
    meta: dict = field(default_factory=dict)

    @property
    def local_id(self):
        return split_id(self.id)[1]


class Document:
    def __init__(self, id, corpus, lang, title, title_raw, source, sections, units, text,
                 pair_id=None, alignment=None, meta=None):
        self.id = id
        self.corpus = corpus
        self.lang = lang
        self.title = title
        self.title_raw = title_raw
        self.source = source
        self.sections = sections          # list[Section], pre-order
        self.units = units                # list[Unit], document order
        self.text = text
        self.pair_id = pair_id            # doc id of the other-language edition, or None
        self.alignment = alignment        # "unit" (A) | "document" (B) | None (C)
        self.meta = meta or {}
        self._sec = {s.id: s for s in sections}
        self._unit = {u.id: u for u in units}
        self._starts = [u.span[0] for u in units]

    # lookups ----------------------------------------------------------------
    def section(self, sid):
        return self._sec[sid]

    def unit(self, uid):
        return self._unit[uid]

    def has_unit(self, uid):
        return uid in self._unit

    def roots(self):
        return [s for s in self.sections if s.parent is None]

    def children(self, sid):
        return [self._sec[c] for c in self._sec[sid].children]

    def descendants(self, sid):
        out = []
        stack = list(reversed(self._sec[sid].children))
        while stack:
            s = self._sec[stack.pop()]
            out.append(s)
            stack.extend(reversed(s.children))
        return out

    def leaves(self):
        return [s for s in self.sections if not s.children]

    def section_units(self, sid, recursive=False):
        secs = [self._sec[sid]] + (self.descendants(sid) if recursive else [])
        return [self._unit[u] for s in secs for u in s.unit_ids]

    def section_text(self, sid, recursive=False):
        """Title + own units (or the whole subtree), exactly as in Document.text."""
        s = self._sec[sid]
        a, b = s.subtree_span if recursive else s.span
        return self.text[a:b]

    def section_path(self, sid):
        """Titles from the root down to this section."""
        out = []
        s = self._sec[sid]
        while s is not None:
            out.append(s.title)
            s = self._sec[s.parent] if s.parent else None
        return list(reversed(out))

    def units_overlapping(self, start, end):
        """Units whose span intersects [start, end) - maps chunks/windows back to unit ids."""
        i = max(bisect.bisect_right(self._starts, start) - 1, 0)
        out = []
        while i < len(self.units) and self.units[i].span[0] < end:
            u = self.units[i]
            if u.span[1] > start and u.span[0] < end:
                out.append(u)
            i += 1
        return out

    def char_len(self):
        return len(self.text)

    # serialisation ------------------------------------------------------------
    def to_dict(self):
        return {"id": self.id, "corpus": self.corpus, "lang": self.lang, "title": self.title,
                "title_raw": self.title_raw, "source": self.source, "pair_id": self.pair_id,
                "alignment": self.alignment, "meta": self.meta, "text": self.text,
                "sections": [asdict(s) for s in self.sections], "units": [asdict(u) for u in self.units]}

    @classmethod
    def from_dict(cls, d):
        return cls(id=d["id"], corpus=d["corpus"], lang=d["lang"], title=d["title"], title_raw=d["title_raw"],
                   source=d["source"], sections=[Section(**s) for s in d["sections"]],
                   units=[Unit(**u) for u in d["units"]], text=d["text"], pair_id=d.get("pair_id"),
                   alignment=d.get("alignment"), meta=d.get("meta") or {})

    def save(self, path):
        from ..util import atomic_write_text
        return atomic_write_text(path, json.dumps(self.to_dict(), ensure_ascii=False))

    @classmethod
    def load(cls, path):
        with open(path, encoding="utf-8") as fh:
            return cls.from_dict(json.load(fh))

    def stats(self):
        kinds = {}
        for u in self.units:
            kinds[u.kind] = kinds.get(u.kind, 0) + 1
        return {"id": self.id, "sections": len(self.sections), "units": len(self.units),
                "chars": len(self.text), "max_depth": max((s.level for s in self.sections), default=0),
                "unit_kinds": kinds}

    def __repr__(self):
        return "Document(%s, %d sections, %d units, %d chars)" % (self.id, len(self.sections), len(self.units), len(self.text))


class DocBuilder:
    """Adapters describe the tree; the builder normalises text, linearises and assigns spans.

        b = DocBuilder("A.rpt-a.fa", "A", "fa", "title", source="...")
        b.add_section("market", "The market")
        b.add_section("market/iran", "Iran inside it", parent="market")
        b.add_unit("market", "market#lede", "paragraph", "raw text")
        doc = b.build()
    """

    def __init__(self, doc_id, corpus, lang, title_raw, source, pair_id=None, alignment=None, meta=None):
        self.doc_id = doc_id
        self.corpus = corpus
        self.lang = lang
        self.title_raw = title_raw or ""
        self.source = source
        self.pair_id = pair_id
        self.alignment = alignment
        self.meta = meta or {}
        self._sections = {}       # local -> dict
        self._roots = []
        self._unit_locals = set()

    def has_section(self, local):
        return local in self._sections

    def add_section(self, local, title_raw, parent=None, meta=None):
        if local in self._sections:
            raise ValueError("duplicate section id %s in %s" % (local, self.doc_id))
        if parent is not None and parent not in self._sections:
            raise ValueError("unknown parent %s for %s" % (parent, local))
        self._sections[local] = {"local": local, "title_raw": title_raw or "", "parent": parent, "children": [],
                                 "units": [], "meta": meta or {}}
        if parent is None:
            self._roots.append(local)
        else:
            self._sections[parent]["children"].append(local)
        return "%s::%s" % (self.doc_id, local)

    def add_unit(self, section_local, local, kind, text_raw, meta=None, keep_empty=False, text_norm=None):
        """text_norm: an already-normalised form from the source (e.g. the extractor's
        dehyphenated text_norm); it is passed through normalize_lines, which is idempotent."""
        if kind not in UNIT_KINDS:
            raise ValueError("unknown unit kind %s" % kind)
        if local in self._unit_locals:
            raise ValueError("duplicate unit id %s in %s" % (local, self.doc_id))
        text = normalize_lines(text_norm if text_norm is not None else (text_raw or ""))
        if not text and not keep_empty:
            return None
        self._unit_locals.add(local)
        self._sections[section_local]["units"].append((local, kind, text_raw or "", text, meta or {}))
        return "%s::%s" % (self.doc_id, local)

    def prune_empty(self):
        """Drop sections with no units and no (remaining) children. Returns number dropped."""
        dropped = 0
        changed = True
        while changed:
            changed = False
            for local in list(self._sections):
                s = self._sections[local]
                if not s["units"] and not s["children"]:
                    if s["parent"] is None:
                        self._roots.remove(local)
                    else:
                        self._sections[s["parent"]]["children"].remove(local)
                    del self._sections[local]
                    dropped += 1
                    changed = True
        return dropped

    def build(self):
        parts = []
        pos = 0
        sections, units = [], []

        def emit(s):
            nonlocal pos
            if parts:
                parts.append("\n\n")
                pos += 2

        order = []
        stack = list(reversed(self._roots))
        depth = {r: 1 for r in self._roots}
        while stack:
            local = stack.pop()
            order.append(local)
            for c in reversed(self._sections[local]["children"]):
                depth[c] = depth[local] + 1
                stack.append(c)

        spans = {}
        for local in order:
            s = self._sections[local]
            title = normalize(s["title_raw"])
            block_start = None
            if title or s["units"]:
                emit(s)
                block_start = pos
            if title:
                parts.append(title)
                pos += len(title)
            first = True
            for (ulocal, kind, text_raw, text, umeta) in s["units"]:
                sep = "\n" if (first and title) else ("" if first else "\n\n")
                parts.append(sep)
                pos += len(sep)
                start = pos
                parts.append(text)
                pos += len(text)
                units.append(Unit(id="%s::%s" % (self.doc_id, ulocal), doc_id=self.doc_id,
                                  section_id="%s::%s" % (self.doc_id, local), kind=kind,
                                  lang=detect_lang(text, default=self.lang), text=text, text_raw=text_raw,
                                  span=[start, pos], order=len(units), meta=umeta))
                first = False
            if block_start is None:
                block_start = pos
            spans[local] = [block_start, pos]
            sections.append(Section(id="%s::%s" % (self.doc_id, local), doc_id=self.doc_id, title=title,
                                    title_raw=s["title_raw"], level=depth[local],
                                    parent=("%s::%s" % (self.doc_id, s["parent"])) if s["parent"] else None,
                                    children=["%s::%s" % (self.doc_id, c) for c in s["children"]],
                                    unit_ids=["%s::%s" % (self.doc_id, u[0]) for u in s["units"]],
                                    span=spans[local], subtree_span=None, order=len(sections), meta=s["meta"]))
        # subtree spans (children come after their parent in pre-order)
        by_local = {sec.local_id: sec for sec in sections}
        for sec in reversed(sections):
            end = sec.span[1]
            for c in sec.children:
                end = max(end, by_local[split_id(c)[1]].subtree_span[1])
            sec.subtree_span = [sec.span[0], end]
        text = "".join(parts)
        return Document(id=self.doc_id, corpus=self.corpus, lang=self.lang, title=normalize(self.title_raw),
                        title_raw=self.title_raw, source=self.source, sections=sections, units=units, text=text,
                        pair_id=self.pair_id, alignment=self.alignment, meta=self.meta)
