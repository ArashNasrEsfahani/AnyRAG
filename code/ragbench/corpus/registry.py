"""Named corpus sets with an on-disk cache.

    from ragbench.corpus import load_set
    A_en = load_set("A_en")            # CorpusSet of 10 English KareOne Documents
    doc = A_en.doc("A.rpt-a.en"); u = A_en.unit("A.rpt-a.en::market#lede")

Set names: "A", "A_fa", "A_en", "B", "B_fa", "B_en", "C" (= "C_fa"), "D" (= "D_en": natural English reports of
global companies, added 2026-09-23 for the fresh holdout; manifest lane "global_en"); and the datasets-v2 domains
(2026-09-25) "BOOK_fa", "BOOK_en" (corpus letter K: public-domain non-fiction books) and "LAW_fa", "LAW_en" (corpus
letter L: statutes and regulations), read from corpus/natural/manifest_v2.csv (lane = set name; the frozen study
manifest is untouched). Their doc ids are "K.<key>.<lang>" / "L.<key>.<lang>"; the fa and en pools are different
documents (not translations), so there is no pair_id and swap_lang does not apply to them.
Cache: work/corpus/docs/<doc_id>.json (rebuilt when the source fingerprint or adapter
changes) and work/corpus/sets/<name>.json (doc ids, fingerprints, skipped sources).
B and C only contain documents whose extraction exists and is complete; everything
skipped is listed in CorpusSet.skipped with the reason.
"""
import csv
import hashlib
from pathlib import Path

from ..config import load_config
from ..util import atomic_write_json, read_json, now_iso
from . import kareone, extracted
from .model import Document, make_doc_id, doc_lang

# datasets v2 (2026-09-25): domain -> corpus letter. Sets are per language only ("BOOK_fa", ...).
V2_DOMAINS = {"BOOK": "K", "LAW": "L"}
V2_LETTERS = {v: k for k, v in V2_DOMAINS.items()}
V2_SET_NAMES = tuple("%s_%s" % (d, l) for d in V2_DOMAINS for l in ("fa", "en"))
V2_MANIFEST = "corpus/natural/manifest_v2.csv"          # default when config.yaml has no paths.manifest_v2

SET_NAMES = ("A", "A_fa", "A_en", "B", "B_fa", "B_en", "C", "C_fa", "D", "D_en") + V2_SET_NAMES
CORPUS_LETTERS = ("A", "B", "C", "D") + tuple(V2_LETTERS)


class CorpusSet:
    def __init__(self, name, docs, skipped=None):
        self.name = name
        self.docs = docs
        self.skipped = skipped or []
        self._by_id = {d.id: d for d in docs}

    @property
    def corpus(self):
        base = self.name.split("_")[0]
        return V2_DOMAINS.get(base, base)

    @property
    def lang(self):
        langs = {d.lang for d in self.docs}
        return langs.pop() if len(langs) == 1 else None

    def doc(self, doc_id):
        return self._by_id[doc_id]

    def has_doc(self, doc_id):
        return doc_id in self._by_id

    def doc_ids(self):
        return [d.id for d in self.docs]

    def unit(self, unit_id):
        return self._by_id[unit_id.split("::", 1)[0]].unit(unit_id)

    def section(self, section_id):
        return self._by_id[section_id.split("::", 1)[0]].section(section_id)

    def subset(self, doc_ids, name=None):
        keep = set(doc_ids)
        return CorpusSet(name or self.name, [d for d in self.docs if d.id in keep], self.skipped)

    def fingerprint(self):
        h = hashlib.sha1(self.name.encode())
        for d in sorted(self.docs, key=lambda x: x.id):
            h.update(("%s:%s;" % (d.id, d.meta.get("fingerprint", ""))).encode())
        return h.hexdigest()

    def stats(self):
        return {"name": self.name, "docs": len(self.docs), "sections": sum(len(d.sections) for d in self.docs),
                "units": sum(len(d.units) for d in self.docs), "chars": sum(len(d.text) for d in self.docs),
                "skipped": len(self.skipped)}

    def __iter__(self):
        return iter(self.docs)

    def __len__(self):
        return len(self.docs)

    def __repr__(self):
        s = self.stats()
        return "CorpusSet(%s: %d docs, %d units, %d chars)" % (self.name, s["docs"], s["units"], s["chars"])


# --------------------------------------------------------------------------
# caching
# --------------------------------------------------------------------------

def _doc_cache_path(cfg, doc_id):
    return cfg.path("corpus_cache") / "docs" / ("%s.json" % doc_id)


def _cached(cfg, doc_id, fp):
    p = _doc_cache_path(cfg, doc_id)
    if not p.is_file():
        return None
    try:
        d = Document.load(p)
    except Exception:  # noqa: BLE001
        return None
    if d.meta.get("fingerprint") != fp or d.meta.get("cfg_rows") != cfg.corpora["table_rows_per_unit"]:
        return None
    return d


def _store(cfg, doc):
    doc.meta["cfg_rows"] = cfg.corpora["table_rows_per_unit"]
    doc.save(_doc_cache_path(cfg, doc.id))


# --------------------------------------------------------------------------
# A
# --------------------------------------------------------------------------

def _load_A(cfg, rebuild=False):
    docs = []
    skipped = []
    rows = cfg.corpora["table_rows_per_unit"]
    for key, rdir in kareone.report_dirs(cfg):
        if not (rdir / "report.meta.json").is_file() or not list(rdir.glob("chapters/*.json")):
            skipped.append({"source": str(rdir), "reason": "missing report.meta.json or chapters/*.json"})
            continue
        fp = kareone.fingerprint(rdir)
        pair = {}
        if not rebuild:
            for l in ("fa", "en"):
                c = _cached(cfg, make_doc_id("A", key, l), fp)
                if c is not None:
                    pair[l] = c
        if len(pair) != 2:
            pair = kareone.build_report(key, rdir, rows_per_unit=rows)
            for l in ("fa", "en"):
                _store(cfg, pair[l])
        docs.extend([pair["fa"], pair["en"]])
    return docs, skipped


# --------------------------------------------------------------------------
# B and C
# --------------------------------------------------------------------------

def read_manifest(cfg):
    p = cfg.path("manifest")
    with open(p, encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def _norm_path(p):
    return (p or "").replace("\\", "/").strip().lower()


def _load_extracted_doc(cfg, doc_dir, doc_id, corpus, lang, title, pair_id, alignment, rebuild, allow_partial=False):
    fp = extracted.fingerprint(doc_dir)
    if not rebuild:
        c = _cached(cfg, doc_id, fp)
        if c is not None:
            return c
    d = extracted.build_extracted(doc_dir, doc_id, corpus, lang, title=title, pair_id=pair_id, alignment=alignment,
                                  rows_per_unit=cfg.corpora["table_rows_per_unit"])
    _store(cfg, d)
    return d


def _load_B(cfg, rebuild=False, extracted_root=None, allow_partial=False):
    root = Path(extracted_root) if extracted_root else cfg.path("extracted_root")
    manifest = {_norm_path(r.get("local_path")): r for r in read_manifest(cfg) if (r.get("local_path") or "").strip()}
    docs, skipped = [], []
    for pair in cfg.corpora["B"]["pairs"]:
        key = pair["key"]
        dirs, ok = {}, True
        for l in ("fa", "en"):
            ddir = extracted.out_dir_for(root, pair[l])
            usable, why = extracted.is_usable(ddir, allow_partial=allow_partial, exclude_verdicts=())
            if not usable:
                skipped.append({"source": pair[l], "reason": why, "pair": key})
                ok = False
            dirs[l] = ddir
        if not ok:
            continue
        for l in ("fa", "en"):
            row = manifest.get(_norm_path(pair[l]), {})
            other = "en" if l == "fa" else "fa"
            docs.append(_load_extracted_doc(cfg, dirs[l], make_doc_id("B", key, l), "B", l, row.get("title_%s" % l),
                                            make_doc_id("B", key, other), "document", rebuild, allow_partial))
    return docs, skipped


def c_key(local_path):
    stem = Path(local_path.replace("\\", "/")).stem
    if stem.endswith("_fa"):
        stem = stem[:-3]
    return stem


def _load_C(cfg, rebuild=False, extracted_root=None):
    root = Path(extracted_root) if extracted_root else cfg.path("extracted_root")
    cc = cfg.corpora["C"]
    b_paths = {_norm_path(p[l]) for p in cfg.corpora["B"]["pairs"] for l in ("fa", "en")}
    docs, skipped = [], []
    seen = set()
    for r in read_manifest(cfg):
        lp = (r.get("local_path") or "").strip()
        if not lp or _norm_path(lp) in b_paths:
            continue
        if (r.get("language") or "").strip().lower() != "fa" or (r.get("parallel_file") or "").strip():
            continue
        ddir = extracted.out_dir_for(root, lp)
        usable, why = extracted.is_usable(ddir, allow_partial=cc.get("allow_partial_extractions", False),
                                          exclude_verdicts=tuple(cc.get("exclude_verdicts", ["fail"])))
        if not usable:
            skipped.append({"source": lp, "reason": why})
            continue
        doc_id = make_doc_id("C", c_key(lp), "fa")
        if doc_id in seen:
            skipped.append({"source": lp, "reason": "duplicate key %s" % doc_id})
            continue
        seen.add(doc_id)
        docs.append(_load_extracted_doc(cfg, ddir, doc_id, "C", "fa", r.get("title_fa"), None, None, rebuild))
    return docs, skipped


def _load_D(cfg, rebuild=False, extracted_root=None):
    """Corpus D: the English annual reports of global companies (manifest lane global_en, language en, no parallel
    file) - the natural-English counterpart of C. Same extraction and cache path as C."""
    root = Path(extracted_root) if extracted_root else cfg.path("extracted_root")
    cc = cfg.corpora["C"]
    docs, skipped, seen = [], [], set()
    for r in read_manifest(cfg):
        lp = (r.get("local_path") or "").strip()
        if not lp or (r.get("lane") or "").strip() != "global_en" or (r.get("language") or "").strip().lower() != "en":
            continue
        ddir = extracted.out_dir_for(root, lp)
        usable, why = extracted.is_usable(ddir, allow_partial=cc.get("allow_partial_extractions", False),
                                          exclude_verdicts=tuple(cc.get("exclude_verdicts", ["fail"])))
        if not usable:
            skipped.append({"source": lp, "reason": why})
            continue
        stem = Path(lp.replace("\\", "/")).stem
        key = stem[:-3] if stem.endswith("_en") else stem
        doc_id = make_doc_id("D", key, "en")
        if doc_id in seen:
            skipped.append({"source": lp, "reason": "duplicate key %s" % doc_id})
            continue
        seen.add(doc_id)
        docs.append(_load_extracted_doc(cfg, ddir, doc_id, "D", "en", r.get("title_en"), None, None, rebuild))
    return docs, skipped


# --------------------------------------------------------------------------
# datasets v2: BOOK (K) and LAW (L), from manifest_v2.csv
# --------------------------------------------------------------------------

def manifest_v2_path(cfg):
    try:
        return cfg.path("manifest_v2")
    except KeyError:
        return cfg.project_root / V2_MANIFEST


def read_manifest_v2(cfg):
    p = manifest_v2_path(cfg)
    if not p.is_file():
        return []
    with open(p, encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def v2_key(local_path, lang):
    """'law_en/eurlex-32016R0679_gdpr_en.html' -> 'eurlex-32016R0679_gdpr' (the file stem without its _<lang>)."""
    stem = Path(local_path.replace("\\", "/")).stem
    return stem[:-3] if stem.endswith("_" + lang) else stem


def _load_v2(cfg, name, rebuild=False, extracted_root=None):
    """One datasets-v2 set: the manifest_v2 rows whose lane is `name` ("BOOK_fa", ...) and whose language matches.
    Same extraction, usability rules and document cache as C and D."""
    domain, lang = name.split("_", 1)
    letter = V2_DOMAINS[domain]
    root = Path(extracted_root) if extracted_root else cfg.path("extracted_root")
    cc = cfg.corpora["C"]
    docs, skipped, seen = [], [], set()
    for r in read_manifest_v2(cfg):
        lp = (r.get("local_path") or "").strip()
        if not lp or (r.get("lane") or "").strip() != name:
            continue
        if (r.get("language") or "").strip().lower() != lang:
            skipped.append({"source": lp, "reason": "language %r is not %s" % (r.get("language"), lang)})
            continue
        ddir = extracted.out_dir_for(root, lp)
        usable, why = extracted.is_usable(ddir, allow_partial=cc.get("allow_partial_extractions", False),
                                          exclude_verdicts=tuple(cc.get("exclude_verdicts", ["fail"])))
        if not usable:
            skipped.append({"source": lp, "reason": why})
            continue
        doc_id = make_doc_id(letter, v2_key(lp, lang), lang)
        if doc_id in seen:
            skipped.append({"source": lp, "reason": "duplicate key %s" % doc_id})
            continue
        seen.add(doc_id)
        title = r.get("title_%s" % lang) or r.get("title_en") or r.get("title_fa")
        docs.append(_load_extracted_doc(cfg, ddir, doc_id, letter, lang, title, None, None, rebuild))
    return docs, skipped


# --------------------------------------------------------------------------
# public
# --------------------------------------------------------------------------

_MEMO = {}


def load_set(name, cfg=None, rebuild=False, extracted_root=None, allow_partial=False):
    """CorpusSet for a set name. Memoised per process; cached on disk per document."""
    cfg = cfg or load_config()
    if name not in SET_NAMES:
        raise ValueError("unknown corpus set %r; choose from %s" % (name, SET_NAMES))
    memo_key = (name, str(extracted_root), allow_partial)
    if not rebuild and memo_key in _MEMO:
        return _MEMO[memo_key]
    base = name.split("_")[0]
    lang = name.split("_")[1] if "_" in name else None
    if base in V2_DOMAINS:
        docs, skipped = _load_v2(cfg, name, rebuild, extracted_root)
    elif base == "A":
        docs, skipped = _load_A(cfg, rebuild)
    elif base == "B":
        docs, skipped = _load_B(cfg, rebuild, extracted_root, allow_partial)
    elif base == "D":
        docs, skipped = _load_D(cfg, rebuild, extracted_root)
    else:
        docs, skipped = _load_C(cfg, rebuild, extracted_root)
    if lang:
        docs = [d for d in docs if d.lang == lang]
    cs = CorpusSet(name, docs, skipped)
    if extracted_root is None and not allow_partial:
        atomic_write_json(cfg.path("corpus_cache") / "sets" / ("%s.json" % name),
                          {"name": name, "built_at": now_iso(), "fingerprint": cs.fingerprint(), "stats": cs.stats(),
                           "docs": [{"id": d.id, "fingerprint": d.meta.get("fingerprint"), **d.stats()} for d in docs],
                           "skipped": skipped}, indent=1)
    _MEMO[memo_key] = cs
    return cs


def set_name_for(corpus, d_lang):
    """Corpus set searched for a question of `corpus` in document language `d_lang`. The v2 domains take their
    letter (K, L) or their domain name (BOOK, LAW)."""
    if corpus in V2_LETTERS or corpus in V2_DOMAINS:
        if d_lang not in ("fa", "en"):
            raise ValueError("bad document language %r" % (d_lang,))
        return "%s_%s" % (V2_LETTERS.get(corpus, corpus), d_lang)
    if corpus == "C":
        if d_lang != "fa":
            raise ValueError("corpus C is Persian-only")
        return "C"
    if corpus == "D" and d_lang != "en":
        raise ValueError("corpus D is English-only")
    return "%s_%s" % (corpus, d_lang)


def load_document(doc_id, cfg=None):
    corpus = doc_id.split(".", 1)[0]
    name = set_name_for(corpus, doc_lang(doc_id))
    return load_set(name, cfg).doc(doc_id)
