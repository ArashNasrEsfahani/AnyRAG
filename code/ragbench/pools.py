"""Frozen search pools for the multi-document experiment (2026-09-24, P0 list of the multi-document design).

A pool is an explicit list of documents, chosen independently of any question's gold evidence and frozen as a JSON
manifest under work/multidoc/pools/<pool_id>.json before questions run in it. A run takes it with
`ragbench.run --pool <manifest>`; multi-document question files refuse --docs-from-questions (which derives the pool
from gold units) and --docs, and require --pool.

The pool is a SCOPE MASK, not a smaller corpus: every index is built over the full frozen corpus set (so BM25/IDF,
vector neighbourhoods and routing statistics are those of the full set), and Approach.make_query replaces a
question's "corpus" scope with the pool's documents in the query's set (a listed scope is intersected with it).

Manifest (schema "pool-v1"):
  pool_id, corpus, description, derived_from ("independent" | "manual" | "random" | "stratified" | ...; never
  gold/question-derived), selection (free text: how the documents were chosen), created,
  frozen_sets {set name: {fingerprint, n_docs}}  - the full sets the indexes are built over,
  docs [{doc_id, set, fingerprint, text_sha1, chars}]  - sorted by doc_id,
  manifest_sha1  - sha1 over everything else; a changed file no longer loads.

    .venv/Scripts/python.exe -m ragbench.pools freeze --pool-id C-tiny-test --corpus C --docs C.a.fa,C.b.fa \
        --derived-from manual --selection "smoke test"
    .venv/Scripts/python.exe -m ragbench.pools verify work/multidoc/pools/C-tiny-test.json
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

from .util import atomic_write_text, now_iso

POOL_SCHEMA = "pool-v1"
GOLD_DERIVED = {"gold", "gold_units", "questions", "docs_from_questions", "question_file"}


class PoolError(ValueError):
    pass


def pool_root(cfg):
    return Path(cfg.project_root) / "work" / "multidoc" / "pools"


def _canon(obj):
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def manifest_sha1(manifest):
    body = {k: v for k, v in manifest.items() if k != "manifest_sha1"}
    return hashlib.sha1(_canon(body).encode("utf-8")).hexdigest()


def doc_entry(doc, set_name):
    return {"doc_id": doc.id, "set": set_name, "fingerprint": (doc.meta or {}).get("fingerprint"),
            "text_sha1": hashlib.sha1((doc.text or "").encode("utf-8")).hexdigest(), "chars": len(doc.text or "")}


def build_manifest(pool_id, corpus, doc_ids, sets, derived_from, selection="", description="", created=None):
    """sets: {set name: CorpusSet} - the FULL frozen sets of the corpus (every doc id must be in one of them)."""
    if not pool_id or any(c in pool_id for c in "/\\: "):
        raise PoolError("pool_id must be a plain name, got %r" % pool_id)
    if str(derived_from).lower() in GOLD_DERIVED:
        raise PoolError("a pool must not be derived from gold evidence or a question file (derived_from=%r)" % derived_from)
    ids = sorted(set(doc_ids))
    if not ids:
        raise PoolError("empty pool")
    docs, missing = [], []
    for d in ids:
        hit = [(name, cs) for name, cs in sets.items() if cs.has_doc(d)]
        if not hit:
            missing.append(d)
            continue
        name, cs = hit[0]
        docs.append(doc_entry(cs.doc(d), name))
    if missing:
        raise PoolError("documents not in the frozen sets %s: %s" % (sorted(sets), missing[:10]))
    used = sorted({e["set"] for e in docs})
    m = {"schema": POOL_SCHEMA, "pool_id": pool_id, "corpus": corpus, "description": description,
         "derived_from": derived_from, "selection": selection, "created": created or now_iso(),
         "frozen_sets": {n: {"fingerprint": sets[n].fingerprint(), "n_docs": len(sets[n])} for n in used},
         "docs": docs}
    m["manifest_sha1"] = manifest_sha1(m)
    return m


def write_manifest(path, manifest):
    """Freeze: write once. Rewriting the same content is a no-op; different content under an existing path is refused."""
    path = Path(path)
    if manifest.get("manifest_sha1") != manifest_sha1(manifest):
        raise PoolError("manifest_sha1 does not match the manifest body")
    if path.exists():
        old = json.loads(path.read_text(encoding="utf-8"))
        if old.get("manifest_sha1") == manifest["manifest_sha1"]:
            return path
        raise PoolError("pool manifest %s is frozen with another content (sha1 %s); choose a new pool_id" % (
            path, str(old.get("manifest_sha1"))[:12]))
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(path, json.dumps(manifest, ensure_ascii=False, indent=1) + "\n")
    return path


class Pool:
    def __init__(self, manifest, path=None):
        self.manifest = manifest
        self.path = str(path) if path else None
        self.pool_id = manifest["pool_id"]
        self.corpus = manifest.get("corpus")
        self.sha1 = manifest["manifest_sha1"]
        self._by_set = {}
        for e in manifest["docs"]:
            self._by_set.setdefault(e["set"], []).append(e["doc_id"])

    def doc_ids(self, set_name=None):
        if set_name is None:
            return [e["doc_id"] for e in self.manifest["docs"]]
        return list(self._by_set.get(set_name, []))

    def mask(self, scope, set_name):
        """A query's doc scope under the pool: "corpus" -> the pool's documents in this set; a list -> its members
        that are in the pool (question order kept)."""
        docs = self.doc_ids(set_name)
        if scope in (None, "corpus"):
            return list(docs)
        keep = set(docs)
        return [d for d in scope if d in keep]

    def identity(self):
        return {"pool_id": self.pool_id, "manifest_sha1": self.sha1}

    def verify(self, sets):
        """Problems (empty = fine) against the current full corpus sets {name: CorpusSet}: a frozen set whose
        fingerprint changed, a missing document, or a document whose fingerprint or text changed."""
        probs = []
        for name, fz in (self.manifest.get("frozen_sets") or {}).items():
            cs = sets.get(name)
            if cs is None:
                probs.append("frozen set %s not loaded" % name)
                continue
            if cs.fingerprint() != fz.get("fingerprint"):
                probs.append("set %s fingerprint is %s, the pool froze %s (the full set changed: index statistics "
                             "would differ)" % (name, cs.fingerprint()[:12], str(fz.get("fingerprint"))[:12]))
        for e in self.manifest["docs"]:
            cs = sets.get(e["set"])
            if cs is None:
                continue
            if not cs.has_doc(e["doc_id"]):
                probs.append("document %s missing from %s" % (e["doc_id"], e["set"]))
                continue
            now = doc_entry(cs.doc(e["doc_id"]), e["set"])
            for k in ("fingerprint", "text_sha1"):
                if now[k] != e.get(k):
                    probs.append("document %s %s changed" % (e["doc_id"], k))
        return probs


def load_pool(path):
    path = Path(path)
    m = json.loads(path.read_text(encoding="utf-8"))
    if m.get("schema") != POOL_SCHEMA:
        raise PoolError("%s: schema %r, expected %s" % (path, m.get("schema"), POOL_SCHEMA))
    if m.get("manifest_sha1") != manifest_sha1(m):
        raise PoolError("%s: manifest_sha1 does not match its content (edited after freezing?)" % path)
    if str(m.get("derived_from")).lower() in GOLD_DERIVED:
        raise PoolError("%s: pool derived from gold evidence (%s)" % (path, m.get("derived_from")))
    return Pool(m, path)


def corpus_set_names(corpus):
    """The set names queries of a corpus use (contracts.make_query: set_name_for per document language)."""
    from .corpus.registry import set_name_for
    langs = ("fa",) if corpus == "C" else ("en",) if corpus == "D" else ("fa", "en")
    return tuple(sorted({set_name_for(corpus, l) for l in langs}))


def resolve_pool_path(arg, cfg):
    p = Path(arg)
    if p.is_file():
        return p
    cand = pool_root(cfg) / (arg if arg.endswith(".json") else arg + ".json")
    if cand.is_file():
        return cand
    raise PoolError("pool manifest %s not found (nor %s)" % (arg, cand))


def main(argv=None):
    from .config import load_config
    from .corpus.registry import load_set
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description="freeze or verify a pool manifest")
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("freeze")
    f.add_argument("--pool-id", required=True)
    f.add_argument("--corpus", required=True, choices=["A", "B", "C", "D", "K", "L"])
    f.add_argument("--docs", required=True, help="comma list of full doc ids")
    f.add_argument("--derived-from", required=True)
    f.add_argument("--selection", default="")
    f.add_argument("--description", default="")
    f.add_argument("--out", default=None)
    v = sub.add_parser("verify")
    v.add_argument("manifest")
    a = ap.parse_args(argv)
    cfg = load_config()
    if a.cmd == "freeze":
        sets = {n: load_set(n, cfg) for n in corpus_set_names(a.corpus)}
        m = build_manifest(a.pool_id, a.corpus, [x.strip() for x in a.docs.split(",") if x.strip()], sets,
                           a.derived_from, a.selection, a.description)
        out = write_manifest(a.out or pool_root(cfg) / ("%s.json" % a.pool_id), m)
        print("frozen %s: %d docs, sha1 %s -> %s" % (a.pool_id, len(m["docs"]), m["manifest_sha1"][:12], out))
        return 0
    pool = load_pool(resolve_pool_path(a.manifest, cfg))
    sets = {n: load_set(n, cfg) for n in (pool.manifest.get("frozen_sets") or {})}
    probs = pool.verify(sets)
    print("pool %s: %d docs; %s" % (pool.pool_id, len(pool.doc_ids()), "OK" if not probs else "; ".join(probs)))
    return 0 if not probs else 1


if __name__ == "__main__":
    raise SystemExit(main())
