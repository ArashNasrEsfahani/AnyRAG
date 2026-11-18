"""OURS index levels: cheap indexes next to the full per-section enrichment, compared by accuracy against index cost.

    get_approach("OURS", flags={"index_level": "L0"})      # variant "OURS+index_level=L0"; L3 (default) = "OURS"

  L0  no model at index time. Per section: @@TITLE <section title> and @@PATH <document title> > <section path>
      (title weight 3, path weight 0.75: the document title and the ancestors rank as context, never above a body
      hit), plus deterministic fact lines "metric | value | unit | period" extracted by lite.extract_facts from the
      section's own text and tables (FACTX_VERSION).
  L1  L0 + ONE bilingual glossary per document (lite_llm.GlossaryBuilder, one call + at most one repair). Each section
      header gets "gl_fa: ..." / "gl_en: ..." lines holding the glossary entries whose document-language term or synonym
      occurs in that section's title or own text (lite.project_glossary, at most l1_max_entries_per_section, best
      first; an entry found in more than l1_max_section_share of the sections is not projected). The glossary is projected at index time rather than shown to the planner: the query side (plan prompt,
      its tokens, the <= 3-call cap) stays identical at every level, and a whole glossary on every header of a
      document would give every section of a doc-scoped question the same header hit (no ranking signal); projected,
      an English pattern reaches exactly the Persian sections that use the term, like L3's kw_en lines.
  L2  L1 + short per-section keyword/synonym lists "kw_fa / kw_en / syn_fa / syn_en" from batched calls
      (lite_llm.KeywordBatcher: several sections per call, capped output, strict JSON, repair, per-section cache).
  L3  the existing enrichment (indexer.py), unchanged: same prompt, cache, store, signature and variant name "OURS".

Every level writes its own store (index_dir/store, swapped in atomically from a .partial directory) and has its own
index fingerprint (level_signature). The executor and planner are the same for every level.

Index cost, for the accuracy-vs-index-cost plot (index.json):
  enrichment.cost_all_cached_calls   the cache-inclusive model cost (what eval.aggregate.cached_index_cost reads)
  index_cost                         per stage (facts, digest, glossary, projection, keywords, store_write): wall time of
                                     this build, new calls / tokens / model wall, cache hits and misses, cache-inclusive
                                     calls / tokens / model wall, s/section and s/document; totals; the Ollama version
                                     of this build and of every cached model result used
Ledger: besides the model calls themselves, one line per stage per document (kind "stage", phase "index", purpose
"index_stage", ok true; no tokens, so call/token/failure sums are unchanged) is written as the stage finishes, so the
timing of interrupted attempts is kept too.
L1/L2 model results are cached in work/indexes/OURS/_levels/ (shared by every variant with the same level settings) and
committed per result, so interrupted builds resume losing at most the call in flight.
"""
import os
import shutil
import time
from collections import Counter
from pathlib import Path

from ...store import StoreBlock, StoreFile, write_store
from ...util import atomic_write_json, dir_size_bytes, now_iso, safe_name
from .indexer import fact_line, fact_value_ok, section_body, text_number_set
from .lite import FACTX_VERSION, L0_DEFAULTS, PROJECTION_VERSION, extract_facts, glossary_header_lines, project_glossary
from .lite_llm import (GLOSSARY_PROMPT_VERSION, KEYWORDS_PROMPT_VERSION, KW_KEYS, LLM_DEFAULTS, GlossaryBuilder,
                       KeywordBatcher)

LEVELS = ("L0", "L1", "L2", "L3")
DEFAULT_LEVEL = "L3"
PROJECTION_DEFAULTS = {"l1_max_entries_per_section": 8,
                       "l1_max_section_share": 0.4}     # an entry in more sections than this share is not projected
LEVEL_DEFAULTS = dict(L0_DEFAULTS, **LLM_DEFAULTS, **PROJECTION_DEFAULTS)
_ZERO = {"calls": 0.0, "prompt_tokens": 0.0, "completion_tokens": 0.0, "wall_ms": 0.0}


def normalize_level(v):
    """'L1' | 'l1' | '1' | 1 -> 'L1'; anything else raises ValueError."""
    s = str(v).strip().upper()
    if not s.startswith("L"):
        s = "L" + s
    if s not in LEVELS:
        raise ValueError("index_level must be one of %s (got %r)" % (LEVELS, v))
    return s


def level_options(opts):
    """LEVEL_DEFAULTS overridden by any level key present in the approach options."""
    o = dict(LEVEL_DEFAULTS)
    o.update({k: v for k, v in (opts or {}).items() if k in LEVEL_DEFAULTS})
    return o


def level_versions(level):
    level = normalize_level(level)
    v = {"index_level": level, "factx": FACTX_VERSION}
    if level in ("L1", "L2"):
        v.update(glossary=GLOSSARY_PROMPT_VERSION, projection=PROJECTION_VERSION)
    if level == "L2":
        v["keywords"] = KEYWORDS_PROMPT_VERSION
    return v


def level_signature(level, opts, *, facts=True, normalized=True):
    """config_signature of an L0-L2 variant (part of the index fingerprint). L3 keeps EnrichedGrep's own signature."""
    level = normalize_level(level)
    if level == DEFAULT_LEVEL:
        raise ValueError("L3 uses EnrichedGrep.config_signature")
    o = level_options(opts)
    sig = {"enrichment": True, "facts": bool(facts), "normalized": bool(normalized), "index_level": level,
           "versions": level_versions(level), "l0": {k: o[k] for k in sorted(L0_DEFAULTS)}}
    if level in ("L1", "L2"):
        sig["l1"] = {k: o[k] for k in sorted(LEVEL_DEFAULTS) if k.startswith("l1_")}
    if level == "L2":
        sig["l2"] = {k: o[k] for k in sorted(LEVEL_DEFAULTS) if k.startswith("l2_")}
    return sig


def cache_paths(cfg):
    model = safe_name(cfg.models["chat"])
    d = cfg.path("index_root") / "OURS" / "_levels"
    return {"glossary": d / ("%s-%s.sqlite" % (model, GLOSSARY_PROMPT_VERSION)),
            "keywords": d / ("%s-%s.sqlite" % (model, KEYWORDS_PROMPT_VERSION))}


# ---------------------------------------------------------------------------------------------------------------
# store
# ---------------------------------------------------------------------------------------------------------------

def level_store_file(doc, section, header_lines, facts_list, *, facts=True, normalized=True):
    """The L3 store layout (indexer.section_store_file) with the document title leading @@PATH and the level's header
    lines (gl_* / kw_* / syn_*) in the @@HEADER block."""
    title = " ".join(((section.title if normalized else section.title_raw) or "").split())
    doc_title = " ".join(((doc.title if normalized else doc.title_raw) or "").split())
    if normalized:
        path = " > ".join(t for t in doc.section_path(section.id) if t)
    else:
        chain, s = [], section
        while s is not None:
            chain.append(" ".join((s.title_raw or "").split()))
            s = doc.section(s.parent) if s.parent else None
        path = " > ".join(t for t in reversed(chain) if t)
    full_path = " > ".join(x for x in (doc_title, path) if x)
    blocks = [StoreBlock("@@SECTION doc=%s sec=%s level=%d" % (doc.id, section.id, section.level), "", "meta"),
              StoreBlock("@@TITLE %s" % title, "", "meta"),
              StoreBlock("@@PATH %s" % full_path, "", "meta")]
    if header_lines:
        blocks.append(StoreBlock("\n".join(["@@HEADER"] + list(header_lines) + ["@@END HEADER"]), "", "header"))
    if facts and facts_list:
        blocks.append(StoreBlock("@@FACTS", "", "fence"))
        for f in facts_list:
            blocks.append(StoreBlock(fact_line(f), f.get("unit_id") or "", "fact"))
        blocks.append(StoreBlock("@@END FACTS", "", "fence"))
    for uid, t in section_body(doc, section, raw=not normalized):
        blocks.append(StoreBlock(t, uid, "body"))
    return StoreFile(doc.id, section.id, "%s/%04d.txt" % (safe_name(doc.id), section.order), section.level, title, blocks)


def write_store_atomic(root, files, variant, extra_manifest=None, attempts=10):
    """store.write_store into <root>.partial, then swap it in: a build interrupted while writing never leaves a
    half-written store behind a manifest (the previous store, if any, stays until the swap)."""
    root = Path(root)
    partial = root.with_name(root.name + ".partial")
    old = root.with_name(root.name + ".old")
    shutil.rmtree(partial, ignore_errors=True)
    write_store(partial, files, variant=variant, extra_manifest=extra_manifest)
    for k in range(attempts):
        try:
            shutil.rmtree(old, ignore_errors=True)
            if root.exists():
                os.replace(root, old)
            os.replace(partial, root)
            break
        except OSError:
            if k == attempts - 1:
                raise
            time.sleep(0.5)
    shutil.rmtree(old, ignore_errors=True)
    return root


def keyword_header_lines(rec, max_chars=120):
    lines = []
    for key in ("kw_fa", "kw_en", "syn_fa", "syn_en"):
        vals = [" ".join(str(x).replace("|", "/").replace(";", ",").split())[:max_chars] for x in (rec.get(key) or [])]
        vals = [v for v in vals if v]
        if vals:
            lines.append("%s: %s" % (key, " ; ".join(vals)))
    return lines


# ---------------------------------------------------------------------------------------------------------------
# timing
# ---------------------------------------------------------------------------------------------------------------

def _add(acc, cost):
    for k in _ZERO:
        acc[k] = acc.get(k, 0.0) + float((cost or {}).get(k) or 0.0)


def ollama_version_of(client):
    """The Ollama server version for the build record (None for a client without health(), or when unreachable)."""
    h = getattr(client, "health", None)
    if not callable(h):
        return None
    try:
        r = h()
        return r.get("version") if r.get("reachable") else None
    except Exception:  # noqa: BLE001
        return None


class StageLog:
    """Per-stage wall time + counters, accumulated for index.json and written to the client's ledger as each stage
    of each document finishes (kind "stage": excluded from every call / token / failure sum)."""

    STAGES = ("facts", "digest", "glossary", "projection", "keywords", "store_write")

    def __init__(self, client, level, corpus_set):
        self.ledger = getattr(client, "ledger", None)
        self.level = level
        self.corpus_set = corpus_set.name
        self.acc = {s: {"wall_ms": 0.0, "docs": 0, "sections": 0} for s in self.STAGES}

    def add(self, stage, wall_ms, doc_id=None, **counts):
        a = self.acc[stage]
        a["wall_ms"] += wall_ms
        a["docs"] += 1 if doc_id else 0
        for k, v in counts.items():
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                a[k] = a.get(k, 0) + v
        if self.ledger is not None:
            try:
                rec = dict(self.ledger.context(), kind="stage", phase="index", purpose="index_stage", stage=stage,
                           index_level=self.level, corpus_set=self.corpus_set, doc_id=doc_id, ok=True,
                           wall_ms=round(wall_ms, 1), ts=now_iso(), **counts)
                rec.setdefault("approach", "OURS+index_level=%s" % self.level)
                self.ledger.record(rec)
            except Exception:  # noqa: BLE001  (timing must never fail a build)
                pass


def build_level_store(corpus_set, root, level, *, cfg, client, opts=None, facts=True, normalized=True, caches=None,
                      log=print, log_every=25):
    """Build the L0/L1/L2 store of a CorpusSet (resumable through the L1/L2 caches). -> (stats, detail)"""
    level = normalize_level(level)
    if level == DEFAULT_LEVEL:
        raise ValueError("L3 is built by indexer.build_enriched_store")
    o = level_options(opts)
    caches = caches or cache_paths(cfg)
    t_all = time.perf_counter()
    version = ollama_version_of(client)
    gb = GlossaryBuilder(cfg, client, caches["glossary"], o, ollama_version=version) if level in ("L1", "L2") else None
    kb = KeywordBatcher(cfg, client, caches["keywords"], o, ollama_version=version) if level == "L2" else None
    stages = StageLog(client, level, corpus_set)
    st = {"index_level": level, "versions": level_versions(level), "ollama_version": version, "started": now_iso(),
          "docs": 0, "sections": 0, "sections_with_body": 0, "fact_candidates": 0, "facts_kept": 0, "facts_dropped": 0,
          "sections_with_facts": 0, "fact_drop_reasons": {}}
    if gb:
        st["glossary"] = {"docs": 0, "ok": 0, "salvaged": 0, "repaired": 0, "partial": 0, "failed": 0, "terms": 0,
                          "terms_projected": 0, "terms_too_common": 0, "sections_with_glossary": 0, "entries_projected": 0}
    if kb:
        st["keywords"] = {"sections": 0, "ok": 0, "repaired": 0, "failed": 0, "too_short": 0, "sections_with_keywords": 0}
    cost = {"glossary": dict(_ZERO), "keywords": dict(_ZERO)}
    cached_versions = {"glossary": Counter(), "keywords": Counter()}
    files, detail_sections, detail_glossary = [], {}, {}
    total = sum(len(d.sections) for d in corpus_set.docs)
    for d in corpus_set.docs:
        st["docs"] += 1
        # ---- L0 facts (program)
        t0 = time.perf_counter()
        facts_by, n_facts = {}, 0
        for s in d.sections:
            units = [(uid, d.unit(uid).text) for uid in s.unit_ids if (d.unit(uid).text or "").strip()]
            if units:
                st["sections_with_body"] += 1
            fl, fst = extract_facts(units, s.title, o)
            st["fact_candidates"] += fst["candidates"]
            nums = text_number_set(s.title + "\n" + "\n".join(t for _, t in units))
            kept = []
            for f in fl:
                ok, why = fact_value_ok(f["value"], nums)
                if ok:
                    kept.append(f)
                else:
                    st["fact_drop_reasons"][why] = st["fact_drop_reasons"].get(why, 0) + 1
            for why in ("dropped_score", "dropped_cap", "dropped_empty", "duplicates"):
                if fst.get(why):
                    st["fact_drop_reasons"][why] = st["fact_drop_reasons"].get(why, 0) + fst[why]
            st["facts_dropped"] += len(fl) - len(kept) + fst["dropped_score"] + fst["dropped_cap"] + fst["dropped_empty"]
            st["facts_kept"] += len(kept)
            st["sections_with_facts"] += 1 if kept else 0
            facts_by[s.id] = kept
            n_facts += len(kept)
        stages.add("facts", (time.perf_counter() - t0) * 1000.0, d.id, sections=len(d.sections), facts=n_facts)
        # ---- L1 glossary (model, cached per document) + projection (program)
        gl_rec, proj = None, {}
        if gb:
            c0 = dict(gb.counters)
            dg0 = gb.digest_ms
            t0 = time.perf_counter()
            gl_rec = gb.build(d)
            wall = (time.perf_counter() - t0) * 1000.0
            dg = gb.digest_ms - dg0
            hit = gb.counters["cache_hits"] > c0["cache_hits"]
            stages.add("digest", dg, d.id, sections=len(d.sections))
            stages.add("glossary", max(0.0, wall - dg), d.id, calls=gb.counters["calls"] - c0["calls"],
                       prompt_tokens=gb.counters["new_prompt_tokens"] - c0["new_prompt_tokens"],
                       completion_tokens=gb.counters["new_completion_tokens"] - c0["new_completion_tokens"],
                       model_wall_ms=round(gb.counters["new_wall_ms"] - c0["new_wall_ms"], 1),
                       cache_hits=int(hit), cache_misses=int(not hit), terms=len(gl_rec["entries"]),
                       cached_calls=gl_rec["calls"], cached_model_wall_ms=gl_rec["wall_ms"])
            _add(cost["glossary"], gl_rec)
            cached_versions["glossary"][gl_rec.get("ollama_version") or "unknown"] += 1
            g = st["glossary"]
            g["docs"] += 1
            g[gl_rec["status"]] = g.get(gl_rec["status"], 0) + 1
            g["terms"] += len(gl_rec["entries"])
            t0 = time.perf_counter()
            common = []
            proj, hits = project_glossary(gl_rec["entries"], d, d.lang, int(o["l1_max_entries_per_section"]),
                                          max_share=float(o["l1_max_section_share"]), too_common=common)
            stages.add("projection", (time.perf_counter() - t0) * 1000.0, d.id, sections=len(d.sections),
                       entries_projected=sum(len(v) for v in proj.values()), terms_too_common=len(common))
            g["terms_projected"] += len(hits)
            g["terms_too_common"] += len(common)
            g["sections_with_glossary"] += len(proj)
            g["entries_projected"] += sum(len(v) for v in proj.values())
            detail_glossary[d.id] = {"status": gl_rec["status"], "calls": gl_rec["calls"], "digest": gl_rec.get("digest"),
                                     "ollama_version": gl_rec.get("ollama_version"), "entries": gl_rec["entries"],
                                     "sections_per_entry": {gl_rec["entries"][i]["fa"] + " = " + gl_rec["entries"][i]["en"]: n
                                                            for i, n in sorted(hits.items())},
                                     "too_common": [gl_rec["entries"][i]["fa"] + " = " + gl_rec["entries"][i]["en"]
                                                    for i in common],
                                     "unprojected": [e["fa"] + " = " + e["en"] for i, e in enumerate(gl_rec["entries"])
                                                     if i not in hits and i not in common]}
        # ---- L2 keywords (model, batched, cached per section)
        kw = {}
        if kb:
            c0 = dict(kb.counters)
            t0 = time.perf_counter()
            kw = kb.run(d, d.sections)
            wall = (time.perf_counter() - t0) * 1000.0
            doc_cost = dict(_ZERO)
            for rec in kw.values():
                _add(doc_cost, rec.get("cost"))
                if rec["status"] != "too_short":
                    cached_versions["keywords"][rec.get("ollama_version") or "unknown"] += 1
                k = st["keywords"]
                k["sections"] += 1
                k[rec["status"]] = k.get(rec["status"], 0) + 1
                k["sections_with_keywords"] += 1 if any(rec.get(x) for x in KW_KEYS) else 0
            _add(cost["keywords"], doc_cost)
            stages.add("keywords", wall, d.id, sections=len(d.sections), calls=kb.counters["calls"] - c0["calls"],
                       repairs=kb.counters["repairs"] - c0["repairs"],
                       prompt_tokens=kb.counters["new_prompt_tokens"] - c0["new_prompt_tokens"],
                       completion_tokens=kb.counters["new_completion_tokens"] - c0["new_completion_tokens"],
                       model_wall_ms=round(kb.counters["new_wall_ms"] - c0["new_wall_ms"], 1),
                       cache_hits=kb.counters["cache_hits"] - c0["cache_hits"],
                       cache_misses=kb.counters["cache_misses"] - c0["cache_misses"],
                       too_short=kb.counters["too_short"] - c0["too_short"],
                       cached_calls=round(doc_cost["calls"], 3), cached_model_wall_ms=round(doc_cost["wall_ms"], 1))
        # ---- store files (in memory; written once below)
        for s in d.sections:
            header = []
            if gl_rec is not None:
                header += glossary_header_lines(gl_rec["entries"], proj.get(s.id, []), int(o["field_max_chars"]))
            if s.id in kw:
                header += keyword_header_lines(kw[s.id], int(o["field_max_chars"]))
            files.append(level_store_file(d, s, header, facts_by[s.id], facts=facts, normalized=normalized))
            detail_sections[s.id] = {"facts": len(facts_by[s.id]), "glossary_entries": len(proj.get(s.id, [])),
                                     **({"keywords": kw[s.id]["status"]} if s.id in kw else {})}
            st["sections"] += 1
            if log and log_every and st["sections"] % log_every == 0:
                new = (gb.counters["calls"] if gb else 0) + (kb.counters["calls"] if kb else 0)
                log("[OURS %s index] %s: %d/%d sections, %d new calls" % (level, corpus_set.name, st["sections"],
                                                                          total, new))
    t0 = time.perf_counter()
    variant = "ours-enriched-%s" % level + ("" if normalized else "-raw") + ("" if facts else "-nofacts")
    write_store_atomic(root, files, variant, extra_manifest={
        "corpus_set": corpus_set.name, "corpus_fingerprint": corpus_set.fingerprint(), "normalized": normalized,
        "facts": facts, "index_level": level, "prompt_version": level_versions(level)})
    stages.add("store_write", (time.perf_counter() - t0) * 1000.0, None, sections=len(files),
               bytes=dir_size_bytes(root))
    denom = st["facts_kept"] + st["facts_dropped"]
    st["fact_drop_rate"] = round(st["facts_dropped"] / denom, 4) if denom else 0.0
    all_cost = dict(_ZERO)
    _add(all_cost, cost["glossary"])
    _add(all_cost, cost["keywords"])
    st["cost_all_cached_calls"] = {k: round(v, 1) for k, v in all_cost.items()}
    st["cost_by_stage"] = {k: {kk: round(vv, 1) for kk, vv in v.items()} for k, v in cost.items()}
    st["new_calls"] = (gb.counters["calls"] if gb else 0) + (kb.counters["calls"] if kb else 0)
    st["cache_hits"] = (gb.counters["cache_hits"] if gb else 0) + (kb.counters["cache_hits"] if kb else 0)
    st["cache_misses"] = (gb.counters["cache_misses"] if gb else 0) + (kb.counters["cache_misses"] if kb else 0)
    st["counters"] = {"glossary": dict(gb.counters) if gb else None, "keywords": dict(kb.counters) if kb else None}
    st["stages"] = {k: {kk: (round(vv, 1) if isinstance(vv, float) else vv) for kk, vv in v.items()}
                    for k, v in stages.acc.items()}
    st["cached_result_ollama_versions"] = {k: dict(v) for k, v in cached_versions.items() if v}
    st["program_ms"] = {k: round(stages.acc[k]["wall_ms"], 1) for k in ("facts", "digest", "projection", "store_write")}
    st["enrich_ms"] = round((time.perf_counter() - t_all) * 1000.0, 1)
    st["finished"] = now_iso()
    return st, {"sections": detail_sections, "glossary": detail_glossary}


def _per(v, n, scale=1000.0, nd=4):
    return round(v / scale / n, nd) if n else None


def index_cost(st):
    """The level's index cost for the accuracy-vs-cost plot. Model cost is cache-inclusive (every cached glossary /
    keyword result counts with the calls that produced it); wall_ms of a stage is this build's elapsed time."""
    n = st["sections"]
    nd = st["docs"]
    c = st["cost_all_cached_calls"]
    stg = st["stages"]
    prog = sum(stg[k]["wall_ms"] for k in ("facts", "digest", "projection", "store_write"))
    by_stage = {}
    for name in StageLog.STAGES:
        a = dict(stg[name])
        if name in ("glossary", "keywords"):
            if not a.get("docs"):
                continue
            cc = st["cost_by_stage"][name]
            a.update(cached_calls=cc["calls"], cached_prompt_tokens=cc["prompt_tokens"],
                     cached_completion_tokens=cc["completion_tokens"], cached_model_wall_ms=cc["wall_ms"],
                     s_per_section_cached_model=_per(cc["wall_ms"], n), s_per_document_cached_model=_per(cc["wall_ms"], nd))
        elif name in ("digest", "projection") and not a.get("docs"):
            continue
        a.update(s_per_section=_per(a["wall_ms"], n, nd=5), s_per_document=_per(a["wall_ms"], nd, nd=5))
        by_stage[name] = a
    return {"index_level": st["index_level"], "ollama_version": st["ollama_version"],
            "cached_result_ollama_versions": st["cached_result_ollama_versions"],
            "docs": nd, "sections": n, "sections_with_body": st["sections_with_body"],
            "llm_calls": round(c["calls"], 1), "prompt_tokens": round(c["prompt_tokens"]),
            "completion_tokens": round(c["completion_tokens"]), "llm_wall_ms": round(c["wall_ms"], 1),
            "new_llm_calls": st["new_calls"], "cache_hits": st["cache_hits"], "cache_misses": st["cache_misses"],
            "program_ms": round(prog, 1), "build_wall_ms": st["enrich_ms"],
            "calls_per_section": round(c["calls"] / n, 4) if n else None,
            "s_per_section_model": _per(c["wall_ms"], n), "s_per_document_model": _per(c["wall_ms"], nd),
            "s_per_section_program": _per(prog, n, nd=5), "s_per_document_program": _per(prog, nd, nd=5),
            "s_per_section": _per(c["wall_ms"] + prog, n), "s_per_document": _per(c["wall_ms"] + prog, nd),
            "by_stage": by_stage,
            "note": "model cost (llm_*, s_per_*_model, cached_*) is cache-inclusive; wall_ms / program_ms / build_wall_ms "
                    "are this build's elapsed times (a build from a warm cache is fast, its model cost is not zero)"}


def build_level_index(approach, corpus_set, index_dir, log=print):
    """EnrichedGrep._build_index for index_level L0-L2 -> index payload (also written to index_dir/index.json)."""
    index_dir = Path(index_dir)
    t0 = time.perf_counter()
    custom = getattr(approach, "level_cache_paths", None)          # tests: throwaway caches
    caches = custom() if callable(custom) else cache_paths(approach.cfg)
    root = index_dir / "store"
    st, detail = build_level_store(corpus_set, root, approach.index_level, cfg=approach.cfg, client=approach.client,
                                   opts=approach.opts, facts=approach.facts, normalized=approach.normalized,
                                   caches=caches, log=log)
    atomic_write_json(index_dir / "enrichment_sections.json", detail["sections"])
    if detail["glossary"]:
        atomic_write_json(index_dir / "glossary.json", detail["glossary"], indent=1)
    meta = {"store": "enriched", "store_root": str(root), "normalized": approach.normalized, "facts": approach.facts,
            "index_level": approach.index_level, "prompt_version": level_versions(approach.index_level),
            "enrichment": st, "index_cost": index_cost(st),
            "cache": {k: str(v) for k, v in caches.items()} if approach.index_level != "L0" else None,
            "corpus_fingerprint": corpus_set.fingerprint()}
    meta["build_ms"] = round((time.perf_counter() - t0) * 1000.0, 1)
    atomic_write_json(index_dir / "index.json", meta, indent=1)
    return meta
