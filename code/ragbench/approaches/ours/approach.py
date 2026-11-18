"""OURS: enriched section index + planned grep (EnrichedGrep).

Query time (at most caps.ours_max_calls = 3 local-model calls):
  call 1  planner: question -> {"<q_lang>": [patterns], "<other>": [patterns], scope, expect}
          language check (code): >= 1 compiling pattern in the document language and in the question
          language (monolingual_patterns: question language only)
  call 2  EITHER the repair call, when the language check failed (the model sees its reply and what is
          missing; the missing languages' patterns are taken from the repair)
          OR, only when no repair was used and nothing matched: the widened plan (shorter stems,
          synonyms), executor again (still nothing: one more grep on the question's own content
          words, no model call)
          executor (no LLM): ripgrep over the store, rank sections, choose granularity
  last    the shared answer call (Approach.generate_answer over pack_evidence)
trace["stats"] records plan_repair / plan_missing_lang(_after_repair) / widen / question_words_fallback /
pattern_fallback per question (0/1); run.py sums them per invocation, aggregate reports their rates.
No BM25 and no embeddings anywhere.

Flags (ragbench.approaches.OURS_FLAGS):
  no_enrichment         grep the plain section store (store.build_plain_store, the one B5 uses); same planner
  no_facts              enriched store without fact lines; executor never picks fact granularity
  no_normalization      store body/titles from raw text, enrichment strings and patterns not normalised,
                        no joiner flex; question passed to the planner as written
  monolingual_patterns  planner writes (and validation keeps) patterns in the question language only
  granularity           always_section | always_window (default: auto)
  facts_bare            a fact item shows the matched fact lines (each with <= 2 verbatim source lines), the
                        dev_fix1 behaviour; default (DEFAULTS fact_evidence="context"): the source text holding each
                        matched value (table header + row, short unit, or sentence). Ranking is the same either way.
  index_level           L0 | L1 | L2 | L3 (default L3 = the enrichment below, variant "OURS"; L3 is dropped from the
                        flags so its variant name, signature and index directory are unchanged). L0: no model at index
                        time (document title + section path headers, regex fact lines); L1: L0 + one bilingual glossary
                        per document projected onto the section headers; L2: L1 + batched per-section keywords. See
                        levels.py; the level modules are imported only when a level other than L3 is used. The query
                        side (planner, executor, call cap) is the same at every level.
"""
import time
from pathlib import Path

from ...contracts import AnswerResult, Approach
from ...store import build_plain_store
from ...util import atomic_write_json, read_json, safe_name
from .defaults import DEFAULTS, FACT_EVIDENCE, GRANULARITIES, PLAN_PROMPT_VERSION, PROMPT_VERSION
from . import executor as ex
from .indexer import Enricher, build_enriched_store
from .planner import Planner, fallback_patterns

KNOWN_FLAGS = ("no_enrichment", "no_facts", "no_normalization", "monolingual_patterns", "granularity", "facts_bare",
               "index_level")
INDEX_LEVELS = ("L0", "L1", "L2", "L3")
DEFAULT_INDEX_LEVEL = "L3"


def parse_flags(flags):
    """Accepts a dict or a CLI-style string "no_facts,granularity=always_window"."""
    if flags is None:
        return {}
    if isinstance(flags, str):
        out = {}
        for part in [p.strip() for p in flags.replace("+", ",").split(",") if p.strip()]:
            if "=" in part:
                k, v = part.split("=", 1)
                out[k.strip()] = v.strip()
            else:
                out[part] = True
        flags = out
    flags = dict(flags)
    for k in flags:
        if k not in KNOWN_FLAGS:
            raise ValueError("unknown OURS flag %r; known: %s" % (k, ", ".join(KNOWN_FLAGS)))
    g = flags.get("granularity")
    if g in (None, False, "auto"):
        flags.pop("granularity", None)
    elif g not in GRANULARITIES:
        raise ValueError("granularity must be one of %s" % (GRANULARITIES,))
    if "index_level" in flags:
        lv = flags["index_level"]
        if lv in (None, False, True):
            raise ValueError("index_level takes a value: one of %s" % (INDEX_LEVELS,))
        lv = str(lv).strip().upper()
        lv = lv if lv.startswith("L") else "L" + lv
        if lv not in INDEX_LEVELS:
            raise ValueError("index_level must be one of %s" % (INDEX_LEVELS,))
        if lv == DEFAULT_INDEX_LEVEL:
            flags.pop("index_level")             # the full enrichment keeps the variant name "OURS"
        else:
            flags["index_level"] = lv
    for k in ("no_enrichment", "no_facts", "no_normalization", "monolingual_patterns", "facts_bare"):
        if k in flags:
            v = flags[k]
            v = v if isinstance(v, bool) else str(v).lower() in ("1", "true", "yes", "on")
            if v:
                flags[k] = True
            else:
                flags.pop(k)
    return flags


class EnrichedGrep(Approach):
    name = "OURS"
    needs_gold = False

    def __init__(self, cfg=None, client=None, flags=None, options=None):
        super().__init__(cfg=cfg, client=client, flags=parse_flags(flags))
        self.opts = dict(DEFAULTS, **(options or {}))
        f = self.flags
        self.enrichment = not f.get("no_enrichment")
        self.facts = self.enrichment and not f.get("no_facts")
        self.normalized = not f.get("no_normalization")
        self.monolingual = bool(f.get("monolingual_patterns"))
        self.granularity = f.get("granularity") or "auto"
        self.index_level = f.get("index_level") or DEFAULT_INDEX_LEVEL
        if self.index_level != DEFAULT_INDEX_LEVEL and not self.enrichment:
            raise ValueError("index_level %s builds an enriched store; it cannot be combined with no_enrichment"
                             % self.index_level)
        self.fact_evidence = "bare" if f.get("facts_bare") else self.opts["fact_evidence"]
        if self.fact_evidence not in FACT_EVIDENCE:
            raise ValueError("fact_evidence must be one of %s" % (FACT_EVIDENCE,))
        self.planner = Planner(self.cfg, self.client, monolingual=self.monolingual, normalized=self.normalized,
                               options=self.opts)

    # ------------------------------------------------------------------ identity
    def config_signature(self):
        if self.index_level != DEFAULT_INDEX_LEVEL:
            from .levels import level_signature
            return level_signature(self.index_level, self.opts, facts=self.facts, normalized=self.normalized)
        o = self.opts
        sig = {"enrichment": self.enrichment, "facts": self.facts, "normalized": self.normalized}
        if self.enrichment:
            sig.update({"prompt": PROMPT_VERSION, "raw_enrichment": bool(o["no_normalization_raw_enrichment"]) and not self.normalized,
                        **{k: o[k] for k in ("enrich_input_chars", "enrich_max_parts", "enrich_min_chars",
                                             "enrich_num_predict", "enrich_attempts", "max_keywords", "max_synonyms",
                                             "min_questions", "max_questions", "max_facts_per_part", "field_max_chars")}})
        return sig

    def enrichment_cache_path(self):
        model = safe_name(self.cfg.models["chat"])
        return self.cfg.path("index_root") / "OURS" / "_enrichment" / ("%s-%s.sqlite" % (model, PROMPT_VERSION))

    # ------------------------------------------------------------------ index
    def _build_index(self, corpus_set, index_dir):
        if self.enrichment and self.index_level != DEFAULT_INDEX_LEVEL:
            from .levels import build_level_index
            return build_level_index(self, corpus_set, index_dir)
        index_dir = Path(index_dir)
        t0 = time.perf_counter()
        if not self.enrichment:
            # the same store.build_plain_store layout B5 greps, written under this index dir so a
            # subset or concurrent rebuild never touches the shared work/stores/<set> copy
            root = build_plain_store(corpus_set, root=index_dir / "plain_store", normalized=self.normalized, cfg=self.cfg)
            meta = {"store": "plain", "store_root": str(root), "normalized": self.normalized,
                    "corpus_fingerprint": corpus_set.fingerprint()}
        else:
            enricher = Enricher(self.cfg, self.client, self.enrichment_cache_path(), self.opts)
            root = index_dir / "store"
            stats, detail = build_enriched_store(corpus_set, root, enricher, facts=self.facts, normalized=self.normalized,
                                                 raw_enrichment=bool(self.opts["no_normalization_raw_enrichment"]))
            atomic_write_json(index_dir / "enrichment_sections.json", detail)
            meta = {"store": "enriched", "store_root": str(root), "normalized": self.normalized, "facts": self.facts,
                    "prompt_version": PROMPT_VERSION, "enrichment": stats, "cache": str(self.enrichment_cache_path()),
                    "corpus_fingerprint": corpus_set.fingerprint()}
        meta["build_ms"] = round((time.perf_counter() - t0) * 1000.0, 1)
        atomic_write_json(index_dir / "index.json", meta, indent=1)
        return meta

    def _load_index(self, corpus_set, index_dir):
        meta = read_json(Path(index_dir) / "index.json")
        if meta and (Path(meta["store_root"]) / "manifest.json").is_file() \
                and meta.get("corpus_fingerprint") == corpus_set.fingerprint():
            return meta
        # store missing (deleted) or stale: rebuild; enrichment comes from the cache, new calls are ledgered
        ledger = self.client.ledger
        with ledger.scope(phase="index", approach=self.variant, corpus_set=corpus_set.name):
            return self._build_index(corpus_set, index_dir)

    # ------------------------------------------------------------------ query
    def _answer(self, query, index):
        cfg = self.cfg
        max_calls = int(cfg.caps["ours_max_calls"])
        budget = int(cfg.evidence["max_chars"])
        min_item = int(cfg.evidence["min_item_chars"])
        root = index.payload["store_root"]
        doc_ids = None if query.doc_scope == "corpus" else [d for d in query.doc_scope if index.corpus.has_doc(d)]
        calls = 0
        trace = {"variant": self.variant, "store": index.payload.get("store"), "granularity": self.granularity,
                 "plan_prompt": PLAN_PROMPT_VERSION, "fact_evidence": self.fact_evidence if self.facts else None}
        prog_ms = 0.0

        stats = {"plan_repair": 0, "plan_missing_lang": 0, "plan_missing_lang_after_repair": 0, "widen": 0,
                 "question_words_fallback": 0, "pattern_fallback": 0,
                 # evidence composition (counts, not 0/1): fact items offered; context mode: matched fact lines shown
                 # as their source text / as the bare fact line (no unit holds the value)
                 "fact_items": 0, "fact_lines_sourced": 0, "fact_lines_fallback": 0}
        trace["stats"] = stats
        plan = self.planner.plan(query)
        calls += 1
        trace["plan"] = _plan_trace(plan)
        repair_used = False
        if plan["check"]["missing"]:
            stats["plan_missing_lang"] = 1
            # the one repair call (counts toward the cap; a repaired question gets no widening call)
            if calls < max_calls - 1:
                plan = self.planner.repair(query, plan)
                calls += 1
                repair_used = True
                stats["plan_repair"] = 1
                trace["repair"] = dict(plan["repair"], patterns=plan["patterns"], check=plan["check"],
                                       dropped=plan["validation"]["dropped"])
            stats["plan_missing_lang_after_repair"] = int(bool(plan["check"]["missing"]))
        plan = self.planner.finalize(query, plan)
        stats["pattern_fallback"] = int(bool(plan["validation"].get("fallback_added")))
        trace["plan"]["final_patterns"] = plan["patterns"]
        trace["plan"]["fallback_added"] = plan["validation"].get("fallback_added")
        t0 = time.perf_counter()
        ranked, info, view = ex.search(root, plan, doc_ids, index.corpus, cfg, self.opts)
        prog_ms += (time.perf_counter() - t0) * 1000.0
        trace["search"] = info
        used_plan = plan
        trace["evidence_from"] = "plan"
        if not ranked and not repair_used and calls < max_calls - 1:
            wplan = self.planner.widen(query, plan)
            calls += 1
            stats["widen"] = 1
            trace["widen"] = _plan_trace(wplan)
            t0 = time.perf_counter()
            ranked, info2, view = ex.search(root, wplan, doc_ids, index.corpus, cfg, self.opts)
            prog_ms += (time.perf_counter() - t0) * 1000.0
            trace["search_widened"] = info2
            used_plan = wplan
            trace["evidence_from"] = "widen"
        if not ranked and self.opts["question_words_fallback"]:
            # last resort without a model call: grep the question's own content words
            words = fallback_patterns(self.planner._question(query), query.q_lang, self.planner.pmax,
                                      normalized=self.normalized, flex=self.opts["flex_joiners"])
            if words:
                fplan = {"patterns": words, "scope": "all", "expect": used_plan.get("expect") or "text"}
                t0 = time.perf_counter()
                ranked, info3, view = ex.search(root, fplan, doc_ids, index.corpus, cfg, self.opts)
                prog_ms += (time.perf_counter() - t0) * 1000.0
                trace["search_question_words"] = info3
                used_plan = fplan
                trace["evidence_from"] = "question_words"
                stats["question_words_fallback"] = 1
        trace["used_plan"] = {k: used_plan.get(k) for k in ("patterns", "scope", "expect")}
        trace["enrich_prompt"] = PROMPT_VERSION if self.enrichment else None
        if self.index_level != DEFAULT_INDEX_LEVEL:
            from .levels import level_versions
            trace["index_level"] = self.index_level
            trace["enrich_prompt"] = level_versions(self.index_level)
        t0 = time.perf_counter()
        items, decisions, retrieved, ranked_x = ex.select_evidence(
            ranked, view, index.corpus, expect=used_plan["expect"], granularity=self.granularity,
            facts_enabled=self.facts, budget=budget, min_item_chars=min_item, opts=self.opts, normalized=self.normalized,
            patterns=used_plan["patterns"], fact_evidence=self.fact_evidence)
        for it in items:
            if it.kind == "fact":
                stats["fact_items"] += 1
                stats["fact_lines_sourced"] += int(it.meta.get("fact_sourced") or 0)
                stats["fact_lines_fallback"] += int(it.meta.get("fact_fallback") or 0)
        chosen = {it.section_id for it in items}
        trace["sections_matched"] = len(ranked)
        trace["ranking"] = [{"rank": i, "section_id": sh.section_id, "score": round(sh.score, 3),
                             "scope_score": round(sh.scope_score, 3),
                             "matched": {str(pi): k for pi, k in sorted(sh.best.items())},
                             "via_parent": sh.via_parent, "selected": sh.section_id in chosen}
                            for i, sh in enumerate(ranked_x[: int(self.opts["ranking_trace_max"])], 1)]
        trace["selection"] = decisions
        prog_ms += (time.perf_counter() - t0) * 1000.0
        trace["program_ms"] = round(prog_ms, 1)        # grep + ranking + selection, model calls excluded

        evidence = self.pack(items, index)
        trace["evidence"] = {"n_offered": evidence.n_offered, "n_included": evidence.n_included,
                             "chars": evidence.chars, "last_item_cut": evidence.last_item_cut}
        if not items and not self.opts["answer_when_empty"]:
            trace["calls"] = calls
            return AnswerResult(answer_text=cfg.evidence["abstain_token"], evidence=evidence, retrieved=[], trace=trace)
        if calls >= max_calls:
            raise RuntimeError("OURS call cap reached before the answer call")
        answer = self.generate_answer(query, evidence)
        calls += 1
        trace["calls"] = calls
        return AnswerResult(answer_text=answer, evidence=evidence, retrieved=retrieved, trace=trace)


def _plan_trace(plan):
    return {"patterns": plan["patterns"], "scope": plan["scope"], "expect": plan["expect"],
            "parse_ok": plan["parse_ok"], "model_output": plan["model_output"], "done_reason": plan["done_reason"],
            "lists": plan.get("lists"), "check": plan.get("check"), "moved": plan["validation"].get("moved"),
            "dropped": plan["validation"]["dropped"], "fallback_added": plan["validation"].get("fallback_added"),
            "langs": plan["validation"].get("langs"), "truncated": plan["validation"].get("truncated"),
            **({"widen": plan["validation"]["widen"]} if "widen" in plan["validation"] else {}),
            **({"scope_model": plan["scope_model"]} if "scope_model" in plan else {})}
