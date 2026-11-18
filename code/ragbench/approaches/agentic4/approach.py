"""AGENTIC4: AGENTIC3 with a graph retrieval path over typed facts. At most 4 model calls and 60 s per question.

    1 plan          (1 call)  v3's schema-constrained plan (kind single | timeline | comparison IS the hop template)
    2 retrieve      (no LLM)  v3's database pool (raw-label regex, sections, units, tree)
                              + the GRAPH pool (graphret.py): card routing when the scope is broad, facts found by
                              their metric / entity NAME in every spelling, each target report's members of the seed
                              families (the soft cross-year hop), qualifier match, hub penalty
    3 route         (no LLM)  v3's
    4 vectors                 v3's (always merged)
    5 merge                   v3's selection, unchanged; a window the graph also found carries its node and
                              qualifiers. `graph_min_slots_multi` new graph windows may be added (0 by default: on dev
                              they took first-round verification seats from v3's evidence - work/analysis/V4_DEV.md)
    6 verify        (1-2)     v3's batched verifier and guard; the first round is v3's
    7 reconcile + gap         v3's SQL reconciliation, then v3's gap round - WITH THE GRAPH: for each year still
                              missing, the graph hops from the verified rows' own wording to that year's report
                              (family and concept walk) and offers `graph_gap_per_year` windows, after v3's own gap
                              candidates, in the call v3 holds back for this round. A verified row the verifier left
                              'unspecified' takes the qualifier of its graph node (active, registered, total, ...)
    8 answer        (1 call)  the SHARED prompt and the shared 6000-char budget

Everything v3 does is inherited, not copied: this module only adds the graph pool, the graph-aware merge, the graph's
gap candidates (through v3's `gap_batch` hook) and the qualifier annotation. `no_graph` turns all of it off and is
AGENTIC3 exactly.
"""
import json
import os
import time
from collections import defaultdict
from contextlib import contextmanager

from ...contracts import Approach
from ...kb import graph as G
from ...kb.concepts import CONCEPTS_VERSION, NameVectors
from ...kb import labels as L
from ...util import atomic_write_json, read_json
from ..agentic2.lazyvec import LazyVectors
from ..agentic3 import retrieve as RT
from ..agentic3.approach import TypedKBAgent, _Session3
from . import graphret as GR
from .defaults import APPROACH_VERSION, DEFAULTS, FLAGS, GRAPH_DEF_CLASSES, VALUE_FLAGS

OPTIONS_ENV = "AGENTIC4_OPTIONS"     # DEV tuning only; a --final run refuses it (run.py TUNING_ENV)


def parse_flags(flags):
    if flags is None:
        return {}
    if isinstance(flags, str):
        d = {}
        for p in flags.replace("+", ",").split(","):
            p = p.strip()
            if p:
                k, _, v = p.partition("=")
                d[k.strip()] = v.strip() if v else True
        flags = d
    out = {}
    for k, v in dict(flags).items():
        if k not in FLAGS:
            raise ValueError("unknown AGENTIC4 flag %r; known: %s" % (k, ", ".join(FLAGS)))
        if k in VALUE_FLAGS:
            n = int(v)
            if n < 1:
                raise ValueError("AGENTIC4 flag %s must be >= 1" % k)
            out[k] = n
        elif v if isinstance(v, bool) else str(v).lower() in ("1", "true", "yes", "on"):
            out[k] = True
    return out


class GraphAgent(TypedKBAgent):
    name = "AGENTIC4"
    needs_gold = False

    def __init__(self, cfg=None, client=None, flags=None, options=None):
        # v3's constructor validates flags against v3's list and reads v3's options and caps, so the shared parts are
        # set here with v4's own; everything the session does is inherited
        Approach.__init__(self, cfg=cfg, client=client, flags=parse_flags(flags))
        env = os.environ.get(OPTIONS_ENV)
        self.env_options = json.loads(env) if env else None
        self.opts = dict(DEFAULTS, **(self.env_options or {}), **(options or {}))
        f, o = self.flags, self.opts
        if f.get("candidates"):
            o["max_candidates"] = o["max_candidates_multi"] = int(f["candidates"])
        if f.get("no_facets"):
            o["facets"] = False
        self.use_graph = not f.get("no_graph")
        if not self.use_graph:               # AGENTIC3's merge budget, exactly
            from ..agentic3.defaults import DEFAULTS as V3
            o["vec_min_slots"] = V3["vec_min_slots"]
        self.use_family = not f.get("no_family")
        self.use_cards = not f.get("no_cards")
        if f.get("no_graph_quals"):
            o["graph_quals_to_def_class"] = False
        self.use_concepts = self.use_graph and not f.get("no_concepts") and bool(o["graph_concepts"])
        self.warm = False
        self.use_trigram = True
        self.use_vectors = not f.get("no_vectors")
        self.use_verify = True
        self.use_tree = True
        self.use_factdb = True
        self.line_windows = False
        self.doc_selection = True
        caps = self.cfg.caps or {}
        self.max_calls = int(caps.get("agentic4_max_calls") or o["max_calls_default"])
        self.max_seconds = float(caps.get("agentic4_max_seconds") or o["max_seconds_default"])
        self._cache = None
        self._cache_for = None
        self._con = None
        self.lazy = LazyVectors(self)
        self.names = NameVectors(self.lazy)

    def config_signature(self):
        sig = super().config_signature()
        sig.update(approach=APPROACH_VERSION, labels=L.LABELS_VERSION, graph=G.GRAPH_VERSION,
                   graphret=GR.GRAPH_RET_VERSION, concepts=CONCEPTS_VERSION)
        return sig

    def _build_index(self, corpus_set, index_dir):
        """v3's ingestion, then the label layer and the graph (one whole-corpus pass, no model)."""
        t0 = time.perf_counter()
        payload = super()._build_index(corpus_set, index_dir)
        g = G.ensure_current(self.kb()) or G.summary(self.kb())
        payload.update(graph=g, graph_ms=round((time.perf_counter() - t0) * 1000.0 - payload["build_ms"], 1))
        atomic_write_json(index_dir / "kb_index.json", payload, indent=1)
        return payload

    def _load_index(self, corpus_set, index_dir):
        payload = super()._load_index(corpus_set, index_dir)
        G.ensure_current(self.kb())
        return payload

    def _answer(self, query, index):
        return _Session4(self, query, index).run()


class _Session4(_Session3):
    def __init__(self, ap, query, index):
        super().__init__(ap, query, index)
        self._stage_stack = []
        self.stats.update({"graph_candidates": 0, "graph_seeds": 0, "graph_walked": 0, "graph_slots": 0,
                           "graph_in_selection": 0, "graph_ms": 0.0, "graph_quals_applied": 0,
                           "cards_routed": 0, "concept_seeds": 0, "concept_walked": 0, "names_embedded": 0,
                           "graph_gap_candidates": 0})

    @contextmanager
    def stage(self, name):
        """v1's stage record, plus the name of the stage it runs INSIDE (`parent`), so the clock can take a nested
        lazy-index out of its parent too. v1-v3 skip a lazy-index record but still count its time inside the stage
        around it; v3 nests its passage embedding inside `vectors`, so its clock includes first-touch embedding."""
        parent = self._stage_stack[-1] if self._stage_stack else None
        self._stage_stack.append(name)
        try:
            with super().stage(name) as rec:
                rec["parent"] = parent
                yield rec
        finally:
            self._stage_stack.pop()

    def clock_s(self):
        """The query clock with EVERY lazy-index excluded, nested ones included (see `stage`)."""
        nested = {}
        for st in self.timing:
            if st["stage"] == "lazy-index" and st.get("parent"):
                nested[st["parent"]] = nested.get(st["parent"], 0.0) + st["wall_ms"]
        total = 0.0
        for st in self.timing:
            if st["stage"] == "lazy-index":
                continue
            calls_wall = sum(c["wall_ms"] for c in self.calls if c.get("stage") == st["stage"])
            total += st["wall_ms"] - st.get("call_elapsed_ms", 0.0) - nested.pop(st["stage"], 0.0) + calls_wall
        return total / 1000.0

    def concepts(self, plan, doc_ids):
        """Name vectors of the target reports (first touch embeds them: a lazy-index stage, off the clock) and the
        vectors of the plan's terms (one query-embed call, on the clock). -> (mats, qvecs) or None."""
        if not (self.a.use_concepts and doc_ids and (len(doc_ids) > 1 or self.o["graph_single_doc"])):
            return None
        with self.stage("lazy-index") as st:
            mats, info = self.a.names.ensure(self.con, doc_ids, self.a.client)
            self.stats["names_embedded"] += info["embedded_now"]
            st.update(kind="names", docs=info["docs"], names=info["names"], embedded=info["embedded_now"],
                      embed_calls=info["embed_calls"], embed_ms=info["embed_ms"])
        texts = GR.concept_texts(plan, self.q.d_lang)
        if not texts:
            return None
        qvecs, qinfo = self.a.lazy.query_vectors(texts, self.a.client, purpose="concept-query-embed")
        self.stats["query_embed_calls"] += qinfo["embed_calls"]
        return mats, qvecs

    # -- the graph in the GAP round (graphret-v4) ---------------------------------------------------------------
    def gap_fill(self, rec, plan, target_periods, scope_ids, rows, meta_by, cands, fallback=None):
        self._gap_ctx = (rec, plan, cands)                    # read by gap_batch, which runs right after
        return super().gap_fill(rec, plan, target_periods, scope_ids, rows, meta_by, cands, fallback=fallback)

    def gap_batch(self, gcands, skipped, gap):
        """v3's batch for the call held back for the gap round, plus the graph's candidates for the years STILL
        missing after the first round. v3's own gap candidates keep their seats and the graph fills the room left
        in the batch (`verify_batch_max`); when v3's gap search found nothing, the graph's candidates go first and
        the skipped first-round candidates fill the rest. Nothing the first round verified is touched."""
        extra, new = super().gap_batch(gcands, skipped, gap)
        ctx = getattr(self, "_gap_ctx", None)
        if not (self.a.use_graph and self.o["graph_gap"] and ctx):
            return extra, new
        g, ginfo = self.graph_gap(*ctx, taken=list(ctx[2]) + list(gcands))
        gap["graph"] = ginfo
        if not g:
            return extra, new
        n = int(self.o["verify_batch_max"])
        if gcands:
            g = g[: max(0, n - len(gcands))]
            both = list(gcands) + g
            return both, both
        return g + list(skipped[: max(0, n - len(g))]), g

    def graph_gap(self, rec, plan, cands, taken=()):
        """The graph's hop, from what the first round VERIFIED to the reports of the years still missing.

        Seeds are the verified rows' own wording (their label and a short definition - the words the other years'
        reports actually use; v3's gap search draws its regex from the same rows), then the plan's terms. The graph
        path runs over all target reports so the family and concept walks can start from the verified years' facts;
        only candidates in a missing year's report, in a window no earlier candidate holds, are kept, at most
        `graph_gap_per_year` per report. -> (candidates, info)."""
        t0 = time.perf_counter()
        missing = sorted({int(y) for y in (rec.get("periods_missing") or []) if str(y).lstrip("-").isdigit()})
        docs = list(getattr(self, "_targets", None) or [])
        info = {"missing": missing, "docs": [], "terms": [], "candidates": 0}
        if not missing or len(docs) < 2:
            return [], info
        years = GR.doc_years(self.con, docs)
        miss_docs = [d for d in docs if d in years and any(years[d][0] <= y <= years[d][1] for y in missing)]
        info["docs"] = miss_docs
        if not miss_docs:
            return [], info
        texts = []
        for r in rec.get("series") or []:
            for t in (r.get("label"), r.get("definition")):
                t = (t or "").strip()
                if t and len(t.split()) <= int(self.o["graph_gap_max_words"]) and t not in texts:
                    texts.append(t)
        d_lang = self.q.d_lang
        terms = list(dict.fromkeys(texts[:4] + GR.plan_terms(plan, d_lang)))
        plan2 = dict(plan, terms={d_lang: terms}, metric=texts[0] if texts else plan.get("metric"))
        info["terms"] = terms
        concepts = self.concepts(plan2, docs)
        g, ginfo = GR.graph_candidates(self.con, self.reader, plan2, docs, d_lang=d_lang, opts=self.o,
                                       tree=self.a.use_tree, family=self.a.use_family, cards=False, concepts=concepts)
        per = defaultdict(list)
        for c in g:
            if c.doc_id in miss_docs and not any(GR._overlaps(c, s) for s in taken) \
                    and len(per[c.doc_id]) < int(self.o["graph_gap_per_year"]) \
                    and not any(GR._overlaps(c, x) for x in per[c.doc_id]):
                per[c.doc_id].append(c)
        out = [c for d in miss_docs for c in per[d]]
        for i, c in enumerate(out, 1):
            c.cid = "h%d" % i
            c.verdict = None
        self.stats["graph_gap_candidates"] = len(out)
        info.update(candidates=len(out), seeds=ginfo.get("seeds", 0), walked=ginfo.get("walked", 0),
                    dropped=ginfo.get("dropped"), ms=round((time.perf_counter() - t0) * 1000.0, 1),
                    picked=[{"cid": c.cid, "doc_id": c.doc_id, "label": (c.features.get("fact") or {}).get("label"),
                             "value": (c.features.get("fact") or {}).get("value")} for c in out])
        return out, info

    def retrieve(self, plan, doc_ids, facets, cap):
        self._targets = list(doc_ids)
        self._plan_unit = plan.get("unit") or "count"          # the merge reserves no graph slot for a text plan
        pool, info = super().retrieve(plan, doc_ids, facets, cap)
        if not self.a.use_graph:
            return pool, info
        concepts = self.concepts(plan, doc_ids)
        g, ginfo = GR.graph_candidates(self.con, self.reader, plan, doc_ids, d_lang=self.q.d_lang, opts=self.o,
                                       tree=self.a.use_tree, family=self.a.use_family, cards=self.a.use_cards,
                                       concepts=concepts)
        self.stats.update(concept_seeds=ginfo.get("concept_seeds", 0), concept_walked=ginfo.get("concept_walked", 0))
        RT.tag_facet(g, None)
        self.stats.update(graph_candidates=len(g), graph_seeds=ginfo.get("seeds", 0),
                          graph_walked=ginfo.get("walked", 0), graph_ms=ginfo.get("ms", 0.0),
                          cards_routed=int("cards" in ginfo))
        info["graph"] = ginfo
        return pool + g, info

    def merge(self, pool, doc_ids, facets, cap):
        if not self.a.use_graph:
            return super().merge(pool, doc_ids, facets, cap)
        slots = GR.graph_slots(self.o, doc_ids, getattr(self, "_plan_unit", "count"))
        out, info = GR.merge_graph(pool, doc_ids, facets, self.o, cap, per_period=bool(self.o["graph_per_period"]),
                                   slots=slots)
        self.stats.update(graph_slots=info["graph_slots"], graph_in_selection=info["graph_in_selection"])
        return out, info

    def annotate_rows(self, out, rows, batch):
        """v3's annotation, then the graph qualifier for a verified row whose definition is still 'unspecified'."""
        super().annotate_rows(out, rows, batch)
        if not (self.a.use_graph and self.o["graph_quals_to_def_class"]):
            return
        quals_by_cid = {c.cid: ((c.features or {}).get("graph") or {}).get("quals") or "" for c in batch}
        for r in rows:
            if r.get("def_class") not in (None, "", "unspecified"):
                continue
            q = [x for x in quals_by_cid.get(r.get("cid"), "").split(",") if x in GRAPH_DEF_CLASSES]
            main = sorted(x for x in q if x != "subgroup")
            if main or q:
                r["def_class"] = "+".join(main) if main else "subgroup"
                r["def_source"] = "graph"
                self.stats["graph_quals_applied"] += 1
