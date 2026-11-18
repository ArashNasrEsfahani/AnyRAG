"""OURS cheap index levels, the model parts: L1 document glossary and L2 batched section keywords.

Both are crash-safe and resumable: every result is committed to a ragbench.util.KVCache (SQLite WAL, one commit per
put) the moment its call returns, keyed by a content hash of the exact input + model + model digest + generation
options + prompt version + output caps. A restarted build skips everything already cached, so an interrupted build
loses at most the call in flight. Transport failures (ModelError after the shared client's own retries, e.g. while the
Ollama tray app restarts the server) are never cached: they stop the build and the rerun resumes from the cache.

L1 glossary (GLOSSARY_PROMPT_VERSION), one call per document (+ at most one repair call):
  input  the document title and lite.doc_digest (section titles + most frequent content terms); never a question
  output strict JSON (Ollama structured output, maxItems) {"terms": [{"<doc lang>": term as the report spells it,
         "<other lang>": translation, "fa_syn": [...], "en_syn": [...]}]}; parse_glossary cleans every entry
         (script check, swapped languages fixed, lengths, dedupe) and salvages the complete entries of a truncated reply
  repair when fewer than l1_min_terms entries survive: one repair call (the model sees its reply and the problem,
         fewer terms asked). The first reply is checkpointed in the cache before the repair call, so an interruption
         between the two resumes with the repair call.
L2 keywords (KEYWORDS_PROMPT_VERSION), several sections per call:
  input  per section: its path and the first l2_section_chars characters of its own text (lite.section_block_text)
  output strict JSON {"sections": [{"id": "S1", "kw_fa": [...], "kw_en": [...], "syn_fa": [...], "syn_en": [...]}]},
         num_predict = l2_num_predict_base + l2_num_predict_per_section * sections in the batch
  repair the sections missing from a reply (bad JSON, truncation, unknown ids, no keyword) are asked again in one
         smaller batch call; still missing -> cached as failed (like L3's failed parts)
  cache  per section (content hash of report title + section path + input text), so batch composition does not matter
         on resume; each call's cost is split evenly over the sections it answered.
"""
import json
import re
import time

from ...llm import ModelError, extract_json
from ...util import KVCache, now_iso, sha1_json
from .lite import clean_term, doc_digest, section_block_text, text_script

GLOSSARY_PROMPT_VERSION = "ours-glossary-v2"
# v1 -> v2 (live check on A.rpt-c, 2026-09-16): v1 chose generic words (customs, technology, capital, data) that occur
# in most sections, and 40 Persian terms overran 1600 tokens; v2 asks for terms that tell sections apart, 30 terms.
KEYWORDS_PROMPT_VERSION = "ours-kwbatch-v1"

LANG_NAMES = {"fa": "Persian", "en": "English"}

LLM_DEFAULTS = {
    # L1 glossary
    "l1_max_terms": 30,
    "l1_max_syn": 2,
    "l1_num_predict": 1800,             # ~55 completion tokens per Persian entry
    "l1_attempts": 2,
    "l1_min_terms": 8,
    "l1_repair_terms": 20,
    "l1_digest_title_chars": 3000,
    "l1_digest_terms": 120,
    "l1_digest_min_count": 2,
    "l1_term_max_chars": 60,
    # L2 keyword batches
    "l2_batch_sections": 6,
    "l2_batch_input_chars": 5000,
    "l2_section_chars": 700,
    "l2_min_chars": 40,                 # sections whose own text is shorter get no keywords (L3 enrich_min_chars)
    "l2_max_keywords": 5,
    "l2_max_synonyms": 3,
    "l2_num_predict_base": 40,
    "l2_num_predict_per_section": 140,
    "l2_attempts": 2,
    "l2_term_max_chars": 60,
}


def _other(lang):
    return "en" if lang == "fa" else "fa"


def _arr(n):
    return {"type": "array", "items": {"type": "string"}, "maxItems": int(n)}


# ---------------------------------------------------------------------------------------------------------------
# tolerant JSON: complete objects of a possibly truncated reply
# ---------------------------------------------------------------------------------------------------------------

def salvage_objects(text, list_key):
    """(items, how): the dicts of obj[list_key] when the reply parses; otherwise every complete flat JSON object
    found in the text (a reply cut at the length limit keeps its finished entries). how: json | salvaged | none."""
    obj = extract_json(text, None)
    if isinstance(obj, dict):
        v = obj.get(list_key)
        if isinstance(v, list):
            return [x for x in v if isinstance(x, dict)], "json"
        lists = [x for x in obj.values() if isinstance(x, list) and x and all(isinstance(y, dict) for y in x)]
        if lists:
            return lists[0], "json"
    if isinstance(obj, list) and obj and all(isinstance(x, dict) for x in obj):
        return obj, "json"
    items, dec, i = [], json.JSONDecoder(), 0
    t = text or ""
    start = t.find("[")
    i = start + 1 if start >= 0 else 0
    while True:
        j = t.find("{", i)
        if j < 0:
            break
        try:
            o, end = dec.raw_decode(t, j)
        except json.JSONDecodeError:
            i = j + 1
            continue
        if isinstance(o, dict) and not any(isinstance(v, list) and v and isinstance(v[0], dict) for v in o.values()):
            items.append(o)
        i = end
    return items, ("salvaged" if items else "none")


def _as_list(v):
    if isinstance(v, str):
        return [x for x in re.split(r"[,،;؛]", v) if x.strip()]
    return [x for x in v if isinstance(x, str)] if isinstance(v, list) else []


# ---------------------------------------------------------------------------------------------------------------
# L1 glossary
# ---------------------------------------------------------------------------------------------------------------

GLOSSARY_SYSTEM = (
    "You build a bilingual search glossary for one report written in {D}. You see the report's title, its section "
    "titles and its most frequent terms. Choose up to {n} terms that tell the report's sections apart: specific topics, "
    "metrics, products, technologies, organisations, places and technical terms, preferably two or three words. Skip "
    "generic words that fit almost every section of this report (such as the report's own subject, report, data, "
    "market, company). For each term return one object: \"{d}\": the term exactly "
    "as the report spells it, in {D}; \"{o}\": the same term in {O}, as a report written in {O} would say it; "
    "\"{d}_syn\" and \"{o}_syn\": up to {s} other words a reader might search for instead (synonyms, abbreviations, "
    "variant spellings), in {D} and in {O}. Every term and synonym is one to three words. "
    "Return JSON {{\"terms\": [...]}} with the most important terms first, and nothing else."
)
GLOSSARY_USER = "Report: {title}\n\n{digest}"
GLOSSARY_REPAIR = ("Your reply was not usable ({problem}). Reply again with the complete JSON object only: at most {n} "
                   "terms and at most 1 synonym per language, each term one to three words.")


def glossary_schema(doc_lang, max_terms, max_syn):
    d, o = doc_lang, _other(doc_lang)
    props = {d: {"type": "string"}, o: {"type": "string"}, d + "_syn": _arr(max_syn), o + "_syn": _arr(max_syn)}
    return {"type": "object",
            "properties": {"terms": {"type": "array", "maxItems": int(max_terms),
                                     "items": {"type": "object", "properties": props, "required": list(props)}}},
            "required": ["terms"]}


def parse_glossary(text, doc_lang, max_terms=40, max_syn=2, max_chars=60):
    """Model reply -> (entries [{fa, en, fa_syn, en_syn}], report{how, raw, kept, dropped{reason: n}, swapped})."""
    items, how = salvage_objects(text, "terms")
    rep = {"how": how, "raw": len(items), "kept": 0, "dropped": {}, "swapped": 0}

    def drop(reason):
        rep["dropped"][reason] = rep["dropped"].get(reason, 0) + 1

    out, seen = [], set()
    for it in items:
        fa = clean_term(it.get("fa"), max_chars)
        en = clean_term(it.get("en"), max_chars)
        if text_script(fa) == "en" and text_script(en) == "fa":
            fa, en = en, fa
            rep["swapped"] += 1
        if not fa or not en:
            drop("empty_or_too_long")
            continue
        if text_script(fa) != "fa" or text_script(en) != "en":
            drop("wrong_script")
            continue
        entry = {"fa": fa, "en": en, "fa_syn": [], "en_syn": []}
        pool = [(clean_term(x, max_chars), "fa_syn") for x in _as_list(it.get("fa_syn"))] + \
               [(clean_term(x, max_chars), "en_syn") for x in _as_list(it.get("en_syn"))]
        low = {fa.lower(), en.lower()}
        for s, key in pool:
            if not s or s.lower() in low:
                continue
            key = "fa_syn" if text_script(s) == "fa" else "en_syn"      # the script decides the list
            if len(entry[key]) < max_syn:
                entry[key].append(s)
                low.add(s.lower())
        sig = (fa.lower(), en.lower())
        if sig in seen:
            drop("duplicate")
            continue
        seen.add(sig)
        out.append(entry)
        if len(out) >= max_terms:
            break
    rep["kept"] = len(out)
    return out, rep


class GlossaryBuilder:
    """One glossary per document, cached (and checkpointed between the first call and its repair)."""

    def __init__(self, cfg, client, cache_path, options=None, ollama_version=None):
        self.cfg = cfg
        self.client = client
        self.opts = dict(LLM_DEFAULTS, **(options or {}))
        self.cache = KVCache(cache_path)
        self.ollama_version = ollama_version
        self.counters = {"docs": 0, "cache_hits": 0, "cache_misses": 0, "calls": 0, "repairs": 0, "failed_docs": 0,
                         "resumed": 0, "new_prompt_tokens": 0, "new_completion_tokens": 0, "new_wall_ms": 0.0}
        self.digest_ms = 0.0                              # program time spent building digests (index cost)

    def messages(self, doc):
        o = self.opts
        d = doc.lang
        t0 = time.perf_counter()
        digest, info = doc_digest(doc, o["l1_digest_title_chars"], o["l1_digest_terms"], o["l1_digest_min_count"])
        self.digest_ms += (time.perf_counter() - t0) * 1000.0
        system = GLOSSARY_SYSTEM.format(D=LANG_NAMES[d], O=LANG_NAMES[_other(d)], d=d, o=_other(d),
                                        n=o["l1_max_terms"], s=o["l1_max_syn"])
        user = GLOSSARY_USER.format(title=doc.title, digest=digest)
        return [{"role": "system", "content": system}, {"role": "user", "content": user}], info

    def cache_key(self, doc, msgs):
        o = self.opts
        return sha1_json({"v": GLOSSARY_PROMPT_VERSION, "model": self.cfg.models["chat"],
                          "digest": self.cfg.models.get("chat_digest"), "gen": self.cfg.generation,
                          "caps": {k: o[k] for k in ("l1_max_terms", "l1_max_syn", "l1_num_predict", "l1_attempts",
                                                     "l1_min_terms", "l1_repair_terms", "l1_term_max_chars")},
                          "schema": glossary_schema(doc.lang, o["l1_max_terms"], o["l1_max_syn"]), "messages": msgs})

    def build(self, doc):
        """-> cached record {status: ok|repaired|salvaged|failed, entries, calls, prompt_tokens, completion_tokens,
        wall_ms, attempts, digest}"""
        o = self.opts
        msgs, info = self.messages(doc)
        key = self.cache_key(doc, msgs)
        self.counters["docs"] += 1
        hit = self.cache.get(key)
        if hit is not None:
            self.counters["cache_hits"] += 1
            return hit
        self.counters["cache_misses"] += 1
        schema = glossary_schema(doc.lang, o["l1_max_terms"], o["l1_max_syn"])
        rec = self.cache.get(key + ":attempt")           # a first reply checkpointed before an interrupted repair
        if rec is not None:
            self.counters["resumed"] += 1
        else:
            rec = {"status": "failed", "entries": [], "calls": 0, "prompt_tokens": 0, "completion_tokens": 0,
                   "wall_ms": 0.0, "attempts": [], "doc_id": doc.id, "digest": info, "created": now_iso(),
                   "ollama_version": self.ollama_version, "last_text": ""}
        convo = list(msgs)
        n_attempts = int(o["l1_attempts"])
        while len(rec["attempts"]) < n_attempts:
            attempt = len(rec["attempts"])
            if attempt:
                convo = list(msgs) + [{"role": "assistant", "content": rec["last_text"][:6000]},
                                      {"role": "user", "content": GLOSSARY_REPAIR.format(
                                          problem=rec["attempts"][-1]["problem"], n=o["l1_repair_terms"])}]
            purpose = "glossary" if attempt == 0 else "glossary_repair"
            try:
                r = self.client.chat(convo, purpose=purpose, format=schema, num_predict=int(o["l1_num_predict"]),
                                     doc_id=doc.id)
            except ModelError:
                self.counters["calls"] += 1
                raise
            self.counters["calls"] += 1
            if attempt:
                self.counters["repairs"] += 1
            self.counters["new_prompt_tokens"] += r.prompt_tokens
            self.counters["new_completion_tokens"] += r.completion_tokens
            self.counters["new_wall_ms"] += r.wall_ms
            rec["calls"] += 1
            rec["prompt_tokens"] += r.prompt_tokens
            rec["completion_tokens"] += r.completion_tokens
            rec["wall_ms"] = round(rec["wall_ms"] + r.wall_ms, 1)
            entries, prep = parse_glossary(r.text, doc.lang, o["l1_max_terms"], o["l1_max_syn"], o["l1_term_max_chars"])
            problem = None
            if len(entries) < int(o["l1_min_terms"]):
                problem = ("truncated at the length limit" if r.done_reason == "length" else
                           "no JSON object" if prep["how"] == "none" else
                           "only %d usable terms" % len(entries))
            rec["attempts"].append({"done_reason": r.done_reason, "problem": problem, "parse": prep,
                                    "prompt_tokens": r.prompt_tokens, "completion_tokens": r.completion_tokens,
                                    "ctx_risk": r.prompt_tokens + int(o["l1_num_predict"]) > int(self.cfg.generation["num_ctx"])})
            if len(entries) > len(rec["entries"]):
                rec["entries"] = entries
            rec["last_text"] = r.text
            if problem is None:
                rec["status"] = ("ok" if prep["how"] == "json" else "salvaged") if attempt == 0 else "repaired"
                break
            if len(rec["attempts"]) < n_attempts:
                self.cache.put(key + ":attempt", rec)     # checkpoint: a restart resumes with the repair call
        if rec["status"] == "failed":
            if rec["entries"]:
                rec["status"] = "partial"
            else:
                self.counters["failed_docs"] += 1
        rec.pop("last_text", None)
        self.cache.put(key, rec)
        return rec


# ---------------------------------------------------------------------------------------------------------------
# L2 keyword batches
# ---------------------------------------------------------------------------------------------------------------

KW_SYSTEM = (
    "You index sections of a report written in {D} for a keyword search engine. For every section below return one "
    "object: \"id\": the section's id (such as \"S1\"); \"kw_fa\": up to {k} Persian search keywords for what the section "
    "is about (topics, entities, metrics); \"kw_en\": the same keywords in English; \"syn_fa\" and \"syn_en\": up to {s} "
    "other Persian and English words a reader might search for instead (synonyms, variant spellings, broader terms). "
    "Each keyword is one to three words. Use only what the section says. "
    "Return JSON {{\"sections\": [...]}} with one object per section, in the order given, and nothing else."
)
KW_USER = "Report: {title}\n\n{blocks}"
KW_RETRY_NOTE = "\n\n(Answer every section above. Keep every list short.)"
KW_KEYS = ("kw_fa", "kw_en", "syn_fa", "syn_en")


def keywords_schema(n_sections, max_kw, max_syn):
    item = {"type": "object",
            "properties": {"id": {"type": "string"}, "kw_fa": _arr(max_kw), "kw_en": _arr(max_kw),
                           "syn_fa": _arr(max_syn), "syn_en": _arr(max_syn)},
            "required": ["id", "kw_fa", "kw_en", "syn_fa", "syn_en"]}
    return {"type": "object", "properties": {"sections": {"type": "array", "maxItems": int(n_sections), "items": item}},
            "required": ["sections"]}


def _norm_id(v):
    m = re.search(r"(\d+)", str(v if v is not None else ""))
    return "S%d" % int(m.group(1)) if m else None


def clean_keywords(item, max_kw, max_syn, max_chars):
    """{kw_fa, kw_en, syn_fa, syn_en} with the script deciding the language (a Latin keyword in kw_fa moves to kw_en)."""
    out = {k: [] for k in KW_KEYS}
    caps = {"kw_fa": max_kw, "kw_en": max_kw, "syn_fa": max_syn, "syn_en": max_syn}
    seen = set()
    moved = 0
    for k in KW_KEYS:
        for x in _as_list(item.get(k)):
            t = clean_term(x, max_chars)
            if not t or t.lower() in seen:
                continue
            lang = text_script(t)
            if lang is None:
                continue
            kind = k.split("_")[0]
            dest = "%s_%s" % (kind, lang)
            if dest != k:
                moved += 1
            if len(out[dest]) < caps[dest]:
                out[dest].append(t)
                seen.add(t.lower())
    return out, moved


def parse_keyword_batch(text, ids, max_kw=5, max_syn=3, max_chars=60):
    """Model reply for sections `ids` (["S1", ...]) -> ({id: keywords}, report{how, raw, by_order, unknown_ids,
    empty, moved}). Sections without any keyword are left out (they are retried)."""
    items, how = salvage_objects(text, "sections")
    rep = {"how": how, "raw": len(items), "by_order": False, "unknown_ids": 0, "empty": 0, "moved": 0}
    want = set(ids)
    got_ids = [_norm_id(it.get("id")) for it in items]
    if items and len(items) == len(ids) and not any(g in want for g in got_ids):
        got_ids = list(ids)                          # no usable ids at all, but one object per section: take the order
        rep["by_order"] = True
    out = {}
    for it, sid in zip(items, got_ids):
        if sid not in want:
            rep["unknown_ids"] += 1
            continue
        if sid in out:
            continue
        kw, moved = clean_keywords(it, max_kw, max_syn, max_chars)
        rep["moved"] += moved
        if not any(kw.values()):
            rep["empty"] += 1
            continue
        out[sid] = kw
    return out, rep


class KeywordBatcher:
    """Batched per-section keywords with a per-section cache (resumable at section granularity)."""

    def __init__(self, cfg, client, cache_path, options=None, ollama_version=None):
        self.cfg = cfg
        self.client = client
        self.opts = dict(LLM_DEFAULTS, **(options or {}))
        self.cache = KVCache(cache_path)
        self.ollama_version = ollama_version
        self.counters = {"sections": 0, "cache_hits": 0, "cache_misses": 0, "calls": 0, "repairs": 0,
                         "failed_sections": 0, "too_short": 0, "batches": 0, "new_prompt_tokens": 0,
                         "new_completion_tokens": 0, "new_wall_ms": 0.0}

    def system(self, doc):
        o = self.opts
        return KW_SYSTEM.format(D=LANG_NAMES[doc.lang], k=o["l2_max_keywords"], s=o["l2_max_synonyms"])

    def section_input(self, doc, section):
        path, body = section_block_text(doc, section, int(self.opts["l2_section_chars"]))
        return path, body

    def section_key(self, doc, path, body):
        o = self.opts
        return sha1_json({"v": KEYWORDS_PROMPT_VERSION, "model": self.cfg.models["chat"],
                          "digest": self.cfg.models.get("chat_digest"), "gen": self.cfg.generation,
                          "caps": {k: o[k] for k in ("l2_section_chars", "l2_max_keywords", "l2_max_synonyms",
                                                     "l2_num_predict_base", "l2_num_predict_per_section",
                                                     "l2_attempts", "l2_term_max_chars")},
                          "lang": doc.lang, "system": self.system(doc), "report": doc.title, "path": path, "body": body})

    def _batches(self, pending):
        o = self.opts
        out, cur, size = [], [], 0
        for p in pending:
            n = len(p["path"]) + len(p["body"]) + 8
            if cur and (len(cur) >= int(o["l2_batch_sections"]) or size + n > int(o["l2_batch_input_chars"])):
                out.append(cur)
                cur, size = [], 0
            cur.append(p)
            size += n
        if cur:
            out.append(cur)
        return out

    def _call(self, doc, batch, purpose, retry=False):
        o = self.opts
        ids = ["S%d" % (i + 1) for i in range(len(batch))]
        blocks = "\n\n".join("[%s] %s\n%s" % (sid, p["path"], p["body"]) for sid, p in zip(ids, batch))
        user = KW_USER.format(title=doc.title, blocks=blocks) + (KW_RETRY_NOTE if retry else "")
        msgs = [{"role": "system", "content": self.system(doc)}, {"role": "user", "content": user}]
        n_predict = int(o["l2_num_predict_base"]) + int(o["l2_num_predict_per_section"]) * len(batch)
        try:
            r = self.client.chat(msgs, purpose=purpose, format=keywords_schema(len(batch), o["l2_max_keywords"],
                                                                               o["l2_max_synonyms"]),
                                 num_predict=n_predict, doc_id=doc.id, n_sections=len(batch))
        except ModelError:
            self.counters["calls"] += 1
            raise
        self.counters["calls"] += 1
        self.counters["batches"] += 1
        self.counters["new_prompt_tokens"] += r.prompt_tokens
        self.counters["new_completion_tokens"] += r.completion_tokens
        self.counters["new_wall_ms"] += r.wall_ms
        parsed, rep = parse_keyword_batch(r.text, ids, o["l2_max_keywords"], o["l2_max_synonyms"], o["l2_term_max_chars"])
        rep.update(done_reason=r.done_reason, prompt_tokens=r.prompt_tokens, completion_tokens=r.completion_tokens,
                   n_sections=len(batch), purpose=purpose,
                   ctx_risk=r.prompt_tokens + n_predict > int(self.cfg.generation["num_ctx"]))
        return r, {p["key"]: parsed.get(sid) for sid, p in zip(ids, batch)}, rep

    def run(self, doc, sections):
        """-> {section_id: record {status: ok|repaired|failed|too_short, kw_fa, kw_en, syn_fa, syn_en, cost{calls,
        prompt_tokens, completion_tokens, wall_ms}}}. Every answered section is cached right after its call."""
        o = self.opts
        results, pending = {}, []
        for s in sections:
            self.counters["sections"] += 1
            path, body = self.section_input(doc, s)
            own = sum(len(doc.unit(u).text.strip()) for u in s.unit_ids)
            if own < int(o["l2_min_chars"]):
                self.counters["too_short"] += 1
                results[s.id] = {"status": "too_short", **{k: [] for k in KW_KEYS},
                                 "cost": {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "wall_ms": 0.0}}
                continue
            key = self.section_key(doc, path, body)
            hit = self.cache.get(key)
            if hit is not None:
                self.counters["cache_hits"] += 1
                results[s.id] = hit
                continue
            self.counters["cache_misses"] += 1
            pending.append({"section": s, "path": path, "body": body, "key": key,
                            "carry": {"calls": 0.0, "prompt_tokens": 0.0, "completion_tokens": 0.0, "wall_ms": 0.0},
                            "reports": []})
        attempts = int(o["l2_attempts"])
        for batch in self._batches(pending):
            todo = batch
            for attempt in range(attempts):
                if not todo:
                    break
                purpose = "kw_batch" if attempt == 0 else "kw_batch_repair"
                if attempt:
                    self.counters["repairs"] += 1
                r, parsed, rep = self._call(doc, todo, purpose, retry=bool(attempt))
                share = {"calls": 1.0 / len(todo), "prompt_tokens": r.prompt_tokens / len(todo),
                         "completion_tokens": r.completion_tokens / len(todo), "wall_ms": r.wall_ms / len(todo)}
                missing = []
                last = attempt == attempts - 1
                for p in todo:
                    for k in share:
                        p["carry"][k] += share[k]
                    p["reports"].append({k: rep[k] for k in ("how", "done_reason", "by_order", "unknown_ids", "empty",
                                                             "n_sections", "purpose", "ctx_risk")})
                    kw = parsed.get(p["key"])
                    if kw is None and not last:
                        missing.append(p)
                        continue
                    rec = {"status": ("ok" if attempt == 0 else "repaired") if kw else "failed",
                           **(kw or {k: [] for k in KW_KEYS}),
                           "cost": {k: round(v, 3) for k, v in p["carry"].items()},
                           "section_id": p["section"].id, "attempts": p["reports"], "created": now_iso(),
                           "ollama_version": self.ollama_version}
                    if rec["status"] == "failed":
                        self.counters["failed_sections"] += 1
                    self.cache.put(p["key"], rec)       # committed now: an interruption loses at most the next call
                    results[p["section"].id] = rec
                todo = missing
        return results
