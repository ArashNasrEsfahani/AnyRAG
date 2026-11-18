"""OURS index time: per-section enrichment with the local model, then the enriched store.

For every section with body text the local model writes (strict JSON, Ollama structured
output, one repair retry):
  keywords_fa / keywords_en, synonyms_fa / synonyms_en   (both languages, always)
  questions                                             (3-5, in the document language)
  facts [{metric, value, unit, period}]                 (numbers stated in the section, tables included)
Fact values are checked against the section text (number tokens, separators tolerated);
facts whose number does not occur are dropped and counted. The model never sees any
question or question-set metadata: its only input is the section title path and text.

Enrichment is cached per section part in one sqlite KV file shared by all OURS variants
(work/indexes/OURS/_enrichment/<model>-<prompt version>.sqlite), keyed by model digest,
prompt version, generation options and the exact input text, so a variant that only
changes the store layout or query side (no_facts, monolingual_patterns, granularity,
no_normalization) reuses it without new calls. Every cached entry keeps the cost of the
calls that produced it, so the full enrichment cost can be reported for any variant.

Enriched store file (store.write_store; tags in the line map in brackets):
  @@SECTION doc=.. sec=.. level=..   [meta]  (ignored by the executor)
  @@TITLE <title>                    [meta]  -> line kind "title"
  @@PATH <a > b > c>                 [meta]  -> line kind "path"
  (blank)
  @@HEADER                           [header] fence, ignored
  kw_fa: ... ; ...                   [header] -> "header"
  kw_en: ... ; ...
  syn_fa: ... / syn_en: ...
  q: <question>                      (one line per question)
  @@END HEADER
  (blank)
  @@FACTS                            [fence]
  <metric> | <value> | <unit> | <period>   [fact, unit_id = the unit the value was found in]
  @@END FACTS                        [fence]
  (blank)
  <unit text> ...                    [body, unit_id]
"""
import json
import re
import time
from pathlib import Path

from ...normalize import normalize, normalize_lines, word_tokens
from ...store import StoreBlock, StoreFile, write_store
from ...util import KVCache, now_iso, safe_name, sha1_json
from ...llm import ModelError
from .defaults import DEFAULTS, PROMPT_VERSION

LANG_NAMES = {"fa": "Persian", "en": "English"}

def enrich_schema(max_keywords, max_synonyms, max_questions, max_facts):
    def arr(n):
        return {"type": "array", "items": {"type": "string"}, "maxItems": int(n)}
    return {
        "type": "object",
        "properties": {
            "keywords_fa": arr(max_keywords),
            "keywords_en": arr(max_keywords),
            "synonyms_fa": arr(max_synonyms),
            "synonyms_en": arr(max_synonyms),
            "questions": arr(max_questions),
            "facts": {
                "type": "array",
                "maxItems": int(max_facts),
                "items": {
                    "type": "object",
                    "properties": {"metric": {"type": "string"}, "value": {"type": "string"},
                                   "unit": {"type": "string"}, "period": {"type": "string"}},
                    "required": ["metric", "value", "unit", "period"],
                },
            },
        },
        "required": ["keywords_fa", "keywords_en", "synonyms_fa", "synonyms_en", "questions", "facts"],
    }


ENRICH_SCHEMA = enrich_schema(DEFAULTS["max_keywords"], DEFAULTS["max_synonyms"], DEFAULTS["max_questions"],
                              DEFAULTS["max_facts_per_part"])

ENRICH_SYSTEM = (
    "You index one section of a report for a keyword search engine. Read the section and return one JSON object:\n"
    "- keywords_fa: {kw} Persian search keywords for the section's topics, entities and metrics, each one to three words.\n"
    "- keywords_en: the same keywords in English.\n"
    "- synonyms_fa: up to {syn} other Persian words a reader might search for instead (synonyms, variant spellings, broader terms), each one or two words.\n"
    "- synonyms_en: up to {syn} such English words.\n"
    "- questions: {qmin} to {qmax} short questions that this section answers, written in {doc_lang}.\n"
    "- facts: at most {facts} numeric facts stated in the section, table cells included. Each fact has "
    "metric (what is measured and for whom, in {doc_lang}, a few words), value (the number exactly as written in the text), "
    "unit (such as %, $M, tonnes; empty if none) and period (year, date or date range; empty if not stated). "
    "If there are more, keep the headline figures and totals.\n"
    "Use only what the section says. Copy numbers exactly; never compute, round or convert them."
)

ENRICH_USER = "Report: {doc_title}\nSection: {path}{part}\n\nSection text:\n{text}"

REPAIR_USER = ("Your reply was not a complete valid JSON object ({problem}). Reply again with the complete JSON object only. "
               "Keep it shorter: at most 8 keywords per list, 5 synonyms per list, 3 questions and 10 facts.")

_NUM = re.compile(r"\d+(?:[.,/]\d+)*")


# --------------------------------------------------------------------------
# numbers: validating fact values against the section text
# --------------------------------------------------------------------------

def _canon_float(s):
    try:
        return "%.10g" % float(s)
    except ValueError:
        return None


def number_variants(tok):
    """Equivalent spellings of one number token: thousands commas removed, a comma or a
    Persian slash used as the decimal mark, float canonical form; dates split on '/'."""
    t = tok.strip()
    out = {t, t.replace(",", "")}
    if "," in t and not re.fullmatch(r"\d{1,3}(,\d{3})+(\.\d+)?", t):
        out.add(t.replace(",", "."))
    if "/" in t:
        out.add(t.replace("/", "."))
        parts = t.split("/")
        if len(parts) >= 3 or any(len(p) == 4 for p in parts):
            out.update(parts)
    for v in list(out):
        c = _canon_float(v)
        if c is not None:
            out.add(c)
    return out


def text_number_set(text):
    s = set()
    for tok in _NUM.findall(text or ""):
        s |= number_variants(tok)
    return s


def value_numbers(value):
    return _NUM.findall(normalize(value or ""))


def fact_value_ok(value, number_set):
    toks = value_numbers(value)
    if not toks:
        return False, "no_number"
    for t in toks:
        if not (number_variants(t) & number_set):
            return False, "not_in_text"
    return True, ""


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def _clean_str(s, max_chars, norm=True):
    if not isinstance(s, str):
        s = "" if s is None else str(s)
    s = normalize(s) if norm else " ".join(s.split())
    s = s.replace("|", "/").replace(";", ",").strip(" -,:")
    return s[:max_chars].strip()


def _clean_list(xs, cap, max_chars, norm=True):
    out, seen = [], set()
    for x in xs if isinstance(xs, list) else []:
        c = _clean_str(x, max_chars, norm)
        if c and c.lower() not in seen:
            seen.add(c.lower())
            out.append(c)
        if len(out) >= cap:
            break
    return out


def section_body(doc, section, raw=False):
    """[(unit_id, text)] of the section's own non-empty units."""
    out = []
    for uid in section.unit_ids:
        u = doc.unit(uid)
        t = u.text_raw if raw else u.text
        if raw:
            t = "\n".join(line.strip() for line in (t or "").replace("\r\n", "\n").split("\n") if line.strip())
        if t and t.strip():
            out.append((uid, t))
    return out


def split_parts(units, max_chars, max_parts):
    """Group consecutive units into parts of at most max_chars (a single longer unit is cut)."""
    parts, cur, size = [], [], 0
    for uid, t in units:
        if len(t) > max_chars:
            if cur:
                parts.append(cur)
                cur, size = [], 0
            parts.append([(uid, t[:max_chars])])
            continue
        add = len(t) + (2 if cur else 0)
        if cur and size + add > max_chars:
            parts.append(cur)
            cur, size = [], 0
            add = len(t)
        cur.append((uid, t))
        size += add
    if cur:
        parts.append(cur)
    return parts[:max_parts], max(0, len(parts) - max_parts)


# --------------------------------------------------------------------------
# enrichment
# --------------------------------------------------------------------------

class Enricher:
    """Calls the local model once (plus at most one repair) per section part, with a durable cache."""

    def __init__(self, cfg, client, cache_path, options=None):
        self.cfg = cfg
        self.client = client
        self.opts = dict(DEFAULTS, **(options or {}))
        self.cache = KVCache(cache_path)
        o = self.opts
        self.schema = enrich_schema(o["max_keywords"], o["max_synonyms"], o["max_questions"], o["max_facts_per_part"])
        self.counters = {"parts": 0, "cache_hits": 0, "calls": 0, "repairs": 0, "failed_parts": 0}

    def messages(self, doc, section, text, part_no, n_parts):
        o = self.opts
        system = ENRICH_SYSTEM.format(kw="5 to %d" % o["max_keywords"], syn=o["max_synonyms"], qmin=o["min_questions"],
                                      qmax=o["max_questions"], facts=o["max_facts_per_part"],
                                      doc_lang=LANG_NAMES.get(doc.lang, doc.lang))
        part = " (part %d of %d)" % (part_no + 1, n_parts) if n_parts > 1 else ""
        user = ENRICH_USER.format(doc_title=doc.title, path=" > ".join(t for t in doc.section_path(section.id) if t),
                                  part=part, text=text)
        return [{"role": "system", "content": system}, {"role": "user", "content": user}]

    def cache_key(self, msgs):
        return sha1_json({"v": PROMPT_VERSION, "model": self.cfg.models["chat"],
                          "digest": self.cfg.models.get("chat_digest"), "gen": self.cfg.generation,
                          "num_predict": self.opts["enrich_num_predict"], "attempts": self.opts["enrich_attempts"],
                          "schema": self.schema, "messages": msgs})

    @staticmethod
    def _problem(obj, done_reason):
        if obj is None or not isinstance(obj, dict):
            return "truncated at the length limit" if done_reason == "length" else "no JSON object"
        missing = [k for k in ENRICH_SCHEMA["required"] if k not in obj]
        if missing:
            return "missing fields: %s" % ", ".join(missing)
        return None

    def enrich_part(self, doc, section, text, part_no, n_parts):
        """-> cached record {status, obj, calls, prompt_tokens, completion_tokens, wall_ms, attempts: [...]}"""
        msgs = self.messages(doc, section, text, part_no, n_parts)
        key = self.cache_key(msgs)
        self.counters["parts"] += 1
        hit = self.cache.get(key)
        if hit is not None:
            self.counters["cache_hits"] += 1
            return hit
        rec = {"status": "failed", "obj": None, "calls": 0, "prompt_tokens": 0, "completion_tokens": 0,
               "wall_ms": 0.0, "attempts": [], "section_id": section.id, "part": part_no, "created": now_iso()}
        convo = list(msgs)
        n_predict = int(self.opts["enrich_num_predict"])
        for attempt in range(int(self.opts["enrich_attempts"])):
            purpose = "enrich" if attempt == 0 else "enrich_repair"
            try:
                r = self.client.chat(convo, purpose=purpose, format=self.schema, num_predict=n_predict,
                                     section_id=section.id)
            except ModelError:
                # transport/server failure after the client's own retries: do not cache anything,
                # stop the build (build_stats records the failed attempt; a rerun resumes from the cache)
                self.counters["calls"] += 1
                raise
            self.counters["calls"] += 1
            if attempt:
                self.counters["repairs"] += 1
            rec["calls"] += 1
            rec["prompt_tokens"] += r.prompt_tokens
            rec["completion_tokens"] += r.completion_tokens
            rec["wall_ms"] = round(rec["wall_ms"] + r.wall_ms, 1)
            obj = r.json(None)
            problem = self._problem(obj, r.done_reason)
            # prompt + completion cap beyond num_ctx: Ollama truncates silently -> make it visible in the stats
            ctx_risk = r.prompt_tokens + n_predict > int(self.cfg.generation["num_ctx"])
            rec["attempts"].append({"done_reason": r.done_reason, "problem": problem, "chars": len(r.text),
                                    "prompt_tokens": r.prompt_tokens, "ctx_risk": ctx_risk})
            if problem is None:
                rec["status"] = "ok" if attempt == 0 else "repaired"
                rec["obj"] = obj
                break
            convo = list(msgs) + [{"role": "assistant", "content": r.text[:4000]},
                                  {"role": "user", "content": REPAIR_USER.format(problem=problem)}]
            n_predict = int(n_predict * 1.5)
        if rec["status"] == "failed":
            self.counters["failed_parts"] += 1
        self.cache.put(key, rec)
        return rec

    def enrich_section(self, doc, section, raw_input=False, normalize_output=True):
        """-> dict(status, keywords_fa, keywords_en, synonyms_fa, synonyms_en, questions, facts, dropped_facts,
        n_parts, skipped_parts, cost{calls, prompt_tokens, completion_tokens, wall_ms}).
        normalize_output=False keeps the model's strings un-normalised (no_normalization store);
        fact validation always compares normalised numbers."""
        o = self.opts
        nrm = bool(normalize_output)
        units = section_body(doc, section, raw=raw_input)
        body_chars = sum(len(t) for _, t in units)
        out = {"status": "no_body", "keywords_fa": [], "keywords_en": [], "synonyms_fa": [], "synonyms_en": [],
               "questions": [], "facts": [], "dropped_facts": [], "n_parts": 0, "skipped_parts": 0,
               "cost": {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "wall_ms": 0.0}}
        if body_chars < o["enrich_min_chars"]:
            out["status"] = "no_body" if body_chars == 0 else "too_short"
            return out
        parts, skipped = split_parts(units, o["enrich_input_chars"], o["enrich_max_parts"])
        out["n_parts"], out["skipped_parts"] = len(parts), skipped
        # numbers of the whole section (normalised, whatever the input variant) for fact validation
        norm_units = [(uid, doc.unit(uid).text) for uid in section.unit_ids]
        sec_numbers = text_number_set(section.title + "\n" + "\n".join(t for _, t in norm_units))
        unit_numbers = [(uid, text_number_set(t), set(word_tokens(t))) for uid, t in norm_units]
        statuses = []
        seen_facts = set()
        mf, ms = o["field_max_chars"], o["max_synonyms"]
        for i, part in enumerate(parts):
            text = "\n\n".join(t for _, t in part)
            rec = self.enrich_part(doc, section, text, i, len(parts))
            statuses.append(rec["status"])
            if any(a.get("ctx_risk") for a in rec.get("attempts") or []):
                out["ctx_risk_parts"] = out.get("ctx_risk_parts", 0) + 1
            for k in ("calls", "prompt_tokens", "completion_tokens", "wall_ms"):
                out["cost"][k] = round(out["cost"][k] + (rec.get(k) or 0), 1)
            obj = rec.get("obj") or {}
            if not obj:
                continue
            for k, cap in (("keywords_fa", o["max_keywords"]), ("keywords_en", o["max_keywords"]),
                           ("synonyms_fa", ms), ("synonyms_en", ms), ("questions", o["max_questions"])):
                merged = out[k] + _clean_list(obj.get(k), cap, 300 if k == "questions" else mf, nrm)
                out[k] = _clean_list(merged, cap, 300 if k == "questions" else mf, nrm)
            for f in (obj.get("facts") or [])[: o["max_facts_per_part"]]:
                if not isinstance(f, dict):
                    continue
                fact = {k: _clean_str(f.get(k), mf, nrm) for k in ("metric", "value", "unit", "period")}
                if not fact["metric"] or not fact["value"]:
                    out["dropped_facts"].append(dict(fact, reason="empty"))
                    continue
                ok, why = fact_value_ok(fact["value"], sec_numbers)
                if not ok:
                    out["dropped_facts"].append(dict(fact, reason=why))
                    continue
                sig = (fact["metric"].lower(), fact["value"], fact["unit"].lower(), fact["period"].lower())
                if sig in seen_facts:
                    continue
                seen_facts.add(sig)
                fact["unit_id"] = self._source_unit(fact, unit_numbers)
                out["facts"].append(fact)
        if all(s == "failed" for s in statuses):
            out["status"] = "failed"
        elif any(s == "failed" for s in statuses):
            out["status"] = "partial"
        elif any(s == "repaired" for s in statuses):
            out["status"] = "repaired"
        else:
            out["status"] = "ok"
        return out

    @staticmethod
    def _source_unit(fact, unit_numbers):
        toks = value_numbers(fact["value"])
        metric_words = set(word_tokens(fact["metric"]))
        best, best_overlap = "", -1
        for uid, nums, words in unit_numbers:
            if all(number_variants(t) & nums for t in toks):
                ov = len(metric_words & words)
                if ov > best_overlap:
                    best, best_overlap = uid, ov
        return best


# --------------------------------------------------------------------------
# the enriched store
# --------------------------------------------------------------------------

def fact_line(f):
    return "%s | %s | %s | %s" % (f["metric"], f["value"], f["unit"] or "-", f["period"] or "-")


def section_store_file(doc, section, enr, *, facts=True, normalized=True):
    title = (section.title if normalized else section.title_raw) or ""
    title = " ".join(title.split())
    if normalized:
        path = " > ".join(t for t in doc.section_path(section.id) if t)
    else:
        chain, s = [], section
        while s is not None:
            chain.append(" ".join((s.title_raw or "").split()))
            s = doc.section(s.parent) if s.parent else None
        path = " > ".join(t for t in reversed(chain) if t)
    blocks = [StoreBlock("@@SECTION doc=%s sec=%s level=%d" % (doc.id, section.id, section.level), "", "meta"),
              StoreBlock("@@TITLE %s" % title, "", "meta"),
              StoreBlock("@@PATH %s" % path, "", "meta")]
    hl = []
    for key, label in (("keywords_fa", "kw_fa"), ("keywords_en", "kw_en"), ("synonyms_fa", "syn_fa"),
                       ("synonyms_en", "syn_en")):
        if enr.get(key):
            hl.append("%s: %s" % (label, " ; ".join(enr[key])))
    for q in enr.get("questions") or []:
        hl.append("q: %s" % q)
    if hl:
        blocks.append(StoreBlock("\n".join(["@@HEADER"] + hl + ["@@END HEADER"]), "", "header"))
    if facts and enr.get("facts"):
        blocks.append(StoreBlock("@@FACTS", "", "fence"))
        for f in enr["facts"]:
            blocks.append(StoreBlock(fact_line(f), f.get("unit_id") or "", "fact"))
        blocks.append(StoreBlock("@@END FACTS", "", "fence"))
    for uid, t in section_body(doc, section, raw=not normalized):
        blocks.append(StoreBlock(t, uid, "body"))
    return StoreFile(doc.id, section.id, "%s/%04d.txt" % (safe_name(doc.id), section.order), section.level, title, blocks)


def build_enriched_store(corpus_set, root, enricher, *, facts=True, normalized=True, raw_enrichment=False,
                         log_every=25, log=print):
    """Enrich every section of every document (resumable through the cache) and write the store.
    Returns build stats (also saved as <root>/../enrichment_stats.json by the caller)."""
    t0 = time.perf_counter()
    files = []
    st = {"sections": 0, "enriched": 0, "no_body": 0, "too_short": 0, "failed": 0, "partial": 0, "repaired": 0,
          "parts": 0, "skipped_parts": 0, "facts_kept": 0, "facts_dropped": 0, "drop_reasons": {},
          "questions": 0, "sections_lt_min_questions": 0,
          "cost_all_cached_calls": {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "wall_ms": 0.0}}
    sections_detail = {}
    total = sum(len(d.sections) for d in corpus_set.docs)
    c0 = dict(enricher.counters)
    for d in corpus_set.docs:
        for s in d.sections:
            enr = enricher.enrich_section(d, s, raw_input=(raw_enrichment and not normalized),
                                          normalize_output=normalized)
            st["sections"] += 1
            if enr["status"] in ("no_body", "too_short"):
                st[enr["status"]] += 1
            else:
                st["enriched"] += 1
                if enr["status"] in ("failed", "partial", "repaired"):
                    st[enr["status"]] += 1
            st["parts"] += enr["n_parts"]
            st["skipped_parts"] += enr["skipped_parts"]
            st["ctx_risk_parts"] = st.get("ctx_risk_parts", 0) + enr.get("ctx_risk_parts", 0)
            st["facts_kept"] += len(enr["facts"])
            st["facts_dropped"] += len(enr["dropped_facts"])
            for df in enr["dropped_facts"]:
                st["drop_reasons"][df["reason"]] = st["drop_reasons"].get(df["reason"], 0) + 1
            st["questions"] += len(enr["questions"])
            if enr["status"] not in ("no_body", "too_short") and len(enr["questions"]) < enricher.opts["min_questions"]:
                st["sections_lt_min_questions"] += 1
            for k in st["cost_all_cached_calls"]:
                st["cost_all_cached_calls"][k] = round(st["cost_all_cached_calls"][k] + enr["cost"][k], 1)
            sections_detail[s.id] = {"status": enr["status"], "facts": len(enr["facts"]),
                                     "dropped": [{k: f[k] for k in ("metric", "value", "reason")} for f in enr["dropped_facts"]],
                                     "questions": len(enr["questions"]), "parts": enr["n_parts"]}
            files.append(section_store_file(d, s, enr, facts=facts, normalized=normalized))
            if log and log_every and st["sections"] % log_every == 0:
                log("[OURS index] %s: %d/%d sections, %d new calls" % (corpus_set.name, st["sections"], total,
                                                                      enricher.counters["calls"] - c0["calls"]))
    denom = st["facts_kept"] + st["facts_dropped"]
    st["fact_drop_rate"] = round(st["facts_dropped"] / denom, 4) if denom else 0.0
    st["new_calls"] = enricher.counters["calls"] - c0["calls"]
    st["cache_hits"] = enricher.counters["cache_hits"] - c0["cache_hits"]
    st["enrich_ms"] = round((time.perf_counter() - t0) * 1000.0, 1)
    write_store(root, files, variant="ours-enriched" + ("" if normalized else "-raw") + ("" if facts else "-nofacts"),
                extra_manifest={"corpus_set": corpus_set.name, "corpus_fingerprint": corpus_set.fingerprint(),
                                "normalized": normalized, "facts": facts, "prompt_version": PROMPT_VERSION})
    return st, sections_detail
