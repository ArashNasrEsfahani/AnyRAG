"""OURS executor: runs the plan with ripgrep over the (enriched or plain) section store,
ranks sections and chooses the evidence granularity. No model calls, no BM25, no
embeddings: the only signals are which line kinds each regex matched in each section.

Line kinds (from the store line map tag and the line prefix):
  title  "@@TITLE ..."            path "@@PATH ..."        header  kw_*/syn_*/q: lines
  fact   "metric | value | unit | period"                  body    unit text
  ignored: "@@SECTION" lines, fences, blank lines, and matches that fall only inside a line prefix
Scope -> line kinds of the ranking tier: headers = title+path+header, facts = fact, body = body,
all = every kind. (A store without fact lines maps scope "facts" to "all".)

Section score = sum over patterns of the best weight that pattern reached in the section
(title/header 3 > fact 2 > body 1 > ancestor path 0.75) + 0.05 * min(hit lines, 20).
Ranking key: (score over the scope's kinds, score over all kinds, document order), so
in-scope matches come first and other matched sections fill the rest of the budget
(DEFAULTS scope_fallback_to_all; False = drop sections with no in-scope match).

Granularity (auto): for sections in rank order while the evidence budget has room:
  expect=number and fact lines matched  -> kind "fact". fact_evidence "context" (default): the verbatim source
                                           text of each matched fact (table header + matching row with the
                                           table's caption, a short unit whole, or the sentence(s) holding the
                                           value), each source once; the bare fact line only when no unit holds
                                           the value. "bare" (ablation facts_bare, dev_fix1): the matching fact
                                           lines, each followed by the verbatim source line(s) holding the number
  own body fits the remaining budget     -> the whole section (kind "section")
  otherwise                              -> +-window_lines store lines around body matches (kind "window")
then, budget permitting, the body context of the fact sections (section or window).
always_section / always_window force one granularity for every section.
"""
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from ...contracts import RetrievedItem, item_header
from ...normalize import word_tokens
from ...store import grep
from ...util import read_json
from .defaults import FACT_EVIDENCE
from .indexer import _NUM as NUM_RE, number_variants, text_number_set, value_numbers

KINDS = ("title", "path", "header", "fact", "body")
SCOPE_KINDS = {"headers": {"title", "path", "header"}, "facts": {"fact"}, "body": {"body"}, "all": set(KINDS)}
WINDOW_GAP = "\n…\n"


# --------------------------------------------------------------------------
# store view
# --------------------------------------------------------------------------

class StoreView:
    """Manifest, line maps and file lines of one store, loaded lazily and cached."""

    def __init__(self, root):
        self.root = Path(root)
        self.manifest = read_json(self.root / "manifest.json") or {"files": {}}
        self.files = self.manifest["files"]
        self.by_section = {info["section_id"]: rel for rel, info in self.files.items()}
        self._linemaps = {}
        self._lines = {}
        self.has_facts = bool(self.manifest.get("facts")) and str(self.manifest.get("variant", "")).startswith("ours-enriched")
        self.enriched = str(self.manifest.get("variant", "")).startswith("ours-enriched")

    def spans(self, rel):
        d = rel.split("/")[0]
        if d not in self._linemaps:
            self._linemaps[d] = read_json(self.root / d / "_linemap.json", {})
        return self._linemaps[d].get(rel, [])

    def lines(self, rel):
        if rel not in self._lines:
            with open(self.root / rel, encoding="utf-8") as fh:
                ls = fh.read().split("\n")
            if ls and ls[-1] == "":
                ls.pop()
            self._lines[rel] = ls
        return self._lines[rel]

    def body_spans(self, rel):
        return [(a, b, uid) for a, b, uid, tag in self.spans(rel) if tag == "body"]

    def body_units(self, rel):
        ls = self.lines(rel)
        return [(uid, "\n".join(ls[a - 1:b]), a, b) for a, b, uid in self.body_spans(rel)]

    def body_text(self, rel):
        return "\n\n".join(t for _, t, _, _ in self.body_units(rel)).strip()

    def fact_lines(self, rel):
        ls = self.lines(rel)
        return {a: (uid, ls[a - 1]) for a, b, uid, tag in self.spans(rel) if tag == "fact"}


def classify(hit):
    """(kind, content_offset) of a GrepHit; kind None = ignore."""
    t = hit.text
    if hit.tag == "meta":
        if t.startswith("@@TITLE "):
            return "title", 8
        if t.startswith("@@PATH "):
            return "path", 7
        return None, 0
    if hit.tag == "header":
        if t.startswith("@@"):
            return None, 0
        i = t.find(": ")
        return "header", (i + 2 if 0 <= i < 12 else 0)
    if hit.tag == "fact":
        return "fact", 0
    if hit.tag == "body":
        return "body", 0
    return None, 0


# --------------------------------------------------------------------------
# search + ranking
# --------------------------------------------------------------------------

@dataclass
class SectionHits:
    rel: str
    doc_id: str
    section_id: str
    order_key: tuple
    hits: list = field(default_factory=list)          # [(line, kind, pattern_index, unit_id)]
    score: float = 0.0                                # over every line kind
    best: dict = field(default_factory=dict)          # pattern_index -> best kind reached
    via_parent: str = None
    scope_score: float = 0.0                          # over the plan's scope kinds only (ranking tier)

    def lines_of(self, kinds, patterns=None):
        return sorted({ln for ln, k, pi, _ in self.hits if k in kinds and (patterns is None or pi in patterns)})


def run_patterns(root, patterns, doc_ids, cfg, max_hits, view=None):
    """One ripgrep run per pattern (so each hit is attributed to its pattern).
    Every store file has one "@@SECTION doc=<id> sec=<id>" line that ASCII patterns can match
    (e.g. "report", "snapp") but the executor ignores; the grep cap is raised by the number of
    files in scope so those lines can never push real hits past max_hits."""
    out = {}
    if doc_ids is not None and not doc_ids:
        return {i: [] for i in range(len(patterns))}, max_hits   # a doc scope with no document in this set
    cap = max_hits
    if max_hits and view is not None:
        if doc_ids is None:
            n_files = len(view.files)
        else:
            scope = set(doc_ids)
            n_files = sum(1 for info in view.files.values() if info.get("doc_id") in scope)
        cap = max_hits + n_files
    for i, p in enumerate(patterns):
        try:
            out[i] = grep(root, p, doc_ids=doc_ids, ignore_case=True, max_hits=cap, cfg=cfg)
        except ValueError as e:
            out[i] = e
        except subprocess.TimeoutExpired as e:          # one pathological pattern must not fail the question
            out[i] = RuntimeError("ripgrep timeout: %s" % e)
    return out, cap


def rank_sections(hits_by_pattern, view, scope, corpus, opts, grep_cap=None):
    """-> (ranked [SectionHits], info)"""
    weights = opts["weights"]
    doc_pos = {d.id: i for i, d in enumerate(corpus.docs)}
    secs = {}
    info = {"per_pattern": [], "scope_requested": scope}
    too_broad = set()
    for pi, hits in hits_by_pattern.items():
        if isinstance(hits, Exception):
            info["per_pattern"].append({"i": pi, "error": str(hits)[:160]})
            continue
        kinds_count = {}
        n_used = 0
        for h in hits:
            kind, off = classify(h)
            if kind is None:
                continue
            if off and h.submatches and all(e <= off for s, e in h.submatches):
                continue
            n_used += 1
            kinds_count[kind] = kinds_count.get(kind, 0) + 1
            sh = secs.get(h.path)
            if sh is None:
                try:
                    order = corpus.section(h.section_id).order
                except KeyError:
                    order = 10 ** 6
                sh = secs[h.path] = SectionHits(h.path, h.doc_id, h.section_id, (doc_pos.get(h.doc_id, 10 ** 6), order))
            sh.hits.append((h.line, kind, pi, h.unit_id))
        truncated = bool(grep_cap) and len(hits) >= grep_cap
        if truncated or n_used >= opts["max_hits_per_pattern"]:
            too_broad.add(pi)
        info["per_pattern"].append({"i": pi, "raw_hits": len(hits), "used_hits": n_used, "kinds": kinds_count,
                                    "sections": len({h.path for h in hits}), "too_broad": pi in too_broad,
                                    "truncated": truncated})
    if too_broad and len(too_broad) < len([h for h in hits_by_pattern.values() if not isinstance(h, Exception) and h]):
        excluded = too_broad
    else:
        excluded = set()
    info["excluded_patterns"] = sorted(excluded)

    def score(sh, allowed):
        best, count = {}, 0
        for ln, kind, pi, _ in sh.hits:
            if kind not in allowed or pi in excluded:
                continue
            count += 1
            if weights[kind] > weights.get(best.get(pi), -1):
                best[pi] = kind
        if not best:
            return 0.0, best
        return sum(weights[k] for k in best.values()) + opts["hit_count_bonus"] * min(count, opts["hit_count_cap"]), best

    eff_scope = scope
    if scope == "facts" and not view.has_facts:
        eff_scope = "all"
    allowed = SCOPE_KINDS.get(eff_scope, SCOPE_KINDS["all"])
    n_in = 0
    for sh in secs.values():
        s_all, b_all = score(sh, SCOPE_KINDS["all"])
        s_in, _ = score(sh, allowed) if eff_scope != "all" else (s_all, b_all)
        sh.score, sh.best = s_all, b_all
        sh.scope_score = s_in
        n_in += 1 if s_in > 0 else 0
    fallback = eff_scope != "all" and n_in == 0 and bool(secs)
    if opts["scope_fallback_to_all"]:
        # scope is a priority tier: in-scope matches first, then every other matched section
        pool = [sh for sh in secs.values() if sh.score > 0]
    else:
        pool = [sh for sh in secs.values() if sh.scope_score > 0]
    ranked = sorted(pool, key=lambda s: (-s.scope_score, -s.score, s.order_key))
    info.update(scope_used=eff_scope, scope_fallback=fallback, sections_in_scope=n_in, sections_matched=len(ranked))
    return ranked, info


def expand_empty(ranked, view, corpus, limit):
    """A ranked section without own body text is replaced by its children that have body
    (they inherit the parent's rank position); keeps the first `limit` entries."""
    out, seen = [], set()
    for sh in ranked:
        if len(out) >= limit:
            break
        if sh.rel in seen:
            continue
        if view.body_spans(sh.rel):
            out.append(sh)
            seen.add(sh.rel)
            continue
        try:
            sec = corpus.section(sh.section_id)
            doc = corpus.doc(sh.doc_id)
        except KeyError:
            continue
        queue = list(sec.children)
        while queue and len(out) < limit:
            cid = queue.pop(0)
            rel = view.by_section.get(cid)
            if rel is None or rel in seen:
                continue
            if view.body_spans(rel):
                child = SectionHits(rel, sh.doc_id, cid, (sh.order_key[0], doc.section(cid).order), [], sh.score,
                                    dict(sh.best), via_parent=sh.section_id, scope_score=sh.scope_score)
                out.append(child)
                seen.add(rel)
            else:
                queue = list(doc.section(cid).children) + queue
    return out


# --------------------------------------------------------------------------
# evidence granularity
# --------------------------------------------------------------------------

def _trim_lines(text, room):
    """Whole lines that fit in `room`; when not even the first line fits (a long paragraph is
    one store line), that line is cut at `room` characters instead of returning nothing."""
    if len(text) <= room:
        return text
    out, used = [], 0
    for line in text.split("\n"):
        add = len(line) + (1 if out else 0)
        if used + add > room:
            break
        out.append(line)
        used += add
    kept = "\n".join(out).strip()
    if not kept and room > 0:
        kept = text[:room].rstrip()
    return kept


def window_text(view, rel, hit_lines, n_lines, room, table_header=True):
    """Body lines within +-n_lines of the hit lines (lead of the section if no hit lines),
    grouped by unit; -> (text, unit_ids)."""
    units = view.body_units(rel)
    if not units:
        return "", []
    ls = view.lines(rel)
    if hit_lines:
        ranges = []
        for h in sorted(hit_lines):
            lo, hi = h - n_lines, h + n_lines
            if ranges and lo <= ranges[-1][1] + 1:
                ranges[-1][1] = max(ranges[-1][1], hi)
            else:
                ranges.append([lo, hi])
    else:
        ranges = [[units[0][2], units[0][2] + 2 * n_lines]]
    segments, uids = [], []
    used = 0
    for lo, hi in ranges:
        pieces = []
        for uid, _t, a, b in units:
            if b < lo or a > hi:
                continue
            s0, s1 = max(a, lo), min(b, hi)
            chunk = ls[s0 - 1:s1]
            if table_header and s0 > a and b > a:
                chunk = [ls[a - 1]] + chunk
            pieces.append(("\n".join(chunk), uid))
        if not pieces:
            continue
        seg_text = "\n\n".join(p for p, _ in pieces)
        sep = len(WINDOW_GAP) if segments else 0
        if used + sep + len(seg_text) > room:
            left = room - used - sep
            if left < 80:
                break
            seg_text = _trim_lines(seg_text, left)
            if not seg_text:
                break
            kept = [uid for p, uid in pieces if p.split("\n")[0][:60] in seg_text]
            segments.append(seg_text)
            uids.extend(u for u in kept if u not in uids)
            break
        segments.append(seg_text)
        used += sep + len(seg_text)
        uids.extend(u for _, u in pieces if u not in uids)
    return WINDOW_GAP.join(segments).strip(), uids


def source_lines(view, rel, uid, value, max_lines=2):
    """Body lines of the fact's source unit that contain the fact's number (e.g. the table row
    naming the subject the model left out of the metric). Verbatim corpus text."""
    toks = value_numbers(value)
    if not uid or not toks:
        return []
    variants = [number_variants(t) for t in toks]
    out = []
    for u, _t, a, b in view.body_units(rel):
        if u != uid:
            continue
        ls = view.lines(rel)[a - 1:b]
        for k, line in enumerate(ls):
            nums = text_number_set(line)
            if all(v & nums for v in variants):
                if k > 0 and len(ls) > 1 and " | " in ls[0] and ls[0] not in out:
                    out.append(ls[0])          # table header row, so the matched row can be read
                out.append(line)
                if len(out) >= max_lines + 1:
                    return out
    return out


def numeric_pattern_ids(patterns):
    """Patterns without letters (years, numbers): they narrow, but never alone select a fact line."""
    return {i for i, p in enumerate(patterns or []) if not any(ch.isalpha() for ch in p.replace("\\d", ""))}


def fact_candidates(sh, numeric):
    """Fact lines of a section matched by at least one non-numeric pattern, keeping only those that
    match the most distinct non-numeric patterns; ordered by all patterns matched, then line."""
    per = {}
    for ln, kind, pi, _ in sh.hits:
        if kind == "fact":
            per.setdefault(ln, set()).add(pi)
    elig = {ln: ps for ln, ps in per.items() if ps - numeric}
    if not elig:
        return []
    best = max(len(ps - numeric) for ps in elig.values())
    keep = [ln for ln, ps in elig.items() if len(ps - numeric) == best]
    return sorted(keep, key=lambda ln: (-len(elig[ln]), ln))


def fact_text(view, rel, fact_hit_lines, room, with_source=True):
    """Matched fact lines (in the given order), each followed by the verbatim source line(s) holding its number."""
    facts = view.fact_lines(rel)
    rows, uids, shown = [], [], set()
    used = 0
    for ln in fact_hit_lines:
        if ln not in facts:
            continue
        uid, line = facts[ln]
        block = [line]
        if with_source:
            parts = line.split(" | ")
            for src in source_lines(view, rel, uid, parts[1] if len(parts) > 1 else ""):
                if src not in shown and len(src) <= 400:
                    block.append("  < " + src)
        text = "\n".join(block)
        add = len(text) + (1 if rows else 0)
        if used + add > room:
            break
        rows.append(text)
        shown.update(block[1:])
        shown.update(s[4:] for s in block[1:])
        used += add
        if uid and uid not in uids:
            uids.append(uid)
    return "\n".join(rows), uids


# ---- fact_evidence = "context": the source text of each matched fact instead of the fact line ----

_SENT_SPLIT = re.compile(r"(?<=[.!?؟…])\s+")


def _is_table(lines):
    return len(lines) >= 2 and " | " in lines[0]


def _value_variants(value):
    toks = value_numbers(value)
    return [number_variants(t) for t in toks]


def _has_value(text, variants):
    nums = text_number_set(text)
    return bool(variants) and all(v & nums for v in variants)


def _overlap(words, text):
    return len(words & set(word_tokens(text))) if words else 0


def _pieces(text):
    """[(line_index, sentence)] of a non-table unit: each store line split into sentences
    (after . ! ? ؟ … followed by whitespace, so decimals keep their dot)."""
    out = []
    for li, line in enumerate(text.split("\n")):
        for s in _SENT_SPLIT.split(line):
            if s.strip():
                out.append((li, s.strip()))
    return out


def _clip_around(piece, variants, max_chars):
    """A sentence longer than max_chars cut to a window around the first number that is one of the value's."""
    if len(piece) <= max_chars:
        return piece
    pos = 0
    for m in NUM_RE.finditer(piece):
        if any(number_variants(m.group(0)) & v for v in variants):
            pos = m.start()
            break
    lo = max(0, pos - max_chars // 2)
    hi = min(len(piece), lo + max_chars)
    lo = max(0, hi - max_chars)
    return ("… " if lo > 0 else "") + piece[lo:hi].strip() + (" …" if hi < len(piece) else "")


def fact_source(units, uid, metric, value, opts):
    """Where a fact's value is stated in its section: (unit_id, mode, piece indexes) or None.
    units: [(uid, text, first_line, last_line)] (StoreView.body_units). The fact's own unit (chosen at index time)
    first; otherwise the unit holding the value with the most metric words (then document order).
    mode "table": {0 (header row)} + the matching rows; "whole": the unit; "pieces": matching sentence indexes."""
    variants = _value_variants(value)
    if not variants:
        return None
    words = set(word_tokens(metric))
    by_uid = {u: t for u, t, _, _ in units}
    src = uid if uid in by_uid and _has_value(by_uid[uid], variants) else None
    if src is None:
        best = None
        for k, (u, t, _, _) in enumerate(units):
            if _has_value(t, variants):
                key = (-_overlap(words, t), k)
                if best is None or key < best[0]:
                    best = (key, u)
        if best is None:
            return None
        src = best[1]
    text = by_uid[src]
    lines = text.split("\n")
    if _is_table(lines):
        rows = [k for k in range(1, len(lines)) if _has_value(lines[k], variants)]
        if not rows:            # the value is in the header row itself (e.g. a year column)
            return src, "table", {0}
        rows = _best(rows, lambda k: _overlap(words, lines[k]), int(opts["fact_context_max_rows"]))
        return src, "table", {0} | set(rows)
    if len(text) <= int(opts["fact_context_unit_chars"]):
        return src, "whole", None
    pieces = _pieces(text)
    hit = [i for i, (_, p) in enumerate(pieces) if _has_value(p, variants)]
    if not hit:                 # a range whose numbers sit in different sentences: every sentence with one of them
        hit = [i for i, (_, p) in enumerate(pieces) if any(v & text_number_set(p) for v in variants)]
    if not hit:
        return src, "whole", None
    hit = _best(hit, lambda i: _overlap(words, pieces[i][1]), int(opts["fact_context_max_sentences"]))
    return src, "pieces", set(hit)


def _best(idx, overlap, cap):
    """Candidates holding the value: when some share words with the metric, only those with the most shared
    words; otherwise all of them; at most `cap`, in document order."""
    scores = {i: overlap(i) for i in idx}
    top = max(scores.values()) if scores else 0
    keep = [i for i in idx if scores[i] == top] if top > 0 else list(idx)
    return keep[:cap]


def _render_unit(text, mode, sel, variants_of, opts):
    if mode == "whole":
        return text
    lines = text.split("\n")
    if mode == "table":
        return "\n".join(lines[k] for k in sorted(sel))
    pieces = _pieces(text)
    out, prev = "", None
    for i in sorted(sel):
        li, p = pieces[i]
        p = _clip_around(p, variants_of.get(i) or [], int(opts["fact_context_sentence_chars"]))
        if prev is None:
            out = p
        elif i == prev + 1:
            out += ("\n" if pieces[prev][0] != li else " ") + p
        else:
            out += " … " + p
        prev = i
    return out


def fact_context_text(view, rel, fact_hit_lines, room, opts, corpus=None):
    """fact_evidence="context": for the matched fact lines (in the given order), the verbatim source text holding
    each value (fact_source), every source shown once and in first-use order; a table's short caption unit (the
    unit right before it, kind caption) above the table; the bare fact line only when no unit holds the value.
    -> (text, unit_ids, stats{facts, sourced, fallback})"""
    facts = view.fact_lines(rel)
    units = view.body_units(rel)
    by_uid = {u: t for u, t, _, _ in units}
    pos = {u: k for k, (u, _, _, _) in enumerate(units)}
    entries = []               # [("unit", uid) | ("fact", line)] in first-use order
    sel = {}                   # uid -> (mode, set of piece/row indexes | None)
    variants_of = {}           # uid -> {piece index: value variants} (sentence clipping)
    stats = {"facts": 0, "sourced": 0, "fallback": 0}

    def caption_of(uid):
        k = pos.get(uid, 0)
        if k == 0 or corpus is None:
            return None
        cu = units[k - 1][0]
        try:
            kind = corpus.unit(cu).kind
        except KeyError:
            return None
        if kind == "caption" and len(units[k - 1][1]) <= int(opts["fact_context_caption_chars"]):
            return cu
        return None

    def render(entries_, sel_):
        blocks = []
        for kind, ref in entries_:
            if kind == "fact":
                blocks.append(ref)
                continue
            mode, s = sel_[ref]
            body = _render_unit(by_uid[ref], mode, s, variants_of.get(ref, {}), opts)
            cap = caption_of(ref) if mode == "table" else None
            if cap and cap not in sel_:
                body = by_uid[cap] + "\n" + body
            blocks.append(body)
        return "\n\n".join(blocks)

    text = ""
    for ln in fact_hit_lines:
        if ln not in facts:
            continue
        uid, line = facts[ln]
        parts = line.split(" | ")
        metric, value = parts[0], (parts[1] if len(parts) > 1 else "")
        found = fact_source(units, uid, metric, value, opts)
        new_entries, new_sel = list(entries), {k: (m, set(s) if s is not None else None) for k, (m, s) in sel.items()}
        if found is None:
            if ("fact", line) in new_entries:
                continue
            new_entries.append(("fact", line))
        else:
            src, mode, s = found
            if src not in new_sel:
                new_entries.append(("unit", src))
                new_sel[src] = (mode, s)
            else:
                old_mode, old = new_sel[src]
                if old_mode == "whole" or mode == "whole":
                    new_sel[src] = ("whole", None)
                else:
                    new_sel[src] = (old_mode, old | s)
            if mode == "pieces":
                vv = _value_variants(value)
                for i in s:
                    variants_of.setdefault(src, {}).setdefault(i, vv)
        candidate = render(new_entries, new_sel)
        if len(candidate) > room:
            break
        entries, sel, text = new_entries, new_sel, candidate
        stats["facts"] += 1
        stats["sourced" if found is not None else "fallback"] += 1
    uids = []
    for kind, ref in entries:
        if kind != "unit":
            continue
        cap = caption_of(ref) if sel[ref][0] == "table" else None
        for u in ([cap] if cap else []) + [ref]:
            if u not in uids:
                uids.append(u)
    return text, uids, stats


def section_span(corpus, doc_id, uids, normalized):
    if not normalized or not uids:
        return None
    try:
        doc = corpus.doc(doc_id)
        us = [doc.unit(u) for u in uids]
    except KeyError:
        return None
    return [min(u.span[0] for u in us), max(u.span[1] for u in us)]


def select_evidence(ranked, view, corpus, *, expect, granularity, facts_enabled, budget, min_item_chars, opts,
                    normalized=True, patterns=None, fact_evidence=None):
    """-> (items to pack in order, decisions, retrieved log items, ranked list after empty-section expansion)
    fact_evidence: "context" | "bare" (default: opts["fact_evidence"]); what a fact item shows, see defaults.py."""
    fact_evidence = fact_evidence or opts.get("fact_evidence", "context")
    if fact_evidence not in FACT_EVIDENCE:
        raise ValueError("fact_evidence must be one of %s" % (FACT_EVIDENCE,))
    n_lines = opts["window_lines"]
    numeric = numeric_pattern_ids(patterns)
    ranked = expand_empty(ranked, view, corpus, opts["max_sections"])
    items, decisions = [], []
    used = 0
    fact_sections = []

    def cost_of_header(doc_id, section_id):
        probe = RetrievedItem(text="", doc_id=doc_id, section_id=section_id)
        h = item_header(len(items) + 1, probe, corpus)
        return (2 if items else 0) + len(h) + 1

    def add(sh, kind, text, uids, rank, reason, extra=None):
        nonlocal used
        hc = cost_of_header(sh.doc_id, sh.section_id)
        whole = kind == "section" and not (extra and "cut_from_chars" in extra)
        span = section_span(corpus, sh.doc_id, uids, normalized) if whole else None   # only for verbatim slices
        meta = {"rel": sh.rel, "granularity": kind, "reason": reason,
                "matched": {str(pi): k for pi, k in sorted(sh.best.items())}}
        if sh.via_parent:
            meta["via_parent"] = sh.via_parent
        if extra:
            meta.update(extra)
        items.append(RetrievedItem(text=text, doc_id=sh.doc_id, section_id=sh.section_id, unit_ids=list(uids),
                                   kind=kind, score=round(sh.score, 4), rank=rank, span=span, meta=meta))
        used += hc + len(text)
        decisions.append({"section_id": sh.section_id, "kind": kind, "chars": len(text), "score": round(sh.score, 3),
                          "reason": reason, "rank": rank})

    for rank, sh in enumerate(ranked, 1):
        hc = cost_of_header(sh.doc_id, sh.section_id)
        room = budget - used - hc
        if room < min_item_chars:
            decisions.append({"section_id": sh.section_id, "kind": None, "reason": "budget_full", "rank": rank})
            break
        body = view.body_text(sh.rel)
        body_hits = sh.lines_of({"body"})
        fact_hits = fact_candidates(sh, numeric)
        if granularity == "always_section":
            if not body:
                continue
            units = view.body_units(sh.rel)
            if len(body) <= room:
                add(sh, "section", body, [u for u, _, _, _ in units], rank, "forced_section")
                continue
            # the section does not fit: cut it here exactly as pack_evidence would (text[:room-2] + " …"),
            # but derive unit ids from the store's own unit texts. pack_evidence re-derives them from the
            # NORMALISED corpus text, which silently loses every unit of a raw (no_normalization) store.
            kept = body[: room - 2].rstrip() + " …"
            uids = [u for u, t, _, _ in units if t.strip()[:60] in kept]
            add(sh, "section", kept, uids, rank, "forced_section_cut", extra={"cut_from_chars": len(body)})
            break               # budget used up; pack_evidence would include nothing after a cut item
        if granularity == "always_window":
            text, uids = window_text(view, sh.rel, body_hits, n_lines, room, opts["window_table_header"])
            if text:
                add(sh, "window", text, uids, rank, "forced_window" if body_hits else "forced_window_lead")
            continue
        # auto
        if expect == "number" and facts_enabled and fact_hits:
            if fact_evidence == "bare":
                text, uids = fact_text(view, sh.rel, fact_hits, room, with_source=opts.get("fact_source_lines", True))
                reason, extra = "number_question_fact_match", {"fact_evidence": "bare"}
            else:
                text, uids, fst = fact_context_text(view, sh.rel, fact_hits, room, opts, corpus)
                reason, extra = "number_question_fact_source", {"fact_evidence": "context", "fact_lines": fst["facts"],
                                                                 "fact_sourced": fst["sourced"],
                                                                 "fact_fallback": fst["fallback"]}
            if text:
                add(sh, "fact", text, uids, rank, reason, extra)
                decisions[-1].update({k: v for k, v in extra.items() if k != "fact_evidence"})
                fact_sections.append((rank, sh))
                continue
        if not body:
            continue
        if len(body) <= room:
            uids = [u for u, _, _, _ in view.body_units(sh.rel)]
            add(sh, "section", body, uids, rank, "fits_budget")
        else:
            text, uids = window_text(view, sh.rel, body_hits, n_lines, room, opts["window_table_header"])
            if text:
                add(sh, "window", text, uids, rank, "section_too_long" if body_hits else "section_too_long_lead")

    # auto: body context for the fact sections, while the budget lasts
    if granularity == "auto":
        for rank, sh in fact_sections:
            hc = cost_of_header(sh.doc_id, sh.section_id)
            room = budget - used - hc
            if room < min_item_chars:
                break
            body = view.body_text(sh.rel)
            if not body:
                continue
            if len(body) <= room:
                add(sh, "section", body, [u for u, _, _, _ in view.body_units(sh.rel)], rank, "fact_context_section")
                continue
            fact_uids = {view.fact_lines(sh.rel)[ln][0] for ln in sh.lines_of({"fact"}) if ln in view.fact_lines(sh.rel)}
            anchor = set(sh.lines_of({"body"}))
            for uid, _t, a, b in view.body_units(sh.rel):
                if uid in fact_uids:
                    anchor.add(a)
            text, uids = window_text(view, sh.rel, sorted(anchor), n_lines, room, opts["window_table_header"])
            if text:
                add(sh, "window", text, uids, rank, "fact_context_window")

    # retrieved log (RunRecord.retrieved_units) = exactly the items offered to pack_evidence, like B2/B5.
    # Ranked sections that did not fit the budget are NOT logged as retrieved (that would credit OURS
    # with whole sections the answer model never saw); their ids and scores are in the trace ranking.
    retrieved = list(items)
    return items, decisions, retrieved, ranked


def search(root, plan, doc_ids, corpus, cfg, opts):
    """grep + rank for one plan -> (ranked, info, view)"""
    view = StoreView(root)
    hits, cap = run_patterns(root, plan["patterns"], doc_ids, cfg, opts["max_hits_per_pattern"], view=view)
    ranked, info = rank_sections(hits, view, plan["scope"], corpus, opts, grep_cap=cap)
    info["patterns"] = list(plan["patterns"])
    return ranked, info, view
