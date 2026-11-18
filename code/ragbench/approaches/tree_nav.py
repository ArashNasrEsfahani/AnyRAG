"""B4 - tree navigation (PageIndex-style).

INDEX TIME (question-blind; phase="index"; every call goes through the shared client)
  * One navigation tree per document: a document root, the document's section tree,
    and two kinds of synthetic nodes, so that every choice list stays short and every
    leaf is readable within the evidence budget:
      part  - a section whose own text is longer than DEFAULTS["part_max_chars"] gets its
              own units split into consecutive parts of about part_target_chars (whole
              units; one oversized unit is split by lines, and a split table repeats its
              header line). Parts become the first children of that section.
      group - a node with more than DEFAULTS["max_fanout"] children gets its children
              wrapped in consecutive groups (recursively). This is what makes the flat
              extracted trees navigable (e.g. 360 top-level headings in one C report).
  * A 1-2 sentence summary for every node, written by the local model in the DOCUMENT's
    language, bottom-up. As in PageIndex, a node is summarised from its own text: a leaf
    or part from its passage, and a section that has subsections from its OWN text only
    (PageIndex's "prefix summary"; the subsections are shown to the navigator by title,
    see query time). A node without own text (a group, or a section whose own text was
    split into parts) is described from its children's titles and summaries. A node whose
    summarised text is already no longer than caps.tree_nav_summary_chars uses that text
    as its summary without a call (PageIndex does the same below its token threshold).
    One document-level summary per document, from the top-level entries, for corpus-level
    routing. (Before prompt b4-2026-09-15.2 a parent's prompt mixed its clipped own text with its
    children's summaries, and the 4B model mostly copied the first child's summary, so a
    chapter's own findings never reached the navigator.)
  * Summaries are cached in one sqlite KV file per variant, keyed by the exact prompt,
    model digest and generation options, so an interrupted build resumes and a rebuilt
    corpus set re-uses unchanged summaries. Each cache entry keeps the cost of the calls
    that produced it; <index_dir>/summary_stats.json reports both the calls made by this
    build and the original cost of every summary in the index (honest totals even when
    summaries came from the cache). The base class's build_stats.json has the ledgered
    calls of each build attempt.

QUERY TIME (phase="query")
  1. Documents in query.doc_scope. If there are more than max_docs (always the case for
     doc_scope == "corpus" with several documents), one routing call picks at most
     max_docs from numbered titles + document summaries.
  2. Each chosen document starts as one candidate: its root. Loop:
       - render all candidates exactly as pack_evidence renders them; if they fit
         evidence.max_chars, stop ("fits_budget");
       - otherwise open the largest candidate that still has children. A node with a
         single child and no own text is descended without a call. Otherwise one
         navigation call shows its children (number, title, summary, subsection count)
         and the model replies {"pick": [...]} with at most max_picks numbers (each entry also
         lists the titles of its own subsections, the part of the tree structure PageIndex's
         tree search shows the model), or
         {"pick": ["STOP"]} = keep the current section whole (not offered at a document
         root). Picked children replace the candidate;
       - stop when nothing can be opened ("no_expandable") or the call cap is reached
         ("call_cap").
     The reply is constrained with an Ollama JSON schema; a reply that is still unusable
     (no JSON, an unknown number, STOP where it is not allowed, empty) gets one repair
     call that shows the model its reply and the problem. If that fails too, the node is
     kept whole and the step is marked as a fallback.
       Opening only happens while it can still change the evidence (a candidate is not
       opened once better-ranked final items - taking their size, at most the rest of the
       budget, in rank order - leave less than a minimum item of room: "budget_filled").
  3. Evidence, in the navigator's rank order like B2/B3: the final candidates depth-first by
     pick order, and the own text (intro, key figures) of every section the walk opened
     right after that section's picked descendants and before its next-ranked sibling - the
     model chose those sections one level up, and opening them would otherwise discard
     their own text (DEFAULTS["backfill_opened_own_text"]). Items are taken whole while they
     fit; an item that does not fit the remaining room is NARROWED to its leading whole units
     (document order: a kept-whole section keeps its own introduction first) instead of being
     dropped - only when not even one unit (or min_item_chars) fits is it cut at a line
     boundary - and later, smaller items still fill what room is left. Every offered item
     then fits, so pack_evidence (the shared budget) includes them unchanged, followed by the
     shared answer prompt. (Before prompt b4-2026-09-15.2 the list was fit-first: whole-fitting items
     first and oversized ones deferred to the end, where the first was cut to whatever room
     remained or dropped.) AnswerResult.retrieved (-> RunRecord.retrieved_units,
     ret_hit/ret_recall) lists the same items, each clipped to the evidence budget, so a
     kept-whole chapter or document root does not claim every unit it spans (see
     retrieved_view); the trace names the clipped nodes.
  Call cap: caps.tree_nav_max_calls counts routing + navigation + repair calls + the
  answer call, so navigation gets at most tree_nav_max_calls - 1 calls. Every call is a
  trace step with its raw reply, tokens and wall time; the ledger scope in
  Approach.answer() is what RunRecord counts.

Self-test:
    .venv/Scripts/python.exe -m ragbench.approaches.tree_nav --offline
    .venv/Scripts/python.exe -m ragbench.approaches.tree_nav --live      (GPU; one report)
"""
import math
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from ..contracts import LANG_NAMES, AnswerResult, Approach, RetrievedItem, _units_in_cut, _verbatim_start, item_header
from ..llm import extract_json
from ..normalize import detect_lang, normalize
from ..util import KVCache, atomic_write_json, now_iso, read_json, safe_name, sha1_json, sha1_text

PROMPT_VERSION = "b4-2026-09-15.2"
# 2026-09-15.2: PageIndex-style prefix summaries (a section with subsections is summarised from its OWN text;
# its subsections are shown to the navigator by title instead), subsection titles in the navigation entries,
# rank-order evidence packing with unit-boundary narrowing (was fit-first).
SUMMARY_CACHE_VERSION = "b4-2026-09-15.1"   # summary cache key; the key also holds the exact messages, so an
                                            # unchanged leaf prompt keeps its cached summary across PROMPT_VERSION bumps

# Approach tunables that are not fairness-critical. The fairness-critical ones
# (evidence budget, call cap, summary length shown to the model, generation options)
# come from config.yaml.
DEFAULTS = {
    # query time
    "max_docs": 2,                      # documents chosen by the routing call
    "max_picks": 2,                     # children chosen per navigation call
    "doc_summary_chars": 400,           # document summary shown in the routing list
    "nav_num_predict": 64,
    "nav_prompt_max_chars": 12000,      # entry list is shortened (summaries clipped) to fit
    "nav_title_chars": 120,
    "repair_attempts": 1,
    "json_schema": True,                # constrain navigation replies with a JSON schema
    "backfill_opened_own_text": True,   # offer the own text of sections the walk opened (after their picked descendants)
    "nav_subtitles_chars": 240,         # subsection titles listed under a navigation entry (shrunk to fit the prompt)
    # index time (all in config_signature)
    "part_max_chars": 4000,             # own text above this is split into parts
    "part_target_chars": 3000,
    "max_fanout": 20,
    "summary_input_chars": 3000,        # passage shown to the summariser (leaf / part)
    "summary_own_chars_with_children": 1500,
    "summary_child_list_chars": 1800,
    "summary_child_chars": 120,
    "summary_words": 30,
    "doc_summary_words": 60,
    "summary_num_predict": 160,
    "doc_summary_num_predict": 220,
    "summary_lang_retry": True,         # one retry when the summary is not in the document language
}
INDEX_KEYS = ("part_max_chars", "part_target_chars", "max_fanout", "summary_input_chars",
              "summary_own_chars_with_children", "summary_child_list_chars", "summary_child_chars",
              "summary_words", "doc_summary_words", "summary_num_predict", "doc_summary_num_predict",
              "summary_lang_retry")
QUERY_KEYS = ("max_docs", "max_picks", "doc_summary_chars", "nav_num_predict", "nav_prompt_max_chars",
              "nav_title_chars", "repair_attempts", "json_schema", "backfill_opened_own_text", "nav_subtitles_chars")


# --------------------------------------------------------------------------
# text helpers
# --------------------------------------------------------------------------

def clip(text, n):
    """Whitespace-collapsed text of at most n characters, cut at a word boundary with '…'."""
    t = " ".join((text or "").split())
    if n is None or len(t) <= n:
        return t
    if n <= 1:
        return ""
    cut = t[: n - 1]
    sp = cut.rfind(" ")
    if sp >= int(n * 0.6):
        cut = cut[:sp]
    return cut.rstrip(" ,;:،؛") + "…"


_RE_LABEL = re.compile(r"^\s*(summary|description|خلاصه|توضیح|شرح)\s*[:：]\s*", re.I)


def clean_summary(text):
    t = (text or "").strip()
    t = re.sub(r"^```[a-zA-Z]*\s*|\s*```$", "", t).strip()
    t = _RE_LABEL.sub("", t)
    t = t.replace("**", "").replace("__", "").strip()
    t = t.strip("\"'“”«»").strip()
    return normalize(" ".join(t.split()))


# --------------------------------------------------------------------------
# navigation tree (pure; no model)
# --------------------------------------------------------------------------

def _node(nid, kind, section_id, title, parent, span, prefix=None, own_chars=0):
    return {"nid": nid, "kind": kind, "section_id": section_id, "title": title, "parent": parent,
            "children": [], "span": [int(span[0]), int(span[1])], "prefix": prefix, "own_chars": int(own_chars),
            "chars": 0, "path": [], "summary": None, "summary_src": None}


def node_text(doc, node):
    """The evidence text of a node: a verbatim slice of Document.text (a split table's
    later parts are prefixed with the table header line)."""
    a, b = node["span"]
    t = doc.text[a:b]
    return (node["prefix"] + "\n" + t) if node.get("prefix") else t


def _after_title(doc, s):
    """Offset just after a section's title line (the pack header already shows the path)."""
    a = s.span[0]
    if s.title and doc.text.startswith(s.title, a):
        a += len(s.title)
        while a < len(doc.text) and doc.text[a] == "\n":
            a += 1
    return a


def _section_body(doc, s):
    return doc.text[_after_title(doc, s):s.span[1]]


def _pieces_for_units(doc, unit_ids, target):
    """[(start, end, prefix, solo)] covering the units in order; a unit longer than target
    is split by lines into solo pieces (a table repeats its header on later pieces)."""
    out = []
    for uid in unit_ids:
        u = doc.unit(uid)
        if len(u.text) <= target:
            out.append((u.span[0], u.span[1], None, False))
            continue
        lines = u.text.split("\n")
        header = lines[0] if (u.kind == "table" and len(lines) > 1) else None
        pos = u.span[0]
        cur = None
        for line in lines:
            la, lb = pos, pos + len(line)
            pos = lb + 1
            if cur is None:
                cur = [la, lb]
            elif lb - cur[0] > target:
                out.append((cur[0], cur[1], header if cur[0] != u.span[0] else None, True))
                cur = [la, lb]
            else:
                cur[1] = lb
        if cur is not None:
            out.append((cur[0], cur[1], header if cur[0] != u.span[0] else None, True))
    return out


def _parts_from_pieces(pieces, target):
    parts = []
    for a, b, prefix, solo in pieces:
        if parts and not solo and not parts[-1]["solo"] and b - parts[-1]["span"][0] <= target:
            parts[-1]["span"][1] = b
        else:
            parts.append({"span": [a, b], "prefix": prefix, "solo": solo})
    return parts


def build_nav_tree(doc, params):
    """Navigation tree for one Document: {"doc_id", "root", "nodes": {nid: node}}.
    Node kinds: doc | section | part | group. Children are in document order and their
    spans are ordered, disjoint and inside the parent's span."""
    pmax, ptarget, fan = int(params["part_max_chars"]), int(params["part_target_chars"]), int(params["max_fanout"])
    nodes = {}

    def add(n):
        nodes[n["nid"]] = n
        return n["nid"]

    def group(kids, parent_nid, section_id):
        level = 0
        while len(kids) > fan:
            n_groups = math.ceil(len(kids) / fan)
            size = math.ceil(len(kids) / n_groups)
            new = []
            for gi in range(n_groups):
                members = kids[gi * size:(gi + 1) * size]
                if not members:
                    continue
                first, last = nodes[members[0]], nodes[members[-1]]
                title = "%s … %s" % (clip(first["title"], 50), clip(last["title"], 50))
                g = _node("%s@g%d.%d" % (parent_nid, level, gi), "group", section_id, title, parent_nid,
                          [first["span"][0], last["span"][1]])
                g["children"] = list(members)
                for m in members:
                    nodes[m]["parent"] = g["nid"]
                new.append(add(g))
            kids = new
            level += 1
        return kids

    def section_node(s, parent_nid):
        own_body = _section_body(doc, s)
        title = s.title or clip(own_body, 60) or s.local_id
        n = _node(s.id, "section", s.id, title, parent_nid, [_after_title(doc, s), s.subtree_span[1]],
                  own_chars=len(own_body.strip()))
        add(n)
        kids = []
        if len(own_body) > pmax and s.unit_ids:
            parts = _parts_from_pieces(_pieces_for_units(doc, s.unit_ids, ptarget), ptarget)
            if len(parts) >= 2:
                for i, p in enumerate(parts):
                    kids.append(add(_node("%s#part%d" % (s.id, i + 1), "part", s.id,
                                          "%s (%d/%d)" % (title, i + 1, len(parts)), s.id, p["span"],
                                          prefix=p["prefix"])))
                n["own_chars"] = 0          # the own text now lives in the parts
        for c in doc.children(s.id):
            kids.append(section_node(c, s.id))
        n["children"] = group(kids, s.id, s.id)
        return s.id

    root = _node(doc.id, "doc", None, doc.title or doc.id, None, [0, len(doc.text)])
    add(root)
    root["children"] = group([section_node(r, doc.id) for r in doc.roots()], doc.id, None)

    # paths (titles below the document root) and text sizes
    stack = [(doc.id, [])]
    while stack:
        nid, path = stack.pop()
        n = nodes[nid]
        n["path"] = path
        n["chars"] = len(node_text(doc, n))
        for c in n["children"]:
            stack.append((c, path + [nodes[c]["title"]]))
    return {"doc_id": doc.id, "root": doc.id, "nodes": nodes}


def post_order(tree):
    nodes, out = tree["nodes"], []
    stack = [(tree["root"], False)]
    while stack:
        nid, done = stack.pop()
        if done:
            out.append(nid)
            continue
        stack.append((nid, True))
        for c in reversed(nodes[nid]["children"]):
            stack.append((c, False))
    return out


def node_body(doc, node):
    """Text a childless node would be summarised from (section title stripped)."""
    return node_text(doc, node)


def tree_problems(doc, tree, params):
    """Structural invariants (used by the self-test)."""
    probs = []
    nodes = tree["nodes"]
    L = len(doc.text)
    for nid, n in nodes.items():
        a, b = n["span"]
        if not (0 <= a <= b <= L):
            probs.append("%s span out of range" % nid)
        if len(n["children"]) > int(params["max_fanout"]):
            probs.append("%s fanout %d" % (nid, len(n["children"])))
        prev_end = a
        for c in n["children"]:
            ca, cb = nodes[c]["span"]
            if ca < prev_end or cb > b or ca < a:
                probs.append("%s child %s span %s not inside/ordered in %s" % (nid, c, [ca, cb], [a, b]))
            prev_end = cb
            if nodes[c]["parent"] != nid:
                probs.append("%s parent pointer" % c)
    for s in doc.sections:
        parts = [nodes[k] for k in _flatten(nodes, s.id) if nodes[k]["kind"] == "part"]
        if parts:
            got = []
            for p in parts:
                got += [u.id for u in doc.units_overlapping(*p["span"])]
            if sorted(set(got)) != sorted(s.unit_ids):
                probs.append("%s parts do not cover own units exactly" % s.id)
    return probs


def _flatten(nodes, nid):
    """Descendants reachable through synthetic (part/group) nodes only."""
    out, stack = [], list(nodes[nid]["children"])
    while stack:
        c = stack.pop()
        if nodes[c]["kind"] in ("part", "group"):
            out.append(c)
            stack.extend(nodes[c]["children"])
    return out


# --------------------------------------------------------------------------
# prompts
# --------------------------------------------------------------------------

SUMMARY_SYSTEM = (
    "You write the short descriptions shown under the entries of a report's table of contents. "
    "A reader uses them to decide which section to open when looking for a specific fact.\n"
    "Write in {lang}. Write one or two sentences, at most {words} words. Say what the passage covers: "
    "its topic, and the entities, metrics and years it gives figures for. "
    "Use only what is in the passage. Reply with the description only, without a heading or quotation marks."
)
DOC_SUMMARY_SYSTEM = (
    "You write the short description of a whole report that is shown in a list of reports. "
    "A reader uses it to decide which report to open when looking for a specific fact.\n"
    "Write in {lang}. Write two or three sentences, at most {words} words. Say what the report covers: "
    "its subject, the country, sector or companies, the main metrics and the period. "
    "Use only what is given. Reply with the description only, without a heading or quotation marks."
)
LANG_RETRY = "Rewrite that description in {lang}. Reply with the description only."

NAV_SYSTEM = (
    "You find information in a long report by walking its table of contents, the way a reader opens "
    "the parts most likely to hold the answer.\n"
    "You see a question and the numbered entries under the current position, each with a title, a "
    "short description of its own text and the titles of its subsections. The question and the report may be "
    "in different languages.\n"
    "Choose at most {k} entries whose text most likely contains the answer, best first. "
    "Prefer a specific entry over a general one.{stop_rule}\n"
    "Reply with JSON only, for example {example}."
)
NAV_STOP_RULE = (" If the answer is more likely in the current section as a whole (for example in its own "
                 "introduction) than in any single entry, reply {\"pick\": [\"STOP\"]}.")
ROUTE_SYSTEM = (
    "You choose which reports to open to answer a question. You see the question and a numbered list of "
    "reports, each with a title and a short description. The question and the reports may be in "
    "different languages.\n"
    "Choose at most {k} reports that most likely contain the answer, best first.\n"
    "Reply with JSON only, for example {example}."
)


def _fill(template, **kw):
    out = template
    for k, v in kw.items():
        out = out.replace("{%s}" % k, str(v))
    return out


def _example(k):
    return '{"pick": ["2"]}' if k <= 1 else '{"pick": ["2"]} or {"pick": ["2", "5"]}'


def render_entries(entries, max_chars, summary_chars, title_chars, subtitle_chars=0):
    """entries: [(label, title, summary, tag)] or [(label, title, summary, tag, [subsection titles])].
    Subsection-title lists and summaries are clipped further (lists first, then summaries, down to
    titles only) until the list fits max_chars. Returns (text, summary_chars_used)."""
    n = int(summary_chars)
    m = int(subtitle_chars or 0)
    while True:
        lines = []
        for e in entries:
            label, title, summ, tag = e[:4]
            subs = e[4] if len(e) > 4 else None
            lines.append("[%s] %s%s" % (label, clip(title, title_chars), (" (%s)" % tag) if tag else ""))
            s = clip(summ, n) if n > 0 else ""
            if s:
                lines.append("    " + s)
            if subs and m > 0:
                lines.append("    " + clip("Subsections: " + "; ".join(clip(t, 60) for t in subs), m))
        text = "\n".join(lines)
        if len(text) <= max_chars or (n == 0 and m == 0):
            return text, n
        if m > 0:
            m = int(m * 0.6)
            if m < 60:
                m = 0
            continue
        n = int(n * 0.6)
        if n < 40:
            n = 0


def pick_schema(labels, allow_stop, k):
    enum = list(labels) + (["STOP"] if allow_stop else [])
    return {"type": "object",
            "properties": {"pick": {"type": "array", "items": {"type": "string", "enum": enum},
                                    "minItems": 1, "maxItems": int(k)}},
            "required": ["pick"]}


def parse_pick(text, labels, allow_stop, max_picks):
    """-> (picks: [label], stop: bool, error: str|None, note: str|None)."""
    obj = extract_json(text, None)
    if obj is None:
        return None, False, "the reply is not JSON", None
    if isinstance(obj, dict):
        raw = None
        for key in ("pick", "picks", "choice", "choices"):
            if key in obj:
                raw = obj[key]
                break
        if raw is None:
            return None, False, 'the JSON has no "pick" list', None
    elif isinstance(obj, list):
        raw = obj
    else:
        return None, False, 'the JSON has no "pick" list', None
    if isinstance(raw, (str, int, float)):
        raw = [raw]
    if not isinstance(raw, list):
        return None, False, '"pick" is not a list', None
    toks = []
    for x in raw:
        t = str(int(x)) if isinstance(x, float) and x == int(x) else str(x)
        toks.append(t.strip().strip("[]()#").strip())
    labels = [str(l) for l in labels]
    stop = any(t.upper() == "STOP" for t in toks)
    nums = [t for t in toks if t.upper() != "STOP"]
    bad = [t for t in nums if t not in labels]
    if bad:
        return None, False, "unknown entry number(s) %s; valid numbers are 1 to %d" % (", ".join(bad[:5]), len(labels)), None
    uniq = []
    for t in nums:
        if t not in uniq:
            uniq.append(t)
    if not uniq:
        if stop and allow_stop:
            return [], True, None, None
        if stop:
            return None, False, "STOP is not allowed here; pick at least one entry number", None
        return None, False, "the pick list is empty", None
    note = "STOP ignored because entry numbers were also given" if stop else None
    if len(uniq) > max_picks:
        note = ((note + "; ") if note else "") + "clamped %d picks to %d" % (len(uniq), max_picks)
        uniq = uniq[:max_picks]
    return uniq, False, None, note


def repair_message(error, allow_stop, k):
    ex = _example(k) + (' or {"pick": ["STOP"]}' if allow_stop else "")
    return ("That reply could not be used: %s. Reply again with JSON only, for example %s, "
            "using only entry numbers from the list above." % (error, ex))


# --------------------------------------------------------------------------
# query-time helpers
# --------------------------------------------------------------------------

@dataclass
class _Cand:
    doc: object
    nid: str
    rank: tuple
    final: bool = False


def rendered_size(items, corpus):
    """Characters pack_evidence would need for these items without cutting."""
    total, n = 0, 0
    for it in items:
        body = (it.text or "").strip()
        if not body:
            continue
        n += 1
        total += (2 if n > 1 else 0) + len(item_header(n, it, corpus)) + 1 + len(body)
    return total


def _line_cut(text, room):
    """At most `room` chars of text ending in " …", cut at a line break (else a space) in the last 40%."""
    t = text.strip()
    cut = t[: max(0, room - 2)]
    for sep in ("\n", " "):
        k = cut.rfind(sep)
        if k >= int(len(cut) * 0.6):
            cut = cut[:k]
            break
    return cut.rstrip() + " …"


def narrow_item(item, corpus, room, min_item_chars):
    """The item reduced to at most `room` characters of body, or None when room < min_item_chars.
    A verbatim item (span matching Document.text) keeps its LEADING WHOLE UNITS (in document order,
    so a kept-whole section keeps its own introduction first) and stays verbatim with exact unit ids.
    When fewer than min_item_chars of whole units fit (a single long unit, e.g. a table), or the item
    is not verbatim (a split-table part with a repeated header), the text is cut at a line boundary
    and its unit ids re-derived as pack_evidence would for a cut."""
    body = (item.text or "").strip()
    if room < min_item_chars or not body:
        return None
    if len(body) <= room:
        return item
    meta = dict(item.meta or {}, cut_from_chars=len(body))
    start = _verbatim_start(item, corpus)
    if start is not None:
        doc = corpus.doc(item.doc_id)
        end_item = start + len(body)
        want = set(item.unit_ids)
        end, kept = None, []
        for u in doc.units_overlapping(start, end_item):
            if u.span[0] < start or u.span[1] > end_item:
                continue
            if u.span[1] - start > room:
                break
            end = u.span[1]
            if u.id in want:
                kept.append(u.id)
        if end is not None and end - start >= min_item_chars and kept:
            text = doc.text[start:end].rstrip()
            meta.update(narrowed="leading_units", narrowed_units=len(kept))
            return RetrievedItem(text=text, doc_id=item.doc_id, section_id=item.section_id, unit_ids=kept,
                                 kind=item.kind, score=item.score, rank=item.rank, span=[start, start + len(text)],
                                 meta=meta)
    kept_text = _line_cut(body, room)
    meta.update(narrowed="line_cut")
    return RetrievedItem(text=kept_text, doc_id=item.doc_id, section_id=item.section_id,
                         unit_ids=_units_in_cut(item, kept_text, corpus), kind=item.kind, score=item.score,
                         rank=item.rank, span=None, meta=meta)


def pack_rank_order(items, corpus, budget, min_item_chars):
    """Rank-order packing (like B2/B3): walk the items in the navigator's order; take an item whole
    when it fits, otherwise narrow it to the remaining room (narrow_item) instead of dropping it,
    and keep going while at least a minimum item still fits. Room is counted exactly as
    pack_evidence renders ("[n] title > path" header, newline, blank-line separators), so
    pack_evidence then includes every returned item unchanged.
    -> (items, decisions [{nid, role, action, chars, from_chars}])"""
    out, dec, used = [], [], 0
    for it in items:
        body = (it.text or "").strip()
        m = it.meta or {}
        d = {"nid": m.get("nid"), "role": m.get("role"), "from_chars": len(body)}
        if not body:
            continue
        sep = 2 if out else 0
        room = budget - used - sep - len(item_header(len(out) + 1, it, corpus)) - 1
        new = narrow_item(it, corpus, room, min_item_chars)
        if new is None:
            dec.append(dict(d, action="dropped_budget_full", chars=0))
            continue
        nb = (new.text or "").strip()
        out.append(new)
        used += sep + len(item_header(len(out), new, corpus)) + 1 + len(nb)
        dec.append(dict(d, action="whole" if new is it else "narrowed_" + new.meta.get("narrowed", "cut"), chars=len(nb)))
    return out, dec


def retrieved_view(item, corpus, budget):
    """The item as logged in AnswerResult.retrieved (RunRecord.retrieved_units, the basis of
    ret_hit / ret_recall). A kept-whole node can be far larger than the whole evidence
    budget (a STOP'd chapter, or a document root kept after a failed or capped walk), and
    listing all of its units would make B4's retrieval recall trivially 1.0 for any gold in
    that document, and put whole documents into every run log. No single item can ever
    deliver more than `budget` characters, so the logged view is clipped to its first
    `budget` characters and its unit_ids re-derived. The evidence itself is packed from the
    unclipped items (pack_evidence cuts identically for every approach)."""
    body = item.text or ""
    if len(body) <= budget:
        return item
    keep = body[:budget]
    if item.span:
        doc = corpus.doc(item.doc_id)
        a = item.span[0]
        span = [a, a + len(keep)]
        unit_ids = [u.id for u in doc.units_overlapping(*span)]
    else:
        span = None
        unit_ids = _units_in_cut(item, keep, corpus)
    return RetrievedItem(text=keep, doc_id=item.doc_id, section_id=item.section_id, unit_ids=unit_ids,
                         kind=item.kind, score=item.score, rank=item.rank, span=span,
                         meta=dict(item.meta, retrieved_clipped_from_chars=len(body),
                                   retrieved_clipped_units=len(item.unit_ids)))


class _Navigator:
    """Makes the routing/navigation/repair calls, enforces the cap, records the trace."""

    def __init__(self, approach, cap):
        self.ap = approach
        self.cap = int(cap)
        self.calls = 0
        self.steps = []

    @property
    def exhausted(self):
        return self.calls >= self.cap

    def ask(self, kind, messages, labels, allow_stop, k, meta):
        """-> (picks, stop, status) with status ok | fallback | call_cap."""
        p = self.ap.params
        fmt = pick_schema(labels, allow_stop, k) if p["json_schema"] else "json"
        msgs = list(messages)
        attempt = 0
        while True:
            if self.exhausted:
                return None, False, ("call_cap" if attempt == 0 else "fallback")
            purpose = ("tree_route" if kind == "route" else "tree_nav") if attempt == 0 else "tree_repair"
            r = self.ap.client.chat(msgs, purpose=purpose, num_predict=p["nav_num_predict"], format=fmt)
            self.calls += 1
            picks, stop, err, note = parse_pick(r.text, labels, allow_stop, k)
            step = {"call": self.calls, "kind": kind if attempt == 0 else "repair", "purpose": purpose}
            step.update(meta if attempt == 0 else {key: meta[key] for key in ("doc_id", "node") if key in meta})
            step.update(raw=(r.text or "")[:300], ok=err is None, error=err, note=note, pick=picks, stop=stop,
                        prompt_chars=sum(len(m.get("content") or "") for m in msgs),
                        prompt_tokens=r.prompt_tokens, completion_tokens=r.completion_tokens,
                        wall_ms=round(r.wall_ms, 1), done_reason=r.done_reason, call_id=r.call_id)
            self.steps.append(step)
            if err is None:
                return picks, stop, "ok"
            attempt += 1
            if attempt > int(p["repair_attempts"]):
                return None, False, "fallback"
            msgs = msgs + [{"role": "assistant", "content": r.text or ""},
                           {"role": "user", "content": repair_message(err, allow_stop, k)}]


# --------------------------------------------------------------------------
# the approach
# --------------------------------------------------------------------------

class TreeNavigation(Approach):
    name = "B4"
    needs_gold = False

    def __init__(self, cfg=None, client=None, flags=None):
        super().__init__(cfg, client, flags)
        self.params = dict(DEFAULTS)

    def config_signature(self):
        sig = {k: self.params[k] for k in INDEX_KEYS}
        sig.update(prompt_version=PROMPT_VERSION, summary_chars=int(self.cfg.caps["tree_nav_summary_chars"]),
                   chat_digest=self.cfg.models.get("chat_digest"), generation=self.cfg.hash("generation"))
        return sig

    # ------------------------------------------------------------------ index
    def _cache_path(self):
        return self.cfg.path("index_root") / safe_name(self.variant) / "_summary_cache.sqlite"

    def _build_index(self, corpus_set, index_dir):
        index_dir = Path(index_dir)
        tdir = index_dir / "trees"
        sig = self.config_signature()
        agg = {"approach": self.variant, "corpus_set": corpus_set.name, "prompt_version": PROMPT_VERSION,
               "docs": 0, "trees_built": 0, "trees_reused": 0, "nodes": 0, "by_kind": {},
               "summaries": {"llm": 0, "verbatim": 0, "fallback": 0}, "lang_mismatch": 0, "lang_retries": 0,
               "this_build": {"calls": 0, "cache_hits": 0, "prompt_tokens": 0, "completion_tokens": 0, "wall_ms": 0.0},
               "original_cost": {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "wall_ms": 0.0},
               "per_doc": {}}
        cache = KVCache(self._cache_path())
        try:
            for doc in corpus_set.docs:
                path = tdir / ("%s.json" % safe_name(doc.id))
                dsig = sha1_json({"sig": sig, "text": sha1_text(doc.text), "sections": len(doc.sections),
                                  "units": len(doc.units)})
                tree = read_json(path)
                if tree and tree.get("sig") == dsig:
                    agg["trees_reused"] += 1
                else:
                    tree = build_nav_tree(doc, self.params)
                    tree["sig"] = dsig
                    tree["stats"] = self._summarise_tree(doc, tree, cache)
                    tree["built_at"] = now_iso()
                    atomic_write_json(path, tree)
                    agg["trees_built"] += 1
                    for k in agg["this_build"]:
                        agg["this_build"][k] = round(agg["this_build"][k] + tree["stats"]["this_build"][k], 1)
                st = tree["stats"]
                agg["docs"] += 1
                agg["nodes"] += st["nodes"]
                for k, v in st["by_kind"].items():
                    agg["by_kind"][k] = agg["by_kind"].get(k, 0) + v
                for k in agg["summaries"]:
                    agg["summaries"][k] += st["summaries"][k]
                agg["lang_mismatch"] += st["lang_mismatch"]
                agg["lang_retries"] += st["lang_retries"]
                for k in agg["original_cost"]:
                    agg["original_cost"][k] = round(agg["original_cost"][k] + st["original_cost"][k], 1)
                agg["per_doc"][doc.id] = {"nodes": st["nodes"], "llm_summaries": st["summaries"]["llm"],
                                          "calls": st["original_cost"]["calls"]}
        finally:
            cache.close()
        agg["written_at"] = now_iso()
        atomic_write_json(index_dir / "summary_stats.json", agg, indent=1)
        return self._read_trees(corpus_set, index_dir)

    def _load_index(self, corpus_set, index_dir):
        return self._read_trees(corpus_set, index_dir)

    def _read_trees(self, corpus_set, index_dir):
        docs = {}
        for doc in corpus_set.docs:
            path = Path(index_dir) / "trees" / ("%s.json" % safe_name(doc.id))
            tree = read_json(path)
            if tree is None:
                raise FileNotFoundError("B4 tree missing for %s in %s (delete build_stats.json to rebuild)"
                                        % (doc.id, index_dir))
            docs[doc.id] = tree
        return {"docs": docs}

    def _summarise_tree(self, doc, tree, cache):
        p = self.params
        nodes = tree["nodes"]
        sum_chars = int(self.cfg.caps["tree_nav_summary_chars"])
        st = {"nodes": len(nodes), "by_kind": {}, "summaries": {"llm": 0, "verbatim": 0, "fallback": 0},
              "lang_mismatch": 0, "lang_retries": 0,
              "this_build": {"calls": 0, "cache_hits": 0, "prompt_tokens": 0, "completion_tokens": 0, "wall_ms": 0.0},
              "original_cost": {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "wall_ms": 0.0}}
        for nid in post_order(tree):
            n = nodes[nid]
            st["by_kind"][n["kind"]] = st["by_kind"].get(n["kind"], 0) + 1
            if n["kind"] == "doc":
                msgs, npred, purpose = self._doc_summary_messages(doc, tree), p["doc_summary_num_predict"], "tree_doc_summary"
                fallback = doc.title
            else:
                body = self._summary_source(doc, n)
                if body is not None and len(" ".join(body.split())) <= sum_chars:
                    n["summary"], n["summary_src"] = (clip(body, sum_chars) or n["title"]), "verbatim"
                    st["summaries"]["verbatim"] += 1
                    continue
                body = body if body is not None else node_body(doc, n)
                msgs = self._node_summary_messages(doc, tree, n)
                npred = p["summary_num_predict"]
                purpose = "tree_group_summary" if n["kind"] == "group" else "tree_summary"
                fallback = clip(body, sum_chars) or n["title"]
            key = sha1_json({"v": SUMMARY_CACHE_VERSION, "model": self.cfg.models["chat"],
                             "digest": self.cfg.models.get("chat_digest"), "gen": self.cfg.hash("generation"),
                             "np": npred, "retry": bool(p["summary_lang_retry"]), "lang": doc.lang, "msgs": msgs})
            ent = cache.get(key)
            if ent is None:
                ent = self._summary_call(msgs, doc.lang, purpose, npred)
                cache.put(key, ent)
                for k in ("calls", "prompt_tokens", "completion_tokens", "wall_ms"):
                    st["this_build"][k] = round(st["this_build"][k] + ent["cost"][k], 1)
            else:
                st["this_build"]["cache_hits"] += 1
            for k in ("calls", "prompt_tokens", "completion_tokens", "wall_ms"):
                st["original_cost"][k] = round(st["original_cost"][k] + ent["cost"][k], 1)
            if ent["summary"]:
                n["summary"], n["summary_src"] = ent["summary"], "llm"
                st["summaries"]["llm"] += 1
            else:
                n["summary"], n["summary_src"] = fallback, "fallback"
                st["summaries"]["fallback"] += 1
            st["lang_mismatch"] += 0 if ent["lang_ok"] else 1
            st["lang_retries"] += 1 if ent["retried"] else 0
        return st

    def _summary_call(self, msgs, lang, purpose, npred):
        r = self.client.chat(msgs, purpose=purpose, num_predict=npred)
        cost = {"calls": 1, "prompt_tokens": r.prompt_tokens, "completion_tokens": r.completion_tokens,
                "wall_ms": round(r.wall_ms, 1)}
        s = clean_summary(r.text)
        lang_ok = bool(s) and detect_lang(s, default=lang) == lang
        retried = False
        if not lang_ok and self.params["summary_lang_retry"]:
            retried = True
            msgs2 = msgs + [{"role": "assistant", "content": r.text or ""},
                            {"role": "user", "content": _fill(LANG_RETRY, lang=LANG_NAMES[lang])}]
            r2 = self.client.chat(msgs2, purpose=purpose + "_retry", num_predict=npred)
            cost["calls"] += 1
            cost["prompt_tokens"] += r2.prompt_tokens
            cost["completion_tokens"] += r2.completion_tokens
            cost["wall_ms"] = round(cost["wall_ms"] + r2.wall_ms, 1)
            s2 = clean_summary(r2.text)
            if s2 and detect_lang(s2, default=lang) == lang:
                s, lang_ok = s2, True
            elif not s:
                s = s2
        return {"summary": s, "lang_ok": lang_ok, "retried": retried, "raw": (r.text or "")[:400],
                "done_reason": r.done_reason, "cost": cost, "created": now_iso()}

    def _path_str(self, doc, node):
        return " > ".join([doc.title] + list(node["path"])) if node["path"] else doc.title

    def _child_list(self, tree, node, limit):
        p = self.params
        nodes = tree["nodes"]
        lines, used = [], 0
        for c in node["children"]:
            cn = nodes[c]
            line = "- %s: %s" % (clip(cn["title"], 100), clip(cn["summary"] or "", p["summary_child_chars"]))
            if used + len(line) > limit:
                lines.append("- … (%d more)" % (len(node["children"]) - len(lines)))
                break
            lines.append(line)
            used += len(line) + 1
        return "\n".join(lines)

    def _summary_source(self, doc, node):
        """The text a node is summarised from, or None when it has no own text and is described
        from its children (groups; sections whose own text lives in parts or that have none).
        Leaves and parts: their passage. A section with subsections: its OWN text only (PageIndex
        prefix summary)."""
        if node["kind"] == "group":
            return None
        if not node["children"]:
            return node_body(doc, node)
        if node["kind"] == "section" and node["own_chars"] > 0:
            return _section_body(doc, doc.section(node["section_id"]))
        return None

    def _node_summary_messages(self, doc, tree, node):
        p = self.params
        system = _fill(SUMMARY_SYSTEM, lang=LANG_NAMES[doc.lang], words=p["summary_words"])
        head = "Report: %s\nSection: %s\n\n" % (doc.title, self._path_str(doc, node))
        if node["kind"] == "group":
            user = (head + "This entry groups consecutive sections of the report:\n"
                    + self._child_list(tree, node, p["summary_child_list_chars"] + p["summary_own_chars_with_children"]))
        elif node["children"]:
            own = self._summary_source(doc, node)
            if own is not None:
                # PageIndex prefix summary: the section's own text; the navigator sees its subsections by title
                user = head + "Passage (the section's own text):\n" + clip_block(own, p["summary_input_chars"])
            else:
                user = head + "Its subsections:\n" + self._child_list(tree, node, p["summary_child_list_chars"])
        else:
            user = head + "Passage:\n" + clip_block(node_text(doc, node), p["summary_input_chars"])
        return [{"role": "system", "content": system}, {"role": "user", "content": user}]

    def _doc_summary_messages(self, doc, tree):
        p = self.params
        system = _fill(DOC_SUMMARY_SYSTEM, lang=LANG_NAMES[doc.lang], words=p["doc_summary_words"])
        root = tree["nodes"][tree["root"]]
        user = "Report title: %s\n\nTable of contents:\n%s" % (doc.title, self._child_list(tree, root, 4000))
        return [{"role": "system", "content": system}, {"role": "user", "content": user}]

    # ------------------------------------------------------------------ query
    def _item(self, doc, node, rank=None):
        a, b = node["span"]
        return RetrievedItem(text=node_text(doc, node), doc_id=doc.id, section_id=node["section_id"],
                             unit_ids=[u.id for u in doc.units_overlapping(a, b)],
                             kind="section" if node["kind"] == "section" else "node", rank=rank,
                             span=None if node.get("prefix") else [a, b],
                             meta={"nid": node["nid"], "nav_kind": node["kind"], "title": node["title"]})

    def _answer(self, query, index):
        p = self.params
        corpus = index.corpus
        trees = index.payload["docs"]
        budget = int(self.cfg.evidence["max_chars"])
        nav = _Navigator(self, max(0, int(self.cfg.caps["tree_nav_max_calls"]) - 1))
        docs = [d for d in self.docs_in_scope(query, index) if d.id in trees]
        trace = {"docs_in_scope": [d.id for d in docs], "budget": budget, "nav_call_cap": nav.cap,
                 "routing": None, "steps": nav.steps}

        def node_of(c):
            return trees[c.doc.id]["nodes"][c.nid]

        # 1. corpus-level routing
        chosen = docs
        if len(docs) > p["max_docs"]:
            k = int(p["max_docs"])
            entries = [(str(i + 1), d.title, trees[d.id]["nodes"][d.id]["summary"], None) for i, d in enumerate(docs)]
            text, used = render_entries(entries, p["nav_prompt_max_chars"], p["doc_summary_chars"], p["nav_title_chars"])
            msgs = [{"role": "system", "content": _fill(ROUTE_SYSTEM, k=k, example=_example(k))},
                    {"role": "user", "content": "Question: %s\n\nReports:\n%s" % (query.text.strip(), text)}]
            labels = [e[0] for e in entries]
            meta = {"doc_id": None, "node": None, "summary_chars_shown": used,
                    "options": [[e[0], d.id, clip(d.title, 80)] for e, d in zip(entries, docs)]}
            picks, _stop, status = nav.ask("route", msgs, labels, False, k, meta)
            if status == "ok":
                chosen = [docs[int(l) - 1] for l in picks]
            else:
                chosen = docs[:k]
            trace["routing"] = {"status": status, "chosen": [d.id for d in chosen]}

        # 2. walk the trees
        cands = [_Cand(doc=d, nid=d.id, rank=(i,)) for i, d in enumerate(chosen)]
        opened = []                  # (rank, doc, node) of every node the walk opened
        stop_reason = "no_documents" if not cands else None
        while cands:
            cands.sort(key=lambda c: c.rank)
            size = rendered_size([self._item(c.doc, node_of(c)) for c in cands], corpus)
            if size <= budget:
                stop_reason = "fits_budget"
                break
            openable = [c for c in cands if not c.final and node_of(c)["children"]]
            # Only open a candidate whose children could still reach the evidence. Evidence is
            # ordered fit-first (see pack_order), so walk in rank order counting the room taken
            # by better-ranked items that are final and fit whole; opening is useless once less
            # than a minimum item is left at that rank.
            useful, used_room = [], 0
            min_item = int(self.cfg.evidence["min_item_chars"])
            for cc in cands:
                if cc in openable:
                    # an item reaches the evidence only if min_item_chars of its body fit after the
                    # separator, its header and the newline; a child's header is at least as long as
                    # this node's (its path only grows).
                    hdr = len(item_header(len(cands), self._item(cc.doc, node_of(cc)), corpus))
                    if budget - used_room >= 2 + hdr + 1 + 2 + min_item:
                        useful.append(cc)
                else:
                    # rank-order packing: a better-ranked final item takes its size, at most the rest
                    need = rendered_size([self._item(cc.doc, node_of(cc))], corpus) + 2
                    used_room += min(need, max(0, budget - used_room))
            if not useful:
                stop_reason = "budget_filled" if openable else "no_expandable"
                break
            c = max(useful, key=lambda c: (node_of(c)["chars"], tuple(-x for x in c.rank)))
            n = node_of(c)
            nodes = trees[c.doc.id]["nodes"]
            kids = n["children"]
            base = {"doc_id": c.doc.id, "node": n["nid"], "node_kind": n["kind"], "path": self._path_str(c.doc, n),
                    "node_chars": n["chars"], "candidates_chars": size}
            if len(kids) == 1 and n["own_chars"] == 0:
                nav.steps.append(dict(base, call=None, kind="auto", pick=[kids[0]], note="single child, no own text"))
                opened.append((c.rank, c.doc, n))
                cands.remove(c)
                cands.append(_Cand(doc=c.doc, nid=kids[0], rank=c.rank + (0,)))
                continue
            if nav.exhausted:
                stop_reason = "call_cap"
                break
            allow_stop = n["kind"] != "doc"
            k = int(p["max_picks"])
            entries = []
            for j, kid in enumerate(kids):
                kn = nodes[kid]
                tag = None
                if kn["children"]:
                    tag = "%d %s" % (len(kn["children"]), "sections" if kn["kind"] in ("group", "doc") else "subsections")
                subs = [nodes[g]["title"] for g in kn["children"] if nodes[g]["kind"] != "part"]
                entries.append((str(j + 1), kn["title"], kn["summary"], tag, subs))
            text, used = render_entries(entries, p["nav_prompt_max_chars"], int(self.cfg.caps["tree_nav_summary_chars"]),
                                        p["nav_title_chars"], p["nav_subtitles_chars"])
            system = _fill(NAV_SYSTEM, k=k, example=_example(k), stop_rule=NAV_STOP_RULE if allow_stop else "")
            where = "the top level of the report" if n["kind"] == "doc" else self._path_str(c.doc, n)
            user = "Question: %s\n\nReport: %s\nCurrent position: %s\n" % (query.text.strip(), c.doc.title, where)
            if n["kind"] != "doc" and n.get("summary"):
                user += "About the current position: %s\n" % clip(n["summary"], int(self.cfg.caps["tree_nav_summary_chars"]))
            user += "\nEntries:\n" + text
            msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
            labels = [e[0] for e in entries]
            meta = dict(base, allow_stop=allow_stop, summary_chars_shown=used,
                        options=[[e[0], kid, clip(nodes[kid]["title"], 80)] for e, kid in zip(entries, kids)])
            picks, stop, status = nav.ask("expand", msgs, labels, allow_stop, k, meta)
            last = nav.steps[-1]
            if status == "ok" and not stop:
                chosen_nids = [kids[int(l) - 1] for l in picks]
                last["decision"] = {"open": chosen_nids}
                opened.append((c.rank, c.doc, n))
                cands.remove(c)
                for j, nid in enumerate(chosen_nids):
                    cands.append(_Cand(doc=c.doc, nid=nid, rank=c.rank + (j,)))
            else:
                c.final = True
                last["decision"] = {"keep_whole": n["nid"], "status": "stop" if stop else status}
                if status == "call_cap":
                    stop_reason = "call_cap"
                    break

        # 3. evidence (rank order, oversized items narrowed) and the shared answer call
        cands.sort(key=lambda c: c.rank)
        keyed = []
        for c in cands:
            it = self._item(c.doc, node_of(c))
            it.meta.update(final=c.final, nav_rank=list(c.rank), role="picked")
            keyed.append((tuple(c.rank), it))
        backfill = []
        if p["backfill_opened_own_text"]:
            for rank, doc, n in sorted(opened, key=lambda x: x[0]):
                if n["kind"] == "section" and n["own_chars"] > 0:
                    s = doc.section(n["section_id"])
                    a = _after_title(doc, s)
                    it = RetrievedItem(text=doc.text[a:s.span[1]], doc_id=doc.id, section_id=s.id,
                                       unit_ids=list(s.unit_ids), kind="section", span=[a, s.span[1]],
                                       meta={"nid": n["nid"], "nav_kind": "section_own_text", "title": n["title"],
                                             "nav_rank": list(rank), "role": "opened_own_text"})
                    backfill.append(it)
                    # after every picked descendant of this section (rank + (j, ...)), before its next sibling
                    keyed.append((tuple(rank) + (10 ** 6,), it))
        ordered = [it for _k, it in sorted(keyed, key=lambda x: x[0])]
        for i, it in enumerate(ordered):
            it.rank = i + 1
        min_item = int(self.cfg.evidence["min_item_chars"])
        packed, pack_dec = pack_rank_order(ordered, corpus, budget, min_item)
        evidence = self.pack(packed, index)
        answer = self.generate_answer(query, evidence)
        retrieved = [retrieved_view(it, corpus, budget) for it in ordered]
        repair_calls = sum(1 for s in nav.steps if s.get("kind") == "repair")
        fallbacks = sum(1 for s in nav.steps if (s.get("decision") or {}).get("status") == "fallback")
        stats = {"repair_calls": repair_calls, "fallback_steps": fallbacks,
                 "stop_steps": sum(1 for s in nav.steps if (s.get("decision") or {}).get("status") == "stop"),
                 "narrowed_items": sum(1 for d in pack_dec if d["action"].startswith("narrowed")),
                 "dropped_items": sum(1 for d in pack_dec if d["action"].startswith("dropped")),
                 "call_cap": int(stop_reason == "call_cap")}
        trace.update(prompt_version=PROMPT_VERSION, params={k: p[k] for k in QUERY_KEYS},
                     stop_reason=stop_reason, nav_calls=nav.calls, answer_calls=1, stats=stats,
                     candidates=[{"nid": c.nid, "doc_id": c.doc.id, "kind": node_of(c)["kind"], "rank": list(c.rank),
                                  "chars": node_of(c)["chars"], "kept_whole": c.final} for c in cands],
                     backfill=[it.meta["nid"] for it in backfill],
                     pack_order=[[it.meta["nid"], it.meta["role"]] for it in ordered], packing=pack_dec,
                     candidates_rendered_chars=rendered_size([it for _k, it in keyed if it.meta.get("role") == "picked"], corpus),
                     evidence_chars=evidence.chars,
                     evidence_items=evidence.n_included, evidence_cut=evidence.last_item_cut,
                     retrieved_clipped=[it.meta["nid"] for it in retrieved if "retrieved_clipped_from_chars" in it.meta])
        return AnswerResult(answer_text=answer, evidence=evidence, retrieved=retrieved, trace=trace)


def clip_block(text, n):
    """Clip a multi-line passage to n characters, keeping line breaks."""
    t = (text or "").strip()
    if len(t) <= n:
        return t
    cut = t[:n]
    nl = cut.rfind("\n")
    if nl >= int(n * 0.6):
        cut = cut[:nl]
    return cut.rstrip() + "\n…"


# --------------------------------------------------------------------------
# self-test
# --------------------------------------------------------------------------

# Self-test fixtures. The original fixtures were questions on a report of the development corpus A, which is not
# part of the public release; these neutral placeholders keep the structure (the self-test needs corpus A to run).
HANDMADE = [
    {"id": "b4-smoke-q1", "answer_type": "numeric", "type": "lookup", "answer_value": 1.0, "answer_unit": "$M",
     "question_en": "<placeholder question 1>", "question_fa": "<پرسش نمونهٔ ۱>",
     "answer_en": "$1.0M", "answer_fa": "$1.0M",
     "gold": ["sec1/sec1/table1#table.0", "front/front#takeaway.1"]},
    {"id": "b4-smoke-q2", "answer_type": "numeric", "type": "lookup", "answer_value": 2.0, "answer_unit": "%",
     "question_en": "<placeholder question 2>", "question_fa": "<پرسش نمونهٔ ۲>",
     "answer_en": "2.0%", "answer_fa": "2.0%",
     "gold": ["sec2/sec2/table2#table.0", "sec2/sec2#takeaway.0"]},
    {"id": "b4-smoke-q3", "answer_type": "numeric", "type": "lookup", "answer_value": 3, "answer_unit": "companies",
     "question_en": "<placeholder question 3>", "question_fa": "<پرسش نمونهٔ ۳>",
     "answer_en": "3", "answer_fa": "3",
     "gold": ["sec2/sec2#lede", "sec2/sec2/kpis#kpi.0", "front/front#takeaway.2"]},
]


def handmade_questions(report="rpt-b"):
    from ..contracts import Question
    out = []
    for h in HANDMADE:
        gold = {l: ["A.%s.%s::%s" % (report, l, g) for g in h["gold"]] for l in ("fa", "en")}
        out.append(Question(id=h["id"], corpus="A", question_fa=h["question_fa"], question_en=h["question_en"],
                            answer_fa=h["answer_fa"], answer_en=h["answer_en"], answer_type=h["answer_type"],
                            gold_units=gold, doc_scope=["A.%s.en" % report], type=h["type"],
                            answer_value=h["answer_value"], answer_unit=h["answer_unit"], meta={"source": "b4 self-test"}))
    return out


_RESULTS = []


def _check(name, cond, detail=""):
    _RESULTS.append((name, bool(cond)))
    print("  [%s] %s%s" % ("ok" if cond else "FAIL", name, (" - " + str(detail)) if detail else ""), flush=True)


def _scripted_client(cfg, ledger, responder):
    from ..llm import OllamaClient

    class ScriptedClient(OllamaClient):
        def _post(self, endpoint, payload):
            text = responder(payload)
            data = {"message": {"content": text}, "prompt_eval_count": sum(len(m["content"]) for m in payload["messages"]) // 4,
                    "eval_count": max(1, len(text) // 4), "done_reason": "stop"}
            return data, 1.0, 0.0, 1

    c = ScriptedClient(cfg, ledger)
    c._flock = None
    return c


def _entries(user):
    return re.findall(r"^\[(\d+)\] (.*)$", user, flags=re.M)


class _Responder:
    """Offline stand-in for the model: summaries echo the passage start; navigation picks
    entries whose title contains a keyword, with scripted failure modes."""

    def __init__(self):
        self.keywords = []
        self.garbage_next = 0       # this many navigation replies are unusable
        self.stop_at = None         # substring of the current position that triggers STOP
        self.reverse = False        # return matching picks last-first
        self.seen = []
        self.seen_users = []

    def __call__(self, payload):
        msgs = payload["messages"]
        system, user = msgs[0]["content"], msgs[-1]["content"]
        self.seen.append((system[:40], payload.get("format") is not None))
        self.seen_users.append((system[:40], payload.get("format") is not None, user))
        if "descriptions" in system or "description of a whole report" in system:
            if user.startswith("Rewrite"):
                return "این بخش دربارهٔ داده‌های گزارش است."
            if "Persian" in system and not any(m["role"] == "assistant" for m in msgs):
                return "An English summary that should trigger the language retry."
            return "Covers " + clip(user.split("\n\n", 1)[-1], 80)
        if user.startswith("That reply could not be used") and self.garbage_next <= 0:
            return '{"pick": ["1"]}'
        if self.garbage_next > 0:
            self.garbage_next -= 1
            return "I think entry two looks best."
        entries = _entries(user)
        pos = re.search(r"^Current position: (.*)$", user, flags=re.M)
        if self.stop_at and pos and self.stop_at in pos.group(1):
            return '{"pick": ["STOP"]}'
        picks = [num for num, title in entries if any(k.lower() in title.lower() for k in self.keywords)]
        if self.reverse:
            picks.reverse()
        return '{"pick": %s}' % (str((picks or ["1"])[:2]).replace("'", '"'))


def _offline():
    import tempfile
    from ..config import load_config
    from ..contracts import Question, Setting, pack_evidence
    from ..corpus import load_set
    from ..llm import Ledger

    print("B4 offline self-test")
    base_cfg = load_config()
    params = dict(DEFAULTS)

    # 1. tree invariants on every loaded corpus set, and the summary-call estimate
    for name in ("A_en", "A_fa", "B", "C"):
        cs = load_set(name)
        probs, nodes, llm, big_leaves, kinds = [], 0, 0, 0, {}
        for doc in cs.docs:
            tree = build_nav_tree(doc, params)
            probs += tree_problems(doc, tree, params)
            for n in tree["nodes"].values():
                nodes += 1
                kinds[n["kind"]] = kinds.get(n["kind"], 0) + 1
                verb = (not n["children"]) and len(" ".join(node_text(doc, n).split())) <= base_cfg.caps["tree_nav_summary_chars"]
                llm += 0 if verb else 1
                if not n["children"] and n["chars"] > base_cfg.evidence["max_chars"]:
                    big_leaves += 1
        _check("tree invariants %s (%d docs)" % (name, len(cs)), not probs, probs[:3] if probs else
               "%d nodes %s, ~%d summary calls, %d leaves over budget" % (nodes, kinds, llm, big_leaves))

    # 2. parser
    L = ["1", "2", "3"]
    _check("parse ok", parse_pick('{"pick": ["2", "3"]}', L, True, 2)[:3] == (["2", "3"], False, None))
    _check("parse fenced + ints", parse_pick('```json\n{"pick": [3]}\n```', L, True, 2)[0] == ["3"])
    _check("parse STOP", parse_pick('{"pick": ["STOP"]}', L, True, 2)[:3] == ([], True, None))
    _check("parse STOP not allowed", parse_pick('{"pick": ["STOP"]}', L, False, 2)[2] is not None)
    _check("parse unknown number", parse_pick('{"pick": ["7"]}', L, True, 2)[2] is not None)
    _check("parse not JSON", parse_pick("entry 2", L, True, 2)[2] is not None)
    r = parse_pick('{"pick": ["1", "2", "3"]}', L, True, 2)
    _check("parse clamps", r[0] == ["1", "2"] and "clamped" in (r[3] or ""))

    # 3. scripted index build + navigation on one report
    tmp = Path(tempfile.mkdtemp(prefix="b4_offline_"))
    cfg = base_cfg.copy_with({"paths": {"index_root": str(tmp / "indexes")}})
    ledger = Ledger(path=None)
    resp = _Responder()
    client = _scripted_client(cfg, ledger, resp)
    ap = TreeNavigation(cfg, client)
    en = load_set("A_en").subset(["A.rpt-b.en"])
    h = ap.build_index(en)
    st = read_json(h.path / "summary_stats.json")
    _check("index built", h.stats.get("complete") and st["docs"] == 1, "calls %s, summaries %s" % (h.stats["totals"]["llm_calls"], st["summaries"]))
    _check("index calls ledgered = summary calls", h.stats["totals"]["llm_calls"] == st["this_build"]["calls"])
    ap2 = TreeNavigation(cfg, client)
    before = len(ledger.records)
    h2 = ap2.build_index(en)
    _check("index reload makes no calls", len(ledger.records) == before and h2.payload["docs"].keys() == h.payload["docs"].keys())
    ap2._indexes = {}
    import shutil
    shutil.rmtree(h.path)
    before = len(ledger.records)
    h3 = ap2.build_index(en)
    st3 = read_json(h3.path / "summary_stats.json")
    _check("rebuild re-uses summary cache", len(ledger.records) == before and st3["this_build"]["cache_hits"] > 0
           and st3["original_cost"]["calls"] == st["original_cost"]["calls"], st3["this_build"])

    qs = handmade_questions()
    q1 = qs[0]
    resp.keywords = ["money goes to", "typically costs"]
    rec = ap2.answer(q1, "en-en")
    tr = rec.trace
    _check("answer ran", rec.error is None, rec.error)
    _check("llm_calls == nav calls + 1 (ledger)", rec.llm_calls == tr["nav_calls"] + 1, (rec.llm_calls, tr["nav_calls"]))
    _check("within call cap", rec.llm_calls <= cfg.caps["tree_nav_max_calls"])
    _check("evidence within budget", 0 < rec.evidence_chars <= cfg.evidence["max_chars"], rec.evidence_chars)
    _check("navigated to the gold section", any("c-size" in u for u in rec.evidence_units),
           [s.get("decision") for s in tr["steps"]])
    _check("steps recorded per call", sum(1 for s in tr["steps"] if s.get("call")) == tr["nav_calls"])
    _check("navigation used JSON schema", all(f for s, f in resp.seen if s.startswith("You find")))

    resp.garbage_next = 1
    rec = ap2.answer(q1, "en-en")
    kinds = [s["kind"] for s in rec.trace["steps"]]
    _check("repair call after unusable reply", "repair" in kinds and rec.llm_calls == rec.trace["nav_calls"] + 1, kinds)

    resp.garbage_next = 2
    rec = ap2.answer(q1, "en-en")
    dec = [s.get("decision") for s in rec.trace["steps"] if s.get("decision")]
    _check("fallback keeps node whole after failed repair", dec and dec[0].get("status") == "fallback", dec[:1])
    doc_units = len(en.doc("A.rpt-b.en").units)
    ret_units = {u for it in rec.retrieved_units for u in it["unit_ids"]}
    _check("kept-whole root is clipped to the budget in retrieved_units (no trivial ret_recall)",
           rec.trace["retrieved_clipped"] == ["A.rpt-b.en"]
           and all(len(it["text"]) <= cfg.evidence["max_chars"] for it in rec.retrieved_units)
           and 0 < len(ret_units) < doc_units and set(rec.evidence_units) <= ret_units,
           (rec.trace["retrieved_clipped"], len(ret_units), doc_units))
    _check("trace records prompt version and query params", rec.trace.get("prompt_version") == PROMPT_VERSION
           and set(rec.trace.get("params", {})) == set(QUERY_KEYS))
    resp.garbage_next = 0

    resp.keywords = ["semiconductors"]
    resp.stop_at = "Semiconductors and hardware"
    rec = ap2.answer(qs[2], "en-en")
    dec = [s.get("decision") for s in rec.trace["steps"] if s.get("decision")]
    _check("STOP keeps the chapter whole", any(d.get("status") == "stop" for d in dec) and
           any("chips/chips#lede" in u for u in rec.evidence_units), dec)
    # a large kept-whole chapter ranked first: rank order keeps it first, NARROWED to its leading whole units
    resp.keywords = ["<section title>", "money goes to", "typically costs"]
    resp.stop_at = "Vehicles, batteries and autonomy"
    resp.reverse = True
    rec = ap2.answer(q1, "en-en")
    po = rec.trace["pack_order"]
    pk = rec.trace["packing"]
    first_ev = rec.evidence_items[0] if rec.evidence_items else {}
    _check("rank order: the navigator's first pick comes first", po and pk
           and rec.trace["candidates"][0]["nid"] == po[0][0] == pk[0]["nid"] and pk[0]["action"] != "dropped_budget_full"
           and first_ev.get("section_id") == rec.retrieved_units[0]["section_id"],
           (po[:3], pk[:2], first_ev.get("section_id")))
    big = [d for d in pk if d["action"].startswith("narrowed")]
    _check("an oversized kept-whole pick is narrowed, not dropped", big and big[0]["chars"] > 0
           and rec.evidence_chars <= cfg.evidence["max_chars"] and not rec.trace["evidence_cut"]
           and all(d["action"] != "dropped_budget_full" or d["from_chars"] > 0 for d in pk), pk)
    _check("opened section own text offered as backfill", any(r == "opened_own_text" for _n, r in po), rec.trace["backfill"])
    ranks = [tuple(it["meta"]["nav_rank"]) + ((10 ** 6,) if it["meta"]["role"] == "opened_own_text" else ())
             for it in rec.retrieved_units]
    _check("backfill sits after the section's picks and before the next-ranked pick", ranks == sorted(ranks), ranks)
    resp.stop_at = None
    resp.reverse = False

    # narrowing keeps whole leading units of a verbatim section and exact unit ids
    doc_en = en.doc("A.rpt-b.en")
    tr_en = h2.payload["docs"]["A.rpt-b.en"]
    bigsec = max((n for n in tr_en["nodes"].values() if n["kind"] == "section" and n["children"] and not n.get("prefix")),
                 key=lambda n: n["chars"])
    bit = ap2._item(doc_en, bigsec)
    nar = narrow_item(bit, en, 2500, base_cfg.evidence["min_item_chars"])
    first_u = doc_en.units_overlapping(*bit.span)[0]
    _check("narrow_item: leading whole units, verbatim, exact ids", nar is not None and nar.meta.get("narrowed") == "leading_units"
           and len(nar.text) <= 2500 and nar.text == doc_en.text[nar.span[0]:nar.span[1]] and nar.unit_ids[0] == first_u.id
           and all(doc_en.unit(u).span[1] <= nar.span[1] for u in nar.unit_ids) and nar.meta["cut_from_chars"] == len(bit.text.strip()),
           (bigsec["nid"], len(nar.text) if nar else None, nar.unit_ids[:3] if nar else None))
    _check("narrow_item: no room for a minimum item -> None; fits -> unchanged",
           narrow_item(bit, en, base_cfg.evidence["min_item_chars"] - 1, base_cfg.evidence["min_item_chars"]) is None
           and narrow_item(bit, en, len(bit.text) + 10, base_cfg.evidence["min_item_chars"]) is bit)
    packed, dec = pack_rank_order([bit, ap2._item(doc_en, tr_en["nodes"][tr_en["nodes"][bigsec["nid"]]["children"][0]])], en,
                                  3000, base_cfg.evidence["min_item_chars"])
    pe = pack_evidence(packed, en, budget=3000, cfg=base_cfg)
    _check("pack_rank_order output fits pack_evidence exactly (nothing cut there)", pe.n_included == len(packed)
           and not pe.last_item_cut and pe.chars <= 3000 and dec[0]["action"].startswith("narrowed"), (dec, pe.chars))

    # PageIndex prefix summaries: a section with subsections is summarised from its own text only
    own_node = next(n for n in tr_en["nodes"].values() if n["kind"] == "section" and n["children"] and n["own_chars"] > 300)
    um = ap2._node_summary_messages(doc_en, tr_en, own_node)[1]["content"]
    _check("parent summary prompt = own text, no child summaries", "Passage (the section's own text):" in um
           and "Its subsections" not in um, um[:200])
    nav_users = [u for s, f, u in resp.seen_users if s.startswith("You find")]
    _check("navigation entries list subsection titles", any("Subsections: " in u for u in nav_users), nav_users[:1])

    small_cfg = cfg.copy_with({"evidence": {"max_chars": 700}})
    ap_small = TreeNavigation(small_cfg, client)
    ap_small.build_index(en)
    resp.keywords = ["money goes to", "<section title 2>", "typically costs"]
    rec = ap_small.answer(q1, "en-en")
    _check("no calls spent once the budget is filled", rec.trace["stop_reason"] == "budget_filled"
           and rec.trace["nav_calls"] == 2 and rec.evidence_chars <= 700,
           (rec.trace["stop_reason"], rec.trace["nav_calls"], rec.evidence_chars))

    cap_cfg = cfg.copy_with({"caps": {"tree_nav_max_calls": 2}})
    ap3 = TreeNavigation(cap_cfg, client)
    ap3.build_index(en)
    resp.keywords = ["money goes to"]
    rec = ap3.answer(qs[1], "en-en")
    _check("call cap enforced", rec.llm_calls == 2 and rec.trace["nav_calls"] == 1 and rec.trace["stop_reason"] == "call_cap"
           and rec.evidence_chars <= cfg.evidence["max_chars"], (rec.llm_calls, rec.trace["stop_reason"], rec.evidence_chars))

    # routing over a 3-document set with doc_scope "corpus"
    three = load_set("A_en").subset(["A.rpt-b.en", "A.rpt-c.en", "A.rpt-d.en"])
    ap4 = TreeNavigation(cfg, client)
    ap4.build_index(three)
    qd = q1.to_dict()
    qd["doc_scope"] = "corpus"
    resp.keywords = ["<report title>", "money goes to", "typically costs"]
    rec = ap4.answer(Question.from_dict(qd), "en-en")
    rt = rec.trace
    _check("routing call for corpus scope", rt["routing"] and rt["routing"]["status"] == "ok"
           and rt["routing"]["chosen"][0] == "A.rpt-b.en" and rt["steps"][0]["kind"] == "route", rt["routing"])
    _check("routing counted", rec.llm_calls == rt["nav_calls"] + 1)

    # Persian: language retry at index time
    fa = load_set("A_fa").subset(["A.rpt-b.fa"])
    ap5 = TreeNavigation(cfg, client)
    hfa = ap5.build_index(fa)
    sfa = read_json(hfa.path / "summary_stats.json")
    _check("fa summaries retried into Persian", sfa["lang_retries"] > 0 and sfa["lang_mismatch"] == 0,
           {k: sfa[k] for k in ("lang_retries", "lang_mismatch", "summaries")})
    resp.keywords = ["سرمایه"]
    rec = ap5.answer(q1, "en-fa")
    _check("cross-lingual query runs", rec.error is None and rec.d_lang == "fa", rec.error)
    shutil.rmtree(tmp, ignore_errors=True)
    bad = [n for n, ok in _RESULTS if not ok]
    print("offline: %d checks, %d failed %s" % (len(_RESULTS), len(bad), bad))
    return not bad


def _live(report="rpt-b", langs=("en", "fa"), settings=None):
    from ..config import load_config
    from ..contracts import RunLog
    from ..corpus import load_set
    from ..llm import Ledger, OllamaClient

    cfg = load_config()
    ledger = Ledger(path=str(cfg.ensure_dir("ledger_root") / "smoke_tree_nav.jsonl"))
    client = OllamaClient(cfg, ledger)
    h = client.health()
    print("ollama", {k: h.get(k) for k in ("reachable", "version", "chat_present", "chat_digest_match")}, flush=True)
    if not h.get("reachable") or not h.get("chat_present"):
        return False
    ap = TreeNavigation(cfg, client)
    for lang in langs:
        sub = load_set("A_%s" % lang, cfg).subset(["A.%s.%s" % (report, lang)])
        t0 = time.time()
        hd = ap.build_index(sub)
        st = read_json(hd.path / "summary_stats.json")
        print("index %s: %.0fs  build_stats totals %s  size %d B" % (sub.name, time.time() - t0, hd.stats.get("totals"),
                                                                      hd.stats.get("size_bytes", 0)), flush=True)
        print("   summaries %s  lang_mismatch %d  retries %d  this_build %s" % (
            st["summaries"], st["lang_mismatch"], st["lang_retries"], st["this_build"]), flush=True)
        root = hd.payload["docs"]["A.%s.%s" % (report, lang)]["nodes"]["A.%s.%s" % (report, lang)]
        print("   doc summary:", root["summary"], flush=True)
    log = RunLog(cfg.path("runs_root") / "smoke_tree_nav.jsonl")
    settings = settings or [("en-en", 0), ("en-en", 1), ("en-en", 2), ("fa-fa", 0), ("fa-fa", 1), ("fa-fa", 2),
                            ("fa-en", 1), ("en-fa", 2)]
    qs = handmade_questions(report)
    ok_all = True
    for setting, qi in settings:
        if setting.split("-")[1] not in langs:
            continue
        q = qs[qi]
        rec = ap.answer(q, setting)
        log.append(rec)
        tr = rec.trace or {}
        gold = set(q.gold(rec.d_lang))
        hit = bool(gold & set(rec.evidence_units))
        print("\n%s %s -> %r (gold %s)" % (q.id, setting, rec.answer_text, q.answer(rec.q_lang)), flush=True)
        print("   calls %d (nav %s) tokens %d/%d latency %.1fs evidence %d chars, gold unit in evidence: %s, stop %s, error %s" % (
            rec.llm_calls, tr.get("nav_calls"), rec.prompt_tokens, rec.completion_tokens, rec.latency_ms / 1000,
            rec.evidence_chars, hit, tr.get("stop_reason"), rec.error), flush=True)
        for s in tr.get("steps", []):
            print("   step %s %s at %s -> %s %s %s" % (s.get("call"), s["kind"], clip(s.get("path") or "", 70), s.get("raw", s.get("pick")),
                                                   s.get("decision") or "", ("ERR " + s["error"]) if s.get("error") else ""), flush=True)
        ok_all = ok_all and rec.error is None and rec.llm_calls == tr.get("nav_calls", -99) + 1 \
            and rec.llm_calls <= cfg.caps["tree_nav_max_calls"]
    print("\nlive: consistency %s; ledger %s; runs %s" % (ok_all, ledger.path, log.path))
    return ok_all


if __name__ == "__main__":
    args = sys.argv[1:]
    ok = True
    if "--offline" in args or not args:
        ok = _offline() and ok
    if "--live" in args:
        langs = ("en", "fa")
        for a in args:
            if a.startswith("--langs="):
                langs = tuple(a.split("=", 1)[1].split(","))
        ok = _live(langs=langs) and ok
    sys.exit(0 if ok else 1)
