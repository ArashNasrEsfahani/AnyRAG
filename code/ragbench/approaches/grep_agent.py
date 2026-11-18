"""B5 - free grep agent: a ReAct loop over the PLAIN section store (no enrichment).

The local model drives three tools through a strict JSON action protocol (one JSON
object per reply; small local models are unreliable at native tool calling, so the
reply is parsed, repaired where possible, and bounced back with an error otherwise):

    {"thought": "...", "action": "grep", "pattern": "<regex>", "scope": "corpus" | "D1", "max_hits": 20}
    {"thought": "...", "action": "read", "section": "D1:S12", "start_line": 5, "end_line": 30}
    {"thought": "...", "action": "answer", "text": "...", "cited_sections": ["D1:S12"]}

Index: store.build_plain_store over the CorpusSet, written under the index dir (so
build_stats.size_bytes is the real size of what B5 searches). No index-time model calls.

Query (fairness decisions, see also the notes at the bottom of this docstring):
  * Sections are named by short handles "D<k>:S<order>" (document k of the query's
    doc scope, section pre-order index), not by corpus section ids: A's section ids are
    English slugs even in the Persian edition and would leak English words into fa runs.
  * grep runs ripgrep through store.grep (-e per pattern, -i, UTF-8 JSON output, a
    subprocess timeout) inside the query's doc scope only. "@@SECTION" and "@@PATH" header
    lines are dropped from hits (the first holds ids, the second repeats every ancestor's
    title in each descendant); "@@TITLE" hits are kept and shown as title matches.
    Patterns are guarded: length cap, newlines removed, rejected when they match the
    empty string or a single common character (would flood every line); a regex error or
    timeout comes back to the model as a tool error. Rust regex (rg's default engine) has
    no catastrophic backtracking, so time is bounded by the timeout and the hit cap.
    Patterns are passed through the shared normalize() one run of literal text at a time
    (whitespace, ZWNJ and ASCII punctuation left as written, so regex syntax is untouched):
    Arabic yeh/kaf and Persian/Arabic digits become the store's forms. Hit parsing stops
    after DEFAULTS["grep_scan_cap"] kept hits (counts are then reported as lower bounds), and
    the store is read once by a no-hit ripgrep pass when the index is loaded, so a cold
    8k-file store does not time out the first greps of a question.
  * read shows a section's text with line numbers, at most caps.grep_tool_output_chars
    per call; every character shown is remembered. A line longer than one output resumes
    inside the line on the next read that starts on it. A section with no text of its own
    lists its subsections.
  * Evidence = the text the agent was shown by read, merged per section; sections the
    agent cited in its answer action come first, then the rest in first-read order. Grep
    snippets are pointers, never evidence (the system prompt says so).
  * Final answer: by default the answer action only STOPS the loop; the answer is then
    produced by the shared answer call (Approach.generate_answer) over pack_evidence of the
    read text - the same prompt and the same evidence budget as every other approach. The
    agent's own answer text is kept in the trace. Flag agent_answer=True (variant
    "B5+agent_answer") uses the agent's text directly when it answers, and falls back to
    the shared call when it does not.
  * Call budget: at most caps.grep_agent_max_steps model calls per question, the answer
    included: up to max_steps - 1 agent steps (every reply, valid or not, is a step), then
    one answer call. The loop also stops after DEFAULTS["max_consecutive_invalid"]
    unparseable replies in a row.
  * Context: older tool results are shortened once the transcript passes
    DEFAULTS["context_chars"], and that budget shrinks if a step's prompt_eval_count gets
    close to generation.num_ctx (Ollama would otherwise truncate silently).
  * The system prompt explains the store's normalisation and nothing else about language:
    no translation or bilingual hints. Document titles are listed in the document language.

Retrieved items (RunRecord.retrieved_units) are the read sections, one item per section,
unit_ids = the units whose text the agent was actually shown (a unit counts when at least
one of its lines was shown in full or with >= 60 characters); kind "section" when the whole
section text was shown, "window" otherwise.
"""
import json
import re
import subprocess
from collections import OrderedDict
from pathlib import Path

try:
    import regex as _regex
except ImportError:  # pragma: no cover
    _regex = None

from ..contracts import Approach, AnswerResult, RetrievedItem
from ..llm import extract_json
from ..normalize import ZWNJ, normalize
from ..store import STORE_FORMAT, build_plain_store, grep, line_info, read_lines, section_file
from ..util import read_json

# Query-time tunables that are not fairness-critical (fairness-critical caps live in config.yaml).
DEFAULTS = {
    "step_num_predict": 320,          # tokens for one agent reply (a thought + one action)
    "default_max_hits": 20,           # grep hit lines shown when the model gives no max_hits
    "hit_line_chars": 180,            # characters of a hit line shown around its first match
    "max_pattern_chars": 200,
    "max_patterns": 5,                # a list-valued pattern is OR-ed, at most this many
    "grep_timeout_s": 60,             # a cold corpus-C store (8k files on this disk) took ~76 s to read once
    "max_docs_listed": 25,            # documents listed in the first message (the rest by count)
    "context_chars": 16000,           # transcript size before older tool results are shortened
    "old_observation_chars": 300,
    "keep_recent_observations": 2,
    "ctx_token_guard": 0.85,          # shrink the transcript budget when prompt tokens pass this share of num_ctx
    "max_consecutive_invalid": 3,
    "min_unit_chars_shown": 60,       # a partly shown line still counts its unit when this many chars were shown
    "raw_trace_chars": 1500,
    "normalize_patterns": True,
    "grep_scan_cap": 2000,            # hits parsed per grep call; beyond it the count is reported as "at least"
    "warm_timeout_s": 600,            # one no-hit ripgrep pass over the store when the index is loaded
    "max_subsections_listed": 30,     # a read of a section with no own text lists its subsections
}

FLAGS = ("agent_answer",)

ACTION_ALIASES = {
    "grep": "grep", "search": "grep", "find": "grep", "rg": "grep", "ripgrep": "grep",
    "read": "read", "open": "read", "read_section": "read", "view": "read", "show": "read", "cat": "read",
    "answer": "answer", "final": "answer", "finish": "answer", "final_answer": "answer", "stop": "answer",
    "done": "answer", "respond": "answer",
}
KEY_ALIASES = {
    "pattern": ("pattern", "patterns", "regex", "query", "pat", "search", "expression", "keyword", "keywords",
                "term", "terms"),
    "scope": ("scope", "doc", "document", "doc_id", "in", "where", "target"),
    "max_hits": ("max_hits", "limit", "max", "max_results", "n", "k"),
    "section": ("section", "section_id", "sec", "id", "handle", "file", "path"),
    "start_line": ("start_line", "start", "from", "line_start", "from_line", "first_line", "line"),
    "end_line": ("end_line", "end", "to", "line_end", "to_line", "last_line"),
    "text": ("text", "answer", "final_answer", "content", "response", "value"),
    "cited": ("cited_sections", "cited_section_ids", "citations", "cited", "sources", "sections", "cite"),
}
_WRAPPER_KEYS = ("args", "arguments", "parameters", "params", "input", "tool_input")
_ACTION_KEYS = ("action", "tool", "name", "type", "function")

HANDLE_RE = re.compile(r"[dD]\s*(\d+)\s*[:/.\-_ ,]*\s*[sS]\s*(\d+)")
DOC_RE = re.compile(r"^\s*[dD]\s*(\d+)(?!\d)")     # "D1", "D1: <copied title>", "D1:S5"

EXAMPLE = '{"thought": "...", "action": "grep", "pattern": "word", "scope": "corpus"}'

SYSTEM_PROMPT = """You are a search agent. You find the passages that answer a question in a collection of reports, using tools, and then stop.

The collection is plain text, one file per report section. Sections are named like D1:S12 (document 1, section 12).

Reply with exactly ONE JSON object per turn and nothing else. The three actions:

{{"thought": "<one short sentence>", "action": "grep", "pattern": "<regular expression>", "scope": "corpus", "max_hits": 20}}
  Searches every line for the pattern (ripgrep regular expressions, case-insensitive). scope is "corpus" (all documents) or one document such as "D1". Returns the matching lines with their section names and line numbers.

{{"thought": "<one short sentence>", "action": "read", "section": "D1:S12", "start_line": 5, "end_line": 30}}
  Shows the text of one section with line numbers. start_line and end_line are optional; without them the section is shown from the top. Long output is cut; read again with a later start_line to continue.

{{"thought": "<one short sentence>", "action": "answer", "text": "<the answer, or {abstain}>", "cited_sections": ["D1:S12"]}}
  Stops the search.{answer_rule}

Rules:
- {evidence_rule}
- Each reply is one step. You have at most {steps} steps. Do not repeat a search or a read you have already made.
- A pattern matches an exact run of characters inside one line: "alpha gamma" does not match the line "alpha beta gamma". Search for one distinctive word, or join words with .* (alpha.*gamma), or give alternatives (alpha|delta). Long phrases rarely match.
- The text is normalised; write patterns the same way. Persian yeh is ی (never ي) and kaf is ک (never ك). Digits are ASCII 0-9 (never ۰-۹ or ٠-٩). There is no tatweel (ـ). Inside Persian words the only joiner is ZWNJ (U+200C); a word written with a ZWNJ in one place may be written with a space or joined in another, so a pattern such as می.?شود matches all three forms. Optional marks are kept as written: the hamza in خانهٔ appears in some places and not in others, so خانه.? matches both. Whitespace is single spaces."""

EVIDENCE_RULE_SHARED = ("The final answer is written only from the text you opened with read. Grep lines are pointers, "
                        "not evidence: read the sections that hold the answer before you answer.")
EVIDENCE_RULE_AGENT = ("Answer only from the text you opened with read. Grep lines are pointers, not evidence: "
                       "read the sections that hold the answer before you answer.")
ANSWER_RULE_SHARED = ""
ANSWER_RULE_AGENT = (" Write text in the language of the question, the answer only, with numbers copied exactly as "
                     "written, with their unit and period.")


class _ToolError(Exception):
    pass


# --------------------------------------------------------------------------
# protocol parsing
# --------------------------------------------------------------------------

def _first_dict(obj):
    if isinstance(obj, dict):
        return obj
    if isinstance(obj, list):
        for x in obj:
            if isinstance(x, dict):
                return x
    return None


def parse_action(text):
    """Model reply -> (action, args, repaired, error). action is grep|read|answer or None.
    Tolerates code fences, chatter, a list around the object, nested argument objects,
    alias names for actions and keys, and a reply cut off by the token limit."""
    repaired = False
    obj = _first_dict(extract_json(text))
    if obj is None and text and "{" in text:
        base = text.rstrip().rstrip(",")
        for suffix in ('"}', "}", '"]}', "]}", '"}}', "}}", '"]}}'):
            obj = _first_dict(extract_json(base + suffix))
            if obj is not None:
                repaired = True
                break
    if obj is None:
        return None, {}, False, "your reply was not a JSON object"
    if isinstance(extract_json(text), list):
        repaired = True

    args = {}
    for k in _WRAPPER_KEYS:
        if isinstance(obj.get(k), dict):
            args.update(obj[k])
    act = None
    for k in _ACTION_KEYS:
        if k in obj:
            act = obj[k]
            break
    if isinstance(act, dict):                       # {"action": {"name": "grep", ...}} or {"action": {"grep": {...}}}
        inner = dict(act)
        name = None
        for k in _ACTION_KEYS:
            if isinstance(inner.get(k), str):
                name = inner.pop(k)
                break
        if name is None and len(inner) == 1:
            name, val = next(iter(inner.items()))
            inner = val if isinstance(val, dict) else {}
        for k in _WRAPPER_KEYS:
            if isinstance(inner.get(k), dict):
                args.update(inner.pop(k))
        args.update(inner)
        act = name
        repaired = True
    if act is None:                                 # {"grep": {"pattern": ...}}
        for k in obj:
            if str(k).lower() in ACTION_ALIASES:
                act = k
                if isinstance(obj[k], dict):
                    args.update(obj[k])
                repaired = True
                break
    for k, v in obj.items():
        if k not in _WRAPPER_KEYS and k not in _ACTION_KEYS and k != act:
            args.setdefault(k, v)
    name = ACTION_ALIASES.get(str(act).strip().lower()) if isinstance(act, str) else None
    if name is None:
        return None, args, repaired, "unknown or missing action %s; use grep, read or answer" % (
            json.dumps(act, ensure_ascii=False)[:40] if act is not None else "")
    canon = {}
    for key, aliases in KEY_ALIASES.items():
        for a in aliases:
            if a in args and args[a] not in (None, ""):
                canon[key] = args[a]
                break
    if name == "answer" and "text" not in canon and isinstance(obj.get("answer"), str):
        canon["text"] = obj["answer"]
    if name != str(act).strip().lower():
        repaired = True
    if name == "grep" and "pattern" not in canon:
        return name, canon, repaired, 'grep needs a "pattern"'
    if name == "read" and "section" not in canon:
        return name, canon, repaired, 'read needs a "section" such as "D1:S12"'
    return name, canon, repaired, None


# --------------------------------------------------------------------------
# pattern guard
# --------------------------------------------------------------------------

_PROBES = ("e", "a", "ا", "ی", "1", " ", "-", ".")


def normalize_pattern(p):
    """The shared normalize() applied to each run of literal text in a regex. Whitespace, ZWNJ
    and ASCII punctuation (all regex syntax lives there) are kept as written; every run of
    other characters is normalised as a whole, so NFKC composition inside a word (e.g. Arabic
    yeh + combining hamza -> ئ, alef + madda -> آ) gives the same form as the store. Maps
    ي/ك/Persian digits/tatweel to the store's forms without touching syntax."""
    out, run = [], []

    def flush():
        if run:
            out.append(normalize("".join(run)))
            run.clear()

    for ch in p:
        if ch.isspace() or ch == ZWNJ or (ord(ch) < 128 and not ch.isalnum()):
            flush()
            out.append(ch)
        else:
            run.append(ch)
    flush()
    return "".join(out)


def is_abstain(text, token):
    """'NOT_FOUND', 'not_found.', '"NOT_FOUND"' all count as the abstain token."""
    return re.sub(r"[\s.!\"'`]+", "", text or "").upper() == str(token).upper()


def pattern_problem(p, max_chars):
    """None when the pattern may run; else a message for the model."""
    if not p.strip():
        return "the pattern is empty"
    if len(p) > max_chars:
        return "the pattern is too long (%d characters; the limit is %d)" % (len(p), max_chars)
    if _regex is None:
        return None
    try:
        rx = _regex.compile(p, _regex.IGNORECASE)
    except Exception:  # noqa: BLE001 - syntax differences are left to ripgrep
        return None
    try:
        if rx.search("", timeout=0.5) is not None:
            return "the pattern matches empty text, so it would match every line; search for a word"
        for probe in _PROBES:
            if rx.search(probe, timeout=0.5) is not None:
                return "the pattern matches a single common character, so it would match almost every line; search for a word"
    except Exception:  # noqa: BLE001 - regex timeout: let ripgrep (linear time) judge it
        return None
    return None


def _clean_rg_error(msg):
    m = re.sub(r"\s+", " ", msg or "").replace("ripgrep error: ", "")
    return m[:220]


def _to_int(v):
    if v is None or isinstance(v, bool):
        return None
    try:
        return int(float(str(v).strip().lstrip("Ll")))
    except (TypeError, ValueError):
        return None


def _snippet(text, subs, width):
    if len(text) <= width:
        return text
    s0 = subs[0][0] if subs else 0
    left = max(0, s0 - width // 3)
    right = min(len(text), left + width)
    left = max(0, right - width)
    return ("…" if left > 0 else "") + text[left:right] + ("…" if right < len(text) else "")


# --------------------------------------------------------------------------
# the approach
# --------------------------------------------------------------------------

class GrepAgent(Approach):
    name = "B5"

    def __init__(self, cfg=None, client=None, flags=None):
        super().__init__(cfg=cfg, client=client, flags=flags)
        bad = [k for k, v in self.flags.items() if k not in FLAGS and v not in (False, None)]
        if bad:
            raise ValueError("B5 does not accept flags %s (known: %s)" % (bad, FLAGS))
        self.params = dict(DEFAULTS)
        caps = self.cfg.caps
        self.max_steps = int(caps["grep_agent_max_steps"])
        self.tool_chars = int(caps["grep_tool_output_chars"])
        self.max_hits_cap = int(caps["grep_max_hits"])
        self.agent_answer = bool(self.flags.get("agent_answer"))

    def config_signature(self):
        return {"store": "plain", "store_format": STORE_FORMAT, "normalized": True}

    def _build_index(self, corpus_set, index_dir):
        root = build_plain_store(corpus_set, root=Path(index_dir) / "store", cfg=self.cfg)
        man = read_json(Path(root) / "manifest.json", {}) or {}
        warm_ms = self._warm_store(root)
        return {"store_root": str(root), "n_files": man.get("n_files", 0),
                "corpus_fingerprint": man.get("corpus_fingerprint"), "warm_ms": warm_ms}

    def _warm_store(self, root):
        """One ripgrep pass with a pattern that never matches, so every store file is read once
        at index-load time. Without it the first greps of a question on a freshly written or
        cold store (corpus C: 8k files) hit the per-call timeout and cost the agent steps."""
        import time as _time
        t0 = _time.perf_counter()
        try:
            grep(root, "@@B5-WARM-UP-NO-SUCH-LINE@@", fixed_strings=True,
                 timeout_s=float(self.params["warm_timeout_s"]), cfg=self.cfg)
        except Exception:  # noqa: BLE001 - warm-up is best effort
            pass
        return round((_time.perf_counter() - t0) * 1000.0, 1)

    def _load_index(self, corpus_set, index_dir):
        return self._build_index(corpus_set, index_dir)

    def _answer(self, query, index):
        return _Session(self, query, index).run()


class _Session:
    """State of one question: handles, transcript, what was read."""

    def __init__(self, agent, query, index):
        self.a = agent
        self.q = query
        self.index = index
        self.cfg = agent.cfg
        self.p = agent.params
        self.root = Path(index.payload["store_root"])
        self.docs = agent.docs_in_scope(query, index)
        self.all_docs = query.doc_scope == "corpus"
        self.doc_by_id = {d.id: (i, d) for i, d in enumerate(self.docs, 1)}
        self.agent_steps = max(1, agent.max_steps - 1)
        self.abstain = self.cfg.evidence["abstain_token"]
        self.reads = OrderedDict()      # rel -> {doc, section, handle, first_step, lines {ln: chars shown from 0}, steps}
        self.line_pos = {}              # (rel, ln) -> where a cut line resumes on the next read that starts on it
        self.transcript = []            # [[assistant_content, observation]]
        self.steps = []
        self.seen = {}
        self.ctx_chars = int(self.p["context_chars"])
        self._lines = {}
        self._sec_by_order = {}

    # -- handles --------------------------------------------------------------
    def handle(self, doc, section):
        return "D%d:S%d" % (self.doc_by_id[doc.id][0], section.order)

    def section_by_order(self, doc, order):
        m = self._sec_by_order.get(doc.id)
        if m is None:
            m = self._sec_by_order[doc.id] = {s.order: s for s in doc.sections}
        return m.get(order)

    def resolve_section(self, ref):
        if isinstance(ref, list):
            ref = ref[0] if ref else ""
        if isinstance(ref, (int, float)) and not isinstance(ref, bool):
            ref = "S%d" % int(ref)
        ref = str(ref or "").strip()
        if not ref:
            raise _ToolError('read needs a "section" such as "D1:S12"')
        doc = sec = None
        m = HANDLE_RE.search(ref)
        if m:
            di, so = int(m.group(1)), int(m.group(2))
            if not 1 <= di <= len(self.docs):
                raise _ToolError("there is no document D%d; documents are D1-D%d" % (di, len(self.docs)))
            doc = self.docs[di - 1]
            sec = self.section_by_order(doc, so)
            if sec is None:
                raise _ToolError("D%d has sections S0-S%d; S%d does not exist" % (di, len(doc.sections) - 1, so))
        elif "::" in ref:
            d_id = ref.split("::", 1)[0]
            if d_id in self.doc_by_id:
                doc = self.doc_by_id[d_id][1]
                try:
                    sec = doc.section(ref)
                except KeyError:
                    sec = None
        else:
            m2 = re.fullmatch(r"[sS]?\s*(\d+)", ref)
            if m2 and len(self.docs) == 1:
                doc = self.docs[0]
                sec = self.section_by_order(doc, int(m2.group(1)))
        if doc is None or sec is None:
            raise _ToolError("unknown section %s; use a section name from the grep results, such as D1:S12"
                             % json.dumps(ref[:60], ensure_ascii=False))
        rel = section_file(self.root, sec.id)
        if rel is None:
            raise _ToolError("section %s is not in the store" % self.handle(doc, sec))
        return doc, sec, rel, self.handle(doc, sec)

    def resolve_scope(self, scope):
        """-> (docs, label, doc_ids for store.grep or None for the whole store)."""
        if isinstance(scope, list):
            vals = scope
        else:
            vals = [scope]
        docs = []
        for v in vals:
            s = str(v if v is not None else "").strip()
            if s.lower() in ("", "corpus", "all", "*", "none", "any", "everything", "collection"):
                return list(self.docs), "corpus", (None if self.all_docs else [d.id for d in self.docs])
            m = DOC_RE.match(s) or HANDLE_RE.match(s) or re.match(r"^\s*(\d+)\s*$", s)
            if m:
                di = int(m.group(1))
                if not 1 <= di <= len(self.docs):
                    raise _ToolError('there is no document D%d; use "corpus" or D1-D%d' % (di, len(self.docs)))
                d = self.docs[di - 1]
            elif s in self.doc_by_id:
                d = self.doc_by_id[s][1]
            else:
                raise _ToolError('unknown scope %s; use "corpus" or a document name such as "D1"'
                                 % json.dumps(s[:40], ensure_ascii=False))
            if d not in docs:
                docs.append(d)
        return docs, ",".join("D%d" % self.doc_by_id[d.id][0] for d in docs), [d.id for d in docs]

    # -- store access ---------------------------------------------------------
    def file_lines(self, rel):
        if rel not in self._lines:
            text, _a, _b, _u = read_lines(self.root, rel, 1, 10 ** 9)
            lines = text.split("\n")
            info = [line_info(self.root, rel, i) for i in range(1, len(lines) + 1)]
            self._lines[rel] = (lines, info)
        return self._lines[rel]

    # -- tools ----------------------------------------------------------------
    def tool_grep(self, args, step):
        raw = args.get("pattern")
        pats = raw if isinstance(raw, list) else [raw]
        pats = [str(x) for x in pats if x is not None and str(x).strip()][: int(self.p["max_patterns"])]
        if not pats:
            raise _ToolError('grep needs a "pattern"')
        used = []
        for x in pats:
            y = re.sub(r"[\r\n\t]+", " ", x).strip()
            if self.p["normalize_patterns"]:
                y = normalize_pattern(y)
            problem = pattern_problem(y, int(self.p["max_pattern_chars"]))
            if problem:
                raise _ToolError(problem)
            used.append(y)
        docs, label, doc_ids = self.resolve_scope(args.get("scope"))
        if not docs:
            # store.grep treats doc_ids=[] as "the whole store": never search outside the query's scope
            raise _ToolError("there are no documents to search for this question")
        mh = _to_int(args.get("max_hits"))
        max_hits = max(1, min(self.a.max_hits_cap, mh if mh else int(self.p["default_max_hits"])))
        key = json.dumps(["grep", used, label], ensure_ascii=False)
        info = {"patterns": pats, "patterns_used": used, "scope": label, "max_hits": max_hits}
        if key in self.seen:
            info["repeat_of"] = self.seen[key]
            return ("You already ran this search at step %d; its result is above. Try a different pattern or read a section."
                    % self.seen[key]), info
        # Parsing stops after parse_cap raw hits. Up to two dropped header lines (@@SECTION, @@PATH)
        # per section file can be among them, so the cap leaves room for those: when parsing stops,
        # at least grep_scan_cap kept hits exist and the counts shown are lower bounds.
        scan_cap = int(self.p["grep_scan_cap"])
        parse_cap = scan_cap + 2 * sum(len(d.sections) for d in docs)
        try:
            raw_hits = grep(self.root, used, doc_ids=doc_ids, ignore_case=True, max_hits=parse_cap,
                            timeout_s=float(self.p["grep_timeout_s"]), cfg=self.cfg)
        except ValueError as e:
            raise _ToolError("invalid pattern: %s" % _clean_rg_error(str(e)))
        except subprocess.TimeoutExpired:
            raise _ToolError("the search timed out; try a more specific pattern or a single document scope")
        self.seen[key] = step                          # only searches that ran count as "already run"
        truncated = len(raw_hits) >= parse_cap
        hits = [h for h in raw_hits
                if (h.tag != "meta" or h.text.startswith("@@TITLE ")) and h.doc_id in self.doc_by_id]
        by_sec = OrderedDict()
        for h in hits:
            by_sec.setdefault(h.path, []).append(h)
        info.update(n_hits=len(hits), n_sections=len(by_sec), truncated=truncated)
        pat_show = " | ".join(used)
        if not hits:
            info["sections"] = []
            return ("grep %s in %s: no matches. (A pattern must match a run of characters inside one line; "
                    "try fewer words, word1.*word2, or alternatives.)" % (json.dumps(pat_show, ensure_ascii=False), label)), info
        if truncated:
            head = "grep %s in %s: at least %d matching lines in at least %d sections (too many to count; a more specific pattern helps)." % (
                json.dumps(pat_show, ensure_ascii=False), label, len(hits), len(by_sec))
        else:
            head = "grep %s in %s: %d matching lines in %d sections." % (
                json.dumps(pat_show, ensure_ascii=False), label, len(hits), len(by_sec))
        cap = self.a.tool_chars
        reserve = 160
        out = [head]
        used_chars = len(head)
        shown = 0
        shown_secs = []
        full_secs = set()
        width = int(self.p["hit_line_chars"])
        stop = False
        for rel, hs in by_sec.items():
            h0 = hs[0]
            doc = self.doc_by_id.get(h0.doc_id, (None, None))[1]
            if doc is None:
                continue
            sec = doc.section(h0.section_id)
            sec_line = "%s «%s»" % (self.handle(doc, sec), sec.title[:90])
            if used_chars + 1 + len(sec_line) > cap - reserve or shown >= max_hits:
                break
            out.append(sec_line)
            used_chars += 1 + len(sec_line)
            shown_secs.append(self.handle(doc, sec))
            n_here = 0
            for h in hs:
                if shown >= max_hits:
                    stop = True
                    break
                if h.text.startswith("@@TITLE "):
                    t = h.text[len("@@TITLE "):]
                    subs = [[max(0, s - 8), max(0, e - 8)] for s, e in h.submatches]
                    ln = "  title: " + _snippet(t, subs, width)
                else:
                    ln = "  L%d: %s" % (h.line, _snippet(h.text, h.submatches, width))
                if used_chars + 1 + len(ln) > cap - reserve:
                    stop = True
                    break
                out.append(ln)
                used_chars += 1 + len(ln)
                shown += 1
                n_here += 1
            if n_here == len(hs):
                full_secs.add(rel)
            if stop:
                break
        rest = [(rel, hs) for rel, hs in by_sec.items() if rel not in full_secs]
        if rest:
            parts = []
            for rel, hs in rest:
                h0 = hs[0]
                doc = self.doc_by_id.get(h0.doc_id, (None, None))[1]
                if doc is None:
                    continue
                parts.append("%s (%d)" % (self.handle(doc, doc.section(h0.section_id)), len(hs)))
            tail = "Not shown: %s%d more lines; sections with unshown lines: " % (
                "at least " if truncated else "", len(hits) - shown)
            room = cap - used_chars - len(tail) - 2
            listed = []
            for s in parts:
                if sum(len(x) + 2 for x in listed) + len(s) + 4 > room:
                    listed.append("…")
                    break
                listed.append(s)
            out.append(tail + ", ".join(listed))
        info.update(shown_hits=shown, sections=shown_secs[:40])
        return "\n".join(out)[:cap], info

    def tool_read(self, args, step):
        doc, sec, rel, handle = self.resolve_section(args.get("section"))
        lines, info = self.file_lines(rel)
        body = [i for i in range(1, len(lines) + 1) if lines[i - 1].strip() and info[i - 1][1] != "meta"]
        path = " > ".join(doc.section_path(sec.id)[:-1])
        tinfo = {"section": handle, "section_id": sec.id}
        if not body:
            # a heading-only section (8-9% of sections in corpora B and C): name its subsections,
            # otherwise a title hit on it is a dead end
            tinfo["empty"] = True
            msg = "%s «%s» has no text of its own." % (handle, sec.title[:90])
            kids = doc.children(sec.id)
            if kids:
                lim = int(self.p["max_subsections_listed"])
                room = self.a.tool_chars - len(msg) - 40
                listed = []
                for c in kids[:lim]:
                    s = "%s «%s»" % (self.handle(doc, c), c.title[:60])
                    if sum(len(x) + 2 for x in listed) + len(s) > room:
                        break
                    listed.append(s)
                more = len(kids) - len(listed)
                msg += " Its subsections: " + ", ".join(listed) + ((" and %d more" % more) if more else "") + "."
                tinfo["subsections"] = [self.handle(doc, c) for c in kids[:lim]]
            return msg, tinfo
        b0, b1 = body[0], body[-1]
        sl, el = _to_int(args.get("start_line")), _to_int(args.get("end_line"))
        if sl is not None and el is not None and el < sl:
            sl, el = el, sl
        if el is not None and el < b0:                 # a range inside the header lines: show the text from the top
            sl, el = None, None
            tinfo["header_range_ignored"] = True
        a = max(b0, sl if sl is not None else b0)
        b = min(b1, el if el is not None else b1)
        if a > b1:
            raise _ToolError("%s has text on lines %d-%d; start_line %d is past the end" % (handle, b0, b1, a))
        if b < a:
            raise _ToolError("%s has text on lines %d-%d; end_line %d is before it" % (handle, b0, b1, b))
        # A line cut inside an earlier read resumes where that read stopped when a read starts on it
        # (the cut hint names that line), so the tail of a line longer than one tool output is reachable.
        start_off = self.line_pos.get((rel, a), 0)
        if start_off >= len(lines[a - 1]):
            start_off = 0
        key = json.dumps(["read", rel, a, b, start_off])
        tinfo.update(start_line=a, end_line=b)
        if start_off:
            tinfo["start_char"] = start_off
        if key in self.seen:
            tinfo["repeat_of"] = self.seen[key]
            return ("You already read these lines of %s at step %d; they are above. Read a different section or answer."
                    % (handle, self.seen[key])), tinfo
        self.seen[key] = step
        head = "%s «%s»%s | text lines %d-%d" % (
            handle, sec.title[:90], (" | in: " + path[:120]) if path else "", b0, b1)
        cap = self.a.tool_chars
        reserve = 90
        out = [head]
        used = len(head)
        shown = []                                    # (ln, end): characters [0, end) of line ln have been shown
        cut = False
        for ln in range(a, b + 1):
            t = lines[ln - 1]
            if not t.strip() or info[ln - 1][1] == "meta":
                continue
            off = start_off if ln == a else 0
            seg = t[off:]
            prefix = ("L%d (continued): …" % ln) if off else ("L%d: " % ln)
            if used + 1 + len(prefix) + len(seg) <= cap - reserve:
                out.append(prefix + seg)
                used += 1 + len(prefix) + len(seg)
                shown.append((ln, len(t)))
                continue
            room = cap - reserve - used - 1 - len(prefix) - 1
            if room >= int(self.p["min_unit_chars_shown"]):
                out.append(prefix + seg[:room].rstrip() + "…")
                shown.append((ln, off + room))
            cut = True
            break
        if cut:
            last_ln, last_end = shown[-1] if shown else (None, None)
            if last_ln is not None and last_end < len(lines[last_ln - 1]):
                out.append("(cut inside line %d; to continue: read %s with start_line %d)" % (last_ln, handle, last_ln))
            else:
                nxt = last_ln + 1 if last_ln is not None else a
                more = [i for i in body if nxt <= i <= b]
                if more:
                    out.append("(cut here; to continue: read %s with start_line %d)" % (handle, more[0]))
        elif b < b1:
            out.append("(lines %d-%d of this section not shown)" % (b + 1, b1))
        entry = self.reads.get(rel)
        if entry is None:
            entry = self.reads[rel] = {"doc": doc, "section": sec, "handle": handle, "first_step": step,
                                       "lines": {}, "steps": []}
        entry["steps"].append(step)
        for ln, end in shown:
            # shown ranges of a line are always [0, end): a read starts a line at 0 or where the last cut stopped
            entry["lines"][ln] = max(end, entry["lines"].get(ln, 0))
            if end < len(lines[ln - 1]):
                self.line_pos[(rel, ln)] = max(end, self.line_pos.get((rel, ln), 0))
            else:
                self.line_pos.pop((rel, ln), None)
        tinfo.update(lines_shown=[shown[0][0], shown[-1][0]] if shown else [], cut=cut,
                     unit_ids=self._units_for(rel, [(ln, entry["lines"][ln]) for ln, _e in shown]))
        return "\n".join(out), tinfo

    def _units_for(self, rel, shown):
        """shown: [(line, end)] -> unit ids whose line was shown in full or for >= min_unit_chars_shown chars."""
        lines, info = self.file_lines(rel)
        need = int(self.p["min_unit_chars_shown"])
        out = []
        for ln, end in shown:
            uid = info[ln - 1][0]
            if uid and (end >= len(lines[ln - 1]) or end >= need) and uid not in out:
                out.append(uid)
        return out

    # -- evidence ---------------------------------------------------------------
    def evidence_items(self, cited_handles):
        order, cited_rels = [], set()
        for h in cited_handles:
            try:
                rel = self.resolve_section(h)[2]
            except _ToolError:
                continue
            cited_rels.add(rel)
            if rel in self.reads and rel not in order:
                order.append(rel)
        for rel in self.reads:
            if rel not in order:
                order.append(rel)
        items = []
        for rank, rel in enumerate(order, 1):
            e = self.reads[rel]
            lines, info = self.file_lines(rel)
            body = [i for i in range(1, len(lines) + 1) if lines[i - 1].strip() and info[i - 1][1] != "meta"]
            shown = sorted(e["lines"].items())
            parts = []
            prev_ln, prev_uid = None, None
            for ln, end in shown:
                uid = info[ln - 1][0]
                t = lines[ln - 1]
                if prev_ln is not None:
                    gap = [i for i in body if prev_ln < i < ln]
                    if gap:
                        parts.append("\n…\n")
                    elif uid != prev_uid:
                        parts.append("\n\n")
                    else:
                        parts.append("\n")
                parts.append(t if end >= len(t) else t[:end].rstrip() + " …")
                prev_ln, prev_uid = ln, uid
            whole = all(e["lines"].get(i, 0) >= len(lines[i - 1]) for i in body)
            ranges = []
            for ln, _v in shown:
                if ranges and ln <= ranges[-1][1] + 2:
                    ranges[-1][1] = ln
                else:
                    ranges.append([ln, ln])
            all_units = []
            for i in body:
                u = info[i - 1][0]
                if u and u not in all_units:
                    all_units.append(u)
            items.append(RetrievedItem(
                text="".join(parts), doc_id=e["doc"].id, section_id=e["section"].id,
                unit_ids=self._units_for(rel, shown),
                kind="section" if whole else "window", score=None, rank=rank, span=None,
                meta={"handle": e["handle"], "lines": ranges, "read_steps": e["steps"], "cited": rel in cited_rels,
                      "section_units_total": len(all_units)}))
        return items

    # -- transcript -------------------------------------------------------------
    def system_message(self):
        shared = not self.a.agent_answer
        return SYSTEM_PROMPT.format(
            abstain=self.abstain, steps=self.agent_steps,
            answer_rule=ANSWER_RULE_SHARED if shared else ANSWER_RULE_AGENT,
            evidence_rule=EVIDENCE_RULE_SHARED if shared else EVIDENCE_RULE_AGENT)

    def first_user(self):
        n = len(self.docs)
        lines = ["Question: %s" % self.q.text.strip(), "", "Documents to search (%d):" % n]
        lim = int(self.p["max_docs_listed"])
        for i, d in enumerate(self.docs[:lim], 1):
            lines.append("D%d: %s (sections S0-S%d)" % (i, (d.title or d.id)[:150], len(d.sections) - 1))
        if n > lim:
            lines.append('... and %d more documents (D%d-D%d); grep with scope "corpus" searches all of them.'
                         % (n - lim, lim + 1, n))
        if n == 0:
            lines.append("(none)")
        lines += ["", "Step 1 of %d. Reply with one JSON object." % self.agent_steps]
        return "\n".join(lines)

    def messages(self):
        system = self.system_message()
        first = self.first_user()
        turns = [list(t) for t in self.transcript]
        keep = int(self.p["keep_recent_observations"])
        short = int(self.p["old_observation_chars"])

        def total():
            return len(system) + len(first) + sum(len(x) + len(y) for x, y in turns)

        shortened = 0
        for i in range(max(0, len(turns) - keep)):
            if total() <= self.ctx_chars:
                break
            if len(turns[i][1]) > short:
                turns[i][1] = turns[i][1][:short] + " …[older result shortened]"
                shortened += 1
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": first}]
        for asst, obs in turns:
            msgs.append({"role": "assistant", "content": asst})
            msgs.append({"role": "user", "content": obs})
        return msgs, shortened

    def step_note(self, k):
        left = self.agent_steps - k
        if left <= 0:
            return ""
        if left == 1:
            if self.a.agent_answer:
                return "\n\n[Step %d of %d. This is your last step: reply with the answer action.]" % (k + 1, self.agent_steps)
            return ("\n\n[Step %d of %d. This is your last step; after it the answer is written from the text you have read.]"
                    % (k + 1, self.agent_steps))
        return "\n\n[Step %d of %d.]" % (k + 1, self.agent_steps)

    # -- loop -------------------------------------------------------------------
    def run(self):
        client = self.a.client
        num_ctx = int(self.cfg.generation["num_ctx"])
        stop_reason = "step_cap"
        draft, cited = None, []
        invalid_run = 0
        bounced = False
        n_calls = 0
        for k in range(1, self.agent_steps + 1):
            msgs, shortened = self.messages()
            ctx_chars = sum(len(m["content"]) for m in msgs)
            r = client.chat(msgs, purpose="agent_step", format="json", num_predict=int(self.p["step_num_predict"]))
            n_calls += 1
            if r.prompt_tokens and r.prompt_tokens > float(self.p["ctx_token_guard"]) * num_ctx:
                self.ctx_chars = max(4000, int(self.ctx_chars * 0.7))
            action, args, repaired, err = parse_action(r.text)
            rec = {"step": k, "action": action, "args": _clip(args), "repaired": repaired,
                   "raw": (r.text or "")[: int(self.p["raw_trace_chars"])], "prompt_tokens": r.prompt_tokens,
                   "completion_tokens": r.completion_tokens, "wall_ms": round(r.wall_ms, 1),
                   "done_reason": r.done_reason, "context_chars": ctx_chars, "shortened_observations": shortened}
            done = False
            tool = None
            if err:
                invalid_run += 1
                rec["error"] = err
                obs = "Error: %s. Reply with one JSON object such as %s" % (err, EXAMPLE)
            else:
                invalid_run = 0
                try:
                    if action == "grep":
                        obs, tool = self.tool_grep(args, k)
                    elif action == "read":
                        obs, tool = self.tool_read(args, k)
                    else:
                        text = str(args.get("text") or "").strip()
                        c = args.get("cited")
                        c = c if isinstance(c, list) else ([x for x in re.split(r"[,\s]+", c) if x] if isinstance(c, str) else [])
                        c = [str(x) for x in c][:20]
                        if not self.reads and not bounced and not is_abstain(text, self.abstain):
                            bounced = True
                            tool = {"bounced": True, "text": text[:300]}
                            obs = ("You have not read any section yet. The answer is written only from text you opened "
                                   "with read, so read the section that holds the answer first (or answer %s)." % self.abstain)
                        else:
                            draft, cited = text, c
                            tool = {"text": text[:500], "cited": c}
                            stop_reason = "answer"
                            done = True
                            obs = None
                except _ToolError as e:
                    obs = "Error: %s" % e
                    tool = {"error": str(e)}
            rec["tool"] = tool
            rec["observation"] = obs
            self.steps.append(rec)
            if done:
                break
            own = _first_dict(extract_json(r.text))
            if err:
                asst = (r.text or "")[:400]
            elif own is not None and not repaired:     # the model's own object, thought included
                asst = json.dumps(_clip(own, 400), ensure_ascii=False)
            else:
                asst = json.dumps(dict({"action": action}, **_clip(args, 400)), ensure_ascii=False)
            self.transcript.append([asst, obs + self.step_note(k)])
            if invalid_run >= int(self.p["max_consecutive_invalid"]):
                stop_reason = "invalid_replies"
                break

        items = self.evidence_items(cited)
        ev = self.a.pack(items, self.index)
        if self.a.agent_answer and stop_reason == "answer":
            answer_text = self.abstain if (not draft or is_abstain(draft, self.abstain)) else draft
            final = "agent"
        else:
            answer_text = self.a.generate_answer(self.q, ev)
            final = "shared"
        cited_unread = []
        for h in cited:
            try:
                if self.resolve_section(h)[2] not in self.reads:
                    cited_unread.append(h)
            except _ToolError:
                cited_unread.append(h)
        trace = {
            "params": dict(self.p, max_steps=self.a.max_steps, agent_steps=self.agent_steps,
                           tool_output_chars=self.a.tool_chars, grep_max_hits=self.a.max_hits_cap,
                           agent_answer=self.a.agent_answer),
            "docs": {"D%d" % i: d.id for i, d in enumerate(self.docs, 1)},
            "steps": self.steps,
            "stop_reason": stop_reason,
            "agent_steps_used": n_calls,
            "agent_answer_text": draft,
            "cited_sections": cited,
            "cited_unread": cited_unread,
            "read_sections": [e["handle"] for e in self.reads.values()],
            "final": final,
            "evidence_offered_chars": sum(len(it.text) for it in items),
        }
        return AnswerResult(answer_text=answer_text, evidence=ev, retrieved=items, trace=trace)


def _clip(args, n=300):
    out = {}
    for k, v in (args or {}).items():
        if isinstance(v, str):
            out[k] = v[:n]
        elif isinstance(v, (int, float, bool)) or v is None:
            out[k] = v
        else:
            try:
                s = json.dumps(v, ensure_ascii=False)
            except (TypeError, ValueError):
                s = str(v)
            out[k] = v if len(s) <= n else s[:n]
    return out
