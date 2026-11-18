#!/usr/bin/env python3
"""Strip corpus text from run / trace / pack / replay JSON(L) files, in place.

The report, book and law corpora used in the AnyRAG study are not redistributed.
Run records, answer packs, judge ledgers and replay traces embed retrieved
passages, evidence views and prompts that contain those passages. This script
removes them while keeping everything needed to audit the results: ids, ranks,
scores, answers, verdicts, token counts, latencies and costs.

Rules (applied recursively to every JSON value):
  1. String (or list-of-string) values under keys in DROP_KEYS are replaced by
     a marker "[stripped: N chars]" / "[stripped]"; structural values under
     those keys (offsets, nested dicts) are recursed into. These hold passage / evidence / chunk text,
     prompts that embed evidence, LLM views and excerpts.
  2. `text` is kept only when its parent is a question-side or answer-side
     container (claims, clauses, parts, questions); elsewhere it is stripped.
  3. `raw` (raw model output) is kept only inside judge attempts; selector or
     keyword raw outputs can quote passages and are stripped.
  4. Fallback: any remaining string longer than MAX_FREE chars whose key is not
     in KEEP_LONG is stripped as well.

Usage:
  python tools/strip_corpus_text.py results/            # strip in place
  python tools/strip_corpus_text.py --check results/    # verify only; exit 1 on findings
"""
from __future__ import annotations

import argparse
import json
import os
import sys

MAX_FREE = 400

DROP_KEYS = {
    "content", "prompt", "messages_text", "txt", "full_card", "summary", "prefix",
    "card", "pv", "head", "span", "passage", "passages", "evidence_text", "excerpt",
    "excerpts", "view", "views", "context", "chunk", "chunk_text", "snippet",
    "snippets", "preview", "body", "block_text", "unit_text", "section_text",
    "doc_text", "canon", "window", "evidence_block", "packed", "pack_text",
    # added 2026-11 for the post-hoc agent traces (results/posthoc S3, S9): tool-call
    # outputs (grep hits, document reads) quote corpus text in strings <= 300 chars
    "out",
}
TEXT_OK_PARENTS = {"claims", "clauses", "parts", "questions"}
KEEP_LONG = {
    # answers and judgments (model outputs about the question, not corpus text)
    "answer", "answer_text", "answers", "expected", "expected_answer",
    "expected_answer_gloss_en", "reason", "reasons", "rationale", "explanation",
    "verdict", "note", "notes", "comment", "comments", "raw",
    # questions
    "question", "question_gloss_en", "gloss", "q",
    # bookkeeping
    "rule", "source", "kind", "idea", "gate_result", "traceback", "error",
    "why", "description", "about", "status", "message", "detail", "claim",
}


def _marker(v):
    if isinstance(v, str):
        return f"[stripped: {len(v)} chars]"
    return "[stripped]"


def strip(o, parent: str = "", in_attempts: bool = False):
    """Return (new_value, n_stripped)."""
    n = 0
    if isinstance(o, dict):
        out = {}
        for k, v in o.items():
            kl = k.lower() if isinstance(k, str) else k
            drop = False
            if kl in DROP_KEYS:
                # only text payloads: strings, or lists of strings; structural
                # values (offsets, indexes, nested dicts with ids) are recursed into
                drop = isinstance(v, str) or (
                    isinstance(v, list) and v and all(isinstance(x, str) for x in v))
            elif kl == "text" and parent not in TEXT_OK_PARENTS:
                drop = True
            elif kl == "raw" and not in_attempts:
                drop = isinstance(v, str) and len(v) > 0
            elif isinstance(v, str) and len(v) > MAX_FREE and kl not in KEEP_LONG:
                drop = True
            if drop and v not in (None, "", [], {}):
                out[k] = _marker(v)
                n += 1
                continue
            nv, c = strip(v, kl, in_attempts or kl == "attempts")
            out[k] = nv
            n += c
        return out, n
    if isinstance(o, list):
        out = []
        for v in o:
            # a list element inherits the list's key as parent
            if isinstance(v, str) and len(v) > MAX_FREE and parent not in KEEP_LONG:
                out.append(_marker(v)); n += 1
                continue
            nv, c = strip(v, parent, in_attempts)
            out.append(nv); n += c
        return out, n
    return o, 0


def find_long(o, path="", parent="", in_attempts=False, hits=None):
    """Verification: report strings > MAX_FREE outside answer/reason-type keys."""
    if hits is None:
        hits = []
    if isinstance(o, dict):
        for k, v in o.items():
            kl = k.lower() if isinstance(k, str) else k
            find_long(v, f"{path}.{k}", kl, in_attempts or kl == "attempts", hits)
    elif isinstance(o, list):
        for i, v in enumerate(o):
            find_long(v, f"{path}[{i}]", parent, in_attempts, hits)
    elif isinstance(o, str) and len(o) > MAX_FREE and parent not in KEEP_LONG:
        hits.append((path, len(o), o[:100]))
    return hits


def process(path: str, check: bool):
    hits_all, stripped = [], 0
    if path.endswith(".jsonl"):
        out_lines = []
        with open(path, encoding="utf-8-sig") as fh:
            for i, line in enumerate(fh):
                s = line.strip()
                if not s:
                    continue
                try:
                    rec = json.loads(s)
                except json.JSONDecodeError:
                    hits_all.append((f"line {i}", len(s), "UNPARSEABLE LINE"))
                    out_lines.append(None)
                    continue
                if check:
                    hits_all += [(f"line {i}{p}", L, ex) for p, L, ex in find_long(rec)]
                else:
                    rec, c = strip(rec)
                    stripped += c
                    out_lines.append(json.dumps(rec, ensure_ascii=False))
        if not check:
            if any(x is None for x in out_lines):
                raise SystemExit(f"{path}: unparseable lines; refusing to rewrite")
            with open(path, "w", encoding="utf-8", newline="\n") as fh:
                fh.write("\n".join(out_lines) + ("\n" if out_lines else ""))
    else:
        with open(path, encoding="utf-8-sig") as fh:
            rec = json.load(fh)
        if check:
            hits_all = find_long(rec)
        else:
            rec, stripped = strip(rec)
            with open(path, "w", encoding="utf-8", newline="\n") as fh:
                json.dump(rec, fh, ensure_ascii=False, indent=1)
    return stripped, hits_all


def main():
    if hasattr(sys.stdout, "reconfigure"):  # Persian text in LONG reports on a cp1252 console
        sys.stdout.reconfigure(errors="backslashreplace")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root")
    ap.add_argument("--check", action="store_true", help="verify only")
    a = ap.parse_args()
    total_stripped, total_hits, files = 0, 0, 0
    for dp, _, fns in os.walk(a.root):
        for f in sorted(fns):
            if not (f.endswith(".json") or f.endswith(".jsonl")):
                continue
            p = os.path.join(dp, f)
            files += 1
            try:
                n, hits = process(p, a.check)
            except (json.JSONDecodeError, UnicodeDecodeError) as e:
                print(f"SKIP-UNPARSEABLE {p}: {e}")
                total_hits += 1
                continue
            total_stripped += n
            for path, L, ex in hits[:5]:
                print(f"LONG {p} {path} ({L} chars): {ex!r}")
            total_hits += len(hits)
    print(f"files={files} stripped_fields={total_stripped} long_strings_remaining={total_hits}")
    if a.check and total_hits:
        sys.exit(1)


if __name__ == "__main__":
    main()
