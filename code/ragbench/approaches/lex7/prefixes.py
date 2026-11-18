"""LEX7 free contextual prefixes: question-independent, deterministic, zero LLM cost at query or index time (the cards
were written once by lex6/cards.py).

prefix(chunk) = compact doc card (leading "[XXnn] " display id removed) + "\n" + the covering section card.
Covering section = the section card of the chunk's document with the largest char overlap with the chunk span; ties ->
shortest section span, then list order. Failed / empty section cards are skipped; none -> doc card only.
The prefix is for the retrieval channels only (BM25 / dense / sparse / ColBERT text); the reader sees the chunk.
"""
import json
import re
from pathlib import Path

from ..lex6.cardsrc import CARDS_DIR

PREFIX_VERSION = "lex7-prefix-v1"
_DISPLAY = re.compile(r"^\s*\[[A-Z]{1,4}\d{1,4}\]\s*")


def load_cards(root=None):
    """-> {doc_id: card record} over every <pool>.jsonl."""
    recs = {}
    for f in sorted(Path(root or CARDS_DIR).glob("*.jsonl")):
        with open(f, encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    r = json.loads(line)
                    recs[r["doc_id"]] = r
    return recs


def doc_card(rec):
    return _DISPLAY.sub("", (rec.get("compact_card") or rec.get("full_card") or "")).strip()


def covering_section(rec, span):
    """-> the section dict covering span best (see module doc), or None."""
    a, b = span
    best, key = None, None
    for i, s in enumerate(rec.get("sections") or []):
        if s.get("status") == "failed" or not (s.get("card") or "").strip() or not s.get("span"):
            continue
        sa, sb = s["span"]
        ov = min(b, sb) - max(a, sa)
        if ov <= 0:
            continue
        k = (-ov, sb - sa, i)
        if key is None or k < key:
            best, key = s, k
    return best


def prefix(rec, chunk):
    """-> (prefix text, info {section key or None})."""
    if not rec:
        return "", {"section": None, "doc_card": False}
    d = doc_card(rec)
    s = covering_section(rec, chunk["span"])
    parts = [x for x in (d, (s or {}).get("card", "").strip()) if x]
    return "\n".join(parts), {"section": (s or {}).get("key"), "doc_card": bool(d)}


def prefixed_text(index_text, pre):
    return (pre + "\n" + index_text) if pre else index_text
