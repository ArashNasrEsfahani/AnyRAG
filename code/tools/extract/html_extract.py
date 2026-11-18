"""HTML source (a blog-post report) -> the same block structure as the PDF path.
Standard library only (html.parser)."""
import re
from html.parser import HTMLParser

import fa_text as F

SKIP_TAGS = {"script", "style", "noscript", "svg", "button", "form", "nav", "iframe", "select", "option", "template"}
VOID = {"br", "img", "hr", "meta", "link", "input", "source", "wbr", "area", "base", "col", "embed", "param", "track"}
BLOCK_TAGS = {"p", "h1", "h2", "h3", "h4", "h5", "h6", "li", "blockquote", "figcaption", "pre", "dt", "dd", "caption"}
CONTENT_CLASS_RE = re.compile(r"\b(blog-content|entry-content|post-content|article-content|single-content)\b")


class _Node:
    __slots__ = ("tag", "attrs", "children", "parent")

    def __init__(self, tag, attrs, parent):
        self.tag = tag
        self.attrs = dict(attrs)
        self.children = []
        self.parent = parent


class _TreeBuilder(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.root = _Node("#root", [], None)
        self.cur = self.root

    def handle_starttag(self, tag, attrs):
        node = _Node(tag, attrs, self.cur)
        self.cur.children.append(node)
        if tag not in VOID:
            self.cur = node

    def handle_startendtag(self, tag, attrs):
        self.cur.children.append(_Node(tag, attrs, self.cur))

    def handle_endtag(self, tag):
        n = self.cur
        while n is not None and n.tag != tag:
            n = n.parent
        if n is not None and n.parent is not None:
            self.cur = n.parent

    def handle_data(self, data):
        self.cur.children.append(data)


def _find(node, pred):
    if isinstance(node, str):
        return None
    if pred(node):
        return node
    for ch in node.children:
        r = _find(ch, pred)
        if r is not None:
            return r
    return None


def _text(node):
    if isinstance(node, str):
        return node
    if node.tag in SKIP_TAGS:
        return ""
    if node.tag == "br":
        return " "
    return "".join(_text(c) for c in node.children)


def _clean(s):
    return re.sub(r"\s+", " ", s or "").strip()


def extract_html(path):
    with open(path, "rb") as fh:
        raw = fh.read()
    html = raw.decode("utf-8", errors="replace")
    tb = _TreeBuilder()
    tb.feed(html)
    root = tb.root
    title_node = _find(root, lambda n: n.tag == "title")
    title = _clean(_text(title_node)) if title_node else ""
    h1 = _find(root, lambda n: n.tag == "h1")
    content = _find(root, lambda n: n.tag in ("div", "article", "section", "main") and CONTENT_CLASS_RE.search(n.attrs.get("class", "") or ""))
    used_root = "div.blog-content"
    if content is None:
        content = _find(root, lambda n: n.tag == "article") or _find(root, lambda n: n.tag == "body") or root
        used_root = content.tag
    items = []   # (kind, payload)
    skipped = {"toc": 0}

    if h1 is not None and _clean(_text(h1)):
        items.append(("heading", {"level": 1, "text": _clean(_text(h1)), "tag": "h1"}))
    elif title:
        items.append(("heading", {"level": 1, "text": title.split("|")[0].strip(), "tag": "title"}))

    def walk(node):
        if isinstance(node, str):
            t = _clean(node)
            if t:
                items.append(("paragraph", {"text": t, "tag": "#text"}))
            return
        tag = node.tag
        if tag in SKIP_TAGS:
            return
        if node.attrs.get("id") == "table-of-contents" or "toc" in (node.attrs.get("class", "") or "").split():
            skipped["toc"] += 1
            return
        if tag in ("h1", "h2", "h3", "h4", "h5", "h6"):
            t = _clean(_text(node))
            if t:
                lvl = int(tag[1])
                items.append(("heading", {"level": 1 if lvl == 1 else (2 if lvl == 2 else 3), "text": t, "tag": tag}))
            return
        if tag == "table":
            rows = []
            for tr in _iter(node, "tr"):
                row = []
                for cell in tr.children:
                    if not isinstance(cell, str) and cell.tag in ("td", "th"):
                        row.append({"text": _clean(_text(cell)), "th": cell.tag == "th"})
                if row:
                    rows.append(row)
            if rows:
                items.append(("table", {"rows": rows}))
            return
        if tag == "img":
            items.append(("image", {"src": node.attrs.get("src") or node.attrs.get("data-src"), "alt": _clean(node.attrs.get("alt", ""))}))
            return
        if tag in ("ul", "ol"):
            for ch in node.children:
                if not isinstance(ch, str) and ch.tag == "li":
                    t = _clean(_text(ch))
                    if t:
                        items.append(("list_item", {"text": t, "tag": "li"}))
                    for img in _iter(ch, "img"):
                        items.append(("image", {"src": img.attrs.get("src"), "alt": _clean(img.attrs.get("alt", ""))}))
            return
        if tag in BLOCK_TAGS:
            t = _clean(_text(node))
            if t:
                kind = "caption" if tag in ("figcaption", "caption") else ("list_item" if tag == "li" else "paragraph")
                items.append((kind, {"text": t, "tag": tag}))
            for img in _iter(node, "img"):
                items.append(("image", {"src": img.attrs.get("src"), "alt": _clean(img.attrs.get("alt", ""))}))
            return
        for ch in node.children:
            walk(ch)

    for ch in content.children:
        walk(ch)
    return {"title": title, "items": items, "content_root": used_root, "skipped": skipped, "html_chars": len(html)}


def _iter(node, tag):
    for ch in node.children:
        if isinstance(ch, str):
            continue
        if ch.tag == tag:
            yield ch
        elif ch.tag != "table":
            yield from _iter(ch, tag)
