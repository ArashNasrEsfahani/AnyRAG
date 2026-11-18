"""Tiny stdlib DOM + simplifier: arbitrary HTML -> blocks [(kind, text)] with kind in h1..h4, p, li, table."""
import re, html as H
from html.parser import HTMLParser
VOID={"br","img","hr","meta","link","input","col","area","base","wbr","source","embed","param","track"}
class Node:
    __slots__=("tag","attrs","kids","parent")
    def __init__(s,tag,attrs,parent): s.tag=tag; s.attrs=attrs; s.kids=[]; s.parent=parent
class TB(HTMLParser):
    def __init__(s):
        super().__init__(convert_charrefs=True); s.root=Node("#root",{},None); s.cur=s.root
    def handle_starttag(s,tag,attrs):
        n=Node(tag,{k:(v or "") for k,v in attrs},s.cur); s.cur.kids.append(n)
        if tag not in VOID: s.cur=n
    def handle_startendtag(s,tag,attrs):
        s.cur.kids.append(Node(tag,{k:(v or "") for k,v in attrs},s.cur))
    def handle_endtag(s,tag):
        n=s.cur
        while n is not None and n.tag!=tag: n=n.parent
        if n is not None and n.parent is not None: s.cur=n.parent
    def handle_data(s,d): s.cur.kids.append(d)
def dom(src):
    tb=TB(); tb.feed(src); return tb.root
_SKIP=[None]
def text(n):
    if isinstance(n,str): return n
    if n.tag=="br": return "\n"
    if n.tag in ("script","style") or (_SKIP[0] and _SKIP[0](n)): return ""
    return "".join(text(k) for k in n.kids)
def clean(t): return re.sub(r"[ \t\r\f\v ]+"," ",re.sub(r"\s*\n\s*"," ",t)).strip()
BLOCK={"p","div","li","h1","h2","h3","h4","h5","h6","table","ul","ol","dl","dd","dt","blockquote","section","article","center","pre","tr","td","th","tbody","thead","body","html","main","header","footer","nav","figure","figcaption","poem"}
def simplify(root, skip, head=None):
    """skip(node)->bool drops a subtree. head(node)->None | int level | "merge" (append to previous heading).
    Returns list of (kind,text) blocks."""
    out=[]; buf=[]; _SKIP[0]=skip
    def flush():
        t=clean("".join(buf)); buf.clear()
        if t: out.append(("p",t))
    def walk(n):
        if isinstance(n,str): buf.append(n); return
        if n.tag in ("script","style","noscript","head","title") or skip(n): return
        tg=n.tag
        hv=head(n) if head else None
        if hv is not None:
            flush(); t=clean(text(n))
            if t:
                if hv=="merge" and out and out[-1][0].startswith("h"):
                    out[-1]=(out[-1][0], out[-1][1]+" — "+t)
                else:
                    out.append(("h%d"%(hv if hv!="merge" else 4),t))
            return
        if tg in ("h1","h2","h3","h4","h5","h6"):
            flush(); t=clean(text(n))
            if t: out.append(("h%d"%min(int(tg[1]),4),t)); return
            return
        if tg=="li":
            flush(); t=clean(text(n))
            if t: out.append(("li",t)); return
        if tg=="table":
            flush(); rows=[]
            for tr in iter_tags(n,"tr"):
                cells=[clean(text(c)) for c in tr.kids if not isinstance(c,str) and c.tag in ("td","th")]
                if any(cells): rows.append(cells)
            if rows: out.append(("table",rows))
            return
        if tg=="br": buf.append("\n"); flush(); return
        isb=tg in BLOCK
        if isb: flush()
        for k in n.kids: walk(k)
        if isb: flush()
    walk(root); flush(); return out
def iter_tags(n,tag):
    for k in n.kids:
        if isinstance(k,str): continue
        if k.tag==tag: yield k
        elif k.tag!="table": yield from iter_tags(k,tag)
def find(n,pred):
    if isinstance(n,str): return None
    if pred(n): return n
    for k in n.kids:
        r=find(k,pred)
        if r is not None: return r
    return None
def has_class(n,*cs):
    c=(n.attrs.get("class") or "").split(); return any(x in c for x in cs)
def render(title, blocks, lang, meta=None):
    e=H.escape; parts=[]
    for k,v in blocks:
        if k=="table":
            rows="".join("<tr>%s</tr>"%"".join("<td>%s</td>"%e(c) for c in r) for r in v)
            parts.append("<table>%s</table>"%rows)
        elif k=="li": parts.append("<ul><li>%s</li></ul>"%e(v))
        else: parts.append("<%s>%s</%s>"%(k,e(v),k))
    d=' dir="rtl"' if lang=="fa" else ""
    m="".join('<meta name="%s" content="%s">'%(e(a),e(str(b))) for a,b in (meta or {}).items())
    return "<!doctype html><html lang='%s'%s><head><meta charset='utf-8'><title>%s</title>%s</head><body><article>\n%s\n</article></body></html>"%(lang,d,e(title),m,"\n".join(parts))
