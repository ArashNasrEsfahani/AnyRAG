"""fa.wikisource bundle (_orig/wikisource_*.json) -> one simple HTML document (h1 = work title, h2 = subpage)."""
import sys, json, re, os
from simplify import *
sys.stdout.reconfigure(encoding="utf-8")
INDEX_TAILS=("اعلام","علام","فهرست اعلام","فهرست نامها","فهرست نامها ۲","واژه‌ها","واژه‌نامه","غلطنامه","فهرست اختصارات")
def skip(n):
    if n.tag in ("figure","sup") and (n.tag=="figure" or has_class(n,"reference")): return True
    if n.attrs.get("id") in ("toc","headerContainer","ws-data"): return True
    return has_class(n,"ws-noexport","noprint","headertemplate","pagenum","mw-editsection","searchaux","mw-cite-backlink","wst-rh","toc","mw-empty-elt","sistersitebox","licenseContainer","licenseBanner","ws-summary") 
def norm(t): return re.sub(r"[\s\u200c\u200f\u200e]+","",t)
def convert(bundle, title, exclude=(), keep_main=None, min_dup=40):
    pages=bundle["pages"]; work=bundle["work"]
    multi=len(pages)>1
    blocks=[("h1",title)]; seen=set(); used=[]; dropped=[]
    for p in pages:
        tail=p["title"].split("/",1)[1] if "/" in p["title"] else ""
        if p["title"]==work and multi and not keep_main: dropped.append((p["title"],"main page (table of contents)")); continue
        if tail in INDEX_TAILS or tail.split("/")[-1] in INDEX_TAILS or any(re.search(x,tail) for x in exclude):
            dropped.append((p["title"],"index/excluded")); continue
        root=dom(p["html"])
        hs=find(root,lambda n:n.attrs.get("id")=="header_section_text")
        head=clean(text(hs)) if hs is not None else ""
        _=None
        bl=simplify(root,skip)
        out=[]; new=0
        for k,v in bl:
            if k.startswith("h"):
                out.append(("h%d"%min(int(k[1])+1+(1 if multi else 0),4),v)); continue
            key=norm(v if k!="table" else json.dumps(v,ensure_ascii=False))
            if len(key)>=min_dup:
                if key in seen: continue
                seen.add(key); new+=1
            out.append((k,v))
        if multi:
            if new==0: dropped.append((p["title"],"all content duplicated elsewhere")); continue
            h2=head or tail.replace("/"," — ")
            out=[("h2",h2)]+out
        blocks+=out; used.append(p["title"])
    return blocks, used, dropped
if __name__=="__main__":
    lane, src = sys.argv[1], sys.argv[2]
    b=json.load(open(src,encoding="utf-8"))
    bl,used,dropped=convert(b,b["work"])
    print(len(bl),sum(len(v) for k,v in bl if k!="table"),dropped)
    for k,v in bl[:40]: print(k,str(v)[:120])
