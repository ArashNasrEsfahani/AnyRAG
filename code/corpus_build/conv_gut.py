import sys, os, re, json
from simplify import *
sys.stdout.reconfigure(encoding="utf-8")
BOOKS={1228:"darwin_origin-of-species-1st-ed",2009:"darwin_origin-of-species-6th-ed",34901:"mill_on-liberty",4239:"malthus_essay-on-population",
815:"tocqueville_democracy-in-america-part1",1015:"parkman_oregon-trail",20203:"franklin_autobiography",23:"douglass_narrative",
833:"veblen_theory-of-leisure-class",14474:"faraday_chemical-history-of-a-candle",147:"paine_common-sense"}
base="D:/KareOne/RAG Paper/corpus/natural/book_en"
def skip(n):
    i=n.attrs.get("id","")
    if i in ("pg-header","pg-footer","pg-machine-header","pg-start-separator","pg-end-separator"): return True
    if has_class(n,"pagenum","pgheader","pg-boilerplate","toc","footnote-ref","fnanchor","pageno","linenum","figcenter","figleft","figright","illustration","caption","transnote","tnote","mynote"): return True
    if n.tag=="span" and has_class(n,"pagenum"): return True
    return False
for i,slug in BOOKS.items():
    src=open("%s/_orig/pg%d-images.html"%(base,i),encoding="utf-8",errors="replace").read()
    root=dom(src)
    title=clean(text(find(root,lambda n:n.tag=="title"))).replace("The Project Gutenberg eBook of ","").replace("The Project Gutenberg E-text of ","").split("|")[0].strip()
    body=find(root,lambda n:n.tag=="body")
    bl=simplify(body,skip)
    # drop tables that are tables of contents (most cells short and many links) - keep data tables
    bl2=[]
    for k,v in bl:
        if k=="table" and len(v)>5 and sum(1 for r in v for c in r if re.search(r"^(CHAPTER|Chapter|[IVXLC]+\.?$|\d+$|Page)",c))> len(v): continue
        bl2.append((k,v))
    heads=[v for k,v in bl2 if k in("h1","h2","h3")]
    chars=sum(len(v) for k,v in bl2 if k!="table")
    stem="gutenberg-%d_%s_en"%(i,slug)
    open("%s/%s.html"%(base,stem),"w",encoding="utf-8").write(render(title,bl2,"en",{"source":"https://www.gutenberg.org/ebooks/%d"%i}))
    print(i,stem,"blocks",len(bl2),"heads",len(heads),"chars",chars,"|",heads[:6])
