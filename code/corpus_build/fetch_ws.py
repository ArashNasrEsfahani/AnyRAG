import sys, json, os, time, re
from ws import *
sys.stdout.reconfigure(encoding="utf-8")
def fetch(title, outpath):
    if os.path.exists(outpath):
        return json.load(open(outpath,encoding="utf-8"))
    h,rev=parse_html(title)
    pages=[{"title":title,"revid":rev,"html":h}] if h is not None else []
    if h is None:
        h=""
    order=links_in_order(h,title)
    allp=subpages(title)
    seen={title}
    rest=[t for t in allp if t not in order]
    def nat(t):
        t=t.translate(str.maketrans("۰۱۲۳۴۵۶۷۸۹","0123456789"))
        return [(0,int(x),"") if x.isdigit() else (1,0,x) for x in re.split(r"(\d+)",t)]
    rest.sort(key=lambda t:(0 if t.endswith("/مقدمه") else 1, nat(t)))
    queue=[t for t in order if t in allp]+rest
    if not pages and not allp: raise SystemExit("missing "+title)
    for t in queue:
        if t in seen: continue
        seen.add(t); time.sleep(0.4)
        hh,rv=parse_html(t)
        if hh is None: continue
        pages.append({"title":t,"revid":rv,"html":hh})
    b={"work":title,"source":"https://fa.wikisource.org/wiki/"+title.replace(" ","_"),"retrieved":"2026-09-25","api":"action=parse (rendered HTML)","pages":pages}
    json.dump(b,open(outpath,"w",encoding="utf-8"),ensure_ascii=False)
    return b
if __name__=="__main__":
    lane=sys.argv[1]; 
    for t in sys.argv[2:]:
        slug=re.sub(r"[^\w]+","_",t).strip("_")
        p="D:/KareOne/RAG Paper/corpus/natural/%s/_orig/wikisource_%s.json"%(lane,slug)
        b=fetch(t,p)
        n=sum(len(plain(x["html"])) for x in b["pages"])
        inc=any("ناتمام" in plain(x["html"])[:600] for x in b["pages"])
        print(t,"| pages",len(b["pages"]),"| chars",n,"| incomplete-flag",inc,flush=True)
