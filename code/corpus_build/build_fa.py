import sys, json, re, os, glob
from conv_fa import convert
from simplify import render
from reg import upsert
sys.stdout.reconfigure(encoding="utf-8")
ROOT="D:/KareOne/RAG Paper/corpus/natural"
ENDP=re.compile(r"[.:؛،!?؟»)\]«(]$")
FA_DIG=str.maketrans("0123456789","۰۱۲۳۴۵۶۷۸۹")
def law_post(bl):
    out=[]
    for k,v in bl:
        if k!="p": out.append((k,v)); continue
        v=v.strip("‌‏‎ ")
        m=re.match(r"^=+\s*(.*?)\s*=+$",v)
        if m: out.append(("h3",m.group(1))); continue
        parts=re.split(r"(?<=[.:؛])\s+(?=ماده\s*[۰-۹0-9]+\s*[-ـ–])",v)
        for x in parts:
            if len(x)<120 and re.match(r"^(باب|فصل|بخش|مبحث)\s",x) and not ENDP.search(x): out.append(("h3",x))
            else: out.append(("p",x))
    return out
def book_post(bl):
    out=[]
    for i,(k,v) in enumerate(bl):
        nxt=bl[i+1] if i+1<len(bl) else None
        if k=="p" and 3<=len(v)<=70 and not ENDP.search(v) and not re.fullmatch(r"[\d۰-۹\s\-–—/]+",v) \
           and nxt and nxt[0]=="p" and len(nxt[1])>200 and (not out or out[-1][0]!="h3"):
            out.append(("h3",v))
        else: out.append((k,v))
    return out
def prehtml(b):
    for p in b["pages"]:
        h=p["html"]
        if h.count("<p>")<5 and h.count("\n")>100:
            p["html"]=re.sub("\n(?=\u200c|(?:ماده|فصل|تبصره|بخش|باب)\s|[آ-ی]{1,2} - )","<br>",h)
    return b
def build(lane, work, stem, title_fa, title_en, year_fa, year_g, author, notes, prov, exclude=()):
    slug=re.sub(r"[^\w]+","_",work).strip("_")
    orig="%s/_orig/wikisource_%s.json"%(lane,slug)
    b=prehtml(json.load(open("%s/%s"%(ROOT,orig),encoding="utf-8")))
    bl,used,dropped=convert(b,title_fa,exclude=exclude)
    ZSTAT[0]=0
    bl=zfix_blocks(law_post(bl) if lane=="law_fa" else book_post(bl))
    if ZSTAT[0]: notes=(notes+"; " if notes else "")+"source-side ZWNJ used as word space: %d ZWNJ next to function words/punctuation replaced by a space or removed"%ZSTAT[0]
    fname="%s_fa.html"%stem
    open("%s/%s/%s"%(ROOT,lane,fname),"w",encoding="utf-8").write(render(title_fa,bl,"fa"))
    revs={p["title"]:p["revid"] for p in b["pages"]}
    url="https://fa.wikisource.org/wiki/"+work.replace(" ","_")
    upsert({"lane":lane.upper().replace("_FA","_fa"),"company":author,"title_fa":title_fa,"title_en":title_en,"year_persian":year_fa,"year_gregorian":year_g,
            "language":"fa","source_url":url,"landing_url":url,"local_path":"%s\%s"%(lane,fname),
            "notes":"fa.wikisource transcription, rendered via MediaWiki API action=parse on 2026-09-25 (%d pages used); original bundle %s; converted to simple HTML (headers, page numbers, footnote markers, figures, index pages removed; duplicated transclusions dropped). %s"%(len(used),orig,notes)},
           dict(prov, url=url, originals=[orig], wikisource_pages_used=[(t,revs[t]) for t in used], wikisource_pages_dropped=dropped))
    hs=sum(1 for k,v in bl if k.startswith("h")); ch=sum(len(v) for k,v in bl if k!="table")
    print("%-45s blocks %5d heads %4d chars %8d used %d dropped %d zwnj_fix %d"%(stem,len(bl),hs,ch,len(used),len(dropped),ZSTAT[0]),flush=True)

FW="را|یا|که|در|از|به|با|و|تا|این|آن|هر|بر|اگر|برای|نیز|هم|چون|پس|بی|می"
_ZA=re.compile(r"(?<=[؀-ۿ])‌(?=(?:را|یا|که|در|از|به|با|و|تا|این|آن|هر|بر|اگر|برای|نیز)(?:[\s.,،؛:!؟)»]|$))")
_ZB=re.compile(r"(?:(?<=^)|(?<=[\s(«]))(را|یا|که|در|از|به|با|و|تا|این|آن|هر|بر|اگر|برای|نیز)‌(?=[؀-ۿ])")
_ZP=re.compile(r"‌(?=[^؀-ۿ])|(?<=[^؀-ۿ])‌")
ZSTAT=[0]
def zwnj_fix(t):
    n0=t.count("‌")
    t=_ZA.sub(" ",t); t=_ZB.sub(r"\1 ",t); t=_ZP.sub("",t)
    ZSTAT[0]+=n0-t.count("‌"); return t
def zfix_blocks(bl):
    return [(k,[[zwnj_fix(c) for c in r] for r in v] if k=="table" else zwnj_fix(v)) for k,v in bl]
