import sys, re, os
from simplify import *
sys.stdout.reconfigure(encoding="utf-8")
base="D:/KareOne/RAG Paper/corpus/natural/law_en"
def post(bl):
    out=[]
    for k,v in bl:
        if k=="table" and all(len(r)==2 and len(r[0])<=8 for r in v):
            out+= [("p",(r[0]+" "+r[1]).strip()) for r in v]
        elif k=="table" and all(len(r)==1 for r in v):
            out+= [("p",r[0]) for r in v]
        else: out.append((k,v))
    return out
def eu(src):
    root=dom(src)
    def head(n):
        c=(n.attrs.get("class") or "").split()
        if n.tag!="p" or not c: return None
        if "oj-ti-section-1" in c:
            return 2 if clean(text(n)).upper().startswith(("CHAPTER","TITLE","PART")) else 3
        if "oj-ti-section-2" in c or "oj-sti-art" in c: return "merge"
        if "oj-ti-art" in c: return 4
        if "oj-doc-ti" in c: return 2 if clean(text(n)).upper().startswith("ANNEX") else 1
        if "oj-ti-grseq-1" in c: return 3
        return None
    def skip(n): return has_class(n,"oj-hd-date","oj-hd-lg","oj-hd-oj","oj-hd-coll","oj-hd-uniq")
    return post(simplify(find(root,lambda n:n.tag=="body"),skip,head))
def uk(src):
    src=re.sub(r'<span class="(LegDS|LegPartTitle|LegScheduleTitle|LegPblockTitle|LegSchedulePartTitle)',r' <span class="\1',src)
    root=dom(src)
    def skip(n):
        return has_class(n,"LegAnnotations","LegExtentRestriction","LegCommentaryLink","LegChangeDelimiter","LegClearFix") and n.tag in ("div","span","a") and not has_class(n,"LegP1ContainerFirst","LegP2Container","LegP3Container") if False else has_class(n,"LegAnnotations","LegExtentRestriction","LegCommentaryLink","LegChangeDelimiter")
    body=find(root,lambda n:has_class(n,"DocContainer")) or find(root,lambda n:n.tag=="body")
    return post(simplify(body,skip))
EU={"32016R0679":("gdpr","Regulation (EU) 2016/679 (General Data Protection Regulation)","2016"),
"32024R1689":("ai-act","Regulation (EU) 2024/1689 (Artificial Intelligence Act)","2024"),
"32022R2065":("dsa","Regulation (EU) 2022/2065 (Digital Services Act)","2022"),
"32022R1925":("dma","Regulation (EU) 2022/1925 (Digital Markets Act)","2022"),
"32022R0868":("data-governance-act","Regulation (EU) 2022/868 (Data Governance Act)","2022"),
"32023R2854":("data-act","Regulation (EU) 2023/2854 (Data Act)","2023")}
UK={(2015,30):("modern-slavery-act","Modern Slavery Act 2015"),(2015,15):("consumer-rights-act","Consumer Rights Act 2015"),
(2018,12):("data-protection-act","Data Protection Act 2018"),(2010,23):("bribery-act","Bribery Act 2010")}
if __name__=="__main__":
    res={}
    for c,(slug,t,y) in EU.items():
        bl=eu(open("%s/_orig/eurlex_%s.html"%(base,c),encoding="utf-8").read())
        stem="eurlex-%s_%s_en"%(c,slug); res[stem]=(t,bl)
    for (y,n),(slug,t) in UK.items():
        bl=uk(open("%s/_orig/legislation-gov-uk_ukpga-%d-%d.xht"%(base,y,n),encoding="utf-8").read())
        stem="ukpga-%d-%d_%s_en"%(y,n,slug); res[stem]=(t,bl)
    for stem,(t,bl) in res.items():
        open("%s/%s.html"%(base,stem),"w",encoding="utf-8").write(render(t,bl,"en"))
        hs=[v for k,v in bl if k.startswith("h")]
        print(stem,"blocks",len(bl),"heads",len(hs),"chars",sum(len(v) for k,v in bl if k!="table"),"tables",sum(1 for k,v in bl if k=="table"))
        print("   ",hs[:3],hs[20:23])
