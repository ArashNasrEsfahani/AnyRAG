import requests, re, sys, time, json, html as H
from urllib.parse import unquote
UA={"User-Agent":"RAGStudyCorpus/1.0 (academic research; farhangsarakhavaran.daneshbon@gmail.com)"}
API="https://fa.wikisource.org/w/api.php"
def api(**p):
    p.setdefault("format","json"); p.setdefault("formatversion","2")
    for i in range(8):
        try:
            r=requests.get(API,params=p,headers=UA,timeout=60); r.raise_for_status(); return r.json()
        except Exception as e:
            time.sleep(5*(i+1)); err=e
    raise err
def parse_html(title):
    j=api(action="parse",page=title,prop="text|revid",redirects=1)
    if "error" in j: return None,None
    return j["parse"]["text"], j["parse"]["revid"]
def subpages(prefix):
    out=[];cont={}
    while True:
        j=api(action="query",list="allpages",apprefix=prefix+"/",apnamespace=0,aplimit=500,**cont)
        out+= [p["title"] for p in j["query"]["allpages"]]
        if "continue" in j: cont={"apcontinue":j["continue"]["apcontinue"]}
        else: break
    return out
def links_in_order(html, prefix=None):
    ts=[]
    for m in re.finditer(r'<a href="/wiki/([^"#?]+)[^"]*"[^>]*>', html):
        t=unquote(m.group(1)).replace("_"," ")
        if prefix is None or t.startswith(prefix+"/"):
            if t not in ts: ts.append(t)
    return ts
def plain(html):
    t=re.sub(r"<(script|style)[^>]*>.*?</\1>","",html,flags=re.S)
    t=re.sub(r"<[^>]+>"," ",t); t=H.unescape(t); return re.sub(r"\s+"," ",t).strip()
