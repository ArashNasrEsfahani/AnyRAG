"""Upsert rows into corpus/natural/manifest_v2.csv and provenance into work/datasets_v2/sources/sources_v2.json."""
import csv, json, hashlib, os
ROOT="D:/KareOne/RAG Paper"
MAN=ROOT+"/corpus/natural/manifest_v2.csv"
SRC=ROOT+"/work/datasets_v2/sources/sources_v2.json"
COLS="lane,company,title_fa,title_en,year_persian,year_gregorian,language,parallel_file,format,pages,size_kb,source_url,landing_url,local_path,sha256,notes".split(",")
def sha(p):
    h=hashlib.sha256()
    with open(p,"rb") as f:
        for b in iter(lambda:f.read(1<<20),b""): h.update(b)
    return h.hexdigest()
def upsert(row, prov):
    lp=row["local_path"]; full=os.path.join(ROOT,"corpus/natural",lp.replace("\\","/"))
    row["sha256"]=sha(full); row["size_kb"]=str(round(os.path.getsize(full)/1024)); row.setdefault("format","html"); row.setdefault("pages","1")
    rows=[]
    if os.path.exists(MAN):
        with open(MAN,encoding="utf-8-sig",newline="") as f: rows=[r for r in csv.DictReader(f) if r["local_path"]!=lp]
    rows.append({c:row.get(c,"") for c in COLS})
    rows.sort(key=lambda r:(r["lane"],r["local_path"]))
    tmp=MAN+".tmp"
    with open(tmp,"w",encoding="utf-8",newline="") as f:
        w=csv.DictWriter(f,COLS); w.writeheader(); w.writerows(rows)
    os.replace(tmp,MAN)
    src=json.load(open(SRC,encoding="utf-8")) if os.path.exists(SRC) else {"about":"Provenance of the datasets_v2 BOOK and LAW pools (DESIGN.md source rule 2). One entry per manifest_v2 local_path.","sources":{}}
    origs=prov.pop("originals",[])
    prov["originals"]=[{"path":o,"sha256":sha(os.path.join(ROOT,"corpus/natural",o))} for o in origs]
    prov.update(lane=row["lane"],title=row.get("title_en") or row.get("title_fa"),input_sha256=row["sha256"])
    src["sources"][lp]={**src["sources"].get(lp,{}),**prov}
    tmp=SRC+".tmp"; json.dump(src,open(tmp,"w",encoding="utf-8"),ensure_ascii=False,indent=1); os.replace(tmp,SRC)
