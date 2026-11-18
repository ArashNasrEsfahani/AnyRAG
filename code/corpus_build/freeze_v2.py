import csv,json,hashlib,os,glob,datetime
R="D:/KareOne/RAG Paper"
def sha(p):
    h=hashlib.sha256(); f=open(p,"rb")
    for b in iter(lambda:f.read(1<<20),b""): h.update(b)
    return h.hexdigest()
src=json.load(open(R+"/work/datasets_v2/sources/sources_v2.json",encoding="utf-8"))["sources"]
out={"frozen_at":datetime.datetime.now().isoformat(timespec="seconds"),"manifest":"corpus/natural/manifest_v2.csv","manifest_sha256":sha(R+"/corpus/natural/manifest_v2.csv"),
     "why":"ragbench.tools.freeze_extend writes only into the frozen study manifest (corpus/frozen) and takes no manifest argument; v2 sources are recorded here instead","documents":[]}
for r in csv.DictReader(open(R+"/corpus/natural/manifest_v2.csv",encoding="utf-8-sig")):
    lp=r["local_path"].replace("\\","/"); lane,fn=lp.split("/"); stem=os.path.splitext(fn)[0]
    xd="corpus/extracted/%s/%s"%(lane,stem)
    xf=sorted(glob.glob(R+"/"+xd+"/document.json")+glob.glob(R+"/"+xd+"/document.md")+glob.glob(R+"/"+xd+"/tables/*.json"))
    st=json.load(open(R+"/"+xd+"/extract_status.json",encoding="utf-8"))
    out["documents"].append({"lane":r["lane"],"local_path":"corpus/natural/"+lp,"sha256":sha(R+"/corpus/natural/"+lp),
      "originals":src[r["local_path"]]["originals"],"verdict":st.get("verdict"),
      "extraction":[{"path":os.path.relpath(p,R).replace("\\","/"),"sha256":sha(p)} for p in xf]})
json.dump(out,open(R+"/work/datasets_v2/sources/FROZEN_v2.json","w",encoding="utf-8"),ensure_ascii=False,indent=1)
print(len(out["documents"]))
