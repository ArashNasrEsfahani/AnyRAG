import csv, subprocess, sys, json, os, re
sys.stdout.reconfigure(encoding="utf-8")
ROOT="D:/KareOne/RAG Paper"; lane=sys.argv[1]; extra=sys.argv[2:]
rows=[r for r in csv.DictReader(open(ROOT+"/corpus/natural/manifest_v2.csv",encoding="utf-8-sig")) if r["lane"]==lane]
files=[r["local_path"] for r in rows]
env=dict(os.environ)
p=subprocess.run([sys.executable,"tools/extract/extract.py","--manifest","corpus/natural/manifest_v2.csv","--out","corpus/extracted","--files",*files,*extra],cwd=ROOT,capture_output=True,text=True,encoding="utf-8",env=env)
print(p.stdout[-6000:]); print(p.stderr[-3000:])
