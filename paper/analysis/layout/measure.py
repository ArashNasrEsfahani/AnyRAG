import sys, glob, os, subprocess, numpy as np
from PIL import Image
pdf=sys.argv[1]; out=sys.argv[2]
os.makedirs(out,exist_ok=True)
for f in glob.glob(out+'/p-*'): os.remove(f)
subprocess.run(['pdftoppm','-r','40','-gray',pdf,out+'/p'],check=True)
files=sorted(glob.glob(out+'/p-*.pgm'))
T,B=39,431; H=B-T
flag=[]
for i,f in enumerate(files,1):
    a=np.array(Image.open(f))<200
    res=[]
    for x0,x1 in ((37,163),(167,293)):
        rows=a[T:B,x0:x1].any(1)
        best=cur=0
        for v in rows:
            cur=0 if v else cur+1; best=max(best,cur)
        tail=0
        for v in rows[::-1]:
            if v: break
            tail+=1
        res.append((best/H, tail/H))
    m=max(r[0] for r in res)
    tag='  <-- ' if m>0.25 else ''
    if m>0.25: flag.append(i)
    print(f'p{i:02d} L gap {res[0][0]:.2f} (tail {res[0][1]:.2f})  R gap {res[1][0]:.2f} (tail {res[1][1]:.2f}){tag}')
print('pages',len(files),'flagged(>25%)',flag, 'excluding last:', [p for p in flag if p!=len(files)])
