import sys, glob
from PIL import Image, ImageDraw
d=sys.argv[1]; out=sys.argv[2]; per=int(sys.argv[3]) if len(sys.argv)>3 else 8
fs=sorted(glob.glob(d+'/p-*.pgm'))
w,h=Image.open(fs[0]).size
rows=(len(fs)+per-1)//per
S=Image.new('L',(per*(w+6),rows*(h+6)),128)
for i,f in enumerate(fs):
    im=Image.open(f); x=(i%per)*(w+6); y=(i//per)*(h+6)
    S.paste(im,(x,y)); ImageDraw.Draw(S).text((x+5,y+5),str(i+1),fill=0)
S.save(out)
