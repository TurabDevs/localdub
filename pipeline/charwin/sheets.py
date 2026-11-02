import json,sys
from PIL import Image,ImageDraw,ImageFont
rows=json.load(open('windows.json')); L=json.load(open(sys.argv[2] if len(sys.argv)>2 else 'v9_lines.json')); TAG=sys.argv[3] if len(sys.argv)>3 else 'v9'; OUTD=sys.argv[4] if len(sys.argv)>4 else 'sheets'
try: F=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',15); Fs=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',13)
except Exception: F=Fs=ImageFont.load_default()
W,H,CAP=384,216,52
def line_at(t):
    for l in L:
        if l['start'] is not None and l['start']<=t<l['end']: return l
COL={'F0':'red','F1':'yellow','F2':'cyan','F3':'lime','F4':'magenta','F5':'orange','F6':'white','F7':'pink','F8':'deepskyblue','F9':'gold'}
start=float(sys.argv[1]) if len(sys.argv)>1 else 0
for s0 in range(0,310,10):
    tiles=[r for r in rows if s0<=r['t0']+0.75<s0+10]
    if not tiles: continue
    img=Image.new('RGB',(W*5,(H+CAP)*4),'black'); d=ImageDraw.Draw(img)
    for k,r in enumerate(tiles[:20]):
        i=rows.index(r); x,y=(k%5)*W,(k//5)*(H+CAP)
        try: im=Image.open('fr/%04d.jpg'%(i+1)).convert('RGB')
        except Exception: continue
        img.paste(im,(x,y)); 
        lab=[]
        for f in r['faces']:
            bx=f.get('box')
            if not bx: continue
            c=COL.get(f['fid'],'gray'); X0,Y0=x+bx[0]*W,y+bx[1]*H
            d.rectangle([X0,Y0,X0+bx[2]*W,Y0+bx[3]*H],outline=c,width=2)
            d.text((X0+2,Y0+2),f"{f['fid']} {f['lip']:.1f}",fill=c,font=F)
        tc=r['t0']+0.75
        l=line_at(tc)
        d.text((x+4,y+H+2),f"{tc:.2f}s VAD{r['vad']:.2f}{' CUT' if r['cut'] else ''} {r.get('vc','')}",fill='white',font=F)
        d.text((x+4,y+H+19),' '.join(f"{f['fid']}:{f['lip']:.2f}" for f in r['faces'][:4]),fill='yellow',font=Fs)
        d.text((x+4,y+H+35),(f"{TAG} {l['id']} {l['cm']}/{l['face']} {'TTS' if l['variant']!='original' else 'orig'}" if l else 'v9 -'),fill='cyan',font=Fs)
    img.save(f'{OUTD}/s{s0:03d}.jpg',quality=80)
print('ok')
