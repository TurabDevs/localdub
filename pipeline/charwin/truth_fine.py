# word/0.1 s level truth: who is speaking at each 0.1 s step (speech = Whisper word OR energy VAD on Demucs vocals)
import json,numpy as np,soundfile as sf
R='/workspace/projects/db13b071-b36a-44da-bacb-efac377e3057/build/work/'
P=json.load(open('persons.json')); rows=json.load(open('windows.json')); man=json.load(open('manual_fixes.json'))['window_overrides']
cent=np.array([r['t0']+0.75 for r in rows])
v,sr=sf.read(R+'s1/vocals.flac'); v=v.mean(1) if v.ndim>1 else v
hop=int(sr*0.02); n=len(v)//hop
db=20*np.log10(np.array([np.sqrt(np.mean(v[i*hop:(i+1)*hop]**2)+1e-12) for i in range(n)])+1e-9)
sp=db>np.percentile(db,95)-28
W=json.load(open(R+'chars/whisper/whisper.json')); words=[(w['s'],w['e']) for s in W['segments'] for w in s['words']]
STEP=0.1; T=np.arange(0,307.5,STEP); out=[]
for t in T:
    c=t+STEP/2
    inw=any(a<=c<=b for a,b in words); e=sp[int(t/0.02):int((t+STEP)/0.02)]
    speech=inw or (len(e) and e.mean()>=0.5)
    if not speech: out.append(None); continue
    p=how=None
    for m in man:
        if m['from']<=c<=m['to']: p,how=m['person'],'manual'
    if not p:
        for q,vv in P.items():
            if any(a<=c<=b for a,b in vv['segs']): p,how=q,'visual'
    if not p:
        r=rows[int(np.argmin(np.abs(cent-c)))]
        if r['vad']>=0.3 and r['emb_margin']>=0.05: p,how=r['emb_best'],'voice'
        else: p,how='?','unsure'
    out.append([p,how])
json.dump({'step':STEP,'truth':out},open('truth_fine.json','w'))
from collections import Counter
print(Counter((o[1] if o else 'silence') for o in out))
