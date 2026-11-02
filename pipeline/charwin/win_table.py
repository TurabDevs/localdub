# per-window truth table: 1.5 s windows, 0.5 s hop
import json, numpy as np, soundfile as sf, sys, os
R='/workspace/projects/db13b071-b36a-44da-bacb-efac377e3057/build/work/'
WL, HOP = 1.5, 0.5
fi=json.load(open(R+'chars/faces/face_ids.json'))
t2f=fi['track_face']
frames={}; gd={}
for x in 'abcdef':
    a=json.load(open(R+f'avqa/src_faces_{x}.json'))
    for k,v in a['frames'].items():
        t=round(float(k),3)
        frames[t]=[dict(d, fid=t2f.get(f'src_faces_{x}:{d["track"]}','?'), tk=f'{x}:{d["track"]}') for d in v]
    for k,v in a['gdiff'].items(): gd[round(float(k),3)]=v or 0
ts=sorted(frames)
# shot cuts
cuts=[t for t in sorted(gd) if gd[t]>30]
json.dump(cuts,open('cuts.json','w'))
# VAD on vocals
v,sr=sf.read(R+'s1/vocals.flac'); v=v.mean(1) if v.ndim>1 else v
hop=int(sr*0.02); n=len(v)//hop
rms=np.array([np.sqrt(np.mean(v[i*hop:(i+1)*hop]**2)+1e-12) for i in range(n)])
db=20*np.log10(rms+1e-9); thr=np.percentile(db,95)-28
speech=db>thr
# whisper words
W=json.load(open(R+'chars/whisper/whisper.json'))
words=[(w['s'],w['e'],w['w']) for s in W['segments'] for w in s.get('words',[])]
dur=307.5; rows=[]
t=0.0
while t+WL<=dur+0.01:
    a,b=t,t+WL
    fr=[x for x in ts if a<=x<b]
    faces={}
    for x in fr:
        for d in frames[x]:
            f=faces.setdefault(d['tk'],{'fid':d['fid'],'n':0,'open':[],'mot':[],'box':d['box'],'area':0})
            f['n']+=1
            if d.get('open') is not None: f['open'].append(d['open'])
            if d.get('motion') is not None: f['mot'].append(d['motion'])
            f['area']=max(f['area'],d['box'][2]*d['box'][3])
    fl=[]
    for tk,f in faces.items():
        if f['n']<len(fr)*0.3: continue
        op=np.array(f['open']) if f['open'] else np.zeros(1)
        lip=float(np.std(op)*10+ (np.mean(np.abs(np.diff(op))) *10 if len(op)>1 else 0))
        fl.append({'tk':tk,'fid':f['fid'],'lip':round(lip,3),'open':round(float(op.mean()),3),'mot':round(float(np.mean(f['mot'])) if f['mot'] else 0,3),'area':round(f['area'],4),'vis':round(f['n']/max(1,len(fr)),2)})
    sp=speech[int(a/0.02):int(b/0.02)]
    vad=float(sp.mean()) if len(sp) else 0
    wd=[w[2].strip() for w in words if w[0]<b and w[1]>a]
    rows.append({'t0':round(a,2),'t1':round(b,2),'cut':any(a<c<=b for c in cuts),'vad':round(vad,2),'faces':sorted(fl,key=lambda z:-z['area']),'words':' '.join(wd)})
    t=round(t+HOP,2)
json.dump(rows,open('windows.json','w'),ensure_ascii=False)
print(len(rows), 'cuts',len(cuts), 'words',len(words), 'speech frac', speech.mean())
