# character accuracy: % of dubbed speech seconds where the dub voice (identity of the reference audio the line was
# cloned from) == the verified speaker at that moment. Truth = windows.json 'truth' (visual-verified + voice-embedding).
import json,sys,os,numpy as np,soundfile as sf
R='/workspace/projects/db13b071-b36a-44da-bacb-efac377e3057/build/work/'
rows=json.load(open('windows.json'))
TF=json.load(open('truth_fine.json'))['truth']
def truth(t):
    i=int(round(t/0.1)); o=TF[i] if 0<=i<len(TF) else None
    return None if (not o or o[0]=='?') else o[0]
def ident(spans):
    tot={}
    for a,b in spans:
        for t in np.arange(a,b,0.1):
            p=truth(t)
            if p: tot[p]=tot.get(p,0)+0.1
    s=sum(tot.values())
    if not s: return 'UNKNOWN',tot
    p=max(tot,key=tot.get)
    return (p if tot[p]/s>=0.75 else 'MIXED('+'+'.join(sorted(tot,key=lambda k:-tot[k]))+')'),{k:round(v,2) for k,v in tot.items()}
voc=None
def f0(x, fs=44100):
    st = max(1, fs // 16000); x = x[::st]; fs = fs / st; n = int(0.04 * fs); lo, hi = int(fs / 400), int(fs / 70); out = []
    for i in range(0, len(x) - n, n // 2):
        fr = x[i:i + n] - x[i:i + n].mean()
        if np.sqrt(np.mean(fr ** 2)) < 0.01: continue
        ac = np.correlate(fr, fr, "full")[n - 1:]; ac /= ac[0] + 1e-9; k = lo + int(np.argmax(ac[lo:hi]))
        if ac[k] > 0.5: out.append(fs / k)
    return float(np.median(out)) if len(out) >= 5 else None
def ref_spans(ver, x, L, LBY):
    ref=x.get('ref') or ''
    if ver=='v6':
        if ref=='own_line': return [(L['start'],min(L['end'],L['start']+10))]
        if ref.startswith('pitch_matched:'):
            M=LBY[ref.split(':')[1]]; return [(M['start'],min(M['end'],M['start']+10))]
        if ref=='cluster':
            sp=json.load(open(R+'s2/run5/speakers.json'))[L['spk']]['longest']; out=[];tot=0
            for s in sp:
                out.append((s['start'],min(s['end'],s['start']+7))); tot+=out[-1][1]-out[-1][0]
                if tot>9: break
            return out
    if ref.startswith('char:'):
        c=ref[5:]; chars={p['id']:p for p in json.load(open(CHARS+'/characters.json'))}
        if x['id'] in RETAKE:
            global voc
            if voc is None: voc=sf.read(R+'s1/vocals.flac',dtype='float32',always_2d=True)[0].mean(1)
            fl=f0(voc[int(L['start']*44100):int(L['end']*44100)]); pool=[]
            for u in json.load(open(CHARS+'/units.json')):
                if u['cid']!=c or u['end']-u['start']<1.5 or u.get('how') not in ('face','voice','voice-only'): continue
                fu=f0(voc[int(u['start']*44100):int(u['end']*44100)])
                if fu and fl: pool.append((abs(12*np.log2(fu/fl)),u))
            out=[];tot=0
            for d,u in sorted(pool,key=lambda t:t[0]):
                out.append((u['start'],min(u['end'],u['start']+8))); tot+=out[-1][1]-out[-1][0]
                if tot>=6: break
            return out
        return [(r['start'],r['end']) for r in chars[c]['voice_reference']]
    if ref.startswith('person:'):
        return [tuple(s) for s in json.load(open('v10_refs.json'))[ref[7:]]['spans']]
    return []
ver=sys.argv[1]; qa=json.load(open(sys.argv[2])); LBY={l['id']:l for l in json.load(open(sys.argv[3]))}
CHARS=sys.argv[4] if len(sys.argv)>4 else R+'chars/final'
RETAKE=set(open(sys.argv[5]).read().strip().split(',')) if len(sys.argv)>5 and os.path.exists(sys.argv[5]) else set()
PROV=json.load(open(sys.argv[6])) if len(sys.argv)>6 else {}
res=[]; ok=tot=0
for x in qa['lines']:
    if x.get('variant')=='original' or 'tts' not in x: continue
    L=LBY[x['id']]; sp=[tuple(z) for z in PROV[x['id']]] if x['id'] in PROV else ref_spans(ver,x,L,LBY); vid,vb=ident(sp)
    # dubbed speech seconds: the line span where the source has verified speech
    secs={}
    for t in np.arange(L['start'],L['end'],0.1):
        p=truth(t)
        if p: secs[p]=secs.get(p,0)+0.1
    good=secs.get(vid,0) if not vid.startswith(('MIXED','UNKNOWN')) else 0
    s=sum(secs.values()); ok+=good; tot+=s
    res.append({'ref_spans':[list(map(float,z)) for z in sp],'id':x['id'],'start':L['start'],'end':L['end'],'ref':x.get('ref'),'voice_identity':vid,'voice_ref_breakdown':vb,'truth_secs':{k:round(v,2) for k,v in secs.items()},'correct_s':round(good,2),'speech_s':round(s,2),'en':x.get('en')})
out={'version':ver,'character_accuracy':round(ok/tot,4) if tot else None,'correct_s':round(ok,2),'dubbed_speech_s':round(tot,2),'lines':res}
json.dump(out,open(f'characc_{ver}.json','w'),ensure_ascii=False,indent=1)
print(ver,'character accuracy',out['character_accuracy'],ok,tot)
for r in res:
    if r['correct_s']<r['speech_s']: print(' ',r['id'],r['start'],r['end'],'voice=',r['voice_identity'],'truth=',r['truth_secs'],'|',(r['en'] or '')[:50])
