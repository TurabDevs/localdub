import json,numpy as np
P=json.load(open('persons.json')); rows=json.load(open('windows.json')); E=np.load('win_emb.npy')
names=json.load(open('person_names.json')); C=np.load('person_centroids.npy')
man=json.load(open('manual_fixes.json'))['window_overrides'] if __import__('os').path.exists('manual_fixes.json') else []
def seg_person(tc):
    for p,v in P.items():
        for a,b in v['segs']:
            if a<=tc<=b: return p
for i,r in enumerate(rows):
    tc=r['t0']+0.75; s=E[i]@C.T; o=np.argsort(-s)
    r['emb_best']=names[o[0]]; r['emb_sim']=round(float(s[o[0]]),3); r['emb_margin']=round(float(s[o[0]]-s[o[1]]),3); r['emb_2nd']=names[o[1]]
    vp=seg_person(tc)
    r['truth']=None; r['truth_how']=''
    if r['vad']<0.5: continue
    if vp: r['truth'],r['truth_how']=vp,'visual (verified on-screen speaker)'
    else: r['truth'],r['truth_how']=names[o[0]],'voice embedding (nearest verified speaker)'
    for m in man:
        if m['from']<=tc<=m['to']: r['truth'],r['truth_how']=m['person'],'manual: '+m['why']
json.dump(rows,open('windows.json','w'),ensure_ascii=False)
import csv
with open('timeline.csv','w',newline='') as f:
    w=csv.writer(f); w.writerow(['t0','t1','cut','vad','faces(fid:lip)','voice_cluster','emb_best','emb_sim','emb_margin','truth','truth_how','words'])
    for r in rows: w.writerow([r['t0'],r['t1'],int(r['cut']),r['vad'],' '.join(f"{x['fid']}:{x['lip']}" for x in r['faces']),r.get('vc',''),r['emb_best'],r['emb_sim'],r['emb_margin'],r['truth'] or '',r['truth_how'],r['words']])
print('ok')
