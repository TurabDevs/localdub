import json,numpy as np
P=json.load(open('persons.json')); rows=json.load(open('windows.json')); E=np.load('win_emb.npy')
def inside(r,segs): return any(r['t0']>=a-0.01 and r['t1']<=b+0.01 for a,b in segs)
S={p:[i for i,r in enumerate(rows) if r['vad']>=0.6 and inside(r,v['segs'])] for p,v in P.items()}
C={p:E[ix].mean(0)/np.linalg.norm(E[ix].mean(0)) for p,ix in S.items() if ix}
print({p:len(ix) for p,ix in S.items()})
# leave-one-out
names=list(C); ok=tot=0; conf={}
for p,ix in S.items():
    for i in ix:
        cs={}
        for q in names:
            ids=[j for j in S[q] if j!=i]; c=E[ids].mean(0); cs[q]=float(E[i]@c/np.linalg.norm(c))
        best=max(cs,key=cs.get); tot+=1; ok+=best==p; conf.setdefault(p,{}).setdefault(best,0); conf[p][best]+=1
print('LOO window acc',ok,tot,round(ok/tot,3)); 
for p in conf: print(p,conf[p])
np.save('person_centroids.npy',np.stack([C[n] for n in names])); json.dump(names,open('person_names.json','w'))
