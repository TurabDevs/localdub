import json, numpy as np, soundfile as sf, torch, sys, os
from speechbrain.inference.speaker import EncoderClassifier
torch.set_num_threads(2)
R='/workspace/projects/db13b071-b36a-44da-bacb-efac377e3057/build/work/'
m=EncoderClassifier.from_hparams(source=R+'weights/ecapa',savedir='/tmp/ecapa_sv')
v,sr=sf.read(R+'chars/aib_vocals16k.flac'); v=v.mean(1) if v.ndim>1 else v
assert sr==16000
rows=json.load(open('windows.json'))
out='win_emb.npy'
E=np.load(out) if os.path.exists(out) else np.zeros((len(rows),192),np.float32)
done=np.abs(E).sum(1)>0
import time; t0=time.time()
for i,r in enumerate(rows):
    if done[i]: continue
    x=torch.tensor(v[int(r['t0']*sr):int(r['t1']*sr)],dtype=torch.float32)[None]
    with torch.no_grad(): e=m.encode_batch(x)[0,0].numpy()
    E[i]=e/ (np.linalg.norm(e)+1e-9)
    if time.time()-t0>95: break
np.save(out,E); print('done',int((np.abs(E).sum(1)>0).sum()),len(rows))
