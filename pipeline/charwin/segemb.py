import json,sys,numpy as np,soundfile as sf,torch
from speechbrain.inference.speaker import EncoderClassifier
R='/workspace/projects/db13b071-b36a-44da-bacb-efac377e3057/build/work/'
m=EncoderClassifier.from_hparams(source=R+'weights/ecapa',savedir='/tmp/ecapa_sv')
v,sr=sf.read(R+'chars/aib_vocals16k.flac'); v=v.mean(1) if v.ndim>1 else v
names=json.load(open('person_names.json')); C=np.load('person_centroids.npy')
for a,b in json.loads(sys.argv[1]):
    x=torch.tensor(v[int(a*sr):int(b*sr)],dtype=torch.float32)[None]
    with torch.no_grad(): e=m.encode_batch(x)[0,0].numpy()
    e/=np.linalg.norm(e); s=e@C.T; o=np.argsort(-s)
    print(a,b,' '.join(f'{names[i]}:{s[i]:.2f}' for i in o[:4]))
