# "heard voice" check: ECAPA embedding of the actual TTS take used for each dubbed line vs the verified people's
# centroids (built from their original Hindi speech). Nearest person == majority truth speaker -> match. Cross-language,
# so it is a weak secondary signal; the main metric is characc.py.
import json,sys,os,numpy as np,soundfile as sf,torch
from speechbrain.inference.speaker import EncoderClassifier
R='/workspace/projects/db13b071-b36a-44da-bacb-efac377e3057/build/work/'
torch.set_num_threads(1)
m=EncoderClassifier.from_hparams(source=R+'weights/ecapa',savedir='/tmp/ecapa_sv')
names=json.load(open('person_names.json')); C=np.load('person_centroids.npy')
acc=json.load(open(sys.argv[1])); tdir=sys.argv[2]; q={x['id']:x for x in json.load(open(sys.argv[3]))['lines']}
out=[];ok=tot=0
for r in acc['lines']:
    x=q[r['id']]; f=os.path.join(tdir,f"{r['id']}_{x['variant']}.wav")
    if not os.path.exists(f) or not r['truth_secs']: continue
    a,sr=sf.read(f,dtype='float32',always_2d=True); a=a.mean(1)
    if sr!=16000:
        import torchaudio.functional as F
        a=F.resample(torch.tensor(a),sr,16000).numpy()
    with torch.no_grad(): e=m.encode_batch(torch.tensor(a)[None])[0,0].numpy()
    e/=np.linalg.norm(e); s=e@C.T; o=np.argsort(-s)
    tr=max(r['truth_secs'],key=r['truth_secs'].get); w=r['speech_s']
    hit=names[o[0]]==tr; ok+=w*hit; tot+=w
    out.append({'id':r['id'],'truth':tr,'heard_nearest':names[o[0]],'sim':round(float(s[o[0]]),3),'sim_to_truth':round(float(s[names.index(tr)]),3),'match':bool(hit)})
res={'heard_voice_match_pct_of_speech_s':round(ok/tot,4),'lines':out}
json.dump(res,open(sys.argv[4],'w'),indent=1); print('heard match',res['heard_voice_match_pct_of_speech_s'])
for z in out:
    if not z['match']: print(' ',z)
