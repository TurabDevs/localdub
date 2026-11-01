"""LocalDub stage 1b ("who speaks when") - speaker diarization inside the VAD chunks, offline in a Kaggle kernel.

Why: a VAD chunk often holds two people (a phone call, a quick back-and-forth). One chunk = one dubbing line = one cloned
voice, so both people came out in the same voice. Until v5 I split those by hand from the pitch track.

How:
  1. ECAPA speaker embeddings on sliding 1.5 s windows (hop 0.25 s) over every chunk of the Demucs vocals
  2. average-linkage clustering of all windows (cosine), tiny clusters folded into their nearest neighbour
  3. per chunk, smoothed window labels -> speaker change points; a chunk is cut at the quietest point near each change
     when both sides last >= 0.8 s
  4. Omnilingual ASR + one ECAPA embedding per new chunk, written in the same format stage 1 uses, so make_jobs.py
     works unchanged: segments.json, asr.json, emb.npy (+ diar.json with the window labels and the cuts)
Inputs: stage-1 outputs (vocals.flac, segments.json) attached as a kernel/dataset source, bugxdf/localdub-assets.
"""
import glob, json, os, subprocess, time
W = "/kaggle/working"; T0 = time.time()
def log(*a): print(f"[{time.time()-T0:7.1f}s]", *a, flush=True)
def sh(c): log("+", c[:200]); return subprocess.run(c, shell=True).returncode
def find(p): return sorted(glob.glob(f"/kaggle/input/**/{p}", recursive=True))
TAR = (find("assets.tar") or [None])[0]; ASRW = find("model.safetensors")[0]
if TAR and os.path.isfile(TAR): sh(f"mkdir -p /tmp/a && tar xf {TAR} -C /tmp/a")
else:
    root = os.path.dirname(os.path.dirname(find("wheels/*.whl")[0])); sh(f"mkdir -p /tmp/a && cp -r {root}/wheels {root}/weights /tmp/a/")
sh("pip install -q --no-index --no-deps /tmp/a/wheels/*.whl")
for fx in find("ruamel.yaml-0.18*.whl"): sh(f"pip install -q --no-index --no-deps --force-reinstall {fx}")
A = "/tmp/a/weights"
import numpy as np, soundfile as sf, torch
torch.set_num_threads(os.cpu_count() or 4)
VOC = find("vocals.flac")[0]; SEG0 = json.load(open(find("segments.json")[0]))
sh(f"ffmpeg -v error -y -i {VOC} -ac 1 -ar 16000 /tmp/v16.wav"); w, sr = sf.read("/tmp/v16.wav", dtype="float32")

import torchaudio
if not hasattr(torchaudio, "list_audio_backends"): torchaudio.list_audio_backends = lambda: ["soundfile"]
from speechbrain.lobes.features import Fbank
from speechbrain.lobes.models.ECAPA_TDNN import ECAPA_TDNN
fb = Fbank(n_mels=80)
net = ECAPA_TDNN(input_size=80, channels=[1024, 1024, 1024, 1024, 3072], kernel_sizes=[5, 3, 3, 3, 1], dilations=[1, 2, 3, 4, 1],
                 attention_channels=128, lin_neurons=192)
net.load_state_dict(torch.load(f"{A}/ecapa/embedding_model.ckpt", map_location="cpu")); net.eval()
def emb(c):
    c = np.pad(c, (0, max(0, 8000 - len(c))))
    with torch.no_grad():
        f = fb(torch.from_numpy(c)[None]); f = f - f.mean(dim=1, keepdim=True); e = net(f).squeeze().numpy()
    return e / (np.linalg.norm(e) + 1e-9)

WIN, HOP = 1.5, 0.25
wins = []   # (seg index, t_center, emb)
for s in SEG0:
    a, b = s["start"], s["end"]
    if b - a <= WIN: wins.append((s["i"], (a + b) / 2, emb(w[int(a * sr):int(b * sr)]))); continue
    t = a
    while t + WIN <= b + 1e-6:
        wins.append((s["i"], t + WIN / 2, emb(w[int(t * sr):int((t + WIN) * sr)]))); t += HOP
log("windows", len(wins))
E = np.stack([x[2] for x in wins])

def cluster(E, thr):   # average linkage on cosine distance (scipy is preinstalled on Kaggle)
    from scipy.cluster.hierarchy import linkage, fcluster
    if len(E) < 2: return np.zeros(len(E), int)
    return fcluster(linkage(E, method="average", metric="cosine"), t=1 - thr, criterion="distance") - 1
THR = float(os.environ.get("DIAR_THR", "0.45"))
lab = cluster(E, THR)
# fold clusters with < 1.5 s of windows into the most similar bigger cluster
cnt = np.bincount(lab); cent = {k: E[lab == k].mean(0) for k in set(lab)}
big = [k for k in cent if cnt[k] * HOP >= 1.5] or list(cent)
for k in list(cent):
    if k not in big:
        tgt = max(big, key=lambda b: float(cent[k] @ cent[b])); lab[lab == k] = tgt
log("clusters", {int(k): int(v) for k, v in zip(*np.unique(lab, return_counts=True))})

segs, cuts = [], []
for s in SEG0:
    idx = [n for n, x in enumerate(wins) if x[0] == s["i"]]
    L = lab[idx]; T = [wins[n][1] for n in idx]
    if len(L) >= 5:   # majority smoothing over 5 windows
        L = np.array([np.bincount(L[max(0, j - 2): j + 3]).argmax() for j in range(len(L))])
    pieces = [[s["start"], None, int(L[0])]]
    for j in range(1, len(L)):
        if L[j] != pieces[-1][2]:
            c = T[j] - HOP / 2                                     # change between window centres
            a0, b0 = int((c - 0.3) * sr), int((c + 0.3) * sr)       # quietest 20 ms inside +-0.3 s
            k = 320; en = np.convolve(np.abs(w[a0:b0]), np.ones(k) / k, "valid")
            cut = (a0 + int(en.argmin()) + k // 2) / sr if len(en) else c
            if cut - pieces[-1][0] >= 0.8 and s["end"] - cut >= 0.8:
                pieces[-1][1] = cut; pieces.append([cut, None, int(L[j])]); cuts.append({"seg": s["i"], "t": round(cut, 3)})
    pieces[-1][1] = s["end"]
    for a, b, k in pieces: segs.append({"start": round(a, 3), "end": round(b, 3), "orig": s["i"], "diar_spk": k})
for i, s in enumerate(segs): s["i"] = i
log("chunks", len(SEG0), "->", len(segs), "cuts", cuts)

from transformers import AutoProcessor, Wav2Vec2ForCTC
d = f"{A}/omniasr-ctc-1b-v2"
if not os.path.exists(f"{d}/model.safetensors"): os.symlink(ASRW, f"{d}/model.safetensors")
proc = AutoProcessor.from_pretrained(d); mdl = Wav2Vec2ForCTC.from_pretrained(d).eval()
asr = {}; embs = []
for s in segs:
    c = w[int(s["start"] * sr):int(s["end"] * sr)]
    inp = proc(c, sampling_rate=16000, return_tensors="pt")
    with torch.no_grad(): lg = mdl(inp.input_values).logits
    asr[f"{s['i']:04d}"] = proc.batch_decode(lg.argmax(-1))[0]; embs.append(emb(c))
json.dump(segs, open(f"{W}/segments.json", "w"), indent=0)
json.dump({"omniASR_CTC_1B_v2": asr}, open(f"{W}/asr.json", "w"), ensure_ascii=False, indent=0)
np.save(f"{W}/emb.npy", np.stack(embs))
json.dump({"thr": THR, "win": WIN, "hop": HOP, "cuts": cuts, "windows": [[int(x[0]), round(float(x[1]), 3), int(l)] for x, l in zip(wins, lab)]},
          open(f"{W}/diar.json", "w"))
log("diarize complete", list(asr.values())[:4])
