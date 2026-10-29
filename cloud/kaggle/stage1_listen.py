"""LocalDub stage 1 ("listen") - runs in a Kaggle kernel fully OFFLINE (CPU 4 vCPU/30 GB is enough; uses CUDA if present).

Inputs (attached Kaggle datasets):
  bugxdf/localdub-demo    src.flac (original mix 44.1 kHz stereo), proxy.mp4
  bugxdf/localdub-assets  assets.tar (pinned wheels + Demucs htdemucs_ft weights + ECAPA weights + Omnilingual tokenizer/config)
                          model.safetensors (Meta Omnilingual ASR omniASR_CTC_1B_v2, HF Wav2Vec2ForCTC conversion, parity-verified)
Outputs (/kaggle/working):
  vocals.flac, background.flac   Meta Demucs htdemucs_ft two-stem separation
  segments.json                  speech chunks from Silero VAD on the vocals stem
  asr.json                       Hindi transcript per chunk (Meta Omnilingual ASR CTC 1B v2)
  emb.npy                        ECAPA speaker embedding per chunk
Every model here is open weights and runs inside this kernel. Nothing calls a hosted API.
"""
import glob, json, os, subprocess, sys, time

W = "/kaggle/working"; T0 = time.time(); LOG = open(f"{W}/stage1.log", "a")
def log(*a):
    m = f"[{time.time() - T0:7.1f}s] " + " ".join(str(x) for x in a); print(m, flush=True); LOG.write(m + "\n"); LOG.flush()
def sh(cmd, check=True):
    log("+", cmd[:300]); r = subprocess.run(cmd, shell=True)
    if check and r.returncode: log("FAILED", r.returncode, cmd[:200]); raise SystemExit(r.returncode)
    return r.returncode
def find(p): return sorted(glob.glob(f"/kaggle/input/**/{p}", recursive=True))

SRC = find("src.flac")[0]; TAR = (find("assets.tar") or [None])[0]; ASRW = find("model.safetensors")[0]
log("src", SRC, "assets", TAR, "asr", ASRW)
sh("nproc; free -g; python -V; python -c 'import torch,torchaudio,transformers;print(torch.__version__,torchaudio.__version__,transformers.__version__,torch.cuda.is_available())'", check=False)
if TAR and os.path.isfile(TAR): sh(f"mkdir -p /tmp/a && tar xf {TAR} -C /tmp/a")
else:  # Kaggle auto-extracted the tar: copy the extracted tree (wheels/, weights/) to /tmp/a
    root = os.path.dirname(os.path.dirname(find("wheels/*.whl")[0])); sh(f"mkdir -p /tmp/a && cp -r {root}/wheels {root}/weights /tmp/a/")
sh("pip install -q --no-index --no-deps /tmp/a/wheels/*.whl", check=False)
for fx in find("ruamel.yaml-0.18*.whl"):  # hyperpyyaml (speechbrain) breaks with ruamel.yaml 0.19
    sh(f"pip install -q --no-index --no-deps --force-reinstall {fx}", check=False)
A = "/tmp/a/weights"
# resume: reuse outputs of an earlier run attached as a kernel/dataset source (skips Demucs / ASR)
for f in ("vocals.flac", "background.flac", "asr.json"):
    prev = [p for p in find(f) if "localdub-demo" not in p]
    if prev and not os.path.exists(f"{W}/{f}"): sh(f"cp {prev[0]} {W}/{f}"); log("reused", prev[0])
import numpy as np, soundfile as sf, torch
DEV = "cuda" if torch.cuda.is_available() else "cpu"; torch.set_num_threads(os.cpu_count() or 4)

# ------------------------------------------------------------ 1. Meta Demucs htdemucs_ft (bag of 4 fine-tuned models)
if not os.path.exists(f"{W}/vocals.flac"):
    import demucs, shutil
    from demucs.pretrained import get_model
    from demucs.apply import apply_model
    shutil.copy(os.path.join(os.path.dirname(demucs.__file__), "remote", "htdemucs_ft.yaml"), f"{A}/demucs/htdemucs_ft.yaml")
    model = get_model("htdemucs_ft", repo=__import__("pathlib").Path(f"{A}/demucs")); model.eval()
    wav, sr = sf.read(SRC, dtype="float32", always_2d=True); assert sr == model.samplerate, sr
    x = torch.from_numpy(wav.T.copy()); ref = x.mean(0); m, s = ref.mean(), ref.std()
    with torch.no_grad():
        out = apply_model(model, ((x - m) / s)[None], device=DEV, shifts=1, split=True, overlap=0.25, progress=False, num_workers=0)[0]
    out = out * s + m
    names = model.sources; v = out[names.index("vocals")]; bg = out.sum(0) - v
    sf.write(f"{W}/vocals.flac", v.T.numpy(), sr, subtype="PCM_24"); sf.write(f"{W}/background.flac", bg.T.numpy(), sr, subtype="PCM_24")
    del model, out
log("separation done")

# ------------------------------------------------------------ 2. Silero VAD -> dubbing chunks (<= 14 s)
sh(f"ffmpeg -v error -y -i {W}/vocals.flac -ac 1 -ar 16000 /tmp/voc16.wav")
from silero_vad import load_silero_vad, get_speech_timestamps
w16, _ = sf.read("/tmp/voc16.wav", dtype="float32")
ts = get_speech_timestamps(torch.from_numpy(w16), load_silero_vad(), sampling_rate=16000, threshold=0.45,
                           min_speech_duration_ms=180, min_silence_duration_ms=220, speech_pad_ms=60)
segs = []
for t in ts:
    s_, e_ = t["start"] / 16000, t["end"] / 16000
    if segs and s_ - segs[-1][1] < 0.12: segs[-1][1] = e_
    else: segs.append([s_, e_])
def split(s_, e_):
    if e_ - s_ <= 14: return [[s_, e_]]
    a, b = int((s_ + 4) * 16000), int((e_ - 4) * 16000); k = 1600
    en = np.convolve(np.abs(w16[a:b]), np.ones(k) / k, "valid"); c = (a + int(en.argmin()) + k // 2) / 16000
    return split(s_, c) + split(c, e_)
fin = [p for s_, e_ in segs for p in split(s_, e_)]
SEG = [{"i": i, "start": round(s_, 3), "end": round(e_, 3)} for i, (s_, e_) in enumerate(fin)]
json.dump(SEG, open(f"{W}/segments.json", "w"), indent=0)
log("segments", len(SEG), "speech s", round(sum(e_ - s_ for s_, e_ in fin), 1))
chunks = [w16[int(s["start"] * 16000):int(s["end"] * 16000)] for s in SEG]

# ------------------------------------------------------------ 3. Meta Omnilingual ASR CTC 1B v2 (transformers Wav2Vec2ForCTC)
if not os.path.exists(f"{W}/asr.json"):
    try:
        from transformers import AutoProcessor, Wav2Vec2ForCTC
        d = f"{A}/omniasr-ctc-1b-v2"; os.symlink(ASRW, f"{d}/model.safetensors") if not os.path.exists(f"{d}/model.safetensors") else None
        proc = AutoProcessor.from_pretrained(d); mdl = Wav2Vec2ForCTC.from_pretrained(d).to(DEV).eval()
        res = {}
        for s, c in zip(SEG, chunks):
            inp = proc(c, sampling_rate=16000, return_tensors="pt")
            with torch.no_grad(): lg = mdl(inp.input_values.to(DEV)).logits
            res[f"{s['i']:04d}"] = proc.batch_decode(lg.argmax(-1))[0]
        json.dump({"omniASR_CTC_1B_v2": res}, open(f"{W}/asr.json", "w"), ensure_ascii=False, indent=0)
        log("asr ok", len(res), "sample:", list(res.values())[:3])
        del mdl
    except Exception as e:
        import traceback; traceback.print_exc(); log("ASR FAILED", repr(e)[:300])

# ------------------------------------------------------------ 4. ECAPA speaker embeddings per chunk
try:
    import torchaudio
    if not hasattr(torchaudio, "list_audio_backends"): torchaudio.list_audio_backends = lambda: ["soundfile"]  # torchaudio>=2.9 removed it
    # build ECAPA directly from its checkpoint (speechbrain's hub loader breaks with current huggingface_hub)
    from speechbrain.lobes.features import Fbank
    from speechbrain.lobes.models.ECAPA_TDNN import ECAPA_TDNN
    fb = Fbank(n_mels=80)
    net = ECAPA_TDNN(input_size=80, channels=[1024, 1024, 1024, 1024, 3072], kernel_sizes=[5, 3, 3, 3, 1], dilations=[1, 2, 3, 4, 1],
                     attention_channels=128, lin_neurons=192)
    net.load_state_dict(torch.load(f"{A}/ecapa/embedding_model.ckpt", map_location="cpu")); net.eval().to(DEV)
    class _Enc:
        def encode_batch(self, w):
            f = fb(w); f = f - f.mean(dim=1, keepdim=True); return net(f)
    enc = _Enc()
    E = []
    for c in chunks:
        c = np.pad(c, (0, max(0, 8000 - len(c))))
        with torch.no_grad(): e = enc.encode_batch(torch.from_numpy(c)[None].to(DEV)).squeeze().cpu().numpy()
        E.append(e / (np.linalg.norm(e) + 1e-9))
    np.save(f"{W}/emb.npy", np.stack(E)); log("emb", len(E))
except Exception as e:
    import traceback; traceback.print_exc(); log("EMB FAILED", repr(e)[:300])
log("stage1 complete")
