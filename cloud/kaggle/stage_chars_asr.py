"""LocalDub character stage, part 1 (Kaggle CPU, offline): Whisper large-v3 (faster-whisper / CTranslate2 int8) word timestamps
on the Demucs vocals + one SpeechBrain ECAPA speaker embedding per Whisper segment (segments are re-split at word gaps > 0.6 s).
Output: whisper.json (segments + words), wemb.npy (ECAPA per sub-segment), wsub.json (sub-segment spans)."""
import glob, json, os, subprocess, time
T0 = time.time()
def log(*a): print(f"[{time.time()-T0:7.1f}s]", *a, flush=True)
def sh(c): log("+", c[:200]); return subprocess.run(c, shell=True).returncode
def find(p): return sorted(glob.glob(f"/kaggle/input/**/{p}", recursive=True))
for w in find("ctranslate2-*.whl") + find("faster_whisper-*.whl") + find("av-*.whl"): sh(f"pip install -q --no-index --no-deps {w}")
MD = os.path.dirname(find("model.bin")[0]); log("whisper model dir", MD, os.listdir(MD))
AUD = find("aib_vocals16k.flac")[0]
from faster_whisper import WhisperModel
m = WhisperModel(MD, device="cpu", compute_type="int8", cpu_threads=os.cpu_count() or 4)
import soundfile as sf
W16, _sr = sf.read(AUD, dtype="float32"); assert _sr == 16000
segs, info = m.transcribe(W16, language=None, task="transcribe", beam_size=5, word_timestamps=True, vad_filter=False,
                          condition_on_previous_text=False)
out = []
for s in segs:
    out.append({"start": s.start, "end": s.end, "text": s.text, "avg_logprob": s.avg_logprob, "no_speech_prob": s.no_speech_prob,
                "words": [{"w": w.word, "s": w.start, "e": w.end, "p": w.probability} for w in (s.words or [])]})
    log(f"{s.start:7.2f}-{s.end:7.2f} {s.text[:80]}")
json.dump({"language": info.language, "language_probability": info.language_probability, "model": "whisper-large-v3 (faster-whisper int8)",
           "segments": out}, open("/kaggle/working/whisper.json", "w"), ensure_ascii=False, indent=1)
log("whisper done", len(out))
# sub-segments: split at word gaps > 0.6 s
subs = []
for si, s in enumerate(out):
    ws = s["words"]
    if not ws: continue
    cur = [ws[0]]
    for w in ws[1:]:
        if w["s"] - cur[-1]["e"] > 0.6: subs.append((si, cur)); cur = [w]
        else: cur.append(w)
    subs.append((si, cur))
wsub = [{"seg": si, "start": c[0]["s"], "end": c[-1]["e"], "text": "".join(x["w"] for x in c)} for si, c in subs]
json.dump(wsub, open("/kaggle/working/wsub.json", "w"), ensure_ascii=False, indent=1)
try:
    TAR = (find("assets.tar") or [None])[0]
    if TAR: sh(f"mkdir -p /tmp/a && tar xf {TAR} -C /tmp/a")
    else:
        root = os.path.dirname(os.path.dirname(find("wheels/*.whl")[0])); sh(f"mkdir -p /tmp/a && cp -r {root}/wheels {root}/weights /tmp/a/")
    sh("pip install -q --no-index --no-deps /tmp/a/wheels/speechbrain*.whl /tmp/a/wheels/hyperpyyaml*.whl /tmp/a/wheels/ruamel*.whl")
    import numpy as np, soundfile as sf, torch, torchaudio
    if not hasattr(torchaudio, "list_audio_backends"): torchaudio.list_audio_backends = lambda: ["soundfile"]
    from speechbrain.lobes.features import Fbank
    from speechbrain.lobes.models.ECAPA_TDNN import ECAPA_TDNN
    fb = Fbank(n_mels=80)
    net = ECAPA_TDNN(input_size=80, channels=[1024, 1024, 1024, 1024, 3072], kernel_sizes=[5, 3, 3, 3, 1], dilations=[1, 2, 3, 4, 1],
                     attention_channels=128, lin_neurons=192)
    net.load_state_dict(torch.load("/tmp/a/weights/ecapa/embedding_model.ckpt", map_location="cpu")); net.eval()
    w, sr = sf.read(AUD, dtype="float32")
    E = []
    for x in wsub:
        c = w[int(x["start"] * sr):int(max(x["end"], x["start"] + 0.3) * sr)]
        c = np.pad(c, (0, max(0, 8000 - len(c))))
        with torch.no_grad():
            f = fb(torch.from_numpy(c)[None]); f = f - f.mean(dim=1, keepdim=True); e = net(f).squeeze().numpy()
        E.append(e / (np.linalg.norm(e) + 1e-9))
    np.save("/kaggle/working/wemb.npy", np.stack(E)); log("ecapa done", len(E))
except Exception as ex:
    log("ECAPA failed", repr(ex))
log("ALL DONE")
