"""LocalDub QA "listen" pass - re-transcribes the DUBBED English dialogue with Meta Omnilingual ASR (CTC 1B v2), offline in a
Kaggle kernel, so every dubbed line can be checked against the English text it was supposed to say (word error rate,
truncation, mumbled or missing words). This is the automated "listen" step of the dub -> listen -> fix -> re-dub loop.

Inputs (attached datasets): bugxdf/localdub-assets (Omnilingual weights + tokenizer), a listen dataset with
  dialogue_*.flac (dubbed dialogue stem, 16 kHz mono) and lines_*.json ([{id, start, end, en}]).
Output: /kaggle/working/listen_<tag>.json  {id: {"heard": str, "en": str}}
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
import numpy as np, soundfile as sf, torch
from transformers import AutoProcessor, Wav2Vec2ForCTC
torch.set_num_threads(os.cpu_count() or 4)
d = "/tmp/a/weights/omniasr-ctc-1b-v2"
if not os.path.exists(f"{d}/model.safetensors"): os.symlink(ASRW, f"{d}/model.safetensors")
proc = AutoProcessor.from_pretrained(d); mdl = Wav2Vec2ForCTC.from_pretrained(d).eval(); log("model loaded")
for lj in find("lines_*.json"):
    tag = os.path.basename(lj)[6:-5]; fl = [f for f in find(f"dialogue_{tag}.flac")][0]
    w, sr = sf.read(fl, dtype="float32"); assert sr == 16000, sr
    out = {}
    for L in json.load(open(lj)):
        c = w[int(L["start"] * sr):int(L["end"] * sr)]
        if len(c) < 1600: out[L["id"]] = {"heard": "", "en": L["en"]}; continue
        c = c / (np.abs(c).max() + 1e-6) * 0.9
        inp = proc(c, sampling_rate=16000, return_tensors="pt")
        with torch.no_grad(): lg = mdl(inp.input_values).logits
        out[L["id"]] = {"heard": proc.batch_decode(lg.argmax(-1))[0], "en": L["en"]}
        log(tag, L["id"], out[L["id"]]["heard"][:120])
    json.dump(out, open(f"{W}/listen_{tag}.json", "w"), ensure_ascii=False, indent=1)
log("done")
