#!/usr/bin/env python3
"""LocalDub: voice, fit and mix.

usage: dub.py <s1_dir> <jobs_dir> <results.json>[,<results2.json>...] <video> <out.mp4>
  s1_dir   : vocals.flac, background.flac (Demucs), segments.json
  jobs_dir : lines.json, speakers.json (from make_jobs.py)
  results  : Glimmer results (translation tr_* and cast cast_* entries, optional rewrite rw_*)
Steps
  1. per voice cluster: build a clean zero-shot reference (6-12 s of that character's own Demucs vocals + its Hindi text)
  2. per line: Fish Audio TTS (s2.1-pro-free) with the character's reference and an emotion cue
  3. fit: pick en / en_short / rewrite that fits the slot; gentle tempo only within [0.93, 1.08]; trim silence
  4. place at the original start, sidechain-free mix over the Demucs background (music + effects kept), mux to video
  5. qa.json: per-line slot, tts length, tempo, overflow, variant used, Glimmer cast label for the line
v2 (after the v1 listen/QA pass):
  * ASR-garble guard: a line whose transcript is not real Hindi/Urdu (stray CJK/Hangul characters, or Arabic-script text with
    none of the Urdu-specific letters, which is how the CTC model spells short English words, laughs and grunts) keeps the
    original audio instead of being dubbed as filler ("Uh.", "Man.")
  * Glimmer rewrite variant (rw_*) for QA-flagged lines is tried first
  * a last-resort tempo of up to 1.15x only for lines that would otherwise overflow into the next line
Auth for Fish: the sandbox identity gateway injects the key for api.fish.audio (never in code).
"""
import json, os, re, subprocess, sys, time, concurrent.futures as cf
import numpy as np, soundfile as sf, requests, msgpack

S1, JD, RES, VIDEO, OUT = sys.argv[1:6]
WD = os.path.join(os.path.dirname(os.path.abspath(OUT)), "dubwork"); os.makedirs(WD + "/tts", exist_ok=True)
SR = 44100; TMIN, TMAX = 0.93, 1.08
lines = json.load(open(f"{JD}/lines.json")); speakers = json.load(open(f"{JD}/speakers.json"))
R = {}
for p in RES.split(","): R.update(json.load(open(p)))

def parse(c):
    if not c: return None
    c = re.sub(r"^```(json)?|```$", "", c.strip(), flags=re.M).strip()
    try: return json.loads(c)
    except Exception:
        m = re.search(r"\{.*\}", c, re.S)
        try: return json.loads(m.group(0)) if m else None
        except Exception: return None

EN = {}
for k, v in sorted(R.items()):
    if k.startswith(("tr_", "rw_")):
        j = parse(v.get("content")) or {}
        for x in j.get("lines", []):
            if k.startswith("rw_"): EN.setdefault(x["id"], {})["rw"] = x.get("en_fit") or x.get("en")
            else: EN.setdefault(x["id"], {}).update(x)
CAST = {k[5:]: parse(v.get("content")) for k, v in R.items() if k.startswith("cast_")}   # keyed by line id (cast_L002)
URDU = set("ٹڈڑںےہھیکگپچژۓۀ")
def garbled(t):
    if re.search(r"[\u3040-\u30ff\u3400-\u9fff\uac00-\ud7af]", t): return True          # CJK / Hangul = not speech we can dub
    ar = [c for c in t if "\u0600" <= c <= "\u06ff"]
    return len(ar) > 0 and not any(c in URDU for c in t)                    # Arabic letters but no Urdu letters

voc, sr = sf.read(f"{S1}/vocals.flac", dtype="float32", always_2d=True); assert sr == SR
bg, _ = sf.read(f"{S1}/background.flac", dtype="float32", always_2d=True)

def wav_bytes(x):
    import io; b = io.BytesIO(); sf.write(b, x, SR, format="WAV", subtype="PCM_16"); return b.getvalue()

# 1. references
REF = {}
for k, v in speakers.items():
    clips, txt, tot = [], [], 0.0
    for s in v["longest"]:
        a = voc[int(s["start"] * SR):int(min(s["end"], s["start"] + 7.0) * SR)].mean(1)
        clips.append(a); clips.append(np.zeros(int(0.25 * SR), np.float32)); txt.append(s.get("hi", "")); tot += len(a) / SR
        if tot > 9: break
    ref = np.concatenate(clips) if clips else np.zeros(SR, np.float32)
    ref = ref / (np.abs(ref).max() + 1e-6) * 0.8
    REF[k] = {"audio": wav_bytes(ref), "text": " ".join(txt)}

LREF = {}
for L in lines:   # per-line reference: the line's own original vocals carry the right speaker AND the right emotion
    if L["dur"] >= 2.5:
        a = voc[int(L["start"] * SR):int(min(L["end"], L["start"] + 10.0) * SR)].mean(1)
        LREF[L["id"]] = {"audio": wav_bytes(a / (np.abs(a).max() + 1e-6) * 0.8), "text": L["hi"]}

def f0(x, fs=SR):   # median pitch (Hz) of a mono clip, simple autocorrelation; None if too little voiced speech
    st = max(1, fs // 16000); x = x[::st]; fs = fs / st; n = int(0.04 * fs); lo, hi = int(fs / 400), int(fs / 70); out = []
    for i in range(0, len(x) - n, n // 2):
        fr = x[i:i + n] - x[i:i + n].mean()
        if np.sqrt(np.mean(fr ** 2)) < 0.01: continue
        ac = np.correlate(fr, fr, "full")[n - 1:]; ac /= ac[0] + 1e-9; k = lo + int(np.argmax(ac[lo:hi]))
        if ac[k] > 0.5: out.append(fs / k)
    return float(np.median(out)) if len(out) >= 5 else None

# v2 pitch-matched reference for short lines (< 2.5 s): v1 used the whole voice cluster's reference, but a cluster can mix
# speakers, which gave 11 lines a voice > 4 semitones away from the original. Now: the nearest-in-time long line whose
# original pitch is within 3 semitones (same cluster first, then any cluster); cluster reference only as a fallback.
PREF = {}
if os.environ.get("PITCH_REF", "1") == "1":
    F0 = {L["id"]: f0(voc[int(L["start"] * SR):int(L["end"] * SR)].mean(1)) for L in lines}
    longs = [L for L in lines if L["id"] in LREF and F0.get(L["id"])]
    for L in lines:
        if L["id"] in LREF or not F0.get(L["id"]): continue
        ok = [M for M in longs if abs(12 * np.log2(F0[M["id"]] / F0[L["id"]])) <= 3]
        cand = [M for M in ok if M["spk"] == L["spk"]] or ok
        if cand: PREF[L["id"]] = min(cand, key=lambda M: abs(M["start"] - L["start"]))["id"]
    for kv in filter(None, os.environ.get("PREF_OVERRIDE", "").split(",")):   # QA-driven manual reference override "L031:L020"
        a, b = kv.split(":")
        if b == "cluster": PREF.pop(a, None)
        else: PREF[a] = b

def tts(text, spk, emo, path, lid=None):
    if os.path.exists(path): return path
    cue = {"neutral": "", "happy": "[happy] ", "laughing": "[laughing] ", "sarcastic": "[sarcastic] ", "angry": "[angry] ",
           "shouting": "[shouting] ", "sad": "[sad] ", "worried": "[worried] ", "excited": "[excited] ", "whisper": "[whispering] "}.get(emo or "neutral", "")
    body = {"text": cue + text, "references": [LREF.get(lid) or LREF.get(PREF.get(lid)) or REF[spk]], "format": "wav", "normalize": True, "latency": "normal",
            "temperature": 0.7, "top_p": 0.7, "chunk_length": 200}
    for att in range(5):
        r = requests.post("https://api.fish.audio/v1/tts", data=msgpack.packb(body), timeout=180,
                          headers={"Content-Type": "application/msgpack", "model": "s2.1-pro-free"})
        if r.status_code == 200 and len(r.content) > 1000: open(path, "wb").write(r.content); return path
        time.sleep(3 * (att + 1))
    raise RuntimeError(f"tts failed {r.status_code} {r.text[:200]}")

def load_trim(path):
    a, s = sf.read(path, dtype="float32", always_2d=True); a = a.mean(1)
    if s != SR:
        subprocess.run(f"ffmpeg -v error -y -i {path} -ar {SR} -ac 1 {path}.r.wav", shell=True); a, _ = sf.read(f"{path}.r.wav", dtype="float32")
    e = np.abs(a) > 0.01; idx = np.where(e)[0]
    return a[max(0, idx[0] - 400): idx[-1] + 1600] if len(idx) else a

def tempo(a, f):
    if abs(f - 1) < 0.01: return a
    i, o = f"{WD}/t_in.wav", f"{WD}/t_out.wav"; sf.write(i, a, SR)
    subprocess.run(f"ffmpeg -v error -y -i {i} -filter:a atempo={f:.4f} {o}", shell=True); return sf.read(o, dtype="float32")[0]

def latin(t):
    letters = [c for c in t if c.isalpha()]
    return len(letters) > 0 and sum(c.isascii() for c in letters) / len(letters) > 0.8

qa, todo, keep = [], [], []
for L in lines:
    if latin(L["hi"]): keep.append(L); continue      # already English in the original: keep the original voice
    if os.environ.get("GARBLE_GUARD", "1") == "1" and garbled(L["hi"]): keep.append(L); L["kept"] = "asr_garble"; continue
    e = EN.get(L["id"], {})
    variants = [(n, e.get(n)) for n in ("rw", "en", "en_short") if e.get(n)]
    todo.append((L, variants, e.get("emotion", "neutral")))

PHASE = os.environ.get("PHASE", "all"); BUDGET = float(os.environ.get("BUDGET", "100")); T0 = time.time()
if PHASE == "tts":   # pre-generate every variant in parallel within a time budget (re-run until it prints TTS_ALL_DONE)
    work = [(L, n, t, emo) for L, vs, emo in todo for n, t in vs if not os.path.exists(f"{WD}/tts/{L['id']}_{n}.wav")]
    print("pending tts", len(work), flush=True)
    ex = cf.ThreadPoolExecutor(12); futs = [ex.submit(tts, t, L["spk"], emo, f"{WD}/tts/{L['id']}_{n}.wav", L["id"]) for L, n, t, emo in work]
    done, _ = cf.wait(futs, timeout=max(5, BUDGET - (time.time() - T0)))
    print("finished", sum(f.done() and not f.exception() for f in futs), "of", len(futs), flush=True)
    if len(done) == len(futs) and all(not f.exception() for f in futs): print("TTS_ALL_DONE", flush=True)
    os._exit(0)

def render(item):
    L, variants, emo = item; slot = L["dur"] + 0.25
    best = None
    for name, text in variants:
        a = load_trim(tts(text, L["spk"], emo, f"{WD}/tts/{L['id']}_{name}.wav", L["id"])); d = len(a) / SR
        tmax = 1.15 if name == "rw" else TMAX     # a QA-driven Glimmer rewrite may run a little faster before falling back to v1 text
        f = min(max(d / slot, TMIN), tmax) if d > slot or d < slot * 0.6 else 1.0
        fits = d / f <= slot + 0.05
        cand = (name, text, a, d, f, fits)
        if fits: best = cand; break
        if best is None or d < best[3]: best = cand
    if not best[5] and best[3] / slot <= 1.15:      # last resort: slightly faster rather than spilling into the next line
        name, text, a, d, f, fits = best; best = (name, text, a, d, d / slot, True)
    return L, emo, best

dub = np.zeros(len(bg), np.float32)
with cf.ThreadPoolExecutor(6) as ex:
    for L, emo, b in ex.map(render, [t for t in todo if t[1]]):
        name, text, a, d, f, fits = b; a = tempo(a, f) if f != 1.0 else a
        s = int(L["start"] * SR); e = min(len(dub), s + len(a))
        ref = voc[s:int(L["end"] * SR)].mean(1); rms = lambda x: float(np.sqrt(np.mean(x ** 2)) + 1e-7)
        g = float(np.clip(rms(ref) / rms(a), 0.5, 4.0))          # match the original line's loudness
        dub[s:e] += a[:e - s] * g
        qa.append({"id": L["id"], "spk": L["spk"], "slot": L["dur"], "tts": round(d, 2), "tempo": round(f, 3), "fits": fits,
                   "variant": name, "en": text, "hi": L["hi"], "emotion": emo, "gain": round(g, 2),
                   "ref": "own_line" if L["id"] in LREF else (f"pitch_matched:{PREF[L['id']]}" if L["id"] in PREF else "cluster"),
                   "cast": {k: (CAST.get(L["id"]) or {}).get(k) for k in ("speaker", "gender", "age", "role", "face_visible")} if CAST.get(L["id"]) else None})
for L in keep + [t[0] for t in todo if not t[1]]:     # English originals (and any untranslated line) keep original vocals
    s0, s1 = int(L["start"] * SR), int(L["end"] * SR); dub[s0:s1] += voc[s0:s1].mean(1)
    qa.append({"id": L["id"], "spk": L["spk"], "slot": L["dur"], "variant": "original", "fits": True, "hi": L["hi"],
               "kept": L.get("kept", "english_original")})
missing = [t[0]["id"] for t in todo if not t[1]]
qa.sort(key=lambda q: q["id"])
json.dump({"lines": qa, "missing_translation": missing, "overflow": [q["id"] for q in qa if not q["fits"]]},
          open(os.path.join(os.path.dirname(os.path.abspath(OUT)), "qa.json"), "w"), ensure_ascii=False, indent=1)
mix = bg + dub[:, None]; pk = np.abs(mix).max(); mix = mix / pk * 0.97 if pk > 0.97 else mix
sf.write(f"{WD}/mix.wav", mix, SR, subtype="PCM_16"); sf.write(f"{WD}/dialogue_en.wav", dub, SR, subtype="PCM_16")
if OUT.endswith((".webm", ".mkv")):   # no video re-encode: copy the original stream, Opus audio
    subprocess.run(f"ffmpeg -v error -y -i '{VIDEO}' -i {WD}/mix.wav -map 0:v:0 -map 1:a:0 -c:v copy -c:a libopus -b:a 160k -shortest '{OUT}'", shell=True, check=True)
else:
    subprocess.run(f"ffmpeg -v error -y -i '{VIDEO}' -i {WD}/mix.wav -map 0:v:0 -map 1:a:0 -c:v libx264 -preset veryfast -crf 21 "
                   f"-pix_fmt yuv420p -c:a aac -b:a 192k -shortest '{OUT}'", shell=True, check=True)
print("wrote", OUT, "lines", len(qa), "overflow", sum(not q["fits"] for q in qa), "missing", len(missing))
