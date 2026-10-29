#!/usr/bin/env python3
"""LocalDub: build Muse Glimmer jobs from stage-1 outputs.

usage: make_jobs.py <s1_dir> <out_dir> [--video proxy.mp4]
reads  s1/segments.json, s1/asr.json, s1/emb.npy
writes out/speakers.json   voice clusters (ECAPA cosine, average-linkage agglomerative)
       out/lines.json      dubbing lines (segments merged into sentences per speaker, with timing budget)
       out/jobs_cast.json  Glimmer VISION jobs: frames at each speaker's lines -> who is on screen (gender/age/look/role)
       out/jobs_tr.json    Glimmer TEXT jobs: whole-script context, humor-preserving English with a full and a short variant
The script prefix is identical across translation jobs so llama.cpp's prompt cache reuses it (big win on CPU).
"""
import json, os, sys
import numpy as np

S1, OUT = sys.argv[1], sys.argv[2]; os.makedirs(OUT, exist_ok=True)
segs = json.load(open(f"{S1}/segments.json"))
asr = json.load(open(f"{S1}/asr.json")); asr = asr.get("omniASR_CTC_1B_v2", next(iter(asr.values())))
E = np.load(f"{S1}/emb.npy") if os.path.exists(f"{S1}/emb.npy") else None

# ---------------------------------------------------------------- speakers: average-linkage clustering on cosine sim
def cluster(E, thr=0.35, w=None):
    C = [[i] for i in range(len(E))]
    S = E @ E.T
    while True:
        best, bi, bj = -1, -1, -1
        for a in range(len(C)):
            for b in range(a + 1, len(C)):
                s = S[np.ix_(C[a], C[b])].mean()
                if s > best: best, bi, bj = s, a, b
        if best < thr or bi < 0: break
        C[bi] += C[bj]; del C[bj]
    dur = lambda c: sum(segs[i]["end"] - segs[i]["start"] for i in c)
    big = [c for c in C if dur(c) >= 4.0] or C
    cen = [E[c].mean(0) / np.linalg.norm(E[c].mean(0)) for c in big]
    for c in C:                       # fold tiny clusters (< 4 s of speech) into the most similar real voice
        if c in big: continue
        k = int(np.argmax([E[c].mean(0) @ z for z in cen])); big[k] = big[k] + c
    lab = np.zeros(len(E), int)
    for k, c in enumerate(sorted(big, key=lambda c: -dur(c))): lab[c] = k
    return lab
lab = cluster(E) if E is not None and len(E) else None
for k, s in enumerate(segs): s["spk"] = f"S{int(lab[k])}" if lab is not None else "S?"; s["hi"] = asr.get(f"{s['i']:04d}", "").strip()
spk = {}
for s in segs: spk.setdefault(s["spk"], []).append(s)
speakers = {k: {"n_lines": len(v), "seconds": round(sum(x["end"] - x["start"] for x in v), 1),
                "longest": sorted(v, key=lambda x: x["start"] - x["end"])[:4]} for k, v in spk.items()}
json.dump(speakers, open(f"{OUT}/speakers.json", "w"), ensure_ascii=False, indent=1)

# ---------------------------------------------------------------- lines: one dubbing line per VAD chunk (stable ids L<seg>)
# (chunks are already split at >=220 ms pauses and capped at 14 s, which matches how dubbing scripts are cut)
lines = [{"id": f"L{s['i']:03d}", "spk": s["spk"], "start": s["start"], "end": s["end"], "hi": s["hi"], "segs": [s["i"]]}
         for s in segs if s["hi"]]
for L in lines:
    d = L["end"] - L["start"]; L["dur"] = round(d, 2)
    L["max_syll"] = max(2, int(round(d * 4.2)))   # natural English ~4-5 syllables/s
json.dump(lines, open(f"{OUT}/lines.json", "w"), ensure_ascii=False, indent=1)
print(len(segs), "segments ->", len(lines), "lines,", len(speakers), "voice clusters")

# ---------------------------------------------------------------- cast jobs (vision): one frame per spoken line
def latin(t):
    l = [c for c in t if c.isalpha()]; return bool(l) and sum(c.isascii() for c in l) / len(l) > 0.8
cast = []
for L in lines:
    if L["dur"] < 1.5 or latin(L["hi"]): continue
    t = round(L["start"] + min(1.2, L["dur"] / 2), 2)
    cast.append({"id": f"cast_{L['id']}", "json": True, "reasoning": "low", "max_tokens": 700, "temperature": 0.5,
                 "messages": [{"role": "user", "content": [{"type": "frame", "t": t}, {"type": "text", "text":
        "Frame from a Hindi comedy sketch at the moment this line is spoken: " + repr(L["hi"][:160]) + ". "
        "Who is most likely speaking? Reply ONLY JSON: {\"speaker\": \"5-8 word visual description\", \"gender\": \"male|female|unknown\", "
        "\"age\": \"young|adult|older\", \"role\": \"one-word role e.g. Patient, Friend, Mother, Doctor, Aunt\", \"face_visible\": true|false}"}]}]})
json.dump(cast, open(f"{OUT}/jobs_cast.json", "w"), ensure_ascii=False, indent=1)

# ---------------------------------------------------------------- translation jobs (text) with shared script prefix
script = "\n".join(f"{L['id']} ({L['dur']}s): {L['hi']}" for L in lines)   # voice ids are not needed for translation
SYS = ("You are LocalDub's dubbing writer. You turn Hindi/Urdu comedy dialogue into natural, funny, spoken American English "
       "for a lip-timed dub. Keep the joke and the attitude, not the literal words. Rewrite Indian cultural references only when "
       "an English viewer would miss the joke. Never add content. Lines come from an automatic transcript that can contain errors: "
       "use the whole script to infer what was really said.")
PRE = ("FULL SCRIPT (line id, slot length, Hindi ASR):\n" + script +
       "\n\nFor the requested lines produce JSON {\"lines\": [{\"id\": ..., \"en\": full natural English line, "
       "\"en_short\": shorter version with the same meaning for tight slots, \"emotion\": one of neutral|happy|laughing|sarcastic|angry|"
       "shouting|sad|worried|excited|whisper}]}. Lines that are already English (e.g. a YouTube outro) stay English. Each line must be speakable within its slot "
       "length at a natural pace (about 4 syllables per second for en, 3 for en_short).")
B = 10
tr = []
for k in range(0, len(lines), B):
    ids = [L["id"] for L in lines[k:k + B]]
    tr.append({"id": f"tr_{k // B:02d}", "json": True, "reasoning": "low", "max_tokens": 3500, "temperature": 0.8,
               "messages": [{"role": "system", "content": SYS},
                            {"role": "user", "content": PRE + f"\n\nRequested lines: {', '.join(ids)}"}]})
json.dump(tr, open(f"{OUT}/jobs_tr.json", "w"), ensure_ascii=False, indent=1)
print(len(cast), "cast jobs,", len(tr), "translation jobs; script chars", len(script))
