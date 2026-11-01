#!/usr/bin/env python3
"""Compare two dubs turn by turn: does the dub voice follow the person who is actually speaking?

usage: char_voice_check.py <lines.json with speaker turns> <charmap.json> <s1/vocals.flac> <A_dialogue.wav> <B_dialogue.wav> <out.json> [A_name B_name]

This check does not use the face detector or the character map that chose the voices, so it is independent of the
character stage it is grading. For every speaker turn (lines split at speaker changes, e.g. run7/lines.json) it measures
the median pitch of the original turn (Demucs vocals) and of each dub's dialogue stem in the same window, then reports:
  * turn pitch gap: |12*log2(dub/original)| in semitones, on the turns where both dubs speak (lower = the voice follows
    the speaker; a turn voiced by the wrong person shows up as a large gap when the two people's voices differ)
  * turns on a speaker change: only turns from lines that hold 2+ characters, which is where a one-voice-per-line dub fails
  * per-character spread: standard deviation of the dub's median pitch across that character's turns (one consistent
    voice per character = small spread)
With QA_A / QA_B (env, the two qa.json files) only turns that both versions actually dubbed are compared, so lines that
keep the original audio in either version cannot flatter the score. Pitch is a coarse proxy for identity (two people with similar pitch look alike here), so treat it as one signal, not proof.
"""
import json, sys, collections
import numpy as np, soundfile as sf
lines_f, cmap_f, voc_f, a_f, b_f, out_f = sys.argv[1:7]
NA, NB = (sys.argv[7], sys.argv[8]) if len(sys.argv) > 8 else ("A", "B")
LINES = json.load(open(lines_f)); CM = json.load(open(cmap_f))
def mono(p):
    x, sr = sf.read(p, dtype="float32", always_2d=True); return x.mean(1), sr
voc, sr = mono(voc_f); A, sa = mono(a_f); B, sb = mono(b_f); assert sr == sa == sb
def f0(x):
    x = x[::max(1, sr // 16000)]; fs = sr / max(1, sr // 16000); n = int(0.04 * fs); h = n // 2; out = []
    for i in range(0, len(x) - n, h):
        fr = x[i:i + n] - x[i:i + n].mean()
        if np.sqrt(np.mean(fr ** 2)) < 0.01: continue
        ac = np.correlate(fr, fr, "full")[n - 1:]; ac /= ac[0] + 1e-9
        lo, hi = int(fs / 400), int(fs / 70); k = lo + int(np.argmax(ac[lo:hi]))
        if ac[k] > 0.5: out.append(fs / k)
    return float(np.median(out)) if len(out) >= 5 else None
import os
def dubbed(qf):
    if not qf: return None
    return {q["id"] for q in json.load(open(qf))["lines"] if q["variant"] != "original"}
DA, DB = dubbed(os.environ.get("QA_A")), dubbed(os.environ.get("QA_B"))
def isdub(D, L): return D is None or L["id"] in D or L.get("parent") in D
LINES = [L for L in LINES if isdub(DA, L) and isdub(DB, L)]
multi = collections.Counter(L.get("parent") for L in LINES if L.get("parent"))
rows = []
for L in LINES:
    s, e = int(L["start"] * sr), int(L["end"] * sr)
    fo, fa, fb = f0(voc[s:e]), f0(A[s:e]), f0(B[s:e])
    rows.append({"id": L["id"], "char": CM.get(L["id"]), "turn_of_multi": bool(L.get("parent")),
                 "orig_hz": fo, NA + "_hz": fa, NB + "_hz": fb,
                 NA + "_gap_st": round(abs(12 * np.log2(fa / fo)), 2) if fo and fa else None,
                 NB + "_gap_st": round(abs(12 * np.log2(fb / fo)), 2) if fo and fb else None})
def summ(sel):
    both = [r for r in sel if r[NA + "_gap_st"] is not None and r[NB + "_gap_st"] is not None]
    ga, gb = [r[NA + "_gap_st"] for r in both], [r[NB + "_gap_st"] for r in both]
    if not both: return {"turns": 0}
    return {"turns": len(both), NA + "_mean_gap_st": round(float(np.mean(ga)), 2), NB + "_mean_gap_st": round(float(np.mean(gb)), 2),
            NA + "_turns_over_4st": int(sum(g > 4 for g in ga)), NB + "_turns_over_4st": int(sum(g > 4 for g in gb)),
            "closer_in_" + NB: sum(y < x - 0.5 for x, y in zip(ga, gb)), "closer_in_" + NA: sum(x < y - 0.5 for x, y in zip(ga, gb))}
spread = {}
for c in sorted({r["char"] for r in rows if r["char"]}, key=lambda c: int(c[1:])):
    d = {}
    for n in (NA, NB):
        v = [12 * np.log2(r[n + "_hz"] / 100) for r in rows if r["char"] == c and r[n + "_hz"]]
        d[n] = round(float(np.std(v)), 2) if len(v) >= 3 else None
    if d[NA] is not None and d[NB] is not None: spread[c] = d
res = {"all_turns": summ(rows), "turns_in_multi_speaker_lines": summ([r for r in rows if r["turn_of_multi"]]),
       "per_character_pitch_spread_st": spread,
       "mean_character_spread_st": {n: round(float(np.mean([v[n] for v in spread.values()])), 2) for n in (NA, NB)} if spread else {},
       "turns": rows}
json.dump(res, open(out_f, "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
print(json.dumps({k: v for k, v in res.items() if k != "turns"}, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
