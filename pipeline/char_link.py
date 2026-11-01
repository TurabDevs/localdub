#!/usr/bin/env python3
"""LocalDub character stage: link every piece of speech to the character who is actually saying it.

usage: char_link.py <whisper.json> <wsub.json> <wemb.npy> <face_ids.json> <faces_a.json,faces_b.json,...> <vocals.flac>
                    <lines.json> <cast_results.json,...> <out_dir>

Why: up to v6 a dubbing line was a voice-activity chunk and its voice was cloned from that chunk's own audio. When one
chunk held two people, or a short line borrowed a voice from another line, the dub spoke in A's voice while B's mouth
moved on screen.

Steps
  1. speech units: Whisper large-v3 sub-segments (word timestamps, split at gaps > 0.6 s) with one ECAPA embedding each
  2. active speaker per unit: for every face track on screen during the unit, mouth activity (frame-difference motion
     and lip-gap variation from avqa_faces.py) relative to that track's own baseline while nobody speaks; the clearly
     most active face is the speaker. The track's identity comes from char_faces.py (SFace clustering across shots).
  3. units without a clear on-screen speaker are matched by voice (ECAPA cosine) to the face-confirmed characters;
     a voice that matches nobody becomes a voice-only character (off-screen or phone voice)
  4. character profiles: screen time, lines, speaking time, pitch range, estimated gender/age (pitch + Muse Glimmer's
     per-line cast labels, always marked as estimates), energy, sample lines, and a reference voice built only from that
     character's own face-confirmed speech
  5. per dubbing line: which characters speak in it and where the turn changes (used to split lines and pick voices)
Outputs in out_dir: units.json, characters.json, characters/C*.json, line_chars.json
"""
import json, os, re, sys, collections
import numpy as np, soundfile as sf

wj, sj, ej, fj, faces_files, voc_f, lines_f, cast_fs, OUT = sys.argv[1:10]
os.makedirs(f"{OUT}/characters", exist_ok=True)
W = json.load(open(wj)); SUB = json.load(open(sj)); EMB = np.load(ej); FID = json.load(open(fj))
LINES = json.load(open(lines_f))
CAST = {}
for cf in filter(None, cast_fs.split(",")):
    for k, v in json.load(open(cf)).items():
        if k.startswith("cast_"):
            try: CAST[k[5:]] = json.loads(re.search(r"\{.*\}", v.get("content") or "", re.S).group(0))
            except Exception: pass
FPS = 8.0
frames = {}; tag_of = {}
for f in faces_files.split(","):
    d = json.load(open(f)); tag = os.path.basename(f).replace(".json", "")
    for t, fs in d["frames"].items():
        frames[round(float(t), 3)] = [dict(x, key=f"{tag}:{x['track']}") for x in fs]
times = np.array(sorted(frames))
voc, SR = sf.read(voc_f, dtype="float32", always_2d=True); voc = voc.mean(1)

def f0_track(x, fs):
    st = max(1, fs // 16000); x = x[::st]; fs = fs / st; n = int(0.04 * fs); lo, hi = int(fs / 400), int(fs / 70); out = []
    for i in range(0, len(x) - n, n // 2):
        fr = x[i:i + n] - x[i:i + n].mean()
        if np.sqrt(np.mean(fr ** 2)) < 0.01: continue
        ac = np.correlate(fr, fr, "full")[n - 1:]; ac /= ac[0] + 1e-9; k = lo + int(np.argmax(ac[lo:hi]))
        if ac[k] > 0.5: out.append(fs / k)
    return out

# ---------- 1. units
U = []
for i, s in enumerate(SUB):
    if s["end"] - s["start"] < 0.15 or not s["text"].strip(): continue
    U.append({"u": len(U), "sub": i, "start": round(s["start"], 2), "end": round(s["end"], 2), "text": s["text"].strip()})
emb = {u["u"]: EMB[u["sub"]] for u in U}
speech = np.zeros(len(times), bool)
for u in U: speech |= (times >= u["start"]) & (times < u["end"])

# ---------- 2. active speaker
base = collections.defaultdict(lambda: {"m": [], "o": []}); allv = collections.defaultdict(lambda: {"m": [], "o": []})
for j, t in enumerate(times):
    for x in frames[t]:
        b = allv[x["key"]] if speech[j] else base[x["key"]]
        if x.get("motion") is not None: b["m"].append(x["motion"])
        if x.get("open") is not None: b["o"].append(x["open"])
gm = np.median([v for k in base for v in base[k]["m"]] or [0.3])
def baseline(k):
    m = base[k]["m"]; return float(np.median(m)) if len(m) >= 8 else float(gm)

def active(a, b):
    idx = np.where((times >= a) & (times < b))[0]
    if len(idx) == 0: return None, []
    per = collections.defaultdict(lambda: {"n": 0, "m": [], "o": [], "w": []})
    for j in idx:
        for x in frames[times[j]]:
            p = per[x["key"]]; p["n"] += 1; p["w"].append(x["box"][2])
            if x.get("motion") is not None: p["m"].append(x["motion"])
            if x.get("open") is not None: p["o"].append(x["open"])
    cands = []
    for k, p in per.items():
        if p["n"] < 0.4 * len(idx) or np.median(p["w"]) < 0.025 or len(p["m"]) < 2: continue
        sc = float(np.log((np.mean(p["m"]) + 1e-3) / (baseline(k) + 1e-3)))
        if len(p["o"]) >= 3: sc += float(np.clip(np.std(p["o"]) / 0.01 - 1.0, -1, 2)) * 0.5   # lips opening and closing
        cands.append({"track": k, "face": FID["track_face"].get(k), "score": round(sc, 3), "vis": round(p["n"] / len(idx), 2),
                      "w": round(float(np.median(p["w"])), 3)})
    cands.sort(key=lambda c: -c["score"])
    if not cands: return None, []
    top = cands[0]; second = cands[1]["score"] if len(cands) > 1 else -9
    ok = top["face"] is not None and top["score"] > 0.15 and (top["score"] - second > 0.25 or len(cands) == 1)
    return (top if ok else None), cands

for u in U:
    top, c = active(u["start"], u["end"]); u["faces_on_screen"] = [x["face"] or "?" for x in c]
    u["face_cands"] = c[:3]
    if top: u["char_face"] = top["face"]; u["evidence"] = f"mouth activity {top['score']:+.2f} (face {top['face']})"

# ---------- 3. voices
def centroid(us):
    e = np.stack([emb[u["u"]] for u in us]).mean(0); return e / (np.linalg.norm(e) + 1e-9)
byface = collections.defaultdict(list)
for u in U:
    if u.get("char_face") and u["end"] - u["start"] >= 0.8: byface[u["char_face"]].append(u)
C = {f: centroid(us) for f, us in byface.items() if us}
# drop face-confirmed units whose voice is far from their own character (detector mistakes: wrong face picked)
for f, us in byface.items():
    for u in us:
        s = float(emb[u["u"]] @ C[f])
        if s < 0.25: u.pop("char_face"); u["evidence"] = f"face {f} rejected: voice sim {s:.2f}"
byface = collections.defaultdict(list)
for u in U:
    if u.get("char_face"): byface[u["char_face"]].append(u)
C = {f: centroid(us) for f, us in byface.items() if us}
VOICE_SIM = float(os.environ.get("VOICE_SIM", "0.45"))
vo = []   # voice-only groups
for u in U:
    if u.get("char_face"): u["char"] = u["char_face"]; u["how"] = "face"; continue
    sims = {f: float(emb[u["u"]] @ c) for f, c in C.items()}
    f, s = max(sims.items(), key=lambda x: x[1]) if sims else (None, -1)
    on = set(u["faces_on_screen"])
    if f and s >= VOICE_SIM:
        u["char"] = f; u["how"] = "voice"; u["evidence"] = (u.get("evidence", "") + f" voice sim {s:.2f} to {f}").strip(); continue
    best = None
    for g in vo:
        sg = float(emb[u["u"]] @ (g["e"] / np.linalg.norm(g["e"])))
        if sg >= VOICE_SIM and (best is None or sg > best[1]): best = (g, sg)
    if best: g = best[0]; g["us"].append(u); g["e"] = g["e"] + emb[u["u"]]
    else: g = {"id": f"V{len(vo)}", "us": [u], "e": emb[u["u"]].copy()}; vo.append(g)
    u["char"] = g["id"]; u["how"] = "voice-only"; u["evidence"] = (u.get("evidence", "") + f" no on-screen match (best {f} {s:.2f})").strip()

# ---------- 4. profiles
def overlap(a0, a1, b0, b1): return max(0.0, min(a1, b1) - max(a0, b0))
chars = collections.defaultdict(list)
for u in U: chars[u["char"]].append(u)
faces = {f["id"]: f for f in FID["faces"]}
order = sorted(chars, key=lambda k: -sum(u["end"] - u["start"] for u in chars[k]))
cid = {k: f"C{i}" for i, k in enumerate(order)}
for u in U: u["cid"] = cid[u["char"]]
prof = []
for k in order:
    us = chars[k]; sp = sum(u["end"] - u["start"] for u in us)
    f0 = []
    for u in us: f0 += f0_track(voc[int(u["start"] * SR):int(u["end"] * SR)], SR)
    lines_in = sorted({L["id"] for L in LINES for u in us if overlap(L["start"], L["end"], u["start"], u["end"]) > 0.3 * (u["end"] - u["start"])})
    gv = collections.Counter(); av = collections.Counter(); desc = collections.Counter()
    for lid in lines_in:
        c = CAST.get(lid)
        if c and c.get("face_visible", True):
            gv[c.get("gender")] += 1; av[c.get("age")] += 1; desc[c.get("speaker")] += 1
    med = float(np.median(f0)) if len(f0) >= 10 else None
    g_pitch = None if med is None else ("male" if med < 165 else "female")
    rms = [float(np.sqrt(np.mean(voc[int(u["start"] * SR):int(u["end"] * SR)] ** 2) + 1e-9)) for u in us]
    loud = 20 * np.log10(np.median(rms))
    # reference: longest own face-confirmed units (or voice units for voice-only), up to ~10 s
    pool = sorted([u for u in us if u["how"] == "face"] or us, key=lambda u: -(u["end"] - u["start"]))
    ref, tot = [], 0.0
    for u in pool:
        if u["end"] - u["start"] < 1.0 and tot > 0: continue
        ref.append({"start": u["start"], "end": min(u["end"], u["start"] + 6.0), "text": u["text"]}); tot += ref[-1]["end"] - ref[-1]["start"]
        if tot >= 9.0: break
    fc = faces.get(k)
    p = {"id": cid[k], "face_id": k if fc else None, "kind": "on-screen" if fc else "voice-only (off-screen / phone / narrator)",
         "thumb": fc["thumb"] if fc else None, "screen_time_s": fc["screen_time_s"] if fc else 0.0,
         "speech_units": len(us), "speaking_time_s": round(sp, 1), "dub_lines": lines_in,
         "face_confirmed_units": sum(1 for u in us if u["how"] == "face"), "voice_matched_units": sum(1 for u in us if u["how"] != "face"),
         "pitch_hz": None if med is None else {"p10": round(float(np.percentile(f0, 10))), "median": round(med), "p90": round(float(np.percentile(f0, 90)))},
         "gender_estimate": {"from_pitch": g_pitch, "from_glimmer_cast": gv.most_common(1)[0][0] if gv else None},
         "age_estimate_glimmer": av.most_common(1)[0][0] if av else None,
         "glimmer_description": desc.most_common(1)[0][0] if desc else None,
         "energy": {"median_level_dbfs": round(float(loud), 1)},
         "sample_lines": [u["text"] for u in sorted(us, key=lambda u: -(u["end"] - u["start"]))[:3]],
         "voice_reference": ref, "voice_reference_s": round(tot, 1),
         "voice_reason": (f"cloned from {round(tot, 1)} s of this character's own speech ({len(ref)} clips where the face detector saw this face talking)"
                          if pool and pool[0]["how"] == "face" else f"cloned from {round(tot, 1)} s of this off-screen voice ({len(ref)} clips grouped by voice)"),
         "note": "gender/age are estimates (pitch + Muse Glimmer's per-line labels), not facts about the actor"}
    prof.append(p); json.dump(p, open(f"{OUT}/characters/{p['id']}.json", "w"), ensure_ascii=False, indent=1)
lev = sorted(p["energy"]["median_level_dbfs"] for p in prof)
for p in prof:
    e = p["energy"]["median_level_dbfs"]; p["energy"]["style"] = "loud / animated" if e >= lev[int(0.66 * (len(lev) - 1))] else ("soft" if e <= lev[int(0.33 * (len(lev) - 1))] else "medium")
    json.dump(p, open(f"{OUT}/characters/{p['id']}.json", "w"), ensure_ascii=False, indent=1)
json.dump(prof, open(f"{OUT}/characters.json", "w"), ensure_ascii=False, indent=1)
json.dump(U, open(f"{OUT}/units.json", "w"), ensure_ascii=False, indent=1)

# ---------- 5. per line
LC = {}
for L in LINES:
    parts = [u for u in U if overlap(L["start"], L["end"], u["start"], u["end"]) > 0.5 * (u["end"] - u["start"])]
    comp = collections.Counter()
    for u in parts: comp[u["cid"]] += overlap(L["start"], L["end"], u["start"], u["end"])
    turns = []
    for u in sorted(parts, key=lambda u: u["start"]):
        if turns and turns[-1]["cid"] == u["cid"]: turns[-1]["end"] = u["end"]; turns[-1]["text"] += " " + u["text"]; turns[-1]["how"].append(u["how"])
        else: turns.append({"cid": u["cid"], "start": u["start"], "end": u["end"], "text": u["text"], "how": [u["how"]]})
    tot = sum(comp.values()) or 1
    LC[L["id"]] = {"start": L["start"], "end": L["end"], "main": comp.most_common(1)[0][0] if comp else None,
                   "share": {k: round(v / tot, 2) for k, v in comp.most_common()}, "turns": turns}
json.dump(LC, open(f"{OUT}/line_chars.json", "w"), ensure_ascii=False, indent=1)
print("units", len(U), "face-confirmed", sum(1 for u in U if u["how"] == "face"), "voice", sum(1 for u in U if u["how"] == "voice"),
      "voice-only", sum(1 for u in U if u["how"] == "voice-only"))
for p in prof: print(p["id"], p["face_id"], p["kind"][:9], p["speech_units"], "units", p["speaking_time_s"], "s", p["pitch_hz"], p["gender_estimate"], p["age_estimate_glimmer"], len(p["dub_lines"]), "lines")
mixed = [k for k, v in LC.items() if len([1 for t in v["turns"] if t["end"] - t["start"] >= 0.8]) >= 2 and len(v["share"]) >= 2]
print("lines with >=2 characters:", len(mixed), mixed)
