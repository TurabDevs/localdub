#!/usr/bin/env python3
"""listen_prep.py <qa.json> <dialogue_en.wav> <lines.json> <tag> <outdir> -> dialogue_<tag>.flac (16k mono) + lines_<tag>.json"""
import json, sys, subprocess, os
qa, dia, lj, tag, od = sys.argv[1:6]; os.makedirs(od, exist_ok=True)
L = {l["id"]: l for l in json.load(open(lj))}
out = []
for q in json.load(open(qa))["lines"]:
    if q["variant"] == "original": continue
    s = L[q["id"]]["start"]; dur = max(q["slot"], q["tts"] / max(q.get("tempo", 1.0), 0.5)) + 0.3
    out.append({"id": q["id"], "start": round(max(0, s - 0.1), 3), "end": round(s + dur, 3), "en": q["en"]})
json.dump(out, open(f"{od}/lines_{tag}.json", "w"), ensure_ascii=False, indent=0)
subprocess.run(f"ffmpeg -v error -y -i {dia} -ac 1 -ar 16000 {od}/dialogue_{tag}.flac", shell=True, check=True)
print(len(out), "lines")
