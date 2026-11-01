#!/usr/bin/env python3
"""Subtitles for a LocalDub run: dubbed-language, source-language and bilingual tracks, as SRT and WebVTT.

usage: subs.py <qa.json> <lines.json> <out_prefix> [--burn VIDEO OUT.mp4] [--max-chars 42]
writes <out_prefix>.dub.srt/.vtt      what the dub says (the Glimmer line that was voiced)
       <out_prefix>.src.srt/.vtt      the original line (ASR transcript of the source)
       <out_prefix>.bi.srt/.vtt       both, dub on top, original underneath
Timing: each cue starts at the line start and lasts until the dub audio ends (placed_end) or the line ends, whichever
is later, clipped so cues never overlap. Lines that kept the original voice show the original text in every track.
--burn renders the bilingual track into a copy of the video (needs ffmpeg with libass).
"""
import argparse, json, os, subprocess, textwrap

ap = argparse.ArgumentParser(); ap.add_argument("qa"); ap.add_argument("lines"); ap.add_argument("out")
ap.add_argument("--burn", nargs=2, metavar=("VIDEO", "OUT")); ap.add_argument("--max-chars", type=int, default=42)
a = ap.parse_args()
L = {l["id"]: l for l in json.load(open(a.lines))}
cues = []
for q in json.load(open(a.qa))["lines"]:
    l = L.get(q["id"])
    if not l: continue
    src = (l.get("hi") or "").strip()
    dub = (q.get("en") or "").strip() if q.get("variant") != "original" else src
    end = l["start"] + max(l["end"] - l["start"], q.get("placed_end") or 0)
    cues.append({"start": l["start"], "end": end, "dub": dub, "src": src})
cues.sort(key=lambda c: c["start"])
for i in range(len(cues) - 1):
    cues[i]["end"] = min(cues[i]["end"], cues[i + 1]["start"] - 0.04)

def ts(t, sep):
    h, r = divmod(max(0.0, t), 3600); m, s = divmod(r, 60)
    return f"{int(h):02d}:{int(m):02d}:{int(s):02d}{sep}{int(round((s - int(s)) * 1000)):03d}"

def wrap(t): return "\n".join(textwrap.wrap(t, a.max_chars)) or " "

def write(kind, fmt):
    sep = "," if fmt == "srt" else "."; out = ["WEBVTT", ""] if fmt == "vtt" else []
    for n, c in enumerate(cues, 1):
        if kind == "bi": body = wrap(c["dub"]) + ("\n" + wrap(c["src"]) if c["src"] != c["dub"] else "")
        else: body = wrap(c[kind])
        if fmt == "srt": out.append(str(n))
        out += [f"{ts(c['start'], sep)} --> {ts(c['end'], sep)}", body, ""]
    p = f"{a.out}.{kind}.{fmt}"; open(p, "w", encoding="utf-8").write("\n".join(out)); return p

paths = [write(k, f) for k in ("dub", "src", "bi") for f in ("srt", "vtt")]
print(len(cues), "cues ->", ", ".join(os.path.basename(p) for p in paths))
if a.burn:
    v, o = a.burn; srt = os.path.abspath(f"{a.out}.bi.srt").replace(":", r"\:")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", v, "-vf", f"subtitles='{srt}':force_style='FontSize=18,Outline=2'",
                    "-c:v", "libx264", "-preset", "veryfast", "-crf", "21", "-c:a", "copy", o], check=True)
    print("burned ->", o)
