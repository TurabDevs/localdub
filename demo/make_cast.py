#!/usr/bin/env python3
"""Glimmer-sees-the-scene scene(s): frame + Glimmer JSON. usage: make_cast.py (writes $SCENES/s4c_cast.mp4)"""
import json, subprocess, os
ROOT="/workspace/projects/db13b071-b36a-44da-bacb-efac377e3057"; D=f"{ROOT}/build/demo"
SC=os.environ.get("SCENES", f"{D}/scenes_v3"); os.makedirs(SC, exist_ok=True)
MONO="/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"; FONT="/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"; FONTB="/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
V="-c:v libx264 -preset veryfast -crf 22 -pix_fmt yuv420p -r 25 -c:a aac -b:a 160k -ar 48000 -ac 2"
allc={}
for i in (0,1,2): allc.update(json.load(open(f"{ROOT}/build/work/s2/run1/cast{i}/results.json")))
lines=json.load(open(f"{ROOT}/build/work/s2/run1/lines.json")); LT={l['id']:l for l in lines}
tr={}
for i in (0,1): tr.update(json.load(open(f"{ROOT}/build/work/s2/run1/tr{i}/results.json")))
ids=["L002","L003","L022"]; dur=21.0; seg=dur/len(ids)
def esc(t): return t.replace("\\","\\\\").replace(":","\\:").replace("'","\u2019").replace("%","\\%")
parts=[]
for n,i in enumerate(ids):
    c=json.loads(allc["cast_"+i]["content"])
    out=f"{SC}/_cast_{i}.mp4"
    txt=[("Muse Glimmer sees the frame at the moment of each line",FONT,26,"0x9fb3c8",40,40)]
    import textwrap
    sp=textwrap.wrap(c["speaker"],26); sp=[("speaker : " if k==0 else "          ")+w for k,w in enumerate(sp)]
    rows=sp+[f"gender  : {c['gender']}", f"age     : {c['age']}", f"role    : {c['role']}", f"face    : {'visible' if c['face_visible'] else 'hidden'}"]
    dt=[f"drawtext=fontfile={FONT}:text='{esc(txt[0][0])}':fontsize=26:fontcolor=0x9fb3c8:x=40:y=40"]
    dt.append(f"drawtext=fontfile={FONTB}:text='Glimmer returns (JSON)':fontsize=34:fontcolor=0x7fffd4:x=700:y=150")
    for k,r in enumerate(rows):
        dt.append(f"drawtext=fontfile={MONO}:text='{esc(r)}':fontsize=24:fontcolor=white:x=700:y={215+k*40}")
    dt.append(f"drawtext=fontfile={FONTB}:text='-> QA checks the cloned voice against this':fontsize=30:fontcolor=0x7fffd4:x=40:y=540")
    subprocess.run(f"ffmpeg -v error -y -f lavfi -i color=c=0x101418:s=1280x720:d={seg} -loop 1 -t {seg} -i {D}/cast/{i}.png -filter_complex \"[0:v][1:v]overlay=40:150[b];[b]{','.join(dt)}[v]\" -map [v] -an -t {seg} -c:v libx264 -preset veryfast -crf 22 -pix_fmt yuv420p -r 25 {out}",shell=True,check=True)
    parts.append(out)
open(f"{SC}/_cast_list.txt","w").write("".join(f"file '{p}'\n" for p in parts))
subprocess.run(f"ffmpeg -v error -y -f concat -safe 0 -i {SC}/_cast_list.txt -i {D}/vo/vo4c.mp3 -filter_complex \"[1:a]apad,atrim=0:{dur}[a]\" -map 0:v -map [a] -t {dur} {V} {SC}/s4c_cast.mp4",shell=True,check=True)
print("ok")
