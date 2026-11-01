#!/usr/bin/env python3
"""Character profiles -> Markdown table with face thumbnails, plus a contact-sheet image for the demo.

usage: char_table.py <chars_dir> <faces_thumb_dir> <out_dir> [min_speaking_s]
writes out_dir/characters.md, out_dir/thumbs/C*.jpg, out_dir/C*.json, out_dir/sheet.jpg"""
import json, os, shutil, sys
CD, TD, OUT = sys.argv[1:4]; MIN = float(sys.argv[4]) if len(sys.argv) > 4 else 2.0
os.makedirs(f"{OUT}/thumbs", exist_ok=True)
P = json.load(open(f"{CD}/characters.json"))
rows = ["| | id | kind | screen time | speaks in | speaking time | pitch (p10 / median / p90) | gender (estimate) | age (Glimmer, estimate) | style | voice used for the dub |",
        "|---|---|---|---|---|---|---|---|---|---|---|"]
for p in P:
    if p["speaking_time_s"] < MIN: continue
    img = ""
    if p["face_id"]:
        src = os.path.join(TD, f"{p['face_id']}.jpg")
        if os.path.exists(src): shutil.copy(src, f"{OUT}/thumbs/{p['id']}.jpg"); img = f"<img src=\"thumbs/{p['id']}.jpg\" width=\"72\">"
    ph = p["pitch_hz"]; pt = f"{ph['p10']} / {ph['median']} / {ph['p90']} Hz" if ph else "n/a"
    g = p["gender_estimate"]["estimate"] or "n/a"
    rows.append(f"| {img} | {p['id']} | {p['kind']} | {p['screen_time_s']} s | {len(p['dub_lines'])} lines ({p['speech_units']} turns) | "
                f"{p['speaking_time_s']} s | {pt} | {g} | {p['age_estimate_glimmer'] or 'n/a'} | {p['energy']['style']} | {p['voice_reason']} |")
    shutil.copy(f"{CD}/characters/{p['id']}.json", f"{OUT}/{p['id']}.json")
small = [p["id"] for p in P if p["speaking_time_s"] < MIN]
md = ("\n".join(rows) + (f"\n\n{len(small)} more voice groups speak for less than {MIN:.0f} s each ({', '.join(small)}); they are in characters.json.\n" if small else "\n"))
open(f"{OUT}/characters.md", "w").write(md)
shutil.copy(f"{CD}/characters.json", f"{OUT}/characters.json")
try:
    import cv2, numpy as np
    tiles = []
    for p in P[:10]:
        if not p["face_id"] or p["speaking_time_s"] < MIN: continue
        im = cv2.imread(f"{OUT}/thumbs/{p['id']}.jpg")
        if im is None: continue
        im = cv2.resize(im, (200, 250)); tiles.append((p, im))
    W = 1280; card = np.full((720, W, 3), (40, 26, 16), np.uint8)
    n = len(tiles); cw = min(220, (W - 40) // max(1, n)); x0 = (W - cw * n) // 2
    for i, (p, im) in enumerate(tiles):
        im = cv2.resize(im, (cw - 16, int((cw - 16) * 1.25))); y = 250; x = x0 + i * cw + 8
        card[y:y + im.shape[0], x:x + im.shape[1]] = im
        ty = y + im.shape[0] + 26
        for k, t in enumerate([p["id"], f"{p['speaking_time_s']} s speech", f"{len(p['dub_lines'])} lines"]):
            cv2.putText(card, t, (x, ty + 24 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.55 if k else 0.7, (255, 255, 255) if not k else (200, 210, 220), 1 if k else 2, cv2.LINE_AA)
    cv2.imwrite(f"{OUT}/sheet.jpg", card)
except Exception as e:
    print("no sheet:", e)
print(md)
