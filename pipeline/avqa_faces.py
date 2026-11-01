#!/usr/bin/env python3
"""LocalDub AV-QA, visual pass: who is on screen and is their mouth moving?

usage: avqa_faces.py <video> <faces.json> [--fps 8] [--start S] [--end E] [--models DIR]

Runs frame by frame (default 8 fps) on CPU:
  * YuNet face detector (OpenCV zoo, ONNX) -> boxes + 5 points, works on small and three-quarter faces
  * MediaPipe Face Landmarker (478-point mesh + blendshapes) on each enlarged face crop -> inner-lip gap and jawOpen
  * mouth-region motion (frame difference inside the box between the mouth corners) -> works even on profile faces
    where the mesh fails
  * simple IoU tracker -> a stable track id per face
The pass is resumable: every call adds frames in [start, end) to faces.json, so a long video can be done in chunks
that each fit a short CPU job. Results are reused for every dubbed version of the same source.
"""
import argparse, json, os, subprocess, sys
import numpy as np, cv2

ap = argparse.ArgumentParser()
ap.add_argument("video"); ap.add_argument("out")
ap.add_argument("--fps", type=float, default=8.0); ap.add_argument("--start", type=float, default=0.0)
ap.add_argument("--end", type=float, default=None); ap.add_argument("--width", type=int, default=960)
ap.add_argument("--models", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models"))
a = ap.parse_args()

dur = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", a.video],
                           capture_output=True, text=True).stdout.strip())
end = min(a.end or dur, dur)
W = a.width; H = int(round(W * 9 / 16 / 2) * 2)
st = json.load(open(a.out)) if os.path.exists(a.out) else {"video": os.path.basename(a.video), "fps": a.fps, "w": W, "h": H,
                                                            "duration": dur, "frames": {}, "next_track": 0}
assert abs(st["fps"] - a.fps) < 1e-6, "fps differs from the existing cache"

det = cv2.FaceDetectorYN.create(os.path.join(a.models, "yunet.onnx"), "", (W, H), 0.6, 0.3, 50)
lmk = None
try:   # MediaPipe needs libEGL/libGLESv2 present even on CPU; without it we fall back to motion only
    import mediapipe as mp
    from mediapipe.tasks.python import vision, BaseOptions
    lmk = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=os.path.join(a.models, "face_landmarker.task")),
        running_mode=vision.RunningMode.IMAGE, num_faces=1, output_face_blendshapes=True,
        min_face_detection_confidence=0.3, min_face_presence_confidence=0.3))
except Exception as e:
    print("landmarker unavailable, motion only:", e, file=sys.stderr)

def iou(p, q):
    x1, y1 = max(p[0], q[0]), max(p[1], q[1]); x2, y2 = min(p[0] + p[2], q[0] + q[2]), min(p[1] + p[3], q[1] + q[3])
    i = max(0, x2 - x1) * max(0, y2 - y1); return i / (p[2] * p[3] + q[2] * q[3] - i + 1e-9)

# previous frame (for tracking + motion) = last cached frame just before start, if any
keys = sorted(st["frames"], key=float)
prev_faces = []; prev_gray = None
cmd = f"ffmpeg -v error -ss {a.start:.3f} -i '{a.video}' -t {end - a.start:.3f} -vf fps={a.fps},scale={W}:{H} -f rawvideo -pix_fmt bgr24 -"
p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE); n = 0
while True:
    b = p.stdout.read(W * H * 3)
    if len(b) < W * H * 3: break
    t = round(a.start + n / a.fps, 3); n += 1
    img = np.frombuffer(b, np.uint8).reshape(H, W, 3); gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    g = float(np.abs(gray.astype(np.float32) - prev_gray.astype(np.float32)).mean()) if prev_gray is not None else 0.0
    cut = g > 25.0                                                         # scene cut: mouth motion meaningless here
    _, F = det.detect(img); faces = []
    for f in (F if F is not None else []):
        x, y, w, h = [float(v) for v in f[:4]]; sc = float(f[14])
        mr, ml = f[10:12], f[12:14]                                        # mouth corners
        cw = float(np.hypot(*(ml - mr))) + 1e-6; cx, cy = (mr + ml) / 2
        mb = [int(cx - 0.8 * cw), int(cy - 0.6 * cw), int(cx + 0.8 * cw), int(cy + 0.7 * cw)]
        face = {"box": [round(x / W, 4), round(y / H, 4), round(w / W, 4), round(h / H, 4)], "score": round(sc, 3),
                "open": None, "jaw": None, "motion": None}
        # track: best IoU with a face of the previous frame
        best = max(prev_faces, key=lambda q: iou(q["_px"], (x, y, w, h)), default=None)
        if best is not None and iou(best["_px"], (x, y, w, h)) > 0.25:
            face["track"] = best["track"]
            if prev_gray is not None and not cut:                                      # mouth motion vs previous frame, same box
                x0, y0, x1, y1 = max(0, mb[0]), max(0, mb[1]), min(W, mb[2]), min(H, mb[3])
                if x1 - x0 > 4 and y1 - y0 > 4:
                    d = np.abs(gray[y0:y1, x0:x1].astype(np.float32) - prev_gray[y0:y1, x0:x1].astype(np.float32))
                    face["motion"] = round(float(d.mean()) / (float(gray[y0:y1, x0:x1].std()) + 8.0), 4)
        else:
            face["track"] = st["next_track"]; st["next_track"] += 1
        if lmk is not None:                                                # mesh on an enlarged crop
            m = 0.45; X0, Y0 = int(max(0, x - m * w)), int(max(0, y - m * h)); X1, Y1 = int(min(W, x + w * (1 + m))), int(min(H, y + h * (1 + m)))
            crop = cv2.cvtColor(img[Y0:Y1, X0:X1], cv2.COLOR_BGR2RGB)
            if crop.size and min(crop.shape[:2]) > 20:
                s = 256 / max(crop.shape[:2]); crop = cv2.resize(crop, None, fx=s, fy=s) if s > 1 else crop
                r = lmk.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(crop)))
                if r.face_landmarks:
                    L = r.face_landmarks[0]
                    fh = np.hypot(L[10].x - L[152].x, L[10].y - L[152].y) + 1e-6
                    face["open"] = round(float(np.hypot(L[13].x - L[14].x, L[13].y - L[14].y) / fh), 4)
                    bs = {c.category_name: c.score for c in r.face_blendshapes[0]} if r.face_blendshapes else {}
                    face["jaw"] = round(float(bs.get("jawOpen", 0.0)), 4)
        face["_px"] = (x, y, w, h); faces.append(face)
    st["frames"][f"{t:.3f}"] = [{k: v for k, v in f.items() if k != "_px"} for f in faces]
    st.setdefault("gdiff", {})[f"{t:.3f}"] = round(g, 2)
    prev_faces, prev_gray = faces, gray
p.wait()
st["done_until"] = max(st.get("done_until", 0), round(a.start + n / a.fps, 3))
tmp = a.out + ".tmp"; json.dump(st, open(tmp, "w")); os.replace(tmp, a.out)
print(f"frames {n} [{a.start:.1f}, {a.start + n / a.fps:.1f}) -> {a.out}; with faces {sum(1 for k in st['frames'] if st['frames'][k])}/{len(st['frames'])}")
