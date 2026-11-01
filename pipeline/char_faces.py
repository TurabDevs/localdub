#!/usr/bin/env python3
"""LocalDub character stage, visual part: who is on screen, as stable identities across shots.

usage: char_faces.py embed <video> <faces.json> <state.json> [--models DIR]     # resumable: one faces chunk file per call
       char_faces.py cluster <state.json> <out_faces_ids.json> <thumbdir>

Input faces.json files come from avqa_faces.py (YuNet boxes + mouth openness/motion + an IoU track id per shot, 8 fps).
embed:   for each face track that lasts >= 0.75 s with a usable face size, take up to 4 frames with the biggest box,
         re-detect with YuNet on that frame, align and run SFace (OpenCV zoo face recognition, 128-d) -> one mean
         embedding per track + the best crop as a thumbnail.
cluster: average-linkage clustering of track embeddings (cosine; SFace's same-person threshold ~0.36) -> face identities
         F0, F1, ... with screen time and a thumbnail each. Tracks too small to embed stay unassigned.
"""
import argparse, json, os, subprocess, sys
import numpy as np, cv2

ap = argparse.ArgumentParser()
ap.add_argument("cmd"); ap.add_argument("args", nargs="*")
ap.add_argument("--models", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models"))
a = ap.parse_args()

def iou(p, q):
    x1, y1 = max(p[0], q[0]), max(p[1], q[1]); x2, y2 = min(p[0] + p[2], q[0] + q[2]), min(p[1] + p[3], q[1] + q[3])
    i = max(0, x2 - x1) * max(0, y2 - y1); return i / (p[2] * p[3] + q[2] * q[3] - i + 1e-9)

def avg_link(E, sim):
    """average-linkage agglomerative clustering on cosine similarity (numpy only); merge while mean sim >= sim"""
    C = [[i] for i in range(len(E))]; S = E @ E.T
    while len(C) > 1:
        best, bi, bj = -2, -1, -1
        for i in range(len(C)):
            for j in range(i + 1, len(C)):
                m = S[np.ix_(C[i], C[j])].mean()
                if m > best: best, bi, bj = m, i, j
        if best < sim: break
        C[bi] += C[bj]; del C[bj]
    lab = np.zeros(len(E), int)
    for l, c in enumerate(C): lab[c] = l
    return lab

if a.cmd == "embed":
    video, fj, sj = a.args
    F = json.load(open(fj)); tag = os.path.basename(fj).replace(".json", "")
    st = json.load(open(sj)) if os.path.exists(sj) else {"tracks": {}, "done": []}
    if tag in st["done"]: print("already", tag); sys.exit(0)
    W, H, fps = F["w"], F["h"], F["fps"]
    tr = {}
    for t, fs in F["frames"].items():
        for f in fs: tr.setdefault(f["track"], []).append((float(t), f))
    want = {}   # time -> [(track, box)]
    for k, L in tr.items():
        big = [x for x in L if x[1]["box"][2] >= 0.03]
        if len(L) < 6 or len(big) < 3: continue
        for t, f in sorted(big, key=lambda x: -x[1]["box"][2] * x[1]["box"][3])[:4]:
            want.setdefault(round(t, 3), []).append((k, f["box"]))
    t0 = min(float(t) for t in F["frames"]); t1 = max(float(t) for t in F["frames"]) + 1 / fps
    det = cv2.FaceDetectorYN.create(os.path.join(a.models, "yunet.onnx"), "", (W, H), 0.5, 0.3, 50)
    rec = cv2.FaceRecognizerSF.create(os.path.join(a.models, "sface.onnx"), "")
    cmd = f"ffmpeg -v error -ss {t0:.3f} -i '{video}' -t {t1 - t0:.3f} -vf fps={fps},scale={W}:{H} -f rawvideo -pix_fmt bgr24 -"
    p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE); n = 0
    os.makedirs(os.path.join(os.path.dirname(sj), "crops"), exist_ok=True)
    while True:
        b = p.stdout.read(W * H * 3)
        if len(b) < W * H * 3: break
        t = round(t0 + n / fps, 3); n += 1
        if t not in want: continue
        img = np.frombuffer(b, np.uint8).reshape(H, W, 3).copy()
        _, D = det.detect(img); D = D if D is not None else []
        for k, box in want[t]:
            px = (box[0] * W, box[1] * H, box[2] * W, box[3] * H)
            best = max(D, key=lambda d: iou(px, d[:4]), default=None)
            if best is None or iou(px, best[:4]) < 0.3: continue
            crop = rec.alignCrop(img, best); e = rec.feature(crop).flatten(); e = e / (np.linalg.norm(e) + 1e-9)
            key = f"{tag}:{k}"; rec_ = st["tracks"].setdefault(key, {"emb": [], "area": 0.0, "thumb": None})
            rec_["emb"].append([round(float(v), 5) for v in e])
            area = box[2] * box[3]
            if area > rec_["area"]:
                x, y, w, h = [int(v) for v in px]; m = int(0.35 * max(w, h))
                th = img[max(0, y - m):y + h + m, max(0, x - m):x + w + m]
                fn = os.path.join(os.path.dirname(sj), "crops", f"{tag}_{k}.jpg")
                cv2.imwrite(fn, cv2.resize(th, (160, int(160 * th.shape[0] / max(1, th.shape[1]))))); rec_["thumb"] = fn; rec_["area"] = area
    p.wait()
    # per-track spans (all tracks, embedded or not) for later linking
    for k, L in tr.items():
        key = f"{tag}:{k}"; ts = sorted(x[0] for x in L)
        st.setdefault("spans", {})[key] = [ts[0], ts[-1] + 1 / fps, len(ts)]
    st["done"].append(tag); json.dump(st, open(sj + ".tmp", "w")); os.replace(sj + ".tmp", sj)
    print(tag, "frames", n, "embedded tracks", sum(1 for x in st["tracks"] if x.startswith(tag + ":")))

elif a.cmd == "cluster":
    sj, oj, tdir = a.args
    st = json.load(open(sj)); keys = [k for k, v in st["tracks"].items() if v["emb"]]
    E = np.stack([np.mean(st["tracks"][k]["emb"], 0) for k in keys]); E /= np.linalg.norm(E, axis=1, keepdims=True)
    lab = avg_link(E, float(os.environ.get("FACE_SIM", "0.36")))
    ids = {}
    for k, l in zip(keys, lab): ids.setdefault(int(l), []).append(k)
    order = sorted(ids, key=lambda l: -sum(st["spans"][k][2] for k in ids[l]))
    os.makedirs(tdir, exist_ok=True); out = {"faces": [], "track_face": {}}
    for i, l in enumerate(order):
        ks = ids[l]; fid = f"F{i}"
        best = max(ks, key=lambda k: st["tracks"][k]["area"])
        th = os.path.join(tdir, f"{fid}.jpg"); os.system(f"cp '{st['tracks'][best]['thumb']}' '{th}'")
        frames = sum(st["spans"][k][2] for k in ks)
        out["faces"].append({"id": fid, "tracks": ks, "screen_time_s": round(frames / 8.0, 1), "thumb": th})
        for k in ks: out["track_face"][k] = fid
    out["spans"] = st["spans"]
    json.dump(out, open(oj, "w"), indent=1)
    for f in out["faces"]: print(f["id"], len(f["tracks"]), "tracks", f["screen_time_s"], "s")
