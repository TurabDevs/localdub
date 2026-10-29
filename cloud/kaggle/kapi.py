#!/usr/bin/env python3
"""Raw Kaggle API helper (no kaggle CLI). Auth: env KAGGLE_API_TOKEN (Bearer; proxy sentinel OK). Never prints the token.
usage:
  kapi.py ds-create  <slug> <title> <file>...      # new private dataset bugxdf/<slug> (CC0)
  kapi.py ds-version <slug> <notes> <file>...      # new version: REPLACES file set with given files (pass all files you want kept)
  kapi.py ds-status  <slug>
  kapi.py push <name> <script.py> <ds-slug>[,<ds-slug>...]   # private kernel, GPU (KAGGLE_GPU=1, KAGGLE_SHAPE=NvidiaTeslaT4) + internet ON; ds may be "user/slug" or kernel:"user/slug" or model:"owner/model/framework/variation/version"
  kapi.py status <name>
  kapi.py output <name> [destdir]                  # prints log tail; downloads files into destdir
"""
import os, sys, json, time, subprocess, urllib.parse
import requests
GPU = os.environ.get("KAGGLE_GPU", "1") == "1"
SHAPE = os.environ.get("KAGGLE_SHAPE", "NvidiaTeslaT4")

U = os.environ.get("KAGGLE_USER", "bugxdf"); BASE = "https://www.kaggle.com/api/v1"
def H(): return {"Authorization": "Bearer " + os.environ["KAGGLE_API_TOKEN"], "Content-Type": "application/json"}
def post(p, body, t=120):
    r = requests.post(BASE + p, headers=H(), json=body, timeout=t)
    try: return r.status_code, r.json()
    except Exception: return r.status_code, r.text[:500]
def get(p, t=120):
    r = requests.get(BASE + p, headers=H(), timeout=t)
    try: return r.status_code, r.json()
    except Exception: return r.status_code, r.text[:500]

class FileStream:
    def __init__(s, path, off=0): s.f = open(path, "rb"); s.f.seek(off); s.n = os.path.getsize(path) - off
    def __len__(s): return s.n
    def read(s, k=-1): return s.f.read(1 << 20 if k is None or k < 0 else k)

def upload(path, name=None):
    name = name or os.path.basename(path); L = os.path.getsize(path)
    c, j = post("/blobs/upload", {"type": "dataset", "name": name, "contentLength": L,
                                    "lastModifiedEpochSeconds": int(os.path.getmtime(path))})
    if c != 200 or "createUrl" not in j: raise SystemExit(f"blob start failed {c} {j}")
    url, tok = j["createUrl"], j["token"]; t0 = time.time()
    for attempt in range(6):
        off = 0
        if attempt:  # resume: query committed bytes
            r = requests.put(url, headers={"Content-Range": f"bytes */{L}", "Content-Length": "0"}, timeout=60)
            if r.status_code in (200, 201): break
            rg = r.headers.get("Range");  off = int(rg.split("-")[1]) + 1 if rg else 0
        try:
            hdr = {"Content-Length": str(L - off), "Content-Type": "application/octet-stream"}
            if off: hdr["Content-Range"] = f"bytes {off}-{L-1}/{L}"
            r = requests.put(url, data=FileStream(path, off), headers=hdr, timeout=3600)
            if r.status_code in (200, 201): break
            print("put", r.status_code, r.text[:200], file=sys.stderr)
        except Exception as e: print("put err", e, file=sys.stderr)
        time.sleep(5)
    else: raise SystemExit("upload failed " + path)
    dt = time.time() - t0; print(f"uploaded {name} {L/1e6:.1f}MB in {dt:.1f}s = {L/1e6/max(dt,.01):.1f} MB/s", file=sys.stderr)
    return tok

def files_tokens(paths): return [{"token": upload(p)} for p in paths]

def main():
    a = sys.argv[1:]; cmd = a[0]
    if cmd == "ds-create":
        c, j = post("/datasets/create/new", {"ownerSlug": U, "slug": a[1], "title": a[2], "isPrivate": True,
              "licenseName": "CC0-1.0", "files": files_tokens(a[3:]), "subtitle": "", "description": "", "categoryIds": []}, 600)
        print(c, json.dumps(j))
    elif cmd == "ds-version":
        c, j = post(f"/datasets/create/version/{U}/{a[1]}", {"versionNotes": a[2], "files": files_tokens(a[3:]),
              "deleteOldVersions": True, "categoryIds": []}, 600)
        print(c, json.dumps(j))
    elif cmd == "ds-status":
        print(*get(f"/datasets/status/{U}/{a[1]}"))
    elif cmd == "push":
        src = open(a[2]).read(); parts = a[3].split(",") if len(a) > 3 and a[3] else []
        ds = [d if "/" in d else f"{U}/{d}" for d in parts if not d.startswith("kernel:")]
        kds = [d[7:] for d in parts if d.startswith("kernel:")]
        mds = [d[6:] for d in parts if d.startswith("model:")]
        ds = [d for d in ds if not d.split("/")[-1].startswith("model:") and "model:" not in d]
        c, j = post("/kernels/push", {"slug": f"{U}/{a[1]}", "newTitle": a[1], "text": src, "language": "python",
              "kernelType": "script", "isPrivate": True, "enableGpu": GPU, "enableTpu": False, "enableInternet": os.environ.get("KAGGLE_INTERNET","0") == "1", "machineShape": SHAPE,
              "datasetDataSources": ds, "competitionDataSources": [], "kernelDataSources": kds, "modelDataSources": mds, "categoryIds": []})
        print(c, json.dumps(j))
    elif cmd == "status":
        print(*get(f"/kernels/status?userName={U}&kernelSlug={a[1]}"))
    elif cmd == "output":
        c, j = get(f"/kernels/output?userName={U}&kernelSlug={a[1]}", 300)
        if c != 200: print(c, j); return
        log = j.get("log") or ""; print("LOGTAIL:", log[-3000:])
        dest = a[2] if len(a) > 2 else None
        for f in j.get("files", []):
            print("FILE", f.get("fileName"), f.get("size", ""))
            if dest:
                os.makedirs(dest, exist_ok=True); out = os.path.join(dest, os.path.basename(f["fileName"]))
                t0 = time.time(); want = int(f.get("size") or 0)
                for att in range(3):   # 2026-10-21: never resume a stale .part (caused corrupt lectures); verify size
                    if os.path.exists(out + ".part"): os.remove(out + ".part")
                    subprocess.run(["curl", "-sSL", "--retry", "5", "-o", out + ".part", f["url"]], check=False)
                    if os.path.exists(out + ".part") and (not want or os.path.getsize(out + ".part") == want): break
                    print("  size mismatch, retry", att + 1)
                if not os.path.exists(out + ".part"): continue
                os.replace(out + ".part", out); sz = os.path.getsize(out); dt = time.time() - t0
                print(f"  got {sz/1e6:.1f}MB in {dt:.1f}s = {sz/1e6/max(dt,.01):.1f} MB/s")
    else: print(__doc__)

if __name__ == "__main__": main()
