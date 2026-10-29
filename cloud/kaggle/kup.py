#!/usr/bin/env python3
"""Chunked, resumable Kaggle blob upload that survives short-lived shells (each call works <= BUDGET seconds).
usage:
  kup.py up <statefile> <file>...     # uploads as much as fits in BUDGET s; re-run until it prints ALL_DONE
  kup.py create <statefile> <slug> <title>      # create private dataset bugxdf/<slug> from finished uploads
  kup.py version <statefile> <slug> <notes>     # new version of existing dataset (replaces file set)
Auth: env KAGGLE_API_TOKEN (never printed)."""
import os, sys, json, time, requests
BASE = "https://www.kaggle.com/api/v1"; U = os.environ.get("KAGGLE_USER", "bugxdf")
BUDGET = float(os.environ.get("BUDGET", "95")); CH = 64 * 1024 * 1024
def H(): return {"Authorization": "Bearer " + os.environ["KAGGLE_API_TOKEN"], "Content-Type": "application/json"}
T0 = time.time()

def start(path):
    L = os.path.getsize(path)
    r = requests.post(BASE + "/blobs/upload", headers=H(), json={"type": "dataset", "name": os.path.basename(path),
                      "contentLength": L, "lastModifiedEpochSeconds": int(os.path.getmtime(path))}, timeout=120).json()
    return {"path": path, "size": L, "url": r["createUrl"], "token": r["token"], "done": False}

def committed(e):
    r = requests.put(e["url"], headers={"Content-Range": f"bytes */{e['size']}", "Content-Length": "0"}, timeout=60)
    if r.status_code in (200, 201): return e["size"]
    rg = r.headers.get("Range"); return int(rg.split("-")[1]) + 1 if rg else 0

def up(st, files):
    S = json.load(open(st)) if os.path.exists(st) else {"files": []}
    known = {e["path"] for e in S["files"]}
    for f in files:
        if f not in known: S["files"].append(start(f)); json.dump(S, open(st, "w"))
    for e in S["files"]:
        if e["done"]: continue
        off = committed(e)
        with open(e["path"], "rb") as fh:
            while off < e["size"]:
                if time.time() - T0 > BUDGET: json.dump(S, open(st, "w")); print(f"PARTIAL {os.path.basename(e['path'])} {off}/{e['size']}"); return
                fh.seek(off); data = fh.read(CH); end = off + len(data) - 1
                r = requests.put(e["url"], data=data, headers={"Content-Range": f"bytes {off}-{end}/{e['size']}",
                                 "Content-Length": str(len(data))}, timeout=300)
                if r.status_code in (200, 201): off = e["size"]; break
                if r.status_code == 308:
                    rg = r.headers.get("Range"); off = int(rg.split("-")[1]) + 1 if rg else 0
                else: print("put", r.status_code, r.text[:200]); time.sleep(3); off = committed(e)
        e["done"] = True; json.dump(S, open(st, "w")); print("DONE", os.path.basename(e["path"]))
    print("ALL_DONE")

def main():
    a = sys.argv[1:]
    if a[0] == "up": up(a[1], a[2:])
    else:
        S = json.load(open(a[1])); assert all(e["done"] for e in S["files"]), "uploads unfinished"
        files = [{"token": e["token"]} for e in S["files"]]
        if a[0] == "create":
            body = {"ownerSlug": U, "slug": a[2], "title": a[3], "isPrivate": True, "licenseName": "CC0-1.0", "files": files,
                    "subtitle": "", "description": "", "categoryIds": []}
            r = requests.post(BASE + "/datasets/create/new", headers=H(), json=body, timeout=600)
        else:
            r = requests.post(f"{BASE}/datasets/create/version/{U}/{a[2]}", headers=H(),
                              json={"versionNotes": a[3], "files": files, "deleteOldVersions": True, "categoryIds": []}, timeout=600)
        print(r.status_code, r.text[:400])

if __name__ == "__main__": main()
