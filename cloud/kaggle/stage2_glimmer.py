"""LocalDub stage 2 - Meta Muse Glimmer 30B (open weights, Apache-2.0) served by llama.cpp inside a Kaggle kernel.

Runs fully OFFLINE (no internet needed in the kernel):
  * weights  : Kaggle Model source  dangkhoa2016/meta-models-muse-glimmer-30b-gguf/gguf/default/1
               (byte-identical mirror of huggingface.co/meta-models/Muse-Glimmer-30B-GGUF; sha256 checked below)
  * llama.cpp: official release tarball (b11981, ubuntu x64 CPU build with all CPU variants) from dataset bugxdf/localdub-bin
  * jobs     : jobs.json from dataset bugxdf/localdub-jobs (+ proxy.mp4 for frames)
Works on the CPU machine (4 vCPU / 30 GB RAM, ~1.5 tok/s) and on GPU machines if a CUDA build is attached.

A job: {"id": str, "messages": [...OpenAI chat messages...], "max_tokens": int, "temperature": float, "json": bool,
        "reasoning": "low"|"medium"|"high"}
Message content parts may include {"type": "frame", "t": seconds}: that frame of proxy.mp4 is extracted here and sent to
Glimmer's perception encoder as an image (this is how LocalDub "looks" at who is on screen).
Env (via JOB_SHARD in jobs file name or SHARD below): run only jobs where index % NSHARD == SHARD.
Writes /kaggle/working/results.json incrementally (resumable by re-attaching an earlier results.json).
"""
import base64, glob, hashlib, json, os, subprocess, sys, time, urllib.request

W = "/kaggle/working"; T0 = time.time()
CFG = {"SHARD": 0, "NSHARD": 1, "CTX": 16384, "VERIFY": 0}   # patched per push by the launcher
def log(*a): print(f"[{time.time()-T0:7.1f}s]", *a, flush=True)
def sh(c, check=True):
    log("+", c[:250]); r = subprocess.run(c, shell=True)
    if check and r.returncode: raise SystemExit(f"failed: {c[:120]}")
    return r.returncode
def find(pat): return sorted(glob.glob(f"/kaggle/input/**/{pat}", recursive=True))

GGUF = (find("Muse-Glimmer-30B-KQuant-17GB-Q4_K_M.gguf") or [None])[0]
MMPROJ = (find("mmproj-Muse-Glimmer-30B-Q4_K_M.gguf") or [None])[0]
JOBS = (find("jobs*.json") or [None])[0]
PROXY = (find("proxy.mp4") or [None])[0]
PREV = (find("results.json") or [None])[0]
log("gguf", GGUF, "mmproj", MMPROJ, "jobs", JOBS, "proxy", PROXY, "prev", PREV, "cfg", CFG)
sh("nproc; free -g; nvidia-smi --query-gpu=name,memory.total --format=csv 2>/dev/null || echo NO_GPU", check=False)
GPU = subprocess.run("nvidia-smi", shell=True, capture_output=True).returncode == 0

SHA = {"Muse-Glimmer-30B-KQuant-17GB-Q4_K_M.gguf": "4cc57c0f51040a226e5a72cc47b7613f7772950e460a665f7083de89f183f60e",
       "mmproj-Muse-Glimmer-30B-Q4_K_M.gguf": "f48b452316f9b213758e8659444029b961a24a07f99a1abb2a9f88b06f7c00c6"}
if CFG["VERIFY"]:
    for p in (MMPROJ, GGUF):
        h = hashlib.sha256()
        with open(p, "rb") as f:
            for b in iter(lambda: f.read(1 << 24), b""): h.update(b)
        ok = h.hexdigest() == SHA[os.path.basename(p)]
        log("sha256", os.path.basename(p), "OK (matches Hugging Face meta-models)" if ok else "MISMATCH " + h.hexdigest())
        if not ok: raise SystemExit("weights do not match the official Meta release")

# Kaggle auto-extracts uploaded .tar.gz files, so accept either the tarball or the extracted directory
srvs = [b for b in find("llama-server") if ("cuda" in b) == GPU] or find("llama-server")
tgz = [t for t in find("llama-*-bin-ubuntu-*.tar.gz") if ("cuda" in t) == GPU] or find("llama-*-bin-ubuntu-x64.tar.gz")
if srvs:
    sh(f"mkdir -p /tmp/ll/x && cp -r {os.path.dirname(srvs[0])}/. /tmp/ll/x/ && chmod +x /tmp/ll/x/*")
    # Kaggle's auto-extraction turns the tarball's .so symlinks into empty files: re-create them
    for f in glob.glob("/tmp/ll/x/*.so*"):
        if os.path.getsize(f) < 1024:
            real = sorted([g for g in glob.glob(f + "*") if g != f and os.path.getsize(g) >= 1024], key=len)
            if real: os.remove(f); os.symlink(os.path.basename(real[-1]), f); log("relinked", os.path.basename(f), "->", os.path.basename(real[-1]))
else: sh(f"mkdir -p /tmp/ll && tar xzf {tgz[0]} -C /tmp/ll")
for t in (find("cudart-llama-*.tar.gz") if GPU else []): sh(f"tar xzf {t} -C /tmp/ll/*/", check=False)
BIN = glob.glob("/tmp/ll/*/llama-server")[0]; LD = os.path.dirname(BIN)
sh(f"LD_LIBRARY_PATH={LD} {BIN} --version", check=False)

ncpu = os.cpu_count() or 4
cmd = (f"LD_LIBRARY_PATH={LD} {BIN} -m {GGUF} " + (f"--mmproj {MMPROJ} " if MMPROJ else "") +
       f"-c {CFG['CTX']} -np 1 -t {ncpu} --host 127.0.0.1 --port 8080 --jinja " + ("-ngl 999 " if GPU else "") +
       "> /tmp/server.log 2>&1")
srv = subprocess.Popen(cmd, shell=True)
for i in range(1800):
    try:
        if json.load(urllib.request.urlopen("http://127.0.0.1:8080/health", timeout=5)).get("status") == "ok": break
    except Exception: pass
    if srv.poll() is not None: sh("tail -60 /tmp/server.log", check=False); raise SystemExit("server died")
    time.sleep(2)
log("server up"); sh("grep -i -E 'model params|arch|CPU_REPACK|CUDA|offload|n_ctx_slot|multimodal' /tmp/server.log | head -20", check=False)

os.makedirs("/tmp/frames", exist_ok=True)
def frame(t):
    p = f"/tmp/frames/{t:.2f}.jpg"
    if not os.path.exists(p):
        subprocess.run(f"ffmpeg -v error -y -ss {t} -i {PROXY} -frames:v 1 -vf scale=448:-2 -q:v 3 {p}", shell=True)
    return "data:image/jpeg;base64," + base64.b64encode(open(p, "rb").read()).decode()

def fix(msgs):
    out = []
    for m in msgs:
        c = m["content"]
        if isinstance(c, list):
            c = [{"type": "image_url", "image_url": {"url": frame(float(p["t"]))}} if p.get("type") == "frame" else p for p in c]
        out.append({"role": m["role"], "content": c})
    return out

jobs = json.load(open(JOBS)); res = json.load(open(PREV)) if PREV else {}
jobs = [j for k, j in enumerate(jobs) if k % CFG["NSHARD"] == CFG["SHARD"]]
RP = f"{W}/results.json"; json.dump(res, open(RP, "w"), ensure_ascii=False, indent=1)
for j in jobs:
    if j["id"] in res and res[j["id"]].get("content"): continue
    t = time.time()
    body = {"messages": fix(j["messages"]), "max_tokens": j.get("max_tokens", 2048), "temperature": j.get("temperature", 1.0),
            "top_p": 0.95, "top_k": 64, "chat_template_kwargs": {"reasoning_strength": j.get("reasoning", "low")}}
    if j.get("json"): body["response_format"] = {"type": "json_object"}
    try:
        req = urllib.request.Request("http://127.0.0.1:8080/v1/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        r = json.load(urllib.request.urlopen(req, timeout=7200))
        msg = r["choices"][0]["message"]
        res[j["id"]] = {"content": msg.get("content"), "reasoning": msg.get("reasoning_content"), "usage": r.get("usage"),
                        "timings": r.get("timings"), "finish": r["choices"][0].get("finish_reason"), "secs": round(time.time() - t, 1)}
    except Exception as e:
        res[j["id"]] = {"error": repr(e)[:500]}
    log(j["id"], res[j["id"]].get("secs"), (res[j["id"]].get("usage") or {}), str(res[j["id"]].get("content") or res[j["id"]].get("error"))[:300])
    json.dump(res, open(RP, "w"), ensure_ascii=False, indent=1)
srv.terminate(); sh("tail -5 /tmp/server.log", check=False)
log("done", len(res), "jobs")
