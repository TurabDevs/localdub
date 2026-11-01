#!/usr/bin/env python3
"""localdub: one command from a video to a dubbed video, a QA report and subtitles.

  python3 localdub.py dub talk.mp4 --src en --to ur --work runs/talk
  python3 localdub.py dub sketch.webm --src hi --to en --work runs/sketch --chunk 600
  python3 localdub.py status --work runs/talk

Every step writes its outputs into --work and is skipped when they already exist, so the command can be re-run after a
crash, a timeout or a Kaggle queue and it continues where it stopped. Steps that run on Kaggle (stage 1, diarization,
Glimmer) push a kernel, record it in state.json and return "waiting"; re-run the same command to poll and download.
Use --wait N to keep polling for up to N seconds in one go.

Steps
  prep      ffmpeg: 44.1 kHz source audio + 640 px proxy video; long videos are cut at silences into chunks (--chunk s)
  stage1    Kaggle: Demucs vocals/background, Silero VAD, Omnilingual ASR, ECAPA (one kernel per chunk, run in parallel)
  merge     chunks -> one timeline (vocals.flac, background.flac, segments.json, asr.json, emb.npy)
  diarize   Kaggle: speaker changes inside VAD chunks (sliding-window ECAPA), re-ASR of the split chunks
  jobs      make_jobs.py --src/--tgt: voice clusters, lines, Glimmer cast + translation jobs
  glimmer   Kaggle: Muse Glimmer 30B (llama.cpp, CPU, offline) for cast + translation
  tts       Fish Audio voices per line (repeatable until all takes exist)
  mix       dub.py: fit, place, mix over the original music + effects, mux
  avqa      frame-by-frame AV-QA of the dub vs the source (report.html)
  subs      dub / source / bilingual SRT + VTT
  review    out/review.html: check every line in the browser, edit text, mark re-takes, export edits.json;
            then `--edits edits.json --from-step tts` re-voices just those lines and re-mixes
Needs: ffmpeg, numpy, soundfile, requests, msgpack; Kaggle token in KAGGLE_API_TOKEN; for avqa opencv + mediapipe.
"""
import argparse, glob, json, os, shutil, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__)); K = f"{HERE}/cloud/kaggle"; P = f"{HERE}/pipeline"
STEPS = ["prep", "stage1", "merge", "diarize", "jobs", "glimmer", "tts", "mix", "avqa", "subs", "review"]
GLIMMER_MODEL = "model:dangkhoa2016/meta-models-muse-glimmer-30b-gguf/gguf/default/1"

def sh(cmd, **kw):
    print("+", cmd if isinstance(cmd, str) else " ".join(cmd), flush=True)
    return subprocess.run(cmd, shell=isinstance(cmd, str), check=True, **kw)

def dur(p): return float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", p],
                                        capture_output=True, text=True).stdout)

class Run:
    def __init__(s, a):
        s.a = a; s.w = os.path.abspath(a.work); os.makedirs(s.w, exist_ok=True)
        s.sf = f"{s.w}/state.json"; s.st = json.load(open(s.sf)) if os.path.exists(s.sf) else {}
        s.name = s.st.get("name") or (f"ld-{a.name}" if a.name else None) or ("ld-" + "".join(c for c in os.path.basename(s.w).lower() if c.isalnum())[:24])
        s.st["name"] = s.name; s.save()
    def save(s): json.dump(s.st, open(s.sf + ".tmp", "w"), indent=1); os.replace(s.sf + ".tmp", s.sf)
    def kapi(s, *args, capture=True):
        r = subprocess.run([sys.executable, f"{K}/kapi.py", *args], capture_output=capture, text=True,
                           env=dict(os.environ, KAGGLE_GPU="0", KAGGLE_SHAPE=""))
        return (r.stdout or "") + (r.stderr or "")
    def kernel(s, key, script, sources, outdir):
        """push once, then poll; returns True when outputs are downloaded into outdir"""
        k = s.st.setdefault("kernels", {}).get(key)
        if k is None:
            out = s.kapi("push", key, script, ",".join(sources)); print(out[-300:])
            if '"error": ""' not in out and "versionNumber" not in out: raise SystemExit(f"push failed for {key}")
            s.st["kernels"][key] = {"pushed": time.time()}; s.save(); return False
        stat = s.kapi("status", key)
        if "complete" in stat:
            os.makedirs(outdir, exist_ok=True); s.kapi("output", key, outdir); return True
        if "error" in stat and "failureMessage': ''" not in stat: raise SystemExit(f"kernel {key} failed: {stat[-300:]}")
        print(f"{key}: {stat.strip()[-80:]}"); return False

    # ------------------------------------------------------------------ steps
    def prep(s):
        if os.path.exists(f"{s.w}/in/chunks.json"): return True
        os.makedirs(f"{s.w}/in", exist_ok=True); v = os.path.abspath(s.a.video); D = dur(v)
        sh(["ffmpeg", "-v", "error", "-y", "-i", v, "-vn", "-ac", "2", "-ar", "44100", "-c:a", "flac", f"{s.w}/in/src.flac"])
        sh(["ffmpeg", "-v", "error", "-y", "-i", v, "-an", "-vf", "scale=640:-2", "-c:v", "libx264", "-preset", "veryfast", "-crf", "26", f"{s.w}/in/proxy.mp4"])
        cuts = [0.0]
        if D > s.a.chunk:   # cut near each chunk boundary at the quietest 0.5 s (silencedetect over a +-20 s window)
            t = s.a.chunk
            while t < D - 30:
                r = subprocess.run(["ffmpeg", "-v", "info", "-ss", str(t - 20), "-t", "40", "-i", f"{s.w}/in/src.flac", "-af",
                                    "silencedetect=n=-35dB:d=0.4", "-f", "null", "-"], capture_output=True, text=True).stderr
                ss = [float(x.split("silence_start: ")[1].split()[0]) + t - 20 for x in r.splitlines() if "silence_start" in x]
                c = min(ss, key=lambda x: abs(x - t)) + 0.2 if ss else t
                cuts.append(round(c, 3)); t = c + s.a.chunk
        cuts.append(D); ch = []
        for i in range(len(cuts) - 1):
            d = f"{s.w}/in/c{i}"; os.makedirs(d, exist_ok=True)
            sh(["ffmpeg", "-v", "error", "-y", "-ss", str(cuts[i]), "-t", str(cuts[i + 1] - cuts[i]), "-i", f"{s.w}/in/src.flac", "-c:a", "flac", f"{d}/src.flac"])
            ch.append({"i": i, "start": cuts[i], "end": cuts[i + 1]})
        json.dump(ch, open(f"{s.w}/in/chunks.json", "w")); return True
    def stage1(s):
        ch = json.load(open(f"{s.w}/in/chunks.json")); done = True
        for c in ch:
            out = f"{s.w}/s1/c{c['i']}"
            if os.path.exists(f"{out}/segments.json"): continue
            ds = f"{s.name}-c{c['i']}"
            if ds not in s.st.setdefault("datasets", {}):
                print(s.kapi("ds-create", ds, f"localdub {s.name} chunk {c['i']}", f"{s.w}/in/c{c['i']}/src.flac")[-200:])
                s.st["datasets"][ds] = time.time(); s.save(); done = False; continue
            if "ready" not in s.kapi("ds-status", ds): done = False; continue
            done &= s.kernel(f"{s.name}-s1-c{c['i']}", f"{K}/stage1_listen.py", ["localdub-assets", "localdub-fix", ds], out)
        return done
    def merge(s):
        o = f"{s.w}/s1/all"
        if os.path.exists(f"{o}/segments.json"): return True
        import numpy as np, soundfile as sf
        os.makedirs(o, exist_ok=True); ch = json.load(open(f"{s.w}/in/chunks.json")); seg, asr, emb, V, B = [], {}, [], [], []
        for c in ch:
            d = f"{s.w}/s1/c{c['i']}"; V.append(sf.read(f"{d}/vocals.flac", dtype="float32")[0]); B.append(sf.read(f"{d}/background.flac", dtype="float32")[0])
            a = json.load(open(f"{d}/asr.json")); a = next(iter(a.values())); e = np.load(f"{d}/emb.npy") if os.path.exists(f"{d}/emb.npy") else None
            for k, x in enumerate(json.load(open(f"{d}/segments.json"))):
                n = len(seg); seg.append({"i": n, "start": round(x["start"] + c["start"], 3), "end": round(x["end"] + c["start"], 3)})
                asr[f"{n:04d}"] = a.get(f"{x['i']:04d}", ""); emb.append(e[k] if e is not None else np.zeros(192, np.float32))
        sf.write(f"{o}/vocals.flac", np.concatenate(V), 44100, subtype="PCM_24"); sf.write(f"{o}/background.flac", np.concatenate(B), 44100, subtype="PCM_24")
        json.dump(seg, open(f"{o}/segments.json", "w"), indent=0); json.dump({"omniASR_CTC_1B_v2": asr}, open(f"{o}/asr.json", "w"), ensure_ascii=False)
        np.save(f"{o}/emb.npy", np.stack(emb)); return True
    def diarize(s):
        o = f"{s.w}/s1/diar"
        if s.a.no_diarize or os.path.exists(f"{o}/segments.json"): return True
        ds = f"{s.name}-s1"
        if ds not in s.st.setdefault("datasets", {}):
            print(s.kapi("ds-create", ds, f"localdub {s.name} stage1", f"{s.w}/s1/all/vocals.flac", f"{s.w}/s1/all/segments.json")[-200:])
            s.st["datasets"][ds] = time.time(); s.save(); return False
        if "ready" not in s.kapi("ds-status", ds): return False
        ok = s.kernel(f"{s.name}-diar", f"{K}/stage1_diarize.py", ["localdub-assets", "localdub-fix", ds], o)
        if ok:
            for f in ("vocals.flac", "background.flac"): shutil.copy(f"{s.w}/s1/all/{f}", f"{o}/{f}")
        return ok
    def s1dir(s): return f"{s.w}/s1/all" if s.a.no_diarize else f"{s.w}/s1/diar"
    def jobs(s):
        if os.path.exists(f"{s.w}/s2/jobs.json"): return True
        sh([sys.executable, f"{P}/make_jobs.py", s.s1dir(), f"{s.w}/s2", "--src", s.a.src, "--tgt", s.a.to] + (["--context", s.a.context] if s.a.context else []))
        J = json.load(open(f"{s.w}/s2/jobs_cast.json")) + json.load(open(f"{s.w}/s2/jobs_tr.json"))
        json.dump(J, open(f"{s.w}/s2/jobs.json", "w"), ensure_ascii=False); return True
    def glimmer(s):
        o = f"{s.w}/s2/glimmer"
        if os.path.exists(f"{o}/results.json"): return True
        ds = f"{s.name}-jobs"
        if ds not in s.st.setdefault("datasets", {}):
            out = s.kapi("ds-create", ds, f"localdub {s.name} glimmer jobs", f"{s.w}/s2/jobs.json", f"{s.w}/in/proxy.mp4"); print(out[-200:])
            if '"status":"Ok"' not in out.replace(" ", ""): raise SystemExit(f"dataset create failed for {ds}")
            s.st["datasets"][ds] = time.time(); s.save(); return False
        st = s.kapi("ds-status", ds)
        if "403" in st or "404" in st:   # create never went through: forget it and create again next run (was an endless WAITING loop)
            s.st["datasets"].pop(ds, None); s.save(); print(f"dataset {ds} missing ({st.strip()[-80:]}), will re-create"); return False
        if "ready" not in st: return False
        return s.kernel(f"{s.name}-glimmer", f"{K}/stage2_glimmer.py", ["localdub-bin", ds, GLIMMER_MODEL], o)
    def dubargs(s):
        return [f"{P}/dub.py", s.s1dir(), f"{s.w}/s2", f"{s.w}/s2/glimmer/results.json", os.path.abspath(s.a.video), f"{s.w}/out/dub.{s.a.ext}"]
    def env(s):
        e = dict(os.environ, SRC_LANG=s.a.src, TTS_SCRIPT=s.a.tts_script)
        if s.a.edits: e["EDITS"] = os.path.abspath(s.a.edits)
        return e
    def tts(s):
        if s.a.edits and s.st.get("edits_applied") != os.path.getmtime(s.a.edits):   # new edits: redo tts/mix/avqa/subs/review
            s.st["tts_done"] = False; s.st["edits_applied"] = os.path.getmtime(s.a.edits); s.save()
            for f in (f"out/dub.{s.a.ext}", "out/qa.json", "out/avqa/report.html", "out/subs.bi.srt"):
                if os.path.exists(f"{s.w}/{f}"): os.remove(f"{s.w}/{f}")
        if s.st.get("tts_done"): return True
        os.makedirs(f"{s.w}/out", exist_ok=True)
        r = subprocess.run([sys.executable, *s.dubargs()], env=dict(s.env(), PHASE="tts", BUDGET="90"), capture_output=True, text=True)
        print(r.stdout[-300:], r.stderr[-300:])
        if "TTS_ALL_DONE" in r.stdout: s.st["tts_done"] = True; s.save(); return True
        return False
    def mix(s):
        if os.path.exists(f"{s.w}/out/dub.{s.a.ext}") and os.path.exists(f"{s.w}/out/qa.json"): return True
        sh([sys.executable, *s.dubargs()], env=s.env()); return True
    def avqa(s):
        if os.path.exists(f"{s.w}/out/avqa/report.html"): return True
        fj = f"{s.w}/out/faces.json"; D = dur(os.path.abspath(s.a.video)); f = json.load(open(fj)) if os.path.exists(fj) else {}
        start = f.get("done_until", 0.0)
        if start < D - 0.2:   # visual pass in 60 s pieces (resumable)
            sh([sys.executable, f"{P}/avqa_faces.py", os.path.abspath(s.a.video), fj, "--start", str(start), "--end", str(min(D, start + 60)),
                "--models", s.a.models]); return False
        sh([sys.executable, f"{P}/avqa.py", "pair", "--src", os.path.abspath(s.a.video), "--dub", f"{s.w}/out/dub.{s.a.ext}", "--faces", fj,
            "--vocals", f"{s.s1dir()}/vocals.flac", "--background", f"{s.s1dir()}/background.flac", "--segments", f"{s.s1dir()}/segments.json",
            "--dialogue", f"{s.w}/out/dubwork/dialogue_en.wav", "--lines", f"{s.w}/s2/lines.json", "--qa", f"{s.w}/out/qa.json", "--out", f"{s.w}/out/avqa"])
        return True
    def review(s):
        if os.path.exists(f"{s.w}/out/review.html") and not s.a.edits: return True
        sh([sys.executable, f"{P}/make_review.py", f"{s.w}/out/qa.json", f"{s.w}/s2/lines.json", f"{s.w}/out/dub.{s.a.ext}", f"{s.w}/out/review.html",
            "--src-video", os.path.abspath(s.a.video), "--avqa", f"{s.w}/out/avqa/avqa.json", "--title", os.path.basename(s.a.video)]); return True
    def subs(s):
        if os.path.exists(f"{s.w}/out/subs.bi.srt"): return True
        sh([sys.executable, f"{P}/subs.py", f"{s.w}/out/qa.json", f"{s.w}/s2/lines.json", f"{s.w}/out/subs"]); return True

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["dub", "status"]); ap.add_argument("video", nargs="?")
    ap.add_argument("--src", default="hi"); ap.add_argument("--to", default="en"); ap.add_argument("--work", required=True)
    ap.add_argument("--context"); ap.add_argument("--chunk", type=float, default=600.0); ap.add_argument("--wait", type=float, default=0)
    ap.add_argument("--from-step", choices=STEPS); ap.add_argument("--until", choices=STEPS); ap.add_argument("--no-diarize", action="store_true")
    ap.add_argument("--tts-script", default="native", choices=["native", "roman"]); ap.add_argument("--ext", default="mp4")
    ap.add_argument("--models", default=f"{HERE}/models"); ap.add_argument("--name", help="short run name used for Kaggle datasets/kernels")
    ap.add_argument("--edits", help="edits.json exported from out/review.html")
    a = ap.parse_args(); r = Run(a)
    if a.cmd == "status": print(json.dumps(r.st, indent=1)); return
    steps = STEPS[STEPS.index(a.from_step) if a.from_step else 0: STEPS.index(a.until) + 1 if a.until else None]
    t0 = time.time()
    while True:
        for st in steps:
            if not getattr(r, st)():
                if time.time() - t0 < a.wait: time.sleep(30); break
                print(f"WAITING at step '{st}' (re-run the same command to continue)"); return
        else:
            if a.until and a.until != STEPS[-1]: print(f"done through step '{a.until}'"); return
            print("DONE:", f"{r.w}/out/dub.{a.ext}", f"{r.w}/out/avqa/report.html", f"{r.w}/out/subs.bi.srt"); return

if __name__ == "__main__":
    main()
