#!/usr/bin/env python3
"""LocalDub AV-QA: frame-by-frame audio/video check of a dubbed video against its source.

usage:
  avqa.py pair --src VIDEO --dub VIDEO --faces F1.json[,F2.json...] --out DIR
              [--vocals S1/vocals.flac --background S1/background.flac --segments S1/segments.json
               --dialogue DUBWORK/dialogue_en.wav --lines JOBS/lines.json --qa qa.json]
  avqa.py solo --video VIDEO --faces F.json[,...] --out DIR          (a single finished video, e.g. the demo cut)

The visual pass (faces + mouth activity) comes from avqa_faces.py and is computed once per source video.
Everything runs on a 50 ms grid (20 frames per second of audio, video at the faces.json rate):
  source speech   Silero VAD segments from stage 1 (or an energy VAD on the Demucs vocals)
  source vocals   energy VAD on the Demucs vocals: also catches laughs, breaths and shouts that are not dubbing lines
  dub speech      energy VAD on the dub dialogue stem (or on dub minus source background when no stem is given)
  face talking    mouth-region motion + inner-lip gap of every tracked face; the threshold is calibrated against the
                  source speech track, and its AUC is reported so you can see how much to trust the visual flags
  music/effects   Demucs background of the source vs. the dub's bed (dub mix minus its dialogue stem)
  loudness        EBU R128 integrated loudness and true peak (ffmpeg), sample clipping, per-line level jumps
  sync            cross-correlation of the source and dub beds per 30 s window (drift), and mouth activity vs speech
Flags (each with start/end, the dubbing line it overlaps, and a thumbnail): talking face but dub silent, source
speech not dubbed, non-line vocals dropped, dub speaking over source silence, music/effects ducked or missing,
clipping, level jumps, drift. Output: DIR/avqa.json, DIR/report.html, DIR/timeline.png, DIR/thumbs/*.jpg
"""
import argparse, html, json, os, re, subprocess, sys
import numpy as np

HOP = 0.05   # seconds per analysis frame
SR = 16000

def dur_of(p):
    return float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", p],
                                capture_output=True, text=True).stdout.strip())

def load(p, sr=SR, ch=1):
    # decode at the file's own channel count and average in numpy: ffmpeg's stereo<->mono conversions apply a -3 dB
    # pan law (mono->stereo 0.707, stereo->mono sums at 0.707), which would skew a mono stem vs a stereo one by 3 dB
    nch = int(subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries", "stream=channels", "-of", "csv=p=0", p],
                             capture_output=True, text=True).stdout.strip() or 1)
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", p, "-vn", "-ac", str(nch), "-ar", str(sr), "-f", "f32le", "-"],
                         capture_output=True, check=True).stdout
    x = np.frombuffer(raw, np.float32).reshape(-1, nch)
    if ch > 1: return x if nch > 1 else np.repeat(x, 2, 1)
    return x.mean(1)

def frames_db(x, n):
    h = int(HOP * SR); x = np.pad(x, (0, max(0, n * h - len(x))))[: n * h].reshape(n, h)
    return 10 * np.log10(np.mean(x.astype(np.float64) ** 2, 1) + 1e-10)

def smooth_mask(m, close=0.25, min_on=0.15):
    m = m.copy(); g = int(round(close / HOP)); k = int(round(min_on / HOP))
    # close short gaps
    idx = np.flatnonzero(m)
    if len(idx):
        gaps = np.flatnonzero(np.diff(idx) > 1)
        for i in gaps:
            a, b = idx[i], idx[i + 1]
            if b - a - 1 <= g: m[a:b] = True
    # drop short islands
    for a, b in runs(m):
        if b - a < k: m[a:b] = False
    return m

def runs(m):
    d = np.diff(np.concatenate([[0], m.astype(np.int8), [0]])); return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))

def energy_vad(db, rel=38.0, floor=-48.0):
    top = np.percentile(db, 99); noise = np.percentile(db, 15)
    thr = max(floor, noise + 10.0, top - rel)
    return smooth_mask(db > thr), float(thr)

def seg_mask(segs, n):
    m = np.zeros(n, bool)
    for s in segs: m[int(s["start"] / HOP): int(np.ceil(s["end"] / HOP))] = True
    return m

def ebur128(p):
    r = subprocess.run(["ffmpeg", "-nostats", "-v", "info", "-i", p, "-vn", "-af", "ebur128=peak=true", "-f", "null", "-"],
                       capture_output=True, text=True).stderr
    tail = r[r.rfind("Summary:"):]
    I = re.search(r"I:\s+(-?[\d.]+) LUFS", tail); P = re.search(r"Peak:\s+(-?[\d.]+) dBFS", tail); LRA = re.search(r"LRA:\s+([\d.]+) LU", tail)
    return {"integrated_lufs": float(I.group(1)) if I else None, "true_peak_dbtp": float(P.group(1)) if P else None,
            "lra_lu": float(LRA.group(1)) if LRA else None}

def faces_grid(paths, n):
    """per analysis frame: list of faces (nearest video frame) + per-frame talking activity"""
    fr = {}; gd = {}; fps = None
    for p in paths.split(","):
        d = json.load(open(p)); fps = d["fps"]; fr.update(d["frames"]); gd.update(d.get("gdiff", {}))
    ts = np.array(sorted(float(k) for k in fr)); keys = [f"{t:.3f}" for t in ts]
    # per-track series of motion and lip gap -> activity = smoothed motion (z by track) + rolling std of lip gap
    tracks = {}
    for i, k in enumerate(keys):
        for f in fr[k]:
            tracks.setdefault(f["track"], []).append((i, f))
    act_v = np.zeros(len(ts)); face_v = np.zeros(len(ts), bool)
    win = max(1, int(round(0.5 * fps)))
    for tid, items in tracks.items():
        idx = np.array([i for i, _ in items])
        mot = np.array([np.nan if f["motion"] is None else f["motion"] for _, f in items], float)
        opn = np.array([np.nan if f["open"] is None else f["open"] for _, f in items], float)
        a = np.zeros(len(items))
        for j in range(len(items)):
            lo, hi = max(0, j - win), min(len(items), j + win + 1)
            mm = mot[lo:hi]; oo = opn[lo:hi]
            m1 = np.nanmean(mm) if np.isfinite(mm).any() else 0.0
            o1 = np.nanstd(oo) * 10 if np.isfinite(oo).sum() >= 3 else 0.0
            a[j] = min(m1, 3.0) + o1
        act_v[idx] = np.maximum(act_v[idx], a); face_v[idx] = True
    # map to analysis grid
    g = np.clip(np.round(np.arange(n) * HOP * fps).astype(int), 0, len(ts) - 1) if len(ts) else np.zeros(n, int)
    grid_t = np.arange(n) * HOP
    valid = np.abs(ts[g] - grid_t) <= 1.0 / fps + 1e-6 if len(ts) else np.zeros(n, bool)
    return {"t": ts, "keys": keys, "frames": fr, "act": np.where(valid, act_v[g], 0.0), "face": valid & face_v[g], "fps": fps}

def auc(score, label):
    s = score[np.isfinite(score)]; l = label[np.isfinite(score)]
    if l.sum() == 0 or (~l).sum() == 0: return None
    order = np.argsort(s); r = np.empty(len(s)); r[order] = np.arange(1, len(s) + 1)
    return float((r[l].sum() - l.sum() * (l.sum() + 1) / 2) / (l.sum() * (~l).sum()))

def best_threshold(score, label):
    best = (0.0, None)
    for q in np.linspace(5, 95, 37):
        th = np.percentile(score, q); p = score > th
        tpr = (p & label).sum() / max(1, label.sum()); tnr = (~p & ~label).sum() / max(1, (~label).sum())
        if (tpr + tnr) / 2 > best[0]: best = ((tpr + tnr) / 2, float(th))
    return best[1], best[0]

def xcorr_lag(a, b, maxlag):
    a = (a - a.mean()) / (a.std() + 1e-9); b = (b - b.mean()) / (b.std() + 1e-9); best = (-9, 0)
    for L in range(-maxlag, maxlag + 1):
        if L >= 0: c = np.mean(a[L:] * b[: len(b) - L])
        else: c = np.mean(a[: L] * b[-L:])
        if c > best[0]: best = (float(c), L)
    return best[1], best[0]

def events(mask, kind, min_dur, sev, why, close=0.3):
    m = smooth_mask(mask, close=close, min_on=min_dur); return [{"kind": kind, "start": round(a * HOP, 2), "end": round(b * HOP, 2),
                                                                 "dur": round((b - a) * HOP, 2), "severity": sev, "why": why} for a, b in runs(m)]

def line_at(lines, s, e):
    best = None; ov = 0
    for L in lines:
        o = min(e, L["end"]) - max(s, L["start"])
        if o > ov: ov, best = o, L["id"]
    return best

def thumbs(video, evs, faces, outdir, maxn=60):
    os.makedirs(f"{outdir}/thumbs", exist_ok=True)
    try:
        import cv2
    except Exception:
        cv2 = None
    for k, e in enumerate(evs[:maxn]):
        t = (e["start"] + e["end"]) / 2; fn = f"thumbs/{k:03d}_{e['kind']}_{t:.1f}.jpg"
        subprocess.run(f"ffmpeg -v error -y -ss {t:.2f} -i '{video}' -frames:v 1 -vf scale=320:-2 '{outdir}/{fn}'", shell=True)
        if cv2 is not None and faces is not None and len(faces["t"]) and os.path.exists(f"{outdir}/{fn}"):
            img = cv2.imread(f"{outdir}/{fn}"); H, W = img.shape[:2]
            i = int(np.argmin(np.abs(faces["t"] - t)))
            for f in faces["frames"][faces["keys"][i]]:
                x, y, w, h = f["box"]; cv2.rectangle(img, (int(x * W), int(y * H)), (int((x + w) * W), int((y + h) * H)), (0, 200, 255), 2)
            cv2.imwrite(f"{outdir}/{fn}", img)
        e["thumb"] = fn

def timeline_png(path, tracks, evs, n):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    t = np.arange(n) * HOP; fig, ax = plt.subplots(len(tracks) + 1, 1, figsize=(16, 1.1 * (len(tracks) + 1)), sharex=True)
    for a, (name, y, kind) in zip(ax, tracks):
        if kind == "mask": a.fill_between(t, 0, y.astype(float), step="mid", color="#2a7")
        else: a.plot(t, y, lw=0.6, color="#246")
        a.set_ylabel(name, rotation=0, ha="right", fontsize=8); a.set_yticks([])
    col = {"high": "#d22", "medium": "#e90", "low": "#999"}
    for e in evs: ax[-1].axvspan(e["start"], e["end"], color=col[e["severity"]], alpha=0.8)
    ax[-1].set_ylabel("flags", rotation=0, ha="right", fontsize=8); ax[-1].set_yticks([]); ax[-1].set_xlabel("seconds")
    plt.tight_layout(); plt.savefig(path, dpi=80); plt.close(fig)

def write_html(out, rep):
    s = rep["summary"]; rows = []
    for e in rep["events"]:
        img = f'<img src="{e["thumb"]}" width="200">' if e.get("thumb") else ""
        rows.append(f'<tr class="{e["severity"]}"><td>{e["start"]:.2f}-{e["end"]:.2f}</td><td>{e["kind"]}</td><td>{e["severity"]}</td>'
                    f'<td>{e.get("line") or ""}</td><td>{html.escape(e["why"])}</td><td>{html.escape(json.dumps(e.get("detail", {})))}</td><td>{img}</td></tr>')
    summ = "".join(f"<tr><td>{html.escape(k)}</td><td>{html.escape(json.dumps(v))}</td></tr>" for k, v in s.items())
    doc = f"""<!doctype html><meta charset="utf-8"><title>LocalDub AV-QA {html.escape(rep['name'])}</title>
<style>body{{font:14px system-ui;margin:24px;max-width:1300px}}table{{border-collapse:collapse;width:100%}}td,th{{border:1px solid #ddd;padding:4px;vertical-align:top}}
tr.high td:nth-child(3){{color:#c00;font-weight:600}}tr.medium td:nth-child(3){{color:#c70}}img{{display:block}}</style>
<h1>AV-QA: {html.escape(rep['name'])}</h1><p>{html.escape(rep['mode'])} mode, {rep['duration']:.1f} s, analysis every {HOP*1000:.0f} ms, faces at {rep.get('faces_fps')} fps.</p>
<img src="timeline.png" style="width:100%"><h2>Summary</h2><table>{summ}</table>
<h2>Flags ({len(rep['events'])})</h2><table><tr><th>time (s)</th><th>kind</th><th>severity</th><th>line</th><th>why</th><th>detail</th><th>frame</th></tr>{''.join(rows)}</table>"""
    open(f"{out}/report.html", "w").write(doc)

def pair(a):
    D = dur_of(a.dub); n = int(D / HOP)
    dub = load(a.dub); src_bg = load(a.background) if a.background else None
    voc = load(a.vocals) if a.vocals else None; dial = load(a.dialogue) if a.dialogue else None
    lines = json.load(open(a.lines)) if a.lines else []
    qa = {q["id"]: q for q in json.load(open(a.qa))["lines"]} if a.qa else {}
    fit = lambda x: np.pad(x, (0, max(0, len(dub) - len(x))))[: len(dub)]
    src_bg, voc, dial = [fit(x) if x is not None else None for x in (src_bg, voc, dial)]
    # dub bed = dub mix - k * dialogue, k from least squares on (bg + dialogue) (covers peak normalisation + codec)
    if dial is not None and src_bg is not None:
        ref = src_bg + dial; k = float(np.dot(dub, ref) / (np.dot(ref, ref) + 1e-9)); bed = dub - k * dial
    elif src_bg is not None:
        k = float(np.dot(dub, src_bg) / (np.dot(src_bg, src_bg) + 1e-9)); bed = None
    else:
        k = None; bed = None
    dial_est = dial if dial is not None else (dub - k * src_bg if src_bg is not None else dub)
    db_dub = frames_db(dub, n); db_dial = frames_db(dial_est, n)
    db_voc = frames_db(voc, n) if voc is not None else None
    db_bg = frames_db(src_bg, n) if src_bg is not None else None; db_bed = frames_db(bed, n) if bed is not None else None
    dub_sp, thr_d = energy_vad(db_dial)
    src_voc_act, thr_v = energy_vad(db_voc) if db_voc is not None else (None, None)
    src_sp = seg_mask(json.load(open(a.segments)), n) if a.segments else src_voc_act
    F = faces_grid(a.faces, n) if a.faces else None
    rep = {"name": os.path.basename(a.dub), "mode": "pair", "duration": D, "faces_fps": F["fps"] if F else None, "events": []}
    S = {}
    S["speech_seconds_source"] = round(src_sp.sum() * HOP, 1); S["speech_seconds_dub"] = round(dub_sp.sum() * HOP, 1)
    S["source_speech_covered_by_dub_pct"] = round(100 * (src_sp & dub_sp).sum() / max(1, src_sp.sum()), 1)
    S["dub_speech_inside_source_speech_pct"] = round(100 * (src_sp & dub_sp).sum() / max(1, dub_sp.sum()), 1)
    evs = []
    talking = None
    if F is not None:
        has = F["face"]; act = F["act"]
        A = auc(np.where(has, act, np.nan), src_sp); th, bacc = best_threshold(act[has], src_sp[has]) if has.any() else (None, None)
        talking = has & (act > th) if th is not None else np.zeros(n, bool)
        S["face_on_screen_pct"] = round(100 * has.mean(), 1)
        S["mouth_activity_vs_source_speech_auc"] = round(A, 3) if A is not None else None
        S["mouth_threshold_balanced_acc"] = round(bacc, 3) if bacc else None
        conf = "high" if (A or 0) >= 0.7 else ("medium" if (A or 0) >= 0.6 else "low")
        S["visual_flag_confidence"] = conf
        # lip sync: mouth activity vs speech envelope (only frames with a face)
        if has.sum() > 200:
            env_s = np.where(has, np.clip(db_voc if db_voc is not None else db_dub, -60, 0), -60.0)
            env_d = np.where(has, np.clip(db_dial, -60, 0), -60.0); m = np.where(has, act, 0.0)
            ls, cs = xcorr_lag(m, env_s, 12); ld, cd = xcorr_lag(m, env_d, 12)
            S["mouth_vs_source_speech_lag_ms"] = int(ls * HOP * 1000); S["mouth_vs_source_speech_corr"] = round(cs, 3)
            S["mouth_vs_dub_speech_lag_ms"] = int(ld * HOP * 1000); S["mouth_vs_dub_speech_corr"] = round(cd, 3)
        evs += events(talking & src_sp & ~dub_sp, "talking_face_dub_silent", 0.5, "high",
                      "a face on screen is moving its mouth while the original has speech, but the dub is silent")
    evs += events(src_sp & ~dub_sp & ~(talking if talking is not None else np.zeros(n, bool)), "speech_not_dubbed", 0.6, "medium",
                  "the original has speech here but the dub track is silent")
    if src_voc_act is not None:
        evs += events(src_voc_act & ~src_sp & ~dub_sp, "vocals_dropped", 0.6, "low",
                      "the original has voice sound outside any dubbing line (laugh, breath, shout, crowd) that is missing in the dub")
    face_t = talking if talking is not None else np.zeros(n, bool)
    src_any = src_sp | (src_voc_act if src_voc_act is not None else False)
    evs += events(dub_sp & ~src_any & face_t, "dub_over_source_silence", 0.3, "medium",
                  "the dub speaks where the original is silent (spill or early start); a mouth is moving")
    evs += events(dub_sp & ~src_any & ~face_t, "dub_over_source_silence", 0.3, "high" if F is not None else "medium",
                  "the dub speaks where the original is silent and no mouth on screen is moving")
    if db_bed is not None:
        drop = (db_bg > -50) & (db_bg - db_bed > 4.0)
        evs += events(drop, "bed_ducked", 1.0, "medium", "music/effects in the dub are more than 4 dB below the original bed")
        miss = (db_bg > -45) & (db_bed < -60)
        evs += events(miss, "bed_missing", 0.5, "high", "music/effects present in the original are missing in the dub")
        both = db_bg > -50
        S["bed_level_diff_db_median"] = round(float(np.median((db_bed - db_bg)[both])), 2) if both.any() else None
        S["bed_ducked_seconds"] = round(float(smooth_mask(drop, 0.3, 1.0).sum() * HOP), 1)
    if db_voc is not None:
        both = src_sp & dub_sp & (db_voc > -55)
        S["dialogue_vs_source_voice_db_median"] = round(float(np.median((db_dial - db_voc)[both])), 2) if both.any() else None
        # per line: loudness of the voiced frames of the dub vs the original voice (same line window)
        diffs = []
        pw = lambda x: 10 * np.log10(np.mean(10 ** (x[x > np.percentile(x, 98) - 25] / 10)) + 1e-12)
        for L in lines:
            i0, i1 = int(L["start"] / HOP), int(np.ceil((L["end"] + 0.25) / HOP))
            sv, dv = db_voc[i0:i1], db_dial[i0:i1]
            if len(sv) < 4: continue
            dd = float(pw(dv) - pw(sv)); diffs.append(abs(dd))
            if abs(dd) > 6:
                evs.append({"kind": "level_jump", "start": L["start"], "end": L["end"], "dur": round(L["end"] - L["start"], 2), "severity": "medium",
                            "why": f"dub voice is {dd:+.1f} dB vs the original voice on this line"})
        if diffs: S["line_level_diff_db_mean_abs"] = round(float(np.mean(diffs)), 2); S["lines_level_diff_over_6db"] = int(sum(d > 6 for d in diffs))
    # clipping on the delivered audio at its native rate
    raw = load(a.dub, sr=48000, ch=2); clip = np.abs(raw) >= 0.999
    S["clipped_samples"] = int(clip.sum()); S["sample_peak_dbfs"] = round(float(20 * np.log10(np.abs(raw).max() + 1e-9)), 2)
    if clip.any():
        cf = np.zeros(n, bool); idx = np.flatnonzero(clip.any(1)) // int(48000 * HOP); cf[np.clip(idx, 0, n - 1)] = True
        evs += events(cf, "clipping", 0.0, "high", "samples at full scale (audible distortion)", close=0.2)
    # drift: source bed vs dub mix, 30 s windows
    if db_bg is not None:
        w = int(30 / HOP); lags = []
        for s0 in range(0, n - w, w):
            L, c = xcorr_lag(np.clip((db_bed if db_bed is not None else db_dub)[s0:s0 + w], -70, 0), np.clip(db_bg[s0:s0 + w], -70, 0), 6); lags.append((round(s0 * HOP), int(L * HOP * 1000), round(c, 2)))
        S["bed_sync_per_30s_lag_ms"] = lags
        for t0, L, c in lags:
            if abs(L) > 40 and c > 0.3: evs.append({"kind": "drift", "start": t0, "end": t0 + 30, "dur": 30, "severity": "high", "why": f"dub is {L} ms off the source"})
    S["loudness_source"] = ebur128(a.src); S["loudness_dub"] = ebur128(a.dub)
    for e in evs:
        e["line"] = line_at(lines, e["start"], e["end"]) if lines else None
        q = qa.get(e["line"]) if e["line"] else None
        if q: e["detail"] = {k: q.get(k) for k in ("variant", "en", "kept", "tempo", "gain") if q.get(k) is not None}
    evs.sort(key=lambda e: e["start"])
    S["flags"] = {k: sum(1 for e in evs if e["kind"] == k) for k in sorted(set(e["kind"] for e in evs))}
    S["flags_by_severity"] = {s: sum(1 for e in evs if e["severity"] == s) for s in ("high", "medium", "low")}
    S["flagged_seconds_high"] = round(sum(e["dur"] for e in evs if e["severity"] == "high"), 1)
    rep["summary"] = S; rep["events"] = evs
    return rep, F, [("src speech", src_sp, "mask"), ("src vocals", src_voc_act if src_voc_act is not None else src_sp, "mask"),
                    ("dub speech", dub_sp, "mask"), ("face talking", face_t, "mask"),
                    ("mouth act", F["act"] if F else np.zeros(n), "line"),
                    ("bed diff dB", np.clip(db_bed - db_bg, -20, 10) if db_bed is not None else np.zeros(n), "line")], n

def solo(a):
    D = dur_of(a.video); n = int(D / HOP); x = load(a.video); db = frames_db(x, n)
    sp, thr = energy_vad(db, rel=30.0)
    F = faces_grid(a.faces, n) if a.faces else None
    rep = {"name": os.path.basename(a.video), "mode": "solo", "duration": D, "faces_fps": F["fps"] if F else None, "events": []}
    S = {"audio_active_seconds": round(sp.sum() * HOP, 1)}; evs = []
    quiet = db < -50
    evs += events(quiet, "silence", 1.5, "medium", "more than 1.5 s of near silence")
    talking = np.zeros(n, bool)
    if F is not None:
        has = F["face"]; act = F["act"]; th = np.percentile(act[has], 70) if has.any() else 0
        talking = has & (act > th); S["face_on_screen_pct"] = round(100 * has.mean(), 1)
        evs += events(talking & quiet, "talking_face_silent", 0.6, "high", "a mouth is moving on screen while the audio is near silent")
    raw = load(a.video, sr=48000, ch=2); clip = np.abs(raw) >= 0.999
    S["clipped_samples"] = int(clip.sum()); S["sample_peak_dbfs"] = round(float(20 * np.log10(np.abs(raw).max() + 1e-9)), 2)
    if clip.any():
        cf = np.zeros(n, bool); idx = np.flatnonzero(clip.any(1)) // int(48000 * HOP); cf[np.clip(idx, 0, n - 1)] = True
        evs += events(cf, "clipping", 0.0, "high", "samples at full scale", close=0.2)
    # loudness jumps between consecutive 3 s blocks (scene changes in a cut)
    w = int(3 / HOP); blocks = [float(np.percentile(db[i:i + w], 90)) for i in range(0, n - w, w)]
    for i in range(1, len(blocks)):
        if abs(blocks[i] - blocks[i - 1]) > 10 and max(blocks[i], blocks[i - 1]) > -35:
            evs.append({"kind": "loudness_jump", "start": round(i * 3.0, 1), "end": round(i * 3.0 + 3, 1), "dur": 3.0, "severity": "medium",
                        "why": f"loudness jumps {blocks[i] - blocks[i - 1]:+.1f} dB between 3 s blocks"})
    S["loudness"] = ebur128(a.video)
    evs.sort(key=lambda e: e["start"])
    S["flags"] = {k: sum(1 for e in evs if e["kind"] == k) for k in sorted(set(e["kind"] for e in evs))}
    rep["summary"] = S; rep["events"] = evs
    return rep, F, [("audio active", sp, "mask"), ("level dB", np.clip(db, -70, 0), "line"), ("face talking", talking, "mask")], n

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("mode", choices=["pair", "solo"])
    for k in ("src", "dub", "video", "faces", "out", "vocals", "background", "segments", "dialogue", "lines", "qa"): ap.add_argument("--" + k)
    ap.add_argument("--no-thumbs", action="store_true")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    rep, F, tracks, n = pair(a) if a.mode == "pair" else solo(a)
    if not a.no_thumbs: thumbs(a.src if a.mode == "pair" else a.video, rep["events"], F, a.out)
    timeline_png(f"{a.out}/timeline.png", tracks, rep["events"], n)
    json.dump(rep, open(f"{a.out}/avqa.json", "w"), indent=1, ensure_ascii=False); write_html(a.out, rep)
    print(json.dumps(rep["summary"], indent=1))
