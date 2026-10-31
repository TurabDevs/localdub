#!/usr/bin/env python3
"""LocalDub QA metrics ("listen" proxies) for one dub version.
usage: qa_metrics.py <qa.json> <dialogue_en.wav> <s1/vocals.flac> <lines.json> [listen.json] > metrics.json
Per dubbed line: overlap into the next line, loudness vs the original line (dB), median pitch vs the original line
(semitones; a large gap = voice mismatch), and, if a listen.json from the Omnilingual ASR re-transcription is given,
word error rate of what the dub actually says vs the intended English."""
import json, re, sys
import numpy as np, soundfile as sf
qa = json.load(open(sys.argv[1])); dia, sr = sf.read(sys.argv[2], dtype="float32", always_2d=True); dia = dia.mean(1)
voc, sr2 = sf.read(sys.argv[3], dtype="float32", always_2d=True); voc = voc.mean(1); assert sr == sr2
lines = sorted(json.load(open(sys.argv[4])), key=lambda l: l["start"]); LI = {l["id"]: l for l in lines}
nxt = {a["id"]: b["start"] for a, b in zip(lines, lines[1:])}
listen = json.load(open(sys.argv[5])) if len(sys.argv) > 5 else {}
def db(x): return 20 * np.log10(np.sqrt(np.mean(x ** 2)) + 1e-7)
def f0(x):
    x = x[::max(1, sr // 16000)]; fs = sr / max(1, sr // 16000); n = int(0.04 * fs); h = n // 2; out = []
    for i in range(0, len(x) - n, h):
        fr = x[i:i + n] - x[i:i + n].mean()
        if np.sqrt(np.mean(fr ** 2)) < 0.01: continue
        ac = np.correlate(fr, fr, "full")[n - 1:]; ac /= ac[0] + 1e-9
        lo, hi = int(fs / 400), int(fs / 70); k = lo + int(np.argmax(ac[lo:hi]))
        if ac[k] > 0.5: out.append(fs / k)
    return float(np.median(out)) if len(out) >= 5 else None
def words(t): return re.findall(r"[a-z']+", t.lower().replace("’", "'"))
def wer(ref, hyp):
    r, h = words(ref), words(hyp)
    if not r: return None
    d = list(range(len(h) + 1))
    for i in range(1, len(r) + 1):
        p, d[0] = d[0], i
        for j in range(1, len(h) + 1):
            p, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, p + (r[i - 1] != h[j - 1]))
    return d[len(h)] / len(r)
per, dub = {}, [q for q in qa["lines"] if q["variant"] != "original"]
for q in dub:
    L = LI[q["id"]]; s = int(L["start"] * sr); dur = q["tts"] / (q.get("tempo") or 1.0); e = s + int(dur * sr)
    o = voc[int(L["start"] * sr):int(L["end"] * sr)]; dd = dia[s:e]
    end = L["start"] + dur; ov = max(0.0, end - nxt.get(q["id"], 1e9))
    fo, fd = f0(o), f0(dd)
    m = {"overlap_s": round(ov, 2), "loud_diff_db": round(float(db(dd) - db(o)), 1),
         "pitch_diff_st": round(12 * np.log2(fd / fo), 1) if fo and fd else None}
    if q["id"] in listen: m["wer"] = wer(q["en"], listen[q["id"]]["heard"]); m["heard"] = listen[q["id"]]["heard"]
    per[q["id"]] = m
vals = lambda k: [v[k] for v in per.values() if v.get(k) is not None]
ld, pd, wr = vals("loud_diff_db"), vals("pitch_diff_st"), vals("wer")
kept = {}
for q in qa["lines"]:
    if q["variant"] == "original": kept[q.get("kept", "english_original")] = kept.get(q.get("kept", "english_original"), 0) + 1
summary = {"lines": len(qa["lines"]), "dubbed": len(dub), "kept_original": kept, "fit": sum(q["fits"] for q in dub),
           "overflow": len(qa["overflow"]), "no_tempo_change": sum(abs((q.get("tempo") or 1) - 1) < 0.01 for q in dub),
           "max_tempo": max((q.get("tempo") or 1) for q in dub),
           "overlaps_next_line": sum(v["overlap_s"] > 0.05 for v in per.values()), "overlap_total_s": round(sum(vals("overlap_s")), 2),
           "loud_mean_abs_diff_db": round(float(np.mean(np.abs(ld))), 2), "loud_lines_off_by_6db": sum(abs(x) > 6 for x in ld),
           "pitch_mean_abs_diff_st": round(float(np.mean(np.abs(pd))), 2) if pd else None, "pitch_lines_off_by_4st": sum(abs(x) > 4 for x in pd),
           "filler_dubs": sum(len(words(q["en"])) <= 1 for q in dub)}
if wr: summary.update({"listen_wer_mean": round(float(np.mean(wr)), 3), "listen_lines_wer_over_0.5": sum(x > 0.5 for x in wr), "listen_lines": len(wr)})
json.dump({"summary": summary, "per_line": per}, sys.stdout, indent=1, ensure_ascii=False, default=lambda o: o.item() if hasattr(o, "item") else str(o))
