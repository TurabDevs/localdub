"""Placement helpers used by dub.py (v6), added after the first AV-QA run on v5.

AV-QA on v5 showed the same three problems again and again:
  * the dub line finished early while the actor kept talking on screen (TTS shorter than the original line)
                                                                  -> plan(): fill the phrase with a gentle slow-down,
                                                                     and split at pauses when the original pauses
  * dub voice level was matched on raw RMS (the original window includes its pauses) and the whole mix was scaled
    down for a single peak, so the dub came out 2.5 LU quieter than the original -> active_rms() + limiter()
  * laughs, groans and shouts outside the dubbing lines vanished (only VAD lines were mixed) -> nonverbal_regions()
"""
import numpy as np

FR = 0.02   # seconds per energy frame

def _db(x, sr):
    h = int(FR * sr); n = len(x) // h
    if n == 0: return np.array([-100.0])
    return 10 * np.log10(np.mean(x[: n * h].reshape(n, h).astype(np.float64) ** 2, 1) + 1e-10)

def _runs(m):
    d = np.diff(np.concatenate([[0], m.astype(np.int8), [0]])); return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))

def islands(x, sr, gap=0.4, min_len=0.15, rel=30.0, floor=-45.0):
    """speech islands (start_s, end_s) of a mono clip: energy above max(floor, peak - rel), gaps < gap merged"""
    db = _db(x, sr); thr = max(floor, np.percentile(db, 98) - rel); m = db > thr
    out = []
    for a, b in _runs(m):
        if out and (a - out[-1][1]) * FR < gap: out[-1][1] = b
        else: out.append([a, b])
    return [(a * FR, b * FR) for a, b in out if (b - a) * FR >= min_len]

def active_rms(x, sr, rel=30.0):
    """RMS over the voiced frames only (frames within rel dB of the loud part), so pauses do not lower it"""
    db = _db(x, sr)
    if db.max() < -80: return 1e-7
    act = db > np.percentile(db, 98) - rel
    return float(np.sqrt(np.mean(10 ** (db[act] / 10)))) if act.any() else 1e-7

def tts_cuts(a, sr, min_gap=0.12):
    """candidate cut points (sample index, fraction of voiced time before it) at silences inside a TTS take"""
    db = _db(a, sr); thr = np.percentile(db, 98) - 35; m = db > thr; voiced = np.cumsum(m) / max(1, m.sum())
    cuts = []
    for s, e in _runs(~m):
        if s == 0 or e >= len(m) or (e - s) * FR < min_gap: continue
        mid = (s + e) // 2; cuts.append((int(mid * FR * sr), float(voiced[mid])))
    return cuts

def plan(a, ref, sr, slot, tempo, fmin=0.85, fmax=1.12):
    """Split a TTS take into pieces that start where the original speaker starts each phrase, and stretch each
    piece gently so it ends about when the actor stops talking.
    a: TTS (already fitted), ref: original vocals of the line (mono, from line start), slot: seconds available,
    tempo(x, f): time-stretch function. Returns [(offset_seconds, audio)], info."""
    isl = islands(ref, sr)
    if not isl: return [(0.0, a)], {"pieces": 1, "islands": 0, "tempi": [1.0]}
    tot = sum(e - s for s, e in isl); cum = np.cumsum([e - s for s, e in isl]) / tot
    cuts = tts_cuts(a, sr); chosen = []; win = [list(isl[0])]; last = -1.0
    for k in range(len(isl) - 1):                  # map each original pause to the TTS silence at the same share of speech
        best = min(cuts, key=lambda c: abs(c[1] - cum[k]), default=None)
        if best and abs(best[1] - cum[k]) <= 0.2 and best[1] > last and best[0] > (chosen[-1] if chosen else 0):
            chosen.append(best[0]); last = best[1]; win.append(list(isl[k + 1]))
        else:
            win[-1][1] = isl[k + 1][1]            # no matching TTS pause: merge with the next island
    parts = np.split(a, chosen) if chosen else [a]
    out = []; info = {"pieces": len(parts), "islands": len(isl), "tempi": []}
    for j, (p, (ws, we)) in enumerate(zip(parts, win)):
        nz = np.flatnonzero(np.abs(p) > 0.01); p = p[max(0, nz[0] - 200): nz[-1] + 800] if len(nz) else p
        d = len(p) / sr; w = we - ws
        nxt = win[j + 1][0] if j + 1 < len(win) else slot
        f = 1.0
        if d < 0.93 * w: f = max(fmin, d / (0.96 * w))                    # actor still talking: slow down a little
        elif ws + d > nxt - 0.05: f = min(fmax if j + 1 < len(win) else 1.15, d / max(0.1, nxt - 0.05 - ws))
        if abs(f - 1) >= 0.02: p = tempo(p, f)
        info["tempi"].append(round(f, 3)); out.append((ws, p))
    return out, info

def limiter(x, sr, ceil=0.97, block=0.01):
    """peak limiter with ~20 ms look-ahead instead of scaling the whole mix for one peak"""
    h = int(block * sr); n = int(np.ceil(len(x) / h)); pad = np.pad(np.abs(x).max(1) if x.ndim > 1 else np.abs(x), (0, n * h - len(x)))
    pk = pad.reshape(n, h).max(1); g = np.minimum(1.0, ceil / (pk + 1e-9))
    gm = np.array([g[max(0, i - 2): i + 3].min() for i in range(n)])
    gs = np.convolve(np.pad(gm, 1, mode="edge"), np.ones(3) / 3, "valid")
    gg = np.interp(np.arange(len(x)), np.arange(n) * h + h / 2, gs).astype(np.float32)
    y = x * (gg[:, None] if x.ndim > 1 else gg)
    return np.clip(y, -0.99, 0.99), float((gs < 0.999).mean())

def nonverbal_regions(voc, sr, lines, min_len=0.3, pad_before=0.1, pad_after=0.3):
    """regions where the original vocals are loud but no dubbing line covers them (laughs, groans, shouts)"""
    db = _db(voc, sr); thr = max(-40.0, np.percentile(db, 99) - 28); m = db > thr
    cover = np.zeros(len(m), bool)
    for L in lines: cover[max(0, int((L["start"] - pad_before) / FR)): int((L["end"] + pad_after) / FR)] = True
    m &= ~cover; out = []
    for a, b in _runs(m):
        if out and (a - out[-1][1]) * FR < 0.25: out[-1][1] = b
        else: out.append([a, b])
    return [(a * FR, b * FR) for a, b in out if (b - a) * FR >= min_len]
