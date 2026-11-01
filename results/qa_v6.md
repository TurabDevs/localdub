# v6: what the frame-by-frame AV-QA found in v5, and what I changed

v6 is the first version driven by `pipeline/avqa.py`, a frame-by-frame audio/video check of the dub against the source.
Every 50 ms it compares: speech in the original (Silero segments) and in the dub (dialogue stem), other voice sounds in
the original (laughs, groans, shouts), faces on screen and whether their mouth is moving (YuNet + MediaPipe face mesh +
mouth-region motion, 8 fps), the music/effects bed of the original vs the dub, loudness and clipping, and sync.

## What it found in v5

| check | v5 |
|---|---|
| original speech covered by dub speech | 84.4 % |
| "face talking, original speaking, dub silent" spots | 15 (21.2 s) |
| original speech with no dub (no moving mouth detected) | 5 |
| voice sounds outside any line that the dub dropped | 3, incl. a 4.5 s groan at 1:50 and shouts at 3:11 |
| integrated loudness | dub -19.0 LUFS vs original -15.9 LUFS |
| music/effects bed vs original (median) | -1.9 dB |
| clipping / sync drift | 0 samples / 0 ms in every 30 s window |

Causes:
1. Most "dub silent" spots were line endings: the English take was shorter than the actor's line, so the dub stopped
   while the actor kept talking on screen.
2. The whole mix was scaled down to fit one peak, which pulled the music and effects 1.9 dB under the original and
   made the dub 3 LU quieter overall.
3. Only VAD speech lines were mixed, so non-verbal vocals between lines (a groan, shouts during the broom scene) vanished.
   The demo cut showed the groan as 5 s of silence over a wide-open mouth.

## Fixes (pipeline/align.py, pipeline/dub.py with V6=1)
- each dub line follows the actor's phrasing: if the take ends well before the actor stops, it is slowed gently
  (never below 0.85x), and it is split at pauses when the original pauses (12 lines slowed, 0.85-0.94x)
- when a take fills less than 70 % of the line, a longer Glimmer variant that fits at <= 1.16x is used instead
  (L005, L020, L024, L025 now use the full line; L005 keeps "I have typhoid, I have typhoid, I have typhoid")
- voice level matched on voiced frames only; a look-ahead peak limiter replaces whole-mix scaling (limiter active on 0.2 % of the clip)
- non-verbal vocals outside the dubbing lines are kept from the original (2 regions: 109.98-114.20 s, 190.76-193.36 s)

## Result (same analyzer, same 33 dubbed lines)

| | v5 | v6 |
|---|---|---|
| original speech covered by dub speech | 84.4 % | 90.9 % |
| high-severity AV flags | 15 (21.2 s) | 9 (7.0 s) |
| dropped non-verbal vocals | 3 | 1 (0.65 s) |
| integrated loudness (original -15.9 LUFS) | -19.0 LUFS | -16.5 LUFS |
| music/effects bed vs original (median) | -1.9 dB | +0.1 dB |
| mean loudness gap per line (qa_metrics) | 0.61 dB | 0.49 dB |
| listen WER, Omnilingual ASR (33 lines) | 0.151 | 0.131 |
| lines with WER > 0.5 | 3 | 1 |
| mean pitch gap | 1.88 st | 1.94 st |
| spill into next line | 3 lines / 0.58 s | 4 lines / 0.83 s |
| max speed-up | 1.13x | 1.14x |
| clipping / drift | 0 / 0 ms | 0 / 0 ms |

Honest notes:
- Part of the WER gain is the ASR reading the same takes differently after re-timing (e.g. L034 "more poetry more cring"
  instead of mixed-script output); the takes for L002a and L021 are the same files as v5.
- The visual "mouth moving" detector is only moderately reliable: its AUC against the original speech track is 0.62
  (profile faces and dark scenes). The report says so, and every flag has a thumbnail to check by eye.
- Slowing to 0.85x keeps the actor and the dub ending together but can sound a little drawn out; a Glimmer rewrite to a
  medium-length line would be the better fix for those 12 lines.
- One new spill: L024 now runs 0.3 s past its line.

Files: avqa/ (report.html, avqa.json, timeline.png, thumbs/) for v6, avqa_v5/ for v5, metrics.json, listen_v6.json, qa.json.
