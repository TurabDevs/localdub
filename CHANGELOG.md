# Changelog

My dev log for LocalDub during the hackathon. Each entry only lists what I actually found and fixed that day (Pakistan time). Numbers come from the QA scripts in `pipeline/` and the files in `results/`.

## 2026-11-01

**Morning: fixing what the demo clips get wrong**

- The opening phone call was dubbed in one voice for both people. One speech chunk held both sides of the call, so both halves were cloned from the same audio. I split it at the speaker change (the pitch drops from about 175 Hz to about 125 Hz around 20.2 s), and each side now gets its own voice (v5).
- Went through the lines inside the demo windows for meaning, not just timing. Six lines were wrong in a way the metrics can't see, including the sketch's main joke ("there's no such thing as typhoid, bro"; before it said "typhoid isn't like that") and a sarcastic question that had turned into the opposite. Muse Glimmer rewrote all six with the scene context.
- Two of the new lines came out unintelligible on the first take ("Whole neighbourhood to know?" scored WER 1.0 in the listen check). Generated five takes of each and kept the one the listen check and the pitch check liked best.
- Tried re-voicing L048 (pitch 6.8 semitones off the original) with the cluster reference. It got worse (7.9 st), so I threw that version away and kept v5.

**Afternoon: a frame-by-frame audio/video check (AV-QA)**

Until today I only listened to the dub. I wrote `pipeline/avqa.py` + `pipeline/avqa_faces.py`, which go through the video frame by frame (faces, mouth movement, speech activity of the original vs the dub, music/effects level, loudness, clipping). Running it on v5 found problems I had not heard:

- **15 spots (21.2 s in total) where an actor's mouth is moving and the dub is silent.** Most English takes were shorter than the Hindi line and ended before the actor stopped talking.
- **The whole dub was too quiet:** -19.0 LUFS against -15.9 LUFS for the original. The mix was scaled down as a whole to fit one peak, which also pushed the music and effects 1.9 dB under the original.
- **A groan and the shouts were missing.** Only speech lines were mixed, so a 4.5 s groan (109.7-114.2 s) and the shouts in the broom scene (190.7-193.4 s) simply disappeared. In the demo cut that groan was 5 seconds of silence over a wide-open mouth.

Fixes (v6, `pipeline/align.py`): each line now follows the actor's phrasing and is slowed gently (never below 0.85x) when the take is short, a longer Glimmer variant is used when a take fills less than 70 % of its slot, level is matched on voiced frames only, a look-ahead limiter replaces the whole-mix scaling, and laughs/groans/shouts between lines are kept from the original.

Result v5 -> v6: original speech covered by the dub 84.4 % -> 90.9 %, high AV-QA flags 15 spots / 21.2 s -> 9 spots / 7.0 s, loudness -19.0 -> -16.5 LUFS, music + effects -1.9 dB -> +0.1 dB vs the original, listen WER 0.151 -> 0.131. Rebuilt the demo (v5 cut) on the v6 dub.

**Also today**

- Bilingual subtitles (SRT/WebVTT, source + English) and a review page for checking a dub line by line.
- Started the English -> Urdu direction on a Tears of Steel clip (CC BY 3.0). Found that Fish Audio reads the same Urdu sentence much more clearly from Devanagari text than from Urdu script or Roman Urdu (checked by running the TTS output back through ASR), so the Urdu dub will feed the TTS Devanagari text.
- **Bug in my new one-command runner:** the Glimmer step recorded a failed Kaggle dataset upload as done and then kept polling a dataset that didn't exist, so the English -> Urdu run sat at "WAITING" for hours. It now checks the upload result and re-creates the dataset if Kaggle says it is missing. The kernel started right after the fix.
- README: corrected the "Running it" section. It made the runner sound finished; the demo was actually built step by step with the individual scripts, and the runner's first end-to-end run is not done yet.
