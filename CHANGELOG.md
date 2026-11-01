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

**Evening: does the voice belong to the face that is talking?**

v6 sounded right line by line, so I checked something the earlier QA never looked at: whether each dubbed line is spoken in the voice of the person whose mouth is actually moving. I built a character stage for that: face identities across shots (`pipeline/char_faces.py`, YuNet + SFace, 17 face ids), a Whisper large-v3 pass with word timestamps on the separated vocals, and `pipeline/char_link.py`, which links each speech unit to the face whose mouth moves with it (101 speech units, 82 confirmed by a face, 18 characters, 10 of them with real screen time).

What it found in v6 (`pipeline/char_mismatch.py`, `results/characters_aib/mismatch_v6.json`):

- **13 of 33 dubbed lines are voiced as the wrong character** for at least part of the line. Together that is 37.9 s of the 134.1 s of face-confirmed speech, about 28 %, spoken in another character's voice.
- **15 lines use a blended voice.** The voice reference for those lines was cut from an audio chunk where two or more people talk, so the clone is a mix of them.
- **14 dubbed lines are really several people taking turns** (one of them is 6 turns between 4 people in 11.6 seconds). The voice-activity chunking had glued the turns into one line, so one voice spoke for everybody. v5 had exactly the same problem; my pitch and listen checks could not see it.

The fix I started: split those 14 lines at the speaker changes into 40 turns, let Muse Glimmer write the English for each turn with the whole script as context, and voice every line from a single reference per character, cut only from that character's own face-confirmed speech (`dub.py` with `CHARS` / `CHARMAP`). The per-character profiles (face thumbnail, screen time, pitch range, estimated gender/age, style, sample lines) are in `results/characters_aib/`; gender and age there are estimates and are labelled that way.

Late evening, first results of the character dub (v7, still in progress):

- **A Glimmer job that "finished" with nothing in it.** One of the two per-turn writing jobs used its whole 2,200-token budget thinking (counting syllables word by word) and returned an empty answer, so half the split lines had no English. The kernel still reported success. I re-ran it with a bigger budget and an instruction to think briefly, and the stage 2 script now retries a job once with twice the budget when its answer comes back empty.
- **One voice per character made some lines worse.** Measured turn by turn against the original (`pipeline/char_voice_check.py`, which only uses pitch and does not look at faces), 11 lines landed more than 4 semitones away from the actor, mostly shouted lines voiced from a calm clip of the same character. Fix: for those lines the reference is still that character's own speech, but the clips closest in pitch to the line. 10 of 11 got much closer (for example 6.8 -> 0.2 and 8.1 -> 0.6 semitones); one got worse (5.5 -> 9.7), so it keeps its earlier take.
- English -> Urdu: Muse Glimmer's Urdu lines for the Tears of Steel clip came back. Because the voice reads Devanagari better than Urdu script, the runner now has a transliteration step where Glimmer writes the same Urdu words in Devanagari for the voice only; subtitles stay in Urdu script.

**Also today**

- Bilingual subtitles (SRT/WebVTT, source + English) and a review page for checking a dub line by line.
- Started the English -> Urdu direction on a Tears of Steel clip (CC BY 3.0). Found that Fish Audio reads the same Urdu sentence much more clearly from Devanagari text than from Urdu script or Roman Urdu (checked by running the TTS output back through ASR), so the Urdu dub will feed the TTS Devanagari text.
- **Bug in my new one-command runner:** the Glimmer step recorded a failed Kaggle dataset upload as done and then kept polling a dataset that didn't exist, so the English -> Urdu run sat at "WAITING" for hours. It now checks the upload result and re-creates the dataset if Kaggle says it is missing. The kernel started right after the fix.
- README: corrected the "Running it" section. It made the runner sound finished; the demo was actually built step by step with the individual scripts, and the runner's first end-to-end run is not done yet.
