# LocalDub v2 QA report (dub -> listen -> fix iteration 1)
Built 2026-10-31 (PKT). v1 kept intact (build/out/aib_depression_en.webm, results/qa_v1.json). Measurements from `pipeline/qa_metrics.py` on the dubbed dialogue stem vs the original Demucs vocals.

## What the v1 listen/QA pass found
- 2 lines overflowed their slot (L030, L048); 5 more were only saved by tempo-up.
- 8 one-word filler dubs ("Uh.", "Man.", "Just.", "Huh?"...): the CTC ASR had spelled laughs, grunts and short English words in Arabic/CJK script, and they were then "translated".
- 11 lines with the dub's median pitch more than 4 semitones away from the original line (voice mismatch). Almost all were short lines (< 2.5 s) that used the whole voice cluster as reference, and a cluster can mix speakers.
- 4 lines more than 6 dB louder/quieter than the original line.
- Meaning errors from Hinglish ASR garble, e.g. "Go get tender" (Tinder), "Yesterday at Sail" (Sahil's party), "What are they, educated or something?" (he says "you're educated, right?").

## Fixes in v2
1. **ASR-garble guard** (`dub.py`): a line whose transcript has CJK/Hangul characters, or Arabic-script text with none of the Urdu-specific letters, keeps the original audio. 12 lines: L010, L011, L012, L017, L036, L042, L044, L045, L046, L047, L050, L053.
2. **Pitch-matched voice references** for short lines: nearest-in-time long line whose original pitch is within 3 semitones (same cluster first).
3. **Muse Glimmer rewrite pass** (open weights, Kaggle CPU, kernels localdub-rw2-s0/s1, ~48 min each): 20 QA-flagged lines sent back with the script, the v1 English, the measured timing and a QA note; Glimmer returned `en_fit` lines. For 5 lines (L004, L006, L007, L024, L038) the QA note was a human review note on the likely meaning (e.g. that ASR 'टेंडर' after 'Garima broke up with you' is probably 'Tinder'); the other 15 notes were only the measured timing and a syllable target. The rewrites are tried first by the fitter (tempo up to 1.15x allowed for a rewrite before falling back to v1 text). 18 rewrites were used.
4. Glimmer's per-line vision cast labels are now attached to each line in qa.json (v1 looked them up by the wrong key).

## Results (v1 -> v2)
| metric | v1 | v2 |
|---|---|---|
| lines dubbed / kept original | 44 / 14 | 32 / 26 (14 English originals + 12 ASR-garble non-speech) |
| dubbed lines fitting their slot | 42 / 44 | **32 / 32** |
| overflow | 2 | **0** |
| one-word filler dubs | 8 | **0** |
| pitch mismatch > 4 semitones | 11 | **3** (L031, L037, L048) |
| mean pitch gap vs original | 3.72 st | **2.33 st** |
| loudness gap > 6 dB | 4 | **0** |
| mean loudness gap vs original | 1.65 dB | **0.45 dB** |
| lines with no tempo change | 32 | 28 |
| max tempo | 1.08x | 1.127x (L049, a Glimmer rewrite) |
| spill into next line > 0.05 s | 1 (0.16 s) | 2 (0.52 s total; L004, L038, 0.25 s each) |
On the same 32 lines that v2 dubs, v1 had 30/32 fitting, 5 pitch mismatches, 0.64 dB mean loudness gap.

## Glimmer rewrites used in v2
| line | slot s | v1 (tts s) | v2 Glimmer rewrite (tts s, tempo) |
|---|---|---|---|
| L004 | 8.11 | Just step out, have a couple beers, all bacteria will die. Don't overthink. (4.98s) | Just step out, have a couple beers, all the bacteria will die. Ever seen a bartender get typhoid? They live around beer. Don't overthink. (8.71s, x1.042) |
| L006 | 11.62 | The bacteria will die on its own in two months. Just don't... No promotion, no, got typhoid? Yesterday at Saily's party you were enjoying so much, you were laughing, dancing, joking. What happened suddenly? (11.7s) | Bacteria die on their own in two months. Lost promotion? No? Then typhoid? Day before yesterday at Sahil's party you were laughing, dancing, joking. What happened suddenly? (9.43s, x1.0) |
| L007 | 13.78 | This fever burning could be one thing. If you're smiling and you have typhoid, how can you have both? Are you sure you didn't get typhoid because Garima broke up with you? Go get tender. Your typhoid, vomiting, diarrhea will all go away. (12.17s) | This fever burning could be one thing. If you're smiling and you have typhoid, how can you have both? Are you sure you didn't get typhoid because Garima broke up with you? Go get on Tinder. Your typhoid, vomiting, diarrhea will all go away. (12.07s, x1.0) |
| L019 | 5.43 | A smile counts, typhoid's gone, I know it. Nikhil's friend was also saying that (5.62s) | A smile means no typhoid. I know it. Nikhil's friend said that too. (4.75s, x1.0) |
| L021 | 1.24 | Tell the whole neighborhood. (1.37s) | Tell the whole neighborhood. (1.32s, x1.0) |
| L022 | 5.85 | That our son has fever, fever, fever, his temperature is up, his head is burning hot. What will people think? (5.93s) | Our son has fever, fever, fever, temperature up, head burning hot. What will people think? (4.85s, x1.0) |
| L023 | 7.42 | People will say the whole family has fever, bro. They say you have typhoid. These doctors are running a scam, they just want to get money out of you. (7.45s) | People will say the whole family has fever bro. They say you have typhoid. These doctors are running a scam, they just want cash out of you seriously. (7.3s, x1.0) |
| L024 | 4.79 | Typhoid, whatever, they just pick any name. What are they, educated or something? (5.29s) | Typhoid, whatever, they just make up names. Come on, you're educated, right? (3.47s, x1.0) |
| L030 | 0.76 | Wear it like this. (1.16s) | Put it on. (0.72s, x1.0) |
| L032 | 5.11 | Your planets are messed up. This is a protective shield, I made it specially for Libra. (5.25s) | Your planets are messed up. This shield protects you, made specially for Libra. (5.01s, x1.0) |
| L033 | 3.38 | Go do something you like. Follow your passion, everyone's doing it these days. (3.64s) | Go do something you like. Follow your passion, everyone's doing it these days. (3.98s, x1.095) |
| L034 | 2.26 | More poetry, more crying. (2.37s) | More poetry, more crying. (1.93s, x1.0) |
| L038 | 6.65 | He doesn't even know where to go, where to go, where to go, man. I think you're overthinking, overanalyzing, overhyping. (6.62s) | It doesn't even know where to go, where to go? Take a left up ahead. I think you're overthinking, overanalyzing, overhyping. (7.11s, x1.03) |
| L039 | 7.55 | Over hyperventilating, just over it. Don't think too much, like you're on fire and thinking you're not thinking. (7.8s) | Over hyperventilating, just over it. Don't overthink, you're on fire thinking you're not thinking. (6.45s, x1.0) |
| L040 | 5.85 | You don't have typhoid, maybe you just need a long vacation. Or come with us to the movie. (5.92s) | You don't have typhoid, you just need a long vacation. Come to the movies with us. (4.98s, x1.0) |
| L041 | 1.11 | Okay so you're not coming. (1.35s) | Okay, you're not coming. (1.22s, x1.0) |
| L048 | 1.08 | What are you talking about? (1.5s) | What are you saying? (1.13s, x1.0) |
| L049 | 3.64 | Don't tell anyone, I know nothing. Why would I say, guys. (4.1s) | Don't tell anyone. I know nothing. Why would I say? (4.38s, x1.127) |

## Listen check (Omnilingual ASR re-transcription of the dub)
Kernel `localdub-listenk-v1` / `localdub-listenk-v2` re-transcribe each dubbed line with Meta Omnilingual ASR CTC 1B v2 and compare with the intended English (WER). LISTEN_RESULTS_PENDING

## Still open (next iterations)
- 3 pitch outliers (L031, L037, L048): try the 2nd-best pitch-matched reference.
- Mixed-speaker own-line references (e.g. L002 phone call both sides, L038).
- Two 0.25 s spills into the next line (L004, L038).
