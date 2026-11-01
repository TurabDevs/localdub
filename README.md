# LocalDub

LocalDub takes a Hindi or Urdu video and gives it back dubbed in English, with each character speaking in a voice cloned from their own original lines and the original music and sound effects kept underneath.

I'm Muhammad Asim, a student developer from Pakistan. I built this solo for the Meta Global AI Developer Hackathon 2026 (Indie Developer track, "Build for Your World"), with no budget, on free Kaggle CPU machines.

- Demo video (2:36): https://github.com/TurabDevs/localdub/releases/download/demo-v2/localdub_demo_v3.mp4
- Full 5-minute English dub (latest): https://github.com/TurabDevs/localdub/releases/download/demo-v2/aib_depression_en_v4.webm
- All releases: https://github.com/TurabDevs/localdub/releases

## Why I built this

Where I live, almost everything people watch and make is in Urdu, Punjabi, Saraiki or Hindi. There are thousands of creators here making comedy, cooking videos, tutorials and lectures, and their work almost never travels past people who already speak the language. Big channels pay dubbing studios. A kid with a phone and a good sketch can't.

The other direction hurts too. Most good teaching material online is in English, and a lot of students around me (I'm one of them) follow it slower than they should because they are translating in their head the whole time.

Automatic dubbing tools exist, but the ones I tried had the same three problems with our content:

1. Jokes get translated word for word, so they stop being jokes. Hinglish (Hindi and English mixed in one sentence) confuses them badly.
2. Everyone gets the same stock voice, or the wrong voice for the person on screen.
3. Lines don't fit. The dub keeps talking after the actor's mouth has stopped.

LocalDub is my attempt to fix those three things, using models I can actually run.

## Why open-weight models

This part is not a style choice. Pakistan is one of the countries where Meta's hosted Model API is not available. I asked Meta support during the hackathon; they confirmed the restriction and couldn't give an exception or API credits. I also don't have a card or a budget for paid inference.

So every Meta model in LocalDub is an open-weight model that I downloaded and ran myself, offline, inside free Kaggle notebooks. No hosted Meta API, no VPN. That turned out to be the point of the project: if it runs on a free CPU box, it can run for creators who have nothing else either.

The cost of doing it this way is speed. My Kaggle account could only use CPU machines (4 vCPU, 30 GB RAM, no internet inside the kernel), and Muse Glimmer 30B on that hardware generates about 1 token per second. A full first pass on the 5-minute demo clip took around 4 hours of wall-clock time, most of it waiting on Glimmer. It works, it's just slow.

## How it works

```
                     original video (Hindi / Urdu / Hinglish)
                                     |
             +-----------------------+-----------------------+
             |                                               |
     Demucs (Meta) splits the audio                  low-res frames
             |                                               |
   vocals ---+--- music + effects (kept for the final mix)   |
     |                                                       |
  Silero VAD -> speech chunks                                |
     |                                                       |
  Omnilingual ASR (Meta) -> Hindi/Urdu transcript            |
  ECAPA embeddings -> rough voice clusters                   |
     |                                                       |
     +---------------> Muse Glimmer 30B (Meta) <-------------+
                       - sees a frame for each line: who is on screen,
                         gender, age, role, face visible or not
                       - reads the whole script at once and writes
                         English that keeps the joke, a full and a short
                         version per line, plus an emotion tag
                                     |
             voice clone per line (reference = that speaker's own
             original line from the separated vocals), Fish Audio TTS
                                     |
             fit each line into its original time slot
             (pick the version that fits, small tempo change only if needed)
                                     |
             mix over the original music + effects -> dubbed video
                                     |
             LISTEN / QA pass on my own output:
             - timing, overflow, spill into the next line
             - loudness and pitch of each dubbed line vs the original line
             - Omnilingual ASR re-transcribes the English dub -> word error rate
                                     |
             bad lines go back to Muse Glimmer with a note -> rewrite -> re-dub
```

| Step | Model / tool | Runs where |
|---|---|---|
| Separate voice from music | Meta Demucs `htdemucs_ft` (MIT) | Kaggle CPU kernel, offline |
| Hindi/Urdu speech recognition | Meta Omnilingual ASR `omniASR_CTC_1B_v2` (Apache-2.0), through a Hugging Face `Wav2Vec2ForCTC` conversion because the official package needs Python <= 3.12 and Kaggle has 3.13 | Kaggle CPU kernel, offline |
| Who is on screen, per line | Meta Muse Glimmer 30B (Apache-2.0), GGUF Q4_K_M + vision projector, llama.cpp | Kaggle CPU kernel, offline |
| Translation, short versions, emotion tags | Meta Muse Glimmer 30B | Kaggle CPU kernel, offline |
| Rewrites of lines that failed QA | Meta Muse Glimmer 30B | Kaggle CPU kernel, offline |
| Checking what the dub actually says | Meta Omnilingual ASR | Kaggle CPU kernel, offline |
| Speech segments | Silero VAD (MIT) | Kaggle |
| Voice clusters | SpeechBrain ECAPA (Apache-2.0) | Kaggle |
| English voices | Fish Audio `s2.1-pro-free` zero-shot cloning (free tier, hosted) | API call |
| Fitting and mixing | ffmpeg, numpy | my sandbox |

Before llama.cpp starts, the Glimmer weights are checked by sha256 against the official `meta-models/Muse-Glimmer-30B-GGUF` release, because they come from a Kaggle mirror.

## What I measured

The demo clip is a 5 min 08 s comedy sketch by All India Bakchod, "If People Treated Other Illnesses Like They Treat Depression". It's fast, it's Hinglish, people talk over each other, and the whole joke depends on tone, which is exactly why I picked it.

The clip has 58 speech lines. 14 are already in English and keep the original audio. From v2 on, 12 more keep the original audio too: these are laughs, grunts and one-word bits that the ASR spelled in Arabic or Chinese script and that v1 had "translated" into junk like "Uh." or "Man." That leaves 32 lines that are actually dubbed.

I did four rounds of dub -> listen -> fix. All numbers come from `pipeline/qa_metrics.py` and the Omnilingual ASR listen check (`results/`).

| | v1 | v2 | v3 | v4 (current) |
|---|---|---|---|---|
| lines dubbed | 44 | 32 | 32 | 32 |
| dubbed lines that fit their slot | 42 / 44 | 32 / 32 | 32 / 32 | 32 / 32 |
| lines that overflow | 2 | 0 | 0 | 0 |
| junk one-word dubs from ASR garble | 8 | 0 | 0 | 0 |
| voice pitch more than 4 semitones off the original | 11 | 3 | 1 | 1 |
| mean pitch gap vs original line | 3.72 st | 2.33 st | 1.97 st | 1.84 st |
| mean loudness gap vs original line | 1.65 dB | 0.45 dB | 0.35 dB | 0.63 dB |
| listen WER, all dubbed lines | 0.289 | 0.187 | 0.144 | 0.123 |
| listen WER on the same 32 lines | 0.116 | 0.187 | 0.144 | 0.123 |
| max tempo change | 1.08x | 1.13x | 1.13x | 1.13x |

What changed in each round:

- **v2**: added a guard so garbled ASR lines keep the original audio, picked voice references by pitch for short lines (a short line can't clone a voice from itself), and sent 20 flagged lines back to Glimmer for a rewrite (18 used). For 5 of those I wrote a note on what the Hindi probably meant, for example that the ASR's "टेंडर" after "Garima broke up with you" is "Tinder". The other 15 only had the measured timing and a syllable target.
- **v3**: re-voiced the 3 lines whose pitch was way off with a different reference.
- **v4**: fixed "Put it on.", which the listen check heard as "vtdn" (WER 1.0 -> 0.0), and moved one line that came out with a heavy accent back to the cluster reference.

I want to be straight about the last two rows. The big drop in WER from v1 to v2 is mostly because v2 stopped dubbing the junk lines. On the 32 lines both versions dub, v1 was actually slightly clearer (0.116), v2 and v3 got worse, and v4 got back to almost the same level (0.123) while fixing the overflow, junk and voice problems. A lot of the remaining "errors" are the ASR spelling things its own way ("tiphoid", "mikil" for Nikhil), not wrong speech. Full per-line details, including every Glimmer rewrite, are in `results/qa_v2.md` to `results/qa_v4.md`.

## What doesn't work yet

- **One clip.** Everything above is measured on one 5-minute sketch. I haven't tested it on other creators, other genres, or Punjabi/Saraiki audio yet.
- **Mixed speakers in one chunk.** The opening phone call has both people inside one speech chunk, so both get one voice. The speaker clustering also has one big mixed cluster; short lines that borrow a voice from it can get the wrong person.
- **Hinglish ASR.** Omnilingual ASR handles Hindi well, but English words inside Hindi sentences sometimes come out in the wrong script. Glimmer repairs most of it from context; some meaning still slips (see the rewrite notes).
- **26 of 58 lines keep the original audio.** 14 are fine because they are English. 12 are short laughs and interjections I chose not to dub rather than dub badly.
- **No lip-sync.** Lines fit their time slot, but mouths are not re-animated.
- **The voices are not a Meta model.** The TTS is Fish Audio's free hosted tier. Everything that understands the video is Meta and open-weight; the voice synthesis is not. I'd like to replace it with an open TTS I can run myself.
- **Slow.** CPU only, about 1 token per second for Glimmer. Fine for a demo, not for a creator waiting on an upload.

## Where I want to take it

This is the part I care about most. The hackathon version proves the pipeline; this is what I want LocalDub to become.

1. **Our languages, both ways.** Urdu, Punjabi, Saraiki and Pashto first, then Sindhi, Balochi and more. Omnilingual ASR covers more than 1,600 languages, including many that commercial tools ignore, so the listening side already reaches them. The writing side needs testing language by language.
2. **English into Urdu and Hindi.** The same pipeline in reverse: lectures, tutorials and courses dubbed into Urdu so students here can learn in the language they think in. For me this may matter even more than the Hindi -> English direction.
3. **Education and lectures.** Long, mostly single-speaker videos are the easiest case for this pipeline and probably the most useful one: university lectures, coding tutorials, health information.
4. **Lip-sync with face tracking.** Glimmer already tells me who is on screen and whether their face is visible for each line. Next is tracking that face (SAM 2 / SAM 3) and only re-animating mouths when the face is visible and close enough to matter.
5. **Better voices per character.** Split mixed chunks by speaker, build a clean reference per character, and use Glimmer's on-screen labels to check that the voice matches the person.
6. **A simple web app for creators.** Upload a video, review the script line by line (approve, edit, or ask Glimmer for another version), listen, export. The QA report I already generate is basically the review screen.
7. **An open voice model.** Replace the hosted TTS so the whole thing can run on one machine with no API keys.

What a GPU budget would change: on a single 24 GB GPU, Glimmer goes from about 1 token per second to something usable, and a 5-minute video would take minutes instead of hours. That would make a real creator tool possible, run more rewrite rounds per video, and let me test many languages instead of one clip. Right now the whole project runs on free tiers, and I'm a student, so compute is the main thing holding it back.

## Repository layout

- `cloud/kaggle/stage1_listen.py`: separation, VAD, ASR and speaker embeddings (offline Kaggle kernel)
- `cloud/kaggle/stage2_glimmer.py`: starts llama.cpp with Muse Glimmer and runs text and image jobs (resumable, can be split across kernels)
- `cloud/kaggle/stage_listen_check.py`: re-transcribes the English dub with Omnilingual ASR
- `cloud/kaggle/kapi.py`, `cloud/kaggle/kup.py`: small Kaggle API client (datasets, kernels, chunked uploads)
- `pipeline/make_jobs.py`: voice clusters -> dubbing lines -> Glimmer cast and translation jobs
- `pipeline/dub.py`: per-line TTS, slot fitting, mixing and the QA report
- `pipeline/qa_metrics.py`: per-line metrics (fit, spill, loudness, pitch vs original, listen WER)
- `pipeline/make_rw_jobs.py`: builds the Glimmer rewrite job for flagged lines
- `pipeline/listen_prep.py`: cuts the dubbed lines for the listen check
- `results/`: QA reports, metrics and listen results for every version

## Credits

- Demo source: "If People Treated Other Illnesses Like They Treat Depression" by All India Bakchod (AIB), licensed under [CC BY 3.0](https://creativecommons.org/licenses/by/3.0/). Changes made: the dialogue was translated into English and re-voiced by LocalDub; the music and effects come from the original through source separation.
- Meta open-weight models: Muse Glimmer 30B, Omnilingual ASR, Demucs. Also Silero VAD, SpeechBrain ECAPA, llama.cpp, ffmpeg, and Fish Audio for the voices.

Everything in this repository was written during the hackathon (Oct 27 to Nov 4, 2026). I had an earlier, unrelated desktop prototype with the same name in October 2026; none of its code is used here.

## License

MIT for this repository. The models and libraries keep their own licenses (Muse Glimmer and Omnilingual ASR: Apache-2.0; Demucs: MIT).
