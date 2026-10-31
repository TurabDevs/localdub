# LocalDub

**Hindi/Urdu video in, natural English dub out — every AI model that understands the video is a Meta open-weight model, running on our own compute.**

> Meta Global AI Developer Hackathon 2026 · Indie Developer track · Theme: *Build for Your World* · Built by Muhammad Asim

## Why
Hundreds of millions of people in Pakistan and India make video in Hindi, Urdu and Punjabi, and almost none of it reaches viewers who don't speak those languages. Human dubbing costs too much for independent creators. Most AI dubs translate jokes word for word, give everyone the wrong voice, and play speech where nobody's lips are moving. LocalDub is built to fix those three things for South Asian comedy and talk content.

## How it works

```
video ──► Demucs (Meta) ──► vocals ──► Silero VAD ──► chunks ──► Omnilingual ASR (Meta) ──► Hindi script
            │                                  │
            └─► music + effects bed            └─► ECAPA voice embeddings ──► voice clusters
                                                                        │
proxy frames ──► Muse Glimmer VISION (Meta): who is on screen for each voice? gender / age / role ──┤
                                                                        ▼
Hindi script (whole thing as context) ──► Muse Glimmer TEXT (Meta): humor-preserving English, full + short variant,
                                          emotion per line, syllable budget from the original slot
                                                                        ▼
per-line zero-shot voice (reference = that speaker's own separated original line) ──► TTS
──► fit to the original slot (variant choice, tempo only 0.93–1.08) ──► mix over the original music/effects ──► MP4 + QA report
                                                                        ▼
LISTEN / QA pass: timing, overflow, loudness and pitch of every dubbed line vs the original line (qa_metrics.py),
optional re-transcription of the dub with Omnilingual ASR (stage_listen_check.py)
──► flagged lines go back to Muse Glimmer for a rewrite (make_rw_jobs.py) ──► re-dub
```

| Stage | Model / tool | Where it runs |
|---|---|---|
| Source separation | **Meta Demucs** `htdemucs_ft` (MIT) | Kaggle kernel, offline |
| Speech recognition | **Meta Omnilingual ASR** `omniASR_CTC_1B_v2` (Apache-2.0), via a parity-verified Hugging Face `Wav2Vec2ForCTC` conversion | Kaggle kernel, offline |
| Cast labels per line (vision: who is on screen, gender, age, role) | **Meta Muse Glimmer 30B** (Apache-2.0) + its perception-encoder projector, GGUF Q4_K_M, `llama.cpp` | Kaggle kernel, offline |
| Translation (full + short variant + emotion tag per line) | **Meta Muse Glimmer 30B** | Kaggle kernel, offline |
| Rewrite of QA-flagged lines (timing / meaning) | **Meta Muse Glimmer 30B** | Kaggle kernel, offline |
| Listen check: re-transcribe the English dub | **Meta Omnilingual ASR** | Kaggle kernel, offline |
| Speech segmentation | Silero VAD (MIT) | Kaggle kernel |
| Voice clustering | SpeechBrain ECAPA (Apache-2.0) | Kaggle kernel |
| Voice synthesis | Fish Audio API (`s2.1-pro-free`), zero-shot clone per character | API call |
| Fit + mix | ffmpeg, numpy | local |

No hosted Meta API is used. Muse Glimmer weights are checked by sha256 against the official `meta-models/Muse-Glimmer-30B-GGUF` release before the server starts.

## Repository layout
- `cloud/kaggle/stage1_listen.py` – separation, VAD, ASR, speaker embeddings (offline Kaggle kernel)
- `cloud/kaggle/stage2_glimmer.py` – Muse Glimmer server + job runner (text and image jobs, resumable, shardable)
- `cloud/kaggle/kapi.py`, `cloud/kaggle/kup.py` – minimal Kaggle API client (datasets, kernels, chunked uploads)
- `pipeline/make_jobs.py` – voice clusters → dubbing lines → Glimmer cast and translation jobs
- `pipeline/dub.py` – per-line TTS, slot fitting, mixing, QA report (v2: ASR-garble guard, pitch-matched references, rewrite variants)
- `pipeline/qa_metrics.py` – listen/QA metrics per dubbed line (fit, spill, loudness, pitch vs original, optional ASR WER)
- `pipeline/make_rw_jobs.py` – builds the Muse Glimmer rewrite job for QA-flagged lines
- `pipeline/listen_prep.py`, `cloud/kaggle/stage_listen_check.py` – Omnilingual ASR re-transcription of the dub

## Demo
Demo video v2 (2:15): https://github.com/TurabDevs/localdub/releases/download/demo-v2/localdub_demo_v2.mp4
Full 5-minute English dub: v3 (latest) https://github.com/TurabDevs/localdub/releases/download/demo-v2/aib_depression_en_v3.webm · v2 https://github.com/TurabDevs/localdub/releases/download/demo-v2/aib_depression_en_v2.webm
Demo video v1: https://github.com/TurabDevs/localdub/releases/download/demo-v1/localdub_demo.mp4

### Results: dub → listen → fix (measured, `results/`)
| metric (5 min 08 s clip, 58 speech lines) | v1 | v2 |
|---|---|---|
| dubbed lines fitting their slot | 42 / 44 | **32 / 32** |
| overflow | 2 | **0** |
| one-word filler dubs from ASR garble | 8 | **0** |
| pitch mismatch vs original > 4 semitones | 11 | **3** |
| mean loudness gap vs original line | 1.65 dB | **0.45 dB** |
| max tempo change | 1.08x | 1.127x |
v2 dubs 32 lines: 14 lines were already English in the original and 12 were laughs/grunts/short English words that the ASR garbled, and these 26 keep the original audio. Muse Glimmer rewrote 20 QA-flagged lines (18 used). Details and every rewrite: `results/qa_v2.md`. Iteration 2 (v3, `results/qa_v3.md`) re-voiced the 3 remaining pitch outliers with a different reference: pitch mismatches 3 → 1, mean loudness gap 0.45 → 0.35 dB.

Source clip: "If People Treated Other Illnesses Like They Treat Depression" by **All India Bakchod (AIB)**, licensed [CC BY 3.0](https://creativecommons.org/licenses/by/3.0/). Changes made: the dialogue was translated into English and re-voiced by LocalDub; music and effects come from the original via source separation.

## Built during the hackathon
Everything in this repository was written during the hackathon (Oct 27 – Nov 4, 2026): the Kaggle stages, the Muse Glimmer job runner, the cast-detection and translation prompts, the fitting and mixing pipeline, and this README. An earlier, unrelated LocalDub desktop-app prototype (October 2026) is **not** part of this submission and no code from it is used here.

## License
MIT (this repository). Models and libraries keep their own licenses (Muse Glimmer: Apache-2.0; Omnilingual ASR: Apache-2.0; Demucs: MIT).
