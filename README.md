# LocalDub

**Hindi/Urdu video in, natural English dub out — every AI model that understands the video is a Meta open-weight model, running on our own compute.**

> Meta Global AI Developer Hackathon 2026 · Indie Developer track · Theme: *Build for Your World*

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
per-character zero-shot voice (reference = that character's own separated vocals) ──► TTS
──► fit to the original slot (variant choice, tempo only 0.93–1.08) ──► mix over the original music/effects ──► MP4 + QA report
```

| Stage | Model / tool | Where it runs |
|---|---|---|
| Source separation | **Meta Demucs** `htdemucs_ft` (MIT) | Kaggle kernel, offline |
| Speech recognition | **Meta Omnilingual ASR** `omniASR_CTC_1B_v2` (Apache-2.0), via a parity-verified Hugging Face `Wav2Vec2ForCTC` conversion | Kaggle kernel, offline |
| Cast detection (vision) | **Meta Muse Glimmer 30B** (Apache-2.0) + its perception-encoder projector, GGUF Q4_K_M, `llama.cpp` | Kaggle kernel, offline |
| Translation, timing fit, rewrite | **Meta Muse Glimmer 30B** | Kaggle kernel, offline |
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
- `pipeline/dub.py` – per-character TTS, slot fitting, mixing, QA report

## Demo
Demo video: _link added at submission_.

Source clip: "If People Treated Other Illnesses Like They Treat Depression" by **All India Bakchod**, licensed [CC BY 3.0](https://creativecommons.org/licenses/by/3.0/). Changes made: the dialogue was translated into English and re-voiced by LocalDub; music and effects come from the original via source separation.

## Built during the hackathon
Everything in this repository was written during the hackathon (Oct 27 – Nov 4, 2026): the Kaggle stages, the Muse Glimmer job runner, the cast-detection and translation prompts, the fitting and mixing pipeline, and this README. An earlier, unrelated LocalDub desktop-app prototype (October 2026) is **not** part of this submission and no code from it is used here.

## License
MIT (this repository). Models and libraries keep their own licenses (Muse Glimmer: Apache-2.0; Omnilingual ASR: Apache-2.0; Demucs: MIT).
