# LocalDub v4 QA (iteration 3) — built 2026-10-31
Base: v3. Changes: `PREF_OVERRIDE=L031:L020,L037:L035,L048:L004,L034:cluster` (L034 back to the cluster reference) plus a fresh Fish TTS take for L030 ("Put it on.").
| metric (32 dubbed lines) | v1 | v2 | v3 | v4 |
|---|---|---|---|---|
| mean listen WER (Omnilingual ASR re-transcription), same 32 lines | 0.116 | 0.187 | 0.144 | **0.123** |
| lines with WER > 0.5 | – | 4 | 3 | **2** (L013, L034) |
| pitch mismatch > 4 st / mean gap | 11 / 3.72 st | 3 / 2.33 | 1 / 1.97 | **1 / 1.84 st** |
| loudness mean gap / lines > 6 dB | 1.65 dB / 4 | 0.45 / 0 | 0.35 / 0 | 0.63 dB / 1 |
| fit / overflow | 42/44, 2 | 32/32, 0 | 32/32, 0 | 32/32, 0 |
Honest notes: L030 is fixed (heard "put it on", WER 0.0). L034 is still scored 1.25 but the ASR output ("मोरe पoetरy मोरe क्राinग") reads as "more poetry more cringe" in mixed script, so the words are there but the accent is strong. L013 ("typhoid" spelled "tipoid") is ASR spelling noise. One line (L030) is more than 6 dB off in loudness (gain clipped at 0.5).
