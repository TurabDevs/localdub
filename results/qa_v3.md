# LocalDub v3 QA (iteration 2: pitch outliers, queue item Q5) — built 2026-10-31 by builder-3
Base: v2 (build/out/v2). Change: manual QA reference override for the 3 v2 pitch outliers via new env PREF_OVERRIDE in dub.py:
`PREF_OVERRIDE=L031:L020,L037:L035,L048:L004` (next-closest-pitch long lines; L031/L037 same voice cluster). Only those 3 lines re-voiced.
| metric | v2 | v3 |
|---|---|---|
| fit / overflow | 32/32, 0 | 32/32, 0 |
| pitch mismatch > 4 st | 3 (L031 8.7, L037 -4.7, L048 9.4) | **1** (L031 0.9, L037 3.5, L048 6.8) |
| mean pitch gap | 2.33 st | **1.97 st** |
| mean loudness gap | 0.45 dB | **0.35 dB** |
| max tempo / spill | 1.127x / 0.52 s | same |
Not yet shipped: v2 is what the CV submission + demo-v2 release show (the demo clips at 17.3-32.3 s and 100-140 s do not contain L031/L037/L048, so the demo audio is identical). To ship: upload build/out/v3/aib_depression_en.webm as release asset, push results, optionally update the CV numbers (pitch 11 -> 1).


### Listen WER v1 vs v2 vs v3 (kernel localdub-listenk-v13 for v1+v3, ~8 min)
| | v1 | v2 | v3 |
|---|---|---|---|
| mean WER, all dubbed lines | 0.289 (44 lines) | 0.187 (32) | **0.144** (32) |
| lines with WER > 0.5 | 10 | 4 | **3** |
| mean WER on the same 32 lines v2 dubs | **0.116** | 0.187 | 0.144 |
Honest reading: most of v1's errors were the 12 filler/garble lines v2 no longer dubs. On the 32 lines both versions dub, v2/v3 are slightly LESS intelligible to the ASR than v1, driven by L030 ("Put it on." heard "vtdn"; v1 "Wear it like this." was heard right but overflowed), L034 (new pitch-matched reference gave Hindi-accented English, heard "moर poetry moर crाinग") and L004 ("beers" heard "beards"). Real gains: L038 (0.35 -> 0.05), L037 (0.5 -> 0.0 in v3), L031 (v3 0.0), L024 (0.08 -> 0.0). Next fix (v4): L034 back to the cluster reference, fresh take for L030.