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
