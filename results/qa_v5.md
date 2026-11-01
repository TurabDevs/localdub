# v5: fixing what's wrong inside the demo clips (Nov 1, 2026)

The demo video shows two windows of the dub (17-32 s and 100-140 s), so that's what people hear first. I went through every line in those windows against the Hindi and found problems the metrics didn't catch, because they were meaning problems, not timing problems:

- **L002 (phone call)**: both people are inside one speech chunk, so one cloned voice said both sides. The dub also dropped "Hello bro" and "ten-thirty".
- **L003**: "Typhoid isn't like that, bro" - the Hindi says "there's no such thing as typhoid, bro". That line is the joke of the whole sketch (people treat typhoid the way they treat depression), and the dub lost it.
- **L021**: "Tell the whole neighborhood." - the Hindi is a sarcastic question, "you want the whole neighbourhood to know?". The dub said the opposite.
- **L020**: ended with "no problem", which isn't in the Hindi (the last ASR word is garbled).
- **L022 / L023**: small fixes ("forehead always hot", dropped a filler "seriously").

## What I changed

1. Split L002 into L002a (17.54-20.22 s) and L002b (20.30-24.86 s) at the speaker change. The pitch track shows it clearly: ~175 Hz for the excited caller, ~125 Hz for the sick friend. Each half now clones its own speaker. (`pipeline/split_line.py`)
2. Sent the six lines back to Muse Glimmer (Kaggle CPU kernel `localdub-rw3`, 27 min) with the first 27 lines of the script as context and a review note per line on what the Hindi means. Glimmer's rewrites:
   - L002: "Hello bro, ten-thirty movie, okay? / No man don't book mine, puking since morning, high fever." (split at "No man" into L002a / L002b)
   - L003: "Typhoid and stuff? Are you crazy? There's no such thing, bro, just get out and have fun."
   - L020: "There are separate doctors for these fever things."
   - L021: "Whole neighbourhood to know?"
   - L022: "Our son has fever, fever, fever, temperature up, forehead always hot. What will people think?"
   - L023: "People'll say whole family has fever, bro. They say typhoid, these doctors run a scam just to squeeze money."
3. Listen check (Omnilingual ASR, kernel `localdub-listenk-v5`). Two lines came out bad: L021 was gibberish to the ASR and L002a garbled "ten-thirty". I generated 4 more takes of each and ran them all through the listen check in one kernel (`localdub-listenk-takes5`), then picked by intelligibility and pitch:
   - L002a: take 4 (heard "hello bro ten thirty movi okay", pitch +3.9 st). Take 3 was heard perfectly but was 6.4 st too high, so I didn't use it.
   - L021: take 3 (heard "hole neighborhood to no" - the right sounds, homophones; +5.0 st). It was the only take the ASR could make out at all.

## Numbers (honest)

| | v4 | v5 |
|---|---|---|
| dubbed lines (L002 now counts as two) | 32 | 33 |
| fit / overflow | 32/32, 0 | 33/33, 0 |
| pitch > 4 st off / mean gap | 1 / 1.84 st | 2 (L021, L048) / 1.88 st |
| loudness mean gap / lines > 6 dB | 0.63 dB / 1 | 0.61 dB / 1 |
| listen WER, all dubbed lines | 0.123 (32) | 0.151 (33) |
| listen WER, the same 31 lines | 0.123 | 0.152 |
| listen WER, demo-window lines | 0.059 | 0.168 |
| spill into next line | 2 lines, 0.52 s | 3 lines, 0.58 s (new: L002a 0.06 s into L002b) |

By the automatic numbers, v5 is slightly worse than v4. I'm shipping it anyway as the demo version, and here's why: the WER went up mostly because of how the ASR spells things ("hole ... to no", "our sun", "hart", "people'll" counted as a miss for "people will", "brow" for "bro"), while v4's clean scores in these windows were on lines that said the wrong thing. A metric that only checks whether the words came through can't see that "Tell the whole neighborhood" means the opposite of the Hindi. v4 is still in the release if anyone wants the version with better scores.

Files: `qa_v5.json`, `metrics_v5.json`, `listen_v5.json` in `results/`. Full dub: https://github.com/TurabDevs/localdub/releases/download/demo-v2/aib_depression_en_v5.webm
