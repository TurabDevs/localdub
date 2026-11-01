#!/usr/bin/env python3
"""rw3: Muse Glimmer rewrite for lines inside the demo clip windows (17-32 s, 100-140 s) that read wrong on review.
Context is trimmed to L000-L026 so the CPU kernel finishes faster. QA notes below are review notes on the likely
meaning (written by the builder after reading the Hindi ASR), the same way as the v2 rewrite pass."""
import json, re
qa = {q["id"]: q for q in json.load(open("build/out/v4/qa.json"))["lines"]}
L = {l["id"]: l for l in json.load(open("build/work/s2/run1/lines.json"))}
NOTES = {
 "L002": "Phone call. Caller: 'Hello bro, I'm booking the ten-thirty movie, okay?' Other side: 'No man, don't book mine, I've been throwing up since morning and I have a really high fever.' The current dub dropped 'Hello bro' and 'ten-thirty'. Keep both halves, fit the slot.",
 "L003": "Meaning fix, this is the joke of the whole sketch (people treat typhoid the way they treat depression): 'टाइफाइड ऐसा कुछ नहीं होता भाई' = 'there's no such thing as typhoid, bro', not 'typhoid isn't like that'. Keep: 'Typhoid and all that? Are you crazy? ... you just need to get out and have some fun.'",
 "L020": "The last word in the ASR is garbled; do not invent 'no problem'. Meaning: 'there are separate doctors for these fever-type things' (said like it is a shameful secret).",
 "L021": "Meaning fix: 'पूरे मोहल्ले में बताना है?' is a sarcastic QUESTION: 'You want the whole neighbourhood to know?' The current dub 'Tell the whole neighborhood.' says the opposite. Very short slot (1.2 s), about 6-7 syllables.",
 "L022": "'उसका माथा गरम रहता है' = 'his forehead is always hot'. Keep the repeated 'fever, fever, fever' and end with 'What will people think?'",
 "L023": "Drop the filler 'seriously'. Meaning: 'People will say the whole family has fever, bro. They tell you it's typhoid, these doctors are all running a scam just to squeeze money out of you.'",
}
def syl(t): return sum(max(1, len(re.findall(r"[aeiouy]+", w.lower()))) for w in re.findall(r"[A-Za-z']+", t))
ids = sorted(L)[:27]
ctx = []
for i in ids:
    q = qa[i]
    if q["variant"] == "original": ctx.append(f"{i} ({q['slot']}s) [kept original audio]: {L[i]['hi']}")
    elif i in NOTES: ctx.append(f"{i} ({q['slot']}s) HI: {L[i]['hi']} || current EN: {q['en']}")
    else: ctx.append(f"{i} ({q['slot']}s) EN: {q['en']}")
tl = "\n".join(f"- {i} (slot {qa[i]['slot']}s, about {round(qa[i]['slot']*2.9)} syllables max): {n}" for i, n in NOTES.items())
user = ("SCRIPT, first 27 lines (line id, slot seconds, Hindi ASR, current English dub):\n" + "\n".join(ctx) +
        "\n\nA review of the English dub found problems in these lines. Rewrite ONLY these lines:\n" + tl +
        "\n\nRules: spoken, funny, natural English; keep the joke and the speaker's attitude; must fit the slot at a normal "
        "speaking pace; no stage directions. Return JSON only: "
        '{"lines":[{"id":"L003","en_fit":"...","emotion":"neutral|happy|laughing|sarcastic|angry|shouting|sad|worried|excited|whisper"}]}')
jobs = [{"id": "rw_30", "json": True, "reasoning": "low", "max_tokens": 1500, "temperature": 0.6,
         "messages": [{"role": "system", "content": "You are LocalDub's dubbing writer and script doctor. You fix lines of an English dub of a Hindi/Urdu comedy sketch (a parody where people treat typhoid the way they usually treat depression) so they keep the joke and fit the lip-timed slot. The Hindi comes from an automatic transcript that can contain errors: use the script to infer what was really said."},
                      {"role": "user", "content": user}]}]
import os; os.makedirs("build/work/s2/run3", exist_ok=True)
json.dump(jobs, open("build/work/s2/run3/jobs_rw3.json", "w"), ensure_ascii=False, indent=1)
print(len(NOTES), "targets", len(user), "chars")
