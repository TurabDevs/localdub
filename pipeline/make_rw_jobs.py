#!/usr/bin/env python3
"""v2 rewrite job for Muse Glimmer: QA-flagged lines (overflow / tight timing / ASR-garbled meaning) go back to Glimmer
with the script context, the v1 English, the measured timing and a QA note. Output jobs_rw.json (stage2 format)."""
import json, re, sys
qa = {q["id"]: q for q in json.load(open("build/out/qa.json"))["lines"]}
L = {l["id"]: l for l in json.load(open("build/work/s2/run1/lines.json"))}
def syl(t): return sum(max(1, len(re.findall(r"[aeiouy]+", w.lower()))) for w in re.findall(r"[A-Za-z']+", t))
NOTES = {
 "L004": "v1 used the short variant and lost the bartender joke; there is room (slot 8.1s, v1 audio 5.0s). Keep: step out, couple of beers kill the bacteria, ever seen a bartender get typhoid? because they live around beer. don't overthink.",
 "L006": "ASR garbled the middle. Likely meaning: bacteria die on their own in two months. 'you lost the promotion? no? then you got typhoid?' Day before yesterday at Sahil's party you were enjoying so much, laughing, dancing, joking, and now suddenly this? Keep it inside the slot.",
 "L007": "ASR garbled: 'टेंडर' after 'Garima broke up with you, you should get ...' is almost certainly 'Tinder' (the joke: get on Tinder and your typhoid, vomiting, diarrhea will all go away).",
 "L024": "Meaning fix: 'तो क्या है तो पढ़ा लिखा है ना' = 'come on, you're an educated guy, right?' (he mocks doctors' made-up names like 'typhoid-shmyphoid').",
 "L038": "Meaning fix: the subject is the typhoid: 'It won't even know where to go! Where to go? Take a left up ahead!' then 'I think you're overthinking, over-analyzing, overhyping'.",
}
targets = []
for i, q in sorted(qa.items()):
    if q["variant"] == "original": continue
    tight = q["tts"] > q["slot"] or q.get("tempo", 1) > 1.0 or not q["fits"]
    if tight or i in NOTES:
        budget = round(syl(q["en"]) * min(1.0, q["slot"] / q["tts"]) * 0.92) if tight else None
        note = NOTES.get(i, "")
        if tight: note = (note + " " if note else "") + f"TOO LONG: v1 audio {q['tts']}s for a {q['slot']}s slot. Write it in about {budget} syllables."
        targets.append((i, note))
TID = {t[0] for t in targets}
ctx = []
for i, q in sorted(qa.items()):
    if q["variant"] == "original": ctx.append(f"{i} ({q['slot']}s) [kept original English audio]: {L[i]['hi']}")
    elif i in TID: ctx.append(f"{i} ({q['slot']}s) HI: {L[i]['hi']} || v1 EN: {q['en']}")
    else: ctx.append(f"{i} ({q['slot']}s) v1 EN: {q['en']}")
jobs = []
for sh in (0, 1):
  tl = "\n".join(f"- {i}: {n}" for k, (i, n) in enumerate(targets) if k % 2 == sh)
  user = ("SCRIPT (line id, slot, Hindi ASR, the v1 English dub line):\n" + "\n".join(ctx) +
        "\n\nQA found problems in these lines of the v1 dub. Rewrite ONLY these lines:\n" + tl +
        "\n\nRules: spoken, funny, natural American English; keep the joke and the speaker's attitude; must fit the slot when spoken "
        "at a normal pace (respect the syllable target); no stage directions. Return JSON only: "
        '{"lines":[{"id":"L006","en_fit":"...","emotion":"neutral|happy|laughing|sarcastic|angry|shouting|sad|worried|excited|whisper"}]}')
  jobs.append({"id": f"rw_0{sh}", "json": True, "reasoning": "low", "max_tokens": 3000, "temperature": 0.7,
       "messages": [{"role": "system", "content": "You are LocalDub's dubbing writer and script doctor. You fix lines of an English dub of a Hindi/Urdu comedy sketch so they keep the joke and fit the lip-timed slot. The Hindi comes from an automatic transcript that can contain errors: use the whole script to infer what was really said."},
                    {"role": "user", "content": user}]})
json.dump(jobs, open("build/work/s2/run2/jobs_rw.json", "w"), ensure_ascii=False, indent=1)
print(len(targets), "targets:", [t[0] for t in targets]); print(len(user), "chars")
