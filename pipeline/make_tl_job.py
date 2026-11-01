#!/usr/bin/env python3
"""Build the Muse Glimmer job that writes each Urdu dub line in Devanagari letters for the voice.

usage: make_tl_job.py <glimmer results.json> <out jobs_tl.json>
Same words, same pronunciation, only the script changes (no Sanskritised Hindi, English loanwords as pronounced).
Why: on Nov 1 an ASR round trip showed Fish Audio reads an Urdu sentence much more clearly from Devanagari than from Urdu
script or Roman Urdu. dub.py uses the result with TTS_SCRIPT=deva; subtitles keep the Urdu script."""
import json, re, sys
res, out = sys.argv[1:3]
R = json.load(open(res)); L = []
for k, v in sorted(R.items()):
    if not k.startswith("tr_"): continue
    c = v.get("content") or ""
    try: j = json.loads(c)
    except Exception:
        m = re.search(r"\{.*\}", c, re.S); j = json.loads(m.group(0)) if m else {}
    L += [x for x in j.get("lines", []) if x.get("text")]
rows = "\n".join(f'{x["id"]}: text: {x["text"]} | short: {x.get("text_short", "")} | roman: {x.get("roman", "")}' for x in L)
job = [{"id": "tl_00", "json": True, "reasoning": "low", "max_tokens": 2500, "temperature": 0.2, "messages": [
    {"role": "system", "content": "You transliterate spoken Urdu dub lines from Urdu script into Devanagari script for a text-to-speech voice. "
     "Keep exactly the same words and pronunciation (Hindustani spelling as it is spoken in Pakistan), do not translate into Sanskritised Hindi, "
     "do not change or add words. English loanwords are written in Devanagari as they are pronounced (robot = रोबोट, space = स्पेस). Names stay as pronounced."},
    {"role": "user", "content": "Lines (id: Urdu text | short version | Roman Urdu as a pronunciation hint):\n" + rows +
     "\n\nReturn JSON only: {\"lines\":[{\"id\":\"L000\",\"deva\":\"<text in Devanagari>\",\"deva_short\":\"<short version in Devanagari>\"}]} with one entry per id."}]}]
json.dump(job, open(out, "w"), ensure_ascii=False); print(len(L), "lines")
