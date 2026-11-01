#!/usr/bin/env python3
"""AV-QA face <-> voice check: does the dub speak in the voice of the person whose mouth is moving?

usage: char_mismatch.py <qa.json> <lines.json> <speakers.json> <chars_dir> <out.json>

For every dubbed line, the voice identity is the character whose original speech the cloning reference came from
(qa.json "ref": own_line / pitch_matched:<line> / cluster / char:<Cn>); a reference that mixes two characters is
reported as a blended voice. The on-screen speaker comes from char_link.py: speech units where the active-speaker
detector clearly saw one face talking. A mismatch is a face-confirmed unit inside a dubbed line whose character is not
the voice's character ("voice of A while B's mouth moves"). Off-screen units are not counted either way.
Limits: the active-speaker detector is the same one AV-QA uses (moderate reliability on side faces / dark shots), so
counts are an estimate; every mismatch is listed with its time so it can be checked by eye.
"""
import json, sys, collections
qa_f, lines_f, spk_f, CD, out_f = sys.argv[1:6]
QA = {x["id"]: x for x in json.load(open(qa_f))["lines"]}
LINES = {x["id"]: x for x in json.load(open(lines_f))}
SPK = json.load(open(spk_f))
U = json.load(open(f"{CD}/units.json"))
def ov(a0, a1, b0, b1): return max(0.0, min(a1, b1) - max(a0, b0))
def who(spans):
    c = collections.Counter()
    for a, b in spans:
        for u in U: c[u["cid"]] += ov(a, b, u["start"], u["end"])
    tot = sum(c.values())
    return c, tot
rows = []; mis_s = 0.0; mis_lines = 0; blended = 0; checked_s = 0.0
for lid, q in QA.items():
    if q.get("variant") in (None, "original", "orig", "keep") or not q.get("en") or lid not in LINES: continue
    L = LINES[lid]; ref = q.get("ref") or "cluster"
    if ref.startswith("char:"): voice = ref[5:]; share = 1.0
    else:
        if ref == "own_line": spans = [(L["start"], min(L["end"], L["start"] + 10.0))]
        elif ref.startswith("pitch_matched:"):
            M = LINES.get(ref.split(":", 1)[1]); spans = [(M["start"], min(M["end"], M["start"] + 10.0))] if M else []
        else:
            spans = []; tot = 0.0
            for s in SPK.get(q.get("spk") or L.get("spk"), {}).get("longest", []):
                spans.append((s["start"], min(s["end"], s["start"] + 7.0))); tot += spans[-1][1] - spans[-1][0]
                if tot > 9: break
        c, tot = who(spans)
        if not c: voice, share = None, 0.0
        else: voice, v = c.most_common(1)[0]; share = v / tot
    bl = share < 0.75
    blended += bl
    lm = 0.0; bad = []
    for u in U:
        if u.get("how") != "face": continue
        o = ov(L["start"], L["end"], u["start"], u["end"])
        if o < 0.5 * (u["end"] - u["start"]): continue
        checked_s += o
        if u["cid"] != voice: lm += o; bad.append({"t": u["start"], "speaker_on_screen": u["cid"], "text": u["text"][:60]})
    mis_s += lm; mis_lines += lm > 0.3
    rows.append({"id": lid, "ref": ref, "voice_char": voice, "voice_share": round(share, 2), "blended_voice": bl,
                 "mismatch_s": round(lm, 2), "mismatches": bad})
res = {"dubbed_lines": len(rows), "lines_with_mismatch": mis_lines, "mismatch_s": round(mis_s, 1),
       "face_confirmed_s_checked": round(checked_s, 1), "blended_voice_lines": blended, "lines": rows}
json.dump(res, open(out_f, "w"), ensure_ascii=False, indent=1)
print({k: v for k, v in res.items() if k != "lines"})
