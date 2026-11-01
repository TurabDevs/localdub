#!/usr/bin/env python3
"""After kernel localdub-rw3: read Glimmer's rw_30 output, and for the phone-call line L002 (two speakers in one VAD chunk,
split in build/work/s2/run5/lines.json at 20.22 s where the pitch drops from ~175 Hz to ~125 Hz) split Glimmer's L002
sentence at the turn ("No man" / "Nah man" ...) into L002a / L002b. Writes build/work/s2/run3/split.json (rw_31)."""
import json, re, sys
R = json.load(open("build/work/s2/run3/rw3/results.json"))
c = R["rw_30"]["content"]; c = re.sub(r"^```(json)?|```$", "", c.strip(), flags=re.M).strip()
j = json.loads(re.search(r"\{.*\}", c, re.S).group(0))
for x in j["lines"]: print(x["id"], "|", x.get("en_fit"), "|", x.get("emotion"))
L2 = next((x for x in j["lines"] if x["id"] == "L002"), None)
out = []
if L2:
    t = L2["en_fit"]; m = re.search(r"(?i)\b(no,? man|nah,? man|no,? bro|nah,? bro|no,? dude|nah|no)\b", t[5:])
    if m:
        k = m.start() + 5; a, b = t[:k].strip(), t[k:].strip()
        out = [{"id": "L002a", "en_fit": a, "emotion": "excited"}, {"id": "L002b", "en_fit": b, "emotion": "worried"}]
print("split:", out)
json.dump({"rw_31": {"content": json.dumps({"lines": out})}, "tr_zz_split": {"content": json.dumps({"lines": [{"id": o["id"], "emotion": o["emotion"]} for o in out]})}}, open("build/work/s2/run3/split.json", "w"), ensure_ascii=False)
