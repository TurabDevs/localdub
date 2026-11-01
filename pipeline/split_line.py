#!/usr/bin/env python3
"""split_line.py <lines.json> <out.json> <line_id> <cut_s> <gap_s> <hindi_marker>
Split one dubbing line that holds two speakers (e.g. both sides of a phone call in one VAD chunk) into <id>a / <id>b,
so each half gets its own original audio as the voice-cloning reference. cut_s = where the first speaker stops
(found by looking at the pitch track: ~175 Hz caller vs ~125 Hz sick friend for L002), gap_s = start of the second
speaker after the cut, hindi_marker = first words of the second speaker in the ASR text."""
import json, sys
src, out, lid, cut, gap, mark = sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4]), float(sys.argv[5]), sys.argv[6]
res = []
for x in json.load(open(src)):
    if x["id"] != lid: res.append(x); continue
    k = x["hi"].index(mark)
    a = dict(x, id=lid + "a", end=cut, hi=x["hi"][:k].strip()); a["dur"] = round(cut - a["start"], 2)
    b = dict(x, id=lid + "b", start=gap, hi=x["hi"][k:].strip()); b["dur"] = round(b["end"] - gap, 2)
    res += [a, b]; print(a["id"], a["start"], a["end"], "|", b["id"], b["start"], b["end"])
json.dump(res, open(out, "w"), ensure_ascii=False, indent=1)
