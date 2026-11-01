#!/usr/bin/env python3
"""Split dubbing lines that hold more than one character (from char_link.py) into one sub-line per speaker turn, and
build the Muse Glimmer job that writes the English for each turn.

usage: make_char_split.py <lines.json> <qa.json of the current version> <chars_out_dir> <run_dir_out> [nshard]
writes run_dir_out/lines.json (split lines get ids <id>_1, <id>_2, ...), run_dir_out/charmap.json (line id -> Cn),
run_dir_out/split.json (what was split and why), run_dir_out/jobs_s<k>.json (Glimmer rewrite jobs rw_7<k>)
A turn shorter than 0.8 s is folded into its neighbour (too short to voice on its own)."""
import json, os, re, sys
lines_f, qa_f, CD, OUT = sys.argv[1:5]; NSH = int(sys.argv[5]) if len(sys.argv) > 5 else 2
os.makedirs(OUT, exist_ok=True)
LINES = json.load(open(lines_f)); QA = {q["id"]: q for q in json.load(open(qa_f))["lines"]}
LC = json.load(open(f"{CD}/line_chars.json"))
def syl(t): return sum(max(1, len(re.findall(r"[aeiouy]+", w.lower()))) for w in re.findall(r"[A-Za-z']+", t))
out, cmap, split = [], {}, []
for L in LINES:
    v = LC.get(L["id"]); q = QA.get(L["id"], {})
    dubbed = q.get("variant") not in (None, "original")
    if v and v["main"]: cmap[L["id"]] = v["main"]
    turns = [dict(t) for t in (v["turns"] if v else [])]
    # fold short turns into the neighbour
    changed = True
    while changed and len(turns) > 1:
        changed = False
        for i, t in enumerate(turns):
            if t["end"] - t["start"] < 0.8:
                j = i - 1 if i > 0 else i + 1
                a, b = sorted([i, j]); m = turns[a] if turns[a]["end"] - turns[a]["start"] >= turns[b]["end"] - turns[b]["start"] else turns[b]
                turns[a] = {"cid": m["cid"], "start": turns[a]["start"], "end": turns[b]["end"], "text": turns[a]["text"] + " " + turns[b]["text"], "how": turns[a]["how"] + turns[b]["how"]}
                del turns[b]; changed = True; break
        # merge neighbours that ended up with the same character
        k = 0
        while k < len(turns) - 1:
            if turns[k]["cid"] == turns[k + 1]["cid"]:
                turns[k]["end"] = turns[k + 1]["end"]; turns[k]["text"] += " " + turns[k + 1]["text"]; del turns[k + 1]; changed = True
            else: k += 1
    if not dubbed or len(turns) < 2: out.append(L); continue
    subs = []
    for i, t in enumerate(turns):
        s0 = max(L["start"], t["start"]) if i else L["start"]; s1 = min(L["end"], t["end"]) if i < len(turns) - 1 else L["end"]
        sid = f"{L['id']}_{i + 1}"
        subs.append(dict(L, id=sid, start=round(s0, 3), end=round(s1, 3), dur=round(s1 - s0, 2), hi=t["text"], spk=L["spk"], parent=L["id"]))
        cmap[sid] = t["cid"]
    out += subs
    split.append({"id": L["id"], "en_v6": q.get("en"), "hi": L["hi"], "turns": [{"id": s["id"], "char": cmap[s["id"]], "start": s["start"], "end": s["end"], "hi": s["hi"]} for s in subs]})
json.dump(out, open(f"{OUT}/lines.json", "w"), ensure_ascii=False, indent=1)
json.dump(cmap, open(f"{OUT}/charmap.json", "w"), indent=1)
json.dump(split, open(f"{OUT}/split.json", "w"), ensure_ascii=False, indent=1)
# Glimmer jobs: whole script as context, then the turns to write
ctx = []
for L in LINES:
    q = QA.get(L["id"], {})
    ctx.append(f"{L['id']} [{LC.get(L['id'], {}).get('main') or '?'}]: " + (f"EN: {q['en']}" if q.get("en") else f"(kept original) {L['hi'][:80]}"))
for k in range(NSH):
    part = split[k::NSH]
    if not part: continue
    todo = []
    for x in part:
        todo.append(f"- {x['id']} (current English for the whole line: \"{x['en_v6']}\"). It is really {len(x['turns'])} speakers taking turns:")
        for t in x["turns"]:
            d = t["end"] - t["start"]
            todo.append(f"    {t['id']}: speaker {t['char']}, {d:.1f} s (about {max(2, round(d * 2.9))} syllables max), Hindi ASR: {t['hi']}")
    user = ("SCRIPT (line id [character id]: current English dub):\n" + "\n".join(ctx) +
            "\n\nAn audio/video check found that these lines are not one person talking: the characters take turns inside the line "
            "(character ids come from face tracking + voice matching, the same id is the same person everywhere in the script). "
            "Each turn will now be voiced by its own character. Write the English for EACH turn so the conversation still reads "
            "naturally and keeps the joke, and each turn fits its time:\n" + "\n".join(todo) +
            "\n\nRules: spoken, natural, funny English; keep each speaker's attitude; every turn id gets its own text (no empty "
            "turns); fit the syllable budget; no stage directions. The Hindi is an automatic transcript and can contain errors: use "
            "the script around it. Return JSON only: "
            '{"lines":[{"id":"L006_1","en_fit":"...","emotion":"neutral|happy|laughing|sarcastic|angry|shouting|sad|worried|excited|whisper"}]}')
    jobs = [{"id": f"rw_7{k}", "json": True, "reasoning": "low", "max_tokens": 2200, "temperature": 0.5,
             "messages": [{"role": "system", "content": "You are LocalDub's dubbing writer. You write the English dub of a Hindi/Urdu comedy sketch (a parody where people treat typhoid the way they usually treat depression), line by line, so each character keeps their voice, the jokes survive, and every line fits its lip-timed slot."},
                          {"role": "user", "content": user}]}]
    json.dump(jobs, open(f"{OUT}/jobs_s{k}.json", "w"), ensure_ascii=False, indent=1)
    print("shard", k, len(part), "lines", sum(len(x["turns"]) for x in part), "turns", len(user), "chars")
print("split", len(split), "lines ->", sum(len(x["turns"]) for x in split), "turns; total lines", len(out))
