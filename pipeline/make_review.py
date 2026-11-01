#!/usr/bin/env python3
"""Creator review page: one self-contained HTML file to check a dub line by line in the browser.

usage: make_review.py <qa.json> <lines.json> <dub_video> <out.html> [--src-video V] [--avqa avqa.json] [--title T]

The page shows the dubbed video (switch to the original audio/video with one key), every line with its time, the source
transcript, the dubbed text (editable), the variant/tempo/gain that was used and any AV-QA flags on that line with their
thumbnails. Per line you can play it, edit the text, or mark it for a new take. "Export edits" downloads edits.json:
  {"L003": {"text": "new line", "retake": true}, ...}
dub.py reads it with EDITS=edits.json (edited text replaces the Glimmer variants; retake lines get fresh TTS takes).
No server, no upload: video paths are relative to the HTML file.
"""
import argparse, html, json, os

ap = argparse.ArgumentParser(); ap.add_argument("qa"); ap.add_argument("lines"); ap.add_argument("dub"); ap.add_argument("out")
ap.add_argument("--src-video"); ap.add_argument("--avqa"); ap.add_argument("--title", default="LocalDub review")
a = ap.parse_args()
L = {l["id"]: l for l in json.load(open(a.lines))}
qa = json.load(open(a.qa))["lines"]
ev = json.load(open(a.avqa))["events"] if a.avqa else []
od = os.path.dirname(os.path.abspath(a.out))
rel = lambda p: os.path.relpath(os.path.abspath(p), od) if p else None
avdir = os.path.dirname(os.path.abspath(a.avqa)) if a.avqa else None
rows = []
for q in sorted(qa, key=lambda q: L[q["id"]]["start"]):
    l = L[q["id"]]
    fl = [{"kind": e["kind"], "sev": e["severity"], "t": [e["start"], e["end"]], "why": e["why"],
           "thumb": rel(os.path.join(avdir, e["thumb"])) if e.get("thumb") and avdir else None} for e in ev if e.get("line") == q["id"]]
    rows.append({"id": q["id"], "start": l["start"], "end": l["end"], "src": l.get("hi", ""), "dub": q.get("en") or "",
                 "variant": q.get("variant"), "kept": q.get("kept"), "tempo": q.get("tempo"), "gain": q.get("gain"),
                 "align": (q.get("align") or {}).get("tempi"), "flags": fl})
data = {"title": a.title, "dub": rel(a.dub), "src": rel(a.src_video), "rows": rows,
        "other": [{"kind": e["kind"], "sev": e["severity"], "t": [e["start"], e["end"]], "why": e["why"]} for e in ev if not e.get("line")]}
page = r"""<!doctype html><html><head><meta charset="utf-8"><title>__TITLE__</title>
<style>
body{margin:0;font:14px/1.4 system-ui,sans-serif;background:#111418;color:#e8e8e8;display:grid;grid-template-columns:minmax(420px,46%) 1fr;height:100vh}
#left{padding:16px;border-right:1px solid #2a2f36;overflow:auto}#right{overflow:auto}
video{width:100%;background:#000;border-radius:6px}h1{font-size:18px;margin:0 0 8px}
.bar{display:flex;gap:8px;margin:8px 0;flex-wrap:wrap}button{background:#2a3340;color:#e8e8e8;border:1px solid #3a4452;border-radius:5px;padding:5px 10px;cursor:pointer}
button.on{background:#2f6f5e}table{width:100%;border-collapse:collapse}td{border-bottom:1px solid #22272e;padding:6px 8px;vertical-align:top}
tr.cur td{background:#1c2a26}.t{color:#8aa;white-space:nowrap;font-variant-numeric:tabular-nums}.src{color:#9aa4ae;font-size:13px}
textarea{width:100%;background:#0c0f12;color:#e8e8e8;border:1px solid #2a2f36;border-radius:4px;font:inherit;min-height:38px}
.meta{color:#7d8790;font-size:12px}.flag{display:inline-block;font-size:12px;padding:1px 6px;border-radius:10px;margin:2px 4px 0 0}
.high{background:#5a1f1f}.medium{background:#5a4a1f}.low{background:#333}img.th{width:150px;border-radius:4px;display:block;margin-top:4px}
.edited textarea{border-color:#2f6f5e}.retake{outline:2px solid #a66}
</style></head><body>
<div id="left"><h1>__TITLE__</h1><video id="v" controls></video>
<div class="bar"><button id="bd" class="on">Dub</button><button id="bo">Original</button><button id="ex">Export edits</button>
<span class="meta" id="st"></span></div>
<p class="meta">Space: play/pause &middot; O: switch dub/original &middot; click a time to play that line &middot; edits stay in this browser until you export them.</p>
<div id="other"></div></div>
<div id="right"><table id="tb"></table></div>
<script>
const D=__DATA__;const v=document.getElementById('v');let mode='dub';v.src=D.dub;
const key='ld_edits_'+D.title;let E=JSON.parse(localStorage.getItem(key)||'{}');
function save(){localStorage.setItem(key,JSON.stringify(E));document.getElementById('st').textContent=Object.keys(E).length+' line(s) edited/marked'}
function sw(m){if(m==='orig'&&!D.src)return;const t=v.currentTime,p=!v.paused;mode=m;v.src=m==='dub'?D.dub:D.src;v.currentTime=t;if(p)v.play();
 document.getElementById('bd').classList.toggle('on',m==='dub');document.getElementById('bo').classList.toggle('on',m!=='dub')}
document.getElementById('bd').onclick=()=>sw('dub');document.getElementById('bo').onclick=()=>sw('orig');
if(!D.src)document.getElementById('bo').disabled=true;
let stopAt=null;v.ontimeupdate=()=>{if(stopAt&&v.currentTime>=stopAt){v.pause();stopAt=null}
 document.querySelectorAll('tr[data-s]').forEach(r=>r.classList.toggle('cur',v.currentTime>=+r.dataset.s&&v.currentTime<+r.dataset.e))};
function play(s,e){v.currentTime=Math.max(0,s-0.3);stopAt=e+0.4;v.play()}
const tb=document.getElementById('tb');
D.rows.forEach(r=>{const tr=document.createElement('tr');tr.dataset.s=r.start;tr.dataset.e=r.end;
 const ed=E[r.id]||{};const fl=r.flags.map(f=>`<span class="flag ${f.sev}" title="${f.why}">${f.kind} ${f.t[0].toFixed(1)}s</span>${f.thumb?`<img class="th" src="${f.thumb}">`:''}`).join('');
 const meta=r.kept?`kept original (${r.kept})`:`${r.variant} &middot; tempo ${r.tempo}${r.align?' &middot; placed '+r.align.join('/'):''} &middot; gain ${r.gain}`;
 tr.innerHTML=`<td class="t"><a href="#" data-p>${r.start.toFixed(2)}</a><br>${r.id}</td><td><div class="src">${r.src}</div>
 <textarea ${r.kept?'disabled':''}>${ed.text||r.dub}</textarea><div class="meta">${meta}</div>${fl}</td>
 <td><button data-r>${ed.retake?'retake ✓':'retake'}</button></td>`;
 if(ed.text)tr.classList.add('edited');if(ed.retake)tr.classList.add('retake');
 tr.querySelector('[data-p]').onclick=e=>{e.preventDefault();play(r.start,r.end)};
 tr.querySelector('textarea').oninput=e=>{const t=e.target.value.trim();E[r.id]=E[r.id]||{};if(t&&t!==r.dub)E[r.id].text=t;else delete E[r.id].text;
  if(!Object.keys(E[r.id]).length)delete E[r.id];tr.classList.toggle('edited',!!(E[r.id]&&E[r.id].text));save()};
 tr.querySelector('[data-r]').onclick=e=>{E[r.id]=E[r.id]||{};E[r.id].retake=!E[r.id].retake;if(!E[r.id].retake)delete E[r.id].retake;
  if(!Object.keys(E[r.id]).length)delete E[r.id];e.target.textContent=(E[r.id]&&E[r.id].retake)?'retake ✓':'retake';tr.classList.toggle('retake',!!(E[r.id]&&E[r.id].retake));save()};
 tb.appendChild(tr)});
document.getElementById('other').innerHTML=D.other.length?'<h1>Flags outside lines</h1>'+D.other.map(f=>`<div><a href="#" onclick="play(${f.t[0]},${f.t[1]});return false">${f.t[0].toFixed(1)}s</a> <span class="flag ${f.sev}">${f.kind}</span> ${f.why}</div>`).join(''):'';
document.getElementById('ex').onclick=()=>{const b=new Blob([JSON.stringify(E,null,1)],{type:'application/json'});const u=URL.createObjectURL(b);
 const x=document.createElement('a');x.href=u;x.download='edits.json';x.click()};
document.onkeydown=e=>{if(e.target.tagName==='TEXTAREA')return;if(e.code==='Space'){e.preventDefault();v.paused?v.play():v.pause()}if(e.key==='o'||e.key==='O')sw(mode==='dub'?'orig':'dub')};
save();
</script></body></html>"""
page = page.replace("__TITLE__", html.escape(a.title)).replace("__DATA__", json.dumps(data, ensure_ascii=False).replace("</", "<\\/"))
open(a.out, "w", encoding="utf-8").write(page)
print("review page:", a.out, len(rows), "lines,", sum(len(r["flags"]) for r in rows), "flags on lines")
