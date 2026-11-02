# builds the v10 run dir (s2/run10), person voice refs (chars/v10) and edits from manual_fixes.json decisions
import json,copy,os,shutil
os.chdir('/workspace/projects/db13b071-b36a-44da-bacb-efac377e3057/build/work')
W='chars/win/'
R=json.load(open(W+'v10_refs.json')); P=json.load(open(W+'persons.json'))
os.makedirs('chars/v10',exist_ok=True); os.makedirs('s2/run10',exist_ok=True)
json.dump([{'id':p,'desc':P[p]['desc'],'voice_reference':[{'start':a,'end':b,'text':''} for a,b in v['spans']],'ref_from':v['from']} for p,v in R.items()],open('chars/v10/characters.json','w'),indent=1)
shutil.copy('chars/final/units.json','chars/v10/units.json')
L=json.load(open('s2/run7/lines.json')); by={l['id']:l for l in L}
def sp(l,a,b): l['start']=a; l['end']=b; l['dur']=round(b-a,3)
def new(src,i,a,b,hi):
    n=copy.deepcopy(by[src]); n['id']=i; sp(n,a,b); n['hi']=hi; L.append(n)
sp(by['L004'],37.188,44.2); sp(by['L005_1'],44.26,48.77); sp(by['L013'],88.0,92.924)
sp(by['L023_1'],138.084,139.82); sp(by['L023_2'],139.84,143.91)
sp(by['L025_2'],155.34,157.45); by['L025_2']['hi']='अबे तुझे मेडिकेशन की नहीं मोटिवेशन की जरूरत है'
new('L025_2','L025_3',157.45,158.908,'डिटर्मिनेशन है ना ऐसा')
sp(by['L034'],197.75,199.388)
sp(by['L035_4'],205.65,206.55); by['L035_4']['hi']='चेंज योर कंट्री'
new('L035_4','L035_6',206.55,207.43,'चेंज योर कॉंटिनेंट')
sp(by['L016'],102.8,109.692)     # MOM's untranslated words 101.34-102.7 sat inside BROWN's English-kept L016
new('L015_1','L015_3',101.25,102.75,'निखिल, मैं क्यों बताऊंगी')
L.sort(key=lambda l:l['start'])
json.dump(L,open('s2/run10/lines.json','w'),ensure_ascii=False,indent=1)
for f in ['speakers.json','split.json']: shutil.copy('s2/run7/'+f,'s2/run10/'+f)
C2P={'C0':'RAHUL','C8':'SICK','C4':'BROWN','C2':'MOM','C7':'DAD','C6':'BOSS','C10':'BEARD','C5':'PURPLE','C1':'TEE','C3':'CURLYW','C12':'RAHUL','C11':'RAHUL'}
cm={k:C2P[v] for k,v in json.load(open('s2/run7/charmap.json')).items() if v in C2P}
cm.update({'L005_1':'BROWN','L025_3':'BROWN','L035_4':'BROWN','L035_6':'TEE','L015_3':'MOM','L049_1':'RAHUL','L054':'RAHUL','L048':'BEARD'})
json.dump(cm,open('s2/run10/charmap.json','w'),indent=1)
E={'L004':{'text':'Step out, have a few beers, bacteria die. Ever seen a bartender with typhoid? Beer!'},
 'L005_1':{'text':"Don't overthink it. The more you think you have typhoid"},
 'L025_2':{'text':'You need motivation, not medication.'},'L025_3':{'text':'Determination, right?'},
 'L035_4':{'text':'Change your country,'},'L035_6':{'text':'change your continent.'},
 'L015_3':{'text':'Why would I tell?'}}
for k in ['L006_1','L023_2','L005_2','L007_3','L034','L035_1','L039_1','L049_1','L054']: E[k]={'retake':10}
json.dump(E,open('s2/run10/edits.json','w'),indent=1)
print('lines',len(L),'mapped',len(cm),'edits',len(E))
