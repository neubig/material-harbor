"""Blind image-grounded scientific audit of MCQA labels (not solver accuracy)."""
import argparse, base64, hashlib, json, math, os, re, urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from statistics import NormalDist
PROXY='https://llm-proxy.app.all-hands.dev/v1/chat/completions'
def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def wilson(k,n):
 z=NormalDist().inv_cdf(.975); p=k/n; d=1+z*z/n; c=(p+z*z/(2*n))/d; r=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
 return [max(0,c-r),min(1,c+r)]
def task_inputs(task):
 instruction=(task/'instruction.md').read_text()
 if (task/'tests/reference.json').exists():
  ref=json.loads((task/'tests/reference.json').read_text())
 else:
  label=json.loads((task/'tests/label.json').read_text())
  ref={'answer':label['correct_answer'],'options':list('ABCD')}
 images=sorted(p for p in (task/'environment/data').glob('image*') if p.is_file())
 return instruction,ref,images
def freeze(source,out,n):
 tasks=sorted([x for x in source.iterdir() if (x/'task.toml').exists()],key=lambda x:hashlib.sha256(('scientific-audit-v1:'+x.name).encode()).hexdigest())[:n]
 def commitment(task):
  reference=task/'tests/reference.json'
  return digest(reference if reference.exists() else task/'tests/label.json')
 obj={'version':1,'selection':'lowest SHA256(scientific-audit-v1:task_id), frozen before adjudication','n':n,'judge':'openai/gpt-5.1','temperature':0,'blind':'Gold labels withheld from judge; source figures and all choices shown.','decision_contract':'Judge returns every scientifically defensible option. Ambiguous/low-confidence items are unresolved, not forced.','task_ids':[x.name for x in tasks],'task_hashes':{x.name:{'instruction':digest(x/'instruction.md'),'images':[digest(p) for p in task_inputs(x)[2]],'gold_commitment':commitment(x)} for x in tasks}}
 out.write_text(json.dumps(obj,indent=2)+'\n')
def call(task,model):
 instruction,ref,images=task_inputs(task)
 data=task/'environment/data'
 if (data/'input.json').exists():
  value=json.loads((data/'input.json').read_text())
  scientific_prompt=value['question']
 elif (data/'question.json').exists():
  value=json.loads((data/'question.json').read_text())
  options=json.loads(value['options'])
  scientific_prompt='Which captions accurately describe the figure?\n'+'\n'.join(f"{o['label']}. {o['caption']}" for o in options)
 else:
  scientific_prompt=instruction.split('Answer with',1)[0]
 content=[{'type':'text','text':scientific_prompt+'\n\nINDEPENDENT AUDIT: Assess every listed option using the actual figure(s). Return JSON only: {"valid_options":["A"],"ambiguous":false,"confidence":"high","reason":"brief image-grounded reason"}. Include all scientifically defensible options; set ambiguous true if the figure/question does not uniquely determine one option. Do not assume the dataset label.'}]
 for image in images:
  mime='image/png' if image.suffix.lower()=='.png' else 'image/jpeg'; content.append({'type':'image_url','image_url':{'url':f'data:{mime};base64,'+base64.b64encode(image.read_bytes()).decode()}})
 body={'model':model,'temperature':0,'max_tokens':5000,'messages':[{'role':'user','content':content}]}; req=urllib.request.Request(PROXY,data=json.dumps(body).encode(),headers={'Content-Type':'application/json','Authorization':'Bearer '+os.environ['LLM_API_KEY']}); raw=json.load(urllib.request.urlopen(req,timeout=300))['choices'][0]['message'].get('content') or ''; m=re.search(r'\{.*\}',raw,re.S); adjud=json.loads(m.group(0) if m else raw); valid=adjud.get('valid_options'); options=ref['options']
 if not isinstance(valid,list) or not valid or any(x not in options for x in valid): raise ValueError('invalid judge JSON')
 return {'task_id':task.name,'valid_options':valid,'ambiguous':bool(adjud.get('ambiguous')),'confidence':adjud.get('confidence'),'reason':adjud.get('reason'),'gold':ref['answer'],'options':options}
def report(rows,expected):
 usable=[r for r in rows if not r.get('error') and not r['ambiguous'] and r['confidence']=='high']; fp_n=fp=fn_n=fn=0
 for r in usable:
  valid=set(r['valid_options']); gold=r['gold']
  for option in r['options']:
   if option in valid: fn_n+=1; fn+=option!=gold
   else: fp_n+=1; fp+=option==gold
 def rate(k,n): return {'errors':k,'denominator':n,'rate':k/n if n else None,'wilson95':wilson(k,n) if n else None}
 return {'version':1,'expected':expected,'completed':len(rows),'usable_high_confidence_unambiguous':len(usable),'unresolved_or_errors':len(rows)-len(usable),'false_positive':rate(fp,fp_n),'false_negative':rate(fn,fn_n),'interpretation':'Independent image-grounded scientific option audit. Parser perturbation audit is separate. Rates are not inferred from solver accuracy.','rows':rows}
def main():
 p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--manifest',type=Path,required=True);p.add_argument('--raw',type=Path);p.add_argument('--report',type=Path);p.add_argument('--count',type=int,default=30);p.add_argument('--freeze',action='store_true');p.add_argument('--model',default='openai/gpt-5.1');p.add_argument('--concurrency',type=int,default=4);a=p.parse_args()
 if a.freeze: freeze(a.source,a.manifest,a.count);return
 man=json.loads(a.manifest.read_text());tasks=[a.source/x for x in man['task_ids']]
 def one(t):
  try:return call(t,a.model)
  except Exception as e:return {'task_id':t.name,'error':type(e).__name__+': '+str(e)[:300]}
 with ThreadPoolExecutor(max_workers=a.concurrency) as ex: rows=list(ex.map(one,tasks))
 raw=report(rows,len(tasks));a.raw.parent.mkdir(parents=True,exist_ok=True);a.raw.write_text(json.dumps(raw,indent=2)+'\n'); summary={k:v for k,v in raw.items() if k!='rows'};a.report.write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
