import hashlib,json,math
from pathlib import Path
import numpy as np
root=Path.cwd();out=root/'outputs/q2-chinese-step0-block15-official-input-v1';p=json.loads((out/'plan.json').read_text());r=json.loads((out/'result.json').read_text())
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
assert sha(out/'plan.json')==r['plan_sha256'];assert sha(out/'actual.f32')==r['actual_sha256']==sha(out/'trace/block-0.f32')
for file,digest in {**p['bound_sha256'],**p['model_files_sha256']}.items():assert sha(file)==digest
for item in [*p['inputs'].values(),p['expected'],p['baseline_stack_block15']]:assert sha(item['file'])==item['sha256']
base=json.loads(Path(p['base_plan']).read_text());read=lambda q:np.memmap(q,'<f4','r')
a=read(out/'actual.f32');n=read(p['baseline_stack_block15']['file']);o=read(p['expected']['file']);local=prop=total=dot=0.;maxima=[0.,0.,0.]
for t in range(0,a.size,524288):
 aa=a[t:t+524288].astype('f8');nn=n[t:t+524288].astype('f8');oo=o[t:t+524288].astype('f8');assert np.isfinite(aa).all() and np.isfinite(nn).all() and np.isfinite(oo).all()
 l=aa-oo;v=nn-aa;allerror=nn-oo;local+=float(l@l);prop+=float(v@v);total+=float(allerror@allerror);dot+=float(l@v)
 for i,x in enumerate([l,v,allerror]):maxima[i]=max(maxima[i],float(abs(x).max()))
assert math.isclose(local+prop+2*dot,total,rel_tol=1e-12)
for got,key in [(local,'same_input_official_difference'),(prop,'difference_from_accumulated_stack'),(total,'accumulated_stack_official_difference')]:assert math.isclose(math.sqrt(got),r[key]['error_l2'],rel_tol=1e-12)
in_native=read(Path(p['base_plan']).parent/'trace/block-14.f32');in_official=read(p['inputs']['in0']['file']);input_ss=0.
for t in range(0,in_native.size,524288):
 delta=in_native[t:t+524288].astype('f8')-in_official[t:t+524288].astype('f8');input_ss+=float(delta@delta)
w=json.loads((out/'execution/worker-result.json').read_text());samples=[json.loads(x) for x in (out/'execution/samples.jsonl').read_text().splitlines()]
d={'scope':'Conditional native input sensitivity decomposition, not causal percentages or pure error for changed-input difference','input_l2':math.sqrt(input_ss),'local_same_input_l2':math.sqrt(local),'propagated_input_difference_l2':math.sqrt(prop),'total_l2':math.sqrt(total),'local_propagated_inner_product':dot,'local_propagated_cosine':dot/math.sqrt(local*prop),'conditional_l2_gain':math.sqrt(prop/input_ss),'result_sha256':sha(out/'result.json'),'plan_sha256':sha(out/'plan.json'),'worker':w,'minimum_host_available_bytes':min(s['host_available'] for s in samples),'audit_script_sha256':sha(__file__),'inputs':p['inputs']}
(root/'.superpowers/sdd/2026-09-06-surpass-reference/task-Q2-block15-teacher-evidence.json').write_text(json.dumps(d,indent=2)+'\n');print(json.dumps({k:v for k,v in d.items() if k not in ['inputs','worker']},indent=2))
