import hashlib,json,math
from pathlib import Path
import numpy as np
root=Path.cwd();out=root/'outputs/q2-chinese-step0-current-exact-head-stack-v1';p=json.loads((out/'plan.json').read_text());r=json.loads((out/'result.json').read_text());w=json.loads((out/'execution/worker-result.json').read_text())
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(2**20),b''):h.update(b)
 return h.hexdigest()
assert w['passed'] and not w['stop_reason'] and r['denominator']==36 and not r['native_acceptance_eligible']
assert sha(out/'plan.json')==r['plan_sha256']=='3aba6aa5ec1bec491a36187bc0f62c5332549d26df396655010d1966db4201f7'
for file,digest in {**p['bound_sha256'],**p['model_files_sha256']}.items():assert sha(file)==digest
for item in p['inputs'].values():assert sha(item['file'])==sha(item['source'])==item['sha256']
def metrics(a,b,start=0,end=4160*4096):
 x=np.memmap(a,'<f4','r');y=np.memmap(b,'<f4','r');s=norm=maximum=0.
 for t in range(start,end,524288):
  xx=np.array(x[t:min(t+524288,end)],dtype='f8');yy=np.array(y[t:min(t+524288,end)],dtype='f8');assert np.isfinite(xx).all() and np.isfinite(yy).all();delta=xx-yy
  s+=float(np.dot(delta,delta));norm+=float(np.dot(yy,yy));maximum=max(maximum,float(abs(delta).max()))
 return {'nrmse':math.sqrt(s/norm),'error_l2':math.sqrt(s),'max_abs_error':maximum}
rows=[]
for item,row in zip(p['denominator'],r['rows']):
 i=item['layer'];actual=out/'trace'/item['trace'];reference=Path(item['file']);assert sha(actual)==row['actual_sha256'];assert sha(reference)==item['sha256']
 independent=metrics(actual,reference)
 for key,value in independent.items():assert math.isclose(value,row[key],rel_tol=1e-12,abs_tol=1e-12)
 old=root/f'outputs/diagnostic-chinese-exact-heads-v1/trace/block-{i}.f32'
 rows.append({'layer':i,**independent,'image_tokens':metrics(actual,reference,0,4096*4096),'text_tokens':metrics(actual,reference,4096*4096,4160*4096),'old_exact_head':metrics(old,reference),'old_trace_sha256':sha(old)})
assert sha(out/'actual.f32')==r['final_sha256']==sha(out/'trace/block-35.f32')
samples=[json.loads(x) for x in (out/'execution/samples.jsonl').read_text().splitlines()]
report={'scope':'Independent CPU rehash/reduction; no additional GPU run; descriptive not quality gate','worker':w,'minimum_host_available_bytes':min(s['host_available'] for s in samples),'result_sha256':sha(out/'result.json'),'worker_result_sha256':sha(out/'execution/worker-result.json'),'plan_sha256':sha(out/'plan.json'),'final_sha256':r['final_sha256'],'rows':rows,'audit_script_sha256':sha(__file__)}
(root/'.superpowers/sdd/2026-09-06-surpass-reference/task-Q2-exact-head-result.json').write_text(json.dumps(report,indent=2)+'\n')
for i in [0,12,13,14,15,16,17,28,29,34,35]:print(json.dumps(rows[i]))
