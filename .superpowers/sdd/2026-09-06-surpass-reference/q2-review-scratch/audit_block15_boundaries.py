import hashlib,json,math
from pathlib import Path
import numpy as np
root=Path.cwd();dirs=[root/'outputs/q2-block15-boundary-v1',root/'outputs/q2-block15-boundary88-v1']
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
for d in dirs:
 p=json.loads((d/'plan.json').read_text());r=json.loads((d/'result.json').read_text());assert sha(d/'plan.json')==r['plan_sha256']
 for f,h in p['bound'].items():assert sha(f)==h
 for rec in r['out0_checks']:
  side=rec['side'];assert rec['out0_bitwise_equal'];assert sha(d/side/'out0.f32')==rec['expected_out0_sha256'];assert sha(d/side/'boundary.f32')==rec['boundary_sha256']
p=json.loads((dirs[0]/'plan.json').read_text());gate=np.fromfile(Path(p['sides']['official']['fixture'])/'in6.f32','<f4')
read=lambda d,s,f:np.memmap(d/s/f,'<f4','r').reshape(4160,4096)
b75={s:read(dirs[0],s,'boundary.f32') for s in ['official','native']};b88={s:read(dirs[1],s,'boundary.f32') for s in b75};final={s:read(dirs[1],s,'out0.f32') for s in b75}
counts={s:0 for s in b75};maxdiff={s:0. for s in b75};sums={k:0. for k in ['before','down','branch_update','final','residual_rounding_difference']};dots=0.
for start in range(0,4160,64):
 ys={};rounding={}
 for s in b75:
  a=np.array(b75[s][start:start+64]);b=np.array(b88[s][start:start+64]);f=np.array(final[s][start:start+64]);g=(b*gate).astype('f4');reconstructed=(a+g).astype('f4')
  counts[s]+=int(np.count_nonzero(reconstructed.view('u4')!=f.view('u4')));maxdiff[s]=max(maxdiff[s],float(abs(reconstructed-f).max()))
  ys[s]=g.astype('f8');rounding[s]=f.astype('f8')-a.astype('f8')-g.astype('f8')
 delta75=b75['native'][start:start+64].astype('f8')-b75['official'][start:start+64].astype('f8');delta88=b88['native'][start:start+64].astype('f8')-b88['official'][start:start+64].astype('f8');deltafinal=final['native'][start:start+64].astype('f8')-final['official'][start:start+64].astype('f8');deltag=ys['native']-ys['official'];deltar=rounding['native']-rounding['official']
 for k,x in [('before',delta75),('down',delta88),('branch_update',deltag),('final',deltafinal),('residual_rounding_difference',deltar)]:sums[k]+=float(np.sum(x*x))
 dots+=float(np.sum(delta75*deltag))
report={'scope':'Conditional perturbation propagation; actual reconstructed IEEE FP32 operations; not official local-error oracle','reconstructed_final_mismatches':counts,'reconstructed_final_max_diff':maxdiff,'conditional_difference_l2':{k:math.sqrt(v) for k,v in sums.items()},'before_update_cosine':dots/math.sqrt(sums['before']*sums['branch_update']),'gate_min':float(gate.min()),'gate_max':float(gate.max()),'result_sha256':[sha(d/'result.json') for d in dirs],'workers':[json.loads((d/'execution/worker-result.json').read_text()) for d in dirs],'audit_script_sha256':sha(__file__)}
(root/'.superpowers/sdd/2026-09-06-surpass-reference/task-Q2-boundary-evidence.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k!='workers'},indent=2))
