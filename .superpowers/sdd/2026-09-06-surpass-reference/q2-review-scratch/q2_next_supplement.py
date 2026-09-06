import json,hashlib
from pathlib import Path
import numpy as np
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
p=Path('outputs/q2-history-vector-down-v2');i=json.loads((p/'identity.json').read_text())
assert sha(p/'runner')==i['runner'];assert sha(p/'diagnose_text_stages.py')==i['worker']
target=Path('.superpowers/sdd/2026-09-06-surpass-reference/task-Q2-next-evidence.json');d=json.loads(target.read_text())
d['vector_text_provenance']={'runner_sha256':sha(p/'runner'),'worker_sha256':sha(p/'diagnose_text_stages.py'),'launcher_sha256':sha(p/'run.py.snapshot'),'candidate_sha256':sha(p/'chinese/candidate.f32'),'source_inventory':json.loads((p/'source-inventory.json').read_text())}
raw=np.arange(8,0,-1,dtype='f4')/np.float32(8);sigmas=np.float32(4)*raw/(np.float32(1)+np.float32(3)*raw);delta=float(np.float32(sigmas[1]-sigmas[0]));rows=[]
for name in ['pipeline1024-chinese-s64-fp32-chunked-v1','pipeline1024-chinese-s64-fp32-reference-text-v1','pipeline1024-chinese-vectordown-fp32-v1']:
 p=Path('outputs')/name
 read=lambda q:np.fromfile(q,'<f4').astype('f8')
 prediction=read(p/'trace/prediction-0.f32')-read(p/'reference/prediction-0.f32');sample=read(p/'trace/step-0.f32')-read(p/'reference/step-0.f32');residual=sample-delta*prediction
 rows.append({'run':name,'delta':delta,'prediction_injection_l2':float(np.linalg.norm(delta*prediction)),'update_difference_residual_l2':float(np.linalg.norm(residual)),'residual_max':float(abs(residual).max())})
d['step0_euler_decomposition']=rows;d['supplement_script_sha256']=sha(__file__);target.write_text(json.dumps(d,indent=2,ensure_ascii=False)+'\n');print(json.dumps(rows,indent=2))
