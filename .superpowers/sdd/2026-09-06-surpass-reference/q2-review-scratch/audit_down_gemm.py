import json,struct,hashlib
from pathlib import Path
import numpy as np
root=Path.cwd();d=root/'outputs/q2-block15-boundary87-v1';previous=root/'outputs/q2-block15-boundary88-v1';mathdir=root/'outputs/q2-block15-mlp-cpu-v2'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
p=json.loads((d/'plan.json').read_text());r=json.loads((d/'result.json').read_text());assert sha(d/'plan.json')==r['plan_sha256']
for f,h in p['bound'].items():assert sha(f)==h
for row in r['out0_checks']:
 assert row['out0_bitwise_equal'];assert sha(d/row['side']/'out0.f32')==row['expected_out0_sha256'];assert sha(d/row['side']/'boundary.f32')==row['boundary_sha256']
rows=json.loads((mathdir/'result.json').read_text())['rows'];file=root/'models/official/dit-block-15.safetensors';assert sha(file)=='d3c1b748148895cf9f4a72669bd77d69a1315c024bcc4f1e5f4f96ecd283a3b2'
with file.open('rb') as f:n=struct.unpack('<Q',f.read(8))[0];header=json.loads(f.read(n));base=8+n
h=header['layers.15.mlp.linear_fc2.weight'];assert h['shape']==[4096,12288] and h['dtype']=='BF16';b=np.memmap(file,'<u2','r',offset=base+h['data_offsets'][0],shape=(4096,12288));w=(b.astype('u4')<<16).view('f4').astype('f8')
z=np.concatenate([np.array(np.memmap(d/s/'boundary.f32','<f4','r').reshape(4160,12288)[rows],dtype='f8') for s in ['official','native']]);actual=np.concatenate([np.array(np.memmap(previous/s/'boundary.f32','<f4','r').reshape(4160,4096)[rows],dtype='f8') for s in ['official','native']]);truth=z@w.T;full=np.fromfile(mathdir/'float64.bin','f8').reshape(10,4096);local=actual-truth;total=actual-full;upstream=truth-full
stats=lambda a:{'l2':float(np.linalg.norm(a)),'max':float(abs(a).max())}
res={'scope':'Five fixed rows, exact native down-Gemm inputs and official BF16-expanded weight, FP64 dot; not full oracle','rows':rows,'local_down_gemm_error':stats(local),'full_native_mlp_error':stats(total),'upstream_mlp_propagated_error':stats(upstream),'local_down_full_error_cosine':float(np.sum(local*total)/(np.linalg.norm(local)*np.linalg.norm(total))),'native_87_result_sha256':sha(d/'result.json'),'weights_sha256':sha(file),'source_script_sha256':sha(__file__)}
(root/'.superpowers/sdd/2026-09-06-surpass-reference/task-Q2-down-gemm-evidence.json').write_text(json.dumps(res,indent=2)+'\n');print(json.dumps(res,indent=2))
