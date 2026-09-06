import hashlib,json,shutil,subprocess
from pathlib import Path
import numpy as np

root=Path.cwd();out=Path(__file__).resolve().parent
runner=out/'runner.snapshot';shutil.copy2(root/'build/ernie-text-runner',runner)
old=root/'outputs/diagnostic-text-first-norm-v1';package=root/'models/turbo1024-s64-portable'
ref=root/'outputs/pipeline1024-chinese-s64-fp32-v1/reference'
fixture=json.loads((ref/'fixture.json').read_text())
sha=lambda p:hashlib.file_digest(p.open('rb'),'sha256').hexdigest()
assert sha(ref/'text.f32')==fixture['inputs']['text']['sha256']
def run(command,name):
 with (out/(name+'.log')).open('w') as log:subprocess.run(command,check=True,stdout=log,stderr=subprocess.STDOUT,timeout=300)
run([str(runner),'--model',str(old/'model'),'--fixture',str(old/'fixture'),'--tokens','64','--output',str(out/'norm.f32'),'--backend','cpu'],'norm')
command=[str(runner),'--ids',str(ref/'text/ids.txt'),'--embeddings',str(package/'text/embeddings.bf16'),
         '--frequencies',str(package/'text/rope-inv-freq.f32'),'--tokens','64','--output',str(out/'text.f32'),'--backend','cpu']
for i in range(25):command+=['--model',str(package/f'text/block-{i:02d}')]
run(command,'text')
def error(a,b):return dict(nrmse=float(np.linalg.norm(a-b)/np.linalg.norm(b)),max_abs=float(np.max(abs(a-b))))
a=np.fromfile(out/'norm.f32','f4').reshape(64,3072)[:32].astype('f8')
b=np.fromfile(old/'torch.f32','f4').reshape(64,3072)[:32].astype('f8');d=np.fromfile(old/'fp64.f64','f8').reshape(64,3072)[:32]
result=dict(scope='Only CPU RMSNorm reductions changed; saved original official text reference, Chinese valid tokens',
            first_norm_vs_torch=error(a,b),first_norm_vs_fp64=error(a,d),
            full_text_vs_torch=error(np.fromfile(out/'text.f32','f4').astype('f8'),np.fromfile(ref/'text.f32','f4').astype('f8')),
            native_baseline_vs_torch=error(np.fromfile(root/'outputs/pipeline1024-chinese-s64-fp32-v1/trace/text.f32','f4').astype('f8'),np.fromfile(ref/'text.f32','f4').astype('f8')),
            source_sha256=sha(Path(__file__)),runner_sha256=sha(runner),reference_fixture_sha256=sha(ref/'fixture.json'))
(out/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
