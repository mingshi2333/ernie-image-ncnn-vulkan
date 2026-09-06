import hashlib,json,shutil,subprocess
from pathlib import Path
import numpy as np
import torch
from safetensors import safe_open

torch.set_num_threads(4)
root=Path.cwd(); out=Path(__file__).resolve().parent
fixture=out/'fixture'; fixture.mkdir(); model=out/'model'; model.mkdir()
ids=json.loads((root/'outputs/pipeline1024-chinese-s64-fp32-v1/reference/fixture.json').read_text())['ids']
table=np.memmap(root/'models/turbo1024-s64-portable/text/embeddings.bf16',dtype='<u2',mode='r',shape=(131072,3072))
x=np.zeros((64,3072),dtype='f4'); x[:len(ids)]=(table[ids].astype('u4')<<16).view('f4')
with safe_open(root/'models/official/text-block-00.safetensors',framework='pt',device='cpu') as f:
    weight=f.get_tensor('language_model.model.layers.0.input_layernorm.weight').float()
weight.numpy().tofile(model/'text.ncnn.bin')
(model/'text.ncnn.param').write_text('7767517\n5 5\nInput in0 0 1 in0\nInput in1 0 1 in1\nInput in2 0 1 in2\nInput in3 0 1 in3\nRMSNorm norm 1 1 in0 out0 0=3072 1=1e-5 2=1\n')
x.tofile(fixture/'in0.f32')
for i in (1,2):np.zeros((64,128),dtype='f4').tofile(fixture/f'in{i}.f32')
np.zeros((64,64),dtype='f4').tofile(fixture/'in3.f32')
t=torch.from_numpy(x)
expected=(t*torch.rsqrt(t.pow(2).mean(-1,keepdim=True)+1e-5))*weight
double=(t.double()*torch.rsqrt(t.double().pow(2).mean(-1,keepdim=True)+1e-5))*weight.double()
expected.numpy().tofile(out/'torch.f32'); double.numpy().tofile(out/'fp64.f64')
runner=out/'runner.snapshot';shutil.copy2(root/'build/ernie-text-runner',runner)
command=[str(runner),'--model',str(model),'--fixture',str(fixture),'--tokens','64','--output',str(out/'actual.f32'),'--backend','cpu']
with (out/'native.log').open('w') as log:subprocess.run(command,check=True,stdout=log,stderr=subprocess.STDOUT,timeout=30)
a=np.fromfile(out/'actual.f32','f4').reshape(64,3072).astype('f8')[:len(ids)]
b=expected.numpy().astype('f8')[:len(ids)]; d=double.numpy()[:len(ids)]
def error(a,b):return dict(nrmse=float(np.linalg.norm(a-b)/np.linalg.norm(b)),max_abs=float(np.max(abs(a-b))))
sha=lambda p:hashlib.file_digest(p.open('rb'),'sha256').hexdigest()
result=dict(scope='Chinese real embedding and official first text RMSNorm weight; valid rows, CPU only',
            native_vs_torch=error(a,b),native_vs_fp64=error(a,d),torch_vs_fp64=error(b,d),
            source_sha256=sha(Path(__file__)),runner_sha256=sha(runner),
            files_sha256={str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file()})
(out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k!='files_sha256'}))
