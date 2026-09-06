#!/usr/bin/env python3
"""One official MLP with immutable native81; no full official block forward."""
import json,sys
from pathlib import Path
import numpy as np
from diagnose_official_block_hooks import sha,verify,require,imports,freeze_runtime,tensor_bytes
N81='da8bc040080f6674042c860a372c8f004041dae617a0cdeffae96f3c2be61d54'
EXPECTED={'81':N81,'87':'9a34cfce2296c4f2b3ef7f48aad848faf7362d349ee3ae4b7fcb0719ce0ba3c1','88':'5b0f3edbb3c641a4826da737016b59f4d6eee1f708b919298a883234d4e2eba7'}
SHAPES={'81':(1,4160,4096),'84':(1,4160,12288),'87':(1,4160,12288),'88':(1,4160,4096)}
def endpoint(hashes):
 require(all(hashes.get(k)==v for k,v in EXPECTED.items()),'Official matched endpoint changed')
def run(path):
 p=json.loads(path.read_text());verify(p['bound']);out=Path(p['output']);f=out/'input81.f32';require(sha(f)==N81,'Native81 changed')
 torch,_,load_block=imports(out)
 a=np.fromfile(f,'<f4');require(a.size==17039360 and np.isfinite(a).all(),'Input denominator/finite');x=torch.from_numpy(a).reshape(1,4160,4096).transpose(0,1).to('cuda')
 block,component=load_block(Path(p['weight']),15);require(component['sha256']==p['official_component_sha256'],'Weight identity');block=block.to('cuda')
 config={'cuda':torch.version.cuda,'cudnn':torch.backends.cudnn.version(),'tf32_matmul':torch.backends.cuda.matmul.allow_tf32,'tf32_cudnn':torch.backends.cudnn.allow_tf32,'float32_matmul_precision':torch.get_float32_matmul_precision(),'device':torch.cuda.get_device_name(),'threads':torch.get_num_threads(),'attention_processor':type(block.self_attention.processor).__name__,'attention_backend':str(block.self_attention.processor._attention_backend),'training':block.training,'config':torch.__config__.show()}
 require(config==json.loads(Path(p['old_torch_config']).read_text()),'Actual options changed');(out/'execution/torch-config.json').write_text(json.dumps(config,indent=2)+'\n')
 before=freeze_runtime(out/'execution','gpu-before-forward-runtime');captures={};counts={k:0 for k in SHAPES}
 def capture(k,v):counts[k]+=1;require(counts[k]==1,'Repeated hook');captures[k]=v.detach()
 handles=[block.mlp.register_forward_pre_hook(lambda m,a:capture('81',a[0])),block.mlp.up_proj.register_forward_hook(lambda m,a,y:capture('84',y)),block.mlp.linear_fc2.register_forward_pre_hook(lambda m,a:capture('87',a[0])),block.mlp.linear_fc2.register_forward_hook(lambda m,a,y:capture('88',y))]
 final=block.mlp(x);torch.cuda.synchronize()
 for h in handles:h.remove()
 require(all(v==1 for v in counts.values()),'Missing hook');actual=out/'official';actual.mkdir();hashes={}
 for name,value in captures.items():
  file=actual/(name+'.f32');file.write_bytes(tensor_bytes(value.transpose(0,1),SHAPES[name]));hashes[name]=sha(file)
 (out/'official-observation.json').write_text(json.dumps({'output_sha256':hashes,'counts':counts},indent=2)+'\n');endpoint(hashes)
 freeze_runtime(out/'execution','gpu-after-forward-runtime');verify(p['bound']);verify({k:v['sha256'] for k,v in before['files'].items()})
 result={'status':'completed','scope':'Same-native81 up84 diagnostic only','plan_sha256':sha(path),'official_sha256':hashes,'counts':counts,'forwards':{'native_full_block':1,'official_full_block':0,'official_mlp_only':1},'formal_acceptance':False}
 (out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
if __name__=='__main__':run(Path(sys.argv[1]))
